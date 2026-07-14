"""Deterministic tests for the core data pipeline in tools.py."""

import numpy as np
import pandas as pd
import pytest

from tools import (
    calculate_daily_changes,
    classify_county_type,
    compute_national_daily,
    compute_national_timeseries,
    compute_window_outcomes,
    extract_county_state,
    find_data_corrections,
    get_state_bounds_for_zoom,
    monthly_snapshot_long,
    peer_median_series,
    poisson_rate_ci,
    precompute_daily_diffs,
    precompute_per_capita,
    rolling_cfr,
    rolling_cfr_from_daily,
)
from tests.conftest import DATES


def test_extract_county_state():
    assert extract_county_state("Alpha County, AA") == ("Alpha County", "AA")
    # rsplit keeps commas inside county names intact
    assert extract_county_state("Prince, of Wales, AK") == ("Prince, of Wales", "AK")
    assert extract_county_state("NoComma") == ("NoComma", None)


def test_daily_changes_clips_negatives():
    ts = pd.DataFrame({"Date": pd.to_datetime(DATES),
                       "Cases": [0, 5, 10, 20, 30, 25, 40, 50, 60, 70]})
    out = calculate_daily_changes(ts, "Cases")
    daily = out["Daily Cases"].tolist()
    assert daily[5] == 0            # negative diff (30→25) clipped to zero
    assert daily[1] == 5 and daily[3] == 10
    assert min(daily) >= 0


def test_precompute_daily_diffs_matches_single_county(cases_df, deaths_df):
    daily_c, _ = precompute_daily_diffs(cases_df, deaths_df)
    alpha = daily_c[daily_c["countyFIPS"] == "01001"].iloc[0]
    assert alpha[DATES[1]] == 1 and alpha[DATES[4]] == 4
    gamma = daily_c[daily_c["countyFIPS"] == "02001"].iloc[0]
    assert gamma[DATES[5]] == 0     # correction clipped


def test_per_capita_math(cases_df, deaths_df, population_df):
    pc_cases, _ = precompute_per_capita(cases_df, deaths_df, population_df)
    alpha = pc_cases[pc_cases["countyFIPS"] == "01001"].iloc[0]
    assert alpha[DATES[-1]] == pytest.approx(45 / 10_000 * 100_000)
    statewide = pc_cases[pc_cases["countyFIPS"] == "00000"].iloc[0]
    assert np.isnan(statewide[DATES[-1]])  # zero population → NaN, never inflated


def test_national_totals_exclude_statewide(cases_df, deaths_df):
    nat = compute_national_timeseries(cases_df, deaths_df, "Cases")
    # 45 + 40 + 70 (statewide 900 excluded)
    assert nat["Value"].iloc[-1] == 155
    daily_c, _ = precompute_daily_diffs(cases_df, deaths_df)
    nat_daily = compute_national_daily(daily_c)
    assert nat_daily["Value"].sum() == pytest.approx(155 + 5)  # +5 from clipped dip


def test_window_outcomes(cases_df, deaths_df, population_df):
    win = compute_window_outcomes(cases_df, deaths_df, population_df,
                                  DATES[4], DATES[-1])
    alpha = win[win["countyFIPS"] == "01001"].iloc[0]
    assert alpha["cases_per_100k"] == pytest.approx((45 - 10) / 10_000 * 100_000)
    assert alpha["deaths_per_100k"] == pytest.approx((4 - 1) / 10_000 * 100_000)
    assert alpha["case_fatality_rate"] == pytest.approx(3 / 35 * 100)
    assert "00000" not in set(win["countyFIPS"])
    with pytest.raises(ValueError):
        compute_window_outcomes(cases_df, deaths_df, population_df,
                                DATES[-1], DATES[0])


def test_monthly_snapshot_long(cases_df):
    long_df = monthly_snapshot_long(cases_df)
    assert set(long_df["Month"]) == {"2020-01"}      # all fixture dates share a month
    assert "00000" not in set(long_df["countyFIPS"])
    alpha = long_df[long_df["countyFIPS"] == "01001"]
    assert len(alpha) == 1 and alpha["value"].iloc[0] == 0  # first date of month kept


def test_classify_county_type_fallback_and_rucc(population_df):
    fallback = classify_county_type(population_df, urban_threshold=15_000)
    assert dict(zip(fallback["countyFIPS"], fallback["County_Type"])) == {
        "01001": "Rural", "01003": "Urban", "02001": "Urban",
    }
    rucc = pd.DataFrame({"countyFIPS": ["01001", "01003"], "State": ["AA", "AA"],
                         "rucc_code": [2, 8]})
    by_rucc = classify_county_type(population_df, rucc_df=rucc)
    assert dict(zip(by_rucc["countyFIPS"], by_rucc["County_Type"])) == {
        "01001": "Metro", "01003": "Nonmetro",
    }


def test_state_bounds():
    assert get_state_bounds_for_zoom("PA")["zoom"] == 6
    assert get_state_bounds_for_zoom("ZZ") is None


def test_peer_median_series(cases_df, deaths_df, population_df):
    daily_c, _ = precompute_daily_diffs(cases_df, deaths_df)
    pm = peer_median_series(daily_c, population_df, ["01001", "01003", "02001"])
    assert len(pm) == len(DATES)
    # last day's daily diffs: alpha 9, beta 10, gamma 10 → per-100k medians
    per100k = sorted([9 / 10_000, 10 / 20_000, 10 / 50_000])
    assert pm["Value"].iloc[-1] == pytest.approx(per100k[1] * 100_000)
    # fewer than 3 valid peers → empty (graceful degradation)
    assert peer_median_series(daily_c, population_df, ["01001"]).empty


def test_poisson_rate_ci():
    # Zero events: exact Poisson upper bound is -ln(0.025) ≈ 3.689 events
    lo, hi = poisson_rate_ci(0, 100_000)
    assert lo == 0.0 and hi == pytest.approx(3.689, abs=0.05)
    # Large counts converge to the normal approximation k ± 1.96·√k
    k, pop = 10_000, 1_000_000
    lo, hi = poisson_rate_ci(k, pop, per=pop)
    assert lo == pytest.approx(k - 1.96 * np.sqrt(k), rel=0.005)
    assert hi == pytest.approx(k + 1.96 * np.sqrt(k), rel=0.005)
    # Small counts: wide relative interval (the small-county lesson)
    lo_s, hi_s = poisson_rate_ci(2, 1_000)
    assert hi_s / max(lo_s, 1e-9) > 10
    # Invalid inputs degrade to NaN, never raise
    assert all(np.isnan(v) for v in poisson_rate_ci(5, 0))
    assert all(np.isnan(v) for v in poisson_rate_ci(None, 1000))
    assert all(np.isnan(v) for v in poisson_rate_ci(np.nan, 1000))


def test_rolling_cfr():
    # 300 days, constant 100 cases/day; deaths = 2% of cases lagged 14 days
    n, lag, window = 300, 14, 56
    dates = pd.date_range("2020-03-01", periods=n)
    daily_cases = np.full(n, 100.0)
    daily_deaths = np.zeros(n)
    daily_deaths[lag:] = 2.0
    out = rolling_cfr_from_daily(daily_cases, daily_deaths, dates,
                                 window_days=window, lag_days=lag)
    steady = out["cfr"].iloc[-50:]
    assert steady.notna().all()
    assert steady.iloc[-1] == pytest.approx(2.0, rel=1e-6)
    # first window+lag days undefined (insufficient window / lag shift)
    assert out["cfr"].iloc[: window + lag - 1].isna().all()

    # min_cases mask: near-zero case windows must not print absurd CFRs
    sparse = rolling_cfr_from_daily(np.full(n, 0.1), np.full(n, 0.05), dates,
                                    window_days=window, lag_days=lag, min_cases=20)
    assert sparse["cfr"].isna().all()


def test_rolling_cfr_county_wrapper(cases_df, deaths_df):
    out = rolling_cfr(cases_df, deaths_df, "Alpha County", "AA",
                      window_days=3, lag_days=1, min_cases=1)
    assert len(out) == len(DATES)
    assert rolling_cfr(cases_df, deaths_df, "Nowhere", "ZZ").empty


def test_find_data_corrections(cases_df):
    # Gamma County's cumulative series dips 30 → 25 at index 5
    corr = find_data_corrections(cases_df, "Gamma County", "BB")
    assert len(corr) == 1
    assert corr["Date"].iloc[0] == pd.Timestamp(DATES[5])
    assert corr["correction"].iloc[0] == -5.0
    # monotone series → no corrections; missing county → empty
    assert find_data_corrections(cases_df, "Alpha County", "AA").empty
    assert find_data_corrections(cases_df, "Nowhere County", "ZZ").empty
