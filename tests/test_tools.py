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
    get_state_bounds_for_zoom,
    monthly_snapshot_long,
    precompute_daily_diffs,
    precompute_per_capita,
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
