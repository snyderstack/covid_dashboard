"""
Epidemiological lag analysis for COVID-19 county data.

Peaks are the wave peaks found by the Wave Analysis detector
(wave_analysis.find_waves, region-based with a sensitivity preset) run on
daily new cases and deaths per 100k — so the Time Lag and Wave Analysis tabs
report the same waves. Each case-wave peak is paired with the nearest
death-wave peak (up to MAX_LEAD_DAYS before it, or max_lag_days after it),
and the lag (in days) between them is computed.

A single county is analyzed via `analyze_county_lag(...)`, which returns a
self-contained result dict; comparing two counties means calling it twice.
"""

import numpy as np
import pandas as pd

from tools import (
    prepare_county_timeseries,
    calculate_daily_changes,
    get_population_column,
)
from wave_analysis import (
    DEATH_MA_WINDOW,
    calculate_wave_metrics,
    estimate_optimal_smoothing,
    smooth_series,
)


def get_county_population(population_df, county_name, state, fips=None):
    """
    Look up a county's population, returning NaN if not found or invalid.

    When fips is given the row is matched on (countyFIPS, State): county names
    differ between the USAFacts cases and population files for some counties.
    """
    if fips is not None:
        pop_row = population_df[
            (population_df["countyFIPS"] == fips) &
            (population_df["State"] == state)
        ]
    else:
        pop_row = population_df[
            (population_df["County Name"] == county_name) &
            (population_df["State"] == state)
        ]
    if pop_row.empty:
        return np.nan

    pop_col = get_population_column(population_df)
    if pop_col is None:
        return np.nan

    population = pd.to_numeric(pop_row.iloc[0][pop_col], errors="coerce")
    if pd.isna(population) or population <= 0:
        return np.nan

    return population


def prepare_daily_per_capita(df_wide, population_df, county_name, state, metric_name, ma_window=7,
                             fips=None):
    """
    Build a daily, per-100k, smoothed series for one county.

    Pipeline:
        1. prepare_county_timeseries  -> cumulative {metric_name}
        2. calculate_daily_changes    -> "Daily {metric_name}" (new cases/deaths per day)
        3. divide by (population / 100,000) -> per-100k daily rate
        4. centered moving average (ma_window) -> smoothed per-100k rate, the
           same series the wave detector runs on (wave_analysis.smooth_series).
           The incomplete-window days at each end are NaN so charts stop there
           instead of dropping to zero.

    Returns a DataFrame with columns:
        Date, Daily {metric_name}, Per100k, Per100k MA
    or an empty DataFrame if county data is unavailable.
    """
    ts = prepare_county_timeseries(df_wide, county_name, state, metric_name)
    if ts.empty:
        return pd.DataFrame()

    ts = calculate_daily_changes(ts, metric_name)
    daily_col = f"Daily {metric_name}"

    # Clip any residual negatives (calculate_daily_changes now clips, but be
    # explicit here to guard against future changes to that function)
    ts[daily_col] = ts[daily_col].clip(lower=0)

    population = get_county_population(population_df, county_name, state, fips=fips)

    if pd.isna(population):
        ts["Per100k"] = np.nan
        ts["Per100k MA"] = np.nan
    else:
        ts["Per100k"] = (ts[daily_col] / population) * 100_000
        smoothed = smooth_series(ts["Per100k"].values, ma_window)
        half = ma_window // 2
        if half:
            smoothed[:half] = np.nan
            smoothed[len(smoothed) - half:] = np.nan
        ts["Per100k MA"] = smoothed

    return ts[["Date", daily_col, "Per100k", "Per100k MA"]]


def _wave_peaks(ts, wave_metrics):
    """
    Convert calculate_wave_metrics() output to peak dicts for matching.

    peak_value is the smoothed per-100k rate on the peak date (the value the
    chart shows), not the single-day raw count.
    """
    smoothed = ts.set_index("Date")["Per100k MA"]
    peaks = []
    for w in wave_metrics["waves"]:
        peak_date = pd.Timestamp(w["peak_date"])
        peaks.append({
            "peak_date":         peak_date,
            "peak_value":        float(smoothed.get(peak_date, np.nan)),
            "wave_start":        pd.Timestamp(w["start_date"]),
            "wave_end":          pd.Timestamp(w["end_date"]),
            "wave_significance": w.get("wave_significance", np.nan),
        })
    return peaks


# A death peak may precede its case peak by up to this many days. Both peaks
# are read off smoothed curves (7-day cases, 21-day deaths) of the same broad
# outbreak, and the smoothed death peak can land a week or two before the
# case peak (e.g. a case wave whose reported peak is pushed late by holiday
# reporting). Requiring death ≥ case left such outbreaks with no pair at all.
MAX_LEAD_DAYS = 21


def pair_case_death_waves(case_peaks, death_peaks, max_lag_days=90,
                          max_lead_days=MAX_LEAD_DAYS):
    """
    Pair case-wave peaks with death-wave peaks.

    A death peak is eligible for a case peak when
        -max_lead_days <= lag <= max_lag_days   (lag = death − case, days)
    and it does not fall before the case wave's start ("wave_start" or
    "start_date", when present). Case peaks are processed chronologically;
    each takes the nearest eligible unused death peak (smallest |lag|,
    ties to the later death peak). Each death peak is used at most once.

    Returns a list of (case_index, death_index, lag_days) using the indices
    of the input lists, in case-peak chronological order.
    """
    def _start(p):
        s = p.get("wave_start", p.get("start_date"))
        return pd.Timestamp(s) if s is not None else None

    case_order = sorted(range(len(case_peaks)),
                        key=lambda i: pd.Timestamp(case_peaks[i]["peak_date"]))
    used = set()
    pairs = []
    for i in case_order:
        cp_date = pd.Timestamp(case_peaks[i]["peak_date"])
        cp_start = _start(case_peaks[i])
        best = None
        for j, dp in enumerate(death_peaks):
            if j in used:
                continue
            dp_date = pd.Timestamp(dp["peak_date"])
            lag = (dp_date - cp_date).days
            if not (-max_lead_days <= lag <= max_lag_days):
                continue
            if cp_start is not None and dp_date < cp_start:
                continue
            key = (abs(lag), -lag)
            if best is None or key < best[0]:
                best = (key, j, lag)
        if best is not None:
            used.add(best[1])
            pairs.append((i, best[1], best[2]))
    return pairs


def match_case_death_peaks(case_peaks, death_peaks, max_lag_days=90,
                           max_lead_days=MAX_LEAD_DAYS):
    """
    Match case peaks to death peaks (see pair_case_death_waves for the rule).

    Args:
        case_peaks: list of peak dicts (peak_date, peak_value) for the cases series
        death_peaks: list of peak dicts (peak_date, peak_value) for the deaths series
        max_lag_days: latest a death peak may follow its case peak (days)
        max_lead_days: earliest a death peak may precede its case peak (days)

    Returns:
        DataFrame with columns:
            case_peak_date, death_peak_date, lag_days,
            case_peak_value, death_peak_value
        sorted chronologically by case_peak_date. lag_days is negative when
        the smoothed death peak came before the case peak.
    """
    matches = [
        {
            "case_peak_date":   case_peaks[i]["peak_date"],
            "death_peak_date":  death_peaks[j]["peak_date"],
            "lag_days":         lag,
            "case_peak_value":  case_peaks[i]["peak_value"],
            "death_peak_value": death_peaks[j]["peak_value"],
        }
        for i, j, lag in pair_case_death_waves(
            case_peaks, death_peaks, max_lag_days, max_lead_days,
        )
    ]
    matches_df = pd.DataFrame(
        matches,
        columns=[
            "case_peak_date", "death_peak_date", "lag_days",
            "case_peak_value", "death_peak_value",
        ],
    )
    if not matches_df.empty:
        matches_df = matches_df.sort_values("case_peak_date").reset_index(drop=True)
    return matches_df


def analyze_county_lag(
    cases_df,
    deaths_df,
    population_df,
    county_name,
    state,
    sensitivity="standard",
    max_lag_days=90,
    fips=None,
):
    """
    Run the full case-to-death lag analysis pipeline for a single county.

    Args:
        cases_df, deaths_df: wide-format cumulative dataframes
        population_df: wide-format population dataframe
        county_name, state: identify the county
        sensitivity: wave-detection preset ("conservative" | "standard" |
                     "sensitive"), as on the Wave Analysis tab
        max_lag_days: maximum days between a case peak and its matched death peak
        fips: optional 5-char FIPS for the population lookup (see
              get_county_population)

    Case smoothing is chosen by estimate_optimal_smoothing (the Wave Analysis
    tab's auto setting); deaths use DEATH_MA_WINDOW.

    Returns:
        Dict with keys:
            cases_ts, deaths_ts     -- per-100k daily timeseries (with MA)
            case_peaks, death_peaks -- lists of wave-peak dicts
            matches                 -- DataFrame of matched case/death peak pairs
            population              -- county population used for normalization
            case_ma_window, death_ma_window, sensitivity, max_lag_days
        or {"error": "..."} if data is unavailable.
    """
    population = get_county_population(population_df, county_name, state, fips=fips)
    if pd.isna(population):
        return {"error": "No valid population data available for this county."}

    raw_cases = prepare_daily_per_capita(
        cases_df, population_df, county_name, state, "Cases", ma_window=1, fips=fips,
    )
    if raw_cases.empty:
        return {"error": "No case/death timeseries available for this county."}
    case_ma_window = estimate_optimal_smoothing(raw_cases["Per100k"].values)

    cases_ts = prepare_daily_per_capita(
        cases_df, population_df, county_name, state, "Cases",
        ma_window=case_ma_window, fips=fips,
    )
    deaths_ts = prepare_daily_per_capita(
        deaths_df, population_df, county_name, state, "Deaths",
        ma_window=DEATH_MA_WINDOW, fips=fips,
    )
    if cases_ts.empty or deaths_ts.empty:
        return {"error": "No case/death timeseries available for this county."}

    case_waves = calculate_wave_metrics(
        cases_ts["Per100k"].values, pd.DatetimeIndex(cases_ts["Date"]),
        ma_window=case_ma_window, sensitivity=sensitivity,
        count_unit=100_000 / population,
    )
    death_waves = calculate_wave_metrics(
        deaths_ts["Per100k"].values, pd.DatetimeIndex(deaths_ts["Date"]),
        ma_window=DEATH_MA_WINDOW, sensitivity=sensitivity,
        series="deaths", count_unit=100_000 / population,
    )
    case_peaks  = _wave_peaks(cases_ts, case_waves)
    death_peaks = _wave_peaks(deaths_ts, death_waves)

    matches = match_case_death_peaks(case_peaks, death_peaks, max_lag_days=max_lag_days)

    return {
        "cases_ts": cases_ts,
        "deaths_ts": deaths_ts,
        "case_peaks": case_peaks,
        "death_peaks": death_peaks,
        "matches": matches,
        "population": population,
        "case_ma_window": case_ma_window,
        "death_ma_window": DEATH_MA_WINDOW,
        "sensitivity": sensitivity,
        "max_lag_days": max_lag_days,
    }


def summarize_lag_results(results):
    """
    Compute summary statistics from an analyze_county_lag() result.

    Returns a dict with:
        avg_lag, median_lag, min_lag, max_lag, n_matched,
        largest_case_peak, largest_death_peak
    or None values where not computable.
    """
    summary = {
        "avg_lag": np.nan,
        "median_lag": np.nan,
        "min_lag": np.nan,
        "max_lag": np.nan,
        "n_matched": 0,
        "largest_case_peak": np.nan,
        "largest_death_peak": np.nan,
        "mean_severity_ratio": np.nan,
    }

    if "error" in results:
        return summary

    matches = results["matches"]
    if not matches.empty:
        summary["avg_lag"] = matches["lag_days"].mean()
        summary["median_lag"] = matches["lag_days"].median()
        summary["min_lag"] = matches["lag_days"].min()
        summary["max_lag"] = matches["lag_days"].max()
        summary["n_matched"] = len(matches)

        # Severity ratio: how large the death peak is relative to the case peak
        # that caused it.  Captures how efficiently cases translated into
        # mortality — small ratios suggest better outcomes relative to case burden.
        ratios = (
            matches["death_peak_value"]
            / matches["case_peak_value"].replace(0, np.nan)
        ).dropna()
        if len(ratios) > 0:
            summary["mean_severity_ratio"] = float(ratios.mean())

    if results["case_peaks"]:
        summary["largest_case_peak"] = max(p["peak_value"] for p in results["case_peaks"])

    if results["death_peaks"]:
        summary["largest_death_peak"] = max(p["peak_value"] for p in results["death_peaks"])

    return summary
