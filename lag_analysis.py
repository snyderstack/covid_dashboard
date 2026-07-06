"""
Epidemiological lag analysis for COVID-19 county data.

Identifies peaks in daily new cases per 100k and daily new deaths per 100k,
matches each case peak to the nearest subsequent death peak, and computes
the lag (in days) between them.

This module is structured so a single county can be analyzed via
`analyze_county_lag(...)`, returning a self-contained result dict. Comparing
two counties (a future feature) simply means calling this function twice —
no refactoring required.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from tools import (
    prepare_county_timeseries,
    calculate_daily_changes,
    get_population_column,
)


def get_county_population(population_df, county_name, state):
    """Look up a county's population, returning NaN if not found or invalid."""
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


def prepare_daily_per_capita(df_wide, population_df, county_name, state, metric_name, ma_window=7):
    """
    Build a daily, per-100k, moving-average-smoothed series for one county.

    Pipeline:
        1. prepare_county_timeseries  -> cumulative {metric_name}
        2. calculate_daily_changes    -> "Daily {metric_name}" (new cases/deaths per day)
        3. divide by (population / 100,000) -> per-100k daily rate
        4. rolling mean (ma_window)   -> smoothed per-100k rate

    Returns a DataFrame with columns:
        Date, Daily {metric_name}, Per100k, Per100k MA
    or an empty DataFrame if county data / population is unavailable.
    """
    ts = prepare_county_timeseries(df_wide, county_name, state, metric_name)
    if ts.empty:
        return pd.DataFrame()

    ts = calculate_daily_changes(ts, metric_name)
    daily_col = f"Daily {metric_name}"

    # Clip any residual negatives (calculate_daily_changes now clips, but be
    # explicit here to guard against future changes to that function)
    ts[daily_col] = ts[daily_col].clip(lower=0)

    population = get_county_population(population_df, county_name, state)

    if pd.isna(population):
        ts["Per100k"] = np.nan
    else:
        ts["Per100k"] = (ts[daily_col] / population) * 100_000

    ts["Per100k MA"] = (
        ts["Per100k"].rolling(window=ma_window, min_periods=1).mean()
    )

    return ts[["Date", daily_col, "Per100k", "Per100k MA"]]


def detect_peaks(values, dates, prominence=1.0, min_distance=7):
    """
    Detect meaningful peaks in a smoothed per-100k series.

    Args:
        values: 1D array of (smoothed) per-100k values, may contain NaN
        dates: corresponding array/Series of dates (same length as values)
        prominence: minimum prominence (in per-100k units) for a peak to count
        min_distance: minimum number of days between detected peaks

    Returns:
        List of dicts: {peak_date, peak_value, peak_prominence}
    """
    values = np.asarray(values, dtype=float)
    dates = np.asarray(dates)

    if len(values) < 3 or np.all(np.isnan(values)):
        return []

    # find_peaks cannot handle NaN; treat missing data as zero (no signal)
    clean_values = np.nan_to_num(values, nan=0.0)

    distance = max(1, int(min_distance))

    peak_indices, properties = find_peaks(
        clean_values,
        prominence=prominence,
        distance=distance,
    )

    peaks = []
    for i, idx in enumerate(peak_indices):
        peaks.append({
            "peak_date": pd.Timestamp(dates[idx]),
            "peak_value": float(clean_values[idx]),
            "peak_prominence": float(properties["prominences"][i]),
        })

    return peaks


def match_case_death_peaks(case_peaks, death_peaks, max_lag_days=90):
    """
    Match each case peak to the nearest death peak occurring on or after it.

    Each death peak can be matched to at most one case peak (the closest
    preceding case peak claims it first, processing case peaks in
    chronological order).

    Args:
        case_peaks: list of dicts from detect_peaks() for the cases series
        death_peaks: list of dicts from detect_peaks() for the deaths series
        max_lag_days: maximum allowed gap (days) between a case peak and
                      its matched death peak

    Returns:
        DataFrame with columns:
            case_peak_date, death_peak_date, lag_days,
            case_peak_value, death_peak_value
        sorted chronologically by case_peak_date.
    """
    case_peaks_sorted = sorted(case_peaks, key=lambda p: p["peak_date"])
    death_peaks_sorted = sorted(death_peaks, key=lambda p: p["peak_date"])

    used_death_idx = set()
    matches = []

    for cp in case_peaks_sorted:
        best_idx = None
        best_lag = None

        for j, dp in enumerate(death_peaks_sorted):
            if j in used_death_idx:
                continue

            lag = (dp["peak_date"] - cp["peak_date"]).days

            # Death peak must occur on/after the case peak, within the window
            if 0 <= lag <= max_lag_days:
                if best_lag is None or lag < best_lag:
                    best_lag = lag
                    best_idx = j

        if best_idx is not None:
            dp = death_peaks_sorted[best_idx]
            used_death_idx.add(best_idx)
            matches.append({
                "case_peak_date": cp["peak_date"],
                "death_peak_date": dp["peak_date"],
                "lag_days": best_lag,
                "case_peak_value": cp["peak_value"],
                "death_peak_value": dp["peak_value"],
            })

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
    ma_window=7,
    case_prominence=1.0,
    death_prominence=0.05,
    max_lag_days=90,
    min_peak_distance_days=14,
):
    """
    Run the full case-to-death lag analysis pipeline for a single county.

    Args:
        cases_df, deaths_df: wide-format cumulative dataframes
        population_df: wide-format population dataframe
        county_name, state: identify the county
        ma_window: smoothing window (3, 5, or 7 days) applied to per-100k rates
        case_prominence: minimum prominence (per 100k) for a case peak
        death_prominence: minimum prominence (per 100k) for a death peak
        max_lag_days: maximum days between a case peak and its matched death peak
        min_peak_distance_days: minimum spacing between consecutive peaks
                                 in the same series (suppresses noisy peaks)

    Returns:
        Dict with keys:
            cases_ts, deaths_ts   -- per-100k daily timeseries (with MA)
            case_peaks, death_peaks -- lists of detected peak dicts
            matches               -- DataFrame of matched case/death peak pairs
            population            -- county population used for normalization
        or {"error": "..."} if data is unavailable.
    """
    cases_ts = prepare_daily_per_capita(
        cases_df, population_df, county_name, state, "Cases", ma_window=ma_window
    )
    deaths_ts = prepare_daily_per_capita(
        deaths_df, population_df, county_name, state, "Deaths", ma_window=ma_window
    )

    if cases_ts.empty or deaths_ts.empty:
        return {"error": "No case/death timeseries available for this county."}

    population = get_county_population(population_df, county_name, state)
    if pd.isna(population):
        return {"error": "No valid population data available for this county."}

    case_peaks = detect_peaks(
        cases_ts["Per100k MA"].values,
        cases_ts["Date"].values,
        prominence=case_prominence,
        min_distance=min_peak_distance_days,
    )
    death_peaks = detect_peaks(
        deaths_ts["Per100k MA"].values,
        deaths_ts["Date"].values,
        prominence=death_prominence,
        min_distance=min_peak_distance_days,
    )

    matches = match_case_death_peaks(case_peaks, death_peaks, max_lag_days=max_lag_days)

    return {
        "cases_ts": cases_ts,
        "deaths_ts": deaths_ts,
        "case_peaks": case_peaks,
        "death_peaks": death_peaks,
        "matches": matches,
        "population": population,
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
