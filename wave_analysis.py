"""
Wave and outbreak analysis for COVID-19 county data.

Detects COVID waves (local peaks) in smoothed daily case data and computes
wave metrics for each county.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Tuple


def find_waves(daily_values: np.ndarray, ma_window: int = 7, prominence: float = 1000) -> List[Dict]:
    """
    Detect wave peaks in smoothed daily case data.

    Args:
        daily_values: Array of daily case/death counts (can have NaNs)
        ma_window: Moving average window size for smoothing (3, 5, or 7)
        prominence: Minimum prominence for peak detection (relative height threshold)

    Returns:
        List of wave dicts with keys: peak_index, peak_value, start_index, end_index
    """
    # Remove NaN values but keep track of original indices
    valid_mask = ~np.isnan(daily_values)
    valid_indices = np.where(valid_mask)[0]
    valid_values = daily_values[valid_mask]

    if len(valid_values) < ma_window + 2:
        return []

    # Apply moving average smoothing
    smoothed = np.convolve(valid_values, np.ones(ma_window) / ma_window, mode='valid')

    if len(smoothed) == 0:
        return []

    # Find peaks with minimum prominence
    try:
        peak_indices, properties = find_peaks(smoothed, prominence=prominence)
    except:
        return []

    if len(peak_indices) == 0:
        return []

    # Map back to original indices
    waves = []
    for peak_idx in peak_indices:
        # Adjust for convolution padding
        original_peak_idx = valid_indices[peak_idx + ma_window // 2]
        peak_value = valid_values[peak_idx + ma_window // 2]

        # Find wave start and end (where signal rises/falls below threshold)
        # Use 10% of peak as threshold
        threshold = peak_value * 0.1

        # Find start (working backwards from peak)
        start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if valid_values[i] < threshold:
                start_idx = i
                break

        # Find end (working forwards from peak)
        end_idx = peak_idx
        for i in range(peak_idx + 1, len(valid_values)):
            if valid_values[i] < threshold:
                end_idx = i
                break

        waves.append({
            "peak_index": original_peak_idx,
            "peak_value": peak_value,
            "start_index": valid_indices[start_idx],
            "end_index": valid_indices[end_idx],
        })

    return waves


def calculate_wave_metrics(
    daily_cases: np.ndarray,
    dates: pd.DatetimeIndex,
    ma_window: int = 7,
    prominence: float = 1000
) -> Dict:
    """
    Calculate wave metrics for a county's daily cases.

    Args:
        daily_cases: Array of daily case counts
        dates: DatetimeIndex corresponding to daily_cases
        ma_window: Moving average window (3, 5, or 7 days)
        prominence: Peak detection prominence threshold

    Returns:
        Dictionary with wave metrics:
        - number_of_waves
        - waves (list of wave details)
        - largest_wave (peak cases in highest wave)
        - average_wave_height (mean peak value)
        - average_wave_duration (mean wave length in days)
        - date_of_peak_wave (date of highest peak)
        - total_case_burden (cumulative cases)
    """
    metrics = {
        "number_of_waves": 0,
        "waves": [],
        "largest_wave": 0,
        "average_wave_height": 0,
        "average_wave_duration": 0,
        "date_of_peak_wave": None,
        "total_case_burden": float(np.nansum(daily_cases)),
    }

    # Validate input
    if len(daily_cases) != len(dates):
        return metrics

    # Find waves
    waves = find_waves(daily_cases, ma_window=ma_window, prominence=prominence)

    if len(waves) == 0:
        return metrics

    # Process waves
    metrics["number_of_waves"] = len(waves)
    metrics["waves"] = []

    peak_values = []
    wave_durations = []

    for wave_idx, wave in enumerate(waves, 1):
        peak_date = dates[wave["peak_index"]]
        start_date = dates[wave["start_index"]]
        end_date = dates[wave["end_index"]]
        duration = (end_date - start_date).days

        wave_detail = {
            "wave_number": wave_idx,
            "start_date": start_date,
            "peak_date": peak_date,
            "end_date": end_date,
            "peak_cases": wave["peak_value"],
            "duration_days": duration,
        }
        metrics["waves"].append(wave_detail)

        peak_values.append(wave["peak_value"])
        wave_durations.append(max(duration, 1))  # Avoid division by zero

    # Aggregate metrics
    if peak_values:
        metrics["largest_wave"] = max(peak_values)
        metrics["average_wave_height"] = np.mean(peak_values)

        # Find date of peak wave
        peak_wave_idx = peak_values.index(max(peak_values))
        metrics["date_of_peak_wave"] = metrics["waves"][peak_wave_idx]["peak_date"]

    if wave_durations:
        metrics["average_wave_duration"] = np.mean(wave_durations)

    return metrics


def calculate_waves_for_county(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    daily_cases_df: pd.DataFrame,
    daily_deaths_df: pd.DataFrame,
    county_name: str,
    state: str,
    ma_window: int = 7,
    prominence: float = 1000,
) -> Dict:
    """
    Calculate wave metrics for a specific county.

    Args:
        cases_df: Cases dataframe (wide format, cumulative)
        deaths_df: Deaths dataframe (wide format, cumulative)
        daily_cases_df: Daily cases dataframe (wide format)
        daily_deaths_df: Daily deaths dataframe (wide format)
        county_name: County name
        state: State abbreviation
        ma_window: Moving average window (3, 5, or 7)
        prominence: Peak detection prominence

    Returns:
        Dictionary with case and death wave metrics
    """
    # Find the county row
    cases_row = cases_df[(cases_df["County Name"] == county_name) &
                         (cases_df["State"] == state)]
    deaths_row = deaths_df[(deaths_df["County Name"] == county_name) &
                           (deaths_df["State"] == state)]
    daily_cases_row = daily_cases_df[(daily_cases_df["County Name"] == county_name) &
                                     (daily_cases_df["State"] == state)]
    daily_deaths_row = daily_deaths_df[(daily_deaths_df["County Name"] == county_name) &
                                       (daily_deaths_df["State"] == state)]

    if cases_row.empty or daily_cases_row.empty:
        return {"error": "County not found"}

    # Extract date columns and convert to datetime
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in daily_cases_df.columns if col not in identifier_cols]
    dates = pd.to_datetime(date_cols)

    # Extract values
    daily_cases = pd.to_numeric(daily_cases_row.iloc[0, daily_cases_row.columns.get_loc(date_cols[0]):],
                                errors="coerce").values
    daily_deaths = pd.to_numeric(daily_deaths_row.iloc[0, daily_deaths_row.columns.get_loc(date_cols[0]):],
                                 errors="coerce").values

    # Calculate metrics
    cases_metrics = calculate_wave_metrics(daily_cases, dates, ma_window, prominence)
    deaths_metrics = calculate_wave_metrics(daily_deaths, dates, ma_window, prominence)

    return {
        "county_name": county_name,
        "state": state,
        "cases": cases_metrics,
        "deaths": deaths_metrics,
    }


def calculate_waves_for_all_counties(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    daily_cases_df: pd.DataFrame,
    daily_deaths_df: pd.DataFrame,
    ma_window: int = 7,
    prominence: float = 1000,
) -> pd.DataFrame:
    """
    Calculate wave metrics for all counties.

    Args:
        cases_df: Cases dataframe (wide format)
        deaths_df: Deaths dataframe (wide format)
        daily_cases_df: Daily cases dataframe (wide format)
        daily_deaths_df: Daily deaths dataframe (wide format)
        ma_window: Moving average window (3, 5, or 7)
        prominence: Peak detection prominence

    Returns:
        DataFrame with one row per county and wave metrics columns
    """
    results = []

    for idx, row in cases_df.iterrows():
        county_name = row.get("County Name", "Unknown")
        state = row.get("State", "Unknown")

        metrics = calculate_waves_for_county(
            cases_df, deaths_df, daily_cases_df, daily_deaths_df,
            county_name, state, ma_window, prominence
        )

        if "error" not in metrics:
            result_row = {
                "countyFIPS": row.get("countyFIPS"),
                "County Name": county_name,
                "State": state,
                "case_waves": metrics["cases"]["number_of_waves"],
                "case_largest_wave": metrics["cases"]["largest_wave"],
                "case_avg_wave_height": metrics["cases"]["average_wave_height"],
                "case_avg_wave_duration": metrics["cases"]["average_wave_duration"],
                "case_peak_wave_date": metrics["cases"]["date_of_peak_wave"],
                "case_total_burden": metrics["cases"]["total_case_burden"],
                "death_waves": metrics["deaths"]["number_of_waves"],
                "death_largest_wave": metrics["deaths"]["largest_wave"],
                "death_avg_wave_height": metrics["deaths"]["average_wave_height"],
                "death_avg_wave_duration": metrics["deaths"]["average_wave_duration"],
                "death_peak_wave_date": metrics["deaths"]["date_of_peak_wave"],
                "death_total_burden": metrics["deaths"]["total_case_burden"],
            }
            results.append(result_row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    from tools import load_data, precompute_daily_diffs

    print("Loading data...")
    cases, deaths, pop = load_data()
    daily_cases, daily_deaths = precompute_daily_diffs(cases, deaths)

    print("Testing wave detection for Alameda County, CA...")
    metrics = calculate_waves_for_county(
        cases, deaths, daily_cases, daily_deaths,
        "Alameda County", "CA", ma_window=7
    )

    print(f"\nCases:")
    print(f"  Number of waves: {metrics['cases']['number_of_waves']}")
    print(f"  Largest wave: {metrics['cases']['largest_wave']:.0f}")
    print(f"  Avg wave height: {metrics['cases']['average_wave_height']:.0f}")
    print(f"  Avg duration: {metrics['cases']['average_wave_duration']:.1f} days")
    print(f"  Peak wave date: {metrics['cases']['date_of_peak_wave']}")
    print(f"  Total burden: {metrics['cases']['total_case_burden']:.0f}")
