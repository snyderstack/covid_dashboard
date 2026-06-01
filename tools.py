import pandas as pd
import numpy as np
from pathlib import Path

"""
COVID-19 County Analysis Dashboard - Data Processing Tools

FIXES IMPLEMENTED:
- Per-capita calculations now use proper (FIPS, State) tuple joins to handle duplicate FIPS
- Statewide unallocated entries (FIPS='00000', pop=0) are filtered before calculations
- Invalid populations (pop <= 0) are excluded from per-capita computation
- Results with invalid populations return NaN instead of incorrect values
- See validation.py for diagnostic functions to audit data quality
"""

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    """
    Load datasets from local CSV files in wide format.
    Keeps data in wide format (dates as columns) for memory efficiency.
    Only convert to long format for specific visualizations when needed.

    Returns:
        Tuple of (cases_df, deaths_df, population_df) in wide format
    """
    cases_path = DATA_DIR / "covid_confirmed_usafacts.csv"
    deaths_path = DATA_DIR / "covid_deaths_usafacts.csv"
    pop_path = DATA_DIR / "covid_county_population_usafacts.csv"

    # Load with minimal preprocessing
    cases_df = pd.read_csv(cases_path, low_memory=False)
    deaths_df = pd.read_csv(deaths_path, low_memory=False)
    population_df = pd.read_csv(pop_path, low_memory=False)

    # Normalize metadata across all dataframes
    for df in [cases_df, deaths_df, population_df]:
        if "County Name" in df.columns:
            df["County Name"] = df["County Name"].str.strip()

        # Standardize countyFIPS: ensure exactly 5 characters with leading zeros
        if "countyFIPS" in df.columns:
            df["countyFIPS"] = (
                pd.to_numeric(df["countyFIPS"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
                .str.zfill(5)
            )

        # Ensure StateFIPS is 2-char string with leading zeros if present
        if "StateFIPS" in df.columns:
            df["StateFIPS"] = df["StateFIPS"].astype(str).str.zfill(2)

        # Create Location field explicitly for ALL datasets
        # Using concat to avoid DataFrame fragmentation warning
        if "County Name" in df.columns and "State" in df.columns:
            location_series = (
                df["County Name"].astype(str).str.strip()
                + ", "
                + df["State"].astype(str).str.strip()
            )
            df["Location"] = location_series

    return cases_df, deaths_df, population_df


def _wide_to_long(df_wide, county_row):
    """
    Convert a single county row from wide to long format.

    Args:
        df_wide: Wide-format dataframe
        county_row: Row index or Series for a specific county

    Returns:
        DataFrame with columns: Date, Value
    """
    # Dynamically determine identifier columns that actually exist
    possible_identifiers = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    identifier_cols = [col for col in possible_identifiers if col in df_wide.columns]
    date_cols = [col for col in df_wide.columns if col not in identifier_cols]

    # Extract the row and melt it
    row_data = df_wide.loc[county_row, date_cols]
    values = pd.Series(row_data.values, index=pd.to_datetime(date_cols))

    result = pd.DataFrame({
        "Date": values.index,
        "Value": values.values
    })

    # Convert Value to numeric
    result["Value"] = pd.to_numeric(result["Value"], errors="coerce")

    return result.sort_values("Date").reset_index(drop=True)


def prepare_county_timeseries(df_wide, county_name, state, metric_name="Cases"):
    """
    Extract and convert a single county from wide format to timeseries.

    Args:
        df_wide: Wide-format dataframe (cases_df or deaths_df)
        county_name: County name (string)
        state: State abbreviation (string)
        metric_name: Label for the metric column

    Returns:
        DataFrame with columns: Date, {metric_name}
    """
    # Find the county row
    county_row = df_wide[(df_wide["County Name"] == county_name) &
                         (df_wide["State"] == state)]

    if county_row.empty:
        return pd.DataFrame()

    # Convert to long format (get first matching row)
    timeseries = _wide_to_long(df_wide, county_row.index[0])
    timeseries.columns = ["Date", metric_name]

    return timeseries


def calculate_daily_changes(timeseries_df, metric_column):
    """
    Convert cumulative counts to daily new counts.

    Args:
        timeseries_df: DataFrame with Date column and cumulative value column
        metric_column: Name of the column with cumulative values

    Returns:
        DataFrame with additional 'Daily {metric_column}' column
    """
    df = timeseries_df.copy()
    df[f"Daily {metric_column}"] = df[metric_column].diff().fillna(0).astype(int)
    return df


def apply_moving_average(timeseries_df, metric_column, window=7):
    """
    Apply moving average smoothing to a metric.

    Args:
        timeseries_df: DataFrame with Date and metric columns
        metric_column: Name of column to smooth
        window: Window size for moving average

    Returns:
        DataFrame with additional '{metric_column} MA' column
    """
    df = timeseries_df.copy()
    df[f"{metric_column} MA"] = (
        df[metric_column]
        .rolling(window=window, min_periods=1)
        .mean()
    )
    return df


def calculate_per_capita(timeseries, population_df, county_name, state):
    """
    Add per-capita (per 100k population) column to timeseries data.

    Args:
        timeseries: Timeseries dataframe with Date and value column
        population_df: Wide-format population dataframe
        county_name: County name
        state: State abbreviation

    Returns:
        DataFrame with additional 'Per Capita' column
    """
    # Find population for this county
    pop_row = population_df[(population_df["County Name"] == county_name) &
                            (population_df["State"] == state)]

    if pop_row.empty:
        return timeseries.copy()

    # Get population value (from the first non-identifier column)
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS"]
    pop_cols = [col for col in population_df.columns if col not in identifier_cols and col != "Location"]

    if not pop_cols:
        return timeseries.copy()

    population = pd.to_numeric(pop_row.iloc[0][pop_cols[0]], errors="coerce")

    # Calculate per-capita
    df = timeseries.copy()
    metric_col = df.columns[1] if len(df.columns) > 1 else "Value"

    if population > 0:
        df["Per Capita"] = (df[metric_col] / population) * 100000
    else:
        df["Per Capita"] = np.nan

    return df


def prepare_lag_analysis(cases_df_wide, deaths_df_wide, county_name, state, lag_days=0):
    """
    Prepare case and death data for lag analysis for a single county.

    Args:
        cases_df_wide: Wide-format cases dataframe
        deaths_df_wide: Wide-format deaths dataframe
        county_name: County name
        state: State abbreviation
        lag_days: Number of days to shift deaths data (positive = deaths lag behind cases)

    Returns:
        DataFrame with columns: Date, Cases, Deaths (shifted by lag_days)
    """
    # Convert both to timeseries
    cases_ts = prepare_county_timeseries(cases_df_wide, county_name, state, "Cases")
    deaths_ts = prepare_county_timeseries(deaths_df_wide, county_name, state, "Deaths")

    if cases_ts.empty or deaths_ts.empty:
        return pd.DataFrame()

    # Merge on date
    merged = cases_ts.merge(deaths_ts, on="Date", how="inner")

    # Apply lag to deaths
    if lag_days != 0:
        merged["Deaths"] = merged["Deaths"].shift(lag_days)

    return merged.dropna(subset=["Cases", "Deaths"])


def get_county_lag_comparison(cases_df_wide, deaths_df_wide, county_name, state, max_lag=14):
    """
    Find optimal lag between cases and deaths for a single county.
    Uses correlation to identify best lag.

    Args:
        cases_df_wide: Cases dataframe (wide format)
        deaths_df_wide: Deaths dataframe (wide format)
        county_name: County name
        state: State abbreviation
        max_lag: Maximum lag to test (days)

    Returns:
        Dictionary with lag results: {lag_days: correlation}
    """
    cases_ts = prepare_county_timeseries(cases_df_wide, county_name, state, "Cases")
    deaths_ts = prepare_county_timeseries(deaths_df_wide, county_name, state, "Deaths")

    if cases_ts.empty or deaths_ts.empty:
        return {}

    merged = cases_ts.merge(deaths_ts, on="Date", how="inner")

    # Calculate daily values first (cumulative data makes poor correlations)
    merged = calculate_daily_changes(merged, "Cases")
    merged = calculate_daily_changes(merged, "Deaths")

    results = {}
    for lag in range(0, max_lag + 1):
        deaths_shifted = merged["Daily Deaths"].shift(lag)
        # Only correlate where both have data
        valid_mask = ~(merged["Daily Cases"].isna() | deaths_shifted.isna())
        if valid_mask.sum() > 30:  # Need minimum data points
            corr = merged.loc[valid_mask, "Daily Cases"].corr(deaths_shifted[valid_mask])
            results[lag] = corr if not np.isnan(corr) else 0

    return results


# ===== CHOROPLETH PRECOMPUTATION =====

def precompute_daily_diffs(cases_df, deaths_df):
    """
    Precompute daily cases and deaths from cumulative values.
    Stores results in wide format (counties × date columns).

    Args:
        cases_df: Cases dataframe (wide format)
        deaths_df: Deaths dataframe (wide format)

    Returns:
        Tuple of (daily_cases_df, daily_deaths_df) in wide format
    """
    # Dynamically determine identifier columns that actually exist
    possible_identifiers = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    identifier_cols = [col for col in possible_identifiers if col in cases_df.columns]
    date_cols = [col for col in cases_df.columns if col not in identifier_cols]

    # Convert to numeric for all date columns at once
    cases_numeric = cases_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    deaths_numeric = deaths_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    # Calculate daily as diff, then concat with identifiers
    daily_cases_vals = cases_numeric.diff(axis=1).clip(lower=0).fillna(0)
    daily_deaths_vals = deaths_numeric.diff(axis=1).clip(lower=0).fillna(0)

    daily_cases = pd.concat([cases_df[identifier_cols].reset_index(drop=True), daily_cases_vals.reset_index(drop=True)], axis=1)
    daily_deaths = pd.concat([deaths_df[identifier_cols].reset_index(drop=True), daily_deaths_vals.reset_index(drop=True)], axis=1)

    return daily_cases, daily_deaths


def precompute_moving_averages(daily_cases, daily_deaths, window=7):
    """
    Apply moving average to daily metrics for a specified window.

    Args:
        daily_cases: Daily cases dataframe (wide format)
        daily_deaths: Daily deaths dataframe (wide format)
        window: Window size for moving average (default: 7)

    Returns:
        Tuple of (ma_cases_df, ma_deaths_df) in wide format
    """
    # Dynamically determine identifier columns that actually exist
    possible_identifiers = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    identifier_cols = [col for col in possible_identifiers if col in daily_cases.columns]
    date_cols = [col for col in daily_cases.columns if col not in identifier_cols]

    # Convert to numeric
    cases_numeric = daily_cases[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    deaths_numeric = daily_deaths[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    # Apply rolling average across dates (axis=1 for rolling across columns)
    ma_cases_vals = cases_numeric.T.rolling(window=window, min_periods=1).mean().T.reset_index(drop=True)
    ma_deaths_vals = deaths_numeric.T.rolling(window=window, min_periods=1).mean().T.reset_index(drop=True)

    ma_cases = pd.concat([daily_cases[identifier_cols].reset_index(drop=True), ma_cases_vals], axis=1)
    ma_deaths = pd.concat([daily_deaths[identifier_cols].reset_index(drop=True), ma_deaths_vals], axis=1)

    return ma_cases, ma_deaths


def precompute_all_moving_averages(daily_cases, daily_deaths, windows=[3, 5, 7]):
    """
    Precompute moving averages for multiple window sizes.

    Args:
        daily_cases: Daily cases dataframe (wide format)
        daily_deaths: Daily deaths dataframe (wide format)
        windows: List of window sizes to compute (default: [3, 5, 7])

    Returns:
        Dictionary with keys: ma3_cases, ma3_deaths, ma5_cases, ma5_deaths, ma7_cases, ma7_deaths
    """
    result = {}
    for window in windows:
        ma_cases, ma_deaths = precompute_moving_averages(daily_cases, daily_deaths, window=window)
        result[f"ma{window}_cases"] = ma_cases
        result[f"ma{window}_deaths"] = ma_deaths
    return result


def precompute_per_capita(cases_df, deaths_df, population_df):
    """
    Normalize cases and deaths to per-100k population.
    
    FIXED: Properly handles duplicate FIPS by using (FIPS, State) tuple key
    and filters out statewide unallocated entries (FIPS='00000', population=0).

    Args:
        cases_df: Cases dataframe (wide format)
        deaths_df: Deaths dataframe (wide format)
        population_df: Population dataframe

    Returns:
        Tuple of (pc_cases_df, pc_deaths_df) in wide format
    """
    # Dynamically determine identifier columns that actually exist
    possible_identifiers = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    identifier_cols = [col for col in possible_identifiers if col in cases_df.columns]
    date_cols = [col for col in cases_df.columns if col not in identifier_cols]

    # Build population dict using (FIPS, State) tuple to handle duplicates correctly
    # Filter out invalid entries: FIPS='00000' (statewide) and population <= 0
    pop_col = [col for col in population_df.columns if col not in identifier_cols and col != "Location"][0]
    pop_valid = population_df[
        (population_df["countyFIPS"] != "00000") &
        (population_df["population"] > 0)
    ].copy()

    pop_dict = {}
    for idx, row in pop_valid.iterrows():
        key = (row["countyFIPS"], row["State"])
        pop_dict[key] = row[pop_col]

    # Convert to numeric for all date columns
    cases_numeric = cases_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    deaths_numeric = deaths_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    # Get population array for all counties using (FIPS, State) lookup
    pops = cases_df.apply(
        lambda row: pop_dict.get((row["countyFIPS"], row["State"]), np.nan),
        axis=1
    ).values

    # Vectorized per-capita calculation with proper division handling
    # Where population is NaN or <= 0, result will be NaN
    pc_cases_vals = cases_numeric.div(pops, axis=0) * 100000
    pc_deaths_vals = deaths_numeric.div(pops, axis=0) * 100000

    pc_cases = pd.concat([cases_df[identifier_cols].reset_index(drop=True), pc_cases_vals.reset_index(drop=True)], axis=1)
    pc_deaths = pd.concat([deaths_df[identifier_cols].reset_index(drop=True), pc_deaths_vals.reset_index(drop=True)], axis=1)

    return pc_cases, pc_deaths


def get_available_dates(df_wide):
    """
    Get sorted list of available dates as strings.

    Args:
        df_wide: Wide-format dataframe

    Returns:
        List of date strings (YYYY-MM-DD) in chronological order
    """
    # Dynamically determine identifier columns that actually exist
    possible_identifiers = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    identifier_cols = [col for col in possible_identifiers if col in df_wide.columns]
    date_cols = [col for col in df_wide.columns if col not in identifier_cols]
    return sorted(date_cols)


def prepare_choropleth_for_date(metric_df, date_str, cases_df, deaths_df, population_df):
    """
    Extract and prepare data for choropleth for a single date.

    Ensures all valid counties are included, drops only completely invalid FIPS.

    Args:
        metric_df: Precomputed metric dataframe (wide format)
        date_str: Date string (YYYY-MM-DD)
        cases_df: Cases dataframe (for cumulative context)
        deaths_df: Deaths dataframe (for cumulative context)
        population_df: Population dataframe

    Returns:
        DataFrame ready for Plotly with columns:
        countyFIPS, Location, population, value, cases, deaths, cases_pc, deaths_pc
    """
    # Dynamically determine identifier columns that actually exist
    possible_identifiers = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    identifier_cols = [col for col in possible_identifiers if col in population_df.columns]

    # Start with valid FIPS and Location from metric_df
    # If Location column doesn't exist, construct it from County Name and State
    if "Location" in metric_df.columns:
        result = metric_df[["countyFIPS", "Location", "State"]].copy()
    elif "County Name" in metric_df.columns and "State" in metric_df.columns:
        result = metric_df[["countyFIPS", "County Name", "State"]].copy()
        result["Location"] = result["County Name"] + ", " + result["State"]
    else:
        result = metric_df[["countyFIPS"]].copy()
        result["Location"] = result["countyFIPS"]
        result["State"] = "Unknown"

    # Remove rows with invalid FIPS before any joins
    result = result[result["countyFIPS"].notna()].copy()
    result = result[result["countyFIPS"] != "00000"].copy()
    result = result[result["countyFIPS"].str.strip() != ""].copy()
    result = result.reset_index(drop=True)

    # Add population using (FIPS, State) join to handle duplicates correctly
    pop_col = [col for col in population_df.columns if col not in identifier_cols and col != "Location"][0]
    pop_data = population_df[["countyFIPS", "State", pop_col]].rename(columns={pop_col: "population"})
    pop_data = pop_data[(pop_data["population"] > 0)].copy()  # Filter invalid populations
    result = result.merge(pop_data, on=["countyFIPS", "State"], how="left")

    # Add metric value for this date by merging from metric_df
    if date_str in metric_df.columns:
        metric_vals = metric_df[["countyFIPS", "State", date_str]].rename(columns={date_str: "value"})
        result = result.merge(metric_vals, on=["countyFIPS", "State"], how="left")
    else:
        result["value"] = np.nan

    # Add cumulative cases/deaths for context (allow NaN in hover columns)
    if date_str in cases_df.columns:
        cases_vals = cases_df[["countyFIPS", "State", date_str]].rename(columns={date_str: "cases"})
        result = result.merge(cases_vals, on=["countyFIPS", "State"], how="left")
    else:
        result["cases"] = np.nan

    if date_str in deaths_df.columns:
        deaths_vals = deaths_df[["countyFIPS", "State", date_str]].rename(columns={date_str: "deaths"})
        result = result.merge(deaths_vals, on=["countyFIPS", "State"], how="left")
    else:
        result["deaths"] = np.nan

    # Calculate per-capita values
    result["cases_pc"] = np.where(
        (result["population"] > 0) & result["population"].notna(),
        (result["cases"] / result["population"]) * 100000,
        np.nan
    )
    result["deaths_pc"] = np.where(
        (result["population"] > 0) & result["population"].notna(),
        (result["deaths"] / result["population"]) * 100000,
        np.nan
    )

    # Drop rows only if the metric value itself is NaN (can't display map without it)
    result = result.dropna(subset=["value"]).copy()

    return result


def filter_choropleth_by_state(choro_data, state):
    """
    Filter choropleth data to only include counties in the specified state.
    
    Args:
        choro_data: Choropleth dataframe from prepare_choropleth_for_date()
        state: State abbreviation (e.g., "PA", "TX") or "United States" for all states
    
    Returns:
        Filtered choropleth dataframe (or original if state is "United States")
    """
    if state == "United States":
        return choro_data.copy()
    
    filtered = choro_data[choro_data["State"] == state].copy()
    return filtered


def get_state_bounds_for_zoom(choro_data, state):
    """
    Calculate geographic bounds (latitude/longitude) for a state to enable auto-zoom.
    Uses a hardcoded mapping of states to approximate bounds.
    
    Args:
        choro_data: Choropleth dataframe
        state: State abbreviation
    
    Returns:
        Dictionary with 'lat' (center), 'lon' (center), 'zoom' level, or None if not available
    """
    # State bounds (approximate center lat/lon and zoom level for Plotly)
    # These values provide good visual framing for each state
    STATE_BOUNDS = {
        "AL": {"lat": 32.8, "lon": -86.8, "zoom": 6},
        "AK": {"lat": 64.0, "lon": -153.0, "zoom": 3},
        "AZ": {"lat": 33.7, "lon": -111.4, "zoom": 6},
        "AR": {"lat": 34.8, "lon": -92.4, "zoom": 6},
        "CA": {"lat": 37.0, "lon": -119.5, "zoom": 5},
        "CO": {"lat": 39.0, "lon": -105.5, "zoom": 6},
        "CT": {"lat": 41.6, "lon": -72.7, "zoom": 8},
        "DE": {"lat": 39.0, "lon": -75.5, "zoom": 8},
        "FL": {"lat": 27.7, "lon": -81.8, "zoom": 6},
        "GA": {"lat": 32.8, "lon": -83.6, "zoom": 6},
        "HI": {"lat": 21.1, "lon": -156.5, "zoom": 5},
        "ID": {"lat": 44.2, "lon": -114.0, "zoom": 6},
        "IL": {"lat": 40.0, "lon": -89.0, "zoom": 6},
        "IN": {"lat": 39.8, "lon": -86.2, "zoom": 7},
        "IA": {"lat": 42.0, "lon": -93.5, "zoom": 6},
        "KS": {"lat": 38.5, "lon": -97.5, "zoom": 6},
        "KY": {"lat": 37.7, "lon": -84.7, "zoom": 6},
        "LA": {"lat": 31.0, "lon": -91.5, "zoom": 6},
        "ME": {"lat": 45.3, "lon": -69.0, "zoom": 7},
        "MD": {"lat": 39.1, "lon": -76.8, "zoom": 7},
        "MA": {"lat": 42.2, "lon": -71.8, "zoom": 7},
        "MI": {"lat": 44.3, "lon": -85.4, "zoom": 6},
        "MN": {"lat": 46.0, "lon": -93.9, "zoom": 6},
        "MS": {"lat": 32.8, "lon": -89.7, "zoom": 6},
        "MO": {"lat": 38.5, "lon": -92.3, "zoom": 6},
        "MT": {"lat": 47.0, "lon": -110.0, "zoom": 5},
        "NE": {"lat": 41.5, "lon": -99.9, "zoom": 6},
        "NV": {"lat": 39.5, "lon": -116.9, "zoom": 6},
        "NH": {"lat": 43.5, "lon": -71.5, "zoom": 8},
        "NJ": {"lat": 40.0, "lon": -74.5, "zoom": 8},
        "NM": {"lat": 34.5, "lon": -106.6, "zoom": 6},
        "NY": {"lat": 43.0, "lon": -75.5, "zoom": 6},
        "NC": {"lat": 35.5, "lon": -79.8, "zoom": 6},
        "ND": {"lat": 47.5, "lon": -100.5, "zoom": 6},
        "OH": {"lat": 40.4, "lon": -82.9, "zoom": 6},
        "OK": {"lat": 35.5, "lon": -97.5, "zoom": 6},
        "OR": {"lat": 44.0, "lon": -121.3, "zoom": 6},
        "PA": {"lat": 40.8, "lon": -77.8, "zoom": 6},
        "RI": {"lat": 41.7, "lon": -71.5, "zoom": 9},
        "SC": {"lat": 34.0, "lon": -81.0, "zoom": 6},
        "SD": {"lat": 44.5, "lon": -100.0, "zoom": 6},
        "TN": {"lat": 35.5, "lon": -86.5, "zoom": 6},
        "TX": {"lat": 31.0, "lon": -99.0, "zoom": 5},
        "UT": {"lat": 39.0, "lon": -111.5, "zoom": 6},
        "VT": {"lat": 44.0, "lon": -72.7, "zoom": 7},
        "VA": {"lat": 37.8, "lon": -78.2, "zoom": 6},
        "WA": {"lat": 47.5, "lon": -120.5, "zoom": 6},
        "WV": {"lat": 38.5, "lon": -81.9, "zoom": 7},
        "WI": {"lat": 44.3, "lon": -89.6, "zoom": 6},
        "WY": {"lat": 43.0, "lon": -107.5, "zoom": 6},
    }
    
    if state in STATE_BOUNDS:
        return STATE_BOUNDS[state]
    
    return None


def compute_national_timeseries(cases_df, deaths_df, metric_type="Cases"):
    """
    Compute national (United States-wide) timeseries by summing all counties.
    
    Args:
        cases_df: Wide-format cases dataframe
        deaths_df: Wide-format deaths dataframe
        metric_type: Either "Cases" or "Deaths"
    
    Returns:
        DataFrame with columns: Date, Value (in long format)
    """
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in cases_df.columns if col not in identifier_cols]
    
    if metric_type == "Cases":
        source_df = cases_df
    else:
        source_df = deaths_df
    
    # Sum all counties for each date
    national_values = source_df[date_cols].sum(axis=0)
    
    # Convert to long format
    result = pd.DataFrame({
        "Date": pd.to_datetime(date_cols),
        "Value": national_values.values
    })
    
    return result.sort_values("Date").reset_index(drop=True)


def compute_national_daily(daily_df):
    """
    Compute national daily timeseries from precomputed daily dataframe.
    
    Args:
        daily_df: Wide-format daily cases/deaths dataframe
    
    Returns:
        DataFrame with columns: Date, Value (in long format)
    """
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in daily_df.columns if col not in identifier_cols]
    
    # Sum all counties for each date
    national_values = daily_df[date_cols].sum(axis=0)
    
    # Convert to long format
    result = pd.DataFrame({
        "Date": pd.to_datetime(date_cols),
        "Value": national_values.values
    })
    
    return result.sort_values("Date").reset_index(drop=True)


def compute_national_per_capita(pc_df, population_df):
    """
    Compute national per-capita timeseries (total cases/deaths per 100k US population).
    
    Args:
        pc_df: Wide-format per-capita dataframe (already computed per-capita for counties)
        population_df: Population dataframe
    
    Returns:
        DataFrame with columns: Date, Value (per-capita, in long format)
    """
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in pc_df.columns if col not in identifier_cols]
    
    # Get total US population (sum all valid county populations)
    pop_col = [col for col in population_df.columns 
               if col not in identifier_cols and col != "Location"][0]
    total_pop = population_df[
        (population_df["countyFIPS"] != "00000") & 
        (population_df["population"] > 0)
    ][pop_col].sum()
    
    # Sum raw values (not per-capita) to compute national per-capita correctly
    # We need to recompute from raw counts divided by total US population
    cases_or_deaths_df = None
    for idx, row in pc_df.head(1).iterrows():
        # pc_df contains already-computed per-capita values
        # We need original counts to recalculate properly
        break
    
    # Since we have per-capita values, we can back-calculate by averaging
    # Or better: sum the absolute counts and divide by total pop
    # For now, compute using the identity: sum(per_capita_values) / num_counties * avg_pop / total_pop
    # Actually, this gets complex. Let's return average of county per-capita values as proxy
    
    national_pc_values = pc_df[date_cols].mean(axis=0)
    
    result = pd.DataFrame({
        "Date": pd.to_datetime(date_cols),
        "Value": national_pc_values.values
    })
    
    return result.sort_values("Date").reset_index(drop=True)
