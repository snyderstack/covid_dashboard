"""
COVID-19 County Analysis Dashboard — data processing and choropleth utilities.

All DataFrames stay in wide format (counties × date columns) to keep memory
overhead low.  Per-capita joins use (countyFIPS, State) tuple keys to avoid
duplicate-FIPS collisions from statewide-unallocated rows (FIPS == '00000').

See validation.py for diagnostic functions that audit data quality.
"""

import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
METADATA_COLUMNS = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]

GEOJSON_FILENAME = "geojson-counties-fips.json"
GEOJSON_CDN_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)


def load_county_geojson(data_dir=None):
    """
    Return the US county GeoJSON as a dict, or None if unavailable.

    Loads the bundled copy from data/ when present. If missing, downloads it
    once from the Plotly datasets CDN and saves it locally so subsequent runs
    (and the spatial-analysis features that need geometry) work offline.

    Callers should fall back to passing GEOJSON_CDN_URL directly to Plotly
    when this returns None.
    """
    path = (data_dir or DATA_DIR) / GEOJSON_FILENAME
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(f"Could not read bundled county GeoJSON: {exc}")
            return None

    try:
        from urllib.request import urlopen

        with urlopen(GEOJSON_CDN_URL, timeout=30) as resp:
            raw = resp.read()
        geo = json.loads(raw)
        try:
            with open(path, "wb") as f:
                f.write(raw)
        except OSError:
            pass  # read-only data dir — still return the in-memory copy
        return geo
    except Exception as exc:
        warnings.warn(f"County GeoJSON unavailable ({exc}); falling back to CDN URL")
        return None


def get_identifier_columns(df):
    """Return metadata columns present in a dataframe."""
    return [col for col in METADATA_COLUMNS if col in df.columns]


def get_date_columns(df):
    """Return date columns from a wide-format USAFacts dataframe."""
    return [col for col in df.columns if col not in METADATA_COLUMNS]


def get_population_column(population_df):
    """Return the population value column from the population dataframe."""
    metadata_cols = set(METADATA_COLUMNS)
    pop_cols = [col for col in population_df.columns if col not in metadata_cols]
    return pop_cols[0] if pop_cols else None


def extract_county_state(location: str) -> tuple:
    """
    Split a "County Name, ST" location string into (county_name, state_abbr).

    Returns (location, None) if no comma separator is found.
    Used consistently across all UI tabs to avoid duplicated inline parsing.

    Args:
        location: Location string in "County Name, State" format.

    Returns:
        Tuple of (county_name, state_abbr). state_abbr is None if unparseable.
    """
    if ", " in location:
        county, state = location.rsplit(", ", 1)
        return county, state
    return location, None


def normalize_dataset_metadata(df):
    """Normalize county metadata while preserving the dataframe's wide date columns."""
    df = df.copy()

    if "County Name" in df.columns:
        df["County Name"] = df["County Name"].astype(str).str.strip()

    if "countyFIPS" in df.columns:
        df["countyFIPS"] = (
            pd.to_numeric(df["countyFIPS"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )

    if "StateFIPS" in df.columns:
        df["StateFIPS"] = (
            pd.to_numeric(df["StateFIPS"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(2)
        )

    if "County Name" in df.columns and "State" in df.columns:
        location_series = (
            df["County Name"].astype(str).str.strip()
            + ", "
            + df["State"].astype(str).str.strip()
        )
        df = df.assign(Location=location_series)

    return df


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

    cases_df = pd.read_csv(cases_path, low_memory=False)
    deaths_df = pd.read_csv(deaths_path, low_memory=False)
    population_df = pd.read_csv(pop_path, low_memory=False)

    cases_df = normalize_dataset_metadata(cases_df)
    deaths_df = normalize_dataset_metadata(deaths_df)
    population_df = normalize_dataset_metadata(population_df)

    return cases_df, deaths_df, population_df


def _wide_to_long(df_wide, county_row):
    """
    Convert a single county row from wide to long format.

    Args:
        df_wide: Wide-format dataframe
        county_row: Row index for a specific county

    Returns:
        DataFrame with columns: Date, Value
    """
    date_cols = get_date_columns(df_wide)
    row_data = df_wide.loc[county_row, date_cols]
    values = pd.Series(row_data.values, index=pd.to_datetime(date_cols))

    result = pd.DataFrame({
        "Date": values.index,
        "Value": pd.to_numeric(values.values, errors="coerce"),
    })
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
    county_row = df_wide[
        (df_wide["County Name"] == county_name) & (df_wide["State"] == state)
    ]

    if county_row.empty:
        return pd.DataFrame()

    timeseries = _wide_to_long(df_wide, county_row.index[0])
    timeseries.columns = ["Date", metric_name]
    return timeseries


def calculate_daily_changes(timeseries_df, metric_column):
    """
    Convert cumulative counts to daily new counts.

    Negative diffs (from data corrections / backfills) are clipped to zero,
    consistent with precompute_daily_diffs(). The result is kept as float
    to preserve NaN semantics from the source data.

    Args:
        timeseries_df: DataFrame with Date column and cumulative value column
        metric_column: Name of the column with cumulative values

    Returns:
        DataFrame with additional 'Daily {metric_column}' column
    """
    df = timeseries_df.copy()
    df[f"Daily {metric_column}"] = (
        df[metric_column].diff().clip(lower=0).fillna(0)
    )
    return df


def apply_moving_average(timeseries_df, metric_column, window=7):
    """
    Apply moving average smoothing to a metric.

    Note: min_periods=1 means the first (window-1) values are averaged over
    fewer than window observations. These early-period values should be
    interpreted with caution in research contexts.

    Args:
        timeseries_df: DataFrame with Date and metric columns
        metric_column: Name of column to smooth
        window: Window size for moving average

    Returns:
        DataFrame with additional '{metric_column} MA' column
    """
    df = timeseries_df.copy()
    df[f"{metric_column} MA"] = (
        df[metric_column].rolling(window=window, min_periods=1).mean()
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
    pop_row = population_df[
        (population_df["County Name"] == county_name) &
        (population_df["State"] == state)
    ]

    if pop_row.empty:
        return timeseries.copy()

    pop_col = get_population_column(population_df)
    if pop_col is None:
        return timeseries.copy()

    population = pd.to_numeric(pop_row.iloc[0][pop_col], errors="coerce")

    df = timeseries.copy()
    metric_col = df.columns[1] if len(df.columns) > 1 else "Value"

    if pd.notna(population) and population > 0:
        df["Per Capita"] = (df[metric_col] / population) * 100_000
    else:
        df["Per Capita"] = np.nan

    return df


def precompute_daily_diffs(cases_df, deaths_df):
    """
    Precompute daily cases and deaths from cumulative values.
    Stores results in wide format (counties × date columns).

    Negative diffs are clipped to zero to protect downstream analysis from
    source-data backfills or reporting corrections.

    Args:
        cases_df: Cases dataframe (wide format)
        deaths_df: Deaths dataframe (wide format)

    Returns:
        Tuple of (daily_cases_df, daily_deaths_df) in wide format
    """
    identifier_cols = get_identifier_columns(cases_df)
    date_cols = get_date_columns(cases_df)

    cases_numeric = cases_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    deaths_numeric = deaths_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    daily_cases_vals = cases_numeric.diff(axis=1).clip(lower=0).fillna(0)
    daily_deaths_vals = deaths_numeric.diff(axis=1).clip(lower=0).fillna(0)

    daily_cases = pd.concat(
        [cases_df[identifier_cols].reset_index(drop=True),
         daily_cases_vals.reset_index(drop=True)],
        axis=1
    )
    daily_deaths = pd.concat(
        [deaths_df[identifier_cols].reset_index(drop=True),
         daily_deaths_vals.reset_index(drop=True)],
        axis=1
    )

    return daily_cases, daily_deaths


def precompute_moving_averages(daily_cases, daily_deaths, window=7):
    """
    Apply moving average to daily metrics for a specified window.

    Rolling is applied across dates (axis=1 in the transposed form) for all
    counties at once. Each transpose creates a full copy; acceptable at startup
    but should be reconsidered if date columns grow significantly.

    Note: min_periods=1 means early dates (first window-1 values) are averaged
    over fewer observations.

    Args:
        daily_cases: Daily cases dataframe (wide format)
        daily_deaths: Daily deaths dataframe (wide format)
        window: Window size for moving average (default: 7)

    Returns:
        Tuple of (ma_cases_df, ma_deaths_df) in wide format
    """
    identifier_cols = get_identifier_columns(daily_cases)
    date_cols = get_date_columns(daily_cases)

    cases_numeric = daily_cases[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    deaths_numeric = daily_deaths[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    ma_cases_vals = cases_numeric.T.rolling(window=window, min_periods=1).mean().T.reset_index(drop=True)
    ma_deaths_vals = deaths_numeric.T.rolling(window=window, min_periods=1).mean().T.reset_index(drop=True)

    ma_cases = pd.concat(
        [daily_cases[identifier_cols].reset_index(drop=True), ma_cases_vals], axis=1
    )
    ma_deaths = pd.concat(
        [daily_deaths[identifier_cols].reset_index(drop=True), ma_deaths_vals], axis=1
    )

    return ma_cases, ma_deaths


def precompute_all_moving_averages(daily_cases, daily_deaths, windows=None):
    """
    Precompute moving averages for multiple window sizes.

    Args:
        daily_cases: Daily cases dataframe (wide format)
        daily_deaths: Daily deaths dataframe (wide format)
        windows: List of window sizes to compute (default: [3, 5, 7])

    Returns:
        Dictionary with keys: ma3_cases, ma3_deaths, ma5_cases, ma5_deaths, ma7_cases, ma7_deaths
    """
    if windows is None:
        windows = [3, 5, 7]

    result = {}
    for window in windows:
        ma_cases, ma_deaths = precompute_moving_averages(daily_cases, daily_deaths, window=window)
        result[f"ma{window}_cases"] = ma_cases
        result[f"ma{window}_deaths"] = ma_deaths
    return result


def precompute_per_capita(cases_df, deaths_df, population_df):
    """
    Normalize cases and deaths to per-100k population.

    Uses a vectorized merge on (countyFIPS, State) rather than row-wise apply,
    which is faster and idiomatic. Statewide unallocated entries (FIPS='00000')
    and counties with non-positive populations are excluded from the lookup;
    those rows receive NaN per-capita values.

    Args:
        cases_df: Cases dataframe (wide format)
        deaths_df: Deaths dataframe (wide format)
        population_df: Population dataframe

    Returns:
        Tuple of (pc_cases_df, pc_deaths_df) in wide format
    """
    identifier_cols = get_identifier_columns(cases_df)
    date_cols = get_date_columns(cases_df)

    pop_col = get_population_column(population_df)
    if pop_col is None:
        raise ValueError("Population dataframe does not contain a population column")

    # Exclude statewide rows and zero-pop entries from the population lookup
    pop_valid = population_df[
        (population_df["countyFIPS"] != "00000") &
        (population_df[pop_col] > 0)
    ][["countyFIPS", "State", pop_col]].rename(columns={pop_col: "_pop"})

    merged_pop = (
        cases_df[["countyFIPS", "State"]]
        .merge(pop_valid, on=["countyFIPS", "State"], how="left")
    )
    pops = merged_pop["_pop"].values  # NaN where no valid population exists

    cases_numeric = cases_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    deaths_numeric = deaths_df[date_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))

    pc_cases_vals = cases_numeric.div(pops, axis=0) * 100_000
    pc_deaths_vals = deaths_numeric.div(pops, axis=0) * 100_000

    pc_cases = pd.concat(
        [cases_df[identifier_cols].reset_index(drop=True),
         pc_cases_vals.reset_index(drop=True)],
        axis=1
    )
    pc_deaths = pd.concat(
        [deaths_df[identifier_cols].reset_index(drop=True),
         pc_deaths_vals.reset_index(drop=True)],
        axis=1
    )

    return pc_cases, pc_deaths


def get_available_dates(df_wide):
    """
    Get sorted list of available dates as strings.

    Args:
        df_wide: Wide-format dataframe

    Returns:
        List of date strings (YYYY-MM-DD) in chronological order
    """
    return sorted(get_date_columns(df_wide))


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
        countyFIPS, Location, State, population, value, cases, deaths, cases_pc, deaths_pc
    """
    if "Location" in metric_df.columns:
        result = metric_df[["countyFIPS", "Location", "State"]].copy()
    elif "County Name" in metric_df.columns and "State" in metric_df.columns:
        result = metric_df[["countyFIPS", "County Name", "State"]].copy()
        result["Location"] = result["County Name"] + ", " + result["State"]
    else:
        result = metric_df[["countyFIPS"]].copy()
        result["Location"] = result["countyFIPS"]
        result["State"] = "Unknown"

    result = result[result["countyFIPS"].notna()].copy()
    result = result[result["countyFIPS"] != "00000"].copy()
    result = result[result["countyFIPS"].str.strip() != ""].copy()
    result = result.reset_index(drop=True)

    pop_col = get_population_column(population_df)
    if pop_col is None:
        raise ValueError("Population dataframe does not contain a population column")
    pop_data = (
        population_df[["countyFIPS", "State", pop_col]]
        .rename(columns={pop_col: "population"})
    )
    pop_data = pop_data[pop_data["population"] > 0].copy()
    result = result.merge(pop_data, on=["countyFIPS", "State"], how="left")

    if date_str in metric_df.columns:
        metric_vals = (
            metric_df[["countyFIPS", "State", date_str]]
            .rename(columns={date_str: "value"})
        )
        result = result.merge(metric_vals, on=["countyFIPS", "State"], how="left")
    else:
        result["value"] = np.nan

    # Cumulative totals for hover tooltip (independent of the plotted metric)
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

    result["cases_pc"] = np.where(
        (result["population"] > 0) & result["population"].notna(),
        (result["cases"] / result["population"]) * 100_000,
        np.nan,
    )
    result["deaths_pc"] = np.where(
        (result["population"] > 0) & result["population"].notna(),
        (result["deaths"] / result["population"]) * 100_000,
        np.nan,
    )

    # Drop only rows where the plotted metric itself is missing; hover fields may be NaN.
    result = result.dropna(subset=["value"]).copy()
    return result


def filter_choropleth_by_state(choro_data, state):
    """
    Filter choropleth data to only include counties in the specified state.

    Args:
        choro_data: Choropleth dataframe from prepare_choropleth_for_date()
        state: State abbreviation (e.g., "PA") or "United States" for all

    Returns:
        Filtered choropleth dataframe (or copy of original for "United States")
    """
    if state == "United States":
        return choro_data.copy()
    return choro_data[choro_data["State"] == state].copy()


def get_state_bounds_for_zoom(state):
    """
    Return approximate geographic center and zoom level for a state.

    Args:
        state: State abbreviation

    Returns:
        Dict with 'lat', 'lon', 'zoom' keys, or None if state not in table.
    """
    STATE_BOUNDS = {
        "AL": {"lat": 32.8,  "lon": -86.8,  "zoom": 6},
        "AK": {"lat": 64.0,  "lon": -153.0, "zoom": 3},
        "AZ": {"lat": 33.7,  "lon": -111.4, "zoom": 6},
        "AR": {"lat": 34.8,  "lon": -92.4,  "zoom": 6},
        "CA": {"lat": 37.0,  "lon": -119.5, "zoom": 5},
        "CO": {"lat": 39.0,  "lon": -105.5, "zoom": 6},
        "CT": {"lat": 41.6,  "lon": -72.7,  "zoom": 8},
        "DE": {"lat": 39.0,  "lon": -75.5,  "zoom": 8},
        "FL": {"lat": 27.7,  "lon": -81.8,  "zoom": 6},
        "GA": {"lat": 32.8,  "lon": -83.6,  "zoom": 6},
        "HI": {"lat": 21.1,  "lon": -156.5, "zoom": 5},
        "ID": {"lat": 44.2,  "lon": -114.0, "zoom": 6},
        "IL": {"lat": 40.0,  "lon": -89.0,  "zoom": 6},
        "IN": {"lat": 39.8,  "lon": -86.2,  "zoom": 7},
        "IA": {"lat": 42.0,  "lon": -93.5,  "zoom": 6},
        "KS": {"lat": 38.5,  "lon": -97.5,  "zoom": 6},
        "KY": {"lat": 37.7,  "lon": -84.7,  "zoom": 6},
        "LA": {"lat": 31.0,  "lon": -91.5,  "zoom": 6},
        "ME": {"lat": 45.3,  "lon": -69.0,  "zoom": 7},
        "MD": {"lat": 39.1,  "lon": -76.8,  "zoom": 7},
        "MA": {"lat": 42.2,  "lon": -71.8,  "zoom": 7},
        "MI": {"lat": 44.3,  "lon": -85.4,  "zoom": 6},
        "MN": {"lat": 46.0,  "lon": -93.9,  "zoom": 6},
        "MS": {"lat": 32.8,  "lon": -89.7,  "zoom": 6},
        "MO": {"lat": 38.5,  "lon": -92.3,  "zoom": 6},
        "MT": {"lat": 47.0,  "lon": -110.0, "zoom": 5},
        "NE": {"lat": 41.5,  "lon": -99.9,  "zoom": 6},
        "NV": {"lat": 39.5,  "lon": -116.9, "zoom": 6},
        "NH": {"lat": 43.5,  "lon": -71.5,  "zoom": 8},
        "NJ": {"lat": 40.0,  "lon": -74.5,  "zoom": 8},
        "NM": {"lat": 34.5,  "lon": -106.6, "zoom": 6},
        "NY": {"lat": 43.0,  "lon": -75.5,  "zoom": 6},
        "NC": {"lat": 35.5,  "lon": -79.8,  "zoom": 6},
        "ND": {"lat": 47.5,  "lon": -100.5, "zoom": 6},
        "OH": {"lat": 40.4,  "lon": -82.9,  "zoom": 6},
        "OK": {"lat": 35.5,  "lon": -97.5,  "zoom": 6},
        "OR": {"lat": 44.0,  "lon": -121.3, "zoom": 6},
        "PA": {"lat": 40.8,  "lon": -77.8,  "zoom": 6},
        "RI": {"lat": 41.7,  "lon": -71.5,  "zoom": 9},
        "SC": {"lat": 34.0,  "lon": -81.0,  "zoom": 6},
        "SD": {"lat": 44.5,  "lon": -100.0, "zoom": 6},
        "TN": {"lat": 35.5,  "lon": -86.5,  "zoom": 6},
        "TX": {"lat": 31.0,  "lon": -99.0,  "zoom": 5},
        "UT": {"lat": 39.0,  "lon": -111.5, "zoom": 6},
        "VT": {"lat": 44.0,  "lon": -72.7,  "zoom": 7},
        "VA": {"lat": 37.8,  "lon": -78.2,  "zoom": 6},
        "WA": {"lat": 47.5,  "lon": -120.5, "zoom": 6},
        "WV": {"lat": 38.5,  "lon": -81.9,  "zoom": 7},
        "WI": {"lat": 44.3,  "lon": -89.6,  "zoom": 6},
        "WY": {"lat": 43.0,  "lon": -107.5, "zoom": 6},
    }
    return STATE_BOUNDS.get(state)


def compute_national_timeseries(cases_df, deaths_df, metric_type="Cases"):
    """
    Compute national (United States-wide) timeseries by summing all counties.

    Statewide unallocated rows (countyFIPS == '00000') are excluded before
    summing to prevent double-counting.

    Args:
        cases_df: Wide-format cases dataframe
        deaths_df: Wide-format deaths dataframe
        metric_type: Either "Cases" or "Deaths"

    Returns:
        DataFrame with columns: Date, Value (long format)
    """
    source_df = cases_df if metric_type == "Cases" else deaths_df
    date_cols = get_date_columns(source_df)

    valid_rows = source_df[source_df["countyFIPS"] != "00000"]
    national_values = valid_rows[date_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

    result = pd.DataFrame({
        "Date": pd.to_datetime(date_cols),
        "Value": national_values.values,
    })
    return result.sort_values("Date").reset_index(drop=True)


def compute_national_daily(daily_df):
    """
    Compute national daily timeseries from a precomputed daily dataframe.

    Statewide unallocated rows (countyFIPS == '00000') are excluded.

    Args:
        daily_df: Wide-format daily cases/deaths dataframe

    Returns:
        DataFrame with columns: Date, Value (long format)
    """
    date_cols = get_date_columns(daily_df)
    valid_rows = daily_df[daily_df["countyFIPS"] != "00000"]
    national_values = valid_rows[date_cols].sum(axis=0)

    result = pd.DataFrame({
        "Date": pd.to_datetime(date_cols),
        "Value": national_values.values,
    })
    return result.sort_values("Date").reset_index(drop=True)


URBAN_POPULATION_THRESHOLD = 100_000  # fallback threshold when RUCC data is absent


def classify_county_type(population_df, urban_threshold=URBAN_POPULATION_THRESHOLD, rucc_df=None):
    """
    Classify every county as Metro/Nonmetro (preferred) or Urban/Rural (fallback).

    When ``rucc_df`` is supplied (the AHRF feature table containing a
    ``rucc_code`` column), the classification uses USDA Rural-Urban Continuum
    Codes (RUCC 2013):
        Metro    — RUCC 1-3  (metro counties of any size)
        Nonmetro — RUCC 4-9  (non-metro counties of any adjacency)

    When ``rucc_df`` is None the function falls back to the original
    population-threshold approach:
        Urban — population >= urban_threshold (default 100,000)
        Rural — population <  urban_threshold

    The return contract is unchanged: a DataFrame with columns
    (countyFIPS, State, County_Type) so all downstream consumers work without
    modification.  Statewide unallocated rows (FIPS == '00000') and rows with
    invalid/missing data are excluded.

    Args:
        population_df:    Population dataframe (wide format).
        urban_threshold:  Fallback minimum population for "Urban" classification.
        rucc_df:          Optional AHRF feature DataFrame containing columns
                          ``countyFIPS``, ``State`` (or ``state``), and
                          ``rucc_code``.  Produced by
                          ahrf_loader.build_ahrf_feature_table().

    Returns:
        DataFrame with columns: countyFIPS, State, County_Type
        County_Type values: "Metro"|"Nonmetro" (RUCC) or "Urban"|"Rural" (fallback)
    """
    if rucc_df is not None and "rucc_code" in rucc_df.columns:
        state_col = "State" if "State" in rucc_df.columns else "state"
        needed = {"countyFIPS", "rucc_code", state_col}
        if needed.issubset(rucc_df.columns):
            valid = rucc_df[["countyFIPS", state_col, "rucc_code"]].copy()
            valid = valid.rename(columns={state_col: "State"})
            valid = valid[valid["countyFIPS"] != "00000"].copy()

            rucc = pd.to_numeric(valid["rucc_code"], errors="coerce")
            valid["County_Type"] = np.where(
                rucc.between(1, 3), "Metro",
                np.where(rucc.between(4, 9), "Nonmetro", pd.NA)
            )
            valid = valid.dropna(subset=["County_Type"])
            return valid[["countyFIPS", "State", "County_Type"]].reset_index(drop=True)

    # Population-threshold fallback (no RUCC data available)
    pop_col = get_population_column(population_df)
    if pop_col is None:
        raise ValueError("Population dataframe does not contain a population column")

    valid = population_df[
        (population_df["countyFIPS"] != "00000") &
        (pd.to_numeric(population_df[pop_col], errors="coerce") > 0)
    ][["countyFIPS", "State", pop_col]].copy()

    valid[pop_col] = pd.to_numeric(valid[pop_col], errors="coerce")

    valid["County_Type"] = np.where(
        valid[pop_col] >= urban_threshold,
        "Urban",
        "Rural",
    )

    return valid[["countyFIPS", "State", "County_Type"]].reset_index(drop=True)


def filter_choropleth_by_county_type(choro_data, county_type):
    """
    Filter a prepared choropleth DataFrame by county type classification.

    Expects choro_data to contain a 'County_Type' column, added by merging
    the output of classify_county_type() before this call.

    Accepts both the RUCC-based Metro/Nonmetro labels (preferred, produced when
    AHRF data is loaded) and the legacy population-threshold Urban/Rural labels
    (used as a fallback when AHRF data is unavailable).

    Args:
        choro_data:   Choropleth DataFrame (output of prepare_choropleth_for_date
                      after County_Type has been merged in).
        county_type:  One of:
                        "All Counties"
                        "Metro Counties"    / "Urban Counties"   (treated identically)
                        "Nonmetro Counties" / "Rural Counties"   (treated identically)

    Returns:
        Filtered copy of choro_data, or the full copy for "All Counties".
    """
    if county_type == "All Counties":
        return choro_data.copy()

    # Normalise both label families to their target County_Type value
    type_map = {
        # RUCC-based (preferred)
        "Metro Counties":    "Metro",
        "Nonmetro Counties": "Nonmetro",
        # Population-threshold fallback
        "Urban Counties":    "Urban",
        "Rural Counties":    "Rural",
    }
    target = type_map.get(county_type)
    if target is None:
        return choro_data.copy()

    if "County_Type" not in choro_data.columns:
        # County_Type column not present — return unfiltered rather than crashing
        return choro_data.copy()

    return choro_data[choro_data["County_Type"] == target].copy()


def peer_median_series(daily_ma_df, population_df, fips_list):
    """
    Median daily-per-100k trajectory across a set of counties.

    Used to overlay a "structural peers" reference curve on a county's
    pandemic timeline: each peer's smoothed daily counts are converted to
    per-100k rates, then the median across peers is taken date by date.

    Args:
        daily_ma_df:   Wide-format smoothed daily counts (e.g. ma7_cases).
        population_df: Population dataframe.
        fips_list:     Iterable of 5-char FIPS codes (the peer group).

    Returns:
        DataFrame with columns Date, Value (median per-100k rate), or an
        empty DataFrame if fewer than 3 peers have valid population data.
    """
    fips_set = {str(f).zfill(5) for f in fips_list}
    sub = daily_ma_df[daily_ma_df["countyFIPS"].isin(fips_set)]
    if sub.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    pop_col = get_population_column(population_df)
    pops = population_df[
        (population_df["countyFIPS"].isin(fips_set)) &
        (pd.to_numeric(population_df[pop_col], errors="coerce") > 0)
    ][["countyFIPS", "State", pop_col]].rename(columns={pop_col: "_pop"})

    merged = sub.merge(pops, on=["countyFIPS", "State"], how="inner")
    if len(merged) < 3:
        return pd.DataFrame(columns=["Date", "Value"])

    date_cols = get_date_columns(daily_ma_df)
    rates = merged[date_cols].apply(pd.to_numeric, errors="coerce")
    rates = rates.div(pd.to_numeric(merged["_pop"], errors="coerce").values, axis=0) * 100_000

    return pd.DataFrame({
        "Date": pd.to_datetime(date_cols),
        "Value": rates.median(axis=0).values,
    })


def find_data_corrections(df_wide, county_name, state):
    """
    Find dates where a county's cumulative series decreased.

    A cumulative count can only legitimately rise; a decrease means the source
    revised earlier figures (backfill, de-duplication, jurisdiction change).
    The daily-diff pipeline clips these to zero for analysis; this function
    recovers them so charts can flag correction events instead of hiding them.

    Args:
        df_wide: Wide-format cumulative dataframe (cases or deaths).
        county_name, state: Identify the county.

    Returns:
        DataFrame with columns Date (Timestamp) and correction (the negative
        one-day change). Empty DataFrame if the county is missing or its
        series never decreases.
    """
    row = df_wide[
        (df_wide["County Name"] == county_name) & (df_wide["State"] == state)
    ]
    if row.empty:
        return pd.DataFrame(columns=["Date", "correction"])

    date_cols = get_date_columns(df_wide)
    values = pd.to_numeric(row.iloc[0][date_cols], errors="coerce")
    diffs = values.diff()
    mask = diffs < 0

    return pd.DataFrame({
        "Date": pd.to_datetime(pd.Index(date_cols)[mask.values]),
        "correction": diffs[mask.values].astype(float).values,
    })


def monthly_snapshot_long(metric_df):
    """
    Melt a wide metric table to long format with one date per calendar month.

    Keeps the first available date column of each month (~40 snapshots instead
    of ~1,300 daily columns), which is what makes choropleth animation frames
    tractable for Plotly. Statewide unallocated rows are excluded.

    Returns:
        DataFrame with columns: countyFIPS [, Location, State], Month, value
        ('Month' is 'YYYY-MM'; rows with non-numeric values dropped).
    """
    date_cols = get_date_columns(metric_df)
    monthly, seen = [], set()
    for d in sorted(date_cols):
        month = d[:7]
        if month not in seen:
            seen.add(month)
            monthly.append(d)

    base = metric_df[metric_df["countyFIPS"] != "00000"]
    id_cols = [c for c in ["countyFIPS", "Location", "State"] if c in base.columns]
    long_df = base[id_cols + monthly].melt(
        id_vars=id_cols, var_name="Month", value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["Month"] = long_df["Month"].str[:7]
    return long_df.dropna(subset=["value"])


def compute_window_outcomes(cases_df, deaths_df, population_df, start_date, end_date):
    """
    Compute per-100k COVID outcomes restricted to a date window.

    Window counts are cumulative[end_date] − cumulative[start_date], i.e. events
    occurring after start_date through end_date. Restricting the outcome window
    matters for factor analyses: correlating vaccination rates against
    full-pandemic cumulative outcomes mixes pre- and post-rollout periods and
    invites reverse-causality artifacts.

    Args:
        cases_df, deaths_df: Wide-format cumulative dataframes.
        population_df: Population dataframe.
        start_date, end_date: Date strings (YYYY-MM-DD) present in the columns.

    Returns:
        DataFrame with columns:
            countyFIPS, State, cases_per_100k, deaths_per_100k, case_fatality_rate
        (window-restricted values; NaN where population is invalid).
    """
    for d in (start_date, end_date):
        if d not in cases_df.columns:
            raise ValueError(f"Date column '{d}' not found in cases dataframe")
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date")

    pop_col = get_population_column(population_df)
    if pop_col is None:
        raise ValueError("Population dataframe does not contain a population column")

    result = cases_df[["countyFIPS", "State"]].copy()
    result = result[result["countyFIPS"] != "00000"].reset_index(drop=True)

    def _window_counts(df):
        sub = df[df["countyFIPS"] != "00000"].reset_index(drop=True)
        start = pd.to_numeric(sub[start_date], errors="coerce")
        end = pd.to_numeric(sub[end_date], errors="coerce")
        return (end - start).clip(lower=0)

    result["window_cases"] = _window_counts(cases_df)
    result["window_deaths"] = _window_counts(deaths_df)

    pop_valid = population_df[
        (population_df["countyFIPS"] != "00000") &
        (pd.to_numeric(population_df[pop_col], errors="coerce") > 0)
    ][["countyFIPS", "State", pop_col]].rename(columns={pop_col: "_pop"})
    result = result.merge(pop_valid, on=["countyFIPS", "State"], how="left")

    pop = pd.to_numeric(result["_pop"], errors="coerce")
    result["cases_per_100k"] = np.where(pop > 0, result["window_cases"] / pop * 100_000, np.nan)
    result["deaths_per_100k"] = np.where(pop > 0, result["window_deaths"] / pop * 100_000, np.nan)
    result["case_fatality_rate"] = np.where(
        result["window_cases"] > 0,
        result["window_deaths"] / result["window_cases"] * 100,
        np.nan,
    )

    return result[["countyFIPS", "State", "cases_per_100k",
                   "deaths_per_100k", "case_fatality_rate"]]


def compute_national_per_capita(raw_df, population_df):
    """
    Compute national per-capita timeseries directly from raw cumulative counts.

    This avoids the floating-point round-trip of the previous implementation,
    which reconstructed raw counts from precomputed per-capita values.
    Statewide unallocated rows (countyFIPS == '00000') are excluded from both
    the numerator and the total population denominator.

    Args:
        raw_df: Wide-format raw (cumulative) cases or deaths dataframe
        population_df: Population dataframe

    Returns:
        DataFrame with columns: Date, Value (per 100k, long format)
    """
    date_cols = get_date_columns(raw_df)

    pop_col = get_population_column(population_df)
    if pop_col is None:
        raise ValueError("Population dataframe does not contain a population column")

    total_pop = population_df[
        (population_df["countyFIPS"] != "00000") &
        (population_df[pop_col] > 0)
    ][pop_col].apply(pd.to_numeric, errors="coerce").sum()

    valid_rows = raw_df[raw_df["countyFIPS"] != "00000"]
    national_counts = (
        valid_rows[date_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    )

    result = pd.DataFrame({
        "Date": pd.to_datetime(date_cols),
        "Value": (national_counts / total_pop) * 100_000,
    })
    return result.sort_values("Date").reset_index(drop=True)
