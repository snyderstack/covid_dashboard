"""
Vaccination data loader for CDC county-level COVID-19 vaccination dataset.

Source:
    CDC COVID-19 Vaccinations in the United States, County
    File: COVID-19_Vaccinations_in_the_United_States,County_20260623.csv

Coverage:
    Dates:    2020-12-13 → 2023-05-10  (~609 unique dates, near-daily)
    Counties: ~3,224 valid county FIPS (excludes UNK / state-unallocated rows)
    Match:    3,142/3,143 USAFacts COVID counties matched (99.97 %)

Internal column mapping (CDC → internal):
    Administered_Dose1_Pop_Pct     → vax_dose1_pct
    Series_Complete_Pop_Pct        → vax_complete_pct
    Booster_Doses_Vax_Pct          → vax_booster_pct
    Series_Complete_65PlusPop_Pct  → vax_complete_65plus_pct
    Bivalent_Booster_5Plus_Pop_Pct → vax_bivalent_pct

All percentage columns are clipped to [0, 100]. UNK rows (FIPS length ≠ 5)
are excluded. FIPS values are zero-padded to 5 characters.

Public API:
    load_vaccination_latest(data_dir)
        → DataFrame: one row per county, most-recent vaccination snapshot

    load_vaccination_timeseries(data_dir)
        → DataFrame: full date × county time-series, sorted by FIPS then Date

    get_vaccination_at_dates(vax_ts, fips, target_dates)
        → Series: vax_complete_pct for a single county at requested dates
          (forward-filled from nearest prior observation)

    get_county_vax_timeseries(vax_ts, fips)
        → DataFrame: all rows for a single county
"""

from pathlib import Path

import numpy as np
import pandas as pd

VAX_FILE = "COVID-19_Vaccinations_in_the_United_States,County_20260623.csv"

# CDC column → internal name
_CDC_TO_INTERNAL = {
    "Administered_Dose1_Pop_Pct":     "vax_dose1_pct",
    "Series_Complete_Pop_Pct":        "vax_complete_pct",
    "Booster_Doses_Vax_Pct":          "vax_booster_pct",
    "Series_Complete_65PlusPop_Pct":  "vax_complete_65plus_pct",
    "Bivalent_Booster_5Plus_Pop_Pct": "vax_bivalent_pct",
}

VAX_PCT_COLS = list(_CDC_TO_INTERNAL.values())   # all internal % column names

# Human-readable labels (internal column → display label)
VAX_LABELS = {
    "vax_dose1_pct":           "At Least 1 Dose (%)",
    "vax_complete_pct":        "Fully Vaccinated (%)",
    "vax_booster_pct":         "Booster Rate (%)",
    "vax_complete_65plus_pct": "65+ Fully Vaccinated (%)",
    "vax_bivalent_pct":        "Bivalent Booster (%)",
}


def _load_raw(data_dir: str) -> pd.DataFrame:
    """
    Read and minimally clean the CDC vaccination CSV.

    Returns a DataFrame with columns:
        countyFIPS, Date, vax_dose1_pct, vax_complete_pct,
        vax_booster_pct, vax_complete_65plus_pct, vax_bivalent_pct

    Returns an empty DataFrame if the file is not found.
    """
    path = Path(data_dir) / VAX_FILE
    if not path.exists():
        return pd.DataFrame()

    read_cols = {"Date", "FIPS"} | set(_CDC_TO_INTERNAL.keys())

    df = pd.read_csv(
        path,
        dtype={"FIPS": str},
        low_memory=False,
        usecols=lambda c: c in read_cols,
    )

    # Parse dates; drop rows with unparseable dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Exclude state-unallocated rows (FIPS = "UNK" or wrong length)
    df = df[df["FIPS"].str.len() == 5].copy()
    df["FIPS"] = df["FIPS"].str.zfill(5)

    # Rename columns
    df = df.rename(columns=_CDC_TO_INTERNAL)
    df = df.rename(columns={"FIPS": "countyFIPS"})

    # Coerce and clip percentage columns to [0, 100]
    for col in VAX_PCT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100)

    return df[["countyFIPS", "Date"] + [c for c in VAX_PCT_COLS if c in df.columns]]


def load_vaccination_latest(data_dir: str) -> pd.DataFrame:
    """
    Return the most-recent vaccination snapshot per county.

    For each county, selects the row with the latest Date.

    Returns DataFrame with columns:
        countyFIPS, vax_dose1_pct, vax_complete_pct, vax_booster_pct,
        vax_complete_65plus_pct, vax_bivalent_pct, vax_last_date

    One row per county (5-char zero-padded FIPS).
    Returns empty DataFrame if source file is not found.
    """
    df = _load_raw(data_dir)
    if df.empty:
        return pd.DataFrame()

    latest = (
        df.sort_values("Date")
        .groupby("countyFIPS", as_index=False)
        .last()
        .rename(columns={"Date": "vax_last_date"})
    )
    return latest.reset_index(drop=True)


def load_vaccination_timeseries(data_dir: str) -> pd.DataFrame:
    """
    Return full date × county vaccination time-series.

    Returns DataFrame with columns:
        countyFIPS, Date, vax_dose1_pct, vax_complete_pct,
        vax_booster_pct, vax_complete_65plus_pct, vax_bivalent_pct

    Sorted by (countyFIPS, Date) ascending.
    Returns empty DataFrame if source file is not found.
    """
    df = _load_raw(data_dir)
    if df.empty:
        return pd.DataFrame()
    return df.sort_values(["countyFIPS", "Date"]).reset_index(drop=True)


def get_vaccination_at_dates(
    vax_ts: pd.DataFrame,
    fips: str,
    target_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Look up vax_complete_pct for a single county on or before each target date.

    Uses forward-fill (most-recent observation on or before the target date).
    Pre-data dates (before 2020-12-13) return 0.0.
    Post-data dates (after 2023-05-10) return the last available value.

    Args:
        vax_ts:       Full vaccination time-series DataFrame.
        fips:         5-char county FIPS string.
        target_dates: DatetimeIndex of dates to look up.

    Returns:
        pd.Series of vax_complete_pct values, indexed by target_dates.
        NaN for counties with no vaccination data.
    """
    if vax_ts.empty or "countyFIPS" not in vax_ts.columns:
        return pd.Series(np.nan, index=target_dates)

    fips = str(fips).zfill(5)
    county_df = vax_ts[vax_ts["countyFIPS"] == fips]
    if county_df.empty:
        return pd.Series(np.nan, index=target_dates)

    county_ts = (
        county_df[["Date", "vax_complete_pct"]]
        .set_index("Date")
        .sort_index()
    )

    # Union of all dates so we can reindex cleanly
    all_dates = target_dates.union(county_ts.index).sort_values()
    series = county_ts["vax_complete_pct"].reindex(all_dates).ffill().fillna(0.0)
    return series.reindex(target_dates)


def get_county_vax_timeseries(
    vax_ts: pd.DataFrame,
    fips: str,
) -> pd.DataFrame:
    """
    Return all time-series rows for a single county.

    Args:
        vax_ts: Full vaccination time-series DataFrame.
        fips:   5-char county FIPS string.

    Returns:
        DataFrame subset for this county, sorted by Date.
        Empty DataFrame if county not found.
    """
    if vax_ts.empty:
        return pd.DataFrame()
    fips = str(fips).zfill(5)
    return (
        vax_ts[vax_ts["countyFIPS"] == fips]
        .sort_values("Date")
        .reset_index(drop=True)
    )
