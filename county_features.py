"""
County-level feature table for COVID-19 analysis and external data integration.

Provides a modular architecture for aggregating county-level metrics and
merging with external datasets (AHRF healthcare, socioeconomic, demographic).

Public API:
    create_county_feature_table()   — COVID metrics only (backward compatible)
    create_master_county_table()    — COVID + AHRF feature table (full research table)
    add_external_dataset()          — Generic FIPS-join hook for any external dataset
    compute_bivariate_correlation() — Pearson + Spearman with p-values
    compute_correlation_matrix()    — All-vs-all correlation matrix
    normalize_feature()             — Min-max normalization
    standardize_feature()           — Z-score standardization
    prepare_for_regression()        — X, y matrices ready for OLS
    prepare_for_clustering()        — Feature matrix ready for K-means / hierarchical

"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def create_county_feature_table(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    population_df: pd.DataFrame,
    wave_metrics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create a county-level feature table with COVID metrics.

    This is the original function; its interface is unchanged for backward
    compatibility.  For the full research table (COVID + AHRF), use
    create_master_county_table().

    Args:
        cases_df:       Wide-format cumulative cases DataFrame.
        deaths_df:      Wide-format cumulative deaths DataFrame.
        population_df:  Population DataFrame.
        wave_metrics_df: Optional wave metrics from wave_analysis module.

    Returns:
        DataFrame with one row per county containing COVID metrics.
    """
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols  = [c for c in cases_df.columns if c not in identifier_cols]
    latest_date = sorted(date_cols)[-1]

    pop_cols = [c for c in population_df.columns
                if c not in identifier_cols and c != "Location"]
    pop_col  = pop_cols[0] if pop_cols else "population"

    features = (
        population_df[["countyFIPS", "County Name", "State", pop_col]]
        .rename(columns={pop_col: "population"})
        .copy()
    )
    features = features[
        (features["countyFIPS"] != "00000") & (features["population"] > 0)
    ].reset_index(drop=True)

    cases_totals  = cases_df[["countyFIPS", "State", latest_date]].rename(columns={latest_date: "total_cases"})
    deaths_totals = deaths_df[["countyFIPS", "State", latest_date]].rename(columns={latest_date: "total_deaths"})

    features = features.merge(cases_totals,  on=["countyFIPS", "State"], how="left")
    features = features.merge(deaths_totals, on=["countyFIPS", "State"], how="left")

    features["total_cases"]  = pd.to_numeric(features["total_cases"],  errors="coerce")
    features["total_deaths"] = pd.to_numeric(features["total_deaths"], errors="coerce")

    features["cases_per_100k"] = np.where(
        (features["population"] > 0) & features["population"].notna(),
        (features["total_cases"]  / features["population"]) * 100_000, np.nan,
    )
    features["deaths_per_100k"] = np.where(
        (features["population"] > 0) & features["population"].notna(),
        (features["total_deaths"] / features["population"]) * 100_000, np.nan,
    )

    if wave_metrics_df is not None and len(wave_metrics_df) > 0:
        wave_cols = [c for c in wave_metrics_df.columns
                     if c not in ["countyFIPS", "County Name", "State"]]
        features = features.merge(
            wave_metrics_df[["countyFIPS", "State"] + wave_cols],
            on=["countyFIPS", "State"], how="left",
        )

    return features


def create_master_county_table(
    cases_df:       pd.DataFrame,
    deaths_df:      pd.DataFrame,
    population_df:  pd.DataFrame,
    ahrf_df:        Optional[pd.DataFrame] = None,
    wave_metrics_df: Optional[pd.DataFrame] = None,
    vax_df:         Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build the master county feature table combining COVID outcomes with
    AHRF healthcare, socioeconomic, and demographic variables.

    Join key: countyFIPS (5-character zero-padded string).

    Args:
        cases_df:       Wide-format cumulative cases DataFrame.
        deaths_df:      Wide-format cumulative deaths DataFrame.
        population_df:  Population DataFrame from USAFacts.
        ahrf_df:        AHRF feature DataFrame (from ahrf_loader.build_ahrf_feature_table).
                        If None, AHRF columns are omitted.
        wave_metrics_df: Optional pre-computed wave metrics per county.
        vax_df:         Optional vaccination latest-snapshot DataFrame
                        (from vaccination_loader.load_vaccination_latest).
                        Joined on countyFIPS. Columns added: vax_dose1_pct,
                        vax_complete_pct, vax_booster_pct, vax_complete_65plus_pct,
                        vax_bivalent_pct, vax_last_date.
        verbose:        Print join diagnostics.

    Returns:
        (master_df, diagnostics) where master_df has one row per county.
    """
    diag = {}

    covid_features = create_county_feature_table(
        cases_df, deaths_df, population_df, wave_metrics_df
    )

    covid_features["countyFIPS"] = (
        covid_features["countyFIPS"].astype(str).str.zfill(5)
    )
    diag["covid_counties"] = len(covid_features)

    if ahrf_df is None:
        if verbose:
            print("No AHRF data provided — returning COVID-only feature table")
        return covid_features, diag

    ahrf_copy = ahrf_df.copy()
    ahrf_copy["countyFIPS"] = ahrf_copy["countyFIPS"].astype(str).str.zfill(5)

    exclude_from_ahrf = {
        "countyFIPS", "state", "county_name",
        "County Name", "State", "population",  # avoid duplicates
    }
    ahrf_cols = [c for c in ahrf_copy.columns if c not in exclude_from_ahrf]

    master = covid_features.merge(
        ahrf_copy[["countyFIPS"] + ahrf_cols],
        on="countyFIPS",
        how="left",
    )

    n_matched  = master["pcp_per_100k"].notna().sum() if "pcp_per_100k" in master.columns else "N/A"
    n_unmatched = master["pcp_per_100k"].isna().sum()  if "pcp_per_100k" in master.columns else "N/A"

    diag.update({
        "ahrf_counties":   len(ahrf_df),
        "master_counties": len(master),
        "matched_ahrf":    n_matched,
        "unmatched_ahrf":  n_unmatched,
        "ahrf_columns_added": len(ahrf_cols),
    })

    if verbose:
        print(f"Master table: {len(master):,} counties, {len(master.columns)} columns")
        print(f"  COVID counties:  {diag['covid_counties']:,}")
        print(f"  AHRF matched:    {n_matched}")
        print(f"  AHRF unmatched:  {n_unmatched}")

    # Vaccination data join
    if vax_df is not None and not vax_df.empty:
        vax_copy = vax_df.copy()
        vax_copy["countyFIPS"] = vax_copy["countyFIPS"].astype(str).str.zfill(5)
        # Exclude columns already in master (avoid duplicates)
        vax_cols = [c for c in vax_copy.columns if c not in {"countyFIPS"} and c not in master.columns]
        master = master.merge(
            vax_copy[["countyFIPS"] + vax_cols],
            on="countyFIPS",
            how="left",
        )
        n_vax_matched = master["vax_complete_pct"].notna().sum() if "vax_complete_pct" in master.columns else "N/A"
        diag["vax_counties"]    = len(vax_df)
        diag["vax_matched"]     = n_vax_matched
        diag["vax_cols_added"]  = len(vax_cols)
        if verbose:
            print(f"  Vaccination matched: {n_vax_matched}")

    return master, diag


def add_external_dataset(
    features_df:       pd.DataFrame,
    external_df:       pd.DataFrame,
    external_fips_col: str  = "countyFIPS",
    join_state:        bool = True,
) -> pd.DataFrame:
    """
    Merge any county-level external dataset into the feature table.

    Standardises FIPS in the external dataset before joining.  Where
    possible, joins on (countyFIPS, State) to prevent cross-state FIPS
    collisions.

    Args:
        features_df:       Master feature table.
        external_df:       Dataset to merge (must contain a FIPS column).
        external_fips_col: Name of the FIPS column in external_df.
        join_state:        If True, join on (FIPS, State); else FIPS only.

    Returns:
        Updated feature table with external columns appended.
    """
    result = features_df.copy()

    if external_fips_col not in external_df.columns:
        warnings.warn(f"Column '{external_fips_col}' not found in external_df — returning unchanged")
        return result

    ext = external_df.copy()
    ext[external_fips_col] = (
        pd.to_numeric(ext[external_fips_col], errors="coerce")
        .fillna(0).astype(int).astype(str).str.zfill(5)
    )
    ext = ext.rename(columns={external_fips_col: "countyFIPS"})

    on_cols = ["countyFIPS", "State"] if (join_state and "State" in ext.columns) else ["countyFIPS"]

    result = result.merge(ext, on=on_cols, how="left", suffixes=("", "_external"))
    return result


def compute_bivariate_correlation(
    df:           pd.DataFrame,
    x_col:        str,
    y_col:        str,
    filter_mask:  Optional[pd.Series] = None,
    min_n:        int = 30,
) -> dict:
    """
    Compute Pearson and Spearman correlations between two columns.

    Both NaN-containing rows and rows excluded by filter_mask are dropped
    before computation.  Returns NaN statistics when fewer than min_n
    valid rows are available.

    Args:
        df:          Feature DataFrame.
        x_col:       Name of the X (predictor) column.
        y_col:       Name of the Y (outcome) column.
        filter_mask: Boolean Series aligned with df; True rows are kept.
        min_n:       Minimum number of valid pairs required.

    Returns:
        Dict with keys:
            n, pearson_r, pearson_p, spearman_r, spearman_p,
            r_squared, x_mean, y_mean, x_std, y_std
    """
    empty = {
        "n": 0, "pearson_r": np.nan, "pearson_p": np.nan,
        "spearman_r": np.nan, "spearman_p": np.nan,
        "r_squared": np.nan, "x_mean": np.nan, "y_mean": np.nan,
        "x_std": np.nan, "y_std": np.nan,
    }

    if x_col not in df.columns or y_col not in df.columns:
        return empty

    sub = df.copy()
    if filter_mask is not None:
        sub = sub[filter_mask.values if hasattr(filter_mask, "values") else filter_mask]

    valid = sub[[x_col, y_col]].dropna()
    x = pd.to_numeric(valid[x_col], errors="coerce")
    y = pd.to_numeric(valid[y_col], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    n = len(valid)
    if n < min_n:
        return {**empty, "n": n}

    pr, pp = stats.pearsonr(valid["x"], valid["y"])
    sr, sp = stats.spearmanr(valid["x"], valid["y"])

    return {
        "n":          n,
        "pearson_r":  float(pr),
        "pearson_p":  float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "r_squared":  float(pr ** 2),
        "x_mean":     float(valid["x"].mean()),
        "y_mean":     float(valid["y"].mean()),
        "x_std":      float(valid["x"].std()),
        "y_std":      float(valid["y"].std()),
    }


def compute_ols_trend(
    df:          pd.DataFrame,
    x_col:       str,
    y_col:       str,
    filter_mask: Optional[pd.Series] = None,
) -> dict:
    """
    Fit a simple OLS regression (y ~ x) and return slope/intercept/R².

    Args:
        df:          Feature DataFrame.
        x_col:       Predictor column name.
        y_col:       Outcome column name.
        filter_mask: Boolean Series; True rows are kept.

    Returns:
        Dict with keys: slope, intercept, r_squared, n, x_min, x_max,
        y_pred_min, y_pred_max
    """
    empty = {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan,
             "n": 0, "x_min": np.nan, "x_max": np.nan,
             "y_pred_min": np.nan, "y_pred_max": np.nan}

    if x_col not in df.columns or y_col not in df.columns:
        return empty

    sub = df.copy()
    if filter_mask is not None:
        sub = sub[filter_mask.values if hasattr(filter_mask, "values") else filter_mask]

    valid = sub[[x_col, y_col]].dropna()
    x = pd.to_numeric(valid[x_col], errors="coerce")
    y = pd.to_numeric(valid[y_col], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    n = len(valid)
    if n < 3:
        return {**empty, "n": n}

    slope, intercept, r, p, se = stats.linregress(valid["x"], valid["y"])
    x_min, x_max = valid["x"].min(), valid["x"].max()

    return {
        "slope":      float(slope),
        "intercept":  float(intercept),
        "r_squared":  float(r ** 2),
        "n":          n,
        "x_min":      float(x_min),
        "x_max":      float(x_max),
        "y_pred_min": float(slope * x_min + intercept),
        "y_pred_max": float(slope * x_max + intercept),
    }


def compute_correlation_matrix(
    df:              pd.DataFrame,
    feature_cols:    List[str],
    outcome_cols:    List[str],
    method:          str = "pearson",
) -> pd.DataFrame:
    """
    Compute a correlation matrix between feature columns and outcome columns.

    Args:
        df:           Feature DataFrame.
        feature_cols: Column names for the predictor variables (X).
        outcome_cols: Column names for the outcome variables (Y).
        method:       'pearson' or 'spearman'.

    Returns:
        DataFrame with feature_cols as rows and outcome_cols as columns,
        containing correlation coefficients.  NaN where insufficient data.
    """
    method = method.lower()
    corr_data: dict = {oc: {} for oc in outcome_cols}

    for fc in feature_cols:
        for oc in outcome_cols:
            res = compute_bivariate_correlation(df, fc, oc)
            r = res["pearson_r"] if method == "pearson" else res["spearman_r"]
            corr_data[oc][fc] = r

    return pd.DataFrame(corr_data, index=feature_cols)


def normalize_feature(feature_series: pd.Series) -> pd.Series:
    """Min-max normalise a feature to [0, 1]."""
    valid = feature_series.dropna()
    if len(valid) == 0:
        return feature_series
    mn, mx = valid.min(), valid.max()
    if mn == mx:
        return pd.Series([0.5] * len(feature_series), index=feature_series.index)
    return (feature_series - mn) / (mx - mn)


def standardize_feature(feature_series: pd.Series) -> pd.Series:
    """Z-score standardise a feature (mean=0, std=1)."""
    valid = feature_series.dropna()
    if len(valid) == 0:
        return feature_series
    mu, sigma = valid.mean(), valid.std()
    if sigma == 0:
        return pd.Series([0.0] * len(feature_series), index=feature_series.index)
    return (feature_series - mu) / sigma


def prepare_for_regression(
    df:             pd.DataFrame,
    outcome_col:    str,
    feature_cols:   Optional[List[str]] = None,
    standardize:    bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare (X, y) matrices for multiple linear regression.

    Rows with any missing values in X or y are dropped.  Optionally
    standardises all features and the outcome.

    Args:
        df:            Feature DataFrame.
        outcome_col:   Name of the outcome (dependent) variable.
        feature_cols:  Predictor columns; None = all numeric except outcome
                       and identifier columns.
        standardize:   If True, z-score standardise X and y.

    Returns:
        (X_df, y_series) ready for OLS fitting.
    """
    id_cols = {"countyFIPS", "County Name", "State", "county_name",
               "state", "census_region_name", "census_division_name",
               "rucc_group", "is_metro", "hpsa_shortage_flag"}

    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != outcome_col and c not in id_cols
        ]

    X = df[feature_cols].copy()
    y = df[outcome_col].copy()

    valid = (~X.isna().any(axis=1)) & y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    if standardize:
        for col in X.columns:
            X[col] = standardize_feature(X[col])
        y = standardize_feature(y)

    return X, y


def prepare_for_clustering(
    df:             pd.DataFrame,
    feature_cols:   Optional[List[str]] = None,
    standardize:    bool = True,
) -> pd.DataFrame:
    """
    Prepare a feature matrix for county clustering (K-means, hierarchical).

    Args:
        df:            Feature DataFrame.
        feature_cols:  Columns to use; None = all per-100k and pct columns.
        standardize:   If True, z-score standardise all features.

    Returns:
        DataFrame with rows=counties (no NaN), columns=features.
        The countyFIPS column is retained as the index.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if any(kw in c for kw in ["per_100k", "pct_", "rate", "_pct"])
        ]

    X = df.set_index("countyFIPS")[feature_cols].copy() if "countyFIPS" in df.columns \
        else df[feature_cols].copy()
    X = X.dropna()

    if standardize:
        for col in X.columns:
            X[col] = standardize_feature(X[col])

    return X


if __name__ == "__main__":
    from tools import load_data
    from ahrf_loader import build_ahrf_feature_table

    print("Loading COVID data...")
    cases, deaths, pop = load_data()
    covid_fips = set(cases[cases["countyFIPS"] != "00000"]["countyFIPS"].unique())

    print("Loading AHRF feature table...")
    ahrf_df, _ = build_ahrf_feature_table(covid_fips=covid_fips, verbose=False)

    print("Building master county table...")
    master, diag = create_master_county_table(cases, deaths, pop, ahrf_df)
    print(f"Master table: {master.shape}")

    print("\nSample correlation (pcp_per_100k vs deaths_per_100k):")
    res = compute_bivariate_correlation(master, "pcp_per_100k", "deaths_per_100k")
    print(f"  n={res['n']}, Pearson r={res['pearson_r']:.3f}, p={res['pearson_p']:.4f}")
    print(f"  Spearman r={res['spearman_r']:.3f}, p={res['spearman_p']:.4f}")
    print(f"  R² = {res['r_squared']:.3f}")
