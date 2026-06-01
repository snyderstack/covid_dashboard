"""
County-level feature table for COVID-19 analysis and external data integration.

Provides a modular architecture for aggregating county-level metrics and
merging with external datasets (healthcare access, socioeconomic, demographic).

Future use cases:
- Correlation analysis (Pearson, Spearman)
- Linear regression on outcomes
- County clustering
- Rural vs urban comparisons
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_county_feature_table(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    population_df: pd.DataFrame,
    wave_metrics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create master county-level feature table with current metrics.

    Args:
        cases_df: Cases dataframe (wide format, cumulative)
        deaths_df: Deaths dataframe (wide format, cumulative)
        population_df: Population dataframe
        wave_metrics_df: Optional wave metrics dataframe from wave_analysis module

    Returns:
        DataFrame with one row per county and columns:
        - countyFIPS, County Name, State, Population
        - Total Cases, Total Deaths
        - Cases per 100k, Deaths per 100k
        - Wave metrics (if provided)
    """
    # Get most recent date for totals
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in cases_df.columns if col not in identifier_cols]
    latest_date = sorted(date_cols)[-1]

    # Initialize with population and FIPS
    pop_cols = [col for col in population_df.columns if col not in identifier_cols and col != "Location"]
    pop_col = pop_cols[0] if pop_cols else "population"

    features = population_df[[
        "countyFIPS", "County Name", "State", pop_col
    ]].rename(columns={pop_col: "population"}).copy()

    # Filter out statewide entries and zero population
    features = features[(features["countyFIPS"] != "00000") & (features["population"] > 0)].copy()
    features = features.reset_index(drop=True)

    # Add total cases and deaths
    cases_totals = cases_df[["countyFIPS", "State", latest_date]].rename(
        columns={latest_date: "total_cases"}
    )
    deaths_totals = deaths_df[["countyFIPS", "State", latest_date]].rename(
        columns={latest_date: "total_deaths"}
    )

    features = features.merge(cases_totals, on=["countyFIPS", "State"], how="left")
    features = features.merge(deaths_totals, on=["countyFIPS", "State"], how="left")

    # Calculate per-capita metrics
    features["total_cases"] = pd.to_numeric(features["total_cases"], errors="coerce")
    features["total_deaths"] = pd.to_numeric(features["total_deaths"], errors="coerce")

    features["cases_per_100k"] = np.where(
        (features["population"] > 0) & features["population"].notna(),
        (features["total_cases"] / features["population"]) * 100000,
        np.nan
    )
    features["deaths_per_100k"] = np.where(
        (features["population"] > 0) & features["population"].notna(),
        (features["total_deaths"] / features["population"]) * 100000,
        np.nan
    )

    # Merge wave metrics if provided
    if wave_metrics_df is not None and len(wave_metrics_df) > 0:
        wave_cols = [col for col in wave_metrics_df.columns
                     if col not in ["countyFIPS", "County Name", "State"]]
        features = features.merge(
            wave_metrics_df[["countyFIPS", "State"] + wave_cols],
            on=["countyFIPS", "State"],
            how="left"
        )

    return features


def add_external_dataset(
    features_df: pd.DataFrame,
    external_df: pd.DataFrame,
    external_fips_col: str = "countyFIPS",
    join_state: bool = True,
) -> pd.DataFrame:
    """
    Merge external county-level dataset into feature table.

    Args:
        features_df: County feature table (from create_county_feature_table)
        external_df: External dataset to merge
        external_fips_col: Name of FIPS column in external_df
        join_state: If True, join on (FIPS, State); if False, join on FIPS only

    Returns:
        Updated feature table with external columns added
    """
    result = features_df.copy()

    # Standardize FIPS in external data
    if external_fips_col in external_df.columns:
        external_df = external_df.copy()
        external_df[external_fips_col] = (
            pd.to_numeric(external_df[external_fips_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )

        if join_state and "State" in external_df.columns:
            # Join on (FIPS, State)
            merge_cols = [external_fips_col, "State"]
            external_df = external_df.rename(columns={external_fips_col: "countyFIPS"})
            result = result.merge(
                external_df,
                on=merge_cols,
                how="left",
                suffixes=("", "_external")
            )
        else:
            # Join on FIPS only
            external_df = external_df.rename(columns={external_fips_col: "countyFIPS"})
            result = result.merge(
                external_df,
                on="countyFIPS",
                how="left",
                suffixes=("", "_external")
            )

    return result


def normalize_feature(feature_series: pd.Series) -> pd.Series:
    """
    Normalize a feature to [0, 1] range using min-max normalization.

    Args:
        feature_series: Series of feature values

    Returns:
        Normalized series
    """
    valid_values = feature_series.dropna()
    if len(valid_values) == 0:
        return feature_series

    min_val = valid_values.min()
    max_val = valid_values.max()

    if min_val == max_val:
        return pd.Series([0.5] * len(feature_series), index=feature_series.index)

    return (feature_series - min_val) / (max_val - min_val)


def standardize_feature(feature_series: pd.Series) -> pd.Series:
    """
    Standardize a feature using z-score normalization.

    Args:
        feature_series: Series of feature values

    Returns:
        Standardized series (mean=0, std=1)
    """
    valid_values = feature_series.dropna()
    if len(valid_values) == 0:
        return feature_series

    mean_val = valid_values.mean()
    std_val = valid_values.std()

    if std_val == 0:
        return pd.Series([0] * len(feature_series), index=feature_series.index)

    return (feature_series - mean_val) / std_val


# ===== FUTURE ANALYTICS SKELETON =====
# These functions are templates for future implementation.
# They establish the architecture for analytics that will use the feature table.


def prepare_for_correlation_analysis(
    features_df: pd.DataFrame,
    outcome_column: str,
    exclude_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare feature table for correlation analysis (Pearson/Spearman).

    Future implementation: Compute Pearson and Spearman correlations
    between features and outcome variable.

    Args:
        features_df: County feature table
        outcome_column: Column name of outcome variable (e.g., 'deaths_per_100k')
        exclude_columns: Columns to exclude from analysis

    Returns:
        DataFrame ready for correlation analysis
    """
    if exclude_columns is None:
        exclude_columns = ["countyFIPS", "County Name", "State"]

    analysis_df = features_df.drop(columns=exclude_columns, errors="ignore").copy()

    # Remove rows where outcome is missing
    analysis_df = analysis_df[analysis_df[outcome_column].notna()].copy()

    # TODO: Implement Pearson and Spearman correlation computation
    # TODO: Compute p-values and effect sizes
    # TODO: Return ranked feature importance

    return analysis_df


def prepare_for_regression_analysis(
    features_df: pd.DataFrame,
    outcome_column: str,
    feature_columns: Optional[List[str]] = None,
    standardize: bool = True,
) -> tuple:
    """
    Prepare feature table for multiple linear regression.

    Future implementation: Fit OLS regression, compute R², coefficients, p-values.

    Args:
        features_df: County feature table
        outcome_column: Column name of outcome variable
        feature_columns: Specific columns to use as features; if None, use all except identifiers
        standardize: If True, standardize features and outcome

    Returns:
        Tuple of (X, y) ready for regression
    """
    if feature_columns is None:
        exclude = ["countyFIPS", "County Name", "State", outcome_column]
        feature_columns = [col for col in features_df.columns if col not in exclude]

    X = features_df[feature_columns].copy()
    y = features_df[outcome_column].copy()

    # Remove rows with missing data
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    if standardize:
        for col in X.columns:
            X[col] = standardize_feature(X[col])
        y = standardize_feature(y)

    # TODO: Implement OLS regression fitting
    # TODO: Compute R², adjusted R², coefficients, p-values, VIF
    # TODO: Model diagnostics (residuals, heteroscedasticity, normality)

    return X, y


def prepare_for_clustering_analysis(
    features_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Prepare feature table for county clustering (K-means, hierarchical, etc.).

    Future implementation: Determine optimal number of clusters, fit clustering model,
    assign cluster labels.

    Args:
        features_df: County feature table
        feature_columns: Specific columns for clustering; if None, use COVID metrics
        standardize: If True, standardize features

    Returns:
        DataFrame with clustering features prepared
    """
    if feature_columns is None:
        feature_columns = [
            col for col in features_df.columns
            if "cases" in col.lower() or "deaths" in col.lower() or "wave" in col.lower()
        ]

    X = features_df[feature_columns].copy()

    # Remove rows with missing data
    X = X.dropna()

    if standardize:
        for col in X.columns:
            X[col] = standardize_feature(X[col])

    # TODO: Implement clustering (K-means with optimal k, hierarchical, etc.)
    # TODO: Compute silhouette scores and elbow curve
    # TODO: Assign county cluster labels

    return X


def prepare_for_rural_urban_analysis(
    features_df: pd.DataFrame,
    rural_urban_column: str = "rural_urban_class",
) -> pd.DataFrame:
    """
    Prepare feature table for rural vs urban outcome comparison.

    Future implementation: Compare outcome metrics (deaths per 100k, case rates, etc.)
    across rural/urban categories.

    Args:
        features_df: County feature table
        rural_urban_column: Column name indicating rural/urban classification

    Returns:
        DataFrame ready for rural/urban analysis
    """
    if rural_urban_column not in features_df.columns:
        # TODO: Add rural/urban classification using population density or USDA categories
        pass

    # TODO: Implement statistical comparisons (t-tests, ANOVA)
    # TODO: Compute summary statistics by category
    # TODO: Visualize outcome distributions

    return features_df


def prepare_for_feature_importance_analysis(
    features_df: pd.DataFrame,
    outcome_column: str,
) -> pd.DataFrame:
    """
    Prepare feature table for feature importance analysis.

    Future implementation: Use tree-based models or other methods to rank feature importance.

    Args:
        features_df: County feature table
        outcome_column: Column name of outcome variable

    Returns:
        DataFrame with features ranked by importance
    """
    # TODO: Implement feature importance using:
    # - Random Forest permutation importance
    # - SHAP values
    # - Univariate statistical tests (correlation, Mann-Whitney U, etc.)
    # TODO: Return ranked feature list with importance scores

    return features_df


if __name__ == "__main__":
    # Example usage
    from tools import load_data, precompute_daily_diffs
    from wave_analysis import calculate_waves_for_all_counties

    print("Creating county feature table...")
    cases, deaths, pop = load_data()
    daily_cases, daily_deaths = precompute_daily_diffs(cases, deaths)

    # Create base feature table
    features = create_county_feature_table(cases, deaths, pop)
    print(f"Created feature table with {len(features)} counties")
    print(f"Columns: {features.columns.tolist()}")

    # Example: Show a few counties
    print(f"\nSample counties:")
    print(features[["County Name", "State", "population", "total_cases",
                    "total_deaths", "cases_per_100k", "deaths_per_100k"]].head(10).to_string())
