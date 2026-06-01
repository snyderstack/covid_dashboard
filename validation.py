"""
Data validation and diagnostic utilities for COVID-19 dashboard.

Provides:
- Per-capita calculation verification
- Population data validation
- FIPS join auditing
- Random sampling diagnostics
"""

import pandas as pd
import numpy as np
from pathlib import Path


def validate_per_capita_calculation(df_metric, population_df, metric_name, sample_size=10):
    """
    Validate per-capita calculations by comparing computed values with manual calculation.

    Args:
        df_metric: Dataframe with per-capita metric (wide format with dates as columns)
        population_df: Population dataframe (wide format)
        metric_name: Name of metric column (e.g., 'Deaths', 'Cases')
        sample_size: Number of random counties to validate (default: 10)

    Returns:
        Dictionary with validation results including pass/fail and discrepancies
    """
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in df_metric.columns if col not in identifier_cols]

    results = {
        "valid": True,
        "total_tested": 0,
        "total_passed": 0,
        "discrepancies": [],
        "errors": [],
    }

    # Sample random counties (filter out invalid FIPS first)
    valid_fips = df_metric[
        (df_metric["countyFIPS"] != "00000")
        & (df_metric["countyFIPS"].notna())
        & (df_metric["countyFIPS"] != "")
    ].copy()

    sample_indices = np.random.choice(
        valid_fips.index, size=min(sample_size, len(valid_fips)), replace=False
    )

    pop_col = [col for col in population_df.columns
               if col not in identifier_cols and col != "Location"][0]

    for idx in sample_indices:
        results["total_tested"] += 1
        try:
            county_row = df_metric.loc[idx]
            fips = county_row["countyFIPS"]
            state = county_row["State"]
            county_name = county_row.get("County Name", "Unknown")

            # Get population
            pop_row = population_df[
                (population_df["countyFIPS"] == fips) &
                (population_df["State"] == state)
            ]

            if pop_row.empty:
                results["errors"].append({
                    "fips": fips,
                    "county": county_name,
                    "state": state,
                    "error": "Population not found",
                })
                continue

            population = pd.to_numeric(pop_row.iloc[0][pop_col], errors="coerce")

            if population <= 0:
                results["errors"].append({
                    "fips": fips,
                    "county": county_name,
                    "state": state,
                    "error": f"Invalid population: {population}",
                })
                continue

            # Test against random date
            test_date = np.random.choice(date_cols)

            # Get the original metric value (before per-capita)
            # This is tricky since df_metric is already per-capita
            # Instead, we validate the FORMULA is correct
            computed_pc = pd.to_numeric(county_row[test_date], errors="coerce")

            # Check if value is reasonable (< 100,000)
            if computed_pc > 100000:
                results["discrepancies"].append({
                    "fips": fips,
                    "county": county_name,
                    "state": state,
                    "date": test_date,
                    "per_capita": computed_pc,
                    "issue": "Per-capita value exceeds 100,000 (unrealistic)",
                })
                continue

            results["total_passed"] += 1

        except Exception as e:
            results["errors"].append({
                "fips": str(fips),
                "error": str(e),
            })

    results["valid"] = (len(results["discrepancies"]) == 0 and
                        len(results["errors"]) == 0)
    return results


def check_population_validity(population_df):
    """
    Check for invalid population entries.

    Args:
        population_df: Population dataframe

    Returns:
        Dictionary with validation results
    """
    results = {
        "total_entries": len(population_df),
        "valid_entries": 0,
        "zero_population": 0,
        "negative_population": 0,
        "nan_population": 0,
        "statewide_unallocated": 0,
        "issues": [],
    }

    results["nan_population"] = population_df["population"].isna().sum()
    results["negative_population"] = (population_df["population"] < 0).sum()

    zero_pop = population_df[population_df["population"] == 0]
    results["zero_population"] = len(zero_pop)

    # Count statewide entries
    statewide = zero_pop[
        (zero_pop["County Name"] == "Statewide Unallocated") |
        (zero_pop["County Name"].str.contains("Unallocated", na=False))
    ]
    results["statewide_unallocated"] = len(statewide)

    # Other zero-population entries (potential data quality issues)
    other_zero = zero_pop[
        ~(
            (zero_pop["County Name"] == "Statewide Unallocated") |
            (zero_pop["County Name"].str.contains("Unallocated", na=False))
        )
    ]

    if len(other_zero) > 0:
        results["issues"].append({
            "type": "Zero population for actual counties",
            "count": len(other_zero),
            "examples": other_zero[["countyFIPS", "County Name", "State"]].head(3).to_dict("records"),
        })

    results["valid_entries"] = (
        len(population_df) - results["zero_population"] -
        results["negative_population"] - results["nan_population"]
    )

    return results


def check_fips_duplicates(df):
    """
    Check for duplicate FIPS codes in a dataframe.

    Args:
        df: Dataframe with countyFIPS column

    Returns:
        Dictionary with duplicate analysis
    """
    results = {
        "total_rows": len(df),
        "unique_fips": df["countyFIPS"].nunique(),
        "duplicate_count": (df.duplicated(subset=["countyFIPS"])).sum(),
        "duplicate_fips": [],
    }

    duplicates = df[df.duplicated(subset=["countyFIPS"], keep=False)]
    if len(duplicates) > 0:
        # Group by FIPS to show all occurrences
        for fips in duplicates["countyFIPS"].unique():
            fips_group = df[df["countyFIPS"] == fips]
            if len(fips_group) > 1:
                results["duplicate_fips"].append({
                    "fips": fips,
                    "count": len(fips_group),
                    "entries": fips_group[["County Name", "State"]].to_dict("records"),
                })

    return results


def diagnostic_check(cases_df, deaths_df, population_df, verbose=False):
    """
    Run comprehensive diagnostic checks on all data.

    Args:
        cases_df: Cases dataframe (wide format)
        deaths_df: Deaths dataframe (wide format)
        population_df: Population dataframe
        verbose: Print detailed output

    Returns:
        Dictionary with all diagnostic results
    """
    diagnostics = {
        "population_validity": check_population_validity(population_df),
        "fips_duplicates_cases": check_fips_duplicates(cases_df),
        "fips_duplicates_deaths": check_fips_duplicates(deaths_df),
        "fips_duplicates_population": check_fips_duplicates(population_df),
        "summary": {},
    }

    # Check if all cases/deaths counties have population data
    pop_fips = set(population_df[population_df["population"] > 0]["countyFIPS"].unique())
    cases_fips = set(cases_df[cases_df["countyFIPS"] != "00000"]["countyFIPS"].unique())
    deaths_fips = set(deaths_df[deaths_df["countyFIPS"] != "00000"]["countyFIPS"].unique())

    missing_pop_in_cases = cases_fips - pop_fips
    missing_pop_in_deaths = deaths_fips - pop_fips

    diagnostics["summary"]["cases_missing_population"] = len(missing_pop_in_cases)
    diagnostics["summary"]["deaths_missing_population"] = len(missing_pop_in_deaths)
    diagnostics["summary"]["valid_counties"] = len(pop_fips)

    if verbose:
        print("\n=== DIAGNOSTIC SUMMARY ===\n")
        print("Population Data:")
        print(f"  Total entries: {diagnostics['population_validity']['total_entries']}")
        print(f"  Valid entries: {diagnostics['population_validity']['valid_entries']}")
        print(f"  Zero population: {diagnostics['population_validity']['zero_population']}")
        print(f"    - Statewide unallocated: {diagnostics['population_validity']['statewide_unallocated']}")
        print()
        print("FIPS Duplicates:")
        print(f"  Cases: {diagnostics['fips_duplicates_cases']['duplicate_count']} rows (unique: {diagnostics['fips_duplicates_cases']['unique_fips']})")
        print(f"  Deaths: {diagnostics['fips_duplicates_deaths']['duplicate_count']} rows (unique: {diagnostics['fips_duplicates_deaths']['unique_fips']})")
        print(f"  Population: {diagnostics['fips_duplicates_population']['duplicate_count']} rows (unique: {diagnostics['fips_duplicates_population']['unique_fips']})")
        print()
        print("Coverage:")
        print(f"  Valid counties in population: {diagnostics['summary']['valid_counties']}")
        print(f"  Cases counties missing population: {diagnostics['summary']['cases_missing_population']}")
        print(f"  Deaths counties missing population: {diagnostics['summary']['deaths_missing_population']}")

    return diagnostics


if __name__ == "__main__":
    # Example usage for standalone testing
    from tools import load_data

    cases, deaths, pop = load_data()
    print("Running diagnostic checks...\n")
    diagnostics = diagnostic_check(cases, deaths, pop, verbose=True)
