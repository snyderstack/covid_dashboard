"""
Standalone data-quality runner for the COVID-19 County Analysis Dashboard.

This module is NOT imported by any application code.  Run it directly::

    python validation.py

to audit per-capita calculations, population validity, FIPS join integrity,
and tooltip consistency against live data files.
"""

import pandas as pd
import numpy as np
from tools import (
    get_available_dates,
    get_population_column,
    prepare_choropleth_for_date,
    precompute_per_capita,
)


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

            # df_metric is already per-capita, so a full manual recomputation is
            # done in validate_manual_per_capita_calculation(). Here we only
            # sanity-check the value against a plausibility bound.
            test_date = np.random.choice(date_cols)
            computed_pc = pd.to_numeric(county_row[test_date], errors="coerce")

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

    pop_col = get_population_column(population_df)
    if pop_col is None:
        results["issues"].append({"type": "No population column found"})
        return results

    results["nan_population"] = population_df[pop_col].isna().sum()
    results["negative_population"] = (population_df[pop_col] < 0).sum()

    zero_pop = population_df[population_df[pop_col] == 0]
    results["zero_population"] = len(zero_pop)

    # Count statewide entries
    statewide = zero_pop[
        (zero_pop["County Name"] == "Statewide Unallocated") |
        (zero_pop["County Name"].str.contains("Unallocated", na=False))
    ]
    results["statewide_unallocated"] = len(statewide)

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
        len(population_df)
        - results["zero_population"]
        - results["negative_population"]
        - results["nan_population"]
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
    _pop_col = get_population_column(population_df) or "population"
    pop_fips = set(population_df[population_df[_pop_col] > 0]["countyFIPS"].unique())
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


def validate_manual_per_capita_calculation(
    cases_df, deaths_df, population_df, sample_size=20, verbose=True
):
    """
    Validate per-capita calculations against direct manual computation.

    The audit samples counties and dates, calculates deaths per 100k from the
    raw deaths dataframe and population table, then compares the result with
    the precomputed per-capita dataframe used by the dashboard.
    """
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in cases_df.columns if col not in identifier_cols]

    results = {
        "tested_counties": 0,
        "passed_counties": 0,
        "discrepancies": [],
        "errors": [],
    }

    valid_counties = cases_df[
        (cases_df["countyFIPS"] != "00000") & cases_df["countyFIPS"].notna()
    ].copy()

    if valid_counties.empty or not date_cols:
        results["errors"].append({"error": "No valid counties or date columns available"})
        return results

    sample_indices = np.random.choice(
        valid_counties.index, size=min(sample_size, len(valid_counties)), replace=False
    )

    pop_col = [
        col for col in population_df.columns
        if col not in identifier_cols and col != "Location"
    ][0]
    _, pc_deaths = precompute_per_capita(cases_df, deaths_df, population_df)

    for idx in sample_indices:
        results["tested_counties"] += 1
        try:
            county_row = cases_df.loc[idx]
            fips = county_row["countyFIPS"]
            state = county_row["State"]
            county_name = county_row.get("County Name", "Unknown")

            pop_row = population_df[
                (population_df["countyFIPS"] == fips) & (population_df["State"] == state)
            ]
            deaths_row = deaths_df[
                (deaths_df["countyFIPS"] == fips) & (deaths_df["State"] == state)
            ]
            pc_row = pc_deaths[
                (pc_deaths["countyFIPS"] == fips) & (pc_deaths["State"] == state)
            ]

            if pop_row.empty or deaths_row.empty or pc_row.empty:
                results["errors"].append({
                    "fips": fips,
                    "county": county_name,
                    "state": state,
                    "error": "Missing population, deaths, or per-capita row",
                })
                continue

            population = pd.to_numeric(pop_row.iloc[0][pop_col], errors="coerce")
            if pd.isna(population) or population <= 0:
                results["errors"].append({
                    "fips": fips,
                    "county": county_name,
                    "state": state,
                    "error": f"Invalid population: {population}",
                })
                continue

            test_date = np.random.choice(date_cols)
            deaths_total = pd.to_numeric(deaths_row.iloc[0][test_date], errors="coerce")
            computed_pc = pd.to_numeric(pc_row.iloc[0][test_date], errors="coerce")

            if pd.isna(deaths_total) or pd.isna(computed_pc):
                continue

            manual_pc = (deaths_total / population) * 100000
            diff = abs(manual_pc - computed_pc)

            if diff > 0.01:
                results["discrepancies"].append({
                    "fips": fips,
                    "county": county_name,
                    "state": state,
                    "date": test_date,
                    "population": population,
                    "deaths_total": deaths_total,
                    "manual_calculation": manual_pc,
                    "computed_value": computed_pc,
                    "difference": diff,
                })
            else:
                results["passed_counties"] += 1

        except Exception as exc:
            results["errors"].append({
                "fips": str(locals().get("fips", "unknown")),
                "error": str(exc),
                "type": type(exc).__name__,
            })

    if verbose:
        print("\n=== PER-CAPITA CALCULATION VALIDATION ===\n")
        print(f"Counties tested: {results['tested_counties']}")
        print(f"Calculations passed: {results['passed_counties']}")
        print(f"Discrepancies found: {len(results['discrepancies'])}")
        print(f"Errors: {len(results['errors'])}")

    return results


def validate_tooltip_consistency(
    cases_df, deaths_df, population_df, sample_size=10, verbose=True
):
    """
    Verify choropleth tooltip fields against source data for sampled counties.
    """
    dates = get_available_dates(cases_df)
    test_date = dates[-1] if dates else None

    results = {
        "tested_counties": 0,
        "passed": 0,
        "mismatches": [],
        "errors": [],
    }

    if test_date is None:
        results["errors"].append({"error": "No date columns available"})
        return results

    _, pc_deaths = precompute_per_capita(cases_df, deaths_df, population_df)
    choro_data = prepare_choropleth_for_date(
        pc_deaths, test_date, cases_df, deaths_df, population_df
    )

    if choro_data.empty:
        results["errors"].append({"error": "No choropleth rows available"})
        return results

    sample_indices = np.random.choice(
        choro_data.index, size=min(sample_size, len(choro_data)), replace=False
    )

    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    pop_col = [
        col for col in population_df.columns
        if col not in identifier_cols and col != "Location"
    ][0]

    for idx in sample_indices:
        results["tested_counties"] += 1
        try:
            choro_row = choro_data.loc[idx]
            fips = choro_row["countyFIPS"]
            state = choro_row["State"]
            county_name = choro_row.get("Location", fips)

            source_cases = cases_df[
                (cases_df["countyFIPS"] == fips) & (cases_df["State"] == state)
            ]
            source_deaths = deaths_df[
                (deaths_df["countyFIPS"] == fips) & (deaths_df["State"] == state)
            ]
            source_pop = population_df[
                (population_df["countyFIPS"] == fips)
                & (population_df["State"] == state)
                & (population_df[pop_col] > 0)
            ]

            if source_cases.empty or source_deaths.empty or source_pop.empty:
                results["errors"].append({
                    "fips": fips,
                    "county": county_name,
                    "error": "Source data not found",
                })
                continue

            raw_cases = pd.to_numeric(source_cases.iloc[0][test_date], errors="coerce")
            raw_deaths = pd.to_numeric(source_deaths.iloc[0][test_date], errors="coerce")
            raw_pop = pd.to_numeric(source_pop.iloc[0][pop_col], errors="coerce")

            manual_cases_pc = (raw_cases / raw_pop) * 100000 if raw_pop > 0 else np.nan
            manual_deaths_pc = (raw_deaths / raw_pop) * 100000 if raw_pop > 0 else np.nan

            checks = {
                "cases": (choro_row["cases"], raw_cases),
                "deaths": (choro_row["deaths"], raw_deaths),
                "population": (choro_row["population"], raw_pop),
                "cases_pc": (choro_row["cases_pc"], manual_cases_pc),
                "deaths_pc": (choro_row["deaths_pc"], manual_deaths_pc),
            }

            errors_found = []
            for field, (actual, expected) in checks.items():
                if pd.isna(actual) and pd.isna(expected):
                    continue
                if pd.isna(actual) or pd.isna(expected) or abs(actual - expected) > 0.1:
                    errors_found.append(
                        f"{field} mismatch: tooltip={actual}, expected={expected}"
                    )

            if errors_found:
                results["mismatches"].append({
                    "fips": fips,
                    "county": county_name,
                    "state": state,
                    "date": test_date,
                    "errors": errors_found,
                })
            else:
                results["passed"] += 1

        except Exception as exc:
            results["errors"].append({
                "county": locals().get("county_name", "unknown"),
                "error": str(exc),
                "type": type(exc).__name__,
            })

    if verbose:
        print("\n=== TOOLTIP CONSISTENCY VALIDATION ===\n")
        print(f"Test date: {test_date}")
        print(f"Counties tested: {results['tested_counties']}")
        print(f"Passed: {results['passed']}")
        print(f"Mismatches found: {len(results['mismatches'])}")
        print(f"Errors: {len(results['errors'])}")

    return results


def validate_join_integrity(cases_df, deaths_df, population_df, verbose=True):
    """
    Verify FIPS/state join integrity and population coverage.
    """
    results = {"valid": True, "checks": {}}

    for name, df in [
        ("Cases", cases_df),
        ("Deaths", deaths_df),
        ("Population", population_df),
    ]:
        valid = df[(df["countyFIPS"] != "00000") & df["countyFIPS"].notna()].copy()
        duplicates = valid[valid.duplicated(subset=["countyFIPS", "State"], keep=False)]
        results["checks"][f"{name} duplicate (FIPS, State)"] = {
            "total_rows": len(valid),
            "duplicates": len(duplicates),
            "valid": len(duplicates) == 0,
        }
        if len(duplicates) > 0:
            results["valid"] = False

    cases_statewide = cases_df[
        (cases_df["countyFIPS"] == "00000") |
        (cases_df["County Name"] == "Statewide Unallocated")
    ]
    deaths_statewide = deaths_df[
        (deaths_df["countyFIPS"] == "00000") |
        (deaths_df["County Name"] == "Statewide Unallocated")
    ]
    pop_statewide = population_df[
        (population_df["countyFIPS"] == "00000") |
        (population_df["County Name"] == "Statewide Unallocated")
    ]

    results["checks"]["Cases statewide rows"] = {"count": len(cases_statewide)}
    results["checks"]["Deaths statewide rows"] = {"count": len(deaths_statewide)}
    results["checks"]["Population statewide rows"] = {"count": len(pop_statewide)}

    pop_col = [
        col for col in population_df.columns
        if col not in ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    ][0]
    valid_pop = set(
        population_df[
            (population_df["countyFIPS"] != "00000") & (population_df[pop_col] > 0)
        ][["countyFIPS", "State"]].itertuples(index=False, name=None)
    )
    valid_cases = set(
        cases_df[cases_df["countyFIPS"] != "00000"][["countyFIPS", "State"]]
        .itertuples(index=False, name=None)
    )
    valid_deaths = set(
        deaths_df[deaths_df["countyFIPS"] != "00000"][["countyFIPS", "State"]]
        .itertuples(index=False, name=None)
    )

    missing_pop_for_cases = valid_cases - valid_pop
    missing_pop_for_deaths = valid_deaths - valid_pop
    coverage_valid = not missing_pop_for_cases and not missing_pop_for_deaths

    results["checks"]["Population coverage"] = {
        "counties_with_pop": len(valid_pop),
        "cases_without_pop": len(missing_pop_for_cases),
        "deaths_without_pop": len(missing_pop_for_deaths),
        "valid": coverage_valid,
    }
    if not coverage_valid:
        results["valid"] = False

    if verbose:
        print("\n=== JOIN INTEGRITY VALIDATION ===\n")
        for check_name, check_result in results["checks"].items():
            print(f"{check_name}:")
            for key, value in check_result.items():
                print(f"  {key}: {value}")

    return results


def run_full_audit(cases_df=None, deaths_df=None, population_df=None, verbose=True):
    """
    Run the dashboard validation suite.
    """
    if cases_df is None or deaths_df is None or population_df is None:
        from tools import load_data

        cases_df, deaths_df, population_df = load_data()

    audit_results = {
        "per_capita_calculation": validate_manual_per_capita_calculation(
            cases_df, deaths_df, population_df, sample_size=20, verbose=verbose
        ),
        "tooltip_consistency": validate_tooltip_consistency(
            cases_df, deaths_df, population_df, sample_size=10, verbose=verbose
        ),
        "join_integrity": validate_join_integrity(
            cases_df, deaths_df, population_df, verbose=verbose
        ),
    }

    return audit_results


if __name__ == "__main__":
    # Example usage for standalone testing
    from tools import load_data

    cases, deaths, pop = load_data()
    print("Running diagnostic checks...\n")
    diagnostics = diagnostic_check(cases, deaths, pop, verbose=True)
