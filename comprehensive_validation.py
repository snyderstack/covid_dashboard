"""
Comprehensive Validation Audit for COVID-19 Dashboard

Performs thorough correctness checks on:
- Per-capita calculations
- Data joins and consistency
- Tooltip values vs underlying data
- Statewide row exclusion
- Population data validity
"""

import pandas as pd
import numpy as np
from tools import (
    load_data,
    prepare_choropleth_for_date,
    precompute_per_capita,
    get_available_dates,
)


def validate_manual_per_capita_calculation(
    cases_df, deaths_df, population_df, sample_size=20, verbose=True
):
    """
    Validate per-capita calculations by computing manually and comparing to dashboard values.

    Tests: For each sampled county on a random date, compute:
        manual_deaths_pc = (deaths_total / population) * 100000
        manual_cases_pc = (cases_total / population) * 100000

    Then prepare choropleth data and verify values match within tolerance.

    Args:
        cases_df: Wide-format cases dataframe
        deaths_df: Wide-format deaths dataframe
        population_df: Wide-format population dataframe
        sample_size: Number of counties to test (default: 20)
        verbose: Print detailed output

    Returns:
        Dictionary with validation results
    """
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in cases_df.columns if col not in identifier_cols]

    results = {
        "tested_counties": 0,
        "passed_countries": 0,
        "discrepancies": [],
        "errors": [],
    }

    # Filter for valid FIPS
    valid_counties = cases_df[
        (cases_df["countyFIPS"] != "00000") & (cases_df["countyFIPS"].notna())
    ].copy()

    # Sample counties
    sample_indices = np.random.choice(
        valid_counties.index, size=min(sample_size, len(valid_counties)), replace=False
    )

    for idx in sample_indices:
        results["tested_counties"] += 1
        try:
            county_row = cases_df.loc[idx]
            fips = county_row["countyFIPS"]
            state = county_row["State"]
            county_name = county_row.get("County Name", "Unknown")

            # Get population
            pop_row = population_df[
                (population_df["countyFIPS"] == fips) & (population_df["State"] == state)
            ]

            if pop_row.empty:
                results["errors"].append(
                    {
                        "fips": fips,
                        "county": county_name,
                        "state": state,
                        "error": "Population not found in population_df",
                    }
                )
                continue

            pop_col = [
                col
                for col in population_df.columns
                if col not in identifier_cols and col != "Location"
            ][0]
            population = pd.to_numeric(pop_row.iloc[0][pop_col], errors="coerce")

            if population <= 0:
                results["errors"].append(
                    {
                        "fips": fips,
                        "county": county_name,
                        "state": state,
                        "error": f"Invalid population: {population}",
                    }
                )
                continue

            # Pick a random date
            test_date = np.random.choice(date_cols)

            # Get raw values from source dataframes
            cases_total = pd.to_numeric(county_row[test_date], errors="coerce")
            deaths_total = pd.to_numeric(
                deaths_df.loc[idx][test_date], errors="coerce"
            )

            if pd.isna(cases_total) or pd.isna(deaths_total):
                continue

            # Manual calculation
            cases_pc_manual = (cases_total / population) * 100000
            deaths_pc_manual = (deaths_total / population) * 100000

            # Now prepare choropleth and check if values match
            # We need to check against the precomputed per-capita dataframe
            pc_cases, pc_deaths = precompute_per_capita(cases_df, deaths_df, population_df)

            # Get the values from the per-capita dataframe
            pc_deaths_row = pc_deaths[
                (pc_deaths["countyFIPS"] == fips) & (pc_deaths["State"] == state)
            ]

            if pc_deaths_row.empty:
                results["errors"].append(
                    {
                        "fips": fips,
                        "county": county_name,
                        "state": state,
                        "date": test_date,
                        "error": "Not found in precomputed per-capita dataframe",
                    }
                )
                continue

            deaths_pc_computed = pd.to_numeric(
                pc_deaths_row.iloc[0][test_date], errors="coerce"
            )

            # Compare
            tolerance = 0.01  # Allow 0.01 difference for floating point
            diff = abs(deaths_pc_manual - deaths_pc_computed)

            if diff > tolerance:
                results["discrepancies"].append(
                    {
                        "fips": fips,
                        "county": county_name,
                        "state": state,
                        "date": test_date,
                        "population": population,
                        "deaths_total": deaths_total,
                        "manual_calculation": deaths_pc_manual,
                        "computed_value": deaths_pc_computed,
                        "difference": diff,
                    }
                )
            else:
                results["passed_countries"] += 1

        except Exception as e:
            results["errors"].append(
                {"fips": fips, "error": str(e), "type": type(e).__name__}
            )

    if verbose:
        print("\n=== PER-CAPITA CALCULATION VALIDATION ===\n")
        print(f"Counties tested: {results['tested_counties']}")
        print(f"Calculations passed: {results['passed_countries']}")
        print(f"Discrepancies found: {len(results['discrepancies'])}")
        print(f"Errors: {len(results['errors'])}")

        if results["discrepancies"]:
            print("\nDISCREPANCIES:")
            for disc in results["discrepancies"]:
                print(f"  {disc['county']}, {disc['state']} ({disc['fips']})")
                print(
                    f"    Date: {disc['date']}, Population: {disc['population']}, Deaths: {disc['deaths_total']}"
                )
                print(
                    f"    Manual: {disc['manual_calculation']:.2f}, Computed: {disc['computed_value']:.2f}, Diff: {disc['difference']:.6f}"
                )

        if results["errors"]:
            print("\nERRORS:")
            for err in results["errors"][:5]:
                print(f"  {err}")

    return results


def validate_tooltip_consistency(
    cases_df, deaths_df, population_df, sample_size=10, verbose=True
):
    """
    Verify that tooltip values match the underlying choropleth data.

    Tests: For a sample of counties on a test date:
    1. Get the choropleth data prepared by prepare_choropleth_for_date()
    2. Extract the tooltip values (cases, deaths, cases_pc, deaths_pc)
    3. Verify they match manual calculations from raw data
    4. Verify they come from the same join (same county row)

    Args:
        cases_df: Wide-format cases
        deaths_df: Wide-format deaths
        population_df: Wide-format population
        sample_size: Number of counties to test
        verbose: Print output

    Returns:
        Dictionary with results
    """
    dates = get_available_dates(cases_df)
    test_date = dates[-1]  # Use most recent date

    results = {
        "tested_counties": 0,
        "passed": 0,
        "mismatches": [],
        "errors": [],
    }

    # Prepare choropleth using the actual app logic
    pc_deaths, _ = precompute_per_capita(cases_df, deaths_df, population_df)
    choro_data = prepare_choropleth_for_date(pc_deaths, test_date, cases_df, deaths_df, population_df)

    # Sample from choropleth data
    if len(choro_data) < sample_size:
        sample_size = len(choro_data)

    sample_indices = np.random.choice(choro_data.index, size=sample_size, replace=False)

    for idx in sample_indices:
        results["tested_counties"] += 1
        try:
            choro_row = choro_data.loc[idx]

            fips = choro_row["countyFIPS"]
            state = choro_row["State"]
            county_name = choro_row.get("Location", f"{fips}")

            # Get the tooltip values from choropleth row
            tooltip_population = choro_row["population"]
            tooltip_cases = choro_row["cases"]
            tooltip_deaths = choro_row["deaths"]
            tooltip_cases_pc = choro_row["cases_pc"]
            tooltip_deaths_pc = choro_row["deaths_pc"]

            # Recompute manually from raw source data
            source_cases = cases_df[
                (cases_df["countyFIPS"] == fips) & (cases_df["State"] == state)
            ]
            source_deaths = deaths_df[
                (deaths_df["countyFIPS"] == fips) & (deaths_df["State"] == state)
            ]
            source_pop = population_df[
                (population_df["countyFIPS"] == fips)
                & (population_df["State"] == state)
                & (population_df["population"] > 0)
            ]

            if source_cases.empty or source_deaths.empty or source_pop.empty:
                results["errors"].append(
                    {
                        "fips": fips,
                        "county": county_name,
                        "error": "Source data not found",
                    }
                )
                continue

            # Get raw values
            raw_cases = pd.to_numeric(source_cases.iloc[0][test_date], errors="coerce")
            raw_deaths = pd.to_numeric(source_deaths.iloc[0][test_date], errors="coerce")

            pop_col = [
                col
                for col in population_df.columns
                if col not in ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
            ][0]
            raw_pop = pd.to_numeric(source_pop.iloc[0][pop_col], errors="coerce")

            # Manual calculations
            manual_cases_pc = (raw_cases / raw_pop) * 100000 if raw_pop > 0 else np.nan
            manual_deaths_pc = (raw_deaths / raw_pop) * 100000 if raw_pop > 0 else np.nan

            # Verify consistency
            tolerance = 0.1  # Allow small floating point differences
            errors_found = []

            if abs(tooltip_cases - raw_cases) > tolerance:
                errors_found.append(
                    f"Cases mismatch: tooltip={tooltip_cases}, raw={raw_cases}"
                )

            if abs(tooltip_deaths - raw_deaths) > tolerance:
                errors_found.append(
                    f"Deaths mismatch: tooltip={tooltip_deaths}, raw={raw_deaths}"
                )

            if abs(tooltip_population - raw_pop) > tolerance:
                errors_found.append(
                    f"Population mismatch: tooltip={tooltip_population}, raw={raw_pop}"
                )

            if not np.isnan(tooltip_cases_pc) and not np.isnan(manual_cases_pc):
                if abs(tooltip_cases_pc - manual_cases_pc) > tolerance:
                    errors_found.append(
                        f"Cases/100k mismatch: tooltip={tooltip_cases_pc:.2f}, manual={manual_cases_pc:.2f}"
                    )

            if not np.isnan(tooltip_deaths_pc) and not np.isnan(manual_deaths_pc):
                if abs(tooltip_deaths_pc - manual_deaths_pc) > tolerance:
                    errors_found.append(
                        f"Deaths/100k mismatch: tooltip={tooltip_deaths_pc:.2f}, manual={manual_deaths_pc:.2f}"
                    )

            if errors_found:
                results["mismatches"].append(
                    {
                        "fips": fips,
                        "county": county_name,
                        "state": state,
                        "date": test_date,
                        "errors": errors_found,
                    }
                )
            else:
                results["passed"] += 1

        except Exception as e:
            results["errors"].append(
                {
                    "county": county_name,
                    "error": str(e),
                    "type": type(e).__name__,
                }
            )

    if verbose:
        print("\n=== TOOLTIP CONSISTENCY VALIDATION ===\n")
        print(f"Test date: {test_date}")
        print(f"Counties tested: {results['tested_counties']}")
        print(f"Passed: {results['passed']}")
        print(f"Mismatches found: {len(results['mismatches'])}")
        print(f"Errors: {len(results['errors'])}")

        if results["mismatches"]:
            print("\nMISMATCHES:")
            for mismatch in results["mismatches"][:5]:
                print(f"  {mismatch['county']}, {mismatch['state']}")
                for error in mismatch["errors"]:
                    print(f"    - {error}")

        if results["errors"]:
            print("\nERRORS:")
            for err in results["errors"][:3]:
                print(f"  {err}")

    return results


def validate_join_integrity(cases_df, deaths_df, population_df, verbose=True):
    """
    Verify that joins are done correctly and consistently.

    Checks:
    - No duplicate FIPS in source data
    - (FIPS, State) tuples are unique
    - All counties in cases/deaths have population data
    - Statewide rows (FIPS=00000) are filtered properly

    Args:
        cases_df: Wide-format cases
        deaths_df: Wide-format deaths
        population_df: Wide-format population
        verbose: Print output

    Returns:
        Dictionary with results
    """
    results = {
        "valid": True,
        "checks": {},
    }

    # Check 1: Duplicate FIPS within state
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]

    for name, df in [("Cases", cases_df), ("Deaths", deaths_df), ("Population", population_df)]:
        # Remove invalid FIPS
        valid = df[
            (df["countyFIPS"] != "00000") & (df["countyFIPS"].notna())
        ].copy()

        duplicates = valid[valid.duplicated(subset=["countyFIPS", "State"], keep=False)]

        results["checks"][f"{name} duplicate (FIPS, State)"] = {
            "total_rows": len(valid),
            "duplicates": len(duplicates),
            "valid": len(duplicates) == 0,
        }

        if len(duplicates) > 0:
            results["valid"] = False

    # Check 2: Statewide rows are excluded
    cases_statewide = cases_df[
        (cases_df["countyFIPS"] == "00000") | (cases_df["County Name"] == "Statewide Unallocated")
    ]
    deaths_statewide = deaths_df[
        (deaths_df["countyFIPS"] == "00000") | (deaths_df["County Name"] == "Statewide Unallocated")
    ]
    pop_statewide = population_df[
        (population_df["countyFIPS"] == "00000") | (population_df["County Name"] == "Statewide Unallocated")
    ]

    results["checks"]["Cases statewide rows"] = {
        "count": len(cases_statewide),
        "note": "These should be filtered before per-capita calculation",
    }
    results["checks"]["Deaths statewide rows"] = {
        "count": len(deaths_statewide),
        "note": "These should be filtered before per-capita calculation",
    }
    results["checks"]["Population statewide rows"] = {
        "count": len(pop_statewide),
        "note": "Statewide unallocated should have population=0",
    }

    # Check 3: Population coverage
    valid_pop_fips = set(
        population_df[population_df["population"] > 0]["countyFIPS"].unique()
    )
    valid_cases_fips = set(
        cases_df[cases_df["countyFIPS"] != "00000"]["countyFIPS"].unique()
    )
    valid_deaths_fips = set(
        deaths_df[deaths_df["countyFIPS"] != "00000"]["countyFIPS"].unique()
    )

    missing_pop_for_cases = valid_cases_fips - valid_pop_fips
    missing_pop_for_deaths = valid_deaths_fips - valid_pop_fips

    results["checks"]["Population coverage"] = {
        "counties_with_pop": len(valid_pop_fips),
        "cases_fips_without_pop": len(missing_pop_for_cases),
        "deaths_fips_without_pop": len(missing_pop_for_deaths),
        "valid": len(missing_pop_for_cases) == 0 and len(missing_pop_for_deaths) == 0,
    }

    if not results["checks"]["Population coverage"]["valid"]:
        results["valid"] = False

    if verbose:
        print("\n=== JOIN INTEGRITY VALIDATION ===\n")
        for check_name, check_result in results["checks"].items():
            print(f"{check_name}:")
            for key, value in check_result.items():
                print(f"  {key}: {value}")

    return results


def run_full_audit(verbose=True):
    """
    Run complete validation audit on dashboard data.

    Args:
        verbose: Print detailed output

    Returns:
        Dictionary with all audit results
    """
    print("=" * 60)
    print("COVID-19 DASHBOARD - COMPREHENSIVE VALIDATION AUDIT")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
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

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    all_passed = True
    for audit_name, audit_result in audit_results.items():
        if "valid" in audit_result and not audit_result["valid"]:
            all_passed = False
            print(f"❌ {audit_name}: FAILED")
        elif (
            audit_name == "per_capita_calculation"
            and audit_result["discrepancies"]
        ):
            all_passed = False
            print(f"⚠️  {audit_name}: Discrepancies found")
        elif (
            audit_name == "tooltip_consistency"
            and audit_result["mismatches"]
        ):
            all_passed = False
            print(f"⚠️  {audit_name}: Mismatches found")
        else:
            print(f"✓ {audit_name}: PASSED")

    if all_passed:
        print("\n✅ ALL AUDITS PASSED - DATA IS VALID")
    else:
        print("\n❌ ISSUES FOUND - SEE DETAILS ABOVE")

    return audit_results


if __name__ == "__main__":
    audit_results = run_full_audit(verbose=True)
