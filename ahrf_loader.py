"""
AHRF data loader for the COVID-19 County Analysis Dashboard.

Loads and standardises Area Health Resources Files (AHRF) data for county-level
COVID outcome analysis. Three sources are combined:

  ahrf2023.csv (primary) — CSV, 3,231 rows × 4,306 columns; most variables are
    2021/2022 vintage; 100% FIPS match with the USAFacts COVID dataset.

  AHRF2020.asc (supplementary) — fixed-width text, parsed with the
    AHRF_2019-2020/DOC/AHRF2019-2020.sas layout; provides pandemic-era 2018-2020
    physician and HPSA variables not yet in the 2023 CSV.

  AHRF2021.sas7bdat (supplementary) — SAS binary, 3,230 rows × 7,418 columns;
    provides 2019-2021 validation data using f-number variable names.

Note: ahrf2021.asc and ahrf2022.asc are present in data/ but are not parsed
because no fixed-width layout files exist for those release years. The 2023 CSV
already covers those variable vintages.

All sources are joined to the COVID dataset on 5-character zero-padded countyFIPS
strings. AHRF variable names carry a 2-digit year suffix (e.g., _21 = 2021).
"""

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# 2019-2020 SAS layout (covers AHRF2020.asc)
SAS_LAYOUT_2020 = DATA_DIR / "AHRF_2019-2020" / "DOC" / "AHRF2019-2020.sas"
ASC_2020        = DATA_DIR / "AHRF2020.asc"
SAS7BDAT_2021   = DATA_DIR / "AHRF_2020-2021_SAS" / "AHRF2021.sas7bdat"
CSV_2023        = DATA_DIR / "ahrf2023.csv"

# Selected columns from ahrf2023.csv.  Format: csv_column_name → clean_output_name.
# Year suffix conventions:
#   _21 = data year 2021   _20 = data year 2020 or Census 2020
#   _23 = data year 2023   _13 = RUCC classification year 2013
CSV_2023_COLUMNS = {
    # Geographic identifiers
    "fips_st_cnty":                   "fips_raw",
    "st_name_abbrev":                 "state",
    "cnty_name":                      "county_name",
    "cens_regn":                      "census_region_code",
    "cens_regn_name":                 "census_region_name",
    "cens_divsn":                     "census_division_code",
    "cens_divsn_name":                "census_division_name",

    # Rural-urban classification (USDA 2013 vintage — most recent available)
    "rural_urban_contnm_13":          "rucc_code",          # 1-9 scale
    "urban_influnc_13":               "urban_influence_code",# 1-12 scale
    "cbsa_ind_20":                    "cbsa_type",           # 0=nonmetro,1=metro,2=micro

    # Healthcare workforce — 2021
    "phys_nf_prim_care_pc_exc_rsdt_21": "primary_care_physicians",  # count
    "md_nf_activ_21":                   "total_active_md",           # count
    "md_nf_all_med_spec_21":            "total_specialists",         # count

    # Hospital & facility capacity — 2021
    "hosp_beds_21":                   "hospital_beds",         # total beds all facilities
    "stgh_med_surg_icu_beds_21":      "icu_beds",              # med-surg ICU only
    "snf_beds_21":                    "snf_beds",              # skilled nursing facility
    "critcl_access_hosp_21":          "critical_access_hospitals",  # count CAHs
    "hosp_21":                        "total_hospitals",

    # Healthcare shortage designations — 2023
    "hpsa_prim_care_23":              "hpsa_primary_care",    # 0=none,1=whole,2=partial
    "hpsa_mentl_hlth_23":             "hpsa_mental_health",
    "hpsa_dent_23":                   "hpsa_dental",

    # Population — Census 2020 preferred for denominators
    "cens_popn_20":                   "population_2020",
    "popn_est_ge65_21":               "pop_65plus",
    "popn_densty_per_squr_mi_20":     "pop_density_per_sqmi",
    "land_area_mi2_20":               "land_area_sq_mi",
    "medn_age_20":                    "median_age",
    "urban_popn_pct_20":              "pct_urban_pop",

    # Economic — 2021
    "medn_famly_incom_21":            "median_family_income",
    "per_cap_persnl_incom_21":        "per_capita_income",
    "unemply_rate_ge16_21":           "unemployment_rate",
    "child_povty_famls_5_17_pct_21":  "child_poverty_pct",
    "prstnt_povty_typolgy_14":        "persistent_poverty_flag",   # binary
    "hi_povty_typolgy_14":            "high_poverty_flag",          # binary

    # Education — 2021
    "pers_lt_hsd_ge25_pct_21":        "pct_no_hs_diploma",
    "pers_ge_hsd_ge25_pct_21":        "pct_hs_diploma_or_higher",
    "pers_4yrs_collg_ge25_pct_21":    "pct_college_4yr",

    # Background mortality — 3-year average centred on 2021
    "mort_3yr_65_74_avg_21":          "mortality_rate_65_74",
    "mort_3yr_55_64_avg_21":          "mortality_rate_55_64",
}

# Supplementary columns from AHRF2020.asc (adds 2020-era HPSA and historical data)
ASC_2020_COLUMNS = {
    # variable code → (output_name, description)
    "f00002":   ("fips_raw",           "FIPS St & Cty Code"),
    "f12424":   ("state",              "State Name Abbreviation"),
    "f00010":   ("county_name",        "County Name"),
    "f0002013": ("rucc_code",          "Rural-Urban Continuum Code 2013"),
    "f1255913": ("urban_influence_code","Urban Influence Code 2013"),
    "f0978720": ("hpsa_pc_2020",       "HPSA Code Primary Care 2020"),
    "f0978719": ("hpsa_pc_2019",       "HPSA Code Primary Care 2019"),
    "f0892118": ("hospital_beds_2018", "Hospital Beds Total Hospitals 2018"),
    "f1467518": ("pcp_2018",           "Primary Care Physicians (non-fed, excl residents) 2018"),
    "f0978118": ("per_cap_income_2018","Per Capita Personal Income 2018"),
}


def _pad_fips(series: pd.Series) -> pd.Series:
    """Zero-pad FIPS codes to 5-character strings."""
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )


def _parse_sas_layout(sas_path: Path) -> dict:
    """
    Parse a HRSA/AHRF SAS format file and return a dict of
    variable_name → {start, end, width, type, desc}.

    Field positions in the SAS file are 1-based; output is 0-based.
    """
    with open(sas_path, "r", encoding="latin-1") as f:
        content = f.read()

    pattern = r"@(\d+)\s+(\w+)\s+(\$)?\s+(\d+)\.\s+/\*(.+?)\*/"
    specs = {}
    for pos, varname, is_char, width, desc in re.findall(pattern, content):
        specs[varname] = {
            "start": int(pos) - 1,
            "end":   int(pos) - 1 + int(width),
            "width": int(width),
            "type":  "char" if is_char else "num",
            "desc":  desc.strip(),
        }
    return specs


def load_ahrf_2023(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load priority variables from ahrf2023.csv.

    Returns a DataFrame with standardised column names and 5-char FIPS.
    """
    csv_path = (data_dir or DATA_DIR) / "ahrf2023.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ahrf2023.csv not found at {csv_path}")

    available_cols = list(CSV_2023_COLUMNS.keys())
    df = pd.read_csv(csv_path, usecols=available_cols,
                     encoding="latin-1", low_memory=False)

    df = df.rename(columns=CSV_2023_COLUMNS)
    df["countyFIPS"] = _pad_fips(df["fips_raw"])
    df = df.drop(columns=["fips_raw"])

    str_cols = {"state", "county_name", "census_region_name",
                "census_division_name", "countyFIPS"}
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def parse_ahrf_2020_asc(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse AHRF2020.asc using the AHRF_2019-2020/DOC/AHRF2019-2020.sas layout.

    Returns a DataFrame containing a subset of priority columns available
    in the 2020 ASC file (pandemic-era 2018-2020 data).
    """
    d         = data_dir or DATA_DIR
    asc_path  = d / "AHRF2020.asc"
    sas_path  = d / "AHRF_2019-2020" / "DOC" / "AHRF2019-2020.sas"

    if not asc_path.exists():
        warnings.warn(f"AHRF2020.asc not found at {asc_path} — skipping 2020 ASC source")
        return pd.DataFrame()
    if not sas_path.exists():
        warnings.warn(f"AHRF2019-2020.sas not found at {sas_path} — skipping 2020 ASC source")
        return pd.DataFrame()

    specs = _parse_sas_layout(sas_path)

    fwf_specs, names, output_names = [], [], []
    for varcode, (outname, _) in ASC_2020_COLUMNS.items():
        if varcode in specs:
            s = specs[varcode]
            fwf_specs.append((s["start"], s["end"]))
            names.append(varcode)
            output_names.append(outname)
        else:
            warnings.warn(f"Variable {varcode} not found in 2020 SAS layout — skipping")

    df = pd.read_fwf(
        asc_path,
        colspecs=fwf_specs,
        names=names,
        encoding="latin-1",
        header=None,
    )

    rename_map = {vc: on for vc, (on, _) in ASC_2020_COLUMNS.items() if vc in df.columns}
    df = df.rename(columns=rename_map)
    df["countyFIPS"] = _pad_fips(df.get("fips_raw", df.get("countyFIPS")))

    str_cols = {"state", "county_name", "countyFIPS"}
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def load_ahrf_2021_sas(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load priority columns from AHRF2021.sas7bdat.

    Maps f-number column names to human-readable names using the 2020 SAS
    label file.  Only the columns we can identify are retained.

    Returns a DataFrame with standardised column names and 5-char FIPS.
    """
    d         = data_dir or DATA_DIR
    sas_path  = d / "AHRF_2020-2021_SAS" / "AHRF2021.sas7bdat"
    layout_p  = d / "AHRF_2019-2020" / "DOC" / "AHRF2019-2020.sas"

    if not sas_path.exists():
        warnings.warn(f"AHRF2021.sas7bdat not found at {sas_path} — skipping 2021 SAS source")
        return pd.DataFrame()

    # f-number → output name; same variables as the 2020 ASC but may carry 2020/2021 data
    sas21_cols = {
        "f00002":   "fips_raw",
        "f12424":   "state",
        "f00010":   "county_name",
        "f0002013": "rucc_code_2021_src",
        "f1255913": "urban_influence_code_2021_src",
        "f0978720": "hpsa_pc_2020_sas",
        "f0892118": "hospital_beds_2018_sas",
        "f1467518": "pcp_2018_sas",
        "f0978118": "per_cap_income_2018_sas",
        "f0679519": "unemployment_rate_2019_sas",
        "f1434614": "median_family_income_2014_sas",
        "f1484010": "pop_65plus_2010_sas",
        "f0972110": "land_area_sq_mi_2010_sas",
        "f1387610": "pop_density_2010_sas",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_sas(sas_path)

    keep = [c for c in sas21_cols if c in df.columns]
    df = df[keep].copy()

    for col in df.columns:  # decode bytes columns from SAS reader
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("latin-1").strip() if isinstance(x, bytes)
                else (str(x).strip() if pd.notna(x) else "")
            )

    df = df.rename(columns={k: v for k, v in sas21_cols.items() if k in df.columns})
    df["countyFIPS"] = _pad_fips(df.get("fips_raw", df.get("countyFIPS")))

    str_cols = {"state", "county_name", "countyFIPS", "fips_raw"}
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def _compute_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-100k rate columns and categorical flags derived from raw counts.

    All rates use population_2020 as the denominator (Census 2020 count),
    which provides the most accurate baseline for the 2020-2023 pandemic period.

    Derived columns added:
        pcp_per_100k              Primary care physicians per 100k population
        total_md_per_100k         All active MDs per 100k
        hospital_beds_per_100k    Hospital beds per 100k
        icu_beds_per_100k         ICU beds per 100k
        snf_beds_per_100k         SNF beds per 100k
        pct_pop_65plus            Population 65+ as % of total population
        hpsa_shortage_flag        True if county has any HPSA designation
        rucc_group                'Metro' (RUCC 1-3) or 'Nonmetro' (RUCC 4-9)
        is_metro                  Boolean: RUCC 1-3
    """
    df = df.copy()
    pop = df.get("population_2020", pd.Series(dtype=float))

    def per100k(count_col: str, pop_series: pd.Series) -> pd.Series:
        counts = pd.to_numeric(df.get(count_col, pd.Series(dtype=float)), errors="coerce")
        return np.where(
            (pop_series > 0) & pop_series.notna(),
            (counts / pop_series) * 100_000,
            np.nan,
        )

    pop_s = pd.to_numeric(pop, errors="coerce")

    if "primary_care_physicians" in df.columns:
        df["pcp_per_100k"] = per100k("primary_care_physicians", pop_s)
    if "total_active_md" in df.columns:
        df["total_md_per_100k"] = per100k("total_active_md", pop_s)
    if "hospital_beds" in df.columns:
        df["hospital_beds_per_100k"] = per100k("hospital_beds", pop_s)
    if "icu_beds" in df.columns:
        df["icu_beds_per_100k"] = per100k("icu_beds", pop_s)
    if "snf_beds" in df.columns:
        df["snf_beds_per_100k"] = per100k("snf_beds", pop_s)

    if "pop_65plus" in df.columns:
        df["pct_pop_65plus"] = np.where(
            (pop_s > 0) & pop_s.notna(),
            (pd.to_numeric(df["pop_65plus"], errors="coerce") / pop_s) * 100,
            np.nan,
        )

    if "hpsa_primary_care" in df.columns:
        df["hpsa_shortage_flag"] = (
            pd.to_numeric(df["hpsa_primary_care"], errors="coerce") >= 1
        )

    if "rucc_code" in df.columns:
        rucc = pd.to_numeric(df["rucc_code"], errors="coerce")
        df["is_metro"]   = rucc.between(1, 3)
        df["rucc_group"] = np.where(rucc.between(1, 3), "Metro", "Nonmetro")
        df["rucc_group"] = df["rucc_group"].where(rucc.notna(), pd.NA)

    return df


def _run_diagnostics(df: pd.DataFrame, covid_fips: set, label: str) -> dict:
    """Return a diagnostic dict for a loaded/joined DataFrame."""
    ahrf_fips  = set(df["countyFIPS"].dropna().unique())
    matched    = covid_fips & ahrf_fips
    unmatched  = covid_fips - ahrf_fips
    duplicates = df[df.duplicated(subset=["countyFIPS"], keep=False)]["countyFIPS"].unique().tolist()

    key_cols = [
        "rucc_code", "pcp_per_100k", "hospital_beds_per_100k",
        "median_family_income", "unemployment_rate", "pct_no_hs_diploma",
        "pop_density_per_sqmi",
    ]
    missing_pct = {}
    for col in key_cols:
        if col in df.columns:
            missing_pct[col] = round(100 * df[col].isna().mean(), 1)

    return {
        "source":             label,
        "total_rows":         len(df),
        "covid_fips_count":   len(covid_fips),
        "matched_fips":       len(matched),
        "match_rate_pct":     round(100 * len(matched) / len(covid_fips), 1) if covid_fips else 0,
        "unmatched_covid":    sorted(unmatched)[:10],
        "duplicate_fips":     duplicates[:10],
        "missing_pct":        missing_pct,
    }


def build_ahrf_feature_table(
    data_dir:    Optional[Path] = None,
    covid_fips:  Optional[set]  = None,
    verbose:     bool            = True,
) -> tuple:
    """
    Build the master AHRF county feature table.

    Strategy:
    1. Load ahrf2023.csv as the primary source (most complete, most recent).
    2. Parse AHRF2020.asc for supplementary 2018-2020 variables (if available).
    3. Load AHRF2021.sas7bdat for supplementary 2019-2021 validation (if available).
    4. Compute all derived per-100k rates and categorical flags.
    5. Filter to counties that exist in the COVID dataset (if covid_fips provided).

    Args:
        data_dir:   Path to DATA/ directory. Defaults to ../DATA relative to this file.
        covid_fips: Set of 5-char FIPS codes from the COVID dataset for validation.
        verbose:    Print progress and diagnostics to stdout.

    Returns:
        (feature_df, diagnostics_dict) where feature_df has one row per county
        and diagnostics_dict contains match/coverage statistics.
    """
    d = data_dir or DATA_DIR
    diag = {}

    if verbose:
        print("Loading ahrf2023.csv (primary source)...")
    df = load_ahrf_2023(d)
    if verbose:
        print(f"  → {len(df):,} rows, {len(df.columns)} columns")

    if verbose:
        print("Parsing AHRF2020.asc (supplementary 2020 data)...")
    df20 = parse_ahrf_2020_asc(d)
    if not df20.empty:
        supp_cols = [c for c in df20.columns
                     if c not in {"countyFIPS", "state", "county_name", "fips_raw"}
                     and c not in df.columns]
        if supp_cols:
            df = df.merge(df20[["countyFIPS"] + supp_cols],
                          on="countyFIPS", how="left")
        if verbose:
            print(f"  → Added {len(supp_cols)} supplementary 2020 columns: {supp_cols}")
    else:
        if verbose:
            print("  → 2020 ASC not available; skipping")

    if verbose:
        print("Loading AHRF2021.sas7bdat (supplementary 2021 validation data)...")
    df21 = load_ahrf_2021_sas(d)
    if not df21.empty:
        supp_cols21 = [c for c in df21.columns
                       if c not in {"countyFIPS", "state", "county_name", "fips_raw"}
                       and c not in df.columns]
        if supp_cols21:
            df = df.merge(df21[["countyFIPS"] + supp_cols21],
                          on="countyFIPS", how="left")
        if verbose:
            print(f"  → Added {len(supp_cols21)} supplementary 2021 columns: {supp_cols21}")
    else:
        if verbose:
            print("  → 2021 SAS7BDAT not available; skipping")

    if verbose:
        print("Computing derived per-100k rates and categorical flags...")
    df = _compute_derived_variables(df)

    if covid_fips:
        n_before = len(df)
        df = df[df["countyFIPS"].isin(covid_fips)].copy()
        if verbose:
            print(f"  → Filtered to COVID counties: {len(df):,} of {n_before:,} AHRF rows retained")

    df = df.reset_index(drop=True)

    if covid_fips:
        diag = _run_diagnostics(df, covid_fips, "ahrf_master")
        if verbose:
            print(f"\nDiagnostics:")
            print(f"  Matched:      {diag['matched_fips']:,}/{diag['covid_fips_count']:,} "
                  f"({diag['match_rate_pct']:.1f}%)")
            print(f"  Unmatched:    {len(diag['unmatched_covid'])} COVID FIPS "
                  f"not in AHRF: {diag['unmatched_covid'][:5]}")
            print(f"  Duplicates:   {len(diag['duplicate_fips'])} duplicate FIPS "
                  f"{'(none)' if not diag['duplicate_fips'] else diag['duplicate_fips'][:3]}")
            print(f"  Missing data % for key columns:")
            for col, pct in diag["missing_pct"].items():
                print(f"    {col:<35s}: {pct:5.1f}%")

    return df, diag


def get_rucc_classification(
    ahrf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a RUCC-based county classification DataFrame compatible with
    the existing tools.classify_county_type() interface.

    The returned DataFrame has columns (countyFIPS, State, County_Type)
    where County_Type is 'Metro' (RUCC 1-3) or 'Nonmetro' (RUCC 4-9).

    This replaces the population-threshold approach in tools.py when
    AHRF data is available.

    Args:
        ahrf_df: The master AHRF feature DataFrame (output of
                 build_ahrf_feature_table).

    Returns:
        DataFrame with columns: countyFIPS, State, County_Type, rucc_code
    """
    needed = {"countyFIPS", "state", "rucc_code"}
    if not needed.issubset(ahrf_df.columns):
        raise ValueError(
            f"ahrf_df must contain columns: {needed}. "
            f"Got: {set(ahrf_df.columns)}"
        )

    result = ahrf_df[["countyFIPS", "state", "rucc_code"]].copy()
    result = result.rename(columns={"state": "State"})

    rucc = pd.to_numeric(result["rucc_code"], errors="coerce")
    result["County_Type"] = np.where(
        rucc.between(1, 3), "Metro",
        np.where(rucc.between(4, 9), "Nonmetro", pd.NA)
    )

    return result[["countyFIPS", "State", "County_Type", "rucc_code"]].reset_index(drop=True)


VARIABLE_CATALOG = {
    # name: (display_label, category, units, source_year, description)
    "pcp_per_100k":            ("Primary Care Physicians per 100k", "Healthcare", "per 100k pop", "2021", "Non-federal primary care physicians excluding hospital residents, per 100,000 population"),
    "total_md_per_100k":       ("Total MDs per 100k", "Healthcare", "per 100k pop", "2021", "All active non-federal MDs, per 100,000 population"),
    "hospital_beds_per_100k":  ("Hospital Beds per 100k", "Healthcare", "per 100k pop", "2021", "All hospital beds, per 100,000 population"),
    "icu_beds_per_100k":       ("ICU Beds per 100k", "Healthcare", "per 100k pop", "2021", "Med-surg ICU beds in short-term general hospitals, per 100,000 population"),
    "snf_beds_per_100k":       ("SNF Beds per 100k", "Healthcare", "per 100k pop", "2021", "Skilled nursing facility beds, per 100,000 population"),
    "hpsa_primary_care":       ("HPSA Primary Care", "Healthcare", "0/1/2", "2023", "Health Professional Shortage Area designation: 0=none, 1=whole county, 2=partial"),
    "critical_access_hospitals":("Critical Access Hospitals", "Healthcare", "count", "2021", "Number of Critical Access Hospitals in county"),
    "median_family_income":    ("Median Family Income", "Economic", "dollars", "2021", "Median family income in dollars"),
    "per_capita_income":       ("Per Capita Income", "Economic", "dollars", "2021", "Per capita personal income in dollars"),
    "unemployment_rate":       ("Unemployment Rate", "Economic", "percent", "2021", "Unemployment rate, civilian labor force age 16+"),
    "child_poverty_pct":       ("Child Poverty Rate", "Economic", "percent", "2021", "Percentage of families with children 5-17 below poverty line"),
    "pct_no_hs_diploma":       ("% Without HS Diploma", "Education", "percent", "2021", "Percentage of persons age 25+ without a high school diploma"),
    "pct_college_4yr":         ("% 4-Year College Degree", "Education", "percent", "2021", "Percentage of persons age 25+ with a 4-year college degree"),
    "pop_density_per_sqmi":    ("Population Density", "Demographics", "per sq mi", "2020", "Population per square mile, Census 2020"),
    "pct_pop_65plus":          ("% Population 65+", "Demographics", "percent", "2021", "Percentage of population age 65 and older"),
    "median_age":              ("Median Age", "Demographics", "years", "2020", "Median age of county population, Census 2020"),
    "pct_urban_pop":           ("% Urban Population", "Demographics", "percent", "2020", "Percentage of population in urbanized areas, Census 2020"),
    "rucc_code":               ("RUCC Code (1-9)", "Rural-Urban", "code", "2013", "USDA Rural-Urban Continuum Code: 1-3=Metro, 4-9=Nonmetro"),
    "is_metro":                ("Metro County", "Rural-Urban", "boolean", "2013", "True if RUCC code 1-3 (metropolitan)"),
    "mortality_rate_65_74":    ("Pre-COVID Mortality Rate 65-74", "Health Outcomes", "per 100k", "2021", "3-year average mortality rate per 100k for age 65-74"),
    "mortality_rate_55_64":    ("Pre-COVID Mortality Rate 55-64", "Health Outcomes", "per 100k", "2021", "3-year average mortality rate per 100k for age 55-64"),
    "persistent_poverty_flag": ("Persistent Poverty County", "Economic", "0/1", "2014", "USDA persistent poverty county typology flag"),
}


def get_variable_catalog() -> pd.DataFrame:
    """Return the variable catalog as a DataFrame for documentation and UI dropdowns."""
    rows = []
    for col, (label, category, units, year, desc) in VARIABLE_CATALOG.items():
        rows.append({
            "column":    col,
            "label":     label,
            "category":  category,
            "units":     units,
            "data_year": year,
            "description": desc,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("AHRF Loader — Standalone Validation")

    from tools import load_data

    cases, deaths, pop = load_data()
    covid_fips = set(
        cases[cases["countyFIPS"] != "00000"]["countyFIPS"].unique()
    )
    print(f"COVID dataset: {len(covid_fips)} unique county FIPS\n")

    feat_df, diagnostics = build_ahrf_feature_table(
        covid_fips=covid_fips, verbose=True
    )

    print(f"\nMaster table shape: {feat_df.shape}")
    print(f"Columns: {list(feat_df.columns)}")
    print(f"\nSample rows:")
    sample_cols = ["countyFIPS", "state", "county_name", "rucc_code",
                   "rucc_group", "pcp_per_100k", "hospital_beds_per_100k",
                   "unemployment_rate", "pct_no_hs_diploma"]
    print(feat_df[[c for c in sample_cols if c in feat_df.columns]].head(5).to_string())

    print(f"\nRUCC classification:")
    rucc_cls = get_rucc_classification(feat_df)
    print(rucc_cls["County_Type"].value_counts().to_string())
