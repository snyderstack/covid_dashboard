# COVID-19 County Outcomes Analysis Platform

An interactive Streamlit dashboard for county-level COVID-19 analysis in the United States. The platform joins USAFacts case, death, and population data with HRSA Area Health Resources Files (AHRF) and CDC county-level vaccination data, providing geographic mapping, trend comparison, wave detection, case-to-death lag analysis, and statistical modeling across 3,000+ counties.

Developed at Gettysburg College for public-health analysis, coursework, and exploratory research.

## Dashboard Tabs

**County Overview** — the landing page: a public health fact sheet for any county covering COVID outcomes, detected waves, case-to-death lag, healthcare capacity, socioeconomic context, vaccination status, and national-median comparisons. Includes a "Counties Like This One" structural-peer finder, a random-county button, a downloadable one-page HTML report, and shareable `?county=` URLs.

**Geographic Map** — county choropleth with a date slider, cumulative/daily/moving-average/per-capita COVID metrics, vaccination metrics, state zoom, Metro/Nonmetro filtering (USDA RUCC), and configurable color scaling (percentile clip, absolute, log). Optional animated monthly playback and Getis-Ord Gi* hotspot analysis over county contiguity.

**County Comparison** — overlay two counties, a county against the national aggregate, or all three. Supports cumulative and daily views, smoothing, per-100k normalization, index rebasing, dual axes, log scale, and vaccination rollout comparison.

**Wave Analysis** — region-based epidemiological wave detection with three sensitivity presets. Reports each wave's onset, peak, duration, burden, significance score (0–100), and vaccination coverage at the peak, with a validation panel comparing detected waves against national surge windows. Advanced controls expose the legacy prominence-based detector.

**Time Lag Analysis** — detects peaks in smoothed daily cases and deaths per 100k, matches each case peak to the nearest subsequent death peak, and reports the lag in days plus a severity ratio for each matched pair. Supports county vs county comparison.

**County Factors** — scatter-plot explorer relating COVID outcomes to healthcare access, income, education, demographics, rural-urban classification, and vaccination rates, with Pearson/Spearman statistics, OLS trend lines, and factor correlation rankings. Outcomes can be restricted to a date window (pre-vaccine, post-rollout, or custom) — essential when relating vaccination factors to outcomes.

**Statistical Modeling** — correlation matrices, Random Forest feature importance with partial-dependence curves, multivariable OLS regression with HC3 heteroscedasticity-robust standard errors and VIF multicollinearity diagnostics, cross-validated county resilience scores, vaccination efficacy analysis, and K-means county archetype clustering. Shares the outcome-window control with County Factors.

## Data Sources

| Dataset | Files | Coverage |
|---|---|---|
| USAFacts COVID-19 cases/deaths/population | `data/covid_confirmed_usafacts.csv`, `data/covid_deaths_usafacts.csv`, `data/covid_county_population_usafacts.csv` | Jan 2020 – Jul 2023 |
| HRSA Area Health Resources Files | `data/ahrf2023.csv` (primary), `data/AHRF2020.asc`, `data/AHRF_2020-2021_SAS/AHRF2021.sas7bdat` (supplementary) | 2018 – 2023 vintages |
| CDC county vaccination | `data/COVID-19_Vaccinations_in_the_United_States,County_20260623.csv` | Dec 2020 – May 2023 |

All datasets are read from the local `data/` directory; no dataset is downloaded at runtime (the small county boundary file described below is the single exception). All joins use five-character zero-padded county FIPS codes (with `(countyFIPS, State)` compound keys where duplicate FIPS rows exist).

The USAFacts CSVs and `ahrf2023.csv` are included in the repository, so the dashboard runs immediately after cloning. Larger files (the CDC vaccination CSV and supplementary AHRF releases) exceed GitHub size limits and must be downloaded separately — see `data/README.md` for sources and exact paths. The dashboard degrades when they are absent: vaccination features are hidden and AHRF falls back to the primary 2023 CSV.

The US county boundary file (`data/geojson-counties-fips.json`, ~2 MB) is downloaded automatically on first launch if missing and reused offline afterwards; it powers the maps, hotspot analysis, and archetype maps.

## Installation

Requires Python 3.10 or newer. From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

Streamlit prints the local URL, usually `http://localhost:8501`. First launch takes longer while the AHRF and vaccination tables are loaded and startup transforms are computed; results are cached for the session.

## Project Structure

```text
covid_dashboard/
├── app.py                  # Streamlit UI: layout, controls, charts
├── tools.py                # Data loading, transforms, choropleth prep, national series
├── wave_analysis.py        # Region-based epidemic wave detection
├── lag_analysis.py         # Case-to-death peak lag analysis
├── ahrf_loader.py          # AHRF loading, column selection, derived rates
├── vaccination_loader.py   # CDC vaccination data loading and lookups
├── county_features.py      # Master county feature table, correlations, similarity
├── modeling.py             # Correlations, RF, OLS (HC3), VIF, clustering, resilience
├── spatial_analysis.py     # County adjacency and Getis-Ord Gi* hotspots
├── validation.py           # Standalone data-quality audits (python validation.py)
├── tests/                  # Deterministic pytest suite (synthetic fixtures)
├── assets/                 # Logos
├── data/                   # Local source datasets (see Data Sources)
├── README.md
├── TECHNICAL_DOCUMENTATION.md
└── requirements.txt
```

`app.py` holds all Streamlit layout and visualization logic; the other modules are pure data functions with no Streamlit dependency (aside from caching wrappers defined in `app.py`).

## Testing

```bash
pip install pytest
pytest tests/ -q
```

The suite uses small synthetic datasets, so it runs in seconds and requires none of the files in `data/`. It covers the daily-diff and per-capita pipeline, national aggregation, window outcomes, wave detection and significance scoring, lag matching, similarity search, VIF/HC3, clustering, and the spatial statistics.

## Interpretation Notes

All analyses are county-level (ecological) associations, not individual-level or causal effects. Per-capita rates use a single static population per county. Vaccination joins use the final (May 2023) snapshot for cumulative-outcome analyses, and time-series lookups where a date-specific value is needed. See `TECHNICAL_DOCUMENTATION.md` for methodology, assumptions, and the change log.
