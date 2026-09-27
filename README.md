# COVID-19 County Outcomes Analysis Platform

An interactive Streamlit dashboard for county-level COVID-19 analysis in the United States. The platform joins USAFacts case, death, and population data with HRSA Area Health Resources Files (AHRF) and CDC county-level vaccination data, providing geographic mapping, trend comparison, wave detection, case-to-death lag analysis, and statistical modeling across 3,000+ counties.

Developed at Gettysburg College for public-health analysis, coursework, and exploratory research. Built to be classroom-friendly: every tab carries a plain-language "Key terms" glossary, statistical controls have educational tooltips, and the methodology expanders show the actual formulas behind each statistic.

## Dashboard Tabs

**Geographic Map** — the landing view: a full-width county choropleth with a collapsible control panel, a date slider with Play/Pause auto-advance (four speeds), cumulative/daily/moving-average/per-capita COVID metrics, vaccination metrics, state-level political metrics (2020 presidential margin, governor party), state zoom, Metro/Nonmetro filtering (USDA RUCC), configurable color scaling (percentile clip, absolute, log), and a colorblind-safe palette option. Pan and zoom are kept when the date changes or playback advances, and reset when the metric, filter, or color scale changes. Click any county to load its Overview profile. Optional animated monthly playback, spatial clustering analysis (global Moran's I plus Getis-Ord Gi* hotspot mapping), and shareable `?metric=`/`?date=` URLs.

**County Overview** — a public health fact sheet for any county covering COVID outcomes, detected waves, case-to-death lag, healthcare capacity, socioeconomic context, vaccination status, and national- and peer-median comparisons. COVID rates carry exact-Poisson 95% confidence intervals so small-county uncertainty is visible, and a rolling case-fatality chart (with daily deaths per 100k in a second panel) shows how CFR evolved across testing eras and variants. Features a pandemic timeline of stacked panels on one shared time axis — cases with wave spans, peaks, and the structural-peer median; deaths; and vaccination rollout — and an automated Resilience Profile that places the county's mortality among its ten structural peers and identifies which characteristics distinguish it — with each factor's direction derived from its national association with mortality. A State Political Context section lists the state's governor, U.S. senators, legislature control, and 2020 presidential result as of January 2021. Also includes clickable navigation cards, a "Counties Like This One" peer finder with similarity explanations and bordering-county comparisons, classroom example presets, a random-county button, a downloadable HTML research report with methodology and limitations, and shareable `?county=` URLs.

**County Comparison** — overlay two counties, a county against the national aggregate, all three, or up to five counties at once (e.g., every large county in one state). Supports cumulative and daily views, smoothing, per-100k normalization, index rebasing, dual axes, log scale, and vaccination rollout comparison.

**Wave Analysis** — region-based epidemiological wave detection with three sensitivity presets. Reports each wave's onset, peak, duration, burden, significance score (0–100), and vaccination coverage at the peak, with a validation panel comparing detected waves against national surge windows. Advanced controls expose the legacy prominence-based detector.

**Time Lag Analysis** — detects peaks in smoothed daily cases and deaths per 100k, matches each case peak to the nearest subsequent death peak, and reports the lag in days plus a severity ratio for each matched pair. Supports county vs county comparison.

**County Factors** — scatter-plot explorer relating COVID outcomes to healthcare access, income, education, demographics, rural-urban classification, vaccination rates, and the state's 2020 presidential margin, with Pearson/Spearman statistics, OLS trend lines, and factor correlation rankings. Outcomes can be restricted to a date window (pre-vaccine, post-rollout, or custom) — essential when relating vaccination factors to outcomes.

**Statistical Modeling** — correlation matrices, Random Forest feature importance with partial-dependence curves, multivariable OLS regression with HC3 heteroscedasticity-robust standard errors and VIF multicollinearity diagnostics, cross-validated county resilience scores, vaccination efficacy analysis, and K-means county archetype clustering. The state 2020 presidential margin is available as a predictor (it is not used in resilience scores), and the vaccination scatter can be colored by governor party. Model results stay on screen until their own settings change. Shares the outcome-window control with County Factors.

## Data Sources

| Dataset | Files | Coverage |
|---|---|---|
| USAFacts COVID-19 cases/deaths/population | `data/covid_confirmed_usafacts.csv`, `data/covid_deaths_usafacts.csv`, `data/covid_county_population_usafacts.csv` | Jan 2020 – Jul 2023 |
| HRSA Area Health Resources Files | `data/ahrf2023.csv` (primary), `data/AHRF2020.asc`, `data/AHRF_2020-2021_SAS/AHRF2021.sas7bdat` (supplementary) | 2018 – 2023 vintages |
| CDC county vaccination | `data/COVID-19_Vaccinations_in_the_United_States,County_20260623.csv` | Dec 2020 – May 2023 |
| State political context | `data/state_political.csv` | Snapshot, January 2021 |

All datasets are read from the local `data/` directory; no dataset is downloaded at runtime (the small county boundary file described below is the single exception). All joins use five-character zero-padded county FIPS codes (with `(countyFIPS, State)` compound keys where duplicate FIPS rows exist).

The USAFacts CSVs and `ahrf2023.csv` are included in the repository, so the dashboard runs immediately after cloning. Larger files (the CDC vaccination CSV and supplementary AHRF releases) exceed GitHub size limits. They are published as `covid_data_supplement.zip` in the [v1.0 data supplement release](https://github.com/snyderstack/covid_dashboard/releases/tag/v1.0); unzip it in the repository root and the files land at the paths the loaders expect. `data/README.md` lists every file, its source, and its exact path. The dashboard degrades when optional files are absent: vaccination or political features are hidden and AHRF falls back to the primary 2023 CSV.

The US county boundary file (`data/geojson-counties-fips.json`, ~3 MB) is included, and is downloaded automatically on first launch if missing; it powers the maps, hotspot analysis, bordering-county lists, and the resilience and archetype maps.

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
├── map_component.py        # Zoom-preserving map (Streamlit custom component)
├── validation.py           # Standalone data-quality audits (python validation.py)
├── tests/                  # Deterministic pytest suite (synthetic fixtures)
├── assets/                 # Logos
├── data/                   # Local source datasets (see Data Sources)
├── .streamlit/config.toml  # Enables static file serving for the map
├── README.md
├── TECHNICAL_DOCUMENTATION.md
└── requirements.txt
```

`app.py` holds the Streamlit layout and visualization logic, and `map_component.py` holds the custom map renderer. The other modules are pure data functions with no Streamlit dependency (caching wrappers are defined in `app.py`). On first map render, `map_component.py` copies plotly.js and the county GeoJSON into `static/` (generated, git-ignored) so the browser downloads them once.

## Testing

```bash
pip install pytest
pytest tests/ -q
```

The suite uses small synthetic datasets, so it runs in seconds and requires none of the files in `data/`. It covers the daily-diff and per-capita pipeline, national aggregation, window outcomes, wave detection and significance scoring, lag matching, similarity search, VIF/HC3, clustering, and the spatial statistics.

## Interpretation Notes

All analyses are county-level (ecological) associations, not individual-level or causal effects. Per-capita rates use a single static population per county. Vaccination joins use the final (May 2023) snapshot for cumulative-outcome analyses, and time-series lookups where a date-specific value is needed. Political variables are state-level: every county in a state shares the same value, so the 2020 margin has only 51 distinct values across ~3,100 counties and its p-values overstate the evidence. See `TECHNICAL_DOCUMENTATION.md` for methodology, assumptions, and the change log.
