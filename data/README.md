# Data Directory

The dashboard reads all datasets from this directory. Files small enough for GitHub are included in the repository; the rest must be downloaded separately. The app degrades gracefully — anything listed as *optional* simply disables its associated features when absent.

## Included in the repository (no action needed)

| File | Source | Enables |
|---|---|---|
| `covid_confirmed_usafacts.csv` | USAFacts | Core — required |
| `covid_deaths_usafacts.csv` | USAFacts | Core — required |
| `covid_county_population_usafacts.csv` | USAFacts | Core — required |
| `ahrf2023.csv` | HRSA AHRF 2022–2023 (CSV release) | County Factors, Statistical Modeling, RUCC Metro/Nonmetro classification |
| `AHRF_2019-2020/DOC/` | HRSA | SAS layout used to parse `AHRF2020.asc` (if downloaded) |
| `geojson-counties-fips.json` | Plotly datasets (auto-downloaded on first launch if missing) | County map rendering, hotspot analysis, archetype maps |

## Optional downloads (excluded — GitHub size limits)

| File | Size | Source | Enables |
|---|---|---|---|
| `COVID-19_Vaccinations_in_the_United_States,County_20260623.csv` | ~636 MB | [CDC COVID Data Tracker](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh) — export as CSV | All vaccination features (map metrics, rollout charts, vaccination factors and modeling columns) |
| `AHRF_2020-2021_SAS/AHRF2021.sas7bdat` | ~184 MB | [HRSA AHRF downloads](https://data.hrsa.gov/data/download) — 2020–2021 SAS release | Supplementary 2019–2021 AHRF validation columns |
| `AHRF2020.asc` | ~99 MB | HRSA AHRF downloads — 2019–2020 ASCII release | Supplementary 2018–2020 pandemic-era variables (HPSA 2019/2020, 2018 physician counts) |

Place downloaded files at the exact paths shown above (the vaccination CSV filename must match, including the date suffix — or update `VAX_FILE` in `vaccination_loader.py`).

`ahrf2021.asc` and `ahrf2022.asc` are intentionally unused: HRSA published no fixed-width layout files for those release years, and `ahrf2023.csv` already covers the same variable vintages.
