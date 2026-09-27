# Data Directory

The dashboard reads every dataset from this `data/` directory, at the exact paths below — the loaders look for these filenames and nothing else. Files small enough for GitHub are in the repository; the large ones are in the data supplement release. The app degrades gracefully: any *optional* file that is missing just disables its features.

## Quick setup — data supplement

Download **[covid_data_supplement_v2.zip](https://github.com/snyderstack/covid_dashboard/releases/download/v2.0-data/covid_data_supplement_v2.zip)** (220 MB, from release [Data supplement v2.0](https://github.com/snyderstack/covid_dashboard/releases/tag/v2.0-data)) and unzip it **in the repository root** (the folder containing `app.py`). The archive already contains the `data/…` paths, so every file lands where the loaders expect it. It also includes `DATA_SUPPLEMENT_README.txt` (the same file list as a text file) and the HRSA data use agreement for the AHRF files.

```bash
unzip covid_data_supplement_v2.zip   # run from the repository root
```

## All data files

| File | Saved to (exact path) | Size | In repo? | Source / download | Used by | Enables |
|---|---|---|---|---|---|---|
| USAFacts confirmed cases | `data/covid_confirmed_usafacts.csv` | 18 MB | Yes | USAFacts — ⟨paste link⟩ | `tools.load_data` | **Required** — everything |
| USAFacts deaths | `data/covid_deaths_usafacts.csv` | 12 MB | Yes | USAFacts — ⟨paste link⟩ | `tools.load_data` | **Required** — everything |
| USAFacts county population | `data/covid_county_population_usafacts.csv` | 100 KB | Yes | USAFacts — ⟨paste link⟩ | `tools.load_data` | **Required** — per-100k rates |
| AHRF 2022–2023 (CSV) | `data/ahrf2023.csv` | 38 MB | Yes | HRSA AHRF — ⟨paste link⟩ | `ahrf_loader` | County Factors, Statistical Modeling, Metro/Nonmetro (RUCC) |
| AHRF 2019–2020 SAS layout | `data/AHRF_2019-2020/DOC/AHRF2019-2020.sas` | 1.1 MB | Yes | HRSA AHRF 2019–2020 release — ⟨paste link⟩ | `ahrf_loader` | Parsing `AHRF2020.asc` (below) |
| County boundaries (GeoJSON) | `data/geojson-counties-fips.json` | 3.1 MB | Yes | [Plotly datasets](https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json) — auto-downloaded on first launch if missing | `tools.load_county_geojson` | Offline maps, hotspot analysis, bordering counties |
| State political context (Jan 2021) | `data/state_political.csv` | 8 KB | Yes | ⟨paste link / source⟩ | `tools.load_state_political` | Overview political section, map political metrics, margin predictor |
| CDC county vaccinations | `data/COVID-19_Vaccinations_in_the_United_States,County_20260623.csv` | 635 MB | No — supplement zip | [CDC data.cdc.gov](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh) (export as CSV) — ⟨paste link⟩ | `vaccination_loader` | All vaccination features |
| AHRF 2020–2021 (SAS) | `data/AHRF_2020-2021_SAS/AHRF2021.sas7bdat` | 184 MB | No — supplement zip | HRSA AHRF 2020–2021 SAS release — ⟨paste link⟩ | `ahrf_loader` | Supplementary 2019–2021 AHRF columns |
| AHRF 2019–2020 (ASCII) | `data/AHRF2020.asc` | 98 MB | No — supplement zip | HRSA AHRF 2019–2020 ASCII release — ⟨paste link⟩ | `ahrf_loader` | Supplementary 2018–2020 variables (HPSA, physician counts) |

**Filenames must match exactly.** The vaccination CSV name includes the export date (`_20260623`); if you download a newer export, either rename it to the name above or update `VAX_FILE` in `vaccination_loader.py`.

## Not used

`ahrf2021.asc` and `ahrf2022.asc` are intentionally unused — HRSA published no fixed-width layout files for those release years, and `ahrf2023.csv` already covers the same variable vintages. `data/AHRF_CSV_2022-2023-2/` (the full HRSA CSV download) is also unused; only `ahrf2023.csv` is read.

## Git

Large files are excluded by `.gitignore` (`data/COVID-19_Vaccinations_in_the_United_States*.csv`, `*.sas7bdat`, `*.asc`, `*.zip`, `data/AHRF_2020-2021_SAS/`, `data/AHRF_CSV_2022-2023-2/`) and must never be committed — GitHub rejects files over 100 MB.
