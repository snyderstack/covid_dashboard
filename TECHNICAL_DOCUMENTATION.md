# Technical Documentation

This document is the single technical source of truth for the COVID-19 County Analysis Dashboard. Future technical changes should be appended here rather than recorded in separate summary, audit, implementation, notes, or fixes documents.

## 1. Project Architecture

The dashboard is organized around a small set of modules with clear responsibilities:

```text
covid_dashboard/
├── app.py                      # Streamlit UI, controls, chart assembly
├── tools.py                    # Data loading, preprocessing, map slices, national series
├── validation.py               # Data quality, join integrity, tooltip, and per-capita audits
├── wave_analysis.py            # Region-based wave/outbreak detection
├── lag_analysis.py             # Case-to-death peak lag analysis
├── ahrf_loader.py              # AHRF loading, column selection, derived rates
├── vaccination_loader.py       # CDC vaccination data loading and lookups
├── county_features.py          # Master county feature table, correlations, similarity
├── modeling.py                 # Correlations, RF importance, OLS (HC3), VIF, clustering
├── spatial_analysis.py         # County adjacency and Getis-Ord Gi* hotspots
├── tests/                      # Deterministic pytest suite (synthetic fixtures)
├── assets/                     # Logos
├── README.md                   # User-facing documentation
├── TECHNICAL_DOCUMENTATION.md  # Technical source of truth and change log
├── requirements.txt            # Runtime dependencies
└── data/                       # Local USAFacts, AHRF, and CDC vaccination inputs
```

`app.py` should remain focused on Streamlit layout, user controls, and Plotly visualization. Data preparation belongs in `tools.py`; data validation belongs in `validation.py`; analysis logic belongs in the purpose-specific modules (`wave_analysis.py`, `lag_analysis.py`, `county_features.py`, `modeling.py`, `ahrf_loader.py`, `vaccination_loader.py`), all of which are pure data modules with no Streamlit dependency.

The app uses Streamlit caching for raw data loading and expensive transform precomputation. The raw USAFacts tables remain in wide format, with one row per county and date columns across the timeline. Visualization functions extract only the selected county, metric, or date slice when needed.

## 2. Data Sources

The dashboard expects three local USAFacts datasets:

- `covid_confirmed_usafacts.csv`: cumulative confirmed cases by county and date.
- `covid_deaths_usafacts.csv`: cumulative deaths by county and date.
- `covid_county_population_usafacts.csv`: county population values.

The case and death datasets use a wide schema: `countyFIPS`, `County Name`, `State`, `StateFIPS`, followed by daily date columns in `YYYY-MM-DD` format. The population file contains county metadata and a `population` column.

Data is loaded from the local `data/` directory. Runtime network download of datasets is intentionally not part of the application; the single exception is the county boundary GeoJSON, which `tools.load_county_geojson()` fetches once and saves into `data/` if the bundled copy is missing, then reuses offline.

## 3. Data Processing Pipeline

`load_data()` reads the three CSV files and normalizes metadata:

- `countyFIPS` is converted to a five-character string with leading zeros.
- `StateFIPS` is converted to a two-character string when present.
- `County Name` is stripped of whitespace.
- `Location` is created as `"County Name, State"` for every loaded dataset.

The data remains in wide format after loading. Reusable helpers in `tools.py` identify metadata columns dynamically with `get_identifier_columns()` and date columns with `get_date_columns()`. This prevents preprocessing failures when optional metadata columns are absent.

At app startup, `precompute_all_transforms()` creates reusable derived tables:

- daily cases
- daily deaths
- three-day, five-day, and seven-day moving averages for cases and deaths
- cases per 100k
- deaths per 100k
- available date list

These precomputed tables remain wide-format. UI interactions generally select the correct precomputed dataframe and extract the selected date or county subset.

## 4. Choropleth Implementation

The Map tab uses Plotly's county choropleth support with the public Plotly county GeoJSON:

```python
px.choropleth(
    data_frame,
    locations="countyFIPS",
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    featureidkey="id",
    scope="usa",
)
```

`prepare_choropleth_for_date()` builds the map dataframe for one selected date and metric. It returns only the fields needed for map rendering and hover display:

- `countyFIPS`
- `Location`
- `State`
- `population`
- `value`
- `cases`
- `deaths`
- `cases_pc`
- `deaths_pc`

Invalid geographic rows are removed before plotting. Rows with null FIPS, empty FIPS, or `countyFIPS == "00000"` are excluded. The function drops rows only when the selected plotted metric is missing; hover fields are allowed to be missing without removing valid geographic rows.

State filtering is applied after the single-date choropleth dataframe is prepared. `filter_choropleth_by_state()` limits the dataframe to the selected state, while `get_state_bounds_for_zoom()` provides approximate state framing for the map.

## 5. Per-Capita Calculations

Per-capita rates use:

```text
(count / population) * 100,000
```

`precompute_per_capita()` calculates cases and deaths per 100k for all counties and dates. Population lookup uses `(countyFIPS, State)` tuple keys rather than `countyFIPS` alone. This prevents duplicate-FIPS collisions from statewide unallocated records and other non-county rows.

Rows with `countyFIPS == "00000"` or non-positive population are excluded from population lookup. Counties without valid population receive `NaN` per-capita values instead of misleading zeroes or inflated rates.

`calculate_per_capita()` is used for single-county time series in the County Comparison tab. It follows the same population guardrails.

## 6. Moving Average Calculations

Daily values are calculated from cumulative values with a column-wise difference:

```python
daily = cumulative.diff(axis=1).clip(lower=0).fillna(0)
```

Negative daily values are clipped to zero to protect downstream analysis from source-data backfills or reporting corrections. Moving averages are applied only to daily series, never cumulative series.

`precompute_all_moving_averages()` computes three-day, five-day, and seven-day moving averages for cases and deaths at startup. Results are stored in the transform dictionary as:

- `ma3_cases`, `ma3_deaths`
- `ma5_cases`, `ma5_deaths`
- `ma7_cases`, `ma7_deaths`

The app reuses these dataframes for metric switching and map rendering instead of recalculating rolling windows on every UI interaction.

## 7. Lag Analysis

Lag analysis is implemented in `lag_analysis.py`. The primary entry point is `analyze_county_lag()`, which runs the full pipeline for a single county and returns a self-contained result dictionary. `summarize_lag_results()` computes summary statistics from that dictionary.

The internal pipeline:

1. `prepare_daily_per_capita()` converts cumulative counts to daily new counts (via `calculate_daily_changes()`), normalises to per-100k, and applies a moving-average smoother. Negative daily values (from data corrections) are clipped to zero, consistent with the precomputed pipeline.
2. `detect_peaks()` calls `scipy.signal.find_peaks` on the smoothed per-100k series with configurable prominence and minimum-spacing parameters.
3. `match_case_death_peaks()` pairs each case peak with the nearest subsequent death peak within a configurable lag window. The greedy matching processes case peaks chronologically; each death peak can be claimed only once.
4. Lag is computed as `death_peak_date − case_peak_date` in days.

The two functions `prepare_lag_analysis()` and `get_county_lag_comparison()` that previously existed in `tools.py` have been removed. They are superseded by `lag_analysis.py` and were not referenced anywhere in the application.

This analysis is exploratory. Lag estimates should be interpreted alongside reporting practices, testing availability, variant periods, vaccination coverage, and demographic context.

## 8. Wave Detection

`wave_analysis.py` detects outbreak waves in smoothed daily case or death series. It is exposed as the "Wave Analysis" tab and also feeds the wave summary on the County Overview tab.

The primary detection path is region-based (v3.2): an adaptive local baseline (rolling low percentile) is estimated, sustained elevated periods above that baseline are extracted as epidemic regions, and merged regions containing genuinely distinct surges are split at internal valleys that are deep — relatively and absolutely, in elevation-above-baseline terms — and *sustained* for weeks rather than days. Each wave's start/end boundaries are then trimmed to the span where the signal stays above 10% of that wave's peak elevation (measured against the region's background floor, 5-day sustain rule), so reported intervals track the outbreak's rise and fall rather than the region's threshold crossings. Finally, a peak-significance filter discards regions whose peak barely rises above baseline (low-signal plateaus). One wave is reported per surviving region, and each receives a significance score (0–100) combining prominence (30%), burden (30%), duration (20%), and burst intensity (20%). Three sensitivity presets (`conservative`, `standard`, `sensitive`) parameterize the detector; the module docstring documents the full algorithm and the 2026-07-08/09 changelog entries record its calibration on reference counties.

A legacy prominence-based path (`sensitivity=None`, `scipy.signal.find_peaks` with wave boundaries at 10% of peak value) remains available and is used by the dashboard's advanced detection controls when the user overrides prominence manually.

The primary functions are:

- `find_waves()` — smooths with `numpy.convolve(mode='same')` and runs either the region-based or legacy detection; returns wave dicts and a diagnostics dict.
- `calculate_wave_metrics()` — maps wave indices back to dates and computes aggregate metrics including significance scores.
- `calculate_waves_for_county()` — entry point called from `app.py`; returns a results dict with separate `cases` and `deaths` sub-dicts.
- `calculate_waves_for_all_counties()` — batch function used by the cached all-county wave metrics table.

Wave detail dictionaries use the key `"peak_value"` for the peak count (accurate for both cases and deaths).

## 9. County Feature Table

`county_features.py` creates a one-row-per-county feature table for downstream research. `create_county_feature_table()` merges county metadata, population, latest total cases, latest total deaths, cases per 100k, deaths per 100k, and optional wave metrics. `create_master_county_table()` extends this with AHRF healthcare/socioeconomic variables and CDC vaccination columns, joined on countyFIPS; this master table backs the County Overview, County Factors, and Statistical Modeling tabs.

`add_external_dataset()` supports merging additional county-level datasets by FIPS, optionally also requiring state matching. The module also provides `compute_bivariate_correlation()` (Pearson + Spearman with p-values), `compute_ols_trend()`, min-max normalization, z-score standardization, and `prepare_for_regression()` / `prepare_for_clustering()` helpers for external analysis workflows.

Two peer-analysis functions power the County Overview: `find_similar_counties()` ranks counties by z-scored Euclidean distance over structural features (population log-transformed, outcomes deliberately excluded), and `generate_county_insights()` builds the automated Resilience Profile — peer/national mortality percentiles plus risk/protective factor identification with directions taken from each factor's empirical national correlation with mortality (see the 2026-07-09 analytical-platform changelog entry for the full rules). Model fitting beyond simple OLS trends lives in `modeling.py` (see Section 15).

## 10. Known Issues and Accepted Limitations

These are the current, deliberate limitations of the platform. Each is either disclosed in the UI where it matters or inherent to county-level surveillance data. (Historical findings and their resolution status are catalogued in Section 13.0.)

**Data handling.** Statewide unallocated records and zero-population rows are retained in raw loaded data for traceability but excluded from maps, per-capita math, and national totals. Negative daily diffs (source-data corrections) are clipped to zero for analysis, but correction events are flagged with markers on the wave and daily comparison charts so they are visible rather than hidden; cumulative analyses are unaffected. Per-capita rates use a single static population per county for the entire 2020–2023 period — a small systematic bias for high-growth counties.

**Statistical methods.** Moving averages use `min_periods=1`, so the first window−1 values of any smoothed series draw on fewer points (disclosed in the lag and wave methodology expanders). Lag analysis matches peaks greedily in chronological order (not globally optimal) and treats missing days as zero before peak detection. The Index=100 comparison rebases each series at its own first non-zero value, so curves can start from different epidemic moments. All associations everywhere are county-level (ecological); individual-level inference is invalid.

**Legacy wave path.** When the prominence-based detector is manually selected via Advanced controls, wave boundaries use the 10%-of-peak rule and can degenerate on short series (duration is floored at one day). The default region-based detector does not have this issue.

**Performance.** The startup moving-average precompute transposes full tables (acceptable at current data size). Computing wave metrics for all ~3,100 counties takes 60–90 seconds; it is user-triggered and cached for the session. The random-sample validation audit (`validation.py`) is a spot check, complemented by the deterministic suite in `tests/`.

**Geography.** State map zoom uses approximate hardcoded centers and projection scales rather than bounds derived from geometry. County adjacency for hotspot analysis is derived from shared GeoJSON boundary vertices — coastal and island counties may have fewer neighbours than a Census adjacency file would give them.

## 11. Future Development

Extensions that require datasets **not currently in the repository** (do not
attempt without acquiring the data):

- excess-mortality analysis — needs all-cause death registrations (CDC WONDER)
- age- or race-stratified outcomes — USAFacts county data carries no strata
- policy/NPI timelines (mask mandates, closures) — needs OxCGRT or state data
- hospital *utilization* time series — AHRF provides static capacity only
- variant-era attribution — needs genomic surveillance data
- time-varying population denominators — needs annual county estimates

Extensions feasible with current data:

- multi-county map brushing and comparison selection
- wave propagation analysis (which counties' waves lead or lag their region)
- county geometry-based state and regional zooming

Future work should preserve the current division of responsibility: data preparation in `tools.py`, validation in `validation.py`, UI in `app.py`, and research modules in purpose-specific files.

## 12. Change Log

### 2026-06-01

Repository documentation was consolidated into two canonical files: `README.md` and `TECHNICAL_DOCUMENTATION.md`. Older generated summary, audit, implementation, guide, and checklist documents were merged into these files and removed from the working tree.

Validation logic was consolidated into `validation.py`. The dashboard audit section now imports validation functions from that module instead of the removed `comprehensive_validation.py`.

Obsolete ad hoc test script `test_changes.py` was removed after its useful verification content was folded into this documentation and replaced by targeted validation commands.

`tools.py` was cleaned up with shared metadata helpers for identifier columns, date columns, and population column detection. Metadata normalization now returns explicitly normalized dataframes and avoids the prior fragmented-column warning when creating `Location`.

The national per-capita helper was corrected to compute a population-weighted national rate from precomputed county per-capita data instead of using an unweighted mean across counties.

Project hygiene files were added: `requirements.txt` for runtime dependencies and `.gitignore` for local artifacts such as virtual environments, Python bytecode, and macOS metadata.

### 2026-06-15

All audit findings from Section 13 implemented. No existing functionality was removed or changed in behaviour except where explicitly noted.

**tools.py** — Added `extract_county_state()` shared helper. Fixed `calculate_daily_changes()` to apply `.clip(lower=0)` and remove `.astype(int)`, making it consistent with `precompute_daily_diffs()`. Fixed `compute_national_timeseries()` and `compute_national_daily()` to exclude statewide unallocated rows (`countyFIPS == "00000"`) before summing, preventing double-counting in national totals. Fixed `compute_national_per_capita()` to accept raw cases/deaths DataFrames directly instead of reversing precomputed per-capita values (eliminates floating-point round-trip). Vectorised population lookup in `precompute_per_capita()` using a merge instead of row-wise `apply`. Removed dead functions `prepare_lag_analysis()` and `get_county_lag_comparison()` (superseded by `lag_analysis.py`).

**lag_analysis.py** — Added explicit `.clip(lower=0)` after `calculate_daily_changes()` in `prepare_daily_per_capita()` to guard against future changes to that function.

**wave_analysis.py** — Renamed `"peak_cases"` key to `"peak_value"` in all wave detail dicts so the key is accurate when analysing deaths. Changed `find_waves()` to use `numpy.convolve(mode='same')` throughout, matching the wave chart display and eliminating the manual index-offset correction that was required with `mode='valid'`. Edge artefacts from `mode='same'` at array boundaries are zeroed out explicitly.

**validation.py** — Fixed `check_population_validity()` to use `get_population_column()` instead of the hardcoded column name `"population"`, preventing `KeyError` when the column has a different name. Fixed `diagnostic_check()` with the same guard.

**app.py** — Removed dead `st.session_state.pop_dict` code. Replaced the `st.session_state` national-series caching pattern with a `@st.cache_data`-decorated `compute_national_aggregates()` function, which survives cache clears correctly and passes raw DataFrames to `compute_national_per_capita()`. Added `@st.cache_data`-decorated `get_choropleth_data()` wrapper to avoid re-running four merge operations on every map slider movement. Updated all references to `wave['peak_cases']` to `wave['peak_value']`. Replaced all inline location-string parsing (scattered `rsplit(", ", 1)` calls and a locally-defined `extract_county_state()`) with the shared `tools.extract_county_state()`. Extracted all seven tab content blocks into named `render_*_tab()` functions, reducing the main execution body to eight `with tab_*:` calls. Added `min_periods=1` disclosure note to the lag analysis methodology expander. Updated `import` block to include `extract_county_state`.

**requirements.txt** — Pinned `streamlit==1.57.0`, `pandas==3.0.3`, `numpy==2.4.6`, `plotly==6.7.0`. Added `scipy>=1.9.0` (scipy was present on the system but absent from the pin file).

**TECHNICAL_DOCUMENTATION.md** — Rewrote Section 7 to describe `lag_analysis.py` (replacing the description of the now-deleted `tools.py` functions). Updated Section 8 to reflect that wave detection is a primary dashboard tab and documents the `peak_value` key rename and `mode='same'` convolution change.

### 2026-07-06 — Release-candidate audit

Full-repository code review before first release. No functional behavior changed except where noted.

**app.py** — Removed unused imports (`get_vaccination_at_dates`, `VAX_LABELS`, `score_wave_significance`); `get_population_column` is now imported at module top instead of inside `render_wave_tab()` (the previous "circular import" comment was inaccurate — no cycle exists). Removed a redundant in-function `import plotly.graph_objects` alias in the vaccination efficacy section. Removed an `if True:` scoping block in the lag tab. Removed the unused `national` parameter from `render_county_overview_tab()`. Corrected the County Overview wave-table caption to state the actual significance weights (prominence 30%, burden 30%, duration 20%, intensity 20%; it previously claimed 40/30/30). Simplified NaN checks (`val != np.nan` was always true; a nested isinstance conditional replaced with `pd.notna`). Renumbered the internal SECTION comments in the overview tab to match the rendered headers. Removed an empty `.map-ctrl-panel` CSS rule that was referenced nowhere.

**tools.py** — Removed the unused `choro_data` parameter from `get_state_bounds_for_zoom()`; updated the single caller.

**wave_analysis.py** — Removed dead code: `estimate_prominence_threshold()`, `_compute_adaptive_prominence()`, `_filter_peaks_by_width()`, and `_merge_shallow_valleys()` were defined but never called anywhere in the repository. Removed the now-unreferenced `valley_depth_pct` and `min_width_days` preset entries. Type annotation fix in `estimate_optimal_smoothing()`.

**validation.py** — Removed the duplicate `passed_countries` result key; its comment claimed it was consumed by `app.py`, but `validation.py` is not imported by any application code (it is a standalone diagnostic runner, as its docstring states).

**county_features.py / modeling.py** — Removed an unused `pathlib.Path` import; normalized blank-line spacing.

**README.md** — Rewritten to reflect the current seven-tab application, the AHRF and CDC vaccination data sources, and the full module list. The previous version described a "demographics tab" that no longer exists and a lag tool ("shift death trends by a user-selected lag") that does not match the implemented peak-matching design.

**TECHNICAL_DOCUMENTATION.md** — Section 1 module tree updated to include `lag_analysis.py`, `ahrf_loader.py`, `vaccination_loader.py`, and `modeling.py`. Section 8 rewritten to describe the current region-based (v3) wave detector with the legacy prominence path as fallback. Section 9 updated: the former analytics "scaffolding" is implemented. Section 10 updated: national per-capita is computed from raw counts. Section 11 pruned of items already implemented.

**Repository hygiene** — Removed tracked `__pycache__/*.pyc` and `.DS_Store` files from version control (already covered by `.gitignore`).

### 2026-07-06 — Exploration and research-methods feature release

Eleven additive features; no existing calculation or interface changed except
where explicitly noted. New logic lives in the pure data modules and is
covered by a new deterministic test suite (`tests/`, 21 tests, synthetic
fixtures, no data files required).

**Exploration** — County Overview gains a "Counties Like This One" section
(`county_features.find_similar_counties`: z-scored Euclidean distance over
structural features, population log-transformed), a Surprise Me random-county
button, shareable `?county=` URLs (`st.query_params`), and a downloadable
one-page HTML county report. The Geographic Map gains animated monthly
playback (`tools.monthly_snapshot_long` → `animation_frame`, one frame per
month) and Getis-Ord Gi* hotspot analysis.

**Spatial statistics** — new `spatial_analysis.py`: county adjacency derived
from shared GeoJSON boundary vertices (rook contiguity, ≥2 shared vertices)
and the Gi* statistic with binary weights; hot/cold classification at
|z| > 1.96. The county GeoJSON is now bundled: `tools.load_county_geojson()`
loads `data/geojson-counties-fips.json`, auto-downloading it once from the
Plotly CDN if absent — the maps previously fetched the CDN URL on every
session and failed silently offline.

**Research methods** — `tools.compute_window_outcomes()` restricts outcome
columns to a date window; County Factors and Statistical Modeling expose
pre-vaccine / post-rollout / custom windows (addresses the reverse-causality
caveat documented in Section 16.6). `modeling._ols_fit()` now also returns
HC3 heteroscedasticity-robust standard errors and p-values (MacKinnon &
White 1985), shown alongside classical ones in the regression table.
`modeling.compute_vif()` adds multicollinearity diagnostics.
`modeling.run_rf_partial_dependence()` adds one-way partial-dependence curves
for the top Random Forest features. `modeling.compute_county_clusters()` adds
K-means county archetypes over structural features (sklearn with a numpy
Lloyd's-algorithm fallback), rendered as a new Statistical Modeling section.
`wave_analysis.NATIONAL_WAVE_WINDOWS` and `match_waves_to_national_windows()`
back a new validation panel comparing detected waves to national surge
windows.

**Bug fix** — the Statistical Modeling cache wrappers (`_cached_correlations`
and friends) previously excluded the dataframe from their cache keys
(underscore-prefixed parameter) while receiving *filtered* data, so changing
the state/region/metro filter could return stale results computed for a
previous filter. The dataframe is now part of the cache key.

### 2026-07-08 — Documentation ship-readiness review

Re-verified every finding from the 2026-06-15 architecture audit against the
current code and recorded the outcome in new Section 13.0 (Resolution Status):
16 items Fixed, 3 Superseded by the v3 wave detector or bundled GeoJSON,
2 By design (standalone validation), and the remainder Accepted limitations
with UI disclosure. All Priority 1 and 2 roadmap items are complete; P3-B
(surfacing data-correction events) remains open by choice; P4-C is superseded
by the per-dataset loader modules.

Section 10 rewritten as "Known Issues and Accepted Limitations" reflecting the
current codebase. Corrected stale documentation claims: runtime downloads
(the GeoJSON auto-fetch is now the documented single exception in Section 2
and README), Section 15.3 imputation (numpy median, not SimpleImputer) and
feature-importance description (impurity-based, not OOB), and Section 16.6
outcome-data end date (July 2023, not "present"). Added `tests/__init__.py`
for unambiguous pytest package imports. No functional code changes.

### 2026-07-08 — Classroom and refinement release

Eleven additive changes aimed at teaching use; no analytical outputs changed.

**Educational layer** — `GLOSSARY` (21 plain-language term definitions) surfaced
as a "Key terms on this page" popover per tab via `render_learning_aids()`,
which also renders per-tab "Questions to investigate" expanders (2–3 curated
prompts each). "Classroom examples" popover on the County Overview offers six
pedagogically chosen counties. Methodology expanders now include the actual
formulas (`st.latex`): per-capita rate and lag/severity ratio (Time Lag),
wave significance score (Wave Analysis), OLS estimator and HC3 covariance
(Statistical Modeling), and the Gi* statistic (map hotspot expander).

**Interaction** — Clicking a county on the choropleth loads it into the County
Overview (Plotly `on_select`; the selection is stashed in
`_pending_overview_county` and consumed before widgets instantiate on the next
run, with a re-processing guard). The Overview comparison table adds a **Peer
Median** column — medians over the county's ten structural peers — alongside
the national median. The Wave chart gains an optional national per-100k
overlay (same smoothing window) for per-capita metrics.

**Refinements** — Reporting corrections (dates where a cumulative source
series decreased) are detected by new `tools.find_data_corrections()` and
flagged as markers on the wave chart and daily comparison charts, closing
roadmap item P3-B. Colorblind-safe palette toggle (Viridis) on the map.
Shareable map URLs: `?metric=` and `?date=` seed the map widgets and stay in
sync. First launch now narrates its 30–60 s load sequence in an `st.status`
panel (subsequent reruns skip it). The sidebar date slider now sets only the
map's initial date; the map's own slider governs thereafter.

Tests: +1 (`find_data_corrections`), suite at 22.

### 2026-07-08 — Wave detection robustness (v3.1)

User-reported defect: low-population counties produced implausible wave counts
(Abbeville County, SC: 15 "waves" at Standard sensitivity, most of them noise
blips), and the per-wave legend entries overflowed into the chart title.
Instrumentation showed region detection was sound (Abbeville: exactly 2
epidemic envelopes) — the within-region valley splitter caused the explosion
(2 regions → 11 waves; Miami-Dade 4 → 24).

Three detector changes in `wave_analysis.py`:

1. **Inverted preset bug fixed.** `valley_split_pct` values were inverted
   relative to intent: with the split condition
   `valley < (1 − pct) × lower_peak`, Conservative (0.30) split *more*
   eagerly than Sensitive (0.55). Values corrected to 0.75 / 0.60 / 0.45
   (conservative / standard / sensitive) and comments rewritten.

2. **Valley splitting hardened.** All depth tests now run in elevation space
   (smoothed minus adaptive baseline), making them scale-free for both
   high-burden endemic counties and low-count rural ones. A split now
   requires relative depth AND absolute depth (≥ 2× the absolute elevation
   threshold) AND a **sustained trough** — elevation must stay below the
   split level for at least `min_region_duration` consecutive days. The
   sustained-trough test is the decisive discriminator: reporting noise dips
   for days, real inter-wave troughs (Delta→Omicron) persist for weeks.
   Sub-peaks must clear max(10% of region max elevation, 3× absolute
   threshold) and be a minimum sub-wave duration apart.

3. **Peak-significance filter added.** A region whose smoothed peak rises
   less than `peak_significance_mult` (3.5 / 2.5 / 1.5 by preset) times the
   elevation threshold above its local baseline is discarded as a low-signal
   plateau. Dropped candidates appear in the diagnostics audit log as
   "Removed — below significance floor".

Also fixed: waves displaying "peak = 0" when the smoothed argmax landed on a
zero-reporting day (falls back to the smoothed rate), and the wave chart no
longer emits per-wave legend entries (peaks are labelled on-chart), which
eliminates the legend-overflow at any wave count.

Reference-county results at Standard (before → after): Abbeville SC 15 → 4,
Miami-Dade FL 15 → 6, King WA ~10 → 6, Los Angeles CA 8 → 4, Cook IL 8 → 5,
Loving TX 1 → 0 (no epidemiologically significant wave — correct for a
population of 64). All presets now land in their documented ranges.
Tests: +2 (low-count noise regression; sustained-trough split vs brief-dip
non-split), suite at 24.

### 2026-07-09 — Wave boundary refinement (v3.2)

User-reported defect: wave peaks were correct, but the shaded start/end
intervals extended far beyond the actual outbreak (Los Angeles at Standard,
5-day MA: Wave 1 spanned 2020-01-28 → 2021-11-23, 665 days, though the surge
ran roughly November 2020 → February 2021).

Cause: wave boundaries were the epidemic region's threshold crossings against
the adaptive baseline. Low-level activity clears `baseline×rel + abs` months
before a surge, and successive elevated periods chain through the merge gap;
onset refinement only ever moved the start *earlier*, and the end was simply
the last crossing. Region bounds answer "when was anything elevated?", not
"when did this outbreak rise and fall?".

Fix: new `_trim_wave_bounds()` applied after peak selection. Each wave's
boundaries are trimmed to the span where the smoothed signal stays above
10% of that wave's peak elevation (`WAVE_BOUNDARY_FRAC`), walking outward
from the peak and stopping only after 5 consecutive quiet days
(`WAVE_BOUNDARY_SUSTAIN_DAYS`), so brief dips during rise or decline cannot
truncate the wave. Elevation is measured against the region's background
floor (minimum adaptive baseline within the region) — the rolling baseline
climbs during declines and the baseline at the peak sits inside the surge,
so neither is a valid boundary reference. Bounds are clamped inside the
original region and never exclude the peak.

Explicitly unchanged: region detection, valley splitting, sensitivity
presets, peak selection (peak dates verified identical on the reference
counties), peak values, wave counts (verified identical across 8 counties ×
3 presets), and the significance scoring formula. Duration and burden now
describe the trimmed outbreak span, so significance scores shift modestly.

Result (LA, Standard, 5-day MA): Wave 1 now 2020-11-02 → 2021-02-11;
Abbeville's winter wave 2020-11-17 → 2021-03-27. Regression test added
(surge boundaries must exclude a long low lead-in); suite at 25.

### 2026-07-09 — Time Lag Analysis presentation refresh

Presentation-only change to `_render_lag_chart`, `_render_lag_summary_metrics`,
and `_render_lag_pairs_table` in app.py; peak detection, matching, lag math,
and all summary statistics are untouched (verified by diff — no analysis
module modified).

With many matched pairs, the old chart drew two dashed vertical lines, a
top-of-chart connector bar, and a "Lag: Xd" annotation per pair (80+ overlay
elements for 21 pairs), which collided with each other, the legend, and the
title. Changes: each matched pair is now a single translucent band spanning
case peak → death peak; per-pair lag labels render only when ≤ 8 pairs exist
(hover and the pairs table always carry exact values, and a caption says so
at higher densities); the legend is right-aligned above the plot clear of
the title, with shortened trace names (the smoothing window already appears
in the subtitle); peak markers are slightly smaller; the death curve is
lighter-weight so the case curve reads as primary; both y-axes are anchored
at zero; and the chart gained height and margin.

Added plain-language help tooltips (consistent with the dashboard's existing
tooltip style) on the "Matched Pairs" summary metric and the "Matched Case →
Death Peak Pairs" table heading, plus a CSS rule so `st.subheader` headings
match the existing in-tab heading style.

### 2026-07-09 — Final consistency audit

Project-wide review after the day's wave-detector and lag-presentation work.
Static sweeps (compile, unused imports, widget keys, glossary coverage,
emoji) plus a full real-data regression of every tab's backing pipeline
(map/choropleth, overview master table and peers, factor correlations, VIF,
clustering, wave detection with boundary invariants, national series,
spatial adjacency) and the 25-test suite.

Fixes: removed an unused `Optional` import in `spatial_analysis.py`; the
County Overview wave-summary description still cited the retired
"adaptive prominence · width + valley filters" pipeline and now names the
current region-based detection with peak significance filter; Section 8
updated to describe the v3.2 detector (elevation-space sustained-trough
splitting, boundary trimming, significance filter); the `wave_analysis`
module docstring gained the significance-filter step; two code comments
phrased as development history ("the old…", "previously…") rewritten as
present-tense rationale. No functional changes.

### 2026-07-09 — Sidebar repurposed as a session panel

The sidebar previously held an "Analysis Date" slider (redundant: it only
seeded the map's initial date, which the map's own slider and ?date= URL
parameter control) and a "State" default filter that was assigned but never
read anywhere — dead since the standalone Trend Analysis tab was merged into
County Comparison. As the only element visible from every tab, the sidebar
now carries session-level context instead of per-tab controls:

- **Now Viewing** — the county currently loaded in the Overview (name,
  population, deaths/100k, vaccination) with a Surprise Me button.
- **Recently Viewed** — one-click jump-back buttons for the last few
  counties visited (session history maintained by the Overview tab).
- **Display** — the colorblind-safe palette toggle, promoted from the map
  control panel to a global setting (`cb_safe_global` session key).
- **Dataset Status** — loaded/missing indicators for USAFacts, AHRF, CDC
  vaccination, and the county boundary file, with pointers to
  data/README.md when something is absent — a deployment health check at
  a glance.
- The Data Coverage notice is retained.

The map tab's initial date defaults to the latest available date, exactly
as the removed slider's default did; no other behavior changed.

### 2026-07-09 — Lag chart navigation and targeted tooltips

Presentation-only (app.py; analytics untouched).

**Time Lag chart.** Matched-pair bands are now drawn only when ≤ 8 pairs
exist — the same density gate as the lag labels. Beyond that, overlapping
bands tiled the entire timeline into a wash (the very clutter they replaced);
marker colors, hover, and the pairs table carry the matching information at
any density, and the caption says so. The chart now opens zoomed to the
analysis window (first case peak → last death peak, ± 6 weeks) instead of
compressing three years — including dead post-reporting tails — into one
frame, and gains a range slider (mini-map), 3m/6m/1y/All range buttons, and
drag-to-zoom, with height increased to 720 px. The death curve is drawn
lighter still, so the case curve reads as primary.

**Tooltips.** Five targeted help tooltips added to the Statistical Modeling
tab, where student-facing concepts are densest: the correlation-matrix
outcome selector (what a correlation matrix is; Pearson r vs Spearman ρ in
plain language), the Random Forest outcome selector (importance = predictive
usefulness, not causation; correlated factors share credit), the OLS
predictor multiselect (coefficients are conditional on the other predictors;
pointer to the VIF table), the resilience outcome selector (score =
over/under-performance vs expectation), and the archetype k selector (no
single correct k). Other candidates already carry help text or are covered
by the per-tab Key Terms glossary — deliberately not tooltipped everywhere.

### 2026-07-09 — Map-first landing and Overview interaction polish

Presentation/UX changes in app.py; no analytics touched. Tests 25/25.

**Map-first landing.** The Geographic Map is now the first tab, so opening
the dashboard lands on a large interactive choropleth. The control panel
moved from the left to the right of the map (CDC PLACES-style layout) and is
collapsible via a toggle: hiding it gives the map the full content width
(and extra height). Control state persists across collapse — unmounted
widgets normally lose their session entries, so the six panel keys are
re-registered as app state each run, and the hidden-panel path reads the
persisted values with validated fallbacks.

**Overview navigation cards.** The explore chips are now real buttons:
`st.tabs` has no programmatic switching, so the cards render in a small
embedded component whose clicks locate the matching tab button in the parent
document by visible label and click it. If the DOM ever changes, cards
degrade to inert labels rather than erroring. The static chip CSS was
removed along with the old hover-only chips.

**"Questions to Investigate" removed** from all seven tabs at the user's
request (the `render_learning_aids` helper now renders only the Key Terms
glossary popover). Layouts close up cleanly; no orphaned CSS remains.

**Overview mini lag chart** now opens zoomed to the county's outbreak window
(peaks ± ~6 weeks) with a mini-map range slider for full-timeline
exploration, replacing three years compressed into 260 px.

**Header mark** replaced: the college "G" image gave way to a stylized
coronavirus (generated inline SVG data URI — gradient sphere, twelve spike
proteins in the dashboard's orange palette; no asset file or network fetch).
The footer seal is unchanged.

### 2026-07-09 — Analytical platform release

Six additions that shift the platform from visualization toward county-level
epidemiological analysis. All are computed strictly from repository data;
extensions requiring external datasets are catalogued in Section 11 instead
of being half-implemented. Tests at 28.

**Resilience Profile / automated insights** — new
`county_features.generate_county_insights()`: places a county's mortality at
a percentile within its ten structural peers and nationally, then identifies
distinguishing characteristics (peer gaps ≥ 0.8 national SD) and classifies
each as risk or protective using the factor's *empirical* national
correlation with mortality (|r| ≥ 0.05 required; directions are never
hardcoded). Peer-relative by design: traits shared by the whole peer group
(e.g., low vaccination across an Appalachian peer set) are correctly not
flagged as distinguishing. Replaces the Overview's "Research Snapshots"
section; also embedded in the downloadable report. Degrades to None when
data is insufficient.

**Pandemic timeline** — the Overview's mini chart became an integrated
two-panel timeline: smoothed cases and deaths per 100k (dual axes), detected
wave spans with labelled peaks, a structural-peer median trajectory (new
`tools.peer_median_series()`), and the vaccination rollout on a shared time
axis, with outbreak-window default zoom and a mini-map slider.

**Global Moran's I** — new `spatial_analysis.compute_morans_i()` (binary
contiguity, analytic z under normality, Cliff & Ord 1981) answers "does this
metric cluster geographically at all?" above the Gi* map, which answers
"where?". Cached alongside the hotspot analysis.

**Neighboring-county comparison** — the Overview's peer section gains a
bordering-counties table (from the GeoJSON-derived adjacency) contrasting
geographic neighbours (shared exposure) with structural peers (shared
characteristics), plus a "why these counties?" expander showing the
similarity features behind the peer match.

**Statistical communication** — OLS interpretation bullets now classify
significance by HC3-robust p-values when available and report each
significant coefficient with its 95% CI (plus a plain-language note on what
a CI excluding zero means); the County Factors annotation adds a Fisher-z
95% CI for Pearson r.

**Research report** — the downloadable HTML report gains the resilience
profile, the detected-wave table, and explicit Methodology Notes and Data
Limitations sections suitable for classroom or exploratory-research use.

### Change Log Policy

Future changes should be appended to this section. Do not create new `*_SUMMARY.md`, `*_AUDIT.md`, `*_CHANGES.md`, `*_FIXES.md`, `*_NOTES.md`, `*_IMPLEMENTATION.md`, or similar documentation files.

For user-facing changes, update `README.md`. For technical, architectural, validation, or implementation changes, update `TECHNICAL_DOCUMENTATION.md`.

---

## 13. Full Architecture Audit (2026-06-15)

This section records a complete audit of the codebase **as it existed on 2026-06-15** and is retained as a historical record. The audit covers structure, data pipeline, choropleth, statistics, performance, code quality, research readiness, and documentation.

### 13.0 Resolution Status (verified 2026-07-08)

Every finding below was re-verified against the current code. Statuses: **Fixed** (code changed, most covered by `tests/`), **Superseded** (the design it criticized was replaced), **Accepted** (deliberate limitation, disclosed in Section 10 and/or the UI), **By design** (intended behavior of a standalone tool).

| Finding | Status | Where |
|---|---|---|
| BR-1 national statewide double-count | Fixed | `compute_national_timeseries/daily` exclude FIPS 00000; `tests/test_tools.py` |
| BR-2 hardcoded population column | Fixed | `validation.py` uses `get_population_column()` |
| BR-3 negative daily values in lag | Fixed | `calculate_daily_changes` clips; explicit guard in `lag_analysis` |
| BR-4 zero-duration waves | Superseded / mitigated | default detector is region-based; legacy path floors duration at 1 day |
| BR-5 dead `pop_dict` session state | Fixed | removed |
| BR-6 validation recomputes per-capita | By design | `validation.py` is an independent auditor; recomputation is the point |
| BR-7 `peak_cases` key misnomer | Fixed | renamed `peak_value` everywhere |
| BR-8 Section 7 described dead code | Fixed | rewritten for `lag_analysis.py` |
| SR-1 `min_periods=1` early-window MAs | Accepted | disclosed in lag/wave methodology expanders |
| SR-2 negative diffs silently zeroed | Accepted | disclosed; see Section 10 |
| SR-3 static population denominator | Accepted | disclosed; see Section 10 |
| SR-4 arbitrary 10% wave boundary | Superseded | v3 adaptive-baseline region detection (Section 8) |
| SR-5 greedy lag peak matching | Accepted | disclosed in lag methodology expander |
| SR-6 NaN→0 before peak detection | Accepted | documented in `lag_analysis.detect_peaks` |
| SR-7 Index=100 baseline mismatch | Accepted | explained in the display-mode description |
| SR-8 national per-capita round-trip | Fixed | computed from raw counts |
| PR-1 row-wise population `apply` | Fixed | vectorized `(FIPS, State)` merge |
| PR-2 transpose copies in MA precompute | Accepted | startup-only; documented in docstring |
| PR-3 uncached choropleth prep | Fixed | `get_choropleth_data` cached |
| PR-4 validation recomputation cost | By design | standalone diagnostic runner |
| PR-5 `iterrows` in all-county waves | Accepted | user-triggered, cached, 60–90 s warning shown in UI |
| PR-6 GeoJSON CDN dependency | Fixed | bundled with one-time auto-download |
| RISK-1 national totals | Fixed | see BR-1 |
| RISK-2 wave index mapping fragility | Mitigated | `mode='same'` everywhere; deterministic wave tests |
| RISK-3 `(FIPS, State)` join discipline | Mitigated | dead FIPS-only dict removed; join math regression-tested |
| RISK-4 silent analytics stubs | Resolved | stubs replaced with real implementations (`modeling.py`, `county_features.py`) |
| RISK-5 early-pandemic MA reliability | Accepted | disclosure in UI; see SR-1 |
| RISK-6 CDN outage blanks the map | Fixed | see PR-6 |
| Roadmap P1-A…F | All complete | changelog 2026-06-15 |
| Roadmap P2-A…H | All complete | changelog 2026-06-15 |
| Roadmap P3-A, P3-C | Complete | disclosure notes; `mode='same'` |
| Roadmap P3-B (surface data corrections) | Fixed (2026-07-08) | `tools.find_data_corrections()`; correction markers on wave and comparison charts |
| Roadmap P4-A, P4-B, P4-D, P4-E | Complete | modeling implementations, `tests/`, animated map, bundled GeoJSON |
| Roadmap P4-C (`data_sources.py`) | Superseded | per-dataset loaders (`ahrf_loader`, `vaccination_loader`) + `add_external_dataset()` fill the role |

No files were changed as part of the original audit; findings were recommendations only.

---

### 13.1 Executive Summary

The COVID-19 County Analysis Dashboard is a well-organized, maintainable codebase. The core data pipeline is architecturally sound: wide-format data is loaded once, cached with `@st.cache_data`, transformed into reusable derived tables at startup, and sliced per-interaction. Join correctness is handled well — the `(countyFIPS, State)` tuple key in per-capita lookups prevents the duplicate-FIPS contamination that affected earlier versions.

Functionality is broad: choropleth mapping, county trend analysis, county-vs-county comparison, per-capita normalization, wave detection, epidemiological lag analysis, and a validation audit panel are all implemented and working.

The codebase is suitable as a portfolio and teaching project in its current state. Before it is treated as a research publication or used to draw conclusions that inform policy, the items in Priority 1 of the Refactoring Roadmap must be addressed — particularly the statewide double-count in national totals, the `clip(lower=0)` inconsistency between the lag pipeline and the map/wave pipelines, and pinned dependency versions for reproducibility.

Dead code (two unreferenced lag functions in `tools.py`, an unused `pop_dict` in session state) should be removed before the next round of feature additions to avoid confusion about which lag implementation is authoritative.

The `county_features.py` scaffolding is the correct foundation for future socioeconomic research integration. The `add_external_dataset()` function provides the right join contract. The analytics stubs (`prepare_for_correlation_analysis`, etc.) need real implementations before research use.

---

### 13.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         app.py                                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
│  │ Map Tab │ │ County   │ │ Trend    │ │ Demo   │ │County  │  │
│  │         │ │ Compare  │ │ Tab      │ │ Tab    │ │Factors │  │
│  └────┬────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ └───┬────┘  │
│       │           │            │             │          │       │
│  ┌────┴──────────────────────────────────────────────┐         │
│  │         Lag Tab              Wave Tab             │         │
│  └────┬─────────────────────────────┬───────────────-┘         │
└───────┼─────────────────────────────┼─────────────────────────-┘
        │                             │
        ▼                             ▼
┌───────────────┐           ┌──────────────────┐
│  lag_analysis │           │  wave_analysis   │
│  .py          │           │  .py             │
└───────┬───────┘           └────────┬─────────┘
        │                            │
        ▼                            ▼
┌──────────────────────────────────────────────┐
│                  tools.py                    │
│  load_data()         precompute_daily_diffs()│
│  normalize_*()       precompute_per_capita() │
│  prepare_county_timeseries()                 │
│  prepare_choropleth_for_date()               │
│  compute_national_*()                        │
└────────────────────┬─────────────────────────┘
                     │
          ┌──────────┼──────────────┐
          ▼          ▼              ▼
   cases_df     deaths_df    population_df
  (wide CSV)   (wide CSV)     (wide CSV)
```

Research modules (not yet wired to UI tabs):

```
county_features.py
  create_county_feature_table()   ← aggregates COVID metrics + population
  add_external_dataset()          ← FIPS-join hook for external data
  normalize_feature()             ← min-max normalization
  standardize_feature()           ← z-score normalization
  prepare_for_*_analysis()        ← stubs awaiting implementation

validation.py
  validate_manual_per_capita_calculation()
  validate_tooltip_consistency()
  validate_join_integrity()
  diagnostic_check()
  run_full_audit()
  └── surfaced in app.py as the "Data Quality Audit" expander
```

---

### 13.3 Data Flow Diagram

```
Raw USAFacts CSVs
    ├── covid_confirmed_usafacts.csv
    ├── covid_deaths_usafacts.csv
    └── covid_county_population_usafacts.csv
            │
            ▼ load_data()  [@st.cache_data via get_data()]
    normalize_dataset_metadata()
        - countyFIPS  → zero-padded 5-char string
        - StateFIPS   → zero-padded 2-char string
        - County Name → .strip()
        - Location    → "County Name, State"
            │
            ▼ precompute_all_transforms()  [@st.cache_data]
    ┌─────────────────────────────────────┐
    │  precompute_daily_diffs()           │  → daily_cases, daily_deaths
    │    .diff(axis=1).clip(lower=0)      │    (wide format, NaN→0)
    ├─────────────────────────────────────┤
    │  precompute_all_moving_averages()   │  → ma3/ma5/ma7 cases+deaths
    │    .T.rolling().mean().T            │    (wide format)
    ├─────────────────────────────────────┤
    │  precompute_per_capita()            │  → pc_cases, pc_deaths
    │    (FIPS,State) join                │    (wide format, NaN for
    │    / population * 100_000           │     missing/zero pop)
    ├─────────────────────────────────────┤
    │  get_available_dates()              │  → sorted date string list
    └─────────────────────────────────────┘
            │
            ▼ Per-interaction (UI callbacks)
    ┌──────────────────────────────────────────────────────┐
    │ Map Tab                                              │
    │   prepare_choropleth_for_date(metric_df, date)       │
    │     (FIPS,State) joins for pop, cases, deaths        │
    │     filter_choropleth_by_state()                     │
    │     get_state_bounds_for_zoom()                      │
    │   → px.choropleth()                                  │
    ├──────────────────────────────────────────────────────┤
    │ Trend / County Comparison Tabs                        │
    │   prepare_county_timeseries(df, county, state)       │
    │   calculate_daily_changes()                          │
    │   calculate_per_capita()                             │
    │   apply_moving_average()                             │
    │   → go.Scatter / px.line                             │
    ├──────────────────────────────────────────────────────┤
    │ Lag Tab                                              │
    │   lag_analysis.prepare_daily_per_capita()            │
    │     cumulative → daily → per-100k → MA               │
    │   lag_analysis.detect_peaks()                        │
    │   lag_analysis.match_case_death_peaks()              │
    │   → go.Figure (dual-axis with peak markers)          │
    ├──────────────────────────────────────────────────────┤
    │ Wave Tab                                             │
    │   wave_analysis.calculate_waves_for_county()         │
    │     find_waves(): convolve → find_peaks (scipy)      │
    │     calculate_wave_metrics()                         │
    │   → go.Figure (bars + MA line + wave annotations)    │
    └──────────────────────────────────────────────────────┘
```

---

### 13.4 Strengths of Current Design

**Wide-format data preservation.** The decision to keep all three USAFacts tables in wide format (counties as rows, dates as columns) is correct. It allows vectorized column-wise operations for daily diffs and per-capita calculation, avoids the memory cost of long-format expansion (which would be ~3,000 counties × ~1,000 dates × 3 datasets = ~9M rows), and makes the startup precompute phase fast.

**`(countyFIPS, State)` tuple joins.** All population lookups and join operations use `(countyFIPS, State)` as a compound key. This correctly handles the USAFacts quirk where statewide unallocated records share FIPS `00000` with each other and where a small number of counties have historically shared FIPS codes across states. This fix is documented in `tools.py` and enforced in `precompute_per_capita()`, `prepare_choropleth_for_date()`, and `county_features.py`.

**Startup precomputation with `@st.cache_data`.** All expensive transforms (daily diffs, six moving average tables, per-capita normalization) are computed once at startup and cached. Per-interaction operations are limited to slicing the precomputed tables by county or date, which is fast.

**Validation suite.** The presence of `validation.py` with `validate_manual_per_capita_calculation()`, `validate_tooltip_consistency()`, and `validate_join_integrity()` — and the fact that these are surfaced in the app's Data Quality Audit expander — demonstrates commitment to computational correctness. This is above average for dashboard projects.

**Defensive population handling.** Both `precompute_per_capita()` and `prepare_choropleth_for_date()` filter out `countyFIPS == "00000"` and populations ≤ 0 before calculation. Counties without valid population receive `NaN` rather than zero or an inflated rate.

**`county_features.py` research scaffolding.** The `create_county_feature_table()` function and `add_external_dataset()` with FIPS normalization provide the correct foundation for future socioeconomic correlation studies. The architecture is designed for extension.

**Module separation is appropriate.** Data logic in `tools.py`, UI in `app.py`, validation in `validation.py`, and domain-specific analytics in `wave_analysis.py` / `lag_analysis.py` / `county_features.py` follows clean boundaries. The modules are not deeply entangled.

**Inline documentation of known bugs.** The comment block in `app.py` lines 1029-1045 (documenting the previous `KeyError: 'Value'` and its fix) is an example of good institutional memory. Similar documentation exists in `tools.py` at the top of the file.

**Lag analysis as a proper module.** `lag_analysis.py` is self-contained, documented, and structured to support future multi-county comparison by calling `analyze_county_lag()` twice. The module docstring explicitly notes this design decision.

---

### 13.5 Weaknesses of Current Design

**Dead code in `tools.py`.** Two functions — `prepare_lag_analysis()` (lines 231-259) and `get_county_lag_comparison()` (lines 262-298) — are not imported by `app.py`, `lag_analysis.py`, `wave_analysis.py`, or `county_features.py`. They represent an earlier lag implementation that was superseded by `lag_analysis.py`. Their presence creates confusion about which implementation is authoritative.

**`st.session_state.pop_dict` is built but never read.** Lines 297-299 of `app.py` create a FIPS-to-population dictionary in session state. It is not referenced anywhere else in the file. This is dead code from a previous design iteration.

**Statewide records not filtered in national aggregation.** `compute_national_timeseries()` and `compute_national_daily()` sum all rows in the cases/deaths dataframes without excluding `countyFIPS == "00000"` statewide unallocated rows. If any state has an unallocated row, its cases/deaths will be counted twice in the national total: once in the county rows and once in the statewide row. This is a correctness risk for the National Comparison tab.

**`calculate_daily_changes()` vs `precompute_daily_diffs()` inconsistency.** The precompute pipeline uses `.diff(axis=1).clip(lower=0).fillna(0)`, which forces non-negative daily values. The single-county `calculate_daily_changes()` in `tools.py` (used by trend analysis, county comparison, and lag analysis) uses `.diff().fillna(0).astype(int)` without clipping. The lag analysis pipeline (`prepare_daily_per_capita()`) is therefore exposed to negative daily values from data corrections, which can produce negative per-100k rates and affect peak detection.

**`check_population_validity()` references a hardcoded column name.** `validation.py` line 145 calls `population_df["population"]` directly instead of using `get_population_column()`. If the population column has a different name, this raises a `KeyError`. This function is not called from `app.py` and is therefore a silent break in the standalone diagnostic path.

**Wave analysis inconsistency: `mode='valid'` vs `mode='same'`.** The `find_waves()` function in `wave_analysis.py` uses `np.convolve(..., mode='valid')` to smooth the series before peak detection. The wave chart in `app.py` uses `np.convolve(..., mode='same')` for display. The `mode='valid'` output is shorter than the raw series by `window - 1` points (missing early values), while `mode='same'` is the same length. Peak positions are computed on the trimmed `valid` series, but plotted against the full-length `same` series — the alignment is handled by the index mapping logic but this creates a hard-to-audit discrepancy between what is displayed and what was analyzed.

**`compute_national_per_capita()` reconstructs counts from per-capita values.** This function reverse-computes raw counts from the precomputed per-capita dataframe (`count = pc_value / 100,000 × population`) rather than using the raw counts dataframe directly. This introduces unnecessary floating-point error. The function signature should accept the raw cases/deaths dataframes.

**`st.session_state` caching for national aggregates.** The national series (cumulative, daily, per-capita cases and deaths) are computed inside a `with st.spinner()` block guarded by `if "national_cases_ts" not in st.session_state`. This avoids recomputation on re-renders, but the pattern is fragile: if `@st.cache_data` is cleared (e.g., after a data update), the session state values will not update unless the user refreshes the browser. A `@st.cache_data`-decorated function would be more robust.

**`extract_county_state()` duplicated across tabs.** Tab 2 (`tab_national_comparison`) parses location strings inline. Tab 3 (`tab_county_comparison`) defines a local helper function `extract_county_state()` inside the tab block. Tab 6 and Tab 7 parse inline again. There are four slightly different implementations of the same `"County Name, State"` → `(county, state)` split.

**`apply_chart_styling()` is inconsistently applied.** This helper exists and is called in the Trend Analysis tab. The Map, National Comparison, County Comparison, Lag, and Wave tabs all apply styling inline with repeated `fig.update_layout(...)` calls. The styling is broadly consistent but maintained in multiple places.

**`county_features.py` analytics stubs return input unchanged.** Functions like `prepare_for_rural_urban_analysis()`, `prepare_for_feature_importance_analysis()`, and `prepare_for_clustering_analysis()` return `features_df` or a derived frame without any computation. A caller expecting results would receive the original data silently.

**No version pinning in `requirements.txt`.** All five dependencies are unpinned. Reproducing results requires the same library versions; for a research project, this is a liability.

---

### 13.6 Bug Risks

The following are specific code locations with a credible risk of producing incorrect results or runtime errors.

**BR-1 — National totals double-count statewide rows.**
`tools.py` `compute_national_timeseries()` line 645 and `compute_national_daily()` line 669 sum all rows without filtering `countyFIPS != "00000"`. USAFacts includes statewide unallocated records with their own case/death counts. These are summed in addition to the county-level counts they aggregate from. National case and death totals shown in the National Comparison tab are likely overstated.

**BR-2 — `check_population_validity()` will raise `KeyError`.**
`validation.py` line 145: `population_df["population"].isna().sum()`. The population column name is not always `"population"` — it is discovered dynamically by `get_population_column()` in all other code paths. This function crashes with `KeyError: 'population'` if the column has any other name.

**BR-3 — Lag analysis sees negative daily values.**
`lag_analysis.py` `prepare_daily_per_capita()` calls `calculate_daily_changes()`, which does not clip negative values to zero. Data corrections in the USAFacts source (e.g., when a state revises its historical case counts downward) produce negative daily values. These can create spurious peaks in the `Per100k MA` series and false lag detections.

**BR-4 — Wave boundary detection can produce zero-duration waves.**
`wave_analysis.py` `find_waves()` lines 62-77: if no value below the 10% threshold exists before the start of the series, `start_idx` remains equal to `peak_idx`. Similarly at the end. This produces a wave where `start_date == peak_date` or `end_date == peak_date`, yielding zero-duration waves. This is a silent data quality issue in the wave detail table.

**BR-5 — `pop_dict` session state is dead code that could mask a future bug.**
`app.py` lines 297-299 build `st.session_state.pop_dict` using `countyFIPS` as key (not `(FIPS, State)` tuple). If any code path ever accidentally uses this dictionary for a per-capita lookup instead of the correct `(FIPS, State)` keyed approach in `tools.py`, it will produce incorrect results for counties with duplicate FIPS.

**BR-6 — `validate_manual_per_capita_calculation()` calls `precompute_per_capita()` redundantly.**
`validation.py` line 304: `_, pc_deaths = precompute_per_capita(cases_df, deaths_df, population_df)`. This recalculates per-capita from scratch inside the validation audit rather than validating the already-cached result the dashboard is actually using. If the validation passes but the cached version has a subtle difference (e.g., from a different code path), the audit result would be misleading.

**BR-7 — Wave chart uses `wave['peak_cases']` key for death metrics.**
`app.py` line 2132 and `wave_analysis.py` line 148: the wave detail dictionary always uses key `"peak_cases"` regardless of whether the metric being analyzed is cases or deaths. When `wave_metric == "Deaths"`, the value stored under `"peak_cases"` is actually a death count. This is correctly displayed in the chart (the value is right), but the semantic mismatch between the key name and the metric type is confusing and will cause problems if any downstream code branches on the key name.

**BR-8 — TECHNICAL_DOCUMENTATION.md section 7 describes dead code.**
Section 7 ("Lag Analysis") describes `prepare_lag_analysis()` and `get_county_lag_comparison()` from `tools.py` as the lag implementation. These functions are unused; the actual lag tab uses `lag_analysis.py`. A developer reading section 7 would investigate the wrong module.

---

### 13.7 Statistical Risks

The following affect the validity of computed metrics and should be understood before treating output as research-grade.

**SR-1 — `min_periods=1` in moving averages.**
All moving average calculations (both precomputed wide-format and single-county) use `rolling(window=w, min_periods=1)`. This means the first `w-1` values in any smoothed series are computed from fewer data points than the full window, producing unreliable estimates for the earliest dates. For a 7-day MA, the first six daily values each use a 1-to-6 point window. For COVID data starting in January 2020, this affects the early pandemic period specifically.

**SR-2 — Negative daily values are silently zeroed.**
`precompute_daily_diffs()` applies `.clip(lower=0)` to remove negative values from data corrections. While this prevents downstream visualization artifacts, it conceals reporting corrections and underrepresents the magnitude of backfill events. Any trend analysis that depends on total case accumulation across time will be unaffected (cumulative is used directly), but daily trend analysis during periods with large corrections will appear smoother than reality.

**SR-3 — Static population throughout the pandemic timeline.**
Per-capita calculations use a single population value (2019 estimates from USAFacts) for all dates from 2020 through the dataset's end. For most counties this introduces less than 2% error, but for high-growth or high-decline counties it is a material systematic bias. Any cross-county comparison of per-capita rates that aims to control for population differences should note this limitation.

**SR-4 — Wave boundary threshold (10%) is arbitrary.**
The 10% of peak value threshold for wave start and end dates has no epidemiological derivation. Many COVID waves did not return to 10% between surges — the Alpha and Delta waves in particular showed continuous elevated transmission. Waves in such counties may appear as single merged waves when they are epidemiologically distinct, or may be detected as spanning implausibly long periods.

**SR-5 — Greedy peak matching in lag analysis.**
`match_case_death_peaks()` processes case peaks chronologically, each claiming the nearest subsequent available death peak. This one-pass greedy algorithm is not globally optimal. With multiple closely spaced case peaks and a single large death peak, the matching will assign the death peak to the earliest case peak regardless of biological plausibility.

**SR-6 — NaN treated as zero in peak detection.**
`lag_analysis.py` `detect_peaks()` line 100: `clean_values = np.nan_to_num(values, nan=0.0)`. Missing data days are replaced with zero before `find_peaks()`. A genuine reporting gap (several days of missing data followed by a large batch correction) will appear as a valley-then-spike structure that may be incorrectly identified as a distinct peak.

**SR-7 — Normalized (Index=100) comparison uses first non-zero value as baseline.**
`app.py` lines 760-776: the index baseline is the first non-zero, non-NaN value for each series. For counties that had zero cases for weeks at the start of the pandemic, this baseline date differs substantially from the national baseline date, making the relative growth curves start from different epidemic moments and potentially misleading if interpreted as tracking the same event.

**SR-8 — National per-capita double-reconstructs values.**
`compute_national_per_capita()` reverses the per-capita computation (multiplies per-100k values by population / 100,000 to get counts) and then recalculates per-capita from the summed counts. Each county's reconstructed count will have floating-point rounding error. While the magnitude of this error is small (sub-percent), it is unnecessary: the function should use the raw cases/deaths dataframes.

---

### 13.8 Performance Risks

**PR-1 — Row-wise `apply()` in `precompute_per_capita()`.** 
`tools.py` line 422-425:
```python
pops = cases_df.apply(
    lambda row: pop_dict.get((row["countyFIPS"], row["State"]), np.nan),
    axis=1
).values
```
This calls a Python lambda once per county row. For ~3,000 counties this is fast at startup, but it is the slowest way to perform a lookup. A vectorized merge with the population table on `["countyFIPS", "State"]` would be both faster and more idiomatic, consistent with how other join operations in the codebase are performed.

**PR-2 — `precompute_moving_averages()` allocates multiple full dataframe copies.**
`tools.py` lines 352-353:
```python
ma_cases_vals = cases_numeric.T.rolling(window=window, min_periods=1).mean().T.reset_index(drop=True)
```
The transpose `.T` on a (3000 counties × 1000 dates) dataframe creates a (1000 × 3000) intermediate dataframe. Rolling is then applied across 3000 columns. The second `.T` creates another full copy. With three window sizes, this pattern runs six times at startup. For the current dataset this is acceptable, but growth in date columns (data through 2024) will make this increasingly expensive.

**PR-3 — `prepare_choropleth_for_date()` is uncached.**
`app.py` line 501 calls `prepare_choropleth_for_date()` on every map interaction without caching. The function performs four left-merge operations on the full county dataframe (~3,000 rows), which is fast individually but is re-executed on every slider movement. Adding `@st.cache_data` with the metric_df hash, date string, and state arguments as cache keys would eliminate redundant computation on repeated visits to the same date/metric combination.

**PR-4 — Validation functions recompute per-capita from scratch.**
Both `validate_manual_per_capita_calculation()` and `validate_tooltip_consistency()` call `precompute_per_capita()` internally. This means the validation audit reruns the full per-capita precomputation rather than validating the already-cached result from the dashboard's `transforms` dictionary. Each validation button click triggers a full recomputation, which is why the audit warning says "30-60 seconds."

**PR-5 — `calculate_waves_for_all_counties()` iterates with `for idx, row in df.iterrows()`.**
`wave_analysis.py` lines 256-258. This function is not called from `app.py` (which calls `calculate_waves_for_county()` for a single county), but if it is ever invoked for batch analysis of all counties, iterating with `iterrows()` across 3,000 counties while calling `calculate_waves_for_county()` (which itself does four filtered dataframe lookups per county) will be very slow — potentially minutes.

**PR-6 — GeoJSON loaded from CDN on every full page load.**
`app.py` line 545: the choropleth uses the Plotly county GeoJSON hosted on GitHub raw content. Plotly fetches and caches this in the browser, but the URL is fetched on every fresh browser session. This is a network dependency that can slow initial map rendering and will fail in offline or restricted network environments.

---

### 13.9 Documentation Recommendations

The current two-file documentation structure (`README.md`, `TECHNICAL_DOCUMENTATION.md`) is the correct architecture. Both files are well-written. The following specific corrections and additions are needed.

**Correction required — Section 7 (Lag Analysis):** Section 7 describes `prepare_lag_analysis()` and `get_county_lag_comparison()` from `tools.py` as the lag analysis implementation. These functions are not used by the lag tab or any other tab. The actual implementation lives in `lag_analysis.py`. Section 7 should be rewritten to describe `lag_analysis.py`: `prepare_daily_per_capita()`, `detect_peaks()`, `match_case_death_peaks()`, `analyze_county_lag()`, and `summarize_lag_results()`.

**Correction required — Section 8 (Wave Detection):** Section 8 states "Wave detection is implemented as a research module and is not yet a primary dashboard tab." It is now a primary tab (Tab 7: Wave Analysis). This sentence should be removed or updated.

**Addition recommended — Document the wave convolution modes:** The difference between `mode='valid'` in `find_waves()` and `mode='same'` in the wave chart rendering (app.py) should be explicitly noted. The index mapping logic that compensates for this is not obvious and a future developer could break it while intending to simplify the smoothing code.

**Addition recommended — Document `county_features.py` status:** The module should be explicitly labeled as "internal research scaffolding, not user-facing." The analytics stub functions should carry a note that they return the input unchanged until implemented.

**README.md accuracy:** The README description of the lag analysis tab is accurate. The wave analysis section is accurate. No changes are needed in README.md beyond keeping it aligned with future UI changes.

---

### 13.10 Refactoring Roadmap

Items are ordered by priority. For each item: a description, the rationale, the estimated difficulty, and the risk of implementation.

#### Priority 1: Critical Fixes

These should be completed before any results from the dashboard are used in research or cited externally.

**P1-A — Filter statewide rows in national aggregation.**
`compute_national_timeseries()` and `compute_national_daily()` must exclude rows where `countyFIPS == "00000"` before summing. This is consistent with how all other pipeline functions handle statewide unallocated rows.
- Rationale: National totals are likely overstated by the sum of statewide unallocated rows.
- Difficulty: Trivial (two-line addition in `tools.py`).
- Risk: Low. Should be verified with a comparison before/after against an independent total.

**P1-B — Clip negative values in `calculate_daily_changes()`.**
Add `.clip(lower=0)` and remove `.astype(int)` from `calculate_daily_changes()` in `tools.py` to make it consistent with `precompute_daily_diffs()`.
- Rationale: The lag analysis is exposed to negative daily values; the precomputed pipeline is not. This inconsistency affects peak detection.
- Difficulty: Trivial.
- Risk: Low. The behavioral change only affects periods with data corrections (negative diffs).

**P1-C — Fix `check_population_validity()` hardcoded column name.**
Replace `population_df["population"]` with `get_population_column(population_df)` on line 145 of `validation.py`.
- Rationale: The function crashes with `KeyError` for any population dataframe where the column isn't literally named "population."
- Difficulty: Trivial.
- Risk: None.

**P1-D — Remove dead lag functions from `tools.py`.**
Delete `prepare_lag_analysis()` (lines 231-259) and `get_county_lag_comparison()` (lines 262-298). Confirm no callers exist before deletion.
- Rationale: Dead code creates confusion about which lag implementation is authoritative.
- Difficulty: Trivial.
- Risk: Very low. A grep for callers should confirm zero references.

**P1-E — Remove unused `st.session_state.pop_dict`.**
Delete `app.py` lines 297-299.
- Rationale: Dead code with a dangerous property: it uses FIPS alone as a key (not `(FIPS, State)`) and could be accidentally used in a future code path.
- Difficulty: Trivial.
- Risk: None.

**P1-F — Pin `requirements.txt` to specific versions.**
Run `pip freeze` in the project virtual environment and pin all five dependencies to their tested versions.
- Rationale: Unpinned dependencies prevent reproducible research results.
- Difficulty: Trivial.
- Risk: None (the pinned versions are what is already running).

#### Priority 2: Architecture Improvements

**P2-A — Replace `st.session_state` national caching with `@st.cache_data`.**
Wrap `compute_national_timeseries()`, `compute_national_daily()`, and `compute_national_per_capita()` calls in a single `@st.cache_data`-decorated function analogous to `precompute_all_transforms()`.
- Rationale: The session state pattern does not survive a `@st.cache_data` clear; a cached function will recompute correctly and automatically.
- Difficulty: Easy.
- Risk: Low.

**P2-B — Vectorize population lookup in `precompute_per_capita()`.**
Replace the `.apply(lambda row: pop_dict.get(...), axis=1)` with a merge: `cases_df[["countyFIPS", "State"]].merge(pop_table, on=["countyFIPS", "State"], how="left")["population"].values`.
- Rationale: Vectorized merge is more readable and idiomatic; row-wise apply is the slowest Pandas pattern.
- Difficulty: Easy.
- Risk: Low. Must validate results equal the current output using the validation suite.

**P2-C — Fix `compute_national_per_capita()` to use raw counts.**
Change the function signature to accept `cases_df` and `deaths_df` (raw counts) in addition to `population_df`, and compute national per-capita directly from summed raw counts rather than reconstructing counts from the per-capita table.
- Rationale: Eliminates unnecessary floating-point round-trip error and makes the function conceptually simpler.
- Difficulty: Easy.
- Risk: Low. Values will be slightly different (more accurate) than current output.

**P2-D — Consolidate `extract_county_state()` into a shared helper.**
Create a module-level function in `tools.py` and import it into `app.py`. Remove the four inline/redeclared implementations.
- Rationale: Eliminates duplication and ensures consistent handling of edge cases (e.g., county names that contain commas).
- Difficulty: Trivial.
- Risk: Low.

**P2-E — Rename `"peak_cases"` key to `"peak_value"` in wave metrics.**
In `calculate_wave_metrics()`, rename the `"peak_cases"` key in the wave detail dict to `"peak_value"`. Update the one reference in `app.py` (`wave['peak_cases']` on line 2132).
- Rationale: The key is used for both case and death metrics; the name is actively misleading when analyzing deaths.
- Difficulty: Easy.
- Risk: Low. A search for all usages of `['peak_cases']` will identify every reference.

**P2-F — Extract tab content into functions.**
Extract each tab's content in `app.py` into a named function (`render_map_tab()`, `render_national_comparison_tab()`, etc.). Call the functions from the main tab block.
- Rationale: `app.py` is 2,258 lines with all tab logic inline. Extracting into functions makes individual sections testable and navigable.
- Difficulty: Medium (mechanical refactor with no logic changes).
- Risk: Medium. Streamlit widget keys, session state, and cached data references must all be preserved exactly. A tab-by-tab approach with testing between each extraction reduces risk.

**P2-G — Cache `prepare_choropleth_for_date()`.**
Wrap the call in `app.py` with `@st.cache_data` using the metric dataframe's hash, the date string, and the state filter as cache keys.
- Rationale: Eliminates re-running four merge operations on every map slider movement.
- Difficulty: Easy.
- Risk: Low. Streamlit can hash Pandas DataFrames natively.

**P2-H — Update TECHNICAL_DOCUMENTATION.md sections 7 and 8.**
Rewrite section 7 to describe `lag_analysis.py`. Remove the outdated statement in section 8 that wave analysis is "not yet a primary dashboard tab."
- Rationale: Accuracy.
- Difficulty: Trivial.
- Risk: None.

#### Priority 3: Statistical Validity Improvements

These improve the quality of computed metrics for research use but require care to implement correctly.

**P3-A — Document `min_periods=1` limitation.**
Add a visible note to the trend analysis and lag analysis tabs explaining that moving average values for the first `window-1` dates are computed from fewer data points than the full window.
- Rationale: Transparency for research users.
- Difficulty: Trivial (UI text change only).
- Risk: None.

**P3-B — Expose negative daily values as explicit data quality flags.**
Instead of clipping negative values to zero, retain them and surface them in the UI as a "data correction event" indicator on the trend chart.
- Rationale: Data corrections are meaningful events (state revisions, backfills) that analysts should be aware of rather than have silently zeroed.
- Difficulty: Medium (requires changes to the chart rendering and raw data display).
- Risk: Low.

**P3-C — Standardize wave smoothing mode.**
Use `mode='same'` consistently in both `find_waves()` and the wave chart rendering, or document precisely why `mode='valid'` is used for detection and how the index mapping is compensated.
- Rationale: Consistency and auditability.
- Difficulty: Easy (one-line change in `wave_analysis.py` with careful re-testing of index mapping).
- Risk: Medium. The current index mapping code is tied to `mode='valid'` output length. Switching to `mode='same'` changes the offset and requires updating the mapping.

#### Priority 4: Future Feature Preparation

These must be completed before new datasets or research features are added.

**P4-A — Implement `county_features.py` analytics stubs.**
Fill in the `TODO` blocks in `prepare_for_correlation_analysis()`, `prepare_for_regression_analysis()`, `prepare_for_clustering_analysis()`, and `prepare_for_rural_urban_analysis()`. Use `scipy.stats` for correlation (Pearson, Spearman, p-values) and `sklearn` for regression and clustering.
- Rationale: The scaffolding is in place; the analytics are not. These are prerequisite to any external dataset integration with statistical validation.
- Difficulty: High. Statistical validity of each method requires care (distributional assumptions, multiple testing, confounding).
- Risk: Medium. Bad statistical implementations are worse than none.

**P4-B — Create deterministic test fixtures.**
Create a `tests/` directory with pytest tests that use fixed, known counties and dates to verify specific calculation outputs. Example: test that Alameda County, CA on 2021-01-15 produces a specific deaths-per-100k value given the known population.
- Rationale: The current validation suite samples randomly. Deterministic fixtures catch regressions introduced by refactoring.
- Difficulty: Medium.
- Risk: None.

**P4-C — Create `data_sources.py` module.**
Create a new module for loading, normalizing, and joining external datasets (healthcare access, education, socioeconomic indicators, vaccination rates). The module should follow the same FIPS normalization pattern as `normalize_dataset_metadata()` and expose a `load_external_dataset(path, fips_col, ...)` interface compatible with `add_external_dataset()`.
- Rationale: Without a standardized external data loader, each new dataset will be integrated ad hoc in app.py, bypassing the clean join contract established in `county_features.py`.
- Difficulty: Medium.
- Risk: Low (additive change).

**P4-D — Add animated choropleth.**
Use Plotly's `animation_frame` parameter in `px.choropleth()` to enable date animation. The wide-format data structure makes this straightforward: melt the precomputed metric dataframe to long format and pass the date column as `animation_frame`.
- Rationale: Frequently requested feature; enables visual exploration of pandemic wave propagation.
- Difficulty: Medium (melting large dataframes in the browser is memory-intensive; the animation frame data must be prepared efficiently).
- Risk: Low.

**P4-E — Replace GeoJSON CDN fetch with bundled GeoJSON.**
Download the Plotly county GeoJSON file and include it in the project's `data/` directory. Load it at startup with `json.load()` instead of fetching the CDN URL at render time.
- Rationale: Eliminates the network dependency on GitHub raw content for map rendering. Required for offline or restricted-network deployment.
- Difficulty: Easy.
- Risk: Low.

---

### 13.11 High-Risk Areas

These are the specific code locations most likely to cause incorrect results or hard-to-debug failures as the project grows.

**RISK-1 (HIGH) — Statewide double-count in national totals** (`tools.py` `compute_national_timeseries()` and `compute_national_daily()`). Any analysis comparing county rates to national rates will use an inflated national denominator. This is the most likely source of a visible incorrect result for a research user.

**RISK-2 (MEDIUM) — Wave index mapping** (`wave_analysis.py` `find_waves()` lines 49-55). The index chain from `valid_indices → peak_idx → ma_window//2 offset → original_peak_idx` is correct for the current assumptions but is fragile. Any change to NaN handling, smoothing mode, or the validity mask will break peak date assignments silently. The only way to catch this regression is with deterministic test fixtures.

**RISK-3 (MEDIUM) — `(FIPS, State)` join discipline.** The correct join key is `(countyFIPS, State)`, consistently used in `precompute_per_capita()`, `prepare_choropleth_for_date()`, and `county_features.py`. The dead-code `pop_dict` in `app.py` uses FIPS alone as a key. If a future developer copies this pattern instead of the correct one, per-capita values for counties with shared FIPS codes will be silently wrong.

**RISK-4 (MEDIUM) — `county_features.py` stubs return input unchanged.** If a future developer calls `prepare_for_correlation_analysis()` expecting a correlation result, they receive the input dataframe back. There is no error, no warning, and no indication that the function is a stub. The risk increases when a UI tab is wired to one of these functions before it is implemented.

**RISK-5 (LOW-MEDIUM) — `min_periods=1` in early-pandemic moving averages.** For any analysis that emphasizes the early pandemic period (January–March 2020), the first several MA values are computed from 1-6 points rather than the full window. Trend comparisons anchored to this period may produce misleading relative growth estimates. This is a disclosure risk for research use more than a code risk.

**RISK-6 (LOW) — GeoJSON CDN dependency.** A CDN outage or network restriction will cause the map tab to render with no county fill colors. The error will appear as an empty choropleth rather than a visible error message, which may be misinterpreted as a data problem rather than a network problem.

---

*Audit completed 2026-06-15. No files were modified as part of this audit. All findings are recommendations; implementation requires separate review and testing.*

---

## 14. AHRF Dataset Audit (2026-06-16)

This section records the Phase 1 audit of the Area Health Resources Files (AHRF) uploaded to the `DATA/` folder. No dashboard code was modified as part of this audit.

---

### 14.1 Files Found in DATA/

| File | Size | Format | Usable |
|------|------|--------|--------|
| `ahrf2023.csv` | 39 MB | Delimited CSV (latin-1) | **Yes — primary file** |
| `ahrf2022.asc` | 100 MB | Fixed-width text | Requires SAS codebook to parse |
| `ahrf2021.asc` | 101 MB | Fixed-width text | Requires SAS codebook to parse |
| `AHRF2020.asc` | 99 MB | Fixed-width text | Requires SAS codebook to parse |

The three `.asc` files are HRSA's original fixed-width format. Each record is a single long character string with field positions defined in a companion SAS format file (not uploaded). Without the SAS format dictionary, column boundaries cannot be determined reliably. These files are archived for reference but **`ahrf2023.csv` is the sole usable source**.

---

### 14.2 Dataset Profile: ahrf2023.csv

| Property | Value |
|---|---|
| Rows | 3,231 |
| Columns | 4,306 |
| Encoding | latin-1 |
| Row granularity | One row per county (all rows are county-level; no state-aggregate rows present) |
| Year coverage | Primarily 2020–2023, with some historical columns back to 2010 |
| Geographic scope | 50 states + DC + territories (PR, GU, VI, AS, MP) |

---

### 14.3 Geographic Identifiers

| Column | Description | Unique Values | Null Count | Sample |
|---|---|---|---|---|
| `fips_st_cnty` | 5-digit county FIPS (integer, no leading zero) | 3,231 | 0 | 1001 |
| `fips_st` | 2-digit state FIPS | 54 | 0 | 1 |
| `fips_cnty` | 3-digit county FIPS within state | 332 | 0 | 001 |
| `cnty_name` | County name (short form, no state) | 1,937 | 0 | Autauga |
| `cnty_name_st_abbrev` | "County Name, ST" composite | 3,231 | 0 | Autauga, AL |
| `st_name` | Full state name | 54 | 0 | Alabama |
| `st_name_abbrev` | 2-letter state abbreviation | 54 | 0 | AL |

**Join strategy:** Zero-pad `fips_st_cnty` to 5 characters → matches `countyFIPS` in the COVID dataset exactly. This is a pure numeric FIPS join with no string ambiguity.

---

### 14.4 FIPS Match Analysis Against COVID Dataset

| Metric | Count |
|---|---|
| COVID dataset unique county FIPS | 3,142 |
| AHRF 2023 unique county FIPS | 3,231 |
| Counties matched (in both datasets) | **3,142 (100.0%)** |
| COVID FIPS not found in AHRF | **0** |
| AHRF FIPS not in COVID | 89 (Puerto Rico boroughs, territories, Alaska borough changes, and a few extinct VA independent cities) |

**Every county in the COVID dataset has a corresponding AHRF record.** No imputation or fuzzy matching is required.

---

### 14.5 Variable Inventory and Priority Rankings

All 4,306 AHRF columns are multi-year variants of a smaller core variable set (e.g., `phys_nf_prim_care_pc_exc_rsdt_21` and `phys_nf_prim_care_pc_exc_rsdt_20` are the same measure for 2021 and 2020 respectively). For dashboard integration, the most recent year (typically 2021 for physician/hospital data, 2020 for Census-derived data) is used.

#### HIGH PRIORITY — Direct COVID outcome correlates, low missing data

| AHRF Column | Label | Missing % | Notes |
|---|---|---|---|
| `rural_urban_contnm_13` | USDA Rural-Urban Continuum Code 2013 | 0.3% | Official 1–9 RUCC scale; replaces population-threshold proxy |
| `urban_influnc_13` | USDA Urban Influence Code 2013 | 0.3% | 1–12 scale; complementary to RUCC |
| `phys_nf_prim_care_pc_exc_rsdt_21` | Primary care physicians (non-fed, excl. residents) 2021 | 0.2% | Count; must divide by population for rate |
| `md_nf_activ_21` | Total active non-federal MDs 2021 | 0.2% | Count; 236 counties have 0 primary care physicians |
| `hosp_beds_21` | Total hospital beds 2021 | 0.2% | Includes all facility types |
| `stgh_med_surg_icu_beds_21` | Med-surg ICU beds (short-term general hospitals) 2021 | 0.2% | Most relevant ICU capacity measure; median = 0 (most counties) |
| `snf_beds_21` | Skilled nursing facility beds 2021 | 0.2% | Long-term care capacity |
| `critcl_access_hosp_21` | Critical access hospitals 2021 | 0.2% | Count; measures rural hospital infrastructure |
| `hpsa_prim_care_23` | Primary care HPSA designation 2023 | 0.2% | 0=none, 1=whole county, 2=partial; healthcare shortage indicator |
| `cens_popn_20` | Census population 2020 | 0.2% | Denominator for all per-capita calculations |
| `popn_densty_per_squr_mi_20` | Population density per sq mi 2020 | 0.3% | Median 46.4; extreme right skew |
| `popn_est_ge65_21` | Estimated population age 65+ 2021 | 0.3% | COVID mortality most concentrated in this group |
| `medn_age_20` | Median age 2020 | 0.2% | County mean 42.3 years |
| `medn_famly_incom_21` | Median family income 2021 | 0.3% | County mean $71,564 |
| `unemply_rate_ge16_21` | Unemployment rate (age 16+) 2021 | 0.3% | Percentage; mean 4.8% |
| `pers_lt_hsd_ge25_pct_21` | % persons without HS diploma (age 25+) 2021 | 0.3% | Mean 12.3%; direct SES proxy |
| `pers_4yrs_collg_ge25_pct_21` | % persons with 4-year college degree (age 25+) 2021 | 0.3% | Mean 23.0% |
| `urban_popn_pct_20` | Urban population % 2020 | 0.3% | Mean 37.1%; alternative to RUCC for continuous analysis |

#### MEDIUM PRIORITY — Useful but higher missingness or less direct COVID relationship

| AHRF Column | Label | Missing % | Notes |
|---|---|---|---|
| `per_cap_persnl_incom_21` | Per capita personal income 2021 | 3.6% | Alternative income measure to median family |
| `child_povty_famls_5_17_pct_21` | Child poverty % (families with children 5–17) 2021 | 2.8% | SES indicator |
| `hpsa_mentl_hlth_23` | Mental health HPSA designation 2023 | 0.2% | Behavioral health access |
| `cbsa_ind_20` | CBSA metro/micro/nonmetro indicator 2020 | 0.3% | 0=nonmetro, 1=metro, 2=micro |
| `mort_3yr_65_74_avg_21` | Mortality rate age 65–74 (3yr avg) 2021 | 7.4% | Pre-COVID baseline health |
| `mort_3yr_55_64_avg_21` | Mortality rate age 55–64 (3yr avg) 2021 | 12.7% | Pre-COVID baseline health |
| `prstnt_povty_typolgy_14` | Persistent poverty county flag 2014 | 2.7% | Binary typology; structural disadvantage |
| `hi_povty_typolgy_14` | High poverty county typology flag 2014 | 0.3% | Binary |
| `rural_hlth_clincs_21` | Rural health clinics 2021 | 0.2% | Safety-net infrastructure |
| `md_nf_all_med_spec_21` | Total active MD specialists 2021 | 0.2% | Specialist vs. generalist ratio |

#### LOW PRIORITY — Available but less actionable for COVID analysis at this stage

| AHRF Column | Label | Missing % | Notes |
|---|---|---|---|
| `inf_mort_rate_5yr_lt1_avg_21` | Infant mortality rate (5yr avg) 2021 | **85.8%** | Too sparse — exclude |
| `mort_3yr_45_54_avg_21` | Mortality rate age 45–54 (3yr avg) 2021 | 31.9% | High missingness |
| `lo_eductn_typolgy_15` | Low education county typology flag 2015 | 2.7% | Binary; superseded by pct columns |
| `stgh_card_icu_beds_21` | Cardiac ICU beds 2021 | 0.2% | Very skewed; most counties = 0 |
| `snf_beds_22` | SNF beds 2022 | 0.2% | Use 2021 for consistency |

---

### 14.6 USDA RUCC Code Distribution

The AHRF includes the official USDA Rural-Urban Continuum Codes (2013), which replace the population-threshold classification currently used in the dashboard.

| RUCC | Counties | % | Description |
|---|---|---|---|
| 1 | 472 | 14.6% | Metro — 1M+ population |
| 2 | 395 | 12.2% | Metro — 250K–1M population |
| 3 | 369 | 11.4% | Metro — <250K population |
| 4 | 217 | 6.7% | Nonmetro — urban ≥20K, adjacent to metro |
| 5 | 92 | 2.8% | Nonmetro — urban ≥20K, not adjacent |
| 6 | 597 | 18.5% | Nonmetro — urban 2.5K–20K, adjacent |
| 7 | 434 | 13.4% | Nonmetro — urban 2.5K–20K, not adjacent |
| 8 | 220 | 6.8% | Nonmetro — urban <2.5K, adjacent |
| 9 | 425 | 13.2% | Nonmetro — urban <2.5K, not adjacent |

**RUCC 1–3 = Metro (1,236 counties, 38.2%)** | **RUCC 4–9 = Nonmetro (1,985 counties, 61.5%)**

For dashboard filtering: Metro = RUCC 1–3, Rural = RUCC 4–9 (replaces population ≥/< 100K).

---

### 14.7 Key Derived Variables to Compute at Load Time

These are not present in AHRF as rates but must be computed from the raw counts:

| Derived Variable | Formula |
|---|---|
| Primary care physicians per 100k | `(phys_nf_prim_care_pc_exc_rsdt_21 / cens_popn_20) × 100,000` |
| Total MDs per 100k | `(md_nf_activ_21 / cens_popn_20) × 100,000` |
| Hospital beds per 100k | `(hosp_beds_21 / cens_popn_20) × 100,000` |
| ICU beds per 100k | `(stgh_med_surg_icu_beds_21 / cens_popn_20) × 100,000` |
| SNF beds per 100k | `(snf_beds_21 / cens_popn_20) × 100,000` |
| Pct population 65+ | `(popn_est_ge65_21 / cens_popn_20) × 100` |
| HPSA shortage flag | `hpsa_prim_care_23 >= 1` (any designation = shortage area) |
| Metro/rural flag | `rural_urban_contnm_13 <= 3` → Metro; else Nonmetro |

---

### 14.8 Integration Plan

#### Phase 2: Master County Feature Table

A new file `ahrf_loader.py` will be created to load, clean, and export the AHRF feature set. It will be called from `county_features.py`'s `create_county_feature_table()` via `add_external_dataset()`.

**Selected columns for master table (27 variables):**

```
Geographic:     fips_padded, st_name_abbrev, cnty_name,
                rural_urban_contnm_13, urban_influnc_13,
                cbsa_ind_20, urban_popn_pct_20

Healthcare:     phys_nf_prim_care_pc_exc_rsdt_21, md_nf_activ_21,
                hosp_beds_21, stgh_med_surg_icu_beds_21, snf_beds_21,
                critcl_access_hosp_21, hpsa_prim_care_23,
                [derived: pcp_per_100k, hosp_beds_per_100k, icu_beds_per_100k]

Demographics:   cens_popn_20, popn_densty_per_squr_mi_20,
                popn_est_ge65_21, medn_age_20,
                [derived: pct_ge65]

Economic:       medn_famly_incom_21, per_cap_persnl_incom_21,
                unemply_rate_ge16_21, child_povty_famls_5_17_pct_21,
                prstnt_povty_typolgy_14

Education:      pers_lt_hsd_ge25_pct_21, pers_4yrs_collg_ge25_pct_21
```

#### Phase 3: County Factors Tab

A new "County Factors" tab added to the dashboard with:
- COVID outcome selector (cases per 100k, deaths per 100k, wave count, largest wave, lag)
- Comparison variable selector (grouped by category)
- Scatter plot with OLS trend line
- Pearson + Spearman correlation coefficients with p-values
- Summary statistics table
- Rural/urban coloring option

#### Phase 4: RUCC-Based Rural/Urban Upgrade

Replace the population-threshold `classify_county_type()` in `tools.py` with an RUCC-based version once the master table is built. The function contract (returns `County_Type` column) stays the same; only the classification logic changes.

#### Phase 5: Research Analytics Infrastructure

`county_features.py` stubs (`prepare_for_correlation_analysis`, `prepare_for_regression_analysis`, `prepare_for_clustering_analysis`) will be implemented using `scipy.stats` for correlations and `sklearn` or `statsmodels` for regression and clustering once the master table is validated.

---

### 14.9 Integration Risks and Mitigations

| Risk | Mitigation |
|---|---|
| AHRF FIPS includes territories (PR, GU) not in COVID dataset | Filter to matched FIPS only; 89 AHRF-only rows are dropped |
| .asc files from 2020–2022 are unreadable without SAS codebook | Use 2023 CSV only; note year mismatch for COVID-period variables |
| Per-capita rates require 2020 Census population as denominator | Use `cens_popn_20` consistently across all rate calculations |
| Some 2021 economic variables have 2–4% missingness | Impute with state median for scatter plots; flag nulls in regression |
| ICU beds: median = 0 (most counties have no ICU) | Use log+1 transform or binary presence/absence for analysis |
| Infant mortality: 85.8% missing | Exclude from dashboard; document as unavailable |
| RUCC codes are from 2013, not 2020 | Document the vintage limitation; 2013 codes are the most recent USDA release |

---

### 14.10 Files to Create (Implementation Roadmap)

| File | Purpose |
|---|---|
| `ahrf_loader.py` | Load ahrf2023.csv, select priority columns, compute derived rates, export clean DataFrame |
| Update `county_features.py` | Wire `ahrf_loader.py` into `create_county_feature_table()` |
| Update `tools.py` | Upgrade `classify_county_type()` to use RUCC codes |
| Update `app.py` | Add "County Factors" tab; upgrade rural/urban filter to RUCC |
| Update `requirements.txt` | Add `scipy>=1.9.0` (already present); add `statsmodels` if regression is implemented |

*AHRF audit completed 2026-06-16. No dashboard files were modified. Implementation begins in Phase 2.*

---

## Section 15 — Statistical Modeling & Outcome Drivers Tab

### 15.1 Overview

The **Statistical Modeling** tab moves beyond visualization to identify which county-level characteristics are most strongly associated with COVID outcomes. It answers: *"Why did some counties experience worse outcomes than others?"*

New file: `modeling.py` — pure data functions, no Streamlit dependency.
New entry in `requirements.txt`: `scikit-learn>=1.0`.

All model fitting is wrapped in `@st.cache_data` in `app.py` so expensive computations run once per session and are re-run only when inputs (filters, outcome, predictor set) change.

---

### 15.2 Section 1 — Correlation Matrix

**Purpose:** Quickly identify which structural factors have the strongest univariate association with a selected COVID outcome.

**Implementation:** `modeling.compute_all_correlations()` calls `scipy.stats.pearsonr` and `scipy.stats.spearmanr` on each factor × outcome pair using pairwise complete cases (counties missing either column are excluded per pair).

**Output:**
- Sortable table: Factor | Pearson r | Pearson p | Spearman ρ | Spearman p | N
- Full correlation heatmap (Plotly `go.Heatmap`, RdBu diverging scale, −1 to +1)

**Filtering:** Respects the global state / region / metro filter applied at the top of the tab. National medians are recomputed on the filtered subset.

**Assumptions:** Pearson r assumes a roughly linear monotonic relationship. Spearman ρ is rank-based and robust to non-linearity. Neither controls for confounders — see Section 15.4 for multivariate control.

---

### 15.3 Section 2 — Feature Importance (Random Forest)

**Purpose:** Rank all AHRF factors by their contribution to predicting a selected COVID outcome, accounting for non-linear interactions.

**Method:**

```
sklearn.ensemble.RandomForestRegressor(n_estimators=200, random_state=42)
```

Missing feature values are imputed with column-wise medians (`modeling._impute_median`, numpy) before fitting. The model trains on the full filtered dataset in one pass; the reported importances are scikit-learn's impurity-based `feature_importances_` (no cross-validation).

**Fallback (scikit-learn not installed):** Substitutes `|Pearson r|` as a linear importance proxy. The `Method` column in the output table identifies which was used.

**Interpretation note:** Feature importance reflects predictive contribution, not causal influence. Two correlated predictors will split importance between them. Always interpret alongside the correlation matrix and regression results.

---

### 15.4 Section 3 — Multivariable OLS Regression

**Purpose:** Test whether a factor remains significantly associated with an outcome *after accounting for other variables* (confounding control).

**Implementation:** Pure numpy + scipy OLS via normal equations:

```
β = (X'X)⁻¹ X'y       (via np.linalg.lstsq for numerical stability)
Var(β) = s² (X'X)⁻¹   where s² = SSR / (n − p)
SE(βᵢ) = √Var(β)ᵢᵢ
t-stat  = βᵢ / SE(βᵢ)
p-value = 2 · Pr(|T| > |t-stat|)   with T ~ t(n−p)
```

No statsmodels dependency. Standard errors assume **homoscedastic residuals** (OLS SE). Heteroscedasticity-robust (HC3) standard errors are not implemented; caution is warranted when residual variance is clearly non-constant.

**Output:**
- Regression table: Variable | Coefficient | Std Error | t-stat | p-value | 95% CI
- Model summary: R², adjusted R², N, F-statistic, F p-value
- Rule-based interpretation bullets from `modeling.generate_ols_interpretation()`

**Rule-based interpretation logic:**
- Classifies model fit as strong (R² ≥ 0.5), moderate (≥ 0.25), or weak
- Lists each predictor with p < 0.05 and its direction
- Lists non-significant predictors
- Appends a standard causal-inference caveat

**Missing values:** Handled by complete-case (listwise) deletion. Counties missing any selected predictor or the outcome are excluded. N reflects complete cases.

---

### 15.5 Section 4 — County Resilience Score

**Purpose:** Identify counties that performed better or worse than expected given their structural characteristics. This is the primary novel analytical output of the dashboard.

**Concept:**

> Train a model predicting `deaths_per_100k` (or another outcome) from AHRF structural factors. For each county, compare the model's prediction to the actual observed value.
>
> **Resilience Score = Predicted − Actual**
> - Positive → county achieved fewer deaths/cases than the model expected
> - Negative → county experienced more deaths/cases than expected

**Method — cross-validated to prevent data leakage:**

```
sklearn.model_selection.cross_val_predict(
    RandomForestRegressor(n_estimators=200),
    X, y, cv=k
)
```

Each county's prediction comes from a model trained on all *other* counties in its complementary folds. This ensures the score reflects genuine out-of-sample predictability rather than model memorisation.

**Fallback (scikit-learn not installed):** `modeling._kfold_ols_predict()` implements k-fold OLS cross-prediction using `np.linalg.lstsq`. Results are less powerful (linear only) but unbiased.

**Missing feature imputation:** Column-wise medians (before splitting folds).

**Map:** Plotly `px.choropleth` using county FIPS codes. Color scale: `RdBu` diverging at 0, clipped to the 98th percentile of absolute scores to prevent extreme outliers from compressing the colour range.

**Limitations:**
- Structural characteristics explain only a portion of outcome variance. Policy, testing capacity, reporting, and chance also matter.
- Cross-validated predictions are approximately unbiased but the score distribution depends on model specification (included features, RF hyperparameters).
- Positive resilience score does not imply good policy — it may reflect unmeasured protective factors.

---

### 15.6 Section 5 — County Explorer

**Purpose:** Per-county summary combining COVID outcomes, healthcare capacity, socioeconomic conditions, and national percentile context.

**National percentile:** Computed with `scipy.stats.percentileofscore(national_series, county_value, kind="rank")` across all counties in the master table (not the filtered subset, so the reference population is always the full national dataset).

**Color coding:** ≥ 75th percentile → red; ≤ 25th percentile → green. Note that high percentile is *unfavorable* for outcome metrics (cases, deaths) and *favorable* for resource metrics (PCPs, hospital beds).

---

### 15.7 File Changes Summary

| File | Change |
|---|---|
| `modeling.py` | New — contains `compute_all_correlations`, `run_rf_feature_importance`, `run_ols_regression`, `compute_resilience_scores`, `generate_ols_interpretation` |
| `requirements.txt` | Added `scikit-learn>=1.0` |
| `app.py` | Added `from modeling import ...`; added `_cached_*` wrappers; added `render_modeling_tab()`; added "Statistical Modeling" to `st.tabs()` |

---

### 15.8 Statistical Assumptions and Limitations

| Assumption | Where Applied | Consequence if Violated |
|---|---|---|
| Pearson r linearity | Section 1 correlation table | Use Spearman ρ as robustness check |
| OLS homoscedasticity | Section 3 regression SE | SE and p-values may be incorrect; consider log-transforming skewed outcomes |
| RF importance interpretability | Section 2 | Correlated features share importance; do not rank correlated predictors independently |
| Cross-validation IID | Section 4 resilience | Spatial autocorrelation between neighbouring counties may slightly optimise CV scores |
| Missing-at-random imputation | Sections 2, 4 | If AHRF data is systematically missing for a county type, imputed medians introduce bias |
| Ecological fallacy | All sections | All associations are county-level. Individual-level inferences are not valid. |

---

## Section 16 — Vaccination Data Integration

### 16.1 Dataset Overview

**Source:** CDC COVID-19 Vaccinations in the United States, County  
**File:** `DATA/COVID-19_Vaccinations_in_the_United_States,County_20260623.csv`  
**Provenance:** Downloaded from CDC's COVID Data Tracker. Last updated 2026-06-23.

| Attribute | Value |
|---|---|
| Date coverage | 2020-12-13 → 2023-05-10 |
| Temporal resolution | Near-daily (~609 unique dates; varies by county) |
| Valid county FIPS | 3,224 |
| Match rate (vs COVID data) | 3,142 / 3,143 counties (99.97 %) |
| Null rate (key % columns) | < 0.3 % for all primary metrics |
| FIPS exclusions | Rows with FIPS = "UNK" (state-unallocated counts) are excluded |
| Data character | Cumulative — all percentage values are monotonically non-decreasing |

### 16.2 Column Mapping (CDC → Internal)

| CDC Column | Internal Name | Description |
|---|---|---|
| `Administered_Dose1_Pop_Pct` | `vax_dose1_pct` | % of total county pop. with ≥ 1 dose |
| `Series_Complete_Pop_Pct` | `vax_complete_pct` | % of total county pop. fully vaccinated (primary series) |
| `Booster_Doses_Vax_Pct` | `vax_booster_pct` | % of vaccinated population who received a booster |
| `Series_Complete_65PlusPop_Pct` | `vax_complete_65plus_pct` | % of county residents 65+ fully vaccinated |
| `Bivalent_Booster_5Plus_Pop_Pct` | `vax_bivalent_pct` | % of residents 5+ who received bivalent booster |

All percentage columns are clipped to [0, 100] during loading.

### 16.3 Loading Architecture

New file: **`vaccination_loader.py`**

| Function | Purpose |
|---|---|
| `load_vaccination_latest(data_dir)` | One row per county: most recent date's snapshot |
| `load_vaccination_timeseries(data_dir)` | Full date × county time-series (~2M rows) |
| `get_vaccination_at_dates(vax_ts, fips, dates)` | Forward-fill lookup for a single county at specific dates |
| `get_county_vax_timeseries(vax_ts, fips)` | Subset time-series for one county |

Both `load_*` functions are wrapped with `@st.cache_data` in `app.py` as `_get_vaccination_latest()` and `_get_vaccination_timeseries()`. The full time-series (2M rows) is loaded once at startup and held in memory for wave overlay and comparison lookups.

### 16.4 Join Methodology

**Latest snapshot → master county table (FIPS join):**
```
vax_latest_df → county_features.create_master_county_table(vax_df=...) → left join on countyFIPS
```
The join is a left-outer merge on zero-padded 5-character countyFIPS, executed in `county_features.create_master_county_table()` after the existing AHRF join. Vaccination columns are appended to the master table and are therefore available to all downstream analyses that receive `master_county_df`.

**Time-series lookups (direct FIPS lookup):**  
For wave overlays and vaccination comparison charts, the full `vax_ts_df` is passed to the relevant render functions and queried via `get_county_vax_timeseries(vax_ts_df, fips)`.

### 16.5 Dashboard Integration Points

| Tab / Section | Integration | Notes |
|---|---|---|
| **Geographic Map** | 4 new metric options: % Fully Vaccinated, % At Least 1 Dose, % Boosted, % 65+ Fully Vaccinated | Static choropleth (latest snapshot); date slider inactive; blue color scale |
| **County Comparison** | 2 new metric options in selectbox | Full rollout timeline chart; national median overlay; FIPS looked up via population_df |
| **County Profile § 7** | New Vaccination Status section with 5 metric cards | Cards: Fully Vaccinated, At Least 1 Dose, Booster Rate, 65+ Fully Vaccinated, Data As Of |
| **County Profile § 7 (expanded)** | Vaccination Rollout Timeline expander | Plotly chart with 3 traces: dose1, complete, booster |
| **County Profile § 8** | National comparison table now includes 4 vax rows | Higher vaccination = favorable (not in lower_is_better set) |
| **County Profile § 9** | Research Snapshot includes vaccination finding | Compares county vs national median vax rate |
| **County Factors** | 4 new factor options + 2 new outcome options | Available as both X (factor) and Y (outcome) in scatter explorer |
| **Statistical Modeling** | 4 vaccination FACTOR_COLS added to modeling.py | Available in Correlation Matrix, Feature Importance, OLS Regression, Resilience Score |
| **Wave Analysis** | Vaccination rate at each wave peak date added to Individual Wave Details table | Forward-fill lookup from vax_ts_df; pre-rollout waves show "N/A (pre-rollout)" |

### 16.6 Temporal Interpretation Cautions

Vaccination data (Dec 2020 – May 2023) and COVID outcome data (Jan 2020 – Jul 2023) overlap for approximately 2.5 years. Several analytical cautions apply:

**Static snapshot vs dynamic:** The master county table uses the *final* vaccination rate (as of May 2023). This is appropriate for asking "did higher eventual vaccination associate with lower cumulative mortality?" but is inappropriate for causal claims about early-pandemic outcomes (when vaccination was zero for everyone).

**Reverse causality risk:** High death rates early in the pandemic may have increased vaccine uptake motivation in some counties. Correlating final vaccination rates with cumulative deaths conflates pre- and post-vaccination periods. Researchers should consider restricting analyses to post-rollout periods.

**Ecological fallacy:** County-level vaccination rates and county-level outcomes do not identify individual-level effects. All observed correlations are county-level associations.

**Booster vs primary series:** `vax_booster_pct` is expressed as a percentage of the vaccinated population (not total population), making it not directly comparable to `vax_complete_pct` and `vax_dose1_pct` which are population-based.

### 16.7 File Changes Summary

| File | Change |
|---|---|
| `vaccination_loader.py` | New file: `load_vaccination_latest`, `load_vaccination_timeseries`, `get_vaccination_at_dates`, `get_county_vax_timeseries`, `VAX_LABELS` |
| `county_features.py` | Added `vax_df` parameter to `create_master_county_table()`; vaccination columns joined on countyFIPS after AHRF join |
| `modeling.py` | Added 4 vaccination entries to `FACTOR_COLS`: `vax_complete_pct`, `vax_dose1_pct`, `vax_booster_pct`, `vax_complete_65plus_pct` |
| `app.py` | `from vaccination_loader import ...`; `_get_vaccination_latest()` and `_get_vaccination_timeseries()` cached wrappers; `_VAX_DATA_DIR` startup constant; `get_master_county_table` updated signature; vaccination metrics in map, comparison, county profile, county factors, wave analysis |
