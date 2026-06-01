# COVID-19 Dashboard - User Guide for New Features

## Quick Reference

### What Changed?

#### 1. **Geographic Map Tab**
- **New**: 3-day and 5-day moving average metrics
- **Fixed**: Duplicate "Daily Cases (7-day MA)" removed
- **Now shows 12 unique metrics:**
  - Cumulative Cases
  - Daily Cases (with 3-day, 5-day, 7-day MA options)
  - Cumulative Deaths
  - Daily Deaths (with 3-day, 5-day, 7-day MA options)
  - Cases per 100k
  - Deaths per 100k

#### 2. **Trend Analysis Tab** (Complete Refactor)
- **Previous**: Only showed daily cases with 7-day MA
- **Now**: Fully customizable analysis with 4 independent selectors

### How to Use New Features

#### Trend Analysis Tab - New Selectors

**Column 1: County Selection**
- Select any county in the US
- Filtered by state if you set a default state in sidebar

**Column 2: Metric Selection**
- **Cases**: Analyze COVID-19 cases
- **Deaths**: Analyze COVID-19 deaths

**Column 3: View Type**
- **Cumulative**: Show total cases/deaths over time (trend line going up)
- **Daily**: Show new cases/deaths per day (day-to-day changes)

**Row 2, Column 1: Normalization**
- **Raw**: Show actual counts (# of cases/deaths)
- **Per 100k**: Adjust for county population (cases per 100,000 people)
- Useful for comparing counties of different sizes

**Row 2, Column 2: Smoothing** (Only shows for Daily view)
- **None**: Show raw daily counts without smoothing
- **3-day MA**: Smooth over 3 days (more responsive)
- **5-day MA**: Smooth over 5 days (balanced smoothing)
- **7-day MA**: Smooth over 7 days (more stable trends)

### Example Use Cases

#### Use Case 1: Compare Growth Rate
1. Metric: **Cases**
2. View Type: **Daily**
3. Normalization: **Raw**
4. Smoothing: **7-day MA**
→ Shows daily new cases smoothed to identify growth/decline trends

#### Use Case 2: Adjust for Population
1. Metric: **Deaths**
2. View Type: **Cumulative**
3. Normalization: **Per 100k**
4. Smoothing: **None**
→ Shows death rate per 100,000 people (fair comparison between counties)

#### Use Case 3: Identify Daily Fluctuations
1. Metric: **Cases**
2. View Type: **Daily**
3. Normalization: **Raw**
4. Smoothing: **None**
→ Shows actual daily case counts with no smoothing (see day-to-day variation)

#### Use Case 4: Long-term Trend with Smoothing
1. Metric: **Deaths**
2. View Type: **Daily**
3. Normalization: **Per 100k**
4. Smoothing: **5-day MA**
→ Shows population-adjusted daily deaths smoothed to see clear trend

### Performance Notes

✅ **Fast**: All moving averages are precomputed at startup
- No waiting when you change smoothing options
- Instant chart updates when switching between options

✅ **Efficient**: Data cached and reused
- Only computed once per session
- Multiple tabs use same underlying data

### Data Behind the Scenes

- **Moving Averages**: Computed from daily values (not cumulative)
- **Daily Calculations**: Computed as difference from previous day
- **Per-Capita**: Calculated as (count / population) × 100,000
- **All validated**: No negative values, proper date ordering, consistent county matching

### Where to Find It

**All new features in the "Trend Analysis" tab**
- Look for the tab selector at the top: Geographic Map | **Trend Analysis** | Demographics | Time Lag Analysis

### Questions?

**Check the data:**
- Hover over any chart to see exact values
- Click "Show Raw Data" to see all calculations
- Check "Data Integrity Check" in Map tab to verify data quality

---

## Technical Details (For Reference)

### Metrics Available

| Type | Metrics |
|------|---------|
| Cases | Cumulative, Daily, Daily (3-day MA), Daily (5-day MA), Daily (7-day MA) |
| Deaths | Cumulative, Daily, Daily (3-day MA), Daily (5-day MA), Daily (7-day MA) |
| Normalized | Cases per 100k, Deaths per 100k |

### Supported Combinations

**32 total combinations possible:**
- 2 metrics × 2 views × 2 normalizations × 4 smoothing options = 32

**Valid combinations:**
- Cumulative view: No smoothing (smoothing unavailable)
- Daily view: Any smoothing option

### Architecture

- **Storage**: Wide-format DataFrames (counties × dates)
- **Caching**: @st.cache_data for all heavy computations
- **Precomputation**: MA windows [3, 5, 7] computed at startup
- **Updates**: All data in memory, zero latency on control changes

### Key Implementation Files

- `app.py` - Streamlit UI and controls (lines 424-580 for Trend Analysis)
- `tools.py` - Data processing functions
  - `precompute_all_moving_averages()` - Compute all MA windows
  - `precompute_all_transforms()` - Main preprocessing
