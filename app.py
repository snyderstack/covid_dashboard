import base64
import os
import random
from contextlib import nullcontext

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as _scipy_stats
from tools import (
    load_data,
    extract_county_state,
    get_population_column,
    prepare_county_timeseries,
    calculate_daily_changes,
    apply_moving_average,
    calculate_per_capita,
    precompute_daily_diffs,
    precompute_all_moving_averages,
    precompute_per_capita,
    get_available_dates,
    prepare_choropleth_for_date,
    filter_choropleth_by_state,
    filter_choropleth_by_county_type,
    get_state_bounds_for_zoom,
    classify_county_type,
    compute_national_timeseries,
    compute_national_daily,
    compute_national_per_capita,
    compute_window_outcomes,
    find_data_corrections,
    monthly_snapshot_long,
    load_county_geojson,
    GEOJSON_CDN_URL,
)
from wave_analysis import (
    calculate_waves_for_county,
    calculate_waves_from_values,
    estimate_optimal_smoothing,
    calculate_waves_for_all_counties,
    match_waves_to_national_windows,
    SENSITIVITY_PRESETS,
    NATIONAL_WAVE_WINDOWS,
)
from lag_analysis import analyze_county_lag, summarize_lag_results
from ahrf_loader import build_ahrf_feature_table, get_variable_catalog
from vaccination_loader import (
    load_vaccination_latest,
    load_vaccination_timeseries,
    get_county_vax_timeseries,
)
from county_features import (
    create_master_county_table,
    compute_bivariate_correlation,
    compute_ols_trend,
    find_similar_counties,
)
from modeling import (
    FACTOR_COLS as _MOD_FACTOR_COLS,
    OUTCOME_COLS as _MOD_OUTCOME_COLS,
    compute_all_correlations,
    run_rf_feature_importance,
    run_ols_regression,
    compute_resilience_scores,
    generate_ols_interpretation,
    compute_vif,
    run_rf_partial_dependence,
    compute_county_clusters,
)
from spatial_analysis import (
    build_adjacency_from_geojson,
    compute_getis_ord_gi_star,
)

# Page config and global CSS

st.set_page_config(
    page_title="COVID-19 County Outcomes Analysis Platform — Gettysburg College",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* =========================================================================
   Design system — Gettysburg Navy #0B2341 / Orange #F26A21
   All shared UI styling lives in this block; render_* helpers emit the
   matching class names. Orange is reserved for emphasis and selection.
   ========================================================================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy:       #0B2341;
    --navy-mid:   #153A66;
    --navy-light: #1E4D87;
    --orange:     #F26A21;
    --orange-mid: #D45A16;
    --orange-lt:  rgba(242,106,33,0.08);
    --bg:         #F3F6FA;
    --bg-card:    #FFFFFF;
    --border:     #DCE3EC;
    --border-lt:  #ECF0F5;
    --text-1:     #0B2341;
    --text-2:     #3A4556;
    --text-3:     #64707F;
    --text-4:     #97A1AF;
    --radius:     12px;
    --radius-sm:  8px;
    --shadow-sm:  0 1px 2px rgba(11,35,65,0.05), 0 2px 10px rgba(11,35,65,0.05);
    --shadow-md:  0 3px 10px rgba(11,35,65,0.08), 0 8px 24px rgba(11,35,65,0.06);
    --shadow-lg:  0 6px 18px rgba(11,35,65,0.10), 0 14px 40px rgba(11,35,65,0.07);
    --ease:       cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global layout */
.stApp {
    background-color: var(--bg);
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
}
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem;
    max-width: 100% !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
}

/* Motion — content eases in when a tab mounts. Kept under 350 ms so it reads
   as feedback rather than decoration; disabled entirely for users who prefer
   reduced motion. */
@keyframes rise-in {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}

/* Tab navigation */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(180deg, #0D2849 0%, #0B2341 100%) !important;
    border-radius: 0 !important;
    gap: 0.15rem !important;
    padding: 0.35rem 1.5rem 0 !important;
    border-bottom: none !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.20) !important;
    position: sticky; top: 0; z-index: 99;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.55) !important;
    font-weight: 600 !important;
    font-size: 0.79rem !important;
    letter-spacing: 0.045em !important;
    text-transform: uppercase !important;
    padding: 0.95rem 1.15rem !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: 0 !important;
    background: transparent !important;
    transition: color 0.2s var(--ease), border-color 0.2s var(--ease),
                background 0.2s var(--ease) !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid var(--orange) !important;
    background: rgba(242,106,33,0.14) !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color: rgba(255,255,255,0.92) !important;
    background: rgba(255,255,255,0.06) !important;
    border-bottom: 3px solid rgba(255,255,255,0.22) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg) !important;
    padding-top: 2.25rem !important;
    padding-bottom: 4rem !important;
    animation: rise-in 0.32s var(--ease);
}

/* Page-level section header (one per tab) */
.section-header {
    margin: 0 0 2rem 0;
    padding: 0 0 1.15rem 0;
    border-bottom: 1px solid var(--border);
}
.section-header h3 {
    margin: 0 0 0.4rem 0;
    font-size: 1.72rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.03em;
    line-height: 1.12;
}
.section-header h3::after {
    content: '';
    display: block;
    width: 46px;
    height: 3px;
    margin-top: 0.55rem;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--orange), var(--orange-mid));
}
.section-header p {
    margin: 0.65rem 0 0 0;
    color: var(--text-2);
    font-size: 0.93rem;
    line-height: 1.7;
    max-width: 84ch;
}

/* Numbered sub-section headers */
.sub-section-header {
    margin: 2.6rem 0 1.1rem 0;
    padding: 0.55rem 1rem 0.55rem 1.05rem;
    border-left: 3px solid var(--orange);
    background: linear-gradient(90deg, var(--orange-lt) 0%, rgba(242,106,33,0.01) 70%);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.sub-section-header h3 {
    margin: 0;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sub-section-header p {
    margin: 0.3rem 0 0 0;
    font-size: 0.8rem;
    color: var(--text-3);
    line-height: 1.55;
}

/* KPI metric cards. The accent bar sweeps across on hover — a small motion
   cue that rewards pointing without moving any content. */
.metric-card, .wave-metric-card {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border-lt);
    border-radius: var(--radius);
    padding: 1.15rem 1.35rem 1.05rem;
    box-shadow: var(--shadow-sm);
    height: 100%;
    overflow: hidden;
    transition: transform 0.22s var(--ease), box-shadow 0.22s var(--ease);
}
.metric-card::before, .wave-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    height: 3px;
    width: 34px;
    border-radius: 0 2px 2px 0;
    background: var(--navy-mid);
    transition: width 0.3s var(--ease);
}
.wave-metric-card::before { background: var(--orange); }
.metric-card:hover, .wave-metric-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}
.metric-card:hover::before, .wave-metric-card:hover::before { width: 100%; }
.metric-label {
    margin: 0;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-3);
}
.metric-value {
    margin: 0.5rem 0 0 0;
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--navy);
    line-height: 1.05;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.wave-metric-card .metric-value { color: var(--orange); }

/* KPI band label — reads as a kicker with a rule that fills remaining width */
.kpi-section-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--text-4);
    margin: 1.5rem 0 0.75rem 0;
}
.kpi-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Welcome strip + explore chips (County Overview landing) */
.welcome-banner { margin: 0 0 1.4rem 0; }
.welcome-banner .welcome-kicker {
    margin: 0 0 0.3rem 0;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--orange-mid);
}
.welcome-banner h3 {
    margin: 0 0 0.4rem 0;
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.03em;
    line-height: 1.15;
}
.welcome-banner p {
    margin: 0;
    font-size: 0.9rem;
    color: var(--text-2);
    line-height: 1.65;
    max-width: 92ch;
}
.explore-chips {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin: 1.1rem 0 2rem 0;
}
@media (max-width: 1100px) { .explore-chips { grid-template-columns: repeat(2, 1fr); } }
.explore-chip {
    background: var(--bg-card);
    border: 1px solid var(--border-lt);
    border-radius: var(--radius-sm);
    padding: 0.8rem 1rem 0.75rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease),
                border-color 0.2s var(--ease);
}
.explore-chip:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: rgba(242,106,33,0.4);
}
.explore-chip .chip-title {
    display: block;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 0.15rem;
}
.explore-chip .chip-desc {
    font-size: 0.74rem;
    color: var(--text-3);
    line-height: 1.45;
}

/* County hero banner (Overview tab) */
.county-hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 60%, #1a5080 100%);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}
.county-hero::before {
    content: '';
    position: absolute;
    right: -40px; top: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(242,106,33,0.12);
    pointer-events: none;
}
.county-hero::after {
    content: '';
    position: absolute;
    right: 60px; bottom: -60px;
    width: 150px; height: 150px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}
.county-hero-content { position: relative; z-index: 1; }
.county-hero h2 {
    margin: 0 0 0.3rem 0;
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.county-hero-meta {
    margin: 0;
    font-size: 0.83rem;
    color: rgba(255,255,255,0.65);
    letter-spacing: 0.01em;
    line-height: 1.5;
}
.county-hero-badge {
    display: inline-block;
    background: rgba(242,106,33,0.85);
    color: white;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 0.55rem;
}

/* Map control panel — st.container(border=True) renders as stBorderContainer */
[data-testid="stBorderContainer"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 1.3rem 1.2rem 1.15rem !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--border-lt) !important;
}
.map-ctrl-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--text-4);
    margin: 0 0 1rem 0;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--border-lt);
}
.map-ctrl-group {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-4);
    margin: 1.25rem 0 0.45rem 0;
    padding-top: 1rem;
    border-top: 1px solid var(--border-lt);
}

/* Charts are the centerpiece: each Plotly figure sits in its own card */
[data-testid="stPlotlyChart"] {
    background: var(--bg-card);
    border: 1px solid var(--border-lt);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    padding: 0.5rem 0.5rem 0.15rem;
    transition: box-shadow 0.25s var(--ease);
}
[data-testid="stPlotlyChart"]:hover { box-shadow: var(--shadow-md); }

/* st.metric — match the card language used elsewhere */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-lt);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem 0.8rem;
    box-shadow: var(--shadow-sm);
    height: 100%;
    transition: transform 0.22s var(--ease), box-shadow 0.22s var(--ease);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    color: var(--navy) !important;
    font-variant-numeric: tabular-nums;
}

/* st.subheader (used where a heading needs a help tooltip) — match the
   .stMarkdown h4 style so both heading forms look identical inside tabs */
[data-testid="stHeading"] h3 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: var(--navy-mid) !important;
    margin: 0.4rem 0 0.35rem 0 !important;
    padding-bottom: 0.45rem !important;
    letter-spacing: -0.01em !important;
}

/* Markdown sub-headers inside tabs */
.stMarkdown h4 {
    font-size: 1rem;
    font-weight: 700;
    color: var(--navy-mid);
    margin: 1.9rem 0 0.75rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--border-lt);
    letter-spacing: -0.01em;
}
.stMarkdown h5 {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-2);
    margin: 1.2rem 0 0.4rem 0;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* Widget labels and inputs */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label div p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-2) !important;
    letter-spacing: 0.01em !important;
}
[data-baseweb="select"] > div:first-child {
    border-color: #D1DCE8 !important;
    border-radius: var(--radius-sm) !important;
    background-color: #FAFCFF !important;
    font-size: 0.875rem !important;
    transition: border-color 0.15s var(--ease), box-shadow 0.15s var(--ease) !important;
}
[data-baseweb="select"] > div:first-child:hover { border-color: var(--navy-light) !important; }
[data-baseweb="select"] > div:focus-within {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 3px rgba(242,106,33,0.14) !important;
}

/* Buttons */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: box-shadow 0.2s var(--ease), transform 0.2s var(--ease),
                border-color 0.2s var(--ease), background 0.2s var(--ease) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #F26A21 0%, #D45A16 100%) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(242,106,33,0.32) !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 5px 18px rgba(242,106,33,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
    border-color: #C8D4E4 !important;
    color: var(--navy-mid) !important;
    background: white !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--navy-mid) !important;
    background: rgba(21,58,102,0.04) !important;
}
[data-testid="stDownloadButton"] > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    border-color: #C8D4E4 !important;
    color: var(--navy-mid) !important;
    transition: border-color 0.2s var(--ease), background 0.2s var(--ease) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--navy-mid) !important;
    background: rgba(21,58,102,0.04) !important;
}

/* Slider */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--orange) !important;
    border: 2px solid white !important;
    box-shadow: 0 1px 4px rgba(242,106,33,0.40) !important;
    transition: box-shadow 0.15s var(--ease) !important;
}
[data-testid="stSlider"] [role="slider"]:hover {
    box-shadow: 0 0 0 6px rgba(242,106,33,0.15) !important;
}

/* Captions, alerts, expanders, tables */
[data-testid="stCaptionContainer"] p, .stCaption p {
    color: var(--text-3) !important;
    font-size: 0.8rem !important;
    line-height: 1.55 !important;
}
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 0.875rem !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border-lt) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--navy-mid) !important;
    transition: background 0.15s var(--ease), color 0.15s var(--ease);
}
[data-testid="stExpander"] summary:hover {
    background: rgba(21,58,102,0.05);
    color: var(--orange-mid) !important;
}
[data-testid="stDataFrame"] {
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2341 0%, #091D35 100%) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: rgba(255,255,255,0.82) !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stSidebar"] .stCaption p {
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.75rem !important;
}

/* Misc */
hr {
    border-color: var(--border-lt) !important;
    border-top-width: 1px !important;
    margin: 2rem 0 !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 1.1rem !important;
    align-items: stretch !important;
}
[data-baseweb="tag"] {
    background: rgba(242,106,33,0.14) !important;
    color: var(--orange-mid) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# Logo assets — loaded once at module level for use in header/footer.
def _load_asset_b64(stem):
    """
    Return a CSS-ready data URI for a logo asset, or None if not found.
    Searches assets/ for <stem>.png, .jpg, .jpeg in that order.
    Uses CSS background-image embedding which is not subject to Streamlit's
    <img> src sanitisation.
    """
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    for ext, mime in (("png", "image/png"), ("jpg", "image/jpeg"), ("jpeg", "image/jpeg")):
        path = os.path.join(assets_dir, f"{stem}.{ext}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"
    return None

_G_LOGO_B64   = _load_asset_b64("gettysburg_g")
_SEAL_LOGO_B64 = _load_asset_b64("gettysburg_seal")

def render_header(latest_date=None) -> None:
    """Render professional dashboard header with Gettysburg College branding."""
    date_text = latest_date if latest_date else "—"

    # Header logo: real image via CSS background-image when the file exists,
    # plain styled "G" badge as fallback. CSS background-image with a data URI
    # passes through Streamlit's HTML processor safely.
    if _G_LOGO_B64:
        logo_div = (
            '<div style="flex-shrink:0;width:88px;height:88px;'
            f'background-image:url(\'{_G_LOGO_B64}\');'
            'background-size:contain;background-repeat:no-repeat;'
            'background-position:center;margin-left:1.75rem;opacity:0.95;"></div>'
        )
    else:
        logo_div = (
            '<div style="flex-shrink:0;width:82px;height:82px;'
            'background:#F26A21;border-radius:14px;'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:46px;font-weight:900;color:#0B2341;'
            'font-family:Georgia,serif;margin-left:1.75rem;">G</div>'
        )

    st.markdown(
        '<div style="background:linear-gradient(135deg,#0A1F3C 0%,#0B2341 45%,#153A66 100%);'
        'padding:1.35rem 2rem 0 2rem;margin:0;position:relative;overflow:hidden;">'
        # Subtle geometric accent circles
        '<div style="position:absolute;right:-60px;top:-60px;width:280px;height:280px;'
        'border-radius:50%;background:rgba(242,106,33,0.06);pointer-events:none;"></div>'
        '<div style="position:absolute;right:120px;bottom:-80px;width:180px;height:180px;'
        'border-radius:50%;background:rgba(255,255,255,0.03);pointer-events:none;"></div>'
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'padding-bottom:1rem;position:relative;z-index:1;">'
        '<div>'
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.28rem;">'
        '<div style="width:3px;height:1.7rem;background:#F26A21;border-radius:2px;'
        'flex-shrink:0;"></div>'
        '<div style="font-size:1.68rem;font-weight:800;color:#ffffff;'
        'letter-spacing:-0.03em;line-height:1.12;font-family:sans-serif;">'
        'COVID-19 County Outcomes Analysis Platform'
        '</div>'
        '</div>'
        '<div style="font-size:0.86rem;color:rgba(255,255,255,0.72);'
        'margin:0 0 0.4rem 0.5rem;font-family:sans-serif;line-height:1.5;'
        'max-width:68ch;">'
        'Explore how the pandemic unfolded — and why outcomes differed — '
        'across 3,000+ U.S. counties'
        '</div>'
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.42);'
        'font-family:sans-serif;margin-left:0.5rem;">'
        f'Data through {date_text}'
        '&ensp;<span style="color:rgba(255,255,255,0.22);">|</span>&ensp;'
        'USAFacts &amp; HRSA Area Health Resources Files'
        '&ensp;<span style="color:rgba(255,255,255,0.22);">|</span>&ensp;'
        'Gettysburg College &bull; 2026'
        '</div>'
        '</div>'
        + logo_div +
        '</div>'
        '<div style="height:3px;background:linear-gradient(90deg,#F26A21 0%,#D45A16 100%);'
        'margin:0 -2rem;"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

def render_section_header(title, description="") -> None:
    """Render the page-level section header at the top of each tab."""
    desc_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f'<div class="section-header"><h3>{title}</h3>{desc_html}</div>',
        unsafe_allow_html=True,
    )

def render_metric_card(label, value, suffix="") -> None:
    """Render a single KPI metric card."""
    try:
        is_na = pd.isna(value)
    except (TypeError, ValueError):
        is_na = value is None
    if is_na:
        formatted_value = "N/A"
    elif isinstance(value, int):
        formatted_value = f"{value:,}"
    elif isinstance(value, float):
        formatted_value = f"{value:,.1f}"
    else:
        formatted_value = str(value)
    val_display = f"{formatted_value} {suffix}".strip()
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{val_display}</p>
    </div>
    """, unsafe_allow_html=True)

def render_wave_metric_card(label, value, suffix="") -> None:
    """Render a wave-themed KPI metric card."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        formatted_value = "N/A"
    elif isinstance(value, int):
        formatted_value = f"{value:,}"
    elif isinstance(value, float):
        formatted_value = f"{value:,.1f}"
    else:
        formatted_value = str(value)
    val_display = f"{formatted_value} {suffix}".strip()
    st.markdown(f"""
    <div class="wave-metric-card">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{val_display}</p>
    </div>
    """, unsafe_allow_html=True)

# Plain-language definitions surfaced as per-tab "Key terms" popovers.
# Written for undergraduates meeting epidemiology for the first time.
GLOSSARY = {
    "per_100k":       ("Per 100,000 residents", "A rate that removes population size, so a small rural county and a huge metro county can be compared fairly."),
    "cumulative":     ("Cumulative vs. daily", "Cumulative = running total since January 2020. Daily = new events each day, computed as the day-over-day change in the cumulative count."),
    "cfr":            ("Case fatality rate (CFR)", "Deaths divided by confirmed cases, as a percent. Sensitive to how much testing happened — more testing finds milder cases and lowers CFR."),
    "moving_average": ("Moving average (MA)", "The mean of the last N days. Smooths out weekend reporting dips and batch uploads so the underlying trend is visible."),
    "wave":           ("Wave", "A sustained period of elevated transmission — detected here as a region where smoothed daily counts stay well above the county's local baseline."),
    "prominence":     ("Prominence", "How far a peak rises above its surroundings. Low-prominence bumps are usually reporting noise, not epidemiology."),
    "burden":         ("Wave burden", "Total cases (or deaths) accumulated during one wave — the area under the curve, not just its height."),
    "significance":   ("Significance score", "A 0–100 ranking of each wave combining prominence (30%), burden (30%), duration (20%), and burst intensity (20%)."),
    "lag":            ("Case-to-death lag", "Days between a peak in new cases and the following peak in deaths. Reflects disease progression time plus reporting delays."),
    "severity_ratio": ("Severity ratio", "Death-peak height divided by the case-peak height that preceded it. Lower means fewer deaths per unit of case surge."),
    "pearson":        ("Pearson r", "Linear correlation from −1 to +1. Sign gives direction, magnitude gives strength. Says nothing about causation."),
    "spearman":       ("Spearman ρ", "Rank-based correlation — robust when the relationship is curved or has outliers. If Pearson and Spearman disagree, look at the scatter."),
    "p_value":        ("p-value", "Probability of seeing an association this strong if none truly existed. Below 0.05 is conventionally 'statistically significant' — with 3,000 counties, tiny effects can still clear that bar."),
    "r_squared":      ("R²", "Share of the outcome's variation the model explains, from 0 to 1. An R² of 0.3 means 70% of the variation comes from things not in the model."),
    "residual":       ("Residual / resilience", "Actual minus predicted outcome. A county doing better than its characteristics predict has a favorable residual — we call that resilience."),
    "vif":            ("VIF", "Variance Inflation Factor — flags predictors that duplicate each other. Above 5, a coefficient's sign and size become unreliable."),
    "hc3":            ("Robust (HC3) errors", "Standard errors that stay valid when outcome noise varies across counties. Trust these over classical errors for skewed data."),
    "ecological":     ("Ecological fallacy", "County-level patterns need not hold for individuals. 'Higher-income counties had fewer deaths' does not mean richer people were safer."),
    "rucc":           ("RUCC code", "USDA Rural-Urban Continuum Code, 1 (large metro) to 9 (most rural). Codes 1–3 are Metro; 4–9 Nonmetro."),
    "hpsa":           ("HPSA", "Health Professional Shortage Area — a federal designation that a county lacks adequate primary care capacity."),
    "hotspot":        ("Hotspot (Gi*)", "A county whose whole neighbourhood shows unusually high values — statistically significant spatial clustering, not just one high county."),
}

def render_learning_aids(terms=(), questions=()) -> None:
    """Render the per-tab 'Key terms' popover and 'Questions to investigate' expander."""
    if terms:
        with st.popover("Key terms on this page"):
            for key in terms:
                if key in GLOSSARY:
                    label, definition = GLOSSARY[key]
                    st.markdown(f"**{label}** — {definition}")
    if questions:
        with st.expander("Questions to investigate", expanded=False):
            for q in questions:
                st.markdown(f"- {q}")

def apply_chart_styling(fig):
    """Apply consistent professional styling to Plotly charts."""
    fig.update_layout(
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=11, color="#1A1A2E"),
        plot_bgcolor="rgba(247,249,252,0.8)",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.3)",
                     zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.3)",
                     zeroline=False)
    return fig

def render_footer() -> None:
    """Render professional dashboard footer with Gettysburg College branding."""
    st.markdown(
        '<div style="border-top:1px solid #E5E7EB;margin-top:3rem;"></div>',
        unsafe_allow_html=True,
    )

    foot_left, foot_right = st.columns([6, 1])

    with foot_left:
        st.markdown(
            '<div style="padding:1rem 0 0.75rem 0;font-family:sans-serif;">'
            '<p style="margin:0 0 0.25rem 0;font-size:0.78rem;color:#6B7280;">'
            '<strong style="color:#4B5563;">Data</strong>&ensp;'
            'USAFacts County-Level COVID-19 Data &bull; '
            'HRSA Area Health Resources Files (AHRF) 2020&#x2013;2023'
            '</p>'
            '<p style="margin:0 0 0.25rem 0;font-size:0.78rem;color:#9CA3AF;">'
            'County-level analysis of confirmed COVID-19 cases and deaths. '
            'Rural&#x2013;urban classification uses USDA Rural-Urban Continuum Codes (2013).'
            '</p>'
            '<p style="margin:0 0 0.1rem 0;font-size:0.73rem;color:#9CA3AF;">'
            'Developed by <strong style="color:#6B7280;">Dr. Ryan Johnson</strong> &amp; <strong style="color:#6B7280;">Caden Snyder</strong> &bull; '
            'Gettysburg College &bull; 2026'
            '</p>'
            '<p style="margin:0;font-size:0.68rem;color:#C0C8D2;">'
            'Created for academic research purposes in partial fulfillment of requirements '
            'at Gettysburg College. Not intended for clinical or policy use.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with foot_right:
        # Seal logo via CSS background-image (same technique as header).
        if _SEAL_LOGO_B64:
            st.markdown(
                '<div style="width:90px;height:90px;margin-top:0.4rem;margin-left:auto;'
                f'background-image:url(\'{_SEAL_LOGO_B64}\');'
                'background-size:contain;background-repeat:no-repeat;'
                'background-position:center;opacity:0.88;"></div>',
                unsafe_allow_html=True,
            )
        else:
            seal_svg_path = os.path.join(os.path.dirname(__file__), "assets", "gettysburg_seal.svg")
            if os.path.exists(seal_svg_path):
                try:
                    st.image(seal_svg_path, width=90)
                except Exception:
                    pass

# Cached data loaders and precomputed transforms

@st.cache_data
def get_data():
    """Load raw USAFacts CSVs and normalize metadata."""
    return load_data()

@st.cache_data
def precompute_all_transforms(cases, deaths, population):
    """Precompute all metric transforms for choropleth and analysis."""
    with st.spinner("Computing metrics..."):
        daily_cases, daily_deaths = precompute_daily_diffs(cases, deaths)
        ma_results = precompute_all_moving_averages(daily_cases, daily_deaths, windows=[3, 5, 7])
        pc_cases, pc_deaths = precompute_per_capita(cases, deaths, population)
        dates = get_available_dates(cases)

    result = {
        "daily_cases":  daily_cases,
        "daily_deaths": daily_deaths,
        "pc_cases":     pc_cases,
        "pc_deaths":    pc_deaths,
        "dates":        dates,
    }
    result.update(ma_results)
    return result

@st.cache_data
def compute_national_aggregates(cases_df, deaths_df, population_df, daily_cases_df, daily_deaths_df):
    """
    Precompute all national-level time series.

    Uses raw cases/deaths DataFrames directly so compute_national_per_capita
    avoids the floating-point round-trip that the previous implementation used
    when it reversed precomputed per-capita values. Statewide unallocated rows
    (FIPS == '00000') are excluded inside each helper.
    """
    return {
        "cases_ts":     compute_national_timeseries(cases_df, deaths_df, "Cases"),
        "deaths_ts":    compute_national_timeseries(cases_df, deaths_df, "Deaths"),
        "daily_cases":  compute_national_daily(daily_cases_df),
        "daily_deaths": compute_national_daily(daily_deaths_df),
        "pc_cases":     compute_national_per_capita(cases_df, population_df),
        "pc_deaths":    compute_national_per_capita(deaths_df, population_df),
    }

@st.cache_data
def get_choropleth_data(metric_df, date_str, cases_df, deaths_df, population_df):
    """
    Cached wrapper for prepare_choropleth_for_date.

    Streamlit hashes DataFrames by content, so each unique (metric, date)
    combination is a separate cache entry. Eliminates repeated merge operations
    on every map slider movement.
    """
    return prepare_choropleth_for_date(metric_df, date_str, cases_df, deaths_df, population_df)

@st.cache_resource
def get_county_geojson():
    """
    Bundled county GeoJSON as a dict (auto-downloaded once if absent), or
    None when unavailable — callers then fall back to the CDN URL.
    """
    return load_county_geojson()

@st.cache_resource
def get_county_adjacency():
    """County adjacency derived from the GeoJSON geometry (None if no file)."""
    geo = get_county_geojson()
    if geo is None:
        return None
    return build_adjacency_from_geojson(geo)

@st.cache_data
def get_window_outcomes(_cases_df, _deaths_df, _population_df, start_date, end_date):
    """
    Window-restricted per-100k outcomes, cached by (start, end).

    The dataframes never change within a session, so they are excluded from
    the cache key (underscore prefix); the date pair fully identifies a result.
    """
    return compute_window_outcomes(_cases_df, _deaths_df, _population_df,
                                   start_date, end_date)

@st.cache_data
def get_monthly_animation_frames(_metric_df, metric_key):
    """
    Cached monthly long-format snapshots for map animation (~40 frames).

    Cached by metric_key; the underlying wide table is stable per session so
    it is excluded from the hash.
    """
    return monthly_snapshot_long(_metric_df)

@st.cache_data
def get_hotspot_analysis(_choro_data, metric_key, date_str):
    """Getis-Ord Gi* over the national choropleth slice, cached per metric/date."""
    adjacency = get_county_adjacency()
    if adjacency is None:
        return None
    return compute_getis_ord_gi_star(
        _choro_data[["countyFIPS", "Location", "State", "value"]],
        adjacency,
    )

@st.cache_data
def get_ahrf_features(covid_fips_frozenset):
    """
    Load and build the AHRF county feature table.

    Sources (in priority order):
      1. ahrf2023.csv   — primary; 2021-era data, all counties, full coverage
      2. AHRF2020.asc   — supplementary; 2018-2020 pandemic-era variables
      3. AHRF2021.sas7bdat — supplementary; 2019-2021 validation data

    Note: ahrf2021.asc and ahrf2022.asc are excluded because no SAS layout
    files were uploaded for those years; the 2023 CSV already contains 2021/2022
    data for all key variables.

    Returns (ahrf_df, diagnostics) where ahrf_df has one row per county.
    """
    covid_fips = set(covid_fips_frozenset)
    with st.spinner("Loading AHRF county data (one-time startup)..."):
        ahrf_df, diag = build_ahrf_feature_table(
            covid_fips=covid_fips, verbose=False
        )
    return ahrf_df, diag

@st.cache_data
def get_master_county_table(_cases_df, _deaths_df, _population_df, _ahrf_df, _vax_df=None):
    """
    Build the master county table joining COVID outcomes with AHRF and vaccination features.

    Columns added beyond COVID metrics:
      Vaccination: vax_dose1_pct, vax_complete_pct, vax_booster_pct,
                   vax_complete_65plus_pct, vax_bivalent_pct, vax_last_date
                   (Source: CDC COVID-19 Vaccinations in the United States, County)
      Healthcare:  pcp_per_100k, total_md_per_100k, hospital_beds_per_100k,
                   icu_beds_per_100k, snf_beds_per_100k, hpsa_primary_care,
                   critical_access_hospitals
      Economic:    median_family_income, per_capita_income, unemployment_rate,
                   child_poverty_pct, persistent_poverty_flag, high_poverty_flag
      Education:   pct_no_hs_diploma, pct_hs_diploma_or_higher, pct_college_4yr
      Demographic: population_2020, pop_65plus, pct_pop_65plus, pop_density_per_sqmi,
                   land_area_sq_mi, median_age, pct_urban_pop
      Rural-Urban: rucc_code, rucc_group, is_metro, urban_influence_code
      Geography:   census_region_code, census_region_name, census_division_name

    Also adds case_fatality_rate (deaths/cases × 100, in percent).
    Join key: countyFIPS (5-character, zero-padded).
    """
    master, diag = create_master_county_table(
        _cases_df, _deaths_df, _population_df,
        ahrf_df=_ahrf_df,
        vax_df=_vax_df,
        verbose=False,
    )
    # Derived COVID outcome: case fatality rate
    if "total_cases" in master.columns and "total_deaths" in master.columns:
        master["case_fatality_rate"] = np.where(
            master["total_cases"] > 0,
            (master["total_deaths"] / master["total_cases"]) * 100,
            np.nan,
        )
    return master, diag

@st.cache_data
def _get_vaccination_latest(_data_dir: str) -> pd.DataFrame:
    """
    Cached wrapper for load_vaccination_latest().

    Loads the most-recent vaccination snapshot per county once at startup.
    Returns empty DataFrame if the vaccination file is not found.
    """
    with st.spinner("Loading CDC vaccination data (one-time startup)..."):
        return load_vaccination_latest(_data_dir)

@st.cache_data
def _get_vaccination_timeseries(_data_dir: str) -> pd.DataFrame:
    """
    Cached wrapper for load_vaccination_timeseries().

    Loads the full date × county vaccination time-series once at startup.
    The result is used for wave overlays, county comparison, and rollout charts.
    Returns empty DataFrame if the vaccination file is not found.
    """
    return load_vaccination_timeseries(_data_dir)

@st.cache_data
def get_county_wave_metrics(_cases_df, _deaths_df, _daily_cases_df, _daily_deaths_df,
                             _population_df, prominence=50, sensitivity="standard"):
    """
    Compute wave metrics for all counties and return a normalized DataFrame.

    Uses adaptive multi-criteria wave detection when sensitivity is supplied
    ("conservative" | "standard" | "sensitive").  The sensitivity preset
    automatically scales prominence, width, and valley-depth thresholds so
    major outbreaks are detected consistently across rural and urban counties.

    Returns a DataFrame with countyFIPS plus per-100k wave metrics.
    """
    pop_col_candidates = [c for c in _population_df.columns
                          if c not in {"countyFIPS", "County Name", "State",
                                       "StateFIPS", "Location"}]
    pop_col = pop_col_candidates[0] if pop_col_candidates else None

    wave_df = calculate_waves_for_all_counties(
        _cases_df, _deaths_df, _daily_cases_df, _daily_deaths_df,
        ma_window=7, prominence=prominence, min_merge_days=30,
        sensitivity=sensitivity,
    )

    if wave_df.empty or pop_col is None:
        return wave_df

    # Merge population for per-100k normalisation
    pop_lkp = _population_df[_population_df["countyFIPS"] != "00000"][
        ["countyFIPS", "State", pop_col]
    ].rename(columns={pop_col: "_pop"})

    wave_df = wave_df.merge(pop_lkp, on=["countyFIPS", "State"], how="left")
    pop = pd.to_numeric(wave_df["_pop"], errors="coerce")
    pop = pop.where(pop > 0)

    wave_df["peak_wave_cases_per_100k"] = np.where(
        pop.notna(), (wave_df["case_largest_wave"] / pop) * 100_000, np.nan
    )
    wave_df["peak_wave_deaths_per_100k"] = np.where(
        pop.notna(), (wave_df["death_largest_wave"] / pop) * 100_000, np.nan
    )
    wave_df["case_wave_count"]  = wave_df["case_waves"]
    wave_df["death_wave_count"] = wave_df["death_waves"]

    return wave_df.drop(columns=["_pop"], errors="ignore")

@st.cache_data
def get_county_classifications(_population_df, _ahrf_df=None):
    """
    Cached wrapper for classify_county_type().

    When AHRF data is available, uses USDA RUCC codes:
        Metro    — RUCC 1-3
        Nonmetro — RUCC 4-9

    Falls back to population-threshold Urban/Rural when AHRF is unavailable.
    Returns a DataFrame with columns (countyFIPS, State, County_Type).
    """
    return classify_county_type(_population_df, rucc_df=_ahrf_df)

# Startup — load and precompute all data at module scope. On the very first
# run of a session the sequence takes 30-60 s, so it is narrated inside an
# st.status panel; on later reruns everything returns from cache instantly
# and no panel is shown.

_first_boot = "boot_complete" not in st.session_state
_boot = (
    st.status("Preparing the dashboard — first launch loads all datasets…",
              expanded=True)
    if _first_boot else nullcontext()
)

with _boot:
    if _first_boot:
        st.write("Loading USAFacts case, death, and population data…")
    cases_df, deaths_df, population_df = get_data()

    # normalize_dataset_metadata adds Location, but guard for edge cases where
    # the column is absent (e.g. a non-standard CSV header).
    for _df in (cases_df, deaths_df, population_df):
        if "Location" not in _df.columns:
            _df["Location"] = (
                _df["County Name"].astype(str).str.strip()
                + ", "
                + _df["State"].astype(str).str.strip()
            )

    if _first_boot:
        st.write("Computing daily, moving-average, and per-capita transforms…")
    transforms = precompute_all_transforms(cases_df, deaths_df, population_df)
    dates = transforms["dates"]

    locations     = sorted(cases_df["Location"].unique())
    unique_states = sorted(cases_df["State"].unique())

    # County boundaries: bundled GeoJSON dict when available (offline-capable,
    # also powers spatial analysis); CDN URL string as the Plotly fallback.
    county_geojson = get_county_geojson()
    GEO_SOURCE = county_geojson if county_geojson is not None else GEOJSON_CDN_URL

    national = compute_national_aggregates(
        cases_df, deaths_df, population_df,
        transforms["daily_cases"], transforms["daily_deaths"],
    )

    # AHRF feature table — loaded once at startup. Sources: ahrf2023.csv
    # (primary) + AHRF2020.asc + AHRF2021.sas7bdat (supplementary).
    if _first_boot:
        st.write("Loading AHRF healthcare and socioeconomic data…")
    _covid_fips = frozenset(
        cases_df[cases_df["countyFIPS"] != "00000"]["countyFIPS"].unique()
    )
    ahrf_df, _ahrf_diag = get_ahrf_features(_covid_fips)

    # Vaccination data — CDC county-level dataset, loaded once at startup.
    if _first_boot:
        st.write("Loading CDC vaccination data (largest file — most of the wait)…")
    _VAX_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    vax_latest_df = _get_vaccination_latest(_VAX_DATA_DIR)
    vax_ts_df     = _get_vaccination_timeseries(_VAX_DATA_DIR)

    # Master county table: COVID outcomes joined with AHRF + vaccination.
    if _first_boot:
        st.write("Assembling the master county table…")
    master_county_df, _master_diag = get_master_county_table(
        cases_df, deaths_df, population_df, ahrf_df, vax_latest_df
    )

    # Metro/Nonmetro via USDA RUCC codes; falls back to Urban/Rural by
    # population threshold when AHRF is unavailable.
    county_type_df = get_county_classifications(population_df, ahrf_df)

if _first_boot:
    _boot.update(label="Dashboard ready", state="complete", expanded=False)
    st.session_state["boot_complete"] = True

# Shareable views: ?county=<Location> pre-selects the County Overview county.
# Only seeds session state once, so it never fights the user's own selection.
_shared_county = st.query_params.get("county")
if _shared_county in set(locations) and "overview_county" not in st.session_state:
    st.session_state["overview_county"] = _shared_county

# Map click-to-profile: the Geographic Map tab renders after the Overview
# tab, so a clicked county can't set the overview widget's state in the same
# run. The click handler stashes it here and triggers a rerun; this block
# consumes it before any widget instantiates.
if "_pending_overview_county" in st.session_state:
    _pending_county = st.session_state.pop("_pending_overview_county")
    if _pending_county in set(locations):
        st.session_state["overview_county"] = _pending_county
        st.toast(
            f"**{_pending_county}** loaded — open the County Overview tab "
            "for its full profile."
        )

# AHRF integrity check
# Detect whether the AHRF join produced any usable data. If all key AHRF
# columns are NaN for every county, the source file failed to load and
# downstream tabs (County Factors, Resilience Explorer, Modeling) will be empty.
_AHRF_SENTINEL_COLS = ["median_family_income", "pcp_per_100k", "hospital_beds_per_100k"]
_ahrf_loaded = (
    master_county_df is not None
    and not master_county_df.empty
    and any(
        c in master_county_df.columns and master_county_df[c].notna().any()
        for c in _AHRF_SENTINEL_COLS
    )
)

COUNTY_COLOR      = "#F26A21"  # Gettysburg Orange — County A
NATIONAL_COLOR    = "#153A66"  # Gettysburg Navy   — County B / trend lines
NATION_LINE_COLOR = "#059669"  # Emerald green     — national aggregate

# Header

latest_date = dates[-1] if dates else None
render_header(latest_date)

# Top-level KPI cards

# Read the county-type filter from session state so KPIs reflect any active
# map filter. Defaults to "All Counties" before the map tab has been visited.
_kpi_county_type = st.session_state.get("county_type_filter", "All Counties")

if _kpi_county_type == "All Counties":
    _kpi_cases_df    = cases_df
    _kpi_deaths_df   = deaths_df
    _kpi_label       = "Counties Tracked"
    _kpi_count       = int((cases_df["countyFIPS"] != "00000").sum())
else:
    # Support both RUCC-based (Metro/Nonmetro) and legacy (Urban/Rural) labels
    _type_label_map = {
        "Metro Counties":    "Metro",
        "Nonmetro Counties": "Nonmetro",
        "Urban Counties":    "Urban",
        "Rural Counties":    "Rural",
    }
    _target_type = _type_label_map.get(_kpi_county_type, "Metro")
    _type_pairs  = county_type_df[county_type_df["County_Type"] == _target_type][["countyFIPS", "State"]]
    _kpi_cases_df  = cases_df.merge(_type_pairs,  on=["countyFIPS", "State"], how="inner")
    _kpi_deaths_df = deaths_df.merge(_type_pairs, on=["countyFIPS", "State"], how="inner")
    _kpi_label     = f"{_target_type} Counties Tracked"
    _kpi_count     = len(_kpi_cases_df)

st.markdown(
    '<p class="kpi-section-label" style="margin-top:1.75rem;">Cumulative National Totals</p>',
    unsafe_allow_html=True,
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    _kpi_county_cases  = _kpi_cases_df[_kpi_cases_df["countyFIPS"] != "00000"][dates[-1]].sum() if dates else 0
    render_metric_card("Total US Cases",  int(_kpi_county_cases))
with kpi_col2:
    _kpi_county_deaths = _kpi_deaths_df[_kpi_deaths_df["countyFIPS"] != "00000"][dates[-1]].sum() if dates else 0
    render_metric_card("Total US Deaths", int(_kpi_county_deaths))
with kpi_col3:
    _kpi_cfr = (
        round(_kpi_county_deaths / _kpi_county_cases * 100, 2)
        if _kpi_county_cases > 0 else None
    )
    render_metric_card("Case Fatality Rate", _kpi_cfr if _kpi_cfr else "N/A", suffix="%" if _kpi_cfr else "")
with kpi_col4:
    render_metric_card(_kpi_label, _kpi_count)

# Sidebar controls

with st.sidebar:
    st.markdown(
        "<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;"
        "text-transform:uppercase;color:rgba(255,255,255,0.45);margin:0.5rem 0 1rem 0;'>"
        "Dashboard Controls</p>",
        unsafe_allow_html=True,
    )
    selected_date = st.select_slider(
        "Analysis Date",
        options=dates,
        value=dates[-1],
        help="Initial date for the Map tab; the map's own slider takes over after that",
    )
    st.markdown(
        "<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;"
        "text-transform:uppercase;color:rgba(255,255,255,0.45);margin:1rem 0 0.5rem 0;'>"
        "Default Filters</p>",
        unsafe_allow_html=True,
    )
    analysis_state = st.selectbox(
        "State",
        ["All States"] + unique_states,
        help="Pre-select state for Trend Analysis tab (can override within each tab)",
    )
    st.markdown(
        "<p style='font-size:0.72rem;color:rgba(255,255,255,0.38);margin-top:1.5rem;'>"
        "Filters can be adjusted within each tab independently.</p>",
        unsafe_allow_html=True,
    )
    # Data coverage notice — reminds users this is a historical archive
    st.markdown(
        "<div style='margin-top:2rem;padding:0.75rem;border-radius:6px;"
        "background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);'>"
        "<p style='font-size:0.68rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:rgba(255,255,255,0.45);margin:0 0 0.35rem 0;'>"
        "Data Coverage</p>"
        "<p style='font-size:0.75rem;color:rgba(255,255,255,0.6);margin:0;line-height:1.5;'>"
        "COVID cases &amp; deaths: Jan 2020 – Jul 2023<br>"
        "CDC vaccination: Dec 2020 – May 2023<br>"
        "AHRF socioeconomic: 2022–2023<br>"
        "<span style='color:rgba(255,255,255,0.35);font-size:0.68rem;'>"
        "Historical archive — not updated in real time.</span></p>"
        "</div>",
        unsafe_allow_html=True,
    )

def render_map_tab(transforms, cases_df, deaths_df, population_df, dates, unique_states, selected_date, county_type_df, vax_latest_df=None) -> None:
    """Geographic choropleth map tab — control panel left, full-width map right."""

    # Build metric catalogue (needed before columns so _vax_metric_cols is accessible)
    _vax_metric_cols = {
        "% Fully Vaccinated":     "vax_complete_pct",
        "% At Least 1 Dose":      "vax_dose1_pct",
        "% Boosted":              "vax_booster_pct",
        "% 65+ Fully Vaccinated": "vax_complete_65plus_pct",
    }
    _has_vax   = vax_latest_df is not None and not vax_latest_df.empty
    _vax_group = list(_vax_metric_cols.keys()) if _has_vax else []

    metric_options = {
        "Cumulative Cases":        ("cases_df",     cases_df),
        "Daily Cases":             ("daily_cases",  transforms["daily_cases"]),
        "Daily Cases (3-day MA)":  ("ma3_cases",    transforms["ma3_cases"]),
        "Daily Cases (5-day MA)":  ("ma5_cases",    transforms["ma5_cases"]),
        "Daily Cases (7-day MA)":  ("ma7_cases",    transforms["ma7_cases"]),
        "Cumulative Deaths":       ("deaths_df",    deaths_df),
        "Daily Deaths":            ("daily_deaths", transforms["daily_deaths"]),
        "Daily Deaths (3-day MA)": ("ma3_deaths",   transforms["ma3_deaths"]),
        "Daily Deaths (5-day MA)": ("ma5_deaths",   transforms["ma5_deaths"]),
        "Daily Deaths (7-day MA)": ("ma7_deaths",   transforms["ma7_deaths"]),
        "Cases per 100k":          ("pc_cases",     transforms["pc_cases"]),
        "Deaths per 100k":         ("pc_deaths",    transforms["pc_deaths"]),
    }
    all_metric_names = list(metric_options.keys()) + _vax_group

    # Shareable map views: ?metric= and ?date= seed the widgets once, before
    # they instantiate; afterwards the user's own selections take over.
    _qp_metric = st.query_params.get("metric")
    if _qp_metric in all_metric_names and "map_metric" not in st.session_state:
        st.session_state["map_metric"] = _qp_metric
    _qp_date = st.query_params.get("date")
    if _qp_date in dates and "map_date" not in st.session_state:
        st.session_state["map_date"] = _qp_date

    # Two-column layout: control panel + map
    map_ctrl_col, map_viz_col = st.columns([3, 9])

    with map_ctrl_col:
        with st.container(border=True):

            st.markdown(
                '<p class="map-ctrl-title">Geographic Explorer</p>',
                unsafe_allow_html=True,
            )

            _date_kwargs = {} if "map_date" in st.session_state else {"value": selected_date}
            map_selected_date = st.select_slider(
                "Date",
                options=dates,
                key="map_date",
                help="Select the date to visualize. Vaccination metrics always show the most recent CDC snapshot.",
                **_date_kwargs,
            )

            st.markdown('<p class="map-ctrl-group">Metric</p>', unsafe_allow_html=True)
            _metric_kwargs = {} if "map_metric" in st.session_state else {"index": 0}
            metric_name = st.selectbox(
                "Metric",
                all_metric_names,
                key="map_metric",
                label_visibility="collapsed",
                help=(
                    "COVID metrics reflect the selected date above. "
                    "Vaccination metrics show the most recent CDC snapshot (through May 2023)."
                ),
                **_metric_kwargs,
            )

            st.markdown('<p class="map-ctrl-group">Filter</p>', unsafe_allow_html=True)
            map_state_filter = st.selectbox(
                "State",
                ["United States"] + unique_states,
                key="map_state_filter",
                help="Zoom to a specific state or view the entire US",
            )
            county_type_filter = st.selectbox(
                "County Type",
                ["All Counties", "Metro Counties", "Nonmetro Counties"],
                index=0,
                key="county_type_filter",
                help=(
                    "Metro: RUCC 1–3 (metropolitan)  |  "
                    "Nonmetro: RUCC 4–9 (non-metropolitan)  |  "
                    "USDA Rural-Urban Continuum Codes (2013)"
                ),
            )
            color_scale_mode = st.selectbox(
                "Color Scale",
                ["Percentile Clip", "Absolute", "Log Scale"],
                index=0,
                key="color_scale_mode",
                help=(
                    "Percentile Clip: clips top 1% of outliers for clearer spatial patterns.  "
                    "Absolute: full national range (may look pale for right-skewed metrics).  "
                    "Log Scale: log₁₊₁ axis — best for cumulative counts spanning many orders of magnitude."
                ),
            )
            cb_safe = st.checkbox(
                "Colorblind-safe palette",
                value=False,
                key="map_cb_safe",
                help="Render with the Viridis scale (perceptually uniform, "
                     "readable with all common color-vision deficiencies).",
            )

            # Open County Overview
            st.markdown('<p class="map-ctrl-group">County Profile</p>', unsafe_allow_html=True)
            _map_county_pick = st.selectbox(
                "County",
                locations,
                key="map_county_for_overview",
                label_visibility="collapsed",
                help="Select a county, then click the button to open its full public health profile.",
            )
            if st.button("Open County Overview →", key="map_open_overview", type="primary", use_container_width=True):
                st.session_state["overview_county"] = _map_county_pick
                st.toast(
                    f"**{_map_county_pick}** loaded — click the **County Overview** tab to view its full profile.",
                )

    _is_vax_metric = metric_name in _vax_metric_cols

    # Reflect the current map view into the URL for sharing
    if st.query_params.get("metric") != metric_name:
        st.query_params["metric"] = metric_name
    if not _is_vax_metric and st.query_params.get("date") != map_selected_date:
        st.query_params["date"] = map_selected_date

    if _is_vax_metric:
        _vax_col = _vax_metric_cols[metric_name]
        identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
        _date_cols   = [c for c in cases_df.columns if c not in identifier_cols]
        _latest_date = sorted(_date_cols)[-1]

        _pop_col_cands = [c for c in population_df.columns if c not in identifier_cols and c != "Location"]
        _pop_col = _pop_col_cands[0] if _pop_col_cands else None

        _base = population_df[
            population_df["countyFIPS"] != "00000"
        ][["countyFIPS", "County Name", "State", "Location"] + (
            [_pop_col] if _pop_col else []
        )].copy()
        if _pop_col:
            _base = _base.rename(columns={_pop_col: "population"})
        else:
            _base["population"] = np.nan

        _case_latest  = cases_df[["countyFIPS",  "State", _latest_date]].rename(columns={_latest_date: "cases"})
        _death_latest = deaths_df[["countyFIPS", "State", _latest_date]].rename(columns={_latest_date: "deaths"})
        _base = _base.merge(_case_latest,  on=["countyFIPS", "State"], how="left")
        _base = _base.merge(_death_latest, on=["countyFIPS", "State"], how="left")
        for _c in ("population", "cases", "deaths"):
            _base[_c] = pd.to_numeric(_base[_c], errors="coerce").fillna(0)
        _base["cases_pc"]  = np.where(_base["population"] > 0, (_base["cases"]  / _base["population"]) * 100_000, np.nan)
        _base["deaths_pc"] = np.where(_base["population"] > 0, (_base["deaths"] / _base["population"]) * 100_000, np.nan)

        if _vax_col in vax_latest_df.columns:
            _vax_sub = vax_latest_df[["countyFIPS", _vax_col]].copy()
            _vax_sub["countyFIPS"] = _vax_sub["countyFIPS"].astype(str).str.zfill(5)
            _base = _base.merge(_vax_sub, on="countyFIPS", how="left")
            _base = _base.rename(columns={_vax_col: "value"})
        else:
            _base["value"] = np.nan

        choro_data = _base[["countyFIPS", "State", "Location", "population", "cases", "deaths", "cases_pc", "deaths_pc", "value"]].copy()

    else:
        _, metric_df = metric_options[metric_name]
        choro_data = get_choropleth_data(metric_df, map_selected_date, cases_df, deaths_df, population_df)

    # Merge county type classification
    choro_data = choro_data.merge(
        county_type_df[["countyFIPS", "State", "County_Type"]],
        on=["countyFIPS", "State"],
        how="left",
    )

    filtered_choro_data = filter_choropleth_by_state(choro_data, map_state_filter)
    filtered_choro_data = filter_choropleth_by_county_type(filtered_choro_data, county_type_filter)

    geo_config = dict(scope="usa", projection_type="albers usa")
    if map_state_filter != "United States":
        state_bounds = get_state_bounds_for_zoom(map_state_filter)
        if state_bounds:
            geo_config["center"]     = {"lat": state_bounds["lat"], "lon": state_bounds["lon"]}
            geo_config["projection"] = {"scale": state_bounds["zoom"]}

    filtered_choro_data["countyFIPS"] = filtered_choro_data["countyFIPS"].astype(str).str.zfill(5)

    # Color scale anchored to national (pre-filter) range
    _national_vals       = choro_data["value"].dropna()
    _national_actual_max = float(_national_vals.max()) if not _national_vals.empty else 1.0

    color_col    = "value"
    _colorbar_kw = {}

    if color_scale_mode == "Absolute":
        _zmin = 0.0
        _zmax = max(_national_actual_max, 1.0)

    elif color_scale_mode == "Percentile Clip":
        _zmin = 0.0
        _p99  = float(np.percentile(_national_vals, 99)) if len(_national_vals) > 0 else 1.0
        _zmax = max(_p99, 1.0)

    else:  # Log Scale
        _plot_df = filtered_choro_data.copy()
        _plot_df["log_value"] = np.log1p(_plot_df["value"].clip(lower=0))
        filtered_choro_data = _plot_df
        color_col = "log_value"
        _zmin = 0.0
        _zmax = max(float(np.log1p(_national_actual_max)), float(np.log1p(1.0)))
        _round_ticks = [0, 1, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000,
                        50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
        _tick_pairs  = [(v, float(np.log1p(v))) for v in _round_ticks if v <= _national_actual_max]
        if _national_actual_max not in _round_ticks:
            _tick_pairs.append((_national_actual_max, _zmax))
        _colorbar_kw = dict(
            coloraxis_colorbar=dict(
                tickvals=[p[1] for p in _tick_pairs],
                ticktext=[f"{int(p[0]):,}" for p in _tick_pairs],
                title=dict(text=metric_name + "<br>(log scale)"),
            )
        )

    _hover_data: dict = {
        "countyFIPS":  False,
        "Location":    True,
        "population":  ":,",
        "cases":       ":,",
        "deaths":      ":,",
        "cases_pc":    ":.1f",
        "deaths_pc":   ":.1f",
        "value":       ":.1f",
        "County_Type": True,
    }
    if color_col != "value":
        _hover_data[color_col] = False

    color_scale = (
        "Blues" if _is_vax_metric else
        ("OrRd" if "Deaths" in metric_name else "YlOrRd")
    )
    if cb_safe:
        color_scale = "Viridis"
    fig_map = px.choropleth(
        filtered_choro_data,
        locations="countyFIPS",
        color=color_col,
        scope="usa",
        geojson=GEO_SOURCE,
        featureidkey="id",
        color_continuous_scale=color_scale,
        range_color=[_zmin, _zmax],
        hover_data=_hover_data,
        labels={color_col: metric_name},
    )

    title_text = f"<b>{metric_name} by County</b><br><sub>{map_selected_date}"
    filter_parts = []
    if map_state_filter != "United States":
        filter_parts.append(map_state_filter)
    if county_type_filter != "All Counties":
        filter_parts.append(county_type_filter)
    if filter_parts:
        title_text += " — " + " · ".join(filter_parts)
    if _is_vax_metric:
        title_text += " · CDC Vaccination Snapshot (as of May 2023)"
    title_text += "</sub>"

    fig_map.update_layout(
        title_text=title_text,
        geo=geo_config,
        height=800,
        margin={"r": 0, "t": 65, "l": 0, "b": 0},
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=11),
        paper_bgcolor="white",
    )
    if _colorbar_kw:
        fig_map.update_layout(**_colorbar_kw)
    # customdata order: [0]=Location [1]=population [2]=cases [3]=deaths
    #                   [4]=cases_pc [5]=deaths_pc  [6]=value [7]=County_Type
    fig_map.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Population: %{customdata[1]}<br>"
            "Cases: %{customdata[2]}<br>"
            "Deaths: %{customdata[3]}<br>"
            "Cases/100k: %{customdata[4]:.1f}<br>"
            "Deaths/100k: %{customdata[5]:.1f}<br>"
            f"{metric_name}: %{{customdata[6]:.1f}}<br>"
            "Metro/Nonmetro: %{customdata[7]}<extra></extra>"
        )
    )

    with map_viz_col:
        if _is_vax_metric:
            st.caption(
                f"**{metric_name}** — Vaccination data shows the most recent CDC county snapshot "
                "(through May 2023). The date slider above does not apply to this metric."
            )
        _map_event = st.plotly_chart(
            fig_map,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="map_select",
        )
        st.caption("Click any county to load it into the County Overview tab.")

        # Click-to-profile: resolve the clicked polygon to a Location string and
        # stash it for the pre-tab consumer (see startup block). _last_map_click
        # guards against re-processing the same persisted selection every rerun.
        _sel_points = []
        if _map_event is not None:
            _sel_points = (getattr(_map_event, "selection", None) or {}).get("points", [])
        if _sel_points:
            _clicked_loc = None
            _pt = _sel_points[0]
            _cd = _pt.get("customdata")
            if _cd:
                _clicked_loc = _cd[0]  # customdata[0] = Location (see hover map)
            elif _pt.get("location"):
                _match = filtered_choro_data[
                    filtered_choro_data["countyFIPS"] == str(_pt["location"]).zfill(5)
                ]
                if not _match.empty:
                    _clicked_loc = _match.iloc[0]["Location"]
            if _clicked_loc and _clicked_loc != st.session_state.get("_last_map_click"):
                st.session_state["_last_map_click"] = _clicked_loc
                st.session_state["_pending_overview_county"] = _clicked_loc
                st.rerun()

        # Animated monthly playback (COVID metrics only — vaccination is a
        # single snapshot). Frames are built lazily and cached per metric.
        if not _is_vax_metric:
            with st.expander("Animated playback — watch the pandemic move month by month", expanded=False):
                if st.checkbox("Build animation", key="map_animate",
                               help="One frame per month (~40 frames). First build takes a few seconds."):
                    _metric_key, _metric_src = metric_options[metric_name]
                    _anim_df = get_monthly_animation_frames(_metric_src, _metric_key)
                    _anim_vals = _anim_df["value"]
                    _anim_max = float(np.percentile(_anim_vals, 99)) if len(_anim_vals) else 1.0
                    fig_anim = px.choropleth(
                        _anim_df,
                        locations="countyFIPS",
                        color="value",
                        animation_frame="Month",
                        scope="usa",
                        geojson=GEO_SOURCE,
                        featureidkey="id",
                        color_continuous_scale=color_scale,
                        range_color=[0, max(_anim_max, 1.0)],
                        hover_name="Location" if "Location" in _anim_df.columns else None,
                        labels={"value": metric_name},
                    )
                    fig_anim.update_layout(
                        height=620,
                        margin={"r": 0, "t": 30, "l": 0, "b": 0},
                        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=11),
                        paper_bgcolor="white",
                    )
                    st.plotly_chart(fig_anim, use_container_width=True)
                    st.caption(
                        f"**{metric_name}**, one frame per month. Color scale fixed at the "
                        "99th-percentile value across the whole period so frames are comparable. "
                        "Use the play button or drag the month slider."
                    )

        # Spatial clustering: Getis-Ord Gi* hotspots for the current metric/date
        with st.expander("Hotspot analysis — where is this metric spatially clustered?", expanded=False):
            if st.checkbox("Run hotspot analysis", key="map_hotspots",
                           help="Getis-Ord Gi* over county contiguity, national scope"):
                if get_county_adjacency() is None:
                    st.info(
                        "County boundary file unavailable — hotspot analysis needs "
                        "data/geojson-counties-fips.json (downloaded automatically "
                        "when the app has network access)."
                    )
                else:
                    _hs = get_hotspot_analysis(
                        choro_data, metric_name,
                        map_selected_date if not _is_vax_metric else "latest",
                    )
                    if _hs is None or _hs["gi_z"].isna().all():
                        st.info("Not enough data to compute hotspot statistics.")
                    else:
                        fig_hs = px.choropleth(
                            _hs,
                            locations="countyFIPS",
                            color="gi_category",
                            scope="usa",
                            geojson=GEO_SOURCE,
                            featureidkey="id",
                            category_orders={"gi_category": ["Hotspot", "Not significant", "Coldspot"]},
                            color_discrete_map={
                                "Hotspot": "#c41e3a",
                                "Coldspot": "#1E4D87",
                                "Not significant": "#E3E8F0",
                            },
                            hover_name="Location",
                            hover_data={"countyFIPS": False, "gi_z": ":.2f", "value": ":.1f"},
                            labels={"gi_category": "Spatial cluster", "gi_z": "Gi* z-score",
                                    "value": metric_name},
                        )
                        fig_hs.update_layout(
                            height=560,
                            margin={"r": 0, "t": 30, "l": 0, "b": 0},
                            font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=11),
                            paper_bgcolor="white",
                            legend=dict(orientation="h", y=-0.05),
                        )
                        st.plotly_chart(fig_hs, use_container_width=True)
                        _n_hot = int((_hs["gi_category"] == "Hotspot").sum())
                        _n_cold = int((_hs["gi_category"] == "Coldspot").sum())
                        st.caption(
                            f"Getis-Ord Gi* with rook contiguity (national scope, ignores the state filter): "
                            f"**{_n_hot} hotspot** and **{_n_cold} coldspot** counties at p < 0.05. "
                            "A hotspot is a county whose neighbourhood shows unusually high values "
                            "of the selected metric — spatial clustering, not just a high county."
                        )
                        with st.expander("The statistic", expanded=False):
                            st.latex(
                                r"G_i^* = \frac{\sum_j w_{ij} x_j - \bar{x} W_i}"
                                r"{s\sqrt{\left[n W_i - W_i^2\right]/(n-1)}}"
                            )
                            st.caption(
                                "Binary weights (neighbour = 1, including the county itself); "
                                "W is the neighbourhood size, x̄ and s the national mean and SD. "
                                "The result is a z-score: above +1.96 → hotspot, below −1.96 → coldspot."
                            )

        render_learning_aids(
            terms=("per_100k", "cumulative", "moving_average", "hotspot"),
            questions=(
                "Step through early 2020 with the date slider. Where does the map light "
                "up first, and how long before it reaches the middle of the country?",
                "Switch between **Cumulative Cases** and **Cases per 100k**. Which regions "
                "change most, and what does that say about raw counts?",
                "Run the hotspot analysis on **Deaths per 100k**. Do the hotspots follow "
                "state borders? Should they?",
            ),
        )

    # Export controls render back into the control-panel column; they depend on
    # filtered_choro_data, which only exists after the pipeline above has run.
    with map_ctrl_col:
        _export_cols = {
            "countyFIPS": "FIPS",
            "Location":   "County",
            "State":      "State",
            "population": "Population",
            "cases":      "Total Cases",
            "deaths":     "Total Deaths",
            "cases_pc":   "Cases per 100k",
            "deaths_pc":  "Deaths per 100k",
            "value":      metric_name,
        }
        _export_df = filtered_choro_data.rename(
            columns={k: v for k, v in _export_cols.items() if k in filtered_choro_data.columns}
        )
        st.markdown(
            '<p class="map-ctrl-group" style="margin-top:0.6rem;border-top:none;">Export</p>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download map data as CSV",
            data=_export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"covid_map_{metric_name.replace(' ','_').replace('/','-')}_{map_selected_date}.csv",
            mime="text/csv",
            key="map_download",
            use_container_width=True,
        )
        st.caption(f"{len(_export_df):,} counties · {metric_name} · {map_selected_date}")

def render_comparison_tab(cases_df, deaths_df, population_df, locations, national, vax_ts_df=None) -> None:
    """Unified comparison tab: County vs County / County vs Nation / County vs County vs Nation."""
    render_section_header(
        "Trend Comparison",
        "How did COVID-19 unfold differently across communities? Overlay timelines for two counties, "
        "benchmark a county against the national average, or run a three-way comparison to see "
        "which populations were hit earliest, hardest, and longest.",
    )
    render_learning_aids(
        terms=("per_100k", "cumulative", "moving_average"),
        questions=(
            "Compare a dense metro county with a rural one using **Cases per 100k** — "
            "does the raw-count impression survive normalization?",
            "Switch to **Normalized (Index = 100)** for two counties whose outbreaks "
            "started months apart. Which grew faster from its own starting point?",
            "Find a pair of neighbouring counties whose death curves diverge even "
            "though their case curves match. What might differ between them?",
        ),
    )

    # Primary controls (always visible)
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        cmp_series = st.selectbox(
            "Comparison Mode",
            ["County vs County", "County vs Nation", "County vs County vs Nation"],
            key="cmp_series",
            help=(
                "County vs County: side-by-side county trends  |  "
                "County vs Nation: one county against the national total  |  "
                "County vs County vs Nation: all three simultaneously"
            ),
        )
    with ctrl2:
        county_a = st.selectbox("County A", locations, key="cmp_county_a", index=0)
    with ctrl3:
        include_county_b = cmp_series in ("County vs County", "County vs County vs Nation")
        county_b = st.selectbox(
            "County B", locations,
            key="cmp_county_b",
            index=min(1, len(locations) - 1),
            disabled=not include_county_b,
        )
    with ctrl4:
        _vax_cmp_opts = (
            ["Vaccination Rate (Fully Vaccinated %)", "At Least 1 Dose (%)"]
            if (vax_ts_df is not None and not vax_ts_df.empty) else []
        )
        dual_metric = st.selectbox(
            "Metric",
            ["Cases", "Deaths", "Cases per 100k", "Deaths per 100k"] + _vax_cmp_opts,
            key="cmp_metric",
        )

    # Chart options (secondary — in expander)
    with st.expander("Chart options", expanded=False):
        n_series = 3 if cmp_series == "County vs County vs Nation" else 2
        _eo1, _eo2, _eo3, _eo4 = st.columns([1, 1, 2, 1])
        with _eo1:
            dual_view = st.selectbox("View", ["Cumulative", "Daily"], key="cmp_view")
        with _eo2:
            dual_ma = st.selectbox(
                "Smoothing", ["None", "3-day MA", "5-day MA", "7-day MA"], key="cmp_ma",
                help="Rolling average applied to all active series simultaneously",
            )
        with _eo3:
            display_opts = ["Standard", "Normalized (Index = 100)"]
            if n_series == 2:
                display_opts.append("Dual-Axis View")
            display_mode = st.selectbox(
                "Display Mode",
                display_opts,
                key="cmp_display",
                help=(
                    "Standard: shared y-axis in selected units  |  "
                    "Normalized: each series rebased so its first non-zero value = 100  |  "
                    "Dual-Axis: County A on left axis, second series on right (2-series only)"
                ),
            )
        with _eo4:
            use_log_scale = st.checkbox(
                "Log Scale", value=False, key="cmp_log",
                help="Logarithmic y-axis — useful for comparing outbreak trajectories across orders of magnitude",
            )

    pop_col_name = [
        c for c in population_df.columns
        if c not in ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    ][0]
    national_population = (
        population_df[
            (population_df["countyFIPS"] != "00000") &
            (pd.to_numeric(population_df[pop_col_name], errors="coerce") > 0)
        ][pop_col_name]
        .apply(pd.to_numeric, errors="coerce")
        .sum()
    )

    def _get_county_series(county_name, state_abbr):
        """Return (DataFrame[Date, value_col], plot_col) for one county."""
        is_cases  = "cases" in dual_metric.lower()
        source_df = cases_df if is_cases else deaths_df
        base_col  = "Cases" if is_cases else "Deaths"
        ts = prepare_county_timeseries(source_df, county_name, state_abbr, base_col)
        if ts.empty:
            return pd.DataFrame(), base_col
        if dual_view == "Daily":
            ts       = calculate_daily_changes(ts, base_col)
            plot_col = f"Daily {base_col}"
        else:
            plot_col = base_col
        if "per 100k" in dual_metric.lower():
            ts_pc = calculate_per_capita(ts[["Date", plot_col]], population_df, county_name, state_abbr)
            if "Per Capita" in ts_pc.columns:
                ts, plot_col = ts_pc, "Per Capita"
        if dual_ma != "None":
            window   = int(dual_ma.split("-")[0])
            ts       = apply_moving_average(ts, plot_col, window=window)
            plot_col = f"{plot_col} MA"
        return ts[["Date", plot_col]].copy(), plot_col

    def _get_national_series():
        """Return DataFrame[Date, National] for the national aggregate."""
        is_cases = "cases" in dual_metric.lower()
        if "per 100k" in dual_metric.lower():
            if dual_view == "Cumulative":
                ts = (national["pc_cases"] if is_cases else national["pc_deaths"]).copy()
            else:
                # Daily per-capita: divide national daily raw counts by total US population
                ts = (national["daily_cases"] if is_cases else national["daily_deaths"]).copy()
                ts.columns = ["Date", "National"]
                if national_population > 0:
                    ts["National"] = ts["National"] / national_population * 100_000
                if dual_ma != "None":
                    w = int(dual_ma.split("-")[0])
                    ts["National"] = ts["National"].rolling(window=w, min_periods=1).mean()
                return ts
        elif dual_view == "Cumulative":
            ts = (national["cases_ts"] if is_cases else national["deaths_ts"]).copy()
        else:
            ts = (national["daily_cases"] if is_cases else national["daily_deaths"]).copy()
        ts = ts.copy()
        ts.columns = ["Date", "National"]
        if dual_ma != "None":
            w = int(dual_ma.split("-")[0])
            ts["National"] = ts["National"].rolling(window=w, min_periods=1).mean()
        return ts

    county_a_name, county_a_state = extract_county_state(county_a)

    # Vaccination comparison (special path)
    _is_vax_cmp = "Vaccination" in dual_metric or "1 Dose" in dual_metric
    if _is_vax_cmp and vax_ts_df is not None and not vax_ts_df.empty:
        _vax_cmp_col = (
            "vax_complete_pct" if "Fully Vaccinated" in dual_metric else "vax_dose1_pct"
        )
        _vax_y_label = (
            "% Fully Vaccinated" if "Fully Vaccinated" in dual_metric else "% At Least 1 Dose"
        )
        st.caption(
            "Vaccination data covers Dec 2020 – May 2023 (CDC county dataset). "
            "Smoothing and view controls do not apply to vaccination metrics."
        )

        def _get_vax_county_ts(loc_str):
            """Fetch vaccination time-series for a location string."""
            _cname, _cstate = extract_county_state(loc_str)
            _pop_row = population_df[
                (population_df["County Name"] == _cname) &
                (population_df["State"] == _cstate)
            ]
            if _pop_row.empty or "countyFIPS" not in _pop_row.columns:
                return pd.DataFrame()
            _fips = str(_pop_row.iloc[0]["countyFIPS"]).zfill(5)
            return get_county_vax_timeseries(vax_ts_df, _fips)

        _ts_a = _get_vax_county_ts(county_a)
        _ts_b = _get_vax_county_ts(county_b) if include_county_b else pd.DataFrame()

        # National median vaccination over time (mean of all counties per date)
        if cmp_series in ("County vs Nation", "County vs County vs Nation") and _vax_cmp_col in vax_ts_df.columns:
            _nat_vax_ts = (
                vax_ts_df.groupby("Date")[_vax_cmp_col]
                .median()
                .reset_index()
                .rename(columns={_vax_cmp_col: "National Median"})
            )
        else:
            _nat_vax_ts = pd.DataFrame()

        _fig_vax_cmp = go.Figure()
        if not _ts_a.empty and _vax_cmp_col in _ts_a.columns:
            _fig_vax_cmp.add_trace(go.Scatter(
                x=_ts_a["Date"], y=_ts_a[_vax_cmp_col],
                name=county_a, mode="lines",
                line=dict(color=COUNTY_COLOR, width=2.5),
                hovertemplate=f"<b>{county_a}</b><br>Date: %{{x|%Y-%m-%d}}<br>{_vax_y_label}: %{{y:.1f}}%<extra></extra>",
            ))
        if not _ts_b.empty and _vax_cmp_col in _ts_b.columns:
            _fig_vax_cmp.add_trace(go.Scatter(
                x=_ts_b["Date"], y=_ts_b[_vax_cmp_col],
                name=county_b, mode="lines",
                line=dict(color=NATIONAL_COLOR, width=2.5),
                hovertemplate=f"<b>{county_b}</b><br>Date: %{{x|%Y-%m-%d}}<br>{_vax_y_label}: %{{y:.1f}}%<extra></extra>",
            ))
        if not _nat_vax_ts.empty:
            _fig_vax_cmp.add_trace(go.Scatter(
                x=_nat_vax_ts["Date"], y=_nat_vax_ts["National Median"],
                name="National Median", mode="lines",
                line=dict(color=NATION_LINE_COLOR, width=1.8, dash="dot"),
                hovertemplate=f"<b>National Median</b><br>Date: %{{x|%Y-%m-%d}}<br>{_vax_y_label}: %{{y:.1f}}%<extra></extra>",
            ))

        _series_labs = [county_a]
        if include_county_b:
            _series_labs.append(county_b)
        if not _nat_vax_ts.empty:
            _series_labs.append("National Median")

        _fig_vax_cmp.update_layout(
            title=f"<b>{' vs '.join(_series_labs)}</b><br><sub>Vaccination Rollout — {_vax_y_label}</sub>",
            xaxis_title="Date",
            yaxis=dict(title=_vax_y_label, range=[0, 100]),
            hovermode="x unified",
            height=560,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(255,255,255,0.85)"),
            font=dict(family="sans-serif", size=11),
        )
        _fig_vax_cmp.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        _fig_vax_cmp.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        st.plotly_chart(_fig_vax_cmp, use_container_width=True)

        # Summary cards for vaccination comparison
        _vax_cols_cmp = st.columns(2 if not include_county_b else 3)
        for _i, (_loc, _ts) in enumerate(
            [(county_a, _ts_a)] + ([(county_b, _ts_b)] if include_county_b else [])
        ):
            if _i < len(_vax_cols_cmp) and not _ts.empty and _vax_cmp_col in _ts.columns:
                _last_val = _ts.sort_values("Date")[_vax_cmp_col].dropna().iloc[-1] if not _ts.empty else np.nan
                _last_date = str(_ts.sort_values("Date")["Date"].iloc[-1])[:10] if not _ts.empty else "N/A"
                with _vax_cols_cmp[_i]:
                    st.metric(f"{_loc}", f"{_last_val:.1f}%", help=f"As of {_last_date}")
        return  # Skip the rest of the COVID comparison logic

    # Standard COVID metric comparison
    data_a, plot_col_a = _get_county_series(county_a_name, county_a_state)

    if include_county_b:
        county_b_name, county_b_state = extract_county_state(county_b)
        data_b, plot_col_b = _get_county_series(county_b_name, county_b_state)
    else:
        data_b, plot_col_b = pd.DataFrame(), None

    include_national = cmp_series in ("County vs Nation", "County vs County vs Nation")
    data_n = _get_national_series() if include_national else pd.DataFrame()

    if data_a.empty:
        st.warning(f"No valid data for {county_a}. Try a different county.")
        return
    if include_county_b and data_b.empty:
        st.warning(f"No valid data for {county_b}. Try a different county.")
        return

    is_pc            = "per 100k" in dual_metric.lower()
    base_metric_name = "Cases" if "cases" in dual_metric.lower() else "Deaths"
    view_prefix      = "Daily" if dual_view == "Daily" else "Cumulative"
    y_label = (
        f"{view_prefix} {base_metric_name} per 100k" if is_pc
        else f"{view_prefix} {base_metric_name}"
    )
    if dual_ma != "None":
        y_label += f" ({dual_ma})"

    if display_mode == "Normalized (Index = 100)":
        def _rebase(series):
            first_nz = series.replace(0, np.nan).dropna()
            if first_nz.empty:
                return series * np.nan
            return series / first_nz.iloc[0] * 100

        data_a = data_a.copy()
        data_a[plot_col_a] = _rebase(data_a[plot_col_a])
        if include_county_b and not data_b.empty:
            data_b = data_b.copy()
            data_b[plot_col_b] = _rebase(data_b[plot_col_b])
        if include_national and not data_n.empty:
            data_n = data_n.copy()
            data_n["National"] = _rebase(data_n["National"])
        y_label = "Index (first non-zero value = 100)"

    series_labels = [county_a]
    if include_county_b:
        series_labels.append(county_b)
    if include_national:
        series_labels.append("United States")
    view_label  = dual_view + (f", {dual_ma}" if dual_ma != "None" else "")
    chart_title = (
        f"<b>{' vs '.join(series_labels)}</b>"
        f"<br><sub>{view_label} {dual_metric} — {display_mode}</sub>"
    )

    fig = go.Figure()

    if display_mode == "Dual-Axis View" and n_series == 2:
        # County A on left axis, second series on right
        if include_county_b and not data_b.empty:
            second_label = county_b
            second_x     = data_b["Date"]
            second_y     = data_b[plot_col_b]
            second_color = NATIONAL_COLOR
        else:
            second_label = "United States"
            second_x     = data_n["Date"]
            second_y     = data_n["National"]
            second_color = NATION_LINE_COLOR

        fig.add_trace(go.Scatter(
            x=data_a["Date"], y=data_a[plot_col_a],
            name=county_a, mode="lines",
            line=dict(color=COUNTY_COLOR, width=2.5), yaxis="y1",
            hovertemplate=f"<b>{county_a}</b><br>Date: %{{x|%Y-%m-%d}}<br>{y_label}: %{{y:,.2f}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=second_x, y=second_y,
            name=second_label, mode="lines",
            line=dict(color=second_color, width=2.5), yaxis="y2",
            hovertemplate=f"<b>{second_label}</b><br>Date: %{{x|%Y-%m-%d}}<br>{y_label}: %{{y:,.2f}}<extra></extra>",
        ))
        fig.update_layout(
            title=chart_title,
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
            yaxis=dict(
                title=dict(text=f"{county_a} — {y_label}", font=dict(color=COUNTY_COLOR)),
                tickfont=dict(color=COUNTY_COLOR),
                showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                type="log" if use_log_scale else "linear",
            ),
            yaxis2=dict(
                title=dict(text=f"{second_label} — {y_label}", font=dict(color=second_color)),
                tickfont=dict(color=second_color),
                overlaying="y", side="right", showgrid=False,
                type="log" if use_log_scale else "linear",
            ),
            hovermode="x unified", height=600, template="plotly_white",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
            font=dict(family="sans-serif", size=11),
        )
    else:
        # Standard / Normalized single-axis: national → County B → County A (front)
        if include_national and not data_n.empty:
            fig.add_trace(go.Scatter(
                x=data_n["Date"], y=data_n["National"],
                name="United States", mode="lines",
                line=dict(color=NATION_LINE_COLOR, width=2, dash="dot"),
                hovertemplate=f"<b>United States</b><br>Date: %{{x|%Y-%m-%d}}<br>{y_label}: %{{y:,.2f}}<extra></extra>",
            ))
        if include_county_b and not data_b.empty:
            fig.add_trace(go.Scatter(
                x=data_b["Date"], y=data_b[plot_col_b],
                name=county_b, mode="lines",
                line=dict(color=NATIONAL_COLOR, width=2.5),
                hovertemplate=f"<b>{county_b}</b><br>Date: %{{x|%Y-%m-%d}}<br>{y_label}: %{{y:,.2f}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=data_a["Date"], y=data_a[plot_col_a],
            name=county_a, mode="lines",
            line=dict(color=COUNTY_COLOR, width=2.5),
            hovertemplate=f"<b>{county_a}</b><br>Date: %{{x|%Y-%m-%d}}<br>{y_label}: %{{y:,.2f}}<extra></extra>",
        ))
        fig.update_layout(
            title=chart_title,
            xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
            yaxis=dict(
                title=y_label,
                showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                type="log" if use_log_scale else "linear",
            ),
            hovermode="x unified", height=600, template="plotly_white",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
            font=dict(family="sans-serif", size=11),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")

    # Reporting-correction markers (daily views): dates where a county's
    # cumulative source series decreased. The daily pipeline clips these to
    # zero; flagging them explains sudden gaps students would otherwise miss.
    if dual_view == "Daily":
        _corr_specs = [(county_a_name, county_a_state, county_a)]
        if include_county_b and not data_b.empty:
            _corr_specs.append((county_b_name, county_b_state, county_b))
        _corr_src = cases_df if "cases" in dual_metric.lower() else deaths_df
        for _cn, _cs, _clabel in _corr_specs:
            _corr = find_data_corrections(_corr_src, _cn, _cs)
            if not _corr.empty:
                fig.add_trace(go.Scatter(
                    x=_corr["Date"], y=np.zeros(len(_corr)),
                    name=f"Corrections — {_clabel}", mode="markers",
                    marker=dict(symbol="x-thin", size=9, color="#8A94A3",
                                line=dict(width=1.6, color="#8A94A3")),
                    customdata=_corr["correction"].values,
                    hovertemplate=(f"<b>{_clabel}</b> reporting correction: "
                                   "%{customdata:,.0f} on %{x|%Y-%m-%d}"
                                   "<br>(daily value clipped to 0)<extra></extra>"),
                ))

    st.plotly_chart(fig, use_container_width=True)

    a_latest = data_a[plot_col_a].dropna().iloc[-1] if data_a[plot_col_a].dropna().shape[0] > 0 else np.nan
    b_latest = (
        data_b[plot_col_b].dropna().iloc[-1]
        if (include_county_b and not data_b.empty and data_b[plot_col_b].dropna().shape[0] > 0)
        else np.nan
    )
    n_latest = (
        data_n["National"].dropna().iloc[-1]
        if (include_national and not data_n.empty and data_n["National"].dropna().shape[0] > 0)
        else np.nan
    )

    s1, s2, s3, s4 = st.columns(4)
    if cmp_series == "County vs County":
        a_first = data_a[plot_col_a].replace(0, np.nan).dropna()
        b_first = data_b[plot_col_b].replace(0, np.nan).dropna() if not data_b.empty else pd.Series(dtype=float)
        a_chg   = ((a_latest - a_first.iloc[0]) / a_first.iloc[0] * 100) if not a_first.empty and pd.notna(a_latest) else np.nan
        b_chg   = ((b_latest - b_first.iloc[0]) / b_first.iloc[0] * 100) if not b_first.empty and pd.notna(b_latest) else np.nan
        with s1: st.metric(f"{county_a} (Latest)", f"{a_latest:,.2f}" if pd.notna(a_latest) else "N/A")
        with s2: st.metric(f"{county_b} (Latest)", f"{b_latest:,.2f}" if pd.notna(b_latest) else "N/A")
        with s3:
            if pd.notna(a_latest) and pd.notna(b_latest) and a_latest != 0:
                st.metric(f"{county_b} / {county_a}", f"{b_latest / a_latest:.2f}×")
            else:
                st.metric(f"{county_b} / {county_a}", "N/A")
        with s4:
            a_str = f"{a_chg:.0f}%" if pd.notna(a_chg) else "N/A"
            b_str = f"{b_chg:.0f}%" if pd.notna(b_chg) else "N/A"
            st.metric("% Change (A vs B)", f"{a_str} vs {b_str}")

    elif cmp_series == "County vs Nation":
        pop_row      = population_df[(population_df["County Name"] == county_a_name) & (population_df["State"] == county_a_state)]
        county_a_pop = pd.to_numeric(pop_row.iloc[0][pop_col_name], errors="coerce") if not pop_row.empty else np.nan
        with s1: st.metric(f"{county_a} (Latest)", f"{a_latest:,.2f}" if pd.notna(a_latest) else "N/A")
        with s2: st.metric("United States (Latest)", f"{n_latest:,.2f}" if pd.notna(n_latest) else "N/A")
        with s3:
            if pd.notna(a_latest) and pd.notna(n_latest) and n_latest != 0:
                st.metric("County / National", f"{a_latest / n_latest:.2f}×")
            else:
                st.metric("County / National", "N/A")
        with s4:
            st.metric("County Population",
                      f"{int(county_a_pop):,}" if pd.notna(county_a_pop) else "N/A")

    else:  # County vs County vs Nation
        with s1: st.metric(f"{county_a} (Latest)", f"{a_latest:,.2f}" if pd.notna(a_latest) else "N/A")
        with s2: st.metric(f"{county_b} (Latest)", f"{b_latest:,.2f}" if pd.notna(b_latest) else "N/A")
        with s3: st.metric("United States (Latest)", f"{n_latest:,.2f}" if pd.notna(n_latest) else "N/A")
        with s4:
            if pd.notna(a_latest) and pd.notna(n_latest) and n_latest != 0:
                st.metric(f"{county_a} / National", f"{a_latest / n_latest:.2f}×")
            else:
                st.metric(f"{county_a} / National", "N/A")

    # Comparison data export
    with st.expander("Download comparison data", expanded=False):
        _cmp_frames = []
        if not data_a.empty and plot_col_a in data_a.columns:
            _fa = data_a[["Date", plot_col_a]].copy()
            _fa.columns = ["Date", county_a]
            _cmp_frames.append(_fa.set_index("Date"))
        if include_county_b and not data_b.empty and plot_col_b in data_b.columns:
            _fb = data_b[["Date", plot_col_b]].copy()
            _fb.columns = ["Date", county_b]
            _cmp_frames.append(_fb.set_index("Date"))
        if include_national and not data_n.empty:
            _fn = data_n[["Date", "National"]].copy()
            _fn.columns = ["Date", "United States"]
            _cmp_frames.append(_fn.set_index("Date"))
        if _cmp_frames:
            _cmp_export = pd.concat(_cmp_frames, axis=1).reset_index()
            _safe_metric = dual_metric.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
            st.download_button(
                label="Download as CSV",
                data=_cmp_export.to_csv(index=False).encode("utf-8"),
                file_name=f"covid_comparison_{_safe_metric}.csv",
                mime="text/csv",
                key="cmp_download",
            )
            st.caption(f"{len(_cmp_export):,} dates · metric: {dual_metric}")

    _mode_desc = {
        "Standard": (
            "**Standard** — all series plotted in the selected units on a shared y-axis. "
            "Use *Cases per 100k* or *Deaths per 100k* metrics for epidemiologically "
            "comparable rates that account for population size differences."
        ),
        "Normalized (Index = 100)": (
            "**Normalized (Index = 100)** — each series is rebased so its first non-zero "
            "value equals 100. A reading of 200 means the count has doubled. "
            "Reveals relative outbreak trajectories regardless of absolute scale, "
            "making a small rural county directly comparable to a large metro area."
        ),
        "Dual-Axis View": (
            "**Dual-Axis View** — County A uses the left y-axis and the second series "
            "uses the right, each on its own independent scale. Both curves remain "
            "fully visible even when absolute magnitudes differ by orders of magnitude. "
            "Caution: the visual gap between lines does not reflect true difference in magnitude."
        ),
    }
    with st.expander("About this display mode", expanded=False):
        st.markdown(_mode_desc.get(display_mode, ""))

def render_county_overview_tab(
    cases_df, deaths_df, population_df, locations, dates,
    master_county_df, transforms, vax_ts_df=None,
) -> None:
    """County Overview tab — dashboard landing page and per-county fact sheet."""

    # Landing strip: a short orientation plus one chip per analysis tab, so
    # first-time users can see the breadth of the platform at a glance.
    st.markdown(
        '<div class="welcome-banner">'
        '<p class="welcome-kicker">Start Here</p>'
        '<h3>Every county had a different pandemic. Find out why.</h3>'
        '<p>Pick any of 3,000+ U.S. counties below to see its full story — outbreak waves, '
        'vaccination rollout, healthcare capacity, and how it compares to the rest of the '
        'country. Then use the tabs above to dig deeper.</p>'
        '</div>'
        '<div class="explore-chips">'
        '<div class="explore-chip"><span class="chip-title">Geographic Map</span>'
        '<span class="chip-desc">Watch the pandemic move across the country, one date at a time.</span></div>'
        '<div class="explore-chip"><span class="chip-title">County Comparison</span>'
        '<span class="chip-desc">Put two counties side by side — or measure one against the nation.</span></div>'
        '<div class="explore-chip"><span class="chip-title">Wave Analysis</span>'
        '<span class="chip-desc">Find each outbreak wave and see how big, long, and severe it was.</span></div>'
        '<div class="explore-chip"><span class="chip-title">Time Lag Analysis</span>'
        '<span class="chip-desc">Measure how many days deaths trailed behind case surges.</span></div>'
        '<div class="explore-chip"><span class="chip-title">County Factors</span>'
        '<span class="chip-desc">See how income, healthcare access, and vaccination relate to outcomes.</span></div>'
        '<div class="explore-chip"><span class="chip-title">Statistical Modeling</span>'
        '<span class="chip-desc">Let regression and machine learning rank what mattered most.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # County selector. Session state key "overview_county" is shared with the
    # map tab's "Open County Overview" button, the ?county= URL parameter, and
    # the Surprise Me button (whose on_click callback runs before widgets
    # re-instantiate, which is the only safe way to set a widget's state).
    def _pick_random_county():
        st.session_state["overview_county"] = random.choice(locations)

    def _pick_preset_county(loc):
        st.session_state["overview_county"] = loc

    # Pedagogically interesting starting points, chosen to span the extremes
    # students should see: scale, sparsity, timing, and vaccination contrast.
    _CLASSROOM_EXAMPLES = [
        ("Los Angeles County, CA", "The largest county — every national wave is visible."),
        ("Loving County, TX", "The least populous county — see what sparse data does to rates."),
        ("King County, WA", "Site of the first US outbreak; high vaccination uptake."),
        ("McKinley County, NM", "Rural county with severe early burden despite low density."),
        ("Miami-Dade County, FL", "Major metro hit hardest in the Delta era."),
        ("Mingo County, WV", "Low-vaccination Appalachia — contrast with King County."),
    ]

    _ov_sel_col, _ov_rand_col, _ov_ex_col, _ov_desc_col = st.columns(
        [2, 0.8, 1, 2.2], vertical_alignment="bottom"
    )
    with _ov_sel_col:
        location = st.selectbox("Select County", locations, key="overview_county")
    with _ov_rand_col:
        st.button(
            "Surprise me",
            on_click=_pick_random_county,
            use_container_width=True,
            help="Jump to a randomly chosen county",
        )
    with _ov_ex_col:
        with st.popover("Classroom examples", use_container_width=True):
            st.caption("Six counties worth studying — click to load one.")
            for _ex_loc, _ex_note in _CLASSROOM_EXAMPLES:
                if _ex_loc in locations:
                    st.button(
                        _ex_loc,
                        key=f"preset_{_ex_loc}",
                        on_click=_pick_preset_county,
                        args=(_ex_loc,),
                        help=_ex_note,
                        use_container_width=True,
                    )
    with _ov_desc_col:
        st.markdown(
            '<p style="margin:0 0 0.4rem 0;font-size:0.86rem;color:#4B5563;line-height:1.6;">'
            'A complete public health fact sheet for any U.S. county — outcomes, '
            'wave history, vaccination coverage, healthcare access, and economic context.'
            '</p>',
            unsafe_allow_html=True,
        )

    # Keep the URL shareable: reflect the current county into ?county=
    if st.query_params.get("county") != location:
        st.query_params["county"] = location

    render_learning_aids(
        terms=("per_100k", "cfr", "rucc", "hpsa", "ecological"),
        questions=(
            "Load two Classroom examples — King County, WA and Mingo County, WV — "
            "and compare their vaccination rates and deaths per 100k. "
            "What besides vaccination differs between them?",
            "Check Section 7: is this county doing worse than its structural peers, "
            "or just worse than the national median? Why might those disagree?",
        ),
    )

    county_name, state = extract_county_state(location)

    # Lookup helpers
    master_row = pd.Series(dtype=object)
    if master_county_df is not None and not master_county_df.empty:
        _mask = (
            (master_county_df["County Name"] == county_name) &
            (master_county_df["State"] == state)
        )
        if _mask.any():
            master_row = master_county_df[_mask].iloc[0]

    def _v(col, default=np.nan):
        """Safely extract a scalar from master_row."""
        if col in master_row.index:
            val = master_row[col]
            return val if pd.notna(val) else default
        return default

    def _fmt(val, fmt=".1f", fallback="N/A"):
        return f"{val:{fmt}}" if pd.notna(val) else fallback

    def _fmt_int(val, fallback="N/A"):
        try:
            return f"{int(val):,}" if pd.notna(val) else fallback
        except (TypeError, ValueError):
            return fallback

    # National medians (for comparison section)
    nat_med: dict = {}
    if master_county_df is not None and not master_county_df.empty:
        for _col in [
            "cases_per_100k", "deaths_per_100k", "case_fatality_rate",
            "pcp_per_100k", "total_md_per_100k", "hospital_beds_per_100k",
            "icu_beds_per_100k", "snf_beds_per_100k",
            "median_family_income", "per_capita_income",
            "unemployment_rate", "child_poverty_pct",
            "pct_no_hs_diploma", "pct_college_4yr",
            "pop_density_per_sqmi", "pct_pop_65plus", "median_age",
            "vax_complete_pct", "vax_dose1_pct", "vax_booster_pct",
            "vax_complete_65plus_pct",
        ]:
            if _col in master_county_df.columns:
                _s = pd.to_numeric(master_county_df[_col], errors="coerce").dropna()
                if len(_s) > 0:
                    nat_med[_col] = float(_s.median())

    # Identity values (needed for hero banner)
    fips       = _v("countyFIPS", default="—")
    population = _v("population", default=np.nan)
    rucc       = _v("rucc_code",  default=np.nan)
    rucc_group = _v("rucc_group", default="N/A")
    is_metro   = _v("is_metro",   default=None)
    if rucc_group == "N/A" and is_metro is not None:
        rucc_group = "Metro" if is_metro else "Nonmetro"

    # Hero banner
    _pop_str  = f"Population {_fmt_int(population)}" if pd.notna(population) else ""
    _fips_str = f"FIPS {fips}" if fips != "—" else ""
    _meta_parts = [p for p in [_fips_str, _pop_str] if p]
    _meta_str   = " · ".join(_meta_parts) if _meta_parts else state

    _badge_color = "#F26A21" if is_metro is True else "#153A66" if is_metro is False else "#6B7280"
    _badge_label = rucc_group if rucc_group != "N/A" else "Unknown"

    st.markdown(
        f'<div class="county-hero">'
        f'<div class="county-hero-content">'
        f'<h2>{county_name}, {state}</h2>'
        f'<p class="county-hero-meta">{_meta_str}</p>'
        f'<span class="county-hero-badge" style="background:{_badge_color};">'
        f'{_badge_label}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # SECTION 1 — COVID OUTCOMES

    st.markdown(
        '<div class="sub-section-header"><h3>1 — COVID Outcomes</h3></div>',
        unsafe_allow_html=True,
    )

    total_cases   = _v("total_cases",        default=np.nan)
    total_deaths  = _v("total_deaths",       default=np.nan)
    cases_100k    = _v("cases_per_100k",     default=np.nan)
    deaths_100k   = _v("deaths_per_100k",    default=np.nan)
    cfr           = _v("case_fatality_rate", default=np.nan)

    # Fallback: compute directly from raw data if master row is missing
    if pd.isna(total_cases) and dates:
        latest = dates[-1]
        _cr = cases_df[(cases_df["County Name"] == county_name) & (cases_df["State"] == state)]
        _dr = deaths_df[(deaths_df["County Name"] == county_name) & (deaths_df["State"] == state)]
        if not _cr.empty:
            total_cases = pd.to_numeric(_cr.iloc[0].get(latest, np.nan), errors="coerce")
        if not _dr.empty:
            total_deaths = pd.to_numeric(_dr.iloc[0].get(latest, np.nan), errors="coerce")
        if pd.notna(total_cases) and pd.notna(population) and population > 0:
            cases_100k  = (total_cases  / population) * 100_000
            deaths_100k = (total_deaths / population) * 100_000 if pd.notna(total_deaths) else np.nan
        if pd.notna(total_cases) and pd.notna(total_deaths) and total_cases > 0:
            cfr = (total_deaths / total_cases) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric_card("Total Cases",       _fmt_int(total_cases))
    with c2: render_metric_card("Total Deaths",      _fmt_int(total_deaths))
    with c3: render_metric_card("Cases per 100k",    _fmt(cases_100k,  ".1f"))
    with c4: render_metric_card("Deaths per 100k",   _fmt(deaths_100k, ".2f"))
    with c5: render_metric_card("Case Fatality Rate", f"{_fmt(cfr, '.2f')}%" if pd.notna(cfr) else "N/A")

    st.caption(f"Data as of {dates[-1] if dates else '—'}.")

    # SECTION 2 — WAVE ANALYSIS SUMMARY

    st.markdown(
        '<div class="sub-section-header"><h3>2 — Wave Analysis Summary</h3>'
        '<p>Major COVID-19 outbreak waves detected with Standard sensitivity '
        '(7-day MA · adaptive prominence · width + valley filters). '
        'Sorted by wave significance score.</p></div>',
        unsafe_allow_html=True,
    )

    _wave_results = calculate_waves_for_county(
        cases_df, deaths_df,
        transforms["daily_cases"], transforms["daily_deaths"],
        county_name, state,
        ma_window=7, sensitivity="standard",
    )

    if "error" not in _wave_results:
        _wc = _wave_results["cases"]
        _wd = _wave_results["deaths"]

        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            render_wave_metric_card("Major Case Waves", str(_wc["number_of_waves"]))
        with w2:
            render_wave_metric_card("Major Death Waves", str(_wd["number_of_waves"]))
        with w3:
            _peak_raw = _wc["largest_wave"]
            if pd.notna(_peak_raw) and pd.notna(population) and population > 0:
                _peak_label = f"{(_peak_raw / population * 100_000):.1f} /100k"
            else:
                _peak_label = f"{_peak_raw:,.0f}" if pd.notna(_peak_raw) else "N/A"
            render_wave_metric_card("Largest Case Peak", _peak_label)
        with w4:
            _avg_dur = _wc["average_wave_duration"]
            render_wave_metric_card("Avg Wave Duration", f"{_avg_dur:.0f} days" if pd.notna(_avg_dur) and _avg_dur > 0 else "N/A")
        with w5:
            _peak_date = _wc["date_of_peak_wave"]
            render_wave_metric_card("Highest-Significance Peak", str(_peak_date)[:10] if _peak_date else "N/A")

        # Wave detail table sorted by significance
        if _wc["waves"]:
            _profile_wave_rows = []
            _sorted_waves = sorted(_wc["waves"], key=lambda w: w.get("wave_significance", 0), reverse=True)
            for _pw in _sorted_waves:
                _pv_pc = (_pw["peak_value"] / population * 100_000) if population and population > 0 else np.nan
                _dw_match = next(
                    (dw for dw in _wd["waves"]
                     if abs((pd.Timestamp(dw["peak_date"]) - pd.Timestamp(_pw["peak_date"])).days) < 90),
                    None,
                )
                _deaths_pc = None
                if _dw_match and population and population > 0:
                    _deaths_pc = _dw_match["peak_value"] / population * 100_000
                _profile_wave_rows.append({
                    "Significance": f"{_pw.get('wave_significance', 0):.0f}/100",
                    "Start":         str(_pw["start_date"])[:10],
                    "Peak":          str(_pw["peak_date"])[:10],
                    "End":           str(_pw["end_date"])[:10],
                    "Duration":      f"{_pw['duration_days']}d",
                    "Peak Cases /100k": f"{_pv_pc:.1f}" if pd.notna(_pv_pc) else "N/A",
                    "Peak Deaths /100k": f"{_deaths_pc:.3f}" if _deaths_pc is not None else "—",
                })
            if _profile_wave_rows:
                st.dataframe(
                    pd.DataFrame(_profile_wave_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Waves sorted by Significance score (0–100). "
                    "Score combines peak prominence (30%), total burden (30%), duration (20%), "
                    "and burst intensity (20%). "
                    "Peak Deaths /100k shows the matching death wave peak within 90 days of each case peak."
                )

        # Compact mini-chart: smoothed daily cases per 100k with wave peak markers
        if _wc["number_of_waves"] > 0 or _wd["number_of_waves"] > 0:
            _lag_ts_mini = analyze_county_lag(
                cases_df, deaths_df, population_df, county_name, state,
                ma_window=7, case_prominence=1.0, death_prominence=0.05,
                max_lag_days=90, min_peak_distance_days=14,
            )
            if "error" not in _lag_ts_mini:
                _cts = _lag_ts_mini["cases_ts"]
                _dts = _lag_ts_mini["deaths_ts"]
                _cp  = _lag_ts_mini["case_peaks"]

                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    x=_cts["Date"], y=_cts["Per100k MA"],
                    name="Cases /100k (7d MA)", mode="lines",
                    line=dict(color=NATIONAL_COLOR, width=1.8),
                    hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
                ))
                fig_mini.add_trace(go.Scatter(
                    x=_dts["Date"], y=_dts["Per100k MA"],
                    name="Deaths /100k (7d MA)", mode="lines",
                    line=dict(color="#c41e3a", width=1.8), yaxis="y2",
                    hovertemplate="%{x|%b %Y}: %{y:.3f}<extra></extra>",
                ))
                if _cp:
                    fig_mini.add_trace(go.Scatter(
                        x=[p["peak_date"] for p in _cp],
                        y=[p["peak_value"] for p in _cp],
                        name="Case Peaks", mode="markers",
                        marker=dict(color=COUNTY_COLOR, size=10, symbol="diamond",
                                    line=dict(color="white", width=1)),
                        hovertemplate="Peak: %{x|%Y-%m-%d}<br>%{y:.2f} /100k<extra></extra>",
                    ))
                fig_mini.update_layout(
                    height=260, margin=dict(t=10, b=40, l=50, r=50),
                    template="plotly_white", hovermode="x unified",
                    legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
                    yaxis=dict(
                        title=dict(text="Cases /100k", font=dict(color=NATIONAL_COLOR, size=10)),
                        tickfont=dict(color=NATIONAL_COLOR, size=9),
                        showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                    ),
                    yaxis2=dict(
                        title=dict(text="Deaths /100k", font=dict(color="#c41e3a", size=10)),
                        tickfont=dict(color="#c41e3a", size=9),
                        overlaying="y", side="right", showgrid=False,
                    ),
                    xaxis=dict(showgrid=False),
                    font=dict(family="sans-serif", size=10),
                )
                st.plotly_chart(fig_mini, use_container_width=True)
    else:
        st.info("Wave analysis unavailable for this county.")

    # SECTION 3 — TIME LAG SUMMARY

    st.markdown(
        '<div class="sub-section-header"><h3>3 — Time Lag Summary</h3>'
        '<p>Default parameters: 7-day MA, case prominence 1.0, death prominence 0.05, max lag 90 days.</p></div>',
        unsafe_allow_html=True,
    )

    _lag_results = analyze_county_lag(
        cases_df, deaths_df, population_df, county_name, state,
        ma_window=7, case_prominence=1.0, death_prominence=0.05,
        max_lag_days=90, min_peak_distance_days=14,
    )
    _lag_summary = summarize_lag_results(_lag_results)

    if "error" not in _lag_results:
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        with l1:
            render_metric_card("Avg Lag", f"{_lag_summary['avg_lag']:.1f} d" if pd.notna(_lag_summary["avg_lag"]) else "N/A")
        with l2:
            render_metric_card("Median Lag", f"{_lag_summary['median_lag']:.1f} d" if pd.notna(_lag_summary["median_lag"]) else "N/A")
        with l3:
            _lr = (f"{int(_lag_summary['min_lag'])}–{int(_lag_summary['max_lag'])} d"
                   if pd.notna(_lag_summary["min_lag"]) and pd.notna(_lag_summary["max_lag"])
                   else "N/A")
            render_metric_card("Lag Range", _lr)
        with l4:
            render_metric_card("Matched Pairs", str(_lag_summary["n_matched"]))
        with l5:
            _sr = _lag_summary.get("mean_severity_ratio")
            render_metric_card("Mean Severity Ratio", f"{_sr:.4f}" if pd.notna(_sr) else "N/A")
        with l6:
            render_metric_card("Largest Case Peak", f"{_fmt(_lag_summary['largest_case_peak'], '.2f')} /100k")
    else:
        st.info("Time lag analysis unavailable for this county.")

    # SECTION 4 — HEALTHCARE CAPACITY

    st.markdown(
        '<div class="sub-section-header"><h3>4 — Healthcare Capacity</h3>'
        '<p>Source: HRSA Area Health Resources Files (AHRF).</p></div>',
        unsafe_allow_html=True,
    )

    h1, h2, h3, h4, h5, h6, h7 = st.columns(7)
    with h1: render_metric_card("PCP /100k",           _fmt(_v("pcp_per_100k"),          ".1f"))
    with h2: render_metric_card("Active MDs /100k",    _fmt(_v("total_md_per_100k"),      ".1f"))
    with h3: render_metric_card("Hospital Beds /100k", _fmt(_v("hospital_beds_per_100k"), ".1f"))
    with h4: render_metric_card("ICU Beds /100k",      _fmt(_v("icu_beds_per_100k"),      ".1f"))
    with h5: render_metric_card("SNF Beds /100k",      _fmt(_v("snf_beds_per_100k"),      ".1f"))
    with h6:
        _hpsa = _v("hpsa_primary_care", default=None)
        if _hpsa is None or pd.isna(_hpsa):
            _hpsa_label = "N/A"
        else:
            try:
                _hpsa_label = "Yes" if (bool(int(float(_hpsa))) if str(_hpsa).replace('.','',1).isdigit() else bool(_hpsa)) else "No"
            except Exception:
                _hpsa_label = str(_hpsa)
        render_metric_card("HPSA Designation", _hpsa_label)
    with h7:
        render_metric_card("Critical Access Hospitals", _fmt_int(_v("critical_access_hospitals")))

    # SECTION 5 — SOCIOECONOMIC FACTORS

    st.markdown(
        '<div class="sub-section-header"><h3>5 — Socioeconomic Factors</h3>'
        '<p>Source: HRSA Area Health Resources Files (AHRF).</p></div>',
        unsafe_allow_html=True,
    )

    se1, se2, se3, se4, se5, se6 = st.columns(6)
    with se1:
        _mfi = _v("median_family_income")
        render_metric_card("Median Family Income", f"${_fmt_int(_mfi)}" if pd.notna(_mfi) else "N/A")
    with se2:
        _pci = _v("per_capita_income")
        render_metric_card("Per Capita Income", f"${_fmt_int(_pci)}" if pd.notna(_pci) else "N/A")
    with se3:
        render_metric_card("Unemployment Rate", f"{_fmt(_v('unemployment_rate'), '.1f')}%" if pd.notna(_v("unemployment_rate")) else "N/A")
    with se4:
        render_metric_card("Child Poverty Rate", f"{_fmt(_v('child_poverty_pct'), '.1f')}%" if pd.notna(_v("child_poverty_pct")) else "N/A")
    with se5:
        render_metric_card("% Without HS Diploma", f"{_fmt(_v('pct_no_hs_diploma'), '.1f')}%" if pd.notna(_v("pct_no_hs_diploma")) else "N/A")
    with se6:
        render_metric_card("% College Degree (4yr)", f"{_fmt(_v('pct_college_4yr'), '.1f')}%" if pd.notna(_v("pct_college_4yr")) else "N/A")

    # SECTION 6 — VACCINATION STATUS

    st.markdown(
        '<div class="sub-section-header"><h3>6 — Vaccination Status</h3>'
        '<p>Source: CDC COVID-19 Vaccinations in the United States, County. '
        'Values represent the most recent available county-level snapshot (through May 2023).</p></div>',
        unsafe_allow_html=True,
    )

    _vax_complete      = _v("vax_complete_pct")
    _vax_dose1         = _v("vax_dose1_pct")
    _vax_booster       = _v("vax_booster_pct")
    _vax_65plus        = _v("vax_complete_65plus_pct")
    _vax_last_date     = _v("vax_last_date")

    vax_avail = pd.notna(_vax_complete) or pd.notna(_vax_dose1)

    if vax_avail:
        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        with vc1:
            render_metric_card(
                "Fully Vaccinated",
                f"{_fmt(_vax_complete, '.1f')}%" if pd.notna(_vax_complete) else "N/A",
            )
        with vc2:
            render_metric_card(
                "At Least 1 Dose",
                f"{_fmt(_vax_dose1, '.1f')}%" if pd.notna(_vax_dose1) else "N/A",
            )
        with vc3:
            render_metric_card(
                "Booster Rate",
                f"{_fmt(_vax_booster, '.1f')}%" if pd.notna(_vax_booster) else "N/A",
            )
        with vc4:
            render_metric_card(
                "65+ Fully Vacc.",
                f"{_fmt(_vax_65plus, '.1f')}%" if pd.notna(_vax_65plus) else "N/A",
            )
        with vc5:
            _last_str = str(_vax_last_date)[:10] if pd.notna(_vax_last_date) else "N/A"
            render_metric_card("Data As Of", _last_str)

        # National comparison for vaccination
        _nat_vax_complete = nat_med.get("vax_complete_pct", np.nan)
        _nat_vax_dose1    = nat_med.get("vax_dose1_pct", np.nan)
        _nat_vax_booster  = nat_med.get("vax_booster_pct", np.nan)
        if pd.notna(_nat_vax_complete):
            _vdiff = _vax_complete - _nat_vax_complete if pd.notna(_vax_complete) else np.nan
            if pd.notna(_vdiff):
                _vdiff_color = "#059669" if _vdiff > 0 else "#c41e3a"
                st.markdown(
                    f"Fully vaccinated rate: **{_fmt(_vax_complete, '.1f')}%** county vs "
                    f"**{_fmt(_nat_vax_complete, '.1f')}%** national median "
                    f"<span style='color:{_vdiff_color}; font-weight:600'>({_vdiff:+.1f}%)</span>",
                    unsafe_allow_html=True,
                )

        # Vaccination rollout timeline (if time-series data is available)
        county_fips_val = str(_v("countyFIPS", default="")).zfill(5)
        if vax_ts_df is not None and not vax_ts_df.empty and county_fips_val != "00000":
            _cv_ts = get_county_vax_timeseries(vax_ts_df, county_fips_val)
            if not _cv_ts.empty and "vax_complete_pct" in _cv_ts.columns:
                with st.expander("Vaccination Rollout Timeline", expanded=False):
                    _fig_vax = go.Figure()
                    if "vax_dose1_pct" in _cv_ts.columns:
                        _fig_vax.add_trace(go.Scatter(
                            x=_cv_ts["Date"], y=_cv_ts["vax_dose1_pct"],
                            name="At Least 1 Dose",
                            mode="lines",
                            line=dict(color=NATIONAL_COLOR, width=2, dash="dash"),
                            hovertemplate="Date: %{x|%Y-%m-%d}<br>1+ Dose: %{y:.1f}%<extra></extra>",
                        ))
                    _fig_vax.add_trace(go.Scatter(
                        x=_cv_ts["Date"], y=_cv_ts["vax_complete_pct"],
                        name="Fully Vaccinated",
                        mode="lines",
                        line=dict(color=COUNTY_COLOR, width=2.5),
                        fill="tozeroy", fillcolor="rgba(242,106,33,0.12)",
                        hovertemplate="Date: %{x|%Y-%m-%d}<br>Fully Vacc: %{y:.1f}%<extra></extra>",
                    ))
                    if "vax_booster_pct" in _cv_ts.columns:
                        _fig_vax.add_trace(go.Scatter(
                            x=_cv_ts["Date"], y=_cv_ts["vax_booster_pct"],
                            name="Booster Rate",
                            mode="lines",
                            line=dict(color=NATION_LINE_COLOR, width=2),
                            hovertemplate="Date: %{x|%Y-%m-%d}<br>Booster: %{y:.1f}%<extra></extra>",
                        ))
                    _fig_vax.update_layout(
                        title=f"<b>Vaccination Rollout — {location}</b>",
                        xaxis_title="Date",
                        yaxis=dict(title="% of Population", range=[0, 100]),
                        height=380,
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        font=dict(family="sans-serif", size=11),
                    )
                    st.plotly_chart(_fig_vax, use_container_width=True)
    else:
        st.info("Vaccination data not available for this county.")

    # SECTION 7 — COUNTY VS NATIONAL & PEER MEDIANS

    # Structural peers are computed here so the comparison table can show a
    # peer median column; Section 9 reuses the same result for its display.
    _fips_str5 = str(fips).zfill(5) if fips != "—" else None
    _peers = (
        find_similar_counties(master_county_df, _fips_str5, n=10)
        if (_fips_str5 and master_county_df is not None and not master_county_df.empty)
        else pd.DataFrame()
    )
    _peer_pool = (
        master_county_df[master_county_df["countyFIPS"].isin(set(_peers["countyFIPS"]))]
        if not _peers.empty else pd.DataFrame()
    )

    def _peer_med(col):
        if _peer_pool.empty or col not in _peer_pool.columns:
            return np.nan
        _s = pd.to_numeric(_peer_pool[col], errors="coerce").dropna()
        return float(_s.median()) if len(_s) else np.nan

    st.markdown(
        '<div class="sub-section-header"><h3>7 — County vs National &amp; Peer Medians</h3>'
        '<p>National figures are medians across all counties; peer figures are medians '
        'across the ten most structurally similar counties (detailed in Section 9). '
        '"Is this county doing worse than counties like it?" is usually the better question.</p></div>',
        unsafe_allow_html=True,
    )

    _cmp_rows = []
    _cmp_defs = [
        ("Cases per 100k",           "cases_per_100k",          cases_100k,              ".1f"),
        ("Deaths per 100k",          "deaths_per_100k",         deaths_100k,             ".2f"),
        ("Case Fatality Rate (%)",   "case_fatality_rate",      cfr,                     ".2f"),
        ("Fully Vaccinated (%)",     "vax_complete_pct",        _v("vax_complete_pct"),  ".1f"),
        ("At Least 1 Dose (%)",      "vax_dose1_pct",           _v("vax_dose1_pct"),     ".1f"),
        ("Booster Rate (%)",         "vax_booster_pct",         _v("vax_booster_pct"),   ".1f"),
        ("65+ Vaccination Rate (%)", "vax_complete_65plus_pct", _v("vax_complete_65plus_pct"), ".1f"),
        ("PCP per 100k",             "pcp_per_100k",            _v("pcp_per_100k"),      ".1f"),
        ("Hospital Beds per 100k",   "hospital_beds_per_100k",  _v("hospital_beds_per_100k"), ".1f"),
        ("ICU Beds per 100k",        "icu_beds_per_100k",       _v("icu_beds_per_100k"), ".1f"),
        ("Unemployment Rate (%)",    "unemployment_rate",       _v("unemployment_rate"), ".1f"),
        ("Child Poverty Rate (%)",   "child_poverty_pct",       _v("child_poverty_pct"), ".1f"),
        ("Median Family Income ($)", "median_family_income",    _v("median_family_income"), ",.0f"),
        ("% Without HS Diploma",     "pct_no_hs_diploma",       _v("pct_no_hs_diploma"), ".1f"),
        ("% 65+ Population",         "pct_pop_65plus",          _v("pct_pop_65plus"),    ".1f"),
        ("Median Age",               "median_age",              _v("median_age"),        ".1f"),
    ]

    for label, col, county_val, fmt in _cmp_defs:
        nat_val  = nat_med.get(col, np.nan)
        peer_val = _peer_med(col)
        if pd.isna(county_val) and pd.isna(nat_val):
            continue
        county_str = f"{county_val:{fmt}}" if pd.notna(county_val) else "N/A"
        nat_str    = f"{nat_val:{fmt}}"    if pd.notna(nat_val)    else "N/A"
        peer_str   = f"{peer_val:{fmt}}"   if pd.notna(peer_val)   else "—"

        if pd.notna(county_val) and pd.notna(nat_val) and nat_val != 0:
            pct_diff = ((county_val - nat_val) / abs(nat_val)) * 100
            diff_str = f"{pct_diff:+.1f}%"
            # For rates where lower is better (mortality, poverty, unemployment, no-hs-diploma), flip the arrow
            lower_is_better = col in {
                "deaths_per_100k", "case_fatality_rate", "unemployment_rate",
                "child_poverty_pct", "pct_no_hs_diploma",
                # Note: higher vaccination = BETTER, so higher is NOT lower-is-better
                # (vaccination columns intentionally omitted from this set)
            }
            if lower_is_better:
                status = "▼ Below avg" if county_val < nat_val else ("▲ Above avg" if county_val > nat_val else "At avg")
            else:
                status = "▲ Above avg" if county_val > nat_val else ("▼ Below avg" if county_val < nat_val else "At avg")
        else:
            diff_str = "—"
            status   = "—"

        _cmp_rows.append({
            "Metric":           label,
            "This County":      county_str,
            "Peer Median":      peer_str,
            "National Median":  nat_str,
            "Difference":       diff_str,
            "vs National":      status,
        })

    if _cmp_rows:
        _cmp_df = pd.DataFrame(_cmp_rows)

        def _style_vs(v):
            if "Above avg" in str(v):
                return "color: #059669; font-weight: 600"
            if "Below avg" in str(v):
                return "color: #c41e3a; font-weight: 600"
            return ""

        st.dataframe(
            _cmp_df.style.map(_style_vs, subset=["vs National"]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "▲ Above avg = county value exceeds the national median (arrows compare against "
            "the national column). For mortality, poverty, and unemployment, above average "
            "is unfavorable. Peer Median = the ten structurally similar counties from "
            "Section 9 — a fairer benchmark than the whole nation."
        )
    else:
        st.info("Insufficient data for national comparison.")

    # SECTION 8 — RESEARCH SNAPSHOTS

    st.markdown(
        '<div class="sub-section-header"><h3>8 — Research Snapshots</h3>'
        '<p>Rule-based summary derived from the metrics above. Not a causal inference.</p></div>',
        unsafe_allow_html=True,
    )

    _findings = []

    def _above(col, county_v):
        nat = nat_med.get(col, np.nan)
        if pd.isna(county_v) or pd.isna(nat) or nat == 0:
            return None
        return county_v > nat

    # COVID burden
    _c100 = _above("cases_per_100k", cases_100k)
    if _c100 is not None:
        _findings.append(
            f"COVID case burden was **{'above' if _c100 else 'below'}-average** "
            f"({_fmt(cases_100k, '.1f')} vs national median {_fmt(nat_med.get('cases_per_100k'), '.1f')} per 100k)."
        )

    _d100 = _above("deaths_per_100k", deaths_100k)
    if _d100 is not None:
        _findings.append(
            f"COVID mortality was **{'above' if _d100 else 'below'}-average** "
            f"({_fmt(deaths_100k, '.2f')} vs {_fmt(nat_med.get('deaths_per_100k'), '.2f')} deaths per 100k)."
        )

    # Healthcare
    _pcp = _above("pcp_per_100k", _v("pcp_per_100k"))
    if _pcp is not None:
        _findings.append(
            f"Primary care physician density was **{'above' if _pcp else 'below'}-average** "
            f"({_fmt(_v('pcp_per_100k'), '.1f')} vs {_fmt(nat_med.get('pcp_per_100k'), '.1f')} per 100k)."
        )

    _beds = _above("hospital_beds_per_100k", _v("hospital_beds_per_100k"))
    if _beds is not None:
        _findings.append(
            f"Hospital bed capacity was **{'above' if _beds else 'below'}-average** "
            f"({_fmt(_v('hospital_beds_per_100k'), '.1f')} vs {_fmt(nat_med.get('hospital_beds_per_100k'), '.1f')} per 100k)."
        )

    # Socioeconomic
    _pov = _above("child_poverty_pct", _v("child_poverty_pct"))
    if _pov is not None:
        _findings.append(
            f"Child poverty rate was **{'above' if _pov else 'below'}-average** "
            f"({_fmt(_v('child_poverty_pct'), '.1f')}% vs {_fmt(nat_med.get('child_poverty_pct'), '.1f')}% national median)."
        )

    _unemp = _above("unemployment_rate", _v("unemployment_rate"))
    if _unemp is not None:
        _findings.append(
            f"Unemployment was **{'above' if _unemp else 'below'}-average** "
            f"({_fmt(_v('unemployment_rate'), '.1f')}% vs {_fmt(nat_med.get('unemployment_rate'), '.1f')}%)."
        )

    # Rural/urban context
    if rucc_group in ("Metro", "Nonmetro"):
        _findings.append(
            f"This is a **{rucc_group.lower()} county** "
            f"(RUCC {int(rucc) if pd.notna(rucc) else '—'})."
        )

    # Lag summary
    if pd.notna(_lag_summary["avg_lag"]) and _lag_summary["n_matched"] > 0:
        _findings.append(
            f"Across {_lag_summary['n_matched']} matched wave pair(s), the average lag from case peak to death peak "
            f"was **{_lag_summary['avg_lag']:.1f} days** (median {_lag_summary['median_lag']:.1f} d)."
        )

    # Vaccination context
    _vax_c = _v("vax_complete_pct")
    _nat_vax_c = nat_med.get("vax_complete_pct", np.nan)
    if pd.notna(_vax_c) and pd.notna(_nat_vax_c):
        _vax_above = _vax_c > _nat_vax_c
        _findings.append(
            f"Final vaccination rate was **{'above' if _vax_above else 'below'}-average** "
            f"({_fmt(_vax_c, '.1f')}% fully vaccinated vs {_fmt(_nat_vax_c, '.1f')}% national median)."
        )

    # Wave count
    if "error" not in _wave_results:
        _n_waves = _wave_results["cases"]["number_of_waves"]
        if _n_waves > 0:
            _findings.append(
                f"**{_n_waves} distinct case wave(s)** were detected with the default parameters."
            )

    if _findings:
        for i, finding in enumerate(_findings):
            st.markdown(f"- {finding}")
    else:
        st.info("Insufficient data to generate research snapshots for this county.")

    # SECTION 9 — SIMILAR COUNTIES

    st.markdown(
        '<div class="sub-section-header"><h3>9 — Counties Like This One</h3>'
        '<p>The ten most structurally similar counties — matched on population, density, '
        'income, age, education, healthcare access, and rurality — and how their COVID '
        'outcomes compare. Similar inputs, different outcomes: a natural starting point '
        'for research questions.</p></div>',
        unsafe_allow_html=True,
    )

    # _peers was computed just before Section 7 so the comparison table could
    # include the peer-median column.
    if not _peers.empty:
        _peer_cols = {
            "County Name":         "County",
            "State":               "State",
            "similarity_distance": "Distance",
            "cases_per_100k":      "Cases /100k",
            "deaths_per_100k":     "Deaths /100k",
            "case_fatality_rate":  "CFR (%)",
            "vax_complete_pct":    "Fully Vacc. (%)",
        }
        _peer_disp = _peers[[c for c in _peer_cols if c in _peers.columns]].rename(columns=_peer_cols)
        st.dataframe(
            _peer_disp.style.format({
                "Distance":       "{:.2f}",
                "Cases /100k":    "{:,.0f}",
                "Deaths /100k":   "{:.1f}",
                "CFR (%)":        "{:.2f}",
                "Fully Vacc. (%)": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )

        _peer_deaths = pd.to_numeric(_peers.get("deaths_per_100k"), errors="coerce").dropna()
        if len(_peer_deaths) >= 5 and pd.notna(deaths_100k):
            _peer_med = float(_peer_deaths.median())
            _cmp_word = "higher than" if deaths_100k > _peer_med else "lower than"
            st.caption(
                f"COVID mortality here ({_fmt(deaths_100k, '.1f')} deaths/100k) was "
                f"**{_cmp_word}** the median of its ten structural peers "
                f"({_peer_med:.1f}/100k). Distance is Euclidean in standardized "
                "feature space — smaller means more similar."
            )
    else:
        st.info("Not enough complete structural data to find similar counties.")

    # Downloadable one-page report + shareable link
    st.markdown("---")
    _dl_col, _share_col = st.columns([1, 2], vertical_alignment="center")
    with _dl_col:
        _report_html = _build_county_report_html(
            county_name=county_name, state=state, fips=fips,
            population=population, rucc_group=rucc_group,
            total_cases=total_cases, total_deaths=total_deaths,
            cases_100k=cases_100k, deaths_100k=deaths_100k, cfr=cfr,
            wave_results=_wave_results, lag_summary=_lag_summary,
            values=_v, nat_med=nat_med,
            data_through=dates[-1] if dates else "—",
        )
        st.download_button(
            "Download county report (HTML)",
            data=_report_html.encode("utf-8"),
            file_name=f"county_report_{str(fips)}_{county_name.replace(' ', '_')}.html",
            mime="text/html",
            key="overview_report_download",
            use_container_width=True,
        )
    with _share_col:
        st.caption(
            "Share this exact view: copy the browser URL — it now includes "
            f"`?county={location}` and will open directly to this county."
        )

def _build_county_report_html(county_name, state, fips, population, rucc_group,
                              total_cases, total_deaths, cases_100k, deaths_100k,
                              cfr, wave_results, lag_summary, values, nat_med,
                              data_through):
    """Assemble a self-contained one-page HTML fact sheet for download."""
    def fmt(v, spec=",.1f", fallback="N/A"):
        try:
            return f"{float(v):{spec}}" if pd.notna(v) else fallback
        except (TypeError, ValueError):
            return fallback

    def row(label, county_val, nat_key=None, spec=",.1f"):
        nat = fmt(nat_med.get(nat_key), spec) if nat_key else "—"
        return (f"<tr><td>{label}</td><td>{fmt(county_val, spec)}</td>"
                f"<td>{nat}</td></tr>")

    n_waves = "N/A"
    peak_wave = "N/A"
    if isinstance(wave_results, dict) and "error" not in wave_results:
        n_waves = wave_results["cases"]["number_of_waves"]
        pd_date = wave_results["cases"]["date_of_peak_wave"]
        peak_wave = str(pd_date)[:10] if pd_date else "N/A"

    avg_lag = fmt(lag_summary.get("avg_lag"), ".1f") if lag_summary else "N/A"

    rows = "".join([
        row("Total cases", total_cases, spec=",.0f"),
        row("Total deaths", total_deaths, spec=",.0f"),
        row("Cases per 100k", cases_100k, "cases_per_100k"),
        row("Deaths per 100k", deaths_100k, "deaths_per_100k", ".2f"),
        row("Case fatality rate (%)", cfr, "case_fatality_rate", ".2f"),
        row("Fully vaccinated (%)", values("vax_complete_pct"), "vax_complete_pct"),
        row("Primary care physicians /100k", values("pcp_per_100k"), "pcp_per_100k"),
        row("Hospital beds /100k", values("hospital_beds_per_100k"), "hospital_beds_per_100k"),
        row("ICU beds /100k", values("icu_beds_per_100k"), "icu_beds_per_100k"),
        row("Median family income ($)", values("median_family_income"), "median_family_income", ",.0f"),
        row("Unemployment rate (%)", values("unemployment_rate"), "unemployment_rate"),
        row("Child poverty rate (%)", values("child_poverty_pct"), "child_poverty_pct"),
        row("% without HS diploma", values("pct_no_hs_diploma"), "pct_no_hs_diploma"),
        row("% population 65+", values("pct_pop_65plus"), "pct_pop_65plus"),
        row("Population density /sq mi", values("pop_density_per_sqmi"), "pop_density_per_sqmi"),
    ])

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{county_name}, {state} — COVID-19 County Report</title>
<style>
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #0B2341; margin: 2.5rem auto; max-width: 780px; padding: 0 1.5rem; }}
h1 {{ font-size: 1.6rem; margin: 0 0 0.2rem 0; }}
.meta {{ color: #64707F; font-size: 0.85rem; margin-bottom: 1.5rem; }}
.badges span {{ display: inline-block; background: #F26A21; color: white; border-radius: 12px; padding: 2px 12px; font-size: 0.75rem; font-weight: 700; margin-right: 6px; }}
table {{ width: 100%; border-collapse: collapse; margin: 1.25rem 0; font-size: 0.9rem; }}
th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #ECF0F5; }}
th {{ background: #0B2341; color: white; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }}
.summary {{ background: #F3F6FA; border-left: 3px solid #F26A21; padding: 0.9rem 1.2rem; font-size: 0.88rem; line-height: 1.6; }}
.foot {{ color: #97A1AF; font-size: 0.72rem; margin-top: 2rem; border-top: 1px solid #ECF0F5; padding-top: 0.75rem; }}
</style></head><body>
<h1>{county_name}, {state}</h1>
<div class="meta">FIPS {fips} &middot; Population {fmt(population, ',.0f')} &middot; Data through {data_through}</div>
<div class="badges"><span>{rucc_group}</span></div>
<div class="summary">
Detected case waves (Standard sensitivity): <strong>{n_waves}</strong> &middot;
Peak wave date: <strong>{peak_wave}</strong> &middot;
Average case-to-death lag: <strong>{avg_lag} days</strong>
</div>
<table>
<tr><th>Metric</th><th>This county</th><th>National median</th></tr>
{rows}
</table>
<p class="foot">Generated by the COVID-19 County Outcomes Analysis Platform, Gettysburg College.
Sources: USAFacts, HRSA Area Health Resources Files, CDC county vaccination data.
County-level (ecological) statistics — not individual-level or causal estimates.</p>
</body></html>"""

def _render_lag_chart(location_label, results, summary, lag_ma_window):
    """
    Render the dual-axis cases/deaths chart with matched peaks.

    Presentation notes: each matched pair is shown as a translucent band
    spanning case peak → death peak rather than as vertical lines with
    top-of-chart lag brackets — with many pairs, the old per-pair overlays
    (2 dashed lines + connector + label each) collided with one another,
    the legend, and the title. Lag labels are drawn only when few pairs
    exist; hover and the pairs table always carry the exact values.
    """
    cases_ts    = results["cases_ts"]
    deaths_ts   = results["deaths_ts"]
    case_peaks  = results["case_peaks"]
    death_peaks = results["death_peaks"]
    matches     = results["matches"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cases_ts["Date"], y=cases_ts["Per100k MA"],
        name="New Cases /100k", mode="lines",
        line=dict(color=NATIONAL_COLOR, width=2.2), yaxis="y",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Cases/100k: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=deaths_ts["Date"], y=deaths_ts["Per100k MA"],
        name="New Deaths /100k", mode="lines",
        line=dict(color="rgba(196,30,58,0.75)", width=1.5), yaxis="y2",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Deaths/100k: %{y:.3f}<extra></extra>",
    ))

    matched_case_dates  = set(matches["case_peak_date"])  if not matches.empty else set()
    matched_death_dates = set(matches["death_peak_date"]) if not matches.empty else set()
    case_lag_lookup     = dict(zip(matches["case_peak_date"],  matches["lag_days"])) if not matches.empty else {}
    death_lag_lookup    = dict(zip(matches["death_peak_date"], matches["lag_days"])) if not matches.empty else {}

    if case_peaks:
        cp_dates   = [p["peak_date"]  for p in case_peaks]
        cp_values  = [p["peak_value"] for p in case_peaks]
        cp_matched = [d in matched_case_dates for d in cp_dates]
        cp_lags    = [case_lag_lookup.get(d) for d in cp_dates]
        cp_custom  = [[d.strftime("%Y-%m-%d"), v, (f"{lag} days" if lag is not None else "Unmatched")]
                      for d, v, lag in zip(cp_dates, cp_values, cp_lags)]
        fig.add_trace(go.Scatter(
            x=cp_dates, y=cp_values, name="Case Peaks", mode="markers",
            marker=dict(color=["#1a3d6d" if m else "#9fb3d1" for m in cp_matched],
                        size=[10 if m else 7 for m in cp_matched],
                        symbol="diamond", line=dict(color="white", width=1)),
            yaxis="y", customdata=cp_custom,
            hovertemplate=(
                "<b>Case Peak</b><br>Peak Date: %{customdata[0]}<br>"
                "Peak Value: %{customdata[1]:.2f} /100k<br>"
                "Lag to Death Peak: %{customdata[2]}<extra></extra>"
            ),
        ))

    if death_peaks:
        dp_dates   = [p["peak_date"]  for p in death_peaks]
        dp_values  = [p["peak_value"] for p in death_peaks]
        dp_matched = [d in matched_death_dates for d in dp_dates]
        dp_lags    = [death_lag_lookup.get(d) for d in dp_dates]
        dp_custom  = [[d.strftime("%Y-%m-%d"), v, (f"{lag} days" if lag is not None else "Unmatched")]
                      for d, v, lag in zip(dp_dates, dp_values, dp_lags)]
        fig.add_trace(go.Scatter(
            x=dp_dates, y=dp_values, name="Death Peaks", mode="markers",
            marker=dict(color=["#7a0f1f" if m else "#e3a8b1" for m in dp_matched],
                        size=[10 if m else 7 for m in dp_matched],
                        symbol="star", line=dict(color="white", width=1)),
            yaxis="y2", customdata=dp_custom,
            hovertemplate=(
                "<b>Death Peak</b><br>Peak Date: %{customdata[0]}<br>"
                "Peak Value: %{customdata[1]:.3f} /100k<br>"
                "Lag from Case Peak: %{customdata[2]}<extra></extra>"
            ),
        ))

    # Matched pairs: one translucent band per pair spanning case peak → death
    # peak. Lag labels are drawn only when they can't collide (few pairs).
    shapes, annotations = [], []
    show_lag_labels = len(matches) <= 8
    for _, row in matches.iterrows():
        c_date = row["case_peak_date"]
        d_date = row["death_peak_date"]
        lag    = int(row["lag_days"])
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=c_date, x1=d_date, y0=0, y1=1,
            fillcolor="rgba(242,106,33,0.07)", line_width=0, layer="below",
        ))
        if show_lag_labels:
            mid_date = c_date + (d_date - c_date) / 2
            annotations.append(dict(
                x=mid_date, y=1.02, xref="x", yref="paper",
                text=f"{lag}d", showarrow=False,
                font=dict(size=10, color="#64707F"),
            ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>New Cases vs New Deaths per 100k — {location_label}</b>"
                f"<br><sub>{lag_ma_window}-day MA · {summary['n_matched']} matched peak pair(s) · "
                "shaded bands span case peak → death peak</sub>"
            ),
            y=0.97, yanchor="top",
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(title=dict(text="New Cases per 100k",   font=dict(color=NATIONAL_COLOR)),
                   side="left",  tickfont=dict(color=NATIONAL_COLOR),
                   showgrid=True, gridcolor="rgba(200,200,200,0.3)", rangemode="tozero"),
        yaxis2=dict(title=dict(text="New Deaths per 100k", font=dict(color="#c41e3a")),
                    side="right", overlaying="y", tickfont=dict(color="#c41e3a"),
                    showgrid=False, rangemode="tozero"),
        shapes=shapes, annotations=annotations,
        hovermode="closest", height=640,
        margin=dict(t=118, b=55, l=60, r=60),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.9)", font=dict(size=10)),
        font=dict(family="sans-serif", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)
    if not show_lag_labels and not matches.empty:
        st.caption(
            f"{len(matches)} matched pairs — individual lag labels are hidden at this "
            "density to keep the chart readable. Hover any peak marker for its exact "
            "lag, or see the full list in the pairs table below."
        )

def _render_lag_summary_metrics(summary, population):
    """Render the KPI metric row for a single county's lag results."""
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1: st.metric("Avg Lag",    f"{summary['avg_lag']:.1f} d"    if pd.notna(summary["avg_lag"])    else "N/A")
    with s2: st.metric("Median Lag", f"{summary['median_lag']:.1f} d" if pd.notna(summary["median_lag"]) else "N/A")
    with s3:
        lag_range = (
            f"{int(summary['min_lag'])}–{int(summary['max_lag'])} d"
            if pd.notna(summary["min_lag"]) and pd.notna(summary["max_lag"]) else "N/A"
        )
        st.metric("Lag Range", lag_range)
    with s4:
        st.metric(
            "Matched Pairs", summary["n_matched"],
            help="Case peaks that were successfully paired with a death peak "
                 "occurring within the lag window after them. Each pair lets us "
                 "measure how many days deaths trailed behind that surge in "
                 "infections. Peaks with no partner within the window stay unmatched.",
        )
    with s5:
        sr = summary.get("mean_severity_ratio")
        st.metric(
            "Mean Severity Ratio",
            f"{sr:.4f}" if pd.notna(sr) else "N/A",
            help="Mean of (death peak /100k) ÷ (case peak /100k) across matched pairs. "
                 "Lower values indicate mortality remained proportionally smaller relative to case burden.",
        )
    with s6: st.metric("Population", f"{int(population):,}")

def _render_lag_pairs_table(matches):
    """Render the matched case→death peak pairs table with severity ratio column."""
    st.subheader(
        "Matched Case → Death Peak Pairs",
        help="Each row is a surge in cases that was followed by a surge in deaths "
             "within the allowed time window. 'Lag' is the number of days between "
             "the two peaks — a plain-language estimate of how long it took for a "
             "wave of infections to translate into deaths in this county.",
    )
    if not matches.empty:
        display = matches.copy()
        display["Severity Ratio"] = (
            display["death_peak_value"] / display["case_peak_value"].replace(0, np.nan)
        )
        display["case_peak_date"]  = display["case_peak_date"].dt.strftime("%Y-%m-%d")
        display["death_peak_date"] = display["death_peak_date"].dt.strftime("%Y-%m-%d")
        display = display.rename(columns={
            "case_peak_date":   "Case Peak Date",
            "death_peak_date":  "Death Peak Date",
            "lag_days":         "Lag (days)",
            "case_peak_value":  "Peak Cases /100k",
            "death_peak_value": "Peak Deaths /100k",
        })
        st.dataframe(
            display[["Case Peak Date", "Death Peak Date", "Lag (days)",
                      "Peak Cases /100k", "Peak Deaths /100k", "Severity Ratio"]]
            .style.format({
                "Peak Cases /100k":  "{:.2f}",
                "Peak Deaths /100k": "{:.3f}",
                "Severity Ratio":    "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info(
            "No case peak was followed by a matching death peak within the configured lag window. "
            "Try increasing **Max Lag Window**, lowering the prominence thresholds, or selecting a different county."
        )

def _render_lag_all_peaks_expander(case_peaks, death_peaks):
    with st.expander("All Detected Peaks (including unmatched)", expanded=False):
        pcol, dcol = st.columns(2)
        with pcol:
            st.markdown("**Case Peaks**")
            if case_peaks:
                cp_df = pd.DataFrame(case_peaks)
                cp_df["peak_date"] = pd.to_datetime(cp_df["peak_date"]).dt.strftime("%Y-%m-%d")
                cp_df = cp_df.rename(columns={
                    "peak_date": "Date", "peak_value": "Peak Cases /100k", "peak_prominence": "Prominence",
                })
                st.dataframe(cp_df.style.format({"Peak Cases /100k": "{:.2f}", "Prominence": "{:.2f}"}),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("No case peaks detected with current settings.")
        with dcol:
            st.markdown("**Death Peaks**")
            if death_peaks:
                dp_df = pd.DataFrame(death_peaks)
                dp_df["peak_date"] = pd.to_datetime(dp_df["peak_date"]).dt.strftime("%Y-%m-%d")
                dp_df = dp_df.rename(columns={
                    "peak_date": "Date", "peak_value": "Peak Deaths /100k", "peak_prominence": "Prominence",
                })
                st.dataframe(dp_df.style.format({"Peak Deaths /100k": "{:.3f}", "Prominence": "{:.3f}"}),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("No death peaks detected with current settings.")

def render_lag_tab(cases_df, deaths_df, population_df, locations) -> None:
    """Epidemiological lag analysis tab."""
    render_section_header(
        "Time Lag Analysis",
        "Deaths don't follow cases immediately — there's a delay. This tool measures how long "
        "after a case surge mortality followed in any county, identifies matched outbreak and "
        "death peaks, and quantifies the lag. A longer lag can signal early intervention; "
        "a shorter one may reflect healthcare system strain.",
    )
    render_learning_aids(
        terms=("lag", "severity_ratio", "prominence", "per_100k"),
        questions=(
            "Find a county whose death peak trailed its case peak by more than 60 days. "
            "What could explain such a long delay?",
            "Compare the **severity ratio** of a 2020 wave against an Omicron-era wave "
            "in the same county. What changed between them?",
            "In County vs County mode, compare an urban and a rural county's lags. "
            "Is there a consistent difference, and which way does it run?",
        ),
    )

    _lp1, _lp2, _lp3 = st.columns([2, 1, 2])
    with _lp1:
        lag_mode = st.selectbox(
            "Analysis Mode",
            ["Single County", "County vs County"],
            key="lag_mode",
            help="Single County: detailed lag analysis for one county.  "
                 "County vs County: side-by-side comparison of lag patterns.",
        )
    with _lp2:
        lag_ma_window = st.selectbox(
            "Smoothing",
            [3, 5, 7],
            index=2,
            key="lag_ma_window",
            help="Moving-average window applied to daily per-100k rates before peak detection.",
        )

    if lag_mode == "County vs County":
        _lc1, _lc2 = st.columns(2)
        with _lc1:
            location_lag   = st.selectbox("County A", locations, key="timelag_county")
        with _lc2:
            location_lag_b = st.selectbox("County B", locations, key="timelag_county_b",
                                           index=min(1, len(locations) - 1))
    else:
        _ls1, _ = st.columns([2, 3])
        with _ls1:
            location_lag = st.selectbox("Select County", locations, key="timelag_county")
        location_lag_b = None

    # Detection parameters (advanced — collapsed)
    with st.expander("Detection parameters", expanded=False):
        st.caption(
            "Default values work well for most counties. Adjust if the chart shows "
            "too many or too few peaks."
        )
        _dp1, _dp2, _dp3, _dp4 = st.columns(4)
        with _dp1:
            case_prominence = st.number_input(
                "Case Peak Prominence (/100k)",
                min_value=0.05, max_value=100.0,
                value=1.0, step=0.05, key="case_prominence",
                help="Minimum prominence for a case peak. Lower = more sensitive.",
            )
        with _dp2:
            death_prominence = st.number_input(
                "Death Peak Prominence (/100k)",
                min_value=0.005, max_value=10.0,
                value=0.05, step=0.005, format="%.3f", key="death_prominence",
                help="Minimum prominence for a death peak. Deaths /100k are typically much smaller.",
            )
        with _dp3:
            max_lag_days = st.slider(
                "Max Lag Window (days)", 7, 120, 90, 1, key="max_lag_days",
                help="A death peak can only be matched to a case peak within this many days afterward.",
            )
        with _dp4:
            min_peak_distance = st.slider(
                "Min Peak Spacing (days)", 3, 60, 14, 1, key="min_peak_distance",
                help="Minimum days between consecutive peaks (suppresses noisy double-peaks).",
            )

    # Shared kwargs for analyze_county_lag
    lag_kwargs = dict(
        ma_window=lag_ma_window,
        case_prominence=case_prominence,
        death_prominence=death_prominence,
        max_lag_days=max_lag_days,
        min_peak_distance_days=min_peak_distance,
    )

    county_name, state = extract_county_state(location_lag)
    lag_results = analyze_county_lag(cases_df, deaths_df, population_df, county_name, state, **lag_kwargs)

    if "error" in lag_results:
        st.warning(f"{lag_results['error']} Try a different county.")
        return

    summary = summarize_lag_results(lag_results)

    if lag_mode == "County vs County":
        county_b_name, state_b = extract_county_state(location_lag_b)
        lag_results_b = analyze_county_lag(cases_df, deaths_df, population_df, county_b_name, state_b, **lag_kwargs)
        summary_b = summarize_lag_results(lag_results_b)

        tab_a, tab_b, tab_cmp = st.tabs([location_lag, location_lag_b, "Side-by-Side Comparison"])

        with tab_a:
            if "error" in lag_results:
                st.warning(lag_results["error"])
            else:
                _render_lag_summary_metrics(summary, lag_results["population"])
                st.markdown("---")
                _render_lag_chart(location_lag, lag_results, summary, lag_ma_window)
                _render_lag_pairs_table(lag_results["matches"])
                _render_lag_all_peaks_expander(lag_results["case_peaks"], lag_results["death_peaks"])

        with tab_b:
            if "error" in lag_results_b:
                st.warning(lag_results_b["error"])
            else:
                _render_lag_summary_metrics(summary_b, lag_results_b["population"])
                st.markdown("---")
                _render_lag_chart(location_lag_b, lag_results_b, summary_b, lag_ma_window)
                _render_lag_pairs_table(lag_results_b["matches"])
                _render_lag_all_peaks_expander(lag_results_b["case_peaks"], lag_results_b["death_peaks"])

        with tab_cmp:
            st.markdown("#### Summary Comparison")
            def _fmt_s(v, decimals=1, suffix=""):
                return f"{v:.{decimals}f}{suffix}" if pd.notna(v) else "N/A"

            cmp_rows = [
                ("Avg Lag (days)",          _fmt_s(summary["avg_lag"]),          _fmt_s(summary_b["avg_lag"])),
                ("Median Lag (days)",        _fmt_s(summary["median_lag"]),       _fmt_s(summary_b["median_lag"])),
                ("Min Lag (days)",           _fmt_s(summary["min_lag"], 0),       _fmt_s(summary_b["min_lag"], 0)),
                ("Max Lag (days)",           _fmt_s(summary["max_lag"], 0),       _fmt_s(summary_b["max_lag"], 0)),
                ("Matched Pairs",            str(summary["n_matched"]),           str(summary_b["n_matched"])),
                ("Mean Severity Ratio",      _fmt_s(summary["mean_severity_ratio"], 4),
                                             _fmt_s(summary_b["mean_severity_ratio"], 4)),
                ("Largest Case Peak /100k",  _fmt_s(summary["largest_case_peak"], 2),
                                             _fmt_s(summary_b["largest_case_peak"], 2)),
                ("Largest Death Peak /100k", _fmt_s(summary["largest_death_peak"], 3),
                                             _fmt_s(summary_b["largest_death_peak"], 3)),
                ("Population",              f"{int(lag_results['population']):,}" if pd.notna(lag_results["population"]) else "N/A",
                                            f"{int(lag_results_b['population']):,}" if "error" not in lag_results_b and pd.notna(lag_results_b.get("population", np.nan)) else "N/A"),
            ]
            cmp_df = pd.DataFrame(cmp_rows, columns=["Metric", location_lag, location_lag_b])
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # Overlay chart: smoothed cases /100k for both counties on a shared axis
            st.markdown("#### Cases per 100k — Overlay")
            a_ts = lag_results["cases_ts"]
            b_ts = lag_results_b.get("cases_ts", pd.DataFrame()) if "error" not in lag_results_b else pd.DataFrame()
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=a_ts["Date"], y=a_ts["Per100k MA"],
                name=location_lag, mode="lines",
                line=dict(color=COUNTY_COLOR, width=2.5),
                hovertemplate=f"<b>{location_lag}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}} /100k<extra></extra>",
            ))
            if not b_ts.empty:
                fig_cmp.add_trace(go.Scatter(
                    x=b_ts["Date"], y=b_ts["Per100k MA"],
                    name=location_lag_b, mode="lines",
                    line=dict(color=NATIONAL_COLOR, width=2.5),
                    hovertemplate=f"<b>{location_lag_b}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}} /100k<extra></extra>",
                ))
            fig_cmp.update_layout(
                xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
                yaxis=dict(title=f"New Cases per 100k ({lag_ma_window}-day MA)",
                           showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
                hovermode="x unified", height=500, template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
                font=dict(family="sans-serif", size=11),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

    else:

        _render_lag_summary_metrics(summary, lag_results["population"])
        st.markdown("---")
        _render_lag_chart(location_lag, lag_results, summary, lag_ma_window)
        _render_lag_pairs_table(lag_results["matches"])
        _render_lag_all_peaks_expander(lag_results["case_peaks"], lag_results["death_peaks"])

    with st.expander("How this analysis works", expanded=False):
        st.markdown("""
**Pipeline**

1. **Daily values** — cumulative cases/deaths are converted to daily new counts via differencing. Negative values (data corrections) are clipped to zero.
2. **Per-100k normalization** — daily counts are divided by county population and multiplied by 100,000.
3. **Smoothing** — a rolling moving average (3, 5, or 7 days) reduces day-to-day reporting noise.
   _Note: the first (window−1) smoothed values use fewer than `window` data points (`min_periods=1`). Treat early-pandemic smoothed values with caution._
4. **Peak detection** — `scipy.signal.find_peaks` finds local maxima filtered by prominence and minimum spacing.
5. **Peak matching** — each case peak is matched to the nearest death peak occurring on or after it, within the configured lag window. Each death peak can only be matched once.
6. **Lag** — `lag_days = death_peak_date − case_peak_date`.
7. **Severity ratio** — `death_peak_value / case_peak_value` for each matched pair. Reflects how large the mortality peak was relative to the case surge that preceded it.

**Assumptions**
- Per-100k normalization uses a single, static county population applied across all dates.
- A death peak with no preceding case peak within the lag window is reported as **unmatched**.
- Default prominence thresholds (1.0 cases/100k, 0.05 deaths/100k) are starting points; smaller counties or later pandemic waves may need lower thresholds to detect meaningful peaks.
""")
        st.markdown("**The core formulas**")
        st.latex(r"r_t = \frac{c_t - c_{t-1}}{P} \times 100{,}000")
        st.latex(r"\text{lag} = t_{\text{death peak}} - t_{\text{case peak}} \quad\text{(days)}\qquad"
                 r"\text{severity ratio} = \frac{\text{death peak height}}{\text{case peak height}}")
        st.caption(
            "r = daily rate per 100k, c = cumulative count, P = county population "
            "(static across the period)."
        )

def render_wave_tab(cases_df, deaths_df, transforms, locations, population_df, vax_ts_df=None) -> None:
    """Epidemic wave detection and analysis tab."""
    render_section_header(
        "Wave Analysis",
        "COVID-19 didn't spread in a straight line — it arrived in waves. This tool identifies "
        "distinct outbreak peaks for any county, measures each wave's height, duration, and "
        "total burden, and ranks them by epidemiological significance. Compare how the Initial, "
        "Delta, and Omicron surges played out differently across the country.",
    )
    render_learning_aids(
        terms=("wave", "prominence", "burden", "significance", "moving_average"),
        questions=(
            "Pick a county and switch between **Conservative** and **Sensitive** detection. "
            "Which of the extra waves look real to you, and which look like noise?",
            "Find a county whose largest wave by **burden** is not its tallest peak. "
            "Why can a lower, longer wave matter more?",
            "Using the validation panel, find a county with no detected wave during "
            "the Winter 2020-21 window. Detection artifact, or genuinely spared?",
        ),
    )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        wave_location = st.selectbox("Select County", locations, key="wave_county")
    with r1c2:
        wave_metric = st.selectbox(
            "Metric",
            ["Cases (Raw)", "Cases per 100k", "Deaths (Raw)", "Deaths per 100k"],
            key="wave_metric",
            help=(
                "Cases/Deaths (Raw): daily counts  |  "
                "per 100k: normalised by county population — best for inter-county comparisons"
            ),
        )

    wave_county_name, wave_state = extract_county_state(wave_location)
    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    daily_cases_df  = transforms["daily_cases"]
    daily_deaths_df = transforms["daily_deaths"]

    pop_col = get_population_column(population_df)
    pop_row = population_df[
        (population_df["County Name"] == wave_county_name) &
        (population_df["State"] == wave_state)
    ]
    county_population = (
        float(pd.to_numeric(pop_row.iloc[0][pop_col], errors="coerce"))
        if (not pop_row.empty and pop_col) else float("nan")
    )

    is_cases_metric  = "Cases" in wave_metric
    is_percap_metric = "per 100k" in wave_metric
    source_daily_df  = daily_cases_df if is_cases_metric else daily_deaths_df
    date_cols        = [c for c in source_daily_df.columns if c not in identifier_cols]
    plot_dates       = pd.to_datetime(date_cols)

    county_row = source_daily_df[
        (source_daily_df["County Name"] == wave_county_name) &
        (source_daily_df["State"] == wave_state)
    ]

    if county_row.empty:
        st.warning(f"No data found for {wave_location}. Try selecting a different county.")
        return

    raw_vals = pd.to_numeric(
        county_row.iloc[0, county_row.columns.get_loc(date_cols[0]):], errors="coerce",
    ).values.copy()
    raw_vals = np.clip(raw_vals, 0, None)  # guard negatives

    # Normalise for per-capita metrics
    if is_percap_metric:
        if np.isnan(county_population) or county_population <= 0:
            st.warning(
                f"No valid population data for {wave_location}. "
                "Showing raw counts instead. Per-100k analysis requires a population value."
            )
            analysis_vals = raw_vals.copy()
            is_percap_metric = False          # fall back gracefully
        else:
            analysis_vals = (raw_vals / county_population) * 100_000
    else:
        analysis_vals = raw_vals.copy()

    # Sensitivity + smoothing controls
    dc1, dc2, dc3 = st.columns([2, 2, 1])
    with dc1:
        _SENS_OPTIONS = {
            "Conservative — major national waves only (~3–5)":   "conservative",
            "Standard — major + significant regional surges (~4–8)": "standard",
            "Sensitive — includes smaller local surges (~6–15)": "sensitive",
        }
        _sens_label = st.selectbox(
            "Detection Sensitivity",
            list(_SENS_OPTIONS.keys()),
            index=1,  # "Standard" default
            key="wave_sensitivity",
            help=(
                "Controls how many outbreak events are considered a 'wave'. "
                "Standard is the right choice for most users. "
                "Use Conservative to see only the major national surges "
                "(Initial, Alpha, Delta, Omicron). "
                "Use Sensitive to explore smaller regional events."
            ),
        )
        wave_sensitivity = _SENS_OPTIONS[_sens_label]
    with dc2:
        auto_smooth = st.checkbox(
            "Auto-select smoothing window", value=True, key="wave_auto_smooth",
            help="System selects the optimal moving-average window from the signal variance",
        )
        if auto_smooth:
            wave_ma_window = estimate_optimal_smoothing(analysis_vals)
            st.caption(f"Auto: **{wave_ma_window}-day** moving average")
        else:
            wave_ma_window = st.selectbox(
                "Smoothing Window", [3, 5, 7, 14], index=2, key="wave_ma_window_manual",
                help="Larger window → smoother signal, fewer detected waves",
            )
    with dc3:
        show_raw = st.checkbox("Show raw daily bars", value=True, key="wave_show_raw",
            help="Toggle faint raw-count bars behind the smoothed curve")
        show_national = st.checkbox(
            "Overlay national rate", value=False, key="wave_show_national",
            disabled=not is_percap_metric,
            help="Faint US per-100k curve for context — did this county lead or lag "
                 "the country? (per-100k metrics only)",
        )

    # Advanced controls (collapsed by default)
    with st.expander("Advanced detection controls", expanded=False):
        st.caption(
            "These controls bypass the epidemiological region detector and activate the "
            "legacy prominence-based algorithm. Prominence = minimum height a peak must "
            "stand above its surroundings. Merge window = peaks closer than this are joined."
        )
        adv1, adv2 = st.columns(2)
        with adv1:
            _preset_prom = SENSITIVITY_PRESETS[wave_sensitivity]
            _adv_prom_default = float(
                max(
                    analysis_vals.max() * _preset_prom["prominence_pct"],
                    analysis_vals[analysis_vals > 0].std() * _preset_prom["prominence_floor_iqr_mult"]
                    if (analysis_vals > 0).any() else 1.0,
                    1.0,
                )
            )
            _use_custom_prom = st.checkbox("Override prominence", value=False, key="wave_override_prom")
            if _use_custom_prom:
                wave_prominence_override = st.number_input(
                    "Custom prominence", min_value=0.1,
                    value=float(round(_adv_prom_default, 1)),
                    step=float(max(0.1, _adv_prom_default / 10)),
                    key="wave_prom_custom",
                    help="Minimum prominence for a peak. Lower = more sensitive.",
                )
            else:
                wave_prominence_override = None
                st.caption(f"Preset: **{_adv_prom_default:.1f}** (from {wave_sensitivity})")
        with adv2:
            _use_custom_merge = st.checkbox("Override merge window", value=False, key="wave_override_merge")
            if _use_custom_merge:
                merge_days_override = st.slider(
                    "Merge peaks within (days)", 0, 90,
                    value=SENSITIVITY_PRESETS[wave_sensitivity]["min_merge_days"],
                    step=1, key="wave_merge_custom",
                )
            else:
                merge_days_override = None
                st.caption(f"Preset: **{SENSITIVITY_PRESETS[wave_sensitivity]['min_merge_days']} days**")

    # Resolve effective parameters
    wave_prominence = wave_prominence_override   # None = let sensitivity preset decide
    merge_days      = merge_days_override        # None = let sensitivity preset decide

    st.markdown("---")

    # When an advanced override is active, pass it explicitly and clear sensitivity
    # so the preset doesn't override the user's manual value.
    _eff_sensitivity  = wave_sensitivity if wave_prominence is None else None
    _eff_prominence   = wave_prominence  if wave_prominence is not None else 1000
    _eff_merge        = merge_days       if merge_days      is not None else 0

    try:
        active_metrics = calculate_waves_from_values(
            analysis_vals, plot_dates,
            ma_window=wave_ma_window,
            prominence=_eff_prominence,
            min_merge_days=_eff_merge,
            sensitivity=_eff_sensitivity,
        )
    except Exception as e:
        st.warning(f"Wave detection failed: {e}")
        return

    n_waves   = active_metrics["number_of_waves"]
    wave_list = active_metrics["waves"]
    diag      = active_metrics["diagnostics"]

    # Raw-count detection runs separately so the "Compare Case vs Death Waves"
    # expander always shows counts regardless of which metric is on the main chart.
    try:
        raw_county_results = calculate_waves_for_county(
            cases_df, deaths_df, daily_cases_df, daily_deaths_df,
            wave_county_name, wave_state,
            ma_window=wave_ma_window,
            prominence=_eff_prominence,
            min_merge_days=_eff_merge,
            sensitivity=_eff_sensitivity,
        )
    except Exception:
        raw_county_results = {"error": "unavailable"}

    metric_base    = "Cases" if is_cases_metric else "Deaths"
    if is_percap_metric:
        y_label    = f"Daily {metric_base} per 100k"
        val_fmt    = "{:,.2f}"
        burden_lbl = f"Total {metric_base}/100k·Days"
    else:
        y_label    = f"Daily {metric_base}"
        val_fmt    = "{:,.0f}"
        burden_lbl = f"Total {metric_base} During Wave"

    st.markdown("#### Summary Metrics")
    kw1, kw2, kw3, kw4, kw5, kw6 = st.columns(6)

    with kw1:
        render_wave_metric_card("Waves Detected", n_waves)
    with kw2:
        render_wave_metric_card(
            "Largest Peak",
            round(active_metrics["largest_wave"], 2 if is_percap_metric else 0)
            if active_metrics["largest_wave"] else 0,
        )
    with kw3:
        render_wave_metric_card(
            "Avg Wave Height",
            round(active_metrics["average_wave_height"], 2 if is_percap_metric else 0)
            if active_metrics["average_wave_height"] else 0,
        )
    with kw4:
        render_wave_metric_card(
            "Avg Duration",
            round(active_metrics["average_wave_duration"], 1)
            if active_metrics["average_wave_duration"] else 0,
            suffix="days",
        )
    with kw5:
        peak_date = active_metrics["date_of_peak_wave"]
        render_wave_metric_card("Peak Wave Date", str(peak_date)[:10] if peak_date else "N/A")
    with kw6:
        avg_iw = active_metrics["average_time_between_waves"]
        render_wave_metric_card(
            "Avg Time Between Waves",
            f"{avg_iw:.0f} days" if pd.notna(avg_iw) else "N/A",
        )

    st.markdown("---")

    display_fill = analysis_vals.copy()
    display_fill[np.isnan(display_fill)] = 0.0
    smoothed_arr = np.convolve(display_fill, np.ones(wave_ma_window) / wave_ma_window, mode="same")
    half_w = wave_ma_window // 2
    smoothed_arr[:half_w] = 0.0
    smoothed_arr[len(smoothed_arr) - half_w:] = 0.0

    hover_fmt = ".2f" if is_percap_metric else ",.0f"

    fig_waves = go.Figure()

    if show_raw:
        fig_waves.add_trace(go.Bar(
            x=plot_dates, y=analysis_vals,
            name=f"Daily {metric_base} (raw)",
            marker_color="rgba(42, 82, 152, 0.20)",
            hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>Raw: %{{y:{hover_fmt}}}<extra></extra>",
        ))

    fig_waves.add_trace(go.Scatter(
        x=plot_dates, y=smoothed_arr,
        name=f"{wave_ma_window}-day MA",
        mode="lines",
        line=dict(color=NATIONAL_COLOR, width=2.5),
        hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>MA: %{{y:{hover_fmt}}}<extra></extra>",
    ))

    # National context curve (per-100k metrics only, same smoothing window)
    if is_percap_metric and show_national:
        _nat_daily = national["daily_cases"] if is_cases_metric else national["daily_deaths"]
        _us_pop = pd.to_numeric(
            population_df[population_df["countyFIPS"] != "00000"][pop_col],
            errors="coerce",
        ).clip(lower=0).sum()
        if _us_pop > 0:
            _nat_rate = np.nan_to_num(
                _nat_daily["Value"].values.astype(float) / _us_pop * 100_000
            )
            _nat_smoothed = np.convolve(
                _nat_rate, np.ones(wave_ma_window) / wave_ma_window, mode="same"
            )
            fig_waves.add_trace(go.Scatter(
                x=_nat_daily["Date"], y=_nat_smoothed,
                name="United States", mode="lines",
                line=dict(color="#8A94A3", width=1.6, dash="dot"),
                hovertemplate=f"US: %{{y:{hover_fmt}}}<extra></extra>",
            ))

    # Reporting corrections: dates where the cumulative source series decreased
    # (state revisions/backfills). The daily pipeline clips these to zero; the
    # markers keep them visible instead of silently hidden.
    _corrections = find_data_corrections(
        cases_df if is_cases_metric else deaths_df, wave_county_name, wave_state
    )
    if not _corrections.empty:
        fig_waves.add_trace(go.Scatter(
            x=_corrections["Date"], y=np.zeros(len(_corrections)),
            name="Reporting correction", mode="markers",
            marker=dict(symbol="x-thin", size=9, color="#8A94A3",
                        line=dict(width=1.6, color="#8A94A3")),
            customdata=_corrections["correction"].values,
            hovertemplate=("Reporting correction: %{customdata:,.0f} on "
                           "%{x|%Y-%m-%d}<br>(cumulative total revised downward; "
                           "daily value clipped to 0)<extra></extra>"),
        ))

    WAVE_COLORS = [
        "#F26A21", "#c41e3a", "#153A66", "#5b4fcf",
        "#1a7f5b", "#b5651d", "#6b3a8b", "#1f6b9a",
    ]

    for i, wave in enumerate(wave_list):
        color = WAVE_COLORS[i % len(WAVE_COLORS)]

        # Shaded wave region (start → end)
        fig_waves.add_vrect(
            x0=wave["start_date"], x1=wave["end_date"],
            fillcolor=color, opacity=0.09, layer="below", line_width=0,
        )

        # Start boundary line
        fig_waves.add_vline(
            x=wave["start_date"],
            line_dash="dot", line_color=color, line_width=1, opacity=0.5,
        )
        # End boundary line
        fig_waves.add_vline(
            x=wave["end_date"],
            line_dash="dot", line_color=color, line_width=1, opacity=0.5,
        )
        # Peak line
        fig_waves.add_vline(
            x=wave["peak_date"],
            line_dash="solid", line_color=color, line_width=1.5, opacity=0.7,
        )

        # Peak marker + label
        pv = wave["peak_value"]
        burden = wave.get("wave_burden", float("nan"))
        fig_waves.add_trace(go.Scatter(
            x=[wave["peak_date"]],
            y=[pv],
            mode="markers+text",
            marker=dict(color=color, size=13, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            text=[f"W{wave['wave_number']}"],
            textposition="top center",
            textfont=dict(size=11, color=color, family="sans-serif"),
            name=f"Wave {wave['wave_number']} peak",
            hovertemplate=(
                f"<b>Wave {wave['wave_number']}</b><br>"
                f"Peak date:  {str(wave['peak_date'])[:10]}<br>"
                f"Start date: {str(wave['start_date'])[:10]}<br>"
                f"End date:   {str(wave['end_date'])[:10]}<br>"
                f"Peak value: {pv:{hover_fmt}}<br>"
                f"Duration:   {wave['duration_days']} days<br>"
                f"Burden:     {burden:{hover_fmt}}<extra></extra>"
            ),
            # Peaks are labelled on-chart (W1, W2, …); per-wave legend entries
            # only duplicated those labels and overflowed the legend into the
            # title whenever many waves were detected.
            showlegend=False,
        ))

        # Start / end triangle markers at y=0
        fig_waves.add_trace(go.Scatter(
            x=[wave["start_date"], wave["end_date"]],
            y=[0, 0],
            mode="markers",
            marker=dict(
                color=color, size=8,
                symbol=["triangle-right", "triangle-left"],
                line=dict(color="white", width=1),
            ),
            name=f"W{wave['wave_number']} bounds",
            hovertemplate=(
                "<b>Wave boundary</b><br>"
                "Date: %{x|%Y-%m-%d}<extra></extra>"
            ),
            showlegend=False,
        ))

    auto_tag   = "auto" if auto_smooth else "manual"
    _sens_disp = wave_sensitivity.capitalize()
    if wave_prominence is not None:
        _det_tag = f"custom prominence {wave_prominence:{'.1f' if is_percap_metric else ',.0f'}}"
    else:
        _det_tag = f"{_sens_disp} sensitivity"
    _ma_tag = f"{wave_ma_window}-day MA ({auto_tag})"
    fig_waves.update_layout(
        title=(
            f"<b>COVID-19 {wave_metric} Wave Detection — {wave_location}</b>"
            f"<br><sub>{n_waves} major wave(s) detected · {_ma_tag} · {_det_tag}</sub>"
        ),
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        height=580,
        barmode="overlay",
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        font=dict(family="sans-serif", size=11),
    )
    fig_waves.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig_waves.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    st.plotly_chart(fig_waves, use_container_width=True)

    if wave_list:
        st.markdown("#### Individual Wave Details")

        # Vaccination coverage at each wave peak (if data available)
        _wave_fips = None
        _vax_peak_lookup: dict = {}
        if vax_ts_df is not None and not vax_ts_df.empty:
            _pop_row = population_df[
                (population_df["County Name"] == wave_county_name) &
                (population_df["State"] == wave_state)
            ]
            if not _pop_row.empty and "countyFIPS" in _pop_row.columns:
                _wave_fips = str(_pop_row.iloc[0]["countyFIPS"]).zfill(5)
                _cv_vax_ts = get_county_vax_timeseries(vax_ts_df, _wave_fips)
                if not _cv_vax_ts.empty and "vax_complete_pct" in _cv_vax_ts.columns:
                    _cv_vax_ts = _cv_vax_ts.sort_values("Date")
                    for _w in wave_list:
                        _peak_dt = pd.Timestamp(_w["peak_date"])
                        # Find closest vaccination data point on or before peak
                        _prior = _cv_vax_ts[_cv_vax_ts["Date"] <= _peak_dt]
                        if not _prior.empty:
                            _vax_peak_lookup[_w["wave_number"]] = float(
                                _prior.iloc[-1]["vax_complete_pct"]
                            )

        wave_table_rows = []
        for w in wave_list:
            pv     = w["peak_value"]
            burden = w.get("wave_burden", float("nan"))
            sig    = w.get("wave_significance", float("nan"))
            _row = {
                "Wave #":              w["wave_number"],
                "Significance":        f"{sig:.0f}/100" if not np.isnan(sig) else "—",
                "Start Date":          str(w["start_date"])[:10],
                "Peak Date":           str(w["peak_date"])[:10],
                "End Date":            str(w["end_date"])[:10],
                f"Peak ({y_label})":   (f"{pv:.2f}" if is_percap_metric else f"{pv:,.0f}"),
                "Duration (days)":     w["duration_days"],
                burden_lbl:            (f"{burden:.1f}" if is_percap_metric else f"{burden:,.0f}")
                                        if not np.isnan(burden) else "N/A",
            }
            if _vax_peak_lookup:
                _vax_at_peak = _vax_peak_lookup.get(w["wave_number"], np.nan)
                _row["Fully Vacc. at Peak"] = (
                    f"{_vax_at_peak:.1f}%" if not np.isnan(_vax_at_peak) else "N/A (pre-rollout)"
                )
            wave_table_rows.append(_row)

        wave_table_df = pd.DataFrame(wave_table_rows)
        st.dataframe(wave_table_df, use_container_width=True, hide_index=True)
        if _vax_peak_lookup:
            st.caption(
                "**Fully Vacc. at Peak**: % of county population with completed primary COVID-19 "
                "vaccination series as of the wave peak date (CDC county dataset, most recent "
                "prior observation used). Waves before Jan 2021 show N/A — rollout had not begun."
            )
            # Vaccination–severity correlation insight
            # Build numeric pairs: (vax_pct_at_peak, peak_value) for waves
            # where both values are available and vaccination had begun.
            _vax_sev_pairs = []
            for w in wave_list:
                _vn = w["wave_number"]
                _vp = _vax_peak_lookup.get(_vn, np.nan)
                _pp = w["peak_value"]
                if not np.isnan(_vp) and _vp > 0 and not np.isnan(_pp):
                    _vax_sev_pairs.append((_vp, _pp))
            if len(_vax_sev_pairs) >= 3:
                _vax_arr = np.array([p[0] for p in _vax_sev_pairs])
                _sev_arr = np.array([p[1] for p in _vax_sev_pairs])
                _corr = float(np.corrcoef(_vax_arr, _sev_arr)[0, 1])
                _corr_desc = (
                    "negatively correlated (higher vaccination associated with lower peak severity)"
                    if _corr < -0.2
                    else "positively correlated (peaks were more severe during higher-vax waves — consistent with Omicron's immune evasion)"
                    if _corr > 0.2
                    else "weakly correlated (no clear linear relationship between vaccination rate and peak severity for this county)"
                )
                with st.expander("Vaccination vs. Peak Severity", expanded=False):
                    st.markdown(
                        f"Across **{len(_vax_sev_pairs)} waves** with vaccination data, "
                        f"vaccination rate at peak and peak {y_label} are "
                        f"**{_corr_desc}** (r = {_corr:+.2f}). "
                        f"Note: correlation direction is influenced by which variants circulated "
                        f"during each wave — Delta and Omicron peaked at different vaccination levels "
                        f"with different severity profiles."
                    )
                    _vs_df = pd.DataFrame({
                        "Wave #":            [w["wave_number"] for w in wave_list
                                              if _vax_peak_lookup.get(w["wave_number"], np.nan) > 0
                                              and not np.isnan(w["peak_value"])],
                        "Peak Date":         [str(w["peak_date"])[:10] for w in wave_list
                                              if _vax_peak_lookup.get(w["wave_number"], np.nan) > 0
                                              and not np.isnan(w["peak_value"])],
                        f"Peak ({y_label})": [round(w["peak_value"], 2) for w in wave_list
                                              if _vax_peak_lookup.get(w["wave_number"], np.nan) > 0
                                              and not np.isnan(w["peak_value"])],
                        "Fully Vacc. (%)":   [round(_vax_peak_lookup[w["wave_number"]], 1) for w in wave_list
                                              if _vax_peak_lookup.get(w["wave_number"], np.nan) > 0
                                              and not np.isnan(w["peak_value"])],
                    })
                    st.dataframe(_vs_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No major waves detected with the current settings. "
            "Try switching to **Sensitive** detection, or use **Advanced detection controls** "
            "to manually lower the prominence threshold. "
            "Very rural counties with sparse case counts may not exhibit clearly defined waves."
        )

    with st.expander("Compare Case Waves vs Death Waves", expanded=False):
        if "error" in raw_county_results:
            st.caption("Comparison unavailable for this county.")
        else:
            c_waves = raw_county_results["cases"]["number_of_waves"]
            d_waves = raw_county_results["deaths"]["number_of_waves"]
            cmp1, cmp2, cmp3 = st.columns(3)
            with cmp1: st.metric("Case Waves",  c_waves)
            with cmp2: st.metric("Death Waves", d_waves)
            with cmp3:
                c_peak = raw_county_results["cases"]["date_of_peak_wave"]
                d_peak = raw_county_results["deaths"]["date_of_peak_wave"]
                if c_peak and d_peak:
                    lag = (pd.Timestamp(d_peak) - pd.Timestamp(c_peak)).days
                    st.metric("Peak Death Lag", f"{lag} days",
                              help="Days between peak case wave and peak death wave")
                else:
                    st.metric("Peak Death Lag", "N/A")

            c_list = raw_county_results["cases"]["waves"]
            d_list = raw_county_results["deaths"]["waves"]
            if c_list or d_list:
                rows = []
                for i in range(max(len(c_list), len(d_list))):
                    row = {"Wave #": i + 1}
                    if i < len(c_list):
                        cw = c_list[i]
                        row["Case Peak Date"]  = str(cw["peak_date"])[:10]
                        row["Case Peak Count"] = f"{cw['peak_value']:,.0f}"
                        row["Case Duration"]   = f"{cw['duration_days']} days"
                    else:
                        row["Case Peak Date"] = row["Case Peak Count"] = row["Case Duration"] = "—"
                    if i < len(d_list):
                        dw = d_list[i]
                        row["Death Peak Date"]  = str(dw["peak_date"])[:10]
                        row["Death Peak Count"] = f"{dw['peak_value']:,.0f}"
                        row["Death Duration"]   = f"{dw['duration_days']} days"
                    else:
                        row["Death Peak Date"] = row["Death Peak Count"] = row["Death Duration"] = "—"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Detection Diagnostics", expanded=False):
        _audit_log   = diag.get("peak_audit_log", [])
        _region_mode = diag.get("baseline_used", False)
        _dc, _dw, _dv, _dd, _df = (
            diag.get("candidate_peaks", 0),
            diag.get("after_width_filter", 0),
            diag.get("after_valley_merge", 0),
            diag.get("merged_peaks", 0),
            diag.get("final_waves", 0),
        )
        if _region_mode:
            st.markdown(
                f"**Epidemiological region detection:** "
                f"{diag.get('epidemic_regions', _df)} region(s) identified → "
                f"{_df} wave(s) after onset refinement."
            )
        else:
            st.markdown(
                f"**Legacy pipeline:** "
                f"{_dc} candidates → "
                f"{_dw} after width filter → "
                f"{_dv} after valley merge → "
                f"{_df} final waves ({_dd} merged or absorbed)"
            )
        if _audit_log:
            _audit_rows = []
            for _e in _audit_log:
                _pk_date = str(_e.get("peak_date", ""))[:10] if _e.get("peak_date") is not None else "—"
                _val     = _e.get("smoothed_value", 0.0)
                _pct     = _e.get("pct_of_max", 0.0)
                _dur     = _e.get("fwhm_days", "—")
                _min_dur = _e.get("eff_min_width", "—")
                _removed = _e.get("removed_by") or "kept"
                _status  = {
                    "kept":                "Kept",
                    "width_filter":        "Removed — width filter",
                    "valley_merge":        "Merged — valley",
                    "distance_merge":      "Merged — distance",
                    "date_merge":          "Merged — date",
                    "significance_filter": "Removed — below significance floor",
                }.get(_removed, _removed)
                _val_fmt = f"{_val:.2f}" if is_percap_metric else f"{_val:,.0f}"
                _audit_rows.append({
                    "Peak Date":                                             _pk_date,
                    f"Smoothed ({y_label})":                                _val_fmt,
                    "% of Max":                                             f"{_pct:.0f}%",
                    "Duration (days)" if _region_mode else "FWHM (days)":  _dur,
                    "Min Duration"    if _region_mode else "Eff. Min Width": _min_dur,
                    "Status":                                               _status,
                })
            st.dataframe(pd.DataFrame(_audit_rows), use_container_width=True, hide_index=True)
            if _region_mode:
                st.caption(
                    "Each row is one detected epidemic region. "
                    "**Duration** = days from refined onset to end of elevated period. "
                    "**Min Duration** = minimum days required by the sensitivity preset."
                )
            else:
                st.caption(
                    "**FWHM** = full-width at half-maximum (days the signal stayed above 50% of peak). "
                    "**Eff. Min Width** = required FWHM after magnitude scaling. "
                    "Merged peaks still contribute to the wave; width-filtered peaks were removed entirely."
                )
        else:
            st.caption("Diagnostic log unavailable (no candidates found).")

    with st.expander("Validation — detected waves vs national surge windows", expanded=False):
        if wave_list:
            _val_df = match_waves_to_national_windows(wave_list)
            st.dataframe(_val_df, use_container_width=True, hide_index=True)

            _hit_windows = set(_val_df["National Window"]) - {"Outside national windows"}
            _missed = [name for name, _s, _e in NATIONAL_WAVE_WINDOWS
                       if name not in _hit_windows]
            _n_outside = int((_val_df["National Window"] == "Outside national windows").sum())

            _summary_bits = [
                f"Detected waves align with **{len(_hit_windows)} of "
                f"{len(NATIONAL_WAVE_WINDOWS)}** national surge windows."
            ]
            if _missed:
                _summary_bits.append("No wave detected during: " + ", ".join(_missed) + ".")
            if _n_outside:
                _summary_bits.append(
                    f"{_n_outside} wave(s) peaked outside every national window — "
                    "often a genuine local surge rather than a detection error."
                )
            st.caption(
                " ".join(_summary_bits) + " Missing the Winter 2020-21 or Omicron "
                "window usually means detection is set too conservatively for this "
                "county — try the Sensitive preset. National windows are deliberately "
                "generous; county waves lead or trail the national picture by weeks."
            )
        else:
            st.caption("No waves detected — nothing to validate against national windows.")

    with st.expander("How wave detection works", expanded=False):
        _preset_info = SENSITIVITY_PRESETS.get(wave_sensitivity, {})
        _elev_rel  = int(_preset_info.get("elevation_threshold_rel", 0.30) * 100)
        _elev_abs  = _preset_info.get("elevation_threshold_abs", 2.0)
        _min_reg   = _preset_info.get("min_region_duration", 14)
        _reg_gap   = _preset_info.get("region_merge_gap", 35)
        _onset_lb  = _preset_info.get("onset_lookback", 28)
        _bl_win    = _preset_info.get("baseline_window", 42)
        _bl_smooth = _preset_info.get("baseline_smooth_window", 21)
        st.markdown(f"""
**Detection pipeline ({_sens_disp} mode) — epidemiological wave detection**

1. **Smoothing** — {wave_ma_window}-day moving average applied to suppress reporting artefacts
   (weekend effects, batch corrections). {'*Window auto-selected from signal variance.*' if auto_smooth else '*Manually specified.*'}
2. **Adaptive baseline** — local epidemic background estimated as the rolling 10th percentile
   over a ±{_bl_win // 2}-day window, then re-smoothed over {_bl_smooth} days.
   The baseline represents between-wave transmission so that post-Omicron surges
   are measured against their local context rather than the county's all-time maximum.
   This is why the detector can find a BA.5 wave even when it is 10× smaller than Omicron.
3. **Epidemic region detection** — sustained periods where the smoothed signal exceeds
   the local baseline by **+{_elev_rel}% + {_elev_abs:.0f} cases/day** for ≥ **{_min_reg} consecutive days**.
   Nearby elevated periods separated by ≤ **{_reg_gap} days** are merged into one region
   (one continuous epidemic envelope — e.g., the BA.1/BA.2 sub-waves of Omicron).
4. **Valley splitting** — a merged region spanning two genuinely distinct surges (e.g.,
   Delta and Omicron) is split only at valleys that are deep **relative to the flanking
   peaks**, deep in **absolute** terms, and **sustained** (elevation stays low for weeks,
   not days). All three tests run on elevation above the local baseline, so reporting
   noise in small counties cannot masquerade as an inter-wave trough.
5. **Wave boundaries** — a region marks where the signal is elevated above local
   background, which can begin months before the surge itself. Each wave's start and
   end are therefore trimmed to the span where the signal stays above **10% of that
   wave's peak elevation** (measured against the region's background floor, with a
   5-day sustain rule so brief dips don't truncate the wave). The shaded interval
   tracks the outbreak's actual rise and fall.
6. **Peak significance filter** — a region whose peak barely rises above baseline
   (below a preset multiple of the elevation threshold) is discarded as a low-signal
   plateau rather than reported as a wave. Dropped candidates appear in the Detection
   Diagnostics log.
7. **Significance score** — each surviving wave is scored 0–100 combining prominence (30%),
   total burden (30%), duration (20%), and burst intensity (20%), ranking waves by
   epidemiological importance rather than simply peak height.

**Sensitivity presets**

| Preset | Expected waves | Elevation threshold | Min region | Merge gap |
|---|---|---|---|---|
| Conservative | 3–5 | +50% above baseline | 21 days | 56 days |
| Standard | 4–8 | +30% above baseline | 14 days | 35 days |
| Sensitive | 6–15 | +15% above baseline | 7 days | 21 days |

Use **Advanced detection controls** to apply the legacy prominence-based algorithm with manual parameters.
""")
        st.markdown("**Significance score**")
        st.latex(
            r"S = 100\left[0.3\,\frac{p}{p_{\max}} + 0.3\,\frac{b}{B} + "
            r"0.2\,\min\!\left(\frac{d}{180},1\right) + 0.2\,\frac{b/d}{(b/d)_{\max}}\right]"
        )
        st.caption(
            "p = peak height, b = wave burden, B = total series burden, d = duration in days. "
            "The last term is burst intensity — it keeps short, explosive waves from being "
            "penalised for brevity."
        )

def render_county_factors_tab(
    master_df, cases_df, deaths_df, population_df, transforms
) -> None:
    """County Factors tab — scatter-plot explorer of COVID outcomes vs county characteristics."""
    render_section_header(
        "County Factors",
        "What made some counties fare worse than others? Explore how healthcare access, income, "
        "population density, age distribution, and vaccination rates relate to COVID outcomes "
        "across 3,000+ counties. Each dot is a county — patterns in the scatter reveal "
        "structural inequities in pandemic impact.",
    )
    render_learning_aids(
        terms=("pearson", "spearman", "p_value", "r_squared", "ecological"),
        questions=(
            "Correlate **vaccination** with **deaths per 100k** on the full pandemic, then "
            "restrict the outcome window to the post-rollout era. Does the association "
            "strengthen, weaken, or flip — and why would the window matter?",
            "Find a factor where Pearson and Spearman disagree noticeably. "
            "Look at the scatter — what shape explains the gap?",
            "Pick the strongest correlation in the rankings table and propose two "
            "confounders that could produce it without any causal effect.",
        ),
    )

    if master_df is None or master_df.empty:
        st.error("Master county table is unavailable. AHRF data may not have loaded correctly.")
        return

    if not _ahrf_loaded:
        st.warning(
            "AHRF socioeconomic data did not load — healthcare access and economic factors "
            "will show as N/A. Vaccination factors remain available. Check that "
            "`DATA/ahrf2023.csv` is present and readable.",
            icon=None,
        )

    OUTCOME_OPTIONS = {
        "Cases per 100k (cumulative)":   "cases_per_100k",
        "Deaths per 100k (cumulative)":  "deaths_per_100k",
        "Case Fatality Rate (%)":        "case_fatality_rate",
        "Peak Wave Size — cases/100k":   "peak_wave_cases_per_100k",
        "Peak Wave Size — deaths/100k":  "peak_wave_deaths_per_100k",
        "Number of Case Waves":          "case_wave_count",
        # Vaccination outcomes (CDC county dataset)
        "Vaccination Complete (%)":      "vax_complete_pct",
        "At Least 1 Dose (%)":           "vax_dose1_pct",
    }

    FACTOR_OPTIONS = {
        # Vaccination (CDC county-level dataset — as of May 2023)
        "Vaccination Complete (%)":           "vax_complete_pct",
        "At Least 1 Dose (%)":               "vax_dose1_pct",
        "Booster Rate (%)":                  "vax_booster_pct",
        "65+ Vaccination Rate (%)":          "vax_complete_65plus_pct",
        # Healthcare access
        "Primary Care Physicians per 100k":  "pcp_per_100k",
        "Total Active MDs per 100k":         "total_md_per_100k",
        "Hospital Beds per 100k":            "hospital_beds_per_100k",
        "ICU Beds per 100k":                 "icu_beds_per_100k",
        "SNF Beds per 100k":                 "snf_beds_per_100k",
        "HPSA Primary Care Designation":     "hpsa_primary_care",
        "Critical Access Hospitals (count)": "critical_access_hospitals",
        # Economic
        "Median Family Income ($)":          "median_family_income",
        "Per Capita Income ($)":             "per_capita_income",
        "Unemployment Rate (%)":             "unemployment_rate",
        "Child Poverty Rate (%)":            "child_poverty_pct",
        # Education
        "% Without HS Diploma":             "pct_no_hs_diploma",
        "% 4-Year College Degree":          "pct_college_4yr",
        # Demographics
        "Population Density (per sq mi)":   "pop_density_per_sqmi",
        "% Population 65+":                 "pct_pop_65plus",
        "Median Age (years)":               "median_age",
        "% Urban Population":               "pct_urban_pop",
        # Rural-urban
        "RUCC Code (1=most metro → 9=most rural)": "rucc_code",
    }

    WAVE_OUTCOMES = {"peak_wave_cases_per_100k", "peak_wave_deaths_per_100k", "case_wave_count"}

    # Primary axis selectors
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        outcome_label = st.selectbox(
            "COVID Outcome (Y axis)",
            list(OUTCOME_OPTIONS.keys()),
            help="Select the COVID outcome variable to plot on the Y axis",
        )
        outcome_col = OUTCOME_OPTIONS[outcome_label]

    with sel_col2:
        factor_label = st.selectbox(
            "County Factor (X axis)",
            list(FACTOR_OPTIONS.keys()),
            help="Select the county characteristic to plot on the X axis",
        )
        factor_col = FACTOR_OPTIONS[factor_label]

    # Optional filters (collapsed by default to keep chart front-and-center)
    with st.expander("Filter counties", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            all_states = sorted(master_df["State"].dropna().unique())
            state_filter = st.multiselect(
                "State",
                options=all_states,
                default=[],
                placeholder="All states",
                key="cf_state_filter",
            )
        with f_col2:
            region_opts = []
            if "census_region_name" in master_df.columns:
                region_opts = sorted(master_df["census_region_name"].dropna().unique())
            region_filter = st.multiselect(
                "Census Region",
                options=region_opts,
                default=[],
                placeholder="All regions",
                key="cf_region_filter",
            )
        with f_col3:
            metro_filter = st.selectbox(
                "Metro / Nonmetro",
                ["All Counties", "Metro Counties", "Nonmetro Counties"],
                key="cf_metro_filter",
                help="Metro = RUCC 1–3 · Nonmetro = RUCC 4–9",
            )

    # Outcome time window. Restricting outcomes to a sub-period matters most
    # for vaccination factors: correlating final vaccination rates against
    # full-pandemic cumulative outcomes mixes pre- and post-rollout deaths.
    _dates_all = transforms["dates"]
    _win_c1, _win_c2 = st.columns([2, 3], vertical_alignment="bottom")
    with _win_c1:
        window_choice = st.selectbox(
            "Outcome window",
            ["Full pandemic",
             "Pre-vaccine era (through 2020-12-13)",
             "Post-rollout era (2021-07-01 onward)",
             "Custom range"],
            key="cf_window",
            help=(
                "Restrict COVID outcome columns (cases/deaths per 100k, CFR) to events "
                "within a date window. Use Post-rollout when relating vaccination "
                "factors to outcomes."
            ),
        )
    _window_range = None
    if window_choice == "Pre-vaccine era (through 2020-12-13)":
        _window_range = (_dates_all[0], "2020-12-13")
    elif window_choice == "Post-rollout era (2021-07-01 onward)":
        _window_range = ("2021-07-01", _dates_all[-1])
    elif window_choice == "Custom range":
        with _win_c2:
            _window_range = st.select_slider(
                "Window",
                options=_dates_all,
                value=(_dates_all[0], _dates_all[-1]),
                key="cf_window_custom",
            )
        if _window_range[0] >= _window_range[1]:
            st.warning("Window start must be before window end — using full pandemic.")
            _window_range = None

    plot_df = master_df.copy()
    if state_filter:
        plot_df = plot_df[plot_df["State"].isin(state_filter)]
    if region_filter and "census_region_name" in plot_df.columns:
        plot_df = plot_df[plot_df["census_region_name"].isin(region_filter)]
    if metro_filter == "Metro Counties" and "is_metro" in plot_df.columns:
        plot_df = plot_df[plot_df["is_metro"] == True]
    elif metro_filter == "Nonmetro Counties" and "is_metro" in plot_df.columns:
        plot_df = plot_df[plot_df["is_metro"] == False]

    if _window_range is not None:
        _win_outcomes = get_window_outcomes(
            cases_df, deaths_df, population_df, _window_range[0], _window_range[1]
        )
        _outcome_override = ["cases_per_100k", "deaths_per_100k", "case_fatality_rate"]
        plot_df = plot_df.drop(
            columns=[c for c in _outcome_override if c in plot_df.columns]
        ).merge(_win_outcomes, on=["countyFIPS", "State"], how="left")
        st.caption(
            f"Outcome window active: **{_window_range[0]} → {_window_range[1]}** — "
            "cases/deaths per 100k and CFR reflect only events in this window. "
            "Factor columns (income, healthcare, vaccination) are unchanged."
        )

    # Wave metrics are computed lazily on demand (all ~3,100 counties, ~60–90 s).
    if outcome_col in WAVE_OUTCOMES:
        if outcome_col not in plot_df.columns:
            st.info(
                "Wave metrics require computing waves for all ~3,100 counties. "
                "This takes about 60–90 seconds and is cached for the session."
            )
            compute_waves = st.button(
                "Compute wave metrics for all counties",
                key="cf_compute_waves_btn",
            )
            if not compute_waves:
                st.stop()

            with st.spinner("Computing wave metrics for all counties (one time)…"):
                wave_metrics_df = get_county_wave_metrics(
                    cases_df, deaths_df,
                    transforms["daily_cases"], transforms["daily_deaths"],
                    population_df,
                )

            wave_cols = [c for c in wave_metrics_df.columns
                         if c not in {"County Name", "State", "countyFIPS"}]
            plot_df = plot_df.merge(
                wave_metrics_df[["countyFIPS", "State"] + wave_cols],
                on=["countyFIPS", "State"],
                how="left",
                suffixes=("", "_wave"),
            )

    missing_cols = []
    if outcome_col not in plot_df.columns:
        missing_cols.append(f"Outcome column '{outcome_col}' not found in master table.")
    if factor_col not in plot_df.columns:
        missing_cols.append(f"Factor column '{factor_col}' not found in master table.")
    if missing_cols:
        for msg in missing_cols:
            st.warning(msg)
        return

    corr = compute_bivariate_correlation(plot_df, factor_col, outcome_col, min_n=10)
    ols  = compute_ols_trend(plot_df, factor_col, outcome_col)

    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

    def _fmt(v, decimals=3):
        return f"{v:.{decimals}f}" if pd.notna(v) else "N/A"

    def _fmt_p(p):
        if pd.isna(p):
            return "N/A"
        return "< 0.0001" if p < 0.0001 else f"{p:.4f}"

    with stat_col1:
        render_metric_card("Sample Size (N)", corr["n"] if corr["n"] > 0 else "—")
    with stat_col2:
        render_metric_card("Pearson r", _fmt(corr["pearson_r"]))
    with stat_col3:
        render_metric_card("Spearman r", _fmt(corr["spearman_r"]))
    with stat_col4:
        render_metric_card("R²", _fmt(corr["r_squared"]))
    with stat_col5:
        render_metric_card("p-value", _fmt_p(corr["pearson_p"]))

    hover_cols = ["County Name", "State", "countyFIPS"]
    color_col  = "rucc_group" if "rucc_group" in plot_df.columns else None
    keep_cols  = [factor_col, outcome_col] + hover_cols
    if color_col:
        keep_cols.append(color_col)
    valid_df = plot_df[[c for c in keep_cols if c in plot_df.columns]].dropna(
        subset=[factor_col, outcome_col]
    )

    if len(valid_df) < 3:
        st.warning(
            f"Only {len(valid_df)} counties have both '{factor_label}' and "
            f"'{outcome_label}' data — not enough to plot."
        )
        return

    fig = px.scatter(
        valid_df,
        x=factor_col,
        y=outcome_col,
        color=color_col,
        color_discrete_map={"Metro": "#153A66", "Nonmetro": "#F26A21"},
        hover_name="County Name" if "County Name" in valid_df.columns else None,
        hover_data={"State": True, factor_col: ":.2f", outcome_col: ":.2f",
                    color_col: False} if color_col else {"State": True},
        opacity=0.55,
        labels={
            factor_col:  factor_label,
            outcome_col: outcome_label,
            color_col:   "County Type" if color_col else "",
        },
        title=f"<b>{outcome_label}</b> vs <b>{factor_label}</b>",
    )

    if pd.notna(ols.get("slope")):  # OLS trend line
        fig.add_trace(
            go.Scatter(
                x=[ols["x_min"], ols["x_max"]],
                y=[ols["y_pred_min"], ols["y_pred_max"]],
                mode="lines",
                name=f"OLS trend  R²={ols['r_squared']:.3f}",
                line=dict(color="#c0392b", width=2.5),
                showlegend=True,
            )
        )

    if corr["n"] >= 10:  # correlation stats annotation
        ann_lines = [
            f"<b>Pearson r</b> = {_fmt(corr['pearson_r'])}  (p {_fmt_p(corr['pearson_p'])})",
            f"<b>Spearman r</b> = {_fmt(corr['spearman_r'])}  (p {_fmt_p(corr['spearman_p'])})",
            f"<b>R²</b> = {_fmt(corr['r_squared'])}  ·  <b>N</b> = {corr['n']:,}",
        ]
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text="<br>".join(ann_lines),
            showarrow=False,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=6,
            align="left",
            font=dict(size=11),
        )

    apply_chart_styling(fig)
    fig.update_layout(height=580, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

    if corr["n"] >= 30 and pd.notna(corr["pearson_r"]):
        r = corr["pearson_r"]
        strength = (
            "strong" if abs(r) >= 0.6 else
            "moderate" if abs(r) >= 0.3 else
            "weak"
        )
        direction = "positive" if r > 0 else "negative"
        sig = corr["pearson_p"] < 0.05
        sig_text = "statistically significant (p < 0.05)" if sig else "not statistically significant (p ≥ 0.05)"
        st.caption(
            f"The Pearson correlation is **{strength} {direction}** (r = {_fmt(r)}) "
            f"and is **{sig_text}** across {corr['n']:,} counties."
        )

    with st.expander("View underlying data table", expanded=False):
        display_df = valid_df.rename(columns={
            factor_col:  factor_label,
            outcome_col: outcome_label,
        })
        if color_col and color_col in display_df.columns:
            display_df = display_df.rename(columns={color_col: "Metro/Nonmetro"})
        st.dataframe(
            display_df.sort_values(outcome_label, ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(valid_df):,} counties shown · sorted by {outcome_label} descending")

    # Factor Correlation Rankings
    st.markdown("---")
    st.markdown("#### Factor Correlation Rankings")
    st.caption(
        f"Pearson r and Spearman ρ for all {len(FACTOR_OPTIONS)} county factors "
        f"against **{outcome_label}**, using the active state/region/metro filters. "
        "Sorted by |Pearson r| by default. Click a column header to re-sort."
    )

    ranking_rows = []
    for f_lbl, f_col_r in FACTOR_OPTIONS.items():
        if f_col_r not in plot_df.columns:
            continue
        r_corr = compute_bivariate_correlation(plot_df, f_col_r, outcome_col, min_n=10)
        if r_corr["n"] < 10:
            continue
        ranking_rows.append({
            "Factor":        f_lbl,
            "Pearson r":     r_corr["pearson_r"]  if pd.notna(r_corr["pearson_r"])  else None,
            "Spearman ρ":    r_corr["spearman_r"] if pd.notna(r_corr["spearman_r"]) else None,
            "p-value":       r_corr["pearson_p"]  if pd.notna(r_corr["pearson_p"])  else None,
            "N":             r_corr["n"],
        })

    if ranking_rows:
        ranking_df = pd.DataFrame(ranking_rows)
        ranking_df["_abs_r"] = ranking_df["Pearson r"].abs()
        ranking_df = ranking_df.sort_values("_abs_r", ascending=False).drop(columns=["_abs_r"])
        ranking_df = ranking_df.reset_index(drop=True)

        def _style_r(v):
            if pd.isna(v):
                return ""
            c = "#c41e3a" if v > 0 else "#153A66"
            return f"color: {c}; font-weight: bold"

        st.dataframe(
            ranking_df.style
            .format({
                "Pearson r":  lambda v: f"{v:.3f}" if pd.notna(v) else "N/A",
                "Spearman ρ": lambda v: f"{v:.3f}" if pd.notna(v) else "N/A",
                "p-value":    lambda v: ("< 0.0001" if v < 0.0001 else f"{v:.4f}") if pd.notna(v) else "N/A",
                "N":          "{:,}",
            })
            .map(_style_r, subset=["Pearson r", "Spearman ρ"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Not enough data to compute factor rankings with current filters.")

    # County Resilience Explorer
    st.markdown("---")
    st.markdown("#### County Resilience Explorer")
    st.caption(
        "**Exploratory only — not causal inference.** "
        "OLS regression of deaths/100k on cases/100k; residuals measure whether a county's "
        "mortality was higher or lower than expected given its case burden. "
        "Negative residual = fewer deaths than expected (more resilient). "
        "Positive residual = more deaths than expected (more fragile). "
        "Investigate specific counties further using the scatter plot above."
    )

    resil_cols = ["County Name", "State", "cases_per_100k", "deaths_per_100k"]
    resil_optional = ["rucc_group", "census_region_name"]
    resil_available = [c for c in resil_cols + resil_optional if c in plot_df.columns]
    resil_df = plot_df[resil_available].dropna(subset=["cases_per_100k", "deaths_per_100k"]).copy()

    if len(resil_df) >= 30:
        resil_ols = compute_ols_trend(resil_df, "cases_per_100k", "deaths_per_100k")
        if pd.notna(resil_ols.get("slope")):
            resil_df["predicted_deaths"] = (
                resil_ols["slope"] * resil_df["cases_per_100k"] + resil_ols["intercept"]
            )
            resil_df["residual"] = resil_df["deaths_per_100k"] - resil_df["predicted_deaths"]

            resil_display_cols = {
                "County Name":       "County",
                "State":             "State",
                "cases_per_100k":    "Cases /100k",
                "deaths_per_100k":   "Deaths /100k",
                "predicted_deaths":  "Expected Deaths /100k",
                "residual":          "Residual",
            }
            if "rucc_group" in resil_df.columns:
                resil_display_cols["rucc_group"] = "Metro/Nonmetro"
            if "census_region_name" in resil_df.columns:
                resil_display_cols["census_region_name"] = "Region"

            resil_df_disp = resil_df[[c for c in resil_display_cols if c in resil_df.columns]].rename(
                columns=resil_display_cols
            )

            r20_col, f20_col = st.columns(2)

            with r20_col:
                st.markdown("**Most Resilient (lowest residual)**")
                top20 = resil_df_disp.nsmallest(20, "Residual")
                fmt_cols = {c: "{:.2f}" for c in ["Cases /100k", "Deaths /100k", "Expected Deaths /100k"]}
                fmt_cols["Residual"] = "{:.3f}"
                st.dataframe(
                    top20.style.format(fmt_cols),
                    use_container_width=True,
                    hide_index=True,
                )

            with f20_col:
                st.markdown("**Most Fragile (highest residual)**")
                bot20 = resil_df_disp.nlargest(20, "Residual")
                st.dataframe(
                    bot20.style.format(fmt_cols),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                f"Regression: deaths/100k = {resil_ols['slope']:.4f} × cases/100k "
                f"+ {resil_ols['intercept']:.2f}  |  "
                f"R² = {resil_ols['r_squared']:.3f}  ·  N = {len(resil_df):,} counties"
            )
        else:
            st.info("Insufficient variation in cases/deaths data to compute resilience regression.")
    else:
        st.info(
            f"At least 30 counties are required for the resilience analysis ({len(resil_df)} currently match your filters)."
        )

    with st.expander("AHRF variable catalog", expanded=False):
        catalog = get_variable_catalog()
        st.dataframe(catalog, use_container_width=True, hide_index=True)
        st.caption(
            "All variables sourced from HRSA Area Health Resources Files (AHRF). "
            "Primary source: ahrf2023.csv (2021-era data). "
            "Supplementary: AHRF2020.asc (2018-2020) · AHRF2021.sas7bdat (2019-2021)."
        )

    # Master county table export
    st.markdown("---")
    with st.expander("Download full county dataset", expanded=False):
        st.caption(
            "Complete county-level dataset: COVID outcomes, AHRF socioeconomic variables, "
            "and CDC vaccination rates — one row per county. Ready for external analysis."
        )
        _export_master = master_df.copy()
        # Round float columns to keep file size manageable
        _float_cols = _export_master.select_dtypes(include="float").columns
        _export_master[_float_cols] = _export_master[_float_cols].round(4)
        st.download_button(
            label="Download master county dataset (CSV)",
            data=_export_master.to_csv(index=False).encode("utf-8"),
            file_name="covid_county_master_dataset.csv",
            mime="text/csv",
            key="master_download",
        )
        st.caption(f"{len(_export_master):,} counties · {len(_export_master.columns):,} variables")

# Statistical modeling tab — cached wrappers.
# The dataframe parameter is deliberately hashed (no underscore prefix): the
# tab passes a *filtered* dataframe, so the cache key must include its content
# or filter changes would silently return stale results.

@st.cache_data
def _cached_correlations(df, outcome_col, factor_cols_tuple):
    """Cache correlation matrix for a given outcome and factor set."""
    return compute_all_correlations(
        df,
        outcome_cols=[outcome_col],
        factor_cols=list(factor_cols_tuple),
        min_n=10,
    )

@st.cache_data
def _cached_corr_heatmap(df, outcome_cols_tuple, factor_cols_tuple):
    """Cache full outcome × factor correlation matrix for the heatmap."""
    return compute_all_correlations(
        df,
        outcome_cols=list(outcome_cols_tuple),
        factor_cols=list(factor_cols_tuple),
        min_n=10,
    )

@st.cache_data
def _cached_rf_importance(df, outcome_col, feature_cols_tuple):
    return run_rf_feature_importance(df, outcome_col, list(feature_cols_tuple))

@st.cache_data
def _cached_partial_dependence(df, outcome_col, feature_cols_tuple, top_k=3):
    return run_rf_partial_dependence(df, outcome_col, list(feature_cols_tuple), top_k=top_k)

@st.cache_data
def _cached_ols(df, outcome_col, predictor_cols_tuple):
    return run_ols_regression(df, outcome_col, list(predictor_cols_tuple))

@st.cache_data
def _cached_resilience(df, outcome_col, feature_cols_tuple, cv_folds=5):
    return compute_resilience_scores(df, outcome_col, list(feature_cols_tuple), cv_folds=cv_folds)

@st.cache_data
def _cached_clusters(df, feature_cols_tuple, k):
    return compute_county_clusters(df, list(feature_cols_tuple), k=k)

def render_modeling_tab(master_county_df, locations) -> None:
    """Statistical Modeling & Outcome Drivers tab."""
    render_section_header(
        "Statistical Modeling",
        "Move beyond visualization — quantify which county characteristics most strongly "
        "predicted COVID outcomes. Apply Pearson correlations, machine learning feature "
        "importance, or multivariable regression to identify the key structural drivers "
        "of county-level disparities in cases, deaths, and mortality rates.",
    )
    render_learning_aids(
        terms=("r_squared", "p_value", "vif", "hc3", "residual", "ecological"),
        questions=(
            "Fit an OLS model with both income variables as predictors, then check the "
            "**VIF** table. What happens to their coefficients, and why?",
            "Find a predictor that is significant in the correlation matrix but not in "
            "the multivariable regression. What does that tell you?",
            "Run the resilience scores, then look up the most resilient county's profile. "
            "What might the model be missing about it?",
        ),
    )

    if master_county_df is None or master_county_df.empty:
        st.error("Master county table unavailable. AHRF data may not have loaded.")
        return

    if not _ahrf_loaded:
        st.warning(
            "AHRF socioeconomic data did not load — healthcare access and economic "
            "factor columns will be absent from models. Vaccination factors remain "
            "available. Check that `DATA/ahrf2023.csv` is present and readable.",
            icon=None,
        )

    # Shared constants
    _available_factors  = {
        lbl: col for lbl, col in _MOD_FACTOR_COLS.items()
        if col in master_county_df.columns
    }
    _available_outcomes = {
        lbl: col for lbl, col in _MOD_OUTCOME_COLS.items()
        if col in master_county_df.columns
    }

    if not _available_factors:
        st.warning("No factor columns found in master table. Ensure data loaded correctly.")
        return
    if not _available_outcomes:
        st.warning("No COVID outcome columns found in the master table.")
        return

    _factor_cols_all   = list(_available_factors.values())
    _outcome_cols_all  = list(_available_outcomes.values())

    def _fmt_p(p):
        if pd.isna(p):
            return "N/A"
        return "< 0.0001" if p < 0.0001 else f"{p:.4f}"

    def _color_r(v):
        if pd.isna(v):
            return ""
        return f"color: {'#c41e3a' if v > 0 else '#153A66'}; font-weight: 600"

    # Global dataset filter (collapsed — models work on all counties by default)
    with st.expander("Filter dataset", expanded=False):
        gf1, gf2, gf3, gf4 = st.columns(4)
        with gf1:
            all_states = sorted(master_county_df["State"].dropna().unique())
            mod_states = st.multiselect("State", all_states, default=[],
                                        placeholder="All states", key="mod_state_filter")
        with gf2:
            region_opts = (sorted(master_county_df["census_region_name"].dropna().unique())
                           if "census_region_name" in master_county_df.columns else [])
            mod_regions = st.multiselect("Census Region", region_opts, default=[],
                                         placeholder="All regions", key="mod_region_filter")
        with gf3:
            mod_metro = st.selectbox(
                "Metro / Nonmetro",
                ["All Counties", "Metro Counties", "Nonmetro Counties"],
                key="mod_metro_filter",
            )
        with gf4:
            mod_window = st.selectbox(
                "Outcome window",
                ["Full pandemic",
                 "Pre-vaccine era (through 2020-12-13)",
                 "Post-rollout era (2021-07-01 onward)"],
                key="mod_window",
                help=(
                    "Restrict COVID outcomes to a sub-period. Post-rollout is the "
                    "defensible choice when modeling vaccination effects."
                ),
            )

    plot_df = master_county_df.copy()
    if mod_states:
        plot_df = plot_df[plot_df["State"].isin(mod_states)]
    if mod_regions and "census_region_name" in plot_df.columns:
        plot_df = plot_df[plot_df["census_region_name"].isin(mod_regions)]
    if mod_metro == "Metro Counties" and "is_metro" in plot_df.columns:
        plot_df = plot_df[plot_df["is_metro"] == True]
    elif mod_metro == "Nonmetro Counties" and "is_metro" in plot_df.columns:
        plot_df = plot_df[plot_df["is_metro"] == False]

    # Window-restricted outcomes (cases_df etc. are module-level; the master
    # table's factor columns are untouched)
    _mod_window_range = None
    if mod_window == "Pre-vaccine era (through 2020-12-13)":
        _mod_window_range = (dates[0], "2020-12-13")
    elif mod_window == "Post-rollout era (2021-07-01 onward)":
        _mod_window_range = ("2021-07-01", dates[-1])
    if _mod_window_range is not None:
        _mod_win = get_window_outcomes(
            cases_df, deaths_df, population_df,
            _mod_window_range[0], _mod_window_range[1],
        )
        _mod_override = ["cases_per_100k", "deaths_per_100k", "case_fatality_rate"]
        plot_df = plot_df.drop(
            columns=[c for c in _mod_override if c in plot_df.columns]
        ).merge(_mod_win, on=["countyFIPS", "State"], how="left")

    _window_note = (
        f" · outcome window **{_mod_window_range[0]} → {_mod_window_range[1]}**"
        if _mod_window_range else ""
    )
    st.caption(f"**{len(plot_df):,}** counties in model dataset{_window_note}.")

    # SECTION 1 — CORRELATION MATRIX

    st.markdown(
        '<div class="sub-section-header"><h3>1 — Correlation Matrix</h3>'
        '<p>Pearson r and Spearman ρ between every county factor and every COVID outcome.</p></div>',
        unsafe_allow_html=True,
    )

    corr_out_label = st.selectbox(
        "Focus Outcome (table)",
        list(_available_outcomes.keys()),
        key="mod_corr_outcome",
    )
    corr_out_col = _available_outcomes[corr_out_label]

    corr_df = _cached_correlations(
        plot_df, corr_out_col, tuple(_factor_cols_all),
    )

    if corr_df.empty:
        st.info("No correlations computable with current filters.")
    else:
        display_corr = corr_df[["Factor", "Pearson r", "Pearson p",
                                 "Spearman ρ", "Spearman p", "N"]].copy()
        display_corr["_abs_r"] = display_corr["Pearson r"].abs()
        display_corr = display_corr.sort_values("_abs_r", ascending=False).drop(
            columns=["_abs_r"]
        )
        st.dataframe(
            display_corr.style
            .format({
                "Pearson r":  "{:.3f}", "Pearson p":  lambda v: _fmt_p(v),
                "Spearman ρ": "{:.3f}", "Spearman p": lambda v: _fmt_p(v),
                "N": "{:,}",
            })
            .map(_color_r, subset=["Pearson r", "Spearman ρ"]),
            use_container_width=True, hide_index=True,
        )

    # Heatmap: all outcomes × all factors
    with st.expander("Correlation heatmap — all outcomes × all factors", expanded=False):
        heat_df = _cached_corr_heatmap(
            plot_df, tuple(_outcome_cols_all), tuple(_factor_cols_all),
        )
        if not heat_df.empty:
            pivot = (
                heat_df[["Factor", "Outcome", "Pearson r"]]
                .pivot(index="Factor", columns="Outcome", values="Pearson r")
            )
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values.tolist(),
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale="RdBu",
                zmid=0,
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" if not np.isnan(v) else "N/A"
                        for v in row] for row in pivot.values],
                texttemplate="%{text}",
                hovertemplate=(
                    "Factor: %{y}<br>Outcome: %{x}<br>"
                    "Pearson r: %{z:.3f}<extra></extra>"
                ),
                colorbar=dict(title="Pearson r", tickvals=[-1, -0.5, 0, 0.5, 1]),
            ))
            fig_heat.update_layout(
                height=max(350, len(pivot) * 32),
                margin=dict(l=220, r=20, t=20, b=60),
                template="plotly_white",
                font=dict(family="sans-serif", size=11),
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                "Blue = negative association (higher factor → lower outcome). "
                "Red = positive. Intensity reflects strength."
            )

    st.markdown("---")

    # SECTION 2 — FEATURE IMPORTANCE (RANDOM FOREST)

    st.markdown(
        '<div class="sub-section-header"><h3>2 — Feature Importance Analysis</h3>'
        '<p>Random Forest regressor ranks which county factors are most predictive of the '
        'selected outcome. Missing factor values are imputed with column medians before '
        'fitting. Feature importance reflects predictive contribution, not causation.</p></div>',
        unsafe_allow_html=True,
    )

    fi_out_label = st.selectbox(
        "Outcome Variable",
        list(_available_outcomes.keys()),
        key="mod_fi_outcome",
    )
    fi_out_col = _available_outcomes[fi_out_label]

    if st.button("Run Feature Importance", key="mod_fi_run"):
        with st.spinner("Fitting Random Forest…"):
            fi_df, fi_err = _cached_rf_importance(
                plot_df, fi_out_col, tuple(_factor_cols_all),
            )
        if fi_err:
            st.error(fi_err)
        elif fi_df is not None:
            method_used = fi_df["Method"].iloc[0]
            st.caption(f"Method: {method_used}")

            fig_fi = go.Figure(go.Bar(
                y=fi_df["Feature"],
                x=fi_df["Importance"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0, "#d4e6f1"], [1, "#0B2341"]],
                    showscale=False,
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                xaxis=dict(title="Feature Importance"),
                height=max(300, len(fi_df) * 30),
                margin=dict(l=200, r=20, t=20, b=40),
                template="plotly_white",
                font=dict(family="sans-serif", size=11),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            show_cols = [c for c in ["Rank", "Feature", "Importance"] if c in fi_df.columns]
            st.dataframe(
                fi_df[show_cols].style.format({"Importance": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )

            # Partial dependence: how does the predicted outcome move as each
            # top feature sweeps its observed range, all else held fixed?
            pd_curves, pd_err = _cached_partial_dependence(
                plot_df, fi_out_col, tuple(_factor_cols_all), top_k=3,
            )
            if pd_curves:
                st.markdown("##### Partial Dependence — Top 3 Features")
                pd_cols = st.columns(len(pd_curves))
                for _pi, (feat_label, curve) in enumerate(pd_curves.items()):
                    with pd_cols[_pi]:
                        fig_pd = go.Figure(go.Scatter(
                            x=curve["grid_value"], y=curve["avg_prediction"],
                            mode="lines", line=dict(color=NATIONAL_COLOR, width=2.5),
                            hovertemplate=f"{feat_label}: %{{x:,.1f}}<br>Predicted: %{{y:,.2f}}<extra></extra>",
                        ))
                        fig_pd.update_layout(
                            title=dict(text=feat_label, font=dict(size=12)),
                            height=260, margin=dict(t=40, b=35, l=45, r=15),
                            template="plotly_white",
                            xaxis=dict(title=None, tickfont=dict(size=9)),
                            yaxis=dict(title=f"Predicted {fi_out_label}"
                                       if _pi == 0 else None,
                                       tickfont=dict(size=9)),
                            font=dict(family="sans-serif", size=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_pd, use_container_width=True)
                st.caption(
                    "Each curve sweeps one feature across its 5th–95th percentile range "
                    "while all other features keep their observed values; the y-axis is the "
                    "average model prediction. Flat curve = little marginal effect. "
                    "Interpret jointly with the importance ranking — correlated features "
                    "share credit."
                )
            elif pd_err:
                st.caption(f"Partial dependence unavailable: {pd_err}")
    else:
        st.info("Click **Run Feature Importance** to fit the model.")

    st.markdown("---")

    # SECTION 3 — MULTIVARIABLE OLS REGRESSION

    st.markdown(
        '<div class="sub-section-header"><h3>3 — Multivariable Regression</h3>'
        '<p>OLS regression with user-selected predictors. Allows testing whether a factor '
        'remains significant after accounting for other variables. Classical and '
        'HC3-robust standard errors are reported side by side.</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("The math", expanded=False):
        st.latex(r"\hat{\beta} = (X^{\top}X)^{-1}X^{\top}y")
        st.latex(
            r"\widehat{\mathrm{Var}}_{\mathrm{HC3}}(\hat{\beta}) = "
            r"(X^{\top}X)^{-1} X^{\top}\,\mathrm{diag}\!\left(\frac{e_i^2}{(1-h_{ii})^2}\right)"
            r"X\,(X^{\top}X)^{-1}"
        )
        st.caption(
            "e = residuals, h = hat-matrix diagonal (leverage). Classical standard errors "
            "assume every county's noise has the same variance; HC3 lets each county have "
            "its own, which is the safer assumption for skewed outcome data."
        )

    ols_c1, ols_c2 = st.columns([1, 2])
    with ols_c1:
        ols_out_label = st.selectbox(
            "Outcome Variable",
            list(_available_outcomes.keys()),
            key="mod_ols_outcome",
        )
        ols_out_col = _available_outcomes[ols_out_label]

    with ols_c2:
        ols_pred_labels = st.multiselect(
            "Predictor Variables",
            list(_available_factors.keys()),
            default=list(_available_factors.keys())[:6],
            key="mod_ols_predictors",
        )

    if ols_pred_labels and st.button("Fit OLS Model", key="mod_ols_run"):
        ols_pred_cols = tuple(_available_factors[lbl] for lbl in ols_pred_labels)
        with st.spinner("Fitting OLS model…"):
            ols_res, ols_err = _cached_ols(plot_df, ols_out_col, ols_pred_cols)

        if ols_err:
            st.error(ols_err)
        elif ols_res is not None:
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("R²",          f"{ols_res['r_sq']:.4f}")
            with m2: st.metric("Adj. R²",     f"{ols_res['adj_r_sq']:.4f}")
            with m3: st.metric("Sample Size",  f"{ols_res['n']:,}")
            with m4:
                fp = ols_res.get("f_pval", np.nan)
                st.metric("F-test p-value", _fmt_p(fp))

            sdf = ols_res["summary_df"].copy()
            st.dataframe(
                sdf.style.format({
                    "Coefficient":     "{:.4f}",
                    "Std Error":       "{:.4f}",
                    "t-stat":          "{:.3f}",
                    "p-value":         lambda v: _fmt_p(v),
                    "Robust SE (HC3)": "{:.4f}",
                    "Robust p":        lambda v: _fmt_p(v),
                    "CI Lower":        "{:.4f}",
                    "CI Upper":        "{:.4f}",
                }).map(
                    lambda v: "background-color: #fef3c7" if isinstance(v, float) and v < 0.05 else "",
                    subset=["p-value", "Robust p"],
                ),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Highlighted cells: p < 0.05. **Robust SE (HC3)** are "
                "heteroscedasticity-robust standard errors — prefer the Robust p "
                "column when residual variance is non-constant (typical for skewed "
                "county outcomes). Classical and robust p-values agreeing is a good "
                "sign of a stable result."
            )

            # Multicollinearity diagnostics for the selected predictor set
            vif_df = compute_vif(plot_df, list(ols_pred_cols))
            if not vif_df.empty:
                with st.expander("Multicollinearity check (VIF)", expanded=False):
                    st.dataframe(
                        vif_df.style.format({"VIF": "{:.2f}"}).map(
                            lambda v: ("color: #c41e3a; font-weight: 600"
                                       if isinstance(v, float) and v > 5 else ""),
                            subset=["VIF"],
                        ),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(
                        "Variance Inflation Factor: VIF > 5 (red) means the predictor is "
                        "strongly explained by the others, making its coefficient unstable; "
                        "VIF > 10 is a strong signal to drop or combine predictors."
                    )

            st.markdown("**Interpretation**")
            bullets = generate_ols_interpretation(
                sdf, ols_out_label,
                ols_res["r_sq"], ols_res["adj_r_sq"], ols_res["n"],
            )
            for b in bullets:
                st.markdown(f"- {b}")
    elif not ols_pred_labels:
        st.info("Select at least one predictor variable above.")
    else:
        st.info("Click **Fit OLS Model** to run the regression.")

    st.markdown("---")

    # SECTION 4 — COUNTY RESILIENCE SCORE

    st.markdown(
        '<div class="sub-section-header"><h3>4 — County Resilience Score</h3>'
        '<p>Cross-validated model predicts COVID deaths per 100k from structural county '
        'characteristics. '
        '<strong>Resilience Score = Predicted − Actual.</strong> '
        'Positive score → county performed better than expected. '
        'Negative → worse than expected given local conditions.</p></div>',
        unsafe_allow_html=True,
    )

    res_c1, res_c2 = st.columns(2)
    with res_c1:
        res_out_label = st.selectbox(
            "Outcome to Predict",
            list(_available_outcomes.keys()),
            index=list(_available_outcomes.keys()).index("Deaths per 100k")
                  if "Deaths per 100k" in _available_outcomes else 0,
            key="mod_res_outcome",
        )
        res_out_col = _available_outcomes[res_out_label]
    with res_c2:
        res_cv = st.selectbox("CV Folds", [3, 5, 10], index=1, key="mod_res_cv",
                               help="More folds → more stable estimates but slower computation")

    st.caption(
        "Predictors used: all available AHRF factors. "
        "Each county's prediction comes from a model trained on all OTHER counties "
        f"(strict {res_cv}-fold cross-validation — no data leakage)."
    )

    if st.button("Compute Resilience Scores", key="mod_res_run"):
        with st.spinner(f"Running {res_cv}-fold cross-validation across {len(plot_df):,} counties…"):
            res_df, res_err = _cached_resilience(
                plot_df, res_out_col, tuple(_factor_cols_all), cv_folds=res_cv,
            )

        if res_err:
            st.error(res_err)
        elif res_df is not None:
            method_used = res_df["method"].iloc[0] if "method" in res_df.columns else "—"
            n_computed  = len(res_df)
            st.caption(f"Method: {method_used} · {n_computed:,} counties scored.")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                render_metric_card("Counties Scored", f"{n_computed:,}")
            with r2:
                render_metric_card(
                    "Avg Resilience",
                    f"{res_df['resilience_score'].mean():.2f}",
                )
            with r3:
                render_metric_card(
                    "Most Resilient",
                    res_df.loc[res_df["resilience_score"].idxmax(), "County Name"]
                    if "County Name" in res_df.columns else "—",
                )
            with r4:
                render_metric_card(
                    "Least Resilient",
                    res_df.loc[res_df["resilience_score"].idxmin(), "County Name"]
                    if "County Name" in res_df.columns else "—",
                )

            st.markdown("##### Resilience Map")
            if "countyFIPS" in res_df.columns:
                map_df = res_df.copy()
                map_df["countyFIPS"] = map_df["countyFIPS"].astype(str).str.zfill(5)
                _abs_max = float(map_df["resilience_score"].abs().quantile(0.98))
                _abs_max = max(_abs_max, 1.0)

                county_label = "County Name" if "County Name" in map_df.columns else "countyFIPS"
                hover_data   = {"State": True, "actual": ":.2f",
                                "predicted": ":.2f", "resilience_score": ":.2f",
                                "countyFIPS": False}

                fig_res = px.choropleth(
                    map_df,
                    locations="countyFIPS",
                    color="resilience_score",
                    scope="usa",
                    geojson=GEO_SOURCE,
                    featureidkey="id",
                    color_continuous_scale="RdBu",
                    range_color=[-_abs_max, _abs_max],
                    hover_name=county_label if county_label in map_df.columns else None,
                    hover_data=hover_data,
                    labels={
                        "resilience_score": "Resilience Score",
                        "actual":    f"Actual {res_out_label}",
                        "predicted": f"Predicted {res_out_label}",
                    },
                    title=(
                        f"<b>County Resilience — {res_out_label}</b>"
                        f"<br><sub>Blue = better than expected · Red = worse than expected</sub>"
                    ),
                )
                fig_res.update_layout(
                    height=560,
                    margin=dict(t=70, b=10, l=0, r=0),
                    coloraxis_colorbar=dict(
                        title="Resilience<br>Score",
                        tickvals=[-_abs_max, -_abs_max/2, 0, _abs_max/2, _abs_max],
                        ticktext=[
                            f"−{_abs_max:.0f}", f"−{_abs_max/2:.0f}",
                            "0 (expected)", f"+{_abs_max/2:.0f}", f"+{_abs_max:.0f}",
                        ],
                    ),
                    geo=dict(bgcolor="rgba(0,0,0,0)"),
                    font=dict(family="sans-serif", size=11),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                fig_res.update_traces(marker_line_width=0.1, marker_line_color="rgba(0,0,0,0.2)")
                st.plotly_chart(fig_res, use_container_width=True)

            st.markdown("##### Rankings")
            tbl_cols = [c for c in ["County Name", "State", "actual", "predicted",
                                     "resilience_score"] if c in res_df.columns]
            tbl_rename = {
                "actual":           f"Actual {res_out_label}",
                "predicted":        f"Predicted {res_out_label}",
                "resilience_score": "Resilience Score",
            }
            tbl_fmt = {
                f"Actual {res_out_label}":    "{:.2f}",
                f"Predicted {res_out_label}": "{:.2f}",
                "Resilience Score":           "{:.3f}",
            }

            top_col, bot_col = st.columns(2)
            with top_col:
                st.markdown("**Top 25 — Most Resilient** (better than expected)")
                top25 = (
                    res_df.nlargest(25, "resilience_score")[tbl_cols]
                    .rename(columns=tbl_rename)
                )
                st.dataframe(top25.style.format(tbl_fmt), use_container_width=True, hide_index=True)

            with bot_col:
                st.markdown("**Bottom 25 — Least Resilient** (worse than expected)")
                bot25 = (
                    res_df.nsmallest(25, "resilience_score")[tbl_cols]
                    .rename(columns=tbl_rename)
                )
                st.dataframe(bot25.style.format(tbl_fmt), use_container_width=True, hide_index=True)

            with st.expander("Full resilience ranking (all counties)", expanded=False):
                full_tbl = (
                    res_df.sort_values("resilience_score", ascending=False)[tbl_cols]
                    .rename(columns=tbl_rename)
                    .reset_index(drop=True)
                )
                full_tbl.insert(0, "Rank", range(1, len(full_tbl) + 1))
                st.dataframe(
                    full_tbl.style.format(tbl_fmt),
                    use_container_width=True, hide_index=True,
                )

            st.caption(
                "Resilience Score = Predicted − Actual. "
                "Predictions are cross-validated: each county's score uses a model "
                "trained entirely on other counties. "
                "Interpretation is exploratory — structural characteristics not in the model "
                "(e.g. policy responses, testing capacity) also affect outcomes."
            )
    else:
        st.info("Click **Compute Resilience Scores** to run the cross-validated model.")

    st.markdown("---")

    # SECTION 5 — VACCINATION EFFICACY ANALYSIS

    _vax_eff_cols = ["vax_complete_pct", "vax_dose1_pct", "vax_booster_pct"]
    _vax_eff_present = [c for c in _vax_eff_cols if c in plot_df.columns]

    if _vax_eff_present:
        st.markdown(
            '<div class="sub-section-header"><h3>5 — Vaccination Efficacy Analysis</h3>'
            "<p>Cross-county scatter analysis: were higher-vaccinated counties associated "
            "with lower COVID mortality? Each dot is a county. Use the controls to explore "
            "different vaccination metrics and outcomes.</p></div>",
            unsafe_allow_html=True,
        )

        _ve_c1, _ve_c2, _ve_c3 = st.columns(3)
        with _ve_c1:
            _ve_vax_lbl = st.selectbox(
                "Vaccination Metric (x-axis)",
                [l for l, c in _available_factors.items() if c in _vax_eff_cols],
                key="ve_vax_metric",
            )
            _ve_vax_col = _available_factors.get(_ve_vax_lbl, "vax_complete_pct")
        with _ve_c2:
            _ve_out_lbl = st.selectbox(
                "Outcome (y-axis)",
                [l for l, c in _available_outcomes.items()
                 if c in ("deaths_per_100k", "cases_per_100k", "case_fatality_rate")],
                key="ve_out_metric",
            )
            _ve_out_col = _available_outcomes.get(_ve_out_lbl, "deaths_per_100k")
        with _ve_c3:
            _ve_color_by = st.selectbox(
                "Color by",
                ["Metro/Nonmetro", "Census Region", "None"],
                key="ve_color_by",
            )

        _ve_cols_needed = [_ve_vax_col, _ve_out_col, "County Name", "State"]
        if _ve_color_by == "Metro/Nonmetro" and "rucc_group" in plot_df.columns:
            _ve_cols_needed.append("rucc_group")
        if _ve_color_by == "Census Region" and "census_region_name" in plot_df.columns:
            _ve_cols_needed.append("census_region_name")

        _ve_df = plot_df[[c for c in _ve_cols_needed if c in plot_df.columns]].dropna(
            subset=[_ve_vax_col, _ve_out_col]
        ).copy()

        if len(_ve_df) >= 20:
            # Trendline via numpy polyfit (no scipy needed)
            _ve_x = _ve_df[_ve_vax_col].values
            _ve_y = _ve_df[_ve_out_col].values
            _ve_m, _ve_b = np.polyfit(_ve_x, _ve_y, 1)
            _ve_r = float(np.corrcoef(_ve_x, _ve_y)[0, 1])
            _ve_r2 = _ve_r ** 2

            _ve_color_col = None
            if _ve_color_by == "Metro/Nonmetro" and "rucc_group" in _ve_df.columns:
                _ve_color_col = "rucc_group"
            elif _ve_color_by == "Census Region" and "census_region_name" in _ve_df.columns:
                _ve_color_col = "census_region_name"

            _ve_hover = (
                "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                f"{_ve_vax_lbl}: %{{x:.1f}}%<br>"
                f"{_ve_out_lbl}: %{{y:.2f}}<extra></extra>"
            )

            _ve_fig = go.Figure()

            if _ve_color_col:
                _ve_groups = sorted(_ve_df[_ve_color_col].dropna().unique())
                _palette = ["#153A66", "#F26A21", "#059669", "#8B5CF6", "#DC2626",
                            "#0891B2", "#D97706", "#16A34A"]
                for _gi, _grp in enumerate(_ve_groups):
                    _gdf = _ve_df[_ve_df[_ve_color_col] == _grp]
                    _ve_fig.add_trace(go.Scatter(
                        x=_gdf[_ve_vax_col], y=_gdf[_ve_out_col],
                        mode="markers", name=str(_grp),
                        marker=dict(color=_palette[_gi % len(_palette)], size=5, opacity=0.65),
                        customdata=_gdf[["County Name", "State"]].values,
                        hovertemplate=_ve_hover,
                    ))
            else:
                _ve_fig.add_trace(go.Scatter(
                    x=_ve_df[_ve_vax_col], y=_ve_df[_ve_out_col],
                    mode="markers", name="Counties",
                    marker=dict(color=NATIONAL_COLOR, size=5, opacity=0.55),
                    customdata=_ve_df[["County Name", "State"]].values,
                    hovertemplate=_ve_hover,
                ))

            # OLS trendline
            _ve_x_line = np.linspace(_ve_x.min(), _ve_x.max(), 200)
            _ve_y_line = _ve_m * _ve_x_line + _ve_b
            _ve_fig.add_trace(go.Scatter(
                x=_ve_x_line, y=_ve_y_line,
                mode="lines", name=f"OLS trend (r={_ve_r:+.2f})",
                line=dict(color="#c41e3a", width=2, dash="dash"),
                hoverinfo="skip",
            ))

            _ve_fig.update_layout(
                title=dict(
                    text=(
                        f"<b>{_ve_vax_lbl} vs. {_ve_out_lbl}</b>"
                        f"<br><sub>r = {_ve_r:+.2f} · R² = {_ve_r2:.3f} · N = {len(_ve_df):,} counties</sub>"
                    ),
                    font=dict(size=14),
                ),
                xaxis=dict(
                    title=f"{_ve_vax_lbl} (%)",
                    showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                ),
                yaxis=dict(
                    title=_ve_out_lbl,
                    showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                ),
                hovermode="closest",
                height=520,
                template="plotly_white",
                legend=dict(
                    x=0.01, y=0.99,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1,
                ),
                font=dict(family="sans-serif", size=11),
            )
            st.plotly_chart(_ve_fig, use_container_width=True)

            # Interpretation callout
            _ve_interp_dir = "negative" if _ve_m < 0 else "positive"
            _ve_interp_strength = (
                "strong" if abs(_ve_r) > 0.5
                else "moderate" if abs(_ve_r) > 0.3
                else "weak"
            )
            _ve_cav1, _ve_cav2, _ve_cav3 = st.columns(3)
            with _ve_cav1:
                render_metric_card("Correlation (r)", f"{_ve_r:+.3f}")
            with _ve_cav2:
                render_metric_card("R² (variance explained)", f"{_ve_r2:.3f}")
            with _ve_cav3:
                render_metric_card("Counties", f"{len(_ve_df):,}")
            st.caption(
                f"OLS slope: {_ve_m:+.4f} per percentage point of vaccination. "
                f"A **{_ve_interp_strength} {_ve_interp_dir}** association is observed. "
                "This is a cross-sectional ecological analysis — counties with higher "
                "vaccination rates may differ from lower-vaccinated counties on many other "
                "dimensions (urbanicity, income, age structure). Correlation here does not "
                "establish causal efficacy; use this alongside the Factor Importance models above."
            )
        else:
            st.info(f"At least 20 counties with both vaccination and outcome data required ({len(_ve_df)} match current filters).")

    st.markdown("---")

    # SECTION 6 — COUNTY EXPLORER

    st.markdown(
        '<div class="sub-section-header"><h3>6 — County Explorer</h3>'
        '<p>Select a county to view its COVID outcomes, structural characteristics, '
        'and how it compares to the national distribution on each factor.</p></div>',
        unsafe_allow_html=True,
    )

    exp_col, _ = st.columns([2, 3])
    with exp_col:
        exp_location = st.selectbox("Select County", locations, key="mod_exp_county")

    exp_county, exp_state = extract_county_state(exp_location)
    exp_mask = (
        (master_county_df["County Name"] == exp_county) &
        (master_county_df["State"] == exp_state)
    )
    exp_row = master_county_df[exp_mask].iloc[0] if exp_mask.any() else pd.Series(dtype=object)

    def _ev(col, fallback="N/A"):
        if col in exp_row.index:
            v = exp_row[col]
            return v if pd.notna(v) else fallback
        return fallback

    def _efmt(col, fmt=".2f", fallback="N/A"):
        v = _ev(col, None)
        if v is None:
            return fallback
        try:
            return f"{float(v):{fmt}}"
        except (TypeError, ValueError):
            return str(v)

    # COVID outcomes
    st.markdown("**COVID Outcomes**")
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    with ec1: render_metric_card("Cases per 100k",    _efmt("cases_per_100k",    ".1f"))
    with ec2: render_metric_card("Deaths per 100k",   _efmt("deaths_per_100k",   ".2f"))
    with ec3: render_metric_card("Case Fatality Rate", f"{_efmt('case_fatality_rate', '.2f')}%")
    with ec4: render_metric_card("Total Cases",
                                  f"{int(_ev('total_cases', 0)):,}" if pd.notna(_ev("total_cases")) else "N/A")
    with ec5: render_metric_card("Population",
                                  f"{int(_ev('population', 0)):,}" if pd.notna(_ev("population")) else "N/A")

    st.markdown("**Healthcare Capacity**")
    eh1, eh2, eh3, eh4, eh5 = st.columns(5)
    with eh1: render_metric_card("PCP /100k",           _efmt("pcp_per_100k",          ".1f"))
    with eh2: render_metric_card("MDs /100k",           _efmt("total_md_per_100k",     ".1f"))
    with eh3: render_metric_card("Hospital Beds /100k", _efmt("hospital_beds_per_100k",".1f"))
    with eh4: render_metric_card("ICU Beds /100k",      _efmt("icu_beds_per_100k",     ".1f"))
    with eh5: render_metric_card("SNF Beds /100k",      _efmt("snf_beds_per_100k",     ".1f"))

    st.markdown("**Socioeconomic Conditions**")
    es1, es2, es3, es4, es5 = st.columns(5)
    with es1:
        mfi = _ev("median_family_income")
        render_metric_card("Median Family Income",
                           f"${int(mfi):,}" if pd.notna(mfi) else "N/A")
    with es2: render_metric_card("Unemployment %",      _efmt("unemployment_rate",  ".1f"))
    with es3: render_metric_card("Child Poverty %",     _efmt("child_poverty_pct",  ".1f"))
    with es4: render_metric_card("% No HS Diploma",     _efmt("pct_no_hs_diploma",  ".1f"))
    with es5: render_metric_card("RUCC Code",
                                  str(int(_ev("rucc_code"))) if pd.notna(_ev("rucc_code")) else "N/A")

    # Percentile context
    st.markdown("**Where does this county rank nationally?**")
    pct_rows = []
    for lbl, col in list(_available_outcomes.items()) + list(_available_factors.items()):
        val = _ev(col, None)
        if val is None or not pd.notna(val):
            continue
        nat_series = pd.to_numeric(master_county_df[col], errors="coerce").dropna()
        if len(nat_series) < 10:
            continue
        pct = float(_scipy_stats.percentileofscore(nat_series.values, float(val), kind="rank"))
        pct_rows.append({
            "Variable":      lbl,
            "County Value":  float(val),
            "National Pctile": pct,
        })

    if pct_rows:
        pct_df = pd.DataFrame(pct_rows)

        def _pct_color(v):
            try:
                v = float(v)
                if v >= 75:
                    return "color: #c41e3a; font-weight:600"
                if v <= 25:
                    return "color: #059669; font-weight:600"
            except (TypeError, ValueError):
                pass
            return ""

        st.dataframe(
            pct_df.style
            .format({"County Value": "{:.2f}", "National Pctile": "{:.0f}th"})
            .map(_pct_color, subset=["National Pctile"]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Red ≥ 75th percentile · Green ≤ 25th percentile. "
            "For outcomes (cases, deaths) higher is worse; "
            "for healthcare resources higher is generally better."
        )

    st.markdown("---")

    # SECTION 7 — COUNTY ARCHETYPES (K-MEANS)

    st.markdown(
        '<div class="sub-section-header"><h3>7 — County Archetypes</h3>'
        '<p>K-means clustering groups counties into structural archetypes using '
        'demographics, economy, healthcare access, and rurality — outcomes are '
        'deliberately excluded, so comparing COVID outcomes across archetypes '
        'stays meaningful.</p></div>',
        unsafe_allow_html=True,
    )

    _arch_features = [c for c in (
        "population", "pop_density_per_sqmi", "median_family_income",
        "pct_pop_65plus", "median_age", "pcp_per_100k",
        "hospital_beds_per_100k", "pct_college_4yr", "unemployment_rate",
        "pct_urban_pop", "rucc_code",
    ) if c in plot_df.columns]

    _ac1, _ac2 = st.columns([1, 4], vertical_alignment="bottom")
    with _ac1:
        arch_k = st.selectbox("Number of archetypes (k)", [3, 4, 5, 6], index=1, key="mod_arch_k")
    with _ac2:
        run_arch = st.button("Identify archetypes", key="mod_arch_run")

    if run_arch:
        with st.spinner("Clustering counties…"):
            arch_assign, arch_profile, arch_err = _cached_clusters(
                plot_df, tuple(_arch_features), arch_k,
            )
        if arch_err:
            st.error(arch_err)
        elif arch_assign is not None:
            _ARCH_COLORS = ["#153A66", "#F26A21", "#059669", "#8B5CF6", "#DC2626", "#0891B2"]
            arch_map_df = arch_assign.copy()
            arch_map_df["countyFIPS"] = arch_map_df["countyFIPS"].astype(str).str.zfill(5)
            arch_map_df["Archetype"] = "Archetype " + (arch_map_df["cluster"] + 1).astype(str)

            fig_arch = px.choropleth(
                arch_map_df,
                locations="countyFIPS",
                color="Archetype",
                scope="usa",
                geojson=GEO_SOURCE,
                featureidkey="id",
                category_orders={"Archetype": [f"Archetype {i+1}" for i in range(arch_k)]},
                color_discrete_sequence=_ARCH_COLORS[:arch_k],
                hover_name="County Name" if "County Name" in arch_map_df.columns else None,
                hover_data={"State": True, "countyFIPS": False},
                title="<b>County Structural Archetypes</b>",
            )
            fig_arch.update_layout(
                height=560,
                margin=dict(t=60, b=10, l=0, r=0),
                font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=11),
                paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.05),
            )
            st.plotly_chart(fig_arch, use_container_width=True)

            if arch_profile is not None:
                _profile_disp = arch_profile.copy()
                _profile_disp.insert(0, "Archetype",
                                     "Archetype " + (_profile_disp["cluster"] + 1).astype(str))
                _profile_disp = _profile_disp.drop(columns=["cluster"])
                _prof_labels = {
                    "counties":               "Counties",
                    "population":             "Mean Population",
                    "pop_density_per_sqmi":   "Density /sq mi",
                    "median_family_income":   "Median Income ($)",
                    "pct_pop_65plus":         "% 65+",
                    "median_age":             "Median Age",
                    "pcp_per_100k":           "PCP /100k",
                    "hospital_beds_per_100k": "Hosp. Beds /100k",
                    "pct_college_4yr":        "% College",
                    "unemployment_rate":      "Unemp. (%)",
                    "pct_urban_pop":          "% Urban",
                    "rucc_code":              "RUCC",
                    "cases_per_100k":         "Cases /100k",
                    "deaths_per_100k":        "Deaths /100k",
                    "case_fatality_rate":     "CFR (%)",
                    "vax_complete_pct":       "Fully Vacc. (%)",
                }
                _profile_disp = _profile_disp.rename(columns=_prof_labels)
                st.markdown("##### Archetype Profiles (means)")
                st.dataframe(
                    _profile_disp.style.format({
                        c: "{:,.1f}" for c in _profile_disp.columns
                        if c not in ("Archetype", "Counties")
                    } | {"Counties": "{:,.0f}", "Mean Population": "{:,.0f}",
                         "Median Income ($)": "{:,.0f}"}),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Clustering features are z-score standardized (population "
                    "log-transformed). The outcome columns on the right are NOT used "
                    "for clustering — differences in them across archetypes are the "
                    "finding, not the input. Cluster numbering is arbitrary."
                )
    else:
        st.info("Choose k and click **Identify archetypes** to run the clustering.")

# Tab layout

tab_overview, tab_map, tab_county_comparison, tab_waves, tab_lag, tab_factors, tab_modeling = st.tabs([
    "County Overview",
    "Geographic Map",
    "County Comparison",
    "Wave Analysis",
    "Time Lag Analysis",
    "County Factors",
    "Statistical Modeling",
])

with tab_overview:
    render_county_overview_tab(
        cases_df, deaths_df, population_df, locations, dates,
        master_county_df, transforms, vax_ts_df,
    )

with tab_map:
    render_map_tab(transforms, cases_df, deaths_df, population_df, dates, unique_states, selected_date, county_type_df, vax_latest_df)

with tab_county_comparison:
    render_comparison_tab(cases_df, deaths_df, population_df, locations, national, vax_ts_df)

with tab_waves:
    render_wave_tab(cases_df, deaths_df, transforms, locations, population_df, vax_ts_df)

with tab_lag:
    render_lag_tab(cases_df, deaths_df, population_df, locations)

with tab_factors:
    render_county_factors_tab(master_county_df, cases_df, deaths_df, population_df, transforms)

with tab_modeling:
    render_modeling_tab(master_county_df, locations)

# Footer

render_footer()
