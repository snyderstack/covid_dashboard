import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from tools import (
    load_data,
    prepare_county_timeseries,
    calculate_daily_changes,
    apply_moving_average,
    calculate_per_capita,
    prepare_lag_analysis,
    get_county_lag_comparison,
    precompute_daily_diffs,
    precompute_moving_averages,
    precompute_all_moving_averages,
    precompute_per_capita,
    get_available_dates,
    prepare_choropleth_for_date,
    filter_choropleth_by_state,
    get_state_bounds_for_zoom,
    compute_national_timeseries,
    compute_national_daily,
    compute_national_per_capita,
)

# ===== PAGE CONFIG & STYLING =====

st.set_page_config(
    page_title="COVID-19 County Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main container padding */
    .main {
        padding-top: 0;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 1.5rem;
        border-radius: 0;
        color: white;
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }

    .header-subtitle {
        font-size: 1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
        color: white;
    }

    .header-updated {
        font-size: 0.85rem;
        margin-top: 1rem;
        opacity: 0.8;
        color: white;
    }

    /* KPI Cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }

    /* Section divider */
    .section-divider {
        margin: 2rem 0 1.5rem 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        font-size: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 2rem 1.5rem;
        border-top: 1px solid #e0e0e0;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ===== HELPER UI FUNCTIONS =====

def render_header(latest_date=None):
    """Render professional dashboard header."""
    date_text = f"Data through: {latest_date}" if latest_date else "Data through: [Latest]"
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">COVID-19 County Analysis Dashboard</h1>
        <p class="header-subtitle">Real-time surveillance of COVID-19 cases and deaths across US counties</p>
        <p class="header-updated">{date_text} | Interactive analysis of USAFacts county-level data</p>
    </div>
    """, unsafe_allow_html=True)

def render_section_header(title, description=""):
    """Render a professional section header."""
    st.markdown(f"""
    <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
        <h3 style="margin: 0; font-size: 1.3rem; font-weight: 600; color: #1e3c72;">{title}</h3>
        {f'<p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, suffix=""):
    """Render a single KPI metric card."""
    # Format value before embedding in f-string
    if pd.isna(value):
        formatted_value = "N/A"
    elif isinstance(value, int):
        formatted_value = f"{value:,}"
    elif isinstance(value, float):
        formatted_value = f"{value:,.1f}"
    else:
        formatted_value = str(value)
    
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 0; color: #666; font-size: 0.9rem; font-weight: 500;">{label}</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: 700; color: #1e3c72;">
            {formatted_value} {suffix}
        </p>
    </div>
    """, unsafe_allow_html=True)

def apply_chart_styling(fig):
    """Apply consistent professional styling to Plotly charts."""
    fig.update_layout(
        font=dict(family="sans-serif", size=11, color="#333"),
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=40, b=50),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.2)")
    return fig

def location_matches_state(location, state_filter):
    """Check if location matches the selected state filter."""
    if state_filter == "All States":
        return True
    return location.endswith(f", {state_filter}")

def render_footer():
    """Render professional dashboard footer."""
    st.markdown("""
    <div class="footer">
        <p><strong>Data Source:</strong> USAFacts County-Level COVID-19 Data</p>
        <p>This dashboard provides county-level analysis of confirmed COVID-19 cases and deaths in the United States.</p>
        <p style="color: #999; margin-top: 1rem;">Built with Streamlit • Powered by Plotly • Last generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M UTC") + """</p>
    </div>
    """, unsafe_allow_html=True)

# ===== DATA LOADING & PREPROCESSING =====

st.cache_data.clear()

@st.cache_data
def get_data():
    return load_data()

# Load all datasets
cases_df, deaths_df, population_df = get_data()

# Ensure Location column exists
if "Location" not in cases_df.columns:
    cases_df["Location"] = (
        cases_df["County Name"].astype(str).str.strip()
        + ", "
        + cases_df["State"].astype(str).str.strip()
    )
if "Location" not in deaths_df.columns:
    deaths_df["Location"] = (
        deaths_df["County Name"].astype(str).str.strip()
        + ", "
        + deaths_df["State"].astype(str).str.strip()
    )
if "Location" not in population_df.columns:
    population_df["Location"] = (
        population_df["County Name"].astype(str).str.strip()
        + ", "
        + population_df["State"].astype(str).str.strip()
    )

# Precompute transforms
@st.cache_data
def precompute_all_transforms(cases, deaths, population):
    """Precompute all metric transforms for choropleth and analysis."""
    with st.spinner("Computing metrics..."):
        daily_cases, daily_deaths = precompute_daily_diffs(cases, deaths)
        
        # Precompute all moving average windows (3-day, 5-day, 7-day)
        ma_results = precompute_all_moving_averages(daily_cases, daily_deaths, windows=[3, 5, 7])
        
        # Precompute per-capita
        pc_cases, pc_deaths = precompute_per_capita(cases, deaths, population)
        
        # Get available dates
        dates = get_available_dates(cases)
    
    # Build transforms dictionary
    result = {
        "daily_cases": daily_cases,
        "daily_deaths": daily_deaths,
        "pc_cases": pc_cases,
        "pc_deaths": pc_deaths,
        "dates": dates,
    }
    
    # Add all MA results
    result.update(ma_results)
    
    return result

transforms = precompute_all_transforms(cases_df, deaths_df, population_df)
dates = transforms["dates"]

# Ensure Location exists and create locations list
if "Location" not in cases_df.columns:
    cases_df["Location"] = (
        cases_df["County Name"].astype(str).str.strip()
        + ", "
        + cases_df["State"].astype(str).str.strip()
    )

locations = sorted(cases_df["Location"].unique())
unique_states = sorted(cases_df["State"].unique())

# Cache population dict
if "pop_dict" not in st.session_state:
    pop_col = [col for col in population_df.columns if col not in ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]][0]
    st.session_state.pop_dict = dict(zip(population_df["countyFIPS"], population_df[pop_col]))

# ===== HEADER =====

latest_date = dates[-1] if dates else None
render_header(latest_date)

# ===== TOP-LEVEL KPI CARDS =====

render_section_header("Key Metrics", "National overview of COVID-19 impact")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    total_cases = int(cases_df[dates[-1]].sum()) if dates[-1] in cases_df.columns else 0
    render_metric_card("Total US Cases", total_cases)

with kpi_col2:
    total_deaths = int(deaths_df[dates[-1]].sum()) if dates[-1] in deaths_df.columns else 0
    render_metric_card("Total US Deaths", total_deaths)

with kpi_col3:
    counties_tracked = len(cases_df)
    render_metric_card("Counties Tracked", counties_tracked)

with kpi_col4:
    selected_date_display = dates[-1] if dates else "N/A"
    render_metric_card("Latest Data", selected_date_display)

# ===== DATA VALIDATION AUDIT SECTION =====

with st.expander("🔍 **Data Quality Audit** - Verify calculations are correct", expanded=False):
    """
    Shows validation results for per-capita calculations, data joins, and consistency.
    If all checks pass, data is mathematically verified as correct.
    """
    from comprehensive_validation import (
        validate_manual_per_capita_calculation,
        validate_tooltip_consistency,
        validate_join_integrity,
    )
    
    audit_col1, audit_col2 = st.columns(2)
    
    with audit_col1:
        if st.button("🧪 Run Validation Audit", key="run_audit_btn"):
            st.session_state.run_audit = True
    
    if st.session_state.get("run_audit", False):
        with st.spinner("Running validation audit (this may take 30-60 seconds)..."):
            try:
                # Run audits
                pc_results = validate_manual_per_capita_calculation(
                    cases_df, deaths_df, population_df, sample_size=15, verbose=False
                )
                tooltip_results = validate_tooltip_consistency(
                    cases_df, deaths_df, population_df, sample_size=8, verbose=False
                )
                join_results = validate_join_integrity(
                    cases_df, deaths_df, population_df, verbose=False
                )
                
                # Display results
                st.success("✅ Validation audit complete!")
                
                # Per-capita validation
                st.subheader("Per-Capita Calculation Validation")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Counties Tested", pc_results["tested_counties"])
                with col2:
                    st.metric("Calculations Valid", pc_results["passed_countries"])
                with col3:
                    st.metric("Discrepancies", len(pc_results["discrepancies"]))
                
                if pc_results["discrepancies"]:
                    st.warning("⚠️ Discrepancies found in per-capita calculations:")
                    for disc in pc_results["discrepancies"][:3]:
                        st.write(f"  • {disc['county']}, {disc['state']}: diff={disc['difference']:.4f}")
                else:
                    st.success("✓ All tested per-capita calculations verified correct")
                
                # Tooltip consistency validation
                st.subheader("Tooltip Consistency Validation")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Counties Tested", tooltip_results["tested_counties"])
                with col2:
                    st.metric("Passed", tooltip_results["passed"])
                with col3:
                    st.metric("Mismatches", len(tooltip_results["mismatches"]))
                
                if tooltip_results["mismatches"]:
                    st.warning("⚠️ Tooltip mismatches found:")
                    for mismatch in tooltip_results["mismatches"][:2]:
                        st.write(f"  • {mismatch['county']}, {mismatch['state']}")
                else:
                    st.success("✓ All tooltip values consistent with underlying data")
                
                # Join integrity validation
                st.subheader("Data Join Integrity")
                join_valid = join_results["valid"]
                if join_valid:
                    st.success("✓ All joins verified correct")
                else:
                    st.error("❌ Join integrity issues found")
                
                with st.expander("Join Details", expanded=False):
                    for check_name, check_result in join_results["checks"].items():
                        st.write(f"**{check_name}**")
                        for key, value in check_result.items():
                            st.write(f"  • {key}: {value}")
                
                # Overall status
                st.markdown("---")
                all_passed = (
                    not pc_results["discrepancies"]
                    and not tooltip_results["mismatches"]
                    and join_valid
                )
                if all_passed:
                    st.success(
                        "🎯 **ALL AUDITS PASSED** - Dashboard data is verified as mathematically correct. "
                        "Every per-capita value matches the formula: (count / population) × 100,000"
                    )
                else:
                    st.error(
                        "⚠️ **ISSUES DETECTED** - Please review the discrepancies above. "
                        "Contact the dashboard administrator if issues persist."
                    )
                
            except Exception as e:
                st.error(f"Error running audit: {str(e)}")
                st.session_state.run_audit = False

# ===== SIDEBAR CONTROLS =====

st.sidebar.markdown("---")
st.sidebar.markdown("## Dashboard Controls")

with st.sidebar:
    st.markdown("### Data Selection")
    selected_date = st.select_slider(
        "Select Analysis Date",
        options=dates,
        value=dates[-1],
        help="Choose date to analyze geographic distribution"
    )

    st.markdown("### Analysis Filters")
    analysis_state = st.selectbox(
        "Default State Filter",
        ["All States"] + unique_states,
        help="Pre-select state for trend analysis (can override in tabs)"
    )

    st.markdown("---")
    st.caption("Use tabs below to explore different views. Filters can be customized within each tab.")

# ===== MAIN CONTENT TABS =====

tab_map, tab_national_comparison, tab_county_comparison, tab_trends, tab_demographics, tab_lag = st.tabs([
    "Geographic Map",
    "National Comparison",
    "County Comparison",
    "Trend Analysis",
    "Demographics",
    "Time Lag Analysis"
])

# ===== TAB 1: GEOGRAPHIC MAP =====

with tab_map:
    render_section_header(
        "Geographic Distribution",
        "County-level choropleth map of selected metric"
    )

    # Date and metric selector
    col_date, col_metric = st.columns(2)

    with col_date:
        map_selected_date = st.select_slider(
            "Select Date for Map",
            options=dates,
            value=selected_date,
            help="Choose date to view geographic distribution. Use slider to animate through time."
        )

    with col_metric:
        metric_options = {
            "Cumulative Cases": ("cases_df", cases_df),
            "Daily Cases": ("daily_cases", transforms["daily_cases"]),
            "Daily Cases (3-day MA)": ("ma3_cases", transforms["ma3_cases"]),
            "Daily Cases (5-day MA)": ("ma5_cases", transforms["ma5_cases"]),
            "Daily Cases (7-day MA)": ("ma7_cases", transforms["ma7_cases"]),
            "Cumulative Deaths": ("deaths_df", deaths_df),
            "Daily Deaths": ("daily_deaths", transforms["daily_deaths"]),
            "Daily Deaths (3-day MA)": ("ma3_deaths", transforms["ma3_deaths"]),
            "Daily Deaths (5-day MA)": ("ma5_deaths", transforms["ma5_deaths"]),
            "Daily Deaths (7-day MA)": ("ma7_deaths", transforms["ma7_deaths"]),
            "Cases per 100k": ("pc_cases", transforms["pc_cases"]),
            "Deaths per 100k": ("pc_deaths", transforms["pc_deaths"]),
        }
        metric_name = st.selectbox(
            "Select Metric",
            list(metric_options.keys()),
            index=0,
            help="Choose which metric to display on the map"
        )
        metric_key, metric_df = metric_options[metric_name]

    # Prepare choropleth data using the map-specific date selector
    choro_data = prepare_choropleth_for_date(
        metric_df, map_selected_date, cases_df, deaths_df, population_df
    )

    # Add state filter for choropleth (separate from sidebar analysis_state)
    filter_col1, filter_col2 = st.columns([2, 1])
    
    with filter_col1:
        map_state_filter = st.selectbox(
            "Filter by State (for map only)",
            ["United States"] + unique_states,
            key="map_state_filter",
            help="Show only counties in selected state, or show entire nation"
        )
    
    # Apply state filter to choropleth data
    filtered_choro_data = filter_choropleth_by_state(choro_data, map_state_filter)
    
    # Determine zoom and geo settings based on state selection
    geo_config = dict(scope="usa", projection_type="albers usa")
    
    if map_state_filter != "United States":
        state_bounds = get_state_bounds_for_zoom(filtered_choro_data, map_state_filter)
        if state_bounds:
            geo_config["center"] = {"lat": state_bounds["lat"], "lon": state_bounds["lon"]}
            geo_config["projection"] = {"scale": state_bounds["zoom"]}

    # Data integrity check (collapsible)
    with st.expander("🔍 Data Integrity Check", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Counties", len(cases_df))
        with col2:
            st.metric("Unique FIPS", cases_df["countyFIPS"].nunique())
        with col3:
            st.metric("Counties on Map", len(filtered_choro_data))
        with col4:
            coverage = 100 * len(filtered_choro_data) / cases_df["countyFIPS"].nunique() if cases_df["countyFIPS"].nunique() > 0 else 0
            st.metric("Coverage %", f"{coverage:.1f}%")

    # Ensure FIPS formatting
    filtered_choro_data["countyFIPS"] = filtered_choro_data["countyFIPS"].astype(str).str.zfill(5)

    # Create choropleth
    color_scale = "YlOrRd" if "Deaths" not in metric_name else "OrRd"
    fig_map = px.choropleth(
        filtered_choro_data,
        locations="countyFIPS",
        color="value",
        scope="usa",
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        featureidkey="id",
        color_continuous_scale=color_scale,
        hover_data={
            "countyFIPS": False,
            "Location": True,
            "population": ":,",
            "cases": ":,",
            "deaths": ":,",
            "cases_pc": ":.1f",
            "deaths_pc": ":.1f",
            "value": ":.1f"
        },
        labels={"value": metric_name}
    )

    title_text = f"<b>{metric_name} by County</b><br><sub>{map_selected_date}"
    if map_state_filter != "United States":
        title_text += f" - {map_state_filter}"
    title_text += "</sub>"
    
    fig_map.update_layout(
        title_text=title_text,
        geo=geo_config,
        height=700,
        margin={"r": 0, "t": 80, "l": 0, "b": 0},
        font=dict(family="sans-serif", size=11)
    )

    fig_map.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "Population: %{customdata[1]}<br>" +
                      "Cases: %{customdata[2]}<br>" +
                      "Deaths: %{customdata[3]}<br>" +
                      "Cases/100k: %{customdata[4]:.1f}<br>" +
                      "Deaths/100k: %{customdata[5]:.1f}<br>" +
                      f"{metric_name}: %{{customdata[6]:.1f}}<extra></extra>"
    )

    st.plotly_chart(fig_map, use_container_width=True)

# ===== TAB 2: NATIONAL VS COUNTY COMPARISON =====

with tab_national_comparison:
    render_section_header(
        "National vs County Trend Comparison",
        "Compare how a selected county compares against the national trend"
    )
    
    # Precompute national aggregates if not cached
    if "national_cases_ts" not in st.session_state:
        with st.spinner("Computing national aggregates..."):
            st.session_state.national_cases_ts = compute_national_timeseries(cases_df, deaths_df, "Cases")
            st.session_state.national_deaths_ts = compute_national_timeseries(cases_df, deaths_df, "Deaths")
            st.session_state.national_daily_cases = compute_national_daily(transforms["daily_cases"])
            st.session_state.national_daily_deaths = compute_national_daily(transforms["daily_deaths"])
            st.session_state.national_pc_cases = compute_national_per_capita(transforms["pc_cases"], population_df)
            st.session_state.national_pc_deaths = compute_national_per_capita(transforms["pc_deaths"], population_df)
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        comparison_county = st.selectbox(
            "Select County",
            locations,
            key="comparison_county",
            help="Choose a county to compare against national trend"
        )
    
    with col2:
        comparison_metric = st.selectbox(
            "Metric",
            ["Cases", "Deaths", "Cases per 100k", "Deaths per 100k"],
            key="comparison_metric",
            help="Choose metric for comparison"
        )
    
    with col3:
        comparison_view = st.selectbox(
            "View",
            ["Cumulative", "Daily"],
            key="comparison_view",
            help="Show cumulative totals or daily new"
        )
    
    with col4:
        comparison_ma = st.selectbox(
            "Smoothing",
            ["None", "3-day MA", "5-day MA", "7-day MA"],
            key="comparison_ma",
            help="Apply moving average smoothing"
        )
    
    # Extract county name and state from location
    if ", " in comparison_county:
        county_name, state = comparison_county.rsplit(", ", 1)
    else:
        county_name = comparison_county
        state = None
    
    # Get county data
    if comparison_view == "Cumulative":
        if "cases" in comparison_metric.lower():
            county_ts = prepare_county_timeseries(cases_df, county_name, state, "County")
            national_ts = st.session_state.national_cases_ts.copy()
            national_ts.columns = ["Date", "National"]
        else:
            county_ts = prepare_county_timeseries(deaths_df, county_name, state, "County")
            national_ts = st.session_state.national_deaths_ts.copy()
            national_ts.columns = ["Date", "National"]
    else:  # Daily
        if "cases" in comparison_metric.lower():
            county_ts = prepare_county_timeseries(cases_df, county_name, state, "Cases")
            county_ts = calculate_daily_changes(county_ts, "Cases")
            county_ts = county_ts[["Date", "Daily Cases"]].rename(columns={"Daily Cases": "County"})
            national_ts = st.session_state.national_daily_cases.copy()
            national_ts.columns = ["Date", "National"]
        else:
            county_ts = prepare_county_timeseries(deaths_df, county_name, state, "Deaths")
            county_ts = calculate_daily_changes(county_ts, "Deaths")
            county_ts = county_ts[["Date", "Daily Deaths"]].rename(columns={"Daily Deaths": "County"})
            national_ts = st.session_state.national_daily_deaths.copy()
            national_ts.columns = ["Date", "National"]
    
    # Handle per-capita if selected
    if "per 100k" in comparison_metric.lower():
        # Get population for county
        pop_row = population_df[(population_df["County Name"] == county_name) & 
                               (population_df["State"] == state)]
        if not pop_row.empty:
            pop_col = [col for col in population_df.columns 
                      if col not in ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]][0]
            population = pop_row.iloc[0][pop_col]
            if population > 0:
                county_ts["County"] = (county_ts["County"] / population) * 100000
        
        # National per-capita
        if comparison_view == "Cumulative":
            if "cases" in comparison_metric.lower():
                national_ts = st.session_state.national_pc_cases.copy()
            else:
                national_ts = st.session_state.national_pc_deaths.copy()
            national_ts.columns = ["Date", "National"]
        else:
            # For daily per-capita, we need to recalculate
            # Use same approach: daily value / total population * 100k
            pass  # Use computed national average per-capita
    
    # Merge county and national
    comparison_data = county_ts.merge(national_ts, on="Date", how="inner")
    
    # Apply smoothing if selected
    if comparison_ma != "None":
        window = int(comparison_ma.split("-")[0])
        comparison_data["County"] = comparison_data["County"].rolling(window=window, min_periods=1).mean()
        comparison_data["National"] = comparison_data["National"].rolling(window=window, min_periods=1).mean()
    
    # Create plot
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=comparison_data["Date"],
        y=comparison_data["National"],
        name="National",
        mode="lines",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="<b>National</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f}<extra></extra>"
    ))
    fig_comp.add_trace(go.Scatter(
        x=comparison_data["Date"],
        y=comparison_data["County"],
        name=comparison_county,
        mode="lines",
        line=dict(color="#ff7f0e", width=2),
        hovertemplate=f"<b>{comparison_county}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.1f}}<extra></extra>"
    ))
    
    fig_comp.update_layout(
        title=f"<b>{comparison_metric} Trend Comparison</b><br><sub>{comparison_view} - {comparison_ma}</sub>",
        xaxis_title="Date",
        yaxis_title=comparison_metric,
        hovermode="x unified",
        height=600,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        county_latest = comparison_data["County"].iloc[-1] if len(comparison_data) > 0 else 0
        st.metric(f"{comparison_county} (Latest)", f"{county_latest:.1f}")
    with col2:
        national_latest = comparison_data["National"].iloc[-1] if len(comparison_data) > 0 else 0
        st.metric("National (Latest)", f"{national_latest:.1f}")
    with col3:
        if national_latest > 0:
            ratio = county_latest / national_latest
            st.metric("County/National Ratio", f"{ratio:.2f}x")
        else:
            st.metric("County/National Ratio", "N/A")


# ===== TAB 3: COUNTY VS COUNTY COMPARISON =====

with tab_county_comparison:
    render_section_header(
        "County vs County Trend Comparison",
        "Compare trends between two selected counties"
    )
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        county_a = st.selectbox(
            "County A",
            locations,
            key="county_a",
            index=0,
            help="Select first county"
        )
    
    with col2:
        county_b = st.selectbox(
            "County B",
            locations,
            key="county_b",
            index=min(1, len(locations) - 1),
            help="Select second county"
        )
    
    with col3:
        dual_metric = st.selectbox(
            "Metric",
            ["Cases", "Deaths", "Cases per 100k", "Deaths per 100k"],
            key="dual_metric"
        )
    
    with col4:
        dual_view = st.selectbox(
            "View",
            ["Cumulative", "Daily"],
            key="dual_view"
        )
    
    # Second row of controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dual_ma = st.selectbox(
            "Smoothing",
            ["None", "3-day MA", "5-day MA", "7-day MA"],
            key="dual_ma"
        )
    
    with col2:
        st.write("")  # Spacer
    
    with col3:
        st.write("")  # Spacer
    
    # Extract county details
    def extract_county_state(location):
        if ", " in location:
            county, state = location.rsplit(", ", 1)
            return county, state
        return location, None
    
    county_a_name, county_a_state = extract_county_state(county_a)
    county_b_name, county_b_state = extract_county_state(county_b)
    
    # Get data for both counties
    def get_county_comparison_data(county_name, state, metric, view, ma_window="None"):
        if view == "Cumulative":
            if "cases" in metric.lower():
                ts = prepare_county_timeseries(cases_df, county_name, state, "Cases")
            else:
                ts = prepare_county_timeseries(deaths_df, county_name, state, "Deaths")
        else:  # Daily
            if "cases" in metric.lower():
                ts = prepare_county_timeseries(cases_df, county_name, state, "Cases")
                ts = calculate_daily_changes(ts, "Cases")
                ts = ts[["Date", "Daily Cases"]].rename(columns={"Daily Cases": "Value"})
            else:
                ts = prepare_county_timeseries(deaths_df, county_name, state, "Deaths")
                ts = calculate_daily_changes(ts, "Deaths")
                ts = ts[["Date", "Daily Deaths"]].rename(columns={"Daily Deaths": "Value"})
        
        # Handle per-capita
        if "per 100k" in metric.lower():
            pop_row = population_df[(population_df["County Name"] == county_name) & 
                                   (population_df["State"] == state)]
            if not pop_row.empty:
                pop_col = [col for col in population_df.columns 
                          if col not in ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]][0]
                population = pop_row.iloc[0][pop_col]
                if population > 0:
                    ts["Value"] = (ts["Value"] / population) * 100000
        
        # Apply smoothing
        if ma_window != "None":
            window = int(ma_window.split("-")[0])
            ts["Value"] = ts["Value"].rolling(window=window, min_periods=1).mean()
        
        return ts
    
    try:
        data_a = get_county_comparison_data(county_a_name, county_a_state, dual_metric, dual_view, dual_ma)
        data_b = get_county_comparison_data(county_b_name, county_b_state, dual_metric, dual_view, dual_ma)
        
        if data_a.empty or data_b.empty:
            st.warning("Could not find data for one or both counties")
        else:
            # Create comparison plot
            fig_dual = go.Figure()
            
            fig_dual.add_trace(go.Scatter(
                x=data_a["Date"],
                y=data_a["Value"],
                name=county_a,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                hovertemplate=f"<b>{county_a}</b><br>Date: %{{x|%Y-%m-%d}}<br>{dual_metric}: %{{y:.1f}}<extra></extra>"
            ))
            
            fig_dual.add_trace(go.Scatter(
                x=data_b["Date"],
                y=data_b["Value"],
                name=county_b,
                mode="lines",
                line=dict(color="#ff7f0e", width=2),
                hovertemplate=f"<b>{county_b}</b><br>Date: %{{x|%Y-%m-%d}}<br>{dual_metric}: %{{y:.1f}}<extra></extra>"
            ))
            
            fig_dual.update_layout(
                title=f"<b>{dual_metric} Comparison</b><br><sub>{dual_view} - {dual_ma}</sub>",
                xaxis_title="Date",
                yaxis_title=dual_metric,
                hovermode="x unified",
                height=600,
                template="plotly_white",
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
            )
            
            st.plotly_chart(fig_dual, use_container_width=True)
            
            # Statistics comparison
            col1, col2, col3, col4 = st.columns(4)
            
            if len(data_a) > 0:
                a_latest = data_a["Value"].iloc[-1]
                with col1:
                    st.metric(f"{county_a} (Latest)", f"{a_latest:.1f}")
            
            if len(data_b) > 0:
                b_latest = data_b["Value"].iloc[-1]
                with col2:
                    st.metric(f"{county_b} (Latest)", f"{b_latest:.1f}")
            
            if len(data_a) > 0 and len(data_b) > 0 and data_a["Value"].iloc[-1] > 0:
                ratio = b_latest / a_latest
                with col3:
                    st.metric(f"{county_b} / {county_a}", f"{ratio:.2f}x")
            
            if len(data_a) > 0 and len(data_b) > 0:
                a_change = ((data_a["Value"].iloc[-1] - data_a["Value"].iloc[0]) / max(data_a["Value"].iloc[0], 1)) * 100
                b_change = ((data_b["Value"].iloc[-1] - data_b["Value"].iloc[0]) / max(data_b["Value"].iloc[0], 1)) * 100
                with col4:
                    st.metric("% Change (A vs B)", f"{a_change:.0f}% vs {b_change:.0f}%")
    
    except Exception as e:
        st.error(f"Error loading county data: {str(e)}")

# ===== TAB 4: TREND ANALYSIS =====

with tab_trends:
    render_section_header(
        "County Trend Analysis",
        "Analyze trends with flexible options: single county, county comparisons, or vs national"
    )

    # Mode selector (new feature)
    trend_mode = st.selectbox(
        "Analysis Mode",
        ["Single County", "County vs County", "County vs National"],
        key="trend_mode",
        help="Choose comparison mode"
    )

    st.markdown("---")

    # Control columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # County selector
        filtered_locations = [loc for loc in locations if location_matches_state(loc, analysis_state)] if analysis_state != "All States" else locations
        
        if trend_mode == "Single County":
            location = st.selectbox(
                "Select County",
                filtered_locations,
                key="plots_county",
                help="Choose a county to analyze"
            )
        elif trend_mode == "County vs County":
            location = st.selectbox(
                "County A",
                filtered_locations,
                key="plots_county_a",
                help="Choose first county"
            )
        else:  # County vs National
            location = st.selectbox(
                "Select County",
                filtered_locations,
                key="plots_county_national",
                help="Choose county to compare against national"
            )

    with col2:
        if trend_mode == "County vs County":
            location_b = st.selectbox(
                "County B",
                filtered_locations,
                key="plots_county_b",
                index=min(1, len(filtered_locations) - 1),
                help="Choose second county"
            )
        else:
            location_b = None
        
        # Metric selector
        trend_metric = st.selectbox(
            "Metric",
            ["Cases", "Deaths"],
            key="trend_metric",
            help="Choose metric to analyze"
        )

    with col3:
        # View type selector
        trend_view = st.selectbox(
            "View Type",
            ["Cumulative", "Daily"],
            key="trend_view",
            help="Cumulative shows total over time; Daily shows new cases/deaths per day"
        )

    # Smoothing options
    col_norm, col_smooth = st.columns(2)

    with col_norm:
        # Normalization only applicable to raw daily/cumulative
        trend_normalization = st.selectbox(
            "Normalization",
            ["Raw", "Per 100k"],
            key="trend_normalization",
            help="Adjust for population size"
        )

    with col_smooth:
        # Smoothing selector - only show when daily view is selected
        if trend_view == "Daily":
            trend_smoothing = st.selectbox(
                "Smoothing",
                ["None", "3-day MA", "5-day MA", "7-day MA"],
                index=2,  # Default to 7-day
                key="trend_smoothing",
                help="Apply moving average to smooth daily trends"
            )
        else:
            trend_smoothing = "None"
            st.caption("(Smoothing only available for Daily view)")

    # Extract county info
    county_name, state = location.rsplit(", ", 1)

    # Prepare base timeseries
    base_metric_col = "Cases" if trend_metric == "Cases" else "Deaths"
    timeseries = prepare_county_timeseries(cases_df if trend_metric == "Cases" else deaths_df, 
                                            county_name, state, base_metric_col)

    if not timeseries.empty:
        # Apply normalization if requested
        if trend_normalization == "Per 100k":
            timeseries = calculate_per_capita(timeseries, population_df, county_name, state)
            norm_suffix = " (per 100k)"
        else:
            norm_suffix = ""

        # Determine what to plot
        if trend_view == "Cumulative":
            # Plot cumulative directly
            plot_col = "Per Capita" if trend_normalization == "Per 100k" else base_metric_col
            plot_label = f"Cumulative {trend_metric}{norm_suffix}"
            title_suffix = f"Cumulative {trend_metric}{norm_suffix}"
            
            fig_trend = px.line(
                timeseries,
                x="Date",
                y=plot_col,
                title=f"<b>{title_suffix} in {location}</b>",
                labels={plot_col: plot_label}
            )
        else:
            # Daily view - calculate daily changes and optionally smooth
            timeseries = calculate_daily_changes(timeseries, base_metric_col)
            
            # Apply smoothing if requested
            if trend_smoothing != "None":
                window_map = {"3-day MA": 3, "5-day MA": 5, "7-day MA": 7}
                window = window_map[trend_smoothing]
                timeseries = apply_moving_average(timeseries, f"Daily {base_metric_col}", window=window)
                plot_col = f"Daily {base_metric_col} MA"
                plot_label = f"Daily {trend_metric} ({trend_smoothing}){norm_suffix}"
                title_suffix = f"Daily {trend_metric} ({trend_smoothing}){norm_suffix}"
            else:
                plot_col = f"Daily {base_metric_col}"
                plot_label = f"Daily {trend_metric}{norm_suffix}"
                title_suffix = f"Daily {trend_metric}{norm_suffix}"
            
            fig_trend = px.line(
                timeseries,
                x="Date",
                y=plot_col,
                title=f"<b>{title_suffix} in {location}</b>",
                labels={plot_col: plot_label}
            )

        fig_trend = apply_chart_styling(fig_trend)
        fig_trend.update_traces(line=dict(color="#2a5298", width=3))

        st.plotly_chart(fig_trend, use_container_width=True)

        # Raw data display
        with st.expander("Show Raw Data", expanded=False):
            display_cols = ["Date"]
            
            if trend_view == "Cumulative":
                display_cols.append(base_metric_col)
                if "Per Capita" in timeseries.columns:
                    display_cols.append("Per Capita")
            else:
                display_cols.append(base_metric_col)
                display_cols.append(f"Daily {base_metric_col}")
                if f"Daily {base_metric_col} MA" in timeseries.columns:
                    display_cols.append(f"Daily {base_metric_col} MA")
            
            display_data = timeseries[[col for col in display_cols if col in timeseries.columns]]
            
            st.dataframe(
                display_data.style.format({
                    col: "{:,.1f}" if "MA" in col or "Per Capita" in col else "{:,.0f}"
                    for col in display_data.columns if col != "Date"
                }),
                use_container_width=True
            )
    else:
        st.warning("No data available for selected county")

# ===== TAB 5: DEMOGRAPHICS =====

with tab_demographics:
    render_section_header(
        "Population-Normalized Analysis",
        "Cases and deaths adjusted for population size (per 100k) - Compare multiple counties"
    )

    col1, col2 = st.columns(2)

    with col1:
        # Allow multiple county selection
        selected_locations = st.multiselect(
            "Select Counties to Compare",
            locations,
            default=[locations[0]] if locations else [],
            key="demographics_counties",
            help="Choose one or more counties for comparison"
        )

    with col2:
        demo_metric_type = st.radio(
            "View",
            ["Raw Counts", "Per 100k Population"],
            horizontal=True,
            key="demo_metric_type"
        )

    if selected_locations:
        # Build comparison data
        comparison_rows = []
        
        for location in selected_locations:
            if ", " in location:
                county_name, state = location.rsplit(", ", 1)
            else:
                county_name, state = location, None
            
            try:
                # Get population
                pop_row = population_df[
                    (population_df["County Name"] == county_name) & 
                    (population_df["State"] == state)
                ]
                
                identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
                pop_col = [col for col in population_df.columns if col not in identifier_cols and col != "Location"]
                
                population = 0
                if not pop_row.empty and pop_col:
                    population = pd.to_numeric(pop_row.iloc[0][pop_col[0]], errors="coerce")
                    if pd.isna(population):
                        population = 0
                
                # Get latest cases and deaths
                cases_row = cases_df[
                    (cases_df["County Name"] == county_name) & 
                    (cases_df["State"] == state)
                ]
                deaths_row = deaths_df[
                    (deaths_df["County Name"] == county_name) & 
                    (deaths_df["State"] == state)
                ]
                
                total_cases = 0
                total_deaths = 0
                
                if not cases_row.empty:
                    # Get latest value (last column is most recent date)
                    latest_date = dates[-1] if dates else None
                    if latest_date:
                        total_cases = pd.to_numeric(cases_row.iloc[0][latest_date], errors="coerce")
                        if pd.isna(total_cases):
                            total_cases = 0
                
                if not deaths_row.empty:
                    latest_date = dates[-1] if dates else None
                    if latest_date:
                        total_deaths = pd.to_numeric(deaths_row.iloc[0][latest_date], errors="coerce")
                        if pd.isna(total_deaths):
                            total_deaths = 0
                
                # Calculate per-capita
                cases_pc = (total_cases / population * 100000) if population > 0 else 0
                deaths_pc = (total_deaths / population * 100000) if population > 0 else 0
                
                comparison_rows.append({
                    "County": location,
                    "Population": int(population) if population > 0 else 0,
                    "Total Cases": int(total_cases),
                    "Total Deaths": int(total_deaths),
                    "Cases per 100k": round(cases_pc, 1),
                    "Deaths per 100k": round(deaths_pc, 1),
                })
            
            except Exception as e:
                st.warning(f"Error processing {location}: {str(e)}")
        
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            
            # Format display based on view type
            if demo_metric_type == "Raw Counts":
                display_df = comparison_df[["County", "Population", "Total Cases", "Total Deaths"]]
                st.subheader("Raw Case and Death Counts")
            else:
                display_df = comparison_df[["County", "Population", "Cases per 100k", "Deaths per 100k"]]
                st.subheader("Per-Capita Rates (per 100,000 population)")
            
            # Display as styled table
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Show full comparison table in expander
            with st.expander("View Full Comparison Data", expanded=False):
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Summary statistics
            if len(comparison_rows) > 1:
                st.markdown("---")
                st.markdown("### Comparison Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_cases_pc = comparison_df["Cases per 100k"].mean()
                    st.metric("Avg Cases per 100k", f"{avg_cases_pc:.1f}")
                
                with col2:
                    avg_deaths_pc = comparison_df["Deaths per 100k"].mean()
                    st.metric("Avg Deaths per 100k", f"{avg_deaths_pc:.1f}")
                
                with col3:
                    max_cases = comparison_df["Cases per 100k"].max()
                    min_cases = comparison_df["Cases per 100k"].min()
                    if min_cases > 0:
                        ratio = max_cases / min_cases
                        st.metric("Cases Range (High/Low)", f"{ratio:.1f}x")
        else:
            st.warning("No data available for selected counties")
    else:
        st.info("👈 Select one or more counties to view demographic analysis")

# ===== TAB 6: TIME LAG ANALYSIS =====

with tab_lag:
    render_section_header(
        "Epidemiological Lag Analysis",
        "Relationship between case trends and deaths with time lag"
    )

    col1, col2 = st.columns(2)

    with col1:
        location_lag = st.selectbox(
            "Select County",
            locations,
            key="timelag_county",
            help="Choose a county for lag analysis"
        )

    with col2:
        lag_days = st.slider(
            "Time Lag (days)",
            min_value=0,
            max_value=21,
            value=7,
            help="Shift deaths forward relative to cases (positive = deaths lag behind)"
        )

    county_name, state = location_lag.rsplit(", ", 1)
    lag_data = prepare_lag_analysis(cases_df, deaths_df, county_name, state, lag_days=lag_days)

    if not lag_data.empty:
        # Dual-axis plot
        fig_lag = go.Figure()

        fig_lag.add_trace(go.Scatter(
            x=lag_data["Date"],
            y=lag_data["Cases"],
            name="Cases",
            yaxis="y",
            line=dict(color="#2a5298", width=2)
        ))

        fig_lag.add_trace(go.Scatter(
            x=lag_data["Date"],
            y=lag_data["Deaths"],
            name=f"Deaths (lag: {lag_days}d)",
            yaxis="y2",
            line=dict(color="#c41e3a", width=2)
        ))

        fig_lag.update_layout(
            title=f"<b>Cases vs Deaths in {location_lag}</b><br><sub>Deaths shifted {lag_days} days forward</sub>",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Cumulative Cases", side="left"),
            yaxis2=dict(title="Cumulative Deaths", side="right", overlaying="y"),
            hovermode="x unified",
            height=500,
            font=dict(family="sans-serif", size=11),
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_lag, use_container_width=True)

        # Best lag calculation
        with st.expander("Find Optimal Lag", expanded=False):
            lag_results = get_county_lag_comparison(cases_df, deaths_df, county_name, state, max_lag=21)
            if lag_results:
                best_lag = max(lag_results, key=lag_results.get)
                best_corr = lag_results[best_lag]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Optimal Lag (days)", best_lag)
                with col2:
                    st.metric("Best Correlation", f"{best_corr:.3f}")

                st.caption(f"Peak correlation between daily cases and deaths occurs at {best_lag}-day lag (correlation: {best_corr:.3f})")

        # Daily trends
        lag_data_daily = calculate_daily_changes(lag_data, "Cases")
        lag_data_daily = calculate_daily_changes(lag_data_daily, "Deaths")

        fig_lag_daily = go.Figure()
        fig_lag_daily.add_trace(go.Scatter(
            x=lag_data_daily["Date"],
            y=lag_data_daily["Daily Cases"],
            name="Daily Cases",
            line=dict(color="#2a5298", width=2)
        ))
        fig_lag_daily.add_trace(go.Scatter(
            x=lag_data_daily["Date"],
            y=lag_data_daily["Daily Deaths"],
            name="Daily Deaths",
            line=dict(color="#c41e3a", width=2)
        ))

        fig_lag_daily.update_layout(
            title="<b>Daily Cases and Deaths</b>",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Daily Count"),
            hovermode="x unified",
            height=400,
            font=dict(family="sans-serif", size=11),
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_lag_daily, use_container_width=True)

        with st.expander("Show Raw Data", expanded=False):
            st.dataframe(
                lag_data[["Date", "Cases", "Deaths"]].style.format({
                    "Cases": "{:,.0f}",
                    "Deaths": "{:,.0f}"
                }),
                use_container_width=True
            )
    else:
        st.warning("No data available for selected county")

# ===== FOOTER =====

render_footer()

# ===== END OF DASHBOARD =====

