"""
Bike Flow Explorer - Streamlit App

Explore NYC bike ride patterns between neighborhoods (NTAs).
"""

import streamlit as st
import backend as be
import visualizations as viz
import ui_helpers as uih

st.set_page_config(page_title="Bike Flow Explorer", page_icon="🚴", layout="wide")

st.header("🚴 NYC Bike Flow Explorer")

st.markdown("""
This app lets you explore bike ride patterns between NYC neighborhoods (Neighborhood Tabulation Areas or NTAs).
Discover which neighborhoods have the most bike traffic, where riders are coming from and going to, 
and how patterns have changed over time.
""")

# Get available data for dropdowns
boroughs = ["All Boroughs"] + be.get_boroughs()
year_months = be.get_year_months()
default_period1, default_period2 = uih.get_comparison_periods(year_months)

# Let the user select data to view
st.subheader("Select a Neighborhood")
borough_col, nta_col, direction_col, period_col = st.columns(4)

with borough_col:
    borough = st.selectbox(
        "Borough:", 
        boroughs, 
        index=0,
        help="Filter neighborhoods by borough"
    )

with nta_col:
    ntas = be.get_ntas(borough if borough != "All Boroughs" else None)
    # Default to a popular NTA
    default_idx = min(uih.get_nta_index(borough), len(ntas) - 1) if ntas else 0
    selected_nta = st.selectbox(
        "Neighborhood (NTA):", 
        ntas, 
        index=default_idx,
        help="Select a Neighborhood Tabulation Area"
    )

with direction_col:
    direction = st.selectbox(
        "Direction:",
        ["outgoing", "incoming"],
        format_func=lambda x: "Outgoing (rides starting here)" if x == "outgoing" else "Incoming (rides ending here)",
        help="Choose whether to analyze rides leaving or arriving at this neighborhood"
    )

with period_col:
    selected_period = st.selectbox(
        "Period:",
        ["All Time"] + year_months,
        format_func=lambda x: x if x == "All Time" else uih.format_year_month(x),
        help="Filter data by time period"
    )

# Convert period selection to year_month format for backend functions
selected_year_month = None if selected_period == "All Time" else selected_period

# Display tabs
graph_tab, flow_tab, heatmap_tab, table_tab, map_tab, about_tab = st.tabs(
    ["📈 Trends", "🔀 Flows", "🔥 Heat Map", "🔍 Rankings", "🗺️ Overview", "ℹ️ About"]
)

with graph_tab:
    st.subheader("Ride Trends Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Monthly Time Series**")
        fig = viz.get_line_graph(selected_nta, direction)
        st.pyplot(fig)
        st.caption("*Monthly ride totals colored by year.*")
    
    with col2:
        st.markdown("**Year-over-Year Comparison**")
        fig = viz.get_monthly_comparison_graph(selected_nta, direction)
        st.pyplot(fig)
        st.caption("*Compare the same months across different years.*")

with flow_tab:
    st.subheader("Ride Flow Patterns")
    
    # Period selector for flows
    flow_period_col, _ = st.columns([1, 2])
    with flow_period_col:
        flow_period = st.selectbox(
            "Period:",
            ["All Time"] + year_months,
            key="flow_period",
            format_func=lambda x: x if x == "All Time" else uih.format_year_month(x),
            help="Filter flows by time period"
        )
    
    flow_year_month = None if flow_period == "All Time" else flow_period
    
    col1, col2 = st.columns(2)
    
    with col1:
        if direction == "outgoing":
            st.markdown(f"**Top Destinations from {selected_nta}**")
            fig = viz.get_top_destinations_chart(selected_nta, 10, flow_year_month)
        else:
            st.markdown(f"**Top Origins to {selected_nta}**")
            fig = viz.get_top_origins_chart(selected_nta, 10, flow_year_month)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Flow Diagram**")
        fig = viz.get_flow_sankey(selected_nta, direction, 10, flow_year_month)
        st.plotly_chart(fig, use_container_width=True)
    
    # Borough-level heatmap
    st.markdown("---")
    st.markdown("**Borough-to-Borough Flow**")
    fig = viz.get_borough_flow_heatmap(flow_year_month)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("*This heatmap shows the total number of rides between each pair of boroughs.*")

with heatmap_tab:
    st.subheader("🗺️ Geographic Traffic Heat Map")
    
    direction_text = "destinations from" if direction == "outgoing" else "origins to"
    period_text = f" ({uih.format_year_month(selected_period)})" if selected_period != "All Time" else " (All Time)"
    
    st.markdown(f"""
    Showing {direction_text} **{selected_nta}**{period_text}.  
    Darker shades indicate higher traffic volume.
    """)
    
    st.info(f"🔵 The selected neighborhood ({selected_nta}) is highlighted with a blue border.")
    
    # Render the choropleth map using shared controls
    fig = viz.get_nta_choropleth_map(selected_nta, selected_year_month, direction)
    st.plotly_chart(fig, use_container_width=True)
    
    direction_verb = "going to" if direction == "outgoing" else "coming from"
    st.caption(f"*Darker shades indicate more rides {direction_verb} each neighborhood.*")
    
    # Bar chart comparison below the map - show top destinations/origins for selected NTA
    st.markdown("---")
    chart_title = "Top Destinations" if direction == "outgoing" else "Top Origins"
    st.markdown(f"**{chart_title} for {selected_nta}**")
    
    if direction == "outgoing":
        fig = viz.get_top_destinations_chart(selected_nta, 15, selected_year_month)
    else:
        fig = viz.get_top_origins_chart(selected_nta, 15, selected_year_month)
    st.pyplot(fig)

with table_tab:
    st.subheader("Neighborhood Rankings")
    
    st.markdown("Compare how ride counts have changed between two time periods.")
    
    # Period selectors
    period_col1, period_col2 = st.columns(2)
    
    with period_col1:
        period1_idx = year_months.index(default_period1) if default_period1 in year_months else 0
        period1 = st.selectbox(
            "Earlier Period:",
            year_months,
            index=period1_idx,
            key="ranking_period1",
            format_func=uih.format_year_month
        )
    
    with period_col2:
        period2_idx = year_months.index(default_period2) if default_period2 in year_months else len(year_months) - 1
        period2 = st.selectbox(
            "Later Period:",
            year_months,
            index=period2_idx,
            key="ranking_period2",
            format_func=uih.format_year_month
        )
    
    # Get ranking data
    ranking_df = be.get_nta_ranking_df(period1, period2, direction)
    ranking_text = be.get_ranking_text(selected_nta, direction, ranking_df)
    
    # Display ranking info
    direction_text = "outgoing" if direction == "outgoing" else "incoming"
    st.markdown(f"""
This table shows the percent change in **{direction_text}** rides for all NTAs 
between **{uih.format_year_month(period1)}** and **{uih.format_year_month(period2)}**.

{ranking_text}

*Click a column header to sort by it.*
    """)
    
    # Apply styling and display
    styled_df = ranking_df.style.pipe(uih.apply_styles, selected_nta, period1, period2)
    st.dataframe(styled_df, use_container_width=True, height=500)
    
    # Swarm plot
    st.markdown("---")
    st.markdown("**Distribution of Changes Across All NTAs**")
    
    fig = viz.get_swarmplot(period1, period2, direction, selected_nta)
    st.pyplot(fig)
    st.caption("*Each dot represents an NTA. The highlighted dot (if visible) is your selected neighborhood.*")

with map_tab:
    st.subheader("Neighborhood Overview")
    
    map_period_col, map_direction_col = st.columns(2)
    
    with map_period_col:
        map_period = st.selectbox(
            "Period:",
            ["All Time"] + year_months,
            key="map_period",
            format_func=lambda x: x if x == "All Time" else uih.format_year_month(x)
        )
    
    with map_direction_col:
        map_direction = st.selectbox(
            "Direction:",
            ["outgoing", "incoming"],
            key="map_direction",
            format_func=lambda x: "Outgoing Rides" if x == "outgoing" else "Incoming Rides"
        )
    
    map_year_month = None if map_period == "All Time" else map_period
    
    st.markdown("**Top Neighborhoods by Ride Volume**")
    fig = viz.get_nta_totals_map(map_year_month, map_direction)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("*This chart shows the top 20 neighborhoods ranked by total ride count.*")

with about_tab:
    st.subheader("About This App")
    
    st.markdown("""
### Data Source

This app uses bike ride flow data from the `daily_flows` directory, which contains daily ride counts 
between NYC Neighborhood Tabulation Areas (NTAs). The data spans from January 2023 to December 2025.

### What are NTAs?

Neighborhood Tabulation Areas (NTAs) are geographic units defined by the NYC Department of City Planning. 
They are designed to be meaningful geographic areas that roughly correspond to neighborhoods.

### Features

- **📈 Trends**: View time series of ride counts for any neighborhood
- **🔀 Flows**: Explore where riders go to/come from
- **🔍 Rankings**: Compare growth rates across neighborhoods
- **🗺️ Overview**: See which neighborhoods have the most bike traffic

### How to Use

1. **Select a Borough** to filter the list of neighborhoods
2. **Select a Neighborhood (NTA)** to focus your analysis
3. **Choose Direction**: 
   - *Outgoing* = rides that START in this neighborhood
   - *Incoming* = rides that END in this neighborhood
4. **Explore the tabs** to see different visualizations

### Technical Details

- Data is stored as CSV files with columns: `started_date`, `start_NTA`, `end_NTA`, `start_Boro`, `end_Boro`, `ride_count`
- Geographic boundaries are available in `NYC_NTAs.csv` with MULTIPOLYGON geometries
- The app is built with [Streamlit](https://streamlit.io/) and uses matplotlib, seaborn, and plotly for visualizations

---
    """)
    
    # Show data summary
    st.markdown("### Data Summary")
    col1, col2, col3 = st.columns(3)
    
    date_min, date_max = be.get_date_range()
    
    with col1:
        st.metric("Total NTAs", len(be.get_all_ntas()))
    
    with col2:
        st.metric("Time Range", f"{date_min.strftime('%b %Y')} - {date_max.strftime('%b %Y')}")
    
    with col3:
        st.metric("Boroughs", len(be.get_boroughs()))
