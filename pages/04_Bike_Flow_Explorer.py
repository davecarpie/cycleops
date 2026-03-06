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
graph_tab, flow_tab, heatmap_tab = st.tabs(
    ["📈 Trends", "🔀 Flows", "🔥 Heat Map"]
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
    
    # Option: exclude rides that start and end in the same NTA
    exclude_internal = st.checkbox(
        "Exclude rides that both start and end in this NTA",
        help="Remove internal trips from the heatmap so only flows to/from other neighborhoods are shown."
    )
    
    # Render the choropleth map using shared controls
    fig = viz.get_nta_choropleth_map(
        selected_nta, selected_year_month, direction, exclude_self=exclude_internal
    )
    st.plotly_chart(fig, use_container_width=True)
    
    direction_verb = "going to" if direction == "outgoing" else "coming from"
    caption_text = f"*Darker shades indicate more rides {direction_verb} each neighborhood.*"
    if exclude_internal:
        caption_text += " (Internal trips excluded.)"
    st.caption(caption_text)
    
    # Bar chart comparison below the map - show top destinations/origins for selected NTA
    st.markdown("---")
    chart_title = "Top Destinations" if direction == "outgoing" else "Top Origins"
    st.markdown(f"**{chart_title} for {selected_nta}**")
    
    if direction == "outgoing":
        fig = viz.get_top_destinations_chart(selected_nta, 15, selected_year_month)
    else:
        fig = viz.get_top_origins_chart(selected_nta, 15, selected_year_month)
    st.pyplot(fig)
