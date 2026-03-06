"""
Overview Page - Neighborhood overview map

See which neighborhoods have the most bike traffic.
"""

import streamlit as st
import backend as be
import visualizations as viz
import ui_helpers as uih

st.set_page_config(page_title="Overview", page_icon="🗺️", layout="wide")

st.header("🗺️ Neighborhood Overview")

st.markdown("""
Explore which NYC neighborhoods have the highest bike traffic volumes.
""")

# Get available data for dropdowns
year_months = be.get_year_months()

# Controls
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

st.subheader("Top Neighborhoods by Ride Volume")
fig = viz.get_nta_totals_map(map_year_month, map_direction)
st.plotly_chart(fig, use_container_width=True)

st.caption("*This chart shows the top 20 neighborhoods ranked by total ride count.*")
