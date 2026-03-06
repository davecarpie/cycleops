"""
Rankings Page - Neighborhood ranking comparison

Compare how ride counts have changed between two time periods.
"""

import streamlit as st
import backend as be
import visualizations as viz
import ui_helpers as uih

st.set_page_config(page_title="Rankings", page_icon="🔍", layout="wide")

st.header("🔍 Neighborhood Rankings")

st.markdown("""
Compare how ride counts have changed between two time periods for all NYC neighborhoods (NTAs).
""")

# Get available data for dropdowns
boroughs = ["All Boroughs"] + be.get_boroughs()
year_months = be.get_year_months()
default_period1, default_period2 = uih.get_comparison_periods(year_months)

# Direction selector
direction_col, borough_col, nta_col = st.columns(3)

with direction_col:
    direction = st.selectbox(
        "Direction:",
        ["outgoing", "incoming"],
        format_func=lambda x: "Outgoing (rides starting here)" if x == "outgoing" else "Incoming (rides ending here)",
        help="Choose whether to analyze rides leaving or arriving at neighborhoods"
    )

with borough_col:
    borough = st.selectbox(
        "Borough:",
        boroughs,
        index=0,
        help="Filter neighborhoods by borough"
    )

with nta_col:
    ntas = be.get_ntas(borough if borough != "All Boroughs" else None)
    default_idx = min(uih.get_nta_index(borough), len(ntas) - 1) if ntas else 0
    selected_nta = st.selectbox(
        "Highlight Neighborhood:",
        ntas,
        index=default_idx,
        help="Select a neighborhood to highlight in the rankings"
    )

st.subheader("Period Comparison")

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
st.subheader("Distribution of Changes Across All NTAs")

fig = viz.get_swarmplot(period1, period2, direction, selected_nta)
st.pyplot(fig)
st.caption("*Each dot represents an NTA. The highlighted dot (if visible) is your selected neighborhood.*")
