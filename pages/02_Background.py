import streamlit as st

st.set_page_config(page_title="Background", layout="wide")

st.title("Background")

BACKGROUND_MARKUP = """
* The project analyzes Citi Bike trip data to understand how bicycles circulate across New York City's neighborhoods, corridors, and activity hubs.

* Using station-level GPS coordinates and timestamps, we map spatial and temporal patterns of bike flows to reveal rhythms, hotspots, areas of inactivity.

* Neighborhood-to-neighborhood movement is examined to identify dominant origin-destination pairs, internal circulation within districts, and cross-borough travel behavior.

* Historical trends are leveraged to build predictive models that estimate future bike movements based on seasonality, time of day, weather, and past demand.

* The analysis supports operational planning—such as areas for bike infrastructure investment, areas for more stations, and rebalancing  strategies, station placement.

* Ultimately, the project aims to create an interactive visualization tool that helps stakeholders explore historic and near real-time bike flows across the city.

![Background visual](https://images.ctfassets.net/p6ae3zqfb1e3/1pO7gkOffhHXIuqXxFrZNr/7ca4616ed150ed1fc0d1d984d02a2fb4/Screenshot_2025-12-23_at_10.34.24%C3%A2__AM.png?w=&q=60&fm=webp)
"""

st.markdown(BACKGROUND_MARKUP)
