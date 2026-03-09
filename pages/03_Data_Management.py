import streamlit as st
from pathlib import Path
import base64

def load_image(file_name):
    return base64.b64encode(Path(__file__).with_name(file_name).read_bytes()).decode("ascii")

er_diagram_image = load_image("ER Diagram Citibikes(1).png")

DATA_MANAGEMENT_MARKDOWN = f"""
## Data Management
This page provides an overview of how data is managed within the CycleOps project.

* Data comes from the Citibike system data set and the NYC open data Neighborhood Tabulation Areas (NTAs) data set
* Citibike data is from January 2023 through to December 2025
* Significant amount of unnecessary data in raw CSVs which we processed to make more managable

| Field | Explanation |
| --- | --- |
| `ride_id` | A unique identifier for the ride |
| `ridable_type` | The type of bike that was used (classic or electric) |
| `started_at` | The date and time that the ride began at |
| `ended_at` | The date and time the ride ended at |
| `start_station_name` | The name of the station that the ride began at |
| `start_station_id` | The ID of the station that the ride began at |
| `end_station_name` | The name of the station that the ride ended at |
| `end_station_id` | The ID of the station that the ride ended at |
| `start_lat` | The latitude of the station that the ride began at |
| `start_long` | The longitude of the station that the ride began at |
| `end_lat` | The latitude of the station that the ride ended at |
| `end_long` | The longitude of the station that the ride ended at |

* After using GeoPandas and joining our stations to our NTAs we get this ER digram of our data:
![ER Diagram](data:image/png;base64,{er_diagram_image})

* Since we are interested in evaluating the flow of bikes between neighborhoods we created a new table that aggregates the number of rides between each neighborhood pair on each day. 
* This is the table we use for all of our analysis and visualizations.

| Field | Explanation |
| --- | --- |
| `date` | The date these ride counts are for |
| `start_nta` | The start neighborhood of the rides |
| `start_boro` | The start borough of the rides |
| `end_nta` | The end neighborhood of the rides |
| `end_boro` | The end borough of the rides |
| `ride_count` | The count of rides |
"""

st.markdown(DATA_MANAGEMENT_MARKDOWN)


