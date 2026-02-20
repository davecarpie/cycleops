"""
Backend module for loading and processing bike ride flow data.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from functools import lru_cache


# Load all daily flow data from the daily_flows directory
@lru_cache(maxsize=1)
def load_all_data():
    """Load and concatenate all monthly CSV files from daily_flows directory."""
    data_dir = "daily_flows"
    all_dfs = []
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith("_daily.csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            # Extract year-month from filename (e.g., "202301" from "202301_daily.csv")
            year_month = filename.replace("_daily.csv", "")
            df["year_month"] = year_month
            df["year"] = int(year_month[:4])
            df["month"] = int(year_month[4:])
            all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["started_date"] = pd.to_datetime(combined_df["started_date"])
    return combined_df


# Load NTA geographic data
@lru_cache(maxsize=1)
def load_nta_geometry():
    """Load NTA geometry data for mapping."""
    df = pd.read_csv("NYC_NTAs.csv")
    return df


# Get the main dataframe
df = load_all_data()
nta_geo = load_nta_geometry()


def get_boroughs():
    """Get list of unique boroughs."""
    boroughs = sorted(df["start_Boro"].unique())
    return boroughs


def get_ntas(borough=None):
    """Get list of unique NTAs, optionally filtered by borough."""
    if borough and borough != "All Boroughs":
        ntas = sorted(df[df["start_Boro"] == borough]["start_NTA"].unique())
    else:
        ntas = sorted(df["start_NTA"].unique())
    return ntas


def get_all_ntas():
    """Get list of all unique NTAs (both start and end)."""
    start_ntas = set(df["start_NTA"].unique())
    end_ntas = set(df["end_NTA"].unique())
    return sorted(start_ntas | end_ntas)


def get_date_range():
    """Get the date range of available data."""
    return df["started_date"].min(), df["started_date"].max()


def get_year_months():
    """Get list of unique year-month combinations."""
    return sorted(df["year_month"].unique())


def get_years():
    """Get list of unique years in the data."""
    return sorted(df["year"].unique())


def get_nta_time_series(nta, direction="outgoing"):
    """
    Get daily ride counts for an NTA over time.
    
    Args:
        nta: The NTA name
        direction: 'outgoing' (rides starting from NTA) or 'incoming' (rides ending at NTA)
    
    Returns:
        DataFrame with date and ride_count columns
    """
    if direction == "outgoing":
        filtered = df[df["start_NTA"] == nta]
    else:
        filtered = df[df["end_NTA"] == nta]
    
    # Aggregate by date
    time_series = filtered.groupby("started_date")["ride_count"].sum().reset_index()
    time_series.columns = ["date", "ride_count"]
    return time_series


def get_nta_monthly_series(nta, direction="outgoing"):
    """
    Get monthly ride counts for an NTA.
    
    Args:
        nta: The NTA name
        direction: 'outgoing' or 'incoming'
    
    Returns:
        DataFrame with year_month and ride_count columns
    """
    if direction == "outgoing":
        filtered = df[df["start_NTA"] == nta]
    else:
        filtered = df[df["end_NTA"] == nta]
    
    # Aggregate by year-month
    monthly = filtered.groupby("year_month")["ride_count"].sum().reset_index()
    monthly.columns = ["year_month", "ride_count"]
    
    # Add year and month columns for better plotting
    monthly["year"] = monthly["year_month"].str[:4].astype(int)
    monthly["month"] = monthly["year_month"].str[4:].astype(int)
    monthly["date"] = pd.to_datetime(monthly["year_month"], format="%Y%m")
    
    return monthly


def get_top_destinations(nta, top_n=10, year_month=None):
    """
    Get top destination NTAs for rides starting from the given NTA.
    
    Args:
        nta: The source NTA name
        top_n: Number of top destinations to return
        year_month: Optional filter for specific year-month (e.g., "202301")
    
    Returns:
        DataFrame with end_NTA and ride_count columns
    """
    filtered = df[df["start_NTA"] == nta]
    if year_month:
        filtered = filtered[filtered["year_month"] == year_month]
    
    top_dest = (filtered.groupby("end_NTA")["ride_count"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index())
    top_dest.columns = ["destination_NTA", "ride_count"]
    return top_dest


def get_top_origins(nta, top_n=10, year_month=None):
    """
    Get top origin NTAs for rides ending at the given NTA.
    
    Args:
        nta: The destination NTA name
        top_n: Number of top origins to return
        year_month: Optional filter for specific year-month
    
    Returns:
        DataFrame with start_NTA and ride_count columns
    """
    filtered = df[df["end_NTA"] == nta]
    if year_month:
        filtered = filtered[filtered["year_month"] == year_month]
    
    top_orig = (filtered.groupby("start_NTA")["ride_count"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index())
    top_orig.columns = ["origin_NTA", "ride_count"]
    return top_orig


def get_nta_ranking_df(year_month1, year_month2, direction="outgoing"):
    """
    Get ranking dataframe comparing ride counts between two time periods.
    
    Args:
        year_month1: First year-month (e.g., "202301")
        year_month2: Second year-month (e.g., "202401")
        direction: 'outgoing' or 'incoming'
    
    Returns:
        DataFrame with NTA, ride counts for both periods, change, and percent change
    """
    if direction == "outgoing":
        col = "start_NTA"
    else:
        col = "end_NTA"
    
    # Get totals for each period
    df1 = df[df["year_month"] == year_month1].groupby(col)["ride_count"].sum().reset_index()
    df1.columns = ["NTA", year_month1]
    
    df2 = df[df["year_month"] == year_month2].groupby(col)["ride_count"].sum().reset_index()
    df2.columns = ["NTA", year_month2]
    
    # Merge and calculate changes
    merged = pd.merge(df1, df2, on="NTA", how="outer").fillna(0)
    merged["Change"] = merged[year_month2] - merged[year_month1]
    merged["Percent Change"] = np.where(
        merged[year_month1] != 0,
        ((merged[year_month2] - merged[year_month1]) / merged[year_month1] * 100).round(1),
        np.nan
    )
    
    # Sort by percent change and add rank
    merged = merged.sort_values("Percent Change", ascending=False)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["Percent Change"])
    merged["Rank"] = list(range(1, len(merged.index) + 1))
    merged = merged.reset_index(drop=True).set_index("Rank")
    
    return merged


def get_nta_totals_by_month(direction="outgoing"):
    """
    Get total ride counts per NTA per month.
    
    Args:
        direction: 'outgoing' or 'incoming'
    
    Returns:
        DataFrame with NTA, year_month, and ride_count columns
    """
    if direction == "outgoing":
        col = "start_NTA"
    else:
        col = "end_NTA"
    
    totals = df.groupby([col, "year_month"])["ride_count"].sum().reset_index()
    totals.columns = ["NTA", "year_month", "ride_count"]
    return totals


def get_ranking_text(nta, direction, ranking_df):
    """Generate ranking text for the selected NTA."""
    def ordinal_suffix(n):
        if 11 <= n <= 13:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    
    if nta not in list(ranking_df["NTA"]):
        return f"**{nta}** does not have a ranking for this comparison."
    
    rank = ranking_df[ranking_df["NTA"] == nta].index.tolist()[0]
    num_ntas = len(ranking_df.index)
    percentile = round((rank - 1) / (num_ntas - 1) * 100) if num_ntas > 1 else 0
    
    direction_text = "outgoing" if direction == "outgoing" else "incoming"
    
    return (
        f"**{nta}** ranks **{rank}** of {num_ntas} NTAs for {direction_text} ride growth "
        f"(the {percentile}{ordinal_suffix(percentile)} percentile)."
    )


def get_borough_totals_by_month():
    """Get total ride counts per borough pair per month."""
    totals = df.groupby(["start_Boro", "end_Boro", "year_month"])["ride_count"].sum().reset_index()
    return totals


def get_flow_matrix(year_month=None):
    """
    Get a flow matrix showing rides between NTAs.
    
    Args:
        year_month: Optional filter for specific year-month
    
    Returns:
        Pivot table with start_NTA as rows and end_NTA as columns
    """
    filtered = df if year_month is None else df[df["year_month"] == year_month]
    
    matrix = filtered.pivot_table(
        values="ride_count",
        index="start_NTA",
        columns="end_NTA",
        aggfunc="sum",
        fill_value=0
    )
    return matrix


def get_nta_geometry_dict():
    """
    Convert NTA geometry to a format usable by plotly.
    Returns a GeoJSON-like structure.
    """
    features = []
    
    for _, row in nta_geo.iterrows():
        # Parse the WKT geometry string
        geom_str = row["the_geom"]
        if pd.isna(geom_str):
            continue
            
        # Create a simplified feature (in production, you'd parse the full geometry)
        feature = {
            "type": "Feature",
            "properties": {
                "NTAName": row["NTAName"],
                "BoroName": row["BoroName"],
                "NTA2020": row["NTA2020"]
            },
            "id": row["NTAName"]
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


def parse_wkt_multipolygon(wkt_string):
    """
    Parse a WKT MULTIPOLYGON string into GeoJSON coordinates.
    Handles complex multipolygons with multiple polygons and holes.
    
    Args:
        wkt_string: WKT MULTIPOLYGON string
    
    Returns:
        List of polygon coordinates in GeoJSON format, or None if parsing fails
    """
    if pd.isna(wkt_string) or not wkt_string:
        return None
    
    # Remove MULTIPOLYGON prefix and outer parentheses
    wkt_string = wkt_string.strip()
    if not wkt_string.startswith("MULTIPOLYGON"):
        return None
    
    # Extract everything inside MULTIPOLYGON (...)
    match = re.match(r'MULTIPOLYGON\s*\(\s*(.+)\s*\)$', wkt_string, re.DOTALL)
    if not match:
        return None
    
    content = match.group(1)
    
    polygons = []
    
    # Split by polygon boundaries: ((...)), ((...))
    # Each polygon starts with (( and ends with ))
    polygon_pattern = r'\(\(([^()]+(?:\([^()]*\)[^()]*)*)\)\)'
    polygon_matches = re.findall(polygon_pattern, content)
    
    for poly_content in polygon_matches:
        rings = []
        # Split by ), ( to get individual rings (outer ring and holes)
        ring_parts = re.split(r'\)\s*,\s*\(', poly_content)
        
        for ring_str in ring_parts:
            # Clean up the ring string
            ring_str = ring_str.strip().strip('()')
            coords = []
            
            # Parse coordinate pairs
            for pair in ring_str.split(','):
                pair = pair.strip()
                parts = pair.split()
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coords.append([lon, lat])
                    except ValueError:
                        continue
            
            if len(coords) >= 3:
                rings.append(coords)
        
        if rings:
            polygons.append(rings)
    
    # For simple multipolygons with just one polygon
    if not polygons:
        # Try simpler parsing for single polygon
        coords_match = re.search(r'\(\(\((.+?)\)\)\)', wkt_string, re.DOTALL)
        if coords_match:
            coords_str = coords_match.group(1)
            coords = []
            for pair in coords_str.split(','):
                pair = pair.strip()
                parts = pair.split()
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coords.append([lon, lat])
                    except ValueError:
                        continue
            if len(coords) >= 3:
                polygons.append([coords])
    
    return polygons if polygons else None


@lru_cache(maxsize=1)
def create_nta_geojson_full():
    """
    Create a complete GeoJSON FeatureCollection from NTA data with proper geometry.
    
    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []
    
    for _, row in nta_geo.iterrows():
        polygons = parse_wkt_multipolygon(row["the_geom"])
        if polygons is None:
            continue
        
        # Determine geometry type based on number of polygons
        if len(polygons) == 1:
            geometry = {
                "type": "Polygon",
                "coordinates": polygons[0]
            }
        else:
            geometry = {
                "type": "MultiPolygon",
                "coordinates": polygons
            }
        
        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "NTAName": row["NTAName"],
                "BoroName": row["BoroName"],
                "NTA2020": row.get("NTA2020", ""),
            },
            "id": row["NTAName"]
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_nta_traffic_to_others(selected_nta, year_month=None, direction="outgoing"):
    """
    Get traffic from a selected NTA to/from all other NTAs.
    
    Args:
        selected_nta: The NTA to analyze
        year_month: Optional filter for specific year-month
        direction: 'outgoing' (rides from selected_nta) or 'incoming' (rides to selected_nta)
    
    Returns:
        DataFrame with NTA and ride_count columns for all connected NTAs
    """
    filtered = df if year_month is None else df[df["year_month"] == year_month]
    
    if direction == "outgoing":
        # Get rides starting from selected_nta
        nta_df = filtered[filtered["start_NTA"] == selected_nta]
        traffic = nta_df.groupby("end_NTA")["ride_count"].sum().reset_index()
        traffic.columns = ["NTA", "ride_count"]
    else:
        # Get rides ending at selected_nta
        nta_df = filtered[filtered["end_NTA"] == selected_nta]
        traffic = nta_df.groupby("start_NTA")["ride_count"].sum().reset_index()
        traffic.columns = ["NTA", "ride_count"]
    
    return traffic


def get_all_nta_traffic(year_month=None, direction="outgoing"):
    """
    Get total traffic for all NTAs.
    
    Args:
        year_month: Optional filter for specific year-month
        direction: 'outgoing' or 'incoming'
    
    Returns:
        DataFrame with NTA and ride_count columns
    """
    filtered = df if year_month is None else df[df["year_month"] == year_month]
    
    if direction == "outgoing":
        traffic = filtered.groupby("start_NTA")["ride_count"].sum().reset_index()
    else:
        traffic = filtered.groupby("end_NTA")["ride_count"].sum().reset_index()
    
    traffic.columns = ["NTA", "ride_count"]
    return traffic
