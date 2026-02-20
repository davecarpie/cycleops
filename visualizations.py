"""
Visualization functions for the Bike Flow Explorer app.
"""

import backend as be
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import pandas as pd
import json


@st.cache_resource
def get_line_graph(nta, direction="outgoing"):
    """
    Create a time series line graph showing monthly ride counts for an NTA.
    
    Args:
        nta: The NTA name
        direction: 'outgoing' or 'incoming'
    
    Returns:
        matplotlib figure
    """
    df = be.get_nta_monthly_series(nta, direction)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No data available for {nta}", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors by year
    colors = {
        2023: "#1f77b4",  # Blue
        2024: "#ff7f0e",  # Orange
        2025: "#2ca02c",  # Green
    }
    
    # Plot by year with different colors
    for year in df["year"].unique():
        year_data = df[df["year"] == year]
        color = colors.get(year, "#7f7f7f")
        sns.lineplot(
            data=year_data,
            x="date",
            y="ride_count",
            ax=ax,
            marker="o",
            color=color,
            label=str(year),
        )
    
    direction_text = "Outgoing" if direction == "outgoing" else "Incoming"
    ax.set_title(f"{direction_text} Rides\n{nta}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Rides")
    ax.legend(title="Year")
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


@st.cache_resource
def get_monthly_comparison_graph(nta, direction="outgoing"):
    """
    Create a comparison graph showing monthly trends across years.
    
    Args:
        nta: The NTA name
        direction: 'outgoing' or 'incoming'
    
    Returns:
        matplotlib figure
    """
    df = be.get_nta_monthly_series(nta, direction)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No data available for {nta}", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each year as a separate line, with month on x-axis
    colors = {2023: "#1f77b4", 2024: "#ff7f0e", 2025: "#2ca02c"}
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for year in sorted(df["year"].unique()):
        year_data = df[df["year"] == year].copy()
        color = colors.get(year, "#7f7f7f")
        ax.plot(year_data["month"], year_data["ride_count"], 
                marker="o", color=color, label=str(year), linewidth=2)
    
    direction_text = "Outgoing" if direction == "outgoing" else "Incoming"
    ax.set_title(f"{direction_text} Rides by Month\n{nta}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Rides")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45, ha='right')
    ax.legend(title="Year")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    
    plt.tight_layout()
    return fig


@st.cache_resource
def get_top_destinations_chart(nta, top_n=10, year_month=None):
    """
    Create a horizontal bar chart showing top destinations from an NTA.
    
    Args:
        nta: The source NTA name
        top_n: Number of destinations to show
        year_month: Optional filter for specific period
    
    Returns:
        matplotlib figure
    """
    df = be.get_top_destinations(nta, top_n, year_month)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No destination data for {nta}", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Reverse order so highest is at top
    df_plot = df.iloc[::-1]
    
    bars = ax.barh(df_plot["destination_NTA"], df_plot["ride_count"], color="#1f77b4")
    
    ax.set_xlabel("Total Rides")
    ax.set_ylabel("Destination NTA")
    period_text = f" ({year_month})" if year_month else " (All Time)"
    ax.set_title(f"Top {top_n} Destinations from {nta}{period_text}")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    
    # Add value labels on bars
    for bar, value in zip(bars, df_plot["ride_count"]):
        ax.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


@st.cache_resource
def get_top_origins_chart(nta, top_n=10, year_month=None):
    """
    Create a horizontal bar chart showing top origins to an NTA.
    
    Args:
        nta: The destination NTA name
        top_n: Number of origins to show
        year_month: Optional filter for specific period
    
    Returns:
        matplotlib figure
    """
    df = be.get_top_origins(nta, top_n, year_month)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No origin data for {nta}", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Reverse order so highest is at top
    df_plot = df.iloc[::-1]
    
    bars = ax.barh(df_plot["origin_NTA"], df_plot["ride_count"], color="#ff7f0e")
    
    ax.set_xlabel("Total Rides")
    ax.set_ylabel("Origin NTA")
    period_text = f" ({year_month})" if year_month else " (All Time)"
    ax.set_title(f"Top {top_n} Origins to {nta}{period_text}")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    
    # Add value labels on bars
    for bar, value in zip(bars, df_plot["ride_count"]):
        ax.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


@st.cache_resource
def get_swarmplot(year_month1, year_month2, direction, selected_nta):
    """
    Create a swarm plot showing percent change distribution across all NTAs.
    
    Args:
        year_month1: First period (e.g., "202301")
        year_month2: Second period (e.g., "202401")
        direction: 'outgoing' or 'incoming'
        selected_nta: NTA to highlight
    
    Returns:
        matplotlib figure
    """
    df = be.get_nta_ranking_df(year_month1, year_month2, direction)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available for comparison", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors
    normal_color = "black"
    normal_alpha = 0.3
    highlight_color = "#FF4500"  # Orange-red
    
    # Separate highlighted NTA
    df_other = df[df["NTA"] != selected_nta]
    df_highlight = df[df["NTA"] == selected_nta]
    
    # Plot all other NTAs
    sns.swarmplot(
        x=df_other["Percent Change"],
        color=normal_color,
        size=5,
        alpha=normal_alpha,
        ax=ax,
    )
    
    # Highlight selected NTA
    if not df_highlight.empty:
        sns.swarmplot(
            x=df_highlight["Percent Change"],
            color=highlight_color,
            size=8,
            ax=ax,
        )
        
        legend_patch = plt.Line2D(
            [0], [0], marker="o", color=highlight_color, markersize=8,
            label=selected_nta, linestyle="None",
        )
        ax.legend(handles=[legend_patch])
    
    direction_text = "Outgoing" if direction == "outgoing" else "Incoming"
    period1 = f"{year_month1[:4]}-{year_month1[4:]}"
    period2 = f"{year_month2[:4]}-{year_month2[4:]}"
    ax.set_title(f"Percent Change in {direction_text} Rides\nAll NTAs, {period1} to {period2}")
    ax.set_xlabel("Percent Change")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}%"))
    
    return fig


@st.cache_resource
def get_borough_flow_heatmap(year_month=None):
    """
    Create a heatmap showing flow between boroughs.
    
    Args:
        year_month: Optional filter for specific period
    
    Returns:
        plotly figure
    """
    df = be.get_borough_totals_by_month()
    
    if year_month:
        df = df[df["year_month"] == year_month]
    
    # Pivot to create matrix
    matrix = df.pivot_table(
        values="ride_count",
        index="start_Boro",
        columns="end_Boro",
        aggfunc="sum",
        fill_value=0
    )
    
    fig = px.imshow(
        matrix,
        labels=dict(x="Destination Borough", y="Origin Borough", color="Rides"),
        title="Bike Rides Between Boroughs" + (f" ({year_month})" if year_month else " (All Time)"),
        color_continuous_scale="Blues",
        text_auto=True,
    )
    
    fig.update_layout(
        xaxis_title="Destination Borough",
        yaxis_title="Origin Borough",
    )
    
    return fig


@st.cache_resource  
def get_nta_totals_map(year_month, direction="outgoing"):
    """
    Create a choropleth-style visualization of NTA ride totals.
    Since we have complex geometry, we'll use a bar chart as an alternative.
    
    Args:
        year_month: The period to show
        direction: 'outgoing' or 'incoming'
    
    Returns:
        plotly figure
    """
    df = be.get_nta_totals_by_month(direction)
    
    if year_month:
        df = df[df["year_month"] == year_month]
    else:
        df = df.groupby("NTA")["ride_count"].sum().reset_index()
    
    # Get top 20 NTAs
    df_top = df.nlargest(20, "ride_count")
    
    direction_text = "Outgoing" if direction == "outgoing" else "Incoming"
    period_text = f" ({year_month})" if year_month else " (All Time)"
    
    fig = px.bar(
        df_top,
        x="ride_count",
        y="NTA",
        orientation="h",
        title=f"Top 20 NTAs by {direction_text} Rides{period_text}",
        labels={"ride_count": "Total Rides", "NTA": "Neighborhood"},
        color="ride_count",
        color_continuous_scale="Blues",
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=600,
    )
    
    return fig


@st.cache_resource
def get_flow_sankey(nta, direction="outgoing", top_n=10, year_month=None):
    """
    Create a Sankey diagram showing flow to/from an NTA.
    
    Args:
        nta: The selected NTA
        direction: 'outgoing' or 'incoming'
        top_n: Number of connections to show
        year_month: Optional filter
    
    Returns:
        plotly figure
    """
    if direction == "outgoing":
        df = be.get_top_destinations(nta, top_n, year_month)
        source_label = nta
        target_labels = df["destination_NTA"].tolist()
        values = df["ride_count"].tolist()
    else:
        df = be.get_top_origins(nta, top_n, year_month)
        target_label = nta
        source_labels = df["origin_NTA"].tolist()
        values = df["ride_count"].tolist()
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No flow data available", x=0.5, y=0.5, 
                          showarrow=False, font_size=16)
        return fig
    
    if direction == "outgoing":
        all_labels = [source_label] + target_labels
        sources = [0] * len(target_labels)
        targets = list(range(1, len(target_labels) + 1))
    else:
        all_labels = source_labels + [target_label]
        sources = list(range(len(source_labels)))
        targets = [len(source_labels)] * len(source_labels)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(31, 119, 180, 0.4)"
        )
    )])
    
    direction_text = "from" if direction == "outgoing" else "to"
    period_text = f" ({year_month})" if year_month else ""
    fig.update_layout(
        title=f"Bike Flow {direction_text} {nta}{period_text}",
        font_size=12,
        height=500,
    )
    
    return fig


@st.cache_resource
def get_nta_traffic_heatmap(year_month=None, direction="outgoing"):
    """
    Create a heatmap showing traffic intensity for each NTA.
    
    Args:
        year_month: Optional filter for specific period
        direction: 'outgoing' or 'incoming'
    
    Returns:
        plotly figure
    """
    df = be.get_nta_totals_by_month(direction)
    
    if year_month:
        df = df[df["year_month"] == year_month]
        # Group by NTA only since we filtered to one month
        df_totals = df.groupby("NTA")["ride_count"].sum().reset_index()
    else:
        # Aggregate across all months
        df_totals = df.groupby("NTA")["ride_count"].sum().reset_index()
    
    # Sort by ride count descending
    df_totals = df_totals.sort_values("ride_count", ascending=True)
    
    # Take top 30 for readability
    df_top = df_totals.tail(30)
    
    direction_text = "Outgoing" if direction == "outgoing" else "Incoming"
    period_text = f" ({year_month})" if year_month else " (All Time)"
    
    fig = px.bar(
        df_top,
        x="ride_count",
        y="NTA",
        orientation="h",
        title=f"{direction_text} Traffic by Neighborhood{period_text}",
        labels={"ride_count": "Total Rides", "NTA": "Neighborhood"},
        color="ride_count",
        color_continuous_scale="YlOrRd",
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        coloraxis_showscale=True,
        coloraxis_colorbar_title="Rides",
    )
    
    return fig


@st.cache_resource
def get_borough_traffic_heatmap(year_month=None, direction="outgoing"):
    """
    Create a heatmap showing traffic intensity for each borough.
    
    Args:
        year_month: Optional filter for specific period
        direction: 'outgoing' or 'incoming'
    
    Returns:
        plotly figure
    """
    # Get the raw data
    raw_df = be.load_all_data()
    
    if year_month:
        raw_df = raw_df[raw_df["year_month"] == year_month]
    
    # Aggregate by borough based on direction
    if direction == "outgoing":
        df_totals = raw_df.groupby("start_Boro")["ride_count"].sum().reset_index()
        df_totals.columns = ["Borough", "ride_count"]
    else:
        df_totals = raw_df.groupby("end_Boro")["ride_count"].sum().reset_index()
        df_totals.columns = ["Borough", "ride_count"]
    
    # Sort by ride count
    df_totals = df_totals.sort_values("ride_count", ascending=True)
    
    direction_text = "Outgoing" if direction == "outgoing" else "Incoming"
    period_text = f" ({year_month})" if year_month else " (All Time)"
    
    fig = px.bar(
        df_totals,
        x="ride_count",
        y="Borough",
        orientation="h",
        title=f"{direction_text} Traffic by Borough{period_text}",
        labels={"ride_count": "Total Rides", "Borough": "Borough"},
        color="ride_count",
        color_continuous_scale="YlOrRd",
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        coloraxis_showscale=True,
        coloraxis_colorbar_title="Rides",
    )
    
    return fig


@st.cache_resource
def get_nta_flow_heatmap(year_month=None, top_n=20):
    """
    Create a heatmap showing flow between top NTAs.
    
    Args:
        year_month: Optional filter for specific period
        top_n: Number of top NTAs to include
    
    Returns:
        plotly figure
    """
    # Get flow matrix
    matrix = be.get_flow_matrix(year_month)
    
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No flow data available", x=0.5, y=0.5, 
                          showarrow=False, font_size=16)
        return fig
    
    # Get top NTAs by total outgoing rides
    top_ntas = matrix.sum(axis=1).nlargest(top_n).index.tolist()
    
    # Filter matrix to top NTAs only (both rows and columns)
    matrix_filtered = matrix.loc[top_ntas, top_ntas]
    
    period_text = f" ({year_month})" if year_month else " (All Time)"
    
    fig = px.imshow(
        matrix_filtered,
        labels=dict(x="Destination NTA", y="Origin NTA", color="Rides"),
        title=f"Bike Flow Between Top {top_n} Neighborhoods{period_text}",
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    
    fig.update_layout(
        height=700,
        xaxis_tickangle=45,
    )
    
    return fig


@st.cache_resource
def get_nta_choropleth_map(selected_nta=None, year_month=None, direction="outgoing"):
    """
    Create a choropleth map showing traffic intensity for neighborhoods.
    
    If selected_nta is provided, shows traffic to/from that specific NTA.
    Otherwise, shows total traffic for all NTAs.
    
    Args:
        selected_nta: Optional NTA to show connections for
        year_month: Optional filter for specific period
        direction: 'outgoing' or 'incoming'
    
    Returns:
        plotly figure with choropleth map
    """
    import numpy as np
    
    # Get GeoJSON data
    geojson = be.create_nta_geojson_full()
    
    if not geojson["features"]:
        fig = go.Figure()
        fig.add_annotation(text="No geometry data available", x=0.5, y=0.5, 
                          showarrow=False, font_size=16)
        return fig
    
    # Get traffic data
    if selected_nta:
        traffic_df = be.get_nta_traffic_to_others(selected_nta, year_month, direction)
        # Add the selected NTA with 0 rides (to show it on map)
        if selected_nta not in traffic_df["NTA"].values:
            traffic_df = pd.concat([
                traffic_df, 
                pd.DataFrame({"NTA": [selected_nta], "ride_count": [0]})
            ], ignore_index=True)
    else:
        traffic_df = be.get_all_nta_traffic(year_month, direction)
    
    # Get all NTAs from GeoJSON to ensure complete coverage
    geojson_ntas = [f["id"] for f in geojson["features"]]
    
    # Create a complete dataframe with all GeoJSON NTAs
    all_ntas_df = pd.DataFrame({"NTA": geojson_ntas})
    traffic_df = pd.merge(all_ntas_df, traffic_df, on="NTA", how="left")
    traffic_df["ride_count"] = traffic_df["ride_count"].fillna(0)
    
    # Apply log transformation for better color visibility of low values
    # Add 1 to avoid log(0), then use log scale
    max_rides = traffic_df["ride_count"].max()
    if max_rides > 0:
        # Create a normalized value using square root for better spread
        # This makes lower values more visible while still showing the gradient
        traffic_df["color_value"] = np.sqrt(traffic_df["ride_count"])
    else:
        traffic_df["color_value"] = 0
    
    # Build title
    direction_text = "Outgoing" if direction == "outgoing" else "Incoming"
    period_text = f" ({year_month})" if year_month else " (All Time)"
    if selected_nta:
        if direction == "outgoing":
            title = f"Destinations from {selected_nta}{period_text}"
        else:
            title = f"Origins to {selected_nta}{period_text}"
    else:
        title = f"{direction_text} Traffic by Neighborhood{period_text}"
    
    # Create custom hover text with actual ride counts
    traffic_df["hover_text"] = traffic_df.apply(
        lambda row: f"{row['NTA']}<br>Rides: {int(row['ride_count']):,}", axis=1
    )
    
    # Create choropleth map with sqrt-scaled colors but original values in hover
    fig = px.choropleth_mapbox(
        traffic_df,
        geojson=geojson,
        locations="NTA",
        featureidkey="id",
        color="color_value",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 40.7128, "lon": -73.95},
        opacity=0.7,
        hover_name="NTA",
        hover_data={"ride_count": ":,", "color_value": False, "NTA": False},
        labels={"ride_count": "Rides"},
        title=title,
    )
    
    # Update colorbar to show actual ride counts (not sqrt values)
    # Calculate tick values based on actual ride count range
    if max_rides > 0:
        tick_vals = [0, np.sqrt(max_rides * 0.1), np.sqrt(max_rides * 0.25), 
                     np.sqrt(max_rides * 0.5), np.sqrt(max_rides)]
        tick_text = ["0", f"{int(max_rides * 0.1):,}", f"{int(max_rides * 0.25):,}", 
                     f"{int(max_rides * 0.5):,}", f"{int(max_rides):,}"]
    else:
        tick_vals = [0]
        tick_text = ["0"]
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=600,
        coloraxis_colorbar=dict(
            title="Rides",
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
    )
    
    # Highlight the selected NTA if provided
    if selected_nta:
        # Find the selected NTA feature
        selected_feature = None
        for feature in geojson["features"]:
            if feature["id"] == selected_nta:
                selected_feature = feature
                break
        
        if selected_feature:
            # Add a trace to highlight the selected NTA
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson={
                        "type": "FeatureCollection",
                        "features": [selected_feature]
                    },
                    locations=[selected_nta],
                    featureidkey="id",
                    z=[1],  # Dummy value
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    marker=dict(
                        line=dict(width=3, color="blue"),
                        opacity=0.3,
                    ),
                    hoverinfo="text",
                    text=[f"Selected: {selected_nta}"],
                    name=selected_nta,
                )
            )
    
    return fig
