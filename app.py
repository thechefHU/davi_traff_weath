import dash
from dash import html, Input, Output, State, dcc
from pathlib import Path
import plotly.express as px
import pandas as pd
import geopandas as gpd
from time import time
import numpy as np
import logging
import json
from shapely.geometry import Point
from shapely import box

current_plot_type = 'hexbin'  # or 'scatter'

START_COORDINATES = {"lat": 37.0902, "lon": -95.7129}
START_ZOOM = 3

SCATTER_PLOT_ZOOM_THRESHOLD = 7  # zoom level above which we switch to scatter plot

# global variable to hold the last map layout used for geographic filtering
# We do this to avoid excessive geographic filtering when the user is just panning/zooming a little
# around the same area
map_layout_on_last_geofilter = [[START_COORDINATES['lat'], START_COORDINATES['lon']], START_ZOOM]

external_stylesheets = []
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('accidents-dashboard')
logger.info("\nWelcome to the coolest dashboard ever!")
# Load data
data_folder = Path("data")
logger.info("Loading data...")
# Load information about the H3 cells
h3_df = pd.read_parquet(data_folder / "h3_cells.parquet", engine="fastparquet")
with open(data_folder / "h3_cells.geojson", "r") as f:
    geojson_obj = json.load(f)

traffic = pd.read_parquet(data_folder / "traffic.parquet", engine="fastparquet")
#traffic = traffic.sample(200000)  # Use a subset for performance
logger.info("Only keeping relevant columns...")
# ADD MORE AS NEEDED
relevant_columns = [
    'ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
    'End_Lat', 'End_Lng', 'Distance(mi)', #'Description',
    'County', 'State', 'Zipcode', 'Timezone', 'Weather_Condition', "h3cell",
    "geoid"
]
traffic = traffic[relevant_columns]
# Set Start_Lat and Start_Lng to float32 to save memory
traffic['Start_Lat'] = traffic['Start_Lat'].astype('float32')
traffic['Start_Lng'] = traffic['Start_Lng'].astype('float32')
traffic['End_Lat'] = traffic['End_Lat'].astype('float32')
traffic['End_Lng'] = traffic['End_Lng'].astype('float32')
logger.info("Converting to GeoDataFrame and building spatial index...")
traffic = gpd.GeoDataFrame(
    traffic,
    geometry=gpd.points_from_xy(traffic.Start_Lng, traffic.Start_Lat),
)
sindex = traffic.sindex


def bin_data_by_h3(df):
    logger.info("Aggregating based on H3 cells...")
    # Aggregate data based on H3 cells
    filtered_grouped = df.groupby('h3cell').size().reset_index(name='n_accidents')
    # set n_accidents in h3_df based on filtered data
    filtered_cells = h3_df[["h3cell", "h3_lat", "h3_lng"]].merge(filtered_grouped, on='h3cell', how='left')
    return filtered_cells

def filter_data(selected_weather):
    logger.info("Filtering data for weather condition: %s", selected_weather)
    filtered_traffic = traffic[traffic['Weather_Condition'] == selected_weather]
    
    return filtered_traffic

filtered = filter_data("Clear") # a global variable to hold filtered data

filtered_bins = bin_data_by_h3(filtered) # a global variable to hold the current
logger.debug("Initial binned data size: %d", filtered_bins.shape[0])
# binned data

# Get unique weather conditions for dropdown
weather_options = [{'label': w, 'value': w} for w in sorted(traffic['Weather_Condition'].dropna().unique())]








def filter_geographic_bounds(df, lat=None, lng=None, width=6, height=4.0):
    """Only return points visible within a rectangle around the given lat/lng"""
    if lat is None or lng is None:
        return df
    lat_min = lat - height / 2
    lat_max = lat + height / 2
    lng_min = lng - width / 2
    lng_max = lng + width / 2
    start_time = time()
    bbox = box(lng_min, lat_min, lng_max, lat_max)
    global sindex
    matches = sindex.query(bbox)
    filtered_df = df.iloc[matches]
    logger.info("Geographic filtering took: %s seconds", time() - start_time)
    logger.info("Filtered data size: %d", filtered_df.shape[0])
    return filtered_df

def get_point_size(zoom):
    min_zoom, max_zoom = 6, 15
    min_size, max_size = 3, 6
    if zoom < min_zoom:
        return min_size
    if zoom > max_zoom:
        return max_size
    # Linear interpolation between min and max size
    size = min_size + (zoom - min_zoom) * (max_size - min_size) / (max_zoom - min_zoom)
    return size

def get_opacity(zoom):
    min_zoom, max_zoom = 6, 20
    min_opacity, max_opacity = 0.02, 0.6
    if zoom < min_zoom:
        return min_opacity
    if zoom > max_zoom:
        return max_opacity
    # Linear interpolation between min and max opacity
    opacity = min_opacity + (zoom - min_zoom) * (max_opacity - min_opacity) / (max_zoom - min_zoom)
    return opacity

def create_scattermap_figure(df, zoom=3, center=None):
    
    fig = px.scatter_map(
        df,
        lat="Start_Lat",
        lon="Start_Lng",
        #hover_name="Description",
        hover_data={"Start_Lat": False, "Start_Lng": False, "Severity": True, "Start_Time": True},
        zoom=zoom,
        center=center,
        map_style="light",
        width=1000, 
        height=700
    )
    fig.update_traces(marker=dict(size=get_point_size(zoom)),
                       opacity=get_opacity(zoom),
                       marker_color='black')
    return fig

@app.callback(Output('filtered-state', 'value'),
              [Input('weather-dropdown', 'value')],
              prevent_initial_call=True)
def refilter_data(selected_weather):
    global filtered
    filtered = filter_data(selected_weather)

    # If hexbin, update the binned data
    # If scatterplot, rebuild the spatial index
    global current_plot_type
    if current_plot_type == 'hexbin':
        global filtered_bins
        filtered_bins = bin_data_by_h3(filtered)
    elif current_plot_type == 'scatter':
        global sindex
        sindex = filtered.sindex  # rebuild spatial index for filtered data

    return time()


# called by the more general update_figure function below
def update_scattermap_figure(filtered_state, map_layout):

    # filtered_state is a dummy variable to trigger updates whenever we filter
    # so we don't need to pass around the whole dataframe in a dcc.Store
    global filtered
    
    if map_layout is None or len(map_layout) == 0:
        map_layout = [[START_COORDINATES['lat'], START_COORDINATES['lon']], START_ZOOM]
    lat, lng = map_layout[0]
    zoom = map_layout[1]

    # only show points within the current map bounds
    within_bounds = filter_geographic_bounds(filtered, lat=lat, lng=lng)

    fig = create_scattermap_figure(within_bounds, zoom=zoom, center={
        "lat": lat,
        "lon": lng
    })

    return fig

# Updating of the map based on zoom level and panning
@app.callback(
    Output('map_layout', 'data'),
    Input('map_figure', 'relayoutData')
)
def update_map_on_relayout(relayout_data):
    if relayout_data is None:
        return dash.no_update

    # Check if the center and zoom have moved since last time
    global map_layout_on_last_geofilter
    global current_plot_type
    max_dist = 1.5  # degrees
    max_zoom_change = 1.5  # zoom levels

    # Whether we should update the map layout or not
    # This happens if we switch plot types, or if we are in scatter plot mode
    # and the map is panned/zoomed significantly
    should_update_map = False

    # Check if we need to switch plot types
    if relayout_data['map.zoom'] < SCATTER_PLOT_ZOOM_THRESHOLD:
        if current_plot_type != 'hexbin':
            logger.info("Switching to hexbin plot")
            should_update_map = True
        current_plot_type = 'hexbin'
    else:
        if current_plot_type != 'scatter':
            logger.info("Switching to scatter plot")
            # Need to rebuild spatial index for filtered data
            global sindex
            sindex = filtered.sindex
            should_update_map = True
        current_plot_type = 'scatter'

    if 'map.center' in relayout_data and 'map.zoom' in relayout_data:
        if (abs(relayout_data['map.center']['lat'] - map_layout_on_last_geofilter[0][0]) > max_dist or
            abs(relayout_data['map.center']['lon'] - map_layout_on_last_geofilter[0][1]) > max_dist or
            abs(relayout_data['map.zoom'] - map_layout_on_last_geofilter[1]) > max_zoom_change):
            logging.debug("Map center and zoom change changed significantly, updating")
            should_update_map = True

    if not should_update_map:
        logging.debug("Map center and zoom change not significant, not updating")
        return dash.no_update
    else:
        logging.debug("Updating map with new relayout data: %s", relayout_data)
        map_layout = [[
            relayout_data['map.center']['lat'], relayout_data['map.center']['lon']
        ], relayout_data['map.zoom']]
        map_layout_on_last_geofilter = map_layout
    return map_layout


def create_hexbin_figure(df, zoom=3, center=None):
    fig = px.choropleth_map(
        df,
        geojson=geojson_obj,
        locations='h3cell',
        featureidkey='properties.h3cell',
        color='n_accidents',
        color_continuous_scale="Viridis",
        map_style="light",
        zoom=zoom,
        range_color=[0, df['n_accidents'].quantile(0.9)],
        center=center,
        hover_data={'h3cell': True, 'n_accidents': True},
        opacity=0.3,
        width=1000,
        height=700
    )
    fig.update_traces(marker_line_width=0)
    return fig

# called by the more general update_figure function below
def update_hexbin_figure(filtered_state, map_layout):
    # filtered_state is a dummy variable to trigger updates whenever we filter
    global filtered_bins
    if map_layout is None or len(map_layout) <= 1:
        map_layout = [[START_COORDINATES['lat'], START_COORDINATES['lon']], START_ZOOM]
    lat, lng = map_layout[0]
    zoom = map_layout[1]
    fig = create_hexbin_figure(filtered_bins, zoom=zoom, center={"lat": lat, "lon": lng})
    return fig


@app.callback(Output('map_figure', 'figure'),
              [Input('filtered-state', 'value'),
               Input('map_layout', 'data')],
              prevent_initial_call=True)
def update_figure(filtered_state, layout):
    # filtered_state is a dummy variable to trigger updates whenever we filter
    global current_plot_type
    if current_plot_type == 'scatter':
        return update_scattermap_figure(filtered_state, layout)
    elif current_plot_type == 'hexbin':
        return update_hexbin_figure(filtered_state, layout)


# initial_scattermap_data = filter_geographic_bounds(
#     filtered, 
#     lat=START_COORDINATES['lat'], lng=START_COORDINATES['lon'])
# print(f"Initial data size: {initial_scattermap_data.shape}")
# App layout

app.layout = html.Div([
    html.H1("Traffic Incidents by Weather Condition"),
    dcc.Dropdown(
        id='weather-dropdown',
        options=weather_options,
        value='Clear',
        clearable=False
    ),
    dcc.Graph(id="map_figure",
              figure = create_hexbin_figure(filtered_bins,
                 zoom=START_ZOOM, center=START_COORDINATES)),
    dcc.Input(id='filtered-state', type='hidden', value='init'),
    # Format is [[lat, lng], zoom]
    dcc.Store(id='map_layout', data=[
        [[START_COORDINATES['lat'], START_COORDINATES['lon']], START_ZOOM]
    ])
])

if __name__ == '__main__':
    app.run(debug=True)