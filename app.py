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

current_plot_type = 'county'  # or 'scatter' or 'county' or 'state'

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
    h3_cells_geojson = json.load(f)
# Load information about the counties
counties_df = gpd.read_file(data_folder / "counties_processed.geojson")
with open(data_folder / "counties_processed.geojson", "r") as f:
    counties_geojson = json.load(f)
# Load information about the states
states_df = gpd.read_file(data_folder / "us-states.json")
with open(data_folder / "us-states.json", "r") as f:
    states_geojson = json.load(f)
# Load the traffic data
traffic = pd.read_parquet(data_folder / "traffic.parquet", engine="fastparquet")
traffic = traffic.sample(200000)  # Use a subset for performance
logger.info("Only keeping relevant columns...")
# ADD MORE AS NEEDED
relevant_columns = [
    'ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
    'End_Lat', 'End_Lng', 'Distance(mi)', #'Description',
    'County', 'State', 'Zipcode', 'Timezone', 'Weather_Condition', "h3cell",
    "geoid", "Crossing", "Junction", "Stop", "Traffic_Signal","Civil_Twilight"
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

# TODO:  hexagons without accidents are thrown arway in clean data script
def bin_data_by_h3(df):
    logger.info("Aggregating based on H3 cells...")
    # Aggregate data based on H3 cells
    filtered_grouped = df.groupby('h3cell', observed=False).size().reset_index(name='n_accidents')
    # set n_accidents in h3_df based on filtered data
    filtered_cells = h3_df[["h3cell", "h3_lat", "h3_lng"]].merge(filtered_grouped, on='h3cell', how='left')
    filtered_cells['n_accidents'] =  filtered_cells['n_accidents'].fillna(1) #np.log(filtered_cells['n_accidents'].fillna(1))
    return filtered_cells


def bin_data_by_county(df):
    logger.info("Aggregating based on counties...")
    # Aggregate data based on counties
    filtered_grouped = df.groupby('geoid').size().reset_index(name='n_accidents')
    # set n_accidents in counties_df based on filtered data
    filtered_bins = counties_df[["NAME", "GEOID","Density","Population"]].merge(
        filtered_grouped, left_on='GEOID', right_on='geoid', how='left')
    filtered_bins['n_accidents'] = filtered_bins['n_accidents'].fillna(1) #np.log(filtered_bins['n_accidents'].fillna(1))
    return filtered_bins


def bin_data_by_state(df):
    logger.info("Aggregating based on states...")
    # Aggregate data based on states
    filtered_grouped = df.groupby('State', observed=False).size().reset_index(name='n_accidents')
    # set n_accidents in states_df based on filtered data
    filtered_bins = states_df[["name", "id"]].merge(
        filtered_grouped, left_on='id', right_on='State', how='left')
    filtered_bins['n_accidents'] = filtered_bins['n_accidents'].fillna(1) #np.log(filtered_bins['n_accidents'].fillna(1))
    return filtered_bins


def filter_data(selected_weather,Surrounding,Time):
    logger.info("Filtering data for weather condition: %s", selected_weather)
    returnData = traffic
    if selected_weather != "Any":
        returnData =  returnData[returnData['Weather_Condition'] == selected_weather]
    if Surrounding != 'None':
        returnData = returnData[returnData[Surrounding]==True]
    if Time != "Any":
        returnData = returnData[returnData["Civil_Twilight"]==Time]
    return returnData


filtered = filter_data('Any','None',"Any") # a global variable to hold filtered data

# TODO this needs to match the intitial plot type
filtered_bins = bin_data_by_state(filtered) # a global variable to hold the current


logger.debug("Initial binned data size: %d", filtered_bins.shape[0])
# binned data

# Get unique weather conditions for dropdown
weather_options = [{'label': w, 'value': w} for w in sorted(traffic['Weather_Condition'].dropna().unique())]
weather_options.append({'label':'Any','value':'Any'})


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
              [Input('weather-dropdown', 'value'),
              Input('Surrounding', 'value'),
              Input('Time', 'value'),],
              prevent_initial_call=True)
def refilter_data(selected_weather,Surrounding,Time):
    global filtered
    filtered = filter_data(selected_weather,Surrounding,Time)

    # If hexbin, update the binned data
    # If scatterplot, rebuild the spatial index
    global current_plot_type
    global filtered_bins
    if current_plot_type == 'hexbin':
        filtered_bins = bin_data_by_h3(filtered)
    elif current_plot_type == 'county':
        filtered_bins = bin_data_by_county(filtered)
    elif current_plot_type == 'state':
        filtered_bins = bin_data_by_state(filtered)
    elif current_plot_type == 'scatter':
        global sindex
        sindex = filtered.sindex  # rebuild spatial index for filtered data

    return time()


# called by the more general update_figure function below
def update_scattermap_figure(filtering_state, map_layout):

    # filtering_state is a dummy variable to trigger updates whenever we filter
    # so we don't need to pass around the whole dataframe in a dcc.Store
    global filtered
    lat, lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)

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
    [Input('map_figure', 'relayoutData'),
     State('plot-type-radio', 'value')],
)
def update_map_on_relayout(relayout_data, selected_plot_type):
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
        if current_plot_type == 'scatter': # see if we shoud switch back to selected plot type
            logger.info("Switching back to selected plot type: %s", selected_plot_type)
            current_plot_type = selected_plot_type
            should_update_map = True
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


def extract_lat_lng_zoom_from_layout(layout):
    if layout is None or len(layout) <= 1:
        return START_COORDINATES['lat'], START_COORDINATES['lon'], START_ZOOM
    lat, lng = layout[0]
    zoom = layout[1]
    return lat, lng, zoom


def create_hexbin_figure(df, zoom=3, center=None):
    fig = px.choropleth_map(
        df,
        geojson=h3_cells_geojson,
        locations='h3cell',
        featureidkey='properties.h3cell',
        color= np.log10(df['n_accidents']),
        color_continuous_scale="Viridis",
        map_style="light",
        zoom=zoom,
        #range_color= [np.min(df['n_accidents']), np.max(df['n_accidents'])],
        center=center,
        hover_data={'h3cell': True, 'n_accidents': True},
        opacity=1,
        width=1000,
        height=700
    )
    fig.update_traces(marker_line_width=0)
    return fig


# called by the more general update_figure function below
def update_hexbin_figure(filtering_state, map_layout):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    global filtered_bins
    lat, lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)
    fig = create_hexbin_figure(filtered_bins, zoom=zoom, center={"lat": lat, "lon": lng})
    return fig


def create_county_figure(df, zoom=3, center=None):
    fig = px.choropleth_map(
        df,
        geojson=counties_geojson,
        locations='GEOID',
        featureidkey='properties.GEOID',
        color= np.log10(df['n_accidents']),
        color_continuous_scale="Viridis",
        map_style="light",
        zoom=zoom,
        #range_color=[np.min(df['n_accidents']), np.max(df['n_accidents'])],
        center=center,
        hover_data={'NAME': True, 'n_accidents': True},
        opacity=1,
        width=1000,
        height=700
    )
    return fig


def update_county_figure(filtering_state, map_layout):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    global filtered_bins
    lat, lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)
    fig = create_county_figure(filtered_bins, zoom=zoom, center={"lat": lat, "lon": lng})
    return fig


def create_state_figure(df, zoom=3, center=None):
    fig = px.choropleth_map(
        df,
        geojson=states_geojson,
        locations='name',
        featureidkey='properties.name',
        color= np.log10(df['n_accidents']),
        color_continuous_scale="Viridis",
        map_style="light",
        zoom=zoom,
        #range_color=[np.min(df['n_accidents']), np.max(df['n_accidents'])], #done automatically
        center=center,
        hover_data={'name': True, 'n_accidents': True},
        opacity=1,
        width=1000,
        height=700
    )
    return fig


def update_state_figure(filtering_state, map_layout):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    global filtered_bins
    lat, lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)
    fig = create_state_figure(filtered_bins, zoom=zoom, center={"lat": lat, "lon": lng})
    return fig



@app.callback(Output('map_figure', 'figure'),
              [Input('filtered-state', 'value'),
               Input('map_layout', 'data'),
               Input('plot-type-radio', 'value'),
               State('weather-dropdown', 'value'),
               State('Surrounding', 'value'),
               State('Time', 'value'),],
              prevent_initial_call=True)
def update_figure(filtering_state, layout, selected_plot_type, weather_condition, Surrounding,Time):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    global current_plot_type

    # If we are within scatter plot zoom range, always use scatter plot
    # Otherwise, use the selected plot type
    if current_plot_type == 'scatter':
        return update_scattermap_figure(filtering_state, layout)
    else:
        if selected_plot_type != current_plot_type:
            logger.info("Switching to selected plot type: %s", selected_plot_type)
            current_plot_type = selected_plot_type
            refilter_data(weather_condition, Surrounding,Time)
    if current_plot_type == 'county':
        return update_county_figure(filtering_state, layout)
    elif current_plot_type == 'hexbin':
        return update_hexbin_figure(filtering_state, layout)
    elif current_plot_type == 'state':
        return update_state_figure(filtering_state, layout)

width = 100#+"%"
app.layout = html.Div([
    html.H1("Traffic Incidents by Weather Condition"), #title
    html.Div([  #Filters 
        html.Div([ #View filter
            html.Label("View"),
            dcc.Dropdown(
                id='plot-type-radio',
                options=[
                    {'label': 'State', 'value': 'state'},
                    {'label': 'County', 'value': 'county'},
                    {'label': 'Hexagon', 'value': 'hexbin'},
                ],
                value='state',
                clearable=False,
            )     
        ],style = {'padding': 10,'width':100},),
        html.Div([ #Weather filter
            html.Label("Weather"),
            dcc.Dropdown(
                id='weather-dropdown',
                options=weather_options,
                value='Any',
                clearable=False
            ),
        ],style = {'padding': 10,'width':250},),
        html.Div([ #Time filter
            html.Label("Time"),
            dcc.Dropdown(
                id='Time',
                options=["Any","Day","Night"],
                value='Any',
                clearable=False
            ),
        ],style = {'padding': 10,'width':100},),
        html.Div([ #Surrounding filter
            html.Label("Surrounding"),
            dcc.Dropdown(
                options = ['None','Crossing',"Junction","Stop","Traffic_Signal"],
                value="None",
                id="Surrounding",
                clearable=False,              
            ), 
        ],style = {'padding': 10,'width':150},),        
    ], style = {'display': 'flex', 'flexDirection': 'row'},), #End of filters


    dcc.Graph(id="map_figure",
        figure = create_state_figure(filtered_bins,
        zoom=START_ZOOM, center=START_COORDINATES)),
    dcc.Input(id='filtered-state', type='hidden', value='init'),
    # Format is [[lat, lng], zoom]
    dcc.Store(id='map_layout', data=[
        [[START_COORDINATES['lat'], START_COORDINATES['lon']], START_ZOOM]
    ]),
])

if __name__ == '__main__':
    app.run(debug=True)