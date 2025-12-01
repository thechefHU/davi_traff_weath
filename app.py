import dash
from dash import html, Input, Output, State, dcc
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, base64
import matplotlib
matplotlib.use("Agg")  # Must come before pyplot import
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import geopandas as gp
from time import time
import logging
import global_state as gs
from shapely import box
from datetime import date, datetime
import dash_bootstrap_components as dbc
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle

# Import the chart layouts here
import chart_accidents_over_time as chart_time
import chart_accidents_by_weather as chart_weather
import chart_accidents_by_hour as chart_accidents_by_hour
import chart_accidents_by_weekday as chart_accidents_by_weekday
import chart_trend

# Setup variables
current_plot_type = 'county'  # or 'scatter' or 'county' 
START_COORDINATES = {"lat": 36.4, "lon": -118.39} # Center on California
START_ZOOM = 4.5
SCATTER_PLOT_ZOOM_THRESHOLD = 10 # zoom level above which we switch to scatter plot

PLOT_WIDTH = 700
PLOT_HEIGHT = 600

# global variable to hold the last map layout used for geographic filtering
# We do this to avoid excessive geographic filtering when the user is just panning/zooming a little
# around the same area
map_layout_on_last_geofilter = [[START_COORDINATES['lat'], START_COORDINATES['lon']], START_ZOOM, [-1, -1, 1, 1]]
last_plot_type_before_scatter = 'county'  # to remember the last plot type before switching to scatter
# --- Initialize the Dash App with Bootstrap and FontAwesome for icons ---
# Dash will automatically serve any CSS file placed in an 'assets' folder.

#external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]
external_stylesheets = [dbc.icons.FONT_AWESOME]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('accidents-dashboard')
logger.info("\nWelcome to the coolest dashboard ever!")
# Load data

logger.info("Loading data...")

filter_dict = {}  # global variable to hold current filter conditions

gs.load_data(data_folder="data/")  # load a subset for faster testing

min_date = gs.get_data()['Start_Time'].min().date()
max_date = gs.get_data()['Start_Time'].max().date()

# Get unique weather conditions for dropdown
weather_options = sorted(gs.get_data()['Weather_Group'].dropna().unique())


def filter_geographic_bounds(df, lat_min=None, lat_max=None, lng_min=None, lng_max=None):
    """Only return points visible within a rectangle around the given lat/lng"""
    if lat_min is None or lat_max is None or lng_min is None or lng_max is None:
        return df
    start_time = time()
    bbox = box(lng_min, lat_min, lng_max, lat_max)
    sindex = gs.get_spatial_index()
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
        custom_data=["ID"],  # to get point indices for brushing
        zoom=zoom,
        center=center,
        map_style="light",
        width=PLOT_WIDTH, 
        height=PLOT_HEIGHT,
        opacity=0.7
    )
    
    fig.update_traces(cluster=dict(enabled=True))
    return fig

@app.callback(Output('Severity', 'value', allow_duplicate=True),
              [Input('SeveritySelectAll','value'),
               State('Severity','value')],
              prevent_initial_call=True)
def SeveritySelectAll(selectAll,Severity):
    if len(selectAll)==1:
        return [1,2,3,4]
    else:
        return Severity

def create_heatmap_figure(df, zoom=3, center=None, lat_min=None, lat_max=None, lng_min=None, lng_max=None, 
                          detail_level=0,
                           scale=0, opacity=0.6):
    # Create a density heatmap figure

    logger.info("Creating heatmap figure...")
    if lat_min is None or lat_min == -1:
        lng_min = df['Start_Lng'].min()
        lng_max = df['Start_Lng'].max()
        lat_min = df['Start_Lat'].min()
        lat_max = df['Start_Lat'].max()

    if detail_level == 0:
        sample_size = 10000
        nx = 40
        ny = 28
    elif detail_level == 1:
        sample_size = 50000
        nx = 80
        ny = 56
    else:
        sample_size = 200000
        nx = 120
        ny = 84

    if len(df) > sample_size:
        df = df.sample(sample_size)
    # 2. Compute density (heatmap) on a grid
    xi = np.linspace(lng_min, lng_max, nx)
    yi = np.linspace(lat_min, lat_max, ny)
    X, Y = np.meshgrid(xi, yi)
    kde_start_time = time()

    if len(df) > 0: 
        kde = gaussian_kde(np.vstack([df['Start_Lng'], df['Start_Lat']]))
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    else:
        Z = np.zeros(X.shape)
    logger.info("KDE computation took: %s seconds", time() - kde_start_time)
    logger.info("Density computation done.")
    # 3. Render raster image
    encode_start_time = time()
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ax.imshow(Z, extent=[lng_min, lng_max, lat_min, lat_max], origin='lower', cmap='hot')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode()
    logger.info("Encoding image took: %s seconds", time() - encode_start_time)



    # 4. Plotly figure with MapLibre base map and overlay
    fig = px.scatter_map(
        lat=[0],
        lon=[0],
        opacity=0,
        zoom=zoom,
        center=center,
        map_style="dark",
        width=PLOT_WIDTH, 
        height=PLOT_HEIGHT
    )


    fig.update_layout(
        map_layers=[
            {
                "sourcetype": "image",
                "below": '',
                "source": "data:image/png;base64," + encoded,
                "opacity": 0.4,
                "coordinates": [
                    [lng_min, lat_max],  # top-left (lon, lat)
                    [lng_max, lat_max],  # top-right
                    [lng_max, lat_min],  # bottom-right
                    [lng_min, lat_min],  # bottom-left
                ],
            }
        ],
    )


    return fig


def update_heatmap_figure(filtering_state, map_layout, scale=0, opacity=0.6, detail_level=0):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    center_lat, center_lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)
    lat_min, lat_max, lng_min, lng_max, _ = extract_bounds_zoom_from_layout(map_layout)
    within_bounds = filter_geographic_bounds(gs.get_data(), lat_min=lat_min, lat_max=lat_max,
                                            lng_min=lng_min, lng_max=lng_max)
    fig = create_heatmap_figure(within_bounds, zoom=zoom, center={"lat": center_lat, "lon": center_lng},
                                lat_min=lat_min, lat_max=lat_max, lng_min=lng_min, lng_max=lng_max,
                                scale=scale, opacity=opacity, detail_level=detail_level)
    return fig



@app.callback(Output('filter-ui-trigger', 'value', allow_duplicate=True),
              Output('SeveritySelectAll','value'),
              [Input('Severity', 'value')],
              prevent_initial_call=True)
def Severity_updated(Severity):
    global filter_dict
    string = ""
    if not Severity: #No category 
        if "Severity" in filter_dict: # Excessive error catching
            filter_dict.pop("Severity")
    else:
        for i in Severity:
            string += f"Severity == {i} | "
        string = string[:-3]
        filter_dict["Severity"] = string
    if len(Severity)<4:
        return time(), []
    else:
        return time(), ["Select All"] # return a dummy value to trigger the next callback
    


@app.callback(Output('weather-dropdown', 'value', allow_duplicate=True),
              [Input('WeatherSelectAll','value'),
               State('weather-dropdown','value')],
              prevent_initial_call=True)
def weatherSelectAll(selectAll,weather):
    if len(selectAll)==1:
        return weather_options
    else:
        return weather

@app.callback(Output('filter-ui-trigger', 'value', allow_duplicate=True),
              Output('WeatherSelectAll', 'value'),
              [Input('weather-dropdown', 'value')],
              prevent_initial_call=True)
def weather_dropdown_updated(selected_weather):
    global filter_dict
    string = ""
    if not selected_weather: #No category 
        if "weather" in filter_dict: # Excessive error catching
            filter_dict.pop("weather")
    else:
        for i in selected_weather:
            string += f"Weather_Group == '{i}' | "
        string = string[:-3]
        filter_dict["weather"] = string
    if len(selected_weather)<len(weather_options):
        return time(), []
    else:
        return time(), ["Select All"] # return a dummy value to trigger the next callback



@app.callback(Output('Surrounding', 'value', allow_duplicate=True),
              [Input('SurroundingSelectAll','value'),
               State('Surrounding','value')],
              prevent_initial_call=True)
def surroundingSelectAll(selectAll,surrounding):
    if len(selectAll)==1:
        return ['Crossing',"Junction","Stop","Traffic_Signal","No_Surroundings"]
    else:
        return surrounding

@app.callback(Output('filter-ui-trigger', 'value', allow_duplicate=True),
              Output('SurroundingSelectAll','value'),
              [Input('Surrounding', 'value')],
              prevent_initial_call=True)
def surrounding_updated(selected_surrounding):
    global filter_dict
    string = ""
    if not selected_surrounding: #No category 
        if "surrounding" in filter_dict: # Excessive error catching
            filter_dict.pop("surrounding")
    else:
        for i in selected_surrounding:
            if i=="No_Surroundings":
                string += "(Crossing== False & Traffic_Signal == False & Stop== False & Junction == False) | "
            else: 
                string += f"{i} == True | "
        string = string[:-3]
        filter_dict["surrounding"] = string
    print(string)
    if len(selected_surrounding)<5:
        return time(), []
    else:
        return time(), ["Select All"] # return a dummy value to trigger the next callback


@app.callback(Output('Day', 'value', allow_duplicate=True),
              [Input('DaySelectAll','value'),
               State('Day','value')],
              prevent_initial_call=True)
def daySelectAll(selectAll,Day):
    if len(selectAll)==1:
        return ['Day','Night']
    else:
        return Day

@app.callback(Output('filter-ui-trigger', 'value', allow_duplicate=True),
              Output('DaySelectAll','value'),
              [Input('Day', 'value')],
              prevent_initial_call=True)
def day_updated(selected_day):
    global filter_dict
    string = ""
    if not selected_day: #No category 
        if "Day" in filter_dict: # Excessive error catching
            filter_dict.pop("Day")
    else:
        for i in selected_day:
            string += f"Sunrise_Sunset == '{i}' | "
        string = string[:-3]
        filter_dict["Day"] = string
    if len(selected_day)<2:
        return time(), []
    else:
        return time(), ["Select All"] # return a dummy value to trigger the next callback

    return time() # return a dummy value to trigger the next callback

@app.callback(Output('filter-ui-trigger', 'value', allow_duplicate=True),
                Output('dateRange','start_date'),
                Output('dateRange','end_date'),
                [Input('date-range-slider', 'value')],
                prevent_initial_call=True)
def time_range_updated(selected_range):
    global filter_dict
    min_date, max_date = selected_range
    min_date = date.fromordinal(int(min_date))
    max_date = date.fromordinal(int(max_date))
    filter_dict["time"] = f"Start_Time >= '{min_date}' & Start_Time <= '{max_date}'"

    return time(), str(min_date), str(max_date) # return a dummy value to trigger the next callback

@app.callback(Output('date-range-slider', 'value', allow_duplicate=True),
                [Input('dateRange', 'start_date'),
                 Input('dateRange', 'end_date')],
                prevent_initial_call=True)
def time_range_updated(start,end):
    return (datetime.strptime(start,'%Y-%m-%d').toordinal(), datetime.strptime(end,'%Y-%m-%d').toordinal())


@app.callback([Output('filtered-state', 'value'),
            Output('detail-level', 'data', allow_duplicate=True),
            Output('map_layout', 'data', allow_duplicate=True),
            Output('num-filtered-accidents', 'children', allow_duplicate=True)],
              [Input('filter-ui-trigger', 'value'),
               State('map_layout', 'data')],
              prevent_initial_call=True)
def refilter_data(filter_ui_trigger, map_layout=None):
    # filter_ui_trigger is a dummy variable to trigger updates whenever we change the filter UI
    global filter_dict

    # filter the data
    gs.filter_data(filter_dict, logger=logger)
    # If hexbin, update the binned data
    # If scatterplot, rebuild the spatial index
    global current_plot_type
    if current_plot_type == 'hexbin':
        gs.bin_data_by_h3()
    elif current_plot_type == 'county':
        gs.bin_data_by_county()
    elif current_plot_type == 'scatter':
        gs.update_spatial_index()
    elif current_plot_type == 'heatmap':
        gs.update_spatial_index()

    return time(), {"level": 0, "level_zero_signature": list(extract_lat_lng_zoom_from_layout(map_layout))}, map_layout, f"{gs.get_data().shape[0]:,}"  # reset detail level on filter change



# called by the more general update_figure function below
def update_scattermap_figure(filtering_state, map_layout):

    # filtering_state is a dummy variable to trigger updates whenever we filter
    # so we don't need to pass around the whole dataframe in a dcc.Store
    lat, lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)
    lat_min, lat_max, lng_min, lng_max, _ = extract_bounds_zoom_from_layout(map_layout)
    # only show points within the current map bounds
    within_bounds = filter_geographic_bounds(gs.get_data(), lat_min=lat_min, lat_max=lat_max,
                                            lng_min=lng_min, lng_max=lng_max)

    fig = create_scattermap_figure(within_bounds, zoom=zoom, center={
        "lat": lat,
        "lon": lng
    })

    return fig


@app.callback([Output('geoselection-state', 'value'),
               Output('geoselection-info', 'children'),
               Output('clear-selection-button', 'style'),
               Output('num-filtered-selected-accidents', 'children', allow_duplicate=True)],
             [Input('map_figure', 'selectedData'),
              Input('clear-selection-button', 'n_clicks')], prevent_initial_call=True)
def selection_made(relayout_data, clear_selection_clicks):

    # check if the callback was triggered by clear button
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'clear-selection-button':
        gs.set_selection_bounds(None, None, None, None)
        return time(), "No selection, use the box select tool.", {'display': 'none'}, f"{gs.get_data_selected_by_bounds().shape[0]:,}"
    print("HEJEJ", relayout_data)
    
    if relayout_data is None or "range" not in relayout_data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    r = relayout_data["range"]
    if 'map' not in r:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # extract the latitude and longitude bounds from the selection rectangle
    m = r['map']
    lats = [m[0][1], m[1][1]]
    lngs = [m[0][0], m[1][0]]

    lat_min, lat_max = min(lats), max(lats)
    lng_min, lng_max = min(lngs), max(lngs)

    gs.set_selection_bounds(lat_min, lat_max, lng_min, lng_max)

    return time(), f"", {'display': True}, f"{gs.get_data_selected_by_bounds().shape[0]:,}"

    # get lat_min, lat_max, lng_min, lng_max, _ = extract_bounds_zoom_from_layout(map_layout)
    # if selected_data is None:
    #     return

    # logger.info("Brushed data points: %s", selected_data)

    # # Extract point indices from selected data

    # if current_plot_type == 'scatter':
    #     point_indices = [point['customdata'][0] for point in selected_data['points']]
    #     print(point_indices)
    #     #logger.info("Brushed data point indices: %s", point_indices)
    #     filter_str = f"ID in {point_indices}"
    #     filter_dict["brushed"] = filter_str
    # elif current_plot_type in 'county':
    #     counties = [point['location'] for point in selected_data['points']]
    #     filter_str = f"geoid in {counties}"
    #     filter_dict["brushed"] = filter_str
    # elif current_plot_type in 'hexbin':
    #     pass  # hexbin brushing not implemented yet

    
    # refilter_data(filter_ui_trigger=time())






# Updating of the map based on zoom level and panning
@app.callback(
    [Output('map_layout', 'data'),
     Output('plot-type-radio', 'options'),
     Output('plot-type-radio', 'value'),
     Output('detail-level', 'data', allow_duplicate=True)],
    [Input('map_figure', 'relayoutData'),
     State('plot-type-radio', 'value')],
    prevent_initial_call='initial_duplicate',
)
def update_map_on_relayout(relayout_data, selected_plot_type):
    if relayout_data is None or 'map.zoom' not in relayout_data or 'map.center' not in relayout_data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Check if the center and zoom have moved since last time
    global map_layout_on_last_geofilter     
    global current_plot_type

    # Map movement thresholds before updating the geographic filter
    max_dist = 0.0001  # degrees
    max_zoom_change = 0.10  # zoom levels

        # Whether we should update the map layout or not
    # This happens if we switch plot types, or if we are in scatter plot mode
    # and the map is panned/zoomed significantly
    should_update_map = False

    scatter_disabled = relayout_data['map.zoom'] < SCATTER_PLOT_ZOOM_THRESHOLD
    if current_plot_type == 'scatter' and scatter_disabled:
        logging.info("Zoomed out beyond scatter plot threshold, switching back to last plot type")
        global last_plot_type_before_scatter
        current_plot_type = last_plot_type_before_scatter
        should_update_map = True
    


    if 'map.center' in relayout_data and 'map.zoom' in relayout_data:
        if (abs(relayout_data['map.center']['lat'] - map_layout_on_last_geofilter[0][0]) > max_dist or
            abs(relayout_data['map.center']['lon'] - map_layout_on_last_geofilter[0][1]) > max_dist or
            abs(relayout_data['map.zoom'] - map_layout_on_last_geofilter[1]) > max_zoom_change):
            logging.info("Map center and zoom change changed significantly, updating")
            level_zero_signature = [relayout_data['map.center']['lat'], relayout_data['map.center']['lon'], relayout_data['map.zoom']]
            should_update_map = True
    else:
        level_zero_signature = [START_COORDINATES['lat'], START_COORDINATES['lon'], START_ZOOM]

    if not should_update_map:
        logging.info("Map center and zoom change not significant, not updating")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:   
        logging.info("Updating map with new relayout data: %s", relayout_data)
        coordinates = relayout_data['map._derived']['coordinates']

        map_layout = [[
            relayout_data['map.center']['lat'], relayout_data['map.center']['lon']
        ], relayout_data['map.zoom'], [coordinates[3][1], coordinates[3][0], coordinates[1][1], coordinates[1][0]]]
        map_layout_on_last_geofilter = map_layout

    # Update the options for the plot-type-radio element
    
    if scatter_disabled:
        scatter_label = 'Dotmap (zoom in to enable)'
    else:
        scatter_label = 'Dotmap'   
    

    plot_type_options = [
        {'label': 'County', 'value': 'county'},
        {'label': 'Hexagon', 'value': 'hexbin'},
        {'label': 'Heatmap', 'value': 'heatmap'},
        {'label': scatter_label, 'value': 'scatter', 'disabled': scatter_disabled}
    ]

    return map_layout, plot_type_options, current_plot_type, {"level": 0, "level_zero_signature": level_zero_signature}  # reset detail level on map move

def extract_lat_lng_zoom_from_layout(layout):
    if layout is None or len(layout) <= 1:
        return START_COORDINATES['lat'], START_COORDINATES['lon'], START_ZOOM
    lat, lng = layout[0]
    zoom = layout[1]
    return lat, lng, zoom


def extract_bounds_zoom_from_layout(layout):
    # Extract the lat/lng bounds and zoom from the stored layout
    if layout is None or len(layout) <= 2:
        logger.warning("WARNING; No map layout data found, using default bounds")
        logger.warning("Layout data was: %s", layout)
        return START_COORDINATES['lat']-1, START_COORDINATES['lat']+1, START_COORDINATES['lon']-1, START_COORDINATES['lon']+1, START_ZOOM

    lat_min, lng_min, lat_max, lng_max = layout[2]
    zoom = layout[1] if len(layout) > 1 else START_ZOOM

    return lat_min, lat_max, lng_min, lng_max, zoom



def create_hexbin_figure(df, zoom=3, center=None, scale=0, opacity=1):
    color, _ = get_color_and_range(scale==0, df['n_accidents'])
    fig = px.choropleth_map(
        df,
        geojson=gs.get_h3_geojson(),
        locations='h3cell',
        featureidkey='properties.h3cell',
        color= color,
        color_continuous_scale="Viridis",
        map_style="light",
        opacity=opacity,
        zoom=zoom,
        #range_color=[0, df['n_accidents'].quantile(0.9)],
        center=center,
        hover_data={'h3cell': True, 'n_accidents': True},
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )
    fig.update_coloraxes(colorbar_title={"text":"No. of accidents"})
    fig.update_coloraxes(colorbar_ticks="outside")
    tickvals = []
    ticktext = []
    for e in range(8):  # For each exponent
        for i in range(1, 10):  # For each base value 1-9
            tickvals.append(np.log10(i * 10**e))
            ticktext.append(str(i * 10**e) if i == 1 else "")  # Only label powers of 10
    fig.update_coloraxes(colorbar_tickvals=tickvals)
    fig.update_coloraxes(colorbar_ticktext=ticktext)
    fig.update_traces(marker_line_width=0.1)
    return fig


# called by the more general update_figure function below
def update_hexbin_figure(filtering_state, map_layout,scale,opacity):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    lat, lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)
    fig = create_hexbin_figure(gs.get_binned_data(), zoom=zoom, center={"lat": lat, "lon": lng},
                               scale=scale, opacity=opacity)
    return fig


def get_color_and_range(is_log, accidents):
    if is_log:
        accidents = np.array(accidents)
        # replace zeros with small value to avoid -inf in log scale
        accidents[accidents == 0] = 0.1
        accidentsLog = np.log10(
            np.array(accidents),
        )
        return accidentsLog, [np.min(accidentsLog),np.max(accidentsLog)]
    else:
        return accidents, [0, accidents.quantile(0.95)]



def create_county_figure(df, zoom=3, center=None, scale=0,opacity=0.7):
    color, _ = get_color_and_range(scale==0, df['n_accidents'])
    fig = px.choropleth_map(
        df,
        geojson=gs.get_counties_geojson(),
        locations='GEOID',
        featureidkey='properties.GEOID',
        color= color,
        color_continuous_scale="Viridis",
        map_style="light",
        zoom=zoom,
        #range_color=range,
        center=center,
        hover_data={'NAME': True, 'n_accidents': True},
        labels = {'NAME':'County'},
        opacity=opacity,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )   
    fig.update_coloraxes(colorbar_title={"text":"No. of accidents"})
    fig.update_coloraxes(colorbar_ticks="outside")
    tickvals = []
    ticktext = []
    for e in range(8):  # For each exponent
        for i in range(1, 10):  # For each base value 1-9
            tickvals.append(np.log10(i * 10**e))
            ticktext.append(str(i * 10**e) if i == 1 else "")  # Only label powers of 10
    fig.update_coloraxes(colorbar_tickvals=tickvals)
    fig.update_coloraxes(colorbar_ticktext=ticktext)
    return fig


def update_county_figure(filtering_state, map_layout,scale=0,opacity=1):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    lat, lng, zoom = extract_lat_lng_zoom_from_layout(map_layout)
    fig = create_county_figure(gs.get_binned_data(), zoom=zoom, center={"lat": lat, "lon": lng},
                                scale=scale, opacity=opacity)
    return fig




@app.callback(
    Output('detail-level', 'data'),
    [Input('progress-indicator-gif', 'style'),
     State('plot-type-radio', 'value'),
     State('detail-level', 'data'),
     State('map_layout', 'data')],
)
def update_detail_level(progress_indicator_dummy, selected_plot_type, detail_level, map_layout):
    # only increase detail level for heatmap plots
    # and when the progress indicator is invisble (the previous plot has finished rendering)
    if selected_plot_type == 'heatmap' and detail_level["level"] < 2 and progress_indicator_dummy['opacity'] == 0:
        level_zero_signature = list(extract_lat_lng_zoom_from_layout(map_layout))
        if level_zero_signature == detail_level["level_zero_signature"]:
            logger.info(f"Heatmap detail level is {detail_level["level"]}, increasing by one")
            return {"level": detail_level["level"] + 1, "level_zero_signature": detail_level["level_zero_signature"]}
        else:
            logger.info("Level zero signature changed, not increasing detail level")
    return dash.no_update


@app.callback(Output('map_figure', 'figure'),
              [Input('filtered-state', 'value'),
               Input('map_layout', 'data'),
               Input('plot-type-radio', 'value'),
               Input('detail-level', 'data')],
              prevent_initial_call=True,
              running=[
                  (Output('progress-indicator-gif', 'style'),
                   {
                       'top': '10px',
                       'right': '10px',
                       'width': '50px',
                       'height': '50px',
                       'maxWidth': '50px',
                       'opacity': 1,
                       'zIndex': 1000
                   },
                   {
                       'top': '10px',
                       'right': '10px',
                       'width': '50px',
                       'height': '50px',
                       'maxWidth': '50px',
                       'opacity': 0,
                       'zIndex': 1000
                   }
                  )
              ])
def update_figure(filtering_state, layout, selected_plot_type, detail_level, opacity=0.6):
    # filtering_state is a dummy variable to trigger updates whenever we filter
    global current_plot_type
    
    # For color scale
    # If we are within scatter plot zoom range, always use scatter plot
    # Otherwise, use the selected plot type

    if selected_plot_type != current_plot_type:
        logger.info("Switching to selected plot type: %s", selected_plot_type)
        if selected_plot_type == 'scatter':
            logger.info("Switching to scatter plot")
            # Need to rebuild spatial index for filtered data
            gs.update_spatial_index()
            current_plot_type = 'scatter'
        elif selected_plot_type == 'heatmap':
            logger.info("Switching to heatmap plot")
            gs.update_spatial_index()
            current_plot_type = 'heatmap'
        else:
            current_plot_type = selected_plot_type
            global filter_dict
            refilter_data(filter_dict)

    if current_plot_type == 'scatter':
        fig = update_scattermap_figure(filtering_state, layout)
    else:
        global last_plot_type_before_scatter
        last_plot_type_before_scatter = current_plot_type
    scale = 0 #hardcode to log easily this way without deleting all the code
    if current_plot_type == 'county':
        fig = update_county_figure(filtering_state, layout, scale, opacity)
    elif current_plot_type == 'hexbin':
        fig = update_hexbin_figure(filtering_state, layout, scale, opacity)
    elif current_plot_type == 'heatmap':
        fig = update_heatmap_figure(filtering_state, layout, scale, opacity, detail_level["level"])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig

# Add a callback for when "group-1-button" is clicked
@app.callback(
    Output('filtered-state', 'value', allow_duplicate=True),
    Input('group-1-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_group_1_points(n_clicks):
    if n_clicks is None:
        return dash.no_update
    # Count the number of points in the current figure
    gs.set_comparison_group(1)
    return time()
# Add a callback for when "group-2-button" is clicked
@app.callback(
    Output('filtered-state', 'value', allow_duplicate=True),
    Input('group-2-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_group_2_points(n_clicks):
    if n_clicks is None:
        return dash.no_update
    # Count the number of points in the current figure
    gs.set_comparison_group(2)
    return time()
# Add a callback for when "group-3-button" is clicked
@app.callback(
    Output('filtered-state', 'value', allow_duplicate=True),
    Input('clear-groups-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_comparison_groups(n_clicks):
    if n_clicks is None:
        return dash.no_update
    # Count the number of points in the current figure
    gs.clear_comparison_groups()
    return time()


@app.callback(
    [Output('group-1-points', 'children'),
     Output('group-2-points', 'children'),
     Output('clear-groups-button', 'style')],
     Input('filtered-state', 'value'),
)
def update_group_points_display(dummy_input):
    if len(gs.active_comparison_groups()) == 0:
        return "n=0", "n=0", {'display': 'none'}
    else:
        counts = [len(group) for group in gs._comparison_groups]
        return (f"n={counts[0]:,}",
                f"n={counts[1]:,}",
                {'display': 'inline-block'}
        )
app.layout = html.Div(style={'height': '100vh'}, children=[
    PanelGroup(
        id='panel-group',
        direction='horizontal',
        children=[
            # Left slim Filters panel
            Panel(defaultSizePercentage=15,
                  id='filters-panel',
                    style={
                        'flex': '1',
                        'padding': '8px',
                        'boxSizing': 'border-box',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'stretch',
                        'justifyContent': 'stretch',
                        'overflow':'scroll',
                    },
                    children = [
                        html.H2("Filters"),
                        html.H3("Severity"),
                        dcc.Checklist(['Select All'],
                                    [],
                                    id='SeveritySelectAll'),
                        dcc.Dropdown(
                            options=[1, 2, 3, 4],
                            value=[],
                            id="Severity",
                            clearable=True,
                            multi=True,
                        ),
                        html.H3("Weather Condition"),
                        dcc.Checklist(['Select All'],
                                    [],
                                    id='WeatherSelectAll'),
                        dcc.Dropdown(
                            id='weather-dropdown',
                            options=weather_options,
                            value=[],
                            clearable=True,
                            multi=True,
                        ),
                        html.H3("Time of Day"),
                        dcc.Checklist(['Select All'],
                                    [],
                                    id='DaySelectAll'),
                        dcc.Dropdown(
                            id='Day',
                            options=["Day", "Night"],
                            value=[],
                            clearable=True,
                            multi=True,
                        ),
                        html.H3("Surrounding infrastructure"),
                        dcc.Checklist(['Select All'],
                                    [],
                                    id='SurroundingSelectAll'),
                        dcc.Dropdown(
                            options=['Crossing', "Junction", "Stop", "Traffic_Signal","No_Surroundings"],
                            value=[],
                            id="Surrounding",
                            clearable=True,
                            multi=True,
                        ),
                        html.H3("Date"),
                        dcc.RangeSlider(
                            id='date-range-slider',
                            className='date-range-slider',
                            min=min_date.toordinal(),
                            max=max_date.toordinal(),
                            value=[min_date.toordinal(), max_date.toordinal()],
                            marks={date.toordinal(): date.strftime('%Y') for date in pd.date_range(min_date, max_date, freq='YS')},  # Show all years as marks
                        ),
                        dcc.DatePickerRange(
                            id="dateRange",
                            month_format="YYYY-MM-DD",
                            display_format="YYYY-MM-DD",
                            min_date_allowed="2017-01-01",
                            max_date_allowed="2022-12-31",
                            start_date="2017-01-01",
                            end_date= "2022-12-31",
                        ),

                       
                ]),
            PanelResizeHandle(html.Div(
                style={
                    "backgroundColor": "darkgrey",
                    "height": "100%",
                    "width": "5px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "position": "relative",  # Ensure relative positioning for stacking context
                    "zIndex": 1  # Ensure the icon is rendered on top
                },
                children=html.I(className="fa fa-arrows-alt-h", style={"color": "#444444", "fontSize": "14px", "zIndex": 2})
            )),
            # Middle large map panel
            Panel(defaultSizePercentage=50, style={
                'flex': '1',
                'padding': '8px',
                'boxSizing': 'border-box',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'stretch',
                'justifyContent': 'stretch',
                'overflow':'scroll',
            }, children=[
                html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'start'}, children=[
                    html.Div([
                        html.Div("Accidents after filtering ", style={'fontSize': '12px'}),
                        html.Div(f"{gs.get_data().shape[0]:,}", id="num-filtered-accidents", style={'fontSize': '20px', 'color': '#0074D9'}),
                    ]),
                    html.Div(style={'width': '40px'}),  # spacer
                    html.Div([
                        html.Div("Accidents after filtering+selection ", style={'fontSize': '12px'}),
                        html.Div(f"{gs.get_data().shape[0]:,}", id="num-filtered-selected-accidents", style={'fontSize': '20px', 'color': '#0074D9'}),
                    ]),
                     html.Div(style={'width': '40px'}),  # spacer
                    html.Div(children=[
                        html.Div("Geographical Selection", style={'fontSize': '12px'}),
                        html.P("No selection, use the box select tool.", id="geoselection-info", style={'color': "#7A7A7A", 'fontSize': '12px'}),
                        html.Button("Clear geographical selection", id="clear-selection-button", style={'display': 'none'}),
                    ]),
                ]),
                html.Hr(),
                html.Div(style = {'display': 'flex','flexDirection': 'row'},          
                    children = [dcc.RadioItems(
                        id='plot-type-radio',
                        options=[
                            {'label': 'County', 'value': 'county'},
                            {'label': 'Hexagon', 'value': 'hexbin'},
                            {'label': 'Heatmap', 'value': 'heatmap'},
                            {'label': 'Scatter (zoom in to enable)', 'value': 'scatter', 'disabled': True}
                        ],
                        value='county',
                        labelStyle={'display': 'inline-block', 'marginBottom': '6px'}
                    ),
                    html.Div(
                        id='progress-indicator-container',
                        children=[
                            html.Img(
                                id='progress-indicator-gif',
                                src='assets/progress_indicator.gif',
                                style={
                                    'top': '10px',
                                    'right': '10px',
                                    'width': '50px',
                                    'height': '50px',
                                    'maxWidth': '50px',
                                    'objectFit': 'contain',  # Ensure the image scales correctly
                                    'opacity': 0,
                                    'zIndex': 1000,
                                    
                                }
                            ),
                        ]
                    ),
                ]),
                dcc.Graph(
                    id="map_figure",
                    figure=create_county_figure(gs.get_binned_data(), zoom=START_ZOOM, center=START_COORDINATES).update_layout(margin=dict(l=20, r=20, t=20, b=20)),
                    style={'flex': '1', 'minHeight': '0'}  # allow the graph to fill vertical space
                ),
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'space-evenly', 'marginTop': '10px'},
                    children=[
                        html.Div([
                            html.Button("Set comparison group 1", id="group-1-button", style={'backgroundColor': 'rgb(204, 102, 119)'}),
                            html.Div("n=0", id="group-1-points", style={'textAlign': 'center', 'marginTop': '5px'})
                        ]),
                        html.Div([
                            html.Button("Set comparison group 2", id="group-2-button", style={'backgroundColor': 'rgb(221, 204, 119)'}),
                            html.Div("n=0", id="group-2-points", style={'textAlign': 'center', 'marginTop': '5px'})
                        ]),
                        html.Div([
                            html.Button("Clear comparison groups", id="clear-groups-button", style={'display': 'False'}),
                        ]),
                    ]
                ),
            ]),
            PanelResizeHandle(html.Div(
                style={
                    "backgroundColor": "darkgrey",
                    "height": "100%",
                    "width": "5px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "position": "relative",  # Ensure relative positioning for stacking context
                    "zIndex": 1  # Ensure the icon is rendered on top
                },
                children=html.I(className="fa fa-arrows-alt-h", style={"color": "#444444", "fontSize": "14px", "zIndex": 2})
            )),
            Panel(
                style={
                'overflowY': 'auto'
            },
                children=[
                    html.Details([
                        html.Summary("Accidents by hour"),
                        chart_accidents_by_hour.layout
                    ]),
                    html.Details([
                        html.Summary("Accidents by weekday"),
                        chart_accidents_by_weekday.layout
                    ]),
                    html.Details([
                        html.Summary("Accidents Over Time"),
                        chart_time.layout
                    ], open=True),
                    html.Details([
                        html.Summary("Accidents by Weather"),
                        chart_weather.layout
                    ]),
                    html.Details([
                        html.Summary("Accidents Trend"),
                        chart_trend.layout
                    ])
                ]),
        ]),

    # Hidden/input stores (kept at root)
    dcc.Input(id='filtered-state', type='hidden', value='init'),
    dcc.Input(id='geoselection-state', type='hidden', value='init'),
    dcc.Input(id='filter-ui-trigger', type='hidden', value='init'),
    # Format is [[lat, lng], zoom]
    dcc.Store(id='map_layout', data=[
        # [[mid_lat, mid_lon], zoom, [top_lat, left_lon, bottom_lat, right_lon]]
        [[START_COORDINATES['lat'], START_COORDINATES['lon']], START_ZOOM, [-1, -1, 1, 1]]
    ]),
    # Store for detail level in heatmap
    # level zero signature keeps track of the map layout (product of zoom and pan)
    dcc.Store(id='detail-level', data={"level": 0, "level_zero_signature": -1}), # 0: low, 1: medium, 2: high
])

# Register callbacks from other modules
chart_time.register_callbacks(app)
chart_weather.register_callbacks(app)
chart_trend.register_callbacks(app)
chart_accidents_by_hour.register_callbacks(app)
chart_accidents_by_weekday.register_callbacks(app)


if __name__ == '__main__':
    app.run(debug=True)