import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import h3
import json

# If zoom level is above this threshold, show scatter plot instead of hexagons
SCATTER_PLOT_ZOOM_THRESHOLD = 7.0
H3_RESOLUTION = 4

# Initialize Dash app
app = dash.Dash(__name__)

# Load data
data_folder = Path("data")
print("Loading data...")
traffic = pd.read_parquet(data_folder / "traffic.parquet", engine="fastparquet")
traffic_reduced = traffic.head(200000)  # Use a subset for performance

h3_df = pd.read_parquet(data_folder / "h3_cells.parquet", engine="fastparquet")
with open(data_folder / "h3_cells.geojson", "r") as f:
    geojson_obj = json.load(f)



filtered = traffic_reduced
# Get unique weather conditions for dropdown
weather_options = [{'label': w, 'value': w} for w in sorted(traffic_reduced['Weather_Condition'].dropna().unique())]

# App layout
app.layout = html.Div([
    html.H1("Traffic Incidents by Weather Condition"),
    dcc.Dropdown(
        id='weather-dropdown',
        options=weather_options,
        value='Clear',
        clearable=False
    ),
    dcc.Store(id='filtered-data'),
    dcc.Store(id='filtered-cells'),
    dcc.Graph(id='scatter-map')
])

# Callback to update map based on weather condition
@app.callback(
    [
    Output('filtered-cells', 'data'),
    Output('filtered-data', 'data'),
    ],
    [Input('weather-dropdown', 'value')]
)
def filter_data(selected_weather):
    print("Filtering!...")
    filtered = traffic_reduced[traffic_reduced['Weather_Condition'] == selected_weather]
    print("Aggregating based on H3 cells...")
    # Aggregate data based on H3 cells
    filtered_grouped = filtered.groupby('h3cell').size().reset_index(name='n_accidents')
    # set n_accidents in h3_df based on filtered data
    filtered_cells = h3_df[["h3cell", "h3_lat", "h3_lng"]].merge(filtered_grouped, on='h3cell', how='left')
    # Save both filtered_cells and filtered in dcc.Store
    print("Filtering done")
    # TODO: Only return filtered if using scatter plot, and only return filtered_cells if using hexbin
    return filtered_cells.to_dict('records'), filtered.to_dict('records')

@app.callback(
    Output('scatter-map', 'figure'),
    [Input('filtered-cells', 'data'),
     Input('filtered-data', 'data'),
     Input('scatter-map', 'relayoutData')]
)
def update_map(filtered_cells, filtered_data, relayout_data):
    filtered_cells = pd.DataFrame(filtered_cells)
    filtered_data = pd.DataFrame(filtered_data)
    # Default map settings
    current_zoom = 3 # default zoom level
    layout = dict(
        style="carto-positron",
        zoom=current_zoom,
        center={"lat": 37.0902, "lon": -95.7129}
    )
    if relayout_data:
        print(relayout_data)
        if 'map.center' in relayout_data and 'map.zoom' in relayout_data:
            layout['center'] = relayout_data['map.center']
            layout['zoom'] = relayout_data['map.zoom']
            current_zoom = relayout_data['map.zoom']
        else:
            # Check for individual keys
            if 'map.center.lat' in relayout_data and 'map.center.lon' in relayout_data:
                layout['center'] = {
                    "lat": relayout_data['map.center.lat'],
                    "lon": relayout_data['map.center.lon']
                }

        # If we should filter based on what is visible on screen
        if 'map._derived' in relayout_data and current_zoom > SCATTER_PLOT_ZOOM_THRESHOLD:
            coordinates = relayout_data['map._derived']['coordinates']

            # readd the first coordinate to close the polygon
            coordinates.append(coordinates[0])
            # Flip all the lat-lng pairs to lng-lat for h3
            coordinates = [(lng, lat) for lat, lng in coordinates]
            # TODO: If polygon is too small, use polygon of fixed size around center¨
            print("Filtering based on visible area...")
            # create h3 polygon
            poly = h3.LatLngPoly(coordinates)
            # get all h3 cells in the polygon
            h3_cells_in_polygon = h3.h3shape_to_cells(poly, H3_RESOLUTION)
            filtered_data = filtered_data[filtered_data['h3cell'].isin(h3_cells_in_polygon)]



    #zmax_value = 2000 / max(1, layout['zoom']**2)  # Adjust zmax based on zoom level

    # fig = go.Figure(data=go.Densitymap(
    #     lat=filtered['h3_lat'],
    #     lon=filtered['h3_lng'],
    #     hovertext=filtered['h3cell'],
    #     z=filtered['n_accidents'],
    #     radius=25,
    #     zauto=False,
    #     zmin=0,
    #     zmax=zmax_value,
    #     #mode='markers',
    #     #marker=dict(size=filtered['Severity'] * 3)
    # ))

    # Example: Specify locations using the h3cell IDs


    # TODO: only create fig onnce, use update_layout to update zoom and center
    print(current_zoom)

    if current_zoom > SCATTER_PLOT_ZOOM_THRESHOLD:
        print("Using scatter plot")
        fig = px.scatter_map(
        filtered_data,
        lat='Start_Lat',
        lon='Start_Lng',
        color='Severity',
        size='Severity',
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=5,
        zoom=current_zoom,
        center={"lat": layout['center']['lat'], "lon": layout['center']['lon']},
        map_style="carto-positron",
        hover_data={'Severity': True, 'h3cell': True}
    )
    else:
        print("Using hexbin map")
        fig = px.choropleth_map(
            filtered_cells,
            geojson=geojson_obj,
            locations='h3cell',
            featureidkey='properties.h3cell',
            color='n_accidents',
            color_continuous_scale="Viridis",
            range_color=[0, filtered_cells['n_accidents'].quantile(0.9)],
            map_style="carto-positron",
            zoom=current_zoom,
            center={"lat": 37.0902, "lon": -95.7129},
            opacity=0.5,
            hover_data={'h3cell': True, 'n_accidents': True}
        )


    fig.update_layout(
        map=layout,
        width=1000,
        height=700,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)

