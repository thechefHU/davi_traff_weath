import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import h3
import json

h3_resolution = 4 # granularity of hexagons

# Initialize Dash app
app = dash.Dash(__name__)

# Load data
data_folder = Path("data")
print("Loading data...")
traffic = pd.read_parquet(data_folder / "traffic.parquet", engine="fastparquet")
print("Done loading data, putting points into H3 cells...")
traffic_reduced = traffic
traffic_reduced["h3cell"] = traffic_reduced.apply(lambda row: h3.latlng_to_cell(row["Start_Lat"], row["Start_Lng"], h3_resolution), axis=1)
print("H3 cells done.")

# find coordinates of h3 cells
h3_lats = []
h3_lngs = []
h3_cells = traffic_reduced["h3cell"].unique()
for cell in h3_cells:
    lat, lng = h3.cell_to_latlng(cell)
    h3_lats.append(lat)
    h3_lngs.append(lng)

h3_df = pd.DataFrame({
    "h3cell": h3_cells,
    "h3_lat": h3_lats,
    "h3_lng": h3_lngs
})

geojson_obj = {
    "type": "FeatureCollection",
    "features": []
}

for _, row in h3_df.iterrows():
    # The lat-lng pairs need to be in (lng, lat) order for GeoJSON
    geom = {
        "type": "Polygon",
        "coordinates": [[
            (lng, lat) for lat, lng in h3.cell_to_boundary(row["h3cell"])
        ]]
    }

    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "h3cell": row["h3cell"]
        }
    }
    geojson_obj["features"].append(feature)


# Dictionary of h3 cells to indices in traffic_reduced
h3_cells_groups = traffic_reduced.groupby("h3cell").groups



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
    dcc.Graph(id='scatter-map')
])

# Callback to update map based on weather condition
@app.callback(
    Output('filtered-data', 'data'),
    [Input('weather-dropdown', 'value')]
)
def filter_data(selected_weather):
    print("Filtering!")
    filtered = traffic_reduced[traffic_reduced['Weather_Condition'] == selected_weather]
    print("Aggregating based on H3 cells...")
    # Aggregate data based on H3 cells
    filtered = filtered.groupby('h3cell').size().reset_index(name='n_accidents')
    filtered_cells = h3_df[["h3cell", "h3_lat", "h3_lng"]].copy()
    # set n_accidents in h3_df based on filtered data
    filtered_cells = filtered_cells.merge(filtered, on='h3cell', how='left')
    # Convert filtered_cells to GeoJSON format
    # features = []
    # for _, row in filtered_cells.iterrows():
    #     feature = {
    #         "type": "Feature",
    #         "geometry": mapping(row["geometry"]),
    #         "properties": {
    #             "h3cell": row["h3cell"],
    #             "h3_lat": row["h3_lat"],
    #             "h3_lng": row["h3_lng"],
    #             "n_accidents": row.get("n_accidents", 0)
    #         }
    #     }
    #     features.append(feature)
    # geojson_data = {
    #     "type": "FeatureCollection",
    #     "features": features
    # }
    return filtered_cells.to_dict('records')

@app.callback(
    Output('scatter-map', 'figure'),
    [Input('filtered-data', 'data'),
     Input('scatter-map', 'relayoutData')]
)
def update_map(filtered_cells, relayout_data):
    filtered = pd.DataFrame(filtered_cells)

    # Default map settings
    layout = dict(
        style="carto-positron",
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129}
    )
    if relayout_data:
        if 'map.center' in relayout_data and 'map.zoom' in relayout_data:
            layout['center'] = relayout_data['map.center']
            layout['zoom'] = relayout_data['map.zoom']
        else:
            # Check for individual keys
            if 'map.center.lat' in relayout_data and 'map.center.lon' in relayout_data:
                layout['center'] = {
                    "lat": relayout_data['map.center.lat'],
                    "lon": relayout_data['map.center.lon']
                }
            if 'mapbox.zoom' in relayout_data:
                layout['zoom'] = relayout_data['mapbox.zoom']
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
    fig = px.choropleth_map(
        filtered,
        geojson=geojson_obj,
        locations='h3cell',
        featureidkey='properties.h3cell',
        color='n_accidents',
        color_continuous_scale="Viridis",
        range_color=[0, filtered['n_accidents'].quantile(0.9)],
        map_style="carto-positron",
        zoom=3,
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
    return fig

if __name__ == '__main__':
    app.run(debug=True)

