import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# Initialize Dash app
app = dash.Dash(__name__)

# Load data
data_folder = Path("data")
print("Loading data...")
traffic = pd.read_parquet(data_folder / "traffic.parquet", engine="fastparquet")
print("Done loading data.")
traffic_reduced = traffic.head(100000)

# Get unique weather conditions for dropdown
weather_options = [{'label': w, 'value': w} for w in sorted(traffic_reduced['Weather_Condition'].dropna().unique())]

# App layout
app.layout = html.Div([
    html.H1("Traffic Incidents by Weather Condition"),
    dcc.Dropdown(
        id='weather-dropdown',
        options=weather_options,
        value=weather_options[0]['value'],
        clearable=False
    ),
    dcc.Graph(id='scatter-map')
])

# Callback to update map based on weather condition
@app.callback(
    Output('scatter-map', 'figure'),
    Input('weather-dropdown', 'value')
)
def update_map(selected_weather):
    filtered = traffic_reduced[traffic_reduced['Weather_Condition'] == selected_weather]
    fig = go.Figure(data=go.Scattermap(
        lat=filtered['Start_Lat'],
        lon=filtered['Start_Lng'],
        hovertext=filtered['Description'],
        mode='markers',
        marker=dict(size=filtered['Severity'] * 3)
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 37.0902, "lon": -95.7129},
        width=1000,
        height=700,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)

