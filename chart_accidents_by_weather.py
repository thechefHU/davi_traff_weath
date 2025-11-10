""" This is the "Accidents Over Time" chart column in the dashboard layout.
Moved into its own file for clarity and modularity.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import global_state as gs
import dash_daq as daq

# Register callbacks in a function
def register_callbacks(app):
    
    # CALLBACK 1: updates graph when filters change
    @app.callback(
        Output("weather-chart", "figure"),
         Input("filtered-state", "value")
    )
    def update_graph(filtering_state):
        # filtering_state is just a dummy input to trigger the update when filters change
        # (this only happens in the main app)
        fig = update_chart()
        return fig

    # CALLBACK 2: updates graph visibility when the switch is toggled
    @app.callback(
        Output('weather-chart', 'style'),
        Input('hide_weather', 'value')
    )
    def toggle_weather_chart_visibility(switch_value):
        if switch_value == True:
            # If switch is ON, set display to 'block' (visible)
            return {'display': 'block'}
        else:
            # If switch is OFF, set display to 'none' (hidden)
            return {'display': 'none'}


def update_chart():
    counts = gs.get_data().groupby("Weather_Group",
                                   observed=False).size().reset_index(name="count")

    # histogram for all cases
    fig = px.bar(
        counts,
        x="Weather_Group",
        y="count",  
        title=f"Accidents by Weather Group",
        text="count"
    )

    return fig


layout = html.Div([
        html.H4("Accidents by Weather", className="text-center"),
        
        daq.ToggleSwitch(
            id="hide_weather",
            label="Hide Chart",
            value=True # Starts visible
        ),

        # Initialize with an empty figure to prevent cut-off on first load.
        dcc.Graph(
            id="weather-chart",
            figure={}, # Fix for initial load cut-off
            config={'responsive': True},
            className="flex-grow-1", # Bootstrap class for flex-grow: 1
            style={'display': 'block'} 
        )
    ]
)



# If this file is run, just do a simple test load of the data.
# (to test the plot quickly)
if __name__ == "__main__":
    gs.load_data(data_folder="data/", subset_accidents=100000)  # load a subset for faster testing
    app = dash.Dash(__name__, suppress_callback_exceptions=True) 
    app.layout = html.Div([
        layout,
        # Hidden input for filtering state - temporary placeholder, overwritten in main app
        dcc.Input(id="filtered-state", type="hidden", value="init")
    ])
    register_callbacks(app)
    app.run(debug=True)