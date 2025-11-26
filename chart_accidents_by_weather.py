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


def update_chart():
    if len(gs.active_comparison_groups()) == 0:
        # no comparison groups active, just show all data as one group
        counts = gs.get_data().groupby("Weather_Group", observed=False).size().reset_index(name="count")
        counts["group"] = "All Data"
        counts["normalized_count"] = counts["count"] / counts["count"].sum()
        counts["normalized_percentage_text"] = (100*counts["normalized_count"]).map("{:.1f}%".format)
        fig = px.bar(
            counts,
            x="Weather_Group",
            y="normalized_count",  
            title=f"Accidents by Weather",
            text="normalized_percentage_text",
        )
    else:
        counts = gs.get_active_comparison_data().groupby(["Weather_Group", "group"], observed=False).size().reset_index(name="count")
        counts["normalized_count"] = counts.groupby("group")["count"].transform(lambda x: x / x.sum())
        counts["normalized_percentage_text"] = (100*counts["normalized_count"]).map("{:.1f}%".format)
        fig = px.bar(
                counts,
                x="Weather_Group",
                y="normalized_count",  
                title=f"Accidents by Weather",
                text="normalized_percentage_text",
                barmode='group',
                color='group',
                color_discrete_sequence=px.colors.qualitative.Safe
            )

    # histogram for all cases


    return fig


layout = html.Div([
        # Initialize with an empty figure to prevent cut-off on first load.
        dcc.Graph(
            id="weather-chart",
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