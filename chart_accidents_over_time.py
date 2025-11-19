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

    # CALLBACK 1: updates graph when filters or radio buttons change
    @app.callback(
        Output("time-chart", "figure"),
        [Input("time-granularity", "value"),
         Input("filtered-state", "value")]
    )
    def update_graph(granularity, filtering_state):
        # filtering_state is just a dummy input to trigger the update when filters change
        # (this only happens in the main app)
        fig = update_chart(granularity)
        return fig




def update_chart(granularity):
    counts = gs.get_data().groupby(granularity).size().reset_index(name="count")
    # nice ordering
    if granularity == "weekday":
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        counts[granularity] = pd.Categorical(counts[granularity], categories=order, ordered=True)
        counts = counts.sort_values(granularity)
    elif granularity == "month":
        month_names = {
            1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
            7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
        }
        counts["month"] = counts["month"].map(month_names)

    # histogram for all cases
    fig = px.bar(
        counts,
        x=granularity,
        y="count",
        title=f"Accidents by {granularity.capitalize()}",
        text="count"
    )

    return fig


layout = html.Div([
        
        html.Div(id="time-chart-content", style={'display': 'block'}, children=[
            dcc.RadioItems(
                id="time-granularity",
                options=[
                    {"label": "Hourly", "value": "hour"},
                    {"label": "Weekly", "value": "weekday"},
                    {"label": "Monthly", "value": "month"},
                    {"label": "Seasonal", "value": "season"},
                ],
                value="hour",
                inline=True,
                # Add gap-4 for spacing between radio buttons
                className="d-flex justify-content-center mb-3 gap-4"
            ),
            # This Graph will grow to fill the remaining space.
            # Initialize with an empty figure to prevent cut-off on first load.
            dcc.Graph(
                id="time-chart",
            ),
        ])
    ])





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