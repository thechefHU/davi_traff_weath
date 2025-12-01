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
        Output("hour-chart", "figure"),
        [Input("filtered-state", "value"),
         Input("geoselection-state", "value"),
         Input("hour-normalize-toggle", "value")]
    )
    def update_graph(filtering_state, geoselection_state, normalize):
        # filtering_state is just a dummy input to trigger the update when filters change
        # (this only happens in the main app)
        fig = update_chart(normalize)
        return fig




def update_chart(normalize):
    if len(gs.active_comparison_groups()) == 0:
        counts = gs.get_data_geoselected().groupby("hour").size().reset_index(name="count")
        if normalize:
            total = counts["count"].sum()
            counts["count"] = counts["count"] / total
            ylabel = "Proportion of Accidents"
        else:
            ylabel = "Number of Accidents"
        # histogram for all cases
        fig = px.line(
            counts,
            x="hour",
            markers=True,
            y="count",
        )
    else:
        counts = gs.get_active_comparison_data().groupby(["hour", "group"]).size().reset_index(name="count")
        # nice ordering
        if normalize:
            # normalize within each group
            counts["count"] = counts.groupby("group")["count"].transform(lambda x: x / x.sum())
            ylabel = "Proportion of Accidents"
        else:
            ylabel = "Number of Accidents"
        fig = px.line(
            counts,
            x="hour",
            markers=True,
            y="count",
            color='group',
            color_discrete_sequence=px.colors.qualitative.Safe
            )

        # Check if "Selected Data" exists and has the same length as another group
        if "Selected data" in counts["group"].unique():
            selected_data_count = counts[counts["group"] == "Selected data"]["count"].sum()
            other_groups = counts[counts["group"] != "Selected data"]["group"].unique()
            for group in other_groups:
                group_count = counts[counts["group"] == group]["count"].sum()
                if selected_data_count == group_count:
                    print("Hiding legend for 'Selected data' as it matches another group's count.")
                    fig.for_each_trace(
                    lambda trace: trace.update(showlegend=False)
                    if trace.name == "Selected data" else None
                    )
                    break
        


    fig.update_layout(
        xaxis_title="hour".capitalize(),
        yaxis_title=ylabel,
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
    )
    if normalize:
        fig.update_yaxes(tickformat=".0%")  # Format y-axis ticks as percentages

    return fig


layout = html.Div([
            dbc.Checkbox(
                id='hour-normalize-toggle',
                label='Normalize counts',
                value=False,
            ),
            dcc.Graph(
                id="hour-chart",
            )
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