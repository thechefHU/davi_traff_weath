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
        Output("weather-chart", "figure"),
        [Input("filtered-state", "value"),
         Input("geoselection-state", "value"),
         Input("weather-normalize-toggle", "value")]
    )
    def update_graph(filtering_state, geoselection_state, normalize):
        # filtering_state is just a dummy input to trigger the update when filters change
        # (this only happens in the main app)
        fig = update_chart(normalize)
        return fig




def update_chart(normalize):
    order = ["Clear", "Cloudy", "Rain", "Fog", "Wind", "Dust/Smoke", "Snow", "Thunderstorm", "Hail", "Other"]
    if len(gs.active_comparison_groups()) == 0:
        counts = gs.get_data_geoselected().groupby("Weather_Group").size().reset_index(name="count")
        counts["Weather_Group"] = pd.Categorical(counts["Weather_Group"], categories=order, ordered=True)
        counts = counts.sort_values("Weather_Group")
        if normalize:
            total = counts["count"].sum()
            counts["count"] = counts["count"] / total
            ylabel = "Proportion of Accidents"
        else:
            ylabel = "Number of Accidents"
        # histogram for all cases
        fig = px.bar(
            counts,
            x="Weather_Group",
            y="count",
        )
    else:
        counts = gs.get_active_comparison_data().groupby(["Weather_Group", "group"]).size().reset_index(name="count")
        # nice ordering
        counts["Weather_Group"] = pd.Categorical(counts["Weather_Group"], categories=order, ordered=True)
        counts = counts.sort_values("Weather_Group")
        if normalize:
            # normalize within each group
            counts["count"] = counts.groupby("group")["count"].transform(lambda x: x / x.sum())
            ylabel = "Proportion of Accidents"
        else:
            ylabel = "Number of Accidents"

        fig = px.bar(
            counts,
            x="Weather_Group",
            y="count",
            color='group',
            barmode ='group',
            color_discrete_sequence=px.colors.qualitative.Safe
            )

        # Check if "Selected Data" exists and has the same length as another group
        if "Selected data" in counts["group"].unique():
            selected_data_entries = counts[counts["group"] == "Selected data"]
            selected_data_count = selected_data_entries["count"].sum()
            other_groups = counts[counts["group"] != "Selected data"]["group"].unique()
            for group in other_groups:
                group_entries = counts[counts["group"] == group]
                group_count = group_entries["count"].sum()
                if selected_data_count == group_count:
                    # Also check that the var are the same (rough check)
                    if selected_data_entries["count"].var() == group_entries["count"].var():
                        fig.data = [trace for trace in fig.data if trace.name != "Selected data"]
                        break
        


    fig.update_layout(
        xaxis_title="Weather",
        yaxis_title=ylabel,
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
    )
    if normalize:
        fig.update_yaxes(tickformat=".0%")  # Format y-axis ticks as percentages
        hovertemplate = "Weather: %{x}<br>Proportion of accidents: %{y:.1%}"
    else:
        hovertemplate = "Weather: %{x}<br>No. of accidents: %{y}"
    fig.update_traces(hovertemplate=hovertemplate)

    return fig


layout = html.Div([
            dbc.Checkbox(
                id='weather-normalize-toggle',
                label='Normalize counts',
                value=False,
            ),
            dcc.Graph(
                id="weather-chart",
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