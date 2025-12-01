import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import global_state as gs

# --- Helper Function for Dynamic Grouping ---
def get_natural_frequency(df, time_col="Start_Time", target_points=30):
    """
    Determines the best Pandas frequency alias (D, W, M, Q, Y) 
    based on the time range of the data to keep the chart readable.
    """
    if df.empty:
        return "D", "Daily"
        
    min_date = df[time_col].min()
    max_date = df[time_col].max()
    duration = max_date - min_date
    total_days = duration.days
    
    # If data spans less than 2 days, show hourly
    if total_days <= 2:
        return "h", "Hourly"

    # Calculate roughly how many days each "bar/point" represents
    days_per_point = total_days / target_points

    # Snap to the nearest natural human time unit
    if days_per_point <= 1.5:
        return "D", "Daily"      # If points need to be ~1 day apart
    elif days_per_point <= 10:
        return "W-MON", "Weekly" # If points need to be ~1 week apart (Start Monday)
    elif days_per_point <= 60:
        return "ME", "Monthly"   # If points need to be ~1-2 months apart
    elif days_per_point <= 180:
        return "QE", "Quarterly" # If points need to be ~3-6 months apart
    else:
        return "YE", "Yearly"    # Long term trends


# --- Callbacks ---
def register_callbacks(app):
    
    @app.callback(
        Output("trend-chart", "figure"),
        [Input("filtered-state", "value"),
        Input("geoselection-state", "value")]
    )
    def update_graph(filtering_state, geoselection_state):
        fig = update_chart()
        return fig


# --- Main Chart Logic ---
def update_chart():
    df = gs.get_data_selected_by_bounds()
    

    # 5. Resample/Group
    if len(gs.active_comparison_groups()) == 0:
        # No comparison groups, just use main data
        time_rule, freq_label = get_natural_frequency(df, "Start_Time", target_points=30)
        counts = df.resample(time_rule, on="Start_Time").size().reset_index(name="count")

        # 6. Plot
        fig = px.line(
            counts,
            x="Start_Time",
            y="count",  
            markers=True
        )
    else:
        combined = gs.get_active_comparison_data()
        time_rule, freq_label = get_natural_frequency(combined, "Start_Time", target_points=30)
        grouper = pd.Grouper(key="Start_Time", freq=time_rule)
        counts = combined.groupby([grouper, "group"]).size().reset_index(name="count")

        # 6. Plot with comparison groups
        fig = px.line(
                counts,
                x="Start_Time",
                y="count",  
                markers=True,
                color='group',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
    
    # 7. Layout Polish
    fig.update_layout(
        xaxis_title="Time", 
        yaxis_title="Accident Count",
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
        hovermode="x unified" # Shows a clean tooltip line across the graph
    )

    return fig


layout = html.Div([
        dcc.Graph(
            id="trend-chart",
        )
    ]
)


# --- Test Block ---
if __name__ == "__main__":
    # Ensure this matches your global_state setup
    gs.load_data(data_folder="data/", subset_accidents=100000) 
    
    app = dash.Dash(__name__, suppress_callback_exceptions=True) 
    app.layout = html.Div([
        layout,
        dcc.Input(id="filtered-state", type="hidden", value="init")
    ])
    register_callbacks(app)
    app.run(debug=True)