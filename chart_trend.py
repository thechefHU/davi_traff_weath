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
         Input("filtered-state", "value")
    )
    def update_graph(filtering_state):
        fig = update_chart()
        return fig


# --- Main Chart Logic ---
def update_chart():
    df = gs.get_data()

    # Error handling
    if "Start_Time" not in df.columns:
        return px.line(title="Error: 'Start_Time' column missing.")

    # Convert TimeStamp to DateTime
    if not pd.api.types.is_datetime64_any_dtype(df["Start_Time"]):
        df = df.copy()
        df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors='coerce')

    # If parsing failed -> drop. Double-check dropped amount
    initial_count = len(df)
    df = df.dropna(subset=["Start_Time"])
    dropped_count = initial_count - len(df)

    if dropped_count > 0:
        print(f"⚠️ Warning: Dropped {dropped_count} rows due to invalid timestamps.")

    if df.empty:
        return px.line(title="No data available for this selection")

    # 4. Dynamic Grouping. Snapping logic: based on our selected time window it finds the suitable time-frame to group by (quarters,months,weeks,years) to have
    # a visually appealing nr of elements on the trend graph
    # This automatically picks D/W/M/Q/Y based on the data range
    time_rule, freq_label = get_natural_frequency(df, "Start_Time", target_points=30)

    # 5. Resample/Group
    counts = df.resample(time_rule, on="Start_Time").size().reset_index(name="count")

    # 6. Plot
    fig = px.line(
        counts,
        x="Start_Time",
        y="count",  
        title=f"Trend of Accidents",
        markers=True
    )
    
    # 7. Layout Polish
    fig.update_layout(
        xaxis_title="Time", 
        yaxis_title="Accident Count",
        margin=dict(l=40, r=40, t=60, b=40),
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