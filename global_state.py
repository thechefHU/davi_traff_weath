"""
This module manages the state of the data used in the application,
as well as binned versions of the data.
This is to avoid reloading the data multiple times in different modules.
"""
import pandas as pd
from pathlib import Path
import json
import geopandas as gpd

_current_data = pd.DataFrame() # Dataframe holding the current data (one row for each accident)
_current_binned_data = pd.DataFrame() # Dataframe holding the binned data (one row for each hex bin/county/state)
_unfiltered_data = pd.DataFrame() # Dataframe holding the unfiltered data (one row for each accident)   

_sindex = None # Spatial index for the current data

_h3_geojson = None
_counties_geojson = None

_h3_df = None
_counties_df = None
_comparison_groups = [pd.DataFrame(), pd.DataFrame()]

_selection_params = {
    "rectangle": {
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
    }
}
_selected_data = pd.DataFrame()

def load_data(data_folder="/data/", subset_accidents = None, logger=None):
    """
    Loads the initial data from the data folder
    """
    data_folder = Path(data_folder)
    global _current_data
    # Load the accident data
    _current_data = pd.read_parquet(data_folder / "traffic.parquet", engine="fastparquet")
    if subset_accidents is not None:
        _current_data = _current_data.sample(subset_accidents)    # ADD MORE AS NEEDED
    relevant_columns = [
        'ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
        'End_Lat', 'End_Lng', 'Distance(mi)', #'Description',
        'County', 'State', 'Zipcode', 'Timezone', 'Weather_Condition', "h3cell",
        "geoid", 'hour', 'weekday', 'month', 'year', 'season', "Weather_Group",
        "h3cell_fine", "Junction", "Stop", "Traffic_Signal", "Sunrise_Sunset", "Crossing",
    ]
    _current_data = _current_data[relevant_columns]
    # Set Start_Lat and Start_Lng to float32 to save memory
    _current_data['Start_Lat'] = _current_data['Start_Lat'].astype('float32')
    _current_data['Start_Lng'] = _current_data['Start_Lng'].astype('float32')
    _current_data['End_Lat'] = _current_data['End_Lat'].astype('float32')
    _current_data['End_Lng'] = _current_data['End_Lng'].astype('float32')
    if logger is not None:
        logger.info("Converting to GeoDataFrame and building spatial index...")
    _current_data = gpd.GeoDataFrame(
        _current_data,
        geometry=gpd.points_from_xy(_current_data.Start_Lng, _current_data.Start_Lat),
    )
    update_spatial_index()
    global _unfiltered_data
    _unfiltered_data = _current_data.copy() # for later use
    global _h3_df, _counties_df
    global _h3_geojson, _counties_geojson

    # Load information about the H3 cells
    _h3_df = pd.read_parquet(data_folder / "h3_cells.parquet", engine="fastparquet")
    with open(data_folder / "h3_cells.geojson", "r") as f:
        _h3_geojson = json.load(f)
    # Load information about the counties
    _counties_df = gpd.read_file(data_folder / "counties_processed.geojson")
    with open(data_folder / "counties.geojson", "r") as f:
        _counties_geojson = json.load(f)

    # TODO this needs to match the intitial plot type
    _current_binned_data = bin_data_by_county()

    if logger is not None:
        logger.info("Data loaded successfully.")



def get_data():
    """
    Returns the current data
    """
    global _current_data
    return _current_data

def set_data(data):
    """
    Sets the current data
    """
    global _current_data
    _current_data = data


def filter_data(filter_dict : dict, logger=None):
    global _unfiltered_data, _current_data
    if len(filter_dict) == 0:
        if logger:
            logger.info("No filters applied, not filtering")
            _current_data = _unfiltered_data
            return
    
    # Construct the filter string by joining the items
    filter_string = " & ".join(f"({value})" for _, value in filter_dict.items())

    if logger:
        logger.info("Filtering data with condition: %s", filter_string)
    _current_data = _unfiltered_data.query(filter_string)


def get_spatial_index():
    """
    Returns the spatial index of the current data
    """
    global _sindex
    return _sindex


def update_spatial_index():
    """
    Updates the spatial index of the current data
    """
    global _sindex
    global _current_data
    _sindex = _current_data.sindex


def get_binned_data():
    """
    Returns the current binned data
    """
    global _current_binned_data
    return _current_binned_data

def set_binned_data(data):
    """
    Sets the current binned data
    """
    global _current_binned_data
    _current_binned_data = data


# TODO:  hexagons without accidents are thrown arway in clean data script
def bin_data_by_h3():
    # Aggregate data based on H3 cells
    filtered_grouped = get_data().groupby('h3cell', observed=False).size().reset_index(name='n_accidents')
    # set n_accidents in h3_df based on filtered data
    filtered_cells = _h3_df[["h3cell", "h3_lat", "h3_lng"]].merge(
        filtered_grouped,
        left_on ='h3cell',
        right_on ='h3cell',
        how='left')
    filtered_cells['n_accidents'] = filtered_cells['n_accidents'].fillna(0)
    set_binned_data(filtered_cells)


def bin_data_by_county():
    # Aggregate data based on counties
    filtered_grouped = get_data().groupby('geoid').size().reset_index(name='n_accidents')
    # set n_accidents in counties_df based on filtered data
    filtered_bins = _counties_df[["NAME", "GEOID"]].merge(
        filtered_grouped, left_on='GEOID', right_on='geoid', how='left')
    filtered_bins['n_accidents'] = filtered_bins['n_accidents'].fillna(0)
    set_binned_data(filtered_bins)


def set_comparison_group(group_no):
    """
    Sets the comparison group for the data
    """
    global _comparison_groups
    assert group_no in [1, 2, 3], "Group number must be 1, 2, or 3"
    _comparison_groups[group_no - 1] = get_data_geoselected().copy()


def active_comparison_groups():    
    """
    Returns a list of active comparison groups (non-empty)
    """
    global _comparison_groups
    active_groups = []
    for i, group in enumerate(_comparison_groups):
        if not group.empty:
            active_groups.append(i + 1)
    return active_groups


def get_active_comparison_data():
    """
    Return the _current_data concatenated with the active comparison groups.
    Give each group a 'group' column indicating its group number.
    The points from selected data get group 'Selected data'.
    The 'group' column is categorical with the order: "Selected data", "Group 1", "Group 2", "Group 3".
    """
    global _comparison_groups
    frames = []
    selected_data_copy = get_data_geoselected().copy()
    selected_data_copy['group'] = "Selected data"
    frames.append(selected_data_copy)
    for i, group in enumerate(_comparison_groups):
        if not group.empty:
            group = group.copy()  # Avoid modifying the original group
            group['group'] = f"Group {i + 1}"
            frames.append(group)
    combined = pd.concat(frames, ignore_index=True)
    active_categories = ["Selected data"] + [f"Group {i + 1}" for i, group in enumerate(_comparison_groups) if not group.empty]
    combined['group'] = pd.Categorical(
        combined['group'],
        categories=active_categories,
        ordered=True
    )
    return combined


def get_data_geoselected():
    """
    Returns the current data filtered by the selection bounds
    """
    global _selected_data
    global _current_data
    if len(_selected_data) == 0:
        return _current_data
    else:
        return _selected_data


def set_selection_bounds(lat_min, lat_max, lon_min, lon_max):
    """
    Sets the current data to the points within the selection bounds
    """
    global _selection_params
    _selection_params = {
        "rectangle": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    }
    _selection_params["rectangle"]["lat_min"] = lat_min
    _selection_params["rectangle"]["lat_max"] = lat_max
    _selection_params["rectangle"]["lon_min"] = lon_min
    _selection_params["rectangle"]["lon_max"] = lon_max

    global _selected_data
    _selected_data =  _current_data[
        (_current_data['Start_Lat'] >= lat_min) &
        (_current_data['Start_Lat'] <= lat_max) &
        (_current_data['Start_Lng'] >= lon_min) &
        (_current_data['Start_Lng'] <= lon_max)
    ]

def set_selected_counties(geoids):
    """
    Sets the current data to the points within the selected counties
    """
    global _selected_data
    global _current_data

    global _selection_params
    _selection_params = {
        "counties": geoids
    }
    _selected_data =  _current_data[
        _current_data['geoid'].isin(geoids)
    ]

def clear_comparison_groups():
    """
    Clears all comparison groups
    """
    global _comparison_groups
    _comparison_groups = [pd.DataFrame(), pd.DataFrame()]

def get_h3_geojson():
    return _h3_geojson

def get_counties_geojson():
    return _counties_geojson


