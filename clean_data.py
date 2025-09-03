"""This converts data to categorical variables where possible and also save as
a parquet file. Before, the csv is 3gb, but the parquet is only 800mb.

To run you can run 

uv run clean_data.py

May have to sync first using uv sync (as I installed some new packages)
"""


import pandas as pd
import numpy as np
from pathlib import Path
import os

data_folder = Path("data")
os.makedirs(data_folder, exist_ok=True)
print("Loading data from csv")
traffic = pd.read_csv(data_folder / "US_Accidents_March23.csv", engine="c")

print("Starting type conversion")
traffic["ID"] = traffic["ID"].astype("string")
print("ID done")
traffic["Source"] = traffic["Source"].astype("category")
traffic["Severity"] = traffic["Severity"].astype("int64")
traffic["Start_Time"] = pd.to_datetime(traffic["Start_Time"], format='ISO8601')
traffic["End_Time"] = pd.to_datetime(traffic["End_Time"], format='ISO8601')
print("End time done")
traffic["Start_Lat"] = traffic["Start_Lat"].astype("float64")
traffic["Start_Lng"] = traffic["Start_Lng"].astype("float64")
traffic["End_Lat"] = traffic["End_Lat"].astype("float64")
traffic["End_Lng"] = traffic["End_Lng"].astype("float64")
traffic["Distance(mi)"] = traffic["Distance(mi)"].astype("float64")
traffic["Description"] = traffic["Description"].astype("string")
traffic["Street"] = traffic["Street"].astype("string")
traffic["City"] = traffic["City"].astype("category")
traffic["County"] = traffic["County"].astype("category")
traffic["State"] = traffic["State"].astype("category")
traffic["Zipcode"] = traffic["Zipcode"].astype("category")
print("Zipcode done")
traffic["Country"] = traffic["Country"].astype("category")
traffic["Timezone"] = traffic["Timezone"].astype("category")
traffic["Airport_Code"] = traffic["Airport_Code"].astype("category")
traffic["Weather_Timestamp"] = pd.to_datetime(traffic["Weather_Timestamp"], format='ISO8601')
traffic["Temperature(F)"] = traffic["Temperature(F)"].astype("float64")
traffic["Wind_Chill(F)"] = traffic["Wind_Chill(F)"].astype("float64")
traffic["Humidity(%)"] = traffic["Humidity(%)"].astype("float64")
traffic["Pressure(in)"] = traffic["Pressure(in)"].astype("float64")
traffic["Visibility(mi)"] = traffic["Visibility(mi)"].astype("float64")
traffic["Wind_Direction"] = traffic["Wind_Direction"].astype("category")
traffic["Wind_Speed(mph)"] = traffic["Wind_Speed(mph)"].astype("float64")
print("Wind speed done")
traffic["Precipitation(in)"] = traffic["Precipitation(in)"].astype("float64")
traffic["Weather_Condition"] = traffic["Weather_Condition"].astype("category")
traffic["Amenity"] = traffic["Amenity"].astype("bool")
traffic["Bump"] = traffic["Bump"].astype("bool")
traffic["Crossing"] = traffic["Crossing"].astype("bool")
traffic["Give_Way"] = traffic["Give_Way"].astype("bool")
traffic["Junction"] = traffic["Junction"].astype("bool")
traffic["No_Exit"] = traffic["No_Exit"].astype("bool")
traffic["Railway"] = traffic["Railway"].astype("bool")
print("Railway done")
traffic["Roundabout"] = traffic["Roundabout"].astype("bool")
traffic["Station"] = traffic["Station"].astype("bool")
traffic["Traffic_Calming"] = traffic["Traffic_Calming"].astype("bool")
traffic["Traffic_Signal"] = traffic["Traffic_Signal"].astype("bool")
traffic["Turning_Loop"] = traffic["Turning_Loop"].astype("bool")
traffic["Sunrise_Sunset"] = traffic["Sunrise_Sunset"].astype("category")
traffic["Civil_Twilight"] = traffic["Civil_Twilight"].astype("category")
traffic["Nautical_Twilight"] = traffic["Nautical_Twilight"].astype("category")
traffic["Astronomical_Twilight"] = traffic["Astronomical_Twilight"].astype("category")
print("\nAll done")

print(f"New dtypes")
print(traffic.dtypes)

print("Saving as data/traffic.parquet")
traffic.to_parquet(data_folder / "traffic.parquet", engine="fastparquet")
print("Done!")
