"""This converts data to categorical variables where possible and also save as
a parquet file. Before, the csv is 3gb, but the parquet is only 800mb.

To run you can run 

uv run clean_data.py

May have to sync first using uv sync (as I installed some new packages)

Link to direct download of population data
https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv
"""

import geopandas
from shapely.geometry import Polygon, LineString, Point
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import h3


H3_RESOLUTION = 4 # granularity of hexagons

data_folder = Path("data")
os.makedirs(data_folder, exist_ok=True)
print("Loading data from csv")
traffic = pd.read_csv(data_folder / "US_Accidents_March23.csv", engine="c")
geodata = geopandas.read_file(data_folder /"counties.geojson")
population = pd.read_csv("data/co-est2021-alldata.csv",encoding="latin1")

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
print("\nType conversion done")

print(f"New dtypes")
print(traffic.dtypes)

print("Putting points into H3 cells...")
traffic["h3cell"] = traffic.apply(lambda row: h3.latlng_to_cell(row["Start_Lat"], row["Start_Lng"], H3_RESOLUTION), axis=1)

print("Creating H3 DataFrame")
# find coordinates of h3 cells
h3_lats = []
h3_lngs = []
h3_cells = traffic["h3cell"].unique()
for cell in h3_cells:
    lat, lng = h3.cell_to_latlng(cell)
    h3_lats.append(lat)
    h3_lngs.append(lng)

h3_df = pd.DataFrame({
    "h3cell": h3_cells,
    "h3_lat": h3_lats,
    "h3_lng": h3_lngs
})

print("Saving h3 dataframe to parquet")
h3_df.to_parquet(data_folder / "h3_cells.parquet", engine="fastparquet")

print("Creating GeoJSON of H3 cells")
geojson_obj = {
    "type": "FeatureCollection",
    "features": []
}

for _, row in h3_df.iterrows():
    # The lat-lng pairs need to be in (lng, lat) order for GeoJSON
    geom = {
        "type": "Polygon",
        "coordinates": [[
            (lng, lat) for lat, lng in h3.cell_to_boundary(row["h3cell"])
        ]]
    }

    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "h3cell": row["h3cell"]
        }
    }
    geojson_obj["features"].append(feature)

print("Saving H3 GeoJSON to file")
with open(data_folder / "h3_cells.geojson", "w") as f:
    json.dump(geojson_obj, f)

print("H3 cells done.")

print("\n WE ARE NOT ALL DONE, fix geoID, population and density takes around 5 minutes")
#Adding a FIPS code to all traffic accidents, named geoid in the code
polygons = geodata["geometry"]

geoid = geodata["GEOID"]

smallTraffic = traffic[["State","County","Start_Lng","Start_Lat"]] #Chop away most things from the traffic dataset

pair = smallTraffic[["State","County"]].drop_duplicates()

states = pair["State"].unique()

traffic["geoid"] = pd.Series([0]*traffic.shape[0],dtype=str)

#Assign geoid to traffic data for all states and counties except for connecticut
for state in states:
    if state=="CT":
        continue
    stateTraffic = smallTraffic[smallTraffic["State"]==state]
    counties = pair[pair["State"]==state]["County"]

    for county in counties:
        countyTraffic = stateTraffic[stateTraffic["County"]==county]
        index = stateTraffic[stateTraffic["County"]==county].index
        geoidSC = 0
        for idx, row in countyTraffic.iterrows():
            lng = row["Start_Lng"]
            lat = row["Start_Lat"]
            tempGeoid = geoid[polygons.contains(Point(lng,lat))]
            if tempGeoid.empty == False:
                geoidSC = str(geoid[tempGeoid.index[0]])
                break
        traffic.loc[index,"geoid"] = geoidSC

#Assign each accident in connecticut one at the time
conneticutTraffic = smallTraffic[smallTraffic["State"]=="CT"]

for idx, row in conneticutTraffic.iterrows():
    lng = row["Start_Lng"]
    lat = row["Start_Lat"]
    tempGeoid = geoid[polygons.contains(Point(lng,lat))]
    if tempGeoid.empty == False:
        traffic.loc[idx,"geoid"] = geoid[tempGeoid.index[0]]
    elif row["County"]=="New Haven": #Somehow two data points could not be found in the geopandas.contain, but they are both in New haven, so it was done manually
        traffic.loc[idx,"geoid"] = "09009"

#Add data to counties.geojson file, make three new columns population, density and dataExist

geodataPopulation = pd.Series([1]*geodata.shape[0],dtype=int)
geodataDensity = pd.Series([1]*geodata.shape[0],dtype=float)
geodataDataExist = pd.Series([False]*geodata.shape[0],dtype=bool)

uniqueCounties = traffic["geoid"].unique()
count = 0
for geoid in uniqueCounties:
    state = int(geoid[0:2])
    county = int(geoid[2:])
    popIndex = population.loc[(population["STATE"]==state) & (population["COUNTY"]==county)].index[0]
    pop = population.loc[popIndex,"ESTIMATESBASE2020"]
    index = geodata[geodata["GEOID"]==geoid].index[0]
    geodataPopulation[index] = pop
    geodataDensity[index] = pop/geodata.loc[index,"ALAND"]*int(1e6) # *int(1e6) turns ALAND into km2 from m2
    geodataDataExist[index] = True

#Merge new columns into geodata
geodata = pd.concat([geodata,geodataPopulation,geodataDensity,geodataDataExist],axis=1)

geodata = geodata.rename(columns={0:"Population",1:"Density",2:"DataExist"})

geodata.to_file("data/counties_processed.geojson", driver='GeoJSON')


##################################################### Added statesArea.csv to main pull it into data !!!!!!!!!!!!!!!!!!!!!!!!
import pandas as pd
area = pd.Series([1]*50,dtype=int)
area = pd.read_csv("data/statesArea.csv", sep=",",header=None)
area = area.rename(columns={0:"name",1:"area"})
states = pd.read_json("data/us-states.json")
features = states["features"]


for idx, row in features.items():
    properties = row["properties"]
    name = properties["name"]
    stateArea = float(area.loc[area["name"]==name,"area"].iloc[0])
    pop = float(population.loc[(population["STNAME"]==name) & (population["COUNTY"]==0),"ESTIMATESBASE2020"].iloc[0]) #population
    density = pop/stateArea
    properties.update({"Area":stateArea,"Population":pop,"Density":density})

states.to_json("data/us-states_processed.json")


print("Saving as data/traffic.parquet")
##################################################### Added weather groups as a new column
conditions = [
    traffic['Weather_Condition'].str.contains('T-Storm|Thunder|Storm', case=False, na=False),
    traffic['Weather_Condition'].str.contains('Hail|Sleet|Ice', case=False, na=False),
    traffic['Weather_Condition'].str.contains('Snow|Wintry',case=False, na=False),
    traffic['Weather_Condition'].str.contains('Rain|Drizzle|Shower', case=False, na=False),
    traffic['Weather_Condition'].str.contains('Fog|Mist', case=False, na=False),
    traffic['Weather_Condition'].str.contains('Wind|Squall|Tornado',case=False, na=False),
    traffic['Weather_Condition'].str.contains('Dust|Smoke|Sand',case=False, na=False),
    traffic['Weather_Condition'].str.contains('Cloud|Haze|Overcast', case=False, na=False),
    traffic['Weather_Condition'].str.contains('Clear|Sunny|Fair', case=False, na=False)
]

choices = [
    'Thunderstorm',
    'Hail',
    'Snow',
    'Rain',
    'Fog',
    'Wind',
    'Dust/Smoke',
    'Cloudy',
    'Clear'
    
]


traffic['Weather_Group'] = np.select(conditions, choices, default='Other')
traffic.to_parquet(data_folder / "traffic.parquet", engine="fastparquet")
print("Done!")
