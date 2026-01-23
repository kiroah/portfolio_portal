# Databricks notebook source
# MAGIC %md
# MAGIC # Join investigation & otpw creation
# MAGIC This notebook has been created as a workspace to test and verify data source joins, and leave a log of work done. The final cleaned up version will be included in join_clean_imput file. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Always run these first

# COMMAND ----------

!pip install timezonefinder 

# COMMAND ----------


from pyspark.sql.functions import col, count, when, to_timestamp
from pyspark.sql import functions as F
from pyspark.sql.functions import coalesce, lit


import pandas as pd
from io import StringIO
from pathlib import Path
import numpy as np
import math
import os
from sklearn.neighbors import BallTree
from timezonefinder import TimezoneFinder
import pytz
import re
from datetime import datetime
import time


# COMMAND ----------

#Config
DATE_RANGE = '3M' # Pick from 3M, 1Y or 5Y. This will also be the output directory
OVERWRITE = True  # If picked true, then the script will override the existing data
YEARS=[2023,2024] #Only used for 5Y/10Y data generation. Deciding for which year shards to create data for
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic") #Set this if you want to partially overwrite the data

SPARK_DIR = 'dbfs:/student-groups/Group_01_01'
OUTPUT_NAME_1 = 'otpw_v2'

TOPK_NEAREST = 1 # Number of nearest weather stations to consider for weather information (if there wasn't an exact match)
WEATHER_ST_RADIUS = 20 #Only consider nearest weather stations within this radius (in miles) from the airport

WEATHER_TS_BEFORE = 2 #The tied weather data needs to be at least this hours ago
WEATHER_TS_AFTER = 4 #The tied weather data cannot be more than this hours ago



# COMMAND ----------

#Helper functions

def save_spark_df(df, dfname, save_path = SPARK_DIR,date_range = DATE_RANGE, overwrite = OVERWRITE, by_year = False):
    #Quick helper function to save the created spark dataframe with default directory for consistency
    if ('.parquet' not in dfname):
        dfname = dfname + ".parquet"
        print("Added .parquet at the end of name")
    dfname = dfname.lower()
    if date_range is None or date_range == '':
        date_range = ''
    else:
        date_range = date_range.replace('/','')
        date_range = f"/{date_range}/"
        dbutils.fs.mkdirs(save_path+date_range)

    full_path = re.sub(r'/+', '/', save_path + date_range + dfname)

    write_mode = "overwrite" if overwrite else "error"

    #If by_year flag is set, then we will partition the saving data by year
    if by_year == True:
        df.write.mode(write_mode).option("partitionOverwriteMode", "dynamic").partitionBy("YEAR").parquet(full_path)
    else:
        df.write.mode(write_mode).parquet(full_path)
    print(f"Saved {full_path}")

    

def load_spark_df(dfname, load_path = SPARK_DIR, date_range = DATE_RANGE, years = []): 
    #Helper function to load the saved spark dataframe with default directory for consistency   
    dfname = dfname.lower()
    if ('.parquet' not in dfname):
        dfname_parquet = dfname + ".parquet"
    date_range = date_range.replace('/','')
    if date_range is None or date_range == '':
        date_range = ''
    else:
        date_range = f"/{date_range}/"

    full_path = re.sub(r'/+', '/', load_path + date_range + dfname_parquet)
        
    df = spark.read.parquet(full_path)

    if date_range in ['/5Y/','/10Y/'] and years is not None and len(years) != 0:
        yrs = "','".join([str(y) for y in years])
        df = df.filter(f"year IN ('{yrs}')")

    print(f"Loaded {full_path}")
    df.createOrReplaceTempView(dfname) 
    print(f"Created temp view {dfname} for spark SQL reference")
    return df


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Import data
# MAGIC Import data sources and print specs (e.g. # of rows)
# MAGIC
# MAGIC
# MAGIC External links: 
# MAGIC - [261 dataset & cluster info](https://bcourses.berkeley.edu/courses/1546383/pages/mids-261-final-project-dataset-and-cluster?module_item_id=17463611)
# MAGIC - [Flights data](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ)
# MAGIC - [Weather Data](https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf)
# MAGIC - Stations data: Unsure
# MAGIC - Airport code: Need to convert? _"Here you will need to import an external airport code conversion set (source: https://datahub.io/core/airport-codesLinks to an external site.) and join the airport codes to the airline's flights table on the IATA code (3-letter code used by passengers)"_
# MAGIC

# COMMAND ----------

#Import flights, weather and otpw.
if DATE_RANGE == '3M':
    df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/")
    df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
    df_otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
elif DATE_RANGE == '1Y':
    df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/")
    df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_1y/")
    df_otpw = spark.read.option("header","true").csv("dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M/OTPW_12M_2015.csv.gz")
elif DATE_RANGE == '5Y':
    df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")
    df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")
    df_otpw = spark.read.parquet("dbfs:/mnt/mids-w261/OTPW_60M/OTPW_60M")
elif DATE_RANGE == '10Y':
    # the original data has data upto 2021
    df_flights_5y = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")
    df_weather_5y = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")

    # New 3-year extension (2022–2024) you created earlier
    df_flights_new = spark.read.parquet("dbfs:/student-groups/Group_01_01/new_data/flights.parquet")
    df_weather_new = spark.read.parquet("dbfs:/student-groups/Group_01_01/new_data/weather.parquet")

    df_weather_new = df_weather_new.drop('DYTS', 'DYHF')

    # Union by column name (schemas and YEAR partitions are compatible)
    df_flights = df_flights_5y.unionByName(df_flights_new)
    df_weather = df_weather_5y.unionByName(df_weather_new)

    df_otpw = None

#Import airport code <-> IATA+ICAO mapping file
try:    
    dbutils.fs.ls(f"{SPARK_DIR}/airports.parquet")
    df_airports = spark.read.parquet(f"{SPARK_DIR}/airports.parquet")
except:
    df_airports = spark.read.format("csv").option("header","true").load(f"dbfs:/student-groups/Group_01_01/airports.csv")
    df_airports.write.mode("overwrite").parquet(f"{SPARK_DIR}/airports.parquet")

#Import ICAO <-> weather station mapping file
try:    
    dbutils.fs.ls(f"{SPARK_DIR}/isd_history.parquet")
    df_stations = spark.read.parquet(f"{SPARK_DIR}/isd_history.parquet")
except:
    df_stations = spark.read.format("csv").option("header","true").load(f"dbfs:/student-groups/Group_01_01/isd_history.csv")
    df_stations.write.mode("overwrite").parquet(f"{SPARK_DIR}/isd_history.parquet")



#Older script that converted the original csv files into parquet for faster work
# df_stations = spark.read.format("csv").option("header","true").load(f"dbfs:student-groups/Group_01_01/isd_history.csv")
# df_airports = spark.read.format("csv").option("header","true").load(f"dbfs:student-groups/Group_01_01/airports.csv")
# df_stations.write.mode("overwrite").parquet(f"{SPARK_DIR}/isd_history.parquet")
# df_airports.write.mode("overwrite").parquet(f"{SPARK_DIR}/airports.parquet")

#Put into a dictionary for convenience
data_source_map = {
    "flights": df_flights,
    "weather": df_weather,
    "stations": df_stations,
    "otpw": df_otpw,
    "airports": df_airports}
for df_name, df in data_source_map.items():
    if df is not None: df.createOrReplaceTempView(df_name) #Create temp view so it can be queried w/ spark SQL


# COMMAND ----------

if DATE_RANGE == '5Y' or DATE_RANGE == '10Y':
    print("skipping deduping for 5/10 year data")
else:
    #dedup flights data
    df_flights_count = df_flights.count()
    print(f"df_flights rows before: {df_flights_count}")
    df_flights = df_flights.dropDuplicates()

    df_flights_count_dedup = df_flights.count()
    print(f"df_flights rows after: {df_flights_count_dedup}")


    df_otpw_count = df_otpw.count()
    print(f"otpw_df rows: {df_otpw_count}")

# COMMAND ----------






# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyzing data for join
# MAGIC There are two challenges to the join: 
# MAGIC * Flights and weather cannot directly join, requiring airport/IATA code and IKAO code (and possibly others) to have the right join
# MAGIC * The join is not an exact key join, instead more of a range join where the weather data two hours prior to the flight departure time will be joined

# COMMAND ----------

# MAGIC %md
# MAGIC ### Investigation

# COMMAND ----------

# Look for weird airport codes (if any)
spark.sql("""
          SELECT 
          DISTINCT ORIGIN
          FROM flights
          WHERE LEN(ORIGIN) != 3 or ORIGIN is NULL
          """).display()

# COMMAND ----------

# Get list of airport codes of origin from flights data
origins_df = spark.sql("""
          SELECT 
          DISTINCT ORIGIN
          FROM flights
          """
          ).toPandas()

# Get list of IATA and ICAO code from airport data
iata_icao_df = spark.sql("""
          SELECT 
          DISTINCT iata_code, icao_code,name as airport_name, keywords
          FROM airports
          """
          ).toPandas()

# Get list of stations and icao codes from station data
icao_stations_df = spark.sql("""
          SELECT 
          DISTINCT ICAO, USAF, WBAN,
            (CASE WHEN (ABS(LAT) < 1.0 AND (ABS(LAT)*1000 < 90)) THEN LAT * 1000 ELSE LAT END) AS LAT_ORIGIN,
            (CASE WHEN (ABS(LON) < 1.0 AND (ABS(LON)*1000 < 180)) THEN LON * 1000 ELSE LON END) AS LON_ORIGIN
          FROM stations
          WHERE LAT IS NOT NULL AND LON IS NOT NULL
          """
          ).toPandas()
icao_stations_df[['LAT_ORIGIN','LON_ORIGIN']] = icao_stations_df.loc[:,['LAT_ORIGIN','LON_ORIGIN']].astype(float)



# Get list of stations that exists in weather data
stations_weather_df = spark.sql("""
          SELECT 
          DISTINCT STATION, 
            substring(STATION, 0, 6) AS USAF,
            substring(STATION, -5, 5) AS WBAN, 
            LATITUDE AS LAT_STATION,
            LONGITUDE AS LON_STATION
          FROM weather
          WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL
          """
          ).toPandas()

stations_weather_df[['LAT_STATION','LON_STATION']] = stations_weather_df.loc[:,['LAT_STATION','LON_STATION']].astype(float)


# COMMAND ----------

# This table has duplicate ICAOs, which has different WBAN or USAF. Often times one of the is 999999 (or 99999)
# We are going to dedup them, where we only keep the row with lowest USAF then USAF value. 
icao_stations_df = icao_stations_df.sort_values(['ICAO','USAF','WBAN'])

print(f"Number of rows: {len(icao_stations_df)}")
print(f"# of unique ICAOs: {len(icao_stations_df['ICAO'].unique())}")

icao_stations_df = icao_stations_df.drop_duplicates(subset=['ICAO'], keep='first')
print(f"Number of rows after dedup: {len(icao_stations_df)}")
print(f"# of unique ICAOs after dedup: {len(icao_stations_df['ICAO'].unique())}")

# COMMAND ----------

# Some of the IATA and ICAO codes are't populated in airports data, and instead stored in misc column named "keywords" (e.g. KISN). 
# It seems this can happen if the airport is closed. 
# For those missing, we extract the ICAO code from keywords column and derive IATA code from there. After that we update the 

SPECIAL_CHARS = [' ','_',',']
def get_codes(row):
    # Helper function used to extract icao code from keywords column, which are somewhat miscellaneous and comma separated
    # We assume that all airport codes (that we care) starts with "K", and is 4 letters. If not found, return None  
    if row['icao_code'] is None:
        keywords = row['keywords']
        k_split = keywords.split(',') if keywords else []
        k_split = [x.strip() for x in k_split]
        for k in k_split:
            if len(k) == 4 and k.startswith('K') and not any(s_char in k for s_char in SPECIAL_CHARS):
                row['icao_code'] = k
                row['iata_code'] = k[1:4]
                break
    return row
iata_icao_df = iata_icao_df.apply(get_codes, axis=1)



# COMMAND ----------

# --- Join flights → airport codes ---
origins_iata_icao_df = origins_df.merge(
    iata_icao_df, left_on="ORIGIN", right_on="iata_code", how="left"
).sort_values('keywords', ascending=False).drop_duplicates(subset=['ORIGIN'], keep='first')
print("Airport codes existing in flight data, but not in airport codes data")
print(origins_iata_icao_df[origins_iata_icao_df["icao_code"].isnull()])


# --- Join airport codes → station codes ---
origins_icao_stations_df = origins_iata_icao_df.merge(
    icao_stations_df, left_on="icao_code", right_on="ICAO", how="left"
)
missing_station = origins_icao_stations_df[origins_icao_stations_df["ICAO"].isnull()]["ORIGIN"].unique()
print("\n\nAirport codes existing in flight data & airport data, but not in stations data")
print(missing_station)
print("length:", len(missing_station))


# --- Join station codes → weather (USAF+WBAN) ---
origins_weather_df = origins_icao_stations_df.merge(
    stations_weather_df, on=["USAF", "WBAN"], how="left"
).drop_duplicates()
missing_weather = origins_weather_df[origins_weather_df["STATION"].isnull()]["ORIGIN"].unique()
print("\n\nAirport codes existing in flight data & airport data & stations data, but not in weather data (using USAF-WBAN join)")
print(missing_weather)
print(len(missing_weather))



# COMMAND ----------

# For rest of the unmatched airports, top 1 closest stations (within 20 miles)
# We are keeping multiple because the closest station may get retired or relocated in the future, the match doesn't exist anymore. Therefore we keep 2 as a backup. 
# Note that the final code is mostly generated by GenAI, but still required hours of tweaks to make it work. 


def find_closest_points(
    target_lats: pd.Series, 
    target_lons: pd.Series, 
    top_k: int, 
    within: float, 
    ref_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Finds the top_k closest points for a *series* of target latitudes 
    and longitudes, filtered by a maximum distance.

    Uses a vectorized Haversine calculation for efficiency.

    Args:
        target_lats: A pandas Series of target latitudes.
        target_lons: A pandas Series of target longitudes (must be
                     the same length as target_lats).
        top_k: The number of closest points to return per target point.
        within: The maximum distance in miles to include.
        ref_df: A DataFrame with columns 'id', 'lat', and 'lon'.

    Returns:
        A pandas DataFrame with results for all target points, including:
        'target_lat', 'target_lon', 'match_id', 'match_lat', 'match_lon',
        and 'distance_miles'.
    """
    
    # 1. Prepare DataFrames for cross-join
    
    # Create target DataFrame
    target_df = pd.DataFrame({
        'target_lat': target_lats,
        'target_lon': target_lons
    })
    # Add a unique ID for grouping later
    target_df['target_id'] = target_df.reset_index().index

    # Prepare reference DataFrame (rename columns to avoid conflicts)
    new_column_names = ['match_id', 'match_lat', 'match_lon']
    ref_df_renamed = ref_df.copy()
    ref_df_renamed.columns = new_column_names

    # 2. Perform cross-join
    # This creates all possible pairs of (target_point, ref_point)
    target_df['_key'] = 1
    ref_df_renamed['_key'] = 1
    cross_df = pd.merge(target_df, ref_df_renamed, on='_key').drop('_key', axis=1)

    # 3. Vectorized Haversine Calculation
    R_MILES = 3956  # Radius of the Earth in miles

    # Convert all coordinates to radians
    lat1_rad = np.radians(cross_df['target_lat'])
    lon1_rad = np.radians(cross_df['target_lon'])
    lat2_rad = np.radians(cross_df['match_lat'])
    lon2_rad = np.radians(cross_df['match_lon'])

    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Apply Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    cross_df['distance_miles'] = R_MILES * c

    # 4. Filter and get Top-K
    
    # Apply the 'within' filter first
    filtered_df = cross_df[cross_df['distance_miles'] <= within].copy()
    
    # Sort by distance, then group by our original target_id and get the top_k
    final_df = (
        filtered_df.sort_values(by='distance_miles')
        .groupby('target_id')
        .head(top_k)
    )

    # 5. Format Output
    output_columns = [
        'target_lat', 'target_lon', 
        'match_id', 'match_lat', 'match_lon', 
        'distance_miles'
    ]
    
    return final_df[output_columns].reset_index(drop=True)

# COMMAND ----------


missing_weather = origins_weather_df[origins_weather_df["STATION"].isnull()]["ORIGIN"].unique()
print("\n\nAirports missing station codes before nearest station match")
print(missing_weather)
print(len(missing_weather))


missing_airports_df = origins_weather_df[origins_weather_df["STATION"].isnull()][["ORIGIN","LAT_ORIGIN","LON_ORIGIN"]].drop_duplicates()
missing_airports_df = missing_airports_df.loc[~missing_airports_df['LAT_ORIGIN'].isna(),:]
missing_airports_df = find_closest_points(missing_airports_df['LAT_ORIGIN'], missing_airports_df['LON_ORIGIN'],top_k = TOPK_NEAREST, within = WEATHER_ST_RADIUS, ref_df = stations_weather_df[['STATION','LAT_STATION','LON_STATION']])


def merge_missing_airports(r):
    if r['STATION'] is None or r['STATION']=='nan':
        r['STATION'] = r['match_id']
        r['LAT_STATION'] = r['match_lat'] 
        r['LON_STATION'] = r['match_lon']
        r['station_miles'] = r['distance_miles']
    return r

origins_weather_df.drop(missing_airports_df.columns, axis=1, inplace=True, errors='ignore')

origins_weather_df['STATION'] = origins_weather_df['STATION'].astype(str)

origins_weather_df = pd.merge(origins_weather_df, missing_airports_df, left_on=['LAT_ORIGIN','LON_ORIGIN'], right_on=['target_lat','target_lon'], how='left')
origins_weather_df = origins_weather_df.apply(merge_missing_airports, axis=1)

origins_weather_df.drop(missing_airports_df.columns, axis=1, inplace=True, errors='ignore')


missing_weather = origins_weather_df[origins_weather_df["STATION"].isnull()]["ORIGIN"].unique()
print("\n\nAirports missing station codes AFTER nearest station match")
print(missing_weather)
print(len(missing_weather))




# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC
# MAGIC Note that one airport (origin) can be mapped to multiple stations (depending on nearest station match), we need to be careful with it
# MAGIC

# COMMAND ----------

# We will also add the timezone information as we will use that later on. 

# Initialize TimezoneFinder
tf = TimezoneFinder()

# Function to get timezone name
def get_timezone_from_latlon(row):
    if pd.isna(row['LAT_ORIGIN']) or pd.isna(row['LON_ORIGIN']):
        return None
    tz_name = tf.timezone_at(lat=row['LAT_ORIGIN'], lng=row['LON_ORIGIN'])
    try:
        return pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        return None
    return None

origins_weather_df['timezone'] = origins_weather_df.apply(get_timezone_from_latlon, axis=1)
origins_weather_df['timezone'] = origins_weather_df['timezone'].astype(str)

# COMMAND ----------

save_spark_df(spark.createDataFrame(origins_weather_df),'origins_weather',overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actual join

# COMMAND ----------

#Load the mapping dataframe which has been created before
df_origins_weather = load_spark_df('origins_weather')
df_origins_weather.createOrReplaceTempView("origins_weather") 

# COMMAND ----------

#Create a timestamp column for the flights
df_flights = df_flights.withColumn(
    "crs_depart_ts_local",
    F.to_timestamp(
        F.concat_ws(
            " ",
            F.to_date("FL_DATE"),
            F.format_string(
                "%02d:%02d:00",
                F.floor(F.col("CRS_DEP_TIME") / 100),
                (F.col("CRS_DEP_TIME") % 100)
            )
        )
    )
)


df_flights = df_flights.withColumn("TAIL_NUM", coalesce("TAIL_NUM", lit("")))
df_flights = df_flights.withColumn("OP_CARRIER_FL_NUM", coalesce("OP_CARRIER_FL_NUM", lit(0)))
df_flights = df_flights.withColumn("ORIGIN_AIRPORT_ID", coalesce("ORIGIN_AIRPORT_ID", lit("")))
df_flights = df_flights.withColumn("CANCELLED", coalesce("CANCELLED", lit(0)))

df_flights.createOrReplaceTempView("flights") 


# COMMAND ----------

#Also add timestamp column to the weather data
df_weather = df_weather.withColumn(
    "DATE_unixts",
    F.unix_timestamp(F.to_timestamp('DATE'))
)


df_weather.createOrReplaceTempView("weather") 

# COMMAND ----------

# The actual join operation. 
# The join is flights <- mapped_station <- weather and some filtering and ranking. 
# Note that flights <- mapped_station can be one-many join, having multiple stations mapped to flight
# Similarly mapped_station <- weather can be one-many. However it will be filtered to only one row based on 
# Most recent (but beyond 2 hours delay) weather data. 


def actual_join():
  return spark.sql(f"""
  WITH 
    flights_station_base AS (
      SELECT  /*+ BROADCAST(origins_weather) */
      flights.*, 
      to_utc_timestamp(
        flights.crs_depart_ts_local, 
        (CASE WHEN origins_weather.timezone IS NULL OR origins_weather.timezone = 'None' OR origins_weather.timezone = 'null' THEN 'Etc/UCT' ELSE origins_weather.timezone END)
      ) AS crs_depart_ts_utc,
      origins_weather.STATION AS mapped_station, 
      origins_weather.station_miles AS station_miles
      FROM flights LEFT OUTER JOIN origins_weather
      ON flights.ORIGIN = origins_weather.ORIGIN
    ),
    flights_station AS (
        SELECT *, 
        unix_timestamp(crs_depart_ts_utc) as crs_depart_unixts_utc,
        unix_timestamp(crs_depart_ts_utc - INTERVAL {WEATHER_TS_AFTER} HOURS) AS time_after, 
        unix_timestamp(crs_depart_ts_utc - INTERVAL {WEATHER_TS_BEFORE} HOURS) AS time_before     
        FROM flights_station_base
    ),
    weather_filtered AS (
      SELECT * except(YEAR)
      FROM weather
      WHERE weather.station IN (
        SELECT DISTINCT mapped_station FROM flights_station )
    ), 
    flights_weather AS (
    SELECT 
        * 
        FROM 
        flights_station
        LEFT OUTER JOIN weather_filtered
            ON flights_station.mapped_station = weather_filtered.station
            AND weather_filtered.DATE_unixts BETWEEN time_after AND time_before
    ),
   flights_weather_ranked AS (
      SELECT *, 
          ROW_NUMBER() OVER( 
              PARTITION BY TAIL_NUM, OP_CARRIER_FL_NUM, ORIGIN_AIRPORT_ID, crs_depart_unixts_utc, CANCELLED
              ORDER BY DATE_unixts DESC
          ) AS rn
      FROM flights_weather
  ),
  flights_weather_best AS (
      SELECT *
      FROM flights_weather_ranked
      WHERE rn = 1    
  )
  SELECT *,
        ROW_NUMBER() OVER (PARTITION BY TAIL_NUM ORDER BY crs_depart_ts_utc) AS tail_idx
  FROM flights_weather_best
  """)



# COMMAND ----------

# Self-joining to add previous flight information. Note that no filtering is done here, so 
# Some of the information may need to be filtered out based on recency. 
# For example, if the scheduled flight is 3:30PM and the previous flight arrived at 2PM, we should not 
# use the arrival data since it has not arrived yet as of 2 hours ago of the scheduled flight. 

def prev_flight_join(viewname:str):
  return spark.sql(f"""
                  SELECT otpw_new.*,  
                    prev.CRS_DEP_TIME as PREV_CRS_DEP_TIME,
                    prev.DEP_TIME as PREV_DEP_TIME,
                    prev.crs_depart_unixts_utc as PREV_crs_depart_unixts_utc,
                    prev.DEP_DELAY as PREV_DEP_DELAY,
                    prev.DEP_DELAY_NEW as PREV_DEP_DELAY_NEW,
                    prev.DEP_DELAY_GROUP as PREV_DEP_DELAY_GROUP,
                    prev.ARR_DELAY_GROUP as PREV_ARR_DELAY_GROUP,
                    prev.CRS_ARR_TIME as PREV_CRS_ARR_TIME,
                    prev.ARR_TIME as PREV_ARR_TIME,
                    prev.ARR_DELAY as PREV_ARR_DELAY,
                    prev.CANCELLED as PREV_CANCELLED,  
                    prev.DIVERTED as PREV_DIVERTED,
                    prev.ORIGIN as PREV_ORIGIN,
                    prev.DEST as PREV_DEST,
                    prev.OP_CARRIER_FL_NUM as PREV_OP_CARRIER_FL_NUM,
                    prev.DISTANCE as PREV_DISTANCE 
                  FROM {viewname}
                  LEFT OUTER JOIN {viewname}  AS prev on 
                    otpw_new.TAIL_NUM = prev.TAIL_NUM 
                    AND 
                    otpw_new.tail_idx = prev.tail_idx + 1
                    AND otpw_new.TAIL_NUM IS NOT NULL
                  """
                  )


# COMMAND ----------

def generate_otpw_v2(viewname: str):
    #Quick wrapper to do the actual join and self-join for previous flight
    df = actual_join()
    df.createOrReplaceTempView(viewname) 
    df = prev_flight_join(viewname)
    df.persist()
    df.createOrReplaceTempView(viewname) 
    return df

#If the data is for 3 months or 1 year, it's pretty simple
if DATE_RANGE in ['3M','1Y']:
    df_otpw_new = generate_otpw_v2("otpw_new")
    print(f"Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({round(time.time())})")
    save_spark_df(df_otpw_new, OUTPUT_NAME_1, date_range = DATE_RANGE, overwrite = OVERWRITE)
    print(f"Data saved")        
    print(f"Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({round(time.time())})")
    print(f"df_flights rows: {df_flights_count_dedup}")
    print(f"df_otpw rows: {df_otpw.count()}")
    print(f"df_otpw_v2 rows: {df_otpw_new.count()}")
    print(f"Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({round(time.time())})")


#If it's 5 year (or more?) data, we'll process it year by year. Since the weather and flights data is already partitioned by year, 
#we will leverage that partition to process year by year
if DATE_RANGE == '5Y' or DATE_RANGE == '10Y':
    for year in YEARS:
        print(f"Year: {year}")        
        print(f"Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({round(time.time())})")
        df_flights_year = df_flights.filter(f"YEAR={year}")
        print(f"# of rows in Flights before dedup: {df_flights_year.count()}")
        df_flights_year = df_flights_year.dropDuplicates().persist()
        print(f"# of rows in Flights after dedup: {df_flights_year.count()}")
        print(f"Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({round(time.time())})")
        df_weather_year = df_weather.filter(f"YEAR={year}")

        #Since flight data is by local time and weather in utc, some 12/31 flights may have weather data in next year. For that we are extracting
        #one day of data from next year (if it exists)
        df_weather_year_next = df_weather.filter(f"YEAR={year+1}")
        if df_weather_year_next.count()!= 0:
            df_jan_one = df_weather_year_next.filter(
                (col("DATE") >= f"{year+1}-01-01 00:00:00") &
                (col("DATE") <  f"{year+1}-01-02 00:00:00"))
            df_weather_year = df_weather_year.unionByName(df_jan_one)
        
        #Let's process it.
        df_flights_year.createOrReplaceTempView('flights') 
        df_weather_year.createOrReplaceTempView('weather') 
        df_otpw_new_year = generate_otpw_v2("otpw_new")
        save_spark_df(df_otpw_new_year, OUTPUT_NAME_1, date_range = DATE_RANGE, overwrite = OVERWRITE, by_year = True)
        print(f"Saved year {year}")
        print(f"Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({round(time.time())})")
        print(f"df_flights rows: {df_flights_year.count()}")
        print(f"df_otpw_v2 rows: {df_otpw_new_year.count()}")
        print(f"df_weather rows: {df_weather_year.count()}")
        print(f"Current timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({round(time.time())})")
        print(f"Year {year} complete\n\n")        
        df_flights_year.unpersist()  
        df_weather_year.unpersist()  
        df_weather_year_next.unpersist()
        df_otpw_new_year.unpersist()  

      

# COMMAND ----------

df = load_spark_df('otpw_v2', date_range = '10Y')
df.createOrReplaceTempView('yoyoyo') 
display(
    spark.sql("""
        SELECT YEAR, COUNT(*) AS row_count
        FROM yoyoyo
        GROUP BY YEAR
        ORDER BY YEAR
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Misc, random stuff

# COMMAND ----------

# the original data has data upto 2021
df_flights_5y = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")
df_weather_5y = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")

# New 3-year extension (2022–2024) you created earlier
df_flights_new = spark.read.parquet("dbfs:/student-groups/Group_01_01/new_data/flights.parquet")
df_weather_new = spark.read.parquet("dbfs:/student-groups/Group_01_01/new_data/weather.parquet")

df_weather_new = df_weather_new.drop('DYTS', 'DYHF')

# Union by column name (schemas and YEAR partitions are compatible)
df_flights = df_flights_5y.unionByName(df_flights_new)
df_weather = df_weather_5y.unionByName(df_weather_new)


df_flights.createOrReplaceTempView('yoyoyo') 
display(
    spark.sql("""
        SELECT YEAR, COUNT(*) AS row_count
        FROM yoyoyo
        GROUP BY YEAR
        ORDER BY YEAR
    """)
)


# COMMAND ----------

df = load_spark_df('otpw_v2', date_range = '10Y')
df.createOrReplaceTempView('yoyoyo') 
display(
    spark.sql("""
        SELECT YEAR, COUNT(*) AS row_count
        FROM yoyoyo
        GROUP BY YEAR
        ORDER BY YEAR
    """)
)

# COMMAND ----------

# df_timediffs  = spark.sql("""
#                   SELECT (crs_depart_unixts_utc - PREV_crs_depart_unixts_utc) / 3600 AS hours_difference FROM otpw_new""")

# df_timediffs.persist()
# df_timediffs.createOrReplaceTempView("timediffs")

# print(f"Number of rows: {df_timediffs.count()}")
# print(f"Number of rows with diff more than 2 hours: {df_timediffs.filter(col('hours_difference') > 2).count()}")

# COMMAND ----------

df = load_spark_df('otpw_v2', date_range = '10Y')
df.createOrReplaceTempView('otpw')

daily = spark.sql("""
                  SELECT TO_DATE(DATE)  as DATE, COUNT(*) num_flights, SUM(DEP_DEL15) delay_cnt, SUM(DEP_DEL15) / COUNT(*) AS delay_pct
                  FROM otpw GROUP BY TO_DATE(DATE)""")

# COMMAND ----------

daily_df = daily.toPandas()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# --- Prep the data ---
daily_df['DATE'] = pd.to_datetime(daily_df['DATE'])

# Extract year as a series label
daily_df['year'] = daily_df['DATE'].dt.year

# Create a "yearless" date: keep month/day, use a dummy year (e.g. 2000)
# Also drop the time part if you only care about date:
daily_df['date_noyear'] = daily_df['DATE'].dt.normalize().apply(lambda d: d.replace(year=2000))

# Sort by x for nicer lines
daily_df = daily_df.sort_values('date_noyear')

daily_df['month'] = daily_df['DATE'].dt.month

# COMMAND ----------



# COMMAND ----------

len(daily_df)

# COMMAND ----------

plt.figure(figsize=(10, 5))

sns.lineplot(
    data=daily_df.loc[(daily_df['year'] <= 2019) & (daily_df['month'].isin([11,12]))],
    x='date_noyear',
    y='delay_pct',
    hue='year',
    marker='o'
)

ax = plt.gca()
ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
plt.xticks(rotation=45)

plt.xlabel('Date (MM-DD, year ignored)')
plt.ylabel('Delay percent')
plt.title('Delay percentage by date by year (Nov + Dec)')

plt.tight_layout()
plt.show()

# COMMAND ----------

