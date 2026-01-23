# Databricks notebook source
# MAGIC %md
# MAGIC ## Phase 1 - Data description

# COMMAND ----------

# MAGIC %md
# MAGIC ### Always run these first

# COMMAND ----------

!pip install -q altair #a declarative statistical visualization library for Python


# COMMAND ----------

from pyspark.sql.functions import col, count, when, to_timestamp
import pandas as pd
from io import StringIO
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler #Combines multiple columns into a single vector column (required for most ML operations in Spark)
import altair as alt


#Set base directory
SOURCE_DIR = "dbfs:/mnt/mids-w261"


pd.set_option('display.max_rows', 500)   
pd.set_option('display.max_columns', 200)

#Directory to store checkpoint data
DATA_DIR = "dbfs:/student-groups/Fall_2025_Group_01_01"

#This is for storing csv files so it can be downloaded from http
FILE_DIR = "/dbfs/FileStore"
FILE_SUB_DIR = "Fall_2025_Group_01_01"

# COMMAND ----------

#Helper functions

def print_df_specs(df, df_name):
    # Given spark dataframe and dataframe name (string), prints number of rows and columns
    print("Number of rows: " + str(df.count()))  #count all rows in the DataFrame
    print("Number of columns: " + str(len(df.columns))) #Returns number of columns

                      

def print_num_duplicate_rows(df, df_name, subset=None):
    # Given spark dataframe and dataframe name(string), prints number of duplicate rows and percentage
    num_rows = df.count() #count all rows in the DataFrame - full table scan
    if subset:
        num_duplicates = num_rows - df.dropDuplicates(subset).count()
    else:
        num_duplicates = num_rows - df.dropDuplicates().count()
    print(f"Columns used for dedup: {subset} (if None, all)")
    print(f"Number of duplicate rows: {num_duplicates} ({num_duplicates / num_rows * 100:.2f}%)")
    return num_duplicates

def get_missing_pct(df, df_name):
    # Given spark dataframe and dataframe name(string), prints missing values as a percentage for each column
    total_rows = df.count()
    missing_pct = df.select([
        (count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in df.columns
    ]).toPandas().transpose()
    missing_pct.columns = ["missing_pct"]
    missing_pct = missing_pct.reset_index().rename(columns={"index": "field_id"})
    return missing_pct


def save_spark_df(df, fname, save_path = DATA_DIR):
    #Quick helper function to save the created spark dataframe with default directory for consistency
    dbutils.fs.mkdirs(save_path)
    df.write.mode("overwrite").parquet(f"{os.path.join(save_path,fname)}.parquet", )
    print(f"Saved {fname}.parquet into {save_path}")


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Gather data source information
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

#Import data sources. Note that we're only using the 3 months data
df_flights_3m = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/")
df_weather_3m = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
df_otpw_3m = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")

#Put into a dictionary for convenience
data_source_map = {
    "flights_3m": df_flights_3m,
    "weather_3m": df_weather_3m,
    "stations": df_stations,
    "otpw_3m": df_otpw_3m}


# COMMAND ----------

# Get data size of the data sources
!du -sh "/dbfs/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/"
!du -sh "/dbfs/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/"
!du -sh "/dbfs/mnt/mids-w261/OTPW_3M_2015.csv"
!du -sh "/dbfs/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/"

# COMMAND ----------

# Get # of rows & cols, #/% of duplicate rows, and create temp view so it can be queried w/ spark SQL
for df_name, df in data_source_map.items():
    df.createOrReplaceTempView(df_name) #Create temp view so it can be queried w/ spark SQL
    print(df_name, "info:")
    print_df_specs(df, df_name)
    print_num_duplicate_rows(df, df_name)
    print("")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data dictionary building
# MAGIC Build a data dictionary for all data sources. For otpw, we will only include fields that doesn't exist in any of the original data sources. 
# MAGIC
# MAGIC

# COMMAND ----------

#Create base pandas dataframe that includes all fields and descriptions. 
# Field description has been picked from data source websites, however weather is chatGPT assisted so may not be completely accurate
field_info = """
source	field_id	field_desc
flights_3m	YEAR	Year
flights_3m	QUARTER	Quarter (1-4)
flights_3m	MONTH	Month
flights_3m	DAY_OF_MONTH	Day of Month
flights_3m	DAY_OF_WEEK	Day of Week
flights_3m	FL_DATE	Flight Date (yyyymmdd)
flights_3m	OP_UNIQUE_CARRIER	Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years.
flights_3m	OP_CARRIER_AIRLINE_ID	An identification number assigned by US DOT to identify a unique airline (carrier). A unique airline (carrier) is defined as one holding and reporting under the same DOT certificate regardless of its Code, Name, or holding company/corporation.
flights_3m	OP_CARRIER	Code assigned by IATA and commonly used to identify a carrier. As the same code may have been assigned to different carriers over time, the code is not always unique. For analysis, use the Unique Carrier Code.
flights_3m	TAIL_NUM	Tail Number
flights_3m	OP_CARRIER_FL_NUM	Flight Number
flights_3m	ORIGIN_AIRPORT_ID	Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
flights_3m	ORIGIN_AIRPORT_SEQ_ID	Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.
flights_3m	ORIGIN_CITY_MARKET_ID	Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.
flights_3m	ORIGIN	Origin Airport
flights_3m	ORIGIN_CITY_NAME	Origin Airport, City Name
flights_3m	ORIGIN_STATE_ABR	Origin Airport, State Code
flights_3m	ORIGIN_STATE_FIPS	Origin Airport, State Fips
flights_3m	ORIGIN_STATE_NM	Origin Airport, State Name
flights_3m	ORIGIN_WAC	Origin Airport, World Area Code
flights_3m	DEST_AIRPORT_ID	Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
flights_3m	DEST_AIRPORT_SEQ_ID	Destination Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time.
flights_3m	DEST_CITY_MARKET_ID	Destination Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market.
flights_3m	DEST	Destination Airport
flights_3m	DEST_CITY_NAME	Destination Airport, City Name
flights_3m	DEST_STATE_ABR	Destination Airport, State Code
flights_3m	DEST_STATE_FIPS	Destination Airport, State Fips
flights_3m	DEST_STATE_NM	Destination Airport, State Name
flights_3m	DEST_WAC	Destination Airport, World Area Code
flights_3m	CRS_DEP_TIME	CRS Departure Time (local time: hhmm)
flights_3m	DEP_TIME	Actual Departure Time (local time: hhmm)
flights_3m	DEP_DELAY	Difference in minutes between scheduled and actual departure time. Early departures show negative numbers.
flights_3m	DEP_DELAY_NEW	Difference in minutes between scheduled and actual departure time. Early departures set to 0.
flights_3m	DEP_DEL15	Departure Delay Indicator, 15 Minutes or More (1=Yes)
flights_3m	DEP_DELAY_GROUP	Departure Delay intervals, every (15 minutes from <-15 to >180)
flights_3m	DEP_TIME_BLK	CRS Departure Time Block, Hourly Intervals
flights_3m	TAXI_OUT	Taxi Out Time, in Minutes
flights_3m	WHEELS_OFF	Wheels Off Time (local time: hhmm)
flights_3m	WHEELS_ON	Wheels On Time (local time: hhmm)
flights_3m	TAXI_IN	Taxi In Time, in Minutes
flights_3m	CRS_ARR_TIME	CRS Arrival Time (local time: hhmm)
flights_3m	ARR_TIME	Actual Arrival Time (local time: hhmm)
flights_3m	ARR_DELAY	Difference in minutes between scheduled and actual arrival time. Early arrivals show negative numbers.
flights_3m	ARR_DELAY_NEW	Difference in minutes between scheduled and actual arrival time. Early arrivals set to 0.
flights_3m	ARR_DEL15	Arrival Delay Indicator, 15 Minutes or More (1=Yes)
flights_3m	ARR_DELAY_GROUP	Arrival Delay intervals, every (15-minutes from <-15 to >180)
flights_3m	ARR_TIME_BLK	CRS Arrival Time Block, Hourly Intervals
flights_3m	CANCELLED	Cancelled Flight Indicator (1=Yes)
flights_3m	CANCELLATION_CODE	Specifies The Reason For Cancellation
flights_3m	DIVERTED	Diverted Flight Indicator (1=Yes)
flights_3m	CRS_ELAPSED_TIME	CRS Elapsed Time of Flight, in Minutes
flights_3m	ACTUAL_ELAPSED_TIME	Elapsed Time of Flight, in Minutes
flights_3m	AIR_TIME	Flight Time, in Minutes
flights_3m	FLIGHTS	Number of Flights
flights_3m	DISTANCE	Distance between airports (miles)
flights_3m	DISTANCE_GROUP	Distance Intervals, every 250 Miles, for Flight Segment
flights_3m	CARRIER_DELAY	Carrier Delay, in Minutes
flights_3m	WEATHER_DELAY	Weather Delay, in Minutes
flights_3m	NAS_DELAY	National Air System Delay, in Minutes
flights_3m	SECURITY_DELAY	Security Delay, in Minutes
flights_3m	LATE_AIRCRAFT_DELAY	Late Aircraft Delay, in Minutes
flights_3m	FIRST_DEP_TIME	First Gate Departure Time at Origin Airport
flights_3m	TOTAL_ADD_GTIME	Total Ground Time Away from Gate for Gate Return or Cancelled Flight
flights_3m	LONGEST_ADD_GTIME	Longest Time Away from Gate for Gate Return or Cancelled Flight
flights_3m	DIV_AIRPORT_LANDINGS	Number of Diverted Airport Landings
flights_3m	DIV_REACHED_DEST	Diverted Flight Reaching Scheduled Destination Indicator (1=Yes)
flights_3m	DIV_ACTUAL_ELAPSED_TIME	Elapsed Time of Diverted Flight Reaching Scheduled Destination, in Minutes. The ActualElapsedTime column remains NULL for all diverted flights.
flights_3m	DIV_ARR_DELAY	Difference in minutes between scheduled and actual arrival time for a diverted flight reaching scheduled destination. The ArrDelay column remains NULL for all diverted flights.
flights_3m	DIV_DISTANCE	Distance between scheduled destination and final diverted airport (miles). Value will be 0 for diverted flight reaching scheduled destination.
flights_3m	DIV1_AIRPORT	Diverted Airport Code1
flights_3m	DIV1_AIRPORT_ID	Airport ID of Diverted Airport 1. Airport ID is a Unique Key for an Airport
flights_3m	DIV1_AIRPORT_SEQ_ID	Airport Sequence ID of Diverted Airport 1. Unique Key for Time Specific Information for an Airport
flights_3m	DIV1_WHEELS_ON	Wheels On Time (local time: hhmm) at Diverted Airport Code1
flights_3m	DIV1_TOTAL_GTIME	Total Ground Time Away from Gate at Diverted Airport Code1
flights_3m	DIV1_LONGEST_GTIME	Longest Ground Time Away from Gate at Diverted Airport Code1
flights_3m	DIV1_WHEELS_OFF	Wheels Off Time (local time: hhmm) at Diverted Airport Code1
flights_3m	DIV1_TAIL_NUM	Aircraft Tail Number for Diverted Airport Code1
flights_3m	DIV2_AIRPORT	Diverted Airport Code2
flights_3m	DIV2_AIRPORT_ID	Airport ID of Diverted Airport 2. Airport ID is a Unique Key for an Airport
flights_3m	DIV2_AIRPORT_SEQ_ID	Airport Sequence ID of Diverted Airport 2. Unique Key for Time Specific Information for an Airport
flights_3m	DIV2_WHEELS_ON	Wheels On Time (local time: hhmm) at Diverted Airport Code2
flights_3m	DIV2_TOTAL_GTIME	Total Ground Time Away from Gate at Diverted Airport Code2
flights_3m	DIV2_LONGEST_GTIME	Longest Ground Time Away from Gate at Diverted Airport Code2
flights_3m	DIV2_WHEELS_OFF	Wheels Off Time (local time: hhmm) at Diverted Airport Code2
flights_3m	DIV2_TAIL_NUM	Aircraft Tail Number for Diverted Airport Code2
flights_3m	DIV3_AIRPORT	Diverted Airport Code3
flights_3m	DIV3_AIRPORT_ID	Airport ID of Diverted Airport 3. Airport ID is a Unique Key for an Airport
flights_3m	DIV3_AIRPORT_SEQ_ID	Airport Sequence ID of Diverted Airport 3. Unique Key for Time Specific Information for an Airport
flights_3m	DIV3_WHEELS_ON	Wheels On Time (local time: hhmm) at Diverted Airport Code3
flights_3m	DIV3_TOTAL_GTIME	Total Ground Time Away from Gate at Diverted Airport Code3
flights_3m	DIV3_LONGEST_GTIME	Longest Ground Time Away from Gate at Diverted Airport Code3
flights_3m	DIV3_WHEELS_OFF	Wheels Off Time (local time: hhmm) at Diverted Airport Code3
flights_3m	DIV3_TAIL_NUM	Aircraft Tail Number for Diverted Airport Code3
flights_3m	DIV4_AIRPORT	Diverted Airport Code4
flights_3m	DIV4_AIRPORT_ID	Airport ID of Diverted Airport 4. Airport ID is a Unique Key for an Airport
flights_3m	DIV4_AIRPORT_SEQ_ID	Airport Sequence ID of Diverted Airport 4. Unique Key for Time Specific Information for an Airport
flights_3m	DIV4_WHEELS_ON	Wheels On Time (local time: hhmm) at Diverted Airport Code4
flights_3m	DIV4_TOTAL_GTIME	Total Ground Time Away from Gate at Diverted Airport Code4
flights_3m	DIV4_LONGEST_GTIME	Longest Ground Time Away from Gate at Diverted Airport Code4
flights_3m	DIV4_WHEELS_OFF	Wheels Off Time (local time: hhmm) at Diverted Airport Code4
flights_3m	DIV4_TAIL_NUM	Aircraft Tail Number for Diverted Airport Code4
flights_3m	DIV5_AIRPORT	Diverted Airport Code5
flights_3m	DIV5_AIRPORT_ID	Airport ID of Diverted Airport 5. Airport ID is a Unique Key for an Airport
flights_3m	DIV5_AIRPORT_SEQ_ID	Airport Sequence ID of Diverted Airport 5. Unique Key for Time Specific Information for an Airport
flights_3m	DIV5_WHEELS_ON	Wheels On Time (local time: hhmm) at Diverted Airport Code5
flights_3m	DIV5_TOTAL_GTIME	Total Ground Time Away from Gate at Diverted Airport Code5
flights_3m	DIV5_LONGEST_GTIME	Longest Ground Time Away from Gate at Diverted Airport Code5
flights_3m	DIV5_WHEELS_OFF	Wheels Off Time (local time: hhmm) at Diverted Airport Code5
flights_3m	DIV5_TAIL_NUM	Aircraft Tail Number for Diverted Airport Code5
weather_3m	YEAR	Calendar year of the observation or summary.
weather_3m	STATION	Station identifier (typically USAF-WBAN or similar code) for the reporting site.
weather_3m	DATE	Observation or summary date (YYYYMMDD).
weather_3m	LATITUDE	Station latitude in decimal degrees; south is negative.
weather_3m	LONGITUDE	Station longitude in decimal degrees; west is negative.
weather_3m	ELEVATION	Station elevation above mean sea level, in meters (when available).
weather_3m	NAME	Plain-language station name.
weather_3m	REPORT_TYPE	Code indicating report/source type (e.g., METAR, SYNOP, ASOS).
weather_3m	SOURCE	Data provider/source flag or lineage indicator.
weather_3m	HourlyAltimeterSetting	Hourly altimeter setting (pressure reduced to sea level, for aviation).
weather_3m	HourlyDewPointTemperature	Hourly dew point temperature, the saturation temperature at current moisture.
weather_3m	HourlyDryBulbTemperature	Hourly ambient (air) temperature measured in shelter/exposure.
weather_3m	HourlyPrecipitation	Precipitation amount accumulated during the hour (liquid equivalent).
weather_3m	HourlyPresentWeatherType	Codes describing present weather (e.g., rain, snow, fog) observed in the hour.
weather_3m	HourlyPressureChange	Change in station or sea-level pressure over the standard interval.
weather_3m	HourlyPressureTendency	Code describing the character of pressure change (rising/falling/steady).
weather_3m	HourlyRelativeHumidity	Hourly relative humidity, typically derived from temperature and dew point.
weather_3m	HourlySkyConditions	Cloud/sky condition codes, including ceilings and coverage layers.
weather_3m	HourlySeaLevelPressure	Hourly sea-level pressure estimate derived from station pressure and metadata.
weather_3m	HourlyStationPressure	Hourly pressure measured at station elevation.
weather_3m	HourlyVisibility	Prevailing horizontal visibility reported for the hour.
weather_3m	HourlyWetBulbTemperature	Hourly wet-bulb temperature (thermodynamic proxy for moisture/heat).
weather_3m	HourlyWindDirection	Hourly wind direction in degrees from true north (calm/variable when applicable).
weather_3m	HourlyWindGustSpeed	Highest instantaneous wind speed (gust) observed in/near the hour.
weather_3m	HourlyWindSpeed	Mean wind speed for the hour.
weather_3m	Sunrise	Local time of sunrise for the station/date (if available).
weather_3m	Sunset	Local time of sunset for the station/date (if available).
weather_3m	DailyAverageDewPointTemperature	Day's mean dew point temperature.
weather_3m	DailyAverageDryBulbTemperature	Day’s mean air temperature (average of observations or max/min).
weather_3m	DailyAverageRelativeHumidity	Day’s mean relative humidity.
weather_3m	DailyAverageSeaLevelPressure	Day’s mean sea-level pressure.
weather_3m	DailyAverageStationPressure	Day’s mean station pressure.
weather_3m	DailyAverageWetBulbTemperature	Day’s mean wet-bulb temperature.
weather_3m	DailyAverageWindSpeed	Day’s mean wind speed.
weather_3m	DailyCoolingDegreeDays	Cooling degree days for the date relative to a base (often 65°F/18°C).
weather_3m	DailyDepartureFromNormalAverageTemperature	Difference between day’s mean temperature and climatological normal.
weather_3m	DailyHeatingDegreeDays	Heating degree days for the date relative to a base (often 65°F/18°C).
weather_3m	DailyMaximumDryBulbTemperature	Highest air temperature observed for the day.
weather_3m	DailyMinimumDryBulbTemperature	Lowest air temperature observed for the day.
weather_3m	DailyPeakWindDirection	Direction of the day’s peak (maximum) wind.
weather_3m	DailyPeakWindSpeed	Speed of the day’s peak (maximum) wind.
weather_3m	DailyPrecipitation	Total liquid precipitation for the day (may include trace flags).
weather_3m	DailySnowDepth	Snow depth on the ground at observation time for the day.
weather_3m	DailySnowfall	New snow amount (liquid equivalent excluded) accumulated during the day.
weather_3m	DailySustainedWindDirection	Direction associated with the highest sustained wind for the day.
weather_3m	DailySustainedWindSpeed	Highest sustained (averaged) wind speed observed for the day.
weather_3m	DailyWeather	Daily weather summary codes/flags (events such as fog, thunder, etc.).
weather_3m	MonthlyAverageRH	Monthly mean relative humidity.
weather_3m	MonthlyDaysWithGT001Precip	Number of days in the month with precipitation ≥ 0.01 in (0.25 mm).
weather_3m	MonthlyDaysWithGT010Precip	Number of days in the month with precipitation ≥ 0.10 in (2.54 mm).
weather_3m	MonthlyDaysWithGT32Temp	Count of days with maximum temperature > 32°F (0°C) or threshold per dataset.
weather_3m	MonthlyDaysWithGT90Temp	Count of days with maximum temperature > 90°F (32.2°C).
weather_3m	MonthlyDaysWithLT0Temp	Count of days with minimum temperature < 0°F (−17.8°C).
weather_3m	MonthlyDaysWithLT32Temp	Count of days with minimum temperature < 32°F (0°C).
weather_3m	MonthlyDepartureFromNormalAverageTemperature	Difference between monthly mean temperature and its normal.
weather_3m	MonthlyDepartureFromNormalCoolingDegreeDays	Departure of monthly CDD from normal CDD.
weather_3m	MonthlyDepartureFromNormalHeatingDegreeDays	Departure of monthly HDD from normal HDD.
weather_3m	MonthlyDepartureFromNormalMaximumTemperature	Departure of monthly mean of daily maxima from normal.
weather_3m	MonthlyDepartureFromNormalMinimumTemperature	Departure of monthly mean of daily minima from normal.
weather_3m	MonthlyDepartureFromNormalPrecipitation	Departure of monthly total precipitation from normal.
weather_3m	MonthlyDewpointTemperature	Monthly mean dew point temperature.
weather_3m	MonthlyGreatestPrecip	Greatest 24-hour (or event) liquid precipitation amount in the month.
weather_3m	MonthlyGreatestPrecipDate	Date on which the monthly greatest precipitation occurred.
weather_3m	MonthlyGreatestSnowDepth	Maximum snow depth on ground observed in the month.
weather_3m	MonthlyGreatestSnowDepthDate	Date of the maximum snow depth.
weather_3m	MonthlyGreatestSnowfall	Largest single-day snowfall in the month.
weather_3m	MonthlyGreatestSnowfallDate	Date of the largest single-day snowfall.
weather_3m	MonthlyMaxSeaLevelPressureValue	Highest sea-level pressure observed in the month.
weather_3m	MonthlyMaxSeaLevelPressureValueDate	Date of the monthly maximum sea-level pressure.
weather_3m	MonthlyMaxSeaLevelPressureValueTime	Time of the monthly maximum sea-level pressure.
weather_3m	MonthlyMaximumTemperature	Highest daily maximum temperature during the month.
weather_3m	MonthlyMeanTemperature	Mean temperature for the month (often (Tmax+Tmin)/2 or obs average).
weather_3m	MonthlyMinSeaLevelPressureValue	Lowest sea-level pressure observed in the month.
weather_3m	MonthlyMinSeaLevelPressureValueDate	Date of the monthly minimum sea-level pressure.
weather_3m	MonthlyMinSeaLevelPressureValueTime	Time of the monthly minimum sea-level pressure.
weather_3m	MonthlyMinimumTemperature	Lowest daily minimum temperature during the month.
weather_3m	MonthlySeaLevelPressure	Monthly mean sea-level pressure.
weather_3m	MonthlyStationPressure	Monthly mean station pressure.
weather_3m	MonthlyTotalLiquidPrecipitation	Total liquid precipitation for the month.
weather_3m	MonthlyTotalSnowfall	Total snowfall for the month.
weather_3m	MonthlyWetBulb	Monthly mean wet-bulb temperature.
weather_3m	AWND	Average daily wind speed for the period (often from GHCN-D “AWND”).
weather_3m	CDSD	Count of days since last snowfall or cold-season day metric (dataset-specific).
weather_3m	CLDD	Cooling degree days for the month/period (GHCN “CLDD”).
weather_3m	DSNW	Number of days with snow on ground (dataset-specific GHCN metric).
weather_3m	HDSD	Count of days since last snow depth event or heating-season day metric (dataset-specific).
weather_3m	HTDD	Heating degree days for the month/period (GHCN “HTDD”).
weather_3m	NormalsCoolingDegreeDay	Climatological normal CDD for the period/location.
weather_3m	NormalsHeatingDegreeDay	Climatological normal HDD for the period/location.
weather_3m	ShortDurationEndDate005	End date/time stamp for 5-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate010	End date/time stamp for 10-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate015	End date/time stamp for 15-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate020	End date/time stamp for 20-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate030	End date/time stamp for 30-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate045	End date/time stamp for 45-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate060	End date/time stamp for 60-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate080	End date/time stamp for 80-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate100	End date/time stamp for 100-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate120	End date/time stamp for 120-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate150	End date/time stamp for 150-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationEndDate180	End date/time stamp for 180-minute maximum short-duration precipitation event in month.
weather_3m	ShortDurationPrecipitationValue005	Maximum 5-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue010	Maximum 10-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue015	Maximum 15-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue020	Maximum 20-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue030	Maximum 30-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue045	Maximum 45-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue060	Maximum 60-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue080	Maximum 80-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue100	Maximum 100-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue120	Maximum 120-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue150	Maximum 150-minute precipitation amount observed in the month.
weather_3m	ShortDurationPrecipitationValue180	Maximum 180-minute precipitation amount observed in the month.
weather_3m	REM	Free-text remarks or metadata notes associated with the record.
weather_3m	BackupDirection	Direction measurement from a designated backup sensor/system (if primary unavailable).
weather_3m	BackupDistance	Distance to a backup station or sensor used for substitution.
weather_3m	BackupDistanceUnit	Unit for backup distance (e.g., km or miles).
weather_3m	BackupElements	Elements/parameters sourced from the backup station/sensor.
weather_3m	BackupElevation	Elevation of the backup station/sensor.
weather_3m	BackupEquipment	Description or code for backup instrumentation used.
weather_3m	BackupLatitude	Latitude of the backup station/sensor.
weather_3m	BackupLongitude	Longitude of the backup station/sensor.
weather_3m	BackupName	Name/identifier of the backup station/sensor.
weather_3m	WindEquipmentChangeDate	Date when wind instrumentation was changed or reconfigured.
stations	usaf	
stations	wban	
stations	station_id	
stations	lat	
stations	lon	
stations	neighbor_id	
stations	neighbor_name	
stations	neighbor_state	
stations	neighbor_call	
stations	neighbor_lat	
stations	neighbor_lon	
stations	distance_to_neighbor	
"""

df_fields = pd.read_csv(StringIO(field_info), sep="\t")


# COMMAND ----------

# Add otpw columns that don't exist in any of the other tables + Add a column to show if a column is in otpw
otpw_cols = data_source_map["otpw_3m"].columns
otpw_orig_cols = otpw_cols.copy()
for df_name, df in data_source_map.items():
    if "otp" in df_name:
        continue
    source_cols = df.columns
    otpw_orig_cols = [col for col in otpw_orig_cols if col not in source_cols]

#Create new dataframe for otpw original columns and concatenate with df_fields
add_df = pd.DataFrame(data=otpw_orig_cols, columns=["field_id"])
add_df['source'] = 'otpw_3m'
df_fields = pd.concat([df_fields, add_df], ignore_index=True)

#Add a flag if a field exists in otpw dataframe
df_fields['in_otpw'] = df_fields['field_id'].isin(otpw_cols)
df_fields.loc[df_fields['source'] == "otpw", 'in_otpw'] = True
df_fields['in_otpw'] = df_fields['in_otpw'].fillna(False).astype(bool)


# COMMAND ----------

# Add data type column into df_fields that's extracted from the other dataframes
dtype_map = {}
for df_name, df in data_source_map.items():
    for col_name, dtype in df.dtypes:
        dtype_map[(df_name, col_name)] = dtype

df_fields['data_type'] = df_fields.apply(lambda row: dtype_map.get((row['source'], row['field_id']), None), axis=1)

# COMMAND ----------

# Add % of data missing for each field
missing_pct_df = None
df_fields.drop(columns=['missing_pct'],inplace=True, errors='ignore')
for df_name, df in data_source_map.items():
    if missing_pct_df is None:
        missing_pct_df = get_missing_pct(df, df_name)
        missing_pct_df['source'] = df_name
    else:
        missing_pct_df = pd.concat([missing_pct_df, get_missing_pct(df, df_name)], ignore_index=True)
        missing_pct_df['source'].fillna(df_name, inplace=True)
df_fields = pd.merge(df_fields, missing_pct_df, how='left', on=['source','field_id'])

# COMMAND ----------

# Add summary stats for each field. It can be merged with missing_pct_df task, but separting for clarity
summary_stats_df = None
temp_df = None
df_fields.drop(columns=['mean','std','min','max','count', '25%', '50%', '75%'],inplace=True, errors='ignore')

for df_name, df in data_source_map.items():

    temp_df = df.summary().toPandas().set_index('summary').transpose()
    temp_df = temp_df.reset_index().rename(columns={"index":"field_id"})
    temp_df['source'] = df_name
    if summary_stats_df is None:
        summary_stats_df = temp_df
    else:
        summary_stats_df = pd.concat([summary_stats_df, temp_df], ignore_index=True)
df_fields = df_fields.merge(summary_stats_df, how='left', on=['source','field_id'])

# COMMAND ----------

#Results
df_fields.head(10)

# COMMAND ----------

#Save into filestore. 
df_fields.to_csv("/tmp/fields.csv")
!mv /tmp/fields.csv {FILE_DIR / "fields.csv"}
print(f"Download from https://<your-databricks-workspace>/files/{FILE_SUB_DIR}/fields.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prep data for EDA
# MAGIC EDA will be done using the OPTW table for now. We may utilize the origin tables later on for deeper EDA. We extract only subset of columns, and also convert some fields to appropriate data type
# MAGIC

# COMMAND ----------

#Create (or load) a smaller table that only includes the columns to analyze
fields = ["YEAR","QUARTER","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","FL_DATE","OP_UNIQUE_CARRIER","OP_CARRIER_AIRLINE_ID","TAIL_NUM","OP_CARRIER_FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN","ORIGIN_STATE_ABR","DEST_AIRPORT_ID","DEST","DEST_STATE_ABR","CRS_DEP_TIME","DEP_TIME","DEP_DELAY","DEP_DELAY_NEW","DEP_DEL15","ARR_TIME","ARR_DELAY","ARR_DELAY_NEW","ARR_DEL15","CANCELLED","CANCELLATION_CODE","DIVERTED","DISTANCE","CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY","STATION","DATE","HourlyAltimeterSetting","HourlyDewPointTemperature","HourlyDryBulbTemperature","HourlyPrecipitation","HourlyPresentWeatherType","HourlyPressureChange","HourlyPressureTendency","HourlyRelativeHumidity","HourlySkyConditions","HourlyWindGustSpeed","HourlyWindSpeed","DailyAverageDewPointTemperature","sched_depart_date_time_UTC","four_hours_prior_depart_UTC","two_hours_prior_depart_UTC"]

try:
    df_otpw_3m_eda = spark.read.parquet(os.path.join(DATA_DIR, "otpw_3m_eda.parquet"))
    print("loaded from existing file")
except Exception as e:
    print(e)
    print("Creating new file instead")
    df_otpw_3m_eda = df_otpw_3m.select(fields)
    save_spark_df(df_otpw_3m_eda, "otpw_3m_eda.parquet")
    print(f"Snapshot saved to {DATA_DIR}/otpw_3m_eda.parquet")

df_otpw_3m_eda.createOrReplaceTempView("otpw_3m_eda")


# COMMAND ----------

#Since all fields in optw are strings, we need to convert them. We convert some fields into double or datetime

# First, get summary stats (e.g. mean,stdev) information into dataframe
field_summary = df_otpw_3m_eda.summary().toPandas()
field_summary = field_summary.set_index("summary").transpose().reset_index().rename(columns={"index":"field_id"})

# Even though all fields are string, Spark tries to convert to float/double to create the summary. 
# So, we use this information to determine if a fields should be casted to double
convert_to_dbl = field_summary.loc[~field_summary['mean'].isna(),"field_id"].to_list()
for col in convert_to_dbl:
    df_otpw_3m_eda = df_otpw_3m_eda.withColumn(col, df_otpw_3m_eda[col].cast("Double"))

# For dates, we look at the "min" field to see if the field includes regex format of \d\d\d\d-\d\d-\d\d
convert_to_datetime = field_summary.loc[field_summary["min"].str.contains(r"\d\d\d\d-\d\d-\d\d", regex=True),"field_id"].to_list()
for col in convert_to_datetime:
     df_otpw_3m_eda = df_otpw_3m_eda.withColumn(col, to_timestamp(df_otpw_3m_eda[col]))


# COMMAND ----------

display(spark.sql("SELECT * from otpw_3m_eda limit 20"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA

# COMMAND ----------

# and %-age flights delayed
# **Key insights:**
# * Approximately 20% of flights are delayed (excluding cancelled flights)

#Get data
df = spark.sql("SELECT CASE WHEN isnull(DEP_DEL15) THEN 'Cancelled' WHEN DEP_DEL15 = 0 THEN 'On time' ELSE 'Delayed' END AS DEP_DEL15, count(*) as count from otpw_3m_eda GROUP BY DEP_DEL15").toPandas()

#Create graph
ax = sns.barplot(df, x='DEP_DEL15', y='count')

#Edit graph
ax.set_title("# of delayed departure flights (>15 min)")
ax.set_xlabel("Status")
ax.set_ylabel("Count")
plt.ticklabel_format(style='plain', axis='y')
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", label_type="edge", padding=2)
plt.show()

pct = df.loc[df['DEP_DEL15'] == "Delayed","count"].values[0] / (
    df.loc[df['DEP_DEL15'] == "On time","count"].values[0] + 
    df.loc[df['DEP_DEL15'] == "Delayed","count"].values[0]
    )
print(f"Percentage of flights delayed: {pct*100:.4f}%")

# COMMAND ----------

# # Min updated - add another histogram and add % to each category
# and %-age flights delayed
# **Key insights:**
# * Approximately 20% of flights are delayed (excluding cancelled flights)

#Get data
df = spark.sql("SELECT CASE WHEN isnull(DEP_DEL15) THEN 'Cancelled' WHEN DEP_DEL15 = 0 THEN 'On time' ELSE 'Delayed' END AS DEP_DEL15, count(*) as count from otpw_3m_eda GROUP BY DEP_DEL15").toPandas()

# Calculate percentages for all flights
df['percentage'] = (df['count'] / df['count'].sum()) * 100

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Define colors for better distinction
colors = {'On time': '#27ae60', 'Cancelled': '#e74c3c', 'Delayed': '#f39c12'}

# First histogram: All flights (including cancelled)
bars1 = sns.barplot(data=df, x='DEP_DEL15', y='count', ax=ax1, 
                     palette=[colors.get(x, '#3498db') for x in df['DEP_DEL15']])
ax1.set_title("Flight Departure Status - All Flights", fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel("Status", fontsize=12)
ax1.set_ylabel("Number of Flights", fontsize=12)
ax1.ticklabel_format(style='plain', axis='y')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Smart label placement based on bar height
max_height = df['count'].max()
for i, (container, row) in enumerate(zip(ax1.containers, df.itertuples())):
    labels = [f'{int(row.count):,}\n({row.percentage:.2f}%)']
    # If bar is less than 10% of max height, put label on top; otherwise inside
    if row.count < max_height * 0.1:
        ax1.bar_label(container, labels=labels, label_type="edge", padding=3, fontsize=10, fontweight='bold', color='black')
    else:
        ax1.bar_label(container, labels=labels, label_type="center", padding=3, fontsize=10, fontweight='bold', color='white')

# Second histogram: Exclude cancelled flights
df_no_cancel = df[df['DEP_DEL15'] != 'Cancelled'].copy()
df_no_cancel['percentage'] = (df_no_cancel['count'] / df_no_cancel['count'].sum()) * 100

bars2 = sns.barplot(data=df_no_cancel, x='DEP_DEL15', y='count', ax=ax2,
                     palette=[colors.get(x, '#3498db') for x in df_no_cancel['DEP_DEL15']])
ax2.set_title("Flight Departure Status - Excluding Cancelled", fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel("Status", fontsize=12)
ax2.set_ylabel("Number of Flights", fontsize=12)
ax2.ticklabel_format(style='plain', axis='y')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Smart label placement based on bar height
max_height_no_cancel = df_no_cancel['count'].max()
for i, (container, row) in enumerate(zip(ax2.containers, df_no_cancel.itertuples())):
    labels = [f'{int(row.count):,}\n({row.percentage:.2f}%)']
    # If bar is less than 10% of max height, put label on top; otherwise inside
    if row.count < max_height_no_cancel * 0.1:
        ax2.bar_label(container, labels=labels, label_type="edge", padding=3, fontsize=10, fontweight='bold', color='black')
    else:
        ax2.bar_label(container, labels=labels, label_type="center", padding=3, fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.show()

# Calculate percentage of delayed flights (excluding cancelled)
pct = df.loc[df['DEP_DEL15'] == "Delayed","count"].values[0] / (
    df.loc[df['DEP_DEL15'] == "On time","count"].values[0] + 
    df.loc[df['DEP_DEL15'] == "Delayed","count"].values[0]
    )
print(f"\n{'='*60}")
print(f"Key Statistics:")
print(f"{'='*60}")
print(f"Total flights: {df['count'].sum():,}")
print(f"Cancelled flights: {df.loc[df['DEP_DEL15']=='Cancelled', 'count'].values[0]:,} ({df.loc[df['DEP_DEL15']=='Cancelled', 'percentage'].values[0]:.2f}%)")
print(f"Percentage of flights delayed (excluding cancelled): {pct*100:.2f}%")
print(f"{'='*60}")

# COMMAND ----------

# Min Updated: adding analysis to understand between departure delays and arrival delays as many flights that depart late still arrive on time
# Compare departure vs arrival delays
df_delay_comparison = spark.sql("""
    SELECT 
        CASE WHEN DEP_DEL15 = 1 THEN 'Dep Delayed' ELSE 'Dep On-Time' END as dep_status,
        CASE WHEN ARR_DEL15 = 1 THEN 'Arr Delayed' ELSE 'Arr On-Time' END as arr_status,
        COUNT(*) as count
    FROM otpw_3m_eda
    WHERE CANCELLED = 0  -- Exclude cancelled flights
    GROUP BY dep_status, arr_status
""").toPandas()

# Calculate percentages
total = df_delay_comparison['count'].sum()
df_delay_comparison['percentage'] = (df_delay_comparison['count'] / total) * 100

print("\n" + "="*70)
print("Departure vs Arrival Delay Crosstab")
print("="*70)
print(df_delay_comparison.pivot(index='dep_status', columns='arr_status', values='percentage'))
print("\n")

# Key insight: What % of departure delays result in arrival delays?
dep_delayed = df_delay_comparison[df_delay_comparison['dep_status'] == 'Dep Delayed']['count'].sum()
dep_delayed_arr_delayed = df_delay_comparison[
    (df_delay_comparison['dep_status'] == 'Dep Delayed') & 
    (df_delay_comparison['arr_status'] == 'Arr Delayed')
]['count'].values[0]

print(f"Of flights that depart late, {dep_delayed_arr_delayed/dep_delayed*100:.1f}% also arrive late")
print(f"Of flights that depart late, {100 - dep_delayed_arr_delayed/dep_delayed*100:.1f}% make up time and arrive on-time")

# COMMAND ----------

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Departure delays
df_dep = spark.sql("""
    SELECT 
        CASE WHEN DEP_DEL15 = 1 THEN 'Delayed' ELSE 'On time' END as status,
        COUNT(*) as count
    FROM otpw_3m_eda
    WHERE CANCELLED = 0
    GROUP BY status
""").toPandas()
df_dep['percentage'] = (df_dep['count'] / df_dep['count'].sum()) * 100

# Arrival delays
df_arr = spark.sql("""
    SELECT 
        CASE WHEN ARR_DEL15 = 1 THEN 'Delayed' ELSE 'On time' END as status,
        COUNT(*) as count
    FROM otpw_3m_eda
    WHERE CANCELLED = 0
    GROUP BY status
""").toPandas()
df_arr['percentage'] = (df_arr['count'] / df_arr['count'].sum()) * 100

colors = {'On time': '#27ae60', 'Delayed': '#f39c12'}

# Plot 1: Departure delays
sns.barplot(data=df_dep, x='status', y='count', ax=axes[0],
            palette=[colors.get(x, '#3498db') for x in df_dep['status']])
axes[0].set_title("Departure Delays (DEP_DEL15)", fontsize=14, fontweight='bold', pad=20)
axes[0].set_xlabel("Status", fontsize=12)
axes[0].set_ylabel("Number of Flights", fontsize=12)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

max_height_dep = df_dep['count'].max()
for container, row in zip(axes[0].containers, df_dep.itertuples()):
    labels = [f'{int(row.count):,}\n({row.percentage:.2f}%)']
    if row.count < max_height_dep * 0.1:
        axes[0].bar_label(container, labels=labels, label_type="edge", padding=3, fontsize=10, fontweight='bold')
    else:
        axes[0].bar_label(container, labels=labels, label_type="center", padding=3, fontsize=10, fontweight='bold', color='white')

# Plot 2: Arrival delays
sns.barplot(data=df_arr, x='status', y='count', ax=axes[1],
            palette=[colors.get(x, '#3498db') for x in df_arr['status']])
axes[1].set_title("Arrival Delays (ARR_DEL15)", fontsize=14, fontweight='bold', pad=20)
axes[1].set_xlabel("Status", fontsize=12)
axes[1].set_ylabel("Number of Flights", fontsize=12)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

max_height_arr = df_arr['count'].max()
for container, row in zip(axes[1].containers, df_arr.itertuples()):
    labels = [f'{int(row.count):,}\n({row.percentage:.2f}%)']
    if row.count < max_height_arr * 0.1:
        axes[1].bar_label(container, labels=labels, label_type="edge", padding=3, fontsize=10, fontweight='bold')
    else:
        axes[1].bar_label(container, labels=labels, label_type="center", padding=3, fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.show()

print(f"Departure delay rate: {df_dep[df_dep['status']=='Delayed']['percentage'].values[0]:.2f}%")
print(f"Arrival delay rate: {df_arr[df_arr['status']=='Delayed']['percentage'].values[0]:.2f}%")

# COMMAND ----------

# Distribution of delayed minutes
# **Key insights:**
# * Delayed minutes has very long tail, though this was expected


#Get data
df = spark.sql("""
               WITH t as (SELECT ROUND(DEP_DELAY_NEW) AS DEP_DELAY_NEW FROM otpw_3m_eda WHERE DEP_DELAY_NEW != 0)
               
               SELECT DEP_DELAY_NEW, COUNT(*) as count from t WHERE DEP_DELAY_NEW < 240 GROUP BY DEP_DELAY_NEW
               """).toPandas()

#Create graph
ax = sns.histplot(df, x='DEP_DELAY_NEW', weights='count', stat="percent", binwidth=10)

#Edit graph
ax.set_title("Distribution of delayed minutes in percent \nRounded & extreme values removed")
ax.set_xlabel("Departure Delay")
ax.set_ylabel("Percent of flights")
plt.ticklabel_format(style='plain', axis='y')
plt.show()



# COMMAND ----------

# Average & median time of delay by category
# **Key insights:**
# * For both mean and median, late aircraft (i.e. previous flight's delay) is highest
# * Second is carrier delay

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

#Get data
df_avg = spark.sql("""
               SELECT  AVG(CARRIER_DELAY) AS `Carrier`, 
               AVG(WEATHER_DELAY) AS `Weather`,
               AVG(NAS_DELAY) AS `MAS`,
               AVG(SECURITY_DELAY) AS `Security`,
               AVG(LATE_AIRCRAFT_DELAY) AS `Late aircraft`
               FROM otpw_3m_eda WHERE DEP_DEL15 = 1.0
               """).toPandas().transpose()
df_avg = df_avg.reset_index()
df_avg = df_avg.rename(columns={"index":"Cause",0:"value"})


sns.barplot(data=df_avg, x='Cause', y='value', ax=axes[0])
axes[0].set_title("Causes of delay in AVERAGE minutes")
axes[0].set_xlabel("Cause")
axes[0].set_ylabel("Avg. Delay (minutes)")
axes[0].tick_params(axis='x', labelrotation=45)
plt.ticklabel_format(style='plain', axis='y')


df_med = spark.sql(""" SELECT
               MEDIAN(CARRIER_DELAY) AS `Carrier`, 
               MEDIAN(WEATHER_DELAY) AS `Weather`,
               MEDIAN(NAS_DELAY) AS `NAS`,
               MEDIAN(SECURITY_DELAY) AS `Security`,
               MEDIAN(LATE_AIRCRAFT_DELAY) AS `Late aircraft`
               FROM otpw_3m_eda WHERE DEP_DEL15 = 1.0
               """).toPandas().transpose()
df_med = df_med.reset_index()
df_med = df_med.rename(columns={"index":"Cause",0:"value"})

for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.1f", label_type="edge", padding=2)


# Plot on the second subplot (axes[1])
sns.barplot(data=df_med, x='Cause', y='value', ax=axes[1])
axes[1].set_title("Causes of delay in MEDIAN minutes")
axes[1].set_xlabel("Cause")
axes[1].set_ylabel("Median Delay (minutes)")
axes[1].tick_params(axis='x', labelrotation=45)

for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.1f", label_type="edge", padding=2)



# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# COMMAND ----------

# Min updated - add distribution of actual departure delay minutes
# **Key insights:**
# * Shows the frequency distribution of delay durations
# * Helps identify common delay ranges and outliers

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

# Get departure delay data for delayed flights
df_delay_dist = spark.sql("""
    SELECT DEP_DELAY
    FROM otpw_3m_eda 
    WHERE DEP_DEL15 = 1.0 
    AND DEP_DELAY IS NOT NULL
    AND DEP_DELAY > 0
""").toPandas()

# Create histogram with 250 bins
n, bins, patches = ax.hist(df_delay_dist['DEP_DELAY'], 
                           bins=250, 
                           edgecolor='black', 
                           linewidth=0.3,
                           alpha=0.7)

ax.set_title("Distribution of Departure Delay (Minutes) - Delayed Flights", fontsize=14, fontweight='bold')
ax.set_xlabel("Departure Delay (minutes)", fontsize=12)
ax.set_ylabel("Frequency (Number of flights)", fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add statistics text box
mean_delay = df_delay_dist['DEP_DELAY'].mean()
median_delay = df_delay_dist['DEP_DELAY'].median()
max_delay = df_delay_dist['DEP_DELAY'].max()
total_flights = len(df_delay_dist)

stats_text = f'Total Delayed Flights: {total_flights:,}\nMean: {mean_delay:.1f} min\nMedian: {median_delay:.1f} min\nMax: {max_delay:.0f} min'
ax.text(0.98, 0.97, stats_text, 
        transform=ax.transAxes, 
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Optional: Print summary statistics
print(f"Summary Statistics for Departure Delays:")
print(f"  Count: {total_flights:,}")
print(f"  Mean: {mean_delay:.2f} minutes")
print(f"  Median: {median_delay:.2f} minutes")
print(f"  Std Dev: {df_delay_dist['DEP_DELAY'].std():.2f} minutes")
print(f"  Min: {df_delay_dist['DEP_DELAY'].min():.2f} minutes")
print(f"  Max: {max_delay:.2f} minutes")
print(f"\nPercentiles:")
print(f"  75th: {df_delay_dist['DEP_DELAY'].quantile(0.75):.2f} minutes")
print(f"  90th: {df_delay_dist['DEP_DELAY'].quantile(0.90):.2f} minutes")
print(f"  95th: {df_delay_dist['DEP_DELAY'].quantile(0.95):.2f} minutes")
print(f"  99th: {df_delay_dist['DEP_DELAY'].quantile(0.99):.2f} minutes")

# COMMAND ----------

# Min updated - add distribution of actual departure delay minutes with LOG SCALE
# **Key insights:**
# * Log scale helps visualize the long tail of delay distribution
# * Better shows both frequent short delays and rare extreme delays

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

# Get departure delay data for delayed flights
df_delay_dist = spark.sql("""
    SELECT DEP_DELAY
    FROM otpw_3m_eda 
    WHERE DEP_DEL15 = 1.0 
    AND DEP_DELAY IS NOT NULL
    AND DEP_DELAY > 0
""").toPandas()

# Create histogram with 2000 bins and log scale
n, bins, patches = ax.hist(df_delay_dist['DEP_DELAY'], 
                           bins=2000, 
                           edgecolor='black', 
                           linewidth=0.1,
                           alpha=0.7,
                           color='steelblue')

# Set log scale on y-axis
ax.set_yscale('log')

ax.set_title("Distribution of Departure Delay (Minutes) - LOG SCALE", fontsize=14, fontweight='bold')
ax.set_xlabel("Departure Delay (minutes)", fontsize=12)
ax.set_ylabel("Frequency (Number of flights) - Log Scale", fontsize=12)
ax.grid(axis='both', alpha=0.3, linestyle='--', which='both')

# Add statistics text box
mean_delay = df_delay_dist['DEP_DELAY'].mean()
median_delay = df_delay_dist['DEP_DELAY'].median()
max_delay = df_delay_dist['DEP_DELAY'].max()
total_flights = len(df_delay_dist)

stats_text = f'Total Delayed Flights: {total_flights:,}\nMean: {mean_delay:.1f} min\nMedian: {median_delay:.1f} min\nMax: {max_delay:.0f} min'
ax.text(0.98, 0.97, stats_text, 
        transform=ax.transAxes, 
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print percentile distribution to understand the long tail
print(f"\nDelay Distribution Analysis:")
print(f"{'Percentile':<12} {'Delay (min)':<12} {'% of Max':<12}")
print("-" * 40)
percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
for p in percentiles:
    val = df_delay_dist['DEP_DELAY'].quantile(p/100) if p < 100 else max_delay
    pct_of_max = (val / max_delay) * 100
    print(f"{p:>6}th      {val:>8.1f}      {pct_of_max:>8.1f}%")

# COMMAND ----------

# Distribution of delayed minutes
# **Key insights:**
# * For any previous flight that is delayed by a minute or more, there's high chance the next flight is delayed. This may be a feature we want to use by pulling the previous flight's delay. However, because we are predicting 2 hours before scheduled departure, we need to be careful on which data is available prior to the prediction time


#Get data
df = spark.sql("""
               WITH t as 
               (SELECT (CASE WHEN LATE_AIRCRAFT_DELAY IS NULL THEN 0 ELSE LATE_AIRCRAFT_DELAY END) AS LATE_AIRCRAFT_DELAY, 
                        DEP_DEL15,
                        (CASE WHEN LATE_AIRCRAFT_DELAY = 0 THEN TRUE ELSE FALSE END) AS `=0`,
                        (CASE WHEN LATE_AIRCRAFT_DELAY > 0 THEN TRUE ELSE FALSE END) AS `>0`,
                        (CASE WHEN LATE_AIRCRAFT_DELAY > 1 THEN TRUE ELSE FALSE END) AS `>01`,
                        (CASE WHEN LATE_AIRCRAFT_DELAY > 2 THEN TRUE ELSE FALSE END) AS `>02`,
                        (CASE WHEN LATE_AIRCRAFT_DELAY > 4 THEN TRUE ELSE FALSE END) AS `>04`,
                        (CASE WHEN LATE_AIRCRAFT_DELAY > 6 THEN TRUE ELSE FALSE END) AS `>06`,
                        (CASE WHEN LATE_AIRCRAFT_DELAY > 9 THEN TRUE ELSE FALSE END) AS `>09`,
                        (CASE WHEN LATE_AIRCRAFT_DELAY > 14 THEN TRUE ELSE FALSE END) AS `>14`
                        FROM otpw_3m_eda)
               
               SELECT * from t
               """).melt(ids=['LATE_AIRCRAFT_DELAY','DEP_DEL15'], values=['=0','>0','>01','>02','>04','>06','>09','>14'], variableColumnName='criteria', valueColumnName='match').toPandas()

df = df.loc[df['match'] == True,:].groupby(['criteria','match','DEP_DEL15'])[['LATE_AIRCRAFT_DELAY']].count()

#Create graph
ax = sns.barplot(df, x='criteria', y='LATE_AIRCRAFT_DELAY', hue = 'DEP_DEL15')

#Edit graph
ax.set_title("# of Delayed/Non-Delayed Flights by previous aircraft delay")
ax.set_xlabel("# of minutes the previous aircraft was delayed")
ax.set_ylabel("Count")
plt.show()

# COMMAND ----------

#Pearson correlation heatmap for numeric values
# **Key insights:**
# * Focusing on correlation related to DEP_DEL15 (delayed by 15+ minutes), there's only a few that has correlation (based on Pearson's correlation)
# * Weather related delay is low correlation, but it may be useful once we massage them for easier use
# * Note that null-values were imputed with 0.0 without looking at the data, so if we use adequate imputation the correlation may change.
# * Also, note that we are only using Jan~Mar data, so we aren't picking seasonal differences. Likely better to do another correlation analysis with 1 year data. 

df_filled = df_otpw_3m_eda.fillna(0)

# Select only numeric columns
numeric_cols = [field.name for field in df_filled.schema.fields 
                if str(field.dataType) in ("IntegerType()", "DoubleType()", "FloatType()", "LongType()")]

# Assemble into a single features vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid='keep')
vector_df = assembler.transform(df_filled).select("features")

# Compute correlation matrix in Spark
corr_matrix = Correlation.corr(vector_df, "features", method="pearson").head()[0]

# Convert to numpy / pandas
corr_array = corr_matrix.toArray()
corr_df = pd.DataFrame(corr_array, columns=numeric_cols, index=numeric_cols)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=False, cmap="coolwarm", square=True)
plt.title("Correlation heatmap for numeric values \n(NaN values filled w/ zero)")
plt.show()

# COMMAND ----------

# Min updated - Pearson and Spearman correlation heatmaps for numeric values
# **Key insights:**
# * Spearman correlation captures monotonic (not just linear) relationships
# * Comparing both helps identify non-linear relationships that Pearson might miss

df_filled = df_otpw_3m_eda.fillna(0)

# Select only numeric columns
numeric_cols = [field.name for field in df_filled.schema.fields 
                if str(field.dataType) in ("IntegerType()", "DoubleType()", "FloatType()", "LongType()")]

# Assemble into a single features vector
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid='keep')
vector_df = assembler.transform(df_filled).select("features")

# Compute both correlation matrices in Spark
corr_matrix_pearson = Correlation.corr(vector_df, "features", method="pearson").head()[0]
corr_matrix_spearman = Correlation.corr(vector_df, "features", method="spearman").head()[0]

# Convert to numpy / pandas
corr_array_pearson = corr_matrix_pearson.toArray()
corr_df_pearson = pd.DataFrame(corr_array_pearson, columns=numeric_cols, index=numeric_cols)

corr_array_spearman = corr_matrix_spearman.toArray()
corr_df_spearman = pd.DataFrame(corr_array_spearman, columns=numeric_cols, index=numeric_cols)

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Pearson correlation
sns.heatmap(corr_df_pearson, annot=False, cmap="coolwarm", square=True, ax=axes[0], 
            vmin=-1, vmax=1, center=0)
axes[0].set_title("Pearson Correlation Heatmap\n(NaN values filled w/ zero)", fontsize=12, fontweight='bold')

# Spearman correlation
sns.heatmap(corr_df_spearman, annot=False, cmap="coolwarm", square=True, ax=axes[1],
            vmin=-1, vmax=1, center=0)
axes[1].set_title("Spearman Correlation Heatmap\n(NaN values filled w/ zero)", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Optional: Compare correlations for DEP_DEL15 specifically
if 'DEP_DEL15' in numeric_cols:
    print("\nCorrelation with DEP_DEL15 (Departure Delay 15+ min):")
    print(f"{'Feature':<25} {'Pearson':<12} {'Spearman':<12} {'Difference':<12}")
    print("-" * 65)
    
    pearson_del15 = corr_df_pearson['DEP_DEL15'].sort_values(ascending=False)
    spearman_del15 = corr_df_spearman['DEP_DEL15']
    
    for feature in pearson_del15.head(15).index:
        if feature != 'DEP_DEL15':
            p_val = pearson_del15[feature]
            s_val = spearman_del15[feature]
            diff = abs(p_val - s_val)
            print(f"{feature:<25} {p_val:>10.4f}  {s_val:>10.4f}  {diff:>10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Key Insights from Pearson and Spearman Correlation Analysis**
# MAGIC
# MAGIC **1. Strong Linear Relationships (Similar Pearson & Spearman)**
# MAGIC
# MAGIC ARR_DEL15 (0.72): Arrival delays are highly correlated with departure delays - makes intuitive sense
# MAGIC DEP_DELAY (0.64 Pearson, 0.69 Spearman): Raw delay minutes show strong correlation
# MAGIC DEP_TIME & CRS_DEP_TIME: Time-of-day features show consistent correlation across both methods
# MAGIC
# MAGIC **2. Non-Linear Relationships Detected (Large Pearson-Spearman Gaps)**
# MAGIC These features have monotonic but non-linear relationships with delays:
# MAGIC
# MAGIC CARRIER_DELAY: Pearson 0.33 → Spearman 0.60 (Δ 0.28)
# MAGIC
# MAGIC Insight: Carrier delays have a strong monotonic relationship but it's not linear - suggests threshold effects or categorical patterns
# MAGIC
# MAGIC
# MAGIC LATE_AIRCRAFT_DELAY: Pearson 0.44 → Spearman 0.64 (Δ 0.21)
# MAGIC
# MAGIC Insight: Previous flight delays create cascading effects that compound non-linearly
# MAGIC
# MAGIC
# MAGIC DEP_DELAY_NEW: Pearson 0.62 → Spearman 0.79 (Δ 0.16)
# MAGIC
# MAGIC Insight: This feature (likely max(0, DEP_DELAY)) has non-linear behavior, possibly due to zero-truncation
# MAGIC
# MAGIC
# MAGIC ARR_DELAY_NEW: Pearson 0.57 → Spearman 0.69 (Δ 0.13)
# MAGIC
# MAGIC Similar pattern to DEP_DELAY_NEW
# MAGIC
# MAGIC
# MAGIC NAS_DELAY: Pearson 0.22 → Spearman 0.36 (Δ 0.13)
# MAGIC
# MAGIC Insight: Air traffic control delays may have threshold effects
# MAGIC
# MAGIC
# MAGIC 3. Correlation Insight for Modeling Considerations
# MAGIC
# MAGIC Tree-based models will likely perform better than linear models since they can capture the non-linear relationships in CARRIER_DELAY, LATE_AIRCRAFT_DELAY, and NAS_DELAY
# MAGIC Consider feature engineering: Create binned/categorical versions of delay cause features to capture threshold effects
# MAGIC The "_NEW" features show stronger Spearman correlation, suggesting transformation or bucketing could improve linear model performance
# MAGIC
# MAGIC 4. Weather Correlation Still Weak
# MAGIC
# MAGIC WEATHER_DELAY: Pearson 0.12 → Spearman 0.20
# MAGIC While Spearman shows slight improvement, weather remains weakly correlated, confirming our earlier observation that better weather feature engineering is needed (seasonal patterns, location-specific weather impacts, etc.)
# MAGIC
# MAGIC 5. Seasonal Data Limitation - will further explore 1-year, 3-year, and 5-year data in next phase
# MAGIC
# MAGIC The 3-month Jan-Mar window likely misses important seasonal weather patterns
# MAGIC A full-year analysis would better capture weather-delay relationships and seasonal effects

# COMMAND ----------

# Using altair as rendering so you can see tooltips. If getting black background and hard to see, toggle light mode in top right of panel
corr_long = (
    corr_df.reset_index()
           .melt(id_vars=corr_df.index.name or 'index', var_name='col', value_name='corr')
           .rename(columns={corr_df.index.name or 'index': 'row'})
)
x_order = list(corr_df.columns)
y_order = list(corr_df.index)

chart = (
    alt.Chart(corr_long)
    .mark_rect()
    .encode(
        x=alt.X('col:N', sort=x_order, title=None),
        y=alt.Y('row:N', sort=y_order, title=None),
        color=alt.Color('corr:Q', scale=alt.Scale(domain=(-1, 1), scheme='redblue')),
        tooltip=[
            alt.Tooltip('row:N', title='Row'),
            alt.Tooltip('col:N', title='Column'),
            alt.Tooltip('corr:Q', title='Correlation', format='.3f')
        ]
    ).configure_view(
    fill="white"
    )
    .properties(width={'step': 18}, height={'step': 18}, title='Correlation Heatmap for numeric values')
)

chart

# COMMAND ----------

#Looking at distribution of delayed flights based on few categorical variables
# **Key insights:**
# * There are noticeable differences in delays (15+ minutes) by airline and origin/arrival airport/states. 
# * However, these stats may change as time passes (+ we're not including seasonal changes), so if we are to use these as features then we need to use with care then we need to be aware of trend changes.

cat_cols = ['OP_UNIQUE_CARRIER','ORIGIN','ORIGIN_STATE_ABR','DEST','DEST_STATE_ABR']

fig, axes = plt.subplots((len(cat_cols) // 2 + len(cat_cols) % 2), 2, figsize=(12, 4 * (len(cat_cols) // 2 + len(cat_cols) % 2)))

axes = axes.flatten()

for i, col in enumerate(cat_cols):
    df = spark.sql(f"""
                   SELECT DEP_DEL15, {col}, count(*) count 
                   from otpw_3m_eda WHERE DEP_DEL15 IS NOT NULL GROUP BY DEP_DEL15, {col}
                   """).toPandas()
    group_total = df.groupby(col)['count'].sum().reset_index().rename(columns={'count':'group_count'})
    df = df.merge(group_total, on=col, how='left')
    
    #If there are more than 40 groups, truncate to top and bottom 20
    if (len(group_total) > 40):
        top = df.loc[df['DEP_DEL15']==0.0].sort_values('count', ascending=False)[col].head(20).tolist()
        bottom = df.loc[df['DEP_DEL15']==1.0].sort_values('count', ascending=False)[col].head(20).tolist()
        groups_to_include = top + bottom
        df = df.loc[df[col].isin(groups_to_include)]
        truncated = True
    else:
        truncated = False
            
    df['pct'] = df['count'] / df['group_count'] * 100.0
    df.loc[df['DEP_DEL15']==0.0,'pct'] = 100.0 # This is bit of cheating. Since seaborn can't do stacking but can do overlay, 
                                            # setting as 100 for DEP_DEL15 so it will fill the rest of the bar
    df.sort_values('pct', inplace=True)
    sns.barplot(data=df, x=col, y='pct', hue='DEP_DEL15', ax=axes[i], dodge=False)
    axes[i].set_title(f"Percentage of delayed flights by {col} {"" if not truncated else '\n(top/bottom 20 groups)'}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Percentage")
    axes[i].legend(title="Delayed")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Misc, random stuff

# COMMAND ----------

display(df_flights_3m.summary())

# COMMAND ----------

display(spark.sql("SELECT COUNT(*) FROM weather_3m"))

# COMMAND ----------

results = spark.sql("SELECT * from weather_3m").dropDuplicates().count()
display(results)

# COMMAND ----------

display(spark.sql("SELECT OP_CARRIER_FL_NUM,* from flights_3m order by FL_Date, TAIL_NUM limit 100"))

# COMMAND ----------

display(spark.sql("SELECT * from weather_3m limit 100"))

# COMMAND ----------

display(spark.sql("SELECT * from otpw_3m order by FL_Date, TAIL_NUM limit 100"))

# COMMAND ----------

display(spark.sql("SELECT * from stations limit 100"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC

# COMMAND ----------

print (df_otpw_3m.dtypes)

# COMMAND ----------

missing_pct_df = None
df_name = "optw"
df = df_otpw_3m
missing_pct_df = get_missing_pct(df, df_name)
missing_pct_df['source'] = df_name
missing_pct_df

# COMMAND ----------

display(spark.sql("SELECT * from weather_3m limit 50"))