# Databricks notebook source
# MAGIC %md 
# MAGIC # Data joining, cleaning and imputing
# MAGIC This notebook does the following: 
# MAGIC * Join multiple data sources into one datasource
# MAGIC * Convert fields into appropriate datatype (if necessary)
# MAGIC * Remove fields deemed unnecessary or based on amount of missing values
# MAGIC * Impute missing data
# MAGIC
# MAGIC And creates the following files:
# MAGIC * joined_col_filtered
# MAGIC * cleaned_imputed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparation

# COMMAND ----------

#imports
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, to_timestamp
import pandas as pd
from io import StringIO
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

#Config
DATE_RANGE = '3M' # Pick from 3M, 1Y, 5Y or LAST5Y. This will also be the output directory
OVERWRITE = True # If picked true, then the script will override the existing data

SPARK_DIR = 'dbfs:/student-groups/Group_01_01/'
OUTPUT_NAME_1 = 'joined_col_filtered'
OUTPUT_NAME_2 = 'cleaned_imputed'


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
# MAGIC ## Load and Join

# COMMAND ----------

#Load data sources

#Note that for now we're just going to use the otpw directly. 
# if DATE_RANGE == '3M':
#     otpw_df = spark.read.option("header","true").csv(f"dbfs:/mnt/mids-w261/OTPW_3M/OTPW_3M")
# elif DATE_RANGE == '1Y':
#     otpw_df = spark.read.option("header","true").csv("dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M/OTPW_12M_2015.csv.gz")
# elif DATE_RANGE == '5Y':
#     otpw_df = spark.read.option("header","true").parquet("dbfs:/mnt/mids-w261/OTPW_60M/OTPW_60M")
# df = otpw_df


# New otpw. Later on we'll bring in the actual join logic into this notebook
df = load_spark_df("otpw_v2",date_range=DATE_RANGE)



# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean & filter

# COMMAND ----------

#Only keep columns we'll be using
fields = ['YEAR','QUARTER','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','FL_DATE','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_CITY_MARKET_ID','ORIGIN','ORIGIN_STATE_ABR','DEST_AIRPORT_ID','DEST','DEST_STATE_ABR','CRS_DEP_TIME','DEP_TIME','DEP_DELAY','DEP_DELAY_NEW','DEP_DEL15','DEP_DELAY_GROUP','DEP_TIME_BLK','CRS_ARR_TIME','ARR_TIME','ARR_DELAY','ARR_DELAY_NEW','ARR_DEL15','ARR_DELAY_GROUP','CANCELLED','CANCELLATION_CODE','DIVERTED','DISTANCE','DISTANCE_GROUP','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','STATION','DATE','HourlyAltimeterSetting','HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation','HourlyPresentWeatherType','HourlyPressureChange','HourlyPressureTendency','HourlyRelativeHumidity','HourlySkyConditions','HourlyWindGustSpeed','HourlyWindSpeed','DailyAverageDewPointTemperature','crs_depart_ts_local','crs_depart_ts_utc','station_miles','tail_idx','PREV_CRS_DEP_TIME','PREV_DEP_TIME','PREV_crs_depart_unixts_utc','PREV_DEP_DELAY','PREV_DEP_DELAY_NEW','PREV_DEP_DELAY_GROUP','PREV_ARR_DELAY_GROUP','PREV_CRS_ARR_TIME','PREV_ARR_TIME','PREV_ARR_DELAY','PREV_CANCELLED','PREV_DIVERTED','PREV_ORIGIN','PREV_DEST','PREV_OP_CARRIER_FL_NUM','PREV_DISTANCE','ARR_TIME_BLK']
df = df.select(fields)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert datatypes
# MAGIC We are currently using OTPW data, where all fields are stored as strings. We need to convert them to appropriate datatypes

# COMMAND ----------

#Convert datatypes based on the original data source

#Import data sources. Note that we're only using the 3 months data
# df_flights_sample = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/").limit(20)
# df_weather_sample = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/").limit(20)
# df_stations_sample = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/").limit(20)

# source_df_list = [df_flights_sample, df_weather_sample, df_stations_sample]

# #Iterate through each original data source to convert the datatype in otpw. Note that weather data are all strings as well, and therefore we need a different method
# for source_df in source_df_list:
#     # Create mapping of column name → target dataType
#     target_types = {f.name: f.dataType for f in source_df.schema.fields}

#     # Build select expressions
#     cols = []
#     for c in df.columns:
#         if c in target_types:
#             # Cast only if it exists in df_good
#             cols.append(F.col(c).cast(target_types[c]).alias(c))
#         else:
#             # Keep original type (column not in df_good)
#             cols.append(F.col(c))
#     df = df.select(*cols)

# COMMAND ----------

#Convert fields based on summary stats (which Spark tries to interpret any field as double if possible)

dont_dbl = {'FL_DATE','OP_UNIQUE_CARRIER'}
# First, get summary stats (e.g. mean,stdev) information into dataframe
field_summary = df.limit(1000).summary().toPandas()
field_summary = field_summary.set_index("summary").transpose().reset_index().rename(columns={"index":"field_id"})

# Next, we use this information to get list of columns that we can convert to double
convert_to_dbl = field_summary.loc[~field_summary['mean'].isna(),"field_id"].to_list()

# We also filter out columns that are set as non-string in the current otpw table
non_string_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() != "string"]

#Subtract the non_string_cols from convert_to_dbl list
convert_to_dbl = list(set(convert_to_dbl) - set(non_string_cols) - dont_dbl)


#Actual conversion
for col in convert_to_dbl:
    df = df.withColumn(col, df[col].cast("Double"))



# COMMAND ----------

# For dates, we look at the "min" field to see if the field includes regex format of \d\d\d\d-\d\d-\d\d
convert_to_datetime = field_summary.loc[field_summary["min"].fillna('').str.contains(r"\d\d\d\d-\d\d-\d\d", regex=True),"field_id"].to_list()
for col in convert_to_datetime:
     df = df.withColumn(col, to_timestamp(df[col]))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

by_year = True if DATE_RANGE in ['5Y','10Y','LAST5Y'] else False
save_spark_df(df, OUTPUT_NAME_1, by_year = by_year)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Impute

# COMMAND ----------

#df = load_spark_df(OUTPUT_NAME_1)


# COMMAND ----------

# #Blindly filling all NaNs (nulls) with zeroes
# df = df.fillna(0)

# by_year = True if DATE_RANGE in ['5Y','10Y'] else False
# save_spark_df(df, OUTPUT_NAME_2, by_year = by_year)


# COMMAND ----------

