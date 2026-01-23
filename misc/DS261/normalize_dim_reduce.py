# Databricks notebook source
# MAGIC %md 
# MAGIC # Normalization and dimentionality deduction
# MAGIC This notebook does the following: 
# MAGIC * Normalization of values
# MAGIC * Dimentionality deduction via PCA or Lasso
# MAGIC
# MAGIC And creates the following files:
# MAGIC * normalized
# MAGIC * dim_reduced

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparation

# COMMAND ----------

#Config
DATE_RANGE = '3M' # Pick from 3M, 1Y or 5Y. This will also be the output directory
OVERWRITE = True # If picked true, then the script will override the existing data

SPARK_DIR = 'dbfs:/student-groups/Group_01_01/'
INPUT_NAME = 'features_generated'
OUTPUT_NAME_1 = 'normalized'
OUTPUT_NAME_2 = 'dim_reduced'

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
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array

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



# COMMAND ----------

#Import data 
df = load_spark_df(INPUT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalize/standardize

# COMMAND ----------

#As a simple first step, we standardize all double fields


#Manually adding few fields to not normalize
dont_normalize = {"DEP_DEL15",'ARR_DEL15'}

# Get all double field names
double_cols = [f.name for f in df.schema.fields if f.dataType.typeName() == "double"]
double_cols = list(set(double_cols) - set(dont_normalize))

# Combine all doubles into one feature vector
assembler = VectorAssembler(inputCols=double_cols, outputCol="features_vec")

# Apply standardization
scaler = StandardScaler(
    inputCol="features_vec",
    outputCol="scaled_features",
    withMean=True, 
    withStd=True 
)

# Build pipeline
pipeline = Pipeline(stages=[assembler, scaler])

# Fit and transform
model = pipeline.fit(df)
df_scaled = model.transform(df)

# Split scaled_features back into individual standardized columns
df_scaled = df_scaled.withColumn("scaled_array", vector_to_array("scaled_features"))


for i, c in enumerate(double_cols):
    df_scaled = df_scaled.withColumn(c, col("scaled_array")[i])

# (Optional) drop intermediate columns if not needed
df_scaled = df_scaled.drop("features_vec", "scaled_features", "scaled_array")

df = df_scaled
display(df.limit(10))



# COMMAND ----------

by_year = True if DATE_RANGE in ['5Y','10Y'] else False
save_spark_df(df, OUTPUT_NAME_1, by_year = by_year)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimentionality deduction

# COMMAND ----------

#df = load_spark_df(OUTPUT_NAME_1)

# COMMAND ----------

# To be filled!

# COMMAND ----------

by_year = True if DATE_RANGE in ['5Y','10Y'] else False
save_spark_df(df, OUTPUT_NAME_2, by_year = by_year)



# COMMAND ----------

df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")
df_flights= df_flights.filter(f"YEAR IN (2015, 2016, 2017, 2018,2019)")
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_1y/")

display(df_weather.count())
