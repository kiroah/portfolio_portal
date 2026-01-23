# Databricks notebook source
# MAGIC %md 
# MAGIC # Generate Features
# MAGIC This notebook does the following: 
# MAGIC * Generate derived features
# MAGIC
# MAGIC It uses "cleaned_imputed" parquet file(s) as input
# MAGIC
# MAGIC And creates the following files:
# MAGIC * features_generated

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparation

# COMMAND ----------

#Config
DATE_RANGE = '10Y' # Pick from 3M, 1Y or 5Y. This will also be the output directory
OVERWRITE = True # If picked true, then the script will override the existing data

SPARK_DIR = 'dbfs:/student-groups/Group_01_01/'
INPUT_NAME = 'cleaned_imputed'
OUTPUT_NAME = 'features_generated'


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

#Import data 
df = load_spark_df(INPUT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate features
# MAGIC TO BE FILLED!

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Export

# COMMAND ----------

by_year = True if DATE_RANGE in ['5Y','10Y'] else False
save_spark_df(df, OUTPUT_NAME, by_year = by_year)

