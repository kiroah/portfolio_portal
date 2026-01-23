# Databricks notebook source
#!du -h "/dbfs/student-groups/Group_01_01/"
!mkdir "/dbfs/student-groups/Group_01_01/new_data/weather"
!mkdir "/dbfs/student-groups/Group_01_01/new_data/flights"

# COMMAND ----------

!ls  -all "/dbfs/student-groups/Group_01_01/new_data/weather" 

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, to_timestamp


# COMMAND ----------

# MAGIC %md
# MAGIC ## Flights

# COMMAND ----------

# Mapping FROM new flights columns → TO original flights columns
flights_rename_map = {
    "Quarter": "QUARTER",
    "Month": "MONTH",
    "DayofMonth": "DAY_OF_MONTH",
    "DayOfWeek": "DAY_OF_WEEK",
    "FlightDate": "FL_DATE",

    "Reporting_Airline": "OP_UNIQUE_CARRIER",          
    "DOT_ID_Reporting_Airline": "OP_CARRIER_AIRLINE_ID",
    "IATA_CODE_Reporting_Airline": "OP_CARRIER",        

    "Tail_Number": "TAIL_NUM",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",

    "OriginAirportID": "ORIGIN_AIRPORT_ID",
    "OriginAirportSeqID": "ORIGIN_AIRPORT_SEQ_ID",
    "OriginCityMarketID": "ORIGIN_CITY_MARKET_ID",
    "Origin": "ORIGIN",
    "OriginCityName": "ORIGIN_CITY_NAME",
    "OriginState": "ORIGIN_STATE_ABR",
    "OriginStateFips": "ORIGIN_STATE_FIPS",
    "OriginStateName": "ORIGIN_STATE_NM",
    "OriginWac": "ORIGIN_WAC",

    "DestAirportID": "DEST_AIRPORT_ID",
    "DestAirportSeqID": "DEST_AIRPORT_SEQ_ID",
    "DestCityMarketID": "DEST_CITY_MARKET_ID",
    "Dest": "DEST",
    "DestCityName": "DEST_CITY_NAME",
    "DestState": "DEST_STATE_ABR",
    "DestStateFips": "DEST_STATE_FIPS",
    "DestStateName": "DEST_STATE_NM",
    "DestWac": "DEST_WAC",

    "CRSDepTime": "CRS_DEP_TIME",
    "DepTime": "DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "DepDelayMinutes": "DEP_DELAY_NEW",
    "DepDel15": "DEP_DEL15",
    "DepartureDelayGroups": "DEP_DELAY_GROUP",
    "DepTimeBlk": "DEP_TIME_BLK",
    "TaxiOut": "TAXI_OUT",
    "WheelsOff": "WHEELS_OFF",
    "WheelsOn": "WHEELS_ON",
    "TaxiIn": "TAXI_IN",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrTime": "ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "ArrDelayMinutes": "ARR_DELAY_NEW",
    "ArrDel15": "ARR_DEL15",
    "ArrivalDelayGroups": "ARR_DELAY_GROUP",
    "ArrTimeBlk": "ARR_TIME_BLK",

    "Cancelled": "CANCELLED",
    "CancellationCode": "CANCELLATION_CODE",
    "Diverted": "DIVERTED",

    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "ActualElapsedTime": "ACTUAL_ELAPSED_TIME",
    "AirTime": "AIR_TIME",
    "Flights": "FLIGHTS",
    "Distance": "DISTANCE",
    "DistanceGroup": "DISTANCE_GROUP",

    "CarrierDelay": "CARRIER_DELAY",
    "WeatherDelay": "WEATHER_DELAY",
    "NASDelay": "NAS_DELAY",
    "SecurityDelay": "SECURITY_DELAY",
    "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",

    "FirstDepTime": "FIRST_DEP_TIME",
    "TotalAddGTime": "TOTAL_ADD_GTIME",
    "LongestAddGTime": "LONGEST_ADD_GTIME",

    "DivAirportLandings": "DIV_AIRPORT_LANDINGS",
    "DivReachedDest": "DIV_REACHED_DEST",
    "DivActualElapsedTime": "DIV_ACTUAL_ELAPSED_TIME",
    "DivArrDelay": "DIV_ARR_DELAY",
    "DivDistance": "DIV_DISTANCE",

    "Div1Airport": "DIV1_AIRPORT",
    "Div1AirportID": "DIV1_AIRPORT_ID",
    "Div1AirportSeqID": "DIV1_AIRPORT_SEQ_ID",
    "Div1WheelsOn": "DIV1_WHEELS_ON",
    "Div1TotalGTime": "DIV1_TOTAL_GTIME",
    "Div1LongestGTime": "DIV1_LONGEST_GTIME",
    "Div1WheelsOff": "DIV1_WHEELS_OFF",
    "Div1TailNum": "DIV1_TAIL_NUM",

    "Div2Airport": "DIV2_AIRPORT",
    "Div2AirportID": "DIV2_AIRPORT_ID",
    "Div2AirportSeqID": "DIV2_AIRPORT_SEQ_ID",
    "Div2WheelsOn": "DIV2_WHEELS_ON",
    "Div2TotalGTime": "DIV2_TOTAL_GTIME",
    "Div2LongestGTime": "DIV2_LONGEST_GTIME",
    "Div2WheelsOff": "DIV2_WHEELS_OFF",
    "Div2TailNum": "DIV2_TAIL_NUM",

    "Div3Airport": "DIV3_AIRPORT",
    "Div3AirportID": "DIV3_AIRPORT_ID",
    "Div3AirportSeqID": "DIV3_AIRPORT_SEQ_ID",
    "Div3WheelsOn": "DIV3_WHEELS_ON",
    "Div3TotalGTime": "DIV3_TOTAL_GTIME",
    "Div3LongestGTime": "DIV3_LONGEST_GTIME",
    "Div3WheelsOff": "DIV3_WHEELS_OFF",
    "Div3TailNum": "DIV3_TAIL_NUM",

    "Div4Airport": "DIV4_AIRPORT",
    "Div4AirportID": "DIV4_AIRPORT_ID",
    "Div4AirportSeqID": "DIV4_AIRPORT_SEQ_ID",
    "Div4WheelsOn": "DIV4_WHEELS_ON",
    "Div4TotalGTime": "DIV4_TOTAL_GTIME",
    "Div4LongestGTime": "DIV4_LONGEST_GTIME",
    "Div4WheelsOff": "DIV4_WHEELS_OFF",
    "Div4TailNum": "DIV4_TAIL_NUM",

    "Div5Airport": "DIV5_AIRPORT",
    "Div5AirportID": "DIV5_AIRPORT_ID",
    "Div5AirportSeqID": "DIV5_AIRPORT_SEQ_ID",
    "Div5WheelsOn": "DIV5_WHEELS_ON",
    "Div5TotalGTime": "DIV5_TOTAL_GTIME",
    "Div5LongestGTime": "DIV5_LONGEST_GTIME",
    "Div5WheelsOff": "DIV5_WHEELS_OFF",
    "Div5TailNum": "DIV5_TAIL_NUM",

}

# COMMAND ----------

# MAGIC %%bash
# MAGIC BASE_DIR="/dbfs/student-groups/Group_01_01/new_data/flights"
# MAGIC BASE_URL="https://transtats.bts.gov/PREZIP"
# MAGIC
# MAGIC mkdir -p "$BASE_DIR"
# MAGIC
# MAGIC for YEAR in {2021..2024}; do
# MAGIC   for MONTH in {1..12}; do
# MAGIC     FILE="On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${YEAR}_${MONTH}.zip"
# MAGIC     DEST="${BASE_DIR}/${FILE}"
# MAGIC     URL="${BASE_URL}/${FILE}"
# MAGIC
# MAGIC     if [ -f "$DEST" ]; then
# MAGIC       echo "✅ Already downloaded: $FILE"
# MAGIC       continue
# MAGIC     fi
# MAGIC
# MAGIC     echo "⬇️ Downloading $FILE"
# MAGIC     wget -q -O "$DEST" "$URL"
# MAGIC
# MAGIC     if [ $? -ne 0 ]; then
# MAGIC       echo "❌ Failed to download: $FILE"
# MAGIC       rm -f "$DEST"
# MAGIC     else
# MAGIC       echo "✅ Saved: $DEST"
# MAGIC     fi
# MAGIC   done
# MAGIC done

# COMMAND ----------

# MAGIC %%bash
# MAGIC BASE_DIR="/dbfs/student-groups/Group_01_01/new_data/flights"
# MAGIC
# MAGIC cd "$BASE_DIR" || exit 1
# MAGIC
# MAGIC echo "📦 Unzipping all ZIP files..."
# MAGIC
# MAGIC for ZIP in *.zip; do
# MAGIC   echo "➡️ Processing $ZIP"
# MAGIC
# MAGIC   # Extract
# MAGIC   unzip -o "$ZIP" >/dev/null
# MAGIC
# MAGIC   if [ $? -ne 0 ]; then
# MAGIC     echo "❌ Failed to unzip: $ZIP"
# MAGIC     continue
# MAGIC   fi
# MAGIC
# MAGIC   # Verify at least one CSV was created from this ZIP
# MAGIC   CSV_COUNT=$(unzip -l "$ZIP" | grep -i ".csv" | wc -l)
# MAGIC
# MAGIC   if [ "$CSV_COUNT" -gt 0 ]; then
# MAGIC     rm -f "$ZIP"
# MAGIC     echo "✅ Extracted and removed: $ZIP"
# MAGIC   else
# MAGIC     echo "⚠️ No CSV found inside: $ZIP (ZIP retained)"
# MAGIC   fi
# MAGIC
# MAGIC done
# MAGIC
# MAGIC echo "✅ Unzip + cleanup complete."

# COMMAND ----------



spark = SparkSession.builder.getOrCreate()

# Spark uses the "dbfs:/" prefix, not "/dbfs/..."
csv_path = "dbfs:/student-groups/Group_01_01/new_data/flights/*.csv"
parquet_path = "dbfs:/student-groups/Group_01_01/new_data/flights.parquet"

# Read all CSVs. header=True ensures the first line in *each* file is treated as a header,
# so extra headers from each yearly file are not included as data rows.
df = (
    spark.read
         .option("header", "true")
         .option("inferSchema", "true")   # or define schema explicitly if you prefer
         .csv(csv_path)
)

#Update column names
df = (
    df
        .drop("_c109")
        .select(
            *[F.col(old).alias(new) for old, new in flights_rename_map.items()],
            "YEAR"  # keep YEAR as-is
        )
)

# Optional: sanity check
print("Row count:", df.count())
df.printSchema()

# Write partitioned by YEAR
(
    df.write
      .mode("overwrite")
      .partitionBy("YEAR")
      .parquet(parquet_path)
)

print(f"Parquet written to: {parquet_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather

# COMMAND ----------

# MAGIC %sh
# MAGIC cd "/dbfs/student-groups/Group_01_01/new_data/weather"
# MAGIC BASE_URL="https://www.ncei.noaa.gov/data/local-climatological-data/archive"
# MAGIC for YEAR in 2021 2022 2023 2024; do
# MAGIC   wget -c "${BASE_URL}/${YEAR}.tar.gz"
# MAGIC done
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e 
# MAGIC cd "/dbfs/student-groups/Group_01_01/new_data/weather"

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e
# MAGIC
# MAGIC BASE_DIR="/dbfs/student-groups/Group_01_01/new_data/weather"
# MAGIC
# MAGIC cd "$BASE_DIR"
# MAGIC
# MAGIC
# MAGIC # Extract each tar.gz into its own subfolder (2021/, 2022/, 2023/, 2024/)
# MAGIC for YEAR in 2021 2022 2023 2024; do
# MAGIC   TAR_FILE="${YEAR}.tar.gz"
# MAGIC   echo "Extracting $TAR_FILE into $YEAR/ ..."
# MAGIC   mkdir -p "$YEAR"
# MAGIC   tar -xzf "$TAR_FILE" -C "$YEAR"
# MAGIC done
# MAGIC
# MAGIC echo "Done. Directory structure:"
# MAGIC find "$BASE_DIR" -maxdepth 2 -type f | head
# MAGIC

# COMMAND ----------


spark = SparkSession.builder.getOrCreate()

csv_path = "dbfs:/student-groups/Group_01_01/new_data/weather/*/*.csv"
parquet_path = "dbfs:/student-groups/Group_01_01/new_data/weather.parquet"

# Read CSVs (each file has its own header)
df = (
    spark.read
         .option("header", "true")
         .option("inferSchema", "true")
         .csv(csv_path)
)

# Convert DATE string -> timestamp, then extract YEAR
df = df.withColumn(
    "DATE_TS",
    to_timestamp(col("DATE"), "yyyy-MM-dd'T'HH:mm:ss")
)

df = df.withColumn(
    "YEAR",
    year(col("DATE_TS"))
)

# Optional sanity checks
df.select("DATE", "YEAR").show(5, truncate=False)
print("Row count:", df.count())
df.printSchema()

# Write partitioned Parquet
(
    df.drop("DATE_TS")   # helper column not needed in final output
      .write
      .mode("overwrite")
      .partitionBy("YEAR")
      .parquet(parquet_path)
)

print(f"Weather parquet written to: {parquet_path}")