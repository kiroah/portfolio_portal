# Databricks notebook source
# MAGIC %md
# MAGIC # 🛫 Optimizing Airline Operations: Predicting Flight Delays via Scalable ML and Neural Network Architectures
# MAGIC | Hiro Naito | Hong Hu | Micah Collins | Min Yang |
# MAGIC |----------|----------|----------|----------|
# MAGIC | <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/hiro_naito.jpeg" alt="Hiro Naito" width="120"/> | <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/hong_hu.jpeg" alt="Hong Hu" width="120"/> | <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/micah_collins.png" alt="Micah Collins" width="120"/> | <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/min_yang.png" alt="Min Yang" width="120"/> |
# MAGIC | hiro.naito@berkeley.edu | honghu@berkeley.edu | micah_collins@berkeley.edu | yangmindc@berkeley.edu |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2 Report
# MAGIC
# MAGIC **Team 1-1**
# MAGIC
# MAGIC November 23, 2025
# MAGIC
# MAGIC <a href="https://docs.google.com/presentation/d/1lCPovjnTt55cMn3u8OcUd_ydVx_0LIoNvMXYutUeBr0/edit?slide=id.g3a9090b001f_1_0#slide=id.g3a9090b001f_1_0">In-class Presentation Link</a> 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase Leader Plan:
# MAGIC
# MAGIC | **Week #** | **Date** | **Leader** | **Deliverable** | **Description** | **Due Date** |
# MAGIC |---------------|-------------|-------------|---------------|--------------|-------------|
# MAGIC | **Week 10** | 10/27 – 11/2 | Hong Hu | **Phase 1 Report** | Project Plan, Data Preparation, Preliminary EDA, ML Algorithms, Model Performance Metric | **11/2/2025 (Sun) 11:59 PM PST** |
# MAGIC | **Week 11** | 11/3 – 11/17 *Including Fall Break* | Hiro Naito | **In-class Presentation** | Phase II EDA, Baseline Pipeline on all data, Scalability, Efficiency, Distributed/parallel Training, and Scoring Pipeline, Feature Engineering | **11/17/2025 (Mon) In-class** |
# MAGIC | **Week 12** | 11/18 – 11/23 | Hong Hu | **Phase 2 Report** | Baseline Model, Additional Feature Engineering, Hyperparameter Tuning | **11/23/2025 (Sun) 11:59 PM PST** |
# MAGIC | **Week 13** | 12/1 – 12/8 | Micah Collins | **In-class Final Presentation** | Advanced Model Architectures, Loss Functions, Select Optimal Algorithm, Fine-tuning | **12/8/2025 (Mon) In-class** |
# MAGIC | **Week 14** | 12/9 – 12/13 | Min Yang | **Phase 3 Final Report** | Final Report Integration and Write-up | **12/13/2025 (Sat) 11:59 PM PST** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Notes
# MAGIC
# MAGIC - Each phase leader oversees coordination, integration of team work, and on-time delivery.  
# MAGIC - Weekly syncs will ensure alignment on data processing, modeling progress, and documentation updates.  
# MAGIC - Reports will be exported in HTML format and validated for readability in incognito mode per W261 submission guidelines.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit Assignment plan:
# MAGIC
# MAGIC | **Name** | **Phase 1** | **Phase 2 (Planned)** | **Phase 3 (Planned)** |
# MAGIC |-----------------|--------------|------------------|------------------|
# MAGIC | **Hiro Naito** | Data Cleaning, Data Preparation, Data Dictionary creation, Preliminary EDA & Visualizations. (9 hours) | **Phase 2.1 Leader** - Generated 3m, 1y and 5y dataset with custom join, dubbed otpw_v2. Structured data catalog & checkpoints for downstream pipeline. Basic imputation, column filtering and cleanup. Create base template for the phase 2 presentation, contributed to the content for presentation and report. 23 hours | Final report integration, deepdive of results, documentation, and meetings; contribute to Phase 3 report.|
# MAGIC | **Hong Hu** | **Phase 1 Leader** — developed the project plan using data and machine learning pipeline block diagrams in HTML; led overall coordination; handled deliverable submission. (10 hours) | **Phase 2.2 Leader** - Build and fine-tune baseline models using grid search; develop data pipelines with checkpoints; implement ML pipelines; benchmark models on the original OTPW dataset for the 3M, 1Y, and 5Y prediction tasks; handle documentation and meetings; contribute to the Phase 2 report. (30 hours) | Model training and inference; Lead feature engineering on larger datasets; documentation and meetings; contribute to Phase 3 report. |
# MAGIC | **Min Yang** | Preliminary EDA and visualization; authored project abstract; created Phase Leader Plan and Credit Assignment Plan tables. (9 hours)| Perform Phase II EDA on custom joined OTPW-V2 (12-month datasets); contributed to the in-class presentation and Phase 2 report. (30 hours) | **Phase 3 Leader** - Develop Machine Learning Algorithms and Metrics section; build NN-based ML pipeline (MLP and Residual MLP); lead final synthesis and Phase 3 report. |
# MAGIC | **Micah Collins** | Defined ML algorithms and selected evaluation metrics; contributed to model planning. (7.5 hours) | Conducted Feature Engineering and leakage detection on OTPW_V2. Performed feature selection/importance procedures on OTPW_V2. Created modeling pipeline and Time-Series Cross validation pipeline for OTPW_V2. Created model evaluation/selection pipeline for OTPW_V2. Did results/analysis interpretation on OTPW_V2 models (28 hours) | **Phase 3 Leader** - Develop advanced models (NN architectures, ensemble comparisons); refine ML pipeline and contribute to Machine Learning Algorithms and Metrics section; participate in Phase 3 report. |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### SMART Goal Alignment
# MAGIC Each task follows **S.M.A.R.T.** criteria:
# MAGIC - **Specific:** Clear and well-defined roles by phase.  
# MAGIC - **Measurable:** Deliverables include notebook sections, reports, and visual outputs.  
# MAGIC - **Achievable:** Each member’s scope matches expertise and schedule.  
# MAGIC - **Relevant:** Tasks directly align with W261 learning goals (EDA, pipelines, ML modeling, and scalability).  
# MAGIC - **Time-bound:** Due dates and ownership align with the weekly phase leader plan.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *This table will be updated each phase to reflect completed tasks and evolving responsibilities to ensure transparency and credit accuracy.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Project Abstract
# MAGIC
# MAGIC Flight delays remain a major challenge for the U.S. aviation industry, causing operational inefficiencies, financial losses, and passenger dissatisfaction. Accurate flight delay prediction enables airlines and airports to optimize scheduling, allocate resources effectively, and communicate proactively with travelers, improving overall efficiency and customer experience. The objective of this project is to predict U.S. domestic flight departure delays - defined as delays of 15 minutes or more - using large-scale flight and weather data. The datasets combine the U.S. Department of Transportation TranStats On-Time Performance (OTP) records and NOAA Global Hourly Weather data from 2015–2021, joined by airport code and timestamp. 
# MAGIC
# MAGIC Given the significant class imbalance, we selected a Majority Class Baseline (predicting all flights to be non-delayed) to establish a null accuracy benchmark. While this baseline achieved a high accuracy of 82.7%, it resulted in a Recall of 0.0 for delays, proving that accuracy is a deceptive metric for this problem and necessitating a more sophisticated approach. We developed a scalable PySpark pipeline comparing Logistic Regression against Gradient-Boosted Trees (GBT), utilizing an Expanding Window Time-Series Cross-Validation strategy to ensure robust and stable evaluation without data leakage.
# MAGIC
# MAGIC Key features we engineered included `STORM_INDEX` which was composed of weather metrics, and `SCHEDULED_BUFFER` which measured scheduled turnaround time between previous and current flights without data leakage, alongside smoothed target encoding for the high-cardinality airport data. We additionally undersampled the training set non-delay examples (using 75% of them) and weighted the dealy examples by 2x in order to allow for the model to prioritize delays in the imbalanced dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Project Description
# MAGIC
# MAGIC
# MAGIC This Phase 2 report outlines our development of a machine learning system to predict flight departure delays at scale. The document is structured to guide readers through our complete analytical workflow, from data preparation to baseline model evaluation.
# MAGIC
# MAGIC Our report begins by documenting the **custom data integration process** that produced OTPW_V2, explaining why we deviated from the provided dataset and detailing our methodology for joining flight operations, weather observations, and aircraft history data. 
# MAGIC
# MAGIC The **Exploratory Data Analysis** follows with comprehensive findings on data quality, temporal patterns, carrier performance, and weather impacts—revealing the cascading nature of delays and strong time-of-day effects that inform our modeling approach. **Feature Engineering** describes our transformation strategy, including temporal grouping, previous flight indicators with leakage mitigation, weather features composites, and categorical encoding methods for airports and carriers.
# MAGIC
# MAGIC The **Modeling Pipeline** section presents our distributed computing implementation using Spark MLlib, documenting baseline models (Logistic Regression, Random Forest), time-series cross-validation methodology, and hyperparameter tuning via grid search.
# MAGIC
# MAGIC **Results and Discussion** compares model performance across train/validation/test splits, analyzes feature importance, and examines the impact of our custom features on predictive accuracy. We also present preliminary findings on data leakage validation through turnaround-time stratification.
# MAGIC
# MAGIC The **Conclusion** synthesizes key insights, discusses operational implications, and outlines our roadmap for Phase 3, including scaling to five years of data and implementing advanced feature engineering.
# MAGIC
# MAGIC Appendices provide extended technical specifications, custom join implementation details, and code notebook references to ensure reproducibility. This report balances technical rigor with practical applicability, aiming to deliver insights that inform real-world airline operations and delay management strategies.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Tasks
# MAGIC
# MAGIC This report, as well as the entire project, has been conducted as a series of iterative tasks. The iterative tasks are explained below. <br><br>
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/task.png">
# MAGIC
# MAGIC | **Task** | Description |
# MAGIC |---------|-------------|
# MAGIC | **Integrate & clean data** | Combine the identified datasets and resolve issues such as missing values, duplicates, type inconsistencies, and formatting problems to produce a reliable dataset. |
# MAGIC | **EDA** | Explore the data to understand distributions, correlations, potential leakage, and target relationships that guide later feature and modeling decisions. |
# MAGIC | **Feature engineering** | Create, transform, and select features based on EDA insights to improve model signal and remove unnecessary inputs. |
# MAGIC | **Data sampling & splitting** | Prepare the dataset for fair model evaluation by splitting into train/validation/test sets, applying time-based CV, and handling class imbalance. |
# MAGIC | **Model building** | Train and tune multiple model candidates—from baselines to more advanced algorithms—while tracking configurations and performance. |
# MAGIC | **Evaluation** | Assess models using defined metrics, confusion matrices, and error analyses to compare against baselines and identify improvement areas. |
# MAGIC
# MAGIC
# MAGIC __3 months, 1 year and 5 years data__
# MAGIC We work with 3-month, 1-year, and 5-year datasets, and we iterate through these sizes progressively. For example, we perform the data join on the 3-month sample first, then move to the 1-year dataset, and finally the 5-year version. This approach helps us catch issues early and avoid wasting compute on larger datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Description

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data sources
# MAGIC
# MAGIC __Primary data sources__ <br>
# MAGIC The following are considered the main data sources for training and prediction: 
# MAGIC
# MAGIC | **Data source** | **Description** | **# of cols** | **Size** 3M | **Size** 1Y | **Size** 5Y | **Rows** 3M | **Rows** 1Y | **Rows** 5Y |
# MAGIC |------------------|------------------|----------------|-------------------------|-------------------------|-------------------------|--------------|--------------|--------------|
# MAGIC | Flights | Provided by the Bureau of Transportation Statistics (BTS), this dataset contains comprehensive historical flight performance records for U.S. domestic flights from 2015-2019. Each record represents an individual flight, including scheduled and actual departure/arrival times, delay causes (carrier, weather, NAS, security, late aircraft), flight identifiers (airline, tail number, origin, destination), and cancellation information. This dataset forms the foundation for understanding flight delay patterns across airlines, routes, and temporal dimensions. | 109 | 96MB | 595MB | 2.4GB | 1,403,471<br>AFTER DEDUP<br>(before: 2,806,942) | 14,844,074<br>AFTER DEDUP<br>(before: 7,422,037) | 31,746,841<br>AFTER DEDUP<br>before: 63,493,682 |
# MAGIC | Weather | Sourced from NOAA's Integrated Surface Database (ISD), this dataset provides historical weather observations from stations across the U.S., including temperature, humidity, precipitation, wind speed, visibility, and adverse weather events (storms, fog, snow). Weather observations are recorded at hourly intervals, making them critical for capturing environmental conditions that influence flight operations. | 124 | 1.1GB | 4.8GB | 23.6GB | 30,528,602 | 131,937,550 | 639,726,637 |
# MAGIC
# MAGIC __Secondary data sources__ <br>
# MAGIC The following will be utilized as a side source to help join the data. 
# MAGIC | **Data source**   | **Description** | **# of cols** | **Data size** | **# of rows** |
# MAGIC |--------------------|------------------|----------------|----------------|----------------|
# MAGIC | airports.csv       | Sourced from the Airport Codes Dataset (DataHub), this dataset contains comprehensive airport information including IATA and ICAO codes, airport names, cities, states, countries, and geographic coordinates. This enables accurate mapping of airport codes to their geographic positions and facilitates the integration of weather data with flight operations data. | 19 | 12.4MB | 83,798 |
# MAGIC | isd_history.csv    | This supplementary dataset provides geospatial metadata for weather stations, including station IDs (USAF+WBAN codes), latitude, longitude, elevation, and station names, sourced from National Centers of Environment Information (NCEI). It enables precise mapping between weather observations and geographic locations, which is essential for associating weather conditions with specific airports. | 11 | 2.78MB | 29,661 |
# MAGIC
# MAGIC <br><br><br>
# MAGIC __Sources not used__ 
# MAGIC The following sources have not been used for the project's final outcome:
# MAGIC
# MAGIC * OTPW - The OTPW is a pre-joined data of flight and weather provided by the class. However we have opted to create our custom joined data and therefore we have only used at the beginning of the project. The reason for custom join has been explained in the "Joining the data" section
# MAGIC * Airport codes - The airport codes provided by the class is not utilized for the custom join, as the data may be slightly obsolete (if we decide to run this for more recent data). Therefore, we are using the Stations and Airports data stated in the secondary data sources section above. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data fields
# MAGIC Data fields have been moved to appendix A (for all fields) and appendix B (for fields used in EDA and downstream)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Joining the data, i.e. custom join
# MAGIC
# MAGIC Our team has decided to take the path of creating a custom joined data as the source for rest of the tasks. 
# MAGIC
# MAGIC __Why custom join?__
# MAGIC The high level reasons are for reliability and adaptability. Since we don't know the original otpw's join logic, we need to blindly trust the original otpw (i.e. less reliability), and we cannot add newer flights and weather data. The detailed reasons are as follows: 
# MAGIC
# MAGIC * __The original otpw is losing some rows.__ The percentage of loss is less than 1% (e.g. for 3M, 0.2% loss) however as we don't have the code for how the original OTPW is created, we don't know the exact reason and therefore the evaluation results may be biased if we use the original otpw. 
# MAGIC * __The original otpw's Weather data seems to be not tied to the latest reading.__ Although we have confirmed the latest weather data is 2 hours prior to the scheduled flight, it seems not be the latest reading and instead picking others. Similar to above, because we don't know the exact join logic, we may get unexpected results. 
# MAGIC * __Not adaptable to newer data/logic.__ If our team decides to use newer data after 2019, we are stuck with whatever that's available from the original otpw. Also if we decide to add weather data of arrival, the logic of joining the weather data will be inconsistent between departure and arrival, potentially making it hard to debug if issues arise 
# MAGIC * __Original data types are not preserved.__ The original otpw have everything as string data type. While this is a minor issue
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC #### General Steps
# MAGIC The join is divided into 3 steps: 
# MAGIC
# MAGIC | __Step__ | __Description__ | 
# MAGIC | -------- | --------------- |
# MAGIC | Step 1 - Mapping table generation | Preparation work before the actual flight and weather join by generating a mapping table from airport to weather station(s) |
# MAGIC | Step 2 - Flight & Weather join | The actual join of flight data and weather data utilizing the mapping table from step 1 |
# MAGIC | Step 3 - Self join for previous flight data | Self joining the table from step 2 but joining with previous flight's row to get previous flight data (e.g. previous flight departure time) |
# MAGIC
# MAGIC Few notes: 
# MAGIC * Note that while step 1 is persisted, step 2 and step 3 are processed together before presisting to disk. Also for 5 year data, it is processed and persisted year by year. 
# MAGIC * Join investigation and source code stored [here](https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/1500157381699051?o=4021782157704243). It is mostly based on spark SQL. 
# MAGIC * While the join logics are explain below in steps, the actual SQL query and most importantly the spark execution plan is different for optimization purposes

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 1 - Mapping table generation
# MAGIC
# MAGIC The first step is creating the mapping table from origin's airport ID to weather's station ID, utilizing the airports.csv and isd_history.csv 
# MAGIC
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/join-step1.png">
# MAGIC
# MAGIC __Preparation prior to join (but done within same query):__
# MAGIC * There was a minor clean up in the airports.csv file where the IATA code was not populated due to being an old airport. Fortunately the original IATA codes were stored in a supplemental column and therefore was restored from there.
# MAGIC
# MAGIC
# MAGIC __Join logic:__
# MAGIC 1. Get list of distinct ORIGIN airport IDs (e.g. SFO) from Flights table 
# MAGIC 1. Left outer join thea airport IDs to iata_code in the airports table
# MAGIC 1. Left outer join the airport table's icao_code to isd_history table's ICAO field 
# MAGIC 1. Filter distinct station IDs and lat/lon from weather table
# MAGIC 1. Left outer join the concatenated USAF+WBAN field from isd_history to station IDs from #4
# MAGIC 1. __IF__ a match was not found from step 5, look for nearest top K airports within N miles from the airport
# MAGIC
# MAGIC __Other notes:__
# MAGIC * The nearest airport match is calculated based on Haversine formula using lat/lon
# MAGIC * The "top K" and "within N miles" is configurable so we can also aim for higher coverage. As of this writing we're only using Top 1 within 20 miles
# MAGIC   * If K > 1 this will create duplicate origins airport codes in the mapping table, but will be pruned in #2
# MAGIC * The mapping table is persisted prior to the next step

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 2 - Flight & Weather join
# MAGIC We join the flights data with weather data using the origin_weather (i.e. mapping) table + extra logic for finding the most appropriate weather row for the flight. 
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/join-step2.png">
# MAGIC
# MAGIC __Preparation prior to join (but done within same query):__
# MAGIC * Filter the weather data with stations that exist in origin_weather table
# MAGIC * Convert departure date/time to a UTC timestamp based on airport location's timezone. Weather data is already in UTC
# MAGIC * Calculate lower and upper bound timestamp that is allowed when matching flight table with weather table 
# MAGIC
# MAGIC
# MAGIC __Join logic:__
# MAGIC 1. Left outer join flights data with origin_weather (i.e. mapping) table with ORIGIN
# MAGIC 1. Left outer join origin_weather table with the (filtered) weather table by: 
# MAGIC    * STATION id (exact match) and
# MAGIC    * Range join weather table's timestamp with lower/upper bound timestamp precalculated for each flight
# MAGIC 1. Filter the table from #2 by only picking the latest/closest weather timestamp from the scheduled depart time
# MAGIC
# MAGIC __Other notes:__
# MAGIC * The lower and upper bound is a customizable parameter by hours. Currently it's set as lower=2 and upper=4. In this case, if a flight is scheduled to depart at 1/15/2015 10AM, then weather data needs to be between 1/15/2015 6AM~8AM, and the weather closest but not over 8AM will be chosen in #3
# MAGIC * The filtering in #3 is based on windowing function to rank the rows. If a matching weather row doesn't exist in #2, this filtering is ignored
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Step 3 - Self join for previous flight data
# MAGIC Finally we do a self join from Step 2 to retrieve the previous flight data
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/join-step3.png">
# MAGIC
# MAGIC __Preparation prior to join:__
# MAGIC * Assign an index number tail_idx to each row based on TAIL_NUM (tail number, equivalent to car registration # for airplanes) and flight timestamp (ascending).
# MAGIC
# MAGIC __Join logic:__
# MAGIC 1. Left outer join flights_weather data (from Step 2) to itself using TAIL_NUM and tail_idx, but the right side table is using tail_idx - 1 as the previous flight
# MAGIC 1. Project only a few relevant tables from the previous data for the final otpw_v2 table
# MAGIC
# MAGIC __Other notes:__
# MAGIC * If tail number doesn't exist, then they will be ignored. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary results 
# MAGIC
# MAGIC Here are the summary results from the table generation
# MAGIC
# MAGIC | **Metric**                 | **3 Months** | **1 year (2019)** | **5Y (2015–2019)** |
# MAGIC |----------------------------|--------------|--------------------|---------------------|
# MAGIC | **# of rows**              | 1,403,471    | 7,422,037          |  31,746,841         |
# MAGIC | **# of columns**           | 258           | 258                 | 258                  |
# MAGIC | **Mapping table generation time**     | < 1 min      | < 1 min            | < 1 min             |
# MAGIC | **Join (+ self join) generation time**| 4 min        | 30 min             | 6.5h                |
# MAGIC | **Table size (parquet)**           |   255MB         |    1.4GB              |       5.8GB            |
# MAGIC
# MAGIC
# MAGIC Notes: 
# MAGIC * We have confirmed we haven't lost (nor gained) any rows compared to the original flights data
# MAGIC * Cluster size: 6-8 nodes depending on when the availability 
# MAGIC * the 5Y data is taking longer due to some data check querying done for each year generation. For the final report we will have the updated generation time. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Exploratory Data Analysis (EDA)
# MAGIC
# MAGIC ### Summary Description
# MAGIC
# MAGIC This project utilizes a multi-source data ecosystem combining historical flight operations data with real-time environmental observations to understand and predict departure delays. By integrating aviation performance records from the Bureau of Transportation Statistics with NOAA weather observations and geospatial metadata, we created a rich analytical foundation for modeling delay patterns. 
# MAGIC
# MAGIC    - **Airline On-Time Performance Data (Flights)**: Provided by the Bureau of Transportation Statistics (BTS), this dataset contains comprehensive historical flight performance records for U.S. domestic flights from 2015-2019. Each record represents an individual flight, including scheduled and actual departure/arrival times, delay causes (carrier, weather, NAS, security, late aircraft), flight identifiers (airline, tail number, origin, destination), and cancellation information. This dataset forms the foundation for understanding flight delay patterns across airlines, routes, and temporal dimensions.
# MAGIC
# MAGIC    - **Airport Codes Data**: Sourced from the Airport Codes Dataset (DataHub), this dataset contains comprehensive airport information including IATA and ICAO codes, airport names, cities, states, countries, and geographic coordinates. This enables accurate mapping of airport codes to their geographic positions and facilitates the integration of weather data with flight operations data.
# MAGIC
# MAGIC    - **Weather Data**: Sourced from NOAA's Integrated Surface Database (ISD), this dataset provides historical weather observations from stations across the U.S., including temperature, humidity, precipitation, wind speed, visibility, and adverse weather events (storms, fog, snow). Weather observations are recorded at hourly intervals, making them critical for capturing environmental conditions that influence flight operations.
# MAGIC
# MAGIC    - **Weather Stations Metadata**: This supplementary dataset provides geospatial metadata for weather stations, including station IDs (USAF+WBAN codes), latitude, longitude, elevation, and station names, sourced from National Centers of Environment Information (NCEI). It enables precise mapping between weather observations and geographic locations, which is essential for associating weather conditions with specific airports.
# MAGIC
# MAGIC ### Purpose and Approach of EDA
# MAGIC
# MAGIC Before developing predictive models, we conducted comprehensive exploratory data analysis to accomplish three critical objectives: First, to assess data quality and identify issues such as missing values, duplicates, or data type inconsistencies that could compromise model performance. Second, to understand the underlying patterns and distributions in our data, including temporal trends, geographic variations, and the relationships between features and our target variable. Third, to inform our feature engineering strategy by identifying which variables show strong predictive signals and how they might be transformed or combined to improve model accuracy. This systematic exploration guides our modeling decisions and helps us avoid common pitfalls such as data leakage, inappropriate feature transformations, or overlooking important predictive patterns.
# MAGIC
# MAGIC Our EDA has been organized across several analytical dimensions:
# MAGIC
# MAGIC  - **(1) Data Quality Checks:**
# MAGIC
# MAGIC Our custom-joined OTPW_V2 dataset demonstrates high data quality with expected gaps in weather features (15-30%) and previous flight data (~20% structural missingness). No duplicate records were identified, and original data types were successfully preserved.
# MAGIC
# MAGIC Ensuring data integrity is foundational to building reliable predictive models. We conducted comprehensive data quality assessments across four dimensions using cached DataFrames in Spark to optimize computational efficiency given our 7.4 million flight records.
# MAGIC
# MAGIC   - **Data Dictionary with Statistics**: Generated complete schema information including data types, nullable status, and descriptive statistics (count, min, max, mean, std dev for numeric columns; distinct counts for categorical columns). Full details are provided in Appendix B.
# MAGIC   - **Duplicate Detection**: Verified data uniqueness using composite key (FL_DATE, OP_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST, CRS_DEP_TIME) and exact row matching. No duplicates were detected, confirming the integrity of our custom join process.
# MAGIC   - **Data Type Verification**: Validated column data types and scanned for anomalous values (NaN, Inf, -Inf). All numeric fields passed validation, and our custom join correctly preserved original data types from source tables.
# MAGIC   - **Missing Value Analysis**: Missing data analysis is critical for understanding data completeness and informing imputation strategies. Our assessment identified six columns with missing values, as listed in below table.
# MAGIC
# MAGIC   | Column | Missing Count | Missing % | Explanation | Handling Strategy |
# MAGIC |--------|---------------|-----------|-------------|-------------------|
# MAGIC | CANCELLATION_CODE | 7,287,112 | 98.18% | From cross-validation, null value means flight was "not cancelled" | No action needed—missingness is informative; use CANCELLED flag for modeling |
# MAGIC | HourlyPresentWeatherType | 6,625,959 | 89.27% | From deep dive analysis, null value indicates clear/normal weather conditions | No imputation required—treat missing as "no significant weather event" |
# MAGIC | PREV_ORIGIN | 23,728 | 0.32% | Either data error or first flight of the day with no previous leg | Remove affected records or create IS_FIRST_FLIGHT indicator |
# MAGIC | PREV_DEST | 23,728 | 0.32% | Either data error or first flight of the day with no previous leg | Remove affected records or create IS_FIRST_FLIGHT indicator |
# MAGIC | TAIL_NUM | 17,837 | 0.24% | Data error—very small count relative to dataset size | Remove affected records (negligible impact on 7.4M rows) |
# MAGIC | DATE | 8,230 | 0.11% | Data error—can use FL_DATE column as alternative | No action needed—FL_DATE column provides equivalent information |
# MAGIC
# MAGIC In summary, the two columns with highest missing rates (CANCELLATION_CODE at 98.18% and HourlyPresentWeatherType at 89.27%) are missing by design - their absence conveys meaningful information rather than data gaps. A null cancellation code indicates the flight operated normally, while missing weather type suggests clear/normal conditions. For PREV_ORIGIN and PREV_DEST (0.32% missing each), the null values may indicate either data collection errors or first-of-day flights where no previous leg exists. Given the small count (~23,700 records), we will remove these records from the training set; alternatively, a binary IS_FIRST_FLIGHT indicator could be created if first-flight status proves predictive of delays.
# MAGIC The remaining columns (TAIL_NUM, DATE) show minimal missingness (<0.25%) attributed to data errors and can be safely removed or handled using alternative columns (FL_DATE).
# MAGIC
# MAGIC  - **(2) Summary Statistics:**
# MAGIC The dataset contains 7.4 million flight records from 2019 with 16 unique carriers operating across 353 airports, covering 54 U.S. states with typical distributions for U.S. domestic operations. The representation of the Summary Statitics is provided in Appendix B.
# MAGIC
# MAGIC  - **(3) Delay Analysis:**
# MAGIC Our target variable DEP_DEL15 shows a 18.66% overall delay rate, indicating presence of class imbalance that will require consideration during model training.
# MAGIC
# MAGIC   <img src="https://raw.githubusercontent.com/hong-hu/w261/main/flight%20departure%20status_pic1.png" width="1000">
# MAGIC
# MAGIC A critical consideration in our feature engineering is ensuring that all features used for prediction are realistically available at T-2h (two hours before scheduled departure). We conducted a detailed analysis of previous flight features to assess potential data leakage risks. While scheduled information—PREV_ORIGIN, PREV_DEST, PREV_CRS_DEP_TIME, and PREV_CRS_ARR_TIME—is safely available at prediction time, actual performance data presents challenges. PREV_DEP_TIME (actual departure) may or may not be available depending on when the previous flight departed, while PREV_DEP_DELAY and PREV_ARR_TIME are definitively not available at T-2h for many flights, particularly those with short turnarounds. To understand delay patterns using only T-2h safe features, we analyzed delay rates by first flight status and scheduled turnaround time. First flights of the day exhibit a 20.78% delay rate compared to 18.66% for flights with a previous leg—a counterintuitive finding suggesting that having previous flight information available may actually help predict (and potentially mitigate) delays. When examining delay rates by scheduled turnaround time, flights with very short turnarounds (<30 minutes) show the lowest delay rate (~16%), while flights with 30-90+ minute turnarounds experience higher delay rates (21-26%). This pattern suggests that airlines may build additional buffer into tight turnaround schedules, or that short-turnaround flights are prioritized operationally.
# MAGIC These insights validate that scheduled turnaround time—a feature safely available at T-2h—provides meaningful predictive signal without introducing data leakage, and can serve as an alternative to actual previous flight delay information.
# MAGIC
# MAGIC   <img src="https://raw.githubusercontent.com/hong-hu/w261/main/delay_first%20or%20previous_delay%20by%20turnaroundtime_pic5.png" width="1000">
# MAGIC
# MAGIC
# MAGIC  - **(4) Correlation Analysis:**
# MAGIC  Correlation analysis serves multiple critical functions in our predictive modeling workflow. First, it identifies which features have strong linear relationships with our target variable (DEP_DEL15), helping prioritize features for model inclusion. Second, it detects multicollinearity—high correlations between predictor variables—which can inflate variance in regression coefficients and reduce model interpretability. Finally, correlation patterns inform feature engineering decisions, such as DEPART_DELAY related variables to remove redundant ones. Correlation analysis identified critical multicollinearity among DEP_DELAY variants (0.96-1.00), necessitating exclusion of all except our target DEP_DEL15. Top predictive features include LATE_AIRCRAFT_DELAY and CARRIER_DELAY, which show strong non-linear relationships, and PREV_DEP_DELAY (0.23-0.34), confirming cascading delay effects. Surprisingly, raw weather metrics like HourlyRelativeHumidity (0.27-0.33) outperform coded WEATHER_DELAY (0.10-0.20), suggesting weather impacts require feature engineering. Large Pearson-Spearman gaps throughout indicate non-linear relationships, indicating tree-based models may out-perform linear regression.
# MAGIC
# MAGIC   <img src="https://raw.githubusercontent.com/hong-hu/w261/main/correlation_pic4.png">
# MAGIC
# MAGIC
# MAGIC  - **(5) Temporal Patterns:**
# MAGIC Delay rates exhibit strong temporal variation: lowest in morning hours (5-9am: 7-13%) and highest during evening peak (5-11pm: 25-27%), with clear seasonal and day-of-week patterns. 
# MAGIC
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/delay%20rate%20by%20hour%20of%20day_pic2.png" width="600">
# MAGIC
# MAGIC   <img src="https://raw.githubusercontent.com/hong-hu/w261/main/delay%20by%20day%20of%20week_pic6.png" width="700">
# MAGIC
# MAGIC   <img src="https://raw.githubusercontent.com/hong-hu/w261/main/delay%20rate%20by%20month_pic7.png" width="700">
# MAGIC
# MAGIC Weekday flight volumes remain consistent (1.05-1.09M) while Saturday drops 18% (892k). Thursday exhibits the worst delay performance (20.2%, 12.5 min average) as cascading effects accumulate through the week, while Saturday shows the best performance (16.7%, 9.5 min average) benefiting from lower volume and overnight system reset. Day-of-week will be included as a categorical feature given its predictive value.
# MAGIC
# MAGIC Monthly analysis reveals clear seasonality with peak travel in summer (July/August: ~645k flights) and lowest volume in February (520k). Notably, delay rates do not correlate directly with volume—June shows the highest delays (24%) likely due to thunderstorm season, while September achieves the best performance (14%) despite moderate-high volume. This suggests weather conditions outweigh volume as the primary delay driver.
# MAGIC
# MAGIC  - **(6) Carrier Performance:**
# MAGIC Delay rates vary substantially across carriers, reflecting differences in operational practices, hub locations, and network complexity.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/main/carrier%20performace_pic8.png" width="1000">
# MAGIC
# MAGIC
# MAGIC  - **(7) Route Analysis:**
# MAGIC High-volume business routes between major hubs show elevated delays, with short-haul corridors like ORD <-> LGA (~27%) and LGA ->BOS (~25%) experiencing the highest delay rates, suggesting route-specific factors influence delay probability. However, Notably, the Hawaii inter-island route (HNL↔OGG) demonstrates that high volume does not guarantee high delays—its ~6% rate suggests hub congestion and network effects outweigh traffic volume as delay drivers.
# MAGIC
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/top20_routes_volume_delay%20by%20route_pic3.png" width="1000">
# MAGIC
# MAGIC These findings directly inform our feature engineering strategy and model development approach detailed in subsequent sections.
# MAGIC
# MAGIC ## Data Dictionary of Raw Features
# MAGIC
# MAGIC As explained in the Data Description section, we systematically profiled the raw dataset to classify each variable by functional type—continuous numerical, categorical, or text—based on domain context. This classification informs critical preprocessing decisions including encoding methods, scaling techniques, and feature selection approaches. Full feature specifications are documented in Appendix B. 
# MAGIC
# MAGIC ## Dataset size (rows columns, train, test, validation)
# MAGIC The Newly Joined Dataset size is a total of 7,422,037 records. Please refer to details in the Data Description section.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Modeling Pipeline
# MAGIC
# MAGIC ### Machine Learning Algorithms and Metrics
# MAGIC For both mean and median, late aircraft (i.e. previous flight's delay) is highest
# MAGIC We have decided to take a binary classification approach to the problem.
# MAGIC We will predict between two classes: 
# MAGIC * Delayed: the flight departs greater than 15 minutes after the scheduled departure time (Postive Class)
# MAGIC * Not Delayed: the flight departs early or up to 15 minutes after the scheduled departure time (Negative Class)
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Algorithms and Associated Loss Functions:
# MAGIC 1) **Majority Class Classifier**
# MAGIC <br>Always predicts an example to be part of the majority class. In our case, this model will always predict a flight to not be delayed.
# MAGIC
# MAGIC 2) **Logistic Regression**
# MAGIC <br>Predicts the probability of a delay as a sigmoid transformation of some weighted combination of features.
# MAGIC Easy to interpret, simple, easily accessible implementation.
# MAGIC <br>Loss Function: Negative Log-Likelihood in order to penalize incorect predictions based on the confidence of the incorrect prediction
# MAGIC $$\mathcal{L}_{\text{log}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\Big]$$
# MAGIC <br>Pyspark Implementation: `pyspark.ml.classification.LogisticRegression`
# MAGIC <br>Parameters: `regParam`= 0.0001, `elasticNetParam`= 0.0, `maxIter`= 50, `weightCol`= classWeight - we set delayed flights to have double the weight of non-delayed
# MAGIC
# MAGIC 3) **Gradient-Boosted Tree**
# MAGIC <br>An ensemble of shallow decisions trees that are sequentially built through correcting the mistakes of previous iterations.
# MAGIC <br>Good at capturing nonlinear feature interactions on complicated tabular data.
# MAGIC <br>Loss Function: A model updating procedure that fits a shallow tree to the negative gradient of the loss, with the loss per sample being:
# MAGIC $$\mathcal{L}_{\text{GBT}} = \frac{1}{N}\sum_{i=1}^{N}\log(1 + e^{-y_i F(x_i)})$$
# MAGIC <br>Pyspark Implementation: `pyspark.ml.classification.GBTClassifier`
# MAGIC <br>Parameters: `maxDepth`= 8, `maxIter`= 50, `stepSize`= 0.1, `weightCol`= classWeight - we set delayed flights to have double the weight of non-delayed
# MAGIC
# MAGIC ### Metrics and Analysis
# MAGIC We treat “Delayed” as the **positive** class and compute the following using predictions from the test split:
# MAGIC | Metric | Formula | Interpretation |
# MAGIC |--------|----------|----------------|
# MAGIC | **Accuracy** | $$\( \frac{TP+TN}{TP+TN+FP+FN} \)$$ | Overall correctness |
# MAGIC | **Precision** | $$\( \frac{TP}{TP+FP} \)$$ | How many predicted delays were actually delayed |
# MAGIC | **Recall (Sensitivity)** | $$\( \frac{TP}{TP+FN} \)$$ | How many actual delays were correctly identified |
# MAGIC | **F1-Score (Single Class)** | $$F1_c=( 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)$$ | Harmonic mean of precision and recall. We don't directly report this, instead using it to calculate the proceeding Macro F1-Score |
# MAGIC | **Macro F1-Score** | $$ F1_{macro} = \frac{F1_{class0} + F1_{class1}}{2} $$ | The unweighted mean of the F1 scores for both classes. This metric treats both classes as equally important, penalizing the model heavily if it fails on the minority (Delayed) class. |
# MAGIC
# MAGIC **Primary Metric**: Macro F1-score
# MAGIC <br>
# MAGIC **Secondary Metrics**: Accuracy, Recall, and Precision

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization of the modeling pipelines and subpipelines 
# MAGIC
# MAGIC Given the following diagram, the Flight Delay Prediction ML Pipeline consists of a 4-stage machine learning workflow:
# MAGIC
# MAGIC 1. __Common Pipeline__: Index categorical features using StringIndexer
# MAGIC 2. __Hyperparameter Tuning__: For logistic regression (L1, L2, ElasticNet) and ensemble models (decision tree, random forest, gradient boosting)
# MAGIC 3. __Model Training__: With possible preprocessing (one-hot encoding, scaling) and hyperparameter tuning via grid search
# MAGIC 4. __Model Evaluation__: Accuracy, Precision, Recall, F1, confusion matrix, and feature importance
# MAGIC

# COMMAND ----------

with open('img/phase2/w261_proj_presentation_ii_ml_pipeline.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering (OTPW_V2 Custom Data Join)

# COMMAND ----------

# MAGIC %md
# MAGIC The model utilizes a total of **22 raw input variables**, which are processed into **83 distinct input features** for training (after One-Hot Encoding particular categorical features). The features are categorized into four families:
# MAGIC
# MAGIC | Feature Family | Raw Count | Description | Specific Features |
# MAGIC | :--- | :--- | :--- | :--- |
# MAGIC | **Route & Carrier** | 5 | Network structure and airline identifiers. | `ORIGIN`\*, `DEST`\*, `PREV_ORIGIN`\*, `OP_UNIQUE_CARRIER`\*\*, `DISTANCE` |
# MAGIC | **Temporal** | 6 | Cyclical time representations and calendar data. | `MONTH`\*\*, `QUARTER`\*\*, `DAY_OF_WEEK`\*\*, `sched_hour`\*\*, `DEP_HOUR_SIN`, `DEP_HOUR_COS` |
# MAGIC | **Weather** | 8 | Meteorological conditions at departure. | `STORM_INDEX`, `HourlyWindSpeed`, `HourlyAltimeterSetting`, `HourlyDewPointTemperature`, `HourlyDryBulbTemperature`, `HourlyRelativeHumidity`, `HourlyPressureTendency`, `HourlyWindGustSpeed` |
# MAGIC | **Turnaround / Buffer** | 3 | The data from previous flights. | `SCHEDULED_BUFFER`, `PREV_DISTANCE`, `PREV_ORIGIN` |
# MAGIC
# MAGIC \* Target Encoded (Smoothed Mean)<br>
# MAGIC ** One-Hot Encoded
# MAGIC
# MAGIC **Feature Engineering Calculations:<br>**
# MAGIC   * **SCHEDULED_BUFFER**: Calculated as the time between the scheduled arrival of the previous flight and scheduled departure of the current flight<br>
# MAGIC   * **STORM_INDEX**: Precipitation x WindSpeed<br>
# MAGIC   * **DEP_HOUR_SIN and DEP_HOUR_COS**: Sine/Cosine encoding of schedule time, which allows for cyclical nature of time to be recognized while remaining scalar. Allows for the wrapping-around of hours, placing the 23rd hour close to the 1st hour.<br>
# MAGIC   * **Performed target encoding of airports **via using a smoothed global mean of their popularity in place of their names in order to reduce dimensionality.<br>
# MAGIC   * **NOTE:** PREV_DEP_DELAY and PREV_DELAY_GROUP were excluded as features in training models for OTPW_V2 due to data leaks being detected, 20% of examples in these fields were associated with situations where the previous flight departed less than 2 hours from the current flight.<br>
# MAGIC
# MAGIC **Class Imbalance Strategy:<br>**
# MAGIC   * **Undersampled** non-delay examples in training data via only training on 75% of non-delayed flights<br>
# MAGIC   * Weighted delay examples in training data by 2.0x that of non-delays via using weightCol parameter on LogisticRegression and GBTClassifier<br>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Results and Discussion of Results (OTPW_V2 Custom Join Data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experimental Setup
# MAGIC * **Experiments Conducted:** 3 (Baseline, Logistic Regression, GBT), each evaluated across 3 Time-Series Folds.<br>
# MAGIC * All experiments utilized identical input features
# MAGIC * **Cluster Configuration**: 16.4 LTS (incluedes Apache Spark 3.5.2, Scala 2.12)
# MAGIC * **Worker type**: m5d.xlarge, 16 Gb Memory, 4 Cores (Min: 2, Max: 8, Current: 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model Results for 1Y Data (75% Train, 25% Validation Chronological Split)

# COMMAND ----------

# MAGIC %md
# MAGIC | Model | Split | Accuracy | F1 Score (Macro) | Precision (Delayed) | Recall (Delayed) | Time |
# MAGIC | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
# MAGIC | **Baseline** |  Train | 0.828 | 0.750 | 0.0 | 0.0 | 0.33 min |
# MAGIC | |  Val | **0.827** | 0.749 | 0.0 | 0.0 | 0.2 min |
# MAGIC | | | | | | | |
# MAGIC | **Logistic Regression** |  Train | 0.742 | 0.730 | 0.433 | 0.340 | 2.00 min |
# MAGIC | |  Val | 0.804 | 0.775 | 0.359 | 0.176 | 1.33 min |
# MAGIC | | | | | | | |
# MAGIC | **Gradient Boosting** |  Train | 0.768 | 0.759 | 0.504 | 0.413 | 15.75 min |
# MAGIC | |  Val | 0.794 | **0.783** | **0.375** | **0.293** | 5.22 min |

# COMMAND ----------

# MAGIC %md
# MAGIC ####Interpretation of Model Metrics
# MAGIC The Majority Class Baseline achieved the highest overall accuracy of 82.7% simply by predicting "Not Delayed" for every flight.
# MAGIC However, this model provides zero operational value, achieving a Recall of 0.0% and failing to identify a single delay (TP = 0).
# MAGIC This establishes a "Null Accuracy" benchmark: any useful model must outperform the baseline on Recall and macro F1-Score, even if it sacrifices raw Accuracy.
# MAGIC By focusing on the F1-Macro metric, we are measure a model's success in predicting both delays and non-delays in a balanced way.
# MAGIC The results explicitly demonstrate the dangers of relying on Accuracy as a primary metric for imbalanced data.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC The Logistic Regression model achieved a slightly higher raw accuracy (80.4%) compared to the GBT (79.3%). However, this metric is deceptive due to the class imbalance (only around 20% of flights are delayed). The Logistic Regression model was also more conservative, predicting "On Time" more often to maximize accuracy.<br>
# MAGIC The Gradient Boosted Tree has the best Macro F-1 Score with 0.782 for the wholistic 1Y experiment and additionally was able to correctly recall 27.9% of delays.<br>
# MAGIC The GBT model accepted a slightly lower overall accuracy to aggressively target delays, and as a result, outperformed Logistic Regression by 10 percentage points in recall.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC While the Baseline was computationally negligible (0.33 min to train), the GBT required 15.75 minutes to build. However, this additional compute time yielded a system that actually detected delays. In a real-world scenario, the trade-off of 15.75 minutes of training time for the ability to predict nearly 30% of delays two hours in advance is highly favorable.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####Confusion Matrix Analysis (OTPW_V2 Custom Join Data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/confusion_baseline.png" width="700">

# COMMAND ----------

# MAGIC %md
# MAGIC **Baseline (Majority Class)**<br>
# MAGIC This matrix is entirely concentrated in the left column ("Predicted: On Time").<br>
# MAGIC This visualizes the fundamental problem with accuracy as a metric. While the top-left quadrant (True Negatives) is massive (82.7%), the bottom-right quadrant (True Positives) is empty. The model completely ignores the 315,601 actual delays, rendering it operationally useless despite high accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/main/confusion_logistic_M2.png" width="700">

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic Regression**<br>
# MAGIC This model is conservative, but correctly identified 55,061 delays (3.0% of total examples) that the baseline missed. However, the bottom-left quadrant (False Negatives) remains very large (14.2%), indicating that the model still misses the vast majority of disruptions. It prioritizes minimizing False Alarms (Top-Right) over catching delays.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/main/confusion_gbt_M3.png" width="700">

# COMMAND ----------

# MAGIC %md
# MAGIC **Gradient Boosted Tree**<br>
# MAGIC The GBT model is the most aggressive and effective. <br>
# MAGIC It correctly identified 92,579 delays (5.1%), nearly doubling the delay recall of Logistic Regression.<br>
# MAGIC This improved detection comes at the cost of the Top-Right quadrant (False Positives), which increased to 154,063 (8.4%).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top Features by Importance via Logistic Regression (OTPW_V2 Custom Join Data)
# MAGIC
# MAGIC | Feature Name | Coefficient |
# MAGIC | :--- | :--- |
# MAGIC | SCHEDULED_BUFFER | 0.289304 |
# MAGIC | HourlyRelativeHumidity | 0.283907 |
# MAGIC | DEP_HOUR_SIN | -0.245917 |
# MAGIC | sched_hour_ohe_6 | -0.197617 |
# MAGIC | HourlyDewPointTemperature | -0.161314 |
# MAGIC | DEST_encoded | 0.154226 |
# MAGIC | ORIGIN_encoded | 0.143529 |
# MAGIC | sched_hour_ohe_5 | -0.133157 |
# MAGIC | sched_hour_ohe_7 | -0.125106 |
# MAGIC | MONTH_ohe_6 | 0.113700 |
# MAGIC | MONTH_ohe_9 | -0.103340 |
# MAGIC | OP_UNIQUE_CARRIER_ohe_DL | -0.080732 |
# MAGIC | MONTH_ohe_7 | 0.073261 |
# MAGIC | MONTH_ohe_1 | -0.068391 |
# MAGIC | sched_hour_ohe_8 | -0.065886 |
# MAGIC | PREV_ORIGIN_encoded | 0.065303 |
# MAGIC | QUARTER_ohe_1 | -0.062395 |
# MAGIC | MONTH_ohe_4 | -0.059254 |
# MAGIC | MONTH_ohe_8 | 0.058901 |
# MAGIC | HourlyDryBulbTemperature | -0.056930 |
# MAGIC
# MAGIC ### Interpretation of Feature Importance
# MAGIC The Logistic Regression model identifies operational constraints and weather conditions as the strongest predictors of flight delays. Positive coefficients indicate features that increase the log-odds of a delay, while negative coefficients suggest features that reduce delay probability.
# MAGIC
# MAGIC *   Delay Drivers (Positive Coefficients): The strongest predictor of a delay is `SCHEDULED_BUFFER` (0.289) and `HourlyRelativeHumidity` (0.284). This suggests that flights requiring larger schedule padding and flights operating in humid conditions are at the highest risk. Additionally, `DEST_encoded` and `ORIGIN_encoded` show that high-traffic airports significantly contribute to delay probability, as the encoding procedure simply utilizes the popularity of the airport. Summer travel is also a risk factor, indicated by the positive coefficients for June (`MONTH_ohe_6`) and July (`MONTH_ohe_7`).
# MAGIC *   On-Time Drivers (Negative Coefficients): The most effective way to avoid delays appears to be by flying early in the morning as the features `sched_hour_ohe_5`, `_6`, and `_7` (5 AM – 7 AM) all have strong negative coefficients. Operationally, Delta Airlines (`OP_UNIQUE_CARRIER_ohe_DL`) appears to have a protective effect against delays. Finally, September (`MONTH_ohe_9`) and cooler/drier weather metrics (DewPoint interactions) are associated with better on-time performance.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-Fold Expanding Window Time-Series Cross Validation Results (OTPW_V2 Custom Join Data):
# MAGIC | Experiment | Fold | Validation Period | Time (min) | Accuracy | Recall (Delayed) | F1 Macro |
# MAGIC | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
# MAGIC | **1. Baseline (Majority Class)** | 1 | Apr-May | 0.34 | 0.812 | 0.000 | 0.448 |
# MAGIC | | 2 | Jun-Jul | 0.07 | 0.777 | 0.000 | 0.437 |
# MAGIC | | 3 | Aug-Sep | 0.06 | 0.829 | 0.000 | 0.453 |
# MAGIC | **Baseline Average** | | | **0.16** | **0.806** | 0.000 | 0.446 |
# MAGIC | | | | | | | |
# MAGIC | **2. Logistic Regression (Weighted)** | 1 | Apr-May | 0.70 | 0.807 | 0.071 | 0.506 |
# MAGIC | | 2 | Jun-Jul | 0.73 | 0.776 | 0.067 | 0.495 |
# MAGIC | | 3 | Aug-Sep | 0.87 | 0.813 | 0.150 | 0.554 |
# MAGIC | **LR Average** | | | 0.77 | 0.799 | 0.096 | 0.518 |
# MAGIC | | | | | | | |
# MAGIC | **3. Gradient Boosted Tree (Weighted)** | 1 | Apr-May | 1.00 | 0.809 | 0.133 | 0.550 |
# MAGIC | | 2 | Jun-Jul | 1.22 | 0.775 | 0.191 | 0.571 |
# MAGIC | | 3 | Aug-Sep | 1.96 | 0.807 | 0.233 | 0.591 |
# MAGIC | **GBT Average** | | | 1.39 | 0.797 | **0.186** | **0.571** |

# COMMAND ----------

# MAGIC %md
# MAGIC ####Cross-Validation Analysis
# MAGIC The 3-Fold Expanding Window Time-Series Cross-Validation revealed distinct patterns regarding model stability, the impact of seasonality, and the benefits of the expanding training window.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC The Baseline (Majority Class) results highlight the seasonal volatility of flight delays.<br>
# MAGIC **Fold 2 (Jun-Jul Validation):** Baseline accuracy dropped to its lowest point (77.7%), compared to 81.2% in Fold 1 and 82.9% in Fold 3.<br>
# MAGIC Since the Baseline predicts "Not Delayed" for everything, a drop in accuracy indicates a higher frequency of actual delays. This confirms that June and July (Summer peak travel) are the most difficult months to predict, with a significantly higher base rate of delays compared to Spring (Apr-May) or early Autumn (Aug-Sep).
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Across all three folds, the Gradient Boosted Tree (GBT) consistently outperformed Logistic Regression in the key metric of Recall (Delayed).<br>
# MAGIC **Logistic Regression:** Struggled to identify delays in the early folds (Recall near 7%) and only began to improve significantly in Fold 3 (15%).<br>
# MAGIC GBT demonstrated favorable metrics in Fold 1 (13.3% Recall) and consistently improved.<br>
# MAGIC The consistently higher F1 Macro scores for GBT (Avg 0.571 vs LR 0.518) confirm that the non-linear tree architecture is better at capturing complex delay drivers (possibly weather interactions) than the linear logistic model.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC A clear upward trend in performance is visible as the training set expands over time.
# MAGIC GBT Recall Trajectory: 13.3% (Fold 1) 
# MAGIC As the training window expanded from 3 months (Jan-Mar) to 7 months (Jan-Jul), the model successfully utilized the additional historical data to refine its decision boundaries. The peak performance in Fold 3 suggests that more training data directly correlates with higher delay detection capability for this specific problem.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Despite the seasonal difficulty of Fold 2 (Summer), the GBT model's F1 Macro score remained stable and even increased (0.550 to 0.571 to 0.591). This indicates the model is robust in that it did not collapse when faced with the higher chaos of summer travel, but rather adapted and continued to improve its delay detection rate.

# COMMAND ----------

# MAGIC %md
# MAGIC Flight delays cost the aviation industry billions annually in operational disruptions. This project focused on predicting delays (>15 minutes) at the actionable T-minus 2-hour horizon, aiming to enable proactive scheduling adjustments. We hypothesized that non-linear machine learning pipelines utilizing custom engineered features could accurately forecast disruptions where statistical baselines fail.<br>
# MAGIC By implementing an Expanding Window Time-Series Cross-Validation strategy and developing novel features like `SCHEDULED_BUFFER` and `STORM_INDEX`, we validated this hypothesis. While the Majority Class Baseline achieved high accuracy (82.7%) by blindly predicting "On Time," it offered zero operational value (0% Recall). In contrast, our Gradient Boosted Tree (GBT) model successfully identified 27.9% of delays, capturing 92,579 disrupted flights in the validation set alone. This represents a massive efficiency gain, potentially mitigating nearly one-third of delays before they occur. Future work can utilize additional compute resources to perform hyperparameter tuning and test the efficacy of a neural-network-based model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Appendix A: Data Dictionary of Raw Data
# MAGIC
# MAGIC The statistics are based on 3 month data. The source of "otpw_v2" represented derived columns that were created during the custom data join process
# MAGIC | **source** | **field_id** | **field_desc** | **data_type** | **in_otpw** | **missing_pct** | **count** | **mean** | **stddev** | **min** | **25%** | **50%** | **75%** | **max** |
# MAGIC |-----------|--------------|----------------|---------------|-------------|-----------------|-----------|----------|------------|---------|---------|---------|---------|--------|
# MAGIC | flights_3m | YEAR | Year | int | TRUE | 0.00% | 2806942 | 2,015.00 | 0.00 | 2015 | 2015 | 2015 | 2015 | 2015 |
# MAGIC | flights_3m | QUARTER | Quarter (1-4) | int | TRUE | 0.00% | 2806942 | 1.00 | 0.00 | 1 | 1 | 1 | 1 | 1 |
# MAGIC | flights_3m | MONTH | Month | int | TRUE | 0.00% | 2806942 | 2.02 | 0.83 | 1 | 1 | 2 | 3 | 3 |
# MAGIC | flights_3m | DAY_OF_MONTH | Day of Month | int | TRUE | 0.00% | 2806942 | 15.54 | 8.69 | 1 | 8 | 16 | 23 | 31 |
# MAGIC | flights_3m | DAY_OF_WEEK | Day of Week | int | TRUE | 0.00% | 2806942 | 3.94 | 2.00 | 1 | 2 | 4 | 6 | 7 |
# MAGIC | flights_3m | FL_DATE | Flight Date (yyyymmdd) | string | TRUE | 0.00% | 2806942 |  |  | 2015-01-01 |  |  |  | 2015-03-31 |
# MAGIC | flights_3m | OP_UNIQUE_CARRIER | Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years. | string | TRUE | 0.00% | 2806942 |  |  | AA |  |  |  | WN |
# MAGIC | flights_3m | OP_CARRIER_AIRLINE_ID | An identification number assigned by US DOT to identify a unique airline (carrier). A unique airline (carrier) is defined as one holding and reporting under the same DOT certificate regardless of its Code, Name, or holding company/corporation. | int | TRUE | 0.00% | 2806942 | 19,977.27 | 397.96 | 19393 | 19790 | 19977 | 20366 | 21171 |
# MAGIC | flights_3m | OP_CARRIER | Code assigned by IATA and commonly used to identify a carrier. As the same code may have been assigned to different carriers over time, the code is not always unique. For analysis, use the Unique Carrier Code. | string | TRUE | 0.00% | 2806942 |  |  | AA |  |  |  | WN |
# MAGIC | flights_3m | TAIL_NUM | Tail Number | string | TRUE | 0.58% | 2790562 |  |  | D942DN |  |  |  | N9EAMQ |
# MAGIC | flights_3m | OP_CARRIER_FL_NUM | Flight Number | int | TRUE | 0.00% | 2806942 | 2,243.94 | 1,793.08 | 1 | 752 | 1715 | 3465 | 9794 |
# MAGIC | flights_3m | ORIGIN_AIRPORT_ID | Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused. | int | TRUE | 0.00% | 2806942 | 12,670.94 | 1,519.92 | 10135 | 11292 | 12889 | 13930 | 16218 |
# MAGIC | flights_3m | ORIGIN_AIRPORT_SEQ_ID | Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time. | int | TRUE | 0.00% | 2806942 | 1267096.738 | 151991.4914 | 1013503 | 1129202 | 1288903 | 1393003 | 1621801 |
# MAGIC | flights_3m | ORIGIN_CITY_MARKET_ID | Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market. | int | TRUE | 0.00% | 2806942 | 31,712.19 | 1,283.51 | 30070 | 30627 | 31453 | 32467 | 35991 |
# MAGIC | flights_3m | ORIGIN | Origin Airport | string | TRUE | 0.00% | 2806942 |  |  | ABE |  |  |  | YUM |
# MAGIC | flights_3m | ORIGIN_CITY_NAME | Origin Airport, City Name | string | TRUE | 0.00% | 2806942 |  |  | Aberdeen, SD |  |  |  | Yuma, AZ |
# MAGIC | flights_3m | ORIGIN_STATE_ABR | Origin Airport, State Code | string | TRUE | 0.00% | 2806942 |  |  | AK |  |  |  | WY |
# MAGIC | flights_3m | ORIGIN_STATE_FIPS | Origin Airport, State Fips | int | TRUE | 0.00% | 2806942 | 26.04441132 | 16.72778043 | 1 | 12 | 24 | 41 | 78 |
# MAGIC | flights_3m | ORIGIN_STATE_NM | Origin Airport, State Name | string | TRUE | 0.00% | 2806942 |  |  | Alabama |  |  |  | Wyoming |
# MAGIC | flights_3m | ORIGIN_WAC | Origin Airport, World Area Code | int | TRUE | 0.00% | 2806942 | 55.49525498 | 26.44069267 | 1 | 34 | 52 | 81 | 93 |
# MAGIC | flights_3m | DEST_AIRPORT_ID | Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused. | int | TRUE | 0.00% | 2806942 | 12,670.92 | 1,519.93 | 10135 | 11292 | 12889 | 13930 | 16218 |
# MAGIC | flights_3m | DEST_AIRPORT_SEQ_ID | Destination Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time. Airport attributes, such as airport name or coordinates, may change over time. | int | TRUE | 0.00% | 2806942 | 1267094.702 | 151993.1061 | 1013503 | 1129202 | 1288903 | 1393003 | 1621801 |
# MAGIC | flights_3m | DEST_CITY_MARKET_ID | Destination Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market. | int | TRUE | 0.00% | 2806942 | 31712.16108 | 1283.52546 | 30070 | 30627 | 31453 | 32467 | 35991 |
# MAGIC | flights_3m | DEST | Destination Airport | string | TRUE | 0.00% | 2806942 |  |  | ABE |  |  |  | YUM |
# MAGIC | flights_3m | DEST_CITY_NAME | Destination Airport, City Name | string | TRUE | 0.00% | 2806942 |  |  | Aberdeen, SD |  |  |  | Yuma, AZ |
# MAGIC | flights_3m | DEST_STATE_ABR | Destination Airport, State Code | string | TRUE | 0.00% | 2806942 |  |  | AK |  |  |  | WY |
# MAGIC | flights_3m | DEST_STATE_FIPS | Destination Airport, State Fips | int | TRUE | 0.00% | 2806942 | 26.04658522 | 16.72714203 | 1 | 12 | 24 | 41 | 78 |
# MAGIC | flights_3m | DEST_STATE_NM | Destination Airport, State Name | string | TRUE | 0.00% | 2806942 |  |  | Alabama |  |  |  | Wyoming |
# MAGIC | flights_3m | DEST_WAC | Destination Airport, World Area Code | int | TRUE | 0.00% | 2806942 | 55.49300342 | 26.44065317 | 1 | 34 | 52 | 81 | 93 |
# MAGIC | flights_3m | CRS_DEP_TIME | CRS Departure Time (local time: hhmm) | int | TRUE | 0.00% | 2806942 | 1,327.60 | 474.36 | 1 | 924 | 1323 | 1725 | 2359 |
# MAGIC | flights_3m | DEP_TIME | Actual Departure Time (local time: hhmm) | int | TRUE | 3.02% | 2722232 | 1,337.32 | 486.08 | 1 | 929 | 1332 | 1735 | 2400 |
# MAGIC | flights_3m | DEP_DELAY | Difference in minutes between scheduled and actual departure time. Early departures show negative numbers. | double | TRUE | 3.02% | 2722232 | 10.36 | 37.86 | -61 | -5 | -1 | 9 | 1988 |
# MAGIC | flights_3m | DEP_DELAY_NEW | Difference in minutes between scheduled and actual departure time. Early departures set to 0. | double | TRUE | 3.02% | 2722232 | 13.03 | 36.79 | 0 | 0 | 0 | 9 | 1988 |
# MAGIC | flights_3m | DEP_DEL15 | Departure Delay Indicator, 15 Minutes or More (1=Yes) | double | TRUE | 3.02% | 2722232 | 0.20 | 0.40 | 0 | 0 | 0 | 0 | 1 |
# MAGIC | flights_3m | DEP_DELAY_GROUP | Departure Delay intervals, every (15 minutes from <-15 to >180) | int | TRUE | 3.02% | 2722232 | 0.1051570917 | 2.094680114 | -2 | -1 | -1 | 0 | 12 |
# MAGIC | flights_3m | DEP_TIME_BLK | CRS Departure Time Block, Hourly Intervals | string | TRUE | 0.00% | 2806942 |  |  | 0001-0559 |  |  |  | 2300-2359 |
# MAGIC | flights_3m | TAXI_OUT | Taxi Out Time, in Minutes | double | TRUE | 3.08% | 2720600 | 16.39104609 | 9.625138436 | 1 | 11 | 14 | 19 | 225 |
# MAGIC | flights_3m | WHEELS_OFF | Wheels Off Time (local time: hhmm) | int | TRUE | 3.08% | 2720600 | 1360.443576 | 486.6322541 | 1 | 944 | 1345 | 1749 | 2400 |
# MAGIC | flights_3m | WHEELS_ON | Wheels On Time (local time: hhmm) | int | TRUE | 3.16% | 2718206 | 1484.907588 | 508.5931725 | 1 | 1108 | 1517 | 1913 | 2400 |
# MAGIC | flights_3m | TAXI_IN | Taxi In Time, in Minutes | double | TRUE | 3.16% | 2718206 | 7.461946593 | 6.101487762 | 1 | 4 | 6 | 9 | 202 |
# MAGIC | flights_3m | CRS_ARR_TIME | CRS Arrival Time (local time: hhmm) | int | TRUE | 0.00% | 2806942 | 1504.75785 | 492.5939251 | 1 | 1120 | 1525 | 1918 | 2400 |
# MAGIC | flights_3m | ARR_TIME | Actual Arrival Time (local time: hhmm) | int | TRUE | 3.16% | 2718206 | 1,490.84 | 512.40 | 1 | 1113 | 1521 | 1919 | 2400 |
# MAGIC | flights_3m | ARR_DELAY | Difference in minutes between scheduled and actual arrival time. Early arrivals show negative numbers. | double | TRUE | 3.32% | 2713628 | 6.24 | 40.53 | -87 | -12 | -3 | 10 | 1971 |
# MAGIC | flights_3m | ARR_DELAY_NEW | Difference in minutes between scheduled and actual arrival time. Early arrivals set to 0. | double | TRUE | 3.32% | 2713628 | 13.41 | 37.03 | 0 | 0 | 0 | 10 | 1971 |
# MAGIC | flights_3m | ARR_DEL15 | Arrival Delay Indicator, 15 Minutes or More (1=Yes) | double | TRUE | 3.32% | 2713628 | 0.21 | 0.41 | 0 | 0 | 0 | 0 | 1 |
# MAGIC | flights_3m | ARR_DELAY_GROUP | Arrival Delay intervals, every (15-minutes from <-15 to >180) | int | TRUE | 3.32% | 2713628 | -0.07815514875 | 2.246637315 | -2 | -1 | -1 | 0 | 12 |
# MAGIC | flights_3m | ARR_TIME_BLK | CRS Arrival Time Block, Hourly Intervals | string | TRUE | 0.00% | 2806942 |  |  | 0001-0559 |  |  |  | 2300-2359 |
# MAGIC | flights_3m | CANCELLED | Cancelled Flight Indicator (1=Yes) | double | TRUE | 0.00% | 2806942 | 0.03 | 0.17 | 0 | 0 | 0 | 0 | 1 |
# MAGIC | flights_3m | CANCELLATION_CODE | Specifies The Reason For Cancellation | string | TRUE | 96.90% | 87002 |  |  | A |  |  |  | D |
# MAGIC | flights_3m | DIVERTED | Diverted Flight Indicator (1=Yes) | double | TRUE | 0.00% | 2806942 | 0.00 | 0.05 | 0 | 0 | 0 | 0 | 1 |
# MAGIC | flights_3m | CRS_ELAPSED_TIME | CRS Elapsed Time of Flight, in Minutes | double | TRUE | 0.00% | 2806938 | 140.5077918 | 74.64903847 | 20 | 85 | 122 | 173 | 718 |
# MAGIC | flights_3m | ACTUAL_ELAPSED_TIME | Elapsed Time of Flight, in Minutes | double | TRUE | 3.32% | 2713628 | 136.7358267 | 73.7178422 | 15 | 82 | 119 | 169 | 766 |
# MAGIC | flights_3m | AIR_TIME | Flight Time, in Minutes | double | TRUE | 3.32% | 2713628 | 112.8941351 | 71.68069171 | 7 | 60 | 94 | 144 | 690 |
# MAGIC | flights_3m | FLIGHTS | Number of Flights | double | TRUE | 0.00% | 2806942 | 1 | 3.97E-17 | 1 | 1 | 1 | 1 | 1 |
# MAGIC | flights_3m | DISTANCE | Distance between airports (miles) | double | TRUE | 0.00% | 2806942 | 807.10 | 594.87 | 31 | 369 | 644 | 1050 | 4983 |
# MAGIC | flights_3m | DISTANCE_GROUP | Distance Intervals, every 250 Miles, for Flight Segment | int | TRUE | 0.00% | 2806942 | 3.700602292 | 2.342032114 | 1 | 2 | 3 | 5 | 11 |
# MAGIC | flights_3m | CARRIER_DELAY | Carrier Delay, in Minutes | double | TRUE | 79.58% | 573164 | 18.28 | 46.31 | 0 | 0 | 3 | 19 | 1971 |
# MAGIC | flights_3m | WEATHER_DELAY | Weather Delay, in Minutes | double | TRUE | 79.58% | 573164 | 3.15 | 22.34 | 0 | 0 | 0 | 0 | 1152 |
# MAGIC | flights_3m | NAS_DELAY | National Air System Delay, in Minutes | double | TRUE | 79.58% | 573164 | 13.46 | 25.74 | 0 | 0 | 3 | 18 | 1101 |
# MAGIC | flights_3m | SECURITY_DELAY | Security Delay, in Minutes | double | TRUE | 79.58% | 573164 | 0.06 | 1.95 | 0 | 0 | 0 | 0 | 241 |
# MAGIC | flights_3m | LATE_AIRCRAFT_DELAY | Late Aircraft Delay, in Minutes | double | TRUE | 79.58% | 573164 | 22.67 | 41.85 | 0 | 0 | 3 | 29 | 1313 |
# MAGIC | flights_3m | FIRST_DEP_TIME | First Gate Departure Time at Origin Airport | int | TRUE | 99.36% | 17984 | 1269.222086 | 499.6115566 | 1 | 825 | 1225 | 1707 | 2359 |
# MAGIC | flights_3m | TOTAL_ADD_GTIME | Total Ground Time Away from Gate for Gate Return or Cancelled Flight | double | TRUE | 99.36% | 17984 | 34.50044484 | 30.1659832 | 1 | 16 | 26 | 42 | 352 |
# MAGIC | flights_3m | LONGEST_ADD_GTIME | Longest Time Away from Gate for Gate Return or Cancelled Flight | double | TRUE | 99.36% | 17984 | 33.95507117 | 28.75309286 | 1 | 15 | 26 | 41 | 271 |
# MAGIC | flights_3m | DIV_AIRPORT_LANDINGS | Number of Diverted Airport Landings | int | FALSE | 0.00% | 2806942 | 0.004401230948 | 0.1462375647 | 0 | 0 | 0 | 0 | 9 |
# MAGIC | flights_3m | DIV_REACHED_DEST | Diverted Flight Reaching Scheduled Destination Indicator (1=Yes) | double | FALSE | 99.78% | 6312 | 0.7252851711 | 0.4464058277 | 0 | 0 | 1 | 1 | 1 |
# MAGIC | flights_3m | DIV_ACTUAL_ELAPSED_TIME | Elapsed Time of Diverted Flight Reaching Scheduled Destination, in Minutes. The ActualElapsedTime column remains NULL for all diverted flights. | double | FALSE | 99.84% | 4578 | 369.4224552 | 185.6975988 | 91 | 251 | 320 | 447 | 1847 |
# MAGIC | flights_3m | DIV_ARR_DELAY | Difference in minutes between scheduled and actual arrival time for a diverted flight reaching scheduled destination. The ArrDelay column remains NULL for all diverted flights. | double | FALSE | 99.84% | 4578 | 208.2158148 | 175.9105074 | -25 | 111 | 160 | 239 | 1799 |
# MAGIC | flights_3m | DIV_DISTANCE | Distance between scheduled destination and final diverted airport (miles). Value will be 0 for diverted flight reaching scheduled destination. | double | FALSE | 99.78% | 6312 | 84.06147022 | 229.2298792 | 0 | 0 | 0 | 25 | 2917 |
# MAGIC | flights_3m | DIV1_AIRPORT | Diverted Airport Code1 | string | FALSE | 99.75% | 6972 |  |  | ABE |  |  |  | YUM |
# MAGIC | flights_3m | DIV1_AIRPORT_ID | Airport ID of Diverted Airport 1. Airport ID is a Unique Key for an Airport | int | FALSE | 99.75% | 6972 | 12740.01893 | 1568.600682 | 10135 | 11292 | 12889 | 14107 | 16218 |
# MAGIC | flights_3m | DIV1_AIRPORT_SEQ_ID | Airport Sequence ID of Diverted Airport 1. Unique Key for Time Specific Information for an Airport | int | FALSE | 99.75% | 6972 | 1274004.442 | 156859.8752 | 1013503 | 1129202 | 1288903 | 1410702 | 1621801 |
# MAGIC | flights_3m | DIV1_WHEELS_ON | Wheels On Time (local time: hhmm) at Diverted Airport Code1 | int | FALSE | 99.75% | 6972 | 1436.232071 | 550.3571348 | 1 | 1031 | 1439 | 1906 | 2400 |
# MAGIC | flights_3m | DIV1_TOTAL_GTIME | Total Ground Time Away from Gate at Diverted Airport Code1 | double | FALSE | 99.75% | 6972 | 25.65146299 | 25.06835027 | 1 | 10 | 18 | 30 | 221 |
# MAGIC | flights_3m | DIV1_LONGEST_GTIME | Longest Ground Time Away from Gate at Diverted Airport Code1 | double | FALSE | 99.75% | 6972 | 20.55536431 | 22.52168221 | 1 | 8 | 13 | 22 | 212 |
# MAGIC | flights_3m | DIV1_WHEELS_OFF | Wheels Off Time (local time: hhmm) at Diverted Airport Code1 | int | FALSE | 99.83% | 4676 | 1469.735244 | 547.5591494 | 1 | 1108 | 1449 | 1919 | 2400 |
# MAGIC | flights_3m | DIV1_TAIL_NUM | Aircraft Tail Number for Diverted Airport Code1 | string | FALSE | 99.83% | 4676 |  |  | N004AA |  |  |  | N998DL |
# MAGIC | flights_3m | DIV2_AIRPORT | Diverted Airport Code2 | string | FALSE | 100.00% | 120 |  |  | ANC |  |  |  | TWF |
# MAGIC | flights_3m | DIV2_AIRPORT_ID | Airport ID of Diverted Airport 2. Airport ID is a Unique Key for an Airport | string | FALSE | 100.00% | 120 | 12180.66667 | 1480.507858 | 10299 | 10821 | 11921 | 13303 | 15389 |
# MAGIC | flights_3m | DIV2_AIRPORT_SEQ_ID | Airport Sequence ID of Diverted Airport 2. Unique Key for Time Specific Information for an Airport | string | FALSE | 100.00% | 120 | 1218069.617 | 148050.3049 | 1029904 | 1082103 | 1192102 | 1330303 | 1538902 |
# MAGIC | flights_3m | DIV2_WHEELS_ON | Wheels On Time (local time: hhmm) at Diverted Airport Code2 | string | FALSE | 100.00% | 120 | 1314.766667 | 724.7783635 | 1025 | 920 | 1430 | 1829 | 952 |
# MAGIC | flights_3m | DIV2_TOTAL_GTIME | Total Ground Time Away from Gate at Diverted Airport Code2 | string | FALSE | 100.00% | 120 | 13.08333333 | 9.30941814 | 10 | 6 | 8 | 19 | 9 |
# MAGIC | flights_3m | DIV2_LONGEST_GTIME | Longest Ground Time Away from Gate at Diverted Airport Code2 | string | FALSE | 100.00% | 120 | 11.55 | 7.58243215 | 10 | 6 | 8 | 16 | 9 |
# MAGIC | flights_3m | DIV2_WHEELS_OFF | Wheels Off Time (local time: hhmm) at Diverted Airport Code2 | string | FALSE | 100.00% | 26 | 1371.846154 | 576.8103808 | 1018 | 1018 | 1129 | 1848 | 824 |
# MAGIC | flights_3m | DIV2_TAIL_NUM | Aircraft Tail Number for Diverted Airport Code2 | string | FALSE | 100.00% | 26 |  |  | N12567 |  |  |  | N993DL |
# MAGIC | flights_3m | DIV3_AIRPORT | Diverted Airport Code3 | string | FALSE | 100.00% | 4 |  |  | ATL |  |  |  | IAH |
# MAGIC | flights_3m | DIV3_AIRPORT_ID | Airport ID of Diverted Airport 3. Airport ID is a Unique Key for an Airport | string | FALSE | 100.00% | 4 | 11331.5 | 1079.067653 | 10397 | 10397 | 10397 | 12266 | 12266 |
# MAGIC | flights_3m | DIV3_AIRPORT_SEQ_ID | Airport Sequence ID of Diverted Airport 3. Unique Key for Time Specific Information for an Airport | string | FALSE | 100.00% | 4 | 1133154 | 107905.6106 | 1039705 | 1039705 | 1039705 | 1226603 | 1226603 |
# MAGIC | flights_3m | DIV3_WHEELS_ON | Wheels On Time (local time: hhmm) at Diverted Airport Code3 | string | FALSE | 100.00% | 4 | 988 | 1081.954404 | 1925 | 51 | 51 | 1925 | 51 |
# MAGIC | flights_3m | DIV3_TOTAL_GTIME | Total Ground Time Away from Gate at Diverted Airport Code3 | string | FALSE | 100.00% | 4 | 5 | 1.154700538 | 4 | 4 | 4 | 6 | 6 |
# MAGIC | flights_3m | DIV3_LONGEST_GTIME | Longest Ground Time Away from Gate at Diverted Airport Code3 | string | FALSE | 100.00% | 4 | 5 | 1.154700538 | 4 | 4 | 4 | 6 | 6 |
# MAGIC | flights_3m | DIV3_WHEELS_OFF | Wheels Off Time (local time: hhmm) at Diverted Airport Code3 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV3_TAIL_NUM | Aircraft Tail Number for Diverted Airport Code3 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_AIRPORT | Diverted Airport Code4 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_AIRPORT_ID | Airport ID of Diverted Airport 4. Airport ID is a Unique Key for an Airport | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_AIRPORT_SEQ_ID | Airport Sequence ID of Diverted Airport 4. Unique Key for Time Specific Information for an Airport | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_WHEELS_ON | Wheels On Time (local time: hhmm) at Diverted Airport Code4 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_TOTAL_GTIME | Total Ground Time Away from Gate at Diverted Airport Code4 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_LONGEST_GTIME | Longest Ground Time Away from Gate at Diverted Airport Code4 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_WHEELS_OFF | Wheels Off Time (local time: hhmm) at Diverted Airport Code4 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV4_TAIL_NUM | Aircraft Tail Number for Diverted Airport Code4 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_AIRPORT | Diverted Airport Code5 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_AIRPORT_ID | Airport ID of Diverted Airport 5. Airport ID is a Unique Key for an Airport | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_AIRPORT_SEQ_ID | Airport Sequence ID of Diverted Airport 5. Unique Key for Time Specific Information for an Airport | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_WHEELS_ON | Wheels On Time (local time: hhmm) at Diverted Airport Code5 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_TOTAL_GTIME | Total Ground Time Away from Gate at Diverted Airport Code5 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_LONGEST_GTIME | Longest Ground Time Away from Gate at Diverted Airport Code5 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_WHEELS_OFF | Wheels Off Time (local time: hhmm) at Diverted Airport Code5 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | flights_3m | DIV5_TAIL_NUM | Aircraft Tail Number for Diverted Airport Code5 | string | FALSE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | weather_3m | YEAR | Calendar year of the observation or summary. | int | TRUE | 0.00% | 30528602 | 2015 | 1.95E-13 | 2015 | 2015 | 2015 | 2015 | 2015 |
# MAGIC | weather_3m | STATION | Station identifier (typically USAF-WBAN or similar code) for the reporting site. | string | TRUE | 0.00% | 30528602 | * | * | * |  |  |  | * |
# MAGIC | weather_3m | DATE | Observation or summary date (YYYYMMDD). | string | TRUE | 0.00% | 30528602 |  |  | 2015-01-01 0:00:00 |  |  |  | 2015-03-31 23:59:00 |
# MAGIC | weather_3m | LATITUDE | Station latitude in decimal degrees; south is negative. | string | TRUE | 0.79% | 30287448 | 37.86678014 | 21.30808618 | -0.0166667 | 33.355 | 41.4057 | 48.784 | 9.993861 |
# MAGIC | weather_3m | LONGITUDE | Station longitude in decimal degrees; west is negative. | string | TRUE | 0.79% | 30287448 | -38.25671724 | 78.96998694 | -0.005456 | -96.0075 | -77.61667 | 15.525683 | 99.9666666 |
# MAGIC | weather_3m | ELEVATION | Station elevation above mean sea level, in meters (when available). | string | TRUE | 0.79% | 30287448 | 355.4689071 | 529.7921915 | -1 | 32.3 | 164.3 | 399.6 | 999.1 |
# MAGIC | weather_3m | NAME | Plain-language station name. | string | TRUE | 0.79% | 30287448 |  |  | 068 BAFFIN BAY POINT OF ROCKS TX, TX US |  |  |  | ZYRYANKA, RS |
# MAGIC | weather_3m | REPORT_TYPE | Code indicating report/source type (e.g., METAR, SYNOP, ASOS). | string | TRUE | 0.00% | 30528602 |  |  | CRN05 |  |  |  | SY-MT |
# MAGIC | weather_3m | SOURCE | Data provider/source flag or lineage indicator. | string | TRUE | 0.00% | 30528602 | 5.011121096 | 1.399728496 | 1 | 4 | 4 | 7 | O |
# MAGIC | weather_3m | HourlyAltimeterSetting | Hourly altimeter setting (pressure reduced to sea level, for aviation). | string | TRUE | 46.11% | 16450738 | 30.06 | 0.29 | 26.05 | 29.91 | 30.08 | 30.24 | 31.1 |
# MAGIC | weather_3m | HourlyDewPointTemperature | Hourly dew point temperature, the saturation temperature at current moisture. | string | TRUE | 17.57% | 25165189 | 30.60 | 21.88 | * | 19 | 31 | 43 | 9s |
# MAGIC | weather_3m | HourlyDryBulbTemperature | Hourly ambient (air) temperature measured in shelter/exposure. | string | TRUE | 2.13% | 29877539 | 39.47 | 23.04 | * | 27 | 39 | 54 | 9s |
# MAGIC | weather_3m | HourlyPrecipitation | Precipitation amount accumulated during the hour (liquid equivalent). | string | TRUE | 87.12% | 3932455 | 0.01 | 0.05 | * | 0 | 0 | 0 | T |
# MAGIC | weather_3m | HourlyPresentWeatherType | Codes describing present weather (e.g., rain, snow, fog) observed in the hour. | string | TRUE | 87.13% | 3929028 |  |  | * * * \|* * * \| |  |  |  | \|\|s |
# MAGIC | weather_3m | HourlyPressureChange | Change in station or sea-level pressure over the standard interval. | string | TRUE | 72.40% | 8426055 | 0.00 | 0.05 | + | -0.03 | 0 | 0.03 | 1.48 |
# MAGIC | weather_3m | HourlyPressureTendency | Code describing the character of pressure change (rising/falling/steady). | string | TRUE | 71.42% | 8724725 | 4.85 | 2.75 | 0 | 2 | 5 | 7 | 9 |
# MAGIC | weather_3m | HourlyRelativeHumidity | Hourly relative humidity, typically derived from temperature and dew point. | string | TRUE | 17.59% | 25157107 | 72.99 | 20.12 | * | 61 | 77 | 88 | 99 |
# MAGIC | weather_3m | HourlySkyConditions | Cloud/sky condition codes, including ceilings and coverage layers. | string | TRUE | 47.42% | 16052227 | 29.51 | 26.96 | * | 8 | 26 | 41 | X:10s 0s |
# MAGIC | weather_3m | HourlySeaLevelPressure | Hourly sea-level pressure estimate derived from station pressure and metadata. | string | TRUE | 63.77% | 11060358 | 30.02849163 | 0.3419754474 | 27.49 | 29.85 | 30.05 | 30.24 | 32.18 |
# MAGIC | weather_3m | HourlyStationPressure | Hourly pressure measured at station elevation. | string | TRUE | 49.22% | 15503090 | 28.84963413 | 1.676682121 | 13.44s | 28.66 | 29.37 | 29.84 | 32.02 |
# MAGIC | weather_3m | HourlyVisibility | Prevailing horizontal visibility reported for the hour. | string | TRUE | 34.64% | 19952566 | 8.454742865 | 5.562695303 | * | 6.21 | 9 | 10 | 99.42 |
# MAGIC | weather_3m | HourlyWetBulbTemperature | Hourly wet-bulb temperature (thermodynamic proxy for moisture/heat). | string | TRUE | 50.06% | 15247212 | 34.87598412 | 19.43422772 | * | 24 | 35 | 46 | 99 |
# MAGIC | weather_3m | HourlyWindDirection | Hourly wind direction in degrees from true north (calm/variable when applicable). | string | TRUE | 14.25% | 26177554 | 168.8361046 | 118.2680932 | * | 50 | 180 | 270 | VRB |
# MAGIC | weather_3m | HourlyWindGustSpeed | Highest instantaneous wind speed (gust) observed in/near the hour. | string | TRUE | 92.83% | 2187900 | 25.72 | 8.04 | * | 20 | 24 | 30 | 99s |
# MAGIC | weather_3m | HourlyWindSpeed | Mean wind speed for the hour. | string | TRUE | 13.25% | 26483988 | 8.24 | 8.53 | * | 3 | 7 | 11 | 9s |
# MAGIC | weather_3m | Sunrise | Local time of sunrise for the station/date (if available). | string | TRUE | 99.50% | 151432 | 689.3005904 | 80.20007531 | 148 | 636 | 703 | 727 | 1738 |
# MAGIC | weather_3m | Sunset | Local time of sunset for the station/date (if available). | string | TRUE | 99.50% | 151442 | 1765.359497 | 67.61177621 | 155 | 1725 | 1753 | 1819 | 2219 |
# MAGIC | weather_3m | DailyAverageDewPointTemperature | Day's mean dew point temperature. | string | TRUE | 99.91% | 28740 | 29.83 | 17.79 | -1 | 19 | 29 | 41 | 9 |
# MAGIC | weather_3m | DailyAverageDryBulbTemperature | Day’s mean air temperature (average of observations or max/min). | string | TRUE | 99.66% | 104892 | 39.19321272 | 19.24628161 | -1 | 27 | 40 | 53 | 9s |
# MAGIC | weather_3m | DailyAverageRelativeHumidity | Day’s mean relative humidity. | string | TRUE | 99.91% | 28971 | 67.18587553 | 16.42829847 | 10 | 56 | 69 | 80 | 99 |
# MAGIC | weather_3m | DailyAverageSeaLevelPressure | Day’s mean sea-level pressure. | string | TRUE | 99.91% | 28706 | 30.13167839 | 0.2232649633 | 28.52 | 30 | 30.13 | 30.27 | 30.99 |
# MAGIC | weather_3m | DailyAverageStationPressure | Day’s mean station pressure. | string | TRUE | 99.70% | 90368 | 28.78958492 | 1.659583262 | 20.46 | 28.56 | 29.38 | 29.88 | 30.91 |
# MAGIC | weather_3m | DailyAverageWetBulbTemperature | Day’s mean wet-bulb temperature. | string | TRUE | 99.91% | 28740 | 36.67727905 | 16.2869376 | -1 | 27 | 37 | 47 | 9 |
# MAGIC | weather_3m | DailyAverageWindSpeed | Day’s mean wind speed. | string | TRUE | 99.70% | 90784 | 7.840166769 | 4.469818943 | 0 | 4.7 | 7.1 | 10.3 | 91.2 |
# MAGIC | weather_3m | DailyCoolingDegreeDays | Cooling degree days for the date relative to a base (often 65°F/18°C). | string | TRUE | 99.66% | 104892 | 0.5601572662 | 2.352891217 | 0 | 0 | 0 | 0 | 9s |
# MAGIC | weather_3m | DailyDepartureFromNormalAverageTemperature | Difference between day’s mean temperature and climatological normal. | string | TRUE | 99.69% | 94173 | -0.04728871467 | 10.68449798 | -0.1 | -6.9 | 1.2 | 7.4 | 9.9s |
# MAGIC | weather_3m | DailyHeatingDegreeDays | Heating degree days for the date relative to a base (often 65°F/18°C). | string | TRUE | 99.66% | 104892 | 26.36694454 | 18.31239188 | 0 | 12 | 25 | 38 | 9s |
# MAGIC | weather_3m | DailyMaximumDryBulbTemperature | Highest air temperature observed for the day. | string | TRUE | 99.66% | 104921 | 49.47903862 | 20.3602781 | -1 | 35 | 50 | 65 | 9s |
# MAGIC | weather_3m | DailyMinimumDryBulbTemperature | Lowest air temperature observed for the day. | string | TRUE | 99.66% | 104913 | 28.44455036 | 19.27668453 | -1 | 17 | 29 | 41 | 9s |
# MAGIC | weather_3m | DailyPeakWindDirection | Direction of the day’s peak (maximum) wind. | string | TRUE | 99.71% | 87912 | 207.019033 | 109.3512105 | 10 | 120 | 220 | 310 | 360s |
# MAGIC | weather_3m | DailyPeakWindSpeed | Speed of the day’s peak (maximum) wind. | string | TRUE | 99.70% | 91648 | 37.07189213 | 171.4595557 | * | 17 | 23 | 29 | 9s |
# MAGIC | weather_3m | DailyPrecipitation | Total liquid precipitation for the day (may include trace flags). | string | TRUE | 99.66% | 104986 | 0.07765873772 | 0.2527251531 | 0 | 0 | 0 | 0.02 | Ts |
# MAGIC | weather_3m | DailySnowDepth | Snow depth on the ground at observation time for the day. | string | TRUE | 99.88% | 37933 | 1.484023378 | 4.20832185 | 0 | 0 | 0 | 0 | T |
# MAGIC | weather_3m | DailySnowfall | New snow amount (liquid equivalent excluded) accumulated during the day. | string | TRUE | 99.88% | 36824 | 0.1674885979 | 0.8737347617 | 0 | 0 | 0 | 0 | T |
# MAGIC | weather_3m | DailySustainedWindDirection | Direction associated with the highest sustained wind for the day. | string | TRUE | 99.70% | 90899 | 207.6854322 | 109.5457781 | * | 120 | 220 | 310 | 360s |
# MAGIC | weather_3m | DailySustainedWindSpeed | Highest sustained (averaged) wind speed observed for the day. | string | TRUE | 99.70% | 91710 | 29.23364744 | 158.3761863 | * | 13 | 17 | 22 | 9s |
# MAGIC | weather_3m | DailyWeather | Daily weather summary codes/flags (events such as fog, thunder, etc.). | string | TRUE | 99.69% | 93634 |  |  | BLSN |  |  |  | UP SN BR |
# MAGIC | weather_3m | MonthlyAverageRH | Monthly mean relative humidity. | string | TRUE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | weather_3m | MonthlyDaysWithGT001Precip | Number of days in the month with precipitation ≥ 0.01 in (0.25 mm). | string | TRUE | 99.99% | 3240 | 8.20606267 | 4.66157339 | 0 | 5 | 8 | 11 | 9s |
# MAGIC | weather_3m | MonthlyDaysWithGT010Precip | Number of days in the month with precipitation ≥ 0.10 in (2.54 mm). | string | TRUE | 99.99% | 3240 | 4.16187291 | 3.344982886 | 0 | 2 | 3 | 6 | 9s |
# MAGIC | weather_3m | MonthlyDaysWithGT32Temp | Count of days with maximum temperature > 32°F (0°C) or threshold per dataset. | string | TRUE | 99.99% | 3182 | 17.56813081 | 11.26859197 | 0 | 6 | 22 | 28 | 9s |
# MAGIC | weather_3m | MonthlyDaysWithGT90Temp | Count of days with maximum temperature > 90°F (32.2°C). | string | TRUE | 99.99% | 3179 | 0.1188899401 | 0.9266776061 | 0 | 0 | 0 | 0 | 9s |
# MAGIC | weather_3m | MonthlyDaysWithLT0Temp | Count of days with minimum temperature < 0°F (−17.8°C). | string | TRUE | 99.99% | 3182 | 2.632711167 | 5.149123609 | 0 | 0 | 0 | 3 | 9 |
# MAGIC | weather_3m | MonthlyDaysWithLT32Temp | Count of days with minimum temperature < 32°F (0°C). | string | TRUE | 99.99% | 3182 | 6.291106846 | 8.375604437 | 0 | 0 | 2 | 10 | 9s |
# MAGIC | weather_3m | MonthlyDepartureFromNormalAverageTemperature | Difference between monthly mean temperature and its normal. | string | TRUE | 99.99% | 3106 | -0.4599548241 | 5.718595165 | -0.1 | -3.6 | 0.2 | 3.9 | 9.9 |
# MAGIC | weather_3m | MonthlyDepartureFromNormalCoolingDegreeDays | Departure of monthly CDD from normal CDD. | string | TRUE | 99.99% | 3085 | 2.804538088 | 20.82090527 | -1 | 0 | 0 | 0 | 99 |
# MAGIC | weather_3m | MonthlyDepartureFromNormalHeatingDegreeDays | Departure of monthly HDD from normal HDD. | string | TRUE | 99.99% | 3085 | 11.16661264 | 161.0213937 | -1 | -111 | -4 | 102 | 99 |
# MAGIC | weather_3m | MonthlyDepartureFromNormalMaximumTemperature | Departure of monthly mean of daily maxima from normal. | string | TRUE | 99.99% | 3106 | -0.002842377261 | 5.968592947 | -0.1 | -3.6 | 0.2 | 4.4 | 9.9 |
# MAGIC | weather_3m | MonthlyDepartureFromNormalMinimumTemperature | Departure of monthly mean of daily minima from normal. | string | TRUE | 99.99% | 3106 | -0.9424975799 | 5.802626046 | -0.1 | -3.9 | 0 | 3.2 | 9.8 |
# MAGIC | weather_3m | MonthlyDepartureFromNormalPrecipitation | Departure of monthly total precipitation from normal. | string | TRUE | 99.99% | 2775 | -0.2074712644 | 9.806389071 | -0.01 | -1.08 | -0.44 | 0.15 | 7.16 |
# MAGIC | weather_3m | MonthlyDewpointTemperature | Monthly mean dew point temperature. | string | TRUE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | weather_3m | MonthlyGreatestPrecip | Greatest 24-hour (or event) liquid precipitation amount in the month. | string | TRUE | 99.99% | 2788 | 0.81481409 | 0.8532176952 | 0 | 0.24 | 0.56 | 1.12 | T |
# MAGIC | weather_3m | MonthlyGreatestPrecipDate | Date on which the monthly greatest precipitation occurred. | string | TRUE | 99.99% | 2683 |  |  | 01-01 |  |  |  | 31-31 |
# MAGIC | weather_3m | MonthlyGreatestSnowDepth | Maximum snow depth on ground observed in the month. | string | TRUE | 100.00% | 678 | 5.631259484 | 6.977971859 | 0 | 0 | 4 | 8 | T |
# MAGIC | weather_3m | MonthlyGreatestSnowDepthDate | Date of the maximum snow depth. | string | TRUE | 100.00% | 516 | 13.50775194 | 10.0641601 | 1 | 5 | 10 | 23 | 31 |
# MAGIC | weather_3m | MonthlyGreatestSnowfall | Largest single-day snowfall in the month. | string | TRUE | 100.00% | 687 | 3.243438914 | 3.779873749 | 0 | 0.3 | 2.2 | 4.9 | T |
# MAGIC | weather_3m | MonthlyGreatestSnowfallDate | Date of the largest single-day snowfall. | string | TRUE | 100.00% | 585 |  |  | 01-01 |  |  |  | 31-31 |
# MAGIC | weather_3m | MonthlyMaxSeaLevelPressureValue | Highest sea-level pressure observed in the month. | string | TRUE | 99.99% | 2766 | 30.62984 | 0.2071983334 | 26.27s | 30.49 | 30.63 | 30.75 | 31.3 |
# MAGIC | weather_3m | MonthlyMaxSeaLevelPressureValueDate | Date of the monthly maximum sea-level pressure. | string | TRUE | 99.99% | 2767 | 13.35345139 | 9.239371658 | 1 | 6 | 11 | 23 | 31 |
# MAGIC | weather_3m | MonthlyMaxSeaLevelPressureValueTime | Time of the monthly maximum sea-level pressure. | string | TRUE | 99.99% | 2767 | 1039.98699 | 618.5503758 | 0 | 854 | 1009 | 1118 | 9999 |
# MAGIC | weather_3m | MonthlyMaximumTemperature | Highest daily maximum temperature during the month. | string | TRUE | 99.99% | 3231 | 49.16333955 | 17.5495602 | -0.4 | 36.5 | 49.6 | 61.9 | 9.8 |
# MAGIC | weather_3m | MonthlyMeanTemperature | Mean temperature for the month (often (Tmax+Tmin)/2 or obs average). | string | TRUE | 99.99% | 3231 | 38.68358672 | 17.03628783 | -0.1 | 27.1 | 38.8 | 50.2 | 9.9 |
# MAGIC | weather_3m | MonthlyMinSeaLevelPressureValue | Lowest sea-level pressure observed in the month. | string | TRUE | 99.99% | 2763 | 29.60834729 | 0.2071163796 | 25.51s | 29.52 | 29.63 | 29.75 | 30.65s |
# MAGIC | weather_3m | MonthlyMinSeaLevelPressureValueDate | Date of the monthly minimum sea-level pressure. | string | TRUE | 99.99% | 2764 | 17.53437048 | 9.963264679 | 1 | 7 | 20 | 26 | 31 |
# MAGIC | weather_3m | MonthlyMinSeaLevelPressureValueTime | Time of the monthly minimum sea-level pressure. | string | TRUE | 99.99% | 2764 | 1217.671491 | 704.6791144 | 0 | 547 | 1421 | 1650 | 9999 |
# MAGIC | weather_3m | MonthlyMinimumTemperature | Lowest daily minimum temperature during the month. | string | TRUE | 99.99% | 3231 | 28.19150124 | 16.91208058 | -0.2 | 17.3 | 27.3 | 39.5 | 9.9 |
# MAGIC | weather_3m | MonthlySeaLevelPressure | Monthly mean sea-level pressure. | string | TRUE | 99.99% | 2731 | 30.12329916 | 0.1660415838 | 25.96 | 30.08 | 30.13 | 30.18 | 30.44 |
# MAGIC | weather_3m | MonthlyStationPressure | Monthly mean station pressure. | string | TRUE | 99.99% | 2743 | 28.80917244 | 1.652634462 | 20.85 | 28.68 | 29.41 | 29.91 | 30.22 |
# MAGIC | weather_3m | MonthlyTotalLiquidPrecipitation | Total liquid precipitation for the month. | string | TRUE | 99.99% | 3134 | 6.084131148 | 38.71726812 | 0 | 0.57 | 1.46 | 3.36 | T |
# MAGIC | weather_3m | MonthlyTotalSnowfall | Total snowfall for the month. | string | TRUE | 100.00% | 584 | 10.13646833 | 10.85293005 | 0.1 | 2.8 | 7.1 | 13.5 | T |
# MAGIC | weather_3m | MonthlyWetBulb | Monthly mean wet-bulb temperature. | string | TRUE | 100.00% | 0 |  |  |  |  |  |  |  |
# MAGIC | weather_3m | AWND | Average daily wind speed for the period (often from GHCN-D “AWND”). | string | TRUE | 99.99% | 2675 | 7.95211215 | 2.903594062 | 0.2 | 6.3 | 8.1 | 9.6 | 9.8 |
# MAGIC | weather_3m | CDSD | Count of days since last snowfall or cold-season day metric (dataset-specific). | string | TRUE | 99.99% | 3145 | 26.63624801 | 123.6277731 | 0 | 0 | 0 | 0 | 99 |
# MAGIC | weather_3m | CLDD | Cooling degree days for the month/period (GHCN “CLDD”). | string | TRUE | 99.99% | 3169 | 16.45724203 | 62.8073716 | 0 | 0 | 0 | 0 | 99 |
# MAGIC | weather_3m | DSNW | Number of days with snow on ground (dataset-specific GHCN metric). | string | TRUE | 100.00% | 994 | 1.688128773 | 2.473895925 | 0 | 0 | 1 | 3 | 9 |
# MAGIC | weather_3m | HDSD | Count of days since last snow depth event or heating-season day metric (dataset-specific). | string | TRUE | 99.99% | 2957 | 3600.247886 | 2150.566264 | 0 | 1967 | 3449 | 4938 | 999 |
# MAGIC | weather_3m | HTDD | Heating degree days for the month/period (GHCN “HTDD”). | string | TRUE | 99.99% | 3169 | 802.9359419 | 467.9947559 | 0 | 450 | 786 | 1134 | 999 |
# MAGIC | weather_3m | NormalsCoolingDegreeDay | Climatological normal CDD for the period/location. | string | TRUE | 99.99% | 3139 | -1391.138898 | 2999.274574 | -7777 | 0 | 0 | 2 | 99 |
# MAGIC | weather_3m | NormalsHeatingDegreeDay | Climatological normal HDD for the period/location. | string | TRUE | 99.99% | 3139 | 741.1723479 | 772.4074835 | -7777 | 455 | 791 | 1078 | 999 |
# MAGIC | weather_3m | ShortDurationEndDate005 | End date/time stamp for 5-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 5:52 |  |  |  | 2015-03-31 22:54 |
# MAGIC | weather_3m | ShortDurationEndDate010 | End date/time stamp for 10-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 5:52 |  |  |  | 2015-03-31 23:03 |
# MAGIC | weather_3m | ShortDurationEndDate015 | End date/time stamp for 15-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 5:52 |  |  |  | 2015-03-31 23:07 |
# MAGIC | weather_3m | ShortDurationEndDate020 | End date/time stamp for 20-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 5:52 |  |  |  | 2015-03-31 23:07 |
# MAGIC | weather_3m | ShortDurationEndDate030 | End date/time stamp for 30-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 4:22 |  |  |  | 2015-03-31 23:13 |
# MAGIC | weather_3m | ShortDurationEndDate045 | End date/time stamp for 45-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 4:22 |  |  |  | 2015-03-31 23:28 |
# MAGIC | weather_3m | ShortDurationEndDate060 | End date/time stamp for 60-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 4:22 |  |  |  | 2015-03-31 23:47 |
# MAGIC | weather_3m | ShortDurationEndDate080 | End date/time stamp for 80-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2450 |  |  | 2015-01-01 1:30 |  |  |  | 2015-03-31 23:57 |
# MAGIC | weather_3m | ShortDurationEndDate100 | End date/time stamp for 100-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2450 |  |  | 2015-01-01 1:52 |  |  |  | 2015-03-31 23:57 |
# MAGIC | weather_3m | ShortDurationEndDate120 | End date/time stamp for 120-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 2:17 |  |  |  | 2015-03-31 23:57 |
# MAGIC | weather_3m | ShortDurationEndDate150 | End date/time stamp for 150-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2451 |  |  | 2015-01-01 2:52 |  |  |  | 2015-03-31 23:57 |
# MAGIC | weather_3m | ShortDurationEndDate180 | End date/time stamp for 180-minute maximum short-duration precipitation event in month. | string | TRUE | 99.99% | 2445 |  |  | 2015-01-01 2:52 |  |  |  | 2015-03-31 23:57 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue005 | Maximum 5-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.06802937576 | 0.09080677374 | 0.01 | 0.02 | 0.04 | 0.08 | 1.94 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue010 | Maximum 10-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.09902896777 | 0.1248344029 | 0.01 | 0.03 | 0.05 | 0.12 | 2.07 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue015 | Maximum 15-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.1224432522 | 0.1489932302 | 0.01 | 0.04 | 0.07 | 0.15 | 2.11 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue020 | Maximum 20-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.1403467972 | 0.1672270914 | 0.01 | 0.04 | 0.08 | 0.17 | 2.11 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue030 | Maximum 30-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.1716166187 | 0.1964373498 | 0.01 | 0.06 | 0.1 | 0.21 | 2.11 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue045 | Maximum 45-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.2090651907 | 0.2313954074 | 0.01 | 0.07 | 0.13 | 0.26 | 2.11 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue060 | Maximum 60-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.2434512906 | 0.2599247711 | 0.01 | 0.09 | 0.16 | 0.3 | 2.16 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue080 | Maximum 80-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2450 | 0.2767102041 | 0.2927267816 | 0.01 | 0.1 | 0.19 | 0.35 | 2.73 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue100 | Maximum 100-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2450 | 0.308999183 | 0.321015863 | 0.01 | 0.11 | 0.21 | 0.39 | 3.22 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue120 | Maximum 120-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.3414756871 | 0.3466336477 | 0.01 | 0.13 | 0.23 | 0.44 | 3.77 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue150 | Maximum 150-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2451 | 0.3744675643 | 0.3759020607 | 0.01 | 0.14 | 0.26 | 0.48 | 4.41 |
# MAGIC | weather_3m | ShortDurationPrecipitationValue180 | Maximum 180-minute precipitation amount observed in the month. | string | TRUE | 99.99% | 2445 | 0.4134482759 | 0.4092896311 | 0.01 | 0.15 | 0.29 | 0.53 | 5.27 |
# MAGIC | weather_3m | REM | Free-text remarks or metadata notes associated with the record. | string | TRUE | 13.17% | 26506948 |  |  | AWY047NLB SA 0000 AUTO8 M M M M/02/00/2305/M/ M 05MM= |  |  |  | SYN99984542 31670 40000 10170 20098 / / / / / / / / / // // //VFR //VFR //VFR /VFR/ /VFR/ /VFR/ /VFR/ /VFR/ /VFR/ 01 050/ 050/ 050/ 080/ 080/ 1/2///BKN040/ 1/2///SCT030/ 100/ 120/ 20G25KT/ 20G25KT/ 22N 23N///BKN025/ 23N///SCT025 24N/ 32N 37350 400MB/ 57W///CARIBBEAN///GULF 70 75W///BKN030 75W///BKN030/ ALL AND AND ATLANTIC ATLC BARRANQUILLA BKN030 BKN060/ BKN060/ BKN080/ BTN COLD CONDS/ DOMINGO E END FIR FIR FIR FIR FIR FIR FIR// FIR///CURACAO FIR///NRN FIR///PORT/AU/PRINCE FL200/ FL200/ GTR HISPANIOLA/ ICE IFR IMPLY ISLAND JUAN LLWS LYRD LYRD MAIQUETIA MEXICO MIAMI MVFR N NEW NLY NRN NRN NWD///BKN025 NWLY NWRN OCNL OF OF OF OR OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK///VFR///FRQ OVC060/ OVC100/ OVR OVR PANAMA PART PIARCO RDG S S SAN SANTO SCT SCT SCT SCT SCT SCT025 SERN SEV SEV SFC SFC/ SHRA/ SHRA/ SHRA/ SHRA/ SHRA/ SKC/ SRN STNR SWRN SYNOPSIS///LRG TO TO TO TOPS TOPS TOPS TOPS TOPS TOPS TOPS TOPS TOPS TS TURB VIS W W WDLY WDLY WND WND WND/ WND/ WND/ WNDS WRN WTRS///BKN03 |
# MAGIC | weather_3m | BackupDirection | Direction measurement from a designated backup sensor/system (if primary unavailable). | string | TRUE | 98.30% | 518173 |  |  | E |  |  |  | WSW |
# MAGIC | weather_3m | BackupDistance | Distance to a backup station or sensor used for substitution. | string | TRUE | 98.29% | 520992 | 120.9026271 | 716.4806377 | 0.02 | 0.5 | 1 | 2.45 | 7500 |
# MAGIC | weather_3m | BackupDistanceUnit | Unit for backup distance (e.g., km or miles). | string | TRUE | 98.29% | 520992 |  |  | ft |  |  |  | yd |
# MAGIC | weather_3m | BackupElements | Elements/parameters sourced from the backup station/sensor. | string | TRUE | 98.27% | 527470 |  |  | ALL ELEMENTS |  |  |  | TMAX, TMIN, PRECIP |
# MAGIC | weather_3m | BackupElevation | Elevation of the backup station/sensor. | string | TRUE | 98.68% | 402038 | 1392.341876 | 1706.910395 | 0 | 194 | 723 | 1571 | 980 |
# MAGIC | weather_3m | BackupEquipment | Description or code for backup instrumentation used. | string | TRUE | 98.34% | 507639 |  |  | AWPAG, SNOWSTICK, SNOWBOARD |  |  |  | VSALHMP45C, TR-5251-HT |
# MAGIC | weather_3m | BackupLatitude | Latitude of the backup station/sensor. | string | TRUE | 98.71% | 394719 | 41.64417819 | 9.663086174 | 13.4678 | 36.8707 | 40.9386 | 45.4551 | 71.2869 |
# MAGIC | weather_3m | BackupLongitude | Longitude of the backup station/sensor. | string | TRUE | 98.71% | 394719 | -104.7653941 | 24.37958517 | -100.6767 | -116.2101 | -97.4855 | -86.2808 | -99.9692 |
# MAGIC | weather_3m | BackupName | Name/identifier of the backup station/sensor. | string | TRUE | 98.22% | 544771 |  |  | AIR NATIONAL GUARD |  |  |  | WSO NOME |
# MAGIC | weather_3m | WindEquipmentChangeDate | Date when wind instrumentation was changed or reconfigured. | string | TRUE | 93.91% | 1859517 |  |  | 2002-09-24 |  |  |  | 2009-12-13 |
# MAGIC | otpw_v2 | crs_depart_ts_local | Scheduled depart time in local timestamp | timestamp | TRUE | 0 |  |  |  |  |  |  |  |  |
# MAGIC | otpw_v2 | crs_depart_ts_utc | Scheduled depart time in utc timestamp | timestamp | TRUE | 0 |  |  |  |  |  |  |  |  |
# MAGIC | otpw_v2 | mapped_station | Mapped station ID based on mapping table | string | TRUE | 0 | 1403471 | 7.28E+10 | 2.53E+09 | 70026027502 | 7.23E+10 | 7.24E+10 | 7.25E+10 | 99769999999 |
# MAGIC | otpw_v2 | station_miles | # of miles the weather station is away from airport. When it's exactly zero, it's null | double | TRUE | 0.9259877831 | 103874 | 0.311389973 | 0.6374425537 | 0.0054888235 | 0.02989137444 | 0.02989137444 | 0.6884298761 | 3.643752309 |
# MAGIC | otpw_v2 | crs_depart_unixts_utc | Scheduled depart time in utc unix timestamp | bigint | TRUE | 0 | 1403471 | 1.42E+09 | 2256850.36 | 1420060200 | 1422065580 | 1424108400 | 1426015800 | 1427880000 |
# MAGIC | otpw_v2 | time_after | Lower bound unix timestamp when picking weather data | bigint | TRUE | 0 | 1403471 | 1.42E+09 | 2256850.36 | 1420045800 | 1422051180 | 1424094000 | 1426001400 | 1427865600 |
# MAGIC | otpw_v2 | time_before | Upper bound unix timestamp when picking weather data | bigint | TRUE | 0 | 1403471 | 1.42E+09 | 2256850.36 | 1420053000 | 1422058380 | 1424101200 | 1426008600 | 1427872800 |
# MAGIC | otpw_v2 | DATE_unixts | Weather's data field in unix timestamp | bigint | TRUE | 0.0005194264791 | 1402742 | 1.42E+09 | 2256170.483 | 1420084560 | 1422057540 | 1424098560 | 1426006380 | 1427846340 |
# MAGIC | otpw_v2 | rn | Ranking, used for picking the most recent weather data within the bounds | int | TRUE | 0 | 1403471 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
# MAGIC | otpw_v2 | tail_idx | Tail index. Used for finding the previous flight | int | TRUE | 0 | 1403471 | 211.6171456 | 371.4094722 | 1 | 84 | 172 | 276 | 8190 |
# MAGIC | otpw_v2 | PREV_CRS_DEP_TIME | Previous flight's scheduled departure time | int | TRUE | 0.003250512479 | 1398909 | 1326.037636 | 473.8565917 | 1 | 920 | 1320 | 1725 | 2359 |
# MAGIC | otpw_v2 | PREV_DEP_TIME | Previous flight's actual departure time | int | TRUE | 0.03340574903 | 1356587 | 1335.721682 | 485.5926754 | 1 | 928 | 1331 | 1734 | 2400 |
# MAGIC | otpw_v2 | PREV_crs_depart_unixts_utc | Previous flight's scheduled departure time in unix timestamp | bigint | TRUE | 0.003250512479 | 1398909 | 1.42E+09 | 2250751.178 | 1420060200 | 1422060900 | 1424100600 | 1426003800 | 1427874660 |
# MAGIC | otpw_v2 | PREV_DEP_DELAY | Previous flight's departure delay in minutes | double | TRUE | 0.03340574903 | 1356587 | 10.36427446 | 37.85352085 | -61 | -5 | -1 | 9 | 1988 |
# MAGIC | otpw_v2 | PREV_DEP_DELAY_NEW | Previous flight's depature delay in minutes, but early departures are zero | double | TRUE | 0.03340574903 | 1356587 | 13.02745714 | 36.77375832 | 0 | 0 | 0 | 9 | 1988 |
# MAGIC | otpw_v2 | PREV_DEP_DELAY_GROUP | Previous flight's departure delay group | int | TRUE | 0.03340574903 | 1356587 | 0.1053039724 | 2.095042405 | -2 | -1 | -1 | 0 | 12 |
# MAGIC | otpw_v2 | PREV_ARR_DELAY_GROUP | Previous flight's arrival delay group | int | TRUE | 0.0364553311 | 1352307 | -0.07766061996 | 2.247226051 | -2 | -1 | -1 | 0 | 12 |
# MAGIC | otpw_v2 | PREV_CRS_ARR_TIME | Previous flight's scheduled arrival time | int | TRUE | 0.003250512479 | 1398909 | 1503.723253 | 491.7178493 | 1 | 1120 | 1525 | 1916 | 2400 |
# MAGIC | otpw_v2 | PREV_ARR_TIME | Previous flight's actual arrival time | int | TRUE | 0.03483434998 | 1354582 | 1489.856861 | 511.4789347 | 1 | 1112 | 1520 | 1917 | 2400 |
# MAGIC | otpw_v2 | PREV_ARR_DELAY | Previous flight's arrival delay in minutes | double | TRUE | 0.0364553311 | 1352307 | 6.249018159 | 40.52184781 | -87 | -12 | -3 | 10 | 1971 |
# MAGIC | otpw_v2 | PREV_CANCELLED | Previous flight's cancelled status | double | TRUE | 0.003250512479 | 1398909 | 0.03106778211 | 0.1735009987 | 0 | 0 | 0 | 0 | 1 |
# MAGIC | otpw_v2 | PREV_DIVERTED | Previous flight's diverted status | double | TRUE | 0.003250512479 | 1398909 | 0.002245321175 | 0.04733161005 | 0 | 0 | 0 | 0 | 1 |
# MAGIC | otpw_v2 | PREV_ORIGIN | Previous flight's origin airport ID | string | TRUE | 0.003250512479 | 1398909 |  |  | ABE |  |  |  | YUM |
# MAGIC | otpw_v2 | PREV_DEST | Previous flight's destination airport ID | string | TRUE | 0.003250512479 | 1398909 |  |  | ABE |  |  |  | YUM |
# MAGIC | otpw_v2 | PREV_OP_CARRIER_FL_NUM | Previous flight's carrier flight number | int | TRUE | 0.003250512479 | 1398909 | 2244.858591 | 1793.352858 | 1 | 752 | 1716 | 3466 | 9794 |
# MAGIC | otpw_v2 | PREV_DISTANCE | Previous flight's distation from origin to destination | double | TRUE | 0.003250512479 | 1398909 | 806.3177419 | 594.0297248 | 31 | 369 | 643 | 1050 | 4983 |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix B: Summary Statistics (1-year custom joined data)
# MAGIC
# MAGIC | Column_Name | Data_Type | Non_Null_Count | Null_Count | Null_Pct | Min | Max | Mean | Std_Dev | Distinct_Values |
# MAGIC |-------------|-----------|----------------|------------|----------|-----|-----|------|---------|-----------------|
# MAGIC | CANCELLATION_CODE | string | 134,925 | 7,287,112 | 98.18 | null | null | null | null | 4 |
# MAGIC | HourlyPresentWeatherType | string | 796,078 | 6,625,959 | 89.27 | null | null | null | null | 710 |
# MAGIC | PREV_ORIGIN | string | 7,398,309 | 23,728 | 0.32 | null | null | null | null | 353 |
# MAGIC | PREV_DEST | string | 7,398,309 | 23,728 | 0.32 | null | null | null | null | 353 |
# MAGIC | TAIL_NUM | string | 7,404,200 | 17,837 | 0.24 | null | null | null | null | 5563 |
# MAGIC | DATE | timestamp | 7,413,807 | 8,230 | 0.11 | null | null | null | null | null |
# MAGIC | YEAR | int | 7,422,037 | - | 0 | 2019 | 2019 | 2019 | 0 | null |
# MAGIC | QUARTER | int | 7,422,037 | - | 0 | 1 | 4 | 2.53 | 1.11 | null |
# MAGIC | MONTH | int | 7,422,037 | - | 0 | 1 | 12 | 6.58 | 3.4 | null |
# MAGIC | DAY_OF_MONTH | int | 7,422,037 | - | 0 | 1 | 31 | 15.73 | 8.76 | null |
# MAGIC | DAY_OF_WEEK | int | 7,422,037 | - | 0 | 1 | 7 | 3.94 | 2 | null |
# MAGIC | FL_DATE | timestamp | 7,422,037 | - | 0 | null | null | null | null | null |
# MAGIC | OP_UNIQUE_CARRIER | string | 7,422,037 | - | 0 | null | null | null | null | 16 |
# MAGIC | OP_CARRIER_AIRLINE_ID | int | 7,422,037 | - | 0 | 19393 | 20452 | 19986.76 | 374.75 | null |
# MAGIC | OP_CARRIER_FL_NUM | int | 7,422,037 | - | 0 | 1 | 7933 | 2557.2 | 1799.41 | null |
# MAGIC | ORIGIN_AIRPORT_ID | int | 7,422,037 | - | 0 | 10135 | 16869 | 12648.88 | 1523.85 | null |
# MAGIC | ORIGIN_CITY_MARKET_ID | int | 7,422,037 | - | 0 | 30070 | 35991 | 31744.7 | 1304.68 | null |
# MAGIC | ORIGIN | string | 7,422,037 | - | 0 | null | null | null | null | 353 |
# MAGIC | ORIGIN_STATE_ABR | string | 7,422,037 | - | 0 | null | null | null | null | 54 |
# MAGIC | DEST_AIRPORT_ID | int | 7,422,037 | - | 0 | 10135 | 16869 | 12648.81 | 1523.82 | null |
# MAGIC | DEST | string | 7,422,037 | - | 0 | null | null | null | null | 353 |
# MAGIC | DEST_STATE_ABR | string | 7,422,037 | - | 0 | null | null | null | null | 54 |
# MAGIC | CRS_DEP_TIME | int | 7,422,037 | - | 0 | 1 | 2359 | 1330.26 | 492.99 | null |
# MAGIC | DEP_TIME | int | 7,422,037 | - | 0 | 0 | 2400 | 1311.21 | 532.37 | null |
# MAGIC | DEP_DELAY | double | 7,422,037 | - | 0 | -82 | 2710 | 10.73 | 48.55 | null |
# MAGIC | DEP_DELAY_NEW | double | 7,422,037 | - | 0 | 0 | 2710 | 13.86 | 47.51 | null |
# MAGIC | DEP_DEL15 | double | 7,422,037 | - | 0 | 0 | 1 | 0.18 | 0.39 | null |
# MAGIC | ARR_TIME | int | 7,422,037 | - | 0 | 0 | 2400 | 1435.82 | 572.5 | null |
# MAGIC | ARR_DELAY | double | 7,422,037 | - | 0 | -99 | 2695 | 5.3 | 50.54 | null |
# MAGIC | ARR_DELAY_NEW | double | 7,422,037 | - | 0 | 0 | 2695 | 13.87 | 47.19 | null |
# MAGIC | ARR_DEL15 | double | 7,422,037 | - | 0 | 0 | 1 | 0.19 | 0.39 | null |
# MAGIC | CANCELLED | double | 7,422,037 | - | 0 | 0 | 1 | 0.02 | 0.13 | null |
# MAGIC | DIVERTED | double | 7,422,037 | - | 0 | 0 | 1 | 0 | 0.05 | null |
# MAGIC | DISTANCE | double | 7,422,037 | - | 0 | 31 | 5095 | 800.54 | 592.51 | null |
# MAGIC | CARRIER_DELAY | double | 7,422,037 | - | 0 | 0 | 2695 | 3.95 | 29.76 | null |
# MAGIC | WEATHER_DELAY | double | 7,422,037 | - | 0 | 0 | 1847 | 0.71 | 14.08 | null |
# MAGIC | NAS_DELAY | double | 7,422,037 | - | 0 | 0 | 1741 | 3.1 | 18.34 | null |
# MAGIC | SECURITY_DELAY | double | 7,422,037 | - | 0 | 0 | 1078 | 0.02 | 1.45 | null |
# MAGIC | LATE_AIRCRAFT_DELAY | double | 7,422,037 | - | 0 | 0 | 2206 | 5.13 | 25.48 | null |
# MAGIC | STATION | double | 7,422,037 | - | 0 | 0 | 91765061705 | 72721632398 | 3413297119 | null |
# MAGIC | HourlyAltimeterSetting | double | 7,422,037 | - | 0 | 0 | 30.88 | 23.99 | 12.02 | null |
# MAGIC | HourlyDewPointTemperature | double | 7,422,037 | - | 0 | -42 | 96 | 45.25 | 20.48 | null |
# MAGIC | HourlyDryBulbTemperature | double | 7,422,037 | - | 0 | -44 | 128 | 57.9 | 22.16 | null |
# MAGIC | HourlyPrecipitation | double | 7,422,037 | - | 0 | 0 | 4.44 | 0 | 0.03 | null |
# MAGIC | HourlyPressureChange | double | 7,422,037 | - | 0 | -0.28 | 0.38 | 0 | 0.02 | null |
# MAGIC | HourlyPressureTendency | double | 7,422,037 | - | 0 | 0 | 9 | 1.38 | 2.56 | null |
# MAGIC | HourlyRelativeHumidity | double | 7,422,037 | - | 0 | 0 | 100 | 63.41 | 24.3 | null |
# MAGIC | HourlySkyConditions | double | 7,422,037 | - | 0 | 0 | 74 | 3.15 | 11.99 | null |
# MAGIC | HourlyWindGustSpeed | double | 7,422,037 | - | 0 | 0 | 79 | 2.23 | 7.51 | null |
# MAGIC | HourlyWindSpeed | double | 7,422,037 | - | 0 | 0 | 2237 | 7.79 | 5.99 | null |
# MAGIC | DailyAverageDewPointTemperature | double | 7,422,037 | - | 0 | -31 | 81 | 1.64 | 9.2 | null |
# MAGIC | depart_ts_local | timestamp | 7,422,037 | - | 0 | null | null | null | null | null |
# MAGIC | depart_ts_utc | timestamp | 7,422,037 | - | 0 | null | null | null | null | null |
# MAGIC | station_miles | double | 7,422,037 | - | 0 | 0 | 2.94168035 | 0.02 | 0.18 | null |
# MAGIC | PREV_CRS_DEP_TIME | int | 7,422,037 | - | 0 | 0 | 2359 | 1326.46 | 497.76 | null |
# MAGIC | PREV_DEP_TIME | int | 7,422,037 | - | 0 | 0 | 2400 | 1310.48 | 533.4 | null |
# MAGIC | PREV_DEP_DELAY | double | 7,422,037 | - | 0 | -82 | 2710 | 10.72 | 48.53 | null |
# MAGIC | PREV_DEP_DELAY_NEW | double | 7,422,037 | - | 0 | 0 | 2710 | 13.85 | 47.49 | null |
# MAGIC | PREV_CRS_ARR_TIME | int | 7,422,037 | - | 0 | 0 | 2400 | 1481.39 | 527.28 | null |
# MAGIC | PREV_ARR_TIME | int | 7,422,037 | - | 0 | 0 | 2400 | 1434.91 | 573.73 | null |
# MAGIC | PREV_ARR_DELAY | double | 7,422,037 | - | 0 | -99 | 2695 | 5.3 | 50.52 | null |
# MAGIC | PREV_CANCELLED | double | 7,422,037 | - | 0 | 0 | 1 | 0.02 | 0.12 | null |
# MAGIC | PREV_DIVERTED | double | 7,422,037 | - | 0 | 0 | 1 | 0 | 0.05 | null |
# MAGIC | PREV_OP_CARRIER_FL_NUM | int | 7,422,037 | - | 0 | 0 | 7933 | 2550.47 | 1802.56 | null |
# MAGIC | PREV_DISTANCE | double | 7,422,037 | - | 0 | 0 | 5095 | 797.59 | 593.21 | null |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix C: Data checkpoint strategy
# MAGIC Given the following diagram, the Flight Delay Prediction Data Pipeline consists of a 5-stage data processing system for predicting flight delays:
# MAGIC
# MAGIC 1. __Data Mining__: Statistical analysis and feature selection
# MAGIC 2. __Feature Engineering__: Previous flight data, holiday indicators, data cleaning
# MAGIC 3. __Data Imputation__: Handle missing values systematically
# MAGIC 4. __Class Rebalancing__: Monthly-based balancing for training data
# MAGIC 5. __Data Split__: Training, Validation, Test sets
# MAGIC
# MAGIC In addition, we will have separate checkpoint for each data size (3m, 1yr, 3yr)

# COMMAND ----------

from IPython.display import display, HTML

with open('img/phase2/w261_proj_presentation_ii_data_pipeline.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix D: Results and Discussion on Original OTPW 1Y Data
# MAGIC Although our training and primary evaluation use the custom-join dataset, we also benchmark our models on the original OTPW dataset for the 3M, 1Y, and 5Y prediction tasks. This section presents the results and discussion for the 1Y prediction task on the original OTPW dataset.
# MAGIC
# MAGIC ### D.1 Logistic Regression Results on Original OTPW Dataset
# MAGIC Using the best hyperparameters obtained from grid search, the logistic regression model achieves an average F1-score of 70.2% on the original OTPW 1Y dataset. It correctly identifies 66.43% of actual flight delays (recall), while 74.5% of predicted flight delays are correct (precision), indicating strong predictive performance. WYS and DVL are the least likely to experience delays, and PREV_DEP_DELAY_GROUP stands out as the most important feature.

# COMMAND ----------

from IPython.display import display, HTML

with open('img/phase2/OTPW_12M_ogistic_regression_result.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### D.2 Gradient Boosting Results on Original OTPW Dataset
# MAGIC With the best hyperparameters selected via grid search, the gradient boosting model achieves an average F1-score of 70.35% on the original OTPW 1Y dataset. The model’s recall is 65.17%, which is close to that of logistic regression, while precision improves to 76.42%. PREV_DEP_DELAY_GROUP ranks as the most important feature, followed by ORIGIN and DEST, highlighting the role of prior delays and airport characteristics in predicting delays.

# COMMAND ----------

from IPython.display import display, HTML

with open('img/phase2/OTPW_12M_gradient_boosting_result.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC We will also use a common saving function so we use same default path, *dbfs:/student-groups/Fall_2025_Group_01_01*. 
# MAGIC
# MAGIC As for notebook organization, we will likely divide into few a notebooks (maybe 2 or 3). 