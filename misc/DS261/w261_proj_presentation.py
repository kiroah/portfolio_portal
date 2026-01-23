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
# MAGIC ## Phase 1 Report
# MAGIC
# MAGIC **Team 1-1**
# MAGIC
# MAGIC November 2, 2025

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
# MAGIC ### Gantt Chart of Phase Leader Plan
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/gantt.png">
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
# MAGIC | **Hiro Naito** | Data Cleaning, Data Preparation, Data Dictionary creation, Preliminary EDA & Visualizations. (9 hours) | **Phase 2 Leader** - Structure data catalog & checkpoints for downstream use including 1-year and 3-5 yr datasets. Further EDA, and data clean up including datatype conversion and imputation, normalization; **EXTRA CREDIT:** manually join flights and weather data + extra sources; documentation and meetings; contribute to Phase 2 report. | Final report integration, documentation, and meetings; contribute to Phase 3 report.|
# MAGIC | **Hong Hu** | **Phase 1 Leader** — developed the project plan using data and machine learning pipeline block diagrams in HTML; led overall coordination; handled deliverable submission. (10 hours) | **Phase 2 Leader** - Conduct advanced feature engineering (interaction terms, Breiman’s method, etc.); build and fine-tune baseline models using grid search and time-series cross-validation; documentation and meetings; contribute to Phase 2 report. | Model training and inference; Lead feature engineering on larger datasets; documentation and meetings; contribute to Phase 3 report. |
# MAGIC | **Min Yang** | Preliminary EDA and visualization; authored project abstract; created Phase Leader Plan and Credit Assignment Plan tables. (9 hours)| Perform Phase II EDA on OTPW (3-month and 12-month datasets); design ML pipeline using ensemble models; conduct hyperparameter tuning; documentation and meetings; contribute to Phase 2 report. | **Phase 3 Leader** - Develop Machine Learning Algorithms and Metrics section; build NN-based ML pipeline (MLP and Residual MLP); lead final synthesis and Phase 3 report. |
# MAGIC | **Micah Collins** | Defined ML algorithms and selected evaluation metrics; contributed to model planning. (7.5 hours) | Conduct extra credit data extraction for recent years; clean and explore additional datasets; perform Phase II EDA; documentation and meetings; contribute to Phase 2 report. | **Phase 3 Leader** - Develop advanced models (NN architectures, ensemble comparisons); refine ML pipeline and contribute to Machine Learning Algorithms and Metrics section; participate in Phase 3 report. |
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
# MAGIC ## Project Abstract
# MAGIC
# MAGIC Flight delays remain a major challenge for the U.S. aviation industry, causing operational inefficiencies, financial losses, and passenger dissatisfaction. Accurate flight delay prediction enables airlines and airports to optimize scheduling, allocate resources effectively, and communicate proactively with travelers, improving overall efficiency and customer experience. The objective of this project is to predict U.S. domestic flight departure delays - defined as delays of 15 minutes or more - using large-scale flight and weather data. The datasets combine the U.S. Department of Transportation TranStats On-Time Performance (OTP) records and NOAA Global Hourly Weather data from 2015–2021, joined by airport code and timestamp. 
# MAGIC
# MAGIC We select F1-score as the primary evaluation metric which represents the harmonic mean of precision and recall. Because flight delay datasets exhibit significant class imbalance, where accuracy becomes misleading and a model could achieve high accuracy by simply predicting all flights as on-time. F1-score addresses this by harmonically averaging precision and recall, ensuring balanced performance on the minority (delayed) class. Both false positives and false negatives carry operational costs: false positives lead to unnecessary resource allocation and eroded passenger trust, while false negatives result in inadequate preparation and passenger dissatisfaction. F1-score's balanced nature prevents over-optimization for one error type, encouraging models that perform reasonably well on both dimensions while avoiding extreme trade-offs that would limit practical deployment.
# MAGIC
# MAGIC In subsequent phases, we will frame this as a binary classification problem to predict whether a flight will be delayed or on time. A scalable Spark ML pipeline will be developed using logistic regression as the baseline model, while also exploring Random Forest and Gradient-Boosted Trees for improved non-linear performance. In addition, we plan to experiment with neural network architectures, including the Multilayer Perceptron (MLP) and more advanced variants such as Residual MLPs, to evaluate their ability to capture complex feature interactions. The pipeline will incorporate feature engineering, SMOTE for class balancing, and time-based cross-validation to ensure robustness. Model performance will be primarily evaluated using the F1-score, with precision and recall as complementary metrics to provide deeper insight into the trade-off between false delay alerts and missed delay predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Description
# MAGIC
# MAGIC Code used for this section is in [phase1_data_description](https://dbc-fae72cab-cf59.cloud.databricks.com/editor/notebooks/1047316347250649?o=4021782157704243) notebook

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Data sources
# MAGIC We will be utilizing the following data sources. Note that all stats about the tables are from 2015Q1 (i.e. 3 months worth of) data. However, we assume the number of columns and percent of duplicate rows are consistent with larger datasets. 
# MAGIC
# MAGIC | **Data source** | **Description** | **Data size** | **# of rows** | **# of cols** | **% of duplicate rows** |
# MAGIC |-----------------|-----------------|----------------|----------------|----------------|----------------|
# MAGIC | Flights | This is a subset of the passenger flight's on-time performance data taken from the TranStats data collection available from the U.S. Department of Transportation (DOT) | 96 MB | 1403471 <br> AFTER DEDUP (before: 2806942) | 109 | 50% |
# MAGIC | Weather | The weather dataset from the National Oceanic and Atmospheric Administration repository | 1.1 GB | 30528602 | 124 | 0% |
# MAGIC | Stations | Overall the airport dataset provides some metadata about each airport. | 53 MB | 5004169 | 12 | 0% |
# MAGIC | OTPW | A pre-joined table that combines Flights data with Weather data. The join is somewhat complex requiring few mapping data sources. Details are explain below <br>All fields are string datatype | 1.4 GB | 1401363 | 216 | 0% |
# MAGIC | Holiday (stretch) | While the data source is not identified, we are planning to incorporate major US holidays into the dataset as airports tend to be busy during those times. | | | | |
# MAGIC
# MAGIC **Key insights** 
# MAGIC * Flights data have exact duplicate for all rows, therefore doubling the # of rows. 
# MAGIC * After deduping, the # of rows for flights (1403471) and OTPW (1401363) doesn't match, which is strange since OTPW is supposed to 
# MAGIC join the flights data with supplemental weather data. Assuming we use the OTPW dataset, this requires further analysis
# MAGIC * As mentioned in the description of OTPW table, the join seems complex though we can't confirm as we don't have access to original data join mechanics. However it seems the mechanics is: 
# MAGIC    * From flights table, use the origin airport ID (IATA code) and date/time 4 hours before scheduled departure
# MAGIC    * The IATA code is joined with IATA code in [airport code mapping table](https://datahub.io/core/airport-codes) to retrieve ICAO code
# MAGIC    * The ICAO code is joined with neighbor_call field in stations data to retrieve station_id (station that records the weather data)
# MAGIC    * The station_id is mapped to weather data's station field to get the reading, and then joins the row with closest (but earlier) reading from 4 hours before scheduled departure
# MAGIC    * (This is a speculation, and until we decide to do the join manually, we cannot confirm.)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Data fields
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC As of this writing we are planning to use the fields listed below. We may add or remove fields as needed later in the project. 
# MAGIC
# MAGIC Note that the field with source of OTPW means these fields weren't included in the original data source, therefore likely means
# MAGIC it's some derived data. 
# MAGIC
# MAGIC
# MAGIC | **source** | **field_id** | **field_desc** | **data_type** | **missing_pct** | **mean** | **stddev** | **min** | **max** |
# MAGIC |-------------|--------------|----------------|----------------|-----------------|-----------|-------------|-----------|-----------|
# MAGIC | flights_3m | YEAR | Year | int | 0.00% | 2,015.00 | 0.00 | 2015 | 2015 |
# MAGIC | flights_3m | QUARTER | Quarter (1-4) | int | 0.00% | 1.00 | 0.00 | 1 | 1 |
# MAGIC | flights_3m | MONTH | Month | int | 0.00% | 2.02 | 0.83 | 1 | 3 |
# MAGIC | flights_3m | DAY_OF_MONTH | Day of Month | int | 0.00% | 15.54 | 8.69 | 1 | 31 |
# MAGIC | flights_3m | DAY_OF_WEEK | Day of Week | int | 0.00% | 3.94 | 2.00 | 1 | 7 |
# MAGIC | flights_3m | FL_DATE | Flight Date (yyyymmdd) | string | 0.00% |  |  | 2015-01-01 | 2015-03-31 |
# MAGIC | flights_3m | OP_UNIQUE_CARRIER | Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years. | string | 0.00% |  |  | AA | WN |
# MAGIC | flights_3m | OP_CARRIER_AIRLINE_ID | An identification number assigned by US DOT to identify a unique airline (carrier). A unique airline (carrier) is defined as one holding and reporting under the same DOT certificate regardless of its Code, Name, or holding company/corporation. | int | 0.00% | 19,977.27 | 397.96 | 19393 | 21171 |
# MAGIC | flights_3m | TAIL_NUM | Tail Number | string | 0.58% |  |  | D942DN | N9EAMQ |
# MAGIC | flights_3m | OP_CARRIER_FL_NUM | Flight Number | int | 0.00% | 2,243.94 | 1,793.08 | 1 | 9794 |
# MAGIC | flights_3m | ORIGIN_AIRPORT_ID | Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused. | int | 0.00% | 12,670.94 | 1,519.92 | 10135 | 16218 |
# MAGIC | flights_3m | ORIGIN_CITY_MARKET_ID | Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market. Use this field to consolidate airports serving the same city market. | int | 0.00% | 31,712.19 | 1,283.51 | 30070 | 35991 |
# MAGIC | flights_3m | ORIGIN | Origin Airport | string | 0.00% |  |  | ABE | YUM |
# MAGIC | flights_3m | ORIGIN_STATE_ABR | Origin Airport, State Code | string | 0.00% |  |  | AK | WY |
# MAGIC | flights_3m | DEST_AIRPORT_ID | Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport. Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused. | int | 0.00% | 12,670.92 | 1,519.93 | 10135 | 16218 |
# MAGIC | flights_3m | DEST | Destination Airport | string | 0.00% |  |  | ABE | YUM |
# MAGIC | flights_3m | DEST_STATE_ABR | Destination Airport, State Code | string | 0.00% |  |  | AK | WY |
# MAGIC | flights_3m | CRS_DEP_TIME | CRS Departure Time (local time: hhmm) | int | 0.00% | 1,327.60 | 474.36 | 1 | 2359 |
# MAGIC | flights_3m | DEP_TIME | Actual Departure Time (local time: hhmm) | int | 3.02% | 1,337.32 | 486.08 | 1 | 2400 |
# MAGIC | flights_3m | DEP_DELAY | Difference in minutes between scheduled and actual departure time. Early departures show negative numbers. | double | 3.02% | 10.36 | 37.86 | -61 | 1988 |
# MAGIC | flights_3m | DEP_DELAY_NEW | Difference in minutes between scheduled and actual departure time. Early departures set to 0. | double | 3.02% | 13.03 | 36.79 | 0 | 1988 |
# MAGIC | flights_3m | DEP_DEL15 | Departure Delay Indicator, 15 Minutes or More (1=Yes) | double | 3.02% | 0.20 | 0.40 | 0 | 1 |
# MAGIC | flights_3m | ARR_TIME | Actual Arrival Time (local time: hhmm) | int | 3.16% | 1,490.84 | 512.40 | 1 | 2400 |
# MAGIC | flights_3m | ARR_DELAY | Difference in minutes between scheduled and actual arrival time. Early arrivals show negative numbers. | double | 3.32% | 6.24 | 40.53 | -87 | 1971 |
# MAGIC | flights_3m | ARR_DELAY_NEW | Difference in minutes between scheduled and actual arrival time. Early arrivals set to 0. | double | 3.32% | 13.41 | 37.03 | 0 | 1971 |
# MAGIC | flights_3m | ARR_DEL15 | Arrival Delay Indicator, 15 Minutes or More (1=Yes) | double | 3.32% | 0.21 | 0.41 | 0 | 1 |
# MAGIC | flights_3m | CANCELLED | Cancelled Flight Indicator (1=Yes) | double | 0.00% | 0.03 | 0.17 | 0 | 1 |
# MAGIC | flights_3m | CANCELLATION_CODE | Specifies The Reason For Cancellation | string | 96.90% |  |  | A | D |
# MAGIC | flights_3m | DIVERTED | Diverted Flight Indicator (1=Yes) | double | 0.00% | 0.00 | 0.05 | 0 | 1 |
# MAGIC | flights_3m | DISTANCE | Distance between airports (miles) | double | 0.00% | 807.10 | 594.87 | 31 | 4983 |
# MAGIC | flights_3m | CARRIER_DELAY | Carrier Delay, in Minutes | double | 79.58% | 18.28 | 46.31 | 0 | 1971 |
# MAGIC | flights_3m | WEATHER_DELAY | Weather Delay, in Minutes | double | 79.58% | 3.15 | 22.34 | 0 | 1152 |
# MAGIC | flights_3m | NAS_DELAY | National Air System Delay, in Minutes | double | 79.58% | 13.46 | 25.74 | 0 | 1101 |
# MAGIC | flights_3m | SECURITY_DELAY | Security Delay, in Minutes | double | 79.58% | 0.06 | 1.95 | 0 | 241 |
# MAGIC | flights_3m | LATE_AIRCRAFT_DELAY | Late Aircraft Delay, in Minutes | double | 79.58% | 22.67 | 41.85 | 0 | 1313 |
# MAGIC | weather_3m | STATION | Station identifier (typically USAF-WBAN or similar code) for the reporting site. | string | 0.00% | * | * | * | * |
# MAGIC | weather_3m | DATE | Observation or summary date (YYYYMMDD). | string | 0.00% |  |  | 2015-01-01 0:00:00 | 2015-03-31 23:59:00 |
# MAGIC | weather_3m | HourlyAltimeterSetting | Hourly altimeter setting (pressure reduced to sea level, for aviation). | string | 46.11% | 30.06 | 0.29 | 26.05 | 31.1 |
# MAGIC | weather_3m | HourlyDewPointTemperature | Hourly dew point temperature, the saturation temperature at current moisture. | string | 17.57% | 30.60 | 21.88 | * | 9s |
# MAGIC | weather_3m | HourlyDryBulbTemperature | Hourly ambient (air) temperature measured in shelter/exposure. | string | 2.13% | 39.47 | 23.04 | * | 9s |
# MAGIC | weather_3m | HourlyPrecipitation | Precipitation amount accumulated during the hour (liquid equivalent). | string | 87.12% | 0.01 | 0.05 | * | T |
# MAGIC | weather_3m | HourlyPresentWeatherType | Codes describing present weather (e.g., rain, snow, fog) observed in the hour. | string | 87.13% |  |  | * * * |* * * | | ||s |
# MAGIC | weather_3m | HourlyPressureChange | Change in station or sea-level pressure over the standard interval. | string | 72.40% | 0.00 | 0.05 | + | 1.48 |
# MAGIC | weather_3m | HourlyPressureTendency | Code describing the character of pressure change (rising/falling/steady). | string | 71.42% | 4.85 | 2.75 | 0 | 9 |
# MAGIC | weather_3m | HourlyRelativeHumidity | Hourly relative humidity, typically derived from temperature and dew point. | string | 17.59% | 72.99 | 20.12 | * | 99 |
# MAGIC | weather_3m | HourlySkyConditions | Cloud/sky condition codes, including ceilings and coverage layers. | string | 47.42% | 29.51 | 26.96 | * | X:10s 0s |
# MAGIC | weather_3m | HourlyWindGustSpeed | Highest instantaneous wind speed (gust) observed in/near the hour. | string | 92.83% | 25.72 | 8.04 | * | 99s |
# MAGIC | weather_3m | HourlyWindSpeed | Mean wind speed for the hour. | string | 13.25% | 8.24 | 8.53 | * | 9s |
# MAGIC | weather_3m | DailyAverageDewPointTemperature | Day's mean dew point temperature. | string | 99.91% | 29.83 | 17.79 | -1 | 9 |
# MAGIC | otpw_3m | sched_depart_date_time_UTC |  | string | 0.00% |  |  | 2015-01-01 6:55:00 | 2015-04-01 3:59:00 |
# MAGIC | otpw_3m | four_hours_prior_depart_UTC |  | string | 0.00% |  |  | 2015-01-01 2:55:00 | 2015-03-31 23:59:00 |
# MAGIC | otpw_3m | two_hours_prior_depart_UTC |  | string | 0.00% |  |  | 2015-01-01 4:55:00 | 2015-04-01 1:59:00 |
# MAGIC
# MAGIC **Key insights**
# MAGIC * A lot of the weather data are empty/null. It may be because the value is "0", or the station simply doesn't record data. Assuming we use these as features for the model, we need to look into the reasons of empty/null value to decide on the imputation strategy. 
# MAGIC * We will be using DEP_DEL15 as the label, since we are framing this as a classificatin problem. 
# MAGIC * When we see stats for departure delay (DEP_DELAY_NEW), mean is 13.03 with stdev of 36.79, which means it can vary widely. 
# MAGIC * Some fields are included for data analysis only and not for ML features (e.g. CARRIER_DELAY, WEATHER_DELAY). We need to be careful with these so we don't use those as features

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Data checkpoint strategy
# MAGIC We will have at least one checkpoint (i.e. save to persistent disk) for each of the data process steps below: 
# MAGIC 1. Data joining (if any, may use only OTPW)
# MAGIC 1. Data cleaning
# MAGIC 1. Imputations
# MAGIC 1. Feature generation & encoding categorical variables (may have multiple checkpoints)
# MAGIC 1. Normalization
# MAGIC 1. Dimentionality reduction (if any)
# MAGIC 1. Train/test split
# MAGIC
# MAGIC In addition, we will have separate checkpoint for each data size (3m, 1yr, 3yr)
# MAGIC
# MAGIC We will also use a common saving function so we use same default path, *dbfs:/student-groups/Fall_2025_Group_01_01*. 
# MAGIC
# MAGIC As for notebook organization, we will likely divide into few a notebooks (maybe 2 or 3). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA highlights
# MAGIC Below we have added key insights from an early EDA analysis. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### # of delayed flights <br>
# MAGIC * Approximately 20% of flights are delayed (excluding cancelled flights)
# MAGIC <br><img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/delayed_flights.png">
# MAGIC
# MAGIC #### Average & median time of delay by category<br>
# MAGIC * For both mean and median, late aircraft (i.e. previous flight's delay) is highest
# MAGIC * Second is carrier delay
# MAGIC <br><img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/delay_cause.png">
# MAGIC
# MAGIC #### Previous flight's delay and current flight's delay
# MAGIC * For any previous flight that is delayed by a minute or more, there's high chance the next flight is delayed. This may be a feature we want to use by pulling the previous flight's delay. However, because we are predicting 2 hours before scheduled departure, we need to be careful on which data is available prior to the prediction time
# MAGIC <br><img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/prev_flight_delay.png">
# MAGIC
# MAGIC #### Pearson correlation heatmap for numeric values
# MAGIC * Focusing on correlation related to DEP_DEL15 (delayed by 15+ minutes), there's only a few that has correlation (based on Pearson's correlation)
# MAGIC * Weather related delay is low correlation, but it may be useful once we massage them for easier use
# MAGIC * Note that null-values were imputed with 0.0 without looking at the data, so if we use adequate imputation the correlation may change.
# MAGIC * Also, note that we are only using Jan~Mar data, so we aren't picking seasonal differences. Likely better to do another correlation analysis with 1 year data. 
# MAGIC <br><img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/corr_num.png">
# MAGIC
# MAGIC #### Looking delayed flights VS categorical variables
# MAGIC **Key insights:**
# MAGIC * There are noticeable differences in delays (15+ minutes) by airline and origin/arrival airport/states. 
# MAGIC * However, these stats may change as time passes (+ we're not including seasonal changes), so if we are to use these as features then we need to use with care then we need to be aware of trend changes.
# MAGIC <br><img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/delay_by_cat.png">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Machine Learning Algorithms and Metrics
# MAGIC For both mean and median, late aircraft (i.e. previous flight's delay) is highest
# MAGIC We have decided to take a binary classification approach to the problem.
# MAGIC We will predict between two classes: 
# MAGIC * Not Delayed: the flight departs early or up to 15 minutes after the scheduled departure time
# MAGIC * Delayed: the flight departs greater than 15 minutes after the scheduled departure time
# MAGIC
# MAGIC
# MAGIC ### Candidate Algorithms and Associated Loss Functions:
# MAGIC 1) **Logistic Regression (baseline)**
# MAGIC <br>Predicts the probability of a delay as a sigmoid transformation of some weighted combination of features.
# MAGIC Easy to interpret, simple, easily accessible implementation.
# MAGIC <br>Loss Function: Negative Log-Likelihood in order to penalize incorect predictions based on the confidence of the incorrect prediction
# MAGIC $$\mathcal{L}_{\text{log}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\Big]$$
# MAGIC <br>Pyspark Implementation: `pyspark.ml.classification.LogisticRegression`
# MAGIC <br>Parameters: `regParam`, `elasticNetParam`, `maxIter`, `weightCol`
# MAGIC
# MAGIC
# MAGIC 2) **Random Forest**
# MAGIC <br>Uses an ensemble of decision trees trained on random subsets of data and features via bootstrapping through bagging.
# MAGIC Pyspark allows us to either minimize **Gini Impurity** (minimize the uncertainty of class labels) or **Entropy** (maximize information gain) with each split
# MAGIC <br>**Gini Impurity (default):**  
# MAGIC $$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_{k,t}^2$$
# MAGIC **Entropy:**  
# MAGIC $$H(t) = -\sum_{k=1}^{K} p_{k,t}\log(p_{k,t})$$
# MAGIC The split chosen minimizes the weighted impurity across children (example for Gini):  
# MAGIC $$\Delta \text{Gini} = \text{Gini}_{\text{parent}} - \sum_{j} \frac{N_j}{N_{\text{parent}}}\text{Gini}_j$$
# MAGIC <br>PySpark Implementation: `pyspark.ml.classification.RandomForestClassifier`
# MAGIC <br>Parameters: `numTrees`, `maxDepth`, `featureSubsetStrategy`, `weightCol`
# MAGIC
# MAGIC
# MAGIC 3) **Gradient-Boosted Tree**
# MAGIC <br>An ensemble of shallow decisions trees that are sequentially built through correcting the mistakes of previous iterations.
# MAGIC <br>Good at capturing nonlinear feature interactions on complicated tabular data.
# MAGIC <br>Loss Function: A model updating procedure that fits a shallow tree to the negative gradient of the loss, with the loss per sample being:
# MAGIC $$\mathcal{L}_{\text{GBT}} = \frac{1}{N}\sum_{i=1}^{N}\log(1 + e^{-y_i F(x_i)})$$
# MAGIC <br>Pyspark Implementation: `pyspark.ml.classification.GBTClassifier`
# MAGIC <br>Parameters: `maxDepth`, `maxIter`, `stepSize`
# MAGIC
# MAGIC 4) Multilayer Perceptron
# MAGIC <br>Neural Network built on stacked fully connected layers with nonlinear activations. The most simplistic version of a neural network for our purposes.
# MAGIC Loss Function (Binary Cross-Entropy): 
# MAGIC $$\mathcal{L}_{\text{MLP}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\Big]$$
# MAGIC <br>We intend to utilize GPU resources and implement this using Pytorch/Tensorflow
# MAGIC
# MAGIC 5) **Extra: Residual MLPs**
# MAGIC <br>Neural Network built on stacked fully connected layers with nonlinear activations, with the addition of residual connects to stabilize gradients, improve overfitting tendencies, potentially improve convergence in deeper networks, and discover highly embedded signals.
# MAGIC Loss Function (Binary Cross-Entropy): 
# MAGIC $$\mathcal{L}_{\text{MLP}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\Big]$$
# MAGIC Residual Connection (Generalized):
# MAGIC $$h_{l+1} = f(W_l h_l + b_l) + h_l$$
# MAGIC <br>We intend to utilize GPU resources and implement this using Pytorch/Tensorflow.
# MAGIC <br>We will manually implement residual connection logic and ensure GPU compatibility.
# MAGIC
# MAGIC ### Metrics and Analysis
# MAGIC We treat “Not Delayed” as the **positive** class and compute the following using predictions from the test split:
# MAGIC | Metric | Formula | Interpretation |
# MAGIC |--------|----------|----------------|
# MAGIC | **Accuracy** | $$\( \frac{TP+TN}{TP+TN+FP+FN} \)$$ | Overall correctness |
# MAGIC | **Precision** | $$\( \frac{TP}{TP+FP} \)$$ | How many predicted delays were actually delayed |
# MAGIC | **Recall (Sensitivity)** | $$\( \frac{TP}{TP+FN} \)$$ | How many actual delays were correctly identified |
# MAGIC | **F1-Score** | $$\( 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)$$ | Harmonic mean of precision and recall, robust under class imbalance |
# MAGIC
# MAGIC **Primary Metric**: F1-score - robust, allows for balance of other metrics to be taken into account
# MAGIC <br>
# MAGIC **Secondary Metrics**: Accuracy, Precision, and Recall
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pipelines
# MAGIC Below, we have illustrated the planned data and ML pipeline. Note that we may (and likely will) make changes as we proceed with the project

# COMMAND ----------

from IPython.display import display, HTML


with open('img/w261_proj_presentation_eda_pipeline.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

with open('img/w261_proj_presentation_ml_pipeline.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC