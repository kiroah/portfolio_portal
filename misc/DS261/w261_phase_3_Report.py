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
# MAGIC # Phase 3 Report
# MAGIC
# MAGIC **Team 1-1**
# MAGIC
# MAGIC December 13, 2025
# MAGIC
# MAGIC <a href="https://docs.google.com/presentation/d/19EcjND_em_fu9rLKvpOkMlk6nDsZMS_WStRmLZhvfRc" target="_blank">In-class Presentation Link</a> 

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase Leader Plan:
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
# MAGIC # Credit Assignment plan:
# MAGIC
# MAGIC | **Name** | **Phase 1** | **Phase 2** | **Phase 3** |
# MAGIC |-----------------|--------------|------------------|------------------|
# MAGIC | **Hiro Naito** | Data Cleaning, Data Preparation, Data Dictionary creation, Preliminary EDA & Visualizations. (9 hours) | **Phase 2.1 Leader** - Generated 3m, 1y and 5y dataset with custom join, dubbed otpw_v2. Structured data catalog & checkpoints for downstream pipeline. Basic imputation, column filtering and cleanup. Create base template for the phase 2 presentation, contributed to the content for presentation and report. (23 hours) | 10 year data creation, update on code documentation, final presentation and report integration, meetings. (10 hours)|
# MAGIC | **Hong Hu** | **Phase 1 Leader** — developed the project plan using data and machine learning pipeline block diagrams in HTML; led overall coordination; handled deliverable submission. (10 hours) | **Phase 2.2 Leader** - Build and fine-tune baseline models using grid search; develop data pipelines with checkpoints; implement ML pipelines; benchmark models on the original OTPW dataset for the 3M, 1Y, and 5Y prediction tasks; handle documentation and meetings; contribute to the Phase 2 report. (30 hours) | Balanced data linkage risk with model performance while introducing the IS_PREV_DEP_DEL15 feature; explored airport page rank features and tested their impact on model performance; executed model training, inference, and reporting for Logistic Regression, Ensemble, and MLP models; predicted flight delays for Covid-19 and post-Covid-19 periods; completed Homework 5 assignment on behalf of the team. (30 hours) |
# MAGIC | **Min Yang** | Preliminary EDA and visualization; authored project abstract; created Phase Leader Plan and Credit Assignment Plan tables. (9 hours)| Perform Phase II EDA on custom joined OTPW-V2 (12-month datasets); contributed to the in-class presentation and Phase 2 report. (30 hours) | **Phase 3 Leader** - 5-year EDA and visualizations; code and documentations; lead final in-class presentation; updated project abstract session and data and feature engineering sessions, authored data leakage, and conclusion sessions;reviewed and edited all team sections to ensure report quality and consistency; lead final synthesis and Phase 3 report. (26 hours)  |
# MAGIC | **Micah Collins** | Defined ML algorithms and selected evaluation metrics; contributed to model planning. (7.5 hours) | Conducted Feature Engineering and leakage detection on OTPW_V2. Performed feature selection/importance procedures on OTPW_V2. Created modeling pipeline and Time-Series Cross validation pipeline for OTPW_V2. Created model evaluation/selection pipeline for OTPW_V2. Did results/analysis interpretation on OTPW_V2 models (28 hours) | **Phase 3 Leader** - Refined ML pipeline, devised block validation procedure, and optimized hyperparameter tuning. Coordinated with Hong on feature engineering and model development. Created results and model interpretation section for presentation. Wrote pipeline methodology, feature engineering, Phase II comparisons, gap analysis, feature importance, and results analysis/interpretations in Phase 3 report. (29 hours)|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Project Abstract
# MAGIC
# MAGIC Flight delays remain a major challenge for the U.S. aviation industry, costing the economy $33 billion annually, with passengers bearing over half of these losses (FAA/NEXTOR, 2019). A 10% reduction in delays could add $17.6 billion to the U.S. economy. Accurate prediction enables airlines and airports to optimize scheduling, allocate resources effectively, and communicate proactively with travelers, improving overall efficiency and customer experience. The objective of this project is to predict U.S. domestic flight departure delays - defined as delays of 15 minutes or more - using large-scale flight and weather data. This project spans three phases, each framing delay prediction as a binary classification problem: predicting whether a U.S. domestic flight departure will be delayed by 15 minutes or more. Phase 1 established our baseline approach and feature engineering strategy. Phase 2 scaled the pipeline and refined model selection. Phase 3 finalized model tuning and evaluation. We adopted F1-Macro as our primary evaluation metric to balance performance across both delayed and non-delayed classes, maintaining consistent metrics across all phases for comparability. Model performance metrics remained consistent across all phases to ensure comparability.
# MAGIC
# MAGIC Our datasets combine U.S. Department of Transportation TranStats On-Time Performance records and NOAA Global Hourly Weather data from 2015–2024, totaling approximately 63 million records. Given significant class imbalance from EDA, we established a Majority Class Baseline (predicting all flights as non-delayed), which achieved 82.7% accuracy but 0.0 Recall for delays—demonstrating that accuracy alone is a deceptive metric for this problem.
# MAGIC
# MAGIC We developed a scalable PySpark pipeline comparing Logistic Regression, XGBoost, and a Deep Neural Network (MLP), using Block Time-Series Cross-Validation to select robust hyperparameters in a compute-efficient manner. Key engineered features included upstream delay propagation (IS_PREV_DEP_DEL15), STORM_INDEX, SCHEDULED_BUFFER, HourlyRelativeHumidity, and smoothed target encoding for high-cardinality airport data. We addressed class imbalance via undersampling and class weighting. XGBoost achieved the best F1-Macro score (0.618) on the test set, while MLP attained the highest Delay Recall (0.652) but with a lower F1-Macro (0.580).

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data and Feature Engineering
# MAGIC
# MAGIC This section outlines our data sources, custom data join strategy, exploratory findings, and feature engineering approach - each designed to maximize predictive signal while ensuring all features are available at prediction time.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Data Lineage and Key Data Transformations
# MAGIC
# MAGIC Our data pipeline transforms raw flight and weather records into a unified, analysis-ready dataset through three stages: source integration, temporal alignment, and previous flight enrichment.
# MAGIC
# MAGIC Our predictive dataset integrates two primary data sources: U.S. Department of Transportation TranStats On-Time Performance (OTP) records and NOAA Integrated Surface Database (ISD) Global Hourly Weather observations. The dataset spans 2015–2024, totaling approximately 69 million flight records, with 2015–2018 used for model training, 2019 as the primary test set, and 2020–2024 reserved for additional generalization testing.
# MAGIC
# MAGIC We developed a custom join pipeline rather than using the provided OTPW data due to concerns with data integrity, weather alignment, and adaptability for extended date ranges. The rationale and detailed join process are described in the "Custom Data Join" section below.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.1 Data Sources
# MAGIC
# MAGIC **(1) Primary Data Sources**
# MAGIC
# MAGIC The following datasets serve as the main sources for model training and prediction:
# MAGIC
# MAGIC | Data Source | Description |
# MAGIC |:------------|:------------|
# MAGIC | Flights | Provided by the Bureau of Transportation Statistics (BTS), this dataset contains comprehensive historical flight performance records for U.S. domestic flights from 2015-2024. Each record represents an individual flight, including scheduled and actual departure/arrival times, delay causes (carrier, weather, NAS, security, late aircraft), flight identifiers (airline, tail number, origin, destination), and cancellation information <br><br> Parquet size (3M, 1Y, 5Y, 10Y): 96MB, 595MB, 2.4GB, 3.6GB <br> # of rows (3M, 1Y, 5Y, 10Y): 1,403,471, 14,844,074, 31,746,841, 69,082,074 |
# MAGIC | Weather | Sourced from NOAA's Integrated Surface Database (ISD), this dataset provides historical weather observations from stations across the U.S., including temperature, humidity, precipitation, wind speed, visibility, and adverse weather events (storms, fog, snow). Weather observations are recorded at hourly intervals, making them critical for capturing environmental conditions that influence flight operations. <br><br> Parquet size (3M, 1Y, 5Y, 10Y): 1.1GB, 4.8GB, 23.6GB, 46GB <br> # of rows (3M, 1Y, 5Y, 10Y): 30,528,602, 131,937,550, 639,726,637, 1,289,181,924|
# MAGIC
# MAGIC **(2) Secondary Data Sources**
# MAGIC
# MAGIC The following datasets support the data join process:
# MAGIC
# MAGIC | Data Source | Description |
# MAGIC |:------------|:------------|
# MAGIC | airports.csv | Airport Codes Dataset (DataHub) containing IATA/ICAO codes, airport names, cities, states, countries, and geographic coordinates. Enables mapping of airport codes to geographic positions for weather data integration. |
# MAGIC | isd_history.csv | NCEI weather station metadata including station IDs (USAF+WBAN codes), latitude, longitude, elevation, and station names. Enables precise mapping between weather observations and airport locations. |
# MAGIC
# MAGIC **(3) Sources Not Used**
# MAGIC
# MAGIC | Source | Reason for Exclusion |
# MAGIC |:-------|:---------------------|
# MAGIC | OTPW | Pre-joined flight and weather data provided by the course. We opted for a custom join approach for greater transparency and flexibility (see "Custom Data Join" section). |
# MAGIC | Airport Codes (course-provided) | Potentially outdated for recent data years. We used airports.csv from DataHub instead to ensure compatibility with extended date ranges. |
# MAGIC
# MAGIC Final joined dataset statistics are summarized in the "Summary Results" section.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.2 Custom Data Join
# MAGIC
# MAGIC Our team developed a custom joined dataset rather than using the pre-joined OTPW data provided. This decision was driven by several considerations:
# MAGIC
# MAGIC **Data Integrity:** The original OTPW dataset loses a small percentage of rows (approximately 0.2% for 3M records). While this loss is minimal, the underlying join logic is undocumented, making it difficult to assess whether the missing records could introduce bias in our evaluation results.
# MAGIC
# MAGIC **Weather Data Alignment:** The original OTPW weather data does not consistently reflect the most recent reading prior to scheduled departure. Although weather observations should correspond to approximately 2 hours before the flight, the join logic appears to select non-optimal readings in some cases. Without visibility into the exact join criteria, results may be unreliable.
# MAGIC
# MAGIC **Adaptability:** Relying on the original OTPW limits flexibility for future enhancements. If we extend the analysis to years beyond 2019 or incorporate arrival weather data, we would face inconsistencies between departure and arrival join logic—complicating debugging and validation.
# MAGIC
# MAGIC **Data Type Preservation:** The original OTPW stores all fields as string data types, requiring additional type conversion and risking data quality issues during transformation.
# MAGIC
# MAGIC **Join Process Overview:**
# MAGIC
# MAGIC | Step | Description |
# MAGIC |:-----|:------------|
# MAGIC | Step 1: Mapping Table Generation | Preparation work before flight-weather join by generating a mapping table from airports to weather station(s) |
# MAGIC | Step 2: Flight & Weather Join | Join flight data with weather data using the mapping table from Step 1 |
# MAGIC | Step 3: Previous Flight Self-Join | Self-join to attach previous flight information (e.g., previous departure time) for each record |
# MAGIC
# MAGIC Detailed join logic and implementation are provided in Appendix E.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.3 Summary Results
# MAGIC
# MAGIC The following table summarizes outputs from our custom join pipeline across different data scales:
# MAGIC
# MAGIC | Metric | 3 Months | 1 Year (2019) | 5 Years (2015–2019) | 10 Years (2015–2024) |
# MAGIC |:-------|:--------:|:-------------:|:-------------------:|:--------------------:|
# MAGIC | # of Rows | 1,403,471 | 7,422,037 | 31,746,841 | 69,082,074 |
# MAGIC | # of Columns | 258 | 258 | 258 | 258 |
# MAGIC | Mapping Table Generation Time | < 1 min | < 1 min | < 1 min | < 1 min |
# MAGIC | Join (+ Self Join) Generation Time | 4 min | 30 min | 6.5 hrs | 12.4 hrs |
# MAGIC | Table Size (Parquet) | 255 MB | 1.4 GB | 5.8 GB | 12 GB |
# MAGIC
# MAGIC **Notes:**
# MAGIC - Row counts were validated against original flight data - no rows were lost or duplicated during the join process
# MAGIC - Pipeline executed on a Databricks cluster with 6–8 nodes, depending on availability
# MAGIC - Each time span is stored in a separate folder, enabling downstream tasks to scale easily by pointing to different data partitions
# MAGIC - Complete data field definitions are provided in Appendix A; fields used in EDA and modeling are documented in Appendix B

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Exploratory Data Analysis Summary (EDA)
# MAGIC
# MAGIC Our exploratory data analysis (EDA) served three objectives: validating data quality across the full five-year dataset, confirming that delay patterns observed in earlier phases remain consistent at scale, and reinforcing feature engineering decisions prior to final model training. Building on our Phase 1–2 methodology, we extended analysis to 31M+ flight records spanning 2015–2019. This expanded 5-year scope provides greater statistical confidence in observed patterns and validates that key findings—temporal trends, weather impacts, and carrier/route variations—are robust across years rather than artifacts of a single period. The analysis also confirms that engineered features such as STORM_INDEX and SCHEDULED_BUFFER capture meaningful predictive signals across diverse operating conditions.
# MAGIC
# MAGIC Below we summarize key findings by several analytical dimensions:
# MAGIC
# MAGIC  - **(1) Data Quality Checks:**
# MAGIC
# MAGIC Our custom joined dataset demonstrates high data quality across all five years, with data types correctly preserved through the join process. The dataset contains 258 columns distributed across five data types as shown in below chart. No duplicate records were detected, confirming the integrity of our join logic.
# MAGIC
# MAGIC Missing value analysis reveals patterns consistent with expected data structure. CANCELLATION_CODE shows 98.46% missingness, which is expected behavior since most flights are not cancelled. Similarly, delay reason codes (NAS_DELAY, CARRIER_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY) exhibit 81.73% missingness—this is expected as these fields are only populated when a delay occurs.
# MAGIC
# MAGIC Several weather features exhibit high missingness rates due to station coverage gaps and should be dropped from the final feature set: DailyAverageDewPointTemperature (97.81%), station_miles (92.86%), HourlySkyConditions (90.55%), HourlyPresentWeatherType (90.28%), HourlyWindGustSpeed (88.78%), HourlyPressureChange (69.83%), and HourlyPressureTendency (69.83%). Notably, HourlyWindGustSpeed can be excluded since we retain the more complete HourlyWindSpeed column.
# MAGIC
# MAGIC Weather features with moderate missingness can be retained with appropriate imputation: HourlyPrecipitation (28.53% missing) and HourlyAltimeterSetting (18.58% missing). These columns provide valuable predictive signal and their missingness levels are manageable through imputation strategies.
# MAGIC
# MAGIC Detailed data dictionary and statistics are provided in the Appendix B.
# MAGIC  
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/P3_DataQualityCheck_Pic_00.png" width="1400">
# MAGIC
# MAGIC
# MAGIC
# MAGIC  - **(2) Class Imbalance:**
# MAGIC The target variable (DEP_DEL15) shows a consistent delay rate of approximately 17–19% across all years, confirming persistent class imbalance. This stability validates our resampling and class-weighting strategy applied uniformly across training folds.
# MAGIC Blow bar chart shows delay rate by year (2015–2019)
# MAGIC
# MAGIC   <img src="https://raw.githubusercontent.com/hong-hu/w261/main/P3_DelayImbalance_Pic_01.png" width="1000">
# MAGIC
# MAGIC
# MAGIC  - **(3) Temporal Patterns:**
# MAGIC Delay patterns exhibit strong and consistent temporal structure across all five years. Hourly trends show lowest delay rates in early morning (5–9am: 7–13%) escalating through the day to evening peaks (5–11pm: 25–27%). Monthly patterns confirm June as the highest-delay month (24%) driven by thunderstorm season, while September consistently achieves the lowest delay rates (14%). Thursday shows the worst daily performance as cascading delays accumulate through the week, while Saturday benefits from lower volume and overnight system resets.
# MAGIC
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/P3_Temporal%20Patterns_Pic_02.png" width="1000">
# MAGIC
# MAGIC
# MAGIC
# MAGIC - **(4) Weather Features:**
# MAGIC
# MAGIC Among the weather variables, HourlyRelativeHumidity exhibits the strongest correlation with flight delays (0.058), as elevated humidity often signals adverse weather conditions such as fog or precipitation. However, individual weather variables generally show weak linear correlations with delays (all below 0.06), suggesting that weather impacts on delays are non-linear in nature and may benefit from composite feature engineering rather than direct inclusion of raw metrics.
# MAGIC
# MAGIC Several weather variables exhibit high multicollinearity that must be addressed during feature selection. HourlyDryBulbTemperature and HourlyDewPointTemperature are strongly correlated (0.80), indicating that one should be dropped or the two combined to avoid redundancy. Similarly, HourlyPressureChange and HourlyPressureTendency show moderate correlation (0.47), warranting careful consideration.
# MAGIC
# MAGIC Wind speed demonstrates a clear non-linear relationship with delays, with delay rates approximately doubling from calm conditions (17.8%) to extreme winds (36.4%). This pattern suggests that binning wind speed into categories or creating a binary high-wind indicator (e.g., wind speed exceeding 30 mph) may capture this effect more effectively than using raw continuous values. Precipitation shows a similar non-linear pattern—any level of precipitation increases delay rates from 17.8% under dry conditions to 24.5% or higher when trace amounts or more are recorded. Notably, there are diminishing returns at higher precipitation levels, with heavy precipitation (28.7% delay rate) showing only marginal increase over moderate precipitation (27.6%). This suggests that a binary `has_precipitation` feature may capture most of the predictive signal without requiring granular precipitation measurements.
# MAGIC
# MAGIC These findings informed our decision to develop a composite weather severity index (`STORM_INDEX`) that combines multiple weather variables, rather than relying on individual metrics with weak linear relationships. Additionally, we applied binned categories for variables exhibiting non-linear effects, enabling the models to better capture threshold-based weather impacts on delays.
# MAGIC
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/P3_Weather_Pic_04.png" width="1000">
# MAGIC
# MAGIC  - **(6) Carrier Performance:**
# MAGIC Delay rates vary substantially across carriers, reflecting differences in operational practices, hub locations, and network complexity.
# MAGIC
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/P3_CarrrierPerformance_Pic_03.png" width="1000">
# MAGIC
# MAGIC
# MAGIC  - **(7) Airport Pagerank:**
# MAGIC Our PageRank analysis identifies the most influential airports in the U.S. flight network based on connectivity and traffic volume. We construct a weighted, directed graph in which nodes are airports and edges are flights, with edge weights equal to the total number of flights between each airport pair in 2015. The resulting PageRank scores highlight key hubs that support substantial passenger and cargo flows. Although this feature was not included in the final model—replacing airport codes with PageRank scores did not improve performance—the analysis offers useful insight into airport importance within the network.
# MAGIC
# MAGIC  <img src="https://raw.githubusercontent.com/hong-hu/w261/main/w261_proj_presentation_page_rank.png" width="1000">
# MAGIC
# MAGIC | Airport         | Page Rank          |
# MAGIC | :-------------- | -----------------: |
# MAGIC | ATL (Atlanta)   | 16.327479087893977 |
# MAGIC | ORD (Chicago)   | 14.914183583200652 |
# MAGIC | DFW (Dallas)    | 14.701137047790846 |
# MAGIC | DEN (Denver)    | 11.850763383258924 |
# MAGIC | CLT (Charlotte) | 9.669263295135648  |
# MAGIC | LAX (Los Angeles)| 8.295545544957498  |
# MAGIC | MSP (Minneapolis)| 7.991521878718972  |
# MAGIC | DTW (Detroit)   | 7.30569515810981   |
# MAGIC | IAH (Houston)   | 7.279631520573186  |
# MAGIC | PHX (Phoenix)   | 6.747711047955983  |
# MAGIC | ...   | ...  |
# MAGIC
# MAGIC These findings directly inform our feature engineering strategy and model development approach detailed in subsequent sections.
# MAGIC
# MAGIC ## Data Dictionary of Raw Features
# MAGIC
# MAGIC As explained in previous section(s), we systematically profiled the raw dataset to classify each variable by functional type—continuous numerical, categorical, or text—based on domain context. This classification informs critical preprocessing decisions including encoding methods, scaling techniques, and feature selection approaches. Full feature specifications are documented in Appendix B. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Feature Engineering
# MAGIC
# MAGIC **24 input features** across **4 Families** were utilized in modeling. Depending on the model architecture (Linear vs. Tree-based), categorical features were processed differently—utilizing One-Hot Encoding for Logistic Regression to capture specific value effects (e.g., "Flying Delta"), and Ordinal Indexing for XGBoost to reduce dimensionality while preserving split capability.
# MAGIC
# MAGIC | Feature Family | Raw Count | Description | Specific Features |
# MAGIC | :--- | :--- | :--- | :--- |
# MAGIC | **Route & Carrier** | 5 | Network structure and airline identifiers. | `ORIGIN`\*, `DEST`\*, `OP_UNIQUE_CARRIER`\*\*, `DISTANCE`, `PREV_ORIGIN`\* |
# MAGIC | **Temporal** | 7 | Cyclical time representations and calendar data. | `MONTH`\*\*, `QUARTER`\*\*, `DAY_OF_WEEK`\*\*, `sched_hour`\*\*, `DEP_HOUR_SIN` / `COS`, `ARR`/`DEP_TIME_BLK`\*, `IS_HOLIDAY`|
# MAGIC | **Weather** | 8 | Meteorological conditions at departure. | `STORM_INDEX`, `HourlyWindSpeed`, `HourlyAltimeterSetting`, `HourlyDewPointTemperature`, `HourlyDryBulbTemperature`, `HourlyRelativeHumidity`, `HourlyPressureTendency`, `HourlyPrecipitation` |
# MAGIC | **Turnaround / Buffer** | 2 | The data from previous flights. | `SCHEDULED_BUFFER`, `IS_PREV_DEP_DEL15` |
# MAGIC
# MAGIC \* **Preprocessing A:** Target Encoded (Smoothed Mean) or Ordinal Indexed (for Trees)
# MAGIC \** **Preprocessing B:** One-Hot Encoded (for Linear) or Ordinal Indexed (for Trees)
# MAGIC
# MAGIC **Feature Engineering Calculations:**
# MAGIC *   **SCHEDULED_BUFFER:** Calculated as the time between the scheduled arrival of the previous flight and the scheduled departure of the current flight. This captures the "breathing room" available to absorb delays.
# MAGIC *   **STORM_INDEX:** A composite severity metric combining `HourlyPrecipitation` and `HourlyWindSpeed` to isolate storm conditions from standard weather events.
# MAGIC *   **DEP_HOUR_SIN and COS:** Sine/Cosine encoding of the scheduled departure hour. This transforms time into a cyclical feature, correctly mathematically positioning the 23rd hour (11 PM) close to the 0th hour (Midnight), which linear models cannot do with raw integers.
# MAGIC *   **Target Encoding (Airports):** To manage the high cardinality of airport codes without data leakage, we replaced airport identifiers with a smoothed global mean of their delay probability from the training set.
# MAGIC *   **IS_PREV_DEP_DEL15:** A binary indicator of whether the incoming aircraft was delayed on its previous leg. **Leakage Prevention:** This feature is only populated if the previous flight's delay status was finalized at least 2 hours and 15 minutes before the current flight's departure, ensuring strictly causal predictions at the T-2h window.
# MAGIC
# MAGIC
# MAGIC **Class Imbalance Strategy:<br>**
# MAGIC   * **Balanced** the training set via **undersampling** the non-delayed flights until the proportion of delay vs non-delays were equal in the training set<br>
# MAGIC   * Weighted delay examples in training data by 2.0x that of non-delays via using weightCol parameter on LogisticRegression and GBTClassifier<br>
# MAGIC   * Models were later calibrated via threshold manipulation to account for differences in training vs test set class distributions.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Modeling Pipeline
# MAGIC ### 3.1 Machine Learning Algorithms and Metrics
# MAGIC For both mean and median, late aircraft (i.e. previous flight's delay) is highest
# MAGIC We have decided to take a binary classification approach to the problem.
# MAGIC We will predict between two classes: 
# MAGIC * Delayed: the flight departs greater than 15 minutes after the scheduled departure time (Postive Class)
# MAGIC * Not Delayed: the flight departs early or up to 15 minutes after the scheduled departure time (Negative Class)
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### 3.2 Algorithms and Associated Loss Functions:
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
# MAGIC 4) **Multilayer Perceptron**
# MAGIC <br>Neural Network built on stacked fully connected layers with nonlinear activations.
# MAGIC <br>Loss Function: Binary Cross-Entropy:
# MAGIC $$\mathcal{L}_{\text{MLP}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\Big]$$
# MAGIC <br>Parameters: `hidden_layers`, `max_iter`, `step_size`, `block_size`, `threshold`
# MAGIC
# MAGIC ### 3.3 Metrics and Analysis
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
# MAGIC ### 3.4 Visualization of the modeling pipelines and subpipelines 
# MAGIC
# MAGIC Given the following diagram, the Flight Delay Prediction ML Pipeline consists of a 4-stage machine learning workflow:
# MAGIC
# MAGIC 1. __Common Pipeline__: Index categorical features using StringIndexer
# MAGIC 2. __Hyperparameter Tuning__: For logistic regression (L1, L2, ElasticNet) and ensemble models (decision tree, random forest, gradient boosting)
# MAGIC 3. __Model Training__: With possible preprocessing (one-hot encoding, scaling) and hyperparameter tuning via block grid search
# MAGIC 4. __Model Evaluation__: F1-Macro, Accuracy, Precision, Recall, confusion matrix, and feature importance
# MAGIC

# COMMAND ----------

from IPython.display import display, HTML

with open('img/phase3/w261_proj_presentation_iii_ml_pipeline.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC #4.Data Leakage Analysis
# MAGIC
# MAGIC
# MAGIC **Data leakage** occurs when information from outside the training dataset—particularly information that would not be available at prediction time—is inadvertently used to train a model. Leakage leads to overly optimistic performance metrics during development that fail to generalize to real-world deployment, as the model learns patterns it cannot exploit in production.
# MAGIC
# MAGIC **Hypothetical Example in Flight Delay Prediction:**
# MAGIC
# MAGIC Consider a model that includes `ARR_DELAY` (actual arrival delay) as a feature to predict `DEP_DEL15` (departure delay). Since arrival delay is only known after a flight completes, this feature would be unavailable at prediction time. A model trained with this feature might achieve near-perfect accuracy in training but would be useless in practice—we cannot know arrival delay before a flight departs. Similarly, using `WEATHER_DELAY` (a post-hoc categorization assigned after delays occur) would constitute leakage, as this field is populated only after the delay cause is determined.
# MAGIC
# MAGIC **Our Approach to Balancing Leakage Prevention and Model Performance:**
# MAGIC
# MAGIC Previous flight delay (`PREV_DEP_DELAY`) is a highly predictive feature due to cascading effects in airline operations—when one flight is delayed, subsequent flights using the same aircraft are more likely to experience delays. However, including this feature unconditionally would introduce leakage for flights with short turnarounds, where the previous flight's actual delay would not yet be known at prediction time.
# MAGIC
# MAGIC Rather than excluding this valuable feature entirely, we implemented a conditional inclusion strategy. We include previous flight delay only when the previous flight's scheduled departure is at least 2 hours and 15 minutes before the current flight's scheduled departure. This ensures that the previous flight would have already departed (and its delay status known) by our T-2h prediction window, making the information realistically available in a production setting. For flights with tighter connections, we exclude previous flight delay to prevent leakage. This approach allows us to capture the predictive power of cascading delays while maintaining the integrity of our evaluation metrics.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Cardinal Sins of ML
# MAGIC
# MAGIC We reviewed our pipeline against common ML pitfalls:
# MAGIC
# MAGIC | Cardinal Sin | Description | Our Status |
# MAGIC |:-------------|:------------|:-----------|
# MAGIC | **Target Leakage** | Using features derived from or correlated with the target that wouldn't be available at prediction time | ✓ Addressed |
# MAGIC | **Train-Test Contamination** | Information from test set influencing training (e.g., fitting scalers on full data) | ✓ Addressed |
# MAGIC | **Temporal Leakage** | Using future information to predict past events | ✓ Addressed |
# MAGIC | **Preprocessing Leakage** | Applying transformations (e.g., target encoding) before train-test split | ✓ Addressed |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Pipeline Leakage Review
# MAGIC
# MAGIC We systematically audited our pipeline to ensure no leakage exists:
# MAGIC
# MAGIC #### (1). Feature Availability at T-2h
# MAGIC
# MAGIC All features are validated to be realistically available two hours before scheduled departure (T-2h), our designated prediction time:
# MAGIC
# MAGIC | Feature Category | Features | Availability | Leakage Risk |
# MAGIC |:-----------------|:---------|:-------------|:-------------|
# MAGIC | Scheduled Flight Info | `CRS_DEP_TIME`, `ORIGIN`, `DEST`, `OP_CARRIER` | Known at booking | None |
# MAGIC | Weather | `STORM_INDEX`, `HourlyRelativeHumidity`, etc. | Available via forecast/observation at T-2h | None |
# MAGIC | SCHEDULED_BUFFER | Derived from scheduled times only | Known at booking | None |
# MAGIC | Previous Flight (Scheduled) | `PREV_CRS_DEP_TIME`, `PREV_CRS_ARR_TIME` | Known at booking | None |
# MAGIC | Previous Flight Delay | `PREV_DEP_DELAY` | Conditionally available (included only when previous flight scheduled departure ≥2h 15min before current flight) | Mitigated via conditional inclusion |
# MAGIC | Airport Encoding | Target-encoded airport features | Computed on training data only | None |
# MAGIC
# MAGIC **Features Explicitly Excluded:**
# MAGIC
# MAGIC `DEP_DELAY`, `ARR_DELAY`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, and `LATE_AIRCRAFT_DELAY` are all post-hoc fields unavailable at prediction time and are excluded from our feature set. These fields are only populated after a flight completes or after delay causes are determined, making them unsuitable for prediction.
# MAGIC
# MAGIC #### (2). SCHEDULED_BUFFER Design
# MAGIC
# MAGIC Our `SCHEDULED_BUFFER` feature measures the scheduled turnaround time between a previous flight's scheduled arrival and the current flight's scheduled departure. This is intentionally derived from scheduled times only, not actual times, ensuring availability at T-2h without leakage. This feature captures operational constraints—flights with shorter scheduled buffers have less room to absorb upstream delays—without requiring any information that would be unavailable at prediction time.
# MAGIC
# MAGIC #### (3). Previous Flight Delay Feature Engineering
# MAGIC
# MAGIC To balance leakage risk and model performance, we include the previous flight's delay as a feature only when its scheduled departure is at least 2 hours and 15 minutes before the current flight's scheduled departure; closer cases are excluded to prevent potential leakage.
# MAGIC
# MAGIC We are cautious about using prior flight information because some data points fall within this 2-hour-15-minute window. However, the previous flight's delay is an important predictor of the current flight's delay due to cascading effects. To balance these considerations, we only use previous-flight delay information when it falls outside this window. Specifically, if the previous flight's scheduled departure time is earlier than the current flight's scheduled departure time minus 2 hours and 15 minutes, we include whether the previous flight was delayed; otherwise, we exclude this information. This approach is also practical from an engineering standpoint, as airport systems can be queried to obtain previous flight delay status for flights outside the window.
# MAGIC
# MAGIC For example, if a flight is scheduled to depart at 2:15 PM, the cutoff for using the previous flight's delay status is 12:00 PM (2:15 PM minus 2 hours and 15 minutes). If the previous flight was scheduled to depart before 12:00 PM, we check whether it was delayed and use that information as a boolean feature. If the previous flight was scheduled at or after 12:00 PM, we exclude its delay status to avoid potential data leakage.
# MAGIC
# MAGIC <div style="text-align: center;">
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/main/w261_proj_pre_delay.png" width="800">
# MAGIC </div>
# MAGIC
# MAGIC #### (4). Target Encoding Protocol
# MAGIC
# MAGIC Target encoding for high-cardinality airport features was implemented with leakage prevention. Encoding statistics (mean delay rates) were computed only on training data/folds, with smoothing applied to prevent overfitting on low-frequency categories. Test and validation sets were encoded using training-derived statistics only, and encoding was recalculated for each cross-validation fold to prevent information bleeding across temporal boundaries.
# MAGIC
# MAGIC #### (5). Block Time-Series Cross-Validation
# MAGIC
# MAGIC Our cross-validation strategy respects temporal ordering to prevent future data from influencing past predictions. Training data always precedes validation data chronologically, with no random shuffling that could mix future observations into training. Hyperparameters were selected based on forward-looking validation only.
# MAGIC ```
# MAGIC Fold 1: Train [2015 Q1 - 2015 Q4] → Validate [2016 Q1]
# MAGIC Fold 2: Train [2016 Q1 - 2016 Q4] → Validate [2017 Q1]
# MAGIC Fold 3: Train [2017 Q1 - 2017 Q4] → Validate [2018 Q1]
# MAGIC Fold 4: Train [2018 Q1 - 2018 Q3] → Validate [2018 Q4]
# MAGIC Final:  Train [2015-2018] → Test [2019]
# MAGIC ```
# MAGIC
# MAGIC Further detail regarding the cross-validation procedure is provided in the Cross-Validation Analysis section.
# MAGIC
# MAGIC #### (6). Preprocessing Pipeline
# MAGIC
# MAGIC All preprocessing steps are fitted on training data only. Scalers and normalizers are fitted on training folds and applied to validation/test sets. Missing value imputation statistics are derived from training data only. No global statistics are computed across the train-test boundary, ensuring complete separation between training and evaluation phases.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Summary
# MAGIC
# MAGIC Our pipeline is designed to simulate realistic deployment conditions where predictions must be made two hours before scheduled departure using only information available at that time. Through careful feature engineering—including conditional inclusion of previous flight delay data to balance predictive power with leakage prevention—temporal cross-validation, and strict separation of training and evaluation data, we ensure that reported model performance reflects true generalization capability without inflation from data leakage.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Neural Network (MLP)
# MAGIC
# MAGIC
# MAGIC Beyond traditional machine learning approaches, we explored a Multi-Layer Perceptron (MLP) neural network to assess whether deep learning could capture complex non-linear relationships in our flight delay prediction task. While tree-based models like XGBoost excel at handling categorical features and feature interactions, neural networks offer the potential to learn hierarchical representations that may capture subtle patterns missed by other approaches. Additionally, MLPs can naturally model complex interactions between weather conditions, temporal patterns, and operational factors without requiring explicit feature engineering for every interaction.
# MAGIC
# MAGIC Our EDA revealed that many features exhibit non-linear relationships with delays—for example, wind speed and precipitation show threshold effects rather than linear correlations. Neural networks are well-suited to capture such non-linearities through their activation functions and layered architecture. We also sought to compare whether the additional model complexity of a neural network would translate to improved delay detection, particularly for the minority delay class where recall is critical.
# MAGIC
# MAGIC The following sections detail our MLP architecture, hyperparameter tuning process, and results.

# COMMAND ----------

# MAGIC %md
# MAGIC ##5.1 Hyperparameter tuning
# MAGIC
# MAGIC  - We implemented a Neural Network (MLP) using a standard deep learning framework. We defined the following hyperparameters: hidden layers, maximum iterations, step size, block size, and a decision threshold. The first four hyperparameters are used during training, while the threshold is used to convert predicted probabilities to binary labels. We included a threshold because the training set is balanced but the validation set is imbalanced.
# MAGIC
# MAGIC  - We performed a grid search over all hyperparameter combinations and evaluated accuracy, precision, recall, and F1 on the validation set. The best configuration was chosen by highest F1-score
# MAGIC
# MAGIC | Hyperparameter | Search Space                                | Best Value        |
# MAGIC |:---------------|:--------------------------------------------|:------------------|
# MAGIC | hidden_layers  | [64, 32], [128, 64], **[128, 64, 32]**        | [128, 64, 32]     |
# MAGIC | max_iter       | [50, **100**]                                 | 100               |
# MAGIC | step_size      | [**0.01**, 0.03]                              | 0.01              |
# MAGIC | block_size     | [128]                                     | 128               |
# MAGIC | threshold      | [0.5, 0.51, 0.52, 0.53, 0.54, 0.55]       | 0.5               |
# MAGIC
# MAGIC  - As expected, the best model used the most hidden layers ([128, 64, 32]), was trained for the maximum 100 iterations, and employed a smaller step size of 0.01 with a block size of 128. The threshold was set to 0.5 to prioritize recall in our flight delay prediction task.
# MAGIC
# MAGIC ### Calibration via threshold tuning
# MAGIC
# MAGIC The threshold hyperparameter controls the precision–recall trade-off: increasing the threshold generally raises precision and lowers recall, and vice versa. We selected 0.5 not because it maximized F1, but because it yielded the best recall. In this flight delay prediction task, missing a delay (false negative) is more critical than raising a false alarm (false positive), so we prioritized recall. We also kept the threshold at 0.5 across models to maintain consistency for comparison.
# MAGIC
# MAGIC ### 5.2 Epochs Analysis
# MAGIC
# MAGIC  - Convergence and optimal epoch: Metrics level off early and reach their best values near epoch 20 (highest accuracy and macro F1) on the 1-year train/validation set. Consequently, we avoided using an excessively high maximum epoch count during hyperparameter tuning.
# MAGIC  - Best values (though marginal): F1-macro and Accuracy peak at epoch 20; Recall trends slightly upward through epoch 100; Precision is flat and low throughout.
# MAGIC
# MAGIC | Metric | Epoch=5 | Epoch=10 | Epoch=20 | Epoch=50 | Epoch=100 |
# MAGIC |--------|---------|----------|----------|----------|-----------|
# MAGIC | F1 Macro | 0.5935 | 0.5933 | **0.5977** | 0.5927 | 0.5961 |
# MAGIC | Recall | 0.5632 | 0.5650 | 0.5598 | 0.5692 | **0.5711** |
# MAGIC | Precision | 0.2772 | 0.2769 | **0.2821** | 0.2762 | 0.2801 |
# MAGIC | Accuracy | 0.7148 | 0.7141 | **0.7209** | 0.7123 | 0.7161 |
# MAGIC
# MAGIC ### 5.3 MLP Model performance
# MAGIC
# MAGIC The best model's performance on the validation and test datasets is presented, using a 5-year dataset divided into training, validation, and test sets.

# COMMAND ----------

from IPython.display import display, HTML

with open('img/phase3/w261_proj_presentation_mlp_result_merge.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Confusion Matrix Analysis
# MAGIC
# MAGIC The test-set confusion matrix shows that most delayed flights were correctly identified as delayed (863,667). However, a large number of non-delayed flights were incorrectly predicted as delayed (2,066,620), indicating a tendency to over-predict delays. This could result in unnecessary interventions or resource allocations due to false alarms. Overall, the confusion matrix offers valuable insight into model performance and points to areas for improvement.
# MAGIC
# MAGIC <p><img src="https://raw.githubusercontent.com/hong-hu/w261/main/w261_proj_mlp_confusion_matrix.png" width="700"></p>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 6. Results and Discussion of Results (OTPW_V2 Custom Join Data)
# MAGIC
# MAGIC Having established our feature engineering approach, addressed data leakage concerns, and defined our model architectures, we now present the experimental results comparing all models. This section details our experimental setup, cross-validation performance, and final test set evaluation to assess which approach best balances overall predictive accuracy with delay detection capability.

# COMMAND ----------

# MAGIC %md
# MAGIC ##6.1 Experimental Setup
# MAGIC * **Total Models:** 4 (Baseline, Logistic Regression, XGBoost, MLP), with Baseline, Logistic Regression, and XGBoost having 4-Fold Block Cross-Validation for each hyperparameter combination.
# MAGIC * **Cluster Configuration**: 16.4 LTS (incluedes Apache Spark 3.5.2, Scala 2.12)
# MAGIC * **Worker type**: m5d.2xlarge, 16 Gb Memory, 4 Cores (Min: 6, Max: 12, Current: 6)

# COMMAND ----------

# MAGIC %md
# MAGIC ##6.2 Model Results for 5Y Data (Training Set 2015-2018, Test Set 2019)

# COMMAND ----------

# MAGIC %md
# MAGIC | Model | F1-Score (Macro) | Accuracy | Precision (Delayed) | Recall (Delayed) | Time |
# MAGIC | :--- | :--- | :--- | :--- | :--- | :--- | 
# MAGIC | **Baseline** | 0.449 | 0.8133 | 0.000 | 0.000 | 2.55 min |
# MAGIC | **Logistic Regression** | 0.583 | 0.657 | 0.300 | 0.628 | 5.00 min |
# MAGIC | **XGBoost** | 0.618 | 0.703 | 0.336 | 0.630 | 8.50 min |
# MAGIC | **Multilayer Perceptron** | 0.580 | 0.651 | 0.297 | 0.652 | 18.75 min |

# COMMAND ----------

# MAGIC %md
# MAGIC ####Interpretation of Model Metrics
# MAGIC
# MAGIC All models were tested on the same test set.
# MAGIC <br>
# MAGIC The Majority Class Baseline achieved the highest overall accuracy of 81.33% simply by predicting "Not Delayed" for every flight. However, this model provides zero operational value, achieving a Recall of 0.0% and failing to identify a single delay. This establishes a "Null Accuracy" benchmark: any useful model must outperform the baseline on Recall and Macro F1-Score, even if it sacrifices raw Accuracy. By focusing on the F1-Macro metric, we measure a model's success in predicting both delays and non-delays in a balanced way.<br><br>
# MAGIC The machine learning models (Logistic Regression, XGBoost, and MLP) clearly demonstrate the effects of training on undersampled data. Unlike the Baseline, these models sacrificed raw accuracy (dropping from about 81% to a range of 65-70%) to achieve significant gains in Recall.<br><br>
# MAGIC The XGBoost model emerged as the strongest performer, achieving the highest Macro F1-Score of 0.618. It struck the best balance between False Positives and False Negatives. While the Multilayer Perceptron achieved the highest Delay Recall (65.2%), it did so at the cost of the lowest Accuracy (65.1%) and Delay Precision, indicating a higher rate of false alarms. XGBoost maintained a significantly higher accuracy (70.3%) than both Logistic Regression and MLP while still successfully recalling 63.0% of delays.<br><br>
# MAGIC By accepting an 11 percentage point drop in overall accuracy compared to the Baseline, the XGBoost model provides a system capable of predicting nearly 63% of delays, offering actionable intelligence that the high-accuracy Baseline lack.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Comparison to Phase II Results (1Y Data: 75% Train/25% Test Chronological Split)
# MAGIC This table compares the validation accuracy of Phase II modeling to Test accuracy in Phase III:
# MAGIC
# MAGIC | Model | Metric | Phase II Results <br>*(1Y Imbalanced)* | Phase III Results <br>*(5Y Balanced)* | Change |
# MAGIC | :--- | :--- | :--- | :--- | :--- |
# MAGIC | **Baseline** | Accuracy | 82.7% | 81.3% | ▼ 1.4% |
# MAGIC | *(Majority Class)* | **F1-Macro** | **0.453** | **0.449** | **▼ 0.004** |
# MAGIC | | Recall (Delayed) | 0.0% | 0.0% | - |
# MAGIC | | | | | |
# MAGIC | **Logistic Regression** | Accuracy | 80.4% | 65.7% | ▼ 14.7% |
# MAGIC | | **F1-Macro** | **0.562** | **0.583** | **▲ 0.021** |
# MAGIC | | Recall (Delayed) | 17.6% | 62.8% | ▲ 45.2% |
# MAGIC | | Precision (Delayed)| 35.9% | 30.0% | ▼ 5.9% |
# MAGIC | | | | | |
# MAGIC | **Gradient Boosting** | Accuracy | 79.4% | 70.3% | ▼ 9.1% |
# MAGIC | | **F1-Macro** | **0.604** | **0.618** | **▲ 0.014** |
# MAGIC | | Recall (Delayed) | 29.3% | 63.0% | ▲ 33.7% |
# MAGIC | | Precision (Delayed)| 37.5% | 33.6% | ▼ 3.9% |
# MAGIC | | | | | |
# MAGIC | **Multilayer Perceptron** | Accuracy | N/A | 64.8% | *New Model* |
# MAGIC | | F1-Macro | N/A | 0.577 | *New Model* |
# MAGIC | | Recall (Delayed) | N/A | 63.6% | *New Model* |
# MAGIC | | Precision (Delayed) | N/A | 29.5% | *New Model* |

# COMMAND ----------

# MAGIC %md
# MAGIC Phase III represents a significant evolution from the Phase II baseline, expanding the dataset from 1 year to 5 years, implementing additional feature engineering, and introducing a Multilayer Perceptron (MLP) model. Most notably, Phase III addressed the severe class imbalance present in Phase II by applying 1:1 undersampling to the training set. This strategic shift has fundamentally altered the model performance profile, prioritizing the detection of delays over raw accuracy, while maintaining performance on the primary metric of F1-Macro.
# MAGIC <br><br>
# MAGIC In Phase II (imbalanced training), the models struggled to identify the minority class. The best performer in Phase II was the XGBoost model, which only achieved a Recall of 29.3%, effectively missing 7 out of 10 delays.
# MAGIC In Phase III (balanced training), the XGBoost model’s Recall more than doubled to 63.0%. Similarly, Logistic Regression saw a massive jump in sensitivity, rising from a Recall of 17.6% in Phase II to 62.8% in Phase III.
# MAGIC <br><br>
# MAGIC This dramatic improvement in delay detection came at the expected cost of overall accuracy and precision. In Phase II, the models were biased toward the majority class ("No Delay"), resulting in artificially high accuracy (LR: 80.4%, GBT: 79.4%) that closely mirrored the majority-class Baseline.
# MAGIC In Phase III, by forcing the models to pay attention to delays, we accepted a decrease in accuracy (LR: 65.7%, GBT: 70.3%). However, the Phase III XGBoost model showed the most resilience, retaining the highest accuracy of the group while matching the high recall of the other models.
# MAGIC <br><br>
# MAGIC While Logistic Regression saw the most volatility between phases (dropping nearly 15% in accuracy to gain recall), the XGBoost model proved more robust to the dataset changes, maintaining a 70% accuracy threshold. Additionally, the newly added MLP model performed comparably to Logistic Regression, achieving the highest raw Recall (65.2%) but suffering from the lowest Precision (0.297), confirming that tree-based methods (GBT) currently offer the best operational balance for this dataset.
# MAGIC <br><br>
# MAGIC By leveraging 5 years of data and balanced sampling, we have moved from a system that misses the majority of delays to one that captures nearly two-thirds of them.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 Confusion Matrix Analysis (OTPW_V2 Custom Join Data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Baseline (Majority Class)
# MAGIC | | Predicted Negative (No Delay) | Predicted Positive (Delay) |
# MAGIC | :--- | :--- | :--- |
# MAGIC | **Actual Negative** | 5,915,671 | 0 |
# MAGIC | **Actual Positive** | 1,358,381 | 0 |
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ### 2. Logistic Regression
# MAGIC | | Predicted Negative (No Delay) | Predicted Positive (Delay) |
# MAGIC | :--- | :--- | :--- |
# MAGIC | **Actual Negative** | 3,928,232 | 1,987,439 |
# MAGIC | **Actual Positive** | 504,797 | 853,584 |
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ### 3. XGBoost
# MAGIC | | Predicted Negative (No Delay) | Predicted Positive (Delay) |
# MAGIC | :--- | :--- | :--- |
# MAGIC | **Actual Negative** | 4,302,891 | 1,612,780 |
# MAGIC | **Actual Positive** | 537,037 | 821,344 |
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ### 4. Multilayer Perceptron (MLP)
# MAGIC | | Predicted Negative (No Delay) | Predicted Positive (Delay) |
# MAGIC | :--- | :--- | :--- |
# MAGIC | **Actual Negative** | 3,849,051 | 2,066,620 |
# MAGIC | **Actual Positive** | 494,714 | 863,667 |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####Interpretation of Confusion Matrices
# MAGIC The Baseline matrix is entirely concentrated in the left column. This visualizes the fundamental problem with accuracy: while the True Negative quadrant is massive, the True Positive quadrant is empty. The model completely ignores the 1.35 million actual delays. In contrast, all ML models show a significant shift to the right-hand column, now correctly identifying between **821,000 and 863,000 delays**.<br>
# MAGIC
# MAGIC The trade-off for catching these delays is visible in the top-right quadrant (False Positives). By balancing the training set, we forced the models to become hyper-sensitive. Every model now generates "False Alarms," predicting a delay where none occurred.<br>
# MAGIC
# MAGIC The MLP achieved the highest raw volume of **True Positives (863,667)**, followed closely by Logistic Regression (**853,584**). However, they achieved this by casting an extremely wide net, resulting in roughly **2 million False Positives** each. These models are highly "risk-averse" regarding delays, flagging almost any suspicious flight at the cost of high noise.<br>
# MAGIC
# MAGIC The XGBoost matrix reveals why it is the superior operational model. While it identified slightly fewer True Positives (**821,344**) than the MLP, it was significantly more balan. The XGBoost generated about **450,000 fewer False Positives** (1.61M vs 2.06M) than the MLP.<br> 
# MAGIC
# MAGIC While the MLP catches the most delays, the **Gradient Boosting model** offers the most viable real-world solution. It maintains high sensitivity to delays while minimizing the amount of false positives generated by the more aggressive linear and neural network approaches.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 Top 15 Features by Importance (XGBoost - 5 Years Data)
# MAGIC
# MAGIC #### A. Global Feature Importance (XGBoost - Best Model)
# MAGIC *Metric: Gain (Relative contribution of the corresponding feature to the model)*
# MAGIC
# MAGIC | Rank | Feature | Importance | Operational Insight |
# MAGIC | :--- | :--- | :--- | :--- |
# MAGIC | 1 | **ORIGIN_index** | **0.238** | **Infrastructure:** The airport of origin is the single strongest predictor, capturing congestion and capacity limits. |
# MAGIC | 2 | **IS_PREV_DEP_DEL15** | **0.202** | **Propagation:** The "Knock-on" effect. If the incoming aircraft was delayed, the next flight is highly likely to be delayed. |
# MAGIC | 3 | **DEST_index** | 0.160 | **Network:** Destination airport conditions account for air traffic control holds and arrival capacity. |
# MAGIC | 4 | **ARR_TIME_BLK_index** | 0.075 | **Schedule:** The block of time the flight arrives (e.g., peak evening rush vs. quiet night). |
# MAGIC | 5 | **DEP_TIME_BLK_index** | 0.063 | **Schedule:** The block of time the flight departs. |
# MAGIC | 6 | **OP_UNIQUE_CARRIER_index** | 0.062 | **Operations:** The airline operator itself. |
# MAGIC | 7 | **HourlyRelativeHumidity** | 0.061 | **Weather:** The most impactful weather metric, serving as a proxy for fog and storm potential. |
# MAGIC | 8 | **HourlyDryBulbTemperature** | 0.060 | **Weather:** Extreme heat or cold impacting lift and ground crews. |
# MAGIC
# MAGIC #### B. Directional Feature Impact (Logistic Regression)
# MAGIC *Metric: Coefficient (Positive = Increases Delay Risk; Negative = Decreases Delay Risk)*
# MAGIC
# MAGIC | Direction | Feature | Coeff | Interpretation |
# MAGIC | :--- | :--- | :--- | :--- |
# MAGIC | **Increases Risk (+)** | `HourlyRelativeHumidity` | **+0.266** | High humidity is the #1 directional driver of delays in the linear model. |
# MAGIC | | `SCHEDULED_BUFFER` | **+0.254** | Flights with tight turnaround buffers are significantly more prone to delays. |
# MAGIC | | `DEST_encoded` | +0.139 | Flying into historically busy/delayed airports increases risk. |
# MAGIC | | `OP_UNIQUE_CARRIER_ohe_NK`| +0.058 | Flying Spirit Airlines (NK) shows a slight statistical increase in delay odds. |
# MAGIC | | | | |
# MAGIC | **Decreases Risk (-)** | `DEP_HOUR_SIN` | **-0.228** | Cyclical time encoding; aligns with early morning hours reducing risk. |
# MAGIC | | `sched_hour_ohe_6` | **-0.193** | 6:00 AM departures. |
# MAGIC | | `HourlyDewPointTemperature`| -0.152 | Lower dew points (drier air) reduce delay risk. |
# MAGIC | | `OP_UNIQUE_CARRIER_ohe_DL`| **-0.115** | Flying Delta Airlines (DL) significantly reduces delay odds. |
# MAGIC
# MAGIC #### Interpretation of Feature Importance
# MAGIC The most significant finding from the XGBoost model is the massive importance of `IS_PREV_DEP_DEL15` (20.2%). This confirms that **delay propagation** is a dominant force; once a plane is late, the schedule rarely recovers. Interestingly, the Logistic Regression model prioritized `SCHEDULED_BUFFER` (+0.254) instead. This suggests that while the linear model looks at the *planned* gap between flights, the tree model correctly identifies that the *actual* status of the incoming plane is the superior signal.<br><br>
# MAGIC
# MAGIC Both models converged on **Humidity** as the primary weather threat, outranking visible factors like precipitation or wind speed.
# MAGIC *   **Linear View:** Positive coefficient (+0.266) confirms that as humidity rises, delays rise.
# MAGIC *   **Operational View:** High humidity impacts air density (lift), indicates fog potential (low visibility), and often precedes convective storms. It is the "silent killer" of on-time performance.<br><br>
# MAGIC
# MAGIC The Logistic Regression coefficients provide a clear operational recommendation for avoiding delays: **Fly Early.**
# MAGIC The features `sched_hour_ohe_6`, `_5`, and `_7` (5 AM to 7 AM) all possess strong negative coefficients. This is the "reset" period where the airspace is clear, and "knock-on" delays haven't accumulated yet. Conversely, the XGBoost model simply flags "Time Blocks" as important, recognizing that the time of day is crucial, but the Linear model explicitly tells us *which* times are safe.<br><br>
# MAGIC
# MAGIC The models quantified the reputation of specific airlines. The linear model shows a protective effect for **Delta Airlines** (-0.115), statistically validating their higher on-time performance reputation during this period. Conversely, **Spirit Airlines** showed a positive coefficient (+0.058), associating the carrier with a marginally higher risk profile in the model's decision boundary.

# COMMAND ----------

# MAGIC %md
# MAGIC ##6.6 Block 4-Fold Validation Results (5 Years Data):
# MAGIC <img src="https://raw.githubusercontent.com/hong-hu/w261/refs/heads/main/block_cv_rolling_window.jpg"> <br>
# MAGIC | Model | Hyperparameter | Search Space Tested | Best Value Found |
# MAGIC | :--- | :--- | :--- | :--- |
# MAGIC | **Logistic Regression** | `regParam` | [0.1, 0.01, 0.001, 0.0001] | **0.001** |
# MAGIC | | `elasticNetParam` | [0.0, 0.5, 1.0] | **0.0** |
# MAGIC | | `maxIter` | [50] | **50** |
# MAGIC | | | | |
# MAGIC | **XGBoost** | `max_depth` | [6, 10, 14] | **6** |
# MAGIC | | `learning_rate` | [0.25, 0.1, 0.05] | **0.25** |
# MAGIC | | `n_estimators` | [100, 150, 200, 250] | **250** |
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC | Experiment | Fold | Accuracy | Recall (Delayed) | Precision (Delayed) | F1 Macro |
# MAGIC | :--- | :--- | :--- | :--- | :--- | :--- |
# MAGIC | **1. Logistic Regression** | 1 | 0.662 | 0.615 | 0.298 | 0.571 |
# MAGIC | | 2 | 0.635 | 0.542 | 0.274 | 0.548 |
# MAGIC | | 3 | 0.658 | 0.598 | 0.291 | 0.565 |
# MAGIC | | 4 | 0.671 | 0.625 | 0.305 | 0.579 |
# MAGIC | **LR Average** | | **0.657** | **0.595** | **0.292** | **0.566** |
# MAGIC | | | | | | |
# MAGIC | **2. XGBoost** | 1 | 0.715 | 0.638 | 0.342 | 0.612 |
# MAGIC | | 2 | 0.688 | 0.575 | 0.315 | 0.589 |
# MAGIC | | 3 | 0.702 | 0.612 | 0.329 | 0.601 |
# MAGIC | | 4 | 0.724 | 0.641 | 0.351 | 0.619 |
# MAGIC | **XGBoost Average** | | **0.707** | **0.617** | **0.334** | **0.605** |

# COMMAND ----------

# MAGIC %md
# MAGIC ####Cross-Validation Analysis
# MAGIC **Methodology: Block vs. Expanding Window**
# MAGIC <br>
# MAGIC For the 5-Year dataset, Block Cross-Validation was chosen over the Expanding-Window technique employed in Phase II. 
# MAGIC <br>We separated the training data (2015-2018) into 4 folds, with the first 75% of each fold being used for training, and the last 25% being used for validation, with metrics being collected for each fold. 
# MAGIC <br>All 4 folds were run for each hyperparameter combination. The final model for each architecture was trained from the hyperparameter combination with the highest average F1-Macro across all folds.
# MAGIC <br>
# MAGIC
# MAGIC While the expanding window system was useful for demonstrating model scalability with increasing data volume, it inherently biases the model toward the earliest training examples that persist across all windows. Block Cross-Validation eliminates this bias by isolating performance on specific sections of the data.
# MAGIC
# MAGIC Additionally, Block Cross-Validation is computationally more efficient as the training set size remains constant rather than growing as the window expands. This efficiency allowed us to loop the system into wider hyperparameter optimization routines, enabling a more robust determination of the best hyperparameter permutations (as seen in the tuning tables above).
# MAGIC
# MAGIC **The Impact of Balancing on Validation**
# MAGIC <br>
# MAGIC The Cross-Validation results above reflect the balanced training strategy (1:1 undersampling) implemented in Phase III. Accuracy across each fold is in the 65-70% range, while Recall has increased to approximately 60%. This confirms that the models are actively prioritizing delay detection across all time periods, not just in the final test set.
# MAGIC
# MAGIC **Model Stability and Data Volatility**
# MAGIC <br>
# MAGIC The validation results across the four folds reveal distinct patterns regarding model stability and data complexity.
# MAGIC **Fold 2** proved to be the most challenging period for both models, with Recall dropping to its lowest point (LR: 54.2%, XGBoost: 57.5%). This mirrors the seasonal volatility observed in Phase II, suggesting that this specific block of time contains complex delay drivers (likely high-traffic summer months or extreme weather periods) that are statistically harder to distinguish from on-time flights.
# MAGIC
# MAGIC **XGBoost vs. Logistic Regression**
# MAGIC Across all four folds, the XGBoost model consistently outperformed Logistic Regression, particularly in the critical metric of Delayed Flight Recall.
# MAGIC *   **Logistic Regression:** Achieved high recall (Avg 59.5%), but with lower Precision (Avg 29.2%), indicating a high rate of false alarms.
# MAGIC *   **XGBoost:** Achieved a higher Average Recall (61.7%) and higher Accuracy (70.7%). The consistently higher F1 Macro scores for XGBoost (Avg 0.605) confirm that the tree-based architecture is better at capturing the complex, non-linear interactions of flight delays without sacrificing as much overall accuracy as the linear model, even in the context of a subset of the training data.

# COMMAND ----------

# MAGIC %md
# MAGIC ##6.7 Pipeline Gap Analysis
# MAGIC While the current PySpark pipeline successfully demonstrates the viability of predicting flight delays using the XGBoost architecture, a gap analysis reveals distinct divergences between this research-grade batch pipeline and a production-grade inference system.
# MAGIC
# MAGIC **1. Input Data Discrepancy (Training-Serving Skew)**<br>
# MAGIC At the T-minus 2-hour prediction horizon, forecast data contains inherent error margins.
# MAGIC
# MAGIC **2. Data Availability Assumptions (Stream Latency)**:<br>
# MAGIC Our second most important feature, `IS_PREV_DEP_DEL15`, relies on joining the current flight row with the previous flight row. In our batch-processing pipeline, this join is computationally guaranteed because the data is static. In a live environment, this requires a low-latency streaming architecture. If the status update of the incoming aircraft is delayed by even 15 minutes due to telemetry lag, the feature might be null at the moment of prediction. The current pipeline assumes 100% data availability. It does not account for the "staleness" of upstream data feeds.
# MAGIC
# MAGIC **3. Information Loss via Undersampling Strategy**
# MAGIC To address the severe class imbalance, the pipeline utilized **1:1 Random Undersampling**, effectively discarding millions of "On-Time" training examples to create a balanced dataset.
# MAGIC While this successfully forced the model to prioritize Recall (improving from ~20% to ~63%), it introduced a bias. By removing so many "normal" flights, the model may have lost the ability to learn subtle patterns that distinguish "safe" high-traffic periods from "delayed" high-traffic periods.
# MAGIC
# MAGIC **4. Prediction Resolution (Binary vs. Regression)**
# MAGIC he pipeline frames the problem as a Binary Classification (>15 mins or Not).
# MAGIC However, not all delays are equal. A 16-minute delay is a minor inconvenience; a 4-hour delay is a logistical crisis. The current pipeline treats these two scenarios identically.
# MAGIC The model optimizes for the binary boundary. It might confidently predict a "Delay" for a flight that is 16 minutes late, while missing a flight that is 3 hours late but has complex, outlier features. Implementing multiple delay bins or some sort of regression system would alleviate this scenario.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Prediction for Recent Years

# COMMAND ----------

# MAGIC %md
# MAGIC As we have pulled the data for recent years from 2020 to 2024, we can observe the performance trends of three different models: XGBoost, Logistic Regression, and MLP (Multi-Layer Perceptron).
# MAGIC
# MAGIC  - **COVID-19 Impact was Evident:** The data strongly supports the initial premise that the unusual circumstances of 2020 (Covid-19 pandemic) severely hampered the predictive capabilities of all three models.
# MAGIC  - **Post-Pandemic Normalization Led to Recovery:** As the situation normalized from 2021 onwards, all models exhibited significant performance recovery and continued improvement, suggesting that their underlying mechanisms are robust once data patterns stabilize.
# MAGIC  - **XGBoost is the Top Performer:** Across all evaluated metrics, XGBoost consistently delivered the highest performance, indicating its superior robustness and predictive power in this specific problem domain.
# MAGIC  - **Neural Networks (MLP) Outperform Traditional Linear Models (Logistic Regression) marginally:** While Logistic Regression showed good recovery, MLP generally achieved slightly better performance, especially in the later years (2023-2024) for metrics like Accuracy and F1 Macro.
# MAGIC  - **Precision Showed Most Volatility/Improvement:** The Precision metric exhibited the largest percentage increase from 2020 lows to 2024 highs for all models, highlighting its sensitivity to the anomalous 2020 data and its subsequent robust recovery. Recall, while recovering, showed less dramatic proportional shifts compared to Precision.
# MAGIC

# COMMAND ----------

from IPython.display import display, HTML

with open('img/phase3/w261_proj_presentation_prediction.html', 'r', encoding='utf-8') as f:
    display(HTML(f.read()))

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Conclusion
# MAGIC U.S. domestic flight delays impose a significant economic burden - $33 billion annually - with travelers absorbing more than half of these costs (FAA/NEXTOR, 2019). Reducing delays by just 10% could unlock $17.6 billion in economic value. This project aimed to predict departure delays of 15 minutes or more by integrating large-scale flight operations and weather data. Reliable predictions empower airlines to refine scheduling, enable airports to allocate resources more effectively, and help passengers plan with greater confidence.
# MAGIC
# MAGIC We hypothesized that a scalable machine learning pipeline incorporating custom-engineered features—combining flight operations data with weather observations - can accurately predict departure delays while avoiding data leakage. Specifically, we proposed that features engineered to capture weather severity, aircraft turnaround dynamics, and airport network characteristics would provide meaningful predictive signal beyond raw operational data alone.
# MAGIC
# MAGIC ### 8.1 Summary of Key Findings
# MAGIC
# MAGIC **Data and Features:**
# MAGIC
# MAGIC Our analysis leveraged approximately 31 million flight records from 2015–2019, joining U.S. Department of Transportation On-Time Performance data with NOAA Global Hourly Weather observations. Key engineered features included:
# MAGIC
# MAGIC - `STORM_INDEX`: A composite weather severity metric combining precipitation and wind factors
# MAGIC - `SCHEDULED_BUFFER`: Turnaround time derived from scheduled times only, ensuring availability at prediction time (T-2h)
# MAGIC - `HourlyRelativeHumidity`: Raw weather metric that outperformed post-hoc delay categorizations
# MAGIC - PageRank feature: Although the airport PageRank feature greatly reduces the feature space, it does not improve model performance. This aligns with practice: larger airports with more flights do not necessarily experience more delays, likely due to better facilities. Better dimension-reduction techniques are needed.
# MAGIC - Smoothed target encoding for high-cardinality airport features
# MAGIC
# MAGIC **Feature Importance:**
# MAGIC
# MAGIC Among our 24 input features, Origin (ORIGIN_index), Previous Flight Delay (IS_PREV_DEP_DEL15), and Destination (DEST_index) emerged as the top three predictors. This confirms that network position and propagation effects outweigh local weather conditions in the XGBoost decision trees.
# MAGIC Crucially, the Previous Flight Delay feature required strict leakage prevention; it was only populated if the prior flight's scheduled departure occurred at least 2 hours and 15 minutes before the current flight, ensuring the model relied only on information available during the T-2h prediction window.
# MAGIC
# MAGIC **Modeling Approach:**
# MAGIC
# MAGIC We developed a scalable PySpark pipeline comparing Logistic Regression, XGBoost, and a Multilayer Perceptron (MLP). A critical evolution in this phase was the shift to Balanced Training (1:1 Undersampling) to address the severe class imbalance (~18% delay rate). While this lowered raw accuracy compared to the baseline, it drastically improved the operational utility of the models. Expanding upon Phase II, we scaled our analysis to a 5-year dataset and transitioned to a Block 4-Fold Cross-Validation strategy to ensure model robustness across year-to-year data drift and enable more hyperparameter combinations to be searched more quickly.
# MAGIC
# MAGIC **Best Model and Results:**
# MAGIC
# MAGIC XGBoost emerged as the superior operational model, achieving the best balance between precision and recall with an F1-Macro score of 0.618 on the held-out 2019 test set. While the MLP achieved the highest raw Delay Recall (0.652), it suffered from a significantly higher False Positive rate. XGBoost offered a more surgical approach, correctly identifying 63.0% of delays while maintaining higher overall accuracy (70.3%) than the other algorithms.
# MAGIC
# MAGIC **Generalization to Future Years:**
# MAGIC
# MAGIC We evaluated our trained models on 2020~2024 data to assess generalization. Model performance degraded significantly in 2020 due to the unprecedented disruptions of the COVID-19 pandemic, which fundamentally altered flight operations and delay patterns. Performance began recovering from 2021 as conditions normalized, with 2024 being the best performing year for all models. This demonstrates that our models are robust under typical operating conditions but sensitive to extraordinary external shocks.
# MAGIC
# MAGIC ### 8.2 Significance of Results
# MAGIC
# MAGIC Our results validate that machine learning pipelines with carefully engineered features can meaningfully predict flight delays, confirming our initial hypothesis. Key takeaways include:
# MAGIC
# MAGIC 1. **Dimensionality Reduction Requires Care:** Although PageRank-based airport importance successfully reduced feature space, it did not improve model performance. This aligns with operational intuition: larger airports do not necessarily experience higher delay rates, likely due to better infrastructure. Alternative dimensionality reduction techniques warrant exploration.
# MAGIC
# MAGIC 2. **Model Selection Depends on Objectives:** XGBoost delivers balanced performance and handles categorical features effectively. MLP's superior delay recall may suit applications prioritizing delay detection, such as proactive passenger notifications.
# MAGIC
# MAGIC 3. **Robustness Under Normal Conditions:** Models generalize well under typical operating conditions but degrade during extraordinary disruptions like COVID-19, highlighting the need for performance monitoring and adaptive retraining.
# MAGIC
# MAGIC 4. **Scalability is Achievable:** Our PySpark pipeline processed 64M+ records successfully, demonstrating that delay prediction can be operationalized at industry scale.
# MAGIC
# MAGIC ### 8.3 Limitations and Future Work
# MAGIC
# MAGIC An F1-Macro of 0.618 indicates room for improvement. Future directions include:
# MAGIC
# MAGIC - Use of more sophisticated threshold tuning for better calibration and reduction of false postives
# MAGIC - Better dimension-reduction techniques for airport features beyond PageRank
# MAGIC - Real-time weather feed integration for deployment
# MAGIC - Additional data sources such as air traffic control or crew scheduling information
# MAGIC - Adaptive retraining strategies to handle regime shifts
# MAGIC
# MAGIC ### 8.4 Closing Remarks
# MAGIC
# MAGIC This project demonstrates that scalable, leakage-free machine learning pipelines can deliver actionable delay predictions, offering tangible value to airlines, airports, and passengers navigating U.S. domestic air travel.

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC ### Appendix A: Data Dictionary of Raw Data
# MAGIC
# MAGIC The complete data dictionary for all raw data fields is provided in the Phase 2 report. Please refer to the following link:
# MAGIC
# MAGIC [Phase 2 Report - Appendix A: Data Dictionary](insert_link_here)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix B: Summary Statistics (5-year custom joined data)
# MAGIC
# MAGIC | Column_Name | Data_Type | Non_Null_Count | Null_Count | Null_Pct | Min | Max | Mean | Std_Dev | Distinct_Values |
# MAGIC |:------------|:----------|---------------:|-----------:|---------:|----:|----:|-----:|--------:|----------------:|
# MAGIC | CANCELLATION_CODE | string | 489946 | 31256893 | 98.46 | null | null | null | null | 4 |
# MAGIC | HourlyPresentWeatherType | string | 3085361 | 28661478 | 90.28 | null | null | null | null | 1824 |
# MAGIC | PREV_ORIGIN | string | 31720500 | 26339 | 0.08 | null | null | null | null | 361 |
# MAGIC | PREV_DEST | string | 31720500 | 26339 | 0.08 | null | null | null | null | 359 |
# MAGIC | DATE | timestamp | 31732916 | 13923 | 0.04 | null | null | null | null | null |
# MAGIC | QUARTER | int | 31746839 | 0 | 0 | 1 | 4 | 2.52 | 1.11 | null |
# MAGIC | MONTH | int | 31746839 | 0 | 0 | 1 | 12 | 6.55 | 3.4 | null |
# MAGIC | DAY_OF_MONTH | int | 31746839 | 0 | 0 | 1 | 31 | 15.75 | 8.77 | null |
# MAGIC | DAY_OF_WEEK | int | 31746839 | 0 | 0 | 1 | 7 | 3.93 | 1.99 | null |
# MAGIC | FL_DATE | timestamp | 31746839 | 0 | 0 | null | null | null | null | null |
# MAGIC | OP_UNIQUE_CARRIER | string | 31746839 | 0 | 0 | null | null | null | null | 18 |
# MAGIC | OP_CARRIER_AIRLINE_ID | int | 31746839 | 0 | 0 | 19393 | 21171 | 19949.21 | 383.19 | null |
# MAGIC | TAIL_NUM | string | 31746839 | 0 | 0 | null | null | null | null | 7434 |
# MAGIC | OP_CARRIER_FL_NUM | int | 31746839 | 0 | 0 | 1 | 9855 | 2339.57 | 1792.12 | null |
# MAGIC | ORIGIN_AIRPORT_ID | double | 31746839 | 0 | 0 | 10135 | 16869 | 12668.72 | 1526.74 | null |
# MAGIC | ORIGIN_CITY_MARKET_ID | double | 31746839 | 0 | 0 | 30070 | 36133 | 31729.32 | 1289.46 | null |
# MAGIC | ORIGIN | string | 31746839 | 0 | 0 | null | null | null | null | 361 |
# MAGIC | ORIGIN_STATE_ABR | string | 31746839 | 0 | 0 | null | null | null | null | 55 |
# MAGIC | DEST_AIRPORT_ID | double | 31746839 | 0 | 0 | 10135 | 16869 | 12668.67 | 1526.72 | null |
# MAGIC | DEST | string | 31746839 | 0 | 0 | null | null | null | null | 359 |
# MAGIC | DEST_STATE_ABR | string | 31746839 | 0 | 0 | null | null | null | null | null |
# MAGIC | CRS_DEP_TIME | int | 31746839 | 0 | 0 | 1 | 2359 | 1330.09 | 489.87 | null |
# MAGIC | DEP_TIME | int | 31746839 | 0 | 0 | 0 | 2400 | 1314.36 | 525 | null |
# MAGIC | DEP_DELAY | double | 31746839 | 0 | 0 | -234 | 2755 | 9.71 | 43.19 | null |
# MAGIC | DEP_DELAY_NEW | double | 31746839 | 0 | 0 | 0 | 2755 | 12.72 | 42.15 | null |
# MAGIC | DEP_DEL15 | double | 31746839 | 0 | 0 | 0 | 1 | 0.18 | 0.38 | null |
# MAGIC | DEP_DELAY_GROUP | int | 31746839 | 0 | 0 | -2 | 12 | 0.04 | 2.15 | null |
# MAGIC | DEP_TIME_BLK | string | 31746839 | 0 | 0 | null | null | null | null | 19 |
# MAGIC | CRS_ARR_TIME | int | 31746839 | 0 | 0 | 1 | 2400 | 1488.9 | 516.8 | null |
# MAGIC | ARR_TIME | int | 31746839 | 0 | 0 | 0 | 2400 | 1445.67 | 562.77 | null |
# MAGIC | ARR_DELAY | double | 31746839 | 0 | 0 | -238 | 2695 | 4.53 | 45.19 | null |
# MAGIC | ARR_DELAY_NEW | double | 31746839 | 0 | 0 | 0 | 2695 | 12.73 | 41.8 | null |
# MAGIC | ARR_DEL15 | double | 31746839 | 0 | 0 | 0 | 1 | 0.18 | 0.39 | null |
# MAGIC | ARR_DELAY_GROUP | int | 31746839 | 0 | 0 | -2 | 12 | -0.21 | 2.28 | null |
# MAGIC | CANCELLED | double | 31746839 | 0 | 0 | 0 | 1 | 0.02 | 0.12 | null |
# MAGIC | DIVERTED | double | 31746839 | 0 | 0 | 0 | 1 | 0 | 0.05 | null |
# MAGIC | DISTANCE | double | 31746839 | 0 | 0 | 21 | 5095 | 823.22 | 607.68 | null |
# MAGIC | DISTANCE_GROUP | int | 31746839 | 0 | 0 | 0 | 11 | 3.77 | 2.39 | null |
# MAGIC | CARRIER_DELAY | double | 31746839 | 0 | 0 | 0 | 2695 | 3.65 | 26.5 | null |
# MAGIC | WEATHER_DELAY | double | 31746839 | 0 | 0 | 0 | 2692 | 0.59 | 11.53 | null |
# MAGIC | NAS_DELAY | double | 31746839 | 0 | 0 | 0 | 1848 | 2.82 | 16 | null |
# MAGIC | SECURITY_DELAY | double | 31746839 | 0 | 0 | 0 | 1078 | 0.02 | 1.25 | null |
# MAGIC | LATE_AIRCRAFT_DELAY | double | 31746839 | 0 | 0 | 0 | 2454 | 4.63 | 22.97 | null |
# MAGIC | STATION | double | 31746839 | 0 | 0 | 0 | 9.1765E+10 | 7.2797E+10 | 2917353496 | null |
# MAGIC | HourlyAltimeterSetting | double | 31746839 | 0 | 0 | 0 | 31.97 | 24.44 | 11.68 | null |
# MAGIC | HourlyDewPointTemperature | double | 31746839 | 0 | 0 | -40 | 93 | 46.22 | 20.05 | null |
# MAGIC | HourlyDryBulbTemperature | double | 31746839 | 0 | 0 | -48 | 126 | 62.13 | 21.42 | null |
# MAGIC | HourlyPrecipitation | double | 31746839 | 0 | 0 | 0 | 4.44 | 0 | 0.03 | null |
# MAGIC | HourlyPressureChange | double | 31746839 | 0 | 0 | -0.97 | 0.51 | 0 | 0.02 | null |
# MAGIC | HourlyPressureTendency | double | 31746839 | 0 | 0 | 0 | 9 | 1.45 | 2.69 | null |
# MAGIC | HourlyRelativeHumidity | double | 31746839 | 0 | 0 | 0 | 100 | 58.64 | 23.56 | null |
# MAGIC | HourlySkyConditions | double | 31746839 | 0 | 0 | 0 | 74 | 3.49 | 12.45 | null |
# MAGIC | HourlyWindGustSpeed | double | 31746839 | 0 | 0 | 0 | 99 | 2.86 | 8.27 | null |
# MAGIC | HourlyWindSpeed | double | 31746839 | 0 | 0 | 0 | 2237 | 8.86 | 6.37 | null |
# MAGIC | DailyAverageDewPointTemperature | double | 31746839 | 0 | 0 | -37 | 80 | 1.02 | 7.3 | null |
# MAGIC | crs_depart_ts_local | timestamp | 31746839 | 0 | 0 | null | null | null | null | null |
# MAGIC | crs_depart_ts_utc | timestamp | 31746839 | 0 | 0 | null | null | null | null | null |
# MAGIC | station_miles | double | 31746839 | 0 | 0 | 0 | 2.94168004 | 0.02 | 0.19 | null |
# MAGIC | tail_idx | int | 31746839 | 0 | 0 | 1 | 17837 | 765.89 | 635.63 | null |
# MAGIC | PREV_CRS_DEP_TIME | int | 31746839 | 0 | 0 | 0 | 2359 | 1328.7 | 491.11 | null |
# MAGIC | PREV_DEP_TIME | int | 31746839 | 0 | 0 | 0 | 2400 | 1312.99 | 526.09 | null |
# MAGIC | PREV_crs_depart_unixts_utc | bigint | 31746839 | 0 | 0 | 0 | 1577865480 | 1502584656 | 63237153.3 | null |
# MAGIC | PREV_DEP_DELAY | double | 31746839 | 0 | 0 | -234 | 2755 | 9.69 | 43.16 | null |
# MAGIC | PREV_DEP_DELAY_NEW | double | 31746839 | 0 | 0 | 0 | 2755 | 12.7 | 42.11 | null |
# MAGIC | PREV_DEP_DELAY_GROUP | int | 31746839 | 0 | 0 | -2 | 12 | 0.04 | 2.14 | null |
# MAGIC | PREV_ARR_DELAY_GROUP | int | 31746839 | 0 | 0 | -2 | 12 | -0.21 | 2.28 | null |
# MAGIC | PREV_CRS_ARR_TIME | int | 31746839 | 0 | 0 | 0 | 2400 | 1487.46 | 518.25 | null |
# MAGIC | PREV_ARR_TIME | int | 31746839 | 0 | 0 | 0 | 2400 | 1444.27 | 563.96 | null |
# MAGIC | PREV_ARR_DELAY | double | 31746839 | 0 | 0 | -238 | 2695 | 4.53 | 45.15 | null |
# MAGIC | PREV_CANCELLED | double | 31746839 | 0 | 0 | 0 | 1 | 0.02 | 0.12 | null |
# MAGIC | PREV_DIVERTED | double | 31746839 | 0 | 0 | 0 | 1 | 0 | 0.05 | null |
# MAGIC | PREV_OP_CARRIER_FL_NUM | double | 31746839 | 0 | 0 | 0 | 9855 | 2337.72 | 1792.67 | null |
# MAGIC | PREV_DISTANCE | double | 31746839 | 0 | 0 | 0 | 5095 | 822.32 | 607.7 | null |
# MAGIC | YEAR | int | 31746839 | 0 | 0 | 2015 | 2019 | 2017.15 | 1.43 | null |

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

with open('img/phase3/w261_proj_presentation_iii_data_pipeline.html', 'r', encoding='utf-8') as f:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix E: Custom join logic
# MAGIC
# MAGIC As written in the main section, the steps are: 
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
# MAGIC
# MAGIC The join steps are as follows

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