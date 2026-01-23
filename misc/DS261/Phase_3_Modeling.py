# Databricks notebook source
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col, lit, min, max
import datetime
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# COMMAND ----------

dbutils.fs.ls("dbfs:/student-groups/Group_01_01/")

# COMMAND ----------

dbutils.fs.ls("dbfs:/student-groups/Group_01_01/5Y")

# COMMAND ----------

df = spark.read.parquet("dbfs:/student-groups/Group_01_01/5Y/normalized.parquet")

# COMMAND ----------

df.head()

# COMMAND ----------

df.select(min("FL_DATE"), max("FL_DATE")).show()

# COMMAND ----------

# Create data splits, 2015-2018 is train/val with 75/25 split, everything after is test

start_train_year = 2015
end_train_year = 2018
test_year_start = 2019

df = df.filter(col("CANCELLED") < 0)

# Blind Test Set
test_df = df.filter(col("YEAR") >= test_year_start)

# Training/Validation Data (2015-2018)
train_val_pool = df.filter((col("YEAR") >= start_train_year) & (col("YEAR") <= end_train_year))

# 75% time cutoff strictly within the Train/Val pool
train_val_pool = train_val_pool.withColumn("FL_DATE_NUM", col("FL_DATE").cast("long"))
cutoff_list = train_val_pool.approxQuantile("FL_DATE_NUM", [0.75], 0.0)
split_cutoff = cutoff_list[0]

train_df = train_val_pool.filter(col("FL_DATE_NUM") <= split_cutoff)
val_df = train_val_pool.filter(col("FL_DATE_NUM") > split_cutoff)

print(f"Split Cutoff Timestamp: {split_cutoff} ({datetime.datetime.fromtimestamp(split_cutoff)})")

print("\n--- Row Counts ---")
print(f"Train Set (75% of {start_train_year}-{end_train_year}): {train_df.count()}")
print(f"Val Set (25% of {start_train_year}-{end_train_year}):   {val_df.count()}")
print(f"Test Set ({test_year_start}+):              {test_df.count()}")

print("\n--- Date Range Verification ---")
print("Train Max Date:", train_df.agg(max("FL_DATE")).collect()[0][0])
print("Val Min Date:  ", val_df.agg(min("FL_DATE")).collect()[0][0])
print("Test Min Date: ", test_df.agg(min("FL_DATE")).collect()[0][0])

# COMMAND ----------

# training set balancing, we are only using 75% of non-delayed flights in training set
delays_train = train_df.filter(col("DEP_DEL15") == 1)
nondelays_train = train_df.filter(col("DEP_DEL15") == 0)
nondelays_sampled = nondelays_train.sample(
    withReplacement=False,
    fraction=0.60,
    seed=42
)
train_balanced = delays_train.union(nondelays_sampled)
train_balanced_n = train_balanced.count()
print(f"train_balanced: {train_balanced_n}")

# COMMAND ----------

# custom functions, feature engineering/results interpretation

def get_feature_importance_with_names(pipeline_model, df, n=25):
    # Extract the Logistic Regression Model (The last stage)
    lr_model = pipeline_model.stages[-1]
    coeffs = np.array(lr_model.coefficients)
    
    assembler_stage_index = -2 
    assembler_stage = [s for s in pipeline_model.stages if isinstance(s, VectorAssembler)][0]
    
    # We need to run a tiny piece of data through the Indexers/Encoders/Assembler to generate the schema metadata
    stages_before_scaler = []
    for stage in pipeline_model.stages:
        stages_before_scaler.append(stage)
        if isinstance(stage, VectorAssembler):
            break
            
    meta_pipeline = Pipeline(stages=stages_before_scaler)
    meta_df = meta_pipeline.fit(df).transform(df.limit(10))
    output_col = assembler_stage.getOutputCol()
    meta = meta_df.schema[output_col].metadata
    
    feature_names = []
    
    if "ml_attr" in meta and "attrs" in meta["ml_attr"]:
        attr_dict = meta["ml_attr"]["attrs"]
        all_attrs = []
        if "numeric" in attr_dict: all_attrs.extend(attr_dict["numeric"])
        if "binary" in attr_dict:  all_attrs.extend(attr_dict["binary"])
        if "nominal" in attr_dict: all_attrs.extend(attr_dict["nominal"])
        all_attrs.sort(key=lambda x: x['idx'])
        feature_names = [x['name'] for x in all_attrs]
    else:
        feature_names = [f"Feature_{i}" for i in range(len(coeffs))]

    results = pd.DataFrame({
        "Feature_Name": feature_names,
        "Coefficient": coeffs,
        "Impact_Strength": np.abs(coeffs)
    })
    
    return results.sort_values(by="Impact_Strength", ascending=False).head(n)

# smoothed to lower variance of low occurance categories
def target_encode_smooth(df_train, df_val, df_test, cat_col, target_col="DEP_DEL15", m=20):
    # Global Mean
    global_mean = df_train.select(F.mean(target_col)).collect()[0][0]
    
    # Category Stats
    stats = df_train.groupby(cat_col).agg(
        F.count(target_col).alias("n"),
        F.sum(target_col).alias("sum_y")
    )

    # Smoothed Score
    stats = stats.withColumn(
        f"{cat_col}_encoded", 
        (F.col("sum_y") + (m * global_mean)) / (F.col("n") + m)
    ).select(cat_col, f"{cat_col}_encoded")
    
    def apply(df_in):
        return df_in.join(stats, on=cat_col, how="left").na.fill({f"{cat_col}_encoded": global_mean})
    return apply(df_train), apply(df_val), apply(df_test)
    
def engineer_features(df):
    # Fill NAs for T-2 safety
    df = df.na.fill({
        "PREV_DEP_DELAY": 0.0, "PREV_CRS_ARR_TIME": 0,
        "HourlyPrecipitation": 0.0, "HourlyWindSpeed": 0.0
    })

    # Parse Scheduled Time
    df = df.withColumn("sched_hour", F.floor(F.col("CRS_DEP_TIME") / 100))
    df = df.withColumn("sched_min", F.col("CRS_DEP_TIME") % 100)
    df = df.withColumn("sched_mins_midnight", (F.col("sched_hour") * 60) + F.col("sched_min"))

    # Parse Previous Arrival Time
    df = df.withColumn("prev_arr_hour", F.floor(F.col("PREV_CRS_ARR_TIME") / 100))
    df = df.withColumn("prev_arr_min", F.col("PREV_CRS_ARR_TIME") % 100)
    df = df.withColumn("prev_arr_mins_midnight", (F.col("prev_arr_hour") * 60) + F.col("prev_arr_min"))

    # Calculate Buffers
    df = df.withColumn("raw_buffer", F.col("sched_mins_midnight") - F.col("prev_arr_mins_midnight"))
    df = df.withColumn("SCHEDULED_BUFFER", 
                       F.when(F.col("raw_buffer") < 0, F.col("raw_buffer") + 1440)
                        .otherwise(F.col("raw_buffer")))
    
    # not currently using EFFECTIVE_BUFFER
    df = df.withColumn("EFFECTIVE_BUFFER", F.col("SCHEDULED_BUFFER") - F.col("PREV_DEP_DELAY")) #NOTE: IF THERES LEAKAGE, IT'S IN PREV_DEP_DELAY FOR CASE OF SHORT HOPS (check this)

    # Cyclical Time
    # encoding time as trig functions so that 23rd hour is closer to 1st hour (time loops around)
    df = df.withColumn("DEP_HOUR_SIN", F.sin(2 * 3.14159 * F.col("sched_hour") / 24))
    df = df.withColumn("DEP_HOUR_COS", F.cos(2 * 3.14159 * F.col("sched_hour") / 24))

    # Weather Interaction (currently just doing rain x windSpeed)
    df = df.withColumn("STORM_INDEX", F.col("HourlyPrecipitation") + F.col("HourlyWindSpeed")) #probably fine to use ourly precip? can amend to use the previous if we arent already
    
    # Add Class Weights (Fixes Underfitting/Low Recall)
    # weight Delays by 2.0x to force the model to care about them
    df = df.withColumn("classWeight", F.when(F.col("ARR_DEL15") == 1, 2.0).otherwise(1.0)) #classweight only utilized in training
    
    return df

def confusion_matrix(pred_df, label_col="DEP_DEL15", prediction_col="prediction"):
    tp = pred_df.filter((col(prediction_col) == 1) & (col(label_col) == 1)).count()
    tn = pred_df.filter((col(prediction_col) == 0) & (col(label_col) == 0)).count()
    fp = pred_df.filter((col(prediction_col) == 1) & (col(label_col) == 0)).count()
    fn = pred_df.filter((col(prediction_col) == 0) & (col(label_col) == 1)).count()

    print("Confusion Matrix:")
    print(f"""
                 Predicted
                 0        1
Actual   0     {tn:6d}   {fp:6d}
         1     {fn:6d}   {tp:6d}
    """)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def get_metrics(pred_df, label_col="ARR_DEL15", prediction_col="prediction"):
    # 1. Define Evaluators
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="accuracy"
    )
    
    # Class 0 (Not Delayed)
    evaluator_prec_0 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="precisionByLabel", metricLabel=0.0
    )
    evaluator_rec_0 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="recallByLabel", metricLabel=0.0
    )

    # Class 1 (Delayed)
    evaluator_prec_1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="precisionByLabel", metricLabel=1.0
    )
    evaluator_rec_1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="recallByLabel", metricLabel=1.0
    )

    # 2. Run Evaluations (Triggers Spark Jobs)
    # Tip: Ensure pred_df is cached (.cache()) before this, or it will recompute for every line below.
    acc = evaluator_acc.evaluate(pred_df)
    
    p0 = evaluator_prec_0.evaluate(pred_df)
    r0 = evaluator_rec_0.evaluate(pred_df)
    
    p1 = evaluator_prec_1.evaluate(pred_df)
    r1 = evaluator_rec_1.evaluate(pred_df)

    # 3. Calculate F1 Scores Manually (Harmonic Mean)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    def calc_f1(p, r):
        if (p + r) == 0: 
            return 0.0
        return 2 * (p * r) / (p + r)

    f1_0 = calc_f1(p0, r0)
    f1_1 = calc_f1(p1, r1)

    # 4. Calculate Macro F1 (Unweighted Average)
    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "accuracy": acc,
        "f1_macro": macro_f1,           # The Corrected Macro F1
        "f1_delayed": f1_1,             # Often useful to see specifically
        "precision_not_delayed": p0,
        "recall_not_delayed": r0,
        "precision_delayed": p1,
        "recall_delayed": r1,
    }

# COMMAND ----------

# Apply feature engineering
train_eng = engineer_features(train_balanced)
val_eng = engineer_features(val_df)
test_eng = engineer_features(test_df)

# Target Encoding
train_enc, val_enc, test_enc = target_encode_smooth(train_eng, val_eng, test_eng, "ORIGIN")
train_enc, val_enc, test_enc = target_encode_smooth(train_enc, val_enc, test_enc, "DEST")
train_enc, val_enc, test_enc = target_encode_smooth(train_enc, val_enc, test_enc, "PREV_ORIGIN")

# COMMAND ----------

encoded_cols = [
    "ORIGIN_encoded", 
    "DEST_encoded", 
    "PREV_ORIGIN_encoded"
]

# kept as categorical due to low cardinality
cat_cols_to_ohe = [
    "MONTH", 
    "QUARTER", 
    "DAY_OF_WEEK", 
    "sched_hour",
    "OP_UNIQUE_CARRIER",
]

numeric_cols = [
    "DISTANCE", "PREV_DISTANCE",
    "SCHEDULED_BUFFER",
    "DEP_HOUR_SIN", "DEP_HOUR_COS", "STORM_INDEX",#"HourlyWindGustSpeed",
    "HourlyAltimeterSetting", "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
    "HourlyRelativeHumidity", "HourlyWindSpeed" #"HourlyPressureTendency",
]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols_to_ohe]
encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols_to_ohe], 
                        outputCols=[f"{c}_ohe" for c in cat_cols_to_ohe], 
                        handleInvalid="keep")
assembler = VectorAssembler(
    inputCols=encoded_cols + numeric_cols + [f"{c}_ohe" for c in cat_cols_to_ohe],
    outputCol="unscaled_features",
    handleInvalid="skip"
)
# Scale (Crucial for Reg=0.001)
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)

# COMMAND ----------

# Logistic Regression (Weighted + Relaxed)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="ARR_DEL15",
    maxIter=50,
    regParam=0.0001,
    elasticNetParam=0.0,
    weightCol="classWeight" # using weight to prioritize recall of delay
)

pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

model = pipeline.fit(train_enc)
test_pred = model.transform(test_enc)

# COMMAND ----------

print("\n--- Top Features by Importance ---")
print(get_feature_importance_with_names(model, train_enc, n=20)[["Feature_Name", "Coefficient"]])

# COMMAND ----------

train_pred_lr = model.transform(train_enc)
val_pred_lr   = model.transform(val_enc)
test_pred  = model.transform(test_enc)

print("Calculated Metrics for Relaxed Logistic Regression:")

print("\n--- Training Set Results ---")
print(get_metrics(train_pred_lr))
confusion_matrix(train_pred_lr)

print("\n--- Validation Set Results ---")
print(get_metrics(val_pred_lr))
confusion_matrix(val_pred_lr)

print("\n--- Test Set Results (Final Holdout) ---")
print(get_metrics(test_pred))
confusion_matrix(test_pred)

# COMMAND ----------

from pyspark.ml import Pipeline

#  Define the Feature Engineering Pipeline (No Logistic Regression yet)
# Ensure indexers, encoder, assembler, scaler are defined
feature_pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler])

# Fit and Transform only once
print("Fitting Feature Engineering Pipeline...")
feat_model = feature_pipeline.fit(train_enc)

print("Transforming Data...")
train_vec = feat_model.transform(train_enc).select("features", "ARR_DEL15", "classWeight")
val_vec = feat_model.transform(val_enc).select("features", "ARR_DEL15", "classWeight")

train_vec_path = "/tmp/flight_delay_train_vec"
val_vec_path = "/tmp/flight_delay_val_vec"

# Write (overwrite if exists)
train_vec.write.mode("overwrite").parquet(train_vec_path)
val_vec.write.mode("overwrite").parquet(val_vec_path)

# Read back
train_ready = spark.read.parquet(train_vec_path)
val_ready = spark.read.parquet(val_vec_path)

# Now cache the clean, ready-to-go vectors
train_ready.cache()
val_ready.cache()

print(f"Materialized Train Count: {train_ready.count()}")
print(f"Materialized Val Count:   {val_ready.count()}")

# COMMAND ----------

# Select only required columns and repartition for parallelism
train_ready = train_ready.select(
    "features", "ARR_DEL15", "classWeight"
).repartition(200).cache()
val_ready = val_ready.select(
    "features", "ARR_DEL15", "classWeight"
).repartition(100).cache()

# Proceed with your grid search as before

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
import itertools

# Define your Grid
#reg_params = [0.001, 0.0001, 0.00001, 0]
reg_params = [0.0001, 0.00001]
elastic_params = [0.0, 0.25] # 0=L2, 1=L1, 0.5=Mix

# Prepare to store results
results = []

# The Loop
print(f"Starting Grid Search on {len(reg_params) * len(elastic_params)} combinations...")

for reg, elastic in itertools.product(reg_params, elastic_params):
    
    print(f"Training: Reg={reg}, Elastic={elastic}")
    
    # Define LR on the ALREADY VECTORIZED data
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="ARR_DEL15",
        weightCol="classWeight",
        maxIter=75,
        regParam=reg,
        elasticNetParam=elastic
    )
    
    # Fit (Fast, because features are pre-computed)
    model = lr.fit(train_ready)
    
    # Predict on Val
    preds = model.transform(val_ready)
    
    # --- Custom F1-Macro Calculation ---
    # (Using the lightweight aggregation method)
    counts = preds.groupBy("ARR_DEL15", "prediction").count().collect()
    mapping = {(r["ARR_DEL15"], r["prediction"]): r['count'] for r in counts}
    
    tp = mapping.get((1.0, 1.0), 0)
    fp = mapping.get((0.0, 1.0), 0)
    tn = mapping.get((0.0, 0.0), 0)
    fn = mapping.get((1.0, 0.0), 0)
    
    # F1 Class 1
    f1_1 = 0.0
    if (2*tp + fp + fn) > 0: f1_1 = (2*tp) / (2*tp + fp + fn)
        
    # F1 Class 0
    f1_0 = 0.0
    if (2*tn + fn + fp) > 0: f1_0 = (2*tn) / (2*tn + fn + fp)
        
    f1_macro = (f1_1 + f1_0) / 2.0
    
    print(f"  -> F1-Macro: {f1_macro:.4f}")
    
    results.append({
        'regParam': reg,
        'elasticNetParam': elastic,
        'score': f1_macro,
        'model': model 
    })

# 3. Find Best Result
best_result = __builtins__.max(results, key=lambda x: x['score'])
print("\n" + "="*30)
print("BEST HYPERPARAMETERS FOUND")
print(f"RegParam: {best_result['regParam']}")
print(f"ElasticNet: {best_result['elasticNetParam']}")
print(f"F1-Macro: {best_result['score']:.4f}")
print("="*30)

test_vec = feat_model.transform(test_enc) # Transform test using the pipeline we saved in Step 1
test_preds = best_result['model'].transform(test_vec)

# COMMAND ----------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="ARR_DEL15",
    maxIter=50,
    regParam=0.00001,
    elasticNetParam=0.0,
    weightCol="classWeight" # using weight to prioritize recall of delay
)
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

model = pipeline.fit(train_enc)
test_pred = model.transform(test_enc)

# COMMAND ----------

train_pred_lr = model.transform(train_enc)
val_pred_lr   = model.transform(val_enc)
test_pred  = model.transform(test_enc)

print("Calculated Metrics for Relaxed Logistic Regression:")

print("\n--- Training Set Results ---")
print(get_metrics(train_pred_lr))
confusion_matrix(train_pred_lr)

print("\n--- Validation Set Results ---")
print(get_metrics(val_pred_lr))
confusion_matrix(val_pred_lr)

print("\n--- Test Set Results (Final Holdout) ---")
print(get_metrics(test_pred))
confusion_matrix(test_pred)

# COMMAND ----------

df_pool = train_eng.unionByName(val_eng)


# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

assembler_tree = VectorAssembler(
    inputCols=encoded_cols + numeric_cols + [f"{c}_ohe" for c in cat_cols_to_ohe],
    outputCol="features", # Direct to features, no scaling needed
    handleInvalid="skip"
)

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="ARR_DEL15",
    maxIter=25,      
    maxDepth=6,       
    stepSize=0.1,
    weightCol="classWeight",
    seed=42
)

pipeline_gbt = Pipeline(stages=indexers + [encoder, assembler_tree, gbt])

print("Training Gradient Boosted Tree")
model_gbt = pipeline_gbt.fit(train_enc)


# COMMAND ----------

print("Generating GBT Predictions")
train_pred_gbt = model_gbt.transform(train_enc)
val_pred_gbt = model_gbt.transform(val_enc)

print("\n--- Training Set Results (Gradient Boosted Tree ---")
print(get_metrics(train_pred_gbt))
confusion_matrix(train_pred_gbt)

print("\n--- Validation Set Results (Gradient Boosted Tree ---")
print(get_metrics(val_pred_gbt))
confusion_matrix(val_pred_gbt)

# COMMAND ----------

from xgboost.spark import SparkXGBClassifier
import itertools

NUM_WORKERS = 4   

# 1. Define XGBoost Grid
#    - max_depth: How deep the trees go (Higher = more complex/overfitting)
#    - learning_rate: Step size (Lower = slower but more precise)
#    - n_estimators: Number of trees (Equivalent to maxIter)
depth_params = [3, 6, 8] 
lr_params = [0.1, 0.05, 0.01]
n_estimators_params = [50, 100] # Keep fixed or tune [50, 100]

results = []

print(f"Starting XGBoost Grid Search on {len(depth_params) * len(lr_params) * len(n_estimators_params)} combinations...")

for depth, lr, n_est in itertools.product(depth_params, lr_params, n_estimators_params):
    
    print(f"Training: Depth={depth}, LR={lr}, Trees={n_est} ...", end=" ")
    
    # 2. Instantiate XGBoost
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="ARR_DEL15",
        weight_col="classWeight",  # Note: XGBoost usually uses snake_case 'weight_col' or 'weightCol' depending on version
        max_depth=depth,
        learning_rate=lr,
        n_estimators=n_est,
        num_workers=NUM_WORKERS,   # Crucial for Spark XGBoost distribution
        missing=0.0                # Treat 0s as dense values or missing? Usually 0.0 for sparse vectors
    )
    
    # 3. Fit
    # XGBoost can be heavy. Ensure your executors have enough memory.
    try:
        model = xgb.fit(train_ready)
    except Exception as e:
        print(f"Failed to train: {e}")
        continue
        
    # 4. Predict on Val
    preds = model.transform(val_ready)
    
    # --- Custom F1-Macro Calculation ---
    counts = preds.groupBy("ARR_DEL15", "prediction").count().collect()
    mapping = {(r["ARR_DEL15"], r["prediction"]): r['count'] for r in counts}
    
    tp = mapping.get((1.0, 1.0), 0)
    fp = mapping.get((0.0, 1.0), 0)
    tn = mapping.get((0.0, 0.0), 0)
    fn = mapping.get((1.0, 0.0), 0)
    
    f1_1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    f1_0 = (2*tn) / (2*tn + fn + fp) if (2*tn + fn + fp) > 0 else 0.0
    f1_macro = (f1_1 + f1_0) / 2.0
    
    print(f"-> F1-Macro: {f1_macro:.4f}")
    
    results.append({
        'max_depth': depth,
        'learning_rate': lr,
        'n_estimators': n_est,
        'score': f1_macro,
        'model': model 
    })

# 5. Find Best Result
best_result = __builtins__.max(results, key=lambda x: x['score'])

print("="*30)
print("BEST HYPERPARAMETERS FOUND")
print(f"Depth: {best_result['max_depth']}")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Trees: {best_result['n_estimators']}")
print(f"F1-Macro: {best_result['score']:.4f}")
print("="*30)

# 6. Run on Blind Test Set
# Important: XGBoost models in Spark might act differently with VectorAssembler metadata
# Ensure test_vec is created exactly like train_ready
test_vec = feat_model.transform(test_enc) 
test_preds = best_result['model'].transform(test_vec)

# COMMAND ----------

from xgboost.spark import SparkXGBClassifier
import itertools

NUM_WORKERS = 4   

# 1. Define XGBoost Grid
#    - max_depth: How deep the trees go (Higher = more complex/overfitting)
#    - learning_rate: Step size (Lower = slower but more precise)
#    - n_estimators: Number of trees (Equivalent to maxIter)
depth_params = [12] 
lr_params = [0.1]
n_estimators_params = [150] # Keep fixed or tune [50, 100]

results = []

print(f"Starting XGBoost Grid Search on {len(depth_params) * len(lr_params) * len(n_estimators_params)} combinations...")

for depth, lr, n_est in itertools.product(depth_params, lr_params, n_estimators_params):
    
    print(f"Training: Depth={depth}, LR={lr}, Trees={n_est} ...", end=" ")
    
    # 2. Instantiate XGBoost
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="ARR_DEL15",
        weight_col="classWeight",  # Note: XGBoost usually uses snake_case 'weight_col' or 'weightCol' depending on version
        max_depth=depth,
        learning_rate=lr,
        n_estimators=n_est,
        num_workers=NUM_WORKERS,   # Crucial for Spark XGBoost distribution
        missing=0.0                # Treat 0s as dense values or missing? Usually 0.0 for sparse vectors
    )
    
    # 3. Fit
    # XGBoost can be heavy. Ensure your executors have enough memory.
    try:
        model = xgb.fit(train_ready)
    except Exception as e:
        print(f"Failed to train: {e}")
        continue
        
    # 4. Predict on Val
    preds = model.transform(val_ready)
    
    # --- Custom F1-Macro Calculation ---
    counts = preds.groupBy("ARR_DEL15", "prediction").count().collect()
    mapping = {(r["ARR_DEL15"], r["prediction"]): r['count'] for r in counts}
    
    tp = mapping.get((1.0, 1.0), 0)
    fp = mapping.get((0.0, 1.0), 0)
    tn = mapping.get((0.0, 0.0), 0)
    fn = mapping.get((1.0, 0.0), 0)
    
    f1_1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    f1_0 = (2*tn) / (2*tn + fn + fp) if (2*tn + fn + fp) > 0 else 0.0
    f1_macro = (f1_1 + f1_0) / 2.0
    
    print(f"-> F1-Macro: {f1_macro:.4f}")
    
    results.append({
        'max_depth': depth,
        'learning_rate': lr,
        'n_estimators': n_est,
        'score': f1_macro,
        'model': model 
    })

# 5. Find Best Result
best_result = __builtins__.max(results, key=lambda x: x['score'])

print("="*30)
print("BEST HYPERPARAMETERS FOUND")
print(f"Depth: {best_result['max_depth']}")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Trees: {best_result['n_estimators']}")
print(f"F1-Macro: {best_result['score']:.4f}")
print("="*30)

# 6. Run on Blind Test Set
# Important: XGBoost models in Spark might act differently with VectorAssembler metadata
# Ensure test_vec is created exactly like train_ready
test_vec = feat_model.transform(test_enc) 
test_preds = best_result['model'].transform(test_vec)

# COMMAND ----------

print("\n--- Test Set Results (Gradient Boosted Tree ---")
print(get_metrics(test_preds))
confusion_matrix(test_preds)

# COMMAND ----------

from xgboost.spark import SparkXGBClassifier
import itertools

NUM_WORKERS = 4   

# 1. Define XGBoost Grid
#    - max_depth: How deep the trees go (Higher = more complex/overfitting)
#    - learning_rate: Step size (Lower = slower but more precise)
#    - n_estimators: Number of trees (Equivalent to maxIter)
depth_params = [16] 
lr_params = [0.15]
n_estimators_params = [150] # Keep fixed or tune [50, 100]

results = []

print(f"Starting XGBoost Grid Search on {len(depth_params) * len(lr_params) * len(n_estimators_params)} combinations...")

for depth, lr, n_est in itertools.product(depth_params, lr_params, n_estimators_params):
    
    print(f"Training: Depth={depth}, LR={lr}, Trees={n_est} ...", end=" ")
    
    # 2. Instantiate XGBoost
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="ARR_DEL15",
        weight_col="classWeight",  # Note: XGBoost usually uses snake_case 'weight_col' or 'weightCol' depending on version
        max_depth=depth,
        learning_rate=lr,
        n_estimators=n_est,
        num_workers=NUM_WORKERS,   # Crucial for Spark XGBoost distribution
        missing=0.0                # Treat 0s as dense values or missing? Usually 0.0 for sparse vectors
    )
    
    # 3. Fit
    # XGBoost can be heavy. Ensure your executors have enough memory.
    try:
        model = xgb.fit(train_ready)
    except Exception as e:
        print(f"Failed to train: {e}")
        continue
        
    # 4. Predict on Val
    preds = model.transform(val_ready)
    
    # --- Custom F1-Macro Calculation ---
    counts = preds.groupBy("ARR_DEL15", "prediction").count().collect()
    mapping = {(r["ARR_DEL15"], r["prediction"]): r['count'] for r in counts}
    
    tp = mapping.get((1.0, 1.0), 0)
    fp = mapping.get((0.0, 1.0), 0)
    tn = mapping.get((0.0, 0.0), 0)
    fn = mapping.get((1.0, 0.0), 0)
    
    f1_1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    f1_0 = (2*tn) / (2*tn + fn + fp) if (2*tn + fn + fp) > 0 else 0.0
    f1_macro = (f1_1 + f1_0) / 2.0
    
    print(f"-> F1-Macro: {f1_macro:.4f}")
    
    results.append({
        'max_depth': depth,
        'learning_rate': lr,
        'n_estimators': n_est,
        'score': f1_macro,
        'model': model 
    })

# 5. Find Best Result
best_result = __builtins__.max(results, key=lambda x: x['score'])

print("="*30)
print("BEST HYPERPARAMETERS FOUND")
print(f"Depth: {best_result['max_depth']}")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Trees: {best_result['n_estimators']}")
print(f"F1-Macro: {best_result['score']:.4f}")
print("="*30)

# 6. Run on Blind Test Set
# Important: XGBoost models in Spark might act differently with VectorAssembler metadata
# Ensure test_vec is created exactly like train_ready
test_vec = feat_model.transform(test_enc) 
test_preds = best_result['model'].transform(test_vec)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import itertools
import time

# Try importing XGBoost
try:
    from xgboost.spark import SparkXGBClassifier
except ImportError:
    print("Warning: SparkXGBClassifier not found. XGBoost will fail if called.")

def target_encode_smooth(train_df, val_df, col, target_col="ARR_DEL15", smoothing=10):
    """
    Encodes categorical columns based on Train statistics only.
    """
    # 1. Global Mean (from Train)
    global_mean = train_df.select(F.mean(target_col)).collect()[0][0]
    
    # 2. Group Stats (from Train)
    agg = train_df.groupBy(col).agg(
        F.count(col).alias("count"),
        F.mean(target_col).alias("mean")
    )
    
    # 3. Smoothing
    smoother = F.expr(f"(count * mean + {smoothing} * {global_mean}) / (count + {smoothing})")
    agg = agg.withColumn(f"{col}_encoded", smoother).drop("count", "mean")
    
    # 4. Join
    train_enc = train_df.join(agg, on=col, how="left").fillna(global_mean, subset=[f"{col}_encoded"])
    val_enc = val_df.join(agg, on=col, how="left").fillna(global_mean, subset=[f"{col}_encoded"])
    
    return train_enc, val_enc

def get_metrics_efficient(pred_df, label_col="ARR_DEL15", pred_col="prediction"):
    """
    Calculates metrics in ONE pass. Returns 'total' to help debug empty folds.
    """
    # 1. Trigger ONE Spark Job to get counts
    counts = pred_df.groupBy(label_col, pred_col).count().collect()
    
    # 2. Map to Confusion Matrix
    mapping = {(r[label_col], r[pred_col]): r['count'] for r in counts}
    
    tp = mapping.get((1.0, 1.0), 0)
    fp = mapping.get((0.0, 1.0), 0)
    tn = mapping.get((0.0, 0.0), 0)
    fn = mapping.get((1.0, 0.0), 0)
    
    total = tp + fp + tn + fn
    
    def safe_div(n, d): return n / d if d > 0 else 0.0
    
    # Global Metrics
    accuracy = safe_div(tp + tn, total)
    
    # Class 1 (Delayed) Metrics
    prec_1 = safe_div(tp, tp + fp)
    rec_1  = safe_div(tp, tp + fn)
    f1_1   = safe_div(2 * prec_1 * rec_1, prec_1 + rec_1)
    
    # Class 0 (Not Delayed) Metrics
    prec_0 = safe_div(tn, tn + fn)
    rec_0  = safe_div(tn, tn + fp)
    f1_0   = safe_div(2 * prec_0 * rec_0, prec_0 + rec_0)
    
    # F1 Macro
    f1_macro = (f1_1 + f1_0) / 2.0
    
    return {
        "total_rows": total,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_delayed": f1_1,
        "f1_not_delayed": f1_0,
        "precision_delayed": prec_1,
        "recall_delayed": rec_1,
        "precision_not_delayed": prec_0,
        "recall_not_delayed": rec_0,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

# COMMAND ----------

# --------------------------------------------------------------------------------
# 3. MAIN TUNING FUNCTION (Corrected Prints)
# --------------------------------------------------------------------------------
def run_block_cv_tuning(df_pool, model_type, param_grid):
    
    # --- DEBUG: CHECK INPUT DATE RANGE ---
    # This helps explain why Fold 4 might be empty
    print("Checking Input Data Date Range...")
    min_date, max_date = df_pool.select(F.min("FL_DATE"), F.max("FL_DATE")).collect()[0]
    print(f"Input Data Range: {min_date} to {max_date}")

    # --- A. DEFINE SLIDING BLOCKS (2015-2018) ---
    folds = [
        {"id": 1, "train_start": "2015-01-01", "train_end": "2015-12-31", "val_start": "2016-01-01", "val_end": "2016-03-31"},
        {"id": 2, "train_start": "2016-01-01", "train_end": "2016-12-31", "val_start": "2017-01-01", "val_end": "2017-03-31"},
        {"id": 3, "train_start": "2017-01-01", "train_end": "2017-12-31", "val_start": "2018-01-01", "val_end": "2018-03-31"},
        {"id": 4, "train_start": "2018-01-01", "train_end": "2018-09-30", "val_start": "2018-10-01", "val_end": "2018-12-31"}
    ]
    
    # --- B. DEFINE FEATURES ---
    cat_cols_ohe = ["MONTH", "QUARTER", "DAY_OF_WEEK", "sched_hour", "OP_UNIQUE_CARRIER"]
    numeric_cols = [
        "DISTANCE", "PREV_DISTANCE", "SCHEDULED_BUFFER", 
        "DEP_HOUR_SIN", "DEP_HOUR_COS", "STORM_INDEX", 
        "HourlyWindSpeed", "HourlyPrecipitation", "HourlyDryBulbTemperature"
    ]
    
    # --- C. SETUP GRID ---
    if model_type == "baseline":
        param_combinations = [{}] 
    else:
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
    print(f"\n>>> Starting {model_type.upper()} Tuning: {len(param_combinations)} Configs x {len(folds)} Folds")

    best_avg_f1 = -1.0
    best_params = None
    
    results_detailed = [] 
    results_summary = []  

    # --- D. OUTER LOOP: PARAMETERS ---
    for params in param_combinations:
        
        print(f"\nTesting Params: {params}")
        current_param_metrics = []
        
        # --- E. INNER LOOP: FOLDS ---
        for fold in folds:
            # 1. Slice Time
            train_sub = df_pool.filter((F.col("FL_DATE") >= fold['train_start']) & (F.col("FL_DATE") <= fold['train_end']))
            val_sub = df_pool.filter((F.col("FL_DATE") >= fold['val_start']) & (F.col("FL_DATE") <= fold['val_end']))
            
            # 2. Target Encode
            train_enc, val_enc = target_encode_smooth(train_sub, val_sub, "ORIGIN")
            train_enc, val_enc = target_encode_smooth(train_enc, val_enc, "DEST")
            train_enc, val_enc = target_encode_smooth(train_enc, val_enc, "PREV_ORIGIN")
            
            # 3. Build Pipeline
            stages = []
            indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols_ohe]
            encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cat_cols_ohe]
            stages += indexers + encoders
            
            te_feats = ["ORIGIN_encoded", "DEST_encoded", "PREV_ORIGIN_encoded"]
            ohe_feats = [f"{c}_ohe" for c in cat_cols_ohe]
            assembler = VectorAssembler(
                inputCols=numeric_cols + te_feats + ohe_feats, 
                outputCol="unscaled_features", handleInvalid="skip"
            )
            stages.append(assembler)
            
            # 4. Model Logic
            if model_type == "baseline":
                counts = train_enc.groupBy("ARR_DEL15").count().collect()
                maj_class = sorted(counts, key=lambda x: x['count'], reverse=True)[0]['ARR_DEL15']
                preds = val_enc.withColumn("prediction", F.lit(maj_class))
                
            elif model_type == "lr":
                scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)
                lr = LogisticRegression(featuresCol="features", labelCol="ARR_DEL15", weightCol="classWeight", 
                                        maxIter=20, **params)
                stages += [scaler, lr]
                pipeline = Pipeline(stages=stages)
                model = pipeline.fit(train_enc)
                preds = model.transform(val_enc)
                
            elif model_type == "xgb":
                assembler.setOutputCol("features")
                xgb = SparkXGBClassifier(
                    features_col="features", label_col="ARR_DEL15", weight_col="classWeight",
                    missing=0.0, num_workers=12, **params
                )
                stages.append(xgb)
                pipeline = Pipeline(stages=stages)
                model = pipeline.fit(train_enc)
                preds = model.transform(val_enc)

            # 5. Calculate Metrics
            metrics = get_metrics_efficient(preds)
            
            # --- 6. PRINT DETAILED METRICS PER FOLD ---
            print(f"   [Fold {fold['id']}] Val Count: {metrics['total_rows']} | F1-Macro: {metrics['f1_macro']:.3f} | Acc: {metrics['accuracy']:.3f}")
            if metrics['total_rows'] > 0:
                print(f"      > Class 1 (Late): F1={metrics['f1_delayed']:.3f} | Prec={metrics['precision_delayed']:.3f} | Rec={metrics['recall_delayed']:.3f}")
                print(f"      > Class 0 (OK)  : F1={metrics['f1_not_delayed']:.3f} | Prec={metrics['precision_not_delayed']:.3f} | Rec={metrics['recall_not_delayed']:.3f}")
            else:
                print("      > WARNING: Fold Empty (0 rows). Check Data Dates!")

            # Log Results
            metrics['Fold'] = fold['id']
            metrics['Params'] = str(params)
            metrics['Model'] = model_type
            results_detailed.append(metrics)
            
            current_param_metrics.append(metrics)
            
        # --- END FOLD LOOP ---
        
        # --- F. CALCULATE & PRINT AVERAGES ---
        metric_keys = [
            "accuracy", "f1_macro", 
            "f1_delayed", "precision_delayed", "recall_delayed",
            "f1_not_delayed", "precision_not_delayed", "recall_not_delayed"
        ]
        
        avg_metrics = {}
        for k in metric_keys:
            # If a fold was empty, it might drag avg down, but that's expected behavior for valid failures
            avg_metrics[k] = sum(m[k] for m in current_param_metrics) / len(current_param_metrics)
        
        print(f"   >>> AVERAGES (Across {len(folds)} Folds):")
        print(f"       Global  -> F1-Macro: {avg_metrics['f1_macro']:.4f} | Acc: {avg_metrics['accuracy']:.4f}")
        print(f"       Class 1 -> Prec: {avg_metrics['precision_delayed']:.4f} | Rec: {avg_metrics['recall_delayed']:.4f} | F1: {avg_metrics['f1_delayed']:.4f}")
        print(f"       Class 0 -> Prec: {avg_metrics['precision_not_delayed']:.4f} | Rec: {avg_metrics['recall_not_delayed']:.4f} | F1: {avg_metrics['f1_not_delayed']:.4f}") # <--- ADDED F1
        
        # Store Summary
        avg_metrics['Params'] = str(params)
        avg_metrics['Model'] = model_type
        results_summary.append(avg_metrics)

        # Track Best Winner
        if avg_metrics['f1_macro'] > best_avg_f1:
            best_avg_f1 = avg_metrics['f1_macro']
            best_params = params

    print(f"\nWINNER ({model_type}): {best_params} with F1-Macro: {best_avg_f1:.4f}")
    
    return best_params, pd.DataFrame(results_detailed), pd.DataFrame(results_summary)

# COMMAND ----------

from pyspark.storagelevel import StorageLevel

# 1. Merge the Feature-Engineered Train and Val sets
# Use unionByName to ensure columns align correctly
df_pool = train_eng.unionByName(val_eng)
df_pool
# 2. Verify the Date Range
# This ensures you now have data covering 2015 all the way through the end of 2018
print("Checking merged date range...")
df_pool.select(F.min("FL_DATE"), F.max("FL_DATE")).show()

# 3. Repartition and Cache
# Since we just combined two large datasets, we should re-cache the result 
# so the CV loop doesn't re-compute the union every time.
# We use MEMORY_AND_DISK to prevent OOM errors if the combined size is large.
df_pool = df_pool.repartition(200).persist(StorageLevel.MEMORY_AND_DISK)

print(f"Total Rows in CV Pool: {df_pool.count()}")

# COMMAND ----------

# ==============================================================================
# 1. BASELINE (Majority Class)
# ==============================================================================
print("\n" + "="*50 + "\n>>> 1. RUNNING BASELINE (Majority Class)\n" + "="*50)

best_base, df_det_base, df_sum_base = run_block_cv_tuning(df_pool, "baseline", {})

print("\n--- Baseline Summary (Averages) ---")
print(df_sum_base.to_markdown(index=False, floatfmt=".4f"))

# COMMAND ----------

# ==============================================================================
# 2. LOGISTIC REGRESSION
# ==============================================================================
print("\n" + "="*50 + "\n>>> 2. RUNNING LOGISTIC REGRESSION TUNING\n" + "="*50)

# Define LR Grid
lr_grid = {
    'regParam': [0.1, 0.01, 0.001],      # Regularization Strength
    'elasticNetParam': [0.0, 0.5, 1.0]   # 0=Ridge (L2), 1=Lasso (L1)
}

best_lr, df_det_lr, df_sum_lr = run_block_cv_tuning(df_pool, "lr", lr_grid)

print("\n--- Logistic Regression Summary (Sorted by F1-Macro) ---")
print(df_sum_lr.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))

# COMMAND ----------

# ==============================================================================
# 2.1 LOGISTIC REGRESSION
# ==============================================================================
print("\n" + "="*50 + "\n>>> 2. RUNNING LOGISTIC REGRESSION TUNING\n" + "="*50)

# Define LR Grid
lr_grid_2 = {
    'regParam': [0.0001, 0],      # Regularization Strength
    'elasticNetParam': [0.0, 0.5, 1.0]   # 0=Ridge (L2), 1=Lasso (L1)
}

best_lr_2, df_det_lr_2, df_sum_lr_2 = run_block_cv_tuning(df_pool, "lr", lr_grid_2)

print("\n--- Logistic Regression Summary (Sorted by F1-Macro) ---")
print(df_sum_lr_2.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))

# COMMAND ----------

try:
    from xgboost.spark import SparkXGBClassifier
    print("\n" + "="*50 + "\n>>> 3. RUNNING XGBOOST TUNING\n" + "="*50)

    # Define XGB Grid
    # Note: Keep n_estimators lower (e.g., 100) for tuning speed, raise for final training
    xgb_grid = {
        'max_depth': [6, 10],          # Depth of trees (6 is standard, 10 is deep)
        'learning_rate': [0.1, 0.05],  # Step size
        'n_estimators': [100]          # Number of trees
    }

    best_xgb, df_det_xgb, df_sum_xgb = run_block_cv_tuning(df_pool, "xgb", xgb_grid)

    print("\n--- XGBoost Summary (Sorted by F1-Macro) ---")
    print(df_sum_xgb.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))

except ImportError:
    print("\nSkipping XGBoost (Library not found).")

# COMMAND ----------

try:
    from xgboost.spark import SparkXGBClassifier
    print("\n" + "="*50 + "\n>>> 3.1 RUNNING XGBOOST TUNING\n" + "="*50)

    # Define XGB Grid
    # Note: Keep n_estimators lower (e.g., 100) for tuning speed, raise for final training
    xgb_grid_2 = {
        'max_depth': [6, 10, 14],          # Depth of trees (6 is standard, 10 is deep)
        'learning_rate': [0.25, 0.1, 0.05],  # Step size
        'n_estimators': [100, 150, 200, 250]          # Number of trees
    }

    best_xgb_2, df_det_xgb_2, df_sum_xgb_2 = run_block_cv_tuning(df_pool, "xgb", xgb_grid_2)

    print("\n--- XGBoost Summary (Sorted by F1-Macro) ---")
    print(df_sum_xgb.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))

except ImportError:
    print("\nSkipping XGBoost (Library not found).")

# COMMAND ----------

try:
    from xgboost.spark import SparkXGBClassifier
    print("\n" + "="*50 + "\n>>> 3.2 RUNNING XGBOOST TUNING\n" + "="*50)

    # Define XGB Grid
    # Note: Keep n_estimators lower (e.g., 100) for tuning speed, raise for final training
    xgb_grid_3 = {
        'max_depth': [10, 14],          # Depth of trees (6 is standard, 10 is deep)
        'learning_rate': [0.25, 0.1, 0.05],  # Step size
        'n_estimators': [100, 150, 200, 250]          # Number of trees
    }

    best_xgb_3, df_det_xgb_3, df_sum_xgb_3 = run_block_cv_tuning(df_pool, "xgb", xgb_grid_3)

    print("\n--- XGBoost Summary (Sorted by F1-Macro) ---")
    print(df_sum_xgb.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))

except ImportError:
    print("\nSkipping XGBoost (Library not found).")

# COMMAND ----------

try:
    from xgboost.spark import SparkXGBClassifier
    print("\n" + "="*50 + "\n>>> 3.3 RUNNING XGBOOST TUNING\n" + "="*50)

    # Define XGB Grid
    # Note: Keep n_estimators lower (e.g., 100) for tuning speed, raise for final training
    xgb_grid_4 = {
        'max_depth': [14],          # Depth of trees (6 is standard, 10 is deep)
        'learning_rate': [0.25, 0.1, 0.05],  # Step size
        'n_estimators': [100, 150, 200, 250]          # Number of trees
    }

    best_xgb_4, df_det_xgb_4, df_sum_xgb_4 = run_block_cv_tuning(df_pool, "xgb", xgb_grid_4)

    print("\n--- XGBoost Summary (Sorted by F1-Macro) ---")
    print(df_sum_xgb.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))

except ImportError:
    print("\nSkipping XGBoost (Library not found).")

# COMMAND ----------

try:
    from xgboost.spark import SparkXGBClassifier
    print("\n" + "="*50 + "\n>>> 3.3 RUNNING XGBOOST TUNING\n" + "="*50)

    # Define XGB Grid
    # Note: Keep n_estimators lower (e.g., 100) for tuning speed, raise for final training
    xgb_grid_4 = {
        'max_depth': [14],          # Depth of trees (6 is standard, 10 is deep)
        'learning_rate': [0.25, 0.1, 0.05],  # Step size
        'n_estimators': [100, 150, 200, 250]          # Number of trees
    }

    best_xgb_4, df_det_xgb_4, df_sum_xgb_4 = run_block_cv_tuning(df_pool, "xgb", xgb_grid_4)

    print("\n--- XGBoost Summary (Sorted by F1-Macro) ---")
    print(df_sum_xgb.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))

except ImportError:
    print("\nSkipping XGBoost (Library not found).")

# COMMAND ----------

# FINAL Logistic Regression (Weighted + Relaxed)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="ARR_DEL15",
    maxIter=50,
    regParam=0.0001,
    elasticNetParam=0.0,
    weightCol="classWeight"
)

pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

model = pipeline.fit(df_pool)
test_pred = model.transform(test_eng)

# COMMAND ----------

# 4. Predict on Blind Test Set
print("Generating Predictions on Val Set...")
val_pred = model.transform(val_enc)
print("\n--- Test Set Results (Logistic Regression ---")
print(get_metrics(val_pred))
confusion_matrix(val_pred)
print("\n--- Test Set Results (Logistic Regression ---")
print(get_metrics(test_pred))
confusion_matrix(test_pred)

# COMMAND ----------

from xgboost.spark import SparkXGBClassifier
import itertools

NUM_WORKERS = 4   

# 1. Define XGBoost Grid
#    - max_depth: How deep the trees go (Higher = more complex/overfitting)
#    - learning_rate: Step size (Lower = slower but more precise)
#    - n_estimators: Number of trees (Equivalent to maxIter)
depth_params = [6] 
lr_params = [0.25]
n_estimators_params = [250] # Keep fixed or tune [50, 100]

results = []

print(f"Starting XGBoost Grid Search on {len(depth_params) * len(lr_params) * len(n_estimators_params)} combinations...")

for depth, lr, n_est in itertools.product(depth_params, lr_params, n_estimators_params):
    
    print(f"Training: Depth={depth}, LR={lr}, Trees={n_est} ...", end=" ")
    
    # 2. Instantiate XGBoost
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="ARR_DEL15",
        weight_col="classWeight",  # Note: XGBoost usually uses snake_case 'weight_col' or 'weightCol' depending on version
        max_depth=depth,
        learning_rate=lr,
        n_estimators=n_est,
        num_workers=NUM_WORKERS,   # Crucial for Spark XGBoost distribution
        missing=0.0                # Treat 0s as dense values or missing? Usually 0.0 for sparse vectors
    )
    
    # 3. Fit
    # XGBoost can be heavy. Ensure your executors have enough memory.
    try:
        model = xgb.fit(train_ready)
    except Exception as e:
        print(f"Failed to train: {e}")
        continue
        
    # 4. Predict on Val
    preds = model.transform(val_ready)
    
    # --- Custom F1-Macro Calculation ---
    counts = preds.groupBy("ARR_DEL15", "prediction").count().collect()
    mapping = {(r["ARR_DEL15"], r["prediction"]): r['count'] for r in counts}
    
    tp = mapping.get((1.0, 1.0), 0)
    fp = mapping.get((0.0, 1.0), 0)
    tn = mapping.get((0.0, 0.0), 0)
    fn = mapping.get((1.0, 0.0), 0)
    
    f1_1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    f1_0 = (2*tn) / (2*tn + fn + fp) if (2*tn + fn + fp) > 0 else 0.0
    f1_macro = (f1_1 + f1_0) / 2.0
    
    print(f"-> F1-Macro: {f1_macro:.4f}")
    
    results.append({
        'max_depth': depth,
        'learning_rate': lr,
        'n_estimators': n_est,
        'score': f1_macro,
        'model': model 
    })

# 5. Find Best Result
best_result = __builtins__.max(results, key=lambda x: x['score'])

print("="*30)
print("BEST HYPERPARAMETERS FOUND")
print(f"Depth: {results['max_depth']}")
print(f"Learning Rate: {results['learning_rate']}")
print(f"Trees: {results['n_estimators']}")
print(f"F1-Macro: {results['score']:.4f}")
print("="*30)

# 6. Run on Blind Test Set
# Important: XGBoost models in Spark might act differently with VectorAssembler metadata
# Ensure test_vec is created exactly like train_ready
test_vec = feat_model.transform(test_enc) 
test_preds = results['model'].transform(test_vec)

# COMMAND ----------

from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline

# --- Configuration ---
# IMPORTANT: Set this to the number of worker nodes in your cluster.
# If running on Databricks Community Edition, set this to 1.
NUM_WORKERS =  6

# 1. Instantiate the XGBoost Classifier with your specific parameters
xgb = SparkXGBClassifier(
    features_col="features",
    label_col="ARR_DEL15",
    weight_col="classWeight",
    max_depth=6,             # From your image
    learning_rate=0.25,      # From your image
    n_estimators=250,        # From your image
    num_workers=NUM_WORKERS, # Required for Spark distribution
    missing=0.0              # Treat missing/zeros correctly
)

# 2. Build the Pipeline
# We simply replace the 'lr' stage with our new 'xgb' stage
# Ensure indexers, encoder, assembler, scaler are defined from your previous cells
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, xgb])


# COMMAND ----------


# 3. Fit on Training Data
print("Training XGBoost Model...")
model = pipeline.fit(df_pool)

# 4. Predict on Blind Test Set
print("Generating Predictions on Test Set...")
test_pred = model.transform(test_enc)


# COMMAND ----------

print("\n--- Test Set Results (Gradient Boosted Tree ---")
print(get_metrics(test_pred))
confusion_matrix(test_pred)

# COMMAND ----------

# 4. Predict on Blind Test Set
print("Generating Predictions on Val Set...")
val_pred = model.transform(val_enc)
print("\n--- Test Set Results (Gradient Boosted Tree ---")
print(get_metrics(val_pred))
confusion_matrix(val_pred)

# COMMAND ----------

from pyspark.storagelevel import StorageLevel

# 1. Merge the Feature-Engineered Train and Val sets
# Use unionByName to ensure columns align correctly
df_pool = train_enc.unionByName(val_enc)

# 2. Verify the Date Range
# This ensures you now have data covering 2015 all the way through the end of 2018
print("Checking merged date range...")
df_pool.select(F.min("FL_DATE"), F.max("FL_DATE")).show()

# 3. Repartition and Cache
# Since we just combined two large datasets, we should re-cache the result 
# so the CV loop doesn't re-compute the union every time.
# We use MEMORY_AND_DISK to prevent OOM errors if the combined size is large.
df_pool = df_pool.repartition(200).persist(StorageLevel.MEMORY_AND_DISK)

print(f"Total Rows in CV Pool: {df_pool.count()}")

# COMMAND ----------

# FINAL Logistic Regression (Weighted + Relaxed)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="ARR_DEL15",
    maxIter=50,
    regParam=0.0001,
    elasticNetParam=0.0,
    weightCol="classWeight"
)

pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

model = pipeline.fit(df_pool)
test_pred = model.transform(test_enc)

# COMMAND ----------

print("\n--- Test Set Results (Logistic Regression ---")
print(get_metrics(test_pred))
confusion_matrix(test_pred)

# COMMAND ----------

# ==============================================================================
# 2. LOGISTIC REGRESSION
# ==============================================================================
print("\n" + "="*50 + "\n>>> 2. RUNNING LOGISTIC REGRESSION TUNING\n" + "="*50)

# Define LR Grid
lr_grid = {
    'regParam': [0.01, 0.001, 0.0001],      # Regularization Strength
    'elasticNetParam': [0.0, 0.25, 0.5]   # 0=Ridge (L2), 1=Lasso (L1)
}

best_lr, df_det_lr, df_sum_lr = run_block_cv_tuning(df_pool, "lr", lr_grid)

print("\n--- Logistic Regression Summary (Sorted by F1-Macro) ---")
print(df_sum_lr.sort_values("f1_macro", ascending=False).to_markdown(index=False, floatfmt=".4f"))# ============