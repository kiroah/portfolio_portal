# Databricks notebook source
dbutils.fs.ls("dbfs:/student-groups/Group_01_01/")

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/student-groups/Group_01_01/1Y/"))

# COMMAND ----------

df = spark.read.parquet("dbfs:/student-groups/Group_01_01/1Y/normalized.parquet")


# COMMAND ----------

df.head(1)

# COMMAND ----------

df.groupBy("CANCELLED").count().orderBy("count", ascending=False).show(50)

# COMMAND ----------

from pyspark.sql.functions import min, max

df.select(min("FL_DATE"), max("FL_DATE")).show()

# COMMAND ----------

# This is timeseries data so we need to make sure to split train/val/test date-wise, all train must come before all val/test
# I intend to additionally make all val come before all test, cross val can come later- this is initial baseline
from pyspark.sql.functions import col

# date cutoff calculations
df = df.withColumn("FL_DATE_NUM", col("FL_DATE").cast("long"))
cutoffs = df.approxQuantile("FL_DATE_NUM", [0.75, 1.00], 0.0)
train_cutoff, val_cutoff = cutoffs

# COMMAND ----------

print(train_cutoff)
print(val_cutoff)

# COMMAND ----------

#filter out cancelled flights
df = df.filter(df.CANCELLED < 0)

# COMMAND ----------

from pyspark.sql.functions import col

train_df = df.filter(col("FL_DATE").cast("long") <= train_cutoff)
val_df   = df.filter((col("FL_DATE").cast("long") > train_cutoff) & (col("FL_DATE").cast("long") <= val_cutoff))
#test_df  = df.filter(col("FL_DATE").cast("long") > val_cutoff)

# COMMAND ----------

total = df.count()
train_n = train_df.count()
val_n   = val_df.count()
#test_n  = test_df.count()

print(f"Train: {train_n} ({train_n/total:.3%})")
print(f"Val:   {val_n} ({val_n/total:.3%})")
#print(f"Test:  {test_n} ({test_n/total:.3%})")

# COMMAND ----------

#balance training set
delays_train = train_df.filter(col("DEP_DEL15") == 1)
nondelays_train = train_df.filter(col("DEP_DEL15") == 0)
nondelays_sampled = nondelays_train.sample(
    withReplacement=False,
    fraction=0.75, ###now that we are using weight, testing without rebalancing
    seed=42
)
train_balanced = delays_train.union(nondelays_sampled)
train_balanced_n = train_balanced.count()
print(f"train_balanced: {train_balanced_n}")

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler
)
# starting example features: TODO auto feature addition code, determine max features: 20?

numeric_cols = [
    "DISTANCE",
    "HourlyAltimeterSetting", "HourlyDewPointTemperature",
    "HourlyDryBulbTemperature", "HourlyPrecipitation",
    "HourlyPressureTendency",
    "HourlyRelativeHumidity", "HourlyWindGustSpeed",
    "HourlyWindSpeed", "DailyAverageDewPointTemperature", "PREV_DISTANCE",
]
cat_cols = ["ORIGIN", "PREV_ORIGIN", "DAY_OF_WEEK", "MONTH", "QUARTER", "PREV_DEP_DELAY_GROUP"]

# Indexers for each categorical column
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in cat_cols
]

# One-hot encoders
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_ohe" for c in cat_cols],
    handleInvalid="keep"
)

feature_cols = numeric_cols + [f"{c}_ohe" for c in cat_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="ARR_DEL15",
    maxIter=50
)
pipeline = Pipeline(stages=indexers + [encoder, assembler, lr])

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

feature_cols

# COMMAND ----------

logic_regression_model = pipeline.fit(train_balanced)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="ARR_DEL15")
train_pred = logic_regression_model.transform(train_balanced)
train_auc = evaluator.evaluate(train_pred)

print(f"Train AUC: {train_auc:.3f}")

# COMMAND ----------

val_pred = logic_regression_model.transform(val_df)
test_pred = logic_regression_model.transform(test_df)

val_auc = evaluator.evaluate(val_pred)
test_auc = evaluator.evaluate(test_pred)

print(f"Validation AUC: {val_auc:.3f}")
print(f"Test AUC: {test_auc:.3f}")

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def get_metrics(pred_df, label_col="ARR_DEL15", prediction_col="prediction"):
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="f1",
    )
    
    # metrics for NOT delayed (label 0)
    evaluator_prec_0 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="precisionByLabel",
        metricLabel=0.0,
    )
    evaluator_rec_0 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="recallByLabel",
        metricLabel=0.0,
    )

    # metrics for delayed (label 1)
    evaluator_prec_1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="precisionByLabel",
        metricLabel=1.0,
    )
    evaluator_rec_1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="recallByLabel",
        metricLabel=1.0,
    )

    return {
        "accuracy": evaluator_acc.evaluate(pred_df),
        "f1_macro": evaluator_f1.evaluate(pred_df),  # macro-averaged
        "precision_not_delayed": evaluator_prec_0.evaluate(pred_df),
        "recall_not_delayed": evaluator_rec_0.evaluate(pred_df),
        "precision_delayed": evaluator_prec_1.evaluate(pred_df),
        "recall_delayed": evaluator_rec_1.evaluate(pred_df),
    }

# COMMAND ----------

lr_train_metrics = get_metrics(train_pred)
lr_val_metrics   = get_metrics(val_pred)
lr_test_metrics  = get_metrics(test_pred)

print("Training Metrics:", lr_train_metrics)
print("Validation Metrics:", lr_val_metrics)
print("Test Metrics:", lr_test_metrics)

# COMMAND ----------

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

# COMMAND ----------

print("----Train Matrix----")
train_cm = confusion_matrix(train_pred)
print("----Validation Matrix----")
val_cm = confusion_matrix(val_pred)
print("----Test Matrix----")
test_cm = confusion_matrix(test_pred)

# COMMAND ----------

from pyspark.sql import functions as F
import math

label_col = "DEP_DEL15"

# counts per class
counts = (
    train_balanced
    .groupBy(label_col)
    .count()
    .collect()
)

count_dict = {row[label_col]: row["count"] for row in counts}
n0 = count_dict.get(0, 1)   # not delayed
n1 = count_dict.get(1, 1)   # delayed

# how much rarer delays are
ratio = n0 / n1
soft_ratio = math.sqrt(ratio)
print("class 0:", n0, "class 1:", n1, "soft_ratio:", soft_ratio)

# add weight column
train_weighted = train_balanced.withColumn(
    "class_weight",
    F.when(F.col(label_col) == 1, soft_ratio).otherwise(1.0)
)

# COMMAND ----------

#Random forest
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="DEP_DEL15",
    weightCol="class_weight",  
    numTrees=200,
    maxBins=256,
    maxDepth=12,
    impurity="gini",
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
    seed=42
)

# COMMAND ----------

pipeline_rf = Pipeline(stages= indexers + [encoder, assembler, rf])
rf_model = pipeline_rf.fit(train_weighted)

# COMMAND ----------

train_pred_rf = rf_model.transform(train_weighted)
val_pred_rf   = rf_model.transform(val_df)
test_pred_rf  = rf_model.transform(test_df)

# COMMAND ----------

rf_train_metrics = get_metrics(train_pred_rf)
rf_val_metrics   = get_metrics(val_pred_rf)
rf_test_metrics  = get_metrics(test_pred_rf)

print("Random Forest — Training:", rf_train_metrics)
print("Random Forest — Validation:", rf_val_metrics)
print("Random Forest — Test:", rf_test_metrics)

# COMMAND ----------

print("----Train Matrix----")
train_rf_cm = confusion_matrix(train_pred_rf)
print("----Validation Matrix----")
val_rf_cm = confusion_matrix(val_pred_rf)
print("----Test Matrix----")
test_rf_cm = confusion_matrix(test_pred_rf)

# COMMAND ----------

from pyspark.sql import functions as F

label_col = "DEP_DEL15"

pos = train_balanced.filter(F.col(label_col) == 1).count()
neg = train_balanced.filter(F.col(label_col) == 0).count()

# if you have the FULL unbalanced train df around, better to use that here
scale_pos_weight = neg / pos if pos > 0 else 1.0
print("scale_pos_weight:", scale_pos_weight)


# COMMAND ----------

from xgboost.spark import SparkXGBClassifier

xgb = SparkXGBClassifier(
    features_col="features",
    label_col=label_col,
    prediction_col="prediction",
    probability_col="probability",
    raw_prediction_col="rawPrediction",
    
    eval_metric="aucpr",     # good for imbalance

    # tree/boosting params (reasonably "risky" but not insane)
    num_round=70,           # boosting iterations (like n_estimators)
    max_depth=6,
    eta=0.1,                 # learning rate
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,

    scale_pos_weight=scale_pos_weight,

    reg_lambda=1.0,
    reg_alpha=0.0
)


# COMMAND ----------

from pyspark.ml import Pipeline

pipeline_xgb = Pipeline(stages=indexers + [encoder, assembler, xgb])
xgb_model = pipeline_xgb.fit(train_balanced)

# COMMAND ----------

train_pred_xgb = rf_model.transform(train_balanced)
val_pred_xgb   = rf_model.transform(val_df)
test_pred_xgb  = rf_model.transform(test_df)

# COMMAND ----------

from pyspark.ml import PipelineModel
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
#def target_encode_smooth(df_train, df_val, df_test, cat_col, target_col="DEP_DEL15", m=20):
def target_encode_smooth(df_train, df_val, cat_col, target_col="DEP_DEL15", m=20):
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
    #return apply(df_train), apply(df_val), apply(df_test)
    return apply(df_train), apply(df_val)
    
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
    df = df.withColumn("STORM_INDEX", F.col("HourlyPrecipitation") * F.col("HourlyWindSpeed")) #probably fine to use ourly precip? can amend to use the previous if we arent already
    
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
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="f1",
    )
    
    # metrics for NOT delayed (label 0)
    evaluator_prec_0 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="precisionByLabel",
        metricLabel=0.0,
    )
    evaluator_rec_0 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="recallByLabel",
        metricLabel=0.0,
    )

    # metrics for delayed (label 1)
    evaluator_prec_1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="precisionByLabel",
        metricLabel=1.0,
    )
    evaluator_rec_1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="recallByLabel",
        metricLabel=1.0,
    )

    return {
        "accuracy": evaluator_acc.evaluate(pred_df),
        "f1_macro": evaluator_f1.evaluate(pred_df),  # macro-averaged
        "precision_not_delayed": evaluator_prec_0.evaluate(pred_df),
        "recall_not_delayed": evaluator_rec_0.evaluate(pred_df),
        "precision_delayed": evaluator_prec_1.evaluate(pred_df),
        "recall_delayed": evaluator_rec_1.evaluate(pred_df),
    }

# COMMAND ----------

# Feature Engineering
train_eng = engineer_features(train_balanced)
val_eng = engineer_features(val_df)
#test_eng = engineer_features(test_df)

# Target Encoding
#train_enc, val_enc, test_enc = target_encode_smooth(train_eng, val_eng, test_eng, "ORIGIN")
#train_enc, val_enc, test_enc = target_encode_smooth(train_enc, val_enc, test_enc, "DEST")
#train_enc, val_enc, test_enc = target_encode_smooth(train_enc, val_enc, test_enc, "PREV_ORIGIN")
train_enc, val_enc = target_encode_smooth(train_eng, val_eng, "ORIGIN")
train_enc, val_enc = target_encode_smooth(train_enc, val_enc, "DEST")
train_enc, val_enc = target_encode_smooth(train_enc, val_enc, "PREV_ORIGIN")
#train_enc, val_enc, test_enc = target_encode_smooth(train_enc, val_enc, test_enc, "OP_UNIQUE_CARRIER") #just one hot encode: cardinality = 16

# COMMAND ----------

# Encoded High-Cardinality Variables (Numeric Risk Scores)
encoded_cols = [
    "ORIGIN_encoded", 
    "DEST_encoded", 
    "PREV_ORIGIN_encoded"#,
    #"OP_UNIQUE_CARRIER_encoded"
]

# kept as categorical due to low cardinality
cat_cols_to_ohe = [
    "MONTH", 
    "QUARTER", 
    "DAY_OF_WEEK", 
    "sched_hour",
    "OP_UNIQUE_CARRIER"#, # We created this, acts like time of day category
    #"PREV_DEP_DELAY_GROUP"#add previous delay group
]

numeric_cols = [
    "DISTANCE", "PREV_DISTANCE",
    "SCHEDULED_BUFFER", #"EFFECTIVE_BUFFER",
    "DEP_HOUR_SIN", "DEP_HOUR_COS", "STORM_INDEX","HourlyWindGustSpeed",
    "HourlyAltimeterSetting", "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
    "HourlyRelativeHumidity", "HourlyPressureTendency", "HourlyWindSpeed" 
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
#test_pred = model.transform(test_enc)

# COMMAND ----------

# Get Metrics
print("\n--- Top Features by Importance ---")
print(get_feature_importance_with_names(model, train_enc, n=20)[["Feature_Name", "Coefficient"]])

# COMMAND ----------

train_pred_lr = model.transform(train_enc)
val_pred_lr   = model.transform(val_enc)
#test_pred  = model.transform(test_enc)

print("Calculated Metrics for Relaxed Logistic Regression:")

print("\n--- Training Set Results ---")
print(get_metrics(train_pred_lr))
confusion_matrix(train_pred_lr)

print("\n--- Validation Set Results ---")
print(get_metrics(val_pred_lr))
confusion_matrix(val_pred_lr)

#print("\n--- Test Set Results (Final Holdout) ---")
#print(get_metrics(test_pred))
#confusion_matrix(test_pred)

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
    maxIter=50,      
    maxDepth=8,       
    stepSize=0.1,
    weightCol="classWeight",
    seed=42
)

pipeline_gbt = Pipeline(stages=indexers + [encoder, assembler_tree, gbt])


print("Training Gradient Boosted Tree")
model_gbt = pipeline_gbt.fit(train_enc)

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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql import functions as F

def plot_spark_confusion_matrix(pred_df, title="Model Name", label_col="ARR_DEL15"):
    """
    Calculates confusion matrix counts in Spark and plots a Seaborn heatmap.
    """
    # 1. Calculate counts in Spark (Distributed & Fast)
    tp = pred_df.filter((F.col("prediction") == 1) & (F.col(label_col) == 1)).count()
    tn = pred_df.filter((F.col("prediction") == 0) & (F.col(label_col) == 0)).count()
    fp = pred_df.filter((F.col("prediction") == 1) & (F.col(label_col) == 0)).count()
    fn = pred_df.filter((F.col("prediction") == 0) & (F.col(label_col) == 1)).count()

    # 2. Construct Numpy Array (Format: [[TN, FP], [FN, TP]])
    # This matches the sklearn standard layout
    cm = np.array([[tn, fp], [fn, tp]])
    
    # 3. Calculate Percentages for easier reading
    cm_sum = np.sum(cm)
    cm_perc = cm / cm_sum

    # 4. Prepare Labels for the heatmap (Count + Percentage)
    labels = [f"{v1}\n{v2:.1%}" for v1, v2 in zip(cm.flatten(), cm_perc.flatten())]
    labels = np.asarray(labels).reshape(2,2)

    # 5. Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Predicted: On Time', 'Predicted: Delayed'],
                yticklabels=['Actual: On Time', 'Actual: Delayed'])
    
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()



# COMMAND ----------

# GENERATE PLOTS

# Baseline (Majority Class)
plot_spark_confusion_matrix(val_pred_baseline, title="Baseline (Majority Class)")

# Logistic Regression
plot_spark_confusion_matrix(val_pred_lr, title="Logistic Regression")

# Gradient Boosted Tree
plot_spark_confusion_matrix(val_pred_gbt, title="Gradient Boosted Tree")

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
    maxIter=100,      
    maxDepth=10,       
    stepSize=0.1,
    weightCol="classWeight",
    seed=42
)

pipeline_gbt = Pipeline(stages=indexers + [encoder, assembler_tree, gbt])


print("Training Gradient Boosted Tree")
model_gbt = pipeline_gbt.fit(train_enc)

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

from pyspark.sql import functions as F

print("Training Majority Class Baseline")

class_counts = train_enc.groupBy("ARR_DEL15").count().orderBy(F.col("count").desc()).collect()
majority_label = class_counts[0]["ARR_DEL15"]

print(f"Majority Class identified: {majority_label}")

train_pred_baseline = train_enc.withColumn("prediction", F.lit(majority_label).cast("double"))
val_pred_baseline   = val_enc.withColumn("prediction", F.lit(majority_label).cast("double"))

print("\n--- Training Set Results (Baseline) ---")
print(get_metrics(train_pred_baseline))
confusion_matrix(train_pred_baseline)

print("\n--- Validation Set Results (Baseline) ---")
print(get_metrics(val_pred_baseline))
confusion_matrix(val_pred_baseline)

# COMMAND ----------

import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

val_conf_matrix = confusion_matrix(y_val, val_pred_baseline)

plt.figure(figsize=(10, 7))
sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Validation Set')
plt.show()

# COMMAND ----------

count_encoded = len(encoded_cols)
count_numeric = len(numeric_cols)

# 2. Count One-Hot Encoded Features (1-to-Many mapping)
# We need to look at the metadata of the transformed data to see how many columns OHE created
# We transform just 1 row to get the schema metadata
assembler_stage = [s for s in model_gbt.stages if isinstance(s, VectorAssembler)][0]
transformed_sample = model_gbt.transform(val_enc.limit(1))
meta = transformed_sample.schema[assembler_stage.getOutputCol()].metadata

# Iterate through the metadata to count binary features (OHE)
count_ohe = 0
if "ml_attr" in meta and "attrs" in meta["ml_attr"] and "binary" in meta["ml_attr"]["attrs"]:
    count_ohe = len(meta["ml_attr"]["attrs"]["binary"])

# 3. Print the Breakdown
print(f"--- Feature Cardinality Report ---")
print(f"1. Target Encoded Features: {count_encoded} (Original Columns)")
print(f"2. Numeric Features:        {count_numeric} (Original Columns)")
print(f"3. One-Hot Encoded vectors: {count_ohe} (Derived Columns from {len(cat_cols_to_ohe)} categories)")
print(f"------------------------------")
print(f"TOTAL INPUT FEATURES:       {count_encoded + count_numeric + count_ohe}")

# COMMAND ----------

from pyspark.sql import functions as F

# 1. Apply Row-Level Feature Engineering globally 
# (Safe to do once because these calculations don't look across rows)
df_engineered = engineer_features(df)

# 2. Split by Quarter
# Training Pool: Jan 1 to Sep 30 (Q1, Q2, Q3)
train_pool = df_engineered.filter(F.col("QUARTER").isin([1, 2, 3]))

# Blind Test Set: Oct 1 to Dec 31 (Q4)
# NEVER touch this until the very end of the project
blind_test_set = df_engineered.filter(F.col("QUARTER") == 4)

print(f"Training Pool Count: {train_pool.count()}")
print(f"Blind Test Set Count: {blind_test_set.count()}")

# COMMAND ----------

from pyspark.sql import functions as F

def target_encode_smooth(df_train, df_val, df_test, cat_col, target_col="ARR_DEL15", m=20):
    # Calculate Global Mean (Scalar)
    # Check if df_train is actually a dataframe
    global_mean_row = df_train.select(F.mean(target_col)).collect()
    global_mean = global_mean_row[0][0] if global_mean_row else 0.0
    
    # Calculate Stats
    stats = df_train.groupby(cat_col).agg(
        F.count(target_col).alias("n"),
        F.sum(target_col).alias("sum_y")
    )
    
    # Calculate Smoothed Score
    # Formula: (sum_y + m * global_mean) / (n + m)
    stats = stats.withColumn(
        f"{cat_col}_encoded", 
        (F.col("sum_y") + (m * global_mean)) / (F.col("n") + m)
    ).select(cat_col, f"{cat_col}_encoded")
    
    # Join Logic
    def apply_encoding(df_in):
        # Join and fill nulls (unseen categories get the global mean)
        df_out = df_in.join(stats, on=cat_col, how="left")
        return df_out.na.fill({f"{cat_col}_encoded": global_mean})

    return apply_encoding(df_train), apply_encoding(df_val), apply_encoding(df_test)

# COMMAND ----------

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql import functions as F

def run_time_series_cv_updated(df_pool, model_type="lr"):
    results = []
    
    # Define folds (Jan-Sep training pool)
    folds = [
        {"train_end": "2019-03-31", "val_start": "2019-04-01", "val_end": "2019-05-31"},
        {"train_end": "2019-05-31", "val_start": "2019-06-01", "val_end": "2019-07-31"},
        {"train_end": "2019-07-31", "val_start": "2019-08-01", "val_end": "2019-09-30"}
    ]
    
    encoded_cols = [
        "ORIGIN_encoded",
        "DEST_encoded",
        "PREV_ORIGIN_encoded"
    ]
    
    cat_cols_to_ohe = [
        "MONTH",
        "QUARTER",
        "DAY_OF_WEEK",
        "sched_hour",
        "OP_UNIQUE_CARRIER"#,    
        #"PREV_DEP_DELAY_GROUP"  
    ]
    
    numeric_cols = [
        "DISTANCE", "PREV_DISTANCE",
        "SCHEDULED_BUFFER", #"EFFECTIVE_BUFFER",
        "DEP_HOUR_SIN", "DEP_HOUR_COS", "STORM_INDEX", "HourlyWindGustSpeed",
        "HourlyAltimeterSetting", "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
        "HourlyRelativeHumidity", "HourlyPressureTendency", "HourlyWindSpeed"
    ]
    # ----------------------------------

    print(f"Starting Time-Series Cross Validation ({model_type.upper()})...")

    for i, fold in enumerate(folds):
        print(f"\n--- FOLD {i+1} ---")
        print(f"Train Cutoff: {fold['train_end']} | Val Range: {fold['val_start']} to {fold['val_end']}")
        
        # A. TIME SLICING
        train_sub = df_pool.filter(F.col("FL_DATE") <= F.lit(fold['train_end']))
        val_sub = df_pool.filter(
            (F.col("FL_DATE") >= F.lit(fold['val_start'])) & 
            (F.col("FL_DATE") <= F.lit(fold['val_end']))
        )
        
        # B. TARGET ENCODING (Only for ORIGIN, DEST, PREV_ORIGIN)
        # We removed Carrier from here
        train_enc, val_enc, _ = target_encode_smooth(train_sub, val_sub, val_sub, "ORIGIN", target_col="ARR_DEL15")
        train_enc, val_enc, _ = target_encode_smooth(train_enc, val_enc, val_enc, "DEST", target_col="ARR_DEL15")
        train_enc, val_enc, _ = target_encode_smooth(train_enc, val_enc, val_enc, "PREV_ORIGIN", target_col="ARR_DEL15")
        
        # C. PIPELINE SETUP
        # 1. Indexing & OHE for the categorical list (includes Carrier now)
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols_to_ohe]
        encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols_to_ohe], outputCols=[f"{c}_ohe" for c in cat_cols_to_ohe], handleInvalid="keep")
        
        # 2. Vector Assembly
        assembler = VectorAssembler(
            inputCols=encoded_cols + numeric_cols + [f"{c}_ohe" for c in cat_cols_to_ohe],
            outputCol="unscaled_features", 
            handleInvalid="skip"
        )
        
        stages = indexers + [encoder, assembler]
        
        # 3. Model Logic
        if model_type == "lr":
            scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)
            lr = LogisticRegression(featuresCol="features", labelCol="ARR_DEL15", 
                                    maxIter=20, regParam=0.001, elasticNetParam=0.0, weightCol="classWeight")
            stages += [scaler, lr]
        
        elif model_type == "gbt":
            assembler.setOutputCol("features") # Skip scaling for GBT
            gbt = GBTClassifier(featuresCol="features", labelCol="ARR_DEL15", 
                                maxIter=20, maxDepth=5, weightCol="classWeight", seed=42)
            stages += [gbt]

        # D. TRAIN & PREDICT
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(train_enc)
        preds = model.transform(val_enc)
        
        # E. METRICS
        tp = preds.filter((F.col("prediction")==1) & (F.col("ARR_DEL15")==1)).count()
        fn = preds.filter((F.col("prediction")==0) & (F.col("ARR_DEL15")==1)).count()
        tn = preds.filter((F.col("prediction")==0) & (F.col("ARR_DEL15")==0)).count()
        fp = preds.filter((F.col("prediction")==1) & (F.col("ARR_DEL15")==0)).count()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall_delayed = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            "Fold": i + 1,
            "Accuracy": accuracy,
            "Recall_Delayed": recall_delayed,
            "TP": tp, "FN": fn
        }
        results.append(metrics)
        print(f"Fold {i+1}: Acc={accuracy:.3f}, Recall={recall_delayed:.3f}")

    return pd.DataFrame(results)

# --- RUN THE UPDATED CV ---
cv_results = run_time_series_cv_updated(train_pool, model_type="lr")
print(cv_results.to_markdown())

cv_results = run_time_series_cv_updated(train_pool, model_type="gbt")
print(cv_results.to_markdown())

# COMMAND ----------

import pandas as pd
import time
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql import functions as F

def run_time_series_cv_detailed(df_pool, model_type="lr"):
    results = []
    
    # Define folds
    folds = [
        {"train_end": "2019-03-31", "val_start": "2019-04-01", "val_end": "2019-05-31"},
        {"train_end": "2019-05-31", "val_start": "2019-06-01", "val_end": "2019-07-31"},
        {"train_end": "2019-07-31", "val_start": "2019-08-01", "val_end": "2019-09-30"}
    ]
    
    encoded_cols = ["ORIGIN_encoded", "DEST_encoded", "PREV_ORIGIN_encoded"]
    cat_cols_to_ohe = ["MONTH", "QUARTER", "DAY_OF_WEEK", "sched_hour", "OP_UNIQUE_CARRIER"]
    numeric_cols = [
        "DISTANCE", "PREV_DISTANCE", "SCHEDULED_BUFFER", 
        "DEP_HOUR_SIN", "DEP_HOUR_COS", "STORM_INDEX", "HourlyWindGustSpeed",
        "HourlyAltimeterSetting", "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
        "HourlyRelativeHumidity", "HourlyPressureTendency", "HourlyWindSpeed"
    ] # add hourly precipitation
    
    print(f"Starting Time-Series CV ({model_type.upper()}) with Detailed Metrics...")

    for i, fold in enumerate(folds):
        start_time = time.time()
        
        print(f"\n--- FOLD {i+1} ---")
        print(f"Train Cutoff: {fold['train_end']} | Val Range: {fold['val_start']} to {fold['val_end']}")
        
        # A. TIME SLICING
        train_sub = df_pool.filter(F.col("FL_DATE") <= F.lit(fold['train_end']))
        val_sub = df_pool.filter(
            (F.col("FL_DATE") >= F.lit(fold['val_start'])) & 
            (F.col("FL_DATE") <= F.lit(fold['val_end']))
        )
        
        # B. TARGET ENCODING
        train_enc, val_enc, _ = target_encode_smooth(train_sub, val_sub, val_sub, "ORIGIN", target_col="ARR_DEL15")
        train_enc, val_enc, _ = target_encode_smooth(train_enc, val_enc, val_enc, "DEST", target_col="ARR_DEL15")
        train_enc, val_enc, _ = target_encode_smooth(train_enc, val_enc, val_enc, "PREV_ORIGIN", target_col="ARR_DEL15")
        
        # C. MODEL PIPELINE
        if model_type == "baseline":
            # Majority Class Logic (Predict 0 for everything)
            counts = train_enc.groupBy("ARR_DEL15").count().collect()
            majority_class = sorted(counts, key=lambda x: x['count'], reverse=True)[0]['ARR_DEL15']
            preds = val_enc.withColumn("prediction", F.lit(majority_class))
            num_features = 0
            
        else:
            indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols_to_ohe]
            encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols_to_ohe], outputCols=[f"{c}_ohe" for c in cat_cols_to_ohe], handleInvalid="keep")
            
            assembler = VectorAssembler(
                inputCols=encoded_cols + numeric_cols + [f"{c}_ohe" for c in cat_cols_to_ohe],
                outputCol="unscaled_features", handleInvalid="skip"
            )
            
            stages = indexers + [encoder, assembler]
            
            if model_type == "lr":
                scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)
                lr = LogisticRegression(featuresCol="features", labelCol="ARR_DEL15", 
                                        maxIter=20, regParam=0.001, elasticNetParam=0.0, weightCol="classWeight")
                stages += [scaler, lr]
            
            elif model_type == "gbt":
                assembler.setOutputCol("features")
                gbt = GBTClassifier(featuresCol="features", labelCol="ARR_DEL15", 
                                    maxIter=20, maxDepth=5, weightCol="classWeight", seed=42)
                stages += [gbt]

            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(train_enc)
            num_features = model.stages[-1].numFeatures
            preds = model.transform(val_enc)
        
        # D. DETAILED METRICS CALCULATION
        # Get raw counts first
        tp = preds.filter((F.col("prediction")==1) & (F.col("ARR_DEL15")==1)).count()
        fn = preds.filter((F.col("prediction")==0) & (F.col("ARR_DEL15")==1)).count()
        tn = preds.filter((F.col("prediction")==0) & (F.col("ARR_DEL15")==0)).count()
        fp = preds.filter((F.col("prediction")==1) & (F.col("ARR_DEL15")==0)).count()
        
        # Helper for safe division (avoid div by zero errors)
        def safe_div(n, d):
            return n / d if d > 0 else 0.0

        # Calculate metrics manually (Much faster than running 6 Spark Evaluators)
        # 1. Accuracy
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        
        # 2. Precision/Recall for Not Delayed (Label 0)
        # Precision 0 = TN / (TN + FN) (Of all predicted 0s, how many were actually 0?)
        prec_0 = safe_div(tn, tn + fn)
        # Recall 0 = TN / (TN + FP) (Of all actual 0s, how many did we find?)
        rec_0 = safe_div(tn, tn + fp)
        
        # 3. Precision/Recall for Delayed (Label 1)
        prec_1 = safe_div(tp, tp + fp)
        rec_1 = safe_div(tp, tp + fn)
        
        # 4. F1 Scores
        f1_0 = safe_div(2 * prec_0 * rec_0, prec_0 + rec_0)
        f1_1 = safe_div(2 * prec_1 * rec_1, prec_1 + rec_1)
        f1_macro = (f1_0 + f1_1) / 2

        # Timing
        end_time = time.time()
        duration_min = (end_time - start_time) / 60
        
        # Print Summary
        print(f"Confusion Matrix (Fold {i+1}):")
        print(f"                 Predicted")
        print(f"                 0 (OK)   1 (Late)")
        print(f"Actual 0 (OK)   {tn:6d}   {fp:6d}")
        print(f"       1 (Late) {fn:6d}   {tp:6d}")
        print(f"----------------------------------")
        print(f"Time: {duration_min:.2f} min | Acc: {accuracy:.3f} | F1 Macro: {f1_macro:.3f} | Recall(1): {rec_1:.3f}")

        # Store Results
        metrics = {
            "Fold": i + 1,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_not_delayed": prec_0,
            "recall_not_delayed": rec_0,
            "precision_delayed": prec_1,
            "recall_delayed": rec_1,
            "Input_Features": num_features,
            "Duration_Mins": round(duration_min, 2)
        }
        results.append(metrics)

    return pd.DataFrame(results)

# --- EXECUTE ---

print(">>> 1. Baseline (Majority Class)")
cv_baseline = run_time_series_cv_detailed(train_pool, model_type="baseline")
print(cv_baseline.to_markdown())

print("\n>>> 2. Logistic Regression (Weighted)")
cv_lr = run_time_series_cv_detailed(train_pool, model_type="lr")
print(cv_lr.to_markdown())

print("\n>>> 3. Gradient Boosted Tree (Weighted)")
cv_gbt = run_time_series_cv_detailed(train_pool, model_type="gbt")
print(cv_gbt.to_markdown())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC