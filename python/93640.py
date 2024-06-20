# # Notebook #3
# 
# ## Outline
# 
# - [Load Result from Notebook #2](#Load-Result-from-Notebook-#2)
# - [Label Construction](#Label-Construction)
# - [Feature Reduction](#Feature-Reduction)
# - [Prepare Train and Test Data](#Prepare-train-and-test-data)
#    - [Time-dependent Splitting](#Prepare-train-and-test-dataset-using-time-split-method)
#    - [Down Sample Negative Examples](#Down-Sample-Negative-examples:)
#    - [Cache Results](#Cache-results)
# - [Binary Classification Models](#Binary-Classification-Models:)
#    - [Random Forest classifier](#Random-Forest-classifier)
#    - [Gradient-Boosted Tree classifier](#Gradient-Boosted-Tree-classifier)
#    - [Hyper-Parameter Tuning & Cross Validation](#Hyper-Parameter-Tuning-&-Cross-Validation)
#    
#  <br>
# 

import pyspark.sql.functions as F
import time
import pandas as pd
import subprocess
import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import datetime
import atexit

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col,udf,lag,date_add,explode,lit,concat,unix_timestamp,sum, abs
from pandas import DataFrame
from pyspark.sql.dataframe import *
from pyspark.ml.classification import *
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window
from pyspark.sql.types import DateType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import Row
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer, RFormula
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import roc_curve,auc
from pyspark.sql.functions import month, weekofyear, dayofmonth
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.grid_search import ParameterGrid


# ## Load Result from Notebook #2 
# 

# load result from Notebook #2 ("FeatureEngineering_RollingCompute")
df = sqlContext.read.parquet('/mnt/resource/PysparkExample/notebook2_result.parquet')

# check the dimension of the dataset and make sure things look right
print(df.count(), len(df.columns))
df.select('key','deviceid').show(3)


# ## Label Construction
# 
# For predictive maintenance use cases, we usually want to predict failure/problem ahead of time. In our example, we would like to be able to predict machine problem 7 days (failure prediction time window) in advance. That means for the label column, we need to label all the 7 days before the actual failure/problem day as "1". This time window should be picked based on the specific business case: in some situations it may be enough to predict failures hours in advance, while in others days or even weeks might be needed to make meaningful business decision such as allowing enough time for arrival of replacement parts.
# 
# To find more detailed information about the label construction technique, please visit [this link](https://pdfs.semanticscholar.org/284d/f4ec85eed338a87fece985246c5bd4f56495.pdf).
# <br>
# 

#------------------------------------ Create label column ------------------------------------#

# Step 1: 
df = df.withColumn('label_tmp', col('problemreported')) 

# Step 2:
wSpec = Window.partitionBy('deviceid').orderBy(df.date.desc())
lag_window = 7  # Define how many days in advance we want to predict failure

for i in range(lag_window):
    lag_values = lag(df.label_tmp, default=0).over(wSpec)
    df = df.withColumn('label_tmp', F.when((col('label_tmp')==1) | (lag_values==None) | (lag_values<1) | (lag_values>=(lag_window+1)), col('label_tmp')).otherwise(lag_values+1))

# check the results
print(df.select('label_tmp').distinct().rdd.map(lambda r: r[0]).collect()) 
 


# Step 3:
### please note that we need to make "label" column double instead of integer for the pyspark classification models 
df = df.withColumn('label', F.when(col('label_tmp') > 0, 1.0).otherwise(0.0))
df.createOrReplaceTempView("df_view") 
 
# Step 4:
df.orderBy('deviceid', 'date').select('deviceid', 'date', 'problemreported', 'label_tmp', 'label').show(20) 


# Visualize the distribution of "label" column
df.select('label').describe().show()


# ## Feature Reduction
# -  There are not many packages for feature selection in PySpark 2.0.2.,so we decided to use PCA to reduce the demensionality.
# -  There are so many features especially rolling features, we need to perform feature selection to reduce the feature set size
# 

## check the number of rolling features
len([col_n for col_n in df.columns if '_rolling' in col_n])


# Step 1
# Use RFormula to create the feature vector
rolling_features = list(s for s in df.columns if "_rolling" in s)
formula = RFormula(formula = "~" + "+".join(rolling_features))
output = formula.fit(df).transform(df).select("key","features") 


# Step 2 
# Before PCA, we need to standardize the features, it is very important...
# We compared 1) standardization, 2) min-max normalization, 3) combintion of standardization and min-max normalization
# In 2), the 1st PC explained more than 67% of the variance
# 1) & 3) generate exactly the same results for model.explainedVariance. 
# That means min-max normalization does not help in our case

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(output)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(output)


# Step 3
pca = PCA(k=50, inputCol="scaledFeatures", outputCol="pca_roll_features")
model = pca.fit(scaledData)
result = model.transform(scaledData).select("key","pca_roll_features")
print(model.explainedVariance)


# Step 4
df = df.join(result, 'key', 'inner')
rolling_drop_list = [col_n for col_n in df.columns if '_rolling' in col_n]
df = df.select([column for column in df.columns if column not in rolling_drop_list])

df.select('key','pca_roll_features').show(5)


# # Prepare train and test data
# 

# Define list of input columns for downstream modeling
input_features = [
 'usage_count_1',
 'usage_count_2',
 'problem_type_1',
 'problem_type_2',
 'problem_type_3',
 'problem_type_4',
 'error_count_1',
 'error_count_2',
 'error_count_3',
 'error_count_4',
 'error_count_5',
 'error_count_6',
 'error_count_7',
 'error_count_8',
 'month',
 'weekofyear',
 'dayofmonth',
 'warn_type1_total',
 'warn_type2_total',
 'fault_code_type_1_count',
 'fault_code_type_2_count',
 'fault_code_type_3_count',
 'fault_code_type_4_count',
 'problem_type_1_per_usage1',
 'problem_type_2_per_usage1',
 'problem_type_3_per_usage1',
 'problem_type_4_per_usage1',
 'fault_code_type_1_count_per_usage1',
 'fault_code_type_2_count_per_usage1',
 'fault_code_type_3_count_per_usage1',
 'fault_code_type_4_count_per_usage1',
 'problem_type_1_per_usage2',
 'problem_type_2_per_usage2',
 'problem_type_3_per_usage2',
 'problem_type_4_per_usage2',
 'fault_code_type_1_count_per_usage2',
 'fault_code_type_2_count_per_usage2',
 'fault_code_type_3_count_per_usage2',
 'fault_code_type_4_count_per_usage2',   
 'problem_type_1_category_encoded',
 'problem_type_2_category_encoded',
 'problem_type_3_category_encoded',
 'problem_type_4_category_encoded',
 'problem_type_1_per_usage1_category_encoded',
 'problem_type_2_per_usage1_category_encoded',
 'problem_type_3_per_usage1_category_encoded',
 'problem_type_4_per_usage1_category_encoded',
 'problem_type_1_per_usage2_category_encoded',
 'problem_type_2_per_usage2_category_encoded',
 'problem_type_3_per_usage2_category_encoded',
 'problem_type_4_per_usage2_category_encoded',
 'fault_code_type_1_count_category_encoded',
 'fault_code_type_2_count_category_encoded',
 'fault_code_type_3_count_category_encoded',
 'fault_code_type_4_count_category_encoded',
 'fault_code_type_1_count_per_usage1_category_encoded',
 'fault_code_type_2_count_per_usage1_category_encoded',
 'fault_code_type_3_count_per_usage1_category_encoded',
 'fault_code_type_4_count_per_usage1_category_encoded',
 'fault_code_type_1_count_per_usage2_category_encoded',
 'fault_code_type_2_count_per_usage2_category_encoded',
 'fault_code_type_3_count_per_usage2_category_encoded',
 'fault_code_type_4_count_per_usage2_category_encoded',
 'cat1_encoded',
 'cat2_encoded',
 'cat3_encoded',
 'cat4_encoded',     
 'pca_1_warn',
 'pca_2_warn',
 'pca_3_warn',
 'pca_4_warn',
 'pca_5_warn',
 'pca_6_warn',
 'pca_7_warn',
 'pca_8_warn',
 'pca_9_warn',
 'pca_10_warn',
 'pca_11_warn',
 'pca_12_warn',
 'pca_13_warn',
 'pca_14_warn',
 'pca_15_warn',
 'pca_16_warn',
 'pca_17_warn',
 'pca_18_warn',
 'pca_19_warn',
 'pca_20_warn',
 'pca_roll_features'
]

label_var = ['label']
key_cols =['key','deviceid','date']


# Assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')
df = va.transform(df).select('deviceid','date','label','features')


# Set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(df)
    


# ### Remember to do “StringIndexer” on the label column, fit on the entire dataset to include all labels in index. Also, the label column has to be Double instead of Integer type.  
# 

# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)


# ### Prepare train and test dataset using time split method 
# -  training data: year 2012-2014
# -  testing data: year 2015
# 

training = df.filter(df.date > "2011-12-31").filter(df.date < "2015-01-01")
testing = df.filter(df.date > "2014-12-31")

print(training.count())
print(testing.count())


## show the distribution of label "0" and "1"
df.groupby('label').count().show()


# ### Down-Sample Negative examples:
# -  This is a ***highly umbalanced data*** with way more label "0" than "1" ("1" only accounts for 1.5%).
# -  So we need to down sample the negatives while keeping all positive samples.
# -  To make label "1" to "0" ratio close to 1:10 (you can use other ratio for example 1:5), we need to down-sample the "0"s (take 13.5% of all the label "0"s)
# 

# SampleBy returns a stratified sample without replacement based on the fraction given on each stratum
train_downsampled = training.sampleBy('label', fractions={0.0: 0.135, 1.0: 1.0}, seed=123).cache()
train_downsampled.groupby('label').count().show()

testing.groupby('label').count().show()


# ### Cache results 
# 
# Do it when necessary especially if your downstream work (e.g. recursive modeling) use that data over and over again. Here in our case, after the train and test datasets are prepared, we cache them in memory. 
# 

# cache datasets in memory
train_downsampled.cache()
testing.cache()

# check the number of devices in training and testing data
print(train_downsampled.select('deviceid').distinct().count())
print(testing.select('deviceid').distinct().count())


# Set model storage directory path. This is where models will be saved.
modelDir = "/mnt/resource/PysparkExample/Outputs/"; 


# ## Binary Classification Models:
# -  Random Forest classifier
# -  Gradient-Boosted Tree
# 

# ### Random Forest classifier
# 

get_ipython().run_cell_magic('time', '', '\n# Train a RandomForest model.\nrf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=100)\n\n# Chain indexers and forest in a Pipeline\npipeline_rf = Pipeline(stages=[labelIndexer, featureIndexer, rf])\n\n# Train model.  This also runs the indexers.\nmodel_rf = pipeline_rf.fit(train_downsampled)\n\n# Save model\ndatestamp = unicode(datetime.datetime.now()).replace(\' \',\'\').replace(\':\',\'_\');\nrf_fileName = "RandomForest_" + datestamp;\nrfDirfilename = modelDir + rf_fileName;\nmodel_rf.save(rfDirfilename)\n\n# Make predictions.\npredictions_rf = model_rf.transform(testing)\npredictions_rf.groupby(\'indexedLabel\', \'prediction\').count().show()')


predictions_rf.dtypes


get_ipython().run_cell_magic('time', '', '\npredictionAndLabels = predictions_rf.select("indexedLabel", "prediction").rdd\nmetrics = BinaryClassificationMetrics(predictionAndLabels)\nprint("Area under ROC = %g" % metrics.areaUnderROC)\nprint("Area under PR = %g\\n" % metrics.areaUnderPR)\n\n# Select (prediction, true label) and compute test error\nevaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction")\nprint("Accuracy = %g" % evaluator.evaluate(predictions_rf, {evaluator.metricName: "accuracy"}))\nprint("Weighted Precision = %g" % evaluator.evaluate(predictions_rf, {evaluator.metricName: "weightedPrecision"}))\nprint("Weighted Recall = %g" % evaluator.evaluate(predictions_rf, {evaluator.metricName: "weightedRecall"}))\nprint("F1 = %g" % evaluator.evaluate(predictions_rf, {evaluator.metricName: "f1"}))\n\n# PLOT ROC curve after converting predictions to a Pandas dataframe\n%matplotlib inline\npredictions_rf_pddf = predictions_rf.select(\'indexedLabel\',\'probability\').toPandas()\nlabels = predictions_rf_pddf["indexedLabel"]\nprob = []\nfor dv in predictions_rf_pddf["probability"]:\n    prob.append(dv.values[1])\n     \nfpr, tpr, thresholds = roc_curve(labels, prob, pos_label=1.0);\nroc_auc = auc(fpr, tpr)\n\nplt.figure(figsize=(5, 5))\nplt.plot(fpr, tpr, label=\'ROC curve (area = %0.2f)\' % roc_auc)\nplt.plot([0, 1], [0, 1], \'k--\')\nplt.xlim([0.0, 1.0])\nplt.ylim([0.0, 1.05])\nplt.xlabel(\'False Positive Rate\')\nplt.ylabel(\'True Positive Rate\')\nplt.title(\'ROC Curve\')\nplt.legend(loc="lower right")\nplt.show()')


# #### Pyspark MulticlassClassificationEvaluator in version 2.0.2 only gives weighted precision and recall. But we would also like to see the raw precision and recall as well.
# 

# Use sklearn
rf_result = predictions_rf.select('indexedLabel', 'prediction').toPandas()

rf_label = rf_result['indexedLabel'].tolist()
rf_prediction = rf_result['prediction'].tolist()

precision, recall, fscore, support = score(rf_label, rf_prediction)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# ### Gradient-Boosted Tree classifier
# 

get_ipython().run_cell_magic('time', '', '\n# Train a GBT model.\ngbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=10, minInstancesPerNode=5, maxIter=50)\n\n# Chain indexers and GBT in a Pipeline\npipeline_gbt = Pipeline(stages=[labelIndexer, featureIndexer, gbt])\n\n# Train model.  This also runs the indexers.\nmodel_gbt = pipeline_gbt.fit(train_downsampled)\n\n# save model\ndatestamp = unicode(datetime.datetime.now()).replace(\' \',\'\').replace(\':\',\'_\');\ngbt_fileName = "GradientBoostedTree_" + datestamp;\ngbtDirfilename = modelDir + gbt_fileName;\nmodel_gbt.save(gbtDirfilename)\n\n# Make predictions.\npredictions_gbt = model_gbt.transform(testing)')


# show prediction results
predictions_gbt.groupby('indexedLabel', 'prediction').count().show()


get_ipython().run_cell_magic('time', '', '\npredictionAndLabels = predictions_gbt.select("indexedLabel", "prediction").rdd\nmetrics = BinaryClassificationMetrics(predictionAndLabels)\nprint("Area under ROC = %g" % metrics.areaUnderROC)\nprint("Area under PR = %g\\n" % metrics.areaUnderPR)\n\n# Select (prediction, true label) and compute test error\nevaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction")\nprint("Accuracy = %g" % evaluator.evaluate(predictions_gbt, {evaluator.metricName: "accuracy"}))\nprint("Weighted Precision = %g" % evaluator.evaluate(predictions_gbt, {evaluator.metricName: "weightedPrecision"}))\nprint("Weighted Recall = %g" % evaluator.evaluate(predictions_gbt, {evaluator.metricName: "weightedRecall"}))\nprint("F1 = %g" % evaluator.evaluate(predictions_gbt, {evaluator.metricName: "f1"}))')


# Use sklearn to calculate the raw precision and recall

gbt_result = predictions_gbt.select('indexedLabel', 'prediction').toPandas()

gbt_label = gbt_result['indexedLabel'].tolist()
gbt_prediction = gbt_result['prediction'].tolist()

precision, recall, fscore, support = score(gbt_label, gbt_prediction)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# ### Comparing results from Random Forest and Gradient Boosted Tree:
# -  Gradient Boosted Tree gives better recall but worse precision compared with Random Forest. For most of the predictive maintenance use cases, the business cost associated with false positives is usually expensive. There is always a trade-off between precision and recall. We want to achieve higher precision rate (fewer false positives) even though that might compromise the recall rate.
# -  That is why we decided to go with Random Forest model and further optimized it with hyper-parametre tuning.
# 

# ## Hyper-Parameter Tuning & Cross Validation
# 
# Train a random forest classification model using hyper-parameter tuning and cross-validation
# 

get_ipython().run_cell_magic('time', '', '\n# Train a RandomForest model.\nrf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", \n                            featureSubsetStrategy="auto", impurity="gini", seed=123)\n\n# Chain indexers and forest in a Pipeline\npipeline_rf = Pipeline(stages=[labelIndexer, featureIndexer, rf])\n\n\n## Define parameter grid\nparamGrid = ParamGridBuilder() \\\n    .addGrid(rf.numTrees, [20, 50, 100]) \\\n    .addGrid(rf.maxBins, [10, 20]) \\\n    .addGrid(rf.maxDepth, [3, 5, 7]) \\\n    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \\\n    .build()\n\n## Define cross-validation\ncrossval = CrossValidator(estimator=pipeline_rf,\n                          estimatorParamMaps=paramGrid,\n                          evaluator=MulticlassClassificationEvaluator(metricName="weightedPrecision"),\n                          numFolds=3)\n\n## Train model using CV\ncvModel = crossval.fit(train_downsampled)\n\n## Predict and evaluate\npredictions = cvModel.transform(testing)\nevaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")\nr2 = evaluator.evaluate(predictions)\nprint("weightedPrecision on test data = %g" % r2)\n\n## Save the best model\nfileName = "CV_RandomForestClassificationModel_" + datestamp;\nCVDirfilename = modelDir + fileName;\ncvModel.bestModel.save(CVDirfilename);')


# #### Hyper-parameter tuning only improved the model performance a little bit. We will then use that model for future scoring.
# 

