# # Step 4: Model operationalization & Deployment
# 
# In this script, a model is saved as a .model file along with the relevant scheme for deployment. The functions are first tested locally before operationalizing the model using Azure Machine Learning Model Management environment for use in production in realtime.
# 
# **Note:** This notebook will take about 1 minute to execute all cells, depending on the compute configuration you have setup. 
# 

## setup our environment by importing required libraries
import json
import os
import shutil
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

# setup the pyspark environment
from pyspark.sql import SparkSession

from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

# For Azure blob storage access
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess

# For logging model evaluation parameters back into the
# AML Workbench run history plots.
import logging
from azureml.logging import get_azureml_logger

amllog = logging.getLogger("azureml")
amllog.level = logging.INFO

# Turn on cell level logging.
get_ipython().run_line_magic('azureml', 'history on')
get_ipython().run_line_magic('azureml', 'history show')

# Time the notebook execution. 
# This will only make sense if you "Run all cells"
tic = time.time()

logger = get_azureml_logger() # logger writes to AMLWorkbench runtime view
spark = SparkSession.builder.getOrCreate()

# Telemetry
logger.log('amlrealworld.predictivemaintenance.operationalization','true')


# We need to load the feature data set from memory to construct the operationalization schema. We again will require your storage account name and account key to connect to the blob storage.
# 

# Enter your Azure blob storage details here 
ACCOUNT_NAME = "<your blob storage account name>"

# You can find the account key under the _Access Keys_ link in the 
# [Azure Portal](portal.azure.com) page for your Azure storage container.
ACCOUNT_KEY = "<your blob storage account key>"
#-------------------------------------------------------------------------------------------
# We will create this container to hold the results of executing this notebook.
# If this container name already exists, we will use that instead, however
# This notebook will ERASE ALL CONTENTS.
CONTAINER_NAME = "featureengineering"
FE_DIRECTORY = 'featureengineering_files.parquet'

MODEL_CONTAINER = 'modeldeploy'

# Connect to your blob service     
az_blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)

# Create a new container if necessary, otherwise you can use an existing container.
# This command creates the container if it does not already exist. Else it does nothing.
az_blob_service.create_container(CONTAINER_NAME, 
                                 fail_on_exist=False, 
                                 public_access=PublicAccess.Container)

# create a local path where to store the results later.
if not os.path.exists(FE_DIRECTORY):
    os.makedirs(FE_DIRECTORY)

# download the entire parquet result folder to local path for a new run 
for blob in az_blob_service.list_blobs(CONTAINER_NAME):
    if CONTAINER_NAME in blob.name:
        local_file = os.path.join(FE_DIRECTORY, os.path.basename(blob.name))
        az_blob_service.get_blob_to_path(CONTAINER_NAME, blob.name, local_file)

fedata = spark.read.parquet(FE_DIRECTORY)

fedata.limit(5).toPandas().head(5)


# ## Define init and run functions
# Start by defining the init() and run() functions as shown in the cell below. Then write them to the score.py file. This file will load the model, perform the prediction, and return the result.
# 
# The init() function initializes your web service, loading in any data or models that you need to score your inputs. In the example below, we load in the trained model. This command is run when the Docker container containing your service initializes.
# The run() function defines what is executed on a scoring call. In our simple example, we simply load in the input as a data frame, and run our pipeline on the input, and return the prediction.
# 

def init():
    # read in the model file
    from pyspark.ml import PipelineModel
    global pipeline
    
    pipeline = PipelineModel.load(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']+'pdmrfull.model')
    
def run(input_df):
    import json
    response = ''
    try:
        #Get prediction results for the dataframe
        
        # We'll use the known label, key variables and 
        # a few extra columns we won't need.
        key_cols =['label_e','machineID','dt_truncated', 'failure','model_encoded','model' ]

        # Then get the remaing feature names from the data
        input_features = input_df.columns

        # Remove the extra stuff if it's in the input_df
        input_features = [x for x in input_features if x not in set(key_cols)]
        
        # Vectorize as in model building
        va = VectorAssembler(inputCols=(input_features), outputCol='features')
        data = va.transform(input_df).select('machineID','features')
        score = pipeline.transform(data)
        predictions = score.collect()

        #Get each scored result
        preds = [str(x['prediction']) for x in predictions]
        response = ",".join(preds)
    except Exception as e:
        print("Error: {0}",str(e))
        return (str(e))
    
    # Return results
    print(json.dumps(response))
    return json.dumps(response)


# ### Create schema and schema file
# Create a schema for the input to the web service and generate the schema file. This will be used to create a Swagger file for your web service which can be used to discover its input and sample data when calling it.
# 

# define the input data frame
inputs = {"input_df": SampleDefinition(DataTypes.SPARK, 
                                       fedata.drop("dt_truncated","failure","label_e", "model","model_encoded"))}

json_schema = generate_schema(run_func=run, inputs=inputs, filepath='service_schema.json')


# ### Test init and run
# We can then test the init() and run() functions right here in the notebook, before we decide to actually publish a web service.
# 

# We'll use the known label, key variables and 
# a few extra columns we won't need. (machineID is required)
key_cols =['label_e','dt_truncated', 'failure','model_encoded','model' ]

# Then get the remaining feature names from the data
input_features = fedata.columns
# Remove the extra stuff if it's in the input_df
input_features = [x for x in input_features if x not in set(key_cols)]


# this is an example input data record
input_data = [[114, 163.375732902,333.149484586,100.183951698,44.0958812638,164.114723991,
               277.191815232,97.6289110707,50.8853505161,21.0049565219,67.5287259378,12.9361526861,
               4.61359760918,15.5377738062,67.6519885441,10.528274633,6.94129487555,0.0,0.0,0.0,
               0.0,0.0,489.0,549.0,549.0,564.0,18.0]]

df = (spark.createDataFrame(input_data, input_features))

# test init() in local notebook
init()

# test run() in local notebook
run(df)


# ## Persist model assets
# 
# Next we persist the assets we have created to disk for use in operationalization.
# 

# save the schema file for deployment
out = json.dumps(json_schema)
with open(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + 'service_schema.json', 'w') as f:
    f.write(out)


# Now we will use `%%writefile` meta command to save the `init()` and `run()` functions to the save the `pdmscore.py` file.
# 

get_ipython().run_cell_magic('writefile', "{os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']}/pdmscore.py", '\nimport json\nfrom pyspark.ml import Pipeline\nfrom pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier\n\n# for creating pipelines and model\nfrom pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer\n\ndef init():\n    # read in the model file\n    from pyspark.ml import PipelineModel\n    # read in the model file\n    global pipeline\n    pipeline = PipelineModel.load(\'pdmrfull.model\')\n    \ndef run(input_df):\n    response = \'\'\n    try:\n       \n        # We\'ll use the known label, key variables and \n        # a few extra columns we won\'t need.\n        key_cols =[\'label_e\',\'machineID\',\'dt_truncated\', \'failure\',\'model_encoded\',\'model\' ]\n\n        # Then get the remaing feature names from the data\n        input_features = input_df.columns\n\n        # Remove the extra stuff if it\'s in the input_df\n        input_features = [x for x in input_features if x not in set(key_cols)]\n        \n        # Vectorize as in model building\n        va = VectorAssembler(inputCols=(input_features), outputCol=\'features\')\n        data = va.transform(input_df).select(\'machineID\',\'features\')\n        score = pipeline.transform(data)\n        predictions = score.collect()\n\n        #Get each scored result\n        preds = [str(x[\'prediction\']) for x in predictions]\n        response = ",".join(preds)\n    except Exception as e:\n        print("Error: {0}",str(e))\n        return (str(e))\n    \n    # Return results\n    print(json.dumps(response))\n    return json.dumps(response)\n\nif __name__ == "__main__":\n    init()\n    run("{\\"input_df\\":[{\\"machineID\\":114,\\"volt_rollingmean_3\\":163.375732902,\\"rotate_rollingmean_3\\":333.149484586,\\"pressure_rollingmean_3\\":100.183951698,\\"vibration_rollingmean_3\\":44.0958812638,\\"volt_rollingmean_24\\":164.114723991,\\"rotate_rollingmean_24\\":277.191815232,\\"pressure_rollingmean_24\\":97.6289110707,\\"vibration_rollingmean_24\\":50.8853505161,\\"volt_rollingstd_3\\":21.0049565219,\\"rotate_rollingstd_3\\":67.5287259378,\\"pressure_rollingstd_3\\":12.9361526861,\\"vibration_rollingstd_3\\":4.61359760918,\\"volt_rollingstd_24\\":15.5377738062,\\"rotate_rollingstd_24\\":67.6519885441,\\"pressure_rollingstd_24\\":10.528274633,\\"vibration_rollingstd_24\\":6.94129487555,\\"error1sum_rollingmean_24\\":0.0,\\"error2sum_rollingmean_24\\":0.0,\\"error3sum_rollingmean_24\\":0.0,\\"error4sum_rollingmean_24\\":0.0,\\"error5sum_rollingmean_24\\":0.0,\\"comp1sum\\":489.0,\\"comp2sum\\":549.0,\\"comp3sum\\":549.0,\\"comp4sum\\":564.0,\\"age\\":18.0}]}")')


# These files are stored in the `['AZUREML_NATIVE_SHARE_DIRECTORY']` location on the kernel host machine with the model stored in the `3_model_building.ipynb` notebook. In order to share these assets and operationalize the model, we create a new blob container and store a compressed file containing those assets for later retrieval. 
# 

# Compress the operationalization assets for easy blob storage transfer
MODEL_O16N = shutil.make_archive('o16n', 'zip', os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'])

# Create a new container if necessary, otherwise you can use an existing container.
# This command creates the container if it does not already exist. Else it does nothing.
az_blob_service.create_container(MODEL_CONTAINER, 
                                 fail_on_exist=False, 
                                 public_access=PublicAccess.Container)

# Transfer the compressed operationalization assets into the blob container.
az_blob_service.create_blob_from_path(MODEL_CONTAINER, "o16n.zip", str(MODEL_O16N) ) 


# Time the notebook execution. 
# This will only make sense if you "Run All" cells
toc = time.time()
print("Full run took %.2f minutes" % ((toc - tic)/60))

logger.log("Operationalization Run time", ((toc - tic)/60))


# ## Deployment
# 
# Once the assets are stored, we can download them into a local compute context for operationalization on an Azure web service.
# 
# We demonstrate how to setup this web service this through a CLI window opened in the AML Workbench application. 
# 
# ### Download the model
# 
# To download the model we've saved, follow these instructions on a local computer.
# 
# - Open the [Azure Portal](http://portal.azure.com)
# - In the left hand pane, click on __All resources__
# - Search for the storage account using the name you provided earlier in this notebook. 
# - Choose the storage account from search result list, this will open the storage account panel.
# - On the storage account panel, choose __Blobs__
# - On the Blobs panel choose the container __modeldeploy__
# - Select the file o16n.zip and on the properties pane for that blob choose download.
# 
# Once downloaded, unzip the file into the directory of your choosing. The zip file contains three deployment assets:
# 
# - the `pdmscore.py` file
# - a `pdmrfull.model` directory
# - the `service_schema.json` file
# 
# 
# 
# ### Create a model management endpoint 
# 
# Create a modelmanagement under your account. We will call this `pdmmodelmanagement`. The remaining defaults are acceptable.
# 
# `az ml account modelmanagement create --location <ACCOUNT_REGION> --resource-group <RESOURCE_GROUP> --name pdmmodelmanagement`
# 
# 
# ### Check environment settings
# 
# Show what environment is currently active:
# 
# `az ml env show`
# 
# If nothing is set, we setup the environment with the existing model management context first: 
# 
# ` az ml env setup --location <ACCOUNT_REGION> --resource-group <RESOURCE_GROUP> --name pdmmodelmanagement`
# 
# then set the current environment:
# 
# `az ml env set --resource-group <RESOURCE_GROUP> --cluster-name pdmmodelmanagement`
# 
# Check that the environment is now set:
# 
# `az ml env show`
# 
# 
# ### Deploy your web service 
# 
# Once the environment is setup, we'll deploy the web service from the CLI.
# 
# These commands assume the current directory contains the webservice assets we created in throughout the notebooks in this scenario (`pdmscore.py`, `service_schema.json` and `pdmrfull.model`). If your kernel has run locally, the assets will be in the `os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']`. 
# 
# On windows this points to:
# 
# ```
# cd C:\Users\<username>\.azureml\share\<team account>\<Project Name>
# ```
# 
# on linux variants this points to:
# 
# ```
# cd ~\.azureml\share\<team account>\<Project Name>
# ```
# 
# 
# The command to create a web service (`<SERVICE_ID>`) with these operationalization assets in the current directory is:
# 
# ```
# az ml service create realtime -f <filename> -r <TARGET_RUNTIME> -m <MODEL_FILE> -s <SCHEMA_FILE> -n <SERVICE_ID> --cpu 0.1
# ```
# 
# The default cluster has only 2 nodes with 2 cores each. Some cores are taken for system components. AMLWorkbench asks for 1 core per service. To deploy multiple services into this cluster, we specify the cpu requirement in the service create command as (--cpu 0.1) to request 10% of a core. 
# 
# For this example, we will call our webservice `amlworkbenchpdmwebservice`. This `SERVICE_ID` must be all lowercase, with no spaces:
# 
# ```
# az ml service create realtime -f pdmscore.py -r spark-py -m pdmrfull.model -s service_schema.json --cpu 0.1 -n amlworkbenchpdmwebservice
# ```
# 
# This command will take some time to execute. 
# 
# Once complete, the command returns sample usage commands to test the service for both PowerShell and the cmd prompt. We can execute these commands from the command line as well. For our example:
# 
# ```
# az ml service run realtime -i amlworkbenchpdmwebservice --% -d "{\"input_df\": [{\"rotate_rollingstd_24\": 0.3233426394949046, \"error3sum_rollingmean_24\": 0.0, \"age\": 14, \"machineID\": 45, \"error5sum_rollingmean_24\": 0.0, \"pressure_rollingstd_24\": 0.1945085296751734, \"vibration_rollingstd_24\": 0.36239263228769986, \"rotate_rollingmean_3\": 527.816906803798, \"error1sum_rollingmean_24\": 0.0, \"volt_rollingmean_24\": 185.92637096180658, \"pressure_rollingmean_3\": 117.22597085550017, \"volt_rollingstd_24\": 0.03361414142292652, \"comp1sum\": 474.0, \"comp3sum\": 384.0, \"pressure_rollingmean_24\": 113.56479908060074, \"rotate_rollingstd_3\": 2.2898301915618045, \"volt_rollingmean_3\": 174.88172665757065, \"comp2sum\": 459.0, \"error2sum_rollingmean_24\": 0.0, \"rotate_rollingmean_24\": 470.1219658987775, \"vibration_rollingmean_3\": 39.472146777953654, \"vibration_rollingstd_3\": 0.8102848856599294, \"pressure_rollingstd_3\": 0.010565393835276299, \"error4sum_rollingmean_24\": 0.0, \"volt_rollingstd_3\": 8.308641250692387, \"vibration_rollingmean_24\": 39.93637676066078, \"comp4sum\": 579.0}, {\"rotate_rollingstd_24\": 1.5152162169310932, \"error3sum_rollingmean_24\": 0.0, \"age\": 14, \"machineID\": 45, \"error5sum_rollingmean_24\": 0.0, \"pressure_rollingstd_24\": 0.012495480312639678, \"vibration_rollingstd_24\": 0.21106710997624312, \"rotate_rollingmean_3\": 474.63178724391287, \"error1sum_rollingmean_24\": 0.0, \"volt_rollingmean_24\": 186.1033733765524, \"pressure_rollingmean_3\": 124.26190112949568, \"volt_rollingstd_24\": 0.7740120822459206, \"comp1sum\": 474.0, \"comp3sum\": 384.0, \"pressure_rollingmean_24\": 112.46729566613514, \"rotate_rollingstd_3\": 13.920898245623066, \"volt_rollingmean_3\": 188.406673928196, \"comp2sum\": 459.0, \"error2sum_rollingmean_24\": 0.0, \"rotate_rollingmean_24\": 461.1030486200735, \"vibration_rollingmean_3\": 38.869583185731614, \"vibration_rollingstd_3\": 1.9805973022526275, \"pressure_rollingstd_3\": 1.7895872952762106, \"error4sum_rollingmean_24\": 0.0, \"volt_rollingstd_3\": 4.60785082568852, \"vibration_rollingmean_24\": 39.96976455089771, \"comp4sum\": 579.0}, {\"rotate_rollingstd_24\": 2.017971138478601, \"error3sum_rollingmean_24\": 0.0, \"age\": 14, \"machineID\": 45, \"error5sum_rollingmean_24\": 0.0, \"pressure_rollingstd_24\": 0.2620300574897778, \"vibration_rollingstd_24\": 0.16523682934622702, \"rotate_rollingmean_3\": 454.8717742309143, \"error1sum_rollingmean_24\": 0.0, \"volt_rollingmean_24\": 184.4934951791266, \"pressure_rollingmean_3\": 123.02912082922734, \"volt_rollingstd_24\": 0.4103068092842077, \"comp1sum\": 473.6666666666667, \"comp3sum\": 383.6666666666667, \"pressure_rollingmean_24\": 110.24028050598271, \"rotate_rollingstd_3\": 15.91959183377542, \"volt_rollingmean_3\": 171.32900821497492, \"comp2sum\": 458.6666666666667, \"error2sum_rollingmean_24\": 0.0, \"rotate_rollingmean_24\": 458.14752146073414, \"vibration_rollingmean_3\": 37.71234613693027, \"vibration_rollingstd_3\": 2.3594190696788924, \"pressure_rollingstd_3\": 1.808640841551748, \"error4sum_rollingmean_24\": 0.0, \"volt_rollingstd_3\": 7.16544669362819, \"vibration_rollingmean_24\": 39.621269267841434, \"comp4sum\": 578.6666666666666}]}"
# ```
# 
# This submits 3 records to the model through the web service, and returns predictioned output labels for each of the three rows:
# ```
# "0.0,0.0,0.0"
# ```
# 
# Indicating that these records are not predicted to fail with in the requested time.
# 
# ## Conclusion
# 

# # Step 3: Model Building
# 
# Using the labeled feature data set constructed in the `Code/2_feature_engineering.ipynb` Jupyter notebook, this notebook loads the data from the Azure Blob container and splits it into a training and test data set. We then build two machine learning models, a decision tree classifier and a random forest classifier, to predict when different components within our machine population will fail. The two models are compared and we store the better performing model for deployment in an Azure web service. We will prepare and build the web service in the `Code/4_operationalization.ipynb` Jupyter notebook.
# 
# **Note:** This notebook will take about 2-4 minutes to execute all cells, depending on the compute configuration you have setup. 
# 

# import the libraries
import os
import glob
import time

# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import col
from pyspark.sql import SparkSession

# For some data handling
import pandas as pd
import numpy as np
# For Azure blob storage access
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess

# For logging model evaluation parameters back into the
# AML Workbench run history plots.
import logging
from azureml.logging import get_azureml_logger

amllog = logging.getLogger("azureml")
amllog.level = logging.INFO

# Turn on cell level logging.
get_ipython().run_line_magic('azureml', 'history on')
get_ipython().run_line_magic('azureml', 'history show')

# Time the notebook execution. 
# This will only make sense if you "Run all cells"
tic = time.time()

logger = get_azureml_logger() # logger writes to AMLWorkbench runtime view
spark = SparkSession.builder.getOrCreate()

# Telemetry
logger.log('amlrealworld.predictivemaintenance.feature_engineering','true')


# # Load feature data set
# 
# We have previously created the labeled feature data set in the `Code\2_feature_engineering.ipynb` Jupyter notebook. Since the Azure Blob storage account name and account key are not passed between notebooks, you'll need your credentials here again.
# 

# Enter your Azure blob storage details here 
ACCOUNT_NAME = "<your blob storage account name>"

# You can find the account key under the _Access Keys_ link in the 
# [Azure Portal](portal.azure.com) page for your Azure storage container.
ACCOUNT_KEY = "<your blob storage account key>"
#-------------------------------------------------------------------------------------------
# The data from the feature engineering note book is stored in the feature engineering container.
CONTAINER_NAME = CONTAINER_NAME = "featureengineering"

# Connect to your blob service     
az_blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)

# We will store and read each of these data sets in blob storage in an 
# Azure Storage Container on your Azure subscription.
# See https://github.com/Azure/ViennaDocs/blob/master/Documentation/UsingBlobForStorage.md
# for details.

# This is the final feature data file.
FEATURES_LOCAL_DIRECT = 'featureengineering_files.parquet'

# This is where we store the final model data file.
LOCAL_DIRECT = 'model_result.parquet'


# Load the data and dump a short summary of the resulting DataFrame.
# 

# load the previous created final dataset into the workspace
# create a local path where we store results
if not os.path.exists(FEATURES_LOCAL_DIRECT):
    os.makedirs(FEATURES_LOCAL_DIRECT)
    print('DONE creating a local directory!')

# download the entire parquet result folder to local path for a new run 
for blob in az_blob_service.list_blobs(CONTAINER_NAME):
    if FEATURES_LOCAL_DIRECT in blob.name:
        local_file = os.path.join(FEATURES_LOCAL_DIRECT, os.path.basename(blob.name))
        az_blob_service.get_blob_to_path(CONTAINER_NAME, blob.name, local_file)

feat_data = spark.read.parquet(FEATURES_LOCAL_DIRECT)
feat_data.limit(10).toPandas().head(10)


type(feat_data)


# # Prepare the Training/Testing data
# 

# When working with data that comes with time-stamps such as telemetry and errors as in this example, splitting of data into training, validation and test sets should be performed carefully to prevent overestimating the performance of the models. In predictive maintenance, the features are usually generated using laging aggregates and consecutive examples that fall into the same time window may have similar feature values in that window. If a random splitting of training and testing is used, it is possible for some portion of these similar examples that are in the same window to be selected for training and the other portion to leak into the testing data. Also, it is possible for training examples to be ahead of time than validation and testing examples when data is randomly split. However, predictive models should be trained on historical data and valiadted and tested on future data. Due to these problems, validation and testing based on random sampling may provide overly optimistic results. Since random sampling is not a viable approach here, cross validation methods that rely on random samples such as k-fold cross validation is not useful either.
# 
# For predictive maintenance problems, a time-dependent spliting strategy is often a better approach to estimate performance which is done by validating and testing on examples that are later in time than the training examples. For a time-dependent split, a point in time is picked and model is trained on examples up to that point in time, and validated on the examples after that point assuming that the future data after the splitting point is not known. However, this effects the labelling of features falling into the labelling window right before the split as it is assumed that failure information is not known beyond the splitting cut-off. Due to that, those feature records can not be labeled and will not be used. This also prevents the leaking problem at the splitting point.
# 
# Validation can be performed by picking different split points and examining the performance of the models trained on different time splits. In the following, we use a splitting points to train the model and look at the performances for the other split in the evaluation section.
# 

# define list of input columns for downstream modeling

# We'll use the known label, and key variables.
label_var = ['label_e']
key_cols =['machineID','dt_truncated']

# Then get the remaing feature names from the data
input_features = feat_data.columns

# We'll use the known label, key variables and 
# a few extra columns we won't need.
remove_names = label_var + key_cols + ['failure','model_encoded','model' ]

# Remove the extra names if that are in the input_features list
input_features = [x for x in input_features if x not in set(remove_names)]

input_features


# Spark models require a vectorized data frame. We transform the dataset here and then split the data into a training and test set. We use this split data to train the model on 9 months of data (training data), and evaluate on the remaining 3 months (test data) going forward.
# 

# assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')
feat_data = va.transform(feat_data).select('machineID','dt_truncated','label_e','features')

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(feat_data)

# fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol="label_e", outputCol="indexedLabel").fit(feat_data)

# split the data into train/test based on date
split_date = "2015-10-30"
training = feat_data.filter(feat_data.dt_truncated < split_date)
testing = feat_data.filter(feat_data.dt_truncated >= split_date)

print(training.count())
print(testing.count())


# # Classification models
# 
# In predictive maintenance, machine failures are usually rare occurrences in the lifetime of the assets compared to normal operation. This causes an imbalance in the label distribution which typically causes poor performance as algorithms tend to classify majority class examples better at the expense of minority class examples as the total misclassification error is much improved when majority class is labeled correctly. This causes low recall rates although accuracy can be high and becomes a larger problem when the cost of false alarms to the business is very high. To help with this problem, sampling techniques such as oversampling of the minority examples are usually used along with more sophisticated techniques which are not covered in this notebook.
# 
# Also, due to the class imbalance problem, it is important to look at evaluation metrics other than accuracy alone and compare those metrics to the baseline metrics which are computed when random chance is used to make predictions rather than a machine learning model. The comparison will bring out the value and benefits of using a machine learning model better.
# 
# We will build and compare two different classification model approaches:
# 
#  - **Decision Tree Classifier**: Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression. Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.
# 
#  - **Random Forest Classifier**: A random forest is an ensemble of decision trees. Random forests combine many decision trees in order to reduce the risk of overfitting. Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks.
# 
# Remember, we build the model by training on the training data set, then evaluate the model using the testing data set.
# 
# ### Experimentation
# 
# We want to compare these models in the AML Workbench _runs_ screen. The next cell, creates the model. You can choose between a Decision tree or random forest by setting the 'model_type' variable. 
# 
# The next three
# 

model_type = 'RandomForest' # Use 'DecisionTree', or 'RandomForest'

# train a model.
if model_type == 'DecisionTree':
    model = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                                      # Maximum depth of the tree. (>= 0) 
                                      # E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'
                                      maxDepth=15,
                                      # Max number of bins for discretizing continuous features. 
                                      # Must be >=2 and >= number of categories for any categorical feature.
                                      maxBins=32, 
                                      # Minimum number of instances each child must have after split. 
                                      # If a split causes the left or right child to have fewer than 
                                      # minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.
                                      minInstancesPerNode=1, 
                                      # Minimum information gain for a split to be considered at a tree node.
                                      minInfoGain=0.0, 
                                      # Criterion used for information gain calculation (case-insensitive). 
                                      # Supported options: entropy, gini')
                                      impurity="gini")

    ##=======================================================================================================================
    #elif model_type == 'GBTClassifier':
    #    cls_mthd = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    ##=======================================================================================================================
else:    
    model = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", 
                                      # Passed to DecisionTreeClassifier
                                      maxDepth=15, 
                                      maxBins=32, 
                                      minInstancesPerNode=1, 
                                      minInfoGain=0.0,
                                      impurity="gini",
                                      # Number of trees to train (>= 1)
                                      numTrees=50, 
                                      # The number of features to consider for splits at each tree node. 
                                      # Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n].
                                      featureSubsetStrategy="sqrt", 
                                      # Fraction of the training data used for learning each 
                                      # decision tree, in range (0, 1].' 
                                      subsamplingRate = 0.632)

# chain indexers and model in a Pipeline
pipeline_cls_mthd = Pipeline(stages=[labelIndexer, featureIndexer, model])

# train model.  This also runs the indexers.
model_pipeline = pipeline_cls_mthd.fit(training)


# To evaluate this model, we predict the component failures over the test data set. The standard method of viewing this evaluation is with a _confusion matrix_ shown below.
# 

# make predictions. The Pipeline does all the same operations on the test data
predictions = model_pipeline.transform(testing)

# Create the confusion matrix for the multiclass prediction results
# This result assumes a decision boundary of p = 0.5
conf_table = predictions.stat.crosstab('indexedLabel', 'prediction')
confuse = conf_table.toPandas()
confuse.head()


# The confusion matrix lists each true component failure in rows and the predicted value in columns. Labels numbered 0.0 corresponds to no component failures. Labels numbered 1.0 through 4.0 correspond to failures in one of the four components in the machine. As an example, the third number in the top row indicates how many days we predicted component 2 would fail, when no components actually did fail. The second number in the second row, indicates how many days we correctly predicted a component 1 failure within the next 7 days.
# 
# We read the confusion matrix numbers along the diagonal as correctly classifying the component failures. Numbers above the diagonal indicate the model incorrectly predicting a failure when non occured, and those below indicate incorrectly predicting a non-failure for the row indicated component failure.
# 
# When evaluating classification models, it is convenient to reduce the results in the confusion matrix into a single performance statistic. However, depending on the problem space, it is impossible to always use the same statistic in this evaluation. Below, we calculate four such statistics.
# 
# - **Accuracy**: reports how often we correctly predicted the labeled data. Unfortunatly, when there is a class imbalance (a large number of one of the labels relative to others), this measure is biased towards the largest class. In this case non-failure days.
# 
# Because of the class imbalance inherint in predictive maintenance problems, it is better to look at the remaining statistics instead. Here positive predictions indicate a failure.
# 
# - **Precision**: Precision is a measure of how well the model classifies the truely positive samples. Precision depends on falsely classifying negative days as positive.
# 
# - **Recall**: Recall is a measure of how well the model can find the positive samples. Recall depends on falsely classifying positive days as negative.
# 
# - **F1**: F1 considers both the precision and the recall. F1 score is the harmonic average of precision and recall. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# 
# These metrics make the most sense for binary classifiers, though they are still useful for comparision in our multiclass setting. Below we calculate these evaluation statistics for the selected classifier, and post them back to the AML workbench run time page for tracking between experiments.
# 

# select (prediction, true label) and compute test error
# select (prediction, true label) and compute test error
# True positives - diagonal failure terms 
tp = confuse['1.0'][1]+confuse['2.0'][2]+confuse['3.0'][3]+confuse['4.0'][4]

# False positves - All failure terms - True positives
fp = np.sum(np.sum(confuse[['1.0', '2.0','3.0','4.0']])) - tp

# True negatives 
tn = confuse['0.0'][0]

# False negatives total of non-failure column - TN
fn = np.sum(np.sum(confuse[['0.0']])) - tn

# Accuracy is diagonal/total 
acc_n = tn + tp
acc_d = np.sum(np.sum(confuse[['0.0','1.0', '2.0','3.0','4.0']]))
acc = acc_n/acc_d

# Calculate precision and recall.
prec = tp/(tp+fp)
rec = tp/(tp+fn)

# Print the evaluation metrics to the notebook
print("Accuracy = %g" % acc)
print("Precision = %g" % prec)
print("Recall = %g" % rec )
print("F1 = %g" % (2.0 * prec * rec/(prec + rec)))
print("")

# logger writes information back into the AML Workbench run time page.
# Each title (i.e. "Model Accuracy") can be shown as a graph to track
# how the metric changes between runs.
logger.log("Model Accuracy", (acc))
logger.log("Model Precision", (prec))
logger.log("Model Recall", (rec))
logger.log("Model F1", (2.0 * prec * rec/(prec + rec)))


importances = model_pipeline.stages[2].featureImportances

importances


# Remember that this is a simulated data set. We would expect a model built on real world data to behave very differently. The accuracy may still be close to one, but the precision and recall numbers would be much lower.
# 
# ## Persist the model
# 
# We'll save the latest model for use in deploying a webservice for operationalization in the next notebook. We store this local to the Jupyter notebook kernel because the model is stored in a hierarchical format that does not translate to Azure Blob storage well. 
# 

# save model
model_pipeline.write().overwrite().save(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']+'pdmrfull.model')
print("Model saved")

# Time the notebook execution. 
# This will only make sense if you "Run All" cells
toc = time.time()
print("Full run took %.2f minutes" % ((toc - tic)/60))
logger.log("Model Building Run time", ((toc - tic)/60))


# ## Conclusion
# 
# In the next notebook `Code\4_operationalization.ipynb` Jupyter notebook we will create the functions needed to operationalize and deploy any model to get realtime predictions. 
# 

