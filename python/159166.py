# # Copying data from Redshift to S3 and back
# 
# ---
# 
# ---
# ## Contents
# 
# 1. [Introduction](#Introduction)
# 1. [Reading from Redshift](#Reading-from-Redshift)
# 1. [Upload to S3](#Upload-to-S3)
# 1. [Writing back to Redshift](#Writing-back-to-Redshift)
# 
# 
# 
# ## Introduction
# In this notebook we illustrate how to copy data from Redshift to S3 and vice-versa.
# 
# ### Prerequisites
# In order to successfully run this notebook, you'll first need to:
# 1. Have a Redshift cluster within the same VPC.
# 1. Preload that cluster with data from the [iris data set](https://archive.ics.uci.edu/ml/datasets/iris) in a table named public.irisdata.
# 1. Update the credential file (`redshift_creds_template.json.nogit`) file with the appropriate information.
# 
# ### Notebook Setup
# Let's start by installing `psycopg2`, a PostgreSQL database adapter for the Python, adding a few imports and specifying a few configs. 
# 

get_ipython().system('conda install -y -c anaconda psycopg2')


import os
import boto3
import pandas as pd
import json
import psycopg2
import sqlalchemy as sa

region = boto3.Session().region_name

bucket='<your_s3_bucket_name_here>' # put your s3 bucket name here, and create s3 bucket
prefix = 'sagemaker/redshift'
# customize to your bucket where you have stored the data

credfile = 'redshift_creds_template.json.nogit'


# ## Reading from Redshift
# We store the information needed to connect to Redshift in a credentials file. See the file `redshift_creds_template.json.nogit` for an example. 
# 

# Read credentials to a dictionary
with open(credfile) as fh:
    creds = json.loads(fh.read())

# Sample query for testing
query = 'select * from public.irisdata;'


# We create a connection to redshift using our credentials, and use this to query Redshift and store the result in a pandas DataFrame, which we then save.
# 

print("Reading from Redshift...")

def get_conn(creds): 
    conn = psycopg2.connect(dbname=creds['db_name'], 
                            user=creds['username'], 
                            password=creds['password'],
                            port=creds['port_num'],
                            host=creds['host_name'])
    return conn

def get_df(creds, query):
    with get_conn(creds) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result_set = cur.fetchall()
            colnames = [desc.name for desc in cur.description]
            df = pd.DataFrame.from_records(result_set, columns=colnames)
    return df

df = get_df(creds, query)

print("Saving file")
localFile = 'iris.csv'
df.to_csv(localFile, index=False)

print("Done")


# ## Upload to S3
# 

print("Writing to S3...")

fObj = open(localFile, 'rb')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, localFile)).upload_fileobj(fObj)
print("Done")


# ## Writing back to Redshift
# 
# We now demonstrate the reverse process of copying data from S3 to Redshift. We copy back the same data but in an actual application the data would be the output of an algorithm on Sagemaker.
# 

print("Reading from S3...")
# key unchanged for demo purposes - change key to read from output data
key = os.path.join(prefix, localFile)

s3 = boto3.resource('s3')
outfile = 'iris2.csv'
s3.Bucket(bucket).download_file(key, outfile)
df2 = pd.read_csv(outfile)
print("Done")


print("Writing to Redshift...")

connection_str = 'postgresql+psycopg2://' +                   creds['username'] + ':' +                   creds['password'] + '@' +                   creds['host_name'] + ':' +                   creds['port_num'] + '/' +                   creds['db_name'];
                    
df2.to_sql('irisdata_v2', connection_str, schema='public', index=False)
print("Done")


# We read the copied data in Redshift - success!
# 

pd.options.display.max_rows = 2
conn = get_conn(creds)
query = 'select * from irisdata3'
df = pd.read_sql_query(query, conn)
df


# # Regression with Amazon SageMaker XGBoost algorithm
# _**Single machine training for regression with Amazon SageMaker XGBoost algorithm**_
# 
# ---
# 
# ---
# ## Contents
# 1. [Introduction](#Introduction)
# 2. [Setup](#Setup)
#   1. [Fetching the dataset](#Fetching-the-dataset)
#   2. [Data Ingestion](#Data-ingestion)
# 3. [Training the XGBoost model](#Training-the-XGBoost-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#   1. [Import model into hosting](#Import-model-into-hosting)
#   2. [Create endpoint configuration](#Create-endpoint-configuration)
#   3. [Create endpoint](#Create-endpoint)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 
# ---
# ## Introduction
# 
# This notebook demonstrates the use of Amazon SageMaker’s implementation of the XGBoost algorithm to train and host a regression model. We use the [Abalone data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html) originally from the [UCI data repository](https://archive.ics.uci.edu/ml/datasets/abalone). More details about the original dataset can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names).  In the libsvm converted [version](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html), the nominal feature (Male/Female/Infant) has been converted into a real valued feature. Age of abalone is to be predicted from eight physical measurements.  
# 
# ---
# ## Setup
# 
# 
# This notebook was created and tested on an ml.m4.4xlarge notebook instance.
# 
# Let's start by specifying:
# 1. The S3 bucket and prefix that you want to use for training and model data. This should be within the same region as the Notebook Instance, training, and hosting.
# 1. The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these. Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).
# 

get_ipython().run_cell_magic('time', '', "\nimport os\nimport boto3\nimport re\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\nregion = boto3.Session().region_name\n\nbucket='<bucket-name>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/xgboost-regression'\n# customize to your bucket where you have stored the data\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)")


# ### Fetching the dataset
# 
# Following methods split the data into train/test/validation datasets and upload files to S3.
# 

get_ipython().run_cell_magic('time', '', "\nimport io\nimport boto3\nimport random\n\ndef data_split(FILE_DATA, FILE_TRAIN, FILE_VALIDATION, FILE_TEST, PERCENT_TRAIN, PERCENT_VALIDATION, PERCENT_TEST):\n    data = [l for l in open(FILE_DATA, 'r')]\n    train_file = open(FILE_TRAIN, 'w')\n    valid_file = open(FILE_VALIDATION, 'w')\n    tests_file = open(FILE_TEST, 'w')\n\n    num_of_data = len(data)\n    num_train = int((PERCENT_TRAIN/100.0)*num_of_data)\n    num_valid = int((PERCENT_VALIDATION/100.0)*num_of_data)\n    num_tests = int((PERCENT_TEST/100.0)*num_of_data)\n\n    data_fractions = [num_train, num_valid, num_tests]\n    split_data = [[],[],[]]\n\n    rand_data_ind = 0\n\n    for split_ind, fraction in enumerate(data_fractions):\n        for i in range(fraction):\n            rand_data_ind = random.randint(0, len(data)-1)\n            split_data[split_ind].append(data[rand_data_ind])\n            data.pop(rand_data_ind)\n\n    for l in split_data[0]:\n        train_file.write(l)\n\n    for l in split_data[1]:\n        valid_file.write(l)\n\n    for l in split_data[2]:\n        tests_file.write(l)\n\n    train_file.close()\n    valid_file.close()\n    tests_file.close()\n\ndef write_to_s3(fobj, bucket, key):\n    return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fobj)\n\ndef upload_to_s3(bucket, channel, filename):\n    fobj=open(filename, 'rb')\n    key = prefix+'/'+channel\n    url = 's3://{}/{}/{}'.format(bucket, key, filename)\n    print('Writing to {}'.format(url))\n    write_to_s3(fobj, bucket, key)")


# ### Data ingestion
# 
# Next, we read the dataset from the existing repository into memory, for preprocessing prior to training. This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.
# 

get_ipython().run_cell_magic('time', '', 'import urllib.request\n\n# Load the dataset\nFILE_DATA = \'abalone\'\nurllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone", FILE_DATA)\n\n#split the downloaded data into train/test/validation files\nFILE_TRAIN = \'abalone.train\'\nFILE_VALIDATION = \'abalone.validation\'\nFILE_TEST = \'abalone.test\'\nPERCENT_TRAIN = 70\nPERCENT_VALIDATION = 15\nPERCENT_TEST = 15\ndata_split(FILE_DATA, FILE_TRAIN, FILE_VALIDATION, FILE_TEST, PERCENT_TRAIN, PERCENT_VALIDATION, PERCENT_TEST)\n\n#upload the files to the S3 bucket\nupload_to_s3(bucket, \'train\', FILE_TRAIN)\nupload_to_s3(bucket, \'validation\', FILE_VALIDATION)\nupload_to_s3(bucket, \'test\', FILE_TEST)')


# ## Training the XGBoost model
# 
# After setting training parameters, we kick off training, and poll for status until training is completed, which in this example, takes between 5 and 6 minutes.
# 

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]


get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\njob_name = \'xgboost-single-machine-regression-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\n#Ensure that the training and validation data folders generated above are reflected in the "InputDataConfig" parameter below.\n\ncreate_training_params = \\\n{\n    "AlgorithmSpecification": {\n        "TrainingImage": container,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": bucket_path + "/" + prefix + "/single-xgboost"\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.m4.4xlarge",\n        "VolumeSizeInGB": 5\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "max_depth":"5",\n        "eta":"0.2",\n        "gamma":"4",\n        "min_child_weight":"6",\n        "subsample":"0.7",\n        "silent":"0",\n        "objective":"reg:linear",\n        "num_round":"50"\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 3600\n    },\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/" + prefix + \'/train\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "libsvm",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/" + prefix + \'/validation\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "libsvm",\n            "CompressionType": "None"\n        }\n    ]\n}\n\n\nclient = boto3.client(\'sagemaker\')\nclient.create_training_job(**create_training_params)\n\nimport time\n\nstatus = client.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\nprint(status)\nwhile status !=\'Completed\' and status!=\'Failed\':\n    time.sleep(60)\n    status = client.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\n    print(status)')


# ## Set up hosting for the model
# In order to set up hosting, we have to import the model from training to hosting. 
# 
# ### Import model into hosting
# 
# Register the model with hosting. This allows the flexibility of importing models trained elsewhere.
# 

get_ipython().run_cell_magic('time', '', "import boto3\nfrom time import gmtime, strftime\n\nmodel_name=job_name + '-model'\nprint(model_name)\n\ninfo = client.describe_training_job(TrainingJobName=job_name)\nmodel_data = info['ModelArtifacts']['S3ModelArtifacts']\nprint(model_data)\n\nprimary_container = {\n    'Image': container,\n    'ModelDataUrl': model_data\n}\n\ncreate_model_response = client.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response['ModelArn'])")


# ### Create endpoint configuration
# 
# SageMaker supports configuring REST endpoints in hosting with multiple models, e.g. for A/B testing purposes. In order to support this, customers create an endpoint configuration, that describes the distribution of traffic across the models, whether split, shadowed, or sampled in some way. In addition, the endpoint configuration describes the instance type required for model deployment.
# 

from time import gmtime, strftime

endpoint_config_name = 'XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.m4.xlarge',
        'InitialVariantWeight':1,
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# ### Create endpoint
# Lastly, the customer creates the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications. This takes 9-11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = client.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = client.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nwhile status==\'Creating\':\n    time.sleep(60)\n    resp = client.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Status: " + status)\n\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)')


# ## Validate the model for use
# Finally, the customer can now validate the model for use. They can obtain the endpoint from the client library using the result from previous operations, and generate classifications from the trained model using that endpoint.
# 

runtime_client = boto3.client('runtime.sagemaker')


# Start with a single prediction.
# 

get_ipython().system('head -1 abalone.test > abalone.single.test')


get_ipython().run_cell_magic('time', '', 'import json\nfrom itertools import islice\nimport math\nimport struct\n\nfile_name = \'abalone.single.test\' #customize to your test file\nwith open(file_name, \'r\') as f:\n    payload = f.read().strip()\nresponse = runtime_client.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType=\'text/x-libsvm\', \n                                   Body=payload)\nresult = response[\'Body\'].read()\nresult = result.decode("utf-8")\nresult = result.split(\',\')\nresult = [math.ceil(float(i)) for i in result]\nlabel = payload.strip(\' \')[0]\nprint (\'Label: \',label,\'\\nPrediction: \', result[0])')


# OK, a single prediction works. Let's do a whole batch to see how good is the predictions accuracy.
# 

import sys
import math
def do_predict(data, endpoint_name, content_type):
    payload = '\n'.join(data)
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType=content_type, 
                                   Body=payload)
    result = response['Body'].read()
    result = result.decode("utf-8")
    result = result.split(',')
    preds = [float((num)) for num in result]
    preds = [math.ceil(num) for num in preds]
    return preds

def batch_predict(data, batch_size, endpoint_name, content_type):
    items = len(data)
    arrs = []
    
    for offset in range(0, items, batch_size):
        if offset+batch_size < items:
            results = do_predict(data[offset:(offset+batch_size)], endpoint_name, content_type)
            arrs.extend(results)
        else:
            arrs.extend(do_predict(data[offset:items], endpoint_name, content_type))
        sys.stdout.write('.')
    return(arrs)


# The following helps us calculate the Median Absolute Percent Error (MdAPE) on the batch dataset. 
# 

get_ipython().run_cell_magic('time', '', "import json\nimport numpy as np\n\nwith open(FILE_TEST, 'r') as f:\n    payload = f.read().strip()\n\nlabels = [int(line.split(' ')[0]) for line in payload.split('\\n')]\ntest_data = [line for line in payload.split('\\n')]\npreds = batch_predict(test_data, 100, endpoint_name, 'text/x-libsvm')\n\nprint('\\n Median Absolute Percent Error (MdAPE) = ', np.median(np.abs(np.array(labels) - np.array(preds)) / np.array(labels)))")


# ### Delete Endpoint
# Once you are done using the endpoint, you can use the following to delete it. 
# 

client.delete_endpoint(EndpointName=endpoint_name)


# # End-to-End Multiclass Image Classification Example
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
# 3. [Training the ResNet model](#Training-the-ResNet-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#   1. [Import model into hosting](#Import-model-into-hosting)
#   2. [Create endpoint configuration](#Create-endpoint-configuration)
#   3. [Create endpoint](#Create-endpoint)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 

# ## Introduction
# 
# Welcome to our end-to-end example of distributed image classification algorithm. In this demo, we will use the Amazon sagemaker image classification algorithm to train on the [caltech-256 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/). 
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# Here we set up the linkage and authentication to AWS services. There are three parts to this:
# 
# * The roles used to give learning and hosting access to your data. This will automatically be obtained from the role used to start the notebook
# * The S3 bucket that you want to use for training and model data
# * The Amazon sagemaker image classification docker image which need not be changed
# 

get_ipython().run_cell_magic('time', '', "import boto3\nimport re\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nbucket='<<bucket-name>>' # customize to your bucket\ncontainers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest',\n              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',\n              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest',\n              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest'}\ntraining_image = containers[boto3.Session().region_name]")


# ### Data preparation
# Download the data and transfer to S3 for use in training.
# 

import os 
import urllib.request
import boto3

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

        
def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)


# caltech-256
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
upload_to_s3('train', 'caltech-256-60-train.rec')
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
upload_to_s3('validation', 'caltech-256-60-val.rec')


# ## Training the ResNet model
# 
# In this demo, we are using [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) dataset, which contains 30608 images of 256 objects. For the training and validation data, we follow the splitting scheme in this MXNet [example](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/data/caltech256.sh). In particular, it randomly selects 60 images per class for training, and uses the remaining data for validation. The algorithm takes `RecordIO` file as input. The user can also provide the image files as input, which will be converted into `RecordIO` format using MXNet's [im2rec](https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=im2rec) tool. It takes around 50 seconds to converted the entire Caltech-256 dataset (~1.2GB) on a p2.xlarge instance. However, for this demo, we will use record io format. 
# 
# Once we have the data available in the correct format for training, the next step is to actually train the model using the data. After setting training parameters, we kick off training, and poll for status until training is completed.
# 
# ## Training parameters
# There are two kinds of parameters that need to be set for training. The first one are the parameters for the training job. These include:
# 
# * **Input specification**: These are the training and validation channels that specify the path where training data is present. These are specified in the "InputDataConfig" section. The main parameters that need to be set is the "ContentType" which can be set to "rec" or "lst" based on the input data format and the S3Uri which specifies the bucket and the folder where the data is present. 
# * **Output specification**: This is specified in the "OutputDataConfig" section. We just need to specify the path where the output can be stored after training
# * **Resource config**: This section specifies the type of instance on which to run the training and the number of hosts used for training. If "InstanceCount" is more than 1, then training can be run in a distributed manner. 
# 
# Apart from the above set of parameters, there are hyperparameters that are specific to the algorithm. These are:
# 
# * **num_layers**: The number of layers (depth) for the network. We use 101 in this samples but other values such as 50, 152 can be used. 
# * **num_training_samples**: This is the total number of training samples. It is set to 15420 for caltech dataset with the current split
# * **num_classes**: This is the number of output classes for the new dataset. Imagenet was trained with 1000 output classes but the number of output classes can be changed for fine-tuning. For caltech, we use 257 because it has 256 object categories + 1 clutter class
# * **epochs**: Number of training epochs
# * **learning_rate**: Learning rate for training
# * **mini_batch_size**: The number of training samples used for each mini batch. In distributed training, the number of training samples used per batch will be N * mini_batch_size where N is the number of hosts on which training is run
# 

# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 18 layers
num_layers = "18" 
# we need to specify the input image shape for the training data
image_shape = "3,224,224"
# we also need to specify the number of training samples in the training set
# for caltech it is 15420
num_training_samples = "15420"
# specify the number of output classes
num_classes = "257"
# batch size for training
mini_batch_size =  "64"
# number of epochs
epochs = "2"
# learning rate
learning_rate = "0.01"


# # Training
# Run the training using Amazon sagemaker CreateTrainingJob API
# 

get_ipython().run_cell_magic('time', '', 'import time\nimport boto3\nfrom time import gmtime, strftime\n\n\ns3 = boto3.client(\'s3\')\n# create unique job name \njob_name_prefix = \'sagemaker-imageclassification-notebook\'\ntimestamp = time.strftime(\'-%Y-%m-%d-%H-%M-%S\', time.gmtime())\njob_name = job_name_prefix + timestamp\ntraining_params = \\\n{\n    # specify the training docker image\n    "AlgorithmSpecification": {\n        "TrainingImage": training_image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": \'s3://{}/{}/output\'.format(bucket, job_name_prefix)\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.p2.xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "image_shape": image_shape,\n        "num_layers": str(num_layers),\n        "num_training_samples": str(num_training_samples),\n        "num_classes": str(num_classes),\n        "mini_batch_size": str(mini_batch_size),\n        "epochs": str(epochs),\n        "learning_rate": str(learning_rate)\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 360000\n    },\n#Training data should be inside a subdirectory called "train"\n#Validation data should be inside a subdirectory called "validation"\n#The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/train/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/validation/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        }\n    ]\n}\nprint(\'Training job name: {}\'.format(job_name))\nprint(\'\\nInput Data Location: {}\'.format(training_params[\'InputDataConfig\'][0][\'DataSource\'][\'S3DataSource\']))')


# create the Amazon SageMaker training job
sagemaker = boto3.client(service_name='sagemaker')
sagemaker.create_training_job(**training_params)

# confirm that the training job has started
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))


training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
status = training_info['TrainingJobStatus']
print("Training job ended with status: " + status)


# If you see the message,
# 
# > `Training job ended with status: Completed`
# 
# then that means training sucessfully completed and the output model was stored in the output path specified by `training_params['OutputDataConfig']`.
# 
# You can also view information about and the status of a training job using the AWS SageMaker console. Just click on the "Jobs" tab.
# 

# # Inference
# 
# ***
# 
# A trained model does nothing on its own. We now want to use the model to perform inference. For this example, that means predicting the topic mixture representing a given document.
# 
# This section involves several steps,
# 
# 1. [Create Model](#CreateModel) - Create model for the training output
# 1. [Create Endpoint Configuration](#CreateEndpointConfiguration) - Create a configuration defining an endpoint.
# 1. [Create Endpoint](#CreateEndpoint) - Use the configuration to create an inference endpoint.
# 1. [Perform Inference](#Perform Inference) - Perform inference on some input data using the endpoint.
# 

# ## Create Model
# 
# We now create a SageMaker Model from the training output. Using the model we can create an Endpoint Configuration.
# 

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nsage = boto3.Session().client(service_name=\'sagemaker\') \n\nmodel_name="test-image-classification-model"\nprint(model_name)\ninfo = sage.describe_training_job(TrainingJobName=job_name)\nmodel_data = info[\'ModelArtifacts\'][\'S3ModelArtifacts\']\nprint(model_data)\n\ncontainers = {\'us-west-2\': \'433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest\',\n              \'us-east-1\': \'811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest\',\n              \'us-east-2\': \'825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest\',\n              \'eu-west-1\': \'685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest\'}\nhosting_image = containers[boto3.Session().region_name]\nprimary_container = {\n    \'Image\': hosting_image,\n    \'ModelDataUrl\': model_data,\n}\n\ncreate_model_response = sage.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response[\'ModelArn\'])')


# ### Create Endpoint Configuration
# At launch, we will support configuring REST endpoints in hosting with multiple models, e.g. for A/B testing purposes. In order to support this, customers create an endpoint configuration, that describes the distribution of traffic across the models, whether split, shadowed, or sampled in some way.
# 
# In addition, the endpoint configuration describes the instance type required for model deployment, and at launch will describe the autoscaling configuration.
# 

from time import gmtime, strftime

timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
endpoint_config_name = job_name_prefix + '-epc-' + timestamp
endpoint_config_response = sage.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.p2.xlarge',
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print('Endpoint configuration name: {}'.format(endpoint_config_name))
print('Endpoint configuration arn:  {}'.format(endpoint_config_response['EndpointConfigArn']))


# ### Create Endpoint
# Lastly, the customer creates the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications. This takes 9-11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', "import time\n\ntimestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())\nendpoint_name = job_name_prefix + '-ep-' + timestamp\nprint('Endpoint name: {}'.format(endpoint_name))\n\nendpoint_params = {\n    'EndpointName': endpoint_name,\n    'EndpointConfigName': endpoint_config_name,\n}\nendpoint_response = sagemaker.create_endpoint(**endpoint_params)\nprint('EndpointArn = {}'.format(endpoint_response['EndpointArn']))")


# Finally, now the endpoint can be created. It may take sometime to create the endpoint...
# 

# get the status of the endpoint
response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = response['EndpointStatus']
print('EndpointStatus = {}'.format(status))


# wait until the status has changed
sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)


# print the status of the endpoint
endpoint_response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = endpoint_response['EndpointStatus']
print('Endpoint creation ended with EndpointStatus = {}'.format(status))

if status != 'InService':
    raise Exception('Endpoint creation failed.')


# If you see the message,
# 
# > `Endpoint creation ended with EndpointStatus = InService`
# 
# then congratulations! You now have a functioning inference endpoint. You can confirm the endpoint configuration and status by navigating to the "Endpoints" tab in the AWS SageMaker console.
# 
# We will finally create a runtime object from which we can invoke the endpoint.
# 

# ## Perform Inference
# Finally, the customer can now validate the model for use. They can obtain the endpoint from the client library using the result from previous operations, and generate classifications from the trained model using that endpoint.
# 

import boto3
runtime = boto3.Session().client(service_name='runtime.sagemaker') 


# ### Download test image
# 

get_ipython().system('wget -O /tmp/test.jpg http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/008_0007.jpg')
file_name = '/tmp/test.jpg'
# test image
from IPython.display import Image
Image(file_name)  


# ### Evaluation
# 
# Evaluate the image through the network for inteference. The network outputs class probabilities and typically, one selects the class with the maximum probability as the final class output.
# 
# **Note:** The output class detected by the network may not be accurate in this example. To limit the time taken and cost of training, we have trained the model only for a couple of epochs. If the network is trained for more epochs (say 20), then the output class will be more accurate.
# 

import json
import numpy as np

with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-image', 
                                   Body=payload)
result = response['Body'].read()
# result will be in json format and convert it to ndarray
result = json.loads(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))


# ### Clean up
# 
# When we're done with the endpoint, we can just delete it and the backing instances will be released.  Run the following cell to delete the endpoint.
# 

sage.delete_endpoint(EndpointName=endpoint_name)





# # Introduction to Basic Functionality of NTM
# _**Finding Topics in Synthetic Document Data with the Neural Topic Model**_
# 
# ---
# 
# ---
# 
# # Contents
# ***
# 
# 1. [Introduction](#Introduction)
# 1. [Setup](#Setup)
# 1. [Data](#Data)
# 1. [Train](#Train)
# 1. [Host](#Host)
# 1. [Extensions](#Extensions)
# 

# # Introduction
# ***
# 
# Amazon SageMaker NTM (Neural Topic Model) is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. NTM is most commonly used to discover a user-specified number of topics shared by documents within a text corpus. Here each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics. Since the method is unsupervised, the topics are not specified up front, and are not guaranteed to align with how a human may naturally categorize documents. The topics are learned as a probability distribution over the words that occur in each document. Each document, in turn, is described as a mixture of topics.
# 
# In this notebook we will use the Amazon SageMaker NTM algorithm to train a model on some example synthetic data. We will then use this model to classify (perform inference on) the data. The main goals of this notebook are to,
# 
# * learn how to obtain and store data for use in Amazon SageMaker,
# * create an AWS SageMaker training job on a data set to produce a NTM model,
# * use the model to perform inference with an Amazon SageMaker endpoint.
# 

# # Setup
# ***
# 
# _This notebook was created and tested on an ml.m4xlarge notebook instance._
# 
# Let's start by specifying:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.  Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).
# 

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/ntm_synthetic'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# Next we'll import the libraries we'll need throughout the remainder of the notebook.
# 

import numpy as np
from generate_example_data import generate_griffiths_data, plot_topic_data
import io
import os
import time
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import scipy
import sagemaker
import sagemaker.amazon.common as smac
from sagemaker.predictor import csv_serializer, json_deserializer


# # Data
# ***
# 
# We generate some example synthetic document data. For the purposes of this notebook we will omit the details of this process. All we need to know is that each piece of data, commonly called a "document", is a vector of integers representing "word counts" within the document. In this particular example there are a total of 25 words in the "vocabulary".
# 

# generate the sample data
num_documents = 5000
num_topics = 5
vocabulary_size = 25
known_alpha, known_beta, documents, topic_mixtures = generate_griffiths_data(
    num_documents=num_documents, num_topics=num_topics, vocabulary_size=vocabulary_size)

# separate the generated data into training and tests subsets
num_documents_training = int(0.8*num_documents)
num_documents_test = num_documents - num_documents_training

documents_training = documents[:num_documents_training]
documents_test = documents[num_documents_training:]

topic_mixtures_training = topic_mixtures[:num_documents_training]
topic_mixtures_test = topic_mixtures[num_documents_training:]

data_training = (documents_training, np.zeros(num_documents_training))
data_test = (documents_test, np.zeros(num_documents_test))


# ## Inspect Example Data
# 
# *What does the example data actually look like?* Below we print an example document as well as its corresponding *known* topic mixture. Later, when we perform inference on the training data set we will compare the inferred topic mixture to this known one.
# 
# As we can see, each document is a vector of word counts from the 25-word vocabulary
# 

print('First training document = {}'.format(documents[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))


np.set_printoptions(precision=4, suppress=True)

print('Known topic mixture of first training document = {}'.format(topic_mixtures_training[0]))
print('\nNumber of topics = {}'.format(num_topics))


# Because we are visual creatures, let's try plotting the documents. In the below plots, each pixel of a document represents a word. The greyscale intensity is a measure of how frequently that word occurs. Below we plot the first tes documents of the training set reshaped into 5x5 pixel grids.
# 

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_topic_data(documents_training[:10], nrows=2, ncols=5, cmap='gray_r', with_colorbar=False)
fig.suptitle('Example Documents')
fig.set_dpi(160)


# ## Store Data on S3
# 
# A SageMaker training job needs access to training data stored in an S3 bucket. Although training can accept data of various formats recordIO wrapped protobuf is most performant.
# 
# _Note, since NTM is an unsupervised learning algorithm, we simple put 0 in for all label values._
# 

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, data_training[0].astype('float32'))
buf.seek(0)

key = 'ntm.data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)


# # Training
# 
# ***
# 
# Once the data is preprocessed and available in a recommended format the next step is to train our model on the data. There are number of parameters required by the NTM algorithm to configure the model and define the computational environment in which training will take place.  The first of these is to point to a container image which holds the algorithms training and hosting code.
# 

containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/ntm:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/ntm:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/ntm:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/ntm:latest'}


# An NTM model uses the following hyperparameters:
# 
# * **`num_topics`** - The number of topics or categories in the NTM model. This has been pre-defined in our synthetic data to be 5.
# 
# * **`feature_dim`** - The size of the *"vocabulary"*, in topic modeling parlance. In this case, this has been set to 25 by `generate_griffiths_data()`.
# 
# In addition to these NTM model hyperparameters, we provide additional parameters defining things like the EC2 instance type on which training will run, the S3 bucket containing the data, and the AWS access role.
# 

sess = sagemaker.Session()

ntm = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.c4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
ntm.set_hyperparameters(num_topics=num_topics,
                        feature_dim=vocabulary_size)

ntm.fit({'train': s3_train_data})


# # Inference
# 
# ***
# 
# A trained model does nothing on its own. We now want to use the model to perform inference. For this example, that means predicting the topic mixture representing a given document.
# 
# This is simplified by the deploy function provided by the Amazon SageMaker Python SDK.
# 

ntm_predictor = ntm.deploy(initial_instance_count=1,
                           instance_type='ml.c4.xlarge')


# ## Perform Inference
# 
# With this real-time endpoint at our fingertips we can finally perform inference on our training and test data.  We should first discuss the meaning of the SageMaker NTM inference output.
# 
# For each document we wish to compute its corresponding `topic_weights`. Each set of topic weights is a probability distribution over the number of topics, which is 5 in this example. Of the 5 topics discovered during NTM training each element of the topic weights is the proportion to which the input document is represented by the corresponding topic.
# 
# For example, if the topic weights of an input document $\mathbf{w}$ is,
# 
# $$\theta = \left[ 0.3, 0.2, 0, 0.5, 0 \right]$$
# 
# then $\mathbf{w}$ is 30% generated from Topic #1, 20% from Topic #2, and 50% from Topic #4. Below, we compute the topic mixtures for the first ten traning documents.
# 
# First, we setup our serializes and deserializers which allow us to convert NumPy arrays to CSV strings which we can pass into our HTTP POST request to our hosted endpoint.
# 

ntm_predictor.content_type = 'text/csv'
ntm_predictor.serializer = csv_serializer
ntm_predictor.deserializer = json_deserializer


# Now, let's check results for a small sample of records.
# 

results = ntm_predictor.predict(documents_training[:10])
print(results)


# We can see the output format of SageMaker NTM inference endpoint is a Python dictionary with the following format.
# 
# ```
# {
#   'predictions': [
#     {'topic_weights': [ ... ] },
#     {'topic_weights': [ ... ] },
#     {'topic_weights': [ ... ] },
#     ...
#   ]
# }
# ```
# 
# We extract the topic weights, themselves, corresponding to each of the input documents.
# 

predictions = np.array([prediction['topic_weights'] for prediction in results['predictions']])

print(predictions)


# If you decide to compare these results to the known topic weights generated above keep in mind that SageMaker NTM discovers topics in no particular order. That is, the approximate topic mixtures computed above may be (approximate) permutations of the known topic mixtures corresponding to the same documents.
# 

print(topic_mixtures_training[0])  # known topic mixture
print(predictions[0])  # computed topic mixture


# With that said, let's look at how our learned topic weights map to known topic mixtures for the entire training set.  Because NTM inherently creates a soft clustering (meaning that documents can sometimes belong partially to multiple topics), we'll evaluate correlation of topic weights.  This gives us a more relevant picture than just selecting the single topic for each document that happens to have the highest probability.
# 
# To do this, we'll first need to generate predictions for all of our training data.  Because our endpoint has a ~6MB per POST request limit, let's break the training data up into mini-batches and loop over them, creating a full dataset of predictions.
# 

def predict_batches(data, rows=1000):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = []
    for array in split_array:
        results = ntm_predictor.predict(array)
        predictions += [r['topic_weights'] for r in results['predictions']]
    return np.array(predictions)


predictions = predict_batches(documents_training)


# Now we'll look at how the actual and predicted topics correlate.
# 

data = pd.DataFrame(np.concatenate([topic_mixtures_training, predictions], axis=1), 
                    columns=['actual_{}'.format(i) for i in range(5)] + ['predictions_{}'.format(i) for i in range(5)])
display(data.corr())
pd.plotting.scatter_matrix(pd.DataFrame(np.concatenate([topic_mixtures_training, predictions], axis=1)), figsize=(12, 12))
plt.show()


# As we can see:
# - The upper left quadrant of 5 * 5 cells illustrates that the data are synthetic as the correlations are all slightly negative, but too perfectly triangular to occur naturally.
# - The upper right quadrant, which tells us about our model fit, shows some similarities, with many correlations having very near triangular shape, and negative correlations of a similar magnitude.
#   - Notice, actual topic #2 maps to predicted topic #2.  Similarly actual topic #3 maps to predicted topic #3, and #4 to #4.  However, there's a slight bit of uncertainty in topics #0 and #1.  Actual topic #0 appears to map to predicted topic #1, but actual topic #1 also correlates most highly with predicted topic #1.  This is not unexpected given that we're working with manufactured data and unsupervised algorithms.  The important part is that NTM is picking up aggregate structure well and with increased tuning of hyperparameters may fit the data even more closely.
# 
# _Note, specific results may differ due to randomized steps in the data generation and algorithm, but the general story should remain unchanged._
# 

# ## Stop / Close the Endpoint
# 
# Finally, we should delete the endpoint before we close the notebook.
# 
# To restart the endpoint you can follow the code above using the same `endpoint_name` we created or you can navigate to the "Endpoints" tab in the SageMaker console, select the endpoint with the name stored in the variable `endpoint_name`, and select "Delete" from the "Actions" dropdown menu. 
# 

sagemaker.Session().delete_endpoint(ntm_predictor.endpoint)


# # Extensions
# 
# ***
# 
# This notebook was a basic introduction to the NTM .  It was applied on a synthetic dataset merely to show how the algorithm functions and represents data.  Obvious extensions would be to train the algorithm utilizing real data.  We skipped the important step of qualitatively evaluating the outputs of NTM.  Because it is an unsupervised model, we want our topics to make sense.  There is a great deal of subjectivity involved in this, and whether or not NTM is more suitable than another topic modeling algorithm like Amazon SageMaker LDA will depend on your use case.
# 

# ## MNIST Training with MXNet and Gluon
# 
# MNIST is a widely used dataset for handwritten digit classification. It consists of 70,000 labeled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). This tutorial will show how to train and test an MNIST model on SageMaker using MXNet and the Gluon API.
# 
# 

import os
import boto3
import sagemaker
from sagemaker.mxnet import MXNet
from mxnet import gluon
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()


# ## Download training and test data
# 

gluon.data.vision.MNIST('./data/train', train=True)
gluon.data.vision.MNIST('./data/test', train=False)


# ## Uploading the data
# 
# We use the `sagemaker.Session.upload_data` function to upload our datasets to an S3 location. The return value `inputs` identifies the location -- we will use this later when we start the training job.
# 

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')


# ## Implement the training function
# 
# We need to provide a training script that can run on the SageMaker platform. The training scripts are essentially the same as one you would write for local training, except that you need to provide a `train` function. When SageMaker calls your function, it will pass in arguments that describe the training environment. Check the script below to see how this works.
# 
# The script here is an adaptation of the [Gluon MNIST example](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/mnist.py) provided by the [Apache MXNet](https://mxnet.incubator.apache.org/) project. 
# 

get_ipython().system("cat 'mnist.py'")


# ## Run the training script on SageMaker
# 
# The ```MXNet``` class allows us to run our training function on SageMaker infrastructure. We need to configure it with our training script, an IAM role, the number of training instances, and the training instance type. In this case we will run our training job on a single c4.xlarge instance. 
# 

m = MXNet("mnist.py", 
          role=role, 
          train_instance_count=1, 
          train_instance_type="ml.c4.xlarge",
          hyperparameters={'batch_size': 100, 
                         'epochs': 20, 
                         'learning_rate': 0.1, 
                         'momentum': 0.9, 
                         'log_interval': 100})


# After we've constructed our `MXNet` object, we can fit it using the data we uploaded to S3. SageMaker makes sure our data is available in the local filesystem, so our training script can simply read the data from disk.
# 

m.fit(inputs)


# After training, we use the MXNet object to build and deploy an MXNetPredictor object. This creates a SageMaker endpoint that we can use to perform inference. 
# 
# This allows us to perform inference on json encoded multi-dimensional arrays. 
# 

predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


# We can now use this predictor to classify hand-written digits. Drawing into the image box loads the pixel data into a 'data' variable in this notebook, which we can then pass to the mxnet predictor. 
# 

from IPython.display import HTML
HTML(open("input.html").read())


# The predictor runs inference on our input data and returns the predicted digit (as a float value, so we convert to int for display).
# 

response = predictor.predict(data)
print int(response)


# ## Cleanup
# 
# After you have finished with this example, remember to delete the prediction endpoint to release the instance(s) associated with it.
# 

sagemaker.Session().delete_endpoint(predictor.endpoint)


# # Multiclass classification with Amazon SageMaker XGBoost algorithm
# _**Single machine and distributed training for multiclass classification with Amazon SageMaker XGBoost algorithm**_
# 
# ---
# 
# ---
# ## Contents
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
#   2. [Data ingestion](#Data-ingestion)
#   3. [Data conversion](#Data-conversion)
# 3. [Training the XGBoost model](#Training-the-XGBoost-model)
#   1. [Training on a single instance](#Training-on-a-single-instance)
#   2. [Training on multiple instances](#Training-on-multiple-instances)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#   1. [Import model into hosting](#Import-model-into-hosting)
#   2. [Create endpoint configuration](#Create-endpoint-configuration)
#   3. [Create endpoint](#Create-endpoint)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 
# ---
# ## Introduction
# 
# 
# This notebook demonstrates the use of Amazon SageMaker’s implementation of the XGBoost algorithm to train and host a multiclass classification model. The MNIST dataset is used for training. It has a training set of 60,000 examples and a test set of 10,000 examples. To illustrate the use of libsvm training data format, we download the dataset and convert it to the libsvm format before training.
# 
# To get started, we need to set up the environment with a few prerequisites for permissions and configurations.
# 
# ---
# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# Here we set up the linkage and authentication to AWS services.
# 
# 1. The roles used to give learning and hosting access to your data. See the documentation for how to specify these.
# 2. The S3 bucket that you want to use for training and model data.
# 

get_ipython().run_cell_magic('time', '', "\nimport os\nimport boto3\nimport re\nimport copy\nimport time\nfrom time import gmtime, strftime\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nregion = boto3.Session().region_name\n\nbucket='<bucket-name>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/xgboost-multiclass-classification'\n# customize to your bucket where you have stored the data\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)")


# ### Data ingestion
# 
# Next, we read the dataset from the existing repository into memory, for preprocessing prior to training. This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.
# 

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nf = gzip.open(\'mnist.pkl.gz\', \'rb\')\ntrain_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')\nf.close()')


# ### Data conversion
# 
# Since algorithms have particular input and output requirements, converting the dataset is also part of the process that a data scientist goes through prior to initiating training. In this particular case, the data is converted from pickle-ized numpy array to the libsvm format before being uploaded to S3. The hosted implementation of xgboost consumes the libsvm converted data from S3 for training. The following provides functions for data conversions and file upload to S3 and download from S3. 
# 

get_ipython().run_cell_magic('time', '', '\nimport struct\nimport io\nimport boto3\n\n \ndef to_libsvm(f, labels, values):\n     f.write(bytes(\'\\n\'.join(\n         [\'{} {}\'.format(label, \' \'.join([\'{}:{}\'.format(i + 1, el) for i, el in enumerate(vec)])) for label, vec in\n          zip(labels, values)]), \'utf-8\'))\n     return f\n\n\ndef write_to_s3(fobj, bucket, key):\n    return boto3.Session().resource(\'s3\').Bucket(bucket).Object(key).upload_fileobj(fobj)\n\ndef get_dataset():\n  import pickle\n  import gzip\n  with gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n      u = pickle._Unpickler(f)\n      u.encoding = \'latin1\'\n      return u.load()\n\ndef upload_to_s3(partition_name, partition):\n    labels = [t.tolist() for t in partition[1]]\n    vectors = [t.tolist() for t in partition[0]]\n    num_partition = 5                                 # partition file into 5 parts\n    partition_bound = int(len(labels)/num_partition)\n    for i in range(num_partition):\n        f = io.BytesIO()\n        to_libsvm(f, labels[i*partition_bound:(i+1)*partition_bound], vectors[i*partition_bound:(i+1)*partition_bound])\n        f.seek(0)\n        key = "{}/{}/examples{}".format(prefix,partition_name,str(i))\n        url = \'s3n://{}/{}\'.format(bucket, key)\n        print(\'Writing to {}\'.format(url))\n        write_to_s3(f, bucket, key)\n        print(\'Done writing to {}\'.format(url))\n\ndef download_from_s3(partition_name, number, filename):\n    key = "{}/{}/examples{}".format(prefix,partition_name, number)\n    url = \'s3n://{}/{}\'.format(bucket, key)\n    print(\'Reading from {}\'.format(url))\n    s3 = boto3.resource(\'s3\')\n    s3.Bucket(bucket).download_file(key, filename)\n    try:\n        s3.Bucket(bucket).download_file(key, \'mnist.local.test\')\n    except botocore.exceptions.ClientError as e:\n        if e.response[\'Error\'][\'Code\'] == "404":\n            print(\'The object does not exist at {}.\'.format(url))\n        else:\n            raise        \n        \ndef convert_data():\n    train_set, valid_set, test_set = get_dataset()\n    partitions = [(\'train\', train_set), (\'validation\', valid_set), (\'test\', test_set)]\n    for partition_name, partition in partitions:\n        print(\'{}: {} {}\'.format(partition_name, partition[0].shape, partition[1].shape))\n        upload_to_s3(partition_name, partition)')


get_ipython().run_cell_magic('time', '', '\nconvert_data()')


# ## Training the XGBoost model
# 
# Now that we have our data in S3, we can begin training. We'll use Amazon SageMaker XGboost algorithm, and will actually fit two models in order to demonstrate the single machine and distributed training on SageMaker. In the first job, we'll use a single machine to train. In the second job, we'll use two machines and use the ShardedByS3Key mode for the train channel. Since we have 5 part file, one machine will train on three and the other on two part files. Note that the number of instances should not exceed the number of part files. 
# 
# First let's setup a list of training parameters which are common across the two jobs.
# 

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]


#Ensure that the train and validation data folders generated above are reflected in the "InputDataConfig" parameter below.
common_training_params = {
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": bucket_path + "/"+ prefix + "/xgboost"
    },
    "ResourceConfig": {
        "InstanceCount": 1,   
        "InstanceType": "ml.m4.10xlarge",
        "VolumeSizeInGB": 5
    },
    "HyperParameters": {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "silent":"0",
        "objective": "multi:softmax",
        "num_class": "10",
        "num_round": "10"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 86400
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/"+ prefix+ '/train/',
                    "S3DataDistributionType": "FullyReplicated" 
                }
            },
            "ContentType": "libsvm",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/"+ prefix+ '/validation/',
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "libsvm",
            "CompressionType": "None"
        }
    ]
}


# Now we'll create two separate jobs, updating the parameters that are unique to each.
# 
# ### Training on a single instance
# 

#single machine job params
single_machine_job_name = 'xgboost-single-machine-classification' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Job name is:", single_machine_job_name)

single_machine_job_params = copy.deepcopy(common_training_params)
single_machine_job_params['TrainingJobName'] = single_machine_job_name
single_machine_job_params['OutputDataConfig']['S3OutputPath'] = bucket_path + "/"+ prefix + "/xgboost-single"
single_machine_job_params['ResourceConfig']['InstanceCount'] = 1


# ### Training on multiple instances
# 
# You can also run the training job distributed over multiple instances. For larger datasets with multiple partitions, this can significantly boost the training speed. Here we'll still use the small/toy MNIST dataset to demo this feature.
# 

#distributed job params
distributed_job_name = 'xgboost-distributed-classification' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Job name is:", distributed_job_name)

distributed_job_params = copy.deepcopy(common_training_params)
distributed_job_params['TrainingJobName'] = distributed_job_name
distributed_job_params['OutputDataConfig']['S3OutputPath'] = bucket_path + "/"+ prefix + "/xgboost-distributed"
#number of instances used for training
distributed_job_params['ResourceConfig']['InstanceCount'] = 2 # no more than 5 if there are total 5 partition files generated above

# data distribution type for train channel
distributed_job_params['InputDataConfig'][0]['DataSource']['S3DataSource']['S3DataDistributionType'] = 'ShardedByS3Key'
# data distribution type for validation channel
distributed_job_params['InputDataConfig'][1]['DataSource']['S3DataSource']['S3DataDistributionType'] = 'ShardedByS3Key'


# Let's submit these jobs, taking note that the first will be submitted to run in the background so that we can immediately run the second in parallel.
# 

get_ipython().run_cell_magic('time', '', '\nregion = boto3.Session().region_name\nsm = boto3.Session().client(\'sagemaker\')\n\nsm.create_training_job(**single_machine_job_params)\nsm.create_training_job(**distributed_job_params)\n\nstatus = sm.describe_training_job(TrainingJobName=distributed_job_name)[\'TrainingJobStatus\']\nprint(status)\nsm.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=distributed_job_name)\nstatus = sm.describe_training_job(TrainingJobName=distributed_job_name)[\'TrainingJobStatus\']\nprint("Training job ended with status: " + status)\nif status == \'Failed\':\n    message = sm.describe_training_job(TrainingJobName=distributed_job_name)[\'FailureReason\']\n    print(\'Training failed with the following error: {}\'.format(message))\n    raise Exception(\'Training job failed\')')


# Let's confirm both jobs have finished.
# 

print('Single Machine:', sm.describe_training_job(TrainingJobName=single_machine_job_name)['TrainingJobStatus'])
print('Distributed:', sm.describe_training_job(TrainingJobName=distributed_job_name)['TrainingJobStatus'])


# # Set up hosting for the model
# In order to set up hosting, we have to import the model from training to hosting. The step below demonstrated hosting the model generated from the distributed training job. Same steps can be followed to host the model obtained from the single machine job. 
# 
# ### Import model into hosting
# Next, you register the model with hosting. This allows you the flexibility of importing models trained elsewhere.
# 

get_ipython().run_cell_magic('time', '', "import boto3\nfrom time import gmtime, strftime\n\nmodel_name=distributed_job_name + '-model'\nprint(model_name)\n\ninfo = sm.describe_training_job(TrainingJobName=distributed_job_name)\nmodel_data = info['ModelArtifacts']['S3ModelArtifacts']\nprint(model_data)\n\nprimary_container = {\n    'Image': container,\n    'ModelDataUrl': model_data\n}\n\ncreate_model_response = sm.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response['ModelArn'])")


# ### Create endpoint configuration
# SageMaker supports configuring REST endpoints in hosting with multiple models, e.g. for A/B testing purposes. In order to support this, customers create an endpoint configuration, that describes the distribution of traffic across the models, whether split, shadowed, or sampled in some way. In addition, the endpoint configuration describes the instance type required for model deployment.
# 

from time import gmtime, strftime

endpoint_config_name = 'XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.c4.xlarge',
        'InitialVariantWeight':1,
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# ### Create endpoint
# Lastly, the customer creates the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications. This takes 9-11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = sm.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nwhile status==\'Creating\':\n    time.sleep(60)\n    resp = sm.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Status: " + status)\n\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)')


# ## Validate the model for use
# Finally, the customer can now validate the model for use. They can obtain the endpoint from the client library using the result from previous operations, and generate classifications from the trained model using that endpoint.
# 

runtime_client = boto3.client('runtime.sagemaker')


# In order to evaluate the model, we'll use the test dataset previously generated. Let us first download the data from S3 to the local host.
# 

download_from_s3('test', 0, 'mnist.local.test') # reading the first part file within test


# Start with a single prediction. Lets use the first record from the test file.
# 

get_ipython().system('head -1 mnist.local.test > mnist.single.test')


get_ipython().run_cell_magic('time', '', "import json\n\nfile_name = 'mnist.single.test' #customize to your test file 'mnist.single.test' if use the data above\n\nwith open(file_name, 'r') as f:\n    payload = f.read()\n\nresponse = runtime_client.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType='text/x-libsvm', \n                                   Body=payload)\nresult = response['Body'].read().decode('ascii')\nprint('Predicted label is {}.'.format(result))")


# OK, a single prediction works.
# Let's do a whole batch and see how good is the predictions accuracy.
# 

import sys
def do_predict(data, endpoint_name, content_type):
    payload = '\n'.join(data)
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType=content_type, 
                                   Body=payload)
    result = response['Body'].read().decode('ascii')
    preds = [float(num) for num in result.split(',')]
    return preds

def batch_predict(data, batch_size, endpoint_name, content_type):
    items = len(data)
    arrs = []
    for offset in range(0, items, batch_size):
        arrs.extend(do_predict(data[offset:min(offset+batch_size, items)], endpoint_name, content_type))
        sys.stdout.write('.')
    return(arrs)


# The following function helps us calculate the error rate on the batch dataset. 
# 

get_ipython().run_cell_magic('time', '', "import json\n\nfile_name = 'mnist.local.test'\nwith open(file_name, 'r') as f:\n    payload = f.read().strip()\n\nlabels = [float(line.split(' ')[0]) for line in payload.split('\\n')]\ntest_data = payload.split('\\n')\npreds = batch_predict(test_data, 100, endpoint_name, 'text/x-libsvm')\n\nprint ('\\nerror rate=%f' % ( sum(1 for i in range(len(preds)) if preds[i]!=labels[i]) /float(len(preds))))")


# Here are a few predictions
# 

preds[0:10]


# and the corresponding labels
# 

labels[0:10]


# The following function helps us create the confusion matrix on the labeled batch test dataset.
# 

import numpy
def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = numpy.sum(predictions == labels)
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = numpy.zeros([10, 10], numpy.int32)
    bundled = zip(predictions, labels)
    for predicted, actual in bundled:
        confusions[int(predicted), int(actual)] += 1
    
    return error, confusions


# The following helps us visualize the erros that the XGBoost classifier is making. 
# 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

NUM_LABELS = 10  # change it according to num_class in your dataset
test_error, confusions = error_rate(numpy.asarray(preds), numpy.asarray(labels))
print('Test error: %.1f%%' % test_error)

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(False)
plt.xticks(numpy.arange(NUM_LABELS))
plt.yticks(numpy.arange(NUM_LABELS))
plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');

for i, cas in enumerate(confusions):
    for j, count in enumerate(cas):
        if count > 0:
            xoff = .07 * len(str(count))
            plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')


# ### Delete Endpoint
# Once you are done using the endpoint, you can use the following to delete it. 
# 

sm.delete_endpoint(EndpointName=endpoint_name)


# # Image classification transfer learning demo
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
# 3. [Fine-tuning the Image classification model](#Fine-tuning-the-Image-classification-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#   1. [Import model into hosting](#Import-model-into-hosting)
#   2. [Create endpoint configuration](#Create-endpoint-configuration)
#   3. [Create endpoint](#Create-endpoint)
# 5. [Perform Inference](#Perform-Inference)
# 

# ## Introduction
# 
# Welcome to our end-to-end example of distributed image classification algorithm in transfer learning mode. In this demo, we will use the Amazon sagemaker image classification algorithm in transfer learning mode to fine-tune a pre-trained model (trained on imagenet data) to learn to classify a new dataset. In particular, the pre-trained model will be fine-tuned using [caltech-256 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/). 
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# Here we set up the linkage and authentication to AWS services. There are three parts to this:
# 
# * The roles used to give learning and hosting access to your data. This will automatically be obtained from the role used to start the notebook
# * The S3 bucket that you want to use for training and model data
# * The Amazon sagemaker image classification docker image which need not be changed
# 

get_ipython().run_cell_magic('time', '', "import boto3\nimport re\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nbucket='<<bucket-name>>' # customize to your bucket\n\ncontainers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest',\n              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',\n              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest',\n              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest'}\ntraining_image = containers[boto3.Session().region_name]\nprint(training_image)")


# ## Fine-tuning the Image classification model
# 
# The caltech 256 dataset consist of images from 257 categories (the last one being a clutter category) and has 30k images with a minimum of 80 images and a maximum of about 800 images per category. 
# 
# The image classification algorithm can take two types of input formats. The first is a [recordio format](https://mxnet.incubator.apache.org/tutorials/basic/record_io.html) and the other is a [lst format](https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=im2rec). Files for both these formats are available at http://data.dmlc.ml/mxnet/data/caltech-256/. In this example, we will use the recordio format for training and use the training/validation split [specified here](http://data.dmlc.ml/mxnet/data/caltech-256/).
# 

import os
import urllib.request
import boto3

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

        
def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)


# # caltech-256
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
upload_to_s3('validation', 'caltech-256-60-val.rec')
upload_to_s3('train', 'caltech-256-60-train.rec')


# Once we have the data available in the correct format for training, the next step is to actually train the model using the data. Before training the model, we need to setup the training parameters. The next section will explain the parameters in detail.
# 

# ## Training parameters
# There are two kinds of parameters that need to be set for training. The first one are the parameters for the training job. These include:
# 
# * **Input specification**: These are the training and validation channels that specify the path where training data is present. These are specified in the "InputDataConfig" section. The main parameters that need to be set is the "ContentType" which can be set to "application/x-recordio" or "application/x-image" based on the input data format and the S3Uri which specifies the bucket and the folder where the data is present. 
# * **Output specification**: This is specified in the "OutputDataConfig" section. We just need to specify the path where the output can be stored after training
# * **Resource config**: This section specifies the type of instance on which to run the training and the number of hosts used for training. If "InstanceCount" is more than 1, then training can be run in a distributed manner. 
# 
# Apart from the above set of parameters, there are hyperparameters that are specific to the algorithm. These are:
# 
# * **num_layers**: The number of layers (depth) for the network. We use 18 in this samples but other values such as 50, 152 can be used.
# * **num_training_samples**: This is the total number of training samples. It is set to 15420 for caltech dataset with the current split
# * **num_classes**: This is the number of output classes for the new dataset. Imagenet was trained with 1000 output classes but the number of output classes can be changed for fine-tuning. For caltech, we use 257 because it has 256 object categories + 1 clutter class
# * **epochs**: Number of training epochs
# * **learning_rate**: Learning rate for training
# * **mini_batch_size**: The number of training samples used for each mini batch. In distributed training, the number of training samples used per batch will be N * mini_batch_size where N is the number of hosts on which training is run
# 

# After setting training parameters, we kick off training, and poll for status until training is completed, which in this example, takes between 10 to 12 minutes per epoch on a p2.xlarge machine. The network typically converges after 10 epochs.  
# 

# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 18 layers
num_layers = 18
# we need to specify the input image shape for the training data
image_shape = "3,224,224"
# we also need to specify the number of training samples in the training set
# for caltech it is 15420
num_training_samples = 15420
# specify the number of output classes
num_classes = 257
# batch size for training
mini_batch_size =  128
# number of epochs
epochs = 2
# learning rate
learning_rate = 0.01
top_k=2
# Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be 
# initialized with pre-trained weights
use_pretrained_model = 1


# # Training
# Run the training using Amazon sagemaker CreateTrainingJob API
# 

get_ipython().run_cell_magic('time', '', 'import time\nimport boto3\nfrom time import gmtime, strftime\n\n\ns3 = boto3.client(\'s3\')\n# create unique job name \njob_name_prefix = \'sagemaker-imageclassification-notebook\'\ntimestamp = time.strftime(\'-%Y-%m-%d-%H-%M-%S\', time.gmtime())\njob_name = job_name_prefix + timestamp\ntraining_params = \\\n{\n    # specify the training docker image\n    "AlgorithmSpecification": {\n        "TrainingImage": training_image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": \'s3://{}/{}/output\'.format(bucket, job_name_prefix)\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.p2.xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "image_shape": image_shape,\n        "num_layers": str(num_layers),\n        "num_training_samples": str(num_training_samples),\n        "num_classes": str(num_classes),\n        "mini_batch_size": str(mini_batch_size),\n        "epochs": str(epochs),\n        "learning_rate": str(learning_rate),\n        "use_pretrained_model": str(use_pretrained_model)\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 360000\n    },\n#Training data should be inside a subdirectory called "train"\n#Validation data should be inside a subdirectory called "validation"\n#The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/train/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/validation/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        }\n    ]\n}\nprint(\'Training job name: {}\'.format(job_name))\nprint(\'\\nInput Data Location: {}\'.format(training_params[\'InputDataConfig\'][0][\'DataSource\'][\'S3DataSource\']))')


# create the Amazon SageMaker training job
sagemaker = boto3.client(service_name='sagemaker')
sagemaker.create_training_job(**training_params)

# confirm that the training job has started
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))


training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
status = training_info['TrainingJobStatus']
print("Training job ended with status: " + status)


# If you see the message,
# 
# > `Training job ended with status: Completed`
# 
# then that means training sucessfully completed and the output model was stored in the output path specified by `training_params['OutputDataConfig']`.
# 
# You can also view information about and the status of a training job using the AWS SageMaker console. Just click on the "Jobs" tab.
# 

# # Inference
# 
# ***
# 
# A trained model does nothing on its own. We now want to use the model to perform inference. For this example, that means predicting the topic mixture representing a given document.
# 
# This section involves several steps,
# 
# 1. [Create Model](#CreateModel) - Create model for the training output
# 1. [Create Endpoint Configuration](#CreateEndpointConfiguration) - Create a configuration defining an endpoint.
# 1. [Create Endpoint](#CreateEndpoint) - Use the configuration to create an inference endpoint.
# 1. [Perform Inference](#Perform Inference) - Perform inference on some input data using the endpoint.
# 

# ## Create Model
# 
# We now create a SageMaker Model from the training output. Using the model we can create an Endpoint Configuration.
# 

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nsage = boto3.Session().client(service_name=\'sagemaker\') \n\nmodel_name="test-image-classification-model"\nprint(model_name)\ninfo = sage.describe_training_job(TrainingJobName=job_name)\nmodel_data = info[\'ModelArtifacts\'][\'S3ModelArtifacts\']\nprint(model_data)\ncontainers = {\'us-west-2\': \'433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest\',\n              \'us-east-1\': \'811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest\',\n              \'us-east-2\': \'825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest\',\n              \'eu-west-1\': \'685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest\'}\nhosting_image = containers[boto3.Session().region_name]\nprimary_container = {\n    \'Image\': hosting_image,\n    \'ModelDataUrl\': model_data,\n}\n\ncreate_model_response = sage.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response[\'ModelArn\'])')


# ### Create Endpoint Configuration
# At launch, we will support configuring REST endpoints in hosting with multiple models, e.g. for A/B testing purposes. In order to support this, customers create an endpoint configuration, that describes the distribution of traffic across the models, whether split, shadowed, or sampled in some way.
# 
# In addition, the endpoint configuration describes the instance type required for model deployment, and at launch will describe the autoscaling configuration.
# 

from time import gmtime, strftime

timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
endpoint_config_name = job_name_prefix + '-epc-' + timestamp
endpoint_config_response = sage.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.p2.xlarge',
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print('Endpoint configuration name: {}'.format(endpoint_config_name))
print('Endpoint configuration arn:  {}'.format(endpoint_config_response['EndpointConfigArn']))


# ### Create Endpoint
# Lastly, the customer creates the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications. This takes 9-11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', "import time\n\ntimestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())\nendpoint_name = job_name_prefix + '-ep-' + timestamp\nprint('Endpoint name: {}'.format(endpoint_name))\n\nendpoint_params = {\n    'EndpointName': endpoint_name,\n    'EndpointConfigName': endpoint_config_name,\n}\nendpoint_response = sagemaker.create_endpoint(**endpoint_params)\nprint('EndpointArn = {}'.format(endpoint_response['EndpointArn']))")


# Finally, now the endpoint can be created. It may take sometime to create the endpoint...
# 

# get the status of the endpoint
response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = response['EndpointStatus']
print('EndpointStatus = {}'.format(status))


# wait until the status has changed
sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)


# print the status of the endpoint
endpoint_response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = endpoint_response['EndpointStatus']
print('Endpoint creation ended with EndpointStatus = {}'.format(status))

if status != 'InService':
    raise Exception('Endpoint creation failed.')


# If you see the message,
# 
# > `Endpoint creation ended with EndpointStatus = InService`
# 
# then congratulations! You now have a functioning inference endpoint. You can confirm the endpoint configuration and status by navigating to the "Endpoints" tab in the AWS SageMaker console.
# 
# We will finally create a runtime object from which we can invoke the endpoint.
# 

# ## Perform Inference
# Finally, the customer can now validate the model for use. They can obtain the endpoint from the client library using the result from previous operations, and generate classifications from the trained model using that endpoint.
# 

import boto3
runtime = boto3.Session().client(service_name='runtime.sagemaker') 


# ### Download test image
# 

get_ipython().system('wget -O /tmp/test.jpg http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/008_0007.jpg')
file_name = '/tmp/test.jpg'
# test image
from IPython.display import Image
Image(file_name)  


import json
import numpy as np
with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-image', 
                                   Body=payload)
result = response['Body'].read()
# result will be in json format and convert it to ndarray
result = json.loads(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))


# ### Clean up
# 
# When we're done with the endpoint, we can just delete it and the backing instances will be released.  Run the following cell to delete the endpoint.
# 

sage.delete_endpoint(EndpointName=endpoint_name)





# # Mxnet BYOM: Train locally and deploy on SageMaker.
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#     1. [Permissions and environment variables](#Permissions-and-environment-variables)
#     2. [Data Setup](#Data-setup)
# 3. [Training the network locally](#Training)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#     1. [Export from MXNet](#Export-the-model-from-mxnet)
#     2. [Import model into SageMaker](#Import-model-into-SageMaker)
#     3. [Create endpoint](#Create-endpoint) 
# 5. [Validate the endpoint for use](#Validate-the-endpoint-for-use)
# 
# 
# __Note__: Compare this with the [tensorflow bring your own model example](../tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb)
# 

# ## Introduction
# In this notebook, we will train a neural network locally on the location from where this notebook is run using MXNet. We will then see how to create an endpoint from the trained MXNet model and deploy it on SageMaker. We will then inference from the newly created SageMaker endpoint. 
# 
# The neural network that we will use is a simple fully-connected neural network. The definition of the neural network can be found in the accompanying [mnist.py](mnist.py) file. The ``build_graph`` method contains the model defnition (shown below).
# 
# ```python
# def build_graph():
#     data = mx.sym.var('data')
#     data = mx.sym.flatten(data=data)
#     fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
#     act1 = mx.sym.Activation(data=fc1, act_type="relu")
#     fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
#     act2 = mx.sym.Activation(data=fc2, act_type="relu")
#     fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
#     return mx.sym.SoftmaxOutput(data=fc3, name='softmax')
# ```
# 
# From this definitnion we can see that there are two fully-connected layers of 128 and 64 neurons each. The activations of the last fully-connected layer is then fed into a Softmax layer of 10 neurons. We use 10 neurons here because the datatset on which we are going to predict is the MNIST dataset of hand-written digit recognition which has 10 classes. More details can be found about the dataset on the [creator's webpage](http://yann.lecun.com/exdb/mnist/).
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# Here we set up the linkage and authentication to AWS services. In this notebook we only need the roles used to give learning and hosting access to your data. The Sagemaker SDK will use S3 defualt buckets when needed. Supply the role in the variable below.
# 

import boto3, re
from sagemaker import get_execution_role

role = get_execution_role()


# ### Data setup
# 
# Next, we need to pull the data from the author's site to our local box. Since we have ``mxnet`` utilities, we will use the utilities to download the dataset locally.
# 

import mxnet as mx
data = mx.test_utils.get_mnist()


# ### Training
# 
# It is time to train the network. Since we are training the network locally, we can make use of mxnet training tools. The training method is also in the accompanying [mnist.py](mnist.py) file. The method is shown below. 
# 
# ```python 
# def train(data, hyperparameters= {'learning_rate': 0.11}, num_cpus=0, num_gpus =1 , **kwargs):
#     train_labels = data['train_label']
#     train_images = data['train_data']
#     test_labels = data['test_label']
#     test_images = data['test_data']
#     batch_size = 100
#     train_iter = mx.io.NDArrayIter(train_images, train_labels, batch_size, shuffle=True)
#     val_iter = mx.io.NDArrayIter(test_images, test_labels, batch_size)
#     logging.getLogger().setLevel(logging.DEBUG)
#     mlp_model = mx.mod.Module(
#         symbol=build_graph(),
#         context=get_train_context(num_cpus, num_gpus))
#     mlp_model.fit(train_iter,
#                   eval_data=val_iter,
#                   optimizer='sgd',
#                   optimizer_params={'learning_rate': float(hyperparameters.get("learning_rate", 0.1))},
#                   eval_metric='acc',
#                   batch_end_callback=mx.callback.Speedometer(batch_size, 100),
#                   num_epoch=10)
#     return mlp_model
# ```
# 
# The method above collects the ``data`` variable that ``get_mnist`` method gives you (which is a dictionary of data arrays) along with a dictionary of ``hyperparameters`` which only contains learning rate, and other parameters. It creates a [``mxnet.mod.Module``](https://mxnet.incubator.apache.org/api/python/module.html) from the network graph we built in the ``build_graph`` method and trains the network using the ``mxnet.mod.Module.fit`` method. 
# 

from mnist import train
model = train(data = data)


# ## Set up hosting for the model
# 
# ### Export the model from mxnet
# 
# In order to set up hosting, we have to import the model from training to hosting. We will begin by exporting the model from MXNet and saving it down. Analogous to the [TensorFlow example](../tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb), some structure needs to be followed. The exported model has to be converted into a form that is readable by ``sagemaker.mxnet.model.MXNetModel``. The following code describes exporting the model in a form that does the same:
# 

import os
os.mkdir('model')
model.save_checkpoint('model/model', 0000)
import tarfile
with tarfile.open('model.tar.gz', mode='w:gz') as archive:
    archive.add('model', recursive=True)


# ### Import model into SageMaker
# 
# Open a new sagemaker session and upload the model on to the default S3 bucket. We can use the ``sagemaker.Session.upload_data`` method to do this. We need the location of where we exported the model from MXNet and where in our default bucket we want to store the model(``/model``). The default S3 bucket can be found using the ``sagemaker.Session.default_bucket`` method.
# 

import sagemaker

sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')


# Use the ``sagemaker.mxnet.model.MXNetModel`` to import the model into SageMaker that can be deployed. We need the location of the S3 bucket where we have the model, the role for authentication and the entry_point where the model defintion is stored (``mnist.py``). The import call is the following:
# 

from sagemaker.mxnet.model import MXNetModel
sagemaker_model = MXNetModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                  role = role,
                                  entry_point = 'mnist.py')


# ### Create endpoint
# 
# Now the model is ready to be deployed at a SageMaker endpoint. We can use the ``sagemaker.mxnet.model.MXNetModel.deploy`` method to do this. Unless you have created or prefer other instances, we recommend using 1 ``'ml.c4.xlarge'`` instance for this training. These are supplied as arguments. 
# 

predictor = sagemaker_model.deploy(initial_instance_count=1,
                                          instance_type='ml.c4.xlarge')


# ### Validate the endpoint for use
# 
# We can now use this endpoint to classify hand-written digits.
# 

predict_sample = data['test_data'][0][0]
response = predictor.predict(data)
print('Raw prediction result:')
print(response)


# (Optional) Delete the Endpoint
# 

print(predictor.endpoint)


# If you do not want continued use of the endpoint, you can remove it. Remember, open endpoints are charged. If this is a simple test or practice, it is recommended to delete them.
# 

sagemaker.Session().delete_endpoint(predictor.endpoint)


# Clear all stored model data so that we don't overwrite them the next time. 
# 

os.remove('model.tar.gz')
import shutil
shutil.rmtree('export')


# ## Distributed ResNet Training with MXNet and Gluon
# 
# [ResNet_V2](https://arxiv.org/abs/1512.03385) is an architecture for deep convolution networks. In this example, we train a 34 layer network to perform image classification using the CIFAR-10 dataset. CIFAR-10 consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
# 
# ### Setup
# 
# This example requires the `scikit-image` library. Use jupyter's [conda tab](/tree#conda) to install it.
# 

import os
import boto3
import sagemaker
from sagemaker.mxnet import MXNet
from mxnet import gluon
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()


# ## Download training and test data
# 
# We use the helper scripts to download CIFAR10 training data and sample images.
# 

from cifar10_utils import download_training_data
download_training_data()


# ## Uploading the data
# 
# We use the `sagemaker.Session.upload_data` function to upload our datasets to an S3 location. The return value `inputs` identifies the location -- we will use this later when we start the training job.
# 

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/gluon-cifar10')
print('input spec (in this case, just an S3 path): {}'.format(inputs))


# ## Implement the training function
# 
# We need to provide a training script that can run on the SageMaker platform. The training scripts are essentially the same as one you would write for local training, except that you need to provide a `train` function. When SageMaker calls your function, it will pass in arguments that describe the training environment. Check the script below to see how this works.
# 
# The network itself is a pre-built version contained in the [Gluon Model Zoo](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html).
# 

get_ipython().system("cat 'cifar10.py'")


# ## Run the training script on SageMaker
# 
# The ```MXNet``` class allows us to run our training function as a distributed training job on SageMaker infrastructure. We need to configure it with our training script, an IAM role, the number of training instances, and the training instance type. In this case we will run our training job on two `ml.p2.xlarge` instances.
# 
# **Note:** you may need to request a limit increase in order to use two ``ml.p2.xlarge`` instances. If you 
# want to try the example without requesting an increase, just change the ``train_instance_count`` value to ``1``.
# 

m = MXNet("cifar10.py", 
          role=role, 
          train_instance_count=2, 
          train_instance_type="ml.p2.xlarge",
          hyperparameters={'batch_size': 128, 
                           'epochs': 50, 
                           'learning_rate': 0.1, 
                           'momentum': 0.9})


# After we've constructed our `MXNet` object, we can fit it using the data we uploaded to S3. SageMaker makes sure our data is available in the local filesystem, so our training script can simply read the data from disk.
# 

m.fit(inputs)


# ## Prediction
# 
# After training, we use the MXNet estimator object to create and deploy a hosted prediction endpoint. We can use a CPU-based instance for inference (in this case an `ml.c4.xlarge`), even though we trained on GPU instances.
# 
# The predictor object returned by `deploy` lets us call the new endpoint and perform inference on our sample images. 
# 

predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


# ### CIFAR10 sample images
# 
# We'll use these CIFAR10 sample images to test the service:
# 
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/airplane1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/automobile1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/bird1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/cat1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/deer1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/dog1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/frog1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/horse1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/ship1.png" />
# <img style="display: inline; height: 32px; margin: 0.25em" src="images/truck1.png" />
# 
# 

# load the CIFAR10 samples, and convert them into format we can use with the prediction endpoint
from cifar10_utils import read_images

filenames = ['images/airplane1.png',
             'images/automobile1.png',
             'images/bird1.png',
             'images/cat1.png',
             'images/deer1.png',
             'images/dog1.png',
             'images/frog1.png',
             'images/horse1.png',
             'images/ship1.png',
             'images/truck1.png']

image_data = read_images(filenames)


# The predictor runs inference on our input data and returns the predicted class label (as a float value, so we convert to int for display).
# 

for i, img in enumerate(image_data):
    response = predictor.predict(img)
    print('image {}: class: {}'.format(i, int(response)))


# ## Cleanup
# 
# After you have finished with this example, remember to delete the prediction endpoint to release the instance(s) associated with it.
# 

sagemaker.Session().delete_endpoint(predictor.endpoint)


# # Abalone age predictor using tf.layers
# 
# This tutorial covers how to create your own training script using the building
# blocks provided in `tf.layers`, which will predict the ages of
# [abalones](https://en.wikipedia.org/wiki/Abalone) based on their physical
# measurements. You'll learn how to do the following:
# 
# *   Instantiate an `sagemaker.Estimator`
# *   Construct a custom model function
# *   Configure a neural network using `tf.feature_column` and `tf.layers`
# *   Choose an appropriate loss function from `tf.losses`
# *   Define a training op for your model
# *   Generate and return predictions
# 

# ## An Abalone Age Predictor
# 
# It's possible to estimate the age of an
# [abalone](https://en.wikipedia.org/wiki/Abalone) (sea snail) by the number of
# rings on its shell. However, because this task requires cutting, staining, and
# viewing the shell under a microscope, it's desirable to find other measurements
# that can predict age.
# 
# The [Abalone Data Set](https://archive.ics.uci.edu/ml/datasets/Abalone) contains
# the following
# [feature data](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names)
# for abalone:
# 
# | Feature        | Description                                               |
# | -------------- | --------------------------------------------------------- |
# | Length         | Length of abalone (in longest direction; in mm)           |
# | Diameter       | Diameter of abalone (measurement perpendicular to length; in mm)|
# | Height         | Height of abalone (with its meat inside shell; in mm)     |
# | Whole Weight   | Weight of entire abalone (in grams)                       |
# | Shucked Weight | Weight of abalone meat only (in grams)                    |
# | Viscera Weight | Gut weight of abalone (in grams), after bleeding          |
# | Shell Weight   | Weight of dried abalone shell (in grams)                  |
# 
# The label to predict is number of rings, as a proxy for abalone age.
# 

# ### Set up the environment
# 

import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()


# ### Upload the data to a S3 bucket
# 

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/abalone')


# **sagemaker_session.upload_data** will upload the abalone dataset from your machine to a bucket named **sagemaker-{region}-{your aws account number}**, if you don't have this bucket yet, sagemaker_session will create it for you.
# 

# ## Complete source code
# Here is the full code for the network model:
# 

get_ipython().system("cat 'abalone.py'")


# ## Constructing the `model_fn`
# 
# The basic skeleton for an model function looks like this:
# 
# ```python
# def model_fn(features, labels, mode, hyperparameters):
#    # Logic to do the following:
#    # 1. Configure the model via TensorFlow operations
#    # 2. Define the loss function for training/evaluation
#    # 3. Define the training operation/optimizer
#    # 4. Generate predictions
#    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
#    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
# ```
# 
# The **`model_fn`** requires three arguments:
# 
# *   **`features`**: A dict containing the features passed to the model via
#     **`input_fn`**.
# *   **`labels`**: A `Tensor` containing the labels passed to the model via
#     **`input_fn`**. Will be empty for `predict()` calls, as these are the values the
#     model will infer.
# *   **`mode`**: One of the following tf.estimator.ModeKeys string values
#     indicating the context in which the model_fn was invoked:
#     *   **`TRAIN`** The **`model_fn`** was invoked in training
#         mode, namely via a `train()` call.
#     *   **`EVAL`** The **`model_fn`** was invoked in
#         evaluation mode, namely via an `evaluate()` call.
#     *   **`PREDICT`** The **`model_fn`** was invoked in
#         predict mode, namely via a `predict()` call.
# 
# **`model_fn`** may also accept a **`hyperparameters`** argument containing a dict of
# hyperparameters used for training (as shown in the skeleton above).
# 
# The body of the function performs the following tasks (described in detail in the
# sections that follow):
# 
# *   Configuring the model for the abalone predictor, this will be a neural
#     network.
# *   Defining the loss function used to calculate how closely the model's
#     predictions match the target values.
# *   Defining the training operation that specifies the `optimizer` algorithm to
#     minimize the loss values calculated by the loss function.
# 

# The **`model_fn`** must return a tf.estimator.EstimatorSpec
# object, which contains the following values:
# 
# *   **`mode`** (required). The mode in which the model was run. Typically, you will
#     return the `mode` argument of the `model_fn` here.
# 
# *   **`predictions`** (required in `PREDICT` mode). A dict that maps key names of
#     your choice to `Tensor`s containing the predictions from the model, e.g.:
# 
#     ```python
#     predictions = {"results": tensor_of_predictions}
#     ```
# 
#     In `PREDICT` mode, the dict that you return in `EstimatorSpec` will then be
#     returned by `predict()`, so you can construct it in the format in which
#     you'd like to consume it.
# 
# 
# *   **`loss`** (required in `EVAL` and `TRAIN` mode). A `Tensor` containing a scalar
#     loss value: the output of the model's loss function (discussed in more depth
#     later in Defining loss for the model calculated over all
#     the input examples. This is used in `TRAIN` mode for error handling and
#     logging, and is automatically included as a metric in `EVAL` mode.
# 
# *   **`train_op`** (required only in `TRAIN` mode). An Op that runs one step of
#     training.
# 
# *   **`eval_metric_ops`** (optional). A dict of name/value pairs specifying the
#     metrics that will be calculated when the model runs in `EVAL` mode. The name
#     is a label of your choice for the metric, and the value is the result of
#     your metric calculation. The tf.metrics
#     module provides predefined functions for a variety of common metrics. The
#     following `eval_metric_ops` contains an `"accuracy"` metric calculated using
#     `tf.metrics.accuracy`:
# 
#     ```python
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(labels, predictions)
#     }
#     ```
# 
#     If you do not specify `eval_metric_ops`, only `loss` will be calculated
#     during evaluation.
# 

# ### Configuring a neural network with `tf.feature_column` and `tf.layers`
# 
# Constructing a [neural
# network](https://en.wikipedia.org/wiki/Artificial_neural_network) entails
# creating and connecting the input layer, the hidden layers, and the output
# layer.
# 
# The input layer is a series of nodes (one for each feature in the model) that
# will accept the feature data that is passed to the `model_fn` in the `features`
# argument. If `features` contains an n-dimensional `Tensor` with all your feature
# data, then it can serve as the input layer.
# If `features` contains a dict of feature columns passed to
# the model via an input function, you can convert it to an input-layer `Tensor`
# with the tf.feature_column.input_layer function.
# 
# ```python
# input_layer = tf.feature_column.input_layer(features=features, feature_columns=[age, height, weight])
# ```
# 

# As shown above, **`input_layer()`** takes two required arguments:
# 
# *   **`features`**. A mapping from string keys to the `Tensors` containing the
#     corresponding feature data. This is exactly what is passed to the `model_fn`
#     in the `features` argument.
# *   **`feature_columns`**. A list of all the `FeatureColumns`: `age`,
#     `height`, and `weight` in the above example.
# 
# The input layer of the neural network then must be connected to one or more
# hidden layers via an [activation
# function](https://en.wikipedia.org/wiki/Activation_function) that performs a
# nonlinear transformation on the data from the previous layer. The last hidden
# layer is then connected to the output layer, the final layer in the model.
# `tf.layers` provides the `tf.layers.dense` function for constructing fully
# connected layers. The activation is controlled by the `activation` argument.
# Some options to pass to the `activation` argument are:
# 
# *   **`tf.nn.relu`**. The following code creates a layer of `units` nodes fully
#     connected to the previous layer `input_layer` with a
#     [ReLU activation function](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\))
#     (tf.nn.relu):
# 
#     ```python
#     hidden_layer = tf.layers.dense(
#         inputs=input_layer, units=10, activation=tf.nn.relu)
#     ```
# 
# *   **`tf.nn.relu`**. The following code creates a layer of `units` nodes fully
#     connected to the previous layer `hidden_layer` with a ReLU activation
#     function:
# 
#     ```python
#     second_hidden_layer = tf.layers.dense(
#         inputs=hidden_layer, units=20, activation=tf.nn.relu)
#     ```
# 
# *   **`None`**. The following code creates a layer of `units` nodes fully connected
#     to the previous layer `second_hidden_layer` with *no* activation function,
#     just a linear transformation:
# 
#     ```python
#     output_layer = tf.layers.dense(
#         inputs=second_hidden_layer, units=3, activation=None)
#     ```
# 

# Other activation functions are possible, e.g.:
# 
# ```python
# output_layer = tf.layers.dense(inputs=second_hidden_layer,
#                                units=10,
#                                activation_fn=tf.sigmoid)
# ```
# 
# The above code creates the neural network layer `output_layer`, which is fully
# connected to `second_hidden_layer` with a sigmoid activation function
# (tf.sigmoid).
# 
# Putting it all together, the following code constructs a full neural network for
# the abalone predictor, and captures its predictions:
# 
# ```python
# def model_fn(features, labels, mode, params):
#   """Model function for Estimator."""
# 
#   # Connect the first hidden layer to input layer
#   # (features["x"]) with relu activation
#   first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)
# 
#   # Connect the second hidden layer to first hidden layer with relu
#   second_hidden_layer = tf.layers.dense(
#       first_hidden_layer, 10, activation=tf.nn.relu)
# 
#   # Connect the output layer to second hidden layer (no activation fn)
#   output_layer = tf.layers.dense(second_hidden_layer, 1)
# 
#   # Reshape output layer to 1-dim Tensor to return predictions
#   predictions = tf.reshape(output_layer, [-1])
#   predictions_dict = {"ages": predictions}
#   ...
# ```
# 
# Here, because you'll be passing the abalone `Datasets` using `numpy_input_fn`
# as shown below, `features` is a dict `{"x": data_tensor}`, so
# `features["x"]` is the input layer. The network contains two hidden
# layers, each with 10 nodes and a ReLU activation function. The output layer
# contains no activation function, and is
# tf.reshape to a one-dimensional
# tensor to capture the model's predictions, which are stored in
# `predictions_dict`.
# 
# ### Defining loss for the model
# 
# The `EstimatorSpec` returned by the `model_fn` must contain `loss`: a `Tensor`
# representing the loss value, which quantifies how well the model's predictions
# reflect the label values during training and evaluation runs. The tf.losses
# module provides convenience functions for calculating loss using a variety of
# metrics, including:
# 
# *   `absolute_difference(labels, predictions)`. Calculates loss using the
#     [absolute-difference
#     formula](https://en.wikipedia.org/wiki/Deviation_\(statistics\)#Unsigned_or_absolute_deviation)
#     (also known as L<sub>1</sub> loss).
# 
# *   `log_loss(labels, predictions)`. Calculates loss using the [logistic loss
#     forumula](https://en.wikipedia.org/wiki/Loss_functions_for_classification#Logistic_loss)
#     (typically used in logistic regression).
# 
# *   `mean_squared_error(labels, predictions)`. Calculates loss using the [mean
#     squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE; also
#     known as L<sub>2</sub> loss).
# 
# The following example adds a definition for `loss` to the abalone `model_fn`
# using `mean_squared_error()`:
# 

# ```python
# def model_fn(features, labels, mode, params):
#   """Model function for Estimator."""
# 
#   # Connect the first hidden layer to input layer
#   # (features["x"]) with relu activation
#   first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)
# 
#   # Connect the second hidden layer to first hidden layer with relu
#   second_hidden_layer = tf.layers.dense(
#       first_hidden_layer, 10, activation=tf.nn.relu)
# 
#   # Connect the output layer to second hidden layer (no activation fn)
#   output_layer = tf.layers.dense(second_hidden_layer, 1)
# 
#   # Reshape output layer to 1-dim Tensor to return predictions
#   predictions = tf.reshape(output_layer, [-1])
#   predictions_dict = {"ages": predictions}
# 
# 
#   # Calculate loss using mean squared error
#   loss = tf.losses.mean_squared_error(labels, predictions)
#   ...
# ```
# 
# See the [tf.losses](https://www.tensorflow.org/api_docs/python/tf/losses) for a
# full list of loss functions and more details on supported arguments and usage.
# 
# Supplementary metrics for evaluation can be added to an `eval_metric_ops` dict.
# The following code defines an `rmse` metric, which calculates the root mean
# squared error for the model predictions. Note that the `labels` tensor is cast
# to a `float64` type to match the data type of the `predictions` tensor, which
# will contain real values:
# 
# ```python
# eval_metric_ops = {
#     "rmse": tf.metrics.root_mean_squared_error(
#         tf.cast(labels, tf.float64), predictions)
# }
# ```
# 

# ### Defining the training op for the model
# 
# The training op defines the optimization algorithm TensorFlow will use when
# fitting the model to the training data. Typically when training, the goal is to
# minimize loss. A simple way to create the training op is to instantiate a
# `tf.train.Optimizer` subclass and call the `minimize` method.
# 
# The following code defines a training op for the abalone `model_fn` using the
# loss value calculated in [Defining Loss for the Model](https://github.com/tensorflow/tensorflow/blob/eb84435170c694175e38bfa02751c3ef881c7a20/tensorflow/docs_src/extend/estimators.md#defining-loss), the
# learning rate passed to the function in `params`, and the gradient descent
# optimizer. For `global_step`, the convenience function
# tf.train.get_global_step takes care of generating an integer variable:
# 
# ```python
# optimizer = tf.train.GradientDescentOptimizer(
#     learning_rate=params["learning_rate"])
# train_op = optimizer.minimize(
#     loss=loss, global_step=tf.train.get_global_step())
# ```
# 
# ### The complete abalone `model_fn`
# 
# Here's the final, complete `model_fn` for the abalone age predictor. The
# following code configures the neural network; defines loss and the training op;
# and returns a `EstimatorSpec` object containing `mode`, `predictions_dict`, `loss`,
# and `train_op`:
# 
# ```python
# def model_fn(features, labels, mode, params):
#   """Model function for Estimator."""
# 
#   # Connect the first hidden layer to input layer
#   # (features["x"]) with relu activation
#   first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)
# 
#   # Connect the second hidden layer to first hidden layer with relu
#   second_hidden_layer = tf.layers.dense(
#       first_hidden_layer, 10, activation=tf.nn.relu)
# 
#   # Connect the output layer to second hidden layer (no activation fn)
#   output_layer = tf.layers.dense(second_hidden_layer, 1)
# 
#   # Reshape output layer to 1-dim Tensor to return predictions
#   predictions = tf.reshape(output_layer, [-1])
# 
#   # Provide an estimator spec for `ModeKeys.PREDICT`.
#   if mode == tf.estimator.ModeKeys.PREDICT:
#     return tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions={"ages": predictions})
# 
#   # Calculate loss using mean squared error
#   loss = tf.losses.mean_squared_error(labels, predictions)
# 
#   # Calculate root mean squared error as additional eval metric
#   eval_metric_ops = {
#       "rmse": tf.metrics.root_mean_squared_error(
#           tf.cast(labels, tf.float64), predictions)
#   }
# 
#   optimizer = tf.train.GradientDescentOptimizer(
#       learning_rate=params["learning_rate"])
#   train_op = optimizer.minimize(
#       loss=loss, global_step=tf.train.get_global_step())
# 
#   # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
#   return tf.estimator.EstimatorSpec(
#       mode=mode,
#       loss=loss,
#       train_op=train_op,
#       eval_metric_ops=eval_metric_ops)
# ```
# 

# # Submitting script for training
# 

from sagemaker.tensorflow import TensorFlow

abalone_estimator = TensorFlow(entry_point='abalone.py',
                               role=role,
                               training_steps= 100,                                  
                               evaluation_steps= 100,
                               hyperparameters={'learning_rate': 0.001},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge')

abalone_estimator.fit(inputs)


# `estimator.fit` will deploy a script in a container for training and returs the SageMaker model name using the following arguments:
# 
# *   **`entry_point="abalone.py"`** The path to the script that will be deployed to the container.
# *   **`training_steps=100`** The number of training steps of the training job.
# *   **`evaluation_steps=100`** The number of evaluation steps of the training job.
# *   **`role`**. AWS role that gives your account access to SageMaker training and hosting
# *   **`hyperparameters={'learning_rate' : 0.001}`**. Training hyperparameters. 
# 
# Running the code block above will do the following actions:
# * deploy your script in a container with tensorflow installed
# * copy the data from the bucket to the container
# * instantiate the tf.estimator
# * train the estimator with 100 training steps
# * save the estimator model
# 

# # Submiting a trained model for hosting
# 

abalone_predictor = abalone_estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


# `abalone_estimator.deploy` deploys the trained model in a container ready for production.
# 

# # Invoking the endpoint
# 

import tensorflow as tf
import numpy as np

prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=os.path.join('data/abalone_predict.csv'), target_dtype=np.int, features_dtype=np.float32)

data = prediction_set.data[0]
tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)


abalone_predictor.predict(tensor_proto)


# # Deleting the endpoint
# 

sagemaker.Session().delete_endpoint(abalone_predictor.endpoint)


# # End-to-End Example #1
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
#   2. [Data ingestion](#Data-ingestion)
#   3. [Data inspection](#Data-inspection)
#   4. [Data conversion](#Data-conversion)
# 3. [Training the K-Means model](#Training-the-K-Means-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 

# ## Introduction
# 
# Welcome to our first end-to-end example! Today, we're working through a classification problem, specifically of images of handwritten digits, from zero to nine. Let's imagine that this dataset doesn't have labels, so we don't know for sure what the true answer is. In later examples, we'll show the value of "ground truth", as it's commonly known.
# 
# Today, however, we need to get these digits classified without ground truth. A common method for doing this is a set of methods known as "clustering", and in particular, the method that we'll look at today is called k-means clustering. In this method, each point belongs to the cluster with the closest mean, and the data is partitioned into a number of clusters that is specified when framing the problem. In this case, since we know there are 10 clusters, and we have no labeled data (in the way we framed the problem), this is a good fit.
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# Here we set up the linkage and authentication to AWS services. There are two parts to this:
# 
# 1. The role(s) used to give learning and hosting access to your data. Here we extract the role you created earlier for accessing your notebook.  See the documentation if you want to specify  a different role
# 1. The S3 bucket name and locations that you want to use for training and model data.
# 

from sagemaker import get_execution_role

role = get_execution_role()
bucket='<bucket-name>'


# ### Data ingestion
# 
# Next, we read the dataset from the existing repository into memory, for preprocessing prior to training.  In this case we'll use the MNIST dataset, which contains 70K 28 x 28 pixel images of handwritten digits.  For more details, please see [here](http://yann.lecun.com/exdb/mnist/).
# 
# This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.
# 

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nwith gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')')


# ### Data inspection
# 
# Once the dataset is imported, it's typical as part of the machine learning process to inspect the data, understand the distributions, and determine what type(s) of preprocessing might be needed. You can perform those tasks right here in the notebook. As an example, let's go ahead and look at one of the digits that is part of the dataset.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)


def show_digit(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)

show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))


# ## Training the K-Means model
# 
# Once we have the data preprocessed and available in the correct format for training, the next step is to actually train the model using the data. Since this data is relatively small, it isn't meant to show off the performance of the k-means training algorithm.  But Amazon SageMaker's k-means has been tested on, and scales well with, multi-terabyte datasets.
# 
# After setting training parameters, we kick off training, and poll for status until training is completed, which in this example, takes between 7 and 11 minutes.
# 

from sagemaker import KMeans

data_location = 's3://{}/kmeans_highlevel_example/data'.format(bucket)
output_location = 's3://{}/kmeans_example/output'.format(bucket)

print('training data will be uploaded to: {}'.format(data_location))
print('training artifacts will be uploaded to: {}'.format(output_location))

kmeans = KMeans(role=role,
                train_instance_count=2,
                train_instance_type='ml.c4.8xlarge',
                output_path=output_location,
                k=10,
                data_location=data_location)


get_ipython().run_cell_magic('time', '', '\nkmeans.fit(kmeans.record_set(train_set[0]))')


# ## Set up hosting for the model
# Now, we can deploy the model we just trained behind a real-time hosted endpoint.  This next step can take, on average, 7 to 11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', "\nkmeans_predictor = kmeans.deploy(initial_instance_count=1,\n                                 instance_type='ml.c4.xlarge')")


# ## Validate the model for use
# Finally, we'll validate the model for use. Let's generate a classification for a single observation from the trained model using the endpoint we just created.
# 

result = kmeans_predictor.predict(train_set[0][30:31])
print(result)


# OK, a single prediction works.
# 
# Let's do a whole batch and see how well the clustering works.
# 

get_ipython().run_cell_magic('time', '', "\nresult = kmeans_predictor.predict(valid_set[0][0:100])\nclusters = [r.label['closest_cluster'].float32_tensor.values[0] for r in result]")


for cluster in range(10):
    print('\n\n\nCluster {}:'.format(int(cluster)))
    digits = [ img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster ]
    height=((len(digits)-1)//5)+1
    width=5
    plt.rcParams["figure.figsize"] = (width,height)
    _, subplots = plt.subplots(height, width)
    subplots=numpy.ndarray.flatten(subplots)
    for subplot, image in zip(subplots, digits):
        show_digit(image, subplot=subplot)
    for subplot in subplots[len(digits):]:
        subplot.axis('off')

    plt.show()


# ### The bottom line
# 
# K-Means clustering is not the best algorithm for image analysis problems, but we do see pretty reasonable clusters being built.
# 

# ### (Optional) Delete the Endpoint
# If you're ready to be done with this notebook, make sure run the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

print(kmeans_predictor.endpoint)


import sagemaker
sagemaker.Session().delete_endpoint(kmeans_predictor.endpoint)


# # An Introduction to Linear Learner with MNIST
# _**Making a Binary Prediction of Whether a Handwritten Digit is a 0**_
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
#   2. [Data ingestion](#Data-ingestion)
#   3. [Data inspection](#Data-inspection)
#   4. [Data conversion](#Data-conversion)
# 3. [Training the linear model](#Training-the-linear-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 

# ## Introduction
# 
# Welcome to our example introducing Amazon SageMaker's Linear Learner Algorithm!  Today, we're analyzing the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset which consists of images of handwritten digits, from zero to nine.  We'll use the individual pixel values from each 28 x 28 grayscale image to predict a yes or no label of whether the digit is a 0 or some other digit (1, 2, 3, ... 9).
# 
# The method that we'll use is a linear binary classifier.  Linear models are supervised learning algorithms used for solving either classification or regression problems.  As input, the model is given labeled examples ( **`x`**, `y`). **`x`** is a high dimensional vector and `y` is a numeric label.  Since we are doing binary classification, the algorithm expects the label to be either 0 or 1 (but Amazon SageMaker Linear Learner also supports regression on continuous values of `y`).  The algorithm learns a linear function, or linear threshold function for classification, mapping the vector **`x`** to an approximation of the label `y`.
# 
# Amazon SageMaker's Linear Learner algorithm extends upon typical linear models by training many models in parallel, in a computationally efficient manner.  Each model has a different set of hyperparameters, and then the algorithm finds the set that optimizes a specific criteria.  This can provide substantially more accurate models than typical linear algorithms at the same, or lower, cost.
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Let's start by specifying:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.  Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).
# 

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/linear-mnist'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# ### Data ingestion
# 
# Next, we read the dataset from an online URL into memory, for preprocessing prior to training. This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.
# 

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nwith gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')')


# ### Data inspection
# 
# Once the dataset is imported, it's typical as part of the machine learning process to inspect the data, understand the distributions, and determine what type(s) of preprocessing might be needed. You can perform those tasks right here in the notebook. As an example, let's go ahead and look at one of the digits that is part of the dataset.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)


def show_digit(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)

show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))


# ### Data conversion
# 
# Since algorithms have particular input and output requirements, converting the dataset is also part of the process that a data scientist goes through prior to initiating training. In this particular case, the Amazon SageMaker implementation of Linear Learner takes recordIO-wrapped protobuf, where the data we have today is a pickle-ized numpy array on disk.
# 
# Most of the conversion effort is handled by the Amazon SageMaker Python SDK, imported as `sagemaker` below.
# 

import io
import numpy as np
import sagemaker.amazon.common as smac

vectors = np.array([t.tolist() for t in train_set[0]]).astype('float32')
labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype('float32')

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)


# ## Upload training data
# Now that we've created our recordIO-wrapped protobuf, we'll need to upload it to S3, so that Amazon SageMaker training can use it.
# 

import boto3
import os

key = 'recordio-pb-data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# Let's also setup an output S3 location for the model artifact that will be output as the result of training with the algorithm.
# 

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


# ## Training the linear model
# 
# Once we have the data preprocessed and available in the correct format for training, the next step is to actually train the model using the data. Since this data is relatively small, it isn't meant to show off the performance of the Linear Learner training algorithm, although we have tested it on multi-terabyte datasets.
# 
# Again, we'll use the Amazon SageMaker Python SDK to kick off training, and monitor status until it is completed.  In this example that takes between 7 and 11 minutes.  Despite the dataset being small, provisioning hardware and loading the algorithm container take time upfront.
# 
# First, let's specify our containers.  Since we want this notebook to run in all 4 of Amazon SageMaker's regions, we'll create a small lookup.  More details on algorithm containers can be found in [AWS documentation](https://docs-aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).
# 

containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:latest'}


# Next we'll kick off the base estimator, making sure to pass in the necessary hyperparameters.  Notice:
# - `feature_dim` is set to 784, which is the number of pixels in each 28 x 28 image.
# - `predictor_type` is set to 'binary_classifier' since we are trying to predict whether the image is or is not a 0.
# - `mini_batch_size` is set to 200.  This value can be tuned for relatively minor improvements in fit and speed, but selecting a reasonable value relative to the dataset is appropriate in most cases.
# 

import boto3
import sagemaker

sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)
linear.set_hyperparameters(feature_dim=784,
                           predictor_type='binary_classifier',
                           mini_batch_size=200)

linear.fit({'train': s3_train_data})


# ## Set up hosting for the model
# Now that we've trained our model, we can deploy it behind an Amazon SageMaker real-time hosted endpoint.  This will allow out to make predictions (or inference) from the model dyanamically.
# 
# _Note, Amazon SageMaker allows you the flexibility of importing models trained elsewhere, as well as the choice of not importing models if the target of model creation is AWS Lambda, AWS Greengrass, Amazon Redshift, Amazon Athena, or other deployment target._
# 

linear_predictor = linear.deploy(initial_instance_count=1,
                                 instance_type='ml.c4.xlarge')


# ## Validate the model for use
# Finally, we can now validate the model for use.  We can pass HTTP POST requests to the endpoint to get back predictions.  To make this easier, we'll again use the Amazon SageMaker Python SDK and specify how to serialize requests and deserialize responses that are specific to the algorithm.
# 

from sagemaker.predictor import csv_serializer, json_deserializer

linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer


# Now let's try getting a prediction for a single record.
# 

result = linear_predictor.predict(train_set[0][30:31])
print(result)


# OK, a single prediction works.  We see that for one record our endpoint returned some JSON which contains `predictions`, including the `score` and `predicted_label`.  In this case, `score` will be a continuous value between [0, 1] representing the probability we think the digit is a 0 or not.  `predicted_label` will take a value of either `0` or `1` where (somewhat counterintuitively) `1` denotes that we predict the image is a 0, while `0` denotes that we are predicting the image is not of a 0.
# 
# Let's do a whole batch of images and evaluate our predictive accuracy.
# 

import numpy as np

predictions = []
for array in np.array_split(test_set[0], 100):
    result = linear_predictor.predict(array)
    predictions += [r['predicted_label'] for r in result['predictions']]

predictions = np.array(predictions)


import pandas as pd

pd.crosstab(np.where(test_set[1] == 0, 1, 0), predictions, rownames=['actuals'], colnames=['predictions'])


# As we can see from the confusion matrix above, we predict 931 images of 0 correctly, while we predict 44 images as 0s that aren't, and miss predicting 49 images of 0.
# 

# ### (Optional) Delete the Endpoint
# 
# If you're ready to be done with this notebook, please run the delete_endpoint line in the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

import sagemaker

sagemaker.Session().delete_endpoint(linear_predictor.endpoint)


# # End-to-End Example #1
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
#   2. [Data ingestion](#Data-ingestion)
#   3. [Data inspection](#Data-inspection)
#   4. [Data conversion](#Data-conversion)
# 3. [Training the K-Means model](#Training-the-K-Means-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#   1. [Import model into hosting](#Import-model-into-hosting)
#   2. [Create endpoint configuration](#Create-endpoint-configuration)
#   3. [Create endpoint](#Create-endpoint)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 

# ## Introduction
# 
# Welcome to our first end-to-end example! Today, we're working through a classification problem, specifically of images of handwritten digits, from zero to nine. Let's imagine that this dataset doesn't have labels, so we don't know for sure what the true answer is. In later examples, we'll show the value of "ground truth", as it's commonly known.
# 
# Today, however, we need to get these digits classified without ground truth. A common method for doing this is a set of methods known as "clustering", and in particular, the method that we'll look at today is called k-means clustering. In this method, each point belongs to the cluster with the closest mean, and the data is partitioned into a number of clusters that is specified when framing the problem. In this case, since we know there are 10 clusters, and we have no labeled data (in the way we framed the problem), this is a good fit.
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# Here we set up the linkage and authentication to AWS services. There are two parts to this:
# 
# 1. The role(s) used to give learning and hosting access to your data. See the documentation for how to specify these.
# 1. The S3 bucket name and location that you want to use for training and model data.
# 

from sagemaker import get_execution_role

role = get_execution_role()
bucket='<bucket-name>'


# ### Data ingestion
# 
# Next, we read the dataset from the existing repository into memory, for preprocessing prior to training.  In this case we'll use the MNIST dataset, which contains 70K 28 x 28 pixel images of handwritten digits.  For more details, please see [here](http://yann.lecun.com/exdb/mnist/).
# 
# This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.
# 

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nwith gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')')


# ### Data inspection
# 
# Once the dataset is imported, it's typical as part of the machine learning process to inspect the data, understand the distributions, and determine what type(s) of preprocessing might be needed. You can perform those tasks right here in the notebook. As an example, let's go ahead and look at one of the digits that is part of the dataset.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)


def show_digit(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)

show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))


# ### Data conversion and upload
# 
# Since algorithms have particular input and output requirements, converting the dataset is also part of the process that a data scientist goes through prior to initiating training. In this particular case, the hosted implementation of k-means takes recordIO-wrapped protobuf, where the data we have right now is a pickle-ized numpy array on disk.
# 
# To make this process easier, we'll use a function from the Amazon SageMaker Python SDK.  For this dataset, conversion can take up to one minute.
# 

get_ipython().run_cell_magic('time', '', "from sagemaker.amazon.common import write_numpy_to_dense_tensor\nimport io\nimport boto3\n\ndata_key = 'kmeans_lowlevel_example/data'\ndata_location = 's3://{}/{}'.format(bucket, data_key)\nprint('training data will be uploaded to: {}'.format(data_location))\n\n# Convert the training data into the format required by the SageMaker KMeans algorithm\nbuf = io.BytesIO()\nwrite_numpy_to_dense_tensor(buf, train_set[0], train_set[1])\nbuf.seek(0)\n\nboto3.resource('s3').Bucket(bucket).Object(data_key).upload_fileobj(buf)")


# ## Training the K-Means model
# 
# Once we have the data preprocessed and available in the correct format for training, the next step is to actually train the model using the data. Since this data is relatively small, it isn't meant to show off the performance of the k-means training algorithm.  But Amazon SageMaker's k-means has been tested on, and scales well with, multi-terabyte datasets.
# 
# After setting training parameters, we kick off training, and poll for status until training is completed, which in this example, takes between 7 and 11 minutes.
# 

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\njob_name = \'kmeans-lowlevel-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\nimages = {\'us-west-2\': \'174872318107.dkr.ecr.us-west-2.amazonaws.com/kmeans:latest\',\n          \'us-east-1\': \'382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:latest\',\n          \'us-east-2\': \'404615174143.dkr.ecr.us-east-2.amazonaws.com/kmeans:latest\',\n          \'eu-west-1\': \'438346466558.dkr.ecr.eu-west-1.amazonaws.com/kmeans:latest\'}\nimage = images[boto3.Session().region_name]\n\noutput_location = \'s3://{}/kmeans_example/output\'.format(bucket)\nprint(\'training artifacts will be uploaded to: {}\'.format(output_location))\n\ncreate_training_params = \\\n{\n    "AlgorithmSpecification": {\n        "TrainingImage": image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": output_location\n    },\n    "ResourceConfig": {\n        "InstanceCount": 2,\n        "InstanceType": "ml.c4.8xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "k": "10",\n        "feature_dim": "784",\n        "mini_batch_size": "500",\n        "force_dense": "True"\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 60 * 60\n    },\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": data_location,\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "CompressionType": "None",\n            "RecordWrapperType": "None"\n        }\n    ]\n}\n\n\nsagemaker = boto3.client(\'sagemaker\')\n\nsagemaker.create_training_job(**create_training_params)\n\nstatus = sagemaker.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\nprint(status)\n\ntry:\n    sagemaker.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=job_name)\nfinally:\n    status = sagemaker.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\n    print("Training job ended with status: " + status)\n    if status == \'Failed\':\n        message = sagemaker.describe_training_job(TrainingJobName=job_name)[\'FailureReason\']\n        print(\'Training failed with the following error: {}\'.format(message))\n        raise Exception(\'Training job failed\')')


# ## Set up hosting for the model
# In order to set up hosting, we have to import the model from training to hosting. A common question would be, why wouldn't we automatically go from training to hosting?  And, in fact, the [k-means high-level example](/notebooks/sagemaker-python-sdk/1P_kmeans_highlevel/kmeans_mnist.ipynb) shows the functionality to do that.  For this low-level example though it makes sense to show each step in the process to provide a better understanding of the flexibility available.
# 
# ### Import model into hosting
# Next, you register the model with hosting. This allows you the flexibility of importing models trained elsewhere, as well as the choice of not importing models if the target of model creation is AWS Lambda, AWS Greengrass, Amazon Redshift, Amazon Athena, or other deployment target.
# 

get_ipython().run_cell_magic('time', '', "import boto3\nfrom time import gmtime, strftime\n\n\nmodel_name=job_name\nprint(model_name)\n\ninfo = sagemaker.describe_training_job(TrainingJobName=job_name)\nmodel_data = info['ModelArtifacts']['S3ModelArtifacts']\n\nprimary_container = {\n    'Image': image,\n    'ModelDataUrl': model_data\n}\n\ncreate_model_response = sagemaker.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response['ModelArn'])")


# ### Create endpoint configuration
# Now, we'll create an endpoint configuration which provides the instance type and count for model deployment.
# 

from time import gmtime, strftime

endpoint_config_name = 'KMeansEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sagemaker.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.c4.xlarge',
        'InitialInstanceCount':3,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# ### Create endpoint
# Lastly, the customer creates the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications. This takes 9-11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'KMeansEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = sagemaker.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sagemaker.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\ntry:\n    sagemaker.get_waiter(\'endpoint_in_service\').wait(EndpointName=endpoint_name)\nfinally:\n    resp = sagemaker.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Arn: " + resp[\'EndpointArn\'])\n    print("Create endpoint ended with status: " + status)\n\n    if status != \'InService\':\n        message = sagemaker.describe_endpoint(EndpointName=endpoint_name)[\'FailureReason\']\n        print(\'Training failed with the following error: {}\'.format(message))\n        raise Exception(\'Endpoint creation did not succeed\')')


# ## Validate the model for use
# Finally, we'll validate the model for use. Let's generate a classification for a single observation from the trained model using the endpoint we just created.
# 

# Simple function to create a csv from our numpy array
def np2csv(arr):
    csv = io.BytesIO()
    numpy.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


runtime = boto3.Session().client('runtime.sagemaker')


import json

payload = np2csv(train_set[0][30:31])

response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='text/csv', 
                                   Body=payload)
result = json.loads(response['Body'].read().decode())
print(result)


# OK, a single prediction works.
# 
# Let's do a whole batch and see how well the clustering works.
# 

get_ipython().run_cell_magic('time', '', '\npayload = np2csv(valid_set[0][0:100])\nresponse = runtime.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType=\'text/csv\', \n                                   Body=payload)\nresult = json.loads(response[\'Body\'].read().decode())\nclusters = [p[\'closest_cluster\'] for p in result[\'predictions\']]\n\nfor cluster in range(10):\n    print(\'\\n\\n\\nCluster {}:\'.format(int(cluster)))\n    digits = [ img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster ]\n    height=((len(digits)-1)//5)+1\n    width=5\n    plt.rcParams["figure.figsize"] = (width,height)\n    _, subplots = plt.subplots(height, width)\n    subplots=numpy.ndarray.flatten(subplots)\n    for subplot, image in zip(subplots, digits):\n        show_digit(image, subplot=subplot)\n    for subplot in subplots[len(digits):]:\n        subplot.axis(\'off\')\n\n    plt.show()')


# ### The bottom line
# 
# K-Means clustering is not the best algorithm for image analysis problems, but we do see pretty reasonable clusters being built.
# 

# ### Clean up
# 
# If you're ready to be done with this notebook, make sure run the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

sagemaker.delete_endpoint(EndpointName=endpoint_name)


# # Targeting Direct Marketing with Amazon SageMaker XGBoost
# _**Supervised Learning with Gradient Boosted Trees: A Binary Prediction Problem With Unbalanced Classes**_
# 
# ---
# 
# ---
# 
# ## Contents
# 
# 1. [Background](#Background)
# 1. [Prepration](#Preparation)
# 1. [Data](#Data)
#     1. [Exploration](#Exploration)
#     1. [Transformation](#Transformation)
# 1. [Training](#Training)
# 1. [Hosting](#Hosting)
# 1. [Evaluation](#Evaluation)
# 1. [Exentsions](#Extensions)
# 
# ---
# 
# ## Background
# Direct marketing, either through mail, email, phone, etc., is a common tactic to acquire customers.  Because resources and a customer's attention is limited, the goal is to only target the subset of prospects who are likely to engage with a specific offer.  Predicting those potential customers based on readily available information like demographics, past interactions, and environmental factors is a common machine learning problem.
# 
# This notebook presents an example problem to predict if a customer will enroll for a term deposit at a bank, after one or more phone calls.  The steps include:
# 
# * Preparing your Amazon SageMaker notebook
# * Downloading data from the internet into Amazon SageMaker
# * Investigating and transforming the data so that it can be fed to Amazon SageMaker algorithms
# * Estimating a model using the Gradient Boosting algorithm
# * Evaluating the effectiveness of the model
# * Setting the model up to make on-going predictions
# 
# ---
# 
# ## Preparation
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Let's start by specifying:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.  Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).
# 

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/xgboost-dm'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# Now let's bring in the Python libraries that we'll use throughout the analysis
# 

import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
import matplotlib.pyplot as plt                   # For charts and visualizations
from IPython.display import Image                 # For displaying images in the notebook
from IPython.display import display               # For displaying outputs in the notebook
from sklearn.datasets import dump_svmlight_file   # For outputting data to libsvm format for xgboost
from time import gmtime, strftime                 # For labeling SageMaker models, endpoints, etc.
import sys                                        # For writing outputs to notebook
import math                                       # For ceiling function
import json                                       # For parsing hosting outputs
import os                                         # For manipulating filepath names
import sagemaker                                  # Amazon SageMaker's Python SDK provides many helper functions
from sagemaker.predictor import csv_serializer    # Converts strings for HTTP POST requests on inference


# ---
# 
# ## Data
# Let's start by downloading the [direct marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) from UCI's ML Repository.
# 

get_ipython().system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip')
get_ipython().system('unzip -o bank-additional.zip')


# Now lets read this into a Pandas data frame and take a look.
# 

data = pd.read_csv('./bank-additional/bank-additional-full.csv', sep=';')
pd.set_option('display.max_columns', 500)     # Make sure we can see all of the columns
pd.set_option('display.max_rows', 20)         # Keep the output on one page
data


# Let's talk about the data.  At a high level, we can see:
# 
# * We have a little over 40K customer records, and 20 features for each customer
# * The features are mixed; some numeric, some categorical
# * The data appears to be sorted, at least by `time` and `contact`, maybe more
# 
# _**Specifics on each of the features:**_
# 
# *Demographics:*
# * `age`: Customer's age (numeric)
# * `job`: Type of job (categorical: 'admin.', 'services', ...)
# * `marital`: Marital status (categorical: 'married', 'single', ...)
# * `education`: Level of education (categorical: 'basic.4y', 'high.school', ...)
# 
# *Past customer events:*
# * `default`: Has credit in default? (categorical: 'no', 'unknown', ...)
# * `housing`: Has housing loan? (categorical: 'no', 'yes', ...)
# * `loan`: Has personal loan? (categorical: 'no', 'yes', ...)
# 
# *Past direct marketing contacts:*
# * `contact`: Contact communication type (categorical: 'cellular', 'telephone', ...)
# * `month`: Last contact month of year (categorical: 'may', 'nov', ...)
# * `day_of_week`: Last contact day of the week (categorical: 'mon', 'fri', ...)
# * `duration`: Last contact duration, in seconds (numeric). Important note: If duration = 0 then `y` = 'no'.
#  
# *Campaign information:*
# * `campaign`: Number of contacts performed during this campaign and for this client (numeric, includes last contact)
# * `pdays`: Number of days that passed by after the client was last contacted from a previous campaign (numeric)
# * `previous`: Number of contacts performed before this campaign and for this client (numeric)
# * `poutcome`: Outcome of the previous marketing campaign (categorical: 'nonexistent','success', ...)
# 
# *External environment factors:*
# * `emp.var.rate`: Employment variation rate - quarterly indicator (numeric)
# * `cons.price.idx`: Consumer price index - monthly indicator (numeric)
# * `cons.conf.idx`: Consumer confidence index - monthly indicator (numeric)
# * `euribor3m`: Euribor 3 month rate - daily indicator (numeric)
# * `nr.employed`: Number of employees - quarterly indicator (numeric)
# 
# *Target variable:*
# * `y`: Has the client subscribed a term deposit? (binary: 'yes','no')
# 

# ### Exploration
# Let's start exploring the data.  First, let's understand how the features are distributed.
# 

# Frequency tables for each categorical feature
for column in data.select_dtypes(include=['object']).columns:
    display(pd.crosstab(index=data[column], columns='% observations', normalize='columns'))

# Histograms for each numeric features
display(data.describe())
get_ipython().run_line_magic('matplotlib', 'inline')
hist = data.hist(bins=30, sharey=True, figsize=(10, 10))


# Notice that:
# 
# * Almost 90% of the values for our target variable `y` are "no", so most customers did not subscribe to a term deposit.
# * Many of the predictive features take on values of "unknown".  Some are more common than others.  We should think carefully as to what causes a value of "unknown" (are these customers non-representative in some way?) and how we that should be handled.
#   * Even if "unknown" is included as it's own distinct category, what does it mean given that, in reality, those observations likely fall within one of the other categories of that feature?
# * Many of the predictive features have categories with very few observations in them.  If we find a small category to be highly predictive of our target outcome, do we have enough evidence to make a generalization about that?
# * Contact timing is particularly skewed.  Almost a third in May and less than 1% in December.  What does this mean for predicting our target variable next December?
# * There are no missing values in our numeric features.  Or missing values have already been imputed.
#   * `pdays` takes a value near 1000 for almost all customers.  Likely a placeholder value signifying no previous contact.
# * Several numeric features have a very long tail.  Do we need to handle these few observations with extremely large values differently?
# * Several numeric features (particularly the macroeconomic ones) occur in distinct buckets.  Should these be treated as categorical?
# 
# Next, let's look at how our features relate to the target that we are attempting to predict.

for column in data.select_dtypes(include=['object']).columns:
    if column != 'y':
        display(pd.crosstab(index=data[column], columns=data['y'], normalize='columns'))

for column in data.select_dtypes(exclude=['object']).columns:
    print(column)
    hist = data[[column, 'y']].hist(by='y', bins=30)
    plt.show()


# Notice that:
# 
# * Customers who are-- "blue-collar", "married", "unknown" default status, contacted by "telephone", and/or in "may" are a substantially lower portion of "yes" than "no" for subscribing.
# * Distributions for numeric variables are different across "yes" and "no" subscribing groups, but the relationships may not be straightforward or obvious.
# 
# Now let's look at how our features relate to one another.
# 

display(data.corr())
pd.plotting.scatter_matrix(data, figsize=(12, 12))
plt.show()


# Notice that:
# * Features vary widely in their relationship with one another.  Some with highly negative correlation, others with highly positive correlation.
# * Relationships between features is non-linear and discrete in many cases.
# 

# ### Transformation
# 
# Cleaning up data is part of nearly every machine learning project.  It arguably presents the biggest risk if done incorrectly and is one of the more subjective aspects in the process.  Several common techniques include:
# 
# * Handling missing values: Some machine learning algorithms are capable of handling missing values, but most would rather not.  Options include:
#  * Removing observations with missing values: This works well if only a very small fraction of observations have incomplete information.
#  * Removing features with missing values: This works well if there are a small number of features which have a large number of missing values.
#  * Imputing missing values: Entire [books](https://www.amazon.com/Flexible-Imputation-Missing-Interdisciplinary-Statistics/dp/1439868247) have been written on this topic, but common choices are replacing the missing value with the mode or mean of that column's non-missing values.
# * Converting categorical to numeric: The most common method is one hot encoding, which for each feature maps every distinct value of that column to its own feature which takes a value of 1 when the categorical feature is equal to that value, and 0 otherwise.
# * Oddly distributed data: Although for non-linear models like Gradient Boosted Trees, this has very limited implications, parametric models like regression can produce wildly inaccurate estimates when fed highly skewed data.  In some cases, simply taking the natural log of the features is sufficient to produce more normally distributed data.  In others, bucketing values into discrete ranges is helpful.  These buckets can then be treated as categorical variables and included in the model when one hot encoded.
# * Handling more complicated data types: Mainpulating images, text, or data at varying grains is left for other notebook templates.
# 
# Luckily, some of these aspects have already been handled for us, and the algorithm we are showcasing tends to do well at handling sparse or oddly distributed data.  Therefore, let's keep pre-processing simple.
# 

data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)                                 # Indicator variable to capture when pdays takes a value of 999
data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)   # Indicator for individuals not actively employed
model_data = pd.get_dummies(data)                                                                  # Convert categorical variables to sets of indicators


# Another question to ask yourself before building a model is whether certain features will add value in your final use case.  For example, if your goal is to deliver the best prediction, then will you have access to that data at the moment of prediction?  Knowing it's raining is highly predictive for umbrella sales, but forecasting weather far enough out to plan inventory on umbrellas is probably just as difficult as forecasting umbrella sales without knowledge of the weather.  So, including this in your model may give you a false sense of precision.
# 
# Following this logic, let's remove the economic features and `duration` from our data as they would need to be forecasted with high precision to use as inputs in future predictions.
# 
# Even if we were to use values of the economic indicators from the previous quarter, this value is likely not as relevant for prospects contacted early in the next quarter as those contacted later on.
# 

model_data = model_data.drop(['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)


# When building a model whose primary goal is to predict a target value on new data, it is important to understand overfitting.  Supervised learning models are designed to minimize error between their predictions of the target value and actuals, in the data they are given.  This last part is key, as frequently in their quest for greater accuracy, machine learning models bias themselves toward picking up on minor idiosyncrasies within the data they are shown.  These idiosyncrasies then don't repeat themselves in subsequent data, meaning those predictions can actually be made less accurate, at the expense of more accurate predictions in the training phase.
# 
# The most common way of preventing this is to build models with the concept that a model shouldn't only be judged on its fit to the data it was trained on, but also on "new" data.  There are several different ways of operationalizing this, holdout validation, cross-validation, leave-one-out validation, etc.  For our purposes, we'll simply randomly split the data into 3 uneven groups.  The model will be trained on 70% of data, it will then be evaluated on 20% of data to give us an estimate of the accuracy we hope to have on "new" data, and 10% will be held back as a final testing dataset which will be used later on.
# 

train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])   # Randomly sort the data then split out first 70%, second 20%, and last 10%


# Amazon SageMaker's XGBoost container expects data in the libSVM data format.  This expects features and the target variable to be provided as separate arguments.  Let's split these apart.  Notice that although repetitive it's easiest to do this after the train|validation|test split rather than before.  This avoids any misalignment issues due to random reordering.
# 

dump_svmlight_file(X=train_data.drop(['y_no', 'y_yes'], axis=1), y=train_data['y_yes'], f='train.libsvm')
dump_svmlight_file(X=validation_data.drop(['y_no', 'y_yes'], axis=1), y=validation_data['y_yes'], f='validation.libsvm')
dump_svmlight_file(X=test_data.drop(['y_no', 'y_yes'], axis=1), y=test_data['y_yes'], f='test.libsvm')


# Now we'll copy the file to S3 for Amazon SageMaker's managed training to pickup.
# 

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.libsvm')).upload_file('train.libsvm')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.libsvm')).upload_file('validation.libsvm')


# ---
# 
# ## Training
# Now we know that most of our features have skewed distributions, some are highly correlated with one another, and some appear to have non-linear relationships with our target variable.  Also, for targeting future prospects, good predictive accuracy is preferred to being able to explain why that prospect was targeted.  Taken together, these aspects make gradient boosted trees a good candidate algorithm.
# 
# There are several intricacies to understanding the algorithm, but at a high level, gradient boosted trees works by combining predictions from many simple models, each of which tries to address the weaknesses of the previous models.  By doing this the collection of simple models can actually outperform large, complex models.  Other Amazon SageMaker notebooks elaborate on gradient boosting trees further and how they differ from similar algorithms.
# 
# `xgboost` is an extremely popular, open-source package for gradient boosted trees.  It is computationally powerful, fully featured, and has been successfully used in many machine learning competitions.  Let's start with a simple `xgboost` model, trained using Amazon SageMaker's managed, distributed training framework.
# 
# First we'll need to specify the ECR container location for Amazon SageMaker's implementation of XGBoost.
# 

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}


# Then, because we're training with the libSVM file format, we'll create `s3_input`s that our training function can use as a pointer to the files in S3, which also specify that the content type is libSVM.
# 

s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='libsvm')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='libsvm')


# First we'll need to specify training parameters to the estimator.  This includes:
# 1. The `xgboost` algorithm container
# 1. The IAM role to use
# 1. Training instance type and count
# 1. S3 location for output data
# 1. Algorithm hyperparameters
# 
# And then a `.fit()` function which specifies:
# 1. S3 location for output data.  In this case we have both a training and validation set which are passed in.
# 

sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_class=1, 
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 


# ---
# 
# ## Hosting
# Now that we've trained the `xgboost` algorithm on our data, let's deploy a model that's hosted behind a real-time endpoint.
# 

xgb_predictor = xgb.deploy(initial_instance_count=1,
                           instance_type='ml.c4.xlarge')


# ---
# 
# ## Evaluation
# There are many ways to compare the performance of a machine learning model, but let's start by simply comparing actual to predicted values.  In this case, we're simply predicting whether the customer subscribed to a term deposit (`1`) or not (`0`), which produces a simple confusion matrix.
# 
# First we'll need to determine how we pass data into and receive data from our endpoint.  Our data is currently stored as NumPy arrays in memory of our notebook instance.  To send it in an HTTP POST request, we'll serialize it as a CSV string and then decode the resulting CSV.
# 

xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer


# Now, we'll use a simple function to:
# 1. Loop over our test dataset
# 1. Split it into mini-batches of rows 
# 1. Convert those mini-batchs to CSV string payloads
# 1. Retrieve mini-batch predictions by invoking the XGBoost endpoint
# 1. Collect predictions and convert from the CSV output our model provides into a NumPy array
# 

def predict(data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:], sep=',')

predictions = predict(test_data.drop(['y_no', 'y_yes'], axis=1).as_matrix())


# Now we'll check our confusion matrix to see how well we predicted versus actuals.
# 

pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions), rownames=['actuals'], colnames=['predictions'])


# So, of the ~3700 potential customers we predicted would subscribe, 428 of them actually did.  We also had 55 subscribers who subscribed that we did not predict would.  This is less than desirable, but the model can (and should) be tuned to improve this.  Most importantly, note that with minimal effort, our model produced accuracies similar to those published [here](http://media.salford-systems.com/video/tutorial/2015/targeted_marketing.pdf).
# 
# _Note that because there is some element of randomness in the algorithm's subsample, your results may differ slightly from the text written above._
# 

# ---
# 
# ## Extensions
# 
# This example analyzed a relatively small dataset, but utilized Amazon SageMaker features such as distributed, managed training and real-time model hosting, which could easily be applied to much larger problems.  In order to improve predictive accuracy further, we could explore techniques like hyperparameter tuning, as well as spend more time engineering features by hand.  In a real-worl scenario we may also look for additional datasets to include which contain customer information not available in our initial dataset.
# 

# ### (Optional) Clean-up
# 
# If you are done with this notebook, please run the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)


# # Breast Cancer Prediction 
# _**Predict Breast Cancer using SageMaker's Linear-Learner with features derived from images of Breast Mass**_
# 
# ---
# 
# ---
# 
# ## Contents
# 
# 1. [Background](#Background)
# 1. [Setup](#Setup)
# 1. [Data](#Data)
# 1. [Train](#Train)
# 1. [Host](#Host)
# 1. [Predict](#Predict)
# 1. [Extensions](#Extensions)
# 
# ---
# 
# ## Background
# This notebook illustrates how one can use SageMaker's algorithms for solving applications which require `linear models` for prediction. For this illustration, we have taken an example for breast cancer prediction using UCI'S breast cancer diagnostic data set available at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29. The data set is also available on Kaggle at https://www.kaggle.com/uciml/breast-cancer-wisconsin-data. The purpose here is to use this data set to build a predictve model of whether a breast mass image indicates benign or malignant tumor. The data set will be used to illustrate
# 
# * Basic setup for using SageMaker.
# * converting datasets to protobuf format used by the Amazon SageMaker algorithms and uploading to S3. 
# * Training SageMaker's linear learner on the data set.
# * Hosting the trained model.
# * Scoring using the trained model.
# 
# 
# 
# ---
# 
# ## Setup
# 
# Let's start by specifying:
# 
# * The SageMaker role arn used to give learning and hosting access to your data. The snippet below will use the same role used by your SageMaker notebook instance, if you're using other.  Otherwise, specify the full ARN of a role with the SageMakerFullAccess policy attached.
# * The S3 bucket that you want to use for training and storing model objects.
# 

import os
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

bucket = '<your_s3_bucket_name_here>'# enter your s3 bucket where you will copy data and model artifacts
prefix = 'sagemaker/breast_cancer_prediction' # place to upload training files within the bucket


# Now we'll import the Python libraries we'll need.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json
import sagemaker.amazon.common as smac


# ---
# ## Data
# 
# Data Source: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
#         https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# 
# Let's download the data and save it in the local folder with the name data.csv and take a look at it.
# 

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# specify columns extracted from wbdc.names
data.columns = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
                "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                "concave points_worst","symmetry_worst","fractal_dimension_worst"] 

# save the data
data.to_csv("data.csv", sep=',', index=False)

# print the shape of the data file
print(data.shape)

# show the top few rows
display(data.head())

# describe the data object
display(data.describe())

# we will also summarize the categorical field diganosis 
display(data.diagnosis.value_counts())


# #### Key observations:
# * Data has 569 observations and 32 columns.
# * First field is 'id'.
# * Second field, 'diagnosis', is an indicator of the actual diagnosis ('M' = Malignant; 'B' = Benign).
# * There are 30 other numeric features available for prediction.
# 

# ## Create Features and Labels
# #### Split the data into 80% training, 10% validation and 10% testing.
# 

rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]

train_y = ((data_train.iloc[:,1] == 'M') +0).as_matrix();
train_X = data_train.iloc[:,2:].as_matrix();

val_y = ((data_val.iloc[:,1] == 'M') +0).as_matrix();
val_X = data_val.iloc[:,2:].as_matrix();

test_y = ((data_test.iloc[:,1] == 'M') +0).as_matrix();
test_X = data_test.iloc[:,2:].as_matrix();


# Now, we'll convert the datasets to the recordIO-wrapped protobuf format used by the Amazon SageMaker algorithms, and then upload this data to S3.  We'll start with training data.
# 

train_file = 'linear_train.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f)


# Next we'll convert and upload the validation dataset.
# 

validation_file = 'linear_validation.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', train_file)).upload_fileobj(f)


# ---
# ## Train
# 
# Now we can begin to specify our linear model.  Amazon SageMaker's Linear Learner actually fits many models in parallel, each with slightly different hyperparameters, and then returns the one with the best fit.  This functionality is automatically enabled.  We can influence this using parameters like:
# 
# - `num_models` to increase to total number of models run.  The specified parameters will always be one of those models, but the algorithm also chooses models with nearby parameter values in order to find a solution nearby that may be more optimal.  In this case, we're going to use the max of 32.
# - `loss` which controls how we penalize mistakes in our model estimates.  For this case, let's use absolute loss as we haven't spent much time cleaning the data, and absolute loss will be less sensitive to outliers.
# - `wd` or `l1` which control regularization.  Regularization can prevent model overfitting by preventing our estimates from becoming too finely tuned to the training data, which can actually hurt generalizability.  In this case, we'll leave these parameters as their default "auto" though.
# 

# ### Specify container images used for training and hosting SageMaker's linear-learner
# 

# See 'Algorithms Provided by Amazon SageMaker: Common Parameters' in the SageMaker documentation for an explanation of these values.
containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:latest'}


linear_job = 'linear-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())



print("Job name is:", linear_job)

linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": linear_job,
    "AlgorithmSpecification": {
        "TrainingImage": containers[boto3.Session().region_name],
        "TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.c4.2xlarge",
        "VolumeSizeInGB": 10
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "ShardedByS3Key"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        }

    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/".format(bucket, prefix)
    },
    "HyperParameters": {
        "feature_dim": "30",
        "mini_batch_size": "100",
        "predictor_type": "regressor",
        "epochs": "10",
        "num_models": "32",
        "loss": "absolute_loss"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}


# Now let's kick off our training job in SageMaker's distributed, managed training, using the parameters we just created.  Because training is managed, we don't have to wait for our job to finish to continue, but for this case, let's use boto3's 'training_job_completed_or_stopped' waiter so we can ensure that the job has been started.
# 

get_ipython().run_cell_magic('time', '', "\nregion = boto3.Session().region_name\nsm = boto3.client('sagemaker')\n\nsm.create_training_job(**linear_training_params)\n\nstatus = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']\nprint(status)\nsm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)\nif status == 'Failed':\n    message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']\n    print('Training failed with the following error: {}'.format(message))\n    raise Exception('Training job failed')")


# ---
# ## Host
# 
# Now that we've trained the linear algorithm on our data, let's setup a model which can later be hosted.  We will:
# 1. Point to the scoring container
# 1. Point to the model.tar.gz that came from training
# 1. Create the hosting model
# 

linear_hosting_container = {
    'Image': containers[boto3.Session().region_name],
    'ModelDataUrl': sm.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
}

create_model_response = sm.create_model(
    ModelName=linear_job,
    ExecutionRoleArn=role,
    PrimaryContainer=linear_hosting_container)

print(create_model_response['ModelArn'])


# Once we've setup a model, we can configure what our hosting endpoints should be.  Here we specify:
# 1. EC2 instance type to use for hosting
# 1. Initial number of instances
# 1. Our hosting model name
# 

linear_endpoint_config = 'linear-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(linear_endpoint_config)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=linear_endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.c4.2xlarge',
        'InitialInstanceCount': 1,
        'ModelName': linear_job,
        'VariantName': 'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# Now that we've specified how our endpoint should be configured, we can create them.  This can be done in the background, but for now let's run a loop that updates us on the status of the endpoints so that we know when they are ready for use.
# 

get_ipython().run_cell_magic('time', '', '\nlinear_endpoint = \'linear-endpoint-\' + time.strftime("%Y%m%d%H%M", time.gmtime())\nprint(linear_endpoint)\ncreate_endpoint_response = sm.create_endpoint(\n    EndpointName=linear_endpoint,\n    EndpointConfigName=linear_endpoint_config)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm.describe_endpoint(EndpointName=linear_endpoint)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nsm.get_waiter(\'endpoint_in_service\').wait(EndpointName=linear_endpoint)\n\nresp = sm.describe_endpoint(EndpointName=linear_endpoint)\nstatus = resp[\'EndpointStatus\']\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)\n\nif status != \'InService\':\n    raise Exception(\'Endpoint creation did not succeed\')')


# ## Predict
# ### Predict on Test Data
# 
# Now that we have our hosted endpoint, we can generate statistical predictions from it.  Let's predict on our test dataset to understand how accurate our model is.
# 
# There are many metrics to measure classification accuracy.  Common examples include include:
# - Precision
# - Recall
# - F1 measure
# - Area under the ROC curve - AUC
# - Total Classification Accuracy 
# - Mean Absolute Error
# 
# For our example, we'll keep things simple and use total clssification accuracy as our metric of choice. We will also evalute  Mean Absolute  Error (MAE) as the linear-learner has been optimized using this metric, not necessarily because it is a relevant metric from an application point of view. We'll compare the performance of the linear-learner against a naive benchmark prediction which uses majority class observed in the training data set for prediction on the test data.
# 
# 
# 

# ### Function to convert an array to a csv
# 

def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


# Next, we'll invoke the endpoint to get predictions.
# 

runtime= boto3.client('runtime.sagemaker')

payload = np2csv(test_X)
response = runtime.invoke_endpoint(EndpointName=linear_endpoint,
                                   ContentType='text/csv',
                                   Body=payload)
result = json.loads(response['Body'].read().decode())
test_pred = np.array([r['score'] for r in result['predictions']])


# Let's compare linear learner based mean absolute prediction errors from a baseline prediction which uses majority class to predict every instance.
# 

test_mae_linear = np.mean(np.abs(test_y - test_pred))
test_mae_baseline = np.mean(np.abs(test_y - np.median(train_y))) ## training median as baseline predictor

print("Test MAE Baseline :", round(test_mae_baseline, 3))
print("Test MAE Linear:", round(test_mae_linear,3))


# Let's compare predictive accuracy using a classification threshold of 0.5 for the predicted and compare against the majority class prediction from training data set
# 

test_pred_class = (test_pred > 0.5)+0;
test_pred_baseline = np.repeat(np.median(train_y), len(test_y))

prediction_accuracy = np.mean((test_y == test_pred_class))*100
baseline_accuracy = np.mean((test_y == test_pred_baseline))*100

print("Prediction Accuracy:", round(prediction_accuracy,1), "%")
print("Baseline Accuracy:", round(baseline_accuracy,1), "%")


# ###### Run the cell below to delete endpoint once you are done.
# 

sm.delete_endpoint(EndpointName=linear_endpoint)


# ---
# ## Extensions
# 
# - Our linear model does a good job of predicting breast cancer and has an overall accuracy of close to 92%. We can re-run the model with different values of the hyper-parameters, loss functions etc and see if we get improved prediction. Re-running the model with further tweaks to these hyperparameters may provide more accurate out-of-sample predictions.
# - We also did not do much feature engineering. We can create additional features by considering cross-product/intreaction of multiple features, squaring or raising higher powers of the features to induce non-linear effects, etc. If we expand the features using non-linear terms and interactions, we can then tweak the regulaization parameter to optimize the expanded model and hence generate improved forecasts.
# - As a further extension, we can use many of non-linear models available through SageMaker such as XGBoost, MXNet etc.
# 

# ## Sentiment Analysis with MXNet and Gluon
# 
# This tutorial will show how to train and test a Sentiment Analysis (Text Classification) model on SageMaker using MXNet and the Gluon API.
# 
# 

import os
import boto3
import sagemaker
from sagemaker.mxnet import MXNet
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()


# ## Download training and test data
# 

# In this notebook, we will train the **Sentiment Analysis** model on [SST-2 dataset (Stanford Sentiment Treebank 2)](https://nlp.stanford.edu/sentiment/index.html). The dataset consists of movie reviews with one sentence per review. Classification involves detecting positive/negative reviews.  
# We will download the preprocessed version of this dataset from the links below. Each line in the dataset has space separated tokens, the first token being the label: 1 for positive and 0 for negative.
# 

get_ipython().run_cell_magic('bash', '', 'mkdir data\ncurl https://raw.githubusercontent.com/saurabh3949/Text-Classification-Datasets/master/stsa.binary.phrases.train > data/train\ncurl https://raw.githubusercontent.com/saurabh3949/Text-Classification-Datasets/master/stsa.binary.test > data/test ')


# ## Uploading the data
# 
# We use the `sagemaker.Session.upload_data` function to upload our datasets to an S3 location. The return value `inputs` identifies the location -- we will use this later when we start the training job.
# 

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/sentiment')


# ## Implement the training function
# 
# We need to provide a training script that can run on the SageMaker platform. The training scripts are essentially the same as one you would write for local training, except that you need to provide a `train` function. When SageMaker calls your function, it will pass in arguments that describe the training environment. Check the script below to see how this works.
# 
# The script here is a simplified implementation of ["Bag of Tricks for Efficient Text Classification"](https://arxiv.org/abs/1607.01759), as implemented by Facebook's [FastText](https://github.com/facebookresearch/fastText/) for text classification. The model maps each word to a vector and averages vectors of all the words in a sentence to form a hidden representation of the sentence, which is inputted to a softmax classification layer. Please refer to the paper for more details.
# 

get_ipython().system("cat 'sentiment.py'")


# ## Run the training script on SageMaker
# 
# The ```MXNet``` class allows us to run our training function on SageMaker infrastructure. We need to configure it with our training script, an IAM role, the number of training instances, and the training instance type. In this case we will run our training job on a single c4.2xlarge instance. 
# 

m = MXNet("sentiment.py", 
          role=role, 
          train_instance_count=1, 
          train_instance_type="ml.c4.2xlarge",
          hyperparameters={'batch_size': 8, 
                         'epochs': 2, 
                         'learning_rate': 0.01, 
                         'embedding_size': 50, 
                         'log_interval': 1000})


# After we've constructed our `MXNet` object, we can fit it using the data we uploaded to S3. SageMaker makes sure our data is available in the local filesystem, so our training script can simply read the data from disk.
# 

m.fit(inputs)


# As can be seen from the logs, we get > 80% accuracy on the test set using the above hyperparameters.
# 
# After training, we use the MXNet object to build and deploy an MXNetPredictor object. This creates a SageMaker endpoint that we can use to perform inference. 
# 
# This allows us to perform inference on json encoded string array. 
# 

predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


# The predictor runs inference on our input data and returns the predicted sentiment (1 for positive and 0 for negative).
# 

data = ["this movie was extremely good .",
        "the plot was very boring .",
        "this film is so slick , superficial and trend-hoppy .",
        "i just could not watch it till the end .",
        "the movie was so enthralling !"]

response = predictor.predict(data)
print response


# ## Cleanup
# 
# After you have finished with this example, remember to delete the prediction endpoint to release the instance(s) associated with it.
# 

sagemaker.Session().delete_endpoint(predictor.endpoint)


# # Amazon SageMaker XGBoost Bring Your Own Model
# _**Hosting a Pre-Trained scikit-learn Model in Amazon SageMaker XGBoost Algorithm Container**_
# 
# ---
# 
# ---
# 
# ## Contents
# 
# 1. [Background](#Background)
# 1. [Setup](#Setup)
# 1. [Optionally, train a scikit learn XGBoost model](#Optionally,-train-a-scikit-learn-XGBoost-model)
# 1. [Upload the pre-trained model to S3](#Upload-the-pre-trained-model-to-S3)
# 1. [Set up hosting for the model](#Set-up-hosting-for-the-model)
# 1. [Validate the model for use](#Validate-the-model-for-use)
# 
# 
# 
# 
# ---
# ## Background
# 
# Amazon SageMaker includes functionality to support a hosted notebook environment, distributed, serverless training, and real-time hosting. We think it works best when all three of these services are used together, but they can also be used independently.  Some use cases may only require hosting.  Maybe the model was trained prior to Amazon SageMaker existing, in a different service.
# 
# This notebook shows how to use a pre-existing scikit-learn model with the Amazon SageMaker XGBoost Algorithm container to quickly create a hosted endpoint for that model.
# 
# ---
# ## Setup
# 
# Let's start by specifying:
# 
# * AWS region.
# * The IAM role arn used to give learning and hosting access to your data. See the documentation for how to specify these.
# * The S3 bucket that you want to use for training and model data.
# 

get_ipython().run_cell_magic('time', '', "\nimport os\nimport boto3\nimport re\nimport json\nfrom sagemaker import get_execution_role\n\nregion = boto3.Session().region_name\n\nrole = get_execution_role()\n\nbucket='<s3 bucket>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/xgboost-byo'\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)\n# customize to your bucket where you have stored the data")


# ## Optionally, train a scikit learn XGBoost model
# 
# These steps are optional and are needed to generate the scikit-learn model that will eventually be hosted using the SageMaker Algorithm contained. 
# 
# ### Install XGboost
# Note that for conda based installation, you'll need to change the Notebook kernel to the environment with conda and Python3. 
# 

get_ipython().system('conda install -y -c conda-forge xgboost')


# ### Fetch the dataset
# 

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nf = gzip.open(\'mnist.pkl.gz\', \'rb\')\ntrain_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')\nf.close()')


# ### Prepare the dataset for training
# 

get_ipython().run_cell_magic('time', '', "\nimport struct\nimport io\nimport boto3\n\ndef get_dataset():\n  import pickle\n  import gzip\n  with gzip.open('mnist.pkl.gz', 'rb') as f:\n      u = pickle._Unpickler(f)\n      u.encoding = 'latin1'\n      return u.load()")


train_set, valid_set, test_set = get_dataset()

train_X = train_set[0]
train_y = train_set[1]

valid_X = valid_set[0]
valid_y = valid_set[1]

test_X = test_set[0]
test_y = test_set[1]


# ### Train the XGBClassifier
# 

import xgboost as xgb
import sklearn as sk 

bt = xgb.XGBClassifier(max_depth=5,
                       learning_rate=0.2,
                       n_estimators=10,
                       objective='multi:softmax')   # Setup xgboost model
bt.fit(train_X, train_y, # Train it to our data
       eval_set=[(valid_X, valid_y)], 
       verbose=False)


# ### Save the trained model file
# Note that the model file name must satisfy the regular expression pattern: `^[a-zA-Z0-9](-*[a-zA-Z0-9])*;`. The model file also need to tar-zipped. 
# 

model_file_name = "locally-trained-xgboost-model"
bt._Booster.save_model(model_file_name)


get_ipython().system('tar czvf model.tar.gz $model_file_name')


# ## Upload the pre-trained model to S3
# 

fObj = open("model.tar.gz", 'rb')
key= os.path.join(prefix, model_file_name, 'model.tar.gz')
boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fObj)


# ## Set up hosting for the model
# 
# ### Import model into hosting
# This involves creating a SageMaker model from the model file previously uploaded to S3.
# 

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]


get_ipython().run_cell_magic('time', '', 'from time import gmtime, strftime\n\nmodel_name = model_file_name + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nmodel_url = \'https://s3-{}.amazonaws.com/{}/{}\'.format(region,bucket,key)\nsm_client = boto3.client(\'sagemaker\')\n\nprint (model_url)\n\nprimary_container = {\n    \'Image\': container,\n    \'ModelDataUrl\': model_url,\n}\n\ncreate_model_response2 = sm_client.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response2[\'ModelArn\'])')


# ### Create endpoint configuration
# 
# SageMaker supports configuring REST endpoints in hosting with multiple models, e.g. for A/B testing purposes. In order to support this, you can create an endpoint configuration, that describes the distribution of traffic across the models, whether split, shadowed, or sampled in some way. In addition, the endpoint configuration describes the instance type required for model deployment.
# 

from time import gmtime, strftime

endpoint_config_name = 'XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.m4.xlarge',
        'InitialInstanceCount':1,
        'InitialVariantWeight':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# ### Create endpoint
# Lastly, you create the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications. This takes 9-11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = sm_client.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm_client.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nwhile status==\'Creating\':\n    time.sleep(60)\n    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Status: " + status)\n\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)')


# ## Validate the model for use
# Now you can obtain the endpoint from the client library using the result from previous operations and generate classifications from the model using that endpoint.
# 

runtime_client = boto3.client('runtime.sagemaker')


# Lets generate the prediction for a single datapoint. We'll pick one from the test data generated earlier.
# 

import numpy as np
point_X = test_X[0]
point_X = np.expand_dims(point_X, axis=0)
point_y = test_y[0]
np.savetxt("test_point.csv", point_X, delimiter=",")


get_ipython().run_cell_magic('time', '', "import json\n\n\nfile_name = 'test_point.csv' #customize to your test file, will be 'mnist.single.test' if use data above\n\nwith open(file_name, 'r') as f:\n    payload = f.read().strip()\n\nresponse = runtime_client.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType='text/csv', \n                                   Body=payload)\nresult = response['Body'].read().decode('ascii')\nprint('Predicted Class Probabilities: {}.'.format(result))")


# ### Post process the output
# Since the result is a string, let's process it to determine the the output class label. 
# 

floatArr = np.array(json.loads(result))
predictedLabel = np.argmax(floatArr)
print('Predicted Class Label: {}.'.format(predictedLabel))
print('Actual Class Label: {}.'.format(point_y))


# ### (Optional) Delete the Endpoint
# 
# If you're ready to be done with this notebook, please run the delete_endpoint line in the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

sm_client.delete_endpoint(EndpointName=endpoint_name)


# # SageMaker and AWS KMS–Managed Keys
# _**Handling KMS encrypted data with SageMaker model training and encrypting the generated model artifacts**_
# 
# ---
# 
# ---
# 
# ## Contents
# 
# 1. [Background](#Background)
# 1. [Setup](#Setup)
# 1. [Optionally, upload encrypted data files for training](#Optionally,-upload-encrypted-data-files-for-training)
# 1. [Training the XGBoost model](#Training-the-XGBoost-model)
# 1. [Set up hosting for the model](#Set-up-hosting-for-the-model)
# 1. [Validate the model for use](#Validate-the-model-for-use)
# 
# ---
# ## Background
# 
# AWS Key Management Service ([AWS KMS](http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingKMSEncryption.html)) enables 
# Server-side encryption to protect your data at rest. Amazon SageMaker training works with KMS encrypted data if the IAM role used for S3 access has permissions to encrypt and decrypt data with the KMS key. Further, a KMS key can also be used to encrypt the model artifacts at rest using Amazon S3 server-side encryption. In this notebook, we demonstrate SageMaker training with KMS encrypted data. 
# 
# ---
# 
# ## Setup
# 
# ### Prerequisites
# 
# In order to successfully run this notebook, you must first:
# 
# 1. Have an existing KMS key from AWS IAM console or create one ([learn more](http://docs.aws.amazon.com/kms/latest/developerguide/create-keys.html)).
# 2. Allow the IAM role used for SageMaker to encrypt and decrypt data with this key from within applications and when using AWS services integrated with KMS ([learn more](http://docs.aws.amazon.com/console/kms/key-users)).
# 
# We use the `key-id` from the KMS key ARN `arn:aws:kms:region:acct-id:key/key-id`.
# 
# ### General Setup
# Let's start by specifying:
# * AWS region.
# * The IAM role arn used to give learning and hosting access to your data. See the documentation for how to specify these.
# * The S3 bucket that you want to use for training and model data.
# 

get_ipython().run_cell_magic('time', '', "\nimport os\nimport io\nimport boto3\nimport pandas as pd\nimport numpy as np\nimport re\nfrom sagemaker import get_execution_role\n\nregion = boto3.Session().region_name\n\nrole = get_execution_role()\n\nkms_key_id = '<your-kms-key-id>'\n\nbucket='<s3-bucket>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/kms'\n# customize to your bucket where you have stored the data\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)")


# ## Optionally, upload encrypted data files for training
# 
# To demonstrate SageMaker training with KMS encrypted data, we first upload a toy dataset that has Server Side Encryption with customer provided key.
# 
# ### Data ingestion
# 
# We, first, read the dataset from an existing repository into memory. This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as the one used below, reading into memory isn't onerous, though it would be for larger datasets.
# 

from sklearn.datasets import load_boston
boston = load_boston()
X = boston['data']
y = boston['target']
feature_names = boston['feature_names']
data = pd.DataFrame(X, columns=feature_names)
target = pd.DataFrame(y, columns={'MEDV'})
data['MEDV'] = y
local_file_name = 'boston.csv'
data.to_csv(local_file_name, header=False, index=False)


# ### Data preprocessing
# 
# Now that we have the dataset, we need to split it into *train*, *validation*, and *test* datasets which we can use to evaluate the accuracy of the machine learning algorithm. We randomly split the dataset into 60% training, 20% validation and 20% test. Note that SageMaker Xgboost, expects the label column to be the first one in the datasets. So, we'll move the median value column (`MEDV`) from the last to the first position within the `write_file` method below. 
# 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)


def write_file(X, y, fname):
    feature_names = boston['feature_names']
    data = pd.DataFrame(X, columns=feature_names)
    target = pd.DataFrame(y, columns={'MEDV'})
    data['MEDV'] = y
    # bring this column to the front before writing the files
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    data.to_csv(fname, header=False, index=False)


train_file = 'train.csv'
validation_file = 'val.csv'
test_file = 'test.csv'
write_file(X_train, y_train, train_file)
write_file(X_val, y_val, validation_file)
write_file(X_test, y_test, test_file)


# ### Data upload to S3 with Server Side Encryption
# 

s3 = boto3.client('s3')

data_train = open(train_file, 'rb')
key_train = '{}/train/{}'.format(prefix,train_file)


print("Put object...")
s3.put_object(Bucket=bucket,
              Key=key_train,
              Body=data_train,
              ServerSideEncryption='aws:kms',
              SSEKMSKeyId=kms_key_id)
print("Done uploading the training dataset")

data_validation = open(validation_file, 'rb')
key_validation = '{}/validation/{}'.format(prefix,validation_file)

print("Put object...")
s3.put_object(Bucket=bucket,
              Key=key_validation,
              Body=data_validation,
              ServerSideEncryption='aws:kms',
              SSEKMSKeyId=kms_key_id)

print("Done uploading the validation dataset")


# ## Training the SageMaker XGBoost model
# 
# Now that we have our data in S3, we can begin training. We'll use Amazon SageMaker XGboost algorithm as an example to demonstrate model training. Note that nothing needs to be changed in the way you'd call the training algorithm. The only requirement for training to succeed is that the IAM role (`role`) used for S3 access has permissions to encrypt and decrypt data with the KMS key (`kms_key_id`). You can set these permissions using the instructions [here](http://docs.aws.amazon.com/kms/latest/developerguide/key-policies.html#key-policy-default-allow-users). If the permissions aren't set, you'll get the `Data download failed` error.
# 

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]


get_ipython().run_cell_magic('time', '', 'from time import gmtime, strftime\nimport time\n\njob_name = \'xgboost-single-regression\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\ncreate_training_params = \\\n{\n    "AlgorithmSpecification": {\n        "TrainingImage": container,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": bucket_path + "/"+ prefix + "/output"\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.m4.4xlarge",\n        "VolumeSizeInGB": 5\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "max_depth":"5",\n        "eta":"0.2",\n        "gamma":"4",\n        "min_child_weight":"6",\n        "subsample":"0.7",\n        "silent":"0",\n        "objective":"reg:linear",\n        "num_round":"5"\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 86400\n    },\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/"+ prefix + \'/train\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "csv",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/"+ prefix + \'/validation\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "csv",\n            "CompressionType": "None"\n        }\n    ]\n}\n\nclient = boto3.client(\'sagemaker\')\nclient.create_training_job(**create_training_params)\n\ntry:\n    # wait for the job to finish and report the ending status\n    client.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=job_name)\n    training_info = client.describe_training_job(TrainingJobName=job_name)\n    status = training_info[\'TrainingJobStatus\']\n    print("Training job ended with status: " + status)\nexcept:\n    print(\'Training failed to start\')\n     # if exception is raised, that means it has failed\n    message = client.describe_training_job(TrainingJobName=job_name)[\'FailureReason\']\n    print(\'Training failed with the following error: {}\'.format(message))')


# ## Set up hosting for the model
# In order to set up hosting, we have to import the model from training to hosting. 
# 
# ### Import model into hosting
# 
# Register the model with hosting. This allows the flexibility of importing models trained elsewhere.
# 

get_ipython().run_cell_magic('time', '', "import boto3\nfrom time import gmtime, strftime\n\nmodel_name=job_name + '-model'\nprint(model_name)\n\ninfo = client.describe_training_job(TrainingJobName=job_name)\nmodel_data = info['ModelArtifacts']['S3ModelArtifacts']\nprint(model_data)\n\nprimary_container = {\n    'Image': container,\n    'ModelDataUrl': model_data\n}\n\ncreate_model_response = client.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response['ModelArn'])")


# ### Create endpoint configuration
# 
# SageMaker supports configuring REST endpoints in hosting with multiple models, e.g. for A/B testing purposes. In order to support this, customers create an endpoint configuration, that describes the distribution of traffic across the models, whether split, shadowed, or sampled in some way. In addition, the endpoint configuration describes the instance type required for model deployment.
# 

from time import gmtime, strftime

endpoint_config_name = 'XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.m4.xlarge',
        'InitialVariantWeight':1,
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# ### Create endpoint
# Lastly, create the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications. This takes 9-11 minutes to complete.
# 

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-new-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = client.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\n\nprint(\'EndpointArn = {}\'.format(create_endpoint_response[\'EndpointArn\']))\n\n# get the status of the endpoint\nresponse = client.describe_endpoint(EndpointName=endpoint_name)\nstatus = response[\'EndpointStatus\']\nprint(\'EndpointStatus = {}\'.format(status))\n\n\n# wait until the status has changed\nclient.get_waiter(\'endpoint_in_service\').wait(EndpointName=endpoint_name)\n\n\n# print the status of the endpoint\nendpoint_response = client.describe_endpoint(EndpointName=endpoint_name)\nstatus = endpoint_response[\'EndpointStatus\']\nprint(\'Endpoint creation ended with EndpointStatus = {}\'.format(status))\n\nif status != \'InService\':\n    raise Exception(\'Endpoint creation failed.\')')


# ## Validate the model for use
# Finally, you can now validate the model for use. They can obtain the endpoint from the client library using the result from previous operations, and generate classifications from the trained model using that endpoint.
# 

runtime_client = boto3.client('runtime.sagemaker')


import sys
import math
def do_predict(data, endpoint_name, content_type):
    payload = ''.join(data)
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType=content_type, 
                                   Body=payload)
    result = response['Body'].read()
    result = result.decode("utf-8")
    result = result.split(',')
    return result

def batch_predict(data, batch_size, endpoint_name, content_type):
    items = len(data)
    arrs = []
    
    for offset in range(0, items, batch_size):
        if offset+batch_size < items:
            results = do_predict(data[offset:(offset+batch_size)], endpoint_name, content_type)
            arrs.extend(results)
        else:
            arrs.extend(do_predict(data[offset:items], endpoint_name, content_type))
        sys.stdout.write('.')
    return(arrs)


# The following helps us calculate the Median Absolute Percent Error (MdAPE) on the batch dataset. Note that the intent of this example is not to produce the most accurate regressor but to demonstrate how to handle KMS encrypted data with SageMaker. 
# 

get_ipython().run_cell_magic('time', '', "import json\nimport numpy as np\n\n\nwith open('test.csv') as f:\n    lines = f.readlines()\n\n#remove the labels\nlabels = [line.split(',')[0] for line in lines]\nfeatures = [line.split(',')[1:] for line in lines]\n\nfeatures_str = [','.join(row) for row in features]\npreds = batch_predict(features_str, 100, endpoint_name, 'text/csv')\nprint('\\n Median Absolute Percent Error (MdAPE) = ', np.median(np.abs(np.asarray(labels, dtype=float) - np.asarray(preds, dtype=float)) / np.asarray(labels, dtype=float)))")


# ### (Optional) Delete the Endpoint
# 
# If you're ready to be done with this notebook, please run the delete_endpoint line in the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

client.delete_endpoint(EndpointName=endpoint_name)


# # MNIST distributed training  
# 
# The **SageMaker Python SDK** helps you deploy your models for training and hosting in optimized, productions ready containers in SageMaker. The SageMaker Python SDK is easy to use, modular, extensible and compatible with TensorFlow and MXNet. This tutorial focuses on how to create a convolutional neural network model to train the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using **TensorFlow distributed training**.
# 
# 

# ### Set up the environment
# 

import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()


# ### Download the MNIST dataset
# 

import utils
from tensorflow.contrib.learn.python.learn.datasets import mnist
import tensorflow as tf

data_sets = mnist.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)

utils.convert_to(data_sets.train, 'train', 'data')
utils.convert_to(data_sets.validation, 'validation', 'data')
utils.convert_to(data_sets.test, 'test', 'data')


# ### Upload the data
# We use the ```sagemaker.Session.upload_data``` function to upload our datasets to an S3 location. The return value inputs identifies the location -- we will use this later when we start the training job.
# 

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')


# # Construct a script for distributed training 
# Here is the full code for the network model:
# 

get_ipython().system("cat 'mnist.py'")


# The script here is and adaptation of the [TensorFlow MNIST example](https://github.com/tensorflow/models/tree/master/official/mnist). It provides a ```model_fn(features, labels, mode)```, which is used for training, evaluation and inference. 
# 
# ## A regular ```model_fn```
# 
# A regular **```model_fn```** follows the pattern:
# 1. [defines a neural network](https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L96)
# - [applies the ```features``` in the neural network](https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L178)
# - [if the ```mode``` is ```PREDICT```, returns the output from the neural network](https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L186)
# - [calculates the loss function comparing the output with the ```labels```](https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L188)
# - [creates an optimizer and minimizes the loss function to improve the neural network](https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L193)
# - [returns the output, optimizer and loss function](https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L205)
# 
# ## Writing a ```model_fn``` for distributed training
# When distributed training happens, the same neural network will be sent to the multiple training instances. Each instance will predict a batch of the dataset, calculate loss and minimize the optimizer. One entire loop of this process is called **training step**.
# 
# ### Syncronizing training steps
# A [global step](https://www.tensorflow.org/api_docs/python/tf/train/global_step) is a global variable shared between the instances. It necessary for distributed training, so the optimizer will keep track of the number of **training steps** between runs: 
# 
# ```python
# train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
# ```
# 
# That is the only required change for distributed training!
# 

# ## Create a training job using the sagemaker.TensorFlow estimator
# 

from sagemaker.tensorflow import TensorFlow

mnist_estimator = TensorFlow(entry_point='mnist.py',
                             role=role,
                             training_steps=1000, 
                             evaluation_steps=100,
                             train_instance_count=2,
                             train_instance_type='ml.c4.xlarge')

mnist_estimator.fit(inputs)


# The **```fit```** method will create a training job in two **ml.c4.xlarge** instances. The logs above will show the instances doing training, evaluation, and incrementing the number of **training steps**. 
# 
# In the end of the training, the training job will generate a saved model for TF serving.
# 

# # Deploy the trained model to prepare for predictions
# 
# The deploy() method creates an endpoint which serves prediction requests in real-time.
# 

mnist_predictor = mnist_estimator.deploy(initial_instance_count=1,
                                             instance_type='ml.c4.xlarge')


# # Invoking the endpoint
# 

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

for i in range(10):
    data = mnist.test.images[i].tolist()
    tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)
    predict_response = mnist_predictor.predict(tensor_proto)
    
    print("========================================")
    label = np.argmax(mnist.test.labels[i])
    print("label is {}".format(label))
    prediction = predict_response['outputs']['classes']['int64Val'][0]
    print("prediction is {}".format(prediction))


# # Deleting the endpoint
# 

sagemaker.Session().delete_endpoint(mnist_predictor.endpoint)


# # Installing the R kernel
# 
# Installing the R kernel with SageMaker's hosted notebook environment is as simple as:
# 

get_ipython().system('conda install --yes --name JupyterSystemEnv --channel r r-essentials')


# Now just refresh your Jupyter dashboard, and select "R" from the "New" notebook drop-down.
# 

# # Creating Estimators in tf.estimator with Keras
# 
# This tutorial covers how to create your own training script using the building
# blocks provided in `tf.keras`, which will predict the ages of
# [abalones](https://en.wikipedia.org/wiki/Abalone) based on their physical
# measurements. You'll learn how to do the following:
# 
# *   Construct a custom model function
# *   Configure a neural network using `tf.keras`
# *   Choose an appropriate loss function from `tf.losses`
# *   Define a training op for your model
# *   Generate and return predictions
# 

# ## An Abalone Age Predictor
# 
# It's possible to estimate the age of an
# [abalone](https://en.wikipedia.org/wiki/Abalone) (sea snail) by the number of
# rings on its shell. However, because this task requires cutting, staining, and
# viewing the shell under a microscope, it's desirable to find other measurements
# that can predict age.
# 
# The [Abalone Data Set](https://archive.ics.uci.edu/ml/datasets/Abalone) contains
# the following
# [feature data](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names)
# for abalone:
# 
# | Feature        | Description                                               |
# | -------------- | --------------------------------------------------------- |
# | Length         | Length of abalone (in longest direction; in mm)           |
# | Diameter       | Diameter of abalone (measurement perpendicular to length; in mm)|
# | Height         | Height of abalone (with its meat inside shell; in mm)     |
# | Whole Weight   | Weight of entire abalone (in grams)                       |
# | Shucked Weight | Weight of abalone meat only (in grams)                    |
# | Viscera Weight | Gut weight of abalone (in grams), after bleeding          |
# | Shell Weight   | Weight of dried abalone shell (in grams)                  |
# 
# The label to predict is number of rings, as a proxy for abalone age.
# 

# ### Set up the environment¶
# 

import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()


# ### Upload the data to a S3 bucket
# 

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/abalone')


# **sagemaker_session.upload_data** will upload the abalone dataset from your machine to a bucket named **sagemaker-{your aws account number}**, if you don't have this bucket yet, sagemaker_session will create it for you.
# 

# ## Complete source code
# Here is the full code for the network model:
# 

get_ipython().system("cat 'abalone.py'")


# ## Defining a `model_fn`
# 
# The script above implements a `model_fn` as the function responsible for implementing the model for training, evaluation, and prediction. The next section covers how to implement a `model_fn` using `Keras layers`. 
# 
# 
# 
# ### Constructing the `model_fn`
# 
# The basic skeleton for an `model_fn` looks like this:
# 
# ```python
# def model_fn(features, labels, mode, params):
#    # Logic to do the following:
#    # 1. Configure the model via TensorFlow or Keras operations
#    # 2. Define the loss function for training/evaluation
#    # 3. Define the training operation/optimizer
#    # 4. Generate predictions
#    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
#    return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
# ```
# 
# The `model_fn` must accept three arguments:
# 
# *   `features`: A dict containing the features passed to the model via
#     `input_fn`.
# *   `labels`: A `Tensor` containing the labels passed to the model via
#     `input_fn`. Will be empty for `predict()` calls, as these are the values the
#     model will infer.
# *   `mode`: One of the following tf.estimator.ModeKeys string values
#     indicating the context in which the model_fn was invoked:
#     *   `tf.estimator.ModeKeys.TRAIN` The `model_fn` was invoked in training
#         mode, namely via a `train()` call.
#     *   `tf.estimator.ModeKeys.EVAL`. The `model_fn` was invoked in
#         evaluation mode, namely via an `evaluate()` call.
#     *   `tf.estimator.ModeKeys.PREDICT`. The `model_fn` was invoked in
#         predict mode, namely via a `predict()` call.
# 
# `model_fn` may also accept a `params` argument containing a dict of
# hyperparameters used for training (as shown in the skeleton above).
# 
# The body of the function performs the following tasks (described in detail in the
# sections that follow):
# 
# *   Configuring the model for the abalone predictor. This will be a neural
#     network.
# *   Defining the loss function used to calculate how closely the model's
#     predictions match the target values.
# *   Defining the training operation that specifies the `optimizer` algorithm to
#     minimize the loss values calculated by the loss function.
# 
# The `model_fn` must return a tf.estimator.EstimatorSpec
# object, which contains the following values:
# 
# *   `mode` (required). The mode in which the model was run. Typically, you will
#     return the `mode` argument of the `model_fn` here.
# 
# *   `predictions` (required in `PREDICT` mode). A dict that maps key names of
#     your choice to `Tensor`s containing the predictions from the model, e.g.:
# 
#     ```python
#     predictions = {"results": tensor_of_predictions}
#     ```
# 
#     In `PREDICT` mode, the dict that you return in `EstimatorSpec` will then be
#     returned by `predict()`, so you can construct it in the format in which
#     you'd like to consume it.
# 
# 
# *   `loss` (required in `EVAL` and `TRAIN` mode). A `Tensor` containing a scalar
#     loss value: the output of the model's loss function (discussed in more depth
#     later in [Defining loss for the model](https://github.com/tensorflow/tensorflow/blob/eb84435170c694175e38bfa02751c3ef881c7a20/tensorflow/docs_src/extend/estimators.md#defining-loss)) calculated over all
#     the input examples. This is used in `TRAIN` mode for error handling and
#     logging, and is automatically included as a metric in `EVAL` mode.
# 
# *   `train_op` (required only in `TRAIN` mode). An Op that runs one step of
#     training.
# 
# *   `eval_metric_ops` (optional). A dict of name/value pairs specifying the
#     metrics that will be calculated when the model runs in `EVAL` mode. The name
#     is a label of your choice for the metric, and the value is the result of
#     your metric calculation. The tf.metrics
#     module provides predefined functions for a variety of common metrics. The
#     following `eval_metric_ops` contains an `"accuracy"` metric calculated using
#     `tf.metrics.accuracy`:
# 
#     ```python
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(labels, predictions)
#     }
#     ```
# 
#     If you do not specify `eval_metric_ops`, only `loss` will be calculated
#     during evaluation.
# 
# ### Configuring a neural network with `keras layers`
# 
# Constructing a [neural
# network](https://en.wikipedia.org/wiki/Artificial_neural_network) entails
# creating and connecting the input layer, the hidden layers, and the output
# layer.
# 
# The input layer of the neural network then must be connected to one or more
# hidden layers via an [activation
# function](https://en.wikipedia.org/wiki/Activation_function) that performs a
# nonlinear transformation on the data from the previous layer. The last hidden
# layer is then connected to the output layer, the final layer in the model.
# `tf.layers` provides the `tf.layers.dense` function for constructing fully
# connected layers. The activation is controlled by the `activation` argument.
# Some options to pass to the `activation` argument are:
# 
# *   `tf.nn.relu`. The following code creates a layer of `units` nodes fully
#     connected to the previous layer `input_layer` with a
#     [ReLU activation function](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\))
#     (tf.nn.relu):
# 
#     ```python
#     hidden_layer = Dense(10, activation='relu', name='first-layer')(features)
#     ```
# 
# *   `tf.nn.relu6`. The following code creates a layer of `units` nodes fully
#     connected to the previous layer `hidden_layer` with a ReLU activation
#     function:
# 
#     ```python
#     second_hidden_layer = Dense(20, activation='relu', name='first-layer')(hidden_layer)
#     ```
# 
# *   `None`. The following code creates a layer of `units` nodes fully connected
#     to the previous layer `second_hidden_layer` with *no* activation function,
#     just a linear transformation:
# 
#     ```python
#     output_layer = Dense(1, activation='linear')(second_hidden_layer)
#     ```
# 
# Other activation functions are possible, e.g.:
# 
# ```python
# output_layer = Dense(10, activation='sigmoid')(second_hidden_layer)
# ```
# 
# The above code creates the neural network layer `output_layer`, which is fully
# connected to `second_hidden_layer` with a sigmoid activation function
# (tf.sigmoid).
# 
# Putting it all together, the following code constructs a full neural network for
# the abalone predictor, and captures its predictions:
# 
# ```python
# def model_fn(features, labels, mode, params):
#   """Model function for Estimator."""
# 
#   # Connect the first hidden layer to input layer
#   # (features["x"]) with relu activation
#   first_hidden_layer = Dense(10, activation='relu', name='first-layer')(features['x'])
# 
#   # Connect the second hidden layer to first hidden layer with relu
#   second_hidden_layer = Dense(20, activation='relu', name='first-layer')(hidden_layer)
# 
#   # Connect the output layer to second hidden layer (no activation fn)
#   output_layer = Dense(1, activation='linear')(second_hidden_layer)
# 
#   # Reshape output layer to 1-dim Tensor to return predictions
#   predictions = tf.reshape(output_layer, [-1])
#   predictions_dict = {"ages": predictions}
#   ...
# ```
# 
# Here, because you'll be passing the abalone `Datasets` using `numpy_input_fn`
# as shown below, `features` is a dict `{"x": data_tensor}`, so
# `features["x"]` is the input layer. The network contains two hidden
# layers, each with 10 nodes and a ReLU activation function. The output layer
# contains no activation function, and is
# tf.reshape to a one-dimensional
# tensor to capture the model's predictions, which are stored in
# `predictions_dict`.
# 
# ### Defining loss for the model
# 
# The `EstimatorSpec` returned by the `model_fn` must contain `loss`: a `Tensor`
# representing the loss value, which quantifies how well the model's predictions
# reflect the label values during training and evaluation runs. The tf.losses
# module provides convenience functions for calculating loss using a variety of
# metrics, including:
# 
# *   `absolute_difference(labels, predictions)`. Calculates loss using the
#     [absolute-difference
#     formula](https://en.wikipedia.org/wiki/Deviation_statistics#Unsigned_or_absolute_deviation)
#     (also known as L<sub>1</sub> loss).
# 
# *   `log_loss(labels, predictions)`. Calculates loss using the [logistic loss
#     forumula](https://en.wikipedia.org/wiki/Loss_functions_for_classification#Logistic_loss)
#     (typically used in logistic regression).
# 
# *   `mean_squared_error(labels, predictions)`. Calculates loss using the [mean
#     squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE; also
#     known as L<sub>2</sub> loss).
# 
# The following example adds a definition for `loss` to the abalone `model_fn`
# using `mean_squared_error()` (in bold):
# 
# ```python
# def model_fn(features, labels, mode, params):
#   """Model function for Estimator."""
# 
#   # Connect the first hidden layer to input layer
#   # (features["x"]) with relu activation
#     first_hidden_layer = Dense(10, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])
#   
#   # Connect the second hidden layer to first hidden layer with relu
#   second_hidden_layer = Dense(20, activation='relu')(first_hidden_layer)
#   
#   # Connect the output layer to second hidden layer (no activation fn)
#   output_layer = Dense(1, activation='linear')(second_hidden_layer)
# 
#   # Reshape output layer to 1-dim Tensor to return predictions
#   predictions = tf.reshape(output_layer, [-1])
#   predictions_dict = {"ages": predictions}
#   
#   # Calculate loss using mean squared error
#   loss = tf.losses.mean_squared_error(labels, predictions)
#   ...
# ```
# 

# Supplementary metrics for evaluation can be added to an `eval_metric_ops` dict.
# The following code defines an `rmse` metric, which calculates the root mean
# squared error for the model predictions. Note that the `labels` tensor is cast
# to a `float64` type to match the data type of the `predictions` tensor, which
# will contain real values:
# 
# ```python
# eval_metric_ops = {
#     "rmse": tf.metrics.root_mean_squared_error(
#         tf.cast(labels, tf.float64), predictions)
# }
# ```
# 
# ### Defining the training op for the model
# 
# The training op defines the optimization algorithm TensorFlow will use when
# fitting the model to the training data. Typically when training, the goal is to
# minimize loss. A simple way to create the training op is to instantiate a
# `tf.train.Optimizer` subclass and call the `minimize` method.
# 
# The following code defines a training op for the abalone `model_fn` using the
# loss value calculated in [Defining Loss for the Model](https://github.com/tensorflow/tensorflow/blob/eb84435170c694175e38bfa02751c3ef881c7a20/tensorflow/docs_src/extend/estimators.md#defining-loss), the
# learning rate passed to the function in `params`, and the gradient descent
# optimizer. For `global_step`, the convenience function
# tf.train.get_global_step takes care of generating an integer variable:
# 
# ```python
# optimizer = tf.train.GradientDescentOptimizer(
#     learning_rate=params["learning_rate"])
# train_op = optimizer.minimize(
#     loss=loss, global_step=tf.train.get_global_step())
# ```
# 
# ### The complete abalone `model_fn`
# 
# Here's the final, complete `model_fn` for the abalone age predictor. The
# following code configures the neural network; defines loss and the training op;
# and returns a `EstimatorSpec` object containing `mode`, `predictions_dict`, `loss`,
# and `train_op`:
# 
# ```python
# def model_fn(features, labels, mode, params):
#   """Model function for Estimator."""
# 
#   # Connect the first hidden layer to input layer
#   # (features["x"]) with relu activation
#     first_hidden_layer = Dense(10, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])
#   
#   # Connect the second hidden layer to first hidden layer with relu
#   second_hidden_layer = Dense(20, activation='relu')(first_hidden_layer)
#   
#   # Connect the output layer to second hidden layer (no activation fn)
#   output_layer = Dense(1, activation='linear')(second_hidden_layer)
# 
#   # Reshape output layer to 1-dim Tensor to return predictions
#   predictions = tf.reshape(output_layer, [-1])
# 
#   # Provide an estimator spec for `ModeKeys.PREDICT`.
#   if mode == tf.estimator.ModeKeys.PREDICT:
#     return tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions={"ages": predictions})
# 
#   # Calculate loss using mean squared error
#   loss = tf.losses.mean_squared_error(labels, predictions)
# 
#   # Calculate root mean squared error as additional eval metric
#   eval_metric_ops = {
#       "rmse": tf.metrics.root_mean_squared_error(
#           tf.cast(labels, tf.float64), predictions)
#   }
# 
#   optimizer = tf.train.GradientDescentOptimizer(
#       learning_rate=params["learning_rate"])
#   train_op = optimizer.minimize(
#       loss=loss, global_step=tf.train.get_global_step())
# 
#   # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
#   return tf.estimator.EstimatorSpec(
#       mode=mode,
#       loss=loss,
#       train_op=train_op,
#       eval_metric_ops=eval_metric_ops)
# ```
# 

# # Submitting script for training
# 
# We can use the SDK to run our local training script on SageMaker infrastructure.
# 
# 1. Pass the path to the abalone.py file, which contains the functions for defining your estimator, to the sagemaker.TensorFlow init method.
# 2. Pass the S3 location that we uploaded our data to previously to the fit() method.
# 

from sagemaker.tensorflow import TensorFlow

abalone_estimator = TensorFlow(entry_point='abalone.py',
                               role=role,
                               training_steps= 100,                                  
                               evaluation_steps= 100,
                               hyperparameters={'learning_rate': 0.001},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge')

abalone_estimator.fit(inputs)


# `estimator.fit` will deploy a script in a container for training and returns the SageMaker model name using the following arguments:
# 
# *   **`entry_point="abalone.py"`** The path to the script that will be deployed to the container.
# *   **`training_steps=100`** The number of training steps of the training job.
# *   **`evaluation_steps=100`** The number of evaluation steps of the training job.
# *   **`role`**. AWS role that gives your account access to SageMaker training and hosting
# *   **`hyperparameters={'learning_rate' : 0.001}`**. Training hyperparameters. 
# 
# Running the code block above will do the following actions:
# * deploy your script in a container with tensorflow installed
# * copy the data from the bucket to the container
# * instantiate the tf.estimator
# * train the estimator with 10 training steps
# * save the estimator model
# 

# # Submiting a trained model for hosting
# 
# The deploy() method creates an endpoint which serves prediction requests in real-time.
# 

abalone_predictor = abalone_estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')


# # Invoking the endpoint
# 

import tensorflow as tf
import numpy as np

prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=os.path.join('data/abalone_predict.csv'), target_dtype=np.int, features_dtype=np.float32)

data = prediction_set.data[0]
tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)


abalone_predictor.predict(tensor_proto)


# # Deleting the endpoint
# 

sagemaker.Session().delete_endpoint(abalone_predictor.endpoint)


# # A Scientific Deep Dive Into SageMaker LDA
# 
# 1. [Introduction](#Introduction)
# 1. [Setup](#Setup)
# 1. [Data Exploration](#DataExploration)
# 1. [Training](#Training)
# 1. [Inference](#Inference)
# 1. [Epilogue](#Epilogue)
# 

# # Introduction
# ***
# 
# Amazon SageMaker LDA is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. Latent Dirichlet Allocation (LDA) is most commonly used to discover a user-specified number of topics shared by documents within a text corpus. Here each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics. Since the method is unsupervised, the topics are not specified up front, and are not guaranteed to align with how a human may naturally categorize documents. The topics are learned as a probability distribution over the words that occur in each document. Each document, in turn, is described as a mixture of topics.
# 
# This notebook is similar to **LDA-Introduction.ipynb** but its objective and scope are a different. We will be taking a deeper dive into the theory. The primary goals of this notebook are,
# 
# * to understand the LDA model and the example dataset,
# * understand how the Amazon SageMaker LDA algorithm works,
# * interpret the meaning of the inference output.
# 
# Former knowledge of LDA is not required. However, we will run through concepts rather quickly and at least a foundational knowledge of mathematics or machine learning is recommended. Suggested references are provided, as appropriate.
# 

get_ipython().system('conda install -y scipy')


get_ipython().run_line_magic('matplotlib', 'inline')

import os, re, tarfile

import boto3
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# some helpful utility functions are defined in the Python module
# "generate_example_data" located in the same directory as this
# notebook
from generate_example_data import (
    generate_griffiths_data, match_estimated_topics,
    plot_lda, plot_lda_topics)

# accessing the SageMaker Python SDK
import sagemaker
from sagemaker.amazon.common import numpy_to_record_serializer
from sagemaker.predictor import csv_serializer, json_deserializer


# # Setup
# 
# ***
# 
# *This notebook was created and tested on an ml.m4.xlarge notebook instance.*
# 
# We first need to specify some AWS credentials; specifically data locations and access roles. This is the only cell of this notebook that you will need to edit. In particular, we need the following data:
# 
# * `bucket` - An S3 bucket accessible by this account.
#   * Used to store input training data and model data output.
#   * Should be withing the same region as this notebook instance, training, and hosting.
# * `prefix` - The location in the bucket where this notebook's input and and output data will be stored. (The default value is sufficient.)
# * `role` - The IAM Role ARN used to give training and hosting access to your data.
#   * See documentation on how to create these.
#   * The script below will try to determine an appropriate Role ARN.
# 

from sagemaker import get_execution_role

role = get_execution_role()

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/lda_science'


print('Training input/output will be stored in {}/{}'.format(bucket, prefix))
print('\nIAM Role: {}'.format(role))


# ## The LDA Model
# 
# As mentioned above, LDA is a model for discovering latent topics describing a collection of documents. In this section we will give a brief introduction to the model. Let,
# 
# * $M$ = the number of *documents* in a corpus
# * $N$ = the average *length* of a document.
# * $V$ = the size of the *vocabulary* (the total number of unique words)
# 
# We denote a *document* by a vector $w \in \mathbb{R}^V$ where $w_i$ equals the number of times the $i$th word in the vocabulary occurs within the document. This is called the "bag-of-words" format of representing a document.
# 
# $$
# \underbrace{w}_{\text{document}} = \overbrace{\big[ w_1, w_2, \ldots, w_V \big] }^{\text{word counts}},
# \quad
# V = \text{vocabulary size}
# $$
# 
# The *length* of a document is equal to the total number of words in the document: $N_w = \sum_{i=1}^V w_i$.
# 
# An LDA model is defined by two parameters: a topic-word distribution matrix $\beta \in \mathbb{R}^{K \times V}$ and a  Dirichlet topic prior $\alpha \in \mathbb{R}^K$. In particular, let,
# 
# $$\beta = \left[ \beta_1, \ldots, \beta_K \right]$$
# 
# be a collection of $K$ *topics* where each topic $\beta_k \in \mathbb{R}^V$ is represented as probability distribution over the vocabulary. One of the utilities of the LDA model is that a given word is allowed to appear in multiple topics with positive probability. The Dirichlet topic prior is a vector $\alpha \in \mathbb{R}^K$ such that $\alpha_k > 0$ for all $k$.
# 

# # Data Exploration
# 
# ---
# 
# ## An Example Dataset
# 
# Before explaining further let's get our hands dirty with an example dataset. The following synthetic data comes from [1] and comes with a very useful visual interpretation.
# 
# > [1] Thomas Griffiths and Mark Steyvers. *Finding Scientific Topics.* Proceedings of the National Academy of Science, 101(suppl 1):5228-5235, 2004.
# 

print('Generating example data...')
num_documents = 6000
known_alpha, known_beta, documents, topic_mixtures = generate_griffiths_data(
    num_documents=num_documents, num_topics=10)
num_topics, vocabulary_size = known_beta.shape


# separate the generated data into training and tests subsets
num_documents_training = int(0.9*num_documents)
num_documents_test = num_documents - num_documents_training

documents_training = documents[:num_documents_training]
documents_test = documents[num_documents_training:]

topic_mixtures_training = topic_mixtures[:num_documents_training]
topic_mixtures_test = topic_mixtures[num_documents_training:]

print('documents_training.shape = {}'.format(documents_training.shape))
print('documents_test.shape = {}'.format(documents_test.shape))


# Let's start by taking a closer look at the documents. Note that the vocabulary size of these data is $V = 25$. The average length of each document in this data set is 150. (See `generate_griffiths_data.py`.)
# 

print('First training document =\n{}'.format(documents_training[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))
print('Length of first document = {}'.format(documents_training[0].sum()))


average_document_length = documents.sum(axis=1).mean()
print('Observed average document length = {}'.format(average_document_length))


# The example data set above also returns the LDA parameters,
# 
# $$(\alpha, \beta)$$
# 
# used to generate the documents. Let's examine the first topic and verify that it is a probability distribution on the vocabulary.
# 

print('First topic =\n{}'.format(known_beta[0]))

print('\nTopic-word probability matrix (beta) shape: (num_topics, vocabulary_size) = {}'.format(known_beta.shape))
print('\nSum of elements of first topic = {}'.format(known_beta[0].sum()))


# Unlike some clustering algorithms, one of the versatilities of the LDA model is that a given word can belong to multiple topics. The probability of that word occurring in each topic may differ, as well. This is reflective of real-world data where, for example, the word *"rover"* appears in a *"dogs"* topic as well as in a *"space exploration"* topic.
# 
# In our synthetic example dataset, the first word in the vocabulary belongs to both Topic #1 and Topic #6 with non-zero probability.
# 

print('Topic #1:\n{}'.format(known_beta[0]))
print('Topic #6:\n{}'.format(known_beta[5]))


# Human beings are visual creatures, so it might be helpful to come up with a visual representation of these documents.
# 
# In the below plots, each pixel of a document represents a word. The greyscale intensity is a measure of how frequently that word occurs within the document. Below we plot the first few documents of the training set reshaped into 5x5 pixel grids.
# 

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda(documents_training, nrows=3, ncols=4, cmap='gray_r', with_colorbar=True)
fig.suptitle('$w$ - Document Word Counts')
fig.set_dpi(160)


# When taking a close look at these documents we can see some patterns in the word distributions suggesting that, perhaps, each topic represents a "column" or "row" of words with non-zero probability and that each document is composed primarily of a handful of topics.
# 
# Below we plots the *known* topic-word probability distributions, $\beta$. Similar to the documents we reshape each probability distribution to a $5 \times 5$ pixel image where the color represents the probability of that each word occurring in the topic.
# 

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda(known_beta, nrows=1, ncols=10)
fig.suptitle(r'Known $\beta$ - Topic-Word Probability Distributions')
fig.set_dpi(160)
fig.set_figheight(2)


# These 10 topics were used to generate the document corpus. Next, we will learn about how this is done.
# 

# ## Generating Documents
# 
# LDA is a generative model, meaning that the LDA parameters $(\alpha, \beta)$ are used to construct documents word-by-word by drawing from the topic-word distributions. In fact, looking closely at the example documents above you can see that some documents sample more words from some topics than from others.
# 
# LDA works as follows: given 
# 
# * $M$ documents $w^{(1)}, w^{(2)}, \ldots, w^{(M)}$,
# * an average document length of $N$,
# * and an LDA model $(\alpha, \beta)$.
# 
# **For** each document, $w^{(m)}$:
# * sample a topic mixture: $\theta^{(m)} \sim \text{Dirichlet}(\alpha)$
# * **For** each word $n$ in the document:
#   * Sample a topic $z_n^{(m)} \sim \text{Multinomial}\big( \theta^{(m)} \big)$
#   * Sample a word from this topic, $w_n^{(m)} \sim \text{Multinomial}\big( \beta_{z_n^{(m)}} \; \big)$
#   * Add to document
# 
# The [plate notation](https://en.wikipedia.org/wiki/Plate_notation) for the LDA model, introduced in [2], encapsulates this process pictorially.
# 
# ![](http://scikit-learn.org/stable/_images/lda_model_graph.png)
# 
# > [2] David M Blei, Andrew Y Ng, and Michael I Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3(Jan):993–1022, 2003.

# ## Topic Mixtures
# 
# For the documents we generated above lets look at their corresponding topic mixtures, $\theta \in \mathbb{R}^K$. The topic mixtures represent the probablility that a given word of the document is sampled from a particular topic. For example, if the topic mixture of an input document $w$ is,
# 
# $$\theta = \left[ 0.3, 0.2, 0, 0.5, 0, \ldots, 0 \right]$$
# 
# then $w$ is 30% generated from the first topic, 20% from the second topic, and 50% from the fourth topic. In particular, the words contained in the document are sampled from the first topic-word probability distribution 30% of the time, from the second distribution 20% of the time, and the fourth disribution 50% of the time.
# 
# 
# The objective of inference, also known as scoring, is to determine the most likely topic mixture of a given input document. Colloquially, this means figuring out which topics appear within a given document and at what ratios. We will perform infernece later in the [Inference](#Inference) section.
# 
# Since we generated these example documents using the LDA model we know the topic mixture generating them. Let's examine these topic mixtures.
# 

print('First training document =\n{}'.format(documents_training[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))
print('Length of first document = {}'.format(documents_training[0].sum()))


print('First training document topic mixture =\n{}'.format(topic_mixtures_training[0]))
print('\nNumber of topics = {}'.format(num_topics))
print('sum(theta) = {}'.format(topic_mixtures_training[0].sum()))


# We plot the first document along with its topic mixture. We also plot the topic-word probability distributions again for reference.
# 

get_ipython().run_line_magic('matplotlib', 'inline')

fig, (ax1,ax2) = plt.subplots(2, 1)

ax1.matshow(documents[0].reshape(5,5), cmap='gray_r')
ax1.set_title(r'$w$ - Document', fontsize=20)
ax1.set_xticks([])
ax1.set_yticks([])

cax2 = ax2.matshow(topic_mixtures[0].reshape(1,-1), cmap='Reds', vmin=0, vmax=1)
cbar = fig.colorbar(cax2, orientation='horizontal')
ax2.set_title(r'$\theta$ - Topic Mixture', fontsize=20)
ax2.set_xticks([])
ax2.set_yticks([])

fig.set_dpi(100)


get_ipython().run_line_magic('matplotlib', 'inline')

# pot
fig = plot_lda(known_beta, nrows=1, ncols=10)
fig.suptitle(r'Known $\beta$ - Topic-Word Probability Distributions')
fig.set_dpi(160)
fig.set_figheight(1.5)


# Finally, let's plot several documents with their corresponding topic mixtures. We can see how topics with large weight in the document lead to more words in the document within the corresponding "row" or "column".
# 

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda_topics(documents_training, 3, 4, topic_mixtures=topic_mixtures)
fig.suptitle(r'$(w,\theta)$ - Documents with Known Topic Mixtures')
fig.set_dpi(160)


# # Training
# 
# ***
# 
# In this section we will give some insight into how AWS SageMaker LDA fits an LDA model to a corpus, create an run a SageMaker LDA training job, and examine the output trained model.
# 

# ## Topic Estimation using Tensor Decompositions
# 
# Given a document corpus, Amazon SageMaker LDA uses a spectral tensor decomposition technique to determine the LDA model $(\alpha, \beta)$ which most likely describes the corpus. See [1] for a primary reference of the theory behind the algorithm. The spectral decomposition, itself, is computed using the CPDecomp algorithm described in [2].
# 
# The overall idea is the following: given a corpus of documents $\mathcal{W} = \{w^{(1)}, \ldots, w^{(M)}\}, \; w^{(m)} \in \mathbb{R}^V,$ we construct a statistic tensor,
# 
# $$T \in \bigotimes^3 \mathbb{R}^V$$
# 
# such that the spectral decomposition of the tensor is approximately the LDA parameters $\alpha \in \mathbb{R}^K$ and $\beta \in \mathbb{R}^{K \times V}$ which maximize the likelihood of observing the corpus for a given number of topics, $K$,
# 
# $$T \approx \sum_{k=1}^K \alpha_k \; (\beta_k \otimes \beta_k \otimes \beta_k)$$
# 
# This statistic tensor encapsulates information from the corpus such as the document mean, cross correlation, and higher order statistics. For details, see [1].
# 
# 
# > [1] Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham Kakade, and Matus Telgarsky. *"Tensor Decompositions for Learning Latent Variable Models"*, Journal of Machine Learning Research, 15:2773–2832, 2014.
# >
# > [2] Tamara Kolda and Brett Bader. *"Tensor Decompositions and Applications"*. SIAM Review, 51(3):455–500, 2009.
# 
# 
# 

# ## Store Data on S3
# 
# Before we run training we need to prepare the data.
# 
# A SageMaker training job needs access to training data stored in an S3 bucket. Although training can accept data of various formats we convert the documents MXNet RecordIO Protobuf format before uploading to the S3 bucket defined at the beginning of this notebook.
# 

# convert documents_training to Protobuf RecordIO format
recordio_protobuf_serializer = numpy_to_record_serializer()
fbuffer = recordio_protobuf_serializer(documents_training)

# upload to S3 in bucket/prefix/train
fname = 'lda.data'
s3_object = os.path.join(prefix, 'train', fname)
boto3.Session().resource('s3').Bucket(bucket).Object(s3_object).upload_fileobj(fbuffer)

s3_train_data = 's3://{}/{}'.format(bucket, s3_object)
print('Uploaded data to S3: {}'.format(s3_train_data))


# Next, we specify a Docker container containing the SageMaker LDA algorithm. For your convenience, a region-specific container is automatically chosen for you to minimize cross-region data communication
# 

containers = {
    'us-west-2': '266724342769.dkr.ecr.us-west-2.amazonaws.com/lda:latest',
    'us-east-1': '766337827248.dkr.ecr.us-east-1.amazonaws.com/lda:latest',
    'us-east-2': '999911452149.dkr.ecr.us-east-2.amazonaws.com/lda:latest',
    'eu-west-1': '999678624901.dkr.ecr.eu-west-1.amazonaws.com/lda:latest'
}
region_name = boto3.Session().region_name
container = containers[region_name]

print('Using SageMaker LDA container: {} ({})'.format(container, region_name))


# ## Training Parameters
# 
# Particular to a SageMaker LDA training job are the following hyperparameters:
# 
# * **`num_topics`** - The number of topics or categories in the LDA model.
#   * Usually, this is not known a priori.
#   * In this example, howevever, we know that the data is generated by five topics.
# 
# * **`feature_dim`** - The size of the *"vocabulary"*, in LDA parlance.
#   * In this example, this is equal 25.
# 
# * **`mini_batch_size`** - The number of input training documents.
# 
# * **`alpha0`** - *(optional)* a measurement of how "mixed" are the topic-mixtures.
#   * When `alpha0` is small the data tends to be represented by one or few topics.
#   * When `alpha0` is large the data tends to be an even combination of several or many topics.
#   * The default value is `alpha0 = 1.0`.
# 
# In addition to these LDA model hyperparameters, we provide additional parameters defining things like the EC2 instance type on which training will run, the S3 bucket containing the data, and the AWS access role. Note that,
# 
# * Recommended instance type: `ml.c4`
# * Current limitations:
#   * SageMaker LDA *training* can only run on a single instance.
#   * SageMaker LDA does not take advantage of GPU hardware.
#   * (The Amazon AI Algorithms team is working hard to provide these capabilities in a future release!)
# 

# Using the above configuration create a SageMaker client and use the client to create a training job.
# 

session = sagemaker.Session()

# specify general training job information
lda = sagemaker.estimator.Estimator(
    container,
    role,
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    train_instance_count=1,
    train_instance_type='ml.c4.2xlarge',
    sagemaker_session=session,
)

# set algorithm-specific hyperparameters
lda.set_hyperparameters(
    num_topics=num_topics,
    feature_dim=vocabulary_size,
    mini_batch_size=num_documents_training,
    alpha0=1.0,
)

# run the training job on input data stored in S3
lda.fit({'train': s3_train_data})


# If you see the message
# 
# > `===== Job Complete =====`
# 
# at the bottom of the output logs then that means training sucessfully completed and the output LDA model was stored in the specified output path. You can also view information about and the status of a training job using the AWS SageMaker console. Just click on the "Jobs" tab and select training job matching the training job name, below:
# 

print('Training job name: {}'.format(lda.latest_training_job.job_name))


# ## Inspecting the Trained Model
# 
# We know the LDA parameters $(\alpha, \beta)$ used to generate the example data. How does the learned model compare the known one? In this section we will download the model data and measure how well SageMaker LDA did in learning the model.
# 
# First, we download the model data. SageMaker will output the model in 
# 
# > `s3://<bucket>/<prefix>/output/<training job name>/output/model.tar.gz`.
# 
# SageMaker LDA stores the model as a two-tuple $(\alpha, \beta)$ where each LDA parameter is an MXNet NDArray.
# 

# download and extract the model file from S3
job_name = lda.latest_training_job.job_name
model_fname = 'model.tar.gz'
model_object = os.path.join(prefix, 'output', job_name, 'output', model_fname)
boto3.Session().resource('s3').Bucket(bucket).Object(model_object).download_file(fname)
with tarfile.open(fname) as tar:
    tar.extractall()
print('Downloaded and extracted model tarball: {}'.format(model_object))

# obtain the model file
model_list = [fname for fname in os.listdir('.') if fname.startswith('model_')]
model_fname = model_list[0]
print('Found model file: {}'.format(model_fname))

# get the model from the model file and store in Numpy arrays
alpha, beta = mx.ndarray.load(model_fname)
learned_alpha_permuted = alpha.asnumpy()
learned_beta_permuted = beta.asnumpy()

print('\nLearned alpha.shape = {}'.format(learned_alpha_permuted.shape))
print('Learned beta.shape = {}'.format(learned_beta_permuted.shape))


# Presumably, SageMaker LDA has found the topics most likely used to generate the training corpus. However, even if this is case the topics would not be returned in any particular order. Therefore, we match the found topics to the known topics closest in L1-norm in order to find the topic permutation.
# 
# Note that we will use the `permutation` later during inference to match known topic mixtures to found topic mixtures.
# 
# Below plot the known topic-word probability distribution, $\beta \in \mathbb{R}^{K \times V}$ next to the distributions found by SageMaker LDA as well as the L1-norm errors between the two.
# 

permutation, learned_beta = match_estimated_topics(known_beta, learned_beta_permuted)
learned_alpha = learned_alpha_permuted[permutation]

fig = plot_lda(np.vstack([known_beta, learned_beta]), 2, 10)
fig.set_dpi(160)
fig.suptitle('Known vs. Found Topic-Word Probability Distributions')
fig.set_figheight(3)

beta_error = np.linalg.norm(known_beta - learned_beta, 1)
alpha_error = np.linalg.norm(known_alpha - learned_alpha, 1)
print('L1-error (beta) = {}'.format(beta_error))
print('L1-error (alpha) = {}'.format(alpha_error))


# Not bad!
# 
# In the eyeball-norm the topics match quite well. In fact, the topic-word distribution error is approximately 2%.
# 

# # Inference
# 
# ***
# 
# A trained model does nothing on its own. We now want to use the model we computed to perform inference on data. For this example, that means predicting the topic mixture representing a given document.
# 
# We create an inference endpoint using the SageMaker Python SDK `deploy()` function from the job we defined above. We specify the instance type where inference is computed as well as an initial number of instances to spin up.
# 

lda_inference = lda.deploy(
    initial_instance_count=1,
    instance_type='ml.c4.xlarge',  # LDA inference works best on ml.c4 instances
)


# Congratulations! You now have a functioning SageMaker LDA inference endpoint. You can confirm the endpoint configuration and status by navigating to the "Endpoints" tab in the AWS SageMaker console and selecting the endpoint matching the endpoint name, below: 
# 

print('Endpoint name: {}'.format(lda_inference.endpoint))


# With this realtime endpoint at our fingertips we can finally perform inference on our training and test data.
# 
# We can pass a variety of data formats to our inference endpoint. In this example we will demonstrate passing CSV-formatted data. Other available formats are JSON-formatted, JSON-sparse-formatter, and RecordIO Protobuf. We make use of the SageMaker Python SDK utilities `csv_serializer` and `json_deserializer` when configuring the inference endpoint.
# 

lda_inference.content_type = 'text/csv'
lda_inference.serializer = csv_serializer
lda_inference.deserializer = json_deserializer


# We pass some test documents to the inference endpoint. Note that the serializer and deserializer will atuomatically take care of the datatype conversion.
# 

results = lda_inference.predict(documents_test[:12])

print(results)


# It may be hard to see but the output format of SageMaker LDA inference endpoint is a Python dictionary with the following format.
# 
# ```
# {
#   'predictions': [
#     {'topic_mixture': [ ... ] },
#     {'topic_mixture': [ ... ] },
#     {'topic_mixture': [ ... ] },
#     ...
#   ]
# }
# ```
# 
# We extract the topic mixtures, themselves, corresponding to each of the input documents.
# 

inferred_topic_mixtures_permuted = np.array([prediction['topic_mixture'] for prediction in results['predictions']])

print('Inferred topic mixtures (permuted):\n\n{}'.format(inferred_topic_mixtures_permuted))


# ## Inference Analysis
# 
# Recall that although SageMaker LDA successfully learned the underlying topics which generated the sample data the topics were in a different order. Before we compare to known topic mixtures $\theta \in \mathbb{R}^K$ we should also permute the inferred topic mixtures
# 

inferred_topic_mixtures = inferred_topic_mixtures_permuted[:,permutation]

print('Inferred topic mixtures:\n\n{}'.format(inferred_topic_mixtures))


# Let's plot these topic mixture probability distributions alongside the known ones.
# 

get_ipython().run_line_magic('matplotlib', 'inline')

# create array of bar plots
width = 0.4
x = np.arange(10)

nrows, ncols = 3, 4
fig, ax = plt.subplots(nrows, ncols, sharey=True)
for i in range(nrows):
    for j in range(ncols):
        index = i*ncols + j
        ax[i,j].bar(x, topic_mixtures_test[index], width, color='C0')
        ax[i,j].bar(x+width, inferred_topic_mixtures[index], width, color='C1')
        ax[i,j].set_xticks(range(num_topics))
        ax[i,j].set_yticks(np.linspace(0,1,5))
        ax[i,j].grid(which='major', axis='y')
        ax[i,j].set_ylim([0,1])
        ax[i,j].set_xticklabels([])
        if (i==(nrows-1)):
            ax[i,j].set_xticklabels(range(num_topics), fontsize=7)
        if (j==0):
            ax[i,j].set_yticklabels([0,'',0.5,'',1.0], fontsize=7)
        
fig.suptitle('Known vs. Inferred Topic Mixtures')
ax_super = fig.add_subplot(111, frameon=False)
ax_super.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax_super.grid(False)
ax_super.set_xlabel('Topic Index')
ax_super.set_ylabel('Topic Probability')
fig.set_dpi(160)


# In the eyeball-norm these look quite comparable.
# 
# Let's be more scientific about this. Below we compute and plot the distribution of L1-errors from **all** of the test documents. Note that we send a new payload of test documents to the inference endpoint and apply the appropriate permutation to the output.
# 

get_ipython().run_cell_magic('time', '', "\n# create a payload containing all of the test documents and run inference again\n#\n# TRY THIS:\n#   try switching between the test data set and a subset of the training\n#   data set. It is likely that LDA inference will perform better against\n#   the training set than the holdout test set.\n#\npayload_documents = documents_test                    # Example 1\nknown_topic_mixtures = topic_mixtures_test            # Example 1\n#payload_documents = documents_training[:600];         # Example 2\n#known_topic_mixtures = topic_mixtures_training[:600]  # Example 2\n\nprint('Invoking endpoint...\\n')\nresults = lda_inference.predict(payload_documents)\n\ninferred_topic_mixtures_permuted = np.array([prediction['topic_mixture'] for prediction in results['predictions']])\ninferred_topic_mixtures = inferred_topic_mixtures_permuted[:,permutation]\n\nprint('known_topics_mixtures.shape = {}'.format(known_topic_mixtures.shape))\nprint('inferred_topics_mixtures_test.shape = {}\\n'.format(inferred_topic_mixtures.shape))")


get_ipython().run_line_magic('matplotlib', 'inline')

l1_errors = np.linalg.norm((inferred_topic_mixtures - known_topic_mixtures), 1, axis=1)

# plot the error freqency
fig, ax_frequency = plt.subplots()
bins = np.linspace(0,1,40)
weights = np.ones_like(l1_errors)/len(l1_errors)
freq, bins, _ = ax_frequency.hist(l1_errors, bins=50, weights=weights, color='C0')
ax_frequency.set_xlabel('L1-Error')
ax_frequency.set_ylabel('Frequency', color='C0')


# plot the cumulative error
shift = (bins[1]-bins[0])/2
x = bins[1:] - shift
ax_cumulative = ax_frequency.twinx()
cumulative = np.cumsum(freq)/sum(freq)
ax_cumulative.plot(x, cumulative, marker='o', color='C1')
ax_cumulative.set_ylabel('Cumulative Frequency', color='C1')


# align grids and show
freq_ticks = np.linspace(0, 1.5*freq.max(), 5)
freq_ticklabels = np.round(100*freq_ticks)/100
ax_frequency.set_yticks(freq_ticks)
ax_frequency.set_yticklabels(freq_ticklabels)
ax_cumulative.set_yticks(np.linspace(0, 1, 5))
ax_cumulative.grid(which='major', axis='y')
ax_cumulative.set_ylim((0,1))


fig.suptitle('Topic Mixutre L1-Errors')
fig.set_dpi(110)


# Machine learning algorithms are not perfect and the data above suggests this is true of SageMaker LDA. With more documents and some hyperparameter tuning we can obtain more accurate results against the known topic-mixtures.
# 
# For now, let's just investigate the documents-topic mixture pairs that seem to do well as well as those that do not. Below we retreive a document and topic mixture corresponding to a small L1-error as well as one with a large L1-error.
# 

N = 6

good_idx = (l1_errors < 0.05)
good_documents = payload_documents[good_idx][:N]
good_topic_mixtures = inferred_topic_mixtures[good_idx][:N]

poor_idx = (l1_errors > 0.3)
poor_documents = payload_documents[poor_idx][:N]
poor_topic_mixtures = inferred_topic_mixtures[poor_idx][:N]


get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda_topics(good_documents, 2, 3, topic_mixtures=good_topic_mixtures)
fig.suptitle('Documents With Accurate Inferred Topic-Mixtures')
fig.set_dpi(120)


get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda_topics(poor_documents, 2, 3, topic_mixtures=poor_topic_mixtures)
fig.suptitle('Documents With Inaccurate Inferred Topic-Mixtures')
fig.set_dpi(120)


# In this example set the documents on which inference was not as accurate tend to have a denser topic-mixture. This makes sense when extrapolated to real-world datasets: it can be difficult to nail down which topics are represented in a document when the document uses words from a large subset of the vocabulary.
# 

# ## Stop / Close the Endpoint
# 
# Finally, we should delete the endpoint before we close the notebook.
# 
# To do so execute the cell below. Alternately, you can navigate to the "Endpoints" tab in the SageMaker console, select the endpoint with the name stored in the variable `endpoint_name`, and select "Delete" from the "Actions" dropdown menu. 
# 

sagemaker.Session().delete_endpoint(lda_inference.endpoint)


# # Epilogue
# 
# ---
# 
# In this notebook we,
# 
# * learned about the LDA model,
# * generated some example LDA documents and their corresponding topic-mixtures,
# * trained a SageMaker LDA model on a training set of documents and compared the learned model to the known model,
# * created an inference endpoint,
# * used the endpoint to infer the topic mixtures of a test input and analyzed the inference error.
# 
# There are several things to keep in mind when applying SageMaker LDA to real-word data such as a corpus of text documents. Note that input documents to the algorithm, both in training and inference, need to be vectors of integers representing word counts. Each index corresponds to a word in the corpus vocabulary. Therefore, one will need to "tokenize" their corpus vocabulary.
# 
# $$
# \text{"cat"} \mapsto 0, \; \text{"dog"} \mapsto 1 \; \text{"bird"} \mapsto 2, \ldots
# $$
# 
# Each text document then needs to be converted to a "bag-of-words" format document.
# 
# $$
# w = \text{"cat bird bird bird cat"} \quad \longmapsto \quad w = [2, 0, 3, 0, \ldots, 0]
# $$
# 
# Also note that many real-word applications have large vocabulary sizes. It may be necessary to represent the input documents in sparse format. Finally, the use of stemming and lemmatization in data preprocessing provides several benefits. Doing so can improve training and inference compute time since it reduces the effective vocabulary size. More importantly, though, it can improve the quality of learned topic-word probability matrices and inferred topic mixtures. For example, the words *"parliament"*, *"parliaments"*, *"parliamentary"*, *"parliament's"*, and *"parliamentarians"* are all essentially the same word, *"parliament"*, but with different conjugations. For the purposes of detecting topics, such as a *"politics"* or *governments"* topic, the inclusion of all five does not add much additional value as they all essentiall describe the same feature.
# 




# # Bring Your Own R Algorithm
# _**Create a Docker container for training R algorithms and hosting R models**_
# 
# ---
# 
# ---
# 
# ## Contents
# 
# 1. [Background](#Background)
# 1. [Preparation](#Preparation)
# 1. [Code](#Code)
#   1. [Fit](#Fit)
#   1. [Serve](#Serve)
#   1. [Dockerfile](#Dockerfile)
#   1. [Publish](#Publish)
# 1. [Data](#Data)
# 1. [Train](#Train)
# 1. [Host](#Host)
# 1. [Predict](#Predict)
# 1. [Extensions](#Extensions)
# 
# ---
# ## Background
# 
# R is a popular open source statistical programming language, with a lengthy history in Data Science and Machine Learning.  The breadth of algorithms available as an R package is impressive, which fuels a growing community of users.  The R kernel can be installed into Amazon SageMaker Notebooks, and Docker containers which use R can be used to take advantage of Amazon SageMaker's flexible training and hosting functionality.  This notebook illustrates a simple use case for creating an R container and then using it to train and host a model.  In order to take advantage of boto, we'll use Python within the notebook, but this could be done 100% in R by invoking command line arguments.
# 
# ---
# ## Preparation
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Let's start by specifying:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.  Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).
# 

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/r_byo'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# Now we'll import the libraries we'll need for the remainder of the notebook.
# 

import time
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Permissions
# 
# Running this notebook requires permissions in addition to the normal `SageMakerFullAccess` permissions. This is because we'll be creating a new repository in Amazon ECR. The easiest way to add these permissions is simply to add the managed policy `AmazonEC2ContainerRegistryFullAccess` to the role that you used to start your notebook instance. There's no need to restart your notebook instance when you do this, the new permissions will be available immediately.
# 
# ---
# ## Code
# 
# For this example, we'll need 3 supporting code files.
# 
# ### Fit
# 
# `mars.R` creates functions to fit and serve our model.  The algorithm we've chosen to use is [Multivariate Adaptive Regression Splines](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_splines).  This is a suitable example as it's a unique and powerful algorithm, but isn't as broadly used as Amazon SageMaker algorithms, and it isn't available in Python's scikit-learn library.  R's repository of packages is filled with algorithms that share these same criteria. 
# 

# _The top of the code is devoted to setup.  Bringing in the libraries we'll need and setting up the file paths as detailed in Amazon SageMaker documentation on bringing your own container._
# 
# ```
# # Bring in library that contains multivariate adaptive regression splines (MARS)
# library(mda)
# 
# # Bring in library that allows parsing of JSON training parameters
# library(jsonlite)
# 
# # Bring in library for prediction server
# library(plumber)
# 
# 
# # Setup parameters
# # Container directories
# prefix <- '/opt/ml'
# input_path <- paste(prefix, 'input/data', sep='/')
# output_path <- paste(prefix, 'output', sep='/')
# model_path <- paste(prefix, 'model', sep='/')
# param_path <- paste(prefix, 'input/config/hyperparameters.json', sep='/')
# 
# # Channel holding training data
# channel_name = 'train'
# training_path <- paste(input_path, channel_name, sep='/')
# ```
# 

# _Next, we define a train function that actually fits the model to the data.  For the most part this is idiomatic R, with a bit of maneuvering up front to take in parameters from a JSON file, and at the end to output a success indicator._
# 
# ```
# # Setup training function
# train <- function() {
# 
#     # Read in hyperparameters
#     training_params <- read_json(param_path)
# 
#     target <- training_params$target
# 
#     if (!is.null(training_params$degree)) {
#         degree <- as.numeric(training_params$degree)}
#     else {
#         degree <- 2}
# 
#     # Bring in data
#     training_files = list.files(path=training_path, full.names=TRUE)
#     training_data = do.call(rbind, lapply(training_files, read.csv))
#     
#     # Convert to model matrix
#     training_X <- model.matrix(~., training_data[, colnames(training_data) != target])
# 
#     # Save factor levels for scoring
#     factor_levels <- lapply(training_data[, sapply(training_data, is.factor), drop=FALSE],
#                             function(x) {levels(x)})
#     
#     # Run multivariate adaptive regression splines algorithm
#     model <- mars(x=training_X, y=training_data[, target], degree=degree)
#     
#     # Generate outputs
#     mars_model <- model[!(names(model) %in% c('x', 'residuals', 'fitted.values'))]
#     attributes(mars_model)$class <- 'mars'
#     save(mars_model, factor_levels, file=paste(model_path, 'mars_model.RData', sep='/'))
#     print(summary(mars_model))
# 
#     write.csv(model$fitted.values, paste(output_path, 'data/fitted_values.csv', sep='/'), row.names=FALSE)
#     write('success', file=paste(output_path, 'success', sep='/'))}
# ```
# 

# _Then, we setup the serving function (which is really just a short wrapper around our plumber.R file that we'll discuss [next](#Serve)._
# 
# ```
# # Setup scoring function
# serve <- function() {
#     app <- plumb(paste(prefix, 'plumber.R', sep='/'))
#     app$run(host='0.0.0.0', port=8080)}
# ```
# 

# _Finally, a bit of logic to determine if, based on the options passed when Amazon SageMaker Training or Hosting call this script, we are using the container to train an algorithm or host a model._
# 
# ```
# # Run at start-up
# args <- commandArgs()
# if (any(grepl('train', args))) {
#     train()}
# if (any(grepl('serve', args))) {
#     serve()}
# ```
# 

# ### Serve
# `plumber.R` uses the [plumber](https://www.rplumber.io/) package to create a lightweight HTTP server for processing requests in hosting.  Note the specific syntax, and see the plumber help docs for additional detail on more specialized use cases.
# 

# Per the Amazon SageMaker documentation, our service needs to accept post requests to ping and invocations.  plumber specifies this with custom comments, followed by functions that take specific arguments.
# 
# Here invocations does most of the work, ingesting our trained model, handling the HTTP request body, and producing a CSV output of predictions.
# 
# ```
# # plumber.R
# 
# 
# #' Ping to show server is there
# #' @get /ping
# function() {
#     return('')}
# 
# 
# #' Parse input and return the prediction from the model
# #' @param req The http request sent
# #' @post /invocations
# function(req) {
# 
#     # Setup locations
#     prefix <- '/opt/ml'
#     model_path <- paste(prefix, 'model', sep='/')
# 
#     # Bring in model file and factor levels
#     load(paste(model_path, 'mars_model.RData', sep='/'))
# 
#     # Read in data
#     conn <- textConnection(gsub('\\\\n', '\n', req$postBody))
#     data <- read.csv(conn)
#     close(conn)
# 
#     # Convert input to model matrix
#     scoring_X <- model.matrix(~., data, xlev=factor_levels)
# 
#     # Return prediction
#     return(paste(predict(mars_model, scoring_X, row.names=FALSE), collapse=','))}
# ```
# 

# ### Dockerfile
# 
# Smaller containers are preferred for Amazon SageMaker as they lead to faster spin up times in training and endpoint creation, so this container is kept minimal.  It simply starts with Ubuntu, installs R, mda, and plumber libraries, then adds `mars.R` and `plumber.R`, and finally runs `mars.R` when the entrypoint is launched.
# 
# ```Dockerfile
# FROM ubuntu:16.04
# 
# MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>
# 
# RUN apt-get -y update && apt-get install -y --no-install-recommends     wget     r-base     r-base-dev     ca-certificates
# 
# RUN R -e "install.packages(c('mda', 'plumber'), repos='https://cloud.r-project.org')"
# 
# COPY mars.R /opt/ml/mars.R
# COPY plumber.R /opt/ml/plumber.R
# 
# ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/mars.R", "--no-save"]
# ```
# 

# ### Publish
# Now, to publish this container to ECR, we'll run the comands below.
# 
# This command will take several minutes to run the first time.
# 

get_ipython().run_cell_magic('sh', '', '\n# The name of our algorithm\nalgorithm_name=rmars\n\n#set -e # stop if anything fails\n\naccount=$(aws sts get-caller-identity --query Account --output text)\n\n# Get the region defined in the current configuration (default to us-west-2 if none defined)\nregion=$(aws configure get region)\nregion=${region:-us-west-2}\n\nfullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"\n\n# If the repository doesn\'t exist in ECR, create it.\n\naws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1\n\nif [ $? -ne 0 ]\nthen\n    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null\nfi\n\n# Get the login command from ECR and execute it directly\n$(aws ecr get-login --region ${region} --no-include-email)\n\n# On a SageMaker Notebook Instance, the docker daemon may need to be restarted in order\n# to detect your network configuration correctly.  (This is a known issue.)\nif [ -d "/home/ec2-user/SageMaker" ]; then\n  sudo service docker restart\nfi\n\n# Build the docker image locally with the image name and then push it to ECR\n# with the full name.\ndocker build  -t ${algorithm_name} .\ndocker tag ${algorithm_name} ${fullname}\n\ndocker push ${fullname}')


# ---
# ## Data
# For this illustrative example, we'll simply use `iris`.  This a classic, but small, dataset used to test supervised learning algorithms.  Typically the goal is to predict one of three flower species based on various measurements of the flowers' attributes.  Further detail can be found [here](https://en.wikipedia.org/wiki/Iris_flower_data_set).
# 
# Then let's copy the data to S3.
# 

train_file = 'iris.csv'
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_file(train_file)


# _Note: Although we could, we'll avoid doing any preliminary transformations on the data, instead choosing to do those transformations inside the container.  This is not typically the best practice for model efficiency, but provides some benefits in terms of flexibility._
# 

# ---
# ## Train
# 
# Now, let's setup the information needed to train a Multivariate Adaptive Regression Splines (MARS) model on iris data.  In this case, we'll predict `Sepal.Length` rather than the more typical classification of `Species` to show how factors might be included in a model and limit the case to regression.
# 
# First, we'll get our region and account information so that we can point to the ECR container we just created.
# 

region = boto3.Session().region_name
account = boto3.client('sts').get_caller_identity().get('Account')


# 
# - Specify the role to use
# - Give the training job a name
# - Point the algorithm to the container we created
# - Specify training instance resources (in this case our algorithm is only single-threaded so stick to 1 instance)
# - Point to the S3 location of our input data and the `train` channel expected by our algorithm
# - Point to the S3 location for output
# - Provide hyperparamters (keeping it simple)
# - Maximum run time
# 

r_job = 'r-byo-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

print("Training job", r_job)

r_training_params = {
    "RoleArn": role,
    "TrainingJobName": r_job,
    "AlgorithmSpecification": {
        "TrainingImage": '{}.dkr.ecr.{}.amazonaws.com/rmars:latest'.format(account, region),
        "TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.m4.xlarge",
        "VolumeSizeInGB": 10
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        }
    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/output".format(bucket, prefix)
    },
    "HyperParameters": {
        "target": "Sepal.Length",
        "degree": "2"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}


# Now let's kick off our training job on Amazon SageMaker Training, using the parameters we just created.  Because training is managed (AWS takes care of spinning up and spinning down the hardware), we don't have to wait for our job to finish to continue, but for this case, let's setup a waiter so we can monitor the status of our training.
# 

get_ipython().run_cell_magic('time', '', '\nsm = boto3.client(\'sagemaker\')\nsm.create_training_job(**r_training_params)\n\nstatus = sm.describe_training_job(TrainingJobName=r_job)[\'TrainingJobStatus\']\nprint(status)\nsm.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=r_job)\nstatus = sm.describe_training_job(TrainingJobName=r_job)[\'TrainingJobStatus\']\nprint("Training job ended with status: " + status)\nif status == \'Failed\':\n    message = sm.describe_training_job(TrainingJobName=r_job)[\'FailureReason\']\n    print(\'Training failed with the following error: {}\'.format(message))\n    raise Exception(\'Training job failed\')')


# ---
# ## Host
# 
# Hosting the model we just trained takes three steps in Amazon SageMaker.  First, we define the model we want to host, pointing the service to the model artifact our training job just wrote to S3.
# 

r_hosting_container = {
    'Image': '{}.dkr.ecr.{}.amazonaws.com/rmars:latest'.format(account, region),
    'ModelDataUrl': sm.describe_training_job(TrainingJobName=r_job)['ModelArtifacts']['S3ModelArtifacts']
}

create_model_response = sm.create_model(
    ModelName=r_job,
    ExecutionRoleArn=role,
    PrimaryContainer=r_hosting_container)

print(create_model_response['ModelArn'])


# Next, let's create an endpoing configuration, passing in the model we just registered.  In this case, we'll only use a few c4.xlarges.
# 

r_endpoint_config = 'r-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(r_endpoint_config)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=r_endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': r_job,
        'VariantName': 'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# Finally, we'll create the endpoints using our endpoint configuration from the last step.
# 

get_ipython().run_cell_magic('time', '', '\nr_endpoint = \'r-endpoint-\' + time.strftime("%Y%m%d%H%M", time.gmtime())\nprint(r_endpoint)\ncreate_endpoint_response = sm.create_endpoint(\n    EndpointName=r_endpoint,\n    EndpointConfigName=r_endpoint_config)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm.describe_endpoint(EndpointName=r_endpoint)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\ntry:\n    sm.get_waiter(\'endpoint_in_service\').wait(EndpointName=r_endpoint)\nfinally:\n    resp = sm.describe_endpoint(EndpointName=r_endpoint)\n    status = resp[\'EndpointStatus\']\n    print("Arn: " + resp[\'EndpointArn\'])\n    print("Status: " + status)\n\n    if status != \'InService\':\n        raise Exception(\'Endpoint creation did not succeed\')')


# ---
# ## Predict
# To confirm our endpoints are working properly, let's try to invoke the endpoint.
# 
# _Note: The payload we're passing in the request is a CSV string with a header record, followed by multiple new lines.  It also contains text columns, which the serving code converts to the set of indicator variables needed for our model predictions.  Again, this is not a best practice for highly optimized code, however, it showcases the flexibility of bringing your own algorithm._
# 

iris = pd.read_csv('iris.csv')

runtime = boto3.Session().client('runtime.sagemaker')

payload = iris.drop(['Sepal.Length'], axis=1).to_csv(index=False)
response = runtime.invoke_endpoint(EndpointName=r_endpoint,
                                   ContentType='text/csv',
                                   Body=payload)

result = json.loads(response['Body'].read().decode())
result 


# We can see the result is a CSV of predictions for our target variable.  Let's compare them to the actuals to see how our model did.
# 

plt.scatter(iris['Sepal.Length'], np.fromstring(result[0], sep=','))
plt.show()


# ---
# ## Extensions
# 
# This notebook showcases a straightforward example to train and host an R algorithm in Amazon SageMaker.  As mentioned previously, this notebook could also be written in R.  We could even train the algorithm entirely within a notebook and then simply use the serving portion of the container to host our model.
# 
# Other extensions could include setting up the R algorithm to train in parallel.  Although R is not the easiest language to build distributed applications on top of, this is possible.  In addition, running multiple versions of training simultaneously would allow for parallelized grid (or random) search for optimal hyperparamter settings.  This would more fully realize the benefits of managed training.
# 

# ### (Optional) Clean-up
# 
# If you're ready to be done with this notebook, please run the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

sm.delete_endpoint(EndpointName=r_endpoint)


# # TensorFlow BYOM: Train locally and deploy on SageMaker.
# 
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#     1. [Permissions and environment variables](#Permissions-and-environment-variables)
#     2. [Model definitions](#Model-definitions)
#     3. [Data Setup](#Data-setup)
# 3. [Training the network locally](#Training)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#     1. [Export from TensorFlow](#Export-the-model-from-tensorflow)
#     2. [Import model into SageMaker](#Import-model-into-SageMaker)
#     3. [Create endpoint](#Create-endpoint) 
# 5. [Validate the endpoint for use](#Validate-the-endpoint-for-use)
# 
# __Note__: Compare this with the [tensorflow bring your own model example](../tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb)
# 

# ## Introduction 
# 
# This notebook can be compared to [Iris classification example notebook](../tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb) in terms of its functionality. We will do the same classification task, but we will train the same network locally in the box from where this notebook is being run. We then setup a real-time hosted endpoint in SageMaker.
# 
# Consider the following model definition for IRIS classification. This mode uses the ``tensorflow.estimator.DNNClassifier`` which is a pre-defined estimator module for its model definition. The model definition is the same as the one used in the [Iris classification example notebook](../tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb)
# 
# ## Prequisites and Preprocessing
# ### Permissions and environment variables
# 
# Here we set up the linkage and authentication to AWS services. In this notebook we only need the roles used to give learning and hosting access to your data. The Sagemaker SDK will use S3 defualt buckets when needed. If the ``get_execution_role``  does not return a role with the appropriate permissions, you'll need to specify an IAM role arn that does.
# 

import boto3, re
from sagemaker import get_execution_role

role = get_execution_role()


# ### Model Definitions
# 
# We use the [``tensorflow.estimator.DNNClassifier``](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) estimator to set up our network. We also need to write some methods for serving inputs during hosting and training. These methods are all found below.
# 

get_ipython().system('cat iris_dnn_classifier.py')


# Create an estimator object with this model definition.
# 

from iris_dnn_classifier import estimator_fn
classifier = estimator_fn(run_config = None, params = None)


# ### Data setup
# 
# Next, we need to pull the data from tensorflow repository and make them ready for training. The following will code block should do that.
# 

import os 
from six.moves.urllib.request import urlopen

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)


# Create the data input streamer object.
# 

from iris_dnn_classifier import train_input_fn
train_func = train_input_fn('.', params = None)


# ### Training
# 
# It is time to train the network. Since we are training the network locally, we can make use of TensorFlow's ``tensorflow.Estimator.train`` method. The model is trained locally in the box.
# 

classifier.train(input_fn = train_func, steps = 1000)


# ## Set up hosting for the model
# 
# ### Export the model from tensorflow
# 
# In order to set up hosting, we have to import the model from training to hosting. We will begin by exporting the model from TensorFlow and saving it down. Analogous to the [MXNet example](../mxnet_mnist_byom/mxnet_mnist.ipynb), some structure needs to be followed. The exported model has to be converted into a form that is readable by ``sagemaker.mxnet.model.MXNetModel``. The following code describes exporting the model in a form that does the same:
# 
# There is a small difference between a SageMaker model and a TensorFlow model. The conversion is easy and fairly trivial. Simply move the tensorflow exported model into a directory ``export\Servo\`` and tar the entire directory. SageMaker will recognize this as a loadable TensorFlow model.
# 

from iris_dnn_classifier import serving_input_fn

exported_model = classifier.export_savedmodel(export_dir_base = 'export/Servo/', 
                               serving_input_receiver_fn = serving_input_fn)

print (exported_model)
import tarfile
with tarfile.open('model.tar.gz', mode='w:gz') as archive:
    archive.add('export', recursive=True)


# ### Import model into SageMaker
# 
# Open a new sagemaker session and upload the model on to the default S3 bucket. We can use the ``sagemaker.Session.upload_data`` method to do this. We need the location of where we exported the model from MXNet and where in our default bucket we want to store the model(``/model``). The default S3 bucket can be found using the ``sagemaker.Session.default_bucket`` method.
# 

import sagemaker

sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')


# Use the ``sagemaker.mxnet.model.TensorFlowModel`` to import the model into SageMaker that can be deployed. We need the location of the S3 bucket where we have the model, the role for authentication and the entry_point where the model defintion is stored (``iris_dnn_classifier.py``). The import call is the following:
# 

from sagemaker.tensorflow.model import TensorFlowModel
sagemaker_model = TensorFlowModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                  role = role,
                                  entry_point = 'iris_dnn_classifier.py')


# ### Create endpoint
# 
# Now the model is ready to be deployed at a SageMaker endpoint. We can use the ``sagemaker.mxnet.model.TensorFlowModel.deploy`` method to do this. Unless you have created or prefer other instances, we recommend using 1 ``'ml.c4.xlarge'`` instance for this example. These are supplied as arguments. 
# 

get_ipython().run_cell_magic('time', '', "predictor = sagemaker_model.deploy(initial_instance_count=1,\n                                          instance_type='ml.c4.xlarge')")


# ### Validate the endpoint for use
# 
# We can now use this endpoint to classify. Run an example prediction on a sample to ensure that it works.
# 

sample = [6.4,3.2,4.5,1.5]
predictor.predict(sample)


# Delete all temporary directories so that we are not affecting the next run. Also, optionally delete the end points.
# 

os.remove('model.tar.gz')
import shutil
shutil.rmtree('export')


# If you do not want to continue using the endpoint, you can remove it. Remember, open endpoints are charged. If this is a simple test or practice, it is recommended to delete them.
# 

sagemaker.Session().delete_endpoint(predictor.endpoint)


# # An Introduction to PCA with MNIST
# _**Investigating Eigendigits from Principal Components Analysis on Handwritten Digits**_
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
#   2. [Data ingestion](#Data-ingestion)
#   3. [Data inspection](#Data-inspection)
#   4. [Data conversion](#Data-conversion)
# 3. [Training the PCA model](#Training-the-PCA-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#   1. [Import model into hosting](#Import-model-into-hosting)
#   2. [Create endpoint configuration](#Create-endpoint-configuration)
#   3. [Create endpoint](#Create-endpoint)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 

# ## Introduction
# 
# Welcome to our example introducing Amazon SageMaker's PCA Algorithm! Today, we're analyzing the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset which consists of images of handwritten digits, from zero to nine.  We'll ignore the true labels for the time being and instead focus on what information we can obtain from the image pixels alone.
# 
# The method that we'll look at today is called Principal Components Analysis (PCA).  PCA is an unsupervised learning algorithm that attempts to reduce the dimensionality (number of features) within a dataset while still retaining as much information as possible. This is done by finding a new set of feature dimensions called principal components, which are composites of the original features that are uncorrelated with one another. They are also constrained so that the first component accounts for the largest possible variability in the data, the second component the second most variability, and so on.
# 
# PCA is most commonly used as a pre-processing step.  Statistically, many models assume data to be low-dimensional.  In those cases, the output of PCA will actually include much less of the noise and subsequent models can be more accurate.  Taking datasets with a huge number of features and reducing them down can be shown to not hurt the accuracy of the clustering while enjoying significantly improved performance.  In addition, using PCA in advance of a linear model can make overfitting due to multi-collinearity less likely.
# 
# For our current use case though, we focus purely on the output of PCA.  [Eigenfaces](https://en.wikipedia.org/wiki/Eigenface) have been used for years in facial recognition and computer vision.  The eerie images represent a large library of photos as a smaller subset.  These eigenfaces are not necessarily clusters, but instead highlight key features, that when combined, can represent most of the variation in faces throughout the entire library.  We'll follow an analagous path and develop eigendigits from our handwritten digit dataset.
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Let's start by specifying:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.  Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).
# 

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/pca-mnist'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# ### Data ingestion
# 
# Next, we read the dataset from an online URL into memory, for preprocessing prior to training. This processing could be done *in-situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present at the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets such as this one, reading into memory isn't onerous, though it would be for larger datasets.
# 

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nwith gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')')


# ### Data inspection
# 
# Once the dataset is imported, it's typical as part of the machine learning process to inspect the data, understand the distributions, and determine what type(s) of preprocessing might be needed. You can perform those tasks right here in the notebook. As an example, let's go ahead and look at one of the digits that is part of the dataset.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)


def show_digit(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)

show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))


# ### Data conversion
# 
# Since algorithms have particular input and output requirements, converting the dataset is also part of the process that a data scientist goes through prior to initiating training. In this particular case, the Amazon SageMaker implementation of PCA takes recordIO-wrapped protobuf, where the data we have today is a pickle-ized numpy array on disk.
# 
# Most of the conversion effort is handled by the Amazon SageMaker Python SDK, imported as `sagemaker` below.
# 

import io
import numpy as np
import sagemaker.amazon.common as smac

vectors = np.array([t.tolist() for t in train_set[0]]).T

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors)
buf.seek(0)


# ## Upload training data
# Now that we've created our recordIO-wrapped protobuf, we'll need to upload it to S3, so that Amazon SageMaker training can use it.
# 

get_ipython().run_cell_magic('time', '', "import boto3\nimport os\n\nkey = 'recordio-pb-data'\nboto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)\ns3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)\nprint('uploaded training data location: {}'.format(s3_train_data))")


# Let's also setup an output S3 location for the model artifact that will be output as the result of training with the algorithm.
# 

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


# ## Training the PCA model
# 
# Once we have the data preprocessed and available in the correct format for training, the next step is to actually train the model using the data. Since this data is relatively small, it isn't meant to show off the performance of the PCA training algorithm, although we have tested it on multi-terabyte datasets.
# 
# Again, we'll use the Amazon SageMaker Python SDK to kick off training, and monitor status until it is completed.  In this example that takes between 7 and 11 minutes.  Despite the dataset being small, provisioning hardware and loading the algorithm container take time upfront.
# 
# First, let's specify our containers.  Since we want this notebook to run in all 4 of Amazon SageMaker's regions, we'll create a small lookup.  More details on algorithm containers can be found in [AWS documentation](https://docs-aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).
# 

containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/pca:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/pca:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/pca:latest'}


# Next we'll kick off the base estimator, making sure to pass in the necessary hyperparameters.  Notice:
# - `feature_dim` is set to 50000.  We've transposed our datasets relative to most of the other MNIST examples because for eigendigits we're looking to understand pixel relationships, rather than make predictions about individual images.
# - `num_components` has been set to 10.  This could easily be increased for future experimentation.  In practical settings, setting the number of components typically uses a mixture of objective and subjective criteria.  Data Scientists tend to look for the fewest principal components that eat up the most variation in the data.
# - `subtract_mean` standardizes the pixel intensity across all images.  The MNIST data has already been extensively cleaned, but including this shouldn't hurt.
# - `algorithm_mode` is set to 'randomized'.  Because we have a very large number of dimensions, this makes the most sense.  The alternative 'stable' should be used in cases with a lower value for `feature_dim`.
# - `mini_batch_size` has been set to 200.  For PCA, this parameter should not affect fit, but may have slight implications on timing.  Other algorithms may require tuning of this parameter in order to achieve the best results.
# 

import boto3
import sagemaker

sess = sagemaker.Session()

pca = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.c4.xlarge',
                                    output_path=output_location,
                                    sagemaker_session=sess)
pca.set_hyperparameters(feature_dim=50000,
                        num_components=10,
                        subtract_mean=True,
                        algorithm_mode='randomized',
                        mini_batch_size=200)

pca.fit({'train': s3_train_data})


# ## Set up hosting for the model
# Now that we've trained our model, we can deploy it behind an Amazon SageMaker real-time hosted endpoint.  This will allow out to make predictions (or inference) from the model dyanamically.
# 
# _Note, Amazon SageMaker allows you the flexibility of importing models trained elsewhere, as well as the choice of not importing models if the target of model creation is AWS Lambda, AWS Greengrass, Amazon Redshift, Amazon Athena, or other deployment target._
# 

pca_predictor = pca.deploy(initial_instance_count=1,
                           instance_type='ml.c4.xlarge')


# ## Validate the model for use
# Finally, we can now validate the model for use.  We can pass HTTP POST requests to the endpoint to get back predictions.  To make this easier, we'll again use the Amazon SageMaker Python SDK and specify how to serialize requests and deserialize responses that are specific to the algorithm.
# 

from sagemaker.predictor import csv_serializer, json_deserializer

pca_predictor.content_type = 'text/csv'
pca_predictor.serializer = csv_serializer
pca_predictor.deserializer = json_deserializer


# Now let's try getting a prediction for a single record.
# 

result = pca_predictor.predict(train_set[0][:, 0])
print(result)


# OK, a single prediction works.  We see that for one record our endpoint returned some JSON which contains a value for each of the 10 principal components we created when training the model.
# 
# Let's do a whole batch and see what comes out.
# 

import numpy as np

eigendigits = []
for array in np.array_split(train_set[0].T, 50):
    result = pca_predictor.predict(array)
    eigendigits += [r['projection'] for r in result['projections']]


eigendigits = np.array(eigendigits).T


for e in enumerate(eigendigits):
    show_digit(e[1], 'eigendigit #{}'.format(e[0]))


# Not surprisingly, the eigendigits aren't extremely interpretable.  They do show interesting elements of the data, with eigendigit #0 being the "anti-number", eigendigit #1 looking a bit like a `0` combined with the inverse of a `3`, eigendigit #2 showing some shapes resembling a `9`, and so on.
# 

# ### (Optional) Delete the Endpoint
# 
# If you're ready to be done with this notebook, please run the delete_endpoint line in the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

import sagemaker

sagemaker.Session().delete_endpoint(pca_predictor.endpoint)


# # Training and hosting SageMaker Models using the Apache MXNet Module API
# 
# The **SageMaker Python SDK** makes it easy to train and deploy MXNet models. In this example, we train a simple neural network using the Apache MXNet [Module API](https://mxnet.incubator.apache.org/api/python/module.html) and the MNIST dataset. The MNIST dataset is widely used for handwritten digit classification, and consists of 70,000 labeled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). The task at hand is to train a model using the 60,000 training images and subsequently test its classification accuracy on the 10,000 test images.
# 
# ### Setup
# 
# First we need to define a few variables that will be needed later in the example.
# 

from sagemaker import get_execution_role

#Bucket location to save your custom code in tar.gz format.
custom_code_upload_location = 's3://<bucket-name>/customcode/mxnet'

#Bucket location where results of model training are saved.
model_artifacts_location = 's3://<bucket-name>/artifacts'

#IAM execution role that gives SageMaker access to resources in your AWS account.
#We can use the SageMaker Python SDK to get the role from our notebook environment. 
role = get_execution_role()


# ### The training script
# 
# The ``mnist.py`` script provides all the code we need for training and hosting a SageMaker model. The script we will use is adaptated from Apache MXNet [MNIST tutorial (https://mxnet.incubator.apache.org/tutorials/python/mnist.html).
# 

get_ipython().system('cat mnist.py')


# ### SageMaker's MXNet estimator class
# 

# The SageMaker ```MXNet``` estimator allows us to run single machine or distributed training in SageMaker, using CPU or GPU-based instances.
# 
# When we create the estimator, we pass in the filename of our training script, the name of our IAM execution role, and the S3 locations we defined in the setup section. We also provide a few other parameters. ``train_instance_count`` and ``train_instance_type`` determine the number and type of SageMaker instances that will be used for the training job. The ``hyperparameters`` parameter is a ``dict`` of values that will be passed to your training script -- you can see how to access these values in the ``mnist.py`` script above.
# 
# For this example, we will choose one ``ml.m4.xlarge`` instance.
# 

from sagemaker.mxnet import MXNet

mnist_estimator = MXNet(entry_point='mnist.py',
                        role=role,
                        output_path=model_artifacts_location,
                        code_location=custom_code_upload_location,
                        train_instance_count=1, 
                        train_instance_type='ml.m4.xlarge',
                        hyperparameters={'learning_rate': 0.1})


# ### Running the Training Job
# 

# After we've constructed our MXNet object, we can fit it using data stored in S3. Below we run SageMaker training on two input channels: **train** and **test**.
# 
# During training, SageMaker makes this data stored in S3 available in the local filesystem where the mnist script is running. The ```mnist.py``` script simply loads the train and test data from disk.
# 

get_ipython().run_cell_magic('time', '', "import boto3\n\nregion = boto3.Session().region_name\ntrain_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)\ntest_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)\n\nmnist_estimator.fit({'train': train_data_location, 'test': test_data_location})")


# ### Creating an inference Endpoint
# 
# After training, we use the ``MXNet estimator`` object to build and deploy an ``MXNetPredictor``. This creates a Sagemaker **Endpoint** -- a hosted prediction service that we can use to perform inference. 
# 
# The arguments to the ``deploy`` function allow us to set the number and type of instances that will be used for the Endpoint. These do not need to be the same as the values we used for the training job. For example, you can train a model on a set of GPU-based instances, and then deploy the Endpoint to a fleet of CPU-based instances. Here we will deploy the model to a single ``ml.c4.xlarge`` instance.
# 

get_ipython().run_cell_magic('time', '', "\npredictor = mnist_estimator.deploy(initial_instance_count=1,\n                                   instance_type='ml.c4.xlarge')")


# The request handling behavior of the Endpoint is determined by the ``mnist.py`` script. In this case, the script doesn't include any request handling functions, so the Endpoint will use the default handlers provided by SageMaker. These default handlers allow us to perform inference on input data encoded as a multi-dimensional JSON array.
# 
# ### Making an inference request
# 
# Now that our Endpoint is deployed and we have a ``predictor`` object, we can use it to classify handwritten digits.
# 
# To see inference in action, draw a digit in the image box below. The pixel data from your drawing will be loaded into a ``data`` variable in this notebook. 
# 
# *Note: after drawing the image, you'll need to move to the next notebook cell.*
# 

from IPython.display import HTML
HTML(open("input.html").read())


# Now we can use the ``predictor`` object to classify the handwritten digit:
# 

response = predictor.predict(data)
print('Raw prediction result:')
print(response)

labeled_predictions = list(zip(range(10), response[0]))
print('Labeled predictions: ')
print(labeled_predictions)

labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
print('Most likely answer: {}'.format(labeled_predictions[0]))


# # (Optional) Delete the Endpoint
# 
# After you have finished with this example, remember to delete the prediction endpoint to release the instance(s) associated with it.
# 

print("Endpoint name: " + predictor.endpoint)


import sagemaker

sagemaker.Session().delete_endpoint(predictor.endpoint)





# # An Introduction to SageMaker LDA
# 
# ***Finding topics in synthetic document data using Spectral LDA algorithms.***
# 
# ---
# 
# 1. [Introduction](#Introduction)
# 1. [Setup](#Setup)
# 1. [Training](#Training)
# 1. [Inference](#Inference)
# 1. [Epilogue](#Epilogue)
# 

# # Introduction
# ***
# 
# Amazon SageMaker LDA is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. Latent Dirichlet Allocation (LDA) is most commonly used to discover a user-specified number of topics shared by documents within a text corpus. Here each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics. Since the method is unsupervised, the topics are not specified up front, and are not guaranteed to align with how a human may naturally categorize documents. The topics are learned as a probability distribution over the words that occur in each document. Each document, in turn, is described as a mixture of topics.
# 
# In this notebook we will use the Amazon SageMaker LDA algorithm to train an LDA model on some example synthetic data. We will then use this model to classify (perform inference on) the data. The main goals of this notebook are to,
# 
# * learn how to obtain and store data for use in Amazon SageMaker,
# * create an AWS SageMaker training job on a data set to produce an LDA model,
# * use the LDA model to perform inference with an Amazon SageMaker endpoint.
# 
# The following are ***not*** goals of this notebook:
# 
# * understand the LDA model,
# * understand how the Amazon SageMaker LDA algorithm works,
# * interpret the meaning of the inference output
# 
# If you would like to know more about these things take a minute to run this notebook and then check out the SageMaker LDA Documentation and the **LDA-Science.ipynb** notebook.
# 

get_ipython().system('conda install -y scipy')


get_ipython().run_line_magic('matplotlib', 'inline')

import os, re

import boto3
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# some helpful utility functions are defined in the Python module
# "generate_example_data" located in the same directory as this
# notebook
from generate_example_data import generate_griffiths_data, plot_lda, match_estimated_topics

# accessing the SageMaker Python SDK
import sagemaker
from sagemaker.amazon.common import numpy_to_record_serializer
from sagemaker.predictor import csv_serializer, json_deserializer


# # Setup
# 
# ***
# 
# *This notebook was created and tested on an ml.m4.xlarge notebook instance.*
# 
# Before we do anything at all, we need data! We also need to setup our AWS credentials so that AWS SageMaker can store and access data. In this section we will do four things:
# 
# 1. [Setup AWS Credentials](#SetupAWSCredentials)
# 1. [Obtain Example Dataset](#ObtainExampleDataset)
# 1. [Inspect Example Data](#InspectExampleData)
# 1. [Store Data on S3](#StoreDataonS3)
# 

# ## Setup AWS Credentials
# 
# We first need to specify some AWS credentials; specifically data locations and access roles. This is the only cell of this notebook that you will need to edit. In particular, we need the following data:
# 
# * `bucket` - An S3 bucket accessible by this account.
#   * Used to store input training data and model data output.
#   * Should be withing the same region as this notebook instance, training, and hosting.
# * `prefix` - The location in the bucket where this notebook's input and and output data will be stored. (The default value is sufficient.)
# * `role` - The IAM Role ARN used to give training and hosting access to your data.
#   * See documentation on how to create these.
#   * The script below will try to determine an appropriate Role ARN.
# 

from sagemaker import get_execution_role

role = get_execution_role()
bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/lda_introduction'

print('Training input/output will be stored in {}/{}'.format(bucket, prefix))
print('\nIAM Role: {}'.format(role))


# ## Obtain Example Data
# 
# 
# We generate some example synthetic document data. For the purposes of this notebook we will omit the details of this process. All we need to know is that each piece of data, commonly called a *"document"*, is a vector of integers representing *"word counts"* within the document. In this particular example there are a total of 25 words in the *"vocabulary"*.
# 
# $$
# \underbrace{w}_{\text{document}} = \overbrace{\big[ w_1, w_2, \ldots, w_V \big] }^{\text{word counts}},
# \quad
# V = \text{vocabulary size}
# $$
# 
# These data are based on that used by Griffiths and Steyvers in their paper [Finding Scientific Topics](http://psiexp.ss.uci.edu/research/papers/sciencetopics.pdf). For more information, see the **LDA-Science.ipynb** notebook.
# 

print('Generating example data...')
num_documents = 6000
num_topics = 5
known_alpha, known_beta, documents, topic_mixtures = generate_griffiths_data(
    num_documents=num_documents, num_topics=num_topics)
vocabulary_size = len(documents[0])

# separate the generated data into training and tests subsets
num_documents_training = int(0.9*num_documents)
num_documents_test = num_documents - num_documents_training

documents_training = documents[:num_documents_training]
documents_test = documents[num_documents_training:]

topic_mixtures_training = topic_mixtures[:num_documents_training]
topic_mixtures_test = topic_mixtures[num_documents_training:]

print('documents_training.shape = {}'.format(documents_training.shape))
print('documents_test.shape = {}'.format(documents_test.shape))


# ## Inspect Example Data
# 
# *What does the example data actually look like?* Below we print an example document as well as its corresponding known *topic-mixture*. A topic-mixture serves as the "label" in the LDA model. It describes the ratio of topics from which the words in the document are found.
# 
# For example, if the topic mixture of an input document $\mathbf{w}$ is,
# 
# $$\theta = \left[ 0.3, 0.2, 0, 0.5, 0 \right]$$
# 
# then $\mathbf{w}$ is 30% generated from the first topic, 20% from the second topic, and 50% from the fourth topic. For more information see **How LDA Works** in the SageMaker documentation as well as the **LDA-Science.ipynb** notebook.
# 
# Below, we compute the topic mixtures for the first few training documents. As we can see, each document is a vector of word counts from the 25-word vocabulary and its topic-mixture is a probability distribution across the five topics used to generate the sample dataset.
# 

print('First training document =\n{}'.format(documents[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))


print('Known topic mixture of first document =\n{}'.format(topic_mixtures_training[0]))
print('\nNumber of topics = {}'.format(num_topics))
print('Sum of elements = {}'.format(topic_mixtures_training[0].sum()))


# Later, when we perform inference on the training data set we will compare the inferred topic mixture to this known one.
# 
# ---
# 
# Human beings are visual creatures, so it might be helpful to come up with a visual representation of these documents. In the below plots, each pixel of a document represents a word. The greyscale intensity is a measure of how frequently that word occurs. Below we plot the first few documents of the training set reshaped into 5x5 pixel grids.
# 

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda(documents_training, nrows=3, ncols=4, cmap='gray_r', with_colorbar=True)
fig.suptitle('Example Document Word Counts')
fig.set_dpi(160)


# ## Store Data on S3
# 
# A SageMaker training job needs access to training data stored in an S3 bucket. Although training can accept data of various formats we convert the documents MXNet RecordIO Protobuf format before uploading to the S3 bucket defined at the beginning of this notebook. We do so by making use of the SageMaker Python SDK utility `numpy_to_record_serializer`.
# 

# convert documents_training to Protobuf RecordIO format
recordio_protobuf_serializer = numpy_to_record_serializer()
fbuffer = recordio_protobuf_serializer(documents_training)

# upload to S3 in bucket/prefix/train
fname = 'lda.data'
s3_object = os.path.join(prefix, 'train', fname)
boto3.Session().resource('s3').Bucket(bucket).Object(s3_object).upload_fileobj(fbuffer)

s3_train_data = 's3://{}/{}'.format(bucket, s3_object)
print('Uploaded data to S3: {}'.format(s3_train_data))


# # Training
# 
# ***
# 
# Once the data is preprocessed and available in a recommended format the next step is to train our model on the data. There are number of parameters required by SageMaker LDA configuring the model and defining the computational environment in which training will take place.
# 
# First, we specify a Docker container containing the SageMaker LDA algorithm. For your convenience, a region-specific container is automatically chosen for you to minimize cross-region data communication. Information about the locations of each SageMaker algorithm is available in the documentation.
# 

# select the algorithm container based on this notebook's current location
containers = {
    'us-west-2': '266724342769.dkr.ecr.us-west-2.amazonaws.com/lda:latest',
    'us-east-1': '766337827248.dkr.ecr.us-east-1.amazonaws.com/lda:latest',
    'us-east-2': '999911452149.dkr.ecr.us-east-2.amazonaws.com/lda:latest',
    'eu-west-1': '999678624901.dkr.ecr.eu-west-1.amazonaws.com/lda:latest'
}
region_name = boto3.Session().region_name
container = containers[region_name]

print('Using SageMaker LDA container: {} ({})'.format(container, region_name))


# Particular to a SageMaker LDA training job are the following hyperparameters:
# 
# * **`num_topics`** - The number of topics or categories in the LDA model.
#   * Usually, this is not known a priori.
#   * In this example, howevever, we know that the data is generated by five topics.
# 
# * **`feature_dim`** - The size of the *"vocabulary"*, in LDA parlance.
#   * In this example, this is equal 25.
# 
# * **`mini_batch_size`** - The number of input training documents.
# 
# * **`alpha0`** - *(optional)* a measurement of how "mixed" are the topic-mixtures.
#   * When `alpha0` is small the data tends to be represented by one or few topics.
#   * When `alpha0` is large the data tends to be an even combination of several or many topics.
#   * The default value is `alpha0 = 1.0`.
# 
# In addition to these LDA model hyperparameters, we provide additional parameters defining things like the EC2 instance type on which training will run, the S3 bucket containing the data, and the AWS access role. Note that,
# 
# * Recommended instance type: `ml.c4`
# * Current limitations:
#   * SageMaker LDA *training* can only run on a single instance.
#   * SageMaker LDA does not take advantage of GPU hardware.
#   * (The Amazon AI Algorithms team is working hard to provide these capabilities in a future release!)
# 

session = sagemaker.Session()

# specify general training job information
lda = sagemaker.estimator.Estimator(
    container,
    role,
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    train_instance_count=1,
    train_instance_type='ml.c4.2xlarge',
    sagemaker_session=session,
)

# set algorithm-specific hyperparameters
lda.set_hyperparameters(
    num_topics=num_topics,
    feature_dim=vocabulary_size,
    mini_batch_size=num_documents_training,
    alpha0=1.0,
)

# run the training job on input data stored in S3
lda.fit({'train': s3_train_data})


# If you see the message
# 
# > `===== Job Complete =====`
# 
# at the bottom of the output logs then that means training sucessfully completed and the output LDA model was stored in the specified output path. You can also view information about and the status of a training job using the AWS SageMaker console. Just click on the "Jobs" tab and select training job matching the training job name, below:
# 

print('Training job name: {}'.format(lda.latest_training_job.job_name))


# # Inference
# 
# ***
# 
# A trained model does nothing on its own. We now want to use the model we computed to perform inference on data. For this example, that means predicting the topic mixture representing a given document.
# 
# We create an inference endpoint using the SageMaker Python SDK `deploy()` function from the job we defined above. We specify the instance type where inference is computed as well as an initial number of instances to spin up.
# 

lda_inference = lda.deploy(
    initial_instance_count=1,
    instance_type='ml.c4.xlarge',  # LDA inference works best on ml.c4 instances
)


# Congratulations! You now have a functioning SageMaker LDA inference endpoint. You can confirm the endpoint configuration and status by navigating to the "Endpoints" tab in the AWS SageMaker console and selecting the endpoint matching the endpoint name, below: 
# 

print('Endpoint name: {}'.format(lda_inference.endpoint))


# With this realtime endpoint at our fingertips we can finally perform inference on our training and test data.
# 
# We can pass a variety of data formats to our inference endpoint. In this example we will demonstrate passing CSV-formatted data. Other available formats are JSON-formatted, JSON-sparse-formatter, and RecordIO Protobuf. We make use of the SageMaker Python SDK utilities `csv_serializer` and `json_deserializer` when configuring the inference endpoint.
# 

lda_inference.content_type = 'text/csv'
lda_inference.serializer = csv_serializer
lda_inference.deserializer = json_deserializer


# We pass some test documents to the inference endpoint. Note that the serializer and deserializer will atuomatically take care of the datatype conversion from Numpy NDArrays.
# 

results = lda_inference.predict(documents_test[:12])

print(results)


# It may be hard to see but the output format of SageMaker LDA inference endpoint is a Python dictionary with the following format.
# 
# ```
# {
#   'predictions': [
#     {'topic_mixture': [ ... ] },
#     {'topic_mixture': [ ... ] },
#     {'topic_mixture': [ ... ] },
#     ...
#   ]
# }
# ```
# 
# We extract the topic mixtures, themselves, corresponding to each of the input documents.
# 

computed_topic_mixtures = np.array([prediction['topic_mixture'] for prediction in results['predictions']])

print(computed_topic_mixtures)


# If you decide to compare these results to the known topic mixtures generated in the [Obtain Example Data](#ObtainExampleData) Section keep in mind that SageMaker LDA discovers topics in no particular order. That is, the approximate topic mixtures computed above may be permutations of the known topic mixtures corresponding to the same documents.
# 

print(topic_mixtures_test[0])      # known test topic mixture
print(computed_topic_mixtures[0])  # computed topic mixture (topics permuted)


# ## Stop / Close the Endpoint
# 
# Finally, we should delete the endpoint before we close the notebook.
# 
# To do so execute the cell below. Alternately, you can navigate to the "Endpoints" tab in the SageMaker console, select the endpoint with the name stored in the variable `endpoint_name`, and select "Delete" from the "Actions" dropdown menu. 
# 

sagemaker.Session().delete_endpoint(lda_inference.endpoint)


# # Epilogue
# 
# ---
# 
# In this notebook we,
# 
# * generated some example LDA documents and their corresponding topic-mixtures,
# * trained a SageMaker LDA model on a training set of documents,
# * created an inference endpoint,
# * used the endpoint to infer the topic mixtures of a test input.
# 
# There are several things to keep in mind when applying SageMaker LDA to real-word data such as a corpus of text documents. Note that input documents to the algorithm, both in training and inference, need to be vectors of integers representing word counts. Each index corresponds to a word in the corpus vocabulary. Therefore, one will need to "tokenize" their corpus vocabulary.
# 
# $$
# \text{"cat"} \mapsto 0, \; \text{"dog"} \mapsto 1 \; \text{"bird"} \mapsto 2, \ldots
# $$
# 
# Each text document then needs to be converted to a "bag-of-words" format document.
# 
# $$
# w = \text{"cat bird bird bird cat"} \quad \longmapsto \quad w = [2, 0, 3, 0, \ldots, 0]
# $$
# 
# Also note that many real-word applications have large vocabulary sizes. It may be necessary to represent the input documents in sparse format. Finally, the use of stemming and lemmatization in data preprocessing provides several benefits. Doing so can improve training and inference compute time since it reduces the effective vocabulary size. More importantly, though, it can improve the quality of learned topic-word probability matrices and inferred topic mixtures. For example, the words *"parliament"*, *"parliaments"*, *"parliamentary"*, *"parliament's"*, and *"parliamentarians"* are all essentially the same word, *"parliament"*, but with different conjugations. For the purposes of detecting topics, such as a *"politics"* or *governments"* topic, the inclusion of all five does not add much additional value as they all essentiall describe the same feature.
# 




# # An Introduction to Factorization Machines with MNIST
# _**Making a Binary Prediction of Whether a Handwritten Digit is a 0**_
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
#   2. [Data ingestion](#Data-ingestion)
#   3. [Data inspection](#Data-inspection)
#   4. [Data conversion](#Data-conversion)
# 3. [Training the FM model](#Training-the-FM-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
#   1. [Import model into hosting](#Import-model-into-hosting)
#   2. [Create endpoint configuration](#Create-endpoint-configuration)
#   3. [Create endpoint](#Create-endpoint)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 

# ## Introduction
# 
# Welcome to our example introducing Amazon SageMaker's Factorization Machines Algorithm!  Today, we're analyzing the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset which consists of images of handwritten digits, from zero to nine.  We'll use the individual pixel values from each 28 x 28 grayscale image to predict a yes or no label of whether the digit is a 0 or some other digit (1, 2, 3, ... 9).
# 
# The method that we'll use is a factorization machine binary classifier.  A factorization machine is a general-purpose supervised learning algorithm that you which can use for both classification and regression tasks.  It is an extension of a linear model that is designed to parsimoniously capture interactions between features in high dimensional sparse datasets.  For example, in a click prediction system, the factorization machine model can capture click rate patterns observed when ads from a certain ad-category are placed on pages from a certain page-category.  Factorization machines are a good choice for tasks dealing with high dimensional sparse datasets, such as click prediction and item recommendation.
# 
# Amazon SageMaker's Factorization Machine algorithm provides a robust, highly scalable implementation of this algorithm, which has become extremely popular in ad click prediction and recommender systems.  The main purpose of this notebook is to quickly show the basics of implementing Amazon SageMaker Factorization Machines, even if the use case of predicting a digit from an image is not where factorization machines shine.
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.
# 

# ## Prequisites and Preprocessing
# 
# ### Permissions and environment variables
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Let's start by specifying:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.  Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).
# 

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/fm-mnist'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# ### Data ingestion
# 
# Next, we read the dataset from an online URL into memory, for preprocessing prior to training. This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.
# 

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nwith gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')')


# ### Data inspection
# 
# Once the dataset is imported, it's typical as part of the machine learning process to inspect the data, understand the distributions, and determine what type(s) of preprocessing might be needed. You can perform those tasks right here in the notebook. As an example, let's go ahead and look at one of the digits that is part of the dataset.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)


def show_digit(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)

show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))


# ### Data conversion
# 
# Since algorithms have particular input and output requirements, converting the dataset is also part of the process that a data scientist goes through prior to initiating training. In this particular case, the Amazon SageMaker implementation of Factorization Machines takes recordIO-wrapped protobuf, where the data we have today is a pickle-ized numpy array on disk.
# 
# Most of the conversion effort is handled by the Amazon SageMaker Python SDK, imported as `sagemaker` below.
# 
# _Notice, despite the fact that most use cases for factorization machines will utilize spare input, we are writing our data out as dense tensors.  This will be fine since the MNIST dataset is not particularly large or high dimensional._
# 

import io
import numpy as np
import sagemaker.amazon.common as smac

vectors = np.array([t.tolist() for t in train_set[0]]).astype('float32')
labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1.0, 0.0).astype('float32')

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)


# ## Upload training data
# Now that we've created our recordIO-wrapped protobuf, we'll need to upload it to S3, so that Amazon SageMaker training can use it.
# 

import boto3
import os

key = 'recordio-pb-data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# Let's also setup an output S3 location for the model artifact that will be output as the result of training with the algorithm.
# 

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


# ## Training the factorization machine model
# 
# Once we have the data preprocessed and available in the correct format for training, the next step is to actually train the model using the data. Since this data is relatively small, it isn't meant to show off the performance of the Amazon SageMaker's Factorization Machines in training, although we have tested it on multi-terabyte datasets.
# 
# Again, we'll use the Amazon SageMaker Python SDK to kick off training and monitor status until it is completed.  In this example that takes between 7 and 11 minutes.  Despite the dataset being small, provisioning hardware and loading the algorithm container take time upfront.
# 
# First, let's specify our containers.  Since we want this notebook to run in all 4 of Amazon SageMaker's regions, we'll create a small lookup.  More details on algorithm containers can be found in [AWS documentation](https://docs-aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).
# 

containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/factorization-machines:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/factorization-machines:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/factorization-machines:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/factorization-machines:latest'}


# Next we'll kick off the base estimator, making sure to pass in the necessary hyperparameters.  Notice:
# - `feature_dim` is set to 784, which is the number of pixels in each 28 x 28 image.
# - `predictor_type` is set to 'binary_classifier' since we are trying to predict whether the image is or is not a 0.
# - `mini_batch_size` is set to 200.  This value can be tuned for relatively minor improvements in fit and speed, but selecting a reasonable value relative to the dataset is appropriate in most cases.
# - `num_factors` is set to 10.  As mentioned initially, factorization machines find a lower dimensional representation of the interactions for all features.  Making this value smaller provides a more parsimonious model, closer to a linear model, but may sacrifice information about interactions.  Making it larger provides a higher-dimensional representation of feature interactions, but adds computational complexity and can lead to overfitting.  In a practical application, time should be invested to tune this parameter to the appropriate value.
# - `_sparse_input` is set to 'false'.  As mentioned previously, factorization machines are frequently used with sparse data, which is therefore what the algorithm expects.  Setting this to false forces the algorithm to handle the dense recordIO-wrapped protobuf that we created above.
# 

import boto3
import sagemaker

sess = sagemaker.Session()

fm = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                   role, 
                                   train_instance_count=1, 
                                   train_instance_type='ml.c4.xlarge',
                                   output_path=output_location,
                                   sagemaker_session=sess)
fm.set_hyperparameters(feature_dim=784,
                      predictor_type='binary_classifier',
                      mini_batch_size=200,
                      num_factors=10,
                      _sparse_input='false')

fm.fit({'train': s3_train_data})


# ## Set up hosting for the model
# Now that we've trained our model, we can deploy it behind an Amazon SageMaker real-time hosted endpoint.  This will allow out to make predictions (or inference) from the model dyanamically.
# 
# _Note, Amazon SageMaker allows you the flexibility of importing models trained elsewhere, as well as the choice of not importing models if the target of model creation is AWS Lambda, AWS Greengrass, Amazon Redshift, Amazon Athena, or other deployment target._
# 

fm_predictor = fm.deploy(initial_instance_count=1,
                         instance_type='ml.c4.xlarge')


# ## Validate the model for use
# Finally, we can now validate the model for use.  We can pass HTTP POST requests to the endpoint to get back predictions.  To make this easier, we'll again use the Amazon SageMaker Python SDK and specify how to serialize requests and deserialize responses that are specific to the algorithm.
# 
# Since factorization machines are so frequently used with sparse data, making inference requests with a CSV format (as is done in other algorithm examples) can be massively inefficient.  Rather than waste space and time generating all of those zeros, to pad the row to the correct dimensionality, JSON can be used more efficiently.  Since we trained the model using dense data, this is a bit of a moot point, as we'll have to pass all the 0s in anyway.
# 
# Nevertheless, we'll write our own small function to serialize our inference request in the JSON format that Amazon SageMaker Factorization Machines expects.
# 

import json
from sagemaker.predictor import json_deserializer

def fm_serializer(data):
    js = {'instances': []}
    for row in data:
        js['instances'].append({'features': row.tolist()})
    return json.dumps(js)

fm_predictor.content_type = 'application/json'
fm_predictor.serializer = fm_serializer
fm_predictor.deserializer = json_deserializer


# Now let's try getting a prediction for a single record.
# 

result = fm_predictor.predict(train_set[0][30:31])
print(result)


# OK, a single prediction works.  We see that for one record our endpoint returned some JSON which contains `predictions`, including the `score` and `predicted_label`.  In this case, `score` will be a continuous value between [0, 1] representing the probability we think the digit is a 0 or not.  `predicted_label` will take a value of either `0` or `1` where (somewhat counterintuitively) `1` denotes that we predict the image is a 0, while `0` denotes that we are predicting the image is not of a 0.
# 
# Let's do a whole batch of images and evaluate our predictive accuracy.
# 

import numpy as np

predictions = []
for array in np.array_split(test_set[0], 100):
    result = fm_predictor.predict(array)
    predictions += [r['predicted_label'] for r in result['predictions']]

predictions = np.array(predictions)


import pandas as pd

pd.crosstab(np.where(test_set[1] == 0, 1, 0), predictions, rownames=['actuals'], colnames=['predictions'])


# As we can see from the confusion matrix above, we predict 942 images of 0 correctly, while we predict 99 images as 0 when in actuality they aren't, and we miss predicting 38 images of 0 that we should have.
# 

# ### (Optional) Delete the Endpoint
# 
# If you're ready to be done with this notebook, please run the delete_endpoint line in the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.
# 

import sagemaker

sagemaker.Session().delete_endpoint(fm_predictor.endpoint)


