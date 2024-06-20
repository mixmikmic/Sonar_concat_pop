# In this notebook, we're going to use a Generative Adversarial Network to create new MNIST samples.
# 
# <https://medium.com/@julsimon/generative-adversarial-networks-on-apache-mxnet-part-1-b6d39e6b5df1>
# 

from __future__ import print_function
import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
import logging
import cv2
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)


def get_mnist():
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (64,64)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 64, 64))
    X = np.tile(X, (1, 3, 1, 1))
    X_train = X[:60000]
    X_test = X[60000:]

    return X_train, X_test


class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]


mnist_batch_size = 64
random_vector_size = 100
ctx = mx.gpu(0)

X_train, X_test = get_mnist()
train_iter = mx.io.NDArrayIter(X_train, batch_size=mnist_batch_size)
    
rand_iter = RandIter(batch_size, random_vector_size)
label = mx.nd.zeros((batch_size,), ctx=ctx)


def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    plt.imshow(buff)
    plt.title(title)
    plt.show()


def make_dcgan_sym(ngf, ndf, nc, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4,4), num_filter=ngf*8, no_bias=no_bias)
    gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*4, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=no_bias)
    gbn4 = BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g5, name='gact5', act_type='tanh')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=no_bias)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=no_bias)
    dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
    dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=no_bias)
    dbn4 = BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
    d5 = mx.sym.Flatten(d5)

    dloss = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')
    return gout, dloss


dataset = 'mnist'
ndf = 64
ngf = 64
nc = 3
lr = 0.0002
beta1 = 0.5
check_point = False


def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)
    
def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def fentropy(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

mG = mx.metric.CustomMetric(fentropy)
mD = mx.metric.CustomMetric(fentropy)
mACC = mx.metric.CustomMetric(facc)


symG, symD = make_dcgan_sym(ngf, ndf, nc)


modD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
modD.init_params(initializer=mx.init.Normal(0.02))
modD.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })


modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
modG.bind(data_shapes=rand_iter.provide_data)
modG.init_params(initializer=mx.init.Normal(0.02))
modG.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'wd': 0.,
        'beta1': beta1,
    })


print('Training...')
stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')

for epoch in range(100):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            rbatch = rand_iter.next()

            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            label[:] = 0
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            #modD.update()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update discriminator on real
            label[:] = 1
            batch.label = [label]
            modD.forward(batch, is_train=True)
            modD.backward()
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update()

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update generator
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD.get_input_grads()
            modG.backward(diffD)
            modG.update()

            mG.update([label], modD.get_outputs())

            t += 1
            if t % 10 == 0:
                #print 'epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get()
                mACC.reset()
                mG.reset()
                mD.reset()

                visual('gout', outG[0].asnumpy())
                diff = diffD[0].asnumpy()
                diff = (diff - diff.mean())/diff.std()
                visual('diff', diff)
                visual('data', batch.data[0].asnumpy())

        if check_point:
            print('Saving...')
            modG.save_params('%s_G_%s-%04d.params'%(dataset, stamp, epoch))
            modD.save_params('%s_D_%s-%04d.params'%(dataset, stamp, epoch))


modD.save_params('%s_D_%s-%04d.params'%(dataset, stamp, epoch))


# # Image classification - training from scratch demo
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
# Welcome to our end-to-end example of distributed image classification algorithm in transfer learning mode. In this demo, we will use the Amazon sagemaker image classification algorithm to learn to classify the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). 
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

get_ipython().run_cell_magic('time', '', "import boto3\nimport re\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nbucket='jsimon-sagemaker-us' # customize to your bucket\n\ncontainers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest',\n              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',\n              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest',\n              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest'}\ntraining_image = containers[boto3.Session().region_name]\nprint(training_image)")


# ## Training the Image classification model
# 
# The CIFAR-10 dataset consist of images from 10 categories and has 50,000 images with 5,000 images per category. 
# 
# The image classification algorithm can take two types of input formats. The first is a [recordio format](https://mxnet.incubator.apache.org/tutorials/basic/record_io.html) and the other is a [lst format](https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=im2rec). Files for both these formats are available at http://data.mxnet.io/data/cifar10/. In this example, we will use the recordio format for training and use the training/validation split.
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


# CIFAR-10

download('http://data.mxnet.io/data/cifar10/cifar10_train.rec')
download('http://data.mxnet.io/data/cifar10/cifar10_val.rec')
upload_to_s3('validation/cifar10', 'cifar10_val.rec')
upload_to_s3('train/cifar10', 'cifar10_train.rec')


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
# * **num_layers**: The number of layers (depth) for the network. We use 44 in this sample but other values can be used.
# * **num_training_samples**: This is the total number of training samples. It is set to 50000 for CIFAR-10 dataset with the current split
# * **num_classes**: This is the number of output classes for the new dataset. Imagenet was trained with 1000 output classes but the number of output classes can be changed for fine-tuning. For CIFAR-10, we use 10.
# * **epochs**: Number of training epochs
# * **learning_rate**: Learning rate for training
# * **mini_batch_size**: The number of training samples used for each mini batch. In distributed training, the number of training samples used per batch will be N * mini_batch_size where N is the number of hosts on which training is run
# 

# After setting training parameters, we kick off training, and poll for status until training is completed, which in this example, takes between 10 to 12 minutes per epoch on a p2.xlarge machine. The network typically converges after 10 epochs.  
# 

# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 50 layers
num_layers = 44
# we need to specify the input image shape for the training data
image_shape = "3,28,28"
# we also need to specify the number of training samples in the training set
# for CIFAR-10 it is 50000
num_training_samples = 50000
# specify the number of output classes
num_classes = 10
# batch size for training
mini_batch_size =  128
# number of epochs
epochs = 100
# optimizer
optimizer='adam'
# Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be 
# initialized with pre-trained weights
use_pretrained_model = 0


# # Training
# Run the training using Amazon sagemaker CreateTrainingJob API
# 

get_ipython().run_cell_magic('time', '', 'import time\nimport boto3\nfrom time import gmtime, strftime\n\n\ns3 = boto3.client(\'s3\')\n# create unique job name \njob_name_prefix = \'sagemaker-imageclassification-cifar10\'\ntimestamp = time.strftime(\'-%Y-%m-%d-%H-%M-%S\', time.gmtime())\njob_name = job_name_prefix + timestamp\ntraining_params = \\\n{\n    # specify the training docker image\n    "AlgorithmSpecification": {\n        "TrainingImage": training_image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": \'s3://{}/{}/output\'.format(bucket, job_name_prefix)\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.p2.xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "image_shape": image_shape,\n        "num_layers": str(num_layers),\n        "num_training_samples": str(num_training_samples),\n        "num_classes": str(num_classes),\n        "mini_batch_size": str(mini_batch_size),\n        "epochs": str(epochs),\n        "learning_rate": str(learning_rate),\n        "use_pretrained_model": str(use_pretrained_model)\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 360000\n    },\n#Training data should be inside a subdirectory called "train"\n#Validation data should be inside a subdirectory called "validation"\n#The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/train/cifar10\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/validation/cifar10\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        }\n    ]\n}\nprint(\'Training job name: {}\'.format(job_name))\nprint(\'\\nInput Data Location: {}\'.format(training_params[\'InputDataConfig\'][0][\'DataSource\'][\'S3DataSource\']))')


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

# ## Plot training and validation accuracies
# 

import boto3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

client = boto3.client('logs')

lgn='/aws/sagemaker/TrainingJobs'
lsn='sagemaker-imageclassification-cifar10-2018-01-16-10-31-05/algo-1-1516099203'
log=client.get_log_events(logGroupName=lgn, logStreamName=lsn)

trn_accs=[]
val_accs=[]
for e in log['events']:
  msg=e['message']
  if 'Validation-accuracy' in msg:
        val = msg.split("=")
        val = val[1]
        val_accs.append(float(val))
  if 'Train-accuracy' in msg:
        trn = msg.split("=")
        trn = trn[1]
        trn_accs.append(float(trn))

print("Maximum validation accuracy: %f " % max(val_accs))                
fig, ax = plt.subplots()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
trn_plot, = ax.plot(range(epochs), trn_accs, label="Training accuracy")
val_plot, = ax.plot(range(epochs), val_accs, label="Validation accuracy")
plt.legend(handles=[trn_plot,val_plot])
ax.yaxis.set_ticks(np.arange(0.4, 1.05, 0.05))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
plt.show()


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

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nsage = boto3.Session().client(service_name=\'sagemaker\') \n\nmodel_name="test-image-classification-model-cifar-10epochs"\nprint(model_name)\ninfo = sage.describe_training_job(TrainingJobName=job_name)\nmodel_data = info[\'ModelArtifacts\'][\'S3ModelArtifacts\']\nprint(model_data)\ncontainers = {\'us-west-2\': \'433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest\',\n              \'us-east-1\': \'811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest\',\n              \'us-east-2\': \'825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest\',\n              \'eu-west-1\': \'685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest\'}\nhosting_image = containers[boto3.Session().region_name]\nprimary_container = {\n    \'Image\': hosting_image,\n    \'ModelDataUrl\': model_data,\n}\n\ncreate_model_response = sage.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response[\'ModelArn\'])')


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
        'InstanceType':'ml.m4.xlarge',
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

# Bird
#!wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2015/12/19/10/54/bird-1099639_960_720.jpg
# Horse
#!wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2016/02/15/13/26/horse-1201143_960_720.jpg
# Dog
get_ipython().system('wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_960_720.jpg')
# Truck
# Truck
#!wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2015/09/29/10/14/truck-truck-963637_960_720.jpg
    
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
print(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
object_categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))


# ### Clean up
# 
# When we're done with the endpoint, we can just delete it and the backing instances will be released.  Run the following cell to delete the endpoint.
# 

sage.delete_endpoint(EndpointName=endpoint_name)





# # Image classification - transfer learning demo
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
# Welcome to our end-to-end example of distributed image classification algorithm in transfer learning mode. In this demo, we will use the Amazon sagemaker image classification algorithm in transfer learning mode to fine-tune a pre-trained model (trained on imagenet data) to learn to classify a new dataset. In particular, the pre-trained model will be fine-tuned using [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). 
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

get_ipython().run_cell_magic('time', '', "import boto3\nimport re\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nbucket='jsimon-sagemaker-us' # customize to your bucket\n\ncontainers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest',\n              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',\n              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest',\n              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest'}\ntraining_image = containers[boto3.Session().region_name]\nprint(training_image)")


# ## Fine-tuning the Image classification model
# 
# The CIFAR-10 dataset consist of images from 10 categories and has 50,000 images with 5,000 images per category. 
# 
# The image classification algorithm can take two types of input formats. The first is a [recordio format](https://mxnet.incubator.apache.org/tutorials/basic/record_io.html) and the other is a [lst format](https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=im2rec). Files for both these formats are available at http://data.mxnet.io/data/cifar10/. In this example, we will use the recordio format for training and use the training/validation split.
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


# CIFAR-10

download('http://data.mxnet.io/data/cifar10/cifar10_train.rec')
download('http://data.mxnet.io/data/cifar10/cifar10_val.rec')
upload_to_s3('validation/cifar10', 'cifar10_val.rec')
upload_to_s3('train/cifar10', 'cifar10_train.rec')


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
# * **num_layers**: The number of layers (depth) for the network. We use 44 in this sample but other values can be used.
# * **num_training_samples**: This is the total number of training samples. It is set to 50000 for CIFAR-10 dataset with the current split
# * **num_classes**: This is the number of output classes for the new dataset. Imagenet was trained with 1000 output classes but the number of output classes can be changed for fine-tuning. For CIFAR-10, we use 10.
# * **epochs**: Number of training epochs
# * **learning_rate**: Learning rate for training
# * **mini_batch_size**: The number of training samples used for each mini batch. In distributed training, the number of training samples used per batch will be N * mini_batch_size where N is the number of hosts on which training is run
# 

# After setting training parameters, we kick off training, and poll for status until training is completed, which in this example, takes between 10 to 12 minutes per epoch on a p2.xlarge machine. The network typically converges after 10 epochs.  
# 

# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 18 layers
num_layers = 50
# we need to specify the input image shape for the training data
image_shape = "3,28,28"
# we also need to specify the number of training samples in the training set
# for CIFAR-10 it is 50000
num_training_samples = 50000
# specify the number of output classes
num_classes = 10
# batch size for training
mini_batch_size =  128
# number of epochs
epochs = 10
# learning rate
learning_rate = 0.01
# Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be 
# initialized with pre-trained weights
use_pretrained_model = 1


# # Training
# Run the training using Amazon sagemaker CreateTrainingJob API
# 

get_ipython().run_cell_magic('time', '', 'import time\nimport boto3\nfrom time import gmtime, strftime\n\n\ns3 = boto3.client(\'s3\')\n# create unique job name \njob_name_prefix = \'sagemaker-imageclassification-cifar10\'\ntimestamp = time.strftime(\'-%Y-%m-%d-%H-%M-%S\', time.gmtime())\njob_name = job_name_prefix + timestamp\ntraining_params = \\\n{\n    # specify the training docker image\n    "AlgorithmSpecification": {\n        "TrainingImage": training_image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": \'s3://{}/{}/output\'.format(bucket, job_name_prefix)\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.p2.8xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "image_shape": image_shape,\n        "num_layers": str(num_layers),\n        "num_training_samples": str(num_training_samples),\n        "num_classes": str(num_classes),\n        "mini_batch_size": str(mini_batch_size),\n        "epochs": str(epochs),\n        "learning_rate": str(learning_rate),\n        "use_pretrained_model": str(use_pretrained_model)\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 360000\n    },\n#Training data should be inside a subdirectory called "train"\n#Validation data should be inside a subdirectory called "validation"\n#The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/train/cifar10\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/validation/cifar10\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        }\n    ]\n}\nprint(\'Training job name: {}\'.format(job_name))\nprint(\'\\nInput Data Location: {}\'.format(training_params[\'InputDataConfig\'][0][\'DataSource\'][\'S3DataSource\']))')


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

# ## Plot training and validation accuracies
# 

import boto3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

client = boto3.client('logs')

lgn='/aws/sagemaker/TrainingJobs'
lsn='sagemaker-imageclassification-cifar10-2018-01-16-11-05-28/algo-1-1516100993'
log=client.get_log_events(logGroupName=lgn, logStreamName=lsn)

trn_accs=[]
val_accs=[]
for e in log['events']:
  msg=e['message']
  if 'Validation-accuracy' in msg:
        val = msg.split("=")
        val = val[1]
        val_accs.append(float(val))
  if 'Train-accuracy' in msg:
        trn = msg.split("=")
        trn = trn[1]
        trn_accs.append(float(trn))

print("Maximum validation accuracy: %f " % max(val_accs))
fig, ax = plt.subplots()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
trn_plot, = ax.plot(range(epochs), trn_accs, label="Training accuracy")
val_plot, = ax.plot(range(epochs), val_accs, label="Validation accuracy")
plt.legend(handles=[trn_plot,val_plot])
ax.yaxis.set_ticks(np.arange(0.4, 1.05, 0.05))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
plt.show()


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

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nsage = boto3.Session().client(service_name=\'sagemaker\') \n\nmodel_name="image-classification-cifar-transfer"\nprint(model_name)\ninfo = sage.describe_training_job(TrainingJobName=job_name)\nmodel_data = info[\'ModelArtifacts\'][\'S3ModelArtifacts\']\nprint(model_data)\ncontainers = {\'us-west-2\': \'433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest\',\n              \'us-east-1\': \'811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest\',\n              \'us-east-2\': \'825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest\',\n              \'eu-west-1\': \'685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest\'}\nhosting_image = containers[boto3.Session().region_name]\nprimary_container = {\n    \'Image\': hosting_image,\n    \'ModelDataUrl\': model_data,\n}\n\ncreate_model_response = sage.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response[\'ModelArn\'])')


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
        'InstanceType':'ml.m4.xlarge',
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

# Bird
#!wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2015/12/19/10/54/bird-1099639_960_720.jpg
# Horse
#!wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2016/02/15/13/26/horse-1201143_960_720.jpg
# Dog
get_ipython().system('wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_960_720.jpg')
# Truck
#!wget -O /tmp/test.jpg https://cdn.pixabay.com/photo/2015/09/29/10/14/truck-truck-963637_960_720.jpg
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
print(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
object_categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))


# ### Clean up
# 
# When we're done with the endpoint, we can just delete it and the backing instances will be released.  Run the following cell to delete the endpoint.
# 

sage.delete_endpoint(EndpointName=endpoint_name)





# ## Optimizing an Apache MXNet model for AWS DeepLens
# 

# This notebook shows you how to use the Intel Deep Learning Deployment Toolkit to optimize an MXNet model for Deep Lens.
# 

get_ipython().run_line_magic('env', 'TOOLKIT_BUCKET=s3://jsimon-public-us/')
get_ipython().run_line_magic('env', 'TOOLKIT_NAME=toolkit.tgz')
get_ipython().run_line_magic('env', 'TOOLKIT_DIR=l_deeplearning_deploymenttoolkit_2017.1.0.5852')

get_ipython().run_line_magic('env', 'MODEL_BUCKET=s3://jsimon-public-us/')
get_ipython().run_line_magic('env', 'MODEL_NAME=Inception-BN')

get_ipython().run_line_magic('env', 'OPT_DIR=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/model_optimizer/model_optimizer_mxnet')
get_ipython().run_line_magic('env', 'OPT_PRECISION=FP16')
get_ipython().run_line_magic('env', 'OPT_FUSE=YES')


get_ipython().run_cell_magic('bash', '', '\necho "*** Downloading toolkit"\naws s3 cp $TOOLKIT_BUCKET$TOOLKIT_NAME .\necho "*** Installing toolkit"\ntar xfz $TOOLKIT_NAME\ncd $TOOLKIT_DIR\nchmod 755 install.sh\nsudo ./install.sh -s silent.cfg \necho "*** Done"')


get_ipython().run_cell_magic('bash', '', '\n#conda create -n intel_toolkit -y\npython -m ipykernel install --user --name intel_toolkit --display-name "intel_toolkit"\n\nsource activate intel_toolkit\ncd $OPT_DIR\npip install -r requirements.txt ')


get_ipython().run_cell_magic('bash', '', '\necho "*** Downloading model"\naws s3 cp $MODEL_BUCKET$MODEL_NAME"-symbol.json" .\naws s3 cp $MODEL_BUCKET$MODEL_NAME"-0000.params" .\necho "*** Done"')


get_ipython().run_cell_magic('bash', '', '\nsource activate intel_toolkit\necho "*** Converting model"\npython $OPT_DIR/mo_mxnet_converter.py --models-dir . --output-dir . --model-name $MODEL_NAME --precision $OPT_PRECISION --fuse $OPT_FUSE\nls mxnet_$MODEL_NAME*\necho "*** Done"')





# ## Predicting handmade MNIST samples
# 

import mxnet as mx
import numpy as np
import cv2, time
from IPython.display import Image
from collections import namedtuple

np.set_printoptions(precision=4, suppress=True)


# This function loads an image from disk and turns it into a normalized NDArray shaped (1,1,x_size,y_size).
# 

def loadImage(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  img = img / 255
  img = np.expand_dims(img, axis=0)
  img = np.expand_dims(img, axis=0)
  return mx.nd.array(img)


# This function loads an image, forwards it through the model and returns the predicted output.
# 

def predict(model, filename):
  array = loadImage(filename)
  #print(array.shape)
  Batch = namedtuple('Batch', ['data'])
  mod.forward(Batch([array]))
  pred = mod.get_outputs()[0].asnumpy()
  return pred


# This function loads a model and prepares it for inference.
# 

def loadModel(model, epochs):
  model, arg_params, aux_params = mx.model.load_checkpoint(model, epochs)
  mod = mx.mod.Module(model)
  mod.bind(for_training=False, data_shapes=[('data', (1,1,28,28))])
  mod.set_params(arg_params, aux_params)
  return mod


#mod = loadModel("mlp", 50)
mod = loadModel("lenet", 25)


Image(filename="./0.png")


print(predict(mod, "./0.png"))


Image(filename="./1.png")


print(predict(mod, "./1.png"))


Image(filename="./2.png")


print(predict(mod, "./2.png"))


Image(filename="./3.png")


print(predict(mod, "./3.png"))


Image(filename="./4.png")


print(predict(mod, "./4.png"))


Image(filename="./5.png")


print(predict(mod, "./5.png"))


Image(filename="./6.png")


print(predict(mod, "./6.png"))


Image(filename="./7.png")


print(predict(mod, "./7.png"))


Image(filename="./8.png")


print(predict(mod, "./8.png"))


Image(filename="./9.png")


print(predict(mod, "./9.png"))





# ## Learning MNIST with a Multi-Layer Perceptron
# 

# First, let's download the data set.
# 

get_ipython().system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
get_ipython().system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
get_ipython().system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
get_ipython().system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
get_ipython().system('gzip -d train*.gz t10k*.gz')


import mxnet as mx
import logging
import os


logging.basicConfig(level=logging.INFO)

nb_epochs=50


# MXNet provides a convenient iterator for MNIST. We use it to build the training and the validation iterators.
# 

train_iter = mx.io.MNISTIter(shuffle=True)
val_iter = mx.io.MNISTIter(image="./t10k-images-idx3-ubyte", label="./t10k-labels-idx1-ubyte")


# We build a Multi-Layer Perceptron:
# - an input layer receiving a flattened MNIST image (28x28 --> 784),
# - a fully connected hidden layer with 512 neurons activated by the ReLU function,
# - a dropout layer to prevent overfitting,
# - a second fully connected hidden layer with 256 neurons activated by the ReLU function,
# - a second dropout layer to prevent overfitting,
# - an output layer with 10 neurons (because we have 10 categories), holding probabilities computed by the SoftMax function.
# 

data = mx.sym.Variable('data')
data = mx.sym.Flatten(data=data)
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=512)
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
drop1= mx.sym.Dropout(data=act1,p=0.2)
fc2  = mx.sym.FullyConnected(data=drop1, name='fc2', num_hidden = 256)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
drop2= mx.sym.Dropout(data=act2,p=0.2)
fc3  = mx.sym.FullyConnected(data=drop2, name='fc3', num_hidden=10)
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')


# Now, we need to:
# - bind the model to the training set,
# - initialize the parameters, i.e. set initial values for all weights,
# - pick an optimizer and a learning rate, to adjust weights during backpropagation
# 

mod = mx.mod.Module(mlp, context=mx.gpu(0))
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mod.init_params(initializer=mx.init.Xavier())
#mod.init_optimizer('sgd', optimizer_params=(('learning_rate', 0.01),))
mod.init_optimizer('adagrad', optimizer_params=(('learning_rate', 0.1),))


# Time to train!
# 

mod.fit(train_iter, eval_data=val_iter, num_epoch=nb_epochs,
        batch_end_callback=mx.callback.Speedometer(128, 100))


mod.save_checkpoint("mlp", nb_epochs)


# Let's measure validation accuracy.
# 

metric = mx.metric.Accuracy()
mod.score(val_iter, metric)
print(metric.get())





# ## Classifying images with pre-trained Apache MXNet models
# 

# First, let's download three image classification models from the Apache MXNet [model zoo](http://mxnet.io/model_zoo/).
# * **VGG-16**: the 2014 classification winner at the [ImageNet Large Scale Visual Recognition Challenge](http://image-net.org/challenges/LSVRC).
# * **Inception v3**, an evolution of GoogleNet, the 2014 winner for object detection.
# * **ResNet-152**, the 2015 winner in multiple categories.
# 
# Why would we want to try multiple models? Why don't we simply pick the one with the best accuracy? As we will see later on, even though these models have been trained on the same data set and optimized for maximum accuracy, they do behave slightly differently on **specific images**: maybe one of the models them will actually do a better job at solving your business problem. **Prediction speed** can vary a lot as well and that's an important factor for many applications.
# 
# For each model, we need to download two files:
# * the **symbol** file containing the JSON definition of the neural network: layers, connections, activation functions, etc.
# * the **weights** file storing values for all neuron weights, a.k.a. parameters, learned by the network during the training phase.
# 

get_ipython().system('wget http://data.dmlc.ml/models/imagenet/vgg/vgg16-symbol.json -O vgg16-symbol.json')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params -O vgg16-0000.params')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-symbol.json -O Inception-BN-symbol.json')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params -O Inception-BN-0000.params')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-symbol.json -O resnet-152-symbol.json')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-0000.params -O resnet-152-0000.params')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/synset.txt -O synset.txt')


# Let's take a look at the first lines of VGG-16 symbol file. We can see the definition of the input layer ('data'), the input weights and the biases for the first convolution layer. A convolution operation is defined ('conv1_1') as well as a Rectified Linear Unit activation function ('relu1_1').
# 

get_ipython().system('head -48 vgg16-symbol.json')


# All three models have been pre-trained on the ImageNet data set which includes over 1.2 million pictures of objects and animals sorted in 1,000 categories. We can view these categories in the synset.txt file.
# 

get_ipython().system('head -10 synset.txt')


import mxnet as mx
import numpy as np
import cv2,sys,time
from collections import namedtuple
from IPython.core.display import Image, display

print("MXNet version: %s" % mx.__version__)


# Now, let's load a model.
# 
# First, we have to load the **weights** and **model description** from file. MXNet calls this a **checkpoint**: indeed, it's good practice to save weights after each training epoch. Once training is complete, we can look at the training log and pick the weights for the best epoch, i.e. the one with the highest validation accuracy: it's quite likely it won't be the very last one!
# 
# Once loading is complete, we get a *Symbol* object and the weights, a.k.a model parameters. We then create a new *Module* and assign it the input *Symbol*. We could select the *context* where we want to run the model: the default behavior is to use a CPU context. There are two reasons for this:
# * first, this will allow you to test the notebook even if your machine is not equipped with a GPU :)
# * second, we're going to predict a single image and we don't have any specific performance requirements. For production applications where you'd want to predict large batches of images with the best possible throughput, a GPU would definitely be the way to go.
# 
# Then, we bind the input *Symbol* to input data: we have to call it ‘data’ because that’s its name in the **input layer** of the network (remember the first few lines of the JSON file).
# 
# Finally, we define the **shape** of ‘data’ as 1 x 3 x 224 x 224. ‘224 x 224’ is the image resolution, that’s how the model was trained. ‘3’ is the number of channels : red, green and blue (in this order). ‘1’ is the batch size: we’ll predict one image at a time.
# 

def loadModel(modelname, gpu=False):
        sym, arg_params, aux_params = mx.model.load_checkpoint(modelname, 0)
        arg_params['prob_label'] = mx.nd.array([0])
        arg_params['softmax_label'] = mx.nd.array([0])
        if gpu:
            mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))
        else:
            mod = mx.mod.Module(symbol=sym)
        mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
        mod.set_params(arg_params, aux_params)
        return mod


# We also need to load the 1,000 categories stored in the synset.txt file. We'll need the actual descriptions at prediction time.
# 

def loadCategories():
        synsetfile = open('synset.txt', 'r')
        synsets = []
        for l in synsetfile:
                synsets.append(l.rstrip())
        return synsets
    
synsets = loadCategories()
print(synsets[:10])


# Now let's write a function to load an image from file. Remember that the model expects a 4-dimension *NDArray* holding the red, green and blue channels of a single 224 x 224 image. We’re going to use the **OpenCV** library to build this *NDArray* from our input image.
# 
# Here are the steps:
# * read the image: this will return a **numpy array** shaped as (image height, image width, 3), with the three channels in **BGR** order (blue, green and red).
# * convert the image to **RGB**.
# * resize the image to **224 x 224**.
# * **reshape** the array from (image height, image width, 3) to (3, image height, image width).
# * add a **fourth dimension** and build the *NDArray*.
# 

def prepareNDArray(filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224,))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        array = mx.nd.array(img)
        print(array.shape)
        return array


# Let's take care of prediction. Our parameters are an image, a model, a list of categories and the number of top categories we'd like to return. 
# 
# Remember that a *Module* object must feed data to a model in **batches**: the common way to do this is to use a **data iterator**. Here, we’d like to predict a single image, so although we could use a data iterator, it’d probably be overkill. Instead, let's create a named tuple, called *Batch*, which will act as a fake iterator by returning our input *NDArray* when its 'data' attribute is referenced.
# 
# Once the image has been forwarded, the model outputs an *NDArray* holding **1,000 probabilities**, corresponding to the 1,000 categories it has been trained on: the *NDArray* has only one line since batch size is equal to 1. 
# 
# Let’s turn this into an array with *squeeze()*. Then, using *argsort()*, we create a second array holding the **index** of these probabilities sorted in **descending order**. Finally, we return the top n categories and their description.
# 

def predict(filename, model, categories, n):
        array = prepareNDArray(filename)
        Batch = namedtuple('Batch', ['data'])
        t1 = time.time()
        model.forward(Batch([array]))
        prob = model.get_outputs()[0].asnumpy()
        t2 = time.time()
        print("Predicted in %.2f microseconds" % (t2-t1))
        prob = np.squeeze(prob)
        sortedprobindex = np.argsort(prob)[::-1]
        
        topn = []
        for i in sortedprobindex[0:n]:
                topn.append((prob[i], categories[i]))
        return topn


# Time to put everything together. Let's load all three models.
# 

def init(modelname, gpu=False):
        model = loadModel(modelname,gpu)
        categories = loadCategories()
        return model, categories

vgg16,categories = init("vgg16")
resnet152,categories = init("resnet-152")
inceptionv3,categories = init("Inception-BN")


# Before classifying images, let's take a closer look to some of the VGG-16 **parameters** we just loaded from the '.params' file. First, let's print the names of all **layers**.
# 

params = vgg16.get_params()

layers = []
for layer in params[0].keys():
    layers.append(layer)
    
layers.sort()    
print(layers)


# For each layer, we see two components: the weights and the biases. Count the weights and you'll see that there are **sixteen** layers: thirteen convolutional layers and three fully connected layers. Now you know why this model is called **VGG-16** :)
# 
# Now let's print the weights for the last fully connected layer.
# 

print(params[0]['fc8_weight'])


# Did you notice the **shape** of this matrix? **1000x4096**. This layer contains **1,000 neurons**: each of them will store the **probability** of the image belonging to a specific category. Each neuron is also fully connected to all **4,096 neurons** in the previous layer ('fc7').
# 

# OK, enough exploring! Now let's use these models to classify our own images.
# 

image = "violin.jpg"

display(Image(filename=image))

topn = 5
print ("*** VGG16")
print (predict(image,vgg16,categories,topn))
print ("*** ResNet-152")
print (predict(image,resnet152,categories,topn))
print ("*** Inception v3")
print (predict(image,inceptionv3,categories,topn))


# Let's try again with a **GPU context** this time
# 

vgg16,categories = init("vgg16", gpu=True)
resnet152,categories = init("resnet-152", gpu=True)
inceptionv3,categories = init("Inception-BN", gpu=True)

print ("*** VGG16")
print (predict(image,vgg16,categories,topn))
print ("*** ResNet-152")
print (predict(image,resnet152,categories,topn))
print ("*** Inception v3")
print (predict(image,inceptionv3,categories,topn))


# ***If you get an error about GPU support, either your machine or instance is not equipped with a GPU or you're using a version of MXNet that hasn't been built with GPU support (USE_CUDA=1)***
# 
# The difference in performance is quite noticeable: between **15x** and **20x**. If we predicted **multiple images** at the same time, the gap would widen even more due to the **massive parallelism** of GPU architectures.
# 
# Now it's time to try your **own images**. Just copy them in the same folder as this notebook, update the filename in the cell above and run the predict() calls again.
# 
# Have fun with pre-trained models!
# 




