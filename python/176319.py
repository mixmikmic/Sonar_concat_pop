# ## Hotdog or Not HotDog
# 
# Welcome to this SageMaker Notebook! This is an entirely managed notebook service that you can use to create and edit machine learning models. We will be using it today to create a binary image classification model using the Apache MXNet deep learning framework. We will then learn how to delpoy this model onto our DeepLens device.
# 
# In this notebook we will be to using MXNet's Gluon interface, to download and edit a pre-trained ImageNet model and transform it into binary classifier, which we can use to differentiate between hot dogs and not hot dogs.
# 
# ### Setup
# 
# Before we start, make sure the kernel in the the notebook is set to the correct one, `condamxnet3.6` which has all the dependencies we will need for this tutorial already installed.
# 
# First we'll start by importing a bunch of packages into the notebook that you'll need later and installing any required packages that are missing into our notebook kernel.
# 

get_ipython().run_cell_magic('bash', '', 'conda install scikit-image')


from __future__ import print_function
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from collections import OrderedDict
import skimage.io as io
import numpy as np

import mxnet as mx


# ## Model
# 
# The model we will be downloading and editing is [SqueezeNet](https://arxiv.org/abs/1602.07360), an extremely efficient image classification model that achived 2012 State of the Art accuracy on the popular [ImageNet](http://www.image-net.org/challenges/LSVRC/), image classification challenge. SqueezeNet is just a convolutional neural network, with an architecture chosen to have a small number of parameters and to require a minimal amount of computation. It's especially popular for folks that need to run CNNs on low-powered devices like cell phones and other internet-of-things devices, such as DeepLens. The MXNet Deep Learning framework offers squeezenet v1.0 and v1.1 that are pretrained on ImageNet through it's model Zoo.
# 
# ## Pulling the pre-trained model
# The MXNet model zoo  gives us convenient access to a number of popular models,
# both their architectures and their pretrained parameters.
# Let's download SqueezeNet right now with just a few lines of code.
# 

from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

# get pretrained squeezenet
net = models.squeezenet1_1(pretrained=True, prefix='deep_dog_')
# hot dog happens to be a class in imagenet.
# we can reuse the weight for that class for better performance
# here's the index for that class for later use
imagenet_hotdog_index = 713


# ### DeepDog Net
# 
# In vision networks its common that the first set of layers learns the task of recognizing edges, curves and other important visual features of the input image. We call this feature extraction, and once the abstract features are extracted we can leverage a much simpler model to classify images using these features.
# 
# We will use the feature extractor from the pretrained squeezenet (every layer except the last one) to build our own classifier for hotdogs. Conveniently, the MXNet model zoo handles the decaptiation for us. All we have to do is specify the number out of output classes in our new task, which we do via the keyword argument `classes=2`.
# 

deep_dog_net = models.squeezenet1_1(prefix='deep_dog_', classes=2)
deep_dog_net.collect_params().initialize()
deep_dog_net.features = net.features

# Lets take a look at what this network looks like
print(deep_dog_net)


# The network can already be used for prediction. However, since it hasn't been finetuned yet so the network performance could not be optimal.
# 
# Let's test it out by defining a prediction function to feed a local image into the network and get the predicted output
# 

from skimage.color import rgba2rgb

def classify_hotdog(net, url):
    I = io.imread(url)
    if I.shape[2] == 4:
        I = rgba2rgb(I)
    image = mx.nd.array(I).astype(np.uint8)
    image = mx.image.resize_short(image, 256)
    image, _ = mx.image.center_crop(image, (224, 224))
    image = mx.image.color_normalize(image.astype(np.float32)/255,
                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                     std=mx.nd.array([0.229, 0.224, 0.225]))
    image = mx.nd.transpose(image.astype('float32'), (2,1,0))
    image = mx.nd.expand_dims(image, axis=0)
    out = mx.nd.SoftmaxActivation(net(image))
    print('Probabilities are: '+str(out[0].asnumpy()))
    result = np.argmax(out.asnumpy())
    outstring = ['Not hotdog!', 'Hotdog!']
    print(outstring[result])


# Now lets download a hot dog image and an image of another object to our local directory to test this model on
# 

get_ipython().run_cell_magic('bash', '', 'wget http://www.wienerschnitzel.com/wp-content/uploads/2014/10/hotdog_mustard-main.jpg\nwget https://www.what-dog.net/Images/faces2/scroll001.jpg')


# To make the defined network run quickly we usually hybridize it first. 
# This also allows us to serialize and export our model
deep_dog_net.hybridize()

# Let's run the classification on our tow downloaded images to see what our model comes up with
classify_hotdog(deep_dog_net, './hotdog_mustard-main.jpg') # check for hotdog
classify_hotdog(deep_dog_net, './scroll001.jpg') # check for not-hotdog


deep_dog_net.export('hotdog_or_not_model')


# The predictions are a bit off so we can download a set of new parameters for the model that we have pre-optimized through a "fine tuning" process, where we retrained the model with images of hotdogs and not hotdogs. We can then apply these new parameters to our model to make it even more accurate.
# 

from mxnet.test_utils import download

download('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/deep-dog-5a342a6f.params',
         overwrite=True)
deep_dog_net.load_params('deep-dog-5a342a6f.params', mx.cpu())
deep_dog_net.hybridize()
classify_hotdog(deep_dog_net, './hotdog_mustard-main.jpg')
classify_hotdog(deep_dog_net, './scroll001.jpg')


# The predictions seem reasonable, so we can export this as a serialized model to our local dirctory. This is a simple one line command, which produces a set of two files: a json file holding the network architecture, and a params file holding the parameters the network learned.
# 

deep_dog_net.export('hotdog_or_not_model_v2')


# Now let's push this serialized model to S3, where we can then optimize it for our DeepLense device and then push it down onto our device for inference.
# 

import boto3
import re

assumed_role = boto3.client('sts').get_caller_identity()['Arn']
s3_access_role = re.sub(r'^(.+)sts::(\d+):assumed-role/(.+?)/.*$', r'\1iam::\2:role/\3', assumed_role)
print(s3_access_role)
s3 = boto3.resource('s3')
bucket= 'your s3 bucket name here' 

json = open('hotdog_or_not_model-symbol.json', 'rb')
params = open('hotdog_or_not_model-0000.params', 'rb')
s3.Bucket(bucket).put_object(Key='test/hotdog_or_not_model-symbol.json', Body=json)
s3.Bucket(bucket).put_object(Key='test/hotdog_or_not_model-0000.params', Body=params)





# ## Hotdog or Not HotDog
# 
# Welcome to this Amazon SageMaker Notebook! This is an entirely managed notebook service that you can use to create and edit machine learning models with Python. We will be using it today to create a binary image classification model using the Apache MXNet deep learning framework. We will then learn how to delpoy this model onto our AWS DeepLens device.
# 
# In this notebook we will be to using MXNet's Gluon interface, to download and edit a pre-trained [ImageNet](http://www.image-net.org/) model and transform it into binary classifier, which we can use to differentiate between hot dogs and other objects.
# 
# ### Setup
# 
# Before we start, make sure the kernel in the the notebook is set to the correct one, `condamxnet3.6` which has most of the the Python library dependencies we will need for this tutorial already installed.
# 
# First we'll start by importing a bunch of packages into the notebook that you'll need later and installing any required packages that are missing into our notebook kernel.
# 

get_ipython().run_cell_magic('bash', '', 'conda install scikit-image')


from __future__ import print_function
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from collections import OrderedDict
import skimage.io as io
import numpy as np

import mxnet as mx


# ## Model
# 
# The model we will be downloading and editing is [SqueezeNet](https://arxiv.org/abs/1602.07360), an extremely efficient image classification model that achived 2012 State of the Art accuracy on the popular [ImageNet](http://www.image-net.org/challenges/LSVRC/), image classification challenge. SqueezeNet is just a convolutional neural network (CNN), with an architecture chosen to have a small number of parameters and to require a minimal amount of computation. It's especially popular for folks that need to run CNNs on low-powered devices like cell phones and other internet-of-things devices. The MXNet Deep Learning framework offers SqueezeNet v1.0 and v1.1 that are pretrained on ImageNet through it's [model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html).
# 
# ![image](https://community.arm.com/cfs-file/__key/communityserver-discussions-components-files/18/pastedimage1485588767177v1.png)
# Image 1. The layerwise visualization of the SqueezeNet architecture
# 
# ## Pulling the pre-trained model
# The MXNet model zoo  gives us convenient access to a number of popular models,
# both their architectures and their pretrained parameters.
# Let's download a pretrained SqueezeNet right now with just a few lines of code.
# 

from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

# Get pretrained SqueezeNet
net = models.squeezenet1_1(pretrained=True, prefix='deep_dog_')

# hot dog happens to be a class in imagenet, which this model was trained on
# we can reuse the weight for that class for better performance
# here's the index for that class
imagenet_hotdog_index = 713


# ### DeepDog Net
# 
# In vision networks it's common that the first set of layers learns the task of recognizing edges, curves and other important visual features of the input image. We call this feature extraction, and once the abstract features are extracted we can leverage a simpler model at the end of the network to classify images using these features.
# 
# We will use the feature extractor from the pretrained SqueezeNet (every layer except the last one) to build our own classifier for hotdogs. Conveniently, the MXNet model zoo handles the editing of the model for us. All we have to do is specify the number out of output classes in our new task, which we do via the keyword argument `classes=2`.
# 

# Create the model with a two class output classifier and apply the pretrained weights
deep_dog_net = models.squeezenet1_1(prefix='deep_dog_', classes=2)
deep_dog_net.collect_params().initialize()
deep_dog_net.features = net.features

# Lets take a look at what this network looks like
print(deep_dog_net)


# The network can already be used for prediction. However, since it hasn't been fine tuned yet for the hot dog classification task the network performance is not optimal.
# 
# Let's test it out by defining a prediction function to preprocess an image into the shape and color scheme expected by the network and feed it in to get the predicted output.
# 

from skimage.color import rgba2rgb

def classify_hotdog(net, url):

    # Pull in image and ensure there are only 3 color channels (RGB)
    I = io.imread(url)
    if I.shape[2] == 4:
        I = rgba2rgb(I)
        
    # Normalize the color channels and crop the image to the expected input size (224,224)
    image = mx.nd.array(I).astype(np.uint8)
    image = mx.image.resize_short(image, 256)
    image, _ = mx.image.center_crop(image, (224, 224))
    image = mx.image.color_normalize(image.astype(np.float32)/255,
                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                     std=mx.nd.array([0.229, 0.224, 0.225]))

    # Flip the color channels from RGB to the expected BGR input
    image = mx.nd.transpose(image.astype('float32'), (2,1,0))
    image = mx.nd.expand_dims(image, axis=0)
    
    # Feed the pre-processed image into the net and get the predicted result
    inference_result = net(image)
    print('Raw inference output is:'+str(inference_result))
    
    # Squeeze the inference result into a softmax function to turn it into a probability
    out = mx.nd.SoftmaxActivation(inference_result)
    print('Probabilities are: '+str(out[0].asnumpy()))
    
    # Take max probability to predict if the image has a hotdog or not
    result = np.argmax(out.asnumpy())
    outstring = ['Not hotdog!', 'Hotdog!']
    print(outstring[result])


# Now let's download a hot dog image (hotdog_mustard-main.jpg) and an image of a dog (scroll001.jpg) to our local directory to test this model on
# 

get_ipython().run_cell_magic('bash', '', 'wget http://www.wienerschnitzel.com/wp-content/uploads/2014/10/hotdog_mustard-main.jpg\nwget https://www.what-dog.net/Images/faces2/scroll001.jpg')


# Before deploying our net we usually want to run the hybridize function on it, which will essentially "compile" the graph, allowing it to run much faster for both inference and training. This will also allow us to serialize the network as well as its parameters and export it to a file.
# 

deep_dog_net.hybridize()


# Let's run the classification on our tow downloaded images to see what our model comes up with
classify_hotdog(deep_dog_net, './hotdog_mustard-main.jpg') # check for hotdog
classify_hotdog(deep_dog_net, './scroll001.jpg') # check for not hotdog


# As you can see the predictions are not very accurate. The hot dog is classified as not a hot dog, due to the fact that original model was trained on far more images of objects other than hot dogs. This is typically reffered to as a class imbalance problem. To improve the model we can download a set of new parameters for the model that we have pre-optimized through a "fine tuning" process, where we retrained the model using a more balanced set of images of hotdogs and other objects. We can then apply these new parameters to our model to make it even more accurate.
# 

from mxnet.test_utils import download

# Pull the new parameters using the download utility provided by MXNet
download('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/deep-dog-5a342a6f.params',
         overwrite=True)

# This simply applies the new parameters onto the model we already have
deep_dog_net.load_params('deep-dog-5a342a6f.params', mx.cpu())

deep_dog_net.hybridize()
classify_hotdog(deep_dog_net, './hotdog_mustard-main.jpg')
classify_hotdog(deep_dog_net, './scroll001.jpg')


# The predictions seem reasonable, so we can export this as a serialized model to our local dirctory. This is a simple one line command, which produces a set of two files: a json file (hotdog_or_not_model-symbol.json) holding the network architecture, and a params file (hotdog_or_not_model-0000.params) holding the parameters the network learned.
# 

deep_dog_net.export('hotdog_or_not_model')


# Now let's push this serialized model to S3, where we can then optimize it for our AWS DeepLens and then push it down onto our device for inference.
# 

import boto3
import re

assumed_role = boto3.client('sts').get_caller_identity()['Arn']
s3_access_role = re.sub(r'^(.+)sts::(\d+):assumed-role/(.+?)/.*$', r'\1iam::\2:role/\3', assumed_role)
print(s3_access_role)
s3 = boto3.resource('s3')

json = open('hotdog_or_not_model-symbol.json', 'rb')
params = open('hotdog_or_not_model-0000.params', 'rb')
s3.Bucket('test-bucket').put_object(Key='hotdog_or_not_model-symbol.json', Body=json)
s3.Bucket('test-bucket').put_object(Key='hotdog_or_not_model-0000.params', Body=params)


