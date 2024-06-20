# # Demo of JS &harr; Python communication
# 

from IPython.display import HTML


js="""
alert("Hello Javascript (created in python string")
// Lots of pre-written stuff could go here - all generated from Python
"""

# This is Python, printing out the javascript into the browser window
HTML('<script type="text/Javascript">%s</script>' % (js,))

# Nothing will appear be 'output' - but an annoying pop-up will...


# ### Create an HTML placeholder 
# 

html="""
<input type="text" id="textinput" value="12"/>
<input type="submit" id="textsubmit">
"""

# This is Python, printing out the javascript into the browser window
HTML(html)


# ### Create a Python function and Hook up the interactivity
# 

def recalculate_cell_in_python(v):
    if v % 2 == 0: return v/2
    return v*3+1

# Lots more Python could go here
# You can also have side-effects, etc

# This python import will be 'visible' for the python code executed by the javascript callback
# because that happens 'afterwards' as far as the Python kernel is concerned
import json 

js="""
var kernel = IPython.notebook.kernel;
$('#textsubmit').off('click').on('click', function(e) {

    var javascript_cell_value = $('#textinput').val();

    var cmd=[
      'python_new_value = recalculate_cell_in_python('+javascript_cell_value+')',
      'json.dumps( dict( v=python_new_value ) )'
    ].join(';');

    kernel.execute(cmd, {iopub: {output: handle_python_output}}, {silent:false});

    function handle_python_output(msg) {
      //console.log(msg);
      if( msg.msg_type == "error" ) {
        console.log("Javascript received Python error : ", msg.content);
      }
      else {  // execute_result
        var res_str = msg.content.data["text/plain"];
        // Take off surrounding quotes
        var res=JSON.parse( res_str.replace(/^['"](.*)['"]$/, "$1") ); 
        $('#textinput').val( res.v );
      }
    }
    
    return false;
});
"""

# Again,this is a Python cell, printing out the javascript into the browser window
HTML('<script type="text/Javascript">%s</script>' % (js,))


# ### Go back up to the text-box, and play around...
# 




# ## Flickr30k to Features
# 
# *   P. Young, A. Lai, M. Hodosh, and J. Hockenmaier. _From image description to visual denotations: New similarity metrics for semantic inference over event descriptions._ Transactions of the Association for Computational Linguistics (to appear).
# 
# 

import os

import tensorflow.contrib.keras as keras
import numpy as np

import datetime
t_start=datetime.datetime.now()

import pickle


image_folder_path = './data/Flickr30k/flickr30k-images'


output_dir = './data/cache'

output_filepath = os.path.join(output_dir, 
                                'FEATURES_%s_%s.pkl' % ( 
                                 image_folder_path.replace('./', '').replace('/', '_'),
                                 t_start.strftime("%Y-%m-%d_%H-%M"),
                                ), )
output_filepath


from tensorflow.contrib.keras.api.keras.applications.inception_v3 import decode_predictions
from tensorflow.contrib.keras.api.keras.preprocessing import image as keras_preprocessing_image


from tensorflow.contrib.keras.api.keras.applications.inception_v3 import InceptionV3, preprocess_input

BATCHSIZE=16


model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
print("InceptionV3 loaded")


# #### Plan 
# 
# *  Form a list of every file in the image directory
# *  Run InceptionV3 over the list
# *  Save off features to an easy-to-load filetype
# 

import re
good_image = re.compile( r'\.(jpg|png|gif)$', flags=re.IGNORECASE )

img_arr = [ f for f in os.listdir(image_folder_path) if good_image.search(f) ]
', '.join( img_arr[:3] ), ', '.join( img_arr[-3:] )


# Create a generator for preprocessed images
def preprocessed_image_gen():
    #target_size=model.input_shape[1:]
    target_size=(299, 299, 3)
    print("target_size", target_size)
    for img_name in img_arr:
        #print("img_name", img_name)
        img_path = os.path.join(image_folder_path, img_name)
        img = keras_preprocessing_image.load_img(img_path, target_size=target_size)
        yield keras.preprocessing.image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)  # This is to make a single image into a suitable array

def image_batch(batchsize=BATCHSIZE):
    while True:  # This needs to run 'for ever' for Keras input, even if only a fixed number are required
        preprocessed_image_generator = preprocessed_image_gen()
        start = True
        for img in preprocessed_image_generator:
            if start:
                arr, n, start = [], 0, False
            arr.append(img)
            n += 1
            if n>=batchsize: 
                stack = np.stack( arr, axis=0 )
                #print("stack.shape", stack.shape)
                preprocessed = preprocess_input( stack )
                #print("preprocessed.shape", preprocessed.shape)
                yield preprocessed
                start=True
        if len(arr)>0:
            stack = np.stack( arr, axis=0 )
            print("Final stack.shape", stack.shape)
            preprocessed = preprocess_input( stack )
            print("Final preprocessed.shape", preprocessed.shape)
            yield preprocessed


if False:
    image_batcher = image_batch()
    batch = next(image_batcher)
    features = model.predict_on_batch(batch)
    features.shape


# This should do the batch creation on the CPU and the analysis on the GPU asynchronously.
import math  # for ceil

t0=datetime.datetime.now()

features = model.predict_generator(image_batch(), steps = math.ceil( len(img_arr)/BATCHSIZE) )  #, verbose=1

features.shape, (datetime.datetime.now()-t0)/len(img_arr)*1000.


# Save the data into a useful structure

save_me = dict(
    features = features,
    img_arr = img_arr,
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
with open( output_filepath, 'wb') as f:
    pickle.dump(save_me, f)
    
print("Features saved to '%s'" %(output_filepath,))





# # Art Style Transfer
# 
# This notebook is an implementation of the algorithm described in "A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576) by Gatys, Ecker and Bethge. Additional details of their method are available at http://arxiv.org/abs/1505.07376 and https://bethgelab.org/deepneuralart/.
# 
# An image is generated which combines the content of a photograph with the "style" of a painting. This is accomplished by jointly minimizing the squared difference between feature activation maps of the photo and generated image, and the squared difference of feature correlation between painting and generated image. A total variation penalty is also applied to reduce high frequency noise. 
# 

import theano
import theano.tensor as T

import lasagne
from lasagne.utils import floatX

import numpy as np
import pickle

#import skimage.transform
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

AS_PATH='../images/art-style'


# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: non-commercial use only

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

IMAGE_W = 224

# Note: tweaked to use average pooling instead of maxpooling
def build_model():
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    return net


# Download the normalized pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl
# (original source: https://bethgelab.org/deepneuralart/)

# !wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl


# build VGG net and load weights

net = build_model()

values = pickle.load(open('../data/VGG/vgg19_normalized.pkl'))['param values']
lasagne.layers.set_all_param_values(net['pool5'], values)

print("Loaded Model parameters")


MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        #im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W/h), preserve_range=True)
        im = scipy.misc.imresize(im, (IMAGE_W, w*IMAGE_W/h))
    else:
        #im = skimage.transform.resize(im, (h*IMAGE_W/w, IMAGE_W), preserve_range=True)
        im = scipy.misc.imresize(im, (h*IMAGE_W/w, IMAGE_W))

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])


photo = plt.imread('%s/photos/Tuebingen_Neckarfront.jpg' % AS_PATH)
rawim, photo = prep_image(photo)
plt.imshow(rawim)


art = plt.imread('%s/styles/960px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg' % AS_PATH)
rawim, art = prep_image(art)
plt.imshow(rawim)


def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g


def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]
    
    loss = 1./2 * ((x - p)**2).sum()
    return loss


def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()


layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layers = {k: net[k] for k in layers}


# Precompute layer activations for photo and artwork
input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                  for k, output in zip(layers.keys(), outputs)}
art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                for k, output in zip(layers.keys(), outputs)}


# Get expressions for layer activations for generated image
generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

gen_features = lasagne.layers.get_output(layers.values(), generated_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}


# Define loss function
losses = []

# content loss
losses.append(0.001 * content_loss(photo_features, gen_features, 'conv4_2'))

# style loss
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv1_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv2_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv3_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv4_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv5_1'))

# total variation penalty
losses.append(0.1e-7 * total_variation_loss(generated_image))

total_loss = sum(losses)


grad = T.grad(total_loss, generated_image)


# Theano functions to evaluate loss and gradient
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)

# Helper functions to interface with scipy.optimize
def eval_loss(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return f_loss().astype('float64')

def eval_grad(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype('float64')


# Initialize with a noise image
generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

x0 = generated_image.get_value().astype('float64')
xs = []
xs.append(x0)

# Optimize, saving the result periodically
for i in range(8):
    print(i)
    #scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
    scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=7)  # same as Keras
    x0 = generated_image.get_value().astype('float64')
    xs.append(x0)


def deprocess(x):
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x


plt.figure(figsize=(12,12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.gca().xaxis.set_visible(False)    
    plt.gca().yaxis.set_visible(False)    
    plt.imshow(deprocess(xs[i]))
plt.tight_layout()


plt.figure(figsize=(8,8))
plt.imshow(deprocess(xs[-1]), interpolation='nearest')





# ## Examine overfitted feature distributions
# 

# ### Download CIFAR10/CIFAR100 for PyTorch
# 
# Model zoo : https://github.com/aaron-xichen/pytorch-playground
# 
# CIFAR background : http://kele.github.io/cifar10-classification-summary.html
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

use_cuda=torch.cuda.is_available()


random_seed = 42
batch_size = 200

#learning_rate, momentum = 0.01, 0.5  # SGD with momentum
learning_rate = 0.001   # SGD+Adam

log_interval = 20 # Num of batches between log messages


import numpy as np

import os
import time


torch.manual_seed(random_seed)
if use_cuda:
    torch.cuda.manual_seed(random_seed)


#dataset = datasets.CIFAR10    # 170Mb of data download
dataset = datasets.CIFAR100   # 169Mb of data download

data_path = './data'

transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
           ])

train_loader = torch.utils.data.DataLoader(
    dataset(data_path, train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset(data_path, train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)


class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                print("Skipping the Batchnorm for these experiments")
                #layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            #else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def cifar10(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    return model

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    return model


#model = cifar10(128)
#model = cifar10(32)
model = cifar100(32)
if use_cuda:
    model.cuda()


#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#checkpoints_dir = './data/cache/overfitting/cifar10'
checkpoints_dir = './data/cache/overfitting/cifar100'


#torch.save(the_model.state_dict(), PATH)

#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))

def save(epoch):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'saved_%03d.model' % (epoch+1, )))


def train(epoch):
    model.train()
    t0 = time.time()
    tot_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        if True:
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            tot_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            bi1 = batch_idx+1
            print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.4f}\tt_epoch: {:.2f}secs'.format(
                epoch, bi1 * len(data), len(train_loader.dataset),
                100. * bi1 / len(train_loader), loss.data[0], 
                (time.time()-t0)*len(train_loader)/bi1,))
            
    tot_loss = tot_loss # loss function already averages over batch size
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        tot_loss / len(train_loader), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return tot_loss / len(train_loader), correct / len(train_loader.dataset)


def test(epoch):
    model.eval()
    tot_loss, correct = 0, 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #tot_loss += F.nll_loss(output, target).data[0]
        tot_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    tot_loss = tot_loss  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tot_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return tot_loss / len(test_loader), correct / len(test_loader.dataset)    


epoch, losses_by_epoch = 0, []


for _ in range(100):
    epoch+=1
    train_loss, train_correct = train(epoch)
    save(epoch)
    test_loss, test_correct = test(epoch)
    losses_by_epoch.append( [ train_loss, train_correct, test_loss, test_correct ] )
print("Finished %d epochs" % (epoch,))


losses_by_epoch_np = np.array( losses_by_epoch )
np.save(os.path.join(checkpoints_dir, 'losses_by_epoch%03d.npy' % epoch), losses_by_epoch_np)


# ### Plan :
# 
# *  Test saving of model parameters
# *  Run multiple epochs, looking for test curve to move upwards (overfit)
# *  
# 

losses_by_epoch








# ## Examine overfitted feature distributions
# 

# ### Download PyTorch with MNIST/CIFAR10/CIFAR100 Examples
# 
# Have a look at http://pytorch.org/
# 
# Also a (quantized?) model zoo : https://github.com/aaron-xichen/pytorch-playground
# 

# ###  Install PyTorch from binaries
# Since we're on 3.5, and have no ```cuda```: 
# 
# ```
# pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl  # 348Mb
# pip install torchvision
# ```
# 
# *But* this doesn't work : 
# *  https://discuss.pytorch.org/t/blas-performance-on-macos-vs-linux-vs-lua-torch/744
# *  https://discuss.pytorch.org/t/solved-archlinux-using-variable-backwards-appears-to-hang-program-indefinitely/1675
# 
# (see PyTorch from source below)
# 
# ---------
# However, on the graphics-card-installed machine, ```PyTorch``` had trouble finding ```cuda```. so fall back to direct binary install (works fine) : 
# 
# ```
# pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
# ```
# 

# ### PyTorch from source
# So, let's try to install from source...
# 
# ```
# sudo dnf install cmake
# 
# export NO_CUDA=1
# git clone https://github.com/pytorch/pytorch.git   # 10.29Mb
# cd pytorch
# . ~/env3/bin/activate  # Enter into the right virtualenv
# python setup.py install
# ```
# 
# Actually, that seemed to work on my AMD home machine (Fedora 25, running Python 3.5.3 in a virtualenv).
# 
# It leaves 39.8Mb of files in ```env3/lib64/python3.5/site-packages/torch```.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

use_cuda=torch.cuda.is_available()


random_seed = 42
batch_size = 200

#learning_rate, momentum = 0.01, 0.5  # SGD with momentum
learning_rate = 0.001   # SGD+Adam

log_interval = 20 # Num of batches between log messages


import numpy as np

import os
import time


torch.manual_seed(random_seed)
if use_cuda:
    torch.cuda.manual_seed(random_seed)


mnist_data_path = './data'

mnist_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(mnist_data_path, train=True, download=True, transform=mnist_transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(mnist_data_path, train=False, download=True, transform=mnist_transform),
    batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
if use_cuda:
    model.cuda()


#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


checkpoints_dir = './data/cache/overfitting/mnist'


#torch.save(the_model.state_dict(), PATH)

#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))

def save(epoch):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'saved_%03d.model' % (epoch+1, )))


def train(epoch):
    model.train()
    t0 = time.time()
    tot_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if True:
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            tot_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            bi1 = batch_idx+1
            print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.4f}\tt_epoch: {:.2f}secs'.format(
                epoch, bi1 * len(data), len(train_loader.dataset),
                100. * bi1 / len(train_loader), loss.data[0], 
                (time.time()-t0)*len(train_loader)/bi1,))
            
    tot_loss = tot_loss # loss function already averages over batch size
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        tot_loss / len(train_loader), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return tot_loss / len(train_loader), correct / len(train_loader.dataset)


def test(epoch):
    model.eval()
    tot_loss, correct = 0, 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        tot_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    tot_loss = tot_loss  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tot_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return tot_loss / len(test_loader), correct / len(test_loader.dataset)    


losses_by_epoch = []


for epoch in range(100):
    train_loss, train_correct = train(epoch+1)
    save(epoch+1)
    test_loss, test_correct = test(epoch+1)
    losses_by_epoch.append( [ train_loss, train_correct, test_loss, test_correct ] )
print("Finished %d epochs" % (epoch+1,))


losses_by_epoch_np = np.array( losses_by_epoch )
np.save(os.path.join(checkpoints_dir, 'losses_by_epoch.npy'), losses_by_epoch_np)


# ### Plan :
# 
# *  Test saving of model parameters
# *  Run multiple epochs, looking for test curve to move upwards (overfit)
# *  
# 

losses_by_epoch





# ## Aggregate Parallel Texts in directory
# 
# This assumes that we have a bunch of ```.csv``` files with the filename in the format ```${source}-${lang}.csv```, where each file has the header ```ts,txt``` to read in the text at each numeric timestamp.
# 

import os
import csv
import time, random
import re


lang_from, lang_to = 'en', 'ko'

data_path = './data'


# Go through all the files in the directory, and find the ```source``` prefixes that have both ```lang_from``` and ```lang_to``` CSVs available.
# 

stub_from, stub_to = set(),set()
stub_matcher = re.compile(r"(.*)\-(\w+)\.csv")
for fname in os.listdir(data_path):
    #print(fname)
    m = stub_matcher.match(fname)
    if m:
        stub, lang = m.group(1), m.group(2)
        if lang == lang_from: stub_from.add(stub)
        if lang == lang_to:   stub_to.add(stub)
stub_both = stub_from.intersection(stub_to)


# Now, go through ```stub_both``` and for each CSVs, read in both languages, and take all the ```txt``` entries at the same timestamps, and build the correspondence.
# 

correspondence_loc,txt_from,txt_to=[],[],[]

def read_dict_from_csv(fname):
    d=dict()
    with open(fname, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            d[float(row['ts'])]=row['txt']
    return d

for stub in stub_both:
    #print("Reading stub %s" % (stub,))
    data_from = read_dict_from_csv( os.path.join(data_path, stub+'-'+lang_from+'.csv') )
    data_to   = read_dict_from_csv( os.path.join(data_path, stub+'-'+lang_to+'.csv') )
    
    valid, skipped=0, 0
    for ts, txt in data_from.items():
        if ts in data_to:
            correspondence_loc.append( (stub, ts) )
            txt_from.append( txt )
            txt_to.append( data_to[ts] )
            valid += 1
        else:
            skipped += 1
    print("%3d valid of %3d fragments from '%s'" % (valid, valid+skipped, stub))
print("  Total data : %d text fragments" % (len(correspondence_loc),)) 


for _ in range(10):
    i = random.randrange(len(correspondence_loc))
    print( txt_from[i], txt_to[i]  )


# ### Tokenize the correspondences
# NB: Japanese requires word-splitting too
# 

sub_punctuation = re.compile(r'[\,\.\:\;\?\!\-\—\s\"0-9\(\)]+')
sub_apostrophes = re.compile(r'\'(\w+)')
sub_multispaces = re.compile(r'\s\s+')
    
if lang_from=='ja' or lang_to=='ja':
    import tinysegmenter
    ja_segmenter = tinysegmenter.TinySegmenter()
    sub_punc_ja  = re.compile(r'[\」\「\？\。\、\・\（\）\―]+')

def tokenize_txt(arr, lang):
    tok=[]
    for txt in arr:
        t = txt.lower()
        t = re.sub(sub_punctuation, u' ', t)
        if "'" in t:
            t = re.sub(sub_apostrophes, r" '\1", t)
        if lang=='ja':
            t = ' '.join( ja_segmenter.tokenize(t) )
            t = re.sub(sub_punc_ja, u' ', t)
        t = re.sub(sub_multispaces, ' ', t)
        tok.append(t.strip())
    return tok


tok_from = tokenize_txt(txt_from, lang_from)
tok_to   = tokenize_txt(txt_to, lang_to)


tok_from[220:250]


tok_to[220:250]


# ### Build frequency dictionaries
# 

def build_freq(tok_arr):
    f=dict()
    for tok in tok_arr:
        for w in tok.split():
            if w not in f: f[w]=0
            f[w]+=1
    return f


freq_from=build_freq(tok_from)
freq_to  =build_freq(tok_to)


len(freq_from),len(freq_to), 


def most_frequent(freq, n=50, start=0):
    return ', '.join( sorted(freq,key=lambda w:freq[w], reverse=True)[start:n+start] )

print(most_frequent(freq_from))
print(most_frequent(freq_to, n=100))


print(most_frequent(freq_from, n=20, start=9000))


print( len( [_ for w,f in freq_from.items() if f>=10]))
print( len( [_ for w,f in freq_to.items() if f>=10]))


def build_rank(freq):
    return { w:i for i,w in enumerate( sorted(freq, key=lambda w:freq[w], reverse=True) ) }


rank_from = build_rank(freq_from)
rank_to   = build_rank(freq_to)


print(rank_from['robot'])


def max_rank(tok, rank):  # Find the most infrequent word in this tokenized sentence
    r = -1
    for w in tok.split():
        if rank[w]>r: r=rank[w] 
    return r
tok_max_rank_from = [ max_rank(tok, rank_from) for tok in tok_from ]
tok_max_rank_to   = [ max_rank(tok, rank_to)   for tok in tok_to ]


start=0;print(tok_max_rank_from[start:start+15], '\n', tok_max_rank_to[start:start+15],)
i=0; tok_max_rank_from[i], tok_from[i], tok_to[i], tok_max_rank_to[i], 


# ### Build a fragment coincidence matrix
# This might allow us to do single word translations...
# 




# # Copy a Pretrained Network between Frameworks
# 
# Since a large CNN is very time-consuming to train (even on a GPU), and requires huge amounts of data, is there any way to use a pre-calculated one instead of retraining the whole thing from scratch?  
# 
# This notebook shows how this can be done, so that the same data can be used in a different framework.
# 
# The code here is slightly rough-and-ready, since to be interested in doing it assumes some level of familiarity...
# 

import tensorflow as tf
import numpy as np


# ### Add TensorFlow Slim Model Zoo to path
# 

import os, sys
better_instructions = '2-CNN/4-ImageNet/4-ImageClassifier-inception_tf.ipynb'

if not os.path.isfile( '../models/tensorflow_zoo/models/README.md' ):
    print("Please follow the instructions in %s to get the Slim-Model-Zoo installed" % better_instructions)
else:
    sys.path.append('../models/tensorflow_zoo/models/slim')
    print("Model Zoo model code installed")


from datasets import dataset_utils

checkpoint_file = '../data/tensorflow_zoo/checkpoints/inception_v1.ckpt'
if not os.path.isfile( checkpoint_file ):
    print("Please follow the instructions in %s to get the Checkpoint installed" % better_instructions)
else:
    print("Checkpoint available locally")


if not os.path.isfile('../data/imagenet_synset_words.txt'):
    print("Please follow the instructions in %s to get the synset_words file" % better_instructions)
else:    
    print("ImageNet synset labels available")


# ### Build the model in TensorFlow
# 

slim = tf.contrib.slim
from nets import inception
#from preprocessing import inception_preprocessing

#image_size = inception.inception_v1.default_image_size
#image_size


tf.reset_default_graph()

if False:
    # Define the pre-processing chain within the graph - from a raw image
    input_image = tf.placeholder(tf.uint8, shape=[None, None, None, 3], name='input_image')
    processed_image = inception_preprocessing.preprocess_image(input_image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

processed_images = tf.placeholder(tf.float32, shape=[None, None, None, 3])

# Create the model - which uses the above pre-processing on image
#   it also uses the default arg scope to configure the batch norm parameters.
print("Model builder starting")

# Here is the actual model zoo model being instantiated :
with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits, end_points = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
#probabilities = tf.nn.softmax(logits)

# Create an operation that loads the pre-trained model from the checkpoint
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('InceptionV1') )

print("Model defined")


# ### Get the values from the TF model into a NumPy structure
# Mostly because it's easier for me to reason about NumPy...
# 

capture_names =[] 
capture_values=dict()

# Now let's run the pre-trained model
with tf.Session() as sess:
    # This is the loader 'op' we defined above
    init_fn(sess)  
    
    #variables = tf.trainable_variables()
    variables = tf.model_variables()  # includes moving average information
    for variable in variables:
        name, value = variable.name, variable.eval()
        capture_names.append(name)
        capture_values[name] = value
        print("%20s %8d %s " % (value.shape, np.prod(value.shape), name, ))
        
    """
    BatchNorm variables are (beta,moving_mean,moving_variance) separately (trainable==beta only)
               (64,)       64 InceptionV1/Conv2d_1a_7x7/BatchNorm/beta:0 
               (64,)       64 InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean:0 
               (64,)       64 InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance:0     
    """

# This fixes a typo in the original slim library...
if 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/weights:0' in capture_values:
    for w in ['weights:0', 'BatchNorm/beta:0', 'BatchNorm/moving_mean:0', 'BatchNorm/moving_variance:0']:
        capture_values['InceptionV1/Mixed_5b/Branch_2/Conv2d_0b_3x3/'+w] = (
            capture_values['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/'+w]
        )      


# #### And show the ```end_points```
# 

for e in sorted(end_points.keys()):
    print(e)


model_old = dict(names=capture_names, values=capture_values, 
                 check_names=end_points.keys(), check_values=dict())


# ###  Now grab outputs for a sample image at the ```end_points```
# 

import matplotlib.pyplot as plt
img_raw = plt.imread('../images/cat-with-tongue_224x224.jpg')
#img_raw.shape

# This is how the model_old does it in the pre-processing stages (so must be the same for model_new)
img = ( img_raw.astype('float32')/255.0 - 0.5 ) * 2.0
imgs = img[np.newaxis, :, :, :]


with tf.Session() as sess:
    # This is the loader 'op' we defined above
    init_fn(sess)  
    
    # This run grabs all the layer constants for the original photo image input
    check_names = model_old['check_names']
    end_points_values = sess.run([ end_points[k] for k in check_names ], feed_dict={processed_images: imgs})
    
    #model_old['check_values']={ k:end_points_values[i] for i,k in enumerate(check_names) }
    model_old['check_values']=dict( zip(check_names, end_points_values) )


# ----------
# ## Define the Model Structure in Framework 'B'
# 
# ( choosing Keras here )
# 

# This is taken from https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image


# This is taken from https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normalizer=True,
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution, `name + '_bn'` for the
            batch norm layer and `name + '_act'` for the
            activation layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
            filters, (num_row, num_col),
            strides=strides, padding=padding,
            use_bias=False, name=conv_name)(x)
    if normalizer:
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation:
        x = Activation(activation, name=act_name)(x)
    return x


# Convenience function for 'standard' Inception concatenated blocks
def concatenated_block(x, specs, channel_axis, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))
    
    branch_0 = conv2d_bn(x, br0[0], 1, 1, name=name+"_Branch_0_a_1x1")

    branch_1 = conv2d_bn(x, br1[0], 1, 1, name=name+"_Branch_1_a_1x1")
    branch_1 = conv2d_bn(branch_1, br1[1], 3, 3, name=name+"_Branch_1_b_3x3")

    branch_2 = conv2d_bn(x, br2[0], 1, 1, name=name+"_Branch_2_a_1x1")
    branch_2 = conv2d_bn(branch_2, br2[1], 3, 3, name=name+"_Branch_2_b_3x3")

    branch_3 = MaxPooling2D( (3, 3), strides=(1, 1), padding='same', name=name+"_Branch_3_a_max")(x)  
    branch_3 = conv2d_bn(branch_3, br3[0], 1, 1, name=name+"_Branch_3_b_1x1")

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name=name+"_Concatenated")
    return x


def InceptionV1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Inception v1 architecture.

    This architecture is defined in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 224x224.
    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        #default_size=299,
        default_size=224,
        min_size=139,
        data_format=K.image_data_format(),
        include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # 'Sequential bit at start'
    x = img_input
    x = conv2d_bn(x,  64, 7, 7, strides=(2, 2), padding='same',  name='Conv2d_1a_7x7')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_2a_3x3')(x)  
    
    x = conv2d_bn(x,  64, 1, 1, strides=(1, 1), padding='same', name='Conv2d_2b_1x1')  
    x = conv2d_bn(x, 192, 3, 3, strides=(1, 1), padding='same', name='Conv2d_2c_3x3')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_3a_3x3')(x)  
    
    # Now the '3' level inception units
    x = concatenated_block(x, (( 64,), ( 96,128), (16, 32), ( 32,)), channel_axis, 'Mixed_3b')
    x = concatenated_block(x, ((128,), (128,192), (32, 96), ( 64,)), channel_axis, 'Mixed_3c')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_4a_3x3')(x)  

    # Now the '4' level inception units
    x = concatenated_block(x, ((192,), ( 96,208), (16, 48), ( 64,)), channel_axis, 'Mixed_4b')
    x = concatenated_block(x, ((160,), (112,224), (24, 64), ( 64,)), channel_axis, 'Mixed_4c')
    x = concatenated_block(x, ((128,), (128,256), (24, 64), ( 64,)), channel_axis, 'Mixed_4d')
    x = concatenated_block(x, ((112,), (144,288), (32, 64), ( 64,)), channel_axis, 'Mixed_4e')
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_4f')

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='MaxPool_5a_2x2')(x)  

    # Now the '5' level inception units
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_5b')
    x = concatenated_block(x, ((384,), (192,384), (48,128), (128,)), channel_axis, 'Mixed_5c')
    

    if include_top:
        # Classification block
        
        # 'AvgPool_0a_7x7'
        x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)  
        
        # 'Dropout_0b'
        x = Dropout(0.2)(x)  # slim has keep_prob (@0.8), keras uses drop_fraction
        
        #logits = conv2d_bn(x,  classes+1, 1, 1, strides=(1, 1), padding='valid', name='Logits',
        #                   normalizer=False, activation=None, )  
        
        # Write out the logits explictly, since it is pretty different
        x = Conv2D(classes+1, (1, 1), strides=(1,1), padding='valid', use_bias=True, name='Logits')(x)
        
        x = Flatten(name='Logits_flat')(x)
        #x = x[:, 1:]  # ??Shift up so that first class ('blank background') vanishes
        # Would be more efficient to strip off position[0] from the weights+bias terms directly in 'Logits'
        
        x = Activation('softmax', name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(    name='global_pooling')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Finally : Create model
    model = Model(inputs, x, name='inception_v1')
    
    # LOAD model weights (TODO)
    
    return model


# #### Let's try and instantiate the new model
# 

include_top=True

model_new = InceptionV1(weights='imagenet', include_top=include_top)


model_new.summary()
# 'Connected to' isn't showing up due to Keras bug https://github.com/fchollet/keras/issues/6286


# ### Map the weights from Old to New
# 

def show_old_model_expected_shapes(model, name):
    v = model['values'][name]
    print('OLD    :', v.shape, name)

def show_new_model_expected_shapes(model, name):
    layer = model.get_layer(name)
    weights = layer.get_weights()
    for i, w in enumerate(weights):
        print('NEW[%d] : %s %s' % (i, w.shape, name))

# This depends on the naming conventions...
def copy_CNN_weight_with_bn(model_old, name_old, model_new, name_new):
    # See : https://github.com/fchollet/keras/issues/1671 
    weights = model_old['values'][name_old+'/weights:0']
    layer = model_new.get_layer(name_new+"_conv")
    layer.set_weights([weights])
    
    weights0 = model_old['values'][name_old+'/BatchNorm/beta:0']
    weights1 = model_old['values'][name_old+'/BatchNorm/moving_mean:0']
    weights2 = model_old['values'][name_old+'/BatchNorm/moving_variance:0']
    layer = model_new.get_layer(name_new+"_bn")
    weights_all = layer.get_weights()
    weights_all[0]=weights0
    weights_all[1]=weights1
    weights_all[2]=weights2
    layer.set_weights(weights_all)
    #print( weights_all[0] )
    #layer.set_weights([weights, np.zeros_like(weights), np.ones_like(weights), ])


show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/weights:0')
show_new_model_expected_shapes(model_new, 'Conv2d_1a_7x7_conv')

show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/BatchNorm/beta:0')
show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean:0')
show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance:0')
show_new_model_expected_shapes(model_new, 'Conv2d_1a_7x7_bn')

#model_old['values']['InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean:0']
#model_old['values']['InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance:0']

#copy_CNN_weight_with_bn(model_old, 'InceptionV1/Conv2d_1a_7x7', model_new, 'Conv2d_1a_7x7')
#copy_CNN_weight_with_bn(model_old, 'InceptionV1/Conv2d_2b_1x1', model_new, 'Conv2d_2b_1x1')
#copy_CNN_weight_with_bn(model_old, 'InceptionV1/Conv2d_2c_3x3', model_new, 'Conv2d_2c_3x3')
for block in [
        'Conv2d_1a_7x7', 
        'Conv2d_2b_1x1',
        'Conv2d_2c_3x3',
    ]:
    print("Copying %s" % (block,))    
    copy_CNN_weight_with_bn(model_old, 'InceptionV1/'+block, model_new, block)

print("Finished All")


# This depends on the naming conventions...
def copy_inception_block_weights(model_old, block_old, model_new, block_new):
    # e.g. FROM : InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1
    #        TO : Mixed_3b_Branch_1_a_1x1
    # block_old = 'InceptionV1/Mixed_3b'
    # block_new = 'Mixed_3b'
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_0/Conv2d_0a_1x1', model_new, block_new+'_Branch_0_a_1x1')
    
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_1/Conv2d_0a_1x1', model_new, block_new+'_Branch_1_a_1x1')
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_1/Conv2d_0b_3x3', model_new, block_new+'_Branch_1_b_3x3')

    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_2/Conv2d_0a_1x1', model_new, block_new+'_Branch_2_a_1x1')
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_2/Conv2d_0b_3x3', model_new, block_new+'_Branch_2_b_3x3')

    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_3/Conv2d_0b_1x1', model_new, block_new+'_Branch_3_b_1x1')

#copy_inception_block_weights(model_old, 'InceptionV1/Mixed_3b', model_new, 'Mixed_3b')
#copy_inception_block_weights(model_old, 'InceptionV1/Mixed_3c', model_new, 'Mixed_3c')

for block in [
        'Mixed_3b', 'Mixed_3c', 
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 
        'Mixed_5b', 'Mixed_5c', 
    ]:
    print("Copying %s" % (block,))    
    copy_inception_block_weights(model_old, 'InceptionV1/'+block, model_new, block)

print("Finished All")


# This depends on the naming conventions...
def copy_CNN_weight_with_bias(model_old, name_old, model_new, name_new):
    weights0 = model_old['values'][name_old+'/weights:0']
    weights1 = model_old['values'][name_old+'/biases:0']
    layer = model_new.get_layer(name_new)
    weights_all = layer.get_weights()
    weights_all[0]=weights0
    weights_all[1]=weights1
    layer.set_weights(weights_all)

if include_top:
    show_old_model_expected_shapes(model_old, 'InceptionV1/Logits/Conv2d_0c_1x1/weights:0')
    show_old_model_expected_shapes(model_old, 'InceptionV1/Logits/Conv2d_0c_1x1/biases:0')
    show_new_model_expected_shapes(model_new, 'Logits')

    print("Copying Logits")
    copy_CNN_weight_with_bias(model_old, 'InceptionV1/Logits/Conv2d_0c_1x1', model_new, 'Logits')
print("Finished All")


# ### Test the intermediate values in layers on a Sample Image
# 

imgs.shape


def check_image_outputs(model_old, name_old, model_new, name_new, images, idx):
    images_old = model_old['check_values'][name_old]
    print("OLD :", images_old.shape, np.min(images_old), np.max(images_old) )
    
    # See : http://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    output_layer = model_new.get_layer(name_new)
    get_check_value = K.function([model_new.input, K.learning_phase()], 
                                 [output_layer.output,])
    check_value = get_check_value([images, 0])  # '0' is for 'learning_phase'
    images_new = check_value[0]
    print("NEW :", images_new.shape, np.min(images_new), np.max(images_new) )
    
    total_diff = np.sum( np.abs(images_new - images_old) )
    print("total_diff =", total_diff)
    
    if len(images_old.shape)<4: return
    
    def no_axes():
        plt.gca().xaxis.set_visible(False)    
        plt.gca().yaxis.set_visible(False)    
        
    plt.figure(figsize=(9,4))

    # https://matplotlib.org/examples/color/colormaps_reference.html  bwr(+/-) or Blues(0+)
    plt.subplot2grid( (1,2), (0,0) ); no_axes()
    plt.imshow(images_old[0, :,:,idx], cmap='Blues', vmin=0.) # , vmin=0. , vmax=1.
    plt.subplot2grid( (1,2), (0,1) ); no_axes()
    plt.imshow(images_new[0, :,:,idx], cmap='Blues', vmin=0.) # ,vmin=-1., vmax=1.

    #plt.tight_layout()    
    plt.show()

# These should show attractive, identical images on left and right sides
#check_image_outputs(model_old, 'Conv2d_1a_7x7', model_new, 'Conv2d_1a_7x7_act', imgs, 31)
#check_image_outputs(model_old, 'MaxPool_2a_3x3', model_new, 'MaxPool_2a_3x3', imgs, 11)
#check_image_outputs(model_old, 'Conv2d_2b_1x1', model_new, 'Conv2d_2b_1x1_act', imgs, 5)
#check_image_outputs(model_old, 'Conv2d_2c_3x3', model_new, 'Conv2d_2c_3x3_act', imgs, 35)
check_image_outputs(model_old, 'MaxPool_3a_3x3', model_new, 'MaxPool_3a_3x3', imgs, 5)
#check_image_outputs(model_old, 'Mixed_3b', model_new, 'Mixed_3b_Concatenated', imgs, 25)
#check_image_outputs(model_old, 'Mixed_3c', model_new, 'Mixed_3c_Concatenated', imgs, 25)
#check_image_outputs(model_old, 'MaxPool_4a_3x3', model_new, 'MaxPool_4a_3x3', imgs, 25)
#check_image_outputs(model_old, 'MaxPool_5a_2x2', model_new, 'MaxPool_5a_2x2', imgs, 25)

if include_top:
    # No images for these ones...
    #check_image_outputs(model_old, 'Logits', model_new, 'Logits_flat', imgs, -1)
    check_image_outputs(model_old, 'Predictions', model_new, 'Predictions', imgs, -1)


# ### Save the NEW model to a file
# 

if include_top:
    model_file = 'inception_v1_weights_tf_dim_ordering_tf_kernels.h5'
else:
    model_file = 'inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5'

# This assumes that the model_weights will be loaded back into the same structure
model_new.save_weights(model_file)





# ## Test the Numerical Libraries being Used
# 

import os

import numpy as np

import theano
import theano.tensor as T

import time

def show_config():
    print("OMP_NUM_THREADS                       = %s" % 
           os.environ.get('OMP_NUM_THREADS','#CAREFUL : OMP_NUM_THREADS Not-defined!'))

    print("theano.config.device                  = %s" % theano.config.device)
    print("theano.config.floatX                  = %s" % theano.config.floatX)
    print("theano.config.blas.ldflags            = '%s'" % theano.config.blas.ldflags)
    print("theano.config.openmp                  = %s" % theano.config.openmp)
    print("theano.config.openmp_elemwise_minsize = %d" % theano.config.openmp_elemwise_minsize)

    # IDEA for pretty-printing : http://stackoverflow.com/questions/32026727/format-output-of-code-cell-with-markdown

def show_timing(iters=8, order='C'):
    M, N, K = 2000, 2000, 2000
    
    a = theano.shared(np.ones((M, N), dtype=theano.config.floatX, order=order))
    b = theano.shared(np.ones((N, K), dtype=theano.config.floatX, order=order))
    c = theano.shared(np.ones((M, K), dtype=theano.config.floatX, order=order))
    
    f = theano.function([], updates=[(c, 0.4 * c + 0.8 * T.dot(a, b))])
    
    if any([x.op.__class__.__name__ == 'Gemm' for x in f.maker.fgraph.toposort()]):
        c_impl = [hasattr(thunk, 'cthunk')
                  for node, thunk in zip(f.fn.nodes, f.fn.thunks)
                  if node.op.__class__.__name__ == "Gemm"]
        assert len(c_impl) == 1
        
        if c_impl[0]:
            impl = 'CPU (with direct Theano binding to blas)'
        else:
            impl = 'CPU (no direct Theano binding to blas, using numpy/scipy)'
            
    elif any([x.op.__class__.__name__ == 'GpuGemm' for x in
              f.maker.fgraph.toposort()]):
        impl = 'GPU'
        
    else:
        impl = 'ERROR, unable to tell if Theano used the cpu or the gpu:\n'
        impl += str(f.maker.fgraph.toposort())
    
    print("\nRunning operations using              : %s" % impl)
    
    t0 = time.time()
    for i in range(iters):
        f()
    if False:
        theano.sandbox.cuda.synchronize()
        
    print("Time taken for each of %2d iterations  : %.0f msec" % (iters, 1000.*(time.time()-t0)/iters))


# Now show the existing configuration and time an operation
# 

show_config()
show_timing()


#os.environ['OMP_NUM_THREADS']="1"
#os.environ['OMP_NUM_THREADS']="4"
#theano.config.floatX = 'float64'
theano.config.floatX = 'float32'
theano.config.openmp = False
#theano.config.openmp = True
#theano.config.blas.ldflags = ''
#theano.config.blas.ldflags = '-L/lib64/atlas -lsatlas'
theano.config.blas.ldflags = '-L/lib64/atlas -ltatlas'

show_config()
show_timing()





# ## Upgrade this VM from online sources
# 
# Execute the cells below to upgrade from online sources.  This is typically done when you've downloaded a VirtualBox Applicance before, and just want to update the notebooks and other data.  
# 
# NB : *Don't do this if you're not running the VM!  Just ```git pull``` for the latest updates*
# 

# Define the version number of this VirtualBox Appliance - dates will sort 'alphabetically'
with open('../config/vbox_name', 'rt') as f:
    repo,dt,tm=f.read().strip().split('_')
dt_tm = "%s_%s" % (dt,tm)
repo,dt,tm


# Repo to download from :
repo_base='https://raw.githubusercontent.com/mdda/deep-learning-workshop/'
path_to_root='master'
root_to_updates='/notebooks/model/updates.py'

#  Download the changes 'script', so we can find the changes that were made after this VM was created
import requests

updates = requests.get(repo_base+path_to_root+root_to_updates)
if updates.status_code == 200:
    with open('model/updates_current.py', 'wb') as f:
        f.write(updates.content)
        print("file : updates_current.py downloaded successfully")
else:
    print("Download unsuccessful : Complain!")    


# #### With the ```updates_current.py``` downloaded ...
# 
# Execute the following cell, to pull the content into the workbook.  
# Then execute that content (i.e. the same cell twice) to perform the update itself.
# 

get_ipython().magic('load model/updates_current.py')


# ### That's all folks!
# 
# If you want to 're-upgrade', you'll need to edit the cell above to reload the newest ```model/updates_current.py```, by making it look like : 
# 
# ```
# %load model/updates_current.py
# ```
# 
# and executing it again ... twice.

# # Art Style Transfer
# 
# This notebook is an implementation of the algorithm described in "A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576) by Gatys, Ecker and Bethge. Additional details of their method are available at http://arxiv.org/abs/1505.07376 and http://bethgelab.org/deepneuralart/.
# 
# An image is generated which combines the content of a photograph with the "style" of a painting. This is accomplished by jointly minimizing the squared difference between feature activation maps of the photo and generated image, and the squared difference of feature correlation between painting and generated image. A total variation penalty is also applied to reduce high frequency noise. 
# 
# This notebook was originally sourced from [Lasagne Recipes](https://github.com/Lasagne/Recipes/tree/master/examples/styletransfer), but has been modified to use a GoogLeNet network (pre-trained and pre-loaded), and given some features to make it easier to experiment with.
# 

import theano
import theano.tensor as T

import lasagne
from lasagne.utils import floatX

import numpy as np
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os # for directory listings
import pickle
import time

AS_PATH='./images/art-style'


from model import googlenet

net = googlenet.build_model()
net_input_var = net['input'].input_var
net_output_layer = net['prob']


# Load the pretrained weights into the network :
# 

params = pickle.load(open('./data/googlenet/blvc_googlenet.pkl', 'rb'), encoding='iso-8859-1')
model_param_values = params['param values']
#classes = params['synset words']
lasagne.layers.set_all_param_values(net_output_layer, model_param_values)

IMAGE_W=224
print("Loaded Model parameters")


# ### Choose the Photo to be *Enhanced*
# 

photos = [ '%s/photos/%s' % (AS_PATH, f) for f in os.listdir('%s/photos/' % AS_PATH) if not f.startswith('.')]
photo_i=-1 # will be incremented in next cell (i.e. to start at [0])


# Executing the cell below will iterate through the images in the ```./images/art-style/photos``` directory, so you can choose the one you want
# 

photo_i += 1
photo = plt.imread(photos[photo_i % len(photos)])
photo_rawim, photo = googlenet.prep_image(photo)
plt.imshow(photo_rawim)


# ### Choose the photo with the required 'Style'
# 

styles = [ '%s/styles/%s' % (AS_PATH, f) for f in os.listdir('%s/styles/' % AS_PATH) if not f.startswith('.')]
style_i=-1 # will be incremented in next cell (i.e. to start at [0])


# Executing the cell below will iterate through the images in the ```./images/art-style/styles``` directory, so you can choose the one you want
# 

style_i += 1
art = plt.imread(styles[style_i % len(styles)])
art_rawim, art = googlenet.prep_image(art)
plt.imshow(art_rawim)


# This defines various measures of difference that we'll use to compare the current output image with the original sources.
# 

def plot_layout(combined):
    def no_axes():
        plt.gca().xaxis.set_visible(False)    
        plt.gca().yaxis.set_visible(False)    
        
    plt.figure(figsize=(9,6))

    plt.subplot2grid( (2,3), (0,0) )
    no_axes()
    plt.imshow(photo_rawim)

    plt.subplot2grid( (2,3), (1,0) )
    no_axes()
    plt.imshow(art_rawim)

    plt.subplot2grid( (2,3), (0,1), colspan=2, rowspan=2 )
    no_axes()
    plt.imshow(combined, interpolation='nearest')

    plt.tight_layout()


def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g

def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]
    
    loss = 1./2 * ((x - p)**2).sum()
    return loss

def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()


# Here are the GoogLeNet layers that we're going to pay attention to :
# 

layers = [
    # used for 'content' in photo - a mid-tier convolutional layer 
    'inception_4b/output', 
    
    # used for 'style' - conv layers throughout model (not same as content one)
    'conv1/7x7_s2', 'conv2/3x3', 'inception_3b/output', 'inception_4d/output',
]
#layers = [
#    # used for 'content' in photo - a mid-tier convolutional layer 
#    'pool4/3x3_s2', 
#    
#    # used for 'style' - conv layers throughout model (not same as content one)
#    'conv1/7x7_s2', 'conv2/3x3', 'pool3/3x3_s2', 'inception_5b/output',
#]
layers = {k: net[k] for k in layers}


# ### Precompute layer activations for photo and artwork 
# This takes ~ 20 seconds
# 

input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                  for k, output in zip(layers.keys(), outputs)}
art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                for k, output in zip(layers.keys(), outputs)}


# Get expressions for layer activations for generated image
generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

gen_features = lasagne.layers.get_output(layers.values(), generated_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}


# ### Define the overall loss / badness function
# 

losses = []

# content loss
cl = 10 /1000.
losses.append(cl * content_loss(photo_features, gen_features, 'inception_4b/output'))

# style loss
sl = 20 *1000.
losses.append(sl * style_loss(art_features, gen_features, 'conv1/7x7_s2'))
losses.append(sl * style_loss(art_features, gen_features, 'conv2/3x3'))
losses.append(sl * style_loss(art_features, gen_features, 'inception_3b/output'))
losses.append(sl * style_loss(art_features, gen_features, 'inception_4d/output'))
#losses.append(sl * style_loss(art_features, gen_features, 'inception_5b/output'))

# total variation penalty
vp = 0.01 /1000. /1000.
losses.append(vp * total_variation_loss(generated_image))

total_loss = sum(losses)


# ### The *Famous* Symbolic Gradient operation
# 

grad = T.grad(total_loss, generated_image)


# ### Get Ready for Optimisation by SciPy
# 

# Theano functions to evaluate loss and gradient - takes around 1 minute (!)
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)

# Helper functions to interface with scipy.optimize
def eval_loss(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return f_loss().astype('float64')

def eval_grad(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype('float64')


# Initialize with the original ```photo```, since going from noise (the code that's commented out) takes many more iterations.
# 

generated_image.set_value(photo)
#generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

x0 = generated_image.get_value().astype('float64')
iteration=0


# ### Optimize all those losses, and show the image
# 
# To refine the result, just keep hitting 'run' on this cell (each iteration is about 60 seconds) :
# 

t0 = time.time()

scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40) 

x0 = generated_image.get_value().astype('float64')
iteration += 1

if False:
    plt.figure(figsize=(8,8))
    plt.imshow(googlenet.deprocess(x0), interpolation='nearest')
    plt.axis('off')
    plt.text(270, 25, '# {} in {:.1f}sec'.format(iteration, (float(time.time() - t0))), fontsize=14)
else:
    plot_layout(googlenet.deprocess(x0))
    print('Iteration {}, ran in {:.1f}sec'.format(iteration, float(time.time() - t0)))





# ## Flickr30k Captions to Corpus
# 
# *   P. Young, A. Lai, M. Hodosh, and J. Hockenmaier. _From image description to visual denotations: New similarity metrics for semantic inference over event descriptions._ Transactions of the Association for Computational Linguistics (to appear).
# 

import os

import numpy as np

import datetime
t_start=datetime.datetime.now()

import pickle


data_path = './data/Flickr30k'

output_dir = './data/cache'

output_filepath = os.path.join(output_dir, 
                                'CAPTIONS_%s_%s.pkl' % ( 
                                 data_path.replace('./', '').replace('/', '_'),
                                 t_start.strftime("%Y-%m-%d_%H-%M"),
                                ), )
output_filepath


# #### Plan 
# 
# *  Have a look inside the captions ```flickr30k.tar.gz``` : includes ```results_20130124.token```
# *  Extract contents of ```flickr30k.tar.gz``` to ```dict( photo_id -> [captions] )```
# *  Filter out a subset of those ```photo_id``` to convert
# *  Save off image array and corpus to an easy-to-load filetype
# 

WORD_FREQ_MIN=5
IMG_WORD_FREQ_MIN=5


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords


img_to_captions=dict()

tarfilepath = os.path.join(data_path, 'flickr30k.tar.gz')
if os.path.isfile(tarfilepath):
    import tarfile
    with tarfile.open(tarfilepath, 'r:gz').extractfile('results_20130124.token') as tokenized:
        n_captions = 0
        for l in tokenized.readlines():
            #print(l)  # This is bytes
            img_num, caption = l.decode("utf-8").strip().split("\t")
            img, num = img_num.split("#")
            #print(img, caption); break
            if img not in img_to_captions:  img_to_captions[img]=[]
            img_to_captions[img].append(caption)
            n_captions += 1
            
print("Found %d images, with a total of %d captions" % (len(img_to_captions),n_captions, ))
# Found 31783 images, with a total of 158915 captions


good_img_to_captions, good_img_to_captions_title = img_to_captions, 'all'
len(good_img_to_captions)


# Filter for the images that we care about
if False:  
    # This is a super-small list, which means we won't get the chance to see 
    #   enough Text to figure out how to make sentences.  ABANDON THIS 'SIMPLIFICATION'
    import re
    good_caption = re.compile( r'\b(cat|kitten)s?\b', flags=re.IGNORECASE )
    good_img_to_captions = { img:captions
                                for img, captions in img_to_captions.items() 
                                for caption in captions 
                                if good_caption.search( caption )
                           }  # img=='3947306345.jpg'
    good_img_to_captions_title = 'feline'
    #good_img_to_captions
    len(good_img_to_captions)


img_arr = sorted(good_img_to_captions.keys())


# extract the vocab where each word is required to occur WORD_FREQ_MIN times overall
word_freq_all=dict()

#for img in img_to_captions.keys():  # everything
for img in img_arr:  # Our selection
    for caption in img_to_captions[img]:
        for w in caption.lower().split():
            if not w in word_freq_all: word_freq_all[w]=0
            word_freq_all[w] += 1
            
word_freq = { w:f for w,f in word_freq_all.items() if f>=WORD_FREQ_MIN }

freq_word = sorted([ (f,w) for w,f in word_freq.items() ], reverse=True)
vocab = set( word_freq.keys() )

len(vocab), freq_word[0:20]
# 7734,  [(271698, 'a'), (151039, '.'), (83466, 'in'), (62978, 'the'), (45669, 'on'), (44263, 'and'), ...


# extract the vocab where each word is required to occur in IMG_WORD_FREQ_MIN *images* overall
word_freq_imgs=dict()

#for img in img_to_captions.keys():  # everything
for img in img_arr:  # Our selection
    img_caption_words=set()
    for caption in img_to_captions[img]:
        for w in caption.lower().split():
            img_caption_words.add(w)
    for w in img_caption_words:
        if not w in word_freq_imgs: word_freq_imgs[w]=0
        word_freq_imgs[w] += 1
            
word_freq = { w:f for w,f in word_freq_imgs.items() if f>=IMG_WORD_FREQ_MIN }

freq_word = sorted([ (f,w) for w,f in word_freq.items() ], reverse=True)
vocab = set( word_freq.keys() )

len(vocab), freq_word[0:20]
# 7219,  [(31783, '.'), (31635, 'a'), (28076, 'in'), (24180, 'the'), (21235, 'is'), (21201, 'and'), ...


sorted([ (f,w) for w,f in word_freq.items() if not w.isalpha() and '-' not in w ], reverse=True)


stop_words = set ( stopwords.words('english') )
punc = set ("- . , : ; ' \" & $ % ( ) ! ? #".split())

[ (w, w in stop_words) for w in "while with of at in".split() ]


stop_words_seen = vocab.intersection( stop_words.union(punc) )

', '.join(stop_words_seen)
len(stop_words_seen), len(stop_words)


# ### Now for the word Embeddings
# 

glove_dir = './data/RNN/'
glove_100k_50d = 'glove.first-100k.6B.50d.txt'
glove_100k_50d_path = os.path.join(glove_dir, glove_100k_50d)

if not os.path.isfile( glove_100k_50d_path ):
    raise RuntimeError("You need to download GloVE Embeddings "+
                       ": Use the downloader in 5-Text-Corpus-and-Embeddings.ipynb")
else:
    print("GloVE available locally")


# Due to size constraints, only use the first 100k vectors (i.e. 100k most frequently used words)
import glove
embedding_full = glove.Glove.load_stanford( glove_100k_50d_path )
embedding_full.word_vectors.shape


# Find words in word_arr that don't appear in GloVe
#word_arr = stop_words_seen  # Great : these all have embeddings
#word_arr = [ w for w,f in word_freq.items() if f>WORD_FREQ_MIN]  # This seems we're not missing much...
word_arr = vocab

missing_arr=[]
for w in word_arr:
    if not w in embedding_full.dictionary:
        missing_arr.append(w)
len(missing_arr), ', '.join( sorted(missing_arr) )


# ### Filter images and vocab jointly
# 

# Let's filter out the captions for the words that appear in our GloVe embedding
#  And ignore the images that then have no captions
img_to_valid_captions, words_used = dict(), set()
captions_total, captions_valid_total = 0,0

for img, captions in good_img_to_captions.items():
    captions_total += len(captions)
    captions_valid=[]
    for caption in captions:
        c = caption.lower()
        caption_valid=True
        for w in c.split():
            if w not in embedding_full.dictionary:
                caption_valid=False
            if w not in vocab:
                caption_valid=False
        if caption_valid:
            captions_valid.append( c )
            words_used.update( c.split() )
            
    if len(captions_valid)>0:
        img_to_valid_captions[img]=captions_valid
        captions_valid_total += len(captions_valid)
    else:
        #print("Throwing out %s" % (img,), captions)
        pass
    
print("%d images remain of %d.  %d captions remain of %d. Words used : %d" % (
            len(img_to_valid_captions.keys()), len(good_img_to_captions.keys()), 
            captions_valid_total, captions_total, 
            len(words_used),)
     )
# 31640 images remain of 31783.  135115 captions remain of 158915. Words used : 7399 (5 min appearances overall)
# 31522 images remain of 31783.  133106 captions remain of 158915. Words used : 6941 (5 min images)


# So, we only got rid of ~150 images, but 23k captions... if we require 5 mentions minimum
# And only got rid of ~250 images, but 25k captions... if we require 5 minimum image appearances


# ### Assemble a ready-for-use embedding
# 
# Let's filter the embedding to make it sleeker, and add some entries up front for RNN convenience
# 

# Construct an ordered word list:
action_words = "{MASK} {UNK} {START} {STOP} {EXTRA}".split(' ')

# Then want the 'real words' to have :
#  all the stop_words_seen (so that these can be identified separately)
#  followed by the remainder of the words_used, in frequency order

def words_in_freq_order(word_arr, word_freq=word_freq):
    # Create list of freq, word pairs
    word_arr_freq = [ (word_freq[w], w) for w in word_arr]
    return [ w for f,w in sorted(word_arr_freq, reverse=True) ]

stop_words_sorted = words_in_freq_order( stop_words_seen )
rarer_words_sorted = words_in_freq_order( words_used - stop_words_seen )

#", ".join( stop_words_sorted )
#", ".join( words_in_freq_order( words_used )[0:100] ) 
#", ".join( rarer_words_sorted[0:100] ) 
len(words_used), len(action_words), len(stop_words_sorted), len(rarer_words_sorted)


EMBEDDING_DIM = embedding_full.word_vectors.shape[1]

action_embeddings = np.zeros( (len(action_words), EMBEDDING_DIM,), dtype='float32')
for idx,w  in enumerate(action_words):
    if idx>0:  # Ignore {MASK}
        action_embeddings[idx, idx] = 1.0  # Make each row a very simple (but distinct) vector for simplicity

stop_words_idx  = [ embedding_full.dictionary[w] for w in stop_words_sorted ]
rarer_words_idx = [ embedding_full.dictionary[w] for w in rarer_words_sorted ]

embedding = np.vstack([ 
        action_embeddings,
        embedding_full.word_vectors[ stop_words_idx ],
        embedding_full.word_vectors[ rarer_words_idx ],
    ])

embedding_word_arr = action_words + stop_words_sorted + rarer_words_sorted
#stop_words_idx


# Check that this arrangement makes sense :
# 

embedding_dictionary = { w:i for i,w in enumerate(embedding_word_arr) }

# Check that this all ties together...
#word_check='{START}'  # an action word - not found in GloVe
#word_check='this'     # a stop word
word_check='hammer'   # a 'rare' word

#embedding_dictionary[word_check]
(  embedding[ embedding_dictionary[word_check] ] [0:6], 
   embedding_full.word_vectors[ embedding_full.dictionary.get( word_check, 0) ] [0:6], )


# Finally, save the data into a useful structure
# 

np.random.seed(1)  # Consistent values for train/test (for this )
save_me = dict(
    img_to_captions = img_to_valid_captions,
    
    action_words = action_words, 
    stop_words = stop_words_sorted,
    
    embedding = embedding,
    embedding_word_arr = embedding_word_arr,
    
    img_arr = img_arr_save,
    train_test = np.random.random( (len(img_arr_save),) ),
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open( output_filepath, 'wb') as f:
    pickle.dump(save_me, f)
    
print("Corpus saved to '%s'" % (output_filepath,))





"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

#  FROM : https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier
#  CODE : https://www.tensorflow.org/code/tensorflow/examples/tutorials/layers/cnn_mnist.py

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pickle

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)  # Quite a lot...
#tf.logging.set_verbosity(tf.logging.WARN)   # This prevents Logging ...

do_training = False


import sys
print(sys.version)
print('Tensorflow:',tf.__version__)


# Expecting:
# ```
# Tensorflow: 1.0.0
# 3.5.2 (default, Sep 14 2016, 11:28:32) 
# [GCC 6.2.1 20160901 (Red Hat 6.2.1-1)]
# ```
# 

def cnn_model_fn(features, integer_labels, mode):
  """Model function for CNN."""
  #print("Run cnn_model_fn, mode=%s" % (mode,))

  if type(features) is dict:
    #print("New-style feature input")
    features_images=features['images']
  else:
    print("OLD-style feature input (DEPRECATED)")
    features_images=features

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features_images, [-1, 28, 28, 1], name='input_layer')

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training= (mode == learn.ModeKeys.TRAIN) )

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  #logits = tf.Print(logits, [input_layer.get_shape(), integer_labels.get_shape()], "Debug size information : ", first_n=1)
  #logits = tf.layers.dense(inputs=dense, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(integer_labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=onehot_labels)
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=[ cls_targets[0] ])

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001,
      #optimizer="SGD")
      optimizer="Adam")

  # Generate Predictions
  predictions = {
    "classes":       tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor"), 
    "logits":        logits,
    #"before_and_after":( input_layer, logits ),
    #"before_and_after":dict(input_layer=input_layer, logits=logits),
  }
    
  # For OLD-STYLE inputs (needs wierd 'evaluate' metric)
  if mode == model_fn_lib.ModeKeys.EVAL:  
    predictions['input_grad'] = tf.gradients(loss, [input_layer])[0]
    
  # For NEW-STYLE inputs (can smuggle in extra parameters)
  if type(features) is dict and 'fake_targets' in features: 
    loss_vs_target = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, 
        labels=features['fake_targets']
    )
    predictions['image_gradient_vs_fake_target'] = tf.gradients(loss_vs_target, [input_layer])[0]

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# Create the Estimator : https://www.tensorflow.org/extend/estimators
mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="mnist_model/cnn")  # This is relative to the ipynb

# Check : the checkpoints file in 'mnist_model/cnn' has filenames that are in same directory


if False:
    print( mnist_classifier.get_variable_names() )
    #mnist_classifier.get_variable_value('conv2d/bias')

    #mnist_classifier.save()

    #tf.get_variable('input_layer')
    print( tf.global_variables() )
    print( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) )
    print( [n.name for n in tf.get_default_graph().as_graph_def().node] )


# Load training and eval data
mnist = learn.datasets.load_dataset("mnist")

train_data   = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data    = mnist.test.images  # Returns np.array
eval_labels  = np.asarray(mnist.test.labels, dtype=np.int32)

#print(eval_labels[7])
print("Data Loaded")


#https://www.tensorflow.org/get_started/input_fn#passing_input_fn_data_to_your_model
def mnist_batch_input_fn(dataset, batch_size=100, seed=None, num_epochs=1):  
    # If seed is defined, this will shuffle data into batches
    
    if False:  # This is the idea (but numpy, rather than Tensors)
        feature_dict = dict( images = dataset.images )
        labels       = np.asarray( dataset.labels, dtype=np.int32)
        return feature_dict, labels # but batch_size==EVERYTHING_AT_ONCE, unless we batch it up...
        
    np_labels = np.asarray( dataset.labels, dtype=np.int32)
    
    # Instead, build a Tensor dict 
    all_images = tf.constant( dataset.images, shape=dataset.images.shape, verify_shape=True )
    all_labels = tf.constant( np_labels,      shape=np_labels.shape, verify_shape=True )

    print("mnist_batch_input_fn sizing : ", 
          dataset.images.shape, 
          np.asarray( dataset.labels, dtype=np.int32).shape, 
          np.asarray( [dataset.labels], dtype=np.int32).T.shape,
         )
    
    # And create a 'feeder' to batch up the data appropriately...
    image, label = tf.train.slice_input_producer( [all_images, all_labels], 
                                                  num_epochs=num_epochs,
                                                  shuffle=(seed is not None), seed=seed,
                                                )
    
    dataset_dict = dict( images=image, labels=label ) # This becomes pluralized into batches by .batch()
    
    batch_dict = tf.train.batch( dataset_dict, batch_size,
                                num_threads=1, capacity=batch_size*2, 
                                enqueue_many=False, shapes=None, dynamic_pad=False, 
                                allow_smaller_final_batch=False, 
                                shared_name=None, name=None)
    
    
    batch_labels = batch_dict.pop('labels')
    
    # Return : 
    # 1) a mapping of feature columns to Tensors with the corresponding feature data, and 
    # 2) a Tensor containing labels
    return batch_dict, batch_labels

batch_size=100


if do_training:
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook( tensors=tensors_to_log, every_n_secs=20 ) #every_n_iter=1000 )

    # Train the model
    epochs=5

    if False:
        mnist_classifier.fit(
          x=train_data,
          y=train_labels,
          batch_size=batch_size,
          steps=train_labels.shape[0]/batch_size * epochs,
          monitors=[logging_hook]
        )

    mnist_classifier.fit(
        input_fn=lambda: mnist_batch_input_fn(mnist.train, batch_size=batch_size, seed=42, num_epochs=epochs), 
        #steps=train_labels.shape[0] / batch_size * epochs,
        #monitors=[logging_hook],
    )


if False: # This should log 'hi[1]' to the console (not to the Jupyter window...)
    # http://stackoverflow.com/questions/37898478
    #   /is-there-a-way-to-get-tensorflow-tf-print-output-to-appear-in-jupyter-notebook-o
    a = tf.constant(1.0)
    a = tf.Print(a, [a], 'hi')
    sess = tf.Session()
    a.eval(session=sess)


# Configure the accuracy metric for evaluation
cnn_metrics = {
  "accuracy":
      learn.MetricSpec(
          metric_fn=tf.metrics.accuracy, prediction_key="classes"),
}

# Evaluate the model and print results
#cnn_eval_results = mnist_classifier.evaluate( x=eval_data, y=eval_labels, metrics=cnn_metrics)

cnn_eval_results = mnist_classifier.evaluate(
    input_fn=lambda: mnist_batch_input_fn(mnist.test, batch_size=batch_size), 
    metrics=cnn_metrics,
    #steps=eval_labels.shape[0]/batch_size,
)

print(cnn_eval_results)


# Ok, so the built Estimator gets ~99% accuracy on the test set in <20 secs on CPU.
# 

# ### Adversarial Images
# 
# Let's create some adversarial digits for MNIST that fool the original Estimator
# 

train_offset = 17

image_orig = train_data[train_offset]     # This is a flat numpy array with an image in it
label_orig = train_labels[train_offset]   # This the digit label for that image

#label_target = (label_orig+1) % 10
label_target = 3

label_orig, label_target


if False: # Works, but 'old-style'
    #class_predictions = mnist_classifier.predict( x=np.array([image_orig]), batch_size=1, as_iterable=False)
    class_predictions = mnist_classifier.predict( x=image_orig, as_iterable=False)
    class_predictions['probabilities'][0]

    #class_predictions = mnist_classifier.predict( x=image_orig, outputs=['probabilities'], as_iterable=False)
    #class_predictions

def mnist_direct_data_input_fn(features_np_dict, targets_np):
    features_dict = { k:tf.constant(v) for k,v in features_np_dict.items()}
    targets = None if targets_np is None else tf.constant(targets_np)

    return features_dict, targets

class_predictions_generator = mnist_classifier.predict( 
    input_fn=lambda: mnist_direct_data_input_fn(dict(images=np.array([image_orig])), None), 
    outputs=['probabilities'],
)

for class_predictions in class_predictions_generator:
    break # Get the first one...

class_predictions['probabilities']


# ### Intuition behind 'gradient' for explicit inception version ...
# 

## Set the graph for the Inception model as the default graph,
## so that all changes inside this with-block are done to that graph.
#with model.graph.as_default():
#    # Add a placeholder variable for the target class-number.
#    # This will be set to e.g. 300 for the 'bookcase' class.
#    pl_cls_target = tf.placeholder(dtype=tf.int32)
#
#    # Add a new loss-function. This is the cross-entropy.
#    # See Tutorial #01 for an explanation of cross-entropy.
#    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])
#
#    # Get the gradient for the loss-function with regard to
#    # the resized input image.
#    gradient = tf.gradients(loss, resized_image)


# This is the way to do it 'OLD style', where we smuggle out the information during an EVALUATE() call
if False:
    # FIGURING-IT-OUT STEP : WORKS
    def metric_accuracy(cls_targets, predictions):
      return tf.metrics.accuracy(cls_targets, predictions)

    # FIGURING-IT-OUT STEP : WORKS
    def metric_accuracy_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
      if labels.dtype != predictions.dtype:
        predictions = tf.cast(predictions, labels.dtype)
      is_correct = tf.to_float(tf.equal(predictions, labels))
      return tf.metrics.mean(is_correct, weights, metrics_collections, updates_collections, name or 'accuracy')

    # FIGURING-IT-OUT STEP : WORKS
    def metric_mean_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
      return tf.metrics.mean(labels, weights, metrics_collections, updates_collections, name or 'gradient_mean')

    # FINALLY! :: WORKS
    def metric_concat_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
      return tf.contrib.metrics.streaming_concat(labels, axis=0, max_size=None, 
                                         metrics_collections=metrics_collections, 
                                         updates_collections=updates_collections, 
                                         name = name or 'gradient_concat')

    model_gradient = {
    #  "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,  prediction_key="classes"), # WORKS
    #  "accuracy": learn.MetricSpec(metric_fn=metric_accuracy,      prediction_key="classes"), # WORKS
    #  "accuracy": learn.MetricSpec(metric_fn=metric_accuracy_here, prediction_key="classes"), # WORKS
    #  "accuracy": learn.MetricSpec(metric_fn=metric_mean_here,     prediction_key="classes"), # WORKS
      "gradient": learn.MetricSpec(metric_fn=metric_concat_here,   prediction_key="input_grad"), # WORKS!   
    }

    # Evaluate the model and print results  OLD-STYLE
    cnn_gradient = mnist_classifier.evaluate( 
        x=np.array([ image_orig ], dtype='float32'), y=np.array([ label_target ], dtype='int32'), 
        batch_size=1,
        #input_fn = (lambda: (np.array([ image_orig ], dtype='float32'), np.array([7], dtype='int32'))),
        metrics=model_gradient)

    #cnn_gradient = mnist_classifier.evaluate( x=image_orig, y=np.int32(7), metrics=model_gradient)

    cnn_gradient['gradient'].shape


# NEW-STYLE : We can get the data from a .PREDICT() directly (outputs=[xyz] is passed through)

def mnist_direct_data_input_fn(features_np_dict, targets_np):
    features_dict = { k:tf.constant(v) for k,v in features_np_dict.items()}
    targets = None if targets_np is None else tf.constant(targets_np)
    return features_dict, targets

tensor_prediction_generator = mnist_classifier.predict( 
    input_fn=lambda: mnist_direct_data_input_fn(
        dict(
            images=np.array([ image_orig ]),
            fake_targets=np.array([ label_target ], dtype=np.int),
        ), None), 
    outputs=['image_gradient_vs_fake_target'],
)

for tensor_predictions in tensor_prediction_generator:
    break # Get the first one...

grads = tensor_predictions['image_gradient_vs_fake_target']
grads.shape,grads.min(),grads.max()


# Plot the gradients
plt.figure(figsize=(12,3))
for i in range(1):
    plt.subplot(1, 10, i+1)
    plt.imshow(((grads+8.)/11.).reshape((28, 28)), cmap='gray', interpolation='nearest')
    plt.axis('off')


def find_adversarial_noise(image_np, cls_target, model, 
                           pixel_max=255, noise_limit=None, 
                           required_score=0.99, max_iterations=50):
    """
    Find the noise that must be added to the given image so
    that it is classified as the target-class by the given model.
    
    image_np: numpy image in correct 'picture-like' format 
    cls_target: Target class-number (integer between 0-n_classes).
    noise_limit: Limit for pixel-values in the noise (scaled for 0...255 image)
    required_score: Stop when target-class 'probabilty' reaches this.
    max_iterations: Max number of optimization iterations to perform.
    """

    # Initialize the noise to zero.
    noise = np.zeros_like( image_np )

    # Perform a number of optimization iterations to find
    # the noise that causes mis-classification of the input image.
    for i in range(max_iterations):
        print("Iteration:", i)

        # The noisy image is just the sum of the input image and noise.
        noisy_image = image_np + noise
        
        # Ensure the pixel-values of the noisy image are between
        # 0 and pixel_max like a real image. If we allowed pixel-values
        # outside this range then maybe the mis-classification would
        # be due to this 'illegal' input breaking the Inception model.
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=float(pixel_max))
        
        # Calculate the predicted class-scores as well as the gradient.
        #pred, grad = session.run([y_pred, gradient], feed_dict=feed_dict)
         
        tensor_prediction_generator = model.predict( 
            input_fn=lambda: mnist_direct_data_input_fn(
                dict(
                    images=np.array([ noisy_image ]),
                    fake_targets=np.array([ cls_target ], dtype=np.int),
                ), None), 
            outputs=['probabilities','logits','image_gradient_vs_fake_target'],
        )

        for tensor_predictions in tensor_prediction_generator:
            break # Get the first one...

        #tensor_predictions['image_gradient_vs_fake_target'].shape            
        
        pred   = tensor_predictions['probabilities']
        logits = tensor_predictions['logits']
        grad   = tensor_predictions['image_gradient_vs_fake_target']
        
        print( ','.join([ ("%.4f" % p) for p in pred ]))
        #print(pred.shape, grad.shape)
        
        # The scores (probabilities) for the source and target classes.
        # score_source = pred[cls_source]
        score_target = pred[cls_target]

        # The gradient now tells us how much we need to change the
        # noisy input image in order to move the predicted class
        # closer to the desired target-class.

        # Calculate the max of the absolute gradient values.
        # This is used to calculate the step-size.
        grad_absmax = np.abs(grad).max()
        
        # If the gradient is very small then use a lower limit,
        # because we will use it as a divisor.
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        # Calculate the step-size for updating the image-noise.
        # This ensures that at least one pixel colour is changed by 7 out of 255
        # Recall that pixel colours can have 255 different values.
        # This step-size was found to give fast convergence.
        step_size = 7/255.0*pixel_max / grad_absmax

        # Print the score etc. for the source-class.
        #msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        #print(msg.format(score_source, cls_source, name_source))

        # Print the score etc. for the target-class.
        print("Target class (%d) score: %7.4f" % (cls_target, score_target, ))

        # Print statistics for the gradient.
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.6f}"
        print(msg.format(grad.min(), grad.max(), step_size))
        
        # Newline.
        print()
        
        # If the score for the target-class is not high enough.
        if score_target < required_score:
            # Update the image-noise by subtracting the gradient
            # scaled by the step-size.
            noise -= step_size * grad

            # Ensure the noise is within the desired range.
            # This avoids distorting the image too much.
            if noise_limit is not None:
                noise = np.clip(a     =  noise, 
                                a_min = -noise_limit/255.0*pixel_max, 
                                a_max =  noise_limit/255.0*pixel_max)
            
        else:
            # Abort the optimization because the score is high enough.
            break

    return (
        noisy_image, noise, score_target, logits
        #name_source, name_target, \
        #score_source, score_source_org, score_target
    )


np.min(image_orig), np.max(image_orig)


print(label_orig, label_target)

image_orig_sq = np.reshape(image_orig, (28,28,1))
res = find_adversarial_noise(image_orig_sq, label_target, mnist_classifier, 
                         pixel_max=1.0,   # for 0.0 ... 1.0 images (MNIST)
                         #pixel_max=255.0, # for 0..255 images (ImageNet)
                         #noise_limit=7.0,  
                         required_score=0.99, max_iterations=50)
adversarial_image, adversarial_noise, adversarial_score, adversarial_logits = res

# Plot the image, alterted image and noise
plt.figure(figsize=(12,3))
for i,im in enumerate( [image_orig, adversarial_image, adversarial_noise] ):
    plt.subplot(1, 10, 1+i)
    plt.imshow(im.reshape((28, 28)), cmap='gray', interpolation='nearest')
    plt.axis('off')


# tf.getDefaultGraph().finalize()


# ### Next Steps
# 
# Let's :
# 
# *  go through the training set and store the logits for [the valid?] training examples;
# 
# *  build an AutoEncoder on the logits, which minimises reconstruction error;
# 
# *  histogram the reconstruction error to find a bound above which we can reject an input image;
# 
# *  attempt to create adversarial examples on an updated network that includes the autoencoder bound as a gating function on the rest of the outputs;
# 
# *  create an infoGAN network for MNIST that allows us to create digits that are 'between' two classes;
# 
# *  score the reconstruction error of the between images to look at the rejection regions (which hopefully isolate the islands of acceptance from one another)
# 

# #### Get logit representation for all training examples
# 

# Evaluate the model and gather the results.  NB: no seed, since we want to preserve the ordering

# Predictions take ~ 60secs

predictions = mnist_classifier.predict( 
    input_fn=lambda: mnist_batch_input_fn(mnist.train, batch_size=batch_size),
    outputs=['logits'],
    as_iterable=True)

train_data_logits = np.array([ p['logits'] for p in predictions ])


predictions = mnist_classifier.predict( 
    input_fn=lambda: mnist_batch_input_fn(mnist.test, batch_size=batch_size),
    outputs=['logits'],
    as_iterable=True)
eval_data_logits  = np.array([ p['logits'] for p in predictions ])

train_data_logits.shape, eval_data_logits.shape


# Optionally save the logits for quicker iteration...
logits_filename = './mnist_model/logits.pkl'

if not tf.gfile.Exists(logits_filename):
    logits_saver = ( train_data_logits, train_labels, eval_data_logits, eval_labels )
    pickle.dump(logits_saver, open(logits_filename,'wb'), protocol=pickle.HIGHEST_PROTOCOL)


# #### Explore the logit representations
# 

# Load the logits 
if True:
    res = pickle.load( open(logits_filename, 'rb'), encoding='iso-8859-1')
    train_data_logits, train_labels, eval_data_logits, eval_labels = res     


# Show an example #s, target_classes, and logits
print("            %s" % ( ', '.join(["%7s" % l for l in range(10)]),) )
for train_data_example in [99, 98, 84]: # all have a true label of '6'
    print("#%4d : '%d'  [ %s ]" % (
                    train_data_example,
                    train_labels[train_data_example], 
                     ', '.join(["%+7.3f" % l for l in train_data_logits[train_data_example,:]]),
         ))


# Ok, so how about the reconstruction error for the training logits that it gets wrong?

# Create an indicator function that is 1 iff the label doesn't match the best logit answer
train_labels_predicted = np.argmax( train_data_logits, axis=1 )
print("train_labels_predicted.shape     :", train_labels_predicted.shape)
print( 'predicted : ',train_labels_predicted[80:100], '\nactual    : ', train_labels[80:100] )

#train_error_indices = np.where( train_labels_predicted == train_labels, 0, 1)
train_error_indices = train_labels_predicted != train_labels
print( "Total # of bad training examples : ", np.sum( train_error_indices ) )  # [80:90]

# Gather the 'badly trained logits'
train_error_logits = train_data_logits[train_error_indices]
print("train_error_logits.shape         :", train_error_logits.shape)

train_valid_indices = train_labels_predicted == train_labels
train_valid_logits  = train_data_logits[train_valid_indices]


# Histogram various pre-processings of the input logits

#def n(x): return x
#def n(x): return ( (x - x.mean(axis=1, keepdims=True))/x.std(axis=1, keepdims=True)  )
#def n(x): return ((x - x.min(axis=1, keepdims=True))/(x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 0.0001))
#def n(x): return np.fabs(x)

def n(x):
  len_em = len_except_max = (x.shape[1]-1)
  x_max = x.max(axis=1, keepdims=True)
  x_argmax = x.argmax(axis=1)
  mean_em  = (x.sum(axis=1, keepdims=True) - x_max) / len_em
  sumsq_em = np.sum(np.square(x - mean_em), axis=1, keepdims=True)  -  np.square(x_max - mean_em)
  std_em  = np.sqrt( sumsq_em / len_em )
  y = (x - mean_em) / std_em
  y = np.clip(y, -4.0, +4.0)
  y[np.arange(x.shape[0]), x_argmax]=5.0
  return y

count, bins, patches = plt.hist(n(train_valid_logits).flatten(), 50, normed=1, facecolor='green', alpha=1.0)
count, bins, patches = plt.hist(n(train_error_logits).flatten(), 50, normed=1, facecolor='blue', alpha=0.5)

plt.xlabel('logit')
plt.ylabel('density')
#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([-4, 6, 0, 0.8])
plt.grid(True)

plt.show()


# ### Build an autoencoder for the preprocessed logits
# Let's build an autoencoder 'regression' model with a hidden layer 'fewer' units
# 

def autoencoder_model_fn(features, unused_labels, mode):
  logits_dim = 10
  #hidden_dim = logits_dim
  hidden_dim = int(logits_dim*.75)

  input_layer = features['logits']  

  # One-hot on the input logit that's > 4.5
  one_hot = tf.div( tf.add( tf.sign( tf.subtract(input_layer, 4.5) ), 1.0), 2.0)
  one_hot = tf.Print(one_hot, [one_hot], message="one_hot: ", first_n=1, summarize=30 )

  # This summary is the inputs with the 'top-1' set to zero
  input_remainder = tf.subtract( input_layer, tf.multiply(one_hot, 5.0) )

  input_summary = tf.layers.dense(inputs=input_layer, units=int(logits_dim*.5), activation=tf.nn.relu)
    
  combined = tf.concat( [input_summary, one_hot], 1)
    
  # Encoder Dense Layer
    
  #dense1 = tf.layers.dense(inputs=input_layer, units=hidden_dim, activation=tf.nn.relu)
  #dense1 = tf.layers.dense(inputs=input_layer, units=logits_dim, activation=tf.nn.relu)
  #dense = tf.layers.dense(inputs=input_layer, units=hidden_dim, activation=tf.nn.elu)  # ELU!

  #dense1 = tf.layers.dense(inputs=input_layer, units=hidden_dim, activation=tf.nn.tanh)
  #dense1 = tf.layers.dense(inputs=input_layer, units=logits_dim, activation=tf.nn.tanh)
  #dense1 = tf.layers.dense(inputs=combined, units=logits_dim, activation=tf.nn.tanh)

  #dense2 = tf.layers.dense(inputs=dense1, units=hidden_dim, activation=tf.nn.tanh)
  #dense2 = tf.layers.dense(inputs=dense1, units=logits_dim*2, activation=tf.nn.tanh)
  #dense2 = tf.layers.dense(inputs=dense1, units=logits_dim, activation=tf.nn.tanh)

  #dense2 = dense1
  dense2 = combined
    
  # Add dropout operation; 0.6 probability that element will be kept
  #dropout = tf.layers.dropout(
  #    inputs=dense2, rate=0.9, training=mode == learn.ModeKeys.TRAIN)

  # Decoder Dense Layer

  #output_layer = tf.layers.dense(inputs=dropout, units=logits_dim)
  output_layer = tf.layers.dense(inputs=dense2, units=logits_dim)  # Linear activation

  loss = None
  train_op = None

  ## Calculate Loss (for both TRAIN and EVAL modes)
  #if mode != learn.ModeKeys.INFER:
  #  loss = tf.losses.mean_squared_error( input_layer, output_layer )

  if False:
      loss = tf.losses.mean_squared_error( input_layer, output_layer )

  if True:
      weighted_diff = tf.multiply( tf.subtract(1.0, one_hot), tf.subtract(input_layer, output_layer) )
      #weighted_diff = tf.multiply( 1.0, tf.subtract(input_layer, output_layer) )
      loss = tf.reduce_mean( tf.multiply (weighted_diff, weighted_diff) )

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="Adam")

  # Generate Predictions
  predictions = {
      "mse": loss,
      "regenerated":output_layer, 
      "gradient": tf.gradients(loss, input_layer),
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


mnist_autoencoder = learn.Estimator(
      model_fn=autoencoder_model_fn, model_dir="mnist_model/autoencoder")


def mnist_logit_batch_input_fn(logits, batch_size=100, seed=None, num_epochs=1):  
    # If seed is defined, this will shuffle data into batches

    all_logits  = tf.constant( logits, shape=logits.shape, verify_shape=True )
    fake_labels = tf.constant( np.zeros((logits.shape[0],)) )
    
    print("mnist_logit_batch_input_fn sizing : ", all_logits.shape, )
    
    # And create a 'feeder' to batch up the data appropriately...
    logit, label = tf.train.slice_input_producer( [ all_logits, fake_labels ], 
                                           num_epochs=num_epochs,
                                           shuffle=(seed is not None), seed=seed,
                                         )
    
    dataset_dict = dict( logits=logit, labels=label ) # This becomes pluralized into batches by .batch()
    
    batch_dict = tf.train.batch( dataset_dict, batch_size,
                                num_threads=1, capacity=batch_size*2, 
                                enqueue_many=False, shapes=None, dynamic_pad=False, 
                                allow_smaller_final_batch=False, 
                                shared_name=None, name=None)

    batch_labels = batch_dict.pop('labels')
    #batch_labels = batch_dict.pop('logits')
    
    # Return : 
    # 1) a mapping of feature columns to Tensors with the corresponding feature data, and 
    # 2) fake_labels (all 0)
    return batch_dict, batch_labels

autoenc_batch_size, autoenc_epochs = 100, 20


# Fit the autoencoder to the logits

mnist_autoencoder.fit(
    input_fn=lambda: mnist_logit_batch_input_fn( n(train_valid_logits), #train_data_logits, 
                                                batch_size=autoenc_batch_size, 
                                                seed=42, 
                                                num_epochs=autoenc_epochs), 
)


# *  n/2 hidden INFO:tensorflow:Saving checkpoints for 25000 into mnist_model/autoencoder/model.ckpt.
# *  n/2 hidden INFO:tensorflow:Loss for final step: 1.2686.
# 
# *  2xReLU INFO:tensorflow:Saving checkpoints for 25000 into mnist_model/autoencoder/model.ckpt.
# *  2xReLU INFO:tensorflow:Loss for final step: 1.47784e-05.
# 
# *  ELU+ReLU INFO:tensorflow:Saving checkpoints for 5000 into mnist_model/autoencoder/model.ckpt.
# *  ELU+ReLU INFO:tensorflow:Loss for final step: 0.00331942.
# 

# Configure the accuracy metric for evaluation
def metric_mean_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
  return tf.metrics.mean(labels, weights, metrics_collections, updates_collections, name or 'gradient_mean')

autoenc_metrics = {
  "loss":learn.MetricSpec(metric_fn=metric_mean_here, prediction_key="mse"),
}

# Evaluate the model and print results
#autoencoder_eval_results = mnist_autoencoder.evaluate( x=eval_data_logits, y=eval_data_logits, metrics=auto_metrics)
autoencoder_train_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(train_valid_logits), # train_data_logits, 
                                                batch_size=train_valid_logits.shape[0], 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_train_results)

autoencoder_eval_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(eval_data_logits), 
                                                batch_size=eval_data_logits.shape[0], 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_eval_results)


# *   {'loss': 1.1115935e-06, 'global_step': 18250} => This autoencoder thing works
# 

if False:  # Double up train_error_logits to check whether mean() is working
    train_error_logits = np.vstack( [train_error_logits,train_error_logits] )
    train_error_logits.shape


# What is the mean reconstruction error for the incorrectly trained digits?

autoencoder_error_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(train_error_logits), 
                                                batch_size=train_error_logits.shape[0], 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_error_results)


adversarial_logits


autoencoder_adversarial_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(np.array([
                    #train_data_logits[84],
                    adversarial_logits,
                ])),   
                                                batch_size=1, 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_adversarial_results)


get_ipython().magic('pinfo tf.reduce_sum')








# ## Extract Parallel Texts from TED talks
# 
# Derived / inspired by : [Ajinkya Kulkarni's GitHub](https://github.com/ajinkyakulkarni14/How-I-Extracted-TED-talks-for-parallel-Corpus-/blob/master/Ipython_notebook.ipynb).
# 

import requests
from bs4 import BeautifulSoup
#import shutil
#import codecs
import os, glob
import csv
import time, random


def enlist_talk_names(url, dict_):
    time.sleep( random.random()*5.0+5.0 )
    r = requests.get(url)
    print("  Got %d bytes from %s" % (len(r.text), url))
    soup = BeautifulSoup(r.text, 'html.parser')
    talks= soup.find_all("a", class_='')
    for i in talks:
        if i.attrs['href'].find('/talks/')==0 and dict_.get(i.attrs['href'])!=1:
            dict_[i.attrs['href']]=1
    return dict_


all_talk_names={}

# Get all pages of talks (seems a bit abusive)
#for i in xrange(1,61):
#    url='https://www.ted.com/talks?page=%d'%(i)
#    all_talk_names=enlist_talk_names(url, all_talk_names)

# A specific seach term
#url='https://www.ted.com/talks?sort=newest&q=ai'

# Specific topics
url='https://www.ted.com/talks?sort=newest&topics[]=AI'
#url='https://www.ted.com/talks?sort=newest&topics[]=machine+learning'
#url='https://www.ted.com/talks?sort=newest&topics[]=mind'
#url='https://www.ted.com/talks?sort=newest&topics[]=mind&page=2'
all_talk_names=enlist_talk_names(url, all_talk_names)
len(all_talk_names)


data_path = './data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

def extract_talk_languages(url, talk_name, language_list=['en', 'ko', 'ja']):
    need_more_data=False
    for lang in language_list:
        talk_lang_file = os.path.join(data_path, talk_name+'-'+lang+'.csv')
        if not os.path.isfile( talk_lang_file ) :
            need_more_data=True
    if not need_more_data:
        print("  Data already retrieved for %s" % (url,))
        return

    time.sleep( random.random()*5.0+5.0 )
    r = requests.get(url)
    print("  Got %d bytes from %s" % (len(r.text), url))
    if len(r.text)<1000: return # FAIL!
    soup = BeautifulSoup(r.text, 'html.parser')
    for i in soup.findAll('link'):
        if i.get('href')!=None and i.attrs['href'].find('?language=')!=-1:
            #print i.attrs['href']
            lang=i.attrs['hreflang']
            url_lang=i.attrs['href']
            if not lang in language_list:
                continue
                
            talk_lang_file = os.path.join(data_path, talk_name+'-'+lang+'.csv')
            if os.path.isfile( talk_lang_file ) :
                continue
                
            time.sleep( random.random()*5.0+5.0 )
            r_lang = requests.get(url_lang)
            print("    Lang[%s] : Got %d bytes" % (lang, len(r_lang.text), ))
            if len(r.text)<1000: return # FAIL!
            lang_soup = BeautifulSoup(r_lang.text, 'html.parser')

            talk_data = []
            for i in lang_soup.findAll('span',class_='talk-transcript__fragment'):
                d = [ int( i.attrs['data-time'] ), i.text.replace('\n',' ') ]
                talk_data.append(d)
            
            with open(talk_lang_file, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ts', 'txt'])
                writer.writerows(talk_data)            

if False:
    # Now flatten out the talk_data into time_step order
    talk_data_csv = [ ['ts']+language_list, ]
    for ts in sorted(talk_data.keys(), key=int):
        row = [ts] + [ talk_data[ts].get(lang, '') for lang in language_list]
        talk_data_csv.append(row)
        
    with open(os.path.join(data_path, talk_name+'.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(talk_data_csv)


for name in all_talk_names:
    extract_talk_languages('https://www.ted.com'+name+'/transcript', name[7:])
    #break
print("Finished extract_talk_languages for all_talk_names")





# # Modern Network :: Pre-Trained for ImageNet 
# This example demonstrates using a network pretrained on ImageNet for classification.  This image recognition task involved recognising 1000 different classes.  
# 
# ### The Model 'inception v3'
# This model was created by Google, and detailed in ["Rethinking the Inception Architecture for Computer Vision"](http://arxiv.org/abs/1512.00567), and was state-of-the-art until Dec-2015.  
# 
# The model parameter file is licensed Apache 2.0, and has already been downloaded into the ./data/inception_v3 directory.  The parameter file is ~80Mb of data.  And that's considered *small* for this type of model.
# 

import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer, Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.normalization import batch_norm

import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def bn_conv(input_layer, **kwargs):
  l = Conv2DLayer(input_layer, **kwargs)
  l = batch_norm(l, epsilon=0.001)
  return l

def inceptionA(input_layer, nfilt):
  # Corresponds to a modified version of figure 5 in the paper
  l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

  l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
  l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

  l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
  l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
  l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

  l4 = Pool2DLayer(input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
  l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

  return ConcatLayer([l1, l2, l3, l4])

def inceptionB(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionC(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

    l4 = Pool2DLayer(input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionD(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionE(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

    l4 = Pool2DLayer(input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])


def build_network():
  net = {}

  net['input'] = InputLayer((None, 3, 299, 299))
  net['conv']   = bn_conv(net['input'],    num_filters=32, filter_size=3, stride=2)
  net['conv_1'] = bn_conv(net['conv'],   num_filters=32, filter_size=3)
  net['conv_2'] = bn_conv(net['conv_1'], num_filters=64, filter_size=3, pad=1)
  net['pool']   = Pool2DLayer(net['conv_2'],   pool_size=3, stride=2, mode='max')

  net['conv_3'] = bn_conv(net['pool'],   num_filters=80, filter_size=1)

  net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

  net['pool_1'] = Pool2DLayer(net['conv_4'], pool_size=3, stride=2, mode='max')
  
  net['mixed/join'] = inceptionA(
      net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
  net['mixed_1/join'] = inceptionA(
      net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

  net['mixed_2/join'] = inceptionA(
      net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

  net['mixed_3/join'] = inceptionB(
      net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

  net['mixed_4/join'] = inceptionC(
      net['mixed_3/join'],
      nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

  net['mixed_5/join'] = inceptionC(
      net['mixed_4/join'],
      nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

  net['mixed_6/join'] = inceptionC(
      net['mixed_5/join'],
      nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

  net['mixed_7/join'] = inceptionC(
      net['mixed_6/join'],
      nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

  net['mixed_8/join'] = inceptionD(
      net['mixed_7/join'],
      nfilt=((192, 320), (192, 192, 192, 192)))

  net['mixed_9/join'] = inceptionE(
      net['mixed_8/join'],
      nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
      pool_mode='average_exc_pad')

  net['mixed_10/join'] = inceptionE(
      net['mixed_9/join'],
      nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
      pool_mode='max')

  net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

  net['softmax'] = DenseLayer(net['pool3'], num_units=1008, nonlinearity=lasagne.nonlinearities.softmax)

  return net


# ### Load the model parameters and metadata¶
# 

net = build_network()
output_layer = net['softmax']
print("Defined Inception3 model")


import pickle
params = pickle.load(open('./data/inception3/inception_v3.pkl', 'rb'), encoding='iso-8859-1')
#print("Saved model params.keys = ", params.keys())
#print("  License : "+params['LICENSE'])   # Apache 2.0
classes = params['synset words']
lasagne.layers.set_all_param_values(output_layer, params['param values'])
print("Loaded Model")

from model import inception_v3  # This is for image preprocessing functions


# ## Trying it out
# ### On pre-downloaded images
# 

# NB: If this is running on a single CPU core (likely in a VM), expect each image to take ~ 15 seconds (!)
# 
# NB: So, since there are 4 images, that means expect a **full 1 minute delay** ...
# 

image_files = [
    './images/grumpy-cat_224x224.jpg',
    './images/sad-owl_224x224.jpg',
    './images/cat-with-tongue_224x224.jpg',
    './images/doge-wiki_224x224.jpg',
]

import time
t0 = time.time()
for i, f in enumerate(image_files):
    #print("Image File:%s" % (f,))
    im = inception_v3.imagefile_to_np(f)
    
    prob = np.array( lasagne.layers.get_output(output_layer, inception_v3.preprocess(im), deterministic=True).eval() )
    top5 = np.argsort(prob[0])[-1:-6:-1]    

    plt.figure()
    plt.imshow(im.astype('uint8'))
    plt.axis('off')
    for n, label in enumerate(top5):
        plt.text(350, 50 + n * 25, '{}. {}'.format(n+1, classes[label]), fontsize=14)
print("DONE : %6.2f seconds each" %(float(time.time() - t0)/len(image_files),))


# ### On some test images from the web
# We'll download the ILSVRC2012 validation URLs and pick a few at random
# 

import requests

index = requests.get('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').text
image_urls = index.split('<br>')

np.random.seed(23)
np.random.shuffle(image_urls)
image_urls = image_urls[:5]

image_urls


# ### Process test images and print top 5 predicted labels¶
# (uses image pre-processing functions from ./model/inception_v3.py)
# 

import io

for url in image_urls:
    try:
        ext = url.split('.')[-1]
        im = plt.imread(io.BytesIO(requests.get(url).content), ext)
        
        prob = np.array( lasagne.layers.get_output(output_layer, inception_v3.preprocess(im), deterministic=True).eval() )
        top5 = np.argsort(prob[0])[-1:-6:-1]

        plt.figure()
        plt.imshow(inception_v3.resize_image(im))
        plt.axis('off')
        for n, label in enumerate(top5):
            plt.text(350, 50 + n * 25, '{}. {}'.format(n+1, classes[label]), fontsize=14)
            
    except IOError:
        print('bad url: ' + url)





# # ImageNet with GoogLeNet
# 
# ### Input
# GoogLeNet (the neural network structure which this notebook uses) was created to analyse 224x224 pictures from the ImageNet competition.
# 
# ### Output
# This notebook classifies each input image into exatly one output classification (out of 1000 possibilities).
# 

import theano
import theano.tensor as T

import lasagne
from lasagne.utils import floatX

import numpy as np
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
import json
import pickle


# Functions for building the GoogLeNet model with Lasagne are defined in model.googlenet:
# 

from model import googlenet


# The actual structure of the model is somewhat complex, to see the code, uncomment the line below (don't execute the code that appears in the cell, though)
# 

# Uncomment and execute this cell to see the GoogLeNet source
# %load models/imagenet_theano/googlenet.py


# The 27Mb parameter set has already been downloaded...
# 

# !wget -N --directory-prefix=./data/googlenet https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl


# Build the model and select layers we need - the features are taken from the final network layer, before the softmax nonlinearity.
# 

cnn_layers = googlenet.build_model()
cnn_input_var = cnn_layers['input'].input_var
cnn_feature_layer = cnn_layers['loss3/classifier']
cnn_output_layer = cnn_layers['prob']

get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))

print("Defined GoogLeNet model")


# Load the pretrained weights into the network
# 

params = pickle.load(open('./data/googlenet/blvc_googlenet.pkl', 'rb'), encoding='iso-8859-1')
model_param_values = params['param values']
classes = params['synset words']
lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)


# The images need some preprocessing before they can be fed to the CNN
# 

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        #im = skimage.transform.resize(im, (224, w*224/h), preserve_range=True)
        im = scipy.misc.imresize(im, (224, w*224/h))
        
    else:
        #im = skimage.transform.resize(im, (h*224/w, 224), preserve_range=True)
        im = scipy.misc.imresize(im, (h*224/w, 224))

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])


# ### Quick Test on an Example Image
# 
# Let's verify that GoogLeNet and our preprocessing are functioning properly :
# 

im = plt.imread('./images/cat-with-tongue_224x224.jpg')
plt.imshow(im)


rawim, cnn_im = prep_image(im)


plt.imshow(rawim)


p = get_cnn_features(cnn_im)
print(classes[p.argmax()])


# ### Test on Multiple Images in a Directory
# 
# -  Feel free to upload more images into the given directory (or create a new one), and see what the results are...
# 

image_dir = './images/'

image_files = [ '%s/%s' % (image_dir, f) for f in os.listdir(image_dir) 
                 if (f.lower().endswith('png') or f.lower().endswith('jpg')) and f!='logo.png' ]

import time
t0 = time.time()
for i, f in enumerate(image_files):
    im = plt.imread(f)
    #print("Image File:%s" % (f,))
    rawim, cnn_im = prep_image(im)
        
    prob = get_cnn_features(cnn_im)
    top5 = np.argsort(prob[0])[-1:-6:-1]    

    plt.figure()
    plt.imshow(im.astype('uint8'))
    plt.axis('off')
    for n, label in enumerate(top5):
        plt.text(350, 50 + n * 25, '{}. {}'.format(n+1, classes[label]), fontsize=14)
        
print("DONE : %6.2f seconds each" %(float(time.time() - t0)/len(image_files),))





# Theano + Lasagne :: MNIST CNN
# ====================================
# 
# This is a quick illustration of a Convolutional Neural Network being trained on the MNIST data.
# 
# ( Credit for initially creating this workbook : Eben Olson :: https://github.com/ebenolson/pydata2015 )
# 

import numpy as np
import theano
import theano.tensor as T
import lasagne

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import gzip
import pickle


# Seed for reproduciblity
np.random.seed(42)


# ### Get the MNIST data
# Put it into useful subsets, and show some of it as a sanity check
# 

# Download the MNIST digits dataset (Already downloaded locally)
# !wget -N --directory-prefix=./data/MNIST/ http://deeplearning.net/data/mnist/mnist.pkl.gz


train, val, test = pickle.load(gzip.open('./data/MNIST/mnist.pkl.gz'), encoding='iso-8859-1')

X_train, y_train = train
X_val, y_val = val


def batch_gen(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')


# ### Create the Network
# This is a Convolutional Neural Network (CNN), where each 'filter' in a given layer is produced by scanning a small (here 3x3) matrix over the whole of the previous layer (a convolution operation).  These filters can produce effects like : averaging, edge detection, etc.  
# 

# We need to reshape from a 1D feature vector to a 1 channel 2D image.
# Then we apply 3 convolutional filters with 3x3 kernel size.
l_in = lasagne.layers.InputLayer((None, 784))

l_shape = lasagne.layers.ReshapeLayer(l_in, (-1, 1, 28, 28))

l_conv = lasagne.layers.Conv2DLayer(l_shape, num_filters=3, filter_size=3, pad=1)

l_out = lasagne.layers.DenseLayer(l_conv,
                                  num_units=10,
                                  nonlinearity=lasagne.nonlinearities.softmax)


# ###  Compile and train the network.
# Accuracy is much better than the single layer network, despite the small number of filters.
# 

X_sym = T.matrix()
y_sym = T.ivector()

output = lasagne.layers.get_output(l_out, X_sym)
pred = output.argmax(-1)

loss = T.mean(lasagne.objectives.categorical_crossentropy(output, y_sym))

acc = T.mean(T.eq(pred, y_sym))

params = lasagne.layers.get_all_params(l_out)
grad = T.grad(loss, params)
updates = lasagne.updates.adam(grad, params, learning_rate=0.005)

f_train = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
f_val = theano.function([X_sym, y_sym], [loss, acc])
f_predict = theano.function([X_sym], pred)
print("Built network")


BATCH_SIZE = 64
N_BATCHES = len(X_train) // BATCH_SIZE
N_VAL_BATCHES = len(X_val) // BATCH_SIZE


train_batches = batch_gen(X_train, y_train, BATCH_SIZE)
val_batches = batch_gen(X_val, y_val, BATCH_SIZE)

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for _ in range(N_BATCHES):
        X, y = next(train_batches)
        loss, acc = f_train(X, y)
        train_loss += loss
        train_acc += acc
    train_loss /= N_BATCHES
    train_acc /= N_BATCHES

    val_loss = 0
    val_acc = 0
    for _ in range(N_VAL_BATCHES):
        X, y = next(val_batches)
        loss, acc = f_val(X, y)
        val_loss += loss
        val_acc += acc
    val_loss /= N_VAL_BATCHES
    val_acc /= N_VAL_BATCHES
    
    print('Epoch {:2d}, Train loss {:.03f}     (validation loss     : {:.03f}) ratio {:.03f}'.format(
            epoch, train_loss, val_loss, val_loss/train_loss))
    print('          Train accuracy {:.03f} (validation accuracy : {:.03f})'.format(train_acc, val_acc))
print("DONE")


# ### Look at the Output after the Convolutional Layer 
# Since the convolutional layer only has 3 filters, we can map these to red, green and blue for easier visualisation.
# 

filtered = lasagne.layers.get_output(l_conv, X_sym)
f_filter = theano.function([X_sym], filtered)


# Filter the first few training examples
im = f_filter(X_train[:10])
print(im.shape)


# Rearrange dimension so we can plot the result as RGB images
im = np.rollaxis(np.rollaxis(im, 3, 1), 3, 1)


# We can see that each filter detected different features in the images, i.e. horizontal / diagonal / vertical segments
# 

plt.figure(figsize=(16,8))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(im[i], interpolation='nearest')
    plt.axis('off')





# # Art Style Transfer
# 
# This notebook is a re-implementation of the algorithm described in "A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576) by Gatys, Ecker and Bethge. Additional details of their method are available at http://arxiv.org/abs/1505.07376 and http://bethgelab.org/deepneuralart/.
# 
# An image is generated which combines the content of a photograph with the "style" of a painting. This is accomplished by jointly minimizing the squared difference between feature activation maps of the photo and generated image, and the squared difference of feature correlation between painting and generated image. A total variation penalty is also applied to reduce high frequency noise. 
# 
# This notebook was originally sourced from [Lasagne Recipes](https://github.com/Lasagne/Recipes/tree/master/examples/styletransfer), but has been modified to use a GoogLeNet network (pre-trained and pre-loaded), in TensorFlow and given some features to make it easier to experiment with.
# 
# Other implementations : 
#   *  https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb (with [video](https://www.youtube.com/watch?v=LoePx3QC5Js))
#   *  https://github.com/cysmith/neural-style-tf
#   *  https://github.com/anishathalye/neural-style
# 

import tensorflow as tf

import numpy as np
import scipy
import scipy.misc  # for imresize

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import time

from urllib.request import urlopen  # Python 3+ version (instead of urllib2)

import os # for directory listings
import pickle

AS_PATH='./images/art-style'


# ### Add TensorFlow Slim Model Zoo to path
# 

import os, sys

tf_zoo_models_dir = './models/tensorflow_zoo'

if not os.path.exists(tf_zoo_models_dir):
    print("Creating %s directory" % (tf_zoo_models_dir,))
    os.makedirs(tf_zoo_models_dir)
if not os.path.isfile( os.path.join(tf_zoo_models_dir, 'models', 'README.md') ):
    print("Cloning tensorflow model zoo under %s" % (tf_zoo_models_dir, ))
    get_ipython().system('cd {tf_zoo_models_dir}; git clone https://github.com/tensorflow/models.git')

sys.path.append(tf_zoo_models_dir + "/models/slim")

print("Model Zoo model code installed")


# ### The Inception v1 (GoogLeNet) Architecture|
# 
# ![GoogLeNet Architecture](../../images/presentation/googlenet-arch_1228x573.jpg)

# ### Download the Inception V1 checkpoint¶
# 
# Functions for building the GoogLeNet model with TensorFlow / slim and preprocessing the images are defined in ```model.inception_v1_tf``` - which was downloaded from the TensorFlow / slim [Model Zoo](https://github.com/tensorflow/models/tree/master/slim).
# 
# The actual code for the ```slim``` model will be <a href="model/tensorflow_zoo/models/slim/nets/inception_v1.py" target=_blank>here</a>.
# 

from datasets import dataset_utils

targz = "inception_v1_2016_08_28.tar.gz"
url = "http://download.tensorflow.org/models/"+targz
checkpoints_dir = './data/tensorflow_zoo/checkpoints'

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if not os.path.isfile( os.path.join(checkpoints_dir, 'inception_v1.ckpt') ):
    tarfilepath = os.path.join(checkpoints_dir, targz)
    if os.path.isfile(tarfilepath):
        import tarfile
        tarfile.open(tarfilepath, 'r:gz').extractall(checkpoints_dir)
    else:
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
        
    # Get rid of tarfile source (the checkpoint itself will remain)
    os.unlink(tarfilepath)
        
print("Checkpoint available locally")


slim = tf.contrib.slim

from nets import inception
from preprocessing import inception_preprocessing

image_size = inception.inception_v1.default_image_size

IMAGE_W=224
image_size


def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
        
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = scipy.misc.imresize(im, (224, int(w*224/h)))
    else:
        im = scipy.misc.imresize(im, (int(h*224/w), 224))

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Now rescale it to [-1,+1].float32 from [0..255].unit8
    im = ( im.astype('float32')/255.0 - 0.5 ) * 2.0
    return rawim, im


# ### Choose the Photo to be *Enhanced*
# 

photos = [ '%s/photos/%s' % (AS_PATH, f) for f in os.listdir('%s/photos/' % AS_PATH) if not f.startswith('.')]
photo_i=-1 # will be incremented in next cell (i.e. to start at [0])


# Executing the cell below will iterate through the images in the ```./images/art-style/photos``` directory, so you can choose the one you want
# 

photo_i += 1
photo = plt.imread(photos[photo_i % len(photos)])
photo_rawim, photo = prep_image(photo)
plt.imshow(photo_rawim)


# ### Choose the photo with the required 'Style'
# 

styles = [ '%s/styles/%s' % (AS_PATH, f) for f in os.listdir('%s/styles/' % AS_PATH) if not f.startswith('.')]
style_i=-1 # will be incremented in next cell (i.e. to start at [0])


# Executing the cell below will iterate through the images in the ```./images/art-style/styles``` directory, so you can choose the one you want
# 

style_i += 1
style = plt.imread(styles[style_i % len(styles)])
style_rawim, style = prep_image(style)
plt.imshow(style_rawim)


def plot_layout(artwork):
    def no_axes():
        plt.gca().xaxis.set_visible(False)    
        plt.gca().yaxis.set_visible(False)    
        
    plt.figure(figsize=(9,6))

    plt.subplot2grid( (2,3), (0,0) )
    no_axes()
    plt.imshow(photo_rawim)

    plt.subplot2grid( (2,3), (1,0) )
    no_axes()
    plt.imshow(style_rawim)

    plt.subplot2grid( (2,3), (0,1), colspan=2, rowspan=2 )
    no_axes()
    plt.imshow(artwork, interpolation='nearest')

    plt.tight_layout()


# ### Precompute layer activations for photo and artwork 
# This takes ~ 20 seconds
# 

tf.reset_default_graph()

# This creates an image 'placeholder' - image inputs should be (224,224,3).float32 each [-1.0,1.0]
input_image_float = tf.placeholder(tf.float32, shape=[None, None, 3], name='input_image_float')
#input_image_var = tf.Variable(tf.zeros([image_size,image_size,3], dtype=tf.uint8), name='input_image_var' )

# Define the pre-processing chain within the graph - based on the input 'image' above
#processed_image = inception_preprocessing.preprocess_image(input_image, image_size, image_size, is_training=False)

processed_image = input_image_float
processed_images = tf.expand_dims(processed_image, 0)

print("Model builder starting")

# Here is the actual model zoo model being instantiated :
with slim.arg_scope(inception.inception_v1_arg_scope()):
    _, end_points = inception.inception_v1(processed_images, num_classes=1001, is_training=False)

# Create an operation that loads the pre-trained model from the checkpoint
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
    slim.get_model_variables('InceptionV1')
)

print("Model defined")


#dir(slim.get_model_variables('InceptionV1')[10])
#[ v.name for v in slim.get_model_variables('InceptionV1') ]
sorted(end_points.keys())
#dir(end_points['Mixed_4b'])
#end_points['Mixed_4b'].name


# So that gives us a pallette of GoogLeNet layers from which we can choose to pay attention to :
# 

photo_layers = [
    # used for 'content' in photo - a mid-tier convolutional layer 
    'Mixed_4b',      #Theano : 'inception_4b/output', 
#    'pool4/3x3_s2', 
]

style_layers = [
    # used for 'style' - conv layers throughout model (not same as content one)
    'Conv2d_1a_7x7', #Theano : 'conv1/7x7_s2',        
    'Conv2d_2c_3x3', #Theano : 'conv2/3x3', 
    'Mixed_3b',      #Theano : 'inception_3b/output',  
    'Mixed_4d',      #Theano : 'inception_4d/output',

#    'conv1/7x7_s2', 'conv2/3x3', 'pool3/3x3_s2', 'inception_5b/output',
]
all_layers = photo_layers+style_layers


# Actually, we'll capture more data than necessary, so we can compare the how they look (below)
photo_layers_capture = all_layers  # more minimally = photo_layers
style_layers_capture = all_layers  # more minimally = style_layers


# Let's grab (constant) values for all the layers required for the original photo, and the style image :
# 

# Now let's run the pre-trained model on the photo and the style
style_features={}
photo_features={}

with tf.Session() as sess:
    # This is the loader 'op' we defined above
    init_fn(sess)  
    
    # This run grabs all the layer constants for the original photo image input
    photo_layers_np = sess.run([ end_points[k] for k in photo_layers_capture ], feed_dict={input_image_float: photo})
    
    for i,l in enumerate(photo_layers_np):
        photo_features[ photo_layers_capture[i] ] = l

    # This run grabs all the layer constants for the style image input
    style_layers_np = sess.run([ end_points[k] for k in style_layers_capture ], feed_dict={input_image_float: style})
    
    for i,l in enumerate(style_layers_np):
        style_features[ style_layers_capture[i] ] = l

    # Helpful display of 
    for i,name in enumerate(all_layers):
        desc = []
        if name in style_layers:
            desc.append('style')
            l=style_features[name]
        if name in photo_layers:
            desc.append('photo')
            l=photo_features[name]
        print("  Layer[%d].shape=%18s, %s.name = '%s'" % (i, str(l.shape), '+'.join(desc), name,))


# Here are what the layers each see (photo on the top, style on the bottom for each set) :
# 

for name in all_layers:
    print("Layer Name : '%s'" % (name,))
    plt.figure(figsize=(12,6))
    for i in range(4):
        if name in photo_features:
            plt.subplot(2, 4, i+1)
            plt.imshow(photo_features[ name ][0, :, :, i], interpolation='nearest') # , cmap='gray'
            plt.axis('off')
        
        if name in style_features:
            plt.subplot(2, 4, 4+i+1)
            plt.imshow(style_features[ name ][0, :, :, i], interpolation='nearest') #, cmap='gray'
            plt.axis('off')
    plt.show()


# ### Define the overall loss / badness function
# 

# Let's now create model losses, which involve the ```end_points``` evaluated from the generated image, coupled with the appropriate constant layer losses from above : 
# 

art_features = {}
for name in all_layers:  
    art_features[name] = end_points[name]


# This defines various measures of difference that we'll use to compare the current output image with the original sources.
# 

def gram_matrix(tensor):
    shape = tensor.get_shape()
    
    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(shape[3])

    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    
    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram

def content_loss(P, X, layer):
    p = tf.constant( P[layer] )
    x = X[layer]
    
    loss = 1./2. * tf.reduce_mean(tf.square(x - p))
    return loss

def style_loss(S, X, layer):
    s = tf.constant( S[layer] )
    x = X[layer]
    
    S_gram = gram_matrix(s)
    X_gram = gram_matrix(x)
    
    layer_shape = s.get_shape()
    N = layer_shape[1]
    M = layer_shape[2] * layer_shape[3]
    
    loss = tf.reduce_mean(tf.square(X_gram - S_gram)) / (4. * tf.cast( tf.square(N) * tf.square(M), tf.float32))
    return loss

def total_variation_loss_l1(x):
    loss = tf.add( 
            tf.reduce_sum(tf.abs(x[1:,:,:] - x[:-1,:,:])), 
            tf.reduce_sum(tf.abs(x[:,1:,:] - x[:,:-1,:]))
           )
    return loss

def total_variation_loss_lX(x):
    loss = tf.reduce_sum(
            tf.pow( 
                tf.square( x[1:,:-1,:] - x[:-1,:-1,:]) + tf.square( x[:-1,1:,:] - x[:-1,:-1,:]),
                1.25)
           )
    return loss


# And here are some more TF nodes, to compute the losses using the layer values 'saved off' earlier
losses = []

# content loss
cl = 10.
losses.append(cl *1.     * content_loss(photo_features, art_features, 'Mixed_4b'))

# style loss
sl = 2. *1000. *1000.
losses.append(sl *1.     * style_loss(style_features, art_features, 'Conv2d_1a_7x7'))
losses.append(sl *1.     * style_loss(style_features, art_features, 'Conv2d_2c_3x3'))
losses.append(sl *10.    * style_loss(style_features, art_features, 'Mixed_3b'))
losses.append(sl *10.    * style_loss(style_features, art_features, 'Mixed_4d'))

# total variation penalty
vp = 10. /1000. /1000.
losses.append(vp *1.     * total_variation_loss_lX(input_image_float))
#losses.append(vp *1.     * total_variation_loss_l1(input_image_float))


# ['193.694946', '5.038591', '1.713539', '8.238111', '0.034608', '9.986152']
# ['0.473700', '0.034096', '0.010799', '0.021023', '0.164272', '0.539243']
# ['2.659750', '0.238304', '0.073061', '0.190739', '0.806217', '3.915816']
# ['1.098473', '0.169444', '0.245660', '0.109285', '0.938582', '0.028973']
# ['0.603620', '1.707279', '0.498789', '0.181227', '0.060200', '0.002774']
# ['0.788231', '0.920096', '0.358549', '0.806517', '0.256121', '0.002777']

total_loss = tf.reduce_sum(losses)

# And define the overall symbolic gradient operation
total_grad = tf.gradients(total_loss, [input_image_float])[0]


# ### Get Ready for Optimisation by SciPy
# 
# This uses the BFGS routine : 
#   *  R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
# 

# Initialize with the original ```photo```, since going from noise (the code that's commented out) takes many more iterations : 
# 

art_image = photo
#art_image = np.random.uniform(-1.0, +1.0, (image_size, image_size, 3))

x0 = art_image.flatten().astype('float64')
iteration=0


# ### Optimize all those losses, and show the image
# 
# To refine the result, just keep hitting 'run' on this cell (each iteration is about 60 seconds) :
# 

t0 = time.time()

with tf.Session() as sess:
    init_fn(sess)
    
    # This helper function (to interface with scipy.optimize) must close over sess
    def eval_loss_and_grad(x):  # x0 is a 3*image_size*image_size float64 vector
        x_image = x.reshape(image_size,image_size,3).astype('float32')
        x_loss, x_grad = sess.run( [total_loss, total_grad], feed_dict={input_image_float: x_image} )
        print("\nEval Loss @ ", [ "%.6f" % l for l in x[100:106]], " = ", x_loss)
        #print("Eval Grad = ", [ "%.6f" % l for l in x_grad.flatten()[100:106]] )
        
        losses_ = sess.run( losses, feed_dict={input_image_float: x_image} )
        print("Eval loss components = ", [ "%.6f" % l for l in losses_])
        
        return x_loss.astype('float64'), x_grad.flatten().astype('float64')

    x0, x0_loss, state = scipy.optimize.fmin_l_bfgs_b( eval_loss_and_grad, x0, maxfun=50) 
    iteration += 1

print("Iteration %d, in %.1fsec, Current loss : %.4f" % (iteration, float(time.time() - t0), x0_loss))

art_raw = np.clip( ((x0*0.5 + 0.5) * 255.0), a_min=0.0, a_max=255.0 )
plot_layout( art_raw.reshape(image_size,image_size,3).astype('uint8') )


# ### Now try it on your own images and styles...
# 




