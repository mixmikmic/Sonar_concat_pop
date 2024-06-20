# This notebook demonstrates a simple convolutional neural network, a variation of the LeNet5 model.
# 
# It is written using Lasagne for simplicity, which should make it easier to play with the structure of the model, its regularization, update rule, etc.  A pure Theano implementation is available [here](http://deeplearning.net/tutorial/lenet.html#lenet) if you are curious.
# 
# The training loop is written in regular Python and implements early stopping on a validation set.
# 
# This example should take a total of around 9 minutes to train on a GRID K520 with cuDNN v5.1.
# 
# Possible changes you could try:
# - change the nonlinearity of the convolution to rectifier unit
# - add an extra mlp layer
# - add dropout
# - change the update rule to Adam
# - limit the number of epoch of training to allow iterating more rapidly on code change.
# 

import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, NonlinearityLayer, DenseLayer

from load_data import load_data

# For reproducibility
np.random.seed(23455)


# To enable the GPU, run the following code
import theano.gpuarray
theano.config.floatX = 'float32'
theano.gpuarray.use('cuda')


# This implementation simplifies the model in the following ways:
# 
#  - LeNetConvPool doesn't implement location-specific gain and bias parameters
#  - LeNetConvPool doesn't implement pooling by average, it implements pooling
#    by max.
#  - Digit classification is implemented with a logistic regression rather than
#    an RBF network
#  - LeNet5 was not fully-connected convolutions at second layer
# 
# References:
#  - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
#    Gradient-Based Learning Applied to Document
#    Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
#    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
# 

# Define symbolic inputs
x = T.matrix('x')
y = T.ivector('y')

nonlinearity = lasagne.nonlinearities.tanh

## Build the architecture of the network
# Input
input_var = x.reshape((-1, 1, 28, 28))
layer0 = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

# First conv / pool / nonlinearity block
conv1 = Conv2DLayer(layer0, num_filters=20, filter_size=(5, 5), nonlinearity=None)
pool1 = MaxPool2DLayer(conv1, pool_size=(2, 2))
act1 = NonlinearityLayer(pool1, nonlinearity=nonlinearity)

# Second conv / pool / nonlinearity block
conv2 = Conv2DLayer(act1, num_filters=50, filter_size=(5, 5), nonlinearity=None)
pool2 = MaxPool2DLayer(conv2, pool_size=(2, 2))
act2 = NonlinearityLayer(pool2, nonlinearity=nonlinearity)

# Fully-connected layer
dense1 = DenseLayer(act2, num_units=500, nonlinearity=nonlinearity)

# Fully-connected layer for the output
network = DenseLayer(dense1, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

## Training
# Prediction and cost
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, y)
loss = loss.mean()

# Gradients and updates
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=0.1)
train_fn = theano.function([x, y], loss, updates=updates)

## Monitoring and evaluation
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)
test_loss = test_loss.mean()

# Misclassification rate
test_err = T.mean(T.neq(T.argmax(test_prediction, axis=1), y),
                  dtype=theano.config.floatX)

valid_fn = theano.function([x, y], test_err)


def evaluate_model(train_fn, valid_fn, datasets, n_epochs, batch_size):
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size
    n_test_batches = test_set_x.shape[0] // batch_size

    ## early-stopping parameters
    # look as this many examples regardless
    patience = 10000
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # Go through this many minibatches before checking the network
    # on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = %i' % iter)
            cost_ij = train_fn(train_set_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                               train_set_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size])

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [valid_fn(valid_set_x[i * batch_size:(i + 1) * batch_size],
                                              valid_set_y[i * batch_size:(i + 1) * batch_size])
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *                         improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        valid_fn(test_set_x[i * batch_size:(i + 1) * batch_size],
                                 test_set_y[i * batch_size:(i + 1) * batch_size])
                        for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))  


datasets = load_data('mnist.pkl.gz')


evaluate_model(train_fn, valid_fn, datasets, n_epochs=200, batch_size=500)





# This notebook allows you to play with Theano and Lasagne.
# 
# It uses a pre-trained VGG16 network. It reuses the last layers before the final prediction as the features for a new very simple predictor. We use those features to build a KNN on a new dataset of 2 classes (dogs and donuts). This shows we can reuse the pre-trained network with new classes. This was used in order to be super fast to train and allow you to play with it during this lab.
# 
# What you can try:
# - Use a different layer's outputs as your features. How does this changes the prediction performance?
# - If you keep the noise from the dropout to compute the features, does this change the prediction?

import collections
import glob
import io
import sys
import six
from six.moves import cPickle, xrange

from lasagne.utils import floatX
import numpy as np
import lasagne, theano

#To enable the GPU, run the following code
import theano.gpuarray
theano.config.floatX='float32'
theano.gpuarray.use('cuda')

# vgg16 includes the model definition and function to read and preprocess images from VGG16
from vgg16 import build_model, prep_image

# Populating the interactive namespace from numpy and matplotlib
get_ipython().magic('pylab inline')


# Functions for the KNN
# 

def distance_mat(x, m, p=2.0):
    """ Compute the L-p distance between a feature vector `x`
    and a matrix of feature vectors `x`.
    """
    diff = (np.abs(x - m)**p).sum(axis=1)**(1.0/p)
    return diff

def knn_idx(x, features, p=2):
    """Return the row index of the most similar features compared to `x`."""
    dist = distance_mat(x, features, p=p)
    return np.argmin(dist)


# Create a new datasets with 2 new classes
# 

class1_dir = './dog/'
class1_name = 'dog'
class2_dir = './donut/'
class2_name = 'donut'
test_dir = './test/'

# List files under the "dog/" directory
class1_files = glob.glob(class1_dir + '*')
# Load the images
class1_images = [plt.imread(io.BytesIO(open(f, 'rb').read()), f.split('.')[-1]) for f in class1_files]
# Build the target classes
class1_targets = [class1_name] * len(class1_files)

# Do the same for the second class
class2_files = glob.glob(class2_dir + '*')
class2_images = [plt.imread(io.BytesIO(open(f, 'rb').read()), f.split('.')[-1]) for f in class2_files]
class2_targets = [class2_name] * len(class2_files)

# Create the dataset by combining both classes
train_files = class1_files + class2_files
train_images = class1_images + class2_images
train_targets = class1_targets + class2_targets

# Read the test files
test_files = glob.glob(test_dir + '*')
test_images = [plt.imread(io.BytesIO(open(f, 'rb').read()), f.split('.')[-1]) for f in test_files]


# Load the model and the pre-trained weights.
# 
# Here the model is stored in a dict `d`. The keys are the layer names and the values of the corresponding layers.
# 
# It also prints the different layer names in the model.
# 

# vgg16.pkl contains the trained weights and the mean values needed for the preprocessing.
with open('vgg16.pkl', 'rb') as f:
    if six.PY3:
        d = cPickle.load(f, encoding='latin1')
    else:
        d = cPickle.load(f)

MEAN_IMAGE = d['mean value']
# Get the Lasagne model
net = build_model()
# Set the pre-trained weights
lasagne.layers.set_all_param_values(net['prob'], d['param values'])

# The different layer outputs you can reuse for the prediction
print(net.keys())


# Compile the Theano function and compute the features
# 
# This is the part that you can change
# 

# Get the graph that computes the last feature layers (fc8) of the model
# deterministic=True makes the Dropout layers do nothing as we don't train it
output = lasagne.layers.get_output(net['fc8'], deterministic=True)
# Compile the Theano function to be able to execute it.
compute_last = theano.function([net['input'].input_var], output)

def compute_feats(images):
    """Compute the features of many images."""
    preps = []
    for img in images:
        # prep_image returns a 4d tensor with only 1 image
        # remove the first dimensions to batch them ourself
        preps.append(prep_image(img, MEAN_IMAGE)[1][0])
    # Batch compute the features.
    return compute_last(preps)


# Compute the features of the train and test datasets
train_feats = compute_feats(train_images)
test_feats = compute_feats(test_images)

# Show the name of the file corresponding to example 0
print(test_files[0])

# Call knn_idx to get the nearest neighbor of this example
idx0 = knn_idx(test_feats[0], train_feats)

# Show the name of this training file
print(train_files[idx0])

# Show the predicted class
print(train_targets[idx0])


# Some functions to plot the prediction and the closest images
# 

def most_frequent(label_list):
    return collections.Counter(label_list).most_common()[0][0]

def knn_idx(x, features, p=2, k=1):
    dist = distance_mat(x, features, p=p)
    return np.argsort(dist)[:k]


def plot_knn(test_image, test_feat, train_images, train_feats, train_classes, k=1):
    knn_i = knn_idx(test_feat, train_feats, k=k)
    knn_images = [train_images[i] for i in knn_i]
    knn_classes = [train_classes[i] for i in knn_i]
    pred_class = most_frequent(knn_classes)
    figure(figsize=(12, 4))
    subplot(1, k+2, 1)
    imshow(prep_image(test_image, MEAN_IMAGE)[0])
    axis('off')
    title('prediction : ' + pred_class)
    for i in xrange(k):
        knn_preproc = prep_image(knn_images[i], MEAN_IMAGE)[0]
        subplot(1, k+2, i+3)
        imshow(knn_preproc)
        axis('off')
        title(knn_classes[i])


for i in range(len(test_images)):
    plot_knn(test_images[i], test_feats[i], train_images, train_feats, train_targets, k=7)





