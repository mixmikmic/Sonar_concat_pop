# # MNIST handwritten digits visualization with scikit-learn
# 
# In this notebook, we'll use some popular visualization techniques to visualize MNIST digits.  This notebook is based on the scikit-learn embedding examples found [here](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html).
# 
# First, the needed imports.
# 

get_ipython().magic('matplotlib inline')

from time import time

import numpy as np
from sklearn import random_projection, decomposition, manifold

import matplotlib.pyplot as plt
import seaborn as sns


# Then we load the MNIST data. First time it downloads the data, which can take a while.
# 

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Let's inspect only 1024 first training samples in this notebook
X = X_train[:1024]
y = y_train[:1024]

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X:', X.shape)
print('y:', y.shape)


# In this notebook, we only use 1024 first samples of the training data.  This reduces the time needed to calculate the visualizations and makes the visualizations appear less crowded.
# 
# Let's start by inspecting our data.  For such a small dataset, we can draw all samples at once:
# 

n_img_per_row = 32 # 32*32=1024
img = np.zeros((28 * n_img_per_row, 28 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 28 * i
    for j in range(n_img_per_row):    
        iy = 28 * j
        img[ix:ix + 28, iy:iy + 28] = X[i * n_img_per_row + j,:,:]

plt.figure(figsize=(9, 9))
plt.imshow(img)
plt.title('1024 first MNIST digits')
ax=plt.axis('off')


# Let's define a helper function to plot the different visualizations:
# 

def plot_embedding(X, title=None, t0=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(9,6))
    plt.axis('off')
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        if t0 is not None:
            plt.title("%s (%.2fs)" % (title, (time()-t0)))
        else:
            plt.title(title)


# ## 1. Random projection
# 
# A simple first visualization is a [random projection](http://scikit-learn.org/stable/modules/random_projection.html#random-projection) of the data into two dimensions.
# 
# Notice the `reshape(-1,28*28)` function which flattens the 2-D images into 1-D vectors (from 28*28 pixel images to 784-dimensional vectors).
# 

t0 = time()
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X.reshape(-1,28*28))
plot_embedding(X_projected, "Random projection", t0)


# ## 2. PCA
# 
# [Principal component analysis](http://scikit-learn.org/stable/modules/decomposition.html#pca) (PCA) is a standard method to decompose a high-dimensional dataset in a set of successive orthogonal components that explain a maximum amount of the variance. Here we project the data into two first principal components. The components have the maximal possible variance under the orthogonality constraint.
# 

t0 = time()
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X.reshape(-1,28*28))
plot_embedding(X_pca, "PCA projection", t0)


# ## 3. MDS
# 
# [Multidimensional scaling](http://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling) (MDS) seeks a low-dimensional representation of the data in which the distances try to respect the distances in the original high-dimensional space.  
# 

t0 = time()
mds = manifold.MDS(n_components=2, max_iter=500)
X_mds = mds.fit_transform(X.reshape(-1,28*28))
plot_embedding(X_mds, "MDS embedding", t0)


# ## 4. t-SNE
# 
# [t-distributed Stochastic Neighbor Embedding](http://scikit-learn.org/stable/modules/manifold.html#t-sne) (t-SNE) is a relatively new and popular tool to visualize high-dimensional data.  t-SNE is particularly sensitive to local structure and can often reveal clusters in the data.
# 
# t-SNE has an important tuneable parameter called `perplexity`, that can have a large effect on the resulting visualization, depending on the data.  Typical values for perplexity are between 5 and 50.  
# 

t0 = time()
perplexity=30
tsne = manifold.TSNE(n_components=2, perplexity=perplexity)
X_tsne = tsne.fit_transform(X.reshape(-1,28*28))
plot_embedding(X_tsne, "t-SNE embedding with perplexity=%d" % perplexity, t0)


# ## 5. Further visualizations
# 
# Take a look at the original scikit-learn [embedding examples](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html) for more visualizations.  Try some of these (for example LLE and isomap) on the MNIST data.
# 




# # MNIST handwritten digits classification with MLPs
# 
# In this notebook, we'll train a multi-layer perceptron model to classify MNIST digits using **Keras** (version $\ge$ 2 required). 
# 
# First, the needed imports. Keras tells us which backend (Theano or Tensorflow) it will be using.
# 

get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))


# Next we'll load the MNIST data.  First time we may have to download the data, which can take a while.
# 

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding:
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)


# The training data (`X_train`) is a 3rd-order tensor of size (60000, 28, 28), i.e. it consists of 60000 images of size 28x28 pixels. `y_train` is a 60000-dimensional vector containing the correct classes ("0", "1", ..., "9") for each training digit, and `Y_train` is a [one-hot](https://en.wikipedia.org/wiki/One-hot) encoding of `y_train`.
# 
# Let's take a closer look. Here are the first 10 training digits:
# 

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:], cmap="gray")
    plt.title('Class: '+str(y_train[i]))
    print('Training sample',i,': class:',y_train[i], ', one-hot encoded:', Y_train[i])


# ## Linear model
# 
# ### Initialization
# 
# Let's begin with a simple linear model.  We first initialize the model with `Sequential()`.  Then we add a `Dense` layer that has 28*28=784 input nodes (one for each pixel in the input image) and 10 output nodes. The `Dense` layer connects each input to each output with some weight parameter. 
# 
# Finally, we select *categorical crossentropy* as the loss function, select [*stochastic gradient descent*](https://keras.io/optimizers/#sgd) as the optimizer, add *accuracy* to the list of metrics to be evaluated, and `compile()` the model. Note there are [several different options](https://keras.io/optimizers/) for the optimizer in Keras that we could use instead of *sgd*.
# 

linmodel = Sequential()
linmodel.add(Dense(units=10, input_dim=28*28, activation='softmax'))

linmodel.compile(loss='categorical_crossentropy', 
                 optimizer='sgd', 
                 metrics=['accuracy'])
print(linmodel.summary())


# The summary shows that there are 7850 parameters in our model, as the weight matrix is of size 785x10 (not 784, as there's an additional bias term).
# 
# We can also draw a fancier graph of our model.
# 

SVG(model_to_dot(linmodel, show_shapes=True).create(prog='dot', format='svg'))


# ### Learning
# 
# Now we are ready to train our first model.  An *epoch* means one pass through the whole training data. 
# 
# The `reshape()` function flattens our 28x28 images into vectors of length 784.  (This means we are not using any information about the spatial neighborhood relations of pixels.  This setup is known as the *permutation invariant MNIST*.)  
# 
# You can run code below multiple times and it will continue the training process from where it left off.  If you want to start from scratch, re-initialize the model using the code a few cells ago. 
# 

get_ipython().run_cell_magic('time', '', 'epochs = 10 # one epoch takes about 3 seconds\n\nlinhistory = linmodel.fit(X_train.reshape((-1,28*28)), \n                          Y_train, \n                          epochs=epochs, \n                          batch_size=32,\n                          verbose=2)')


# Let's now see how the training progressed. 
# 
# * *Loss* is a function of the difference of the network output and the target values.  We are minimizing the loss function during training so it should decrease over time.
# * *Accuracy* is the classification accuracy for the training data.  It gives some indication of the real accuracy of the model but cannot be fully trusted, as it may have overfitted and just memorizes the training data.
# 

plt.figure(figsize=(8,5))
plt.plot(linhistory.epoch,linhistory.history['loss'])
plt.title('loss')

plt.figure(figsize=(8,5))
plt.plot(linhistory.epoch,linhistory.history['acc'])
plt.title('accuracy');


# ### Inference
# 
# For a better measure of the quality of the model, let's see the model accuracy for the test data. 
# 

linscores = linmodel.evaluate(X_test.reshape((-1,28*28)), 
                              Y_test, 
                              verbose=2)
print("%s: %.2f%%" % (linmodel.metrics_names[1], linscores[1]*100))


# We can now take a closer look on the results.
# 
# Let's define a helper function to show the failure cases of our classifier. 
# 

def show_failures(predictions, trueclass=None, predictedclass=None, maxtoshow=10):
    rounded = np.argmax(predictions, axis=1)
    errors = rounded!=y_test
    print('Showing max', maxtoshow, 'first failures. '
          'The predicted class is shown first and the correct class in parenthesis.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(X_test.shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            if trueclass is not None and y_test[i] != trueclass:
                continue
            if predictedclass is not None and predictions[i] != predictedclass:
                continue
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            plt.imshow(X_test[i,:,:], cmap="gray")
            plt.title("%d (%d)" % (rounded[i], y_test[i]))
            ii = ii + 1


# Here are the first 10 test digits the linear model classified to a wrong class:
# 

linpredictions = linmodel.predict(X_test.reshape((-1,28*28)))

show_failures(linpredictions)


# ## Multi-layer perceptron (MLP) network
# 
# ### Activation functions
# 
# Let's start by plotting some common activation functions for neural networks. `'relu'` stands for rectified linear unit, $y=\max(0,x)$, a very simple non-linearity we will be using in our MLP network below.
# 

x = np.arange(-4,4,.01)
plt.figure()
plt.plot(x, np.maximum(x,0), label='relu')
plt.plot(x, 1/(1+np.exp(-x)), label='sigmoid')
plt.plot(x, np.tanh(x), label='tanh')
plt.axis([-4, 4, -1.1, 1.5])
plt.title('Activation functions')
l = plt.legend()


# ### Initialization
# 
# Let's now create a more complex MLP model that has multiple layers, non-linear activation functions, and dropout layers.  `Dropout()` randomly sets a fraction of inputs to zero during training, which is one approach to regularization and can sometimes help to prevent overfitting.
# 
# There are two options below, a simple and a bit more complex model.  Select either one.
# 
# The output of the last layer needs to be a softmaxed 10-dimensional vector to match the groundtruth (`Y_train`). 
# 
# Finally, we again `compile()` the model, this time using [*Adam*](https://keras.io/optimizers/#adam) as the optimizer.
# 

# Model initialization:
model = Sequential()

# A simple model:
model.add(Dense(units=20, input_dim=28*28))
model.add(Activation('relu'))

# A bit more complex model:
#model.add(Dense(units=50, input_dim=28*28))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

#model.add(Dense(units=50))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

# The last layer needs to be like this:
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
print(model.summary())


SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# ### Learning
# 

get_ipython().run_cell_magic('time', '', 'epochs = 10 # one epoch with simple model takes about 4 seconds\n\nhistory = model.fit(X_train.reshape((-1,28*28)), \n                    Y_train, \n                    epochs=epochs, \n                    batch_size=32,\n                    verbose=2)')


plt.figure(figsize=(8,5))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(8,5))
plt.plot(history.epoch,history.history['acc'])
plt.title('accuracy');


# ### Inference
# 
# Accuracy for test data.  The model should be somewhat better than the linear model. 
# 

get_ipython().run_cell_magic('time', '', 'scores = model.evaluate(X_test.reshape((-1,28*28)), Y_test, verbose=2)\nprint("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))')


# We can again take a closer look on the results, using the `show_failures()` function defined earlier.
# 
# Here are the first 10 test digits the MLP classified to a wrong class:
# 

predictions = model.predict(X_test.reshape((-1,28*28)))

show_failures(predictions)


# We can use `show_failures()` to inspect failures in more detail. For example, here are failures in which the true class was "6":
# 

show_failures(predictions, trueclass=6)


# ## Model tuning
# 
# Modify the MLP model.  Try to improve the classification accuracy, or experiment with the effects of different parameters.  If you are interested in the state-of-the-art performance on permutation invariant MNIST, see e.g. this [recent paper](https://arxiv.org/abs/1507.02672) by Aalto University / The Curious AI Company researchers.
# 
# You can also consult the Keras documentation at https://keras.io/.  For example, the Dense, Activation, and Dropout layers are described at https://keras.io/layers/core/.
# 




# # MNIST handwritten digits classification with CNNs
# 
# In this notebook, we'll train a convolutional neural network (CNN, ConvNet) to classify MNIST digits using Keras (with either Theano or Tensorflow as the compute backend).  Keras version $\ge$ 2 is required. 
# 
# This notebook builds on the MNIST-MLP notebook, so the recommended order is to go through the MNIST-MLP notebook before starting with this one. 
# 
# First, the needed imports. Note that there are a few new layers compared to the MNIST-MLP notebook: Flatten, MaxPooling2D, Conv2D.
# 

get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D 
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding:
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)


# We'll have to do a bit of tensor manipulations, depending on the used backend (Theano or Tensorflow).
# 

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
print('X_train:', X_train.shape)


# ### Initialization
# 
# Now we are ready to create a convolutional model.
# 
#  * The `Convolution2D` layers operate on 2D matrices so we input the digit images directly to the model.  
#  * The `MaxPooling2D` layer reduces the spatial dimensions, that is, makes the image smaller.
#  * The `Flatten` layer flattens the 2D matrices into vectors, so we can then switch to  `Dense` layers as in the MLP model. 
# 
# See https://keras.io/layers/convolutional/, https://keras.io/layers/pooling/ for more information.
# 

# number of convolutional filters to use
nb_filters = 32
# convolution kernel size
kernel_size = (3, 3)
# size of pooling area for max pooling
pool_size = (2, 2)

model = Sequential()

model.add(Conv2D(nb_filters, kernel_size,
                 padding='valid',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())


SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# ### Learning
# 
# Now let's train the CNN model. Note that we do not need the `reshape()` function as in the MLP case. 
# 
# This is a relatively complex model, so training is considerably slower than with MLPs. 
# 

get_ipython().run_cell_magic('time', '', '\nepochs = 3 # one epoch takes about 80 seconds\n\nhistory = model.fit(X_train, \n                    Y_train, \n                    epochs=epochs, \n                    batch_size=128,\n                    verbose=2)')


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'])
plt.title('accuracy');


# ### Inference
# 
# With enough training epochs, the test accuracy should exceed 99%.  
# 
# You can compare your result with the state-of-the art [here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).  Even more results can be found [here](http://yann.lecun.com/exdb/mnist/). 
# 

get_ipython().run_cell_magic('time', '', 'scores = model.evaluate(X_test, Y_test, verbose=2)\nprint("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))')


# We can again take a closer look on the results. Let's begin by defining
# a helper function to show the failure cases of our classifier. 
# 

def show_failures(predictions, trueclass=None, predictedclass=None, maxtoshow=10):
    rounded = np.argmax(predictions, axis=1)
    errors = rounded!=y_test
    print('Showing max', maxtoshow, 'first failures. '
          'The predicted class is shown first and the correct class in parenthesis.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(X_test.shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            if trueclass is not None and y_test[i] != trueclass:
                continue
            if predictedclass is not None and predictions[i] != predictedclass:
                continue
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            if K.image_dim_ordering() == 'th':
                plt.imshow(X_test[i,0,:,:], cmap="gray")
            else:
                plt.imshow(X_test[i,:,:,0], cmap="gray")
            plt.title("%d (%d)" % (rounded[i], y_test[i]))
            ii = ii + 1


# Here are the first 10 test digits the CNN classified to a wrong class:
# 

predictions = model.predict(X_test)

show_failures(predictions)


# We can use `show_failures()` to inspect failures in more detail. For example, here are failures in which the true class was "6":
# 

show_failures(predictions, trueclass=6)


# ## Bonus: train the model in taito-gpu
# 
# The above model can also be run in taito-gpu in a couple of easy steps:
# 
# ```sh
# ssh -l USERNAME taito-gpu.csc.fi
# 
# module purge 
# module load python-env/2.7.10 cuda/8.0
#     
# # the following two commands need to be entered only once
# PYTHONUSERBASE=$USERAPPL/tensorflow.0.11.0 pip install --user /wrk/jppirhon/tensorflow.0.11.0-gcc493_pkg/tensorflow-0.11.0-py2-none-any.whl
# pip install --user keras h5py Pillow
#     
# export PYTHONPATH=$USERAPPL/tensorflow.0.11.0/lib/python2.7/site-packages
# sbatch /wrk/makoskel/run-python27-gputest.sh /wrk/makoskel/keras-mnist-cnn-taitogpu.py
# ```
# 
# One epoch should take about 8 seconds.  With 10 epochs, the model should have > 99% accuracy.
# 




# # MNIST handwritten digits classification with RNNs
# 
# In this notebook, we'll train a recurrent neural network (RNN) to classify MNIST digits using Keras (with either Theano or Tensorflow as the compute backend).  Keras version $\ge$ 2 is required. 
# 
# This notebook builds on the MNIST-MLP notebook, so the recommended order is to go through the MNIST-MLP notebook before starting with this one. 
# 
# First, the needed imports. Note that there are a few new recurrent layers compared to the MNIST-MLP notebook: SimpleRNN, LSTM, GRU.
# 

get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding:
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)


# ### Images as sequences
# 
# Note that in this notebook we are using *a sequence model* for image classification.  Therefore, we consider here an image to be a sequence of (pixel) input vectors.
# 
# More exactly, we consider each MNIST digit image (of size 28x28 pixels) to be a sequence of length 28 (number of image rows) with a 28-dimensional input vector (each image row, having 28 columns) associated with each time step. 
# 
# ### Initialization
# 
# Now we are ready to create a recurrent model.  Keras contains three types of recurrent layers:
# 
#  * `SimpleRNN`, a fully-connected RNN where the output is fed back to input.
#  * `LSTM`, the Long-Short Term Memory unit layer.
#  * `GRU`, the Gated Recurrent Unit layer.
# 
# See https://keras.io/layers/recurrent/ for more information.
# 

# Number of hidden units to use:
nb_units = 50

model = Sequential()

# Recurrent layers supported: SimpleRNN, LSTM, GRU:
model.add(SimpleRNN(nb_units,
                    input_shape=(img_rows, img_cols)))

# To stack multiple RNN layers, all RNN layers except the last one need
# to have "return_sequences=True".  An example of using two RNN layers:
#model.add(SimpleRNN(16,
#                    input_shape=(img_rows, img_cols),
#                    return_sequences=True))
#model.add(SimpleRNN(32))

model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())


SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# ### Learning
# 
# Now let's train the RNN model. Note that we do not need the `reshape()` function as in the MLP case. 
# 
# This is a relatively complex model, so training (especially with LSTM and GRU layers) can be considerably slower than with MLPs. 
# 

get_ipython().run_cell_magic('time', '', '\nepochs = 3\n\nhistory = model.fit(X_train, \n                    Y_train, \n                    epochs=epochs, \n                    batch_size=128,\n                    verbose=2)')


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'])
plt.title('accuracy');


# ### Inference
# 
# With enough training epochs and a large enough model, the test accuracy should exceed 98%.  
# 
# You can compare your result with the state-of-the art [here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).  Even more results can be found [here](http://yann.lecun.com/exdb/mnist/). 
# 

get_ipython().run_cell_magic('time', '', 'scores = model.evaluate(X_test, Y_test, verbose=2)\nprint("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))')


# We can again take a closer look on the results. Let's begin by defining
# a helper function to show the failure cases of our classifier. 
# 

def show_failures(predictions, trueclass=None, predictedclass=None, maxtoshow=10):
    rounded = np.argmax(predictions, axis=1)
    errors = rounded!=y_test
    print('Showing max', maxtoshow, 'first failures. '
          'The predicted class is shown first and the correct class in parenthesis.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(X_test.shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            if trueclass is not None and y_test[i] != trueclass:
                continue
            if predictedclass is not None and predictions[i] != predictedclass:
                continue
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            if K.image_dim_ordering() == 'th':
                plt.imshow(X_test[i,0,:,:], cmap="gray")
            else:
                plt.imshow(X_test[i,:,:,0], cmap="gray")
            plt.title("%d (%d)" % (rounded[i], y_test[i]))
            ii = ii + 1


# Here are the first 10 test digits the RNN classified to a wrong class:
# 

predictions = model.predict(X_test)

show_failures(predictions)


# We can use `show_failures()` to inspect failures in more detail. For example, here are failures in which the true class was "6":
# 

show_failures(predictions, trueclass=6)





# # Notebook for testing the PyTorch setup
# 
# This netbook is for testing the [PyTorch](http://pytorch.org/) setup for the ML hands-on.  Below is a set of required imports.  
# 
# Run the cell, and no error messages should appear.
# 
# Some warnings may appear, this should be fine.
# 

get_ipython().magic('matplotlib inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using PyTorch version:', torch.__version__, 'CUDA:', torch.cuda.is_available())





# # MNIST handwritten digits classification with MLPs
# 
# In this notebook, we'll train a multi-layer perceptron model to classify MNIST digits using **PyTorch**. 
# 
# First, the needed imports. 
# 

get_ipython().magic('matplotlib inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)


# ## Data
# 
# Next we'll load the MNIST data.  First time we may have to download the data, which can take a while.
# 

batch_size = 32

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)


# The train and test data are provided via data loaders that provide iterators over the datasets. The first element of training data (`X_train`) is a 4th-order tensor of size (`batch_size`, 1, 28, 28), i.e. it consists of a batch of images of size 1x28x28 pixels. `y_train` is a vector containing the correct classes ("0", "1", ..., "9") for each training digit.
# 

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break


# Here are the first 10 training digits:
# 

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray")
    plt.title('Class: '+str(y_train[i]))


# ## MLP network definition
# 
# Let's define the network as a Python class.  We have to write the `__init__()` and `forward()` methods, and PyTorch will automatically generate a `backward()` method for computing the gradients for the backward pass.
# 
# Finally, we define an optimizer to update the model parameters based on the computed gradients.  We select *stochastic gradient descent (with momentum)* as the optimization algorithm, and set *learning rate* to 0.01.  Note that there are [several different options](http://pytorch.org/docs/optim.html#algorithms) for the optimizer in PyTorch that we could use instead of *SGD*.
# 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x))

model = Net()
if cuda:
    model.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print(model)


# ## Learning
# 
# Let's now define functions to `train()` and `test()` the model. 
# 

def train(epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(loss_vector, accuracy_vector):
    model.eval()
    test_loss, correct = 0, 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)
    loss_vector.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


# Now we are ready to train our model using the `train()` function.  An *epoch* means one pass through the whole training data. After each epoch, we evaluate the model using `test()`. 
# 

get_ipython().run_cell_magic('time', '', 'epochs = 10\n\nlossv, accv = [], []\nfor epoch in range(1, epochs + 1):\n    train(epoch)\n    test(lossv, accv)')


# Let's now visualize how the training progressed. 
# 
# * *Loss* is a function of the difference of the network output and the target values.  We are minimizing the loss function during training so it should decrease over time.
# * *Accuracy* is the classification accuracy for the test data.
# 

plt.figure(figsize=(8,5))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('test loss')

plt.figure(figsize=(8,5))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('test accuracy');


# ## Model tuning
# 
# Modify the MLP model.  Try to improve the classification accuracy, or experiment with the effects of different parameters.  If you are interested in the state-of-the-art performance on permutation invariant MNIST, see e.g. this [recent paper](https://arxiv.org/abs/1507.02672) by Aalto University / The Curious AI Company researchers.
# 
# You can also consult the PyTorch documentation at http://pytorch.org/.
# 




# # MNIST handwritten digits classification with nearest neighbors 
# 
# In this notebook, we'll use [nearest-neighbor classifiers](http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification) to classify MNIST digits using scikit-learn.
# 
# First, the needed imports. 
# 

get_ipython().magic('matplotlib inline')

from time import time
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


# Then we load the MNIST data. First time it downloads the data, which can take a while.
# 

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# The training data (`X_train`) is a 3rd-order tensor of size (60000, 28, 28), i.e. it consists of 60000 images of size 28x28 pixels. `y_train` is a 60000-dimensional vector containing the correct classes ("0", "1", ..., "9") for each training digit.
# 
# Let's take a closer look. Here are the first 10 training digits:
# 

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:], cmap="gray")
    plt.title('Class: '+str(y_train[i]))


# ## k-NN (k-nearest neighbors) classifier
# 
# ![title](imgs/500px-KnnClassification.svg.png)
# 
# <br/>
# 
# <center><small>Image by Antti Ajanki AnAj (Own work) [<a href="http://www.gnu.org/copyleft/fdl.html">GFDL</a>, <a href="http://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA-3.0</a> or <a href="http://creativecommons.org/licenses/by-sa/2.5-2.0-1.0">CC BY-SA 2.5-2.0-1.0</a>], <a href="https://commons.wikimedia.org/wiki/File%3AKnnClassification.svg">via Wikimedia Commons</a></small></center>
# 
# 
# ## 1-NN classifier
# 
# ### Initialization
# 
# Let's create first a 1-NN classifier.  Note that with nearest-neighbor classifiers there is no internal (parameterized) model and therefore no learning required.  Instead, calling the `fit()` function simply stores the samples of the training data in a suitable data structure.
# 
# Notice also the `reshape(-1,28*28)` function which flattens the 2-D images into 1-D vectors (from 28*28 pixel images to 784-dimensional vectors). 

n_neighbors = 1
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X_train.reshape(-1,28*28), y_train)


# ### Inference
# 
# And try to classify some test samples with it.
# 

t0 = time()
predictions = clf.predict(X_test[:100,:,:].reshape(-1,28*28))
print('Time elapsed: %.2fs' % (time()-t0))


# We observe that the classifier is rather slow, and classifying the whole test set would take quite some time. What is the reason for this?
# 
# The accuracy of the classifier:

print('Predicted', len(predictions), 'digits with accuracy:', accuracy_score(y_test[:100], predictions))


# ## Faster 1-NN classifier
# 
# ### Initialization
# 
# One way to make our 1-NN classifier faster is to use less training data:
# 

n_neighbors = 1
clf_reduced = neighbors.KNeighborsClassifier(n_neighbors)
clf_reduced.fit(X_train[:1024,:,:].reshape(-1,28*28), y_train[:1024])


# ### Inference
# 
# Now we can use the classifier created with reduced data to classify our whole test set in a reasonable amount of time.
# 

t0 = time()
predictions_reduced = clf_reduced.predict(X_test.reshape(-1,28*28))
print('Time elapsed: %.2fs' % (time()-t0))


# The classification accuracy is however now not as good:
# 

print('Predicted', len(predictions_reduced), 'digits with accuracy:', accuracy_score(y_test, predictions_reduced))


# We can also inspect the results in more detail. Let's define and use a helper function to show the wrongly classified test digits.
# 

def show_failures(predictions, trueclass=None, predictedclass=None, maxtoshow=10):
    errors = predictions!=y_test
    print('Showing max', maxtoshow, 'first failures. '
          'The predicted class is shown first and the correct class in parenthesis.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(X_test.shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            if trueclass is not None and y_test[i] != trueclass:
                continue
            if predictedclass is not None and predictions[i] != predictedclass:
                continue
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            plt.imshow(X_test[i,:,:], cmap="gray")
            plt.title("%d (%d)" % (predictions[i], y_test[i]))
            ii = ii + 1
            
show_failures(predictions_reduced)


# We can observe that the classifier makes rather "easy" mistakes, and there might be room for improvement.
# 

# ## Model tuning
# 
# Try to improve the accuracy of the nearest-neighbor classifier while preserving a reasonable runtime to classify the whole test set.  Things to try include using more than one neighbor (with or without weights) or increasing the amount of training data.  Other possible modifications are [nearest centroid classification](http://scikit-learn.org/stable/modules/neighbors.html#nearest-centroid-classifier) and [approximate nearest neighbors](http://scikit-learn.org/stable/modules/neighbors.html#approximate-nearest-neighbors).
# 
# See also http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification for more information.
# 




# # MNIST handwritten digits dimensionality reduction with scikit-learn
# 
# In this notebook, we'll use some popular methods to reduce the dimensionality of MNIST digits data before classification.  
# 
# First, the needed imports.
# 

get_ipython().magic('matplotlib inline')

import numpy as np
from sklearn import decomposition, feature_selection
from skimage.measure import block_reduce
from skimage.feature import canny

import matplotlib.pyplot as plt
import seaborn as sns


# Then we load the MNIST data. First time it may download the data, which can take a while.
# 

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)


# ## 1. Feature extraction
# 
# ### 1.1 PCA
# 
# [Principal component analysis](http://scikit-learn.org/stable/modules/decomposition.html#pca) (PCA) is a standard method to decompose a high-dimensional dataset in a set of successive orthogonal components that explain a maximum amount of the variance. Here we project the data into `n_components` principal components. The components have the maximal possible variance under the orthogonality constraint.
# 
# The option `whiten=True` can be used to whiten the outputs to have unit component-wise variances.  Its usefulness depends on the model to be used.
# 
# Notice the `reshape(-1,28*28)` function which flattens the 2-D images into 1-D vectors (from 28*28 pixel images to 784-dimensional vectors).
# 

get_ipython().run_cell_magic('time', '', "n_components = 50\npca = decomposition.PCA(n_components=n_components, whiten=True)\nX_pca = pca.fit_transform(X_train.reshape(-1,28*28))\nprint('X_pca:', X_pca.shape)")


# We can inspect the amount of variance explained by the principal components.
# 

plt.figure()
plt.plot(np.arange(n_components)+1, pca.explained_variance_)
plt.title('Explained variance by PCA components')


# ### 1.2 Image feature extraction
# 
# There are a lot of different feature extraction methods for image data.  Common ones include extraction of colors, textures, and shapes from images, or detection of edges, corners, lines, blobs, or templates.  Let's try a simple filtering-based method to reduce the dimensionality of the features, and a widely-used edge detector.
# 
# The [`measure.block_reduce()`](http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce) function from scikit-image applies a function (for_example `np.mean`, `np.max` or `np.median`) to blocks of the image, resulting in a downsampled image.
# 

filter_size = 2
X_train_downsampled = block_reduce(X_train, 
                                   block_size=(1, filter_size, filter_size), 
                                   func=np.mean)
print('X_train:', X_train.shape)
print('X_train_downsampled:', X_train_downsampled.shape)


# The [`feature.canny()`](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny) function applies the [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) to extract edges from the image.  Processing all images may take a couple of minutes.
# 

get_ipython().run_cell_magic('time', '', "sigma = 1.0\nX_train_canny = np.zeros(X_train.shape)\nfor i in range(X_train.shape[0]):\n    X_train_canny[i,:,:] = canny(X_train[i,:,:], sigma=sigma)\nprint('X_train_canny:', X_train_canny.shape)")


# Let's compare the original and filtered digit images:
# 

pltsize=1

plt.figure(figsize=(10*pltsize, pltsize))
plt.suptitle('Original')
plt.subplots_adjust(top=0.8)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:], cmap="gray", interpolation='none')

plt.figure(figsize=(10*pltsize, pltsize))
plt.suptitle('Downsampled with a %dx%d filter' % (filter_size, filter_size))
plt.subplots_adjust(top=0.8)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train_downsampled[i,:,:], cmap="gray", interpolation='none')
    
plt.figure(figsize=(10*pltsize, pltsize))
plt.suptitle('Canny edge detection with sigma=%.2f' % sigma)
plt.subplots_adjust(top=0.8)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train_canny[i,:,:], cmap="gray", interpolation='none')


# ## 2. Feature selection
# 
# ### 2.1 Low variance
# 
# The MNIST digits data has a lot of components with little variance.  These components are not particularly useful for discriminating between the classes, so they can probably be removed safely.  Let's first draw the component-wise variances of MNIST data.
# 

variances = np.var(X_train.reshape(-1,28*28), axis=0)
plt.figure()
plt.plot(variances)
plt.title('Component-wise variance of MNIST digits')


# The variances can also be plotted for each pixel.
# 

plt.figure()
with sns.axes_style("white"):
    plt.imshow(variances.reshape(28,28), interpolation='none')
plt.title('Pixel-wise variance of MNIST digits')
plt.grid(False)


# Select an appropriate `variance_threshold` based on the *"Component-wise variance of MNIST digits"* figure above.
# 

get_ipython().run_cell_magic('time', '', "variance_threshold = 1000\nlv = feature_selection.VarianceThreshold(threshold=variance_threshold)\nX_lv = lv.fit_transform(X_train.reshape(-1,28*28))\nprint('X_lv:', X_lv.shape)")


# ### 2.2 Univariate feature selection
# 
# Another method for feature selection is to select the *k* best features based on univariate statistical tests between the features and the class of each sample.  Therefore, this is a supervised method and we need to include `y_train` in `fit_transform()`.
# See [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection) for the set of available statistical tests and other further options.
# 

get_ipython().run_cell_magic('time', '', "k = 50\nukb = feature_selection.SelectKBest(k=k)\nX_ukb = ukb.fit_transform(X_train.reshape(-1,28*28), y_train)\nprint('X_ukb:', X_ukb.shape)")


# We can check which features (that is, pixels in case) got selected:
# 

support = ukb.get_support()
plt.figure()
with sns.axes_style("white"):
    plt.imshow(support.reshape(28,28), interpolation='none')
plt.title('Support of SelectKBest() with k=%d' % k)
plt.grid(False)


# ## 3. Classification with dimension-reduced data 
# 
# Test nearest neighbor classifiers and/or decision trees with the lower-dimensional data.  Compare to classification using the original input data.
# 
# Note that you need to transform the test data into the lower-dimensional space using `transform()`.  Here is an example for PCA:
# 

X_test_pca = pca.transform(X_test.reshape(-1,28*28))
print('X_test_pca:', X_test_pca.shape)


# ## 4. Other methods for dimensionality reduction
# 
# Study and experiment with additional dimensionality reduction methods based on [decomposing](http://scikit-learn.org/stable/modules/decomposition.html) or [feature selection](http://scikit-learn.org/stable/modules/feature_selection.html).  See also [unsupervised dimensionality reduction](http://scikit-learn.org/stable/modules/unsupervised_reduction.html).
# 




# # Notebook for testing the Keras setup
# 
# This netbook is for testing the [Keras](https://keras.io/) setup for the ML hands-on.  Below is a set of required imports.  
# 
# Run the cell, and no error messages should appear.  In particular, **Keras 2 is required**. Keras also informs which backend (Theano or Tensorflow) it will be using. 
# 
# Some warnings may appear, this should be fine.
# 

get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D 
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))


# ## Getting started: 30 seconds to Keras
# 
# (This section is adapted from https://keras.io/)
# 
# The core data structure of Keras is a **model**, a way to organize layers. The main type of model is the `Sequential` model, a linear stack of layers.
# 
# A model is initialized by calling `Sequential()`:
# 

model = Sequential()


# Stacking layers is as easy as `.add()`:
# 

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))


# Once your model looks good, configure its learning process with `.compile()`:
# 

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])


# You can now begin training your model with `.fit()`.  Let's generate some random data and use it to train the model:
# 

X_train = np.random.rand(128, 100)
Y_train = np_utils.to_categorical(np.random.randint(10, size=128))

model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=2);


# Evaluate your performance on test data with `.evaluate():`
# 

X_test = np.random.rand(64, 100)
Y_test = np_utils.to_categorical(np.random.randint(10, size=64))

loss, acc = model.evaluate(X_test, Y_test, batch_size=32)
print()
print('loss:', loss, 'acc:', acc)





# # MNIST handwritten digits visualization with the self-organizing map
# 
# In this notebook, we'll use a classical visualization technique, the self-organizing map (SOM), to visualize MNIST digits.  Unfortunately, scikit-learn does not include the SOM algorithm, so we'll use an external package [minisom](https://github.com/JustGlowing/minisom).  This notebook is based on the minisom [digits example script](https://github.com/JustGlowing/minisom/blob/master/examples/example_digits.py).
# 
# First, the needed imports.
# 

get_ipython().magic('matplotlib inline')

from time import time

import numpy as np
from minisom import MiniSom

from pylab import text,show,cm,axis,figure,subplot,imshow,zeros
import matplotlib.pyplot as plt
import seaborn as sns


# Then we load the MNIST data. First time it downloads the data, which can take a while.
# 

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Let's inspect only 1024 first training samples in this notebook
X = X_train[:1024]
y = y_train[:1024]

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X:', X.shape)
print('y:', y.shape)


# ## Learning
# 
# As the SOM visualizations use a regular grid, we could use the whole MNIST training data to train the SOM.  Let's however use only a subset of the data to reduce training time. 
# 

xsize = 16
ysize = 10
epochs = 20

t0 = time()
som = MiniSom(xsize, ysize, 28*28 ,sigma=.5, learning_rate=0.2)
som.train_random(X.reshape(-1,28*28), X.shape[0]*epochs)
print('Time elapsed: %.2fs' % (time()-t0))


# Next, let's compute the nearest training sample for each SOM unit. 
# 

t0 = time()
wmap = {}
qerrors = np.empty((xsize,ysize))
qerrors.fill(np.nan)
for im,x in enumerate(X.reshape(-1,28*28)):
    (i,j) = som.winner(x)
    qe = np.linalg.norm(x-som.weights[i,j])
    if np.isnan(qerrors[i,j]) or qe<qerrors[i,j]:
        wmap[(i,j)] = im
        qerrors[i,j] = qe
print('Time elapsed: %.2fs' % (time()-t0))


# ## Visualization
# 
# We can visualize each SOM unit by the label of the nearest training sample.  The empty slots correspond to SOM units that have no associated data. 
# 

figure(1)
for j in range(ysize): # images mosaic
	for i in range(xsize):
		if (i,j) in wmap:
			text(i+.5, j+.5, str(y[wmap[(i,j)]]), 
                 color=cm.Dark2(y[wmap[(i,j)]]/9.), 
                 fontdict={'weight': 'bold', 'size': 11})
ax = axis([0,som.weights.shape[0],0,som.weights.shape[1]])


# Alternatively, as we are working with image data, we can draw the actual nearest samples for each SOM unit.
# 

figure(facecolor='white')
cnt = 0
for j in reversed(range(ysize)):
	for i in range(xsize):
		subplot(ysize,xsize,cnt+1,frameon=False, xticks=[], yticks=[])
		if (i,j) in wmap:
			imshow(X[wmap[(i,j)]])
		else:
			imshow(zeros((28,28)))
		cnt = cnt + 1


# As the SOM weights are also vectors in the input space, we can also draw the weights as images. 
# 

figure(facecolor='white')
cnt = 0
for j in reversed(range(ysize)):
	for i in range(xsize):
		subplot(ysize,xsize,cnt+1,frameon=False, xticks=[], yticks=[])
		imshow(som.weights[i,j].reshape(28,28))
		cnt = cnt + 1





