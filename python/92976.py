# ## Wrappers for the Scikit-Learn API
# 
# You can use Sequential Keras models (single-input only) as part of your Scikit-Learn workflow via the wrappers found at keras.wrappers.scikit_learn.py.
# 
# There are two wrappers available:
# 
# keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params), which implements the Scikit-Learn classifier interface,
# 
# keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params), which implements the Scikit-Learn regressor interface.
# 
# #### Arguments
# 
# * build_fn: callable function or class instance
# * sk_params: model parameters & fitting parameters
# 
# build_fn should construct, compile and return a Keras model, which will then be used to fit/predict. One of the following three values could be passed to build_fn:
# 
# 1. A function
# 2. An instance of a class that implements the call method
# 3. None. This means you implement a class that inherits from either KerasClassifier or KerasRegressor. The call method of the present class will then be treated as the default build_fn.
# 
# sk_params takes both model parameters and fitting parameters. Legal model parameters are the arguments of build_fn. Note that like all other estimators in scikit-learn, 'build_fn' should provide default values for its arguments, so that you could create the estimator without passing any values to sk_params.
# 
# sk_params could also accept parameters for calling fit, predict, predict_proba, and score methods (e.g., epochs, batch_size). fitting (predicting) parameters are selected in the following order:
# 
# 1. Values passed to the dictionary arguments of fit, predict, predict_proba, and score methods
# 2. Values passed to sk_params
# 3. The default values of the keras.models.Sequential fit, predict, predict_proba and score methods
# 
# When using scikit-learn's grid_search API, legal tunable parameters are those you could pass to sk_params, including fitting parameters. In other words, you could use grid_search to search for the best batch_size or epochs as well as the model parameters.
# 

from keras.models import Sequential
from keras.layers import Dense, Activation

def build_model(optimizer='rmsprop', dense_dims=32):
    model = Sequential()
    model.add(Dense(dense_dims, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
    


from keras.wrappers.scikit_learn import KerasClassifier

keras_classifier = KerasClassifier(build_model, epochs=2)


import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

keras_classifier.fit(data, labels)


keras_classifier.predict_proba(data[:2])


from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(keras_classifier, {'epochs': [2, 3], 'dense_dims':[16, 32]})


gs.fit(data, labels)


gs.best_params_





# ## Usage of optimizers
# 
# An optimizer is one of the two arguments required for compiling a Keras model:
# 

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(64, input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)


# You can either instantiate an optimizer before passing it to model.compile() , as in the above example, or you can call it by its name. In the latter case, the default parameters for the optimizer will be used.
# 

# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')


# ## Parameters common to all Keras optimizers
# 
# The parameters clipnorm and clipvalue can be used with all optimizers to control gradient clipping:
# 

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)


# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)


import tensorflow as tf

# Use a tensorflow optimizer
pgd = optimizers.TFOptimizer(tf.train.ProximalGradientDescentOptimizer(0.01))


# ## Usage of loss functions
# 
# A loss function (or objective function, or optimization score function) is the other of the two parameters required to compile a model:
# 

model.compile(loss='mean_squared_error', optimizer='sgd')


from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')


# You can either pass the name of an existing loss function, or pass a TensorFlow/Theano symbolic function that returns a scalar for each data-point and takes the following two arguments:
# 
# * y_true: True labels. TensorFlow/Theano tensor.
# * y_pred: Predictions. TensorFlow/Theano tensor of the same shape as y_true.
# 
# The actual optimized objective is the mean of the output array across all datapoints.
# 

model.compile(loss=tf.nn.log_poisson_loss, optimizer='sgd')


# ## Usage of metrics
# 
# A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the metrics parameter when a model is compiled.
# 

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])


from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])


# A metric function is similar to an loss function, except that the results from evaluating a metric are not used when training the model. All loss functions are metrics.
# 
# You can either pass the name of an existing metric, or pass a Theano/TensorFlow symbolic function (see Custom metrics).
# 
# #### Arguments
# 
# * y_true: True labels. Theano/TensorFlow tensor.
# * y_pred: Predictions. Theano/TensorFlow tensor of the same shape as y_true.
# 
# #### Returns
# 
# Single tensor value representing the mean of the output array across all datapoints.
# 

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])





# ## About Keras layers
# 
# All Keras layers have a number of methods in common:
# 
# 

from keras.layers import Dense, Input

inputs = Input((8,))
layer = Dense(8)

layer.get_weights()


x = layer(inputs)


layer.name


layer.__class__.__name__


layer.trainable = True


layer.get_weights()


import numpy as np

new_bias = np.array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.])
layer.set_weights([layer.get_weights()[0], new_bias])


layer.get_weights()


layer.input, layer.output, layer.input_shape, layer.output_shape


x = layer(x)


layer.input, layer.output, layer.input_shape, layer.output_shape


layer.get_input_at(0), layer.get_output_at(0), layer.get_input_shape_at(0), layer.get_output_shape_at(0)


# ## Saving and Loading Individual Layer Configs
# 

layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
config


from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})


# ## Lambda Method
# 
# The lambda method allows you to make any stateless transformation (available in our backend which is tensorflow) to our tensors. Including transformation logic outside a layer will mess up any model, so make sure to put it inside a lambda layer.
# 
# The output shape is not necessary in tensorflow as it will try to impute it. That being said, for some complex functions it might be better to specify it yourself.
# 

from keras.layers import Lambda
from keras import backend as K

Lambda(lambda x: x ** 2)


def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

Lambda(antirectifier, output_shape=antirectifier_output_shape)


import tensorflow as tf
Lambda(tf.reduce_mean, output_shape=(1,))


def to_pow(x, po=2):
    return x ** po

Lambda(to_pow, arguments={'po':5})


# ## Writing your own Keras layers
# 
# For simple, stateless custom operations, you are probably better off using layers.core.Lambda layers. But for any custom operation that has trainable weights, you should implement your own layer.
# 
# Here is the skeleton of a Keras layer, as of Keras 2.0 (if you have an older version, please upgrade). There are only three methods you need to implement:
# 
# * build(input_shape): this is where you will define your weights. This method must set self.built = True, which can be done by calling super([Layer], self).build().
# * call(x): this is where the layer's logic lives. Unless you want your layer to support masking, you only have to care about the first argument passed to call: the input tensor.
# * compute_output_shape(input_shape): in case your layer modifies the shape of its input, you should specify here the shape transformation logic. This allows Keras to do automatic shape inference.
# 

from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)











# ## Inspecting a model
# 

from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(100,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid', name='output')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

history = model.fit(data, labels, epochs=2)


model.summary()


model.get_config()


model.get_layer('output')


model.layers


history.history


model.history.history


# ## Saving and Loading Models
# 

from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)


json_string


from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)


config = model.get_config()
model = Model.from_config(config)


for weight in model.get_weights():
    print weight.shape


model.set_weights(model.get_weights())


get_ipython().magic('pinfo model.save_weights')


get_ipython().magic('pinfo model.load_weights')


from keras.models import load_model

get_ipython().magic('pinfo model.save')


get_ipython().magic('pinfo load_model')





# ## Model visualization
# 
# The keras.utils.vis_utils module provides utility functions to plot a Keras model (using graphviz).
# 
# plot_model takes two optional arguments:
# 
# * show_shapes (defaults to False) controls whether output shapes are shown in the graph.
# * show_layer_names (defaults to True) controls whether layer names are shown in the graph.
# 
# You can also directly obtain the pydot.Graph object and render it yourself, for example to show it in an ipython notebook :
# 

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.summary()


# pip install pydot-ng
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))


# ## Applications
# 
# Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning.
# 
# Weights are downloaded automatically when instantiating a model. They are stored at ~/.keras/models/.
# 
# #### Available models
# 
# Models for image classification with weights trained on ImageNet:
# 
# * Xception
# * VGG16
# * VGG19
# * ResNet50
# * InceptionV3
# 
# All of these architectures (except Xception) are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image data format set in your Keras configuration file at ~/.keras/keras.json. For instance, if you have set image_data_format=tf, then any model loaded from this repository will get built according to the TensorFlow data format convention, "Width-Height-Depth".
# 
# The Xception model is only available for TensorFlow, due to its reliance on SeparableConvolution layers.
# 

from keras.applications.vgg16 import VGG16

model = VGG16(weights=None, include_top=False)

SVG(model_to_dot(model).create(prog='dot', format='svg'))


from keras.applications.resnet50 import ResNet50

model = ResNet50(weights=None)


SVG(model_to_dot(model).create(prog='dot', format='svg'))


# All these models also come with these two functions
from keras.applications.resnet50 import preprocess_input, decode_predictions

# Make sure to use them when making predictions!!!





# ## Getting started with the Keras functional API
# 
# The Keras functional API is the way to go for defining complex models, such as multi-output models, directed acyclic graphs, or models with shared layers.
# 
# In my opinion this is the api that you want to always use!
# 

# ## First example: a densely-connected network
# 
# Some things to note:
# 
# * A layer instance is callable (on a tensor), and it returns a tensor
# * Input tensor(s) and output tensor(s) can then be used to define a Model
# * Such a model can be trained just like Keras Sequential models.
# 

from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Notice that the only things we had to reference are the input and output tensors. We could make as complex interactions as we want, but all the model cares about are the inputs and outputs.
# 

# ## All models are callable, just like layers
# 
# With the functional API, it is easy to re-use trained models: you can treat any model as if it were a layer, by calling it on a tensor. Note that by calling a model you aren't just re-using the architecture of the model, you are also re-using its weights.
# 
# 

x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)


# This can allow, for instance, to quickly create models that can process sequences of inputs. You could turn an image classification model into a video classification model, in just one line.
# 

from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)


# ## Multi-input and multi-output models
# 
# Here's a good use case for the functional API: models with multiple inputs and outputs. The functional API makes it easy to manipulate a large number of intertwined datastreams.
# 
# consider the below:
# 

from keras.layers import concatenate

x_in = Input(shape=(100,), name='x_in')
y_in = Input(shape=(100,), name='y_in')

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(x_in)
y = Dense(64, activation='relu')(y_in)

z = concatenate([x, y])

x = Dense(1, activation='sigmoid', name='x_out')(z)
y = Dense(10, activation='softmax', name='y_out')(z)


# To define a model with multiple inputs or outputs, you just need to specify a list:
# 

model = Model(inputs=[x_in, y_in], outputs=[x, y])

model.summary()


# There are now a couple of ways to compile the model. First is just by passing in lists of losses and loss weights:
# 

from keras.utils import to_categorical

import numpy as np
data = np.random.random((1000, 100))
xs = np.random.randint(2, size=(1000, 1))
ys = np.random.randint(10, size=(1000, 1))

model.compile(optimizer='rmsprop', loss=['binary_crossentropy', 'categorical_crossentropy'],
              loss_weights=[1., 0.2])

model.fit([data, data], [xs, to_categorical(ys)],
          epochs=1, batch_size=32)


# The second is to specify a dictionary (refering to the names of the output tensors):
# 

model.compile(optimizer='rmsprop',
              loss={'x_out': 'binary_crossentropy', 'y_out': 'categorical_crossentropy'},
              loss_weights={'x_out': 1., 'y_out': 0.2})

# And trained it via:
model.fit({'x_in': data, 'y_in': data},
          {'x_out': xs, 'y_out': to_categorical(ys)},
          epochs=1, batch_size=32)


# ## Shared layers
# 
# Another good use for the functional API are models that use shared layers. Let's take a look at shared layers.
# 
# The use is somewhat simple. We save the layer we want to use and apply it multiple times.
# 

inputs = Input(shape=(64,))

# a layer instance is callable on a tensor, and returns a tensor
layer_we_share = Dense(64, activation='relu')

# Now we apply the layer twice
x = layer_we_share(inputs)
x = layer_we_share(x)

predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ## The concept of layer "node"
# 
# Whenever you are calling a layer on some input, you are creating a new tensor (the output of the layer), and you are adding a "node" to the layer, linking the input tensor to the output tensor. When you are calling the same layer multiple times, that layer owns multiple nodes indexed as 0, 1, 2...
# 
# In previous versions of Keras, you could obtain the output tensor of a layer instance via layer.get_output(), or its output shape via layer.output_shape. You still can (except get_output() has been replaced by the property output). But what if a layer is connected to multiple inputs?
# 
# As long as a layer is only connected to one input, there is no confusion, and .output will return the one output of the layer:

a = Input(shape=(140, 256))

dense = Dense(32)
affine_a = dense(a)

assert dense.output == affine_a


# Not so if the layer has multiple inputs:
# 

a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

dense = Dense(32)
affine_a = dense(a)
affine_b = dense(b)

dense.output


# Okay then. The following works:
# 

assert dense.get_output_at(0) == affine_a
assert dense.get_output_at(1) == affine_b


# Simple enough, right?
# 
# The same is true for the properties input_shape and output_shape: as long as the layer has only one node, or as long as all nodes have the same input/output shape, then the notion of "layer output/input shape" is well defined, and that one shape will be returned by layer.output_shape/layer.input_shape. But if, for instance, you apply a same Conv2D layer to an input of shape (3, 32, 32), and then to an input of shape (3, 64, 64), the layer will have multiple input/output shapes, and you will have to fetch them by specifying the index of the node they belong to.




# ## Simple Preprocessing
# 
# You may have already seen me use these guys, but these are ubiquitous:
# 
# * to_categorical
# * normalize
# 

# Generate dummy data
import numpy as np
data = np.random.random((2, 5))
labels = np.random.randint(3, size=(2, 3))

data, labels


from keras.utils import to_categorical, normalize

to_categorical(labels, num_classes=4)


normalize(data, order=1)


# ## Sequence Preprocessing
# 
# The next set of tools is specific to sequences, and really useful for RNNs:
# 
# * pad_sequences: Transform a list of num_samples sequences (lists of scalars) into a 2D Numpy array of shape (num_samples, num_timesteps). num_timesteps is either the maxlen argument if provided, or the length of the longest sequence otherwise. Sequences that are shorter than num_timesteps are padded with value at the end. Sequences longer than num_timesteps are truncated so that it fits the desired length. Position where padding or truncation happens is determined by padding or truncating, respectively.
# * skipgrams: Transforms a sequence of word indexes (list of int) into couples of the form:
#     - (word, word in the same window), with label 1 (positive samples).
#     - (word, random word from the vocabulary), with label 0 (negative samples).
# 

sequences = [
    [1,2,4,4],
    [3],
    [5,6,4,2,1,7,7,4,3],
    [3,3,4,3,2]
]


from keras.preprocessing.sequence import pad_sequences

padded_sequences = pad_sequences(sequences, maxlen=None, dtype='int32',
    padding='pre', truncating='post', value=0.)

padded_sequences


from keras.preprocessing.sequence import skipgrams

grams = skipgrams(padded_sequences[0], vocabulary_size=8,
    window_size=1, negative_samples=1., shuffle=True,
    categorical=False)

grams 


# ## Text preprocessing
# 
# We are starting to get a bit more complex. Realize the the output of text preprocessing can be fed to sequence preprocessing. There are three important functions here:
# 
# * text_to_word_sequence: Split a sentence into a list of words.
# * one_hot: One-hot encode a text into a list of word indexes in a vocabulary of size n.
# * Tokenizer: Class for vectorizing texts, or/and turning texts into sequences (=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i).
# 

text ="""
    My name is Nathaniel.
    I like data science.
    Let's do deep learning.
    Keras, my fave lib.
    """


from keras.preprocessing.text import text_to_word_sequence

words = text_to_word_sequence(text, lower=True, split=" ")

words


# we can change the filter chars too
text_to_word_sequence(text, filters="'", lower=True, split=" ")


from keras.preprocessing.text import one_hot

one_hot(text, n=8, lower=True, split=" ")


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=None, lower=True, split=" ")


# tokenizer.fit_on_sequences
tokenizer.fit_on_texts([text])


tokenizer.texts_to_sequences([text])


tokenizer.texts_to_matrix([text], 'count')


tokenizer.texts_to_matrix(['Data Science is fun'], 'count')


# ## Image Preprocessing
# 
# The image preprocessing is considerably more complex, but comprises only a single function:
# 
# * ImageDataGenerator: Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely.
# 
# So we will go over this carefully
# 

from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


dataset = load_sample_images()


plt.imshow(dataset.images[0])


plt.imshow(dataset.images[1])


from keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,
    preprocessing_function=None)


idg.fit(dataset.images)


import numpy
it = idg.flow(numpy.array(dataset.images), numpy.array([1, 1,]), batch_size=1)


plt.imshow(numpy.array(next(it)[0][0, :, :, :], dtype='uint8'))


# finally there is the option to flow_from_directory





# ## Keras: Deep Learning library for Theano and TensorFlow
# 
# Keras is a high-level neural networks API, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
# 
# Use Keras if you need a deep learning library that:
# 
# * Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
# * Supports both convolutional networks and recurrent networks, as well as combinations of the two.
# * Runs seamlessly on CPU and GPU.
# 

# ### Guiding principles
# 
# * __User friendliness.__ Keras is an API designed for human beings, not machines. It puts user experience front and center. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.
# * __Modularity.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.
# * __Easy extensibility.__ New modules are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Keras suitable for advanced research.
# * __Work with Python.__ No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.
# 

# ### Getting started
# 
# The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers. For more complex architectures, you should use the Keras functional API, which allows to build arbitrary graphs of layers.
# 
# We will only very breifly touch on the Sequential model because the functional model is much more expressive and frankly easier to use.
# 

from keras.models import Sequential

model = Sequential()


# quickly grab the data
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


x_train.shape


from keras.layers import Dense

model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))


# Once the model is created, you will need to compile it (and specify some opitonal params) 
# 

model.compile(loss='mean_squared_error', 
              optimizer='adam',
              metrics=['mean_absolute_percentage_error'])


# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=20, batch_size=404)


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


loss_and_metrics


prices = model.predict(x_test, batch_size=128)


prices[:5]





# ## Usage of callbacks
# 
# A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument callbacks) to the .fit() method of the Sequential or Model classes. The relevant methods of the callbacks will then be called at each stage of the training.
# 

# ## Base Callbacks
# 
# There are a couple of callbacks that you are already using without knowing it:
# 
# * BaseLogger: Callback that accumulates epoch averages of metrics.
# * ProgbarLogger: Callback that prints metrics to stdout.
# * History: Callback that records events into a History object.
# 

# ## Even More Callbacks
# 
# I'll show off a set of callbacks available to you to use with any model, and then we will talk about custom callbacks.
# 

from keras.callbacks import ModelCheckpoint

mc = ModelCheckpoint(
    filepath='tmp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    period=5)


from keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=5,
    verbose=1,
    mode='max')


from keras.callbacks import LearningRateScheduler

lrs = LearningRateScheduler(lambda epoch: 1./epoch)


from keras.callbacks import ReduceLROnPlateau

rlrop = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=10, 
    verbose=0, 
    mode='auto', 
    epsilon=0.0001, 
    cooldown=4, 
    min_lr=10e-7)


from keras.callbacks import CSVLogger

csvl = CSVLogger(
    filename='tmp/training.log',
    separator=',', 
    append=False)


from keras.callbacks import TensorBoard

TensorBoard(
    log_dir='./logs', 
    histogram_freq=0, 
    write_graph=True, 
    write_images=False,
    embeddings_freq=100,
    embeddings_layer_names=None, # this list of embedding layers...
    embeddings_metadata=None)      # with this metadata associated with them.)


# ## Lambda Callback
# 
# If that was not enough for you, here is the big one. 
# 
# This callback is constructed with anonymous functions that will be called at the appropriate time. Note that the callbacks expects positional arguments, as: - on_epoch_begin and on_epoch_end expect two positional arguments: epoch, logs - on_batch_begin and on_batch_end expect two positional arguments: batch, logs - on_train_begin and on_train_end expect one positional argument: logs
# 
# #### Arguments
# 
# * on_epoch_begin: called at the beginning of every epoch.
# * on_epoch_end: called at the end of every epoch.
# * on_batch_begin: called at the beginning of every batch.
# * on_batch_end: called at the end of every batch.
# * on_train_begin: called at the beginning of model training.
# * on_train_end: called at the end of model training.
# 

from keras.callbacks import LambdaCallback

# Print the batch number at the beginning of every batch.
def print_batch(batch, logs):
    print batch
batch_print_callback = LambdaCallback(
    on_batch_begin=print_batch)

# Terminate some processes after having finished model training.
processes = []
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
    p.terminate() for p in processes if p.is_alive()])


# ## Super Custom Callbacks
# 
# You can create a custom callback by extending the base class keras.callbacks.Callback. A callback has access to its associated model through the class property self.model.
# 
# Abstract base class used to build new callbacks.
# 
# #### Properties
# 
# * params: dict. Training parameters (eg. verbosity, batch size, number of epochs...).
# * model: instance of keras.models.Model. Reference of the model being trained.
# 
# The logs dictionary that callback methods take as argument will contain keys for quantities relevant to the current batch or epoch.
# 
# Currently, the .fit() method of the Sequential model class will include the following quantities in the logs that it passes to its callbacks:
# 
# * on_epoch_end: logs include acc and loss, and optionally include val_loss (if validation is enabled in fit), and val_acc (if validation and accuracy monitoring are enabled).
# * on_batch_begin: logs include size, the number of samples in the current batch.
# * on_batch_end: logs include loss, and optionally acc (if accuracy monitoring is enabled).
# 
# Here's a simple example saving a list of losses over each batch during training:
# 

import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# ## Usage of initializers
# 
# Initializations define the way to set the initial random weights of Keras layers.
# 
# The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply kernel_initializer and bias_initializer:
# 

from keras.layers import Dense

get_ipython().magic('pinfo Dense')


layer = Dense(10, kernel_initializer='lecun_uniform', bias_initializer='ones')


from keras.initializers import Constant

layer = Dense(10, kernel_initializer='he_normal', bias_initializer=Constant(7))


# As you can see there are plenty of initializers, you can even make your own:
# 

from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

Dense(64, kernel_initializer=my_init)


# ## Usage of activations
# 
# Activations can either be used through an Activation layer, or through the activation argument supported by all forward layers:
# 

from keras.layers import Activation, Dense, Input

x = Input((1,))
x = Dense(64)(x)
x = Activation('tanh')(x)


# This is equivalent to:
# 

x = Input((1,))
x = Dense(64, activation='tanh')(x)


# You can also pass an element-wise Tensorflow/Theano function as an activation:
# 

from keras import backend as K

x = Input((1,))
x = Dense(64, activation=K.tanh)(x)
x = Activation(K.tanh)(x)


# ## Usage of regularizers
# 
# Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
# 
# The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers Dense, Conv1D, Conv2D and Conv3D have a unified API.
# 
# These layers expose 3 keyword arguments:
# 
# * kernel_regularizer: instance of keras.regularizers.Regularizer
# * bias_regularizer: instance of keras.regularizers.Regularizer
# * activity_regularizer: instance of keras.regularizers.Regularizer
# 

from keras import regularizers
Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))


# available regularizers
regularizers.l1(0.)
regularizers.l2(0.)
regularizers.l1_l2(0.)


# Custom regularizer
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

Dense(64, input_dim=64,
                kernel_regularizer=l1_reg)


# In addition there is the activity regularization layer that can help with this:
# 

from keras.layers import ActivityRegularization

get_ipython().magic('pinfo ActivityRegularization')


# ## Usage of constraints
# 
# Functions from the constraints module allow setting constraints (eg. non-negativity) on network parameters during optimization.
# 
# The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers Dense, Conv1D, Conv2D and Conv3D have a unified API.
# 
# These layers expose 2 keyword arguments:
# 
# * kernel_constraint for the main weights matrix
# * bias_constraint for the bias.
# 

from keras.constraints import max_norm

Dense(64, kernel_constraint=max_norm(2.))


# Available constraints
# 
# * max_norm(max_value=2, axis=0): maximum-norm constraint
# * non_neg(): non-negativity constraint
# * unit_norm(): unit-norm constraint, enforces the matrix to have unit norm along the last axis
# 

# ## Putting it all together
# 
# So you can apply all the concepts to a core layer or use them to make your own! Below I show you how to make a layer that uses all of the above:
# 

from keras.engine.topology import Layer
from keras.activations import hard_sigmoid
from keras import regularizers
import numpy as np


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      contraint='unit_norm',
                                      regularizer=regularizers.l1(1.),
                                      trainable=True)
        
        # Another way to enable this regularization is with the add loss function
        # self.add_loss(self.kernel, inputs=None)
        
        super(MyLayer, self).build(input_shape) 

    def call(self, x):
        return hard_sigmoid(K.dot(x, self.kernel))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# Notice how this all applied to the add_weight function. Both the initializer and the constraint can only be applied there! Regularization as you can see, can be applied up and down the pipe. And activations are pretty self evident.
# 




