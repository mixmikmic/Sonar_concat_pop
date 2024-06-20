# # Advanced Convolutional Neural Networks (CNN) - 2
# - Objective: try different structures of CNNs
# - Note: examples are performed on **i5 7600 + gtx 1060 6GB **
# 

# ## CNN for Sentence Classification
# - It is widely known that CNNs are good for snapshot-like data, like images
# - However, CNNs are effectve for NLP tasks as well
# - For more information, refer to:
#     - Kim 2014 (http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)
#     - Zhang et al 2015 (https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
#     
# <br>
# - In this section, we perform sentence classification with CNNs (Kim 2014)
# </br>
# <img src="http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-8.03.47-AM.png" style="width: 800px"/>
# 
# <br>
# - Pixels are made of embedding vectors of each word in a sentence
# - Convolutions are performed based on word-level
# - Classify each sentence as positive (1) or negative (0)
# 
# <img src="http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-12.05.40-PM.png" style="width: 600px"/>
# 

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


# ## Load Dataset
# - IMDb Movie reviews sentiment classification Dataset
# - Doc: https://keras.io/datasets/
# - Parameter description
#     - num_features: number of words to account for (i.e., only frequent n words are considered)
#     - sequence_length: maximum number of words for a sentence (if sentence is too short, pad by zeros)
#     - embedding_dimension: dimensionality of embedding space (i.e., dimensionality of vector representation for each word)
# 

num_features = 3000
sequence_length = 300
embedding_dimension = 100


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_features)


X_train = pad_sequences(X_train, maxlen = sequence_length)
X_test = pad_sequences(X_test, maxlen = sequence_length)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## 0. Basic CNN sentence classificationmodel
# - Basic CNN using 1D convolution and pooling
# - Known as "temporal convolution"
# 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Embedding, Flatten
from keras import optimizers


def imdb_cnn():
    model = Sequential()
    
    # use Embedding layer to create vector representation of each word => it is fine-tuned every iteration
    model.add(Embedding(input_dim = 3000, output_dim = embedding_dimension, input_length = sequence_length))
    model.add(Conv1D(filters = 50, kernel_size = 5, strides = 1, padding = 'valid'))
    model.add(MaxPooling1D(2, padding = 'valid'))
    
    model.add(Flatten())
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    adam = optimizers.Adam(lr = 0.001)
    
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    
    return model


model = imdb_cnn()


get_ipython().run_cell_magic('time', '', 'history = model.fit(X_train, y_train, batch_size = 50, epochs = 100, validation_split = 0.2, verbose = 0)')


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 1. Advanced CNN sentence classification model - 1
# - Advanced CNN using 2D convolution and pooling
#     - Embedding layer is "reshaped" to 4D to fit into 2D convolutional layer
# - Perform global max pooling for each window
# 

from keras.layers import Reshape, Conv2D, GlobalMaxPooling2D


def imdb_cnn_2():
    model = Sequential()

    model.add(Embedding(input_dim = 3000, output_dim = embedding_dimension, input_length = sequence_length))
    model.add(Reshape((sequence_length, embedding_dimension, 1), input_shape = (sequence_length, embedding_dimension)))
    model.add(Conv2D(filters = 50, kernel_size = (5, embedding_dimension), strides = (1,1), padding = 'valid'))
    model.add(GlobalMaxPooling2D())

    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr = 0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    
    return model


model = imdb_cnn_2()


get_ipython().run_cell_magic('time', '', 'history = model.fit(X_train, y_train, batch_size = 50, epochs = 100, validation_split = 0.2, verbose = 0)')


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 3. Advanced CNN sentence classification model - 2
# - Structure more similar to that proposed in **Kim 2014**
#     - Three convoltion operations with different filter sizes are performed and their results are merged
# 

from keras.models import Model
from keras.layers import concatenate, Input


filter_sizes = [3, 4, 5]


def convolution():
    inn = Input(shape = (sequence_length, embedding_dimension, 1))
    convolutions = []
    # we conduct three convolutions & poolings then concatenate them.
    for fs in filter_sizes:
        conv = Conv2D(filters = 100, kernel_size = (fs, embedding_dimension), strides = 1, padding = "valid")(inn)
        nonlinearity = Activation('relu')(conv)
        maxpool = MaxPooling2D(pool_size = (sequence_length - fs + 1, 1), padding = "valid")(nonlinearity)
        convolutions.append(maxpool)
        
    outt = concatenate(convolutions)
    model = Model(inputs = inn, outputs = outt)
        
    return model


def imdb_cnn_3():
    
    model = Sequential()
    model.add(Embedding(input_dim = 3000, output_dim = embedding_dimension, input_length = sequence_length))
    model.add(Reshape((sequence_length, embedding_dimension, 1), input_shape = (sequence_length, embedding_dimension)))
    
    # call convolution method defined above
    model.add(convolution())
    
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr = 0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    
    return model


model = imdb_cnn_3()


get_ipython().run_cell_magic('time', '', 'history = model.fit(X_train, y_train, batch_size = 50, epochs = 100, validation_split = 0.2, verbose = 0)')


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 3. Advanced CNN sentence classification model - 3
# - Structure more similar to that proposed in **Kim 2014**
#     - More techniques are applied to generate more stable results
# 

from keras.layers import BatchNormalization


filter_sizes = [3, 4, 5]


def convolution():
    inn = Input(shape = (sequence_length, embedding_dimension, 1))
    convolutions = []
    # we conduct three convolutions & poolings then concatenate them.
    for fs in filter_sizes:
        conv = Conv2D(filters = 100, kernel_size = (fs, embedding_dimension), strides = 1, padding = "valid")(inn)
        nonlinearity = Activation('relu')(conv)
        maxpool = MaxPooling2D(pool_size = (sequence_length - fs + 1, 1), padding = "valid")(nonlinearity)
        convolutions.append(maxpool)
        
    outt = concatenate(convolutions)
    model = Model(inputs = inn, outputs = outt)
        
    return model


def imdb_cnn_4():
    
    model = Sequential()
    model.add(Embedding(input_dim = 3000, output_dim = embedding_dimension, input_length = sequence_length))
    model.add(Reshape((sequence_length, embedding_dimension, 1), input_shape = (sequence_length, embedding_dimension)))
    model.add(Dropout(0.5))
    # call convolution method defined above
    model.add(convolution())
    
    model.add(Flatten())
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr = 0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    
    return model


model = imdb_cnn_4()


get_ipython().run_cell_magic('time', '', 'history = model.fit(X_train, y_train, batch_size = 50, epochs = 100, validation_split = 0.2, verbose = 0)')


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# # Text preprocessing with Keras
# - Basic text preprocessing using Keras API
# - Doc: https://keras.io/preprocessing/text/
# 

from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras.preprocessing.sequence import pad_sequences


# ### Tokenization of a sentence
# - Tokenization: the process of converting a sequence of characters into a sequence of tokens (https://en.wikipedia.org/wiki/Lexical_analysis#Token)
# 

sentences = ['Curiosity killed the cat.', 'But satisfaction brought it back']


tk = Tokenizer()    # create Tokenizer instance


tk.fit_on_texts(sentences)    # tokenizer should be fit with text data in advance


# #### Converting sentence into (integer) sequence
# - One of simple ways of modeling text is to create sequence of integers for each sentence
# - By doing so, information regarding order of words can be preserved
# 

seq = tk.texts_to_sequences(sentences)
print(seq)


# #### One-hot encoding of sentence
# - Sometimes, it is preferred to check only whether certain word appeared in sentence or not
# - This way of characterizing sentence is called "one-hot encoding"
#     - IF word appeared in sentence, it is encoded as **"one"**
#     - IF not, it is encoded as **"zero"**
# 

mat = tk.sequences_to_matrix(seq)
print(mat)


# #### Padding sequences
# - Oftentimes, to preserve the dimensionality of sentences, zero padding is performed
# - Idea is similar to that of padding exterior of image-format data, but applied to sequences
# 

# if set padding to 'pre', zeros are appended to start of sentences
pad_seq = pad_sequences(seq, padding='pre')     
print(pad_seq)


# if set padding to 'post', zeros are appended to end of sentences
pad_seq = pad_sequences(seq, padding='post')
print(pad_seq)


# # Advanced MLP
# - Advanced techniques for training neural networks
#     - Weight Initialization
#     - Nonlinearity (Activation function)
#     - Optimizers
#     - Batch Normalization
#     - Dropout (Regularization)
#     - Model Ensemble
# 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical


# ## Load Dataset
# - MNIST dataset
# - source: http://yann.lecun.com/exdb/mnist/
# 

(X_train, y_train), (X_test, y_test) = mnist.load_data()


plt.imshow(X_train[0])    # show first number in the dataset
plt.show()
print('Label: ', y_train[0])


plt.imshow(X_test[0])    # show first number in the dataset
plt.show()
print('Label: ', y_test[0])


# reshaping X data: (n, 28, 28) => (n, 784)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))


# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# use only 33% of training data to expedite the training process
X_train, _ , y_train, _ = train_test_split(X_train, y_train, test_size = 0.67, random_state = 7)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Basic MLP model
# - Naive MLP model without any alterations
# 

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers


model = Sequential()


model.add(Dense(50, input_shape = (784, )))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))


sgd = optimizers.SGD(lr = 0.001)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])


history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


# Training and validation accuracy seems to improve after around 60 epochs
# 

results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 1. Weight Initialization
# - Changing weight initialization scheme can significantly improve training of the model by preventing vanishing gradient problem up to some degree
# - He normal or Xavier normal initialization schemes are SOTA at the moment
# - Doc: https://keras.io/initializers/
# 

# from now on, create a function to generate (return) models
def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, ), kernel_initializer='he_normal'))     # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(50, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(50, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(50, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(10, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


# Training and validation accuracy seems to improve after around 60 epochs
# 

results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 2. Nonlinearity (Activation function)
# - Sigmoid functions suffer from gradient vanishing problem, making training slower
# - There are many choices apart from sigmoid and tanh; try many of them!
#     - **'relu'** (rectified linear unit) is one of the most popular ones
#     - **'selu'** (scaled exponential linear unit) is one of the most recent ones
# - Doc: https://keras.io/activations/
# 

# <img src="http://cs231n.github.io/assets/nn1/sigmoid.jpeg" style="width: 400px"/>
# <center> **Sigmoid Activation Function** </center>
# <img src="http://cs231n.github.io/assets/nn1/relu.jpeg" style="width: 400px"/>
# <center> **Relu Activation Function** </center>
# 

def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, )))
    model.add(Activation('relu'))    # use relu
    model.add(Dense(50))
    model.add(Activation('relu'))    # use relu
    model.add(Dense(50))
    model.add(Activation('relu'))    # use relu
    model.add(Dense(50))
    model.add(Activation('relu'))    # use relu
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


# Training and validation accuracy improve instantaneously, but reach a plateau after around 30 epochs
# 

results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 3. Optimizers
# - Many variants of SGD are proposed and employed nowadays
# - One of the most popular ones are Adam (Adaptive Moment Estimation)
# - Doc: https://keras.io/optimizers/
# 

# <img src="http://cs231n.github.io/assets/nn3/opt2.gif" style="width: 400px"/>
# <br><center> **Relative convergence speed of different optimizers** </center></br>
# 

def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, )))
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(Activation('sigmoid'))  
    model.add(Dense(50))
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(Activation('sigmoid'))    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)                     # use Adam optimizer
    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


# Training and validation accuracy improve instantaneously, but reach plateau after around 50 epochs
# 

results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 4. Batch Normalization
# - Batch Normalization, one of the methods to prevent the "internal covariance shift" problem, has proven to be highly effective
# - Normalize each mini-batch before nonlinearity
# - Doc: https://keras.io/optimizers/
# 

# <img src="https://raw.githubusercontent.com/torch/torch.github.io/master/blog/_posts/images/resnets_modelvariants.png" style="width: 500px"/>
# 
# <br> Batch normalization layer is usually inserted after dense/convolution and before nonlinearity
# 

from keras.layers import BatchNormalization


def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, )))
    model.add(BatchNormalization())                    # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(BatchNormalization())                    # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(BatchNormalization())                    # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(BatchNormalization())                    # Add Batchnorm layer before Activation
    model.add(Activation('sigmoid'))    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


# Training and validation accuracy improve consistently, but reach plateau after around 60 epochs
# 

results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 5. Dropout (Regularization)
# - Dropout is one of powerful ways to prevent overfitting
# - The idea is simple. It is disconnecting some (randomly selected) neurons in each layer
# - The probability of each neuron to be disconnected, namely 'Dropout rate', has to be designated
# - Doc: https://keras.io/layers/core/#dropout
# 

# <img src="https://image.slidesharecdn.com/lecture29-convolutionalneuralnetworks-visionspring2015-150504114140-conversion-gate02/95/lecture-29-convolutional-neural-networks-computer-vision-spring2015-62-638.jpg?cb=1430740006" style="width: 500px"/>
# 

from keras.layers import Dropout


def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, )))
    model.add(Activation('sigmoid'))    
    model.add(Dropout(0.2))                        # Dropout layer after Activation
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))                        # Dropout layer after Activation
    model.add(Dense(50))
    model.add(Activation('sigmoid'))    
    model.add(Dropout(0.2))                        # Dropout layer after Activation
    model.add(Dense(50))
    model.add(Activation('sigmoid'))    
    model.add(Dropout(0.2))                         # Dropout layer after Activation
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


model = mlp_model()
history = model.fit(X_train, y_train, validation_split = 0.3, epochs = 100, verbose = 0)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()


# Validation results does not improve since it did not show signs of overfitting, yet.
# <br> Hence, the key takeaway message is that apply dropout when you see a signal of overfitting.
# 

results = model.evaluate(X_test, y_test)


print('Test accuracy: ', results[1])


# ## 6. Model Ensemble
# - Model ensemble is a reliable and promising way to boost performance of the model
# - Usually create 8 to 10 independent networks and merge their results
# - Here, we resort to scikit-learn API, **VotingClassifier**
# - Doc: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
# 

# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRs1CBSEtpp5yj6SJ5K_nHd1FNfyEYa9KLjWfoMY_v7ARTq3tdpVw" style="width: 300px"/>
# 

import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)


def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, )))
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(Activation('sigmoid'))    
    model.add(Dense(50))
    model.add(Activation('sigmoid'))    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


model1 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model2 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model3 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)


ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3)], voting = 'soft')


ensemble_clf.fit(X_train, y_train)


y_pred = ensemble_clf.predict(X_test)


print('Test accuracy:', accuracy_score(y_pred, y_test))


# Slight boost in the test accuracy from the outset **(0.2144 => 0.3045)**
# 

# ## Summary
# 
# |Model           | Naive Model | He normal  | Relu        | Adam        | Batchnorm  | Dropout   | Ensemble   |
# |----------------|-------------|------------|-------------|-------------|------------|-----------|------------|
# |Test Accuracy   | 0.2144      | 0.4105     | 0.9208      | 0.9248      | 0.9154     | 0.1135    | 0.3045     |
# 
# <br>
# It turns out that most methods improve the model training & test performance.
# Why don't try them out altogether?

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D, Input
from keras.models import Model
from keras.preprocessing import image


# ## 0. Basics
# - Input of image-format data is usually 4-D array in Tensorflow
# <br> **(num_instance, width, height, depth)** </br>
#     - **num_instance:** number of data instances. Usually designated as **None** to accomodate fluctuating data size
#     - **width:** width of an image
#     - **height:** height of an image
#     - **depth:** depth of an image. Color images are usually with depth = 3 (3 channels for RGB). Black/white images are usually with depth = 1 (only one channel)
#     
# <img src="http://xrds.acm.org/blog/wp-content/uploads/2016/06/Figure1.png" style="width: 400px"/>
# 

# - Loading image
#     - Images can be loaded using load_img() function
#     - Images can be converted to numpy array using img_to_array() function
# 

img = image.load_img('dog.jpg', target_size = (100, 100))


img


img = image.img_to_array(img)


print(img.shape)


# ## 1. Padding
# - Two types of padding options
#     - **'valid'**: no padding (drop right-most columns & bottom-most rows)
#     - **'same'**: padding size **p = [k/2]** when kernel size = **k**
# - Customized paddings can be given with ZeroPadding**n**D layer
# 

# when padding = 'valid'
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'valid'))


print(model.output_shape)


# when padding = 'same'
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))


print(model.output_shape)


# user-customized padding
input_layer = Input(shape = (10, 10, 3))
padding_layer = ZeroPadding2D(padding = (1,1))(input_layer)

model = Model(inputs = input_layer, outputs = padding_layer)


print(model.output_shape)


# ## 2. FIlter/kernels
# - Number of filters can be designated
# - Number of filters equals to the **depth of next layer**
# 

# when filter size = 10
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))


# you could see that the depth of output = 10
print(model.output_shape)


# when filter size = 20
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 20, kernel_size = (3,3), strides = (1,1), padding = 'same'))


# you could see that the depth of output = 20
print(model.output_shape)


# ## 3. Pooling
# - Usually, max pooling is applied for rectangular region
# - pooling size, padding type, and strides can be set similar to convolutional layer
# 

model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))


print(model.output_shape)


# when 'strides' parameter is not defined, strides are equal to 'pool_size'
model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))


print(model.output_shape)


model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (1,1), padding = 'valid'))


print(model.output_shape)


model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(AveragePooling2D(pool_size = (2,2), padding = 'valid'))


print(model.output_shape)


# globalmaxpooling performs maxpooling over whole channel with depth = 1
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(GlobalMaxPooling2D())


# as the number of filters = 10, 10 values are returned as result of globalmaxpooling2D
print(model.output_shape)


# ## 4. Flattening
# - To be connected to fully connected layer (dense layer), convolutional/pooling layer should be **"flattened"**
# - Resulting shape = **(Number of instances, width X height X depth)**
# 

model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))


print(model.output_shape)


model.add(Flatten())


print(model.output_shape)


# ## 5. Fully Connected (Dense)
# - After flattening layer, fully connected layer can be added
# - output shape (number of nodes) should be designated
# 

model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Flatten())
model.add(Dense(50))


print(model.output_shape)


# ## Using pretrained models 
# - Keras Applications provided deep learning models with pre-trained weights
# - Documentation: https://keras.io/applications/
# 

import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import preprocess_input, decode_predictions


# ### Load image
# - Images can be loaded using load_img function
# 

# img src = 'https://gfp-2a3tnpzj.stackpathdns.com/wp-content/uploads/2016/07/Dachshund-600x600.jpg'
img = load_img('dog.jpg')


img


# ### Import model
# - Currently, seven models are supported
#     - Xception
#     - VGG16
#     - VGG19
#     - ResNet50
#     - InceptionV3
#     - InceptionResNetV2
#     - MobileNet
# 

from keras.applications.resnet50 import ResNet50


model = ResNet50(weights='imagenet')


img = load_img('dog.jpg', target_size = (224, 224))    # image size can be calibrated with target_size parameter
img


img = img_to_array(img)
print(img.shape)


img = np.expand_dims(img, axis=0)
print(img.shape)


## prediction wo preprocessing
pred_class = model.predict(img)
# print(pred_class)


# print only top 10 predicted classes
n = 10
top_n = decode_predictions(pred_class, top=n)


for c in top_n[0]:
    print(c)


img = preprocess_input(img)    # preprocess image with preprocess_input function
print(img.shape)


## prediction with preprocessing
pred_class = model.predict(img)
# print(pred_class)


n = 10
top_n = decode_predictions(pred_class, top=n)


for c in top_n[0]:
    print(c)


# # Using GPUs
# - Training on GPUs (graphic cards) makes training neural networks much faster than running on CPUs
# - Keras supports training on GPUs with both Tensorflow & Theano backend
#     - docs: https://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu
# 
# 
# <br>
# <img src="https://blogs.nvidia.com/wp-content/uploads/2016/03/titanxfordeeplearning.png" style="width: 600px"/>
# 

# ## Installation and checkups 
# - First, download and install **CUDA & CuDNN** (assuming that you are using NVIDIA gpus)
#     - Note: Installing CuDNN will enable you to use CuDNNGRU & CuDNNLSTM layers, which is about x10 times faster than GRU & LSTM layers (for more information, refer to: [CuDNN layers](https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/tree/master/3.%20RNN/4-Advanced-RNN-3))
#           - doc: https://keras.io/layers/recurrent/#cudnngru
#           - doc: https://keras.io/layers/recurrent/#cudnnlstm
#     - url: https://developer.nvidia.com/cudnn
# - Then, install **tensorflow-gpu** (gpu-enabled version of Tensorflow) by typing below in cmd or terminal
#     - pip install tensorflow-gpu
# - Then check if your machine is utilizing GPU device
#     - In my case, I have one GPU device (whose name is "/device:GPU:0")
# 

import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# just checking if tensorflow is using GPU device
with tf.device('/device:GPU:0'):
    x = tf.constant([1.0, 2.0, 3.0], shape = [1,3], name = 'x')
    y = tf.constant([4.0, 5.0, 6.0], shape = [3,1], name = 'y')
    z = tf.matmul(x,y)
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
    print(sess.run(z))


# # Importing images
# - Import image from url and change into array
# 

import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from urllib.request import urlopen
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# url of an image
url = 'https://gfp-2a3tnpzj.stackpathdns.com/wp-content/uploads/2016/07/Dachshund-600x600.jpg'


image =  BytesIO(urlopen(url).read())


image = Image.open(image)
image


# images can be resized using resize() method
image = image.resize((100,100))
image


# images can be converted to arrays using img_to_array() method
image_arr = img_to_array(image)
print(image_arr.shape)


# by reshaping the image array, we could get array with rank =4 
image_arr = image_arr.reshape((1,) + image_arr.shape) 
print(image_arr.shape)    # first element in shape means that we have one image data instance


url1 = 'http://cdn2-www.dogtime.com/assets/uploads/2011/01/file_23020_dachshund-dog-breed.jpg'
url2 = 'http://lovedachshund.com/wp-content/uploads/2016/06/short-haired-dachshund.jpg'
url3 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTavNocbsgwxukPI9eO3jgNaxP_DupqzBqE2M1oQi_GzdMHIZ-'


urls = [url1, url2, url3]


for url in urls:
    image =  BytesIO(urlopen(url).read())
    image = Image.open(image).resize((100, 100), Image.ANTIALIAS)
    image = img_to_array(image)
    image = image.reshape((1,) + image.shape) 
    print(image.shape)
    image_arr = np.concatenate((image_arr, image), axis = 0)


image_arr.shape


# visualizing imported images
num = len(image_arr)
for i in range(num):
    ax = plt.subplot(1, num, i+1 )
    plt.imshow(image_arr[i])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()    


