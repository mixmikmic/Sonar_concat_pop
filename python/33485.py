# # C15: Ploting Metrics of Training Process
# 
# Summary:
# 
#     1. history = model.fit() fit function will return a dict containing 'acc', 'loss', 'val_acc', 'val_loss' 
#     2. You can plot get training history data by history.history['acc']
# 

# ## 1.Preparation
# 

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np

# fix random seed
seed = 7
np.random.seed(seed)

# load and split dataset
dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)

# define create model
def create_nn():
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ## 2. Plot Training History Data
# 

model = create_nn()

# get history data
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=150, batch_size=10, verbose=0)
print(history.history.keys())


# in MacOs system, the backend of matplotlib shoud be change to TkAgg
# import matplotlib as mpl
# mpl.use('TkAgg')

import matplotlib.pyplot as plt
# set to plot at ipython notebook
get_ipython().magic('matplotlib inline')

plt.subplot(111)
plt.plot(history.history['val_acc'])
plt.plot(history.history['val_loss'])
plt.ylabel('val_acc and val_loss')
plt.xlabel('epoch')
plt.title('model train history')
plt.show()





# # Sequence Classification with LSTM
# 

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# fix random seed
seed = 7
np.random.seed(seed)


# load dataset, only keep the top 5000 words, zero the rest
top_words = 5000
(X_train, Y_train), (X_val, Y_val) = imdb.load_data(nb_words=top_words)

# pad input sequence
maxlen = 500
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_val = sequence.pad_sequences(X_val, maxlen=maxlen)


Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_val = Y_val.reshape(Y_train.shape[0], 1)


# define and build a model
def create_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=maxlen, dropout=0.2))
    model.add(LSTM(64, stateful=False, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

lstm = create_model()


# train model
lstm.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, nb_epoch=3, verbose=1)


# evaluate mode
scores = lstm.evaluate(X_val, Y_val)
print("Acc: %.2f%%"%(scores[1]*100))








# # C7: Develop Your First Neural Network With Keras
# 

# ## 7.1 Overview
# 
# Six steps:
# 
# 1. Load Data.
# 2. Define Model.
# 3. Compile Model.
# 4. Fit Model.
# 5. Evaluate Model.
# 6. Tie It All Together.
# 

# ## 7.2 Get Data Set
# 
# In this tutorial we are going to use the Pima Indians onset of diabetes dataset.
# 
# The introduction of this data set: [pima-indians-diabetes.names](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names)
# 
# The data set file: [pima-indians-diabetes.data](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)
# 
# You can get the data by the fellowing step:
# 
# ```
# $ cd ./data_set/
# $ ./get_pima_indians_diabetes_data.sh
# ```
# 
# pima-indians-diabetes.data will be downloaded at data_set/
# 

# ## 7.3 Load Data
# 
# ### Initialize random seed
# 
# It is a good idea to initialize the random number generator with a fixed seed value. This is so that you can run the same code again and again and get the same result. 
# 

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# ### load data
# 
# Load data by numpy
# 

dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]
print 'X', X.shape, X.dtype
print 'Y', Y.shape, Y.dtype


# ## 7.4 Define Model
# 
# Create a model by keras.models.Sequential and add the layers we designed:
# 
#     The first hidden layer has 12 neurons and expects 8 input variables, and a relu activation.
#     The second hidden layer has 8 neurons, and a relu activation.
#     Finally the output layer has 1 neuron to predict the class, and a sigmoid activation.
# 

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# ## 7.5 Compile Model
# 
# Compiling the model uses the efficient numercial libraries (Tensorflow or Theano). In this step, we shoule specify some hyperperemeters for training process:
# 
#     loss function: binary_crossentropy
#     optimizer: adam
#     metrics" accuracy
# 


# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ## 7.6 Fit Model
# 
# Fit moel means training model, the peremeters in this step are:
# 
#     nb_epoch: 150
#     batch_size: 10
# 
# nb_epoch means the number of epoch, which fix the number of iterations. batch_size means the batch size in the method "mini-batch gradient descent"
# 
# The training process is runing on your CPU or GPU.
# 

import time
start_time = time.time()
# fit model
model.fit(X, Y, nb_epoch=150, batch_size=10)
end_time = time.time()
print "Fit time Cost %s s"%(end_time - start_time)


# ## 7.7 Evaluate Model
# 
# In this part, we can calculate the accuracy fo this model on training dataset 
# 

# evaluate the model
scores = model.evaluate(X, Y)
print "Training Dataset %s: %.2f"%(model.metrics_names[0], scores[1])
print "Training Dataset %s: %.2f%%"%(model.metrics_names[1], scores[1]*100)


# ## 7.8 Switch to GPU model
# 
# 
# ### 7.8.1 For MacOS with Nvidia GPU
# 
# [mac osx/linux下如何将keras运行在GPU上](http://blog.csdn.net/u014205968/article/details/50166651)
# 
# Note: there are some question when install CUDA if your xcode version is 8.0 above, see [here](http://blog.cycleuser.org/use-cuda-80-with-macos-sierra-1012.html)
# 
# 使用下面这个脚本来验证是否启动GPU:
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

from theano import function, config, shared, sandbox  
import theano.tensor as T  
import numpy  
import time  
  
vlen = 10 * 30 * 768  # 10 x #cores x # threads per core  
iters = 1000  
  
rng = numpy.random.RandomState(22)  
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))  
f = function([], T.exp(x))  
print(f.maker.fgraph.toposort())  
t0 = time.time()  
for i in xrange(iters):  
    r = f()  
t1 = time.time()  
print("Looping %d times took %f seconds" % (iters, t1 - t0))  
print("Result is %s" % (r,))  
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):  
    print('Used the cpu')  
else:  
    print('Used the gpu')





# ## Chapter 22: Project: Predict Sentiment From Movie Review
# 

# ## 1. Preparation
# 

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding

# fix random seed
seed = 7
np.random.seed(seed)


# ## 2. IMDB Dataset
# 
# ### What is IMDB Dataset
# 
# The Large Movie Review Dataset (often referred to as the IMDB dataset) contains 25,000 highly-polar movie reviews (good or bad) for training and the same amount again for testing. The problem is to determine whether a given moving review has a positive or negative sentiment.
# 
# ### Download Dataset
# 
# Keras offer a API to load IMDB dataset. When the dataset is stored at ~/.keras/datasets/imdb.pkl as a 32M file.  
# 

from keras.datasets import imdb

(X_train, Y_train), (X_val, Y_val) = imdb.load_data(nb_words=5000)
print X_train.shape, X_train.dtype


print type(X_train[0])


# ### Preprocess
# 
# Bound the length of word sequence to 500, truncating longer reviews and zero-padding shorter reviews.
# 

from keras.preprocessing import sequence

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)

print len(X_train[0])


# ## 3. Build a Simple Model
# 
# We will use an Embedding layer as the input layer, setting the vocabulary to 5,000, the word vector size to 32 dimensions and the input length to 500. The output of this first layer will be a 32x500 sized matrix.
# 

# define a simple model
def create_simple_model():
    model = Sequential()
    model.add(Embedding(5000, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

simple_model = create_simple_model()


simple_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=10, batch_size=30, verbose=1)


# ## 4. Build 1-D Conv Layer Model
# 

# define conv model
def create_conv_model():
    model = Sequential()
    model.add(Embedding(5000, 32, input_length=max_words))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

conv_model = create_conv_model()


conv_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=10, batch_size=30, verbose=1)


























# # C8: Evaluate The Performance of Deep Learning Models
# 

# ## 8.1 Data Split
# 
# ### 8.1.1 Split automatically when fit model
# 
# 在fit model的时候添加validation_split参数来设定validation set占比，然后训练的时候就会在打印出验证集的验证结果
# 
# ```
# model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)
# ```
# example:
# 

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# set random seed
seed = 7
np.random.seed(seed)

# load data
dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)


# ### 8.1.2 Split manually
# 
# 在训练之前手动将数据集分成训练集和验证集，这里推荐sklearn.model_selection里的 train_test_split()函数
# 
# ```
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, random_state=seed)
# ```
# 
# 然后在fit model里的时候设定validation_data=(X_val, Y_val)
# 
# ```
# model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=150, batch_size=10)
# ```
# 
# 例子：
# 

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# set random seed
seed = 7
np.random.seed(seed)

# load data
dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

# split data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=150, batch_size=10)

# evaluate model
scores = model.evaluate(X_val, Y_val, verbose=0)
print model.metrics_names
print 'val loss:', scores[0], 'val acc:', scores[0]


# ### 8.3 Manual k-Fold Cross Validation
# 
# Use sklearn.model_selection.StratifiedKFold Class to split the dataset into 10 folds
# 
# ```
# from sklearn.model_selection import StratifiedKFold
# 
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# for train_idx, val_idx in kfold.split(X, Y)
#     X_train = X[train_idx]
#     Y_train = Y[train_idx]
# ```
# 

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train_idx, val_idx in kfold.split(X, Y):
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=150, batch_size=10, verbose=0)
    scores = model.evaluate(X_val, Y_val, verbose=0)
    
    print("%s: %.2f%%"%(model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)

print("%.2f%% (+/- %.2f%%)"%(np.mean(cvscores), np.std(cvscores)))




























































































# # C17: Lift Performance With Learning Rate Schedules
# 
# Summary:
# 
#     1. learning_rate = initial_learning_rate * drop_rate(epoch), drop_rate() is a function return a 0~1 value.
#     2. Define a drop_rate(epoch) function, and create a LearningRateScheduler callback instance
#     3. Pass the LearningRateScheduler callback instance to the model.fit()
# 
# You can changle the learning_rate at each epoch by multiply the initial learning_rate with a 0~1 value.
# 

# ## 1. Preparation
# 

import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

# fix random seed
seed = 7
np.random.seed(7)

# loaddata
dataframe = pd.read_csv("./data_set/sonar.data", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
encoder = LabelEncoder()
encoder.fit(Y)
Y_enc = encoder.transform(Y)

# define a model
def create_model():
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
    model.add(Dense(30, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# ## 2. Use Learning Rate Schedules
# 

from keras.callbacks import LearningRateScheduler

# define a drop_rate function
def drop_rate(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


model = create_model()

# create a LearningRateScheduler instance
lrate = LearningRateScheduler(drop_rate)
model.fit(X, Y_enc, validation_split=0.2, nb_epoch=150, batch_size=10, callbacks=[lrate], verbose=2)





# # C21: Project: Image Classification With CNN
# 

# For Summary
# 

# ## 0. Preparation
# 

import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# ## 1. Load CIFAR-10 Dataset
# 
# ### What is CIFAT-10 Dataset
# 
# The CIFAR-10 dataset consists of 60,000 photos divided into 10 classes (hence the name CIFAR-10)1. Classes include common objects such as airplanes, automobiles, birds, cats and so on. The dataset is split in a standard way, where 50,000 images are used for training a model and the remaining 10,000 for evaluating its performance. The photos are in color with red, green and blue channels, but are small measuring 32x32 pixel squares.
# 
# State-of-the-art results can be achieved using very large convolutional neural networks. You can learn about state-of-the-art results on CIFAR-10 on Rodrigo Benenson’s webpage2. Model performance is reported in classification accuracy, with very good performance above 90% with human performance on the problem at 94% and state-of-the-art results at 96% at the time of writing.
# 
# ### Download CIFAR-10 By Keras
# 
# Keras has the facility to automatically download CIFAR-10 dataset, and store in the ~/.keras/datasets/ directory when using the cifar10.load_data() function at the first time. The dataset is about 164Mb.
# 

from keras.datasets import cifar10

# load dataset
(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()

print X_train.shape, X_train.dtype
print Y_train.shape, Y_train.dtype


# Show some images in CIFAR-10
# 

for i in range(0, 9):
    plt.subplot(331 + i)
    img = np.rollaxis(X_train[i], 0, 3)  # change axis ordering to [height][width][chanel]
    plt.imshow(img)
plt.show()


# Preprocess the dataset
# 

# normalize inputs from 0-255 to 0-1
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# one hot vector
Y_train = np_utils.to_categorical(Y_train.reshape(Y_train.shape[0],))
Y_val = np_utils.to_categorical(Y_val.reshape(Y_val.shape[0], ))
num_classes = Y_val.shape[1]


# ## 2. Build CNN
# 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import backend
backend.set_image_dim_ordering('th')


# fix random seed
seed = 7
np.random.seed(seed)


# definea a CNN model
def create_cnn():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same',
                            activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# build the model
cnn = create_cnn()


# fit model
cnn.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=1, batch_size=32, verbose=1)


# evaluate model
scores = cnn.evaluate(X_val, Y_val, verbose=0)
print('Val_Acc: %.2f%%'%(scores[1]*100))











# # Fine Tune VGG-16 for Cifar-10
# 
# ### Download VGG16
# 
# The model weights will be saved at ~/.keras/models/vgg16_weights_th_dim_ordering_th_kernels.h5, which is downloaded from [here](https://github.com/fchollet/deep-learning-models/releases/), and you can see the source of vgg16 at this site.
# 

from keras.applications.vgg16 import VGG16

# arguments see https://keras.io/applications/#vgg16
model_vgg16 = VGG16(include_top=True, weights='imagenet')


print(model_vgg16.summary())


print model_vgg16.layers[1].get_weights()








