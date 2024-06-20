# # Convolutional Neural Network for MNIST dataset classification task.
# References:
#     Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
#     learning applied to document recognition." Proceedings of the IEEE,
#     86(11):2278-2324, November 1998.
# Links:
#     [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
# 

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist


# Data loading and preprocessing
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])


# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)

network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')


# Training
model = tflearn.DNN(network, tensorboard_verbose=0)


model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')








# 
# # Linear Regression Example
# 
# This example uses the only the first feature of the `diabetes` dataset, in
# order to illustrate a two-dimensional plot of this regression technique. The
# straight line can be seen in the plot, showing how linear regression attempts
# to draw a straight line that will best minimize the residual sum of squares
# between the observed responses in the dataset, and the responses predicted by
# the linear approximation.
# 
# The coefficients, the residual sum of squares and the variance score are also
# calculated.
# 
# 

get_ipython().magic('matplotlib inline')

print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


# Load the diabetes dataset
diabetes = datasets.load_diabetes()


diabetes


diabetes.data[1]


diabetes.target[1]


diabetes.data.shape


diabetes.target.shape


diabetes.target.shape


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]


# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)


# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))


predictions = regr.predict(diabetes_X_test)



predictions


# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_train,  color='black')
plt.plot(diabetes_X_test, predictions, color='blue',
         linewidth=3)

plt.show()





import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()


iris


X = iris.data
y = iris.target


X.shape


y.shape


# ## Split train and test datasets
# 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


X_train.shape


X_test.shape


y_test


# # Train and test a model
# 

estimator = DecisionTreeClassifier(max_depth=2)


estimator


estimator.fit(X_train, y_train)


y_predict = estimator.predict(X_test)


y_predict


y_test


correct_labels = sum(y_predict == y_test)
correct_labels


len(y_predict)


print("Accuracy: %f" % (correct_labels/len(y_predict)))


X





from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file="tree.dot", class_names=iris.target_names,
                feature_names=iris.feature_names, impurity=False, filled=True)


import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# # Improve the Model
# 

estimator = DecisionTreeClassifier(max_depth=5)


estimator


estimator.fit(X_train, y_train)


y_predict = estimator.predict(X_test)


y_predict


y_test


correct_labels = sum(y_predict == y_test)
correct_labels


len(y_predict)


print("Accuracy: %f" % (correct_labels/len(y_predict)))


X





from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file="tree.dot", class_names=iris.target_names,
                feature_names=iris.feature_names, impurity=False, filled=True)


import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# # CIFAR
# Convolutional network applied to CIFAR-10 dataset classification task.
# References:
#     Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
# Links:
#     [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
# 

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.datasets import cifar10


# Data loading and preprocessing
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()


# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)


# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)


# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)


model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=48, run_id='cifar10_cnn')


model.save('models/cifar/cifar.tflearn')


# # AlexNet.
# Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
# References:
#     - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
#     Classification with Deep Convolutional Neural Networks. NIPS, 2012.
#     - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
# Links:
#     - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
#     - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
# 

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17


# Load dataset
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))


# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])

network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)

network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)

network = fully_connected(network, 17, activation='softmax')

network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)


# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)


model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')





# ## TFLearn - Quick Start
# 
# In this tutorial, you will learn to use TFLearn and TensorFlow to estimate the surviving chance of Titanic passengers using their personal information (such as gender, age, etc...). To tackle this classic machine learning task, we are going to build a deep neural network classifier.
# 

# ### Titanic Dataset
# On April 15, 1912, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this tutorial, we carry an analysis to find out who these people are.
# 

from __future__ import print_function

import numpy as np
import tflearn

from tflearn.datasets import titanic
from tflearn.data_utils import load_csv


# ### Loading the data
# 

# Download the Titanic dataset
titanic.download_dataset('titanic_dataset.csv')


# Load CSV file, indicate that the first column represents labels
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# ### Preprocessing
# Data are given 'as it' and need some preprocessing to be ready to be used in our deep neural network classifier.
# 
# First, we will discard the fields that are not likely to help in our analysis. For example, we make the assumption that 'name' field will not be very useful in our task, because we estimate that a passenger name and his chance of surviving are not correlated. With such thinking, we discard 'name' and 'ticket' fields.
# 
# Then, we need to convert all our data to numerical values, because a neural network model can only perform operations over numbers. However, our dataset contains some non numerical values, such as 'name' or 'sex'. Because 'name' is discarded, we just need to handle 'sex' field. In this simple case, we will just assign '0' to males and '1' to females.
# 

# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)


# ### Build a Deep Neural Network
# 
# We are building a 3-layers neural network using TFLearn. We need to specify the shape of our input data. In our case, each sample has a total of 6 features and we will process samples per batch to save memory, so our data input shape is [None, 6] ('None' stands for an unknown dimension, so we can change the total number of samples that are processed in a batch).
# 

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)


# ### Training
# 
# TFLearn provides a model wrapper 'DNN' that can automatically performs a neural network classifier tasks, such as training, prediction, save/restore, etc... We will run it for 10 epochs (the network will see all data 10 times) with a batch size of 16.
# 

# Define model
model = tflearn.DNN(net)


# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)


# ### Try the Model
# 
# It is time to try out our model. For fun, let's take Titanic movie protagonists (DiCaprio and Winslet) and calculate their chance of surviving (class 1).
# 

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]


# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)


# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])


print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])


# Impressive! Our model accurately predicted the outcome of the movie. Odds were against DiCaprio, but Winslet had a high chance of surviving.
# 
# More generally, it can bee seen through this study that women and children passengers from first class have the highest chance of surviving, while third class male passengers have the lowest.
# 




# # K-means Clustering
# 

get_ipython().magic('matplotlib inline')


from sklearn import datasets, cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# note: I deliberately chose a random seed that ends up 
# labeling the clusters with the same numbering convention 
# as the original y values 
np.random.seed(2)


# load data
iris = datasets.load_iris()


X_iris = iris.data
y_iris = iris.target


# do the clustering
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris) 
labels = k_means.labels_


# check how many of the samples were correctly labeled
correct_labels = sum(y_iris == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y_iris.size))


# plot the clusters in color
fig = plt.figure(1, figsize=(8, 8))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)
plt.cla()

ax.scatter(X_iris[:, 3], X_iris[:, 0], X_iris[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')

plt.show()





# # Convolutional Neural Network
# ## recognizing MNIST digits using Keras
# 

import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


# get train and test datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# preprocess input data

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


X_test.shape


# preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


Y_train.shape


# define the topology
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# 9train the model
model.fit(X_train, Y_train, 
          batch_size=64, epochs=3, verbose=1)


# evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)


score





