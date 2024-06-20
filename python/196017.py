# Image classifier with cifar10 dataset.
# 

import numpy as np
# random seed for reproducibility
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
#Now we will import some utilities
from keras.utils import np_utils
#Fixed dimension ordering issue
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train,y_train),(X_test, y_test)=cifar10.load_data()
#Preprocess imput data for Keras
# Reshape input data.
# reshape to be [samples][channels][width][height]
X_train=X_train.reshape(X_train.shape[0],3,32,32)
X_test=X_test.reshape(X_test.shape[0],3,32,32)
# to convert our data type to float32 and normalize our database
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
print(X_train.shape)
# Z-scoring or Gaussian Normalization
X_train=X_train - np.mean(X_train) / X_train.std()
X_test=X_test - np.mean(X_test) / X_test.std()

# convert 1-dim class arrays to 10 dim class metrices
#one hot encoding outputs
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]
print(num_classes)
#10
#Define a simple CNN model
print(X_train.shape)




model=Sequential()
model.add(Conv2D(32, (5,5), input_shape=(3,32,32), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))      # Dropout, one form of regularization
model.add(Flatten())
model.add(Dense(240,activation='elu'))
model.add(Dense(10, activation='softmax'))
print(model.output_shape)



model.compile(loss='binary_crossentropy', optimizer='adagrad')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)
# Final evaluation of the model
scores =model.evaluate(X_test, y_test, verbose=0)
print('CNN error: % .2f%%' % (scores))





# MLP on randomly generated data
# 

import warnings
warnings.simplefilter("ignore")


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np


#Generate the data using the random function.
# Generate dummy data
x_train = np.random.random((1000, 20))
# Y having 10 possible categories
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)



#Creating a sequential model.
#Create a model 
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# In the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.


model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#Compile the model


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#Using the ‘model.fit’ function to train the model.
# Fit the model
model.fit(x_train, y_train,epochs=20,batch_size=128)
#Evaluating the performance of the model using the ‘model.evaluate’ function.
# Evaluate the model
score = model.evaluate(x_test, y_test, batch_size=128)





# Fashion MNIST Data: Logistic Regression in Keras
# 

from __future__ import print_function
from keras.models import load_model
import keras
import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
batch_size = 128
num_classes = 10
epochs = 2


# We will be using the Fashion MNIST dataset. Store the data and the label in two different variables.
# 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


#Normalizing the dataset
#Gaussian Normalization of the dataset
x_train = (x_train-np.mean(x_train))/np.std(x_train)
x_test = (x_test-np.mean(x_test))/np.std(x_test)



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes) 



#Defining the model
model = Sequential()
model.add(Dense(256, activation='elu', input_shape=(784,)))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) 



# Saving the model in .h5 file(so that we can use it later directly with ‘model.load’ function) and printing the accuracy of the model in testset.
# #saving the model using the 'model.save' function
# 

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.save('my_model.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 





# Working with pre-trained Models
# 

from keras import applications
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
model = VGG16(weights='imagenet', include_top=True)
model.summary()
#predicting for any new image based on the pre-trained model
# Loading Image


import numpy as np
from keras.preprocessing import image
img = image.load_img('horse.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img=preprocess_input(img)
# Predict the output
preds = model.predict(img)
# decode the predictions
pred_class = decode_predictions(preds, top=3)[0][0]
print('Predicted Class: %s' %pred_class[1])
print('Confidance: %s'% pred_class[2])
#Predicted Class: hartebeest
#Confidance: 0.964784
#ResNet50 and InceptionV3 models can be easily utilized for prediction/classification of new images.
from keras.applications import ResNet50
model = ResNet50(weights='imagenet' , include_top=True)
model.summary()
# create the base pre-trained model
from keras.applications import InceptionV3
model = InceptionV3(weights='imagenet')
model.summary()





# #Logistic Regression using scikit learn and Keras
# 

import warnings
warnings.simplefilter("ignore")


from __future__ import print_function
from keras.models import load_model
import keras
from keras.utils import np_utils
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from sklearn.linear_model import LogisticRegressionCV
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import numpy as np


# ################################scikit Learn for  Logistic Regression################################
# 

iris = load_iris()
#Iris Dataset has five attributes out of which we will be using the first four attributes to predict the species, whose class is defined in the fifth attribute of the dataset.
X, y = iris.data[:, :4], iris.target
# Split both independent and dependent variables in half for cross-validation
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0)
#print(type(train_X),len(train_y),len(test_X),len(test_y))
lr = LogisticRegressionCV()
lr.fit(train_X, train_y)
pred_y = lr.predict(test_X)
print("Test fraction correct (LR-Accuracy) = {:.2f}".format(lr.score(test_X, test_y)))



# ########################################Keras Neural Network for Logistic Regression################################
# 

# Use ONE-HOT enconding for converting into categorical variable
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))



# Dividing data into train and test data
train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)


#Creating a model
model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))


# Compiling the model 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# Actual modelling
model.fit(train_X, train_y_ohe, verbose=0, batch_size=1, nb_epoch=100)

score, accuracy = model.evaluate(test_X, test_y_ohe, batch_size=16, verbose=0)

print("\n Test fraction correct (LR-Accuracy) logistic regression = {:.2f}".format(lr.score(test_X, test_y))) # Accuracy is 0.83 
print("Test fraction correct (NN-Accuracy) keras  = {:.2f}".format(accuracy)) # Accuracy is 0.99





# Time Series Forcasting with LSTM model
# 

import warnings
warnings.simplefilter("ignore")


#importing the necessary packages
import pandas
import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np


dataset = pandas.read_csv('sp500.csv', usecols=[0], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))



# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


print(dataset)


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


trainX.shape


print(trainX.shape)



# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=10, verbose=2)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()





