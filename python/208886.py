import sklearn.datasets
import re
from sklearn.cross_validation import train_test_split
import numpy as np


def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)
print (trainset.target_names)
print (len(trainset.data))
print (len(trainset.target))


vocabulary = list(set(' '.join(trainset.data).split()))
len(vocabulary)


# calculate IDF
idf = {}
for i in vocabulary:
    idf[i] = 0
    for k in trainset.data:
        if i in k.split():
            idf[i] += 1
    idf[i] = np.log(idf[i] / len(trainset.data))

# calculate TF
X = np.zeros((len(trainset.data),len(vocabulary)))
for no, i in enumerate(trainset.data):
    for text in i.split():
        X[no, vocabulary.index(text)] += 1
    for text in i.split():
        # calculate TF * IDF
        X[no, vocabulary.index(text)] = X[no, vocabulary.index(text)] * idf[text]


train_X, test_X, train_Y, test_Y = train_test_split(X, trainset.target, test_size = 0.2)


class GaussianNB:
    def __init__(self, epsilon):
        self.EPSILON = epsilon
        pass

    def fit(self, X, y):
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                    for i in separated])

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / ((2 * std**2)+self.EPSILON)))
        return np.log((exponent / ((np.sqrt(2 * np.pi) * std)+self.EPSILON)))

    def predict_log_proba(self, X):
        return [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
                for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)


gaussian_bayes = GaussianNB(1e-8)
gaussian_bayes.fit(train_X, train_Y)


# ## accuracy training
# 

gaussian_bayes.score(train_X, train_Y)


# ## accuracy testing
# 

gaussian_bayes.score(test_X, test_Y)


# What is atrous convolution? atrous is a french word, means hole.
# 
# ![alt text](http://liangchiehchen.com/fig/deeplab_aspp.jpg)

import numpy as np


x = np.zeros((3))
rate = 2
x


atrous = np.ones(np.array(x.shape) + rate)
atrous


for i in range(0, atrous.shape[0], rate //2+1):
    atrous[i] = atrous[i] * x[int(i/rate/2)+1]
atrous


x = np.random.rand(1,7,3)
kernel = np.random.rand(3,3,7)
filter_size = kernel.shape[0]
stride = 2
rate = 2


def padding(x, filter_size, pad='SAME'):
    if pad == 'SAME':
        pad_h_min = int(np.floor((filter_size - 1)/2))
        pad_h_max = int(np.ceil((filter_size - 1)/2))
        pad_h = (pad_h_min, pad_h_max)
        return np.pad(x, ((0, 0), pad_h, (0, 0)), mode='constant')
    else:
        return x
    
def get_shape(x):
    output_height = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
    return int(output_height)


x_padded = padding(x, filter_size)
h = get_shape(x_padded)
out_atrous = np.zeros((1, h, kernel.shape[2]))
out_atrous.shape


def atrous(x, w):
    for i in range(0, x.shape[0], rate //2+1):
        x[i,:] = x[i,:] * w[int(i/rate/2)+1,:]
    return x

def conv(x, w, out):
    for k in range(x.shape[0]):
        for z in range(w.shape[2]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                atroused = atrous(x[k, _h * stride:_h * stride + filter_size + rate, :], w[:, :, z])
                out[k, _h, z] = np.sum(atroused)
    return out


out_atrous = conv(x_padded, kernel, out_atrous)
out_atrous.shape


def deatrous_w(x, w, de):
    for i in range(0, x.shape[0], rate //2+1):
        w[int(i/rate/2)+1,:] = np.sum(x[i,:] * de[i,:])
    return w

def deconv_w(x, w, de):
    for k in range(x.shape[0]):
        for z in range(w.shape[2]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                weighted = deatrous_w(x[k, _h * stride:_h * stride + filter_size + rate, :], w[:, :, z],
                                         de[k, _h * stride:_h * stride + filter_size + rate, :])
                w[:, :, z] = weighted
    return w

def deconv_x(x, w, de):
    for k in range(x.shape[0]):
        for z in range(x.shape[2]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                atroused = atrous(de[k, _h * stride:_h * stride + filter_size + rate, :], w[:, z, :])
                x[k, _h, z] = np.sum(atroused)
    return x


dkernel = np.zeros(kernel.shape)
deconv_w(out_atrous, dkernel, out_atrous).shape


dx = np.zeros(x.shape)
deconv_x(dx, kernel, out_atrous).shape





import numpy as np
import time
from evolution_strategy import *
from function import *
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


df = pd.read_csv('Iris.csv')
df.head()


X = PCA(n_components=2).fit_transform(MinMaxScaler().fit_transform(df.iloc[:, 1:-1]))
Y = LabelEncoder().fit_transform(df.iloc[:, -1])
one_hot = np.zeros((Y.shape[0], 3))
for i in range(Y.shape[0]):
    one_hot[i, Y[i]] = 1.0
    
train_X, test_X, train_Y, test_Y, train_label, test_label = train_test_split(X,one_hot,Y, test_size = 0.2)


X.shape


size_population = 50
sigma = 0.1
learning_rate = 0.001
epoch = 500

'''
class Deep_Evolution_Strategy:
    
    def __init__(self, weights, inputs, solutions, reward_function, population_size, sigma, learning_rate):
    
weights = array of weights, no safe checking
inputs = our input matrix
solutions = our Y matrix
reward_function = cost function, can check function.py

Check example below on how to initialize the model and train any dataset

len(activations) == len(weights)

def train(self, epoch = 100, print_every = 5, activation_function = None):
'''

weights = [np.random.randn(X.shape[1]),
           np.random.randn(X.shape[1],20),
           np.random.randn(20,one_hot.shape[1])]
activations = [sigmoid, sigmoid, softmax]
deep_evolution = Deep_Evolution_Strategy(weights, train_X, train_Y, cross_entropy, size_population, sigma, learning_rate)
deep_evolution.train(epoch=1000,print_every = 50, activation_function = activations)


# ### Accuracy training
# 

predicted= np.argmax(deep_evolution.predict(deep_evolution.get_weight(), train_X, activation_function = activations),axis=1)
print(metrics.classification_report(predicted, np.argmax(train_Y, axis=1), target_names = ['flower 1', 'flower 2', 'flower 3']))


# ## Accuracy testing
# 

predicted= np.argmax(deep_evolution.predict(deep_evolution.get_weight(), test_X, activation_function = activations),axis=1)
print(metrics.classification_report(predicted, np.argmax(test_Y, axis=1), target_names = ['flower 1', 'flower 2', 'flower 3']))


accuracy_test = np.mean(predicted == np.argmax(test_Y, axis=1))



plt.figure(figsize=(15,10))
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.argmax(deep_evolution.predict(deep_evolution.get_weight(), np.c_[xx.ravel(), yy.ravel()], activation_function = activations),axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title('decision boundary, accuracy validation: %f'%(accuracy_test))
plt.show()





# What is atrous convolution? atrous is a french word, means hole.
# 
# ![alt text](http://liangchiehchen.com/fig/deeplab_aspp.jpg)

import numpy as np


x = np.zeros((3,3))
rate = 2
x


atrous = np.ones(np.array(x.shape) + rate)
atrous


for i in range(0, atrous.shape[0], rate //2+1):
    for k in range(0, atrous.shape[1], rate // 2+1):
        atrous[i,k] = atrous[i,k] * x[int(i/rate/2)+1,int(k/rate/2)+1]
atrous


x = np.random.rand(1,7,7,3)
kernel = np.random.rand(3,3,3,7)
filter_size = kernel.shape[0]
stride = 2
rate = 2


def padding(x, filter_size, pad='SAME'):
    if pad == 'SAME':
        pad_h_min = int(np.floor((filter_size - 1)/2))
        pad_h_max = int(np.ceil((filter_size - 1)/2))
        pad_w_min = int(np.floor((filter_size - 1)/2))
        pad_w_max = int(np.ceil((filter_size - 1)/2))
        pad_h, pad_w = (pad_h_min, pad_h_max), (pad_w_min, pad_w_max)
        return np.pad(x, ((0, 0), pad_h, pad_w, (0, 0)), mode='constant')
    else:
        return x
    
def get_shape(x):
    output_height = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
    output_width = int(np.ceil((x.shape[2] - rate * (filter_size-1)) / stride) + 1)
    return int(output_height), int(output_width)


x_padded = padding(x, filter_size)
h, w = get_shape(x_padded)
out_atrous = np.zeros((1, h, w, kernel.shape[3]))
out_atrous.shape


def atrous(x, w):
    for i in range(0, x.shape[0], rate //2+1):
        for k in range(0, x.shape[1], rate // 2+1):
            x[i,k,:] = x[i,k,:] * w[int(i/rate/2)+1,int(k/rate/2)+1,:]
    return x

def conv(x, w, out):
    for k in range(x.shape[0]):
        for z in range(w.shape[3]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                w_range = int(np.ceil((x.shape[2] - rate * (filter_size-1)) / stride) + 1)
                for _w in range(w_range):
                    atroused = atrous(x[k, 
                                        _h * stride:_h * stride + filter_size + rate, 
                                        _w * stride:_w * stride + filter_size + rate, :],
                                     w[:, :, :, z])
                    out[k, _h, _w, z] = np.sum(atroused)
    return out


out_atrous = conv(x_padded, kernel, out_atrous)


def deatrous_w(x, w, de):
    for i in range(0, x.shape[0], rate //2+1):
        for k in range(0, x.shape[1], rate // 2+1):
            w[int(i/rate/2)+1,int(k/rate/2)+1,:] = np.sum(x[i,k,:] * de[i,k,:])
    return w

def deconv_w(x, w, de):
    for k in range(x.shape[0]):
        for z in range(w.shape[3]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                w_range = int(np.ceil((x.shape[2] - rate * (filter_size-1)) / stride) + 1)
                for _w in range(w_range):
                    weighted = deatrous_w(x[k, 
                                            _h * stride:_h * stride + filter_size + rate, 
                                            _w * stride:_w * stride + filter_size + rate, :],
                                            w[:, :, :, z],
                                         de[k, 
                                            _h * stride:_h * stride + filter_size + rate, 
                                            _w * stride:_w * stride + filter_size + rate, :])
                    w[:, :, :, z] = weighted
    return w

def deconv_x(x, w, de):
    for k in range(x.shape[0]):
        for z in range(x.shape[3]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                w_range = int(np.ceil((x.shape[2] - rate * (filter_size-1)) / stride) + 1)
                for _w in range(w_range):
                    atroused = atrous(de[k, 
                                        _h * stride:_h * stride + filter_size + rate, 
                                        _w * stride:_w * stride + filter_size + rate, :], w[:, :, z, :])
                    x[k, _h, _w, z] = np.sum(atroused)
    return x


dkernel = np.zeros(kernel.shape)
deconv_w(out_atrous, dkernel, out_atrous).shape


dx = np.zeros(x.shape)
deconv_x(dx, kernel, out_atrous).shape





import sklearn.datasets
import re
from sklearn.cross_validation import train_test_split
import numpy as np


def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)
print (trainset.target_names)
print (len(trainset.data))
print (len(trainset.target))


vocabulary = list(set(' '.join(trainset.data).split()))
len(vocabulary)


# calculate IDF
idf = {}
for i in vocabulary:
    idf[i] = 0
    for k in trainset.data:
        if i in k.split():
            idf[i] += 1
    idf[i] = np.log(idf[i] / len(trainset.data))

# calculate TF
X = np.zeros((len(trainset.data),len(vocabulary)))
for no, i in enumerate(trainset.data):
    for text in i.split():
        X[no, vocabulary.index(text)] += 1
    for text in i.split():
        # calculate TF * IDF
        X[no, vocabulary.index(text)] = X[no, vocabulary.index(text)] * idf[text]


train_X, test_X, train_Y, test_Y = train_test_split(X, trainset.target, test_size = 0.2)


class MultinomialNB:
    def __init__(self, epsilon):
        self.EPSILON = epsilon

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample + self.EPSILON) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated])
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T + self.EPSILON)

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)


multinomial_bayes = MultinomialNB(1e-8)
multinomial_bayes.fit(train_X, train_Y)


# ## accuracy training
# 

np.mean(train_Y == multinomial_bayes.predict(train_X))


# ## accuracy testing
# 

np.mean(test_Y == multinomial_bayes.predict(test_X))


