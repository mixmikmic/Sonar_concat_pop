# What is SKFlow
# 
# 
# Explain Tensorflow models
# Explain parameters, how they work
# 

import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)


import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
X = preprocessing.StandardScaler().fit_transform(boston.data)
regressor = skflow.TensorFlowLinearRegressor()
regressor.fit(X, boston.target)
score = metrics.mean_squared_error(regressor.predict(X), boston.target)
print ("MSE: %f" % score)


import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target, logdir='/Users/kendall/projects/python/tensorflow/intro-to-tensorflow/log/')
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)


import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()

def my_model(X, y):
    """This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability."""
    layers = skflow.ops.dnn(X, [10, 20, 10], keep_prob=0.5)
    return skflow.models.logistic_regression(layers, y)

classifier = skflow.TensorFlowEstimator(model_fn=my_model, n_classes=3)
classifier.fit(iris.data, iris.target, logdir='/Users/kendall/projects/python/tensorflow/intro-to-tensorflow/log/')
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)


classifier = skflow.TensorFlowLinearRegressor()
classifier.fit(X, y, logdir='.')


# Add skflow problem set here
# Use a new dataset?
# Use a different datasent from sklearn?

# Problem: Train a few models and come up with the most accurate model
# If you win the competition you can get some Google gear





