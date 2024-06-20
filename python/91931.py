import numpy as np
from sklearn import datasets


# ## Load Data
# 
# Adapted from [sklearn docs](http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)
# 

iris = datasets.load_iris()


iris_X = iris.data
iris_y = iris.target


iris_X[:5]


iris_y[:5]


np.unique(iris_y)


# ## Split iris data into train and test
# 

np.random.seed(42)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]


# ## Fit nearest neighbor classifer
# 

from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier()


knn.fit(iris_X_train, iris_y_train) 


# check prediction
knn.predict(iris_X_test)


iris_y_test


# ## Pickle model to use in REST API
# 
# Adapted from [sklearn docs](http://scikit-learn.org/stable/modules/model_persistence.html)
# 

from sklearn.externals import joblib


joblib.dump(knn, 'iris_knn_model.pkl') 


# ## Load pickled model and use it to predict
# 

knn_from_pkl = joblib.load('iris_knn_model.pkl')


knn_from_pkl


# Get 1 test case
test_case = iris_X_test[:1]


# columns correspond to [Sepal Length, Sepal Width, Petal Length and Petal Width]
test_case


test_target = iris_y_test[:1]


test_target


knn_from_pkl.predict(test_case)


type(test_case)


