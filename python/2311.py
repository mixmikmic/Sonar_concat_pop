# # Getting started in scikit-learn with the famous iris dataset
# *From the video series: [Introduction to machine learning with scikit-learn](https://github.com/justmarkham/scikit-learn-videos)*
# 

# ## Agenda
# 
# - What is the famous iris dataset, and how does it relate to machine learning?
# - How do we load the iris dataset into scikit-learn?
# - How do we describe a dataset using machine learning terminology?
# - What are scikit-learn's four key requirements for working with data?

# ## Introducing the iris dataset
# 

# ![Iris](images/03_iris.png)

# - 50 samples of 3 different species of iris (150 samples total)
# - Measurements: sepal length, sepal width, petal length, petal width
# 

from IPython.display import IFrame
IFrame('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', width=300, height=200)


# ## Machine learning on the iris dataset
# 
# - Framed as a **supervised learning** problem: Predict the species of an iris using the measurements
# - Famous dataset for machine learning because prediction is **easy**
# - Learn more about the iris dataset: [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Iris)
# 

# ## Loading the iris dataset into scikit-learn
# 

# import load_iris function from datasets module
from sklearn.datasets import load_iris


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)


# print the iris data
print(iris.data)


# ## Machine learning terminology
# 
# - Each row is an **observation** (also known as: sample, example, instance, record)
# - Each column is a **feature** (also known as: predictor, attribute, independent variable, input, regressor, covariate)
# 

# print the names of the four features
print(iris.feature_names)


# print integers representing the species of each observation
print(iris.target)


# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)


# - Each value we are predicting is the **response** (also known as: target, outcome, label, dependent variable)
# - **Classification** is supervised learning in which the response is categorical
# - **Regression** is supervised learning in which the response is ordered and continuous
# 

# ## Requirements for working with data in scikit-learn
# 
# 1. Features and response are **separate objects**
# 2. Features and response should be **numeric**
# 3. Features and response should be **NumPy arrays**
# 4. Features and response should have **specific shapes**
# 

# check the types of the features and response
print(type(iris.data))
print(type(iris.target))


# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)


# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)


# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# ## Resources
# 
# - scikit-learn documentation: [Dataset loading utilities](http://scikit-learn.org/stable/datasets/)
# - Jake VanderPlas: Fast Numerical Computing with NumPy ([slides](https://speakerdeck.com/jakevdp/losing-your-loops-fast-numerical-computing-with-numpy-pycon-2015), [video](https://www.youtube.com/watch?v=EEUXKG97YRw))
# - Scott Shell: [An Introduction to NumPy](http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf) (PDF)
# 

# ## Comments or Questions?
# 
# - Email: <kevin@dataschool.io>
# - Website: http://dataschool.io
# - Twitter: [@justmarkham](https://twitter.com/justmarkham)
# 

from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()


# # Efficiently searching for optimal tuning parameters
# *From the video series: [Introduction to machine learning with scikit-learn](https://github.com/justmarkham/scikit-learn-videos)*
# 

# ## Agenda
# 
# - How can K-fold cross-validation be used to search for an **optimal tuning parameter**?
# - How can this process be made **more efficient**?
# - How do you search for **multiple tuning parameters** at once?
# - What do you do with those tuning parameters before making **real predictions**?
# - How can the **computational expense** of this process be reduced?

# ## Review of K-fold cross-validation
# 

# Steps for cross-validation:
# 
# - Dataset is split into K "folds" of **equal size**
# - Each fold acts as the **testing set** 1 time, and acts as the **training set** K-1 times
# - **Average testing performance** is used as the estimate of out-of-sample performance
# 
# Benefits of cross-validation:
# 
# - More **reliable** estimate of out-of-sample performance than train/test split
# - Can be used for selecting **tuning parameters**, choosing between **models**, and selecting **features**
# 
# Drawbacks of cross-validation:
# 
# - Can be computationally **expensive**
# 

# ## Review of parameter tuning using `cross_val_score`
# 

# **Goal:** Select the best tuning parameters (aka "hyperparameters") for KNN on the iris dataset
# 

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target


# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)


# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# ## More efficient parameter tuning using `GridSearchCV`
# 

# Allows you to define a **grid of parameters** that will be **searched** using K-fold cross-validation
# 

from sklearn.grid_search import GridSearchCV


# define the parameter values that should be searched
k_range = list(range(1, 31))
print(k_range)


# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)


# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')


# - You can set **`n_jobs = -1`** to run computations in parallel (if supported by your computer and OS)
# 

# fit the grid with data
grid.fit(X, y)


# view the complete results (list of named tuples)
grid.grid_scores_


# examine the first tuple
print(grid.grid_scores_[0].parameters)
print(grid.grid_scores_[0].cv_validation_scores)
print(grid.grid_scores_[0].mean_validation_score)


# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)


# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# ## Searching multiple parameters simultaneously
# 

# - **Example:** tuning `max_depth` and `min_samples_leaf` for a `DecisionTreeClassifier`
# - Could tune parameters **independently**: change `max_depth` while leaving `min_samples_leaf` at its default value, and vice versa
# - But, best performance might be achieved when **neither parameter** is at its default value
# 

# define the parameter values that should be searched
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']


# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)


# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)


# view the complete results
grid.grid_scores_


# examine the best model
print(grid.best_score_)
print(grid.best_params_)


# ## Using the best parameters to make predictions
# 

# train your model using all data and the best known parameters
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)

# make a prediction on out-of-sample data
knn.predict([[3, 5, 4, 2]])


# shortcut: GridSearchCV automatically refits the best model using all of the data
grid.predict([[3, 5, 4, 2]])


# ## Reducing computational expense using `RandomizedSearchCV`
# 

# - Searching many different parameters at once may be computationally infeasible
# - `RandomizedSearchCV` searches a subset of the parameters, and you control the computational "budget"
# 

from sklearn.grid_search import RandomizedSearchCV


# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)


# - **Important:** Specify a continuous distribution (rather than a list of values) for any continous parameters
# 

# n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
rand.fit(X, y)
rand.grid_scores_


# examine the best model
print(rand.best_score_)
print(rand.best_params_)


# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)


# ## Resources
# 
# - scikit-learn documentation: [Grid search](http://scikit-learn.org/stable/modules/grid_search.html), [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html), [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html)
# - Timed example: [Comparing randomized search and grid search](http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html)
# - scikit-learn workshop by Andreas Mueller: [Video segment on randomized search](https://youtu.be/0wUF_Ov8b0A?t=17m38s) (3 minutes), [related notebook](https://github.com/amueller/pydata-nyc-advanced-sklearn/blob/master/Chapter%203%20-%20Randomized%20Hyper%20Parameter%20Search.ipynb)
# - Paper by Yoshua Bengio: [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
# 

# ## Comments or Questions?
# 
# - Email: <kevin@dataschool.io>
# - Website: http://dataschool.io
# - Twitter: [@justmarkham](https://twitter.com/justmarkham)
# 

from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()


