# ## Search for best parameters and create a pipeline
# 

# ### Easy reading...create and use a pipeline
# 

# > <b>Pipelining</b> (as an aside to this section)
# * `Pipeline(steps=[...])` - where steps can be a list of processes through which to put data or a dictionary which includes the parameters for each step as values
# * For example, here we do a transformation (SelectKBest) and a classification (SVC) all at once in a pipeline we set up.
# 
# See a full example [here](http://scikit-learn.org/stable/auto_examples/feature_stacker.html)
# 
# Note:  If you wish to perform <b>multiple transformations</b> in your pipeline try [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion)
# 

# Imports for python 2/3 compatibility

from __future__ import absolute_import, division, print_function, unicode_literals

# For python 2, comment these out:
# from builtins import range


from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# a feature selection instance
selection = SelectKBest(chi2, k = 2)

# classification instance
clf = SVC(kernel = 'linear')

# make a pipeline
pipeline = Pipeline([("feature selection", selection), ("classification", clf)])

# train the model
pipeline.fit(X, y)


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def plot_fit(X_train, y_train, X_test, y_pred):
    plt.plot(X_test, y_pred, label = "Model")
    #plt.plot(X_test, fun, label = "Function")
    plt.scatter(X_train, y_train, label = "Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))


import numpy as np

y_pred = pipeline.predict(X_test)

#plot_fit(X_train, y_train, X_test, y_pred)


# ### Last, but not least, Searching Parameter Space with `GridSearchCV`
# 

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(include_bias = False)
lm = LinearRegression()

pipeline = Pipeline([("polynomial_features", poly),
                         ("linear_regression", lm)])

param_grid = dict(polynomial_features__degree = list(range(1, 30, 2)),
                  linear_regression__normalize = [False, True])

grid_search = GridSearchCV(pipeline, param_grid=param_grid)
grid_search.fit(X[:, np.newaxis], y)
print(grid_search.best_params_)


# Created by a Microsoft Employee.
# 	
# The MIT License (MIT)<br>
# Copyright (c) 2016 Micheleen Harris
# 

# ### Some References
# * [The iris dataset and an intro to sklearn explained on the Kaggle blog](http://blog.kaggle.com/2015/04/22/scikit-learn-video-3-machine-learning-first-steps-with-the-iris-dataset/)
# * [sklearn: Conference Notebooks and Presentation from Open Data Science Conf 2015](https://github.com/amueller/odscon-sf-2015) by Andreas Mueller
# * [real-world example set of notebooks for learning ML from Open Data Science Conf 2015](https://github.com/cmmalone/malone_OpenDataSciCon) by Katie Malone
# * [PyCon 2015 Workshop, Scikit-learn tutorial](https://www.youtube.com/watch?v=L7R4HUQ-eQ0) by Jake VanDerplas (Univ of Washington, eScience Dept)
# * [Data Science for the Rest of Us](https://channel9.msdn.com/blogs/Cloud-and-Enterprise-Premium/Data-Science-for-Rest-of-Us) great introductory webinar (no math) by Brandon Rohrer (Microsoft)
# * [A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) with useful ML "folk wisdom" by Pedro Domingos (Univ of Washington, CS Dept)
# * [Machine Learning 101](http://www.astroml.org/sklearn_tutorial/general_concepts.html) associated with `sklearn` docs
# 
# ### Some Datasets
# * [Machine learning datasets](http://mldata.org/)
# * [Make your own with sklearn](http://scikit-learn.org/stable/datasets/index.html#sample-generators)
# * [Kaggle datasets](https://www.kaggle.com/datasets)
# 
# ### Contact Info
# 
# Micheleen Harris<br>
# email: michhar@microsoft.com
# 




# ## Learning Algorithms - Unsupervised Learning
# <img src='imgs/ml_process_by_micheleenharris.png' alt="Smiley face" width="400"><br>
# >  Reminder:  In machine learning, the problem of unsupervised learning is that of trying to find hidden structure in unlabeled data. Since the training set given to the learner is unlabeled, there is no error or reward signal to evaluate a potential solution. Basically, we are just finding a way to represent the data and get as much information from it that we can.
# 
# HEY!  Remember PCA from above?  PCA is actually considered unsupervised learning.  We just put it up there because it's a good way to visualize data at the beginning of the ML process.
# 
# Let's revisit it in a little more detail using the `iris` dataset.
# 

# Imports for python 2/3 compatibility

from __future__ import absolute_import, division, print_function, unicode_literals

# For python 2, comment these out:
# from builtins import range


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# ### PCA revisited
# 

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()

# subset data to have only sepal width (cm) and petal length (cm) for simplification
X = iris.data[:, 1:3]
print(iris.feature_names[1:3])

pca = PCA(n_components = 2)
pca.fit(X)

print("% of variance attributed to components: "+       ', '.join(['%.2f' % (x * 100) for x in pca.explained_variance_ratio_]))
print('\ncomponents of each feature:', pca.components_)

print(list(zip(pca.explained_variance_, pca.components_)))


# The `pca.explained_variance_` is like the magnitude of a components influence (amount of variance explained) and the `pca.components_` is like the direction of influence for each feature in each component.
# 
# <p style="text-align:right"><i>Code in next cell adapted from Jake VanderPlas's code [here](https://github.com/jakevdp/sklearn_pycon2015)</i></p>
# 

# plot the original data in X (before PCA)
plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.5)

# grab the component means to get the center point for plot below
means = pca.mean_

# here we use the direction of the components in pca.components_
#  and the magnitude of the variance explaine by that component in
#  pca.explained_variane_

# we plot the vector (manginude and direction) of the components
#  on top of the original data in X
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    print([means[0], v[0]+means[0]], [means[1], v[1]+means[1]])
    plt.plot([means[0], v[0]+means[0]], [means[1], v[1]+means[1]], '-k', lw=3)


# axis limits
plt.xlim(0, max(X[:, 0])+3)
plt.ylim(0, max(X[:, 1])+3)

# original feature labels of our data X
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])


# QUESTION:  In which direction in the data is the most variance explained?

# Recall, in the ML 101 module: unsupervised models have a `fit()`, `transform()` and/or `fit_transform()` in `sklearn`.
# 
# 
# If you want to both get a fit and new dataset with reduced dimensionality, which would you use below? (Fill in blank in code)
# 

# get back to our 4D dataset
X, y = iris.data, iris.target

pca = PCA(n_components = 0.95) # keep 95% of variance
X_trans = pca.___(X) # <- fill in the blank
print(X.shape)
print(X_trans.shape)


plt.scatter(X_trans[:, 0], X_trans[:, 1], c=iris.target, edgecolor='none', alpha=0.5,
           cmap=plt.cm.get_cmap('spring', 10))
plt.ylabel('Component 2')
plt.xlabel('Component 1')


# ### Clustering
# KMeans finds cluster centers that are the mean of the points within them.  Likewise, a point is in a cluster because the cluster center is the closest cluster center for that point.
# 

# > If you don't have ipywidgets package installed, go ahead and install it now by running the cell below uncommented.
# 

get_ipython().system('pip install ipywidgets')


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


from ipywidgets import interact
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X, y = iris.data, iris.target
pca = PCA(n_components = 2) # keep 2 components which explain most variance
X = pca.fit_transform(X)

X.shape


# I have to tell KMeans how many cluster centers I want
n_clusters  = 3

# for consistent results when running the methods below
random_state = 2


# <p style="text-align:right"><i>Code in next cell adapted from Jake VanderPlas's code [here](https://github.com/jakevdp/sklearn_pycon2015)</i></p>
# 

def _kmeans_step(frame=0, n_clusters=n_clusters):
    rng = np.random.RandomState(random_state)
    labels = np.zeros(X.shape[0])
    centers = rng.randn(n_clusters, 2)

    nsteps = frame // 3

    for i in range(nsteps + 1):
        old_centers = centers
        if i < nsteps or frame % 3 > 0:
            dist = euclidean_distances(X, centers)
            labels = dist.argmin(1)

        if i < nsteps or frame % 3 > 1:
            centers = np.array([X[labels == j].mean(0)
                                for j in range(n_clusters)])
            nans = np.isnan(centers)
            centers[nans] = old_centers[nans]


    # plot the data and cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='rainbow',
                vmin=0, vmax=n_clusters - 1);
    plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                c=np.arange(n_clusters),
                s=200, cmap='rainbow')
    plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                c='black', s=50)

    # plot new centers if third frame
    if frame % 3 == 2:
        for i in range(n_clusters):
            plt.annotate('', centers[i], old_centers[i], 
                         arrowprops=dict(arrowstyle='->', linewidth=1))
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c=np.arange(n_clusters),
                    s=200, cmap='rainbow')
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c='black', s=50)

    plt.xlim(-4, 5)
    plt.ylim(-2, 2)
    plt.ylabel('PC 2')
    plt.xlabel('PC 1')

    if frame % 3 == 1:
        plt.text(4.5, 1.7, "1. Reassign points to nearest centroid",
                 ha='right', va='top', size=8)
    elif frame % 3 == 2:
        plt.text(4.5, 1.7, "2. Update centroids to cluster means",
                 ha='right', va='top', size=8)


# KMeans employ the <i>Expectation-Maximization</i> algorithm which works as follows: 
# 
# 1. Guess cluster centers
# * Assign points to nearest cluster
# * Set cluster centers to the mean of points
# * Repeat 1-3 until converged
# 

# suppress future warning
# import warnings
# warnings.filterwarnings('ignore')

min_clusters, max_clusters = 1, 6
interact(_kmeans_step, frame=[0, 20],
                    n_clusters=[min_clusters, max_clusters])


# > <b>Warning</b>! There is absolutely no guarantee of recovering a ground truth. First, choosing the right number of clusters is hard. Second, the algorithm is sensitive to initialization, and can fall into local minima, although scikit-learn employs several tricks to mitigate this issue.<br>  --Taken directly from sklearn docs
# 
# <img src='imgs/pca1.png' alt="Original PCA with Labels" align="center">
# 

# ### Novelty detection aka anomaly detection
# QUICK QUESTION:
# What is the diffrence between outlier detection and anomaly detection?
# 
# Below we will use a one-class support vector machine classifier to decide if a point is weird or not given our original data. (The code was adapted from sklearn docs [here](http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#example-svm-plot-oneclass-py))

get_ipython().magic('matplotlib inline')
from matplotlib import rcParams, font_manager
rcParams['figure.figsize'] = (14.0, 7.0)
fprop = font_manager.FontProperties(size=14)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

xx, yy = np.meshgrid(np.linspace(-2, 9, 500), np.linspace(-2,9, 500))

# Iris data
iris = load_iris()
X, y = iris.data, iris.target
labels = iris.feature_names[1:3]
X = X[:, 1:3]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# make some outliers
X_weird = np.random.uniform(low=-2, high=9, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1, random_state = 0)
clf.fit(X_train)

# predict labels
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_weird)


n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection aka Anomaly Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
c = plt.scatter(X_weird[:, 0], X_weird[:, 1], c='red')
plt.axis('tight')
plt.xlim((-2, 9))
plt.ylim((-2, 9))
plt.ylabel(labels[1], fontsize = 14)
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="best",
           prop=fprop)
plt.xlabel(
    "%s\nerror train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/10"
    % (labels[0], n_error_train, n_error_test, n_error_outliers), fontsize = 14)


# TRY changing the value of the parameters in the SVM classifier above especially `gamma`.  More information on `gamma` and support vector machine classifiers [here](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html).
# 

# Created by a Microsoft Employee.
# 	
# The MIT License (MIT)<br>
# Copyright (c) 2016 Micheleen Harris
# 

# #  (Clean Data) and Transform Data
# <img src='imgs/ml_process_by_micheleenharris.png' alt="Smiley face" width="400"><br>
# 

# ### Make the learning easier or better  beforehand -  feature reduction/selection/creation
# * SelectKBest
# * PCA
# * One-Hot Encoder
# 
# Just to remind you, the features of the irises we are dealing with on the flower are:
# ![Iris with labels](imgs/iris_with_labels.jpg)

# Imports for python 2/3 compatibility

from __future__ import absolute_import, division, print_function, unicode_literals

# For python 2, comment these out:
# from builtins import range


# ### Selecting k top scoring features (also dimensionality reduction)
# * Considered unsupervised learning
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# SelectKBest for selecting top-scoring features

from sklearn import datasets
from sklearn.feature_selection import SelectKBest, chi2

# Our nice, clean data (it's not always going to be this easy)
iris = datasets.load_iris()
X, y = iris.data, iris.target

print('Original shape:', X.shape)


# Let's add a NEW feature - a ratio of two of the iris measurements
df = pd.DataFrame(X, columns = iris.feature_names)
df['petal width / sepal width'] = df['petal width (cm)'] / df['sepal width (cm)']
new_feature_names = df.columns
print('New feature names:', list(new_feature_names))

# We've now added a new column to our data
X = np.array(df)


# Perform feature selection
#  input is scoring function (here chi2) to get univariate p-values
#  and number of top-scoring features (k) - here we get the top 3
dim_red = SelectKBest(chi2, k = 3)
dim_red.fit(X, y)
X_t = dim_red.transform(X)


# Show scores, features selected and new shape
print('Scores:', dim_red.scores_)
print('New shape:', X_t.shape)


# Get back the selected columns
selected = dim_red.get_support() # boolean values
selected_names = new_feature_names[selected]

print('Top k features: ', list(selected_names))


# **Note on scoring function selection in `SelectKBest` tranformations:**
# * For regression - [f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)
# * For classification - [chi2](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2), [f_classif](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)
# 

# ### Principal component analysis (aka PCA)
# * Reduces dimensions (number of features), based on what information explains the most variance (or signal)
# * Considered unsupervised learning
# * Useful for very large feature space (e.g. say the botanist in charge of the iris dataset measured 100 more parts of the flower and thus there were 104 columns instead of 4)
# * More about PCA on wikipedia [here](https://en.wikipedia.org/wiki/Principal_component_analysis)
# 

# PCA for dimensionality reduction

from sklearn import decomposition
from sklearn import datasets

iris = datasets.load_iris()

X, y = iris.data, iris.target

# perform principal component analysis
pca = decomposition.PCA(.95)
pca.fit(X)
X_t = pca.transform(X)
(X_t[:, 0])

# import numpy and matplotlib for plotting (and set some stuff)
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# let's separate out data based on first two principle components
x1, x2 = X_t[:, 0], X_t[:, 1]


# please don't worry about details of the plotting below 
#  (note: you can get the iris names below from iris.target_names, also in docs)
c1 = np.array(list('rbg')) # colors
colors = c1[y] # y coded by color
classes = iris.target_names[y] # y coded by iris name
for (i, cla) in enumerate(set(classes)):
    xc = [p for (j, p) in enumerate(x1) if classes[j] == cla]
    yc = [p for (j, p) in enumerate(x2) if classes[j] == cla]
    cols = [c for (j, c) in enumerate(colors) if classes[j] == cla]
    plt.scatter(xc, yc, c = cols, label = cla)
    plt.ylabel('Principal Component 2')
    plt.xlabel('Principal Component 1')
plt.legend(loc = 4)


# ### More feature selection methods [here](http://scikit-learn.org/stable/modules/feature_selection.html)
# 

# ### One Hot Encoding
# * It's an operation on feature labels - a method of dummying variable
# * Expands the feature space by nature of transform - later this can be processed further with a dimensionality reduction (the dummied variables are now their own features)
# * FYI:  One hot encoding variables is needed for python ML module `tenorflow`
# * Can do this with `pandas` method or a `sklearn` one-hot-encoder system
# 

# #### `pandas` method
# 

# Dummy variables with pandas built-in function

import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

# Convert to dataframe and add a column with actual iris species name
data = pd.DataFrame(X, columns = iris.feature_names)
data['target_name'] = iris.target_names[y]

df = pd.get_dummies(data, prefix = ['target_name'])
df.head()


# #### `sklearn` method
# 

# OneHotEncoder for dummying variables

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

# We encode both our categorical variable and it's labels
enc = OneHotEncoder()
label_enc = LabelEncoder() # remember the labels here

# Encode labels (can use for discrete numerical values as well)
data_label_encoded = label_enc.fit_transform(y)

# Encode and "dummy" variables
data_feature_one_hot_encoded = enc.fit_transform(y.reshape(-1, 1))
print(data_feature_one_hot_encoded.shape)

num_dummies = data_feature_one_hot_encoded.shape[1]
df = pd.DataFrame(data_feature_one_hot_encoded.toarray(), columns = label_enc.inverse_transform(range(num_dummies)))

df.head()


# Created by a Microsoft Employee.
# 	
# The MIT License (MIT)<br>
# Copyright (c) 2016 Micheleen Harris
# 

# ## Learning Algorithms - Supervised Learning
# 
# >  Reminder:  All supervised estimators in scikit-learn implement a `fit(X, y)` method to fit the model and a `predict(X)` method that, given unlabeled observations X, returns the predicted labels y. (direct quote from `sklearn` docs)
# 
# * Given that Iris is a fairly small, labeled dataset with relatively few features...what algorithm would you start with and why?

# > "Often the hardest part of solving a machine learning problem can be finding the right estimator for the job."
# 
# > "Different estimators are better suited for different types of data and different problems."
# 
# <a href = "http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html" style = "float: right">-Choosing the Right Estimator from sklearn docs</a>
# 

# Imports for python 2/3 compatibility

from __future__ import absolute_import, division, print_function, unicode_literals

# For python 2, comment these out:
# from builtins import range


# <b>An estimator for recognizing a new iris from its measurements</b>
# 
# > Or, in machine learning parlance, we <i>fit</i> an estimator on known samples of the iris measurements to <i>predict</i> the class to which an unseen iris belongs.
# 
# Let's give it a try!  (We are actually going to hold out a small percentage of the `iris` dataset and check our predictions against the labels)
# 

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

# Let's load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# split data into training and test sets using the handy train_test_split func
# in this split, we are "holding out" only one value and label (placed into X_test and y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# Let's try a decision tree classification method
from sklearn import tree

t = tree.DecisionTreeClassifier(max_depth = 4,
                                    criterion = 'entropy', 
                                    class_weight = 'balanced',
                                    random_state = 2)
t.fit(X_train, y_train)

t.score(X_test, y_test) # what performance metric is this?


# What was the label associated with this test sample? ("held out" sample's original label)
# Let's predict on our "held out" sample
y_pred = t.predict(X_test)
print(y_pred)

#  fill in the blank below

# how did our prediction do for first sample in test dataset?
print("Prediction: %d, Original label: %d" % (y_pred[0], ___)) # <-- fill in blank


# Here's a nifty way to cross-validate (useful for quick model evaluation!)
from sklearn import cross_validation

t = tree.DecisionTreeClassifier(max_depth = 4,
                                    criterion = 'entropy', 
                                    class_weight = 'balanced',
                                    random_state = 2)

# splits, fits and predicts all in one with a score (does this multiple times)
score = cross_validation.cross_val_score(t, X, y)
score


# QUESTIONS:  What do these scores tell you?  Are they too high or too low you think?  If it's 1.0, what does that mean?

# ### What does the graph look like for this decision tree?  i.e. what are the "questions" and "decisions" for this tree...
# * Note:  You need both Graphviz app and the python package `graphviz` (It's worth it for this cool decision tree graph, I promise!)
# * To install both on OS X:
# ```
# sudo port install graphviz
# sudo pip install graphviz
# ```
# * For general Installation see [this guide](http://graphviz.readthedocs.org/en/latest/manual.html)
# 

from sklearn.tree import export_graphviz
import graphviz

# Let's rerun the decision tree classifier
from sklearn import tree

t = tree.DecisionTreeClassifier(max_depth = 4,
                                    criterion = 'entropy', 
                                    class_weight = 'balanced',
                                    random_state = 2)
t.fit(X_train, y_train)

t.score(X_test, y_test) # what performance metric is this?

export_graphviz(t, out_file="mytree.dot",  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

with open("mytree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph, format = 'png')


# ### From Decision Tree to Random Forest
# 

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(max_depth=4,
                                criterion = 'entropy', 
                                n_estimators = 100, 
                                class_weight = 'balanced',
                                n_jobs = -1,
                               random_state = 2)

#forest = RandomForestClassifier()
forest.fit(X_train, y_train)

y_preds = iris.target_names[forest.predict(X_test)]

forest.score(X_test, y_test)


# Here's a nifty way to cross-validate (useful for model evaluation!)
from sklearn import cross_validation

# reinitialize classifier
forest = RandomForestClassifier(max_depth=4,
                                criterion = 'entropy', 
                                n_estimators = 100, 
                                class_weight = 'balanced',
                                n_jobs = -1,
                               random_state = 2)

score = cross_validation.cross_val_score(forest, X, y)
score


# QUESTION:  Comparing to the decision tree method, what do these accuracy scores tell you?  Do they seem more reasonable?

# ### Splitting into train and test set vs. cross-validation
# 

# <p>We can be explicit and use the `train_test_split` method in scikit-learn ( [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html) ) as in (and as shown above for `iris` data):<p>
# 
# ```python
# # Create some data by hand and place 70% into a training set and the rest into a test set
# # Here we are using labeled features (X - feature data, y - labels) in our made-up data
# import numpy as np
# from sklearn import linear_model
# from sklearn.cross_validation import train_test_split
# X, y = np.arange(10).reshape((5, 2)), range(5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70)
# clf = linear_model.LinearRegression()
# clf.fit(X_train, y_train)
# ```
# 
# OR
# 
# Be more concise and
# 
# ```python
# import numpy as np
# from sklearn import cross_validation, linear_model
# X, y = np.arange(10).reshape((5, 2)), range(5)
# clf = linear_model.LinearRegression()
# score = cross_validation.cross_val_score(clf, X, y)
# ```
# 
# <p>There is also a `cross_val_predict` method to create estimates rather than scores and is very useful for cross-validation to evaluate models ( [cross_val_predict](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_predict.html) )
# 

# Created by a Microsoft Employee.
# 	
# The MIT License (MIT)<br>
# Copyright (c) 2016 Micheleen Harris
# 

