# # Lecture 16:  Boosting 
# ### Data Science 1: CS 109A/STAT 121A/AC 209A/ E 109A <br> Instructors: Pavlos Protopapas, Kevin Rader, Rahul Dave
# #### Harvard University <br> Fall 2017 <br> 
# 
# ---
# 

import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
sns.set(style="ticks")
get_ipython().magic('matplotlib inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from io import StringIO
import pydot 
from IPython.display import display

from IPython.display import Image
import seaborn as sns
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)

sns.set_context('poster')


#--------  fit_and_plot_dt
# Fit decision tree with on given data set with given depth, and plot the data/model
# Input: 
#      fname (string containing file name)
#      depth (depth of tree)

def fit_and_plot_dt(x, y, depth, title, ax, plot_data=True, fill=True, color='Greens'):
    # FIT DECISION TREE MODEL
    dt = tree.DecisionTreeClassifier(max_depth = depth)
    dt.fit(x, y)

    # PLOT DECISION TREE BOUNDARY
    ax = plot_tree_boundary(x, y, dt, title, ax, plot_data, fill, color)
    
    return ax


#--------  plot_tree_boundary
# A function that visualizes the data and the decision boundaries
# Input: 
#      x (predictors)
#      y (labels)
#      model (the classifier you want to visualize)
#      title (title for plot)
#      ax (a set of axes to plot on)
# Returns: 
#      ax (axes with data and decision boundaries)

def plot_tree_boundary(x, y, model, title, ax, plot_data=True, fill=True, color='Greens', alpha=0.1):
    if plot_data:
        # PLOT DATA
        ax.scatter(x[y==1,0], x[y==1,1], c='green')
        ax.scatter(x[y==0,0], x[y==0,1], c='white')
    
    # CREATE MESH
    interval = np.arange(min(x.min(), y.min()),max(x.max(), y.max()),0.01)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    xx = np.concatenate((x1, x2), axis=1)

    # PREDICT ON MESH POINTS
    yy = model.predict(xx)    
    yy = yy.reshape((n, n))

    # PLOT DECISION SURFACE
    x1 = x1.reshape(n, n)
    x2 = x2.reshape(n, n)
    if fill:
        ax.contourf(x1, x2, yy, alpha=alpha, cmap=color)
    else:
        ax.contour(x1, x2, yy, alpha=alpha, cmap=color)
    
    # LABEL AXIS, TITLE
    ax.set_title(title)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    
    return ax


data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=1000)
data = np.hstack((data, np.zeros((1000, 1))))
data[data[:, 0]**2 + data[:, 1]**2 < 3**2, 2] = 1


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
x = data[:, :-1]
y = data[:, -1]
ax.scatter(x[y == 1, 0], x[y == 1, 1], c='green', label='vegetation', alpha=0.5)
ax.scatter(x[y == 0, 0], x[y == 0, 1], c='gray', label='non vegetation', alpha=0.1)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('satellite image')
ax.legend()
plt.tight_layout()
plt.show() 


#Variance reduction: Baggining, RF, Tree, Adaboost
depth = None

fig, ax = plt.subplots(1, 4, figsize=(20, 5))


for i in range(8):
    new_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=1000)
    new_data = np.hstack((new_data, np.zeros((1000, 1))))
    new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2, 2] = 1
    x = new_data[:, :-1]
    y = new_data[:, -1]

    ax[0] = fit_and_plot_dt(x, y, depth, 'Variance of Single Decision Tree', ax[0], plot_data=False, fill=False)
    
    bag = ensemble.BaggingClassifier(n_estimators=30)
    bag.fit(x, y)
    ax[1] = plot_tree_boundary(x, y, bag, 'Variance of Bagging', ax[1], plot_data=False, fill=False, color='Reds')
    
    rf = ensemble.RandomForestClassifier(n_estimators=30)
    rf.fit(x, y)
    ax[2] = plot_tree_boundary(x, y, rf, 'Variance of RF', ax[2], plot_data=False, fill=False, color='Blues')

    rf = ensemble.RandomForestClassifier(n_estimators=30)
    rf.fit(x, y)
    ax[2] = plot_tree_boundary(x, y, rf, 'Variance of RF', ax[2], plot_data=False, fill=False, color='Blues')

    adaboost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=30)
    adaboost.fit(x, y)
    ax[3] = plot_tree_boundary(x, y, adaboost, 'Variance of AdaBoost', ax[3], plot_data=False, fill=False, color='Greys')
    
ax[0].set_xlim(-3.2, 3.2)
ax[0].set_ylim(-3.2, 3.2)
ax[1].set_xlim(-3.2, 3.2)
ax[1].set_ylim(-3.2, 3.2)
ax[2].set_xlim(-3.2, 3.2)
ax[2].set_ylim(-3.2, 3.2)
ax[3].set_xlim(-3.2, 3.2)
ax[3].set_ylim(-3.2, 3.2)
plt.show() 


#Error comparison
data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=200)
data = np.hstack((data, np.zeros((200, 1))))
data[data[:, 0]**2 + data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(data[data[:, 0]**2 + data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
x = data[:, :-1]
y = data[:, -1]

test_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=1000)
test_data = np.hstack((test_data, np.zeros((1000, 1))))
test_data[test_data[:, 0]**2 + test_data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(test_data[test_data[:, 0]**2 + test_data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

dt = tree.DecisionTreeClassifier()
dt.fit(x, y)
tree_score = np.array([dt.score(x_test, y_test)] * len(range(20, 320, 10)))

bag_score = []
bag_oob = []
rf_score = []
rf_oob = []
boost_score = []
for i in range(20, 320, 10):
    bag = ensemble.BaggingClassifier(n_estimators=i, oob_score=True)
    bag.fit(x, y)
    bag_score.append(bag.score(x_test, y_test))

    bag_oob.append(bag.oob_score_)
    
    rf = ensemble.RandomForestClassifier(n_estimators=i, oob_score=True)
    rf.fit(x, y)
    rf_score.append(rf.score(x_test, y_test))
    rf_oob.append(rf.oob_score_)
    
    adaboost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=i, learning_rate=0.5)
    adaboost.fit(x, y)
    boost_score.append(adaboost.score(x_test, y_test))


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(range(20, 320, 10), tree_score, color='black', linestyle='--', label='Single Tree')
ax.plot(range(20, 320, 10), bag_score, color='red', alpha=0.8, label='Bagging')
ax.plot(range(20, 320, 10), rf_score, color='green', alpha=0.8, label='Random Forest')
ax.plot(range(20, 320, 10), bag_oob, color='red', alpha=0.2, label='Bagging OOB')
ax.plot(range(20, 320, 10), rf_oob, color='green', alpha=0.2, label='RF OOB')
ax.plot(range(20, 320, 10), boost_score, color='grey', alpha=0.8, label='AdaBoost')
ax.set_title('Comparison of Errors')
ax.set_xlabel('Number of Trees in Ensemble')
ax.legend(loc='best')
plt.show()


#Choosing learning rate

boost_score_small = []
boost_score = []
boost_score_large = []
for i in range(20, 120, 5):
    adaboost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=i, learning_rate=1e-1)
    adaboost.fit(x, y)
    boost_score_small.append(adaboost.score(x, y))
    
    adaboost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=i, learning_rate=0.5)
    adaboost.fit(x, y)
    boost_score.append(adaboost.score(x, y))
    
    adaboost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=i, learning_rate=0.5e1)
    adaboost.fit(x, y)
    boost_score_large.append(adaboost.score(x, y))
    


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(range(20, 220, 10), boost_score_small, color='blue', label='lambda=1e-1')
ax.plot(range(20, 220, 10), boost_score, color='red', alpha=0.8, label='lambda=0.5')
ax.plot(range(20, 220, 10), boost_score_large, color='green', alpha=0.8, label='lambda=0.5e1')

ax.set_title('Comparison of Learning Rates')
ax.set_xlabel('Number of Trees in Ensemble')
ax.legend(loc='best')
plt.show()





# # Lecture 15:  Random Forest 
# ### Data Science 1: CS 109A/STAT 121A/AC 209A/ E 109A <br> Instructors: Pavlos Protopapas, Kevin Rader, Rahul Dave
# #### Harvard University <br> Fall 2017 <br> 
# 
# ---
# 

import pandas as pd
import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import ensemble
sns.set(style="ticks")
get_ipython().magic('matplotlib inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from io import StringIO
import pydot 
from IPython.display import display

from IPython.display import Image
import seaborn as sns
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)

sns.set_context('poster')


#--------  fit_and_plot_dt
# Fit decision tree with on given data set with given depth, and plot the data/model
# Input: 
#      fname (string containing file name)
#      depth (depth of tree)

def fit_and_plot_dt(x, y, depth, title, ax, plot_data=True, fill=True, color='Blues'):
    # FIT DECISION TREE MODEL
    dt = tree.DecisionTreeClassifier(max_depth = depth)
    dt.fit(x, y)

    # PLOT DECISION TREE BOUNDARY
    ax = plot_tree_boundary(x, y, dt, title, ax, plot_data, fill, color)
    
    return ax


#--------  plot_tree_boundary
# A function that visualizes the data and the decision boundaries
# Input: 
#      x (predictors)
#      y (labels)
#      model (the classifier you want to visualize)
#      title (title for plot)
#      ax (a set of axes to plot on)
# Returns: 
#      ax (axes with data and decision boundaries)

def plot_tree_boundary(x, y, model, title, ax, plot_data=True, fill=True, color='Greens', alpha=0.1):
    if plot_data:
        # PLOT DATA
        ax.scatter(x[y==1,0], x[y==1,1], c='blue')
        ax.scatter(x[y==0,0], x[y==0,1], c='black')
    
    # CREATE MESH
    interval = np.arange(min(x.min(), y.min()),max(x.max(), y.max()),0.01)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    xx = np.concatenate((x1, x2), axis=1)

    # PREDICT ON MESH POINTS
    yy = model.predict(xx)    
    yy = yy.reshape((n, n))

    # PLOT DECISION SURFACE
    x1 = x1.reshape(n, n)
    x2 = x2.reshape(n, n)
    if fill:
        ax.contourf(x1, x2, yy, alpha=alpha, cmap=color)
    else:
        ax.contour(x1, x2, yy, alpha=alpha, cmap=color)
    
    # LABEL AXIS, TITLE
    ax.set_title(title)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    
    return ax


# Create some fake data to illustrate the models 
# 

npoints = 500 
data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=npoints)
data = np.hstack((data, np.zeros((npoints, 1))))
data[data[:, 0]**2 + data[:, 1]**2 < 3**2, 2] = 1


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
x = data[:, :-1]
y = data[:, -1]
ax.scatter(x[y == 1, 0], x[y == 1, 1], c='green', label='vegetation', alpha=0.5)
ax.scatter(x[y == 0, 0], x[y == 0, 1], c='gray', label='non vegetation', alpha=0.1)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('satellite image')
ax.legend()
plt.tight_layout()
plt.show() 


#Bagging
depth = None

fig, ax = plt.subplots(1, 1, figsize=(10, 10))


for i in range(8):
    sample_ind = np.random.choice(range(len(data)), len(data), replace=True)
    bootstrap = data[sample_ind]
    x = bootstrap[:, :-1]
    y = bootstrap[:, -1]
    ax = fit_and_plot_dt(x, y, depth, '', ax, plot_data=False, fill=False) 
 
#KIND OF CHEATING BELOW BECAUSE Bagging will create its own bootstrap samples
bag = ensemble.BaggingClassifier(n_estimators=100)
bag.fit(x, y)

ax = plot_tree_boundary(x, y, bag, 'Bagging', ax, plot_data=False, fill=False, color='Reds', alpha=1.)

ax.scatter(x[y == 1, 0], x[y == 1, 1], c='green', label='vegetation', alpha=0.5)
ax.scatter(x[y == 0, 0], x[y == 0, 1], c='gray', label='non vegetation', alpha=0.1)

ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.2, 3.2)
plt.show() 


#Variance reduction: Baggining, RF, Tree
depth = None

fig, ax = plt.subplots(1, 3, figsize=(15, 5))


for i in range(8):
    new_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=npoints)
    new_data = np.hstack((new_data, np.zeros((npoints, 1))))
    new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2, 2] = 1
    x = new_data[:, :-1]
    y = new_data[:, -1]

    ax[0] = fit_and_plot_dt(x, y, depth, 'Variance of Single Decision Tree', ax[0], plot_data=False, fill=False)
    
    bag = ensemble.BaggingClassifier(n_estimators=30)
    bag.fit(x, y)
    ax[1] = plot_tree_boundary(x, y, bag, 'Variance of Bagging', ax[1], plot_data=False, fill=False, color='Reds')
    
    rf = ensemble.RandomForestClassifier(n_estimators=30)
    rf.fit(x, y)
    ax[2] = plot_tree_boundary(x, y, rf, 'Variance of RF', ax[2], plot_data=False, fill=False, color='Blues')


    
ax[0].set_xlim(-3.2, 3.2)
ax[0].set_ylim(-3.2, 3.2)
ax[1].set_xlim(-3.2, 3.2)
ax[1].set_ylim(-3.2, 3.2)
ax[2].set_xlim(-3.2, 3.2)
ax[2].set_ylim(-3.2, 3.2)
plt.show() 


#Errors
data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=200)
data = np.hstack((data, np.zeros((200, 1))))
data[data[:, 0]**2 + data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(data[data[:, 0]**2 + data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
x = data[:, :-1]
y = data[:, -1]

test_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=1000)
test_data = np.hstack((test_data, np.zeros((1000, 1))))
test_data[test_data[:, 0]**2 + test_data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(test_data[test_data[:, 0]**2 + test_data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

dt = tree.DecisionTreeClassifier()
dt.fit(x, y)
tree_score = np.array([dt.score(x_test, y_test)] * len(range(20, 320, 10)))

bag_score = []
bag_oob = []
rf_score = []
rf_oob = []
for i in range(20, 320, 10):
    bag = ensemble.BaggingClassifier(n_estimators=i, oob_score=True)
    bag.fit(x, y)
    bag_score.append(bag.score(x_test, y_test))

    bag_oob.append(bag.oob_score_)
    
    rf = ensemble.RandomForestClassifier(n_estimators=i, oob_score=True)
    rf.fit(x, y)
    rf_score.append(rf.score(x_test, y_test))
    rf_oob.append(rf.oob_score_)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
tree_score = np.array([dt.score(x_test, y_test)] * len(range(20, 320, 10)))
ax.plot(range(20, 320, 10), tree_score, color='black', linestyle='--', label='Single Tree')
ax.plot(range(20, 320, 10), bag_score, color='red', alpha=0.8, label='Bagging')
ax.plot(range(20, 320, 10), rf_score, color='green', alpha=0.8, label='Random Forest')
ax.plot(range(20, 320, 10), bag_oob, color='red', alpha=0.2, label='Bagging OOB')
ax.plot(range(20, 320, 10), rf_oob, color='green', alpha=0.2, label='RF OOB')
ax.set_title('Comparison of Errors')
ax.set_xlabel('Number of Trees in Ensemble')
ax.legend(loc='best')
plt.show()


# ### The heart data
# 

df = pd.read_csv('data/Heart.csv', index_col=0)
df.head()


df.dtypes


predictors_df = df[['Age', 'RestBP','Chol' , 'Fbs', 'RestECG', 'MaxHR', 'ExAng']]

#cat_predictors_df = df[['ChestPain','Thal', 'AHD', 'Sex']]
cat_predictors_df = df[['ChestPain','Thal',  'Sex']]
dummies_df = pd.get_dummies(cat_predictors_df)
dummies_df.shape





# Join Everything Together
# 

dfpreds = predictors_df.join(dummies_df)
print(dfpreds.shape)
dfpreds.head()


X = dfpreds
y_l = df.iloc[:,-1]
D = {"Yes":1, "No":0}
D['No']
y = [1*D[y_] for y_ in y_l] 






rf = ensemble.RandomForestClassifier(max_features= 1, n_estimators=50, oob_score=True)
rf.fit(X, y)

feature_importance = rf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5





plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')


plt.savefig('fig/variable_importance.png')


bg = ensemble.BaggingClassifier( n_estimators=20, oob_score=True)
bg.fit(X, y)


feature_importance = np.mean([
    tree.feature_importances_ for tree in bg.estimators_
], axis=0)



feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5





plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')



plt.savefig('fig/variable_importance_bagging.png')


# ### Bagging. All trees are correlated ! 
# 

# Created nboots 
nboots = 10 
for nboots in np.arange(0,nboots):  
    
    idx = np.random.randint(len(X), size=len(X))
    Xb = X.iloc[idx, :]
    yb = np.array(y)[idx]
    
    bg = DecisionTreeClassifier(max_depth=3)
    bg.fit(Xb, yb)


    dummy_io = StringIO() 
    export_graphviz(bg, out_file = dummy_io, feature_names=X.columns,                    class_names=['Yes', 'No'], proportion=True, filled=True)
    (graph,)=pydot.graph_from_dot_data(dummy_io.getvalue())
    display(Image(graph.create_png()))


len(X)


idx = np.random.randint(len(X), size=len(X))
Xb = X.iloc[idx, :]
yb = np.array(y)[idx]

bg = DecisionTreeClassifier(max_depth=3)
bg.fit(Xb, yb)


dummy_io = StringIO() 
export_graphviz(bg, out_file = dummy_io, feature_names=X.columns,                class_names=['Yes', 'No'], proportion=True, filled=True)
(graph,)=pydot.graph_from_dot_data(dummy_io.getvalue())
display(Image(graph.create_png()))


graph.write_png("fig/pruning_1")





# # Lecture 14:  Decision Trees 
# ### Data Science 1: CS 109A/STAT 121A/AC 209A/ E 109A <br>  Instructors: P. Protopapas, Kevin Rader, Rahul Dave, Margo Levine
# #### Harvard University <br> Fall 2017 <br> 
# 
# ---
# 

import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import RegressionResults
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import ensemble
sns.set(style="ticks")
get_ipython().magic('matplotlib inline')



import seaborn as sns
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)

sns.set_context('poster')


#--------  fit_and_plot_dt
# Fit decision tree with on given data set with given depth, and plot the data/model
# Input: 
#      fname (string containing file name)
#      depth (depth of tree)

def fit_and_plot_dt(x, y, depth, title, ax, plot_data=True, fill=True, color='Blues'):
    # FIT DECISION TREE MODEL
    dt = tree.DecisionTreeClassifier(max_depth = depth)
    dt.fit(x, y)

    # PLOT DECISION TREE BOUNDARY
    ax = plot_tree_boundary(x, y, dt, title, ax, plot_data, fill, color)
    
    return ax


#--------  plot_tree_boundary
# A function that visualizes the data and the decision boundaries
# Input: 
#      x (predictors)
#      y (labels)
#      model (the classifier you want to visualize)
#      title (title for plot)
#      ax (a set of axes to plot on)
# Returns: 
#      ax (axes with data and decision boundaries)

def plot_tree_boundary(x, y, model, title, ax, plot_data=True, fill=True, color='Greens'):
    if plot_data:
        # PLOT DATA
        ax.scatter(x[y==1,0], x[y==1,1], c='green')
        ax.scatter(x[y==0,0], x[y==0,1], c='grey')
    
    # CREATE MESH
    interval = np.arange(min(x.min(), y.min()),max(x.max(), y.max()),0.01)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    xx = np.concatenate((x1, x2), axis=1)

    # PREDICT ON MESH POINTS
    yy = model.predict(xx)    
    yy = yy.reshape((n, n))

    # PLOT DECISION SURFACE
    x1 = x1.reshape(n, n)
    x2 = x2.reshape(n, n)
    if fill:
        ax.contourf(x1, x2, yy, alpha=0.1, cmap=color)
    else:
        ax.contour(x1, x2, yy, alpha=0.1, cmap=color)
    
    # LABEL AXIS, TITLE
    ax.set_title(title)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    
    return ax


npoints = 200 
data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=npoints)
data = np.hstack((data, np.zeros((npoints, 1))))

data[data[:, 0]**2 + data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(data[data[:, 0]**2 + data[:, 1]**2 < 3**2]), p=[0.2, 0.8])


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
x = data[:, :-1]
y = data[:, -1]
ax.scatter(x[y == 1, 0], x[y == 1, 1], c='green', label='vegetation')
ax.scatter(x[y == 0, 0], x[y == 0, 1], c='black', label='non vegetation', alpha=0.25)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('satellite image')
ax.legend()
plt.tight_layout()
plt.show() 


#Different Depths
depths = [1, 5, 100000]
fig, ax = plt.subplots(1, len(depths), figsize=(15, 5))
x = data[:, :-1]
y = data[:, -1]
ind = 0
for i in depths:
    ax[ind] = fit_and_plot_dt(x, y, i, 'Depth {}'.format(i), ax[ind]) 
    ax[ind].set_xlim(-6, 6)
    ax[ind].set_ylim(-6, 6)
    ind += 1    


#Overfitting

depths = [1, 5, 100, 1000, 100000, 1000000]

test_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=1000)
test_data = np.hstack((test_data, np.zeros((1000, 1))))
test_data[test_data[:, 0]**2 + test_data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(test_data[test_data[:, 0]**2 + test_data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

scores = []
scores_train = []
for depth in depths:
    dt = tree.DecisionTreeClassifier(max_depth = depth)
    dt.fit(x, y)
    scores.append(dt.score(x_test, y_test))
    scores_train.append(dt.score(x, y))
    
plt.plot(depths, scores, 'b*-', label = 'Test')
plt.plot(depths, scores_train, 'g*-', label = 'Train')
plt.xlabel('Depth')
plt.ylabel('Score')
plt.xscale('log')
plt.legend()



#Variance comparison between simple and complex models
depths = [3, 10, 1000]

fig, ax = plt.subplots(1, len(depths), figsize=(15, 5))

for d in range(len(depths)):
    for i in range(10):
        new_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=200)
        new_data = np.hstack((new_data, np.zeros((200, 1))))
        new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
        x = new_data[:, :-1]
        y = new_data[:, -1]
        ax[d] = fit_and_plot_dt(x, y, depths[d], 'Depth {}'.format(depths[d]), ax[d], plot_data=False, fill=False) 
        ax[d].set_xlim(-4, 4)
        ax[d].set_ylim(-4, 4)
plt.tight_layout()
plt.show() 


#Different Splitting Criteria
depth = 15

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

new_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=200)
new_data = np.hstack((new_data, np.zeros((200, 1))))
new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
x = new_data[:, :-1]
y = new_data[:, -1]

dt = tree.DecisionTreeClassifier(max_depth = depth)
dt.fit(x, y)

ax[0] = plot_tree_boundary(x, y, dt, 'Gini', ax[0], color='Reds')

dt = tree.DecisionTreeClassifier(max_depth = depth, criterion='entropy')
dt.fit(x, y)

ax[1] = plot_tree_boundary(x, y, dt, 'Entropy', ax[1], color='Reds')


ax[0].set_xlim(-4, 4)
ax[0].set_ylim(-4, 4)
ax[1].set_xlim(-4, 4)
ax[1].set_ylim(-4, 4)
        
plt.tight_layout()
plt.show() 


#Different Stopping Conditions

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

new_data = np.random.multivariate_normal([0, 0], np.eye(2) * 5, size=200)
new_data = np.hstack((new_data, np.zeros((200, 1))))
new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2, 2] = np.random.choice([0, 1], len(new_data[new_data[:, 0]**2 + new_data[:, 1]**2 < 3**2]), p=[0.2, 0.8])
x = new_data[:, :-1]
y = new_data[:, -1]

dt = tree.DecisionTreeClassifier()
dt.fit(x, y)

ax[0] = plot_tree_boundary(x, y, dt, 'No Stopping Conditions', ax[0])

dt = tree.DecisionTreeClassifier(min_impurity_split=0.32)
dt.fit(x, y)

ax[1] = plot_tree_boundary(x, y, dt, 'Minimum Purity Split = 0.32', ax[1])

dt = tree.DecisionTreeClassifier(min_samples_leaf=10)
dt.fit(x, y)

ax[2] = plot_tree_boundary(x, y, dt, 'Minimum Samples per Leaf = 10', ax[2])

ax[0].set_xlim(-4, 4)
ax[0].set_ylim(-4, 4)
ax[1].set_xlim(-4, 4)
ax[1].set_ylim(-4, 4)
ax[2].set_xlim(-4, 4)
ax[2].set_ylim(-4, 4)      

plt.tight_layout()
plt.show() 


# # Lecture 10: Classification and Logistic Regression
# 

get_ipython().magic('matplotlib inline')
import sys
import numpy as np
import pylab as pl
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import sklearn.linear_model as sk


x = np.linspace(-5, 5, 100)
y1 = np.exp(0+1*x)/(1+np.exp(0+1*x))
y2 = np.exp(2+1*x)/(1+np.exp(2+1*x))
y3 = np.exp(0+3*x)/(1+np.exp(0+3*x))
y4 = np.exp(0-1*x)/(1+np.exp(0-1*x))


plt.plot(x,y1,color='black')
plt.plot(x,y2,color='red')
plt.plot(x,y3,color='blue')
plt.plot(x,y4,color='green')

plt.show()


import random
random.seed(12345)

#read the NFL play-by-play data
nfldata = pd.read_csv("NFLplaybyplay-2015.csv")

# shuffle the data
nfldata = nfldata.reindex(np.random.permutation(nfldata.index))

# For simplicity, we will select only 500 points form the dataset.
N = 500
nfldata_sm = nfldata.sample(N)
nfldata_sm.head()


#genomicdata = pd.read_csv("genomic_subset.csv")
#genomicdata.head()



# The following function creates the polynomial design matrix.
def polynomial_basis (x, degree):
    p = np.arange (1, degree + 1)
    return x[:, np.newaxis] ** p

# We create the design matrix of a polynomial of 1 degree.
X = polynomial_basis (nfldata_sm["YardLine"], 1)

plt.scatter(nfldata_sm["YardLine"],nfldata_sm["IsTouchdown"],  color='black')
plt.xlabel ("Yard Line")
plt.ylabel("A Touchdown was Scored")
#plt.plot(x, logitm.predict_proba(x)[:,1],  color='red' , lw=3)
#plt.show()

# Create linear regression object
lm = sk.LinearRegression()
lm.fit (X, nfldata_sm["IsTouchdown"])

# The coefficients
#print('Coefficients: \n', lm.coef_)

# Create logistic regression object
logitm = sk.LogisticRegression(C = 1000000)
logitm.fit (X, nfldata_sm["IsTouchdown"])

# The coefficients
print('Estimated beta1: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)



# Plot outputs
plt.scatter(nfldata_sm["YardLine"],nfldata_sm["IsTouchdown"],  color='black')
plt.xlim(0,100)
plt.plot(X, lm.predict(X), color='blue',lw=3)
x = np.linspace(0, 300, 100)
x = polynomial_basis (x, 1)
#plt.plot(x, logitm.predict_proba(x),  color='red' , lw=3)
plt.plot(x, logitm.predict_proba(x)[:,1],  color='red' , lw=3)
plt.xlabel ("Yard Line")
plt.ylabel("A Touchdown was Scored")

plt.show()


X2 = polynomial_basis (nfldata_sm["IsPass"], 1)
logitm.fit (X2,nfldata_sm["IsTouchdown"])

# The coefficients
print('Estimated beta1: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)

Y=nfldata_sm["IsTouchdown"]
#passes=nfldata["IsPass"0]==0
print(np.mean(Y[nfldata["IsPass"]==0]))
print(np.mean(Y[nfldata["IsPass"]==1]))


# Create data frame of predictors
X = nfldata[["YardLine","IsPass"]]

# Create logistic regression object
logitm = sk.LogisticRegression(C = 1000000)
logitm.fit (X, nfldata["IsTouchdown"])

# The coefficients
print('Estimated beta1, beta2: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)


x = np.linspace(0, 100, 100)
x = polynomial_basis (x, 1)
x0 = np.insert(x,1,0,axis=1)
x1 = np.insert(x,1,1,axis=1)

# Plot outputs
plt.scatter(nfldata["YardLine"],nfldata["IsTouchdown"],  color='black')
plt.plot(x, logitm.predict_proba(x0)[:,1],  color='red' , lw=3)
plt.plot(x, logitm.predict_proba(x1)[:,1],  color='blue' , lw=3)
plt.xlabel ("Yard Line")
plt.ylabel("A Touchdown was Scored")
plt.xlim(0,100)
plt.show()


# Create data frame of predictors
nfldata['Interaction'] = nfldata["YardLine"]*nfldata["IsPass"]
X = nfldata[["YardLine","IsPass","Interaction"]]

# Create logistic regression object
logitm = sk.LogisticRegression(C = 100000000000000000)
logitm.fit (X, nfldata["IsTouchdown"])

# The coefficients
print('Estimated beta1, beta2, beta3: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)

nfldata['Intercept'] = 1.0
logit_sm = sm.Logit(nfldata['IsTouchdown'], nfldata[["Intercept","YardLine","IsPass","Interaction"]])
fit_sm = logit_sm.fit()
print(fit_sm.summary())

nfldata.head()





# # CS 109A/AC 209A/STAT 121A Data Science: 2016 Midterm 2 Solutions 
# **Harvard University**<br>
# **Fall 2016**<br>
# **Instructors: W. Pan, P. Protopapas, K. Rader**<br>
# **Due Date: ** Tuesday, November 22nd, 2016 at 12:00pm
# 

import numpy as np
import pandas as pd
import scipy as sp
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Part I: Diagnosing the Semian Flu 2016
# 
# You are given the early data for an outbreak of a dangerous virus originating from a group of primates being keeped in a Massechussetts biomedical research lab, this virus is dubbed the "Semian Flu".
# 
# You have the medical records of $n$ number of patients in `'flu_train.csv`. There are two general types of patients in the data, flu patients and healthy (this is recorded in the column labeled `flu`, a 0 indicates the absences of the virus and a 1 indicates presence). Furthermore, scientists have found that there are two strains of the virus, each requiring a different type of treatment (this is recorded in the column labeled `flutype`, a 1 indicates the absences of the virus, a 2 indicates presence of strain 1 and a 3 indicates the presence of strain 2).
# 
# **Your task:** build a model to predict if a given patient has the flu. Your goal is to catch as many flu patients as possible without misdiagnosing too many healthy patients.
# 
# **The deliverable:** a function called `flu_predict` which satisfies:
# 
# - input: `x_test`, a set of medical predictors for a group of patients
# - output: `y_pred`, a set of labels, one for each patient; 0 for healthy and 1 for infected with the flu virus
# 
# The MA state government will use your model to diagnose sets of future patients (held by us). You can expect that there will be an increase in the number of flu patients in any groups of patients in the future.
# 
# We provide you with some benchmarks for comparison.
# 
# **Baseline Model:** 
# - ~50% expected accuracy on healthy patients in observed data
# - ~50% expected accuracy on flu patients in observed data
# - ~50% expected accuracy on healthy patients in future data 
# - ~50% expected accuracy on flu patients in future data
# - time to build: 5 min
# 
# **Reasonable Model:** 
# - ~69% expected accuracy on healthy patients in observed data
# - ~55% expected accuracy on flu patients, in observed data
# - ~69% expected accuracy on healthy patients in future data
# - ~60% expected accuracy on flu patients, in future data
# - time to build: 20 min
# 
# **Grading:**
# Your grade will be based on:
# 1. your model's ability to out-perform our benchmarks
# 2. your ability to carefully and thoroughly follow the data science pipeline (see lecture slides for definition)
# 3. the extend to which all choices are reasonable and defensible by methods you have learned in this class
# 

# **Solutions:**
# 
# ## Step 1: Read the data, clean and explore the data
# 
# There are a large number of missing values in the data. Nearly all predictors have some degree of missingness. Not all missingness are alike: as Mike points out, NaN in the `'pregnancy'` column is meaningful and informative, as patients with NaN's in the pregnancy column are males, where as NaN's in other predictors may appear randomly. 
# 
# 
# **What we do:** We make no attempt to interpret the predictors and we make no attempt to model the missing values in the data in any meaningful way. We replace all missing values with 0.
# 
# However, it would be more complete to look at the data and allow the data to inform your decision on how to address missingness. For columns where NaN values are informative, you might want to treat NaN as a distinct value; You might want to drop predictors with too many missing values and impute the ones with few missing values using KNN or a parametric model. There are many acceptable strategies here, as long as the appropriateness of the method in the context of the task and the data is discussed.
# 

#Train

df = pd.read_csv('data/flu_train.csv')

df = df[~np.isnan(df['flu'])]

df.head()


#Clean and encode

encode = preprocessing.LabelEncoder()

for column in df.columns:
    if df[column].dtype == np.object:
        df[column] = df[column].fillna('')
        df.loc[:, column] = encode.fit_transform(df[column])
        
df = df.fillna(0)

df.head()


#Test

df_test = pd.read_csv('data/flu_test.csv')

df_test = df_test[~np.isnan(df_test['flu'])]

df_test.head()


#Clean and encode

encode = preprocessing.LabelEncoder()

for column in df_test.columns:
    if df_test[column].dtype == np.object:
        df_test[column] = df_test[column].fillna('')
        df_test.loc[:, column] = encode.fit_transform(df_test[column])
        
df_test = df_test.fillna(0)

df_test.head()


#What's up in each set

x = df.values[:, :-2]
y = df.values[:, -2]

x_test = df_test.values[:, :-2]
y_test = df_test.values[:, -2]

print('x train shape:', x.shape)
print('x test shape:', x_test.shape)
print('train class 0: {}, train class 1: {}'.format(len(y[y==0]), len(y[y==1])))
print('train class 0: {}, train class 1: {}'.format(len(y_test[y_test==0]), len(y_test[y_test==1])))


# ## Step 2: Model Choice
# 
# The first task is to decide which, of the large number of classifiers we have learned during this semester, would best suit our task and our data.
# 
# It would be possible to do brute force model comparison here - i.e. tune all models and compare which does best with respect to various benchmarks. However, it is also reasonable to do a first round of model comparison by running models (with out of the box parameter settings) on the training data and eliminating models which performed very poorly. 
# 

def expected_score(model, x_test, y_test):
    overall = 0
    class_0 = 0
    class_1 = 0
    for i in range(100):
        sample = np.random.choice(len(x_test), len(x_test))
        x_sub_test = x_test[sample]
        y_sub_test = y_test[sample]
        
        overall += model.score(x_sub_test, y_sub_test)
        class_0 += model.score(x_sub_test[y_sub_test==0], y_sub_test[y_sub_test==0])
        class_1 += model.score(x_sub_test[y_sub_test==1], y_sub_test[y_sub_test==1])

    return pd.Series([overall / 100., 
                      class_0 / 100.,
                      class_1 / 100.],
                      index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])

score = lambda model, x_test, y_test: pd.Series([model.score(x_test, y_test), 
                                                 model.score(x_test[y_test==0], y_test[y_test==0]),
                                                 model.score(x_test[y_test==1], y_test[y_test==1])], 
                                                index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])


#KNN
knn = KNN(n_neighbors=2)
knn.fit(x, y)

knn_scores = score(knn, x, y)
print('knn')

#Unweighted logistic regression
unweighted_logistic = LogisticRegression(C=1000)
unweighted_logistic.fit(x, y)

unweighted_log_scores = score(unweighted_logistic, x, y)
print('unweighted log')


#Weighted logistic regression
weighted_logistic = LogisticRegression(C=1000, class_weight='balanced')
weighted_logistic.fit(x, y)

weighted_log_scores = score(weighted_logistic, x, y)
print('weighted log')


#LDA
lda = LDA()
lda.fit(x, y)

lda_scores = score(lda, x, y)
print('lda')

#QDA
qda = QDA()
qda.fit(x, y)

qda_scores = score(qda, x, y)
print('qda')

#Decision Tree
tree = DecisionTree(max_depth=50, class_weight='balanced', criterion='entropy')
tree.fit(x, y)

tree_scores = score(tree, x, y)
print('tree')


#Random Forest
rf = RandomForest(class_weight='balanced')
rf.fit(x, y)

rf_scores = score(rf, x, y)

print('rf')

#SVC
svc = SVC(C=100, class_weight='balanced')
svc.fit(x, y)

svc_scores = score(svc, x, y)

print('svc')


#Score Dataframe
score_df = pd.DataFrame({'knn': knn_scores, 
                         'unweighted logistic': unweighted_log_scores,
                         'weighted logistic': weighted_log_scores,
                         'lda': lda_scores,
                         'qda': qda_scores,
                         'tree': tree_scores,
                         'rf': rf_scores, 
                         'svc': svc_scores})
score_df


# It looks like we can rule out KNN, LDA and unweighted logistic. 
# 
# **What we do:** We are going to pick weighted logistic regression and just tune the regularization parameter to beat the test benchmarks.
# 
# **What's probably good to do:** QDA, random forest, tree, SVC and weighted logistic are beating our train benchmarks as is. We will tune them to beat the test benchmarks by picking the model and parameter set with the highest CV accuracy.
# 

Cs = 10.**np.arange(-3, 4, 1)
scores = []
for C in Cs:
    print('C:', C)
    weighted_log_scores = np.array([0., 0., 0.])
    kf = KFold(len(x), n_folds=10, shuffle=True, random_state=10)
    for train_index, test_index in kf:
        x_validate_train, x_validate_test = x[train_index], x[test_index]
        y_validate_train, y_validate_test = y[train_index], y[test_index]

        weighted_logistic = LogisticRegression(C=C, class_weight='balanced')
        weighted_logistic.fit(x_validate_train, y_validate_train)

        weighted_log_scores += score(weighted_logistic, x_validate_test, y_validate_test).values

    scores.append(weighted_log_scores / 10.)

scores = pd.DataFrame(np.array(scores).T, columns=[str(C) for C in Cs], index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])


scores


# To beat the future benchmark, we'll select the parameter which yields the highest accuracy on class 1 (while still beating the benchmark on class 0).
# 
# Now let's test our model on the test data:
# 

#Weighted logistic regression
weighted_logistic = LogisticRegression(C=100, class_weight='balanced')
weighted_logistic.fit(x, y)
weighted_log_scores = score(weighted_logistic, x_test, y_test)
weighted_log_scores


# Yay, we beat all the benchmarks!
# 

# # Part II: Diagnosing Strains of the Semian Flu
# 
# From a public health perspective, we want to balance the cost of vaccinations, early interventions and the cost of treating flu complications of unvaccinated people. 
# 
# There are two different strains of the flu: strain 1 has a cheaper early intervention as well as a cheaper treatment for flu complications, but patients with strain 1 has a higher rate of developing complications if treated with the wrong intervention. Strain 2 has a more expensive early intervention as well as a more costly treatment for flu complications, but patients with strain 2 has a lower rate of developing complications if treated with the wrong intervention. With no intervention, flu patients develop complications at the same rate regardless of the strain. 
# 
# **Your task:** build a model to predict if a given patient has the flu and identify the flu strain. The state government of MA will use your model to inform public health policies: we will vaccinate people you've identified as healthy and apply corresponding interventions to patients with different strains of the flu. We have provided you with a function to compute the total expected cost of this policy decision that takes into account the cost of the vaccine, the interventions and the cost of the treatments for flu complications resulting from misdiagnosing patients. Your goal is to make sure your model produces a public health policy with the lowest associated expected cost.
# 
# **The deliverable:** a function called `flu_predict` which satisfies:
# 
# - input: `x_test`, a set of medical predictors for a group of patients
# - output: `y_pred`, a set of labels, one for each patient; 1 for healthy, 2 for infected with strain 1, and 3 for infected with strain 2.
# 
# The MA state government will use your model to diagnose sets of future patients (held by us). You can expect that there will be an increase in the number of flu patients in any groups of patients in the future.
# 
# We provide you with some benchmarks for comparison.
# 
# **Three Baseline Models:** 
# - expected cost on observed data: \$6,818,206.0, \$7,035,735.0, \$8,297,197.5
# - time to build: 1 min
# 
# **Reasonable Model:** 
# - expected cost on observed data: $6,300,000
# - time to build: 20 min
# 
# **Grading:**
# Your grade will be based on:
# 1. your model's ability to out-perform our benchmarks
# 2. your ability to carefully and thoroughly follow the data science pipeline (see lecture slides for definition)
# 3. the extend to which all choices are reasonable and defensible by methods you have learned in this class
# 

#--------  cost
# A function that computes the expected cost of the public healthy policy based on the 
# classifications generated by your model
# Input: 
#      y_true (true class labels: 0, 1, 2)
#      y_pred (predicted class labels: 0, 1, 2)
# Returns: 
#      total_cost (expected total cost)

def cost(y_true, y_pred):
    cost_of_treatment_1 = 29500
    cost_of_treatment_2 = 45000
    cost_of_intervention_1 = 4150
    cost_of_intervention_2 = 4250
    cost_of_vaccine = 15
    
    prob_complications_untreated = 0.65
    prob_complications_1 = 0.30
    prob_complications_2 = 0.15
    
    trials = 1000    
    
    intervention_cost = cost_of_intervention_1 * len(y_pred[y_pred==1]) + cost_of_intervention_2 * len(y_pred[y_pred==2])

    vaccine_cost = cost_of_vaccine * len(y_pred[y_pred==0])
    
    false_neg_1 = ((y_true == 1) & (y_pred == 2)).sum()
    false_neg_2 = ((y_true == 2) & (y_pred == 1)).sum()
    
    untreated_1 = ((y_true == 1) & (y_pred == 0)).sum()    
    untreated_2 = ((y_true == 2) & (y_pred == 0)).sum()
    
    false_neg_1_cost = np.random.binomial(1, prob_complications_1, (false_neg_1, trials)) * cost_of_treatment_1
    false_neg_2_cost = np.random.binomial(1, prob_complications_2, (false_neg_2, trials)) * cost_of_treatment_2
    untreated_1_cost = np.random.binomial(1, prob_complications_untreated, (untreated_1, trials)) * cost_of_treatment_1
    untreated_2_cost = np.random.binomial(1, prob_complications_untreated, (untreated_2, trials)) * cost_of_treatment_2
    
    false_neg_1_cost = false_neg_1_cost.sum(axis=0)
    expected_false_neg_1_cost = false_neg_1_cost.mean()
    
    false_neg_2_cost = false_neg_2_cost.sum(axis=0)
    expected_false_neg_2_cost = false_neg_2_cost.mean()
    
    untreated_1_cost = untreated_1_cost.sum(axis=0)
    expected_untreated_1_cost = untreated_1_cost.mean()
    
    untreated_2_cost = untreated_2_cost.sum(axis=0)
    expected_untreated_2_cost = untreated_2_cost.mean()
    
    total_cost = vaccine_cost + intervention_cost + expected_false_neg_1_cost + expected_false_neg_2_cost + expected_untreated_1_cost + expected_untreated_2_cost
    
    return total_cost


# We're just going to take the weighted logistic model, again, and tune the regularization parameter to both beat the benchmark on the observed data and minimize expected cost on unseen data (i.e. prevent ***overfitting***). Instead of using 'balanced' class weights, we're using a custom weighting scheme for the three classes (this parameter should really be tuned!).
# 
# It would probally also be go through the whole "choosing a model, tuning these models"-process again, this time to minimize cost.
# 
# **Note:** Be aware that the cost is now sensitive to sample size! The smaller the pool of patients the less the cost. If you are evaluating cost on a held-out test set then you can artificially make the cost very small. The benchmarks we give are for the entire training set.
# 

x = df.values[:, :-2]
y = df.values[:, -1]
y = y - 1

x_test = df_test.values[:, :-2]
y_test = df_test.values[:, -1]

y_test = y_test - 1


score = lambda model, x_test, y_test: pd.Series([model.score(x_test, y_test), 
                                                 model.score(x_test[y_test==0], y_test[y_test==0]),
                                                 model.score(x_test[y_test==1], y_test[y_test==1]), 
                                                 model.score(x_test[y_test==2], y_test[y_test==2]), 
                                                 cost(y_test, model.predict(x_test))],
                                                index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1', 'accuracy on class 2', 'total cost'])


Cs = 10.**np.arange(-3, 4, 1)
scores = []
for C in Cs:
    print('C:', C)
    weighted_log_scores = np.array([0., 0., 0., 0., 0.])
    kf = KFold(len(x), n_folds=10, shuffle=True, random_state=10)
    for train_index, test_index in kf:
        x_validate_train, x_validate_test = x[train_index], x[test_index]
        y_validate_train, y_validate_test = y[train_index], y[test_index]

        weighted_logistic = LogisticRegression(C=C, class_weight={0:0.7, 1:10, 2:10})
        weighted_logistic.fit(x_validate_train, y_validate_train)

        weighted_log_scores += score(weighted_logistic, x_validate_test, y_validate_test).values

    scores.append(weighted_log_scores / 10.)

scores = pd.DataFrame(np.array(scores).T, columns=[str(C) for C in Cs], index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1', 'accuracy on class 2', 'total cost'])


scores


#Weighted logistic regression
weighted_logistic = LogisticRegression(C=100, class_weight={0:0.7, 1:10, 2:10})
weighted_logistic.fit(x, y)
weighted_log_scores = score(weighted_logistic, x, y)
weighted_log_scores


#Weighted logistic regression
weighted_log_scores = score(weighted_logistic, x_test, y_test)
weighted_log_scores


print('minimimum cost on train:', cost(y, y))
print('minimimum cost on test:', cost(y_test, y_test))


print('simple model cost on train:', cost(y, np.array([0] * len(y))))
print('simple model cost on test:', cost(y_test, np.array([0] * len(y_test))))


print('simple model cost on train:', cost(y, np.array([1] * len(y))))
print('simple model cost on test:', cost(y_test, np.array([1] * len(y_test))))


print('simple model cost on train:', cost(y, np.array([2] * len(y))))
print('simple model cost on test:', cost(y_test, np.array([2] * len(y_test))))


# Yay! We beat the benchmarks on the observed data and did pretty good on test data!
# 

# # CS 109A/AC 209A/STAT 121A Data Science: Midterm #1 (October 2016)
# **Harvard University**<br>
# **Fall 2016**<br>
# **Instructors: W. Pan, P. Protopapas, K. Rader**<br>
# 

# It is Oct 13, 2016. NASA’s radars discovered a small, 3 meter iron base meteorite, that just entered the Earth’s atmosphere. A small meteorite will not create a big devastation but still dangerous for the citizens. Local authorities would like to know the location of the impact point so they can warn people and allocate resources based on the population that is affected.
# The Governor has sought out the best data scientist in the state, you, to help save the day!
# 
# You are given two datasets:
# - Radar position estimates (x,y,z; z being the altitude) of the meteorite at various times are released on a web page (URL [here](https://cs109alabs.github.io/lab_files/)). x,y and z are coordinates in kilometers and time is in seconds.
# - Locations and other details of every dwelling in the town can be found in the file *midtermbuildings.csv*.
# 
# 
# 1. Using methods you learned in class estimate the expected point of impact along with the region with 90% certainty.
# 2. Using the dwelling database, estimate the total number of people that will most likely be affected within this region.
# 
# **AC209a students only**: Additional measurements from another radar are released in the file *midterm_a_r2_d1.csv*. The accuracy of this radar is approximately 5 times higher than the first radar. Your model should take into account radar data sets.
# 

# Your code here...


