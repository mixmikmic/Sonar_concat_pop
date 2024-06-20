# ### From the Titanic Dataset
# ![](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)
# <div style='text-align:right;'>By Stephen Milborrow (Own work) [CC BY-SA 3.0 via Wikimedia Commons]</div>

import mglearn # credits to Muller and Guido (https://www.amazon.com/dp/1449369413/)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mglearn.plots.plot_tree_not_monotone()


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))





# ## <center> One-Hot Encoding</center>
# 
# - used on categorical variables
# - it replaces a categorical variable/feature with one or more new features that will take the values of 0 or 1
# - increases data burden
# - increases the efficiency of the process
# 

import pandas as pd
from IPython.display import display

data = pd.read_csv('adult.data', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education', 
                                                                      'education-num', 'marital-status', 'occupation', 
                                                                      'relationship', 'race', 'gender', 'capital-gain', 
                                                                      'capital-loss', 'hours-per-week', 'native-country', 
                                                                      'income'])


data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
display(data)


print('Original Features:\n', list(data.columns), '\n')
data_dummies = pd.get_dummies(data)
print('Features after One-Hot Encoding:\n', list(data_dummies.columns))


features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Logistic Regression score on the test set: {:.2f}'.format(logreg.score(X_test, y_test)))





# ## <center> Neural Networks in scikit-learn </center>
# 
# #### Linear models:
# 
# <center> ŷ = w[0] \* x[0] + w[1] \* x[1] + ... + w[p] \* x[p] + b </center>
# 

import mglearn

mglearn.plots.plot_logistic_regression_graph()


mglearn.plots.plot_single_hidden_layer_graph()


mglearn.plots.plot_two_hidden_layer_graph()








# ## <center> One-Hot Encoding</center>
# 
# - used on categorical variables
# - it replaces a categorical variable/feature with one or more new features that will take the values of 0 or 1
# - increases data burden
# - increases the efficiency of the process
# 

import pandas as pd
from IPython.display import display

data = pd.read_csv('adult.data', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education', 
                                                                      'education-num', 'marital-status', 'occupation', 
                                                                      'relationship', 'race', 'gender', 'capital-gain', 
                                                                      'capital-loss', 'hours-per-week', 'native-country', 
                                                                      'income'])


data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
display(data)


print('Original Features:\n', list(data.columns), '\n')
data_dummies = pd.get_dummies(data)
print('Features after One-Hot Encoding:\n', list(data_dummies.columns))








from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import mglearn

get_ipython().magic('matplotlib inline')


cancer = load_breast_cancer()


# Knowledge Gathering 

#print(cancer.DESCR)
#cancer.data
#cancer.data.shape
#print(cancer.feature_names)
#print(cancer.target_names)


# <center> 
# # Process Outline (for many ML projects)
# #### 1. Get the data (pre-process it)
# #### 2. Pick an algorithm (classifier)
# #### 3. Train the algorithm. Verify accuracy. Optimize.
# #### 4. Predict 
# 

# Looking into the raw dataset (not pre-processed like the one that comes with scikit-learn)

import pandas as pd
raw_data=pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',')
#raw_data.tail(10)


# KNN Classifier Overview

mglearn.plots.plot_knn_classification(n_neighbors=3)


from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))





# ## <center>Supervised Learning</center>
# 
# - making inferences from labeled data.
# 
# #### 1. Classification (categorical data)
# - binary classification (tumor: benign, malignant)
# - multiclass classification (books: maths, physics, stats, psychology, etc.)
# - example algorithms: KNN, Linear Models, Decision Trees, SVMs, etc.
# 
# #### 2. Regression (continuous data)
# - predicting income, price of stock, age, and other continous data 
# - example algorithms: KNN, Linear Models, Decision Trees, SVMs, etc.
# ___
# 
# Linear models (LinReg, LogReg, Lasso, Ridged, etc) - make predictions according to a linear function of the input features. <br>
# Many ML algorithms (including those specified above) can be used for both classification and regression.
# 

# Using LogisticRegression on the cancer dataset. Inspired by Muller and Guido ML book: (https://www.amazon.com/dp/1449369413/)

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


print('Accuracy on the training subset: {:.3f}'.format(log_reg.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg.score(X_test, y_test)))


# ## <center> Neural Networks in scikit-learn </center>
# 
# #### Linear models:
# 
# <center> ŷ = w[0] \* x[0] + w[1] \* x[1] + ... + w[p] \* x[p] + b </center>
# 

import mglearn
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mglearn.plots.plot_logistic_regression_graph()


mglearn.plots.plot_single_hidden_layer_graph()


mglearn.plots.plot_two_hidden_layer_graph()


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test, y_test)))


print('The maximum per each feature:\n{}'.format(cancer.data.max(axis=0)))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)

mlp = MLPClassifier(max_iter=1000, random_state=42)   

mlp.fit(X_train_scaled, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))


mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp.fit(X_train_scaled, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))


plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0], interpolation='None', cmap='GnBu')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()





# # <center> Automatic Feature Selection </center>
# 
# - <b><span style='color:green'>to reduce dimensionality<span></b>
# - common methods: univariate statistics, model-based selection, iterative selection
# 

# ### 1. Univariate Statistics
# 
# - determines the relationship between each feature and output (target)
# - only the features with highest confidence are selected
# - <b><span style='color:blue'>SelectKBest</span></b> - selecting K number of features
# - <b><span style='color:blue'>SelectPercentile</span></b> - selection is made based on a percentage of the original features
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print('X_train.shape is: {}'.format(X_train.shape))
print('X_train_selected.shape is: {}'.format(X_train_selected.shape))


mask = select.get_support()
print(mask)
plt.matshow(mask.reshape(1,-1), cmap='gray_r')


X_test_selected = select.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('The score of Logistic Regression on all features: {:.3f}'.format(logreg.score(X_test, y_test)))

logreg.fit(X_train_selected, y_train)
print('The score of Logistic Regression on the selected features: {:.3f}'.format(logreg.score(X_test_selected, y_test)))


# ### 2. Model-Based Feature Selection
# 
# - uses a supervised model to determine the importance of each feature
# - keeps the most important features
# - needs a measure for the importance of features (DT and RF have the 'feature_importances' attribute)
# 

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')


select.fit(X_train, y_train)
X_train_s = select.transform(X_train)
print('The shape of X_train is: ', X_train.shape)
print('The shape of X_train_s is ', X_train_s.shape)


mask = select.get_support()
plt.matshow(mask.reshape(1,-1), cmap='gray_r')
plt.xlabel('Index of Features')


X_test_s = select.transform(X_test)
score = LogisticRegression().fit(X_train_s, y_train).score(X_test_s, y_test)
print('The score of Logistic Regression with the selected features on the test set: {:.3f}'.format(score))





# # <center> Automatic Feature Selection </center>
# 
# - <b><span style='color:green'>to reduce dimensionality<span></b>
# - common methods: univariate statistics, model-based selection, iterative selection
# 

# ### 1. Univariate Statistics
# 
# - determines the relationship between each feature and output (target)
# - only the features with highest confidence are selected
# - <b><span style='color:blue'>SelectKBest</span></b> - selecting K number of features
# - <b><span style='color:blue'>SelectPercentile</span></b> - selection is made based on a percentage of the original features
# 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile

cancer = load_breast_cancer()

rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print('X_train.shape is: {}'.format(X_train.shape))
print('X_train_selected.shape is: {}'.format(X_train_selected.shape))








# ### From the Titanic Dataset
# ![](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)
# <div style='text-align:right;'>By Stephen Milborrow (Own work) [CC BY-SA 3.0 via Wikimedia Commons]</div>

import mglearn # credits to Muller and Guido (https://www.amazon.com/dp/1449369413/)
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

mglearn.plots.plot_tree_not_monotone()


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


import graphviz
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='cancertree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names,
               impurity=False, filled=True)


# ![](cancertree.png)

print('Feature importances: {}'.format(tree.feature_importances_))
type(tree.feature_importances_)


print(cancer.feature_names)


n_features = cancer.data.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()


# ### Advantages of Decision Trees
# 
#  - easy to view and understand
#  - no need to pre-process, normalize, scale, and/or standardize features
#  
# ### Paramaters to work with
#  
#  - max_depth
#  - min_samples_leaf, max_samples_leaf
#  - max_leaf_nodes
#  - etc.
#  
# ### Main Disadvantages
# 
# - tendency to overfit
# - poor generalization 
# 
# ####  <center> Possible work-around: ensembles of decision trees </center>
# 




from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import mglearn

get_ipython().magic('matplotlib inline')


cancer = load_breast_cancer()


# Knowledge Gathering 

#print(cancer.DESCR)
#cancer.data
#cancer.data.shape
#print(cancer.feature_names)
#print(cancer.target_names)


# <center> 
# # Process Outline (for many ML projects)
# #### 1. Get the data (pre-process it)
# #### 2. Pick an algorithm (classifier)
# #### 3. Train the algorithm. Verify accuracy. Optimize.
# #### 4. Predict 
# 

# Looking into the raw dataset (not pre-processed like the one that comes with scikit-learn)

import pandas as pd
raw_data=pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',')
#raw_data.tail(10)


# KNN Classifier Overview

mglearn.plots.plot_knn_classification(n_neighbors=3)


from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))


# Resplit the data, with a different randomization (inspired by Muller & Guido ML book - https://www.amazon.com/dp/1449369413/)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# Create two lists for training and test accuracies
training_accuracy = []
test_accuracy = []

# Define a range of 1 to 10 (included) neighbors to be tested
neighbors_settings = range(1,11)

# Loop with the KNN through the different number of neighbors to determine the most appropriate (best)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)
plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend()








# ## <center> Neural Networks in scikit-learn </center>
# 
# #### Linear models:
# 
# <center> ŷ = w[0] \* x[0] + w[1] \* x[1] + ... + w[p] \* x[p] + b </center>
# 

import mglearn
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mglearn.plots.plot_logistic_regression_graph()


mglearn.plots.plot_single_hidden_layer_graph()


mglearn.plots.plot_two_hidden_layer_graph()


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test, y_test)))


print('The maximum per each feature:\n{}'.format(cancer.data.max(axis=0)))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)

mlp = MLPClassifier(max_iter=1000, random_state=42)   

mlp.fit(X_train_scaled, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))


mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp.fit(X_train_scaled, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))


plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0], interpolation='None', cmap='GnBu')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()


# 
# 
# 
# 
# 
# ## <center> Advantages and Disadvantages of Neural Nets (scikit-learn) </center>
# 

# ### Stronger points:
#  - can be used efficiently on large datasets
#  - can build very complex models
#  - many parameters for tuning
#  - flexibility and rapid prototyping
#  - etc.
#  
# ### Weaker points:
#  - many parameters for tuning
#  - some solvers are scale sensitive
#  - data may need to be pre-processed
#  - etc.
#  
#  
# ### Alternatives:
#  
#  - theano
#  - tensorflow
#  - keras
#  - lasagna
#  - etc. 
# 




# # <center> Automatic Feature Selection </center>
# 
# - <b><span style='color:green'>to reduce dimensionality<span></b>
# - common methods: univariate statistics, model-based selection, iterative selection
# 

# ### 1. Univariate Statistics
# 
# - determines the relationship between each feature and output (target)
# - only the features with highest confidence are selected
# - <b><span style='color:blue'>SelectKBest</span></b> - selecting K number of features
# - <b><span style='color:blue'>SelectPercentile</span></b> - selection is made based on a percentage of the original features
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print('X_train.shape is: {}'.format(X_train.shape))
print('X_train_selected.shape is: {}'.format(X_train_selected.shape))


mask = select.get_support()
print(mask)
plt.matshow(mask.reshape(1,-1), cmap='gray_r')


X_test_selected = select.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('The score of Logistic Regression on all features: {:.3f}'.format(logreg.score(X_test, y_test)))

logreg.fit(X_train_selected, y_train)
print('The score of Logistic Regression on the selected features: {:.3f}'.format(logreg.score(X_test_selected, y_test)))





from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import mglearn

get_ipython().magic('matplotlib inline')


cancer = load_breast_cancer()
#print(cancer.DESCR)


print(cancer.feature_names)
print(cancer.target_names)


#cancer.data


cancer.data.shape


# <center> 
# # Process Outline
# #### 1. Get the data (pre-process it)
# #### 2. Pick an algorithm (classifier)
# #### 3. Train the algorithm. Verify accuracy. Optimize.
# #### 4. Predict 
# 

import pandas as pd
raw_data=pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',')
#raw_data.tail(10)


# KNN Classifier Overview

mglearn.plots.plot_knn_classification(n_neighbors=3)














# ### From the Titanic Dataset
# ![](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)
# <div style='text-align:right;'>By Stephen Milborrow (Own work) [CC BY-SA 3.0 via Wikimedia Commons]</div>

import mglearn # credits to Muller and Guido (https://www.amazon.com/dp/1449369413/)
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

mglearn.plots.plot_tree_not_monotone()


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


import graphviz
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='cancertree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names,
               impurity=False, filled=True)


# ![](cancertree.png)

print('Feature importances: {}'.format(tree.feature_importances_))
type(tree.feature_importances_)


print(cancer.feature_names)


n_features = cancer.data.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()








# ## <center>Supervised Learning</center>
# 
# - making inferences from labeled data.
# 
# #### 1. Classification (categorical data)
# - binary classification (tumor: benign, malignant)
# - multiclass classification (books: maths, physics, stats, psychology, etc.)
# - example algorithms: KNN, Linear Models, Decision Trees, SVMs, etc.
# 
# #### 2. Regression (continuous data)
# - predicting income, price of stock, age, and other continous data 
# - example algorithms: KNN, Linear Models, Decision Trees, SVMs, etc.
# ___
# 
# Linear models (LinReg, LogReg, Lasso, Ridged, etc) - make predictions according to a linear function of the input features. <br>
# Many ML algorithms (including those specified above) can be used for both classification and regression.
# 

# Using LogisticRegression on the cancer dataset. Inspired by Muller and Guido ML book: (https://www.amazon.com/dp/1449369413/)

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


print('Accuracy on the training subset: {:.3f}'.format(log_reg.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg.score(X_test, y_test)))


# **Regularization**:
# 
# - prevention of overfitting - (according to Muller and Guido ML book)
# - L1 - assumes only a few features are important
# - L2 - does not assume only a few features are important - used by default in scikit-learn LogisticRegression
#                
# **'C'**:
# 
# - parameter to control the strength of regularization
# - lower C => log_reg adjusts to the majority of data points.
# - higher C => correct classification of each data point.
# 

log_reg100 = LogisticRegression(C=100)
log_reg100.fit(X_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(log_reg100.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg100.score(X_test, y_test)))


log_reg001 = LogisticRegression(C=0.01)
log_reg001.fit(X_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(log_reg001.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg001.score(X_test, y_test)))


# ### Linear Models (in general): 
# 
# #### <center> y = w * x + b </center>
# 
# - w - slope (or coefficient) - accessed via <raw>.coef_</raw>
# - b - offset (or intercept) - access via <raw>.intercept_</raw>
# - w and b are learned parameters
# - y - prediction (decision)
# 
# Example (for a dataset with only 1 input features): ŷ = w[0] * x[0] + b 
# 
# ### For Logistic Regression (specifically):
# 
# #### <center> ŷ = w[0] \* x[0] + w[1] \* x[1] + ... + w[p] \* x[p] + b > 0</center>
# 

import mglearn # credits to Muller and Guido 2016 (link above)
mglearn.plots.plot_linear_regression_wave()


plt.plot(log_reg.coef_.T, 'o', label='C=1')
plt.plot(log_reg100.coef_.T, '^', label='C=100')
plt.plot(log_reg001.coef_.T, 'v', label='C=0.01')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0,0, cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Magnitude')
plt.legend()





# ## <center> Neural Networks in scikit-learn </center>
# 
# #### Linear models:
# 
# <center> ŷ = w[0] \* x[0] + w[1] \* x[1] + ... + w[p] \* x[p] + b </center>
# 

import mglearn

mglearn.plots.plot_logistic_regression_graph()


mglearn.plots.plot_single_hidden_layer_graph()


mglearn.plots.plot_two_hidden_layer_graph()


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test, y_test)))








from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(forest.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(forest.score(X_test, y_test)))


n_features = cancer.data.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()


# ### Potential Advantages of Random Forests
# 
#  - powerful and widely implemented
#  - perform well with default settings
#  - dont require scaling of the data
#  - randomization makes them better than single DT
#  
# ### Parameters to Tune
#  
#  - n_jobs - number of cores to use for training (n_jobs=-1, for all cores)
#  - n_estimators - how many trees to use (more is always better)
#  - max_depth, for pre-pruning
#  - max_features, for randomization
#      - max_features = sqrt(n_features), for classification
#      - max_features = log2(n_features), for regression
#  - etc.
#  
# ### Potential Disadvantages of Random Forests
# 
# - not so good performance on very high dimensional and sparse data (text data)
# - large datasets require more resources for training (time, CPUs, etc).
# - cannot be visualized as well as single DT
# 




from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


cancer = load_breast_cancer()
print(cancer.DESCR)


print(cancer.feature_names)
print(cancer.target_names)


cancer.data


cancer.data.shape


# <center> 
# # Process Outline
# #### 1. Get the data (pre-process it)
# #### 2. Pick an algorithm (classifier)
# #### 3. Train the algorithm. Verify accuracy. Optimize.
# #### 4. Predict 
# 

import pandas as pd
raw_data=pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',')
raw_data.tail(10)





# ## <center> Neural Networks in scikit-learn </center>
# 
# #### Linear models:
# 
# <center> ŷ = w[0] \* x[0] + w[1] \* x[1] + ... + w[p] \* x[p] + b </center>
# 

import mglearn

mglearn.plots.plot_logistic_regression_graph()


mglearn.plots.plot_single_hidden_layer_graph()


mglearn.plots.plot_two_hidden_layer_graph()


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test, y_test)))


print('The maximum per each feature:\n{}'.format(cancer.data.max(axis=0)))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)

mlp = MLPClassifier(max_iter=1000, random_state=42)   

mlp.fit(X_train_scaled, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))


mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp.fit(X_train_scaled, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))





# ## <center> Neural Networks in scikit-learn </center>
# 
# #### Linear models:
# 
# <center> ŷ = w[0] \* x[0] + w[1] \* x[1] + ... + w[p] \* x[p] + b </center>
# 

import mglearn

mglearn.plots.plot_logistic_regression_graph()


mglearn.plots.plot_single_hidden_layer_graph()


mglearn.plots.plot_two_hidden_layer_graph()


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test, y_test)))


print('The maximum per each feature:\n{}'.format(cancer.data.max(axis=0)))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)

mlp = MLPClassifier(max_iter=1000, random_state=42)   

mlp.fit(X_train_scaled, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))





# ## <center> Preprocessing Methods </center>
# 
# - binarization
# - scaling
# - normalization
# - mean removal
# - etc.
# 
# ### 1. Binarization
# 

from sklearn import preprocessing
import numpy as np 

data = np.array([[2.2, 5.9, -1.8], [5.4, -3.2, -5.1], [-1.9, 4.2, 3.2]])


bindata = preprocessing.Binarizer(threshold=1.5).transform(data)
print('Binarized data:\n\n', bindata)








# ### From the Titanic Dataset
# ![](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)
# <div style='text-align:right;'>By Stephen Milborrow (Own work) [CC BY-SA 3.0 via Wikimedia Commons]</div>

import mglearn # credits to Muller and Guido (https://www.amazon.com/dp/1449369413/)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mglearn.plots.plot_tree_not_monotone()


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


import graphviz
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='cancertree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names,
               impurity=False, filled=True)


# ![](cancertree.png)




# ## <center>Supervised Learning</center>
# 
# - making inferences from labeled data.
# 
# #### 1. Classification (categorical data)
# - binary classification (tumor: benign, malignant)
# - multiclass classification (books: maths, physics, stats, psychology, etc.)
# - example algorithms: KNN, Linear Models, Decision Trees, SVMs, etc.
# 
# #### 2. Regression (continuous data)
# - predicting income, price of stock, age, and other continous data 
# - example algorithms: KNN, Linear Models, Decision Trees, SVMs, etc.
# ___
# 
# Linear models (LinReg, LogReg, Lasso, Ridged, etc) - make predictions according to a linear function of the input features. <br>
# Many ML algorithms (including those specified above) can be used for both classification and regression.
# 

# Using LogisticRegression on the cancer dataset. Inspired by Muller and Guido ML book: (https://www.amazon.com/dp/1449369413/)

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


print('Accuracy on the training subset: {:.3f}'.format(log_reg.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg.score(X_test, y_test)))


# **Regularization**:
# 
# - prevention of overfitting - (according to Muller and Guido ML book)
# - L1 - assumes only a few features are important
# - L2 - does not assume only a few features are important - used by default in scikit-learn LogisticRegression
#                
# **'C'**:
# 
# - parameter to control the strength of regularization
# - lower C => log_reg adjusts to the majority of data points.
# - higher C => correct classification of each data point.
# 

log_reg100 = LogisticRegression(C=100)
log_reg100.fit(X_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(log_reg100.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg100.score(X_test, y_test)))


log_reg001 = LogisticRegression(C=0.01)
log_reg001.fit(X_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(log_reg001.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg001.score(X_test, y_test)))





# ### From the Titanic Dataset
# ![](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)
# <div style='text-align:right;'>By Stephen Milborrow (Own work) [CC BY-SA 3.0 via Wikimedia Commons]</div>

import mglearn # credits to Muller and Guido (https://www.amazon.com/dp/1449369413/)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mglearn.plots.plot_tree_not_monotone()


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))





