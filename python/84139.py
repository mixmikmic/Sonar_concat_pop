# ## Pipelines
# 
# Pipeline can be used to chain multiple estimators into one. This is useful as there is often a fixed sequence of steps in processing the data, for example feature selection, normalization and classification. Pipeline serves two purposes here:
# 
# * Convenience: You only have to call fit and predict once on your data to fit a whole sequence of estimators.
# * Joint parameter selection: You can grid search over parameters of all estimators in the pipeline at once.
# 
# All estimators in a pipeline, except the last one, must be transformers (i.e. must have a transform method). The last estimator may be any type (transformer, classifier, etc.).
# 

from sklearn.pipeline import Pipeline

get_ipython().magic('pinfo Pipeline')


from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA(n_components=2)), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe 


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)


# Notice no need to PCA the Xs in the score!
pipe.fit(X, y).score(X, y)


# The utility function make_pipeline is a shorthand for constructing pipelines; it takes a variable number of estimators and returns a pipeline, filling in the names automatically:
# 

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
make_pipeline(Binarizer(), MultinomialNB()) 


pipe.steps[0]


pipe.named_steps['reduce_dim']


pipe.set_params(clf__C=10) 


from sklearn.model_selection import GridSearchCV
params = dict(reduce_dim__n_components=[2, 5, 10],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)


from sklearn.linear_model import LogisticRegression
params = dict(reduce_dim=[None, PCA(5), PCA(10)],
              clf=[SVC(), LogisticRegression()],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)


# ## Feature Union
# 
# FeatureUnion combines several transformer objects into a new transformer that combines their output. A FeatureUnion takes a list of transformer objects. During fitting, each of these is fit to the data independently. For transforming data, the transformers are applied in parallel, and the sample vectors they output are concatenated end-to-end into larger vectors.
# 
# FeatureUnion serves the same purposes as Pipeline - convenience and joint parameter estimation and validation.
# 
# FeatureUnion and Pipeline can be combined to create complex models.
# 
# (A FeatureUnion has no way of checking whether two transformers might produce identical features. It only produces a union when the feature sets are disjoint, and making sure they are the caller’s responsibility.)
# 

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
combined 


combined.fit_transform(X).shape


combined.set_params(kernel_pca=None) 


combined.fit_transform(X).shape





# # Model evaluation
# 
# There are 3 different approaches to evaluate the quality of predictions of a model:
# * Estimator score method: Estimators have a score method providing a default evaluation criterion for the problem they are designed to solve. This is not discussed on this page, but in each estimator’s documentation.
# * Scoring parameter: Model-evaluation tools using cross-validation (such as model_selection.cross_val_score and model_selection.GridSearchCV) rely on an internal scoring strategy. This is discussed in the section The scoring parameter: defining model evaluation rules.
# * Metric functions: The metrics module implements functions assessing prediction error for specific purposes. These metrics are detailed in sections on Classification metrics, Multilabel ranking metrics, Regression metrics and Clustering metrics.
# 
# Finally, Dummy estimators are useful to get a baseline value of those metrics for random predictions.
# 

# ## The scoring parameter
# 
# Model selection and evaluation using tools, such as model_selection.GridSearchCV and model_selection.cross_val_score, take a scoring parameter that controls what metric they apply to the estimators evaluated.
# 
# For the most common use cases, you can designate a scorer object with the scoring parameter; the table below shows all possible values. All scorer objects follow the convention that higher return values are better than lower return values. Thus metrics which measure the distance between the model and the data, like metrics.mean_squared_error, are available as neg_mean_squared_error which return the negated value of the metric.
# 
# <table border="1" class="docutils">
# <colgroup>
# <col width="26%">
# <col width="40%">
# <col width="33%">
# </colgroup>
# <thead valign="bottom">
# <tr class="row-odd"><th class="head">Scoring</th>
# <th class="head">Function</th>
# <th class="head">Comment</th>
# </tr>
# </thead>
# <tbody valign="top">
# <tr class="row-even"><td><strong>Classification</strong></td>
# <td>&nbsp;</td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-odd"><td>‘accuracy’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score" title="sklearn.metrics.accuracy_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.accuracy_score</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-even"><td>‘average_precision’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score" title="sklearn.metrics.average_precision_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.average_precision_score</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-odd"><td>‘f1’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score" title="sklearn.metrics.f1_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.f1_score</span></code></a></td>
# <td>for binary targets</td>
# </tr>
# <tr class="row-even"><td>‘f1_micro’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score" title="sklearn.metrics.f1_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.f1_score</span></code></a></td>
# <td>micro-averaged</td>
# </tr>
# <tr class="row-odd"><td>‘f1_macro’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score" title="sklearn.metrics.f1_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.f1_score</span></code></a></td>
# <td>macro-averaged</td>
# </tr>
# <tr class="row-even"><td>‘f1_weighted’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score" title="sklearn.metrics.f1_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.f1_score</span></code></a></td>
# <td>weighted average</td>
# </tr>
# <tr class="row-odd"><td>‘f1_samples’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score" title="sklearn.metrics.f1_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.f1_score</span></code></a></td>
# <td>by multilabel sample</td>
# </tr>
# <tr class="row-even"><td>‘neg_log_loss’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss" title="sklearn.metrics.log_loss"><code class="xref py py-func docutils literal"><span class="pre">metrics.log_loss</span></code></a></td>
# <td>requires <code class="docutils literal"><span class="pre">predict_proba</span></code> support</td>
# </tr>
# <tr class="row-odd"><td>‘precision’ etc.</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score" title="sklearn.metrics.precision_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.precision_score</span></code></a></td>
# <td>suffixes apply as with ‘f1’</td>
# </tr>
# <tr class="row-even"><td>‘recall’ etc.</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score" title="sklearn.metrics.recall_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.recall_score</span></code></a></td>
# <td>suffixes apply as with ‘f1’</td>
# </tr>
# <tr class="row-odd"><td>‘roc_auc’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score" title="sklearn.metrics.roc_auc_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.roc_auc_score</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-even"><td><strong>Clustering</strong></td>
# <td>&nbsp;</td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-odd"><td>‘adjusted_rand_score’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score" title="sklearn.metrics.adjusted_rand_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.adjusted_rand_score</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-even"><td><strong>Regression</strong></td>
# <td>&nbsp;</td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-odd"><td>‘neg_mean_absolute_error’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error" title="sklearn.metrics.mean_absolute_error"><code class="xref py py-func docutils literal"><span class="pre">metrics.mean_absolute_error</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-even"><td>‘neg_mean_squared_error’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error" title="sklearn.metrics.mean_squared_error"><code class="xref py py-func docutils literal"><span class="pre">metrics.mean_squared_error</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-odd"><td>‘neg_median_absolute_error’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error" title="sklearn.metrics.median_absolute_error"><code class="xref py py-func docutils literal"><span class="pre">metrics.median_absolute_error</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# <tr class="row-even"><td>‘r2’</td>
# <td><a class="reference internal" href="generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score" title="sklearn.metrics.r2_score"><code class="xref py py-func docutils literal"><span class="pre">metrics.r2_score</span></code></a></td>
# <td>&nbsp;</td>
# </tr>
# </tbody>
# </table>
# 

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(probability=True, random_state=0)
print cross_val_score(clf, X, y, scoring='neg_log_loss') 

model = svm.SVC()
cross_val_score(model, X, y, scoring='wrong_choice')


# ####  Defining your scoring strategy from metric functions
# 
# The module sklearn.metric also exposes a set of simple functions measuring a prediction error given ground truth and prediction:
# 
# * functions ending with _score return a value to maximize, the higher the better.
# * functions ending with _error or _loss return a value to minimize, the lower the better. When converting into a scorer object using make_scorer, set the greater_is_better parameter to False (True by default; see the parameter description below).
# 
# Metrics available for various machine learning tasks are detailed in sections below.
# 
# Many metrics are not given names to be used as scoring values, sometimes because they require additional parameters, such as fbeta_score. In such cases, you need to generate an appropriate scoring object. The simplest way to generate a callable object for scoring is by using make_scorer. That function converts metrics into callables that can be used for model evaluation.
# 
# One typical use case is to wrap an existing metric function from the library with non-default values for its parameters, such as the beta parameter for the fbeta_score function:
# 

from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer)


# The second use case is to build a completely custom scorer object from a simple python function using make_scorer, which can take several parameters:
# 
# * the python function you want to use (my_custom_loss_func in the example below)
# * whether the python function returns a score (greater_is_better=True, the default) or a loss (greater_is_better=False). If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
# * for classification metrics only: whether the python function you provided requires continuous decision certainties (needs_threshold=True). The default value is False.
# * any additional parameters, such as beta or labels in f1_score.
# 
# Here is an example of building custom scorers, and of using the greater_is_better parameter:
# 

import numpy as np
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

# loss_func will negate the return value of my_custom_loss_func,
#  which will be np.log(2), 0.693, given the values for ground_truth
#  and predictions defined below.
loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
score = make_scorer(my_custom_loss_func, greater_is_better=True)
ground_truth = [[1, 1]]
predictions  = [0, 1]
from sklearn.dummy import DummyClassifier
# What is the dummy classifier!?! Wait one second
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf = clf.fit(ground_truth, predictions)
loss(clf,ground_truth, predictions) 

score(clf,ground_truth, predictions) 


# ## Other scoring functions
# 
# There are so many different scoring functions that there is no way that we are going to go over all of them. But we will go over some extremely useful ones
# 

# #### Confusion Matrix
# 
# The confusion_matrix function evaluates classification accuracy by computing the confusion matrix.
# 
# By definition, entry i, j in a confusion matrix is the number of observations actually in group i, but predicted to be in group j. Here is an example:
# 

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)


y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
tn, fp, fn, tp


# #### Classification Report
# 
# The classification_report function builds a text report showing the main classification metrics. Here is a small example with custom target_names and inferred labels:
# 

from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))



# #### Dummy Estimators
# 
# This was an interesting choice to put this here, but it did not feel right in any of the other sections.
# 
# When doing supervised learning, a simple sanity check consists of comparing one’s estimator against simple rules of thumb. DummyClassifier implements several such simple strategies for classification:
# * stratified generates random predictions by respecting the training set class distribution.
# * most_frequent always predicts the most frequent label in the training set.
# * prior always predicts the class that maximizes the class prior (like most_frequent) and predict_proba returns the class prior.
# * uniform generates predictions uniformly at random.
# * constant always predicts a constant label that is provided by the user. A major motivation of this method is F1-scoring, when the positive class is in the minority.
# 
# Note that with all these strategies, the predict method completely ignores the input data!
# To illustrate DummyClassifier, first let’s create an imbalanced dataset:
# 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
y[y != 1] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Now let's compare the SVC to the most_frequent
# 

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) 



clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)  


# Right, we don't do much better! But with a simple kernel change:
# 

clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)  


# DummyRegressor also implements four simple rules of thumb for regression:
# * mean always predicts the mean of the training targets.
# * median always predicts the median of the training targets.
# * quantile always predicts a user provided quantile of the training targets.
# * constant always predicts a constant value that is provided by the user.
# 




# # Tuning the hyper-parameters of an estimator
# 
# Hyper-parameters are parameters that are not directly learnt within estimators. In scikit-learn they are passed as arguments to the constructor of the estimator classes. Typical examples include C, kernel and gamma for Support Vector Classifier, alpha for Lasso, etc.
# 
# It is possible and recommended to search the hyper-parameter space for the best Cross-validation: evaluating estimator performance score.
# 
# Any parameter provided when constructing an estimator may be optimized in this manner. Specifically, to find the names and current values for all parameters for a given estimator, use:
# 
# `estimator.get_params()`
# 
# A search consists of:
# * an estimator (regressor or classifier such as sklearn.svm.SVC());
# * a parameter space;
# * a method for searching or sampling candidates;
# * a cross-validation scheme; and
# * a score function.
# 
# Some models allow for specialized, efficient parameter search strategies, outlined below. Two generic approaches to sampling search candidates are provided in scikit-learn: for given values, GridSearchCV exhaustively considers all parameter combinations, while RandomizedSearchCV can sample a given number of candidates from a parameter space with a specified distribution. After describing these tools we detail best practice applicable to both approaches.
# 
# Note that it is common that a small subset of those parameters can have a large impact on the predictive or computation performance of the model while others can be left to their default values. It is recommend to read the docstring of the estimator class to get a finer understanding of their expected behavior, possibly by reading the enclosed reference to the literature.
# 

# ## GridSearch
# 
# The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter. For instance, the following param_grid:
# 

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


# specifies that two grids should be explored: one with a linear kernel and C values in [1, 10, 100, 1000], and the second one with an RBF kernel, and the cross-product of C values ranging in [1, 10, 100, 1000] and gamma values in [0.001, 0.0001].
# 
# The GridSearchCV instance implements the usual estimator API: when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.
# 

from sklearn.model_selection import GridSearchCV

get_ipython().magic('pinfo GridSearchCV')


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='f1_macro')
clf.fit(X_train, y_train)


clf.best_params_


clf.cv_results_


y_true, y_pred = y_test, clf.predict(X_test)
print classification_report(y_true, y_pred)


clf.cv_results_.keys()


for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    print param, score


# ## Randomized Search
# 
# While using a grid of parameter settings is currently the most widely used method for parameter optimization, other search methods have more favourable properties. RandomizedSearchCV implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values. This has two main benefits over an exhaustive search:
# 
# * A budget can be chosen independent of the number of parameters and possible values.
# * Adding parameters that do not influence the performance does not decrease efficiency.
# 
# Specifying how parameters should be sampled is done using a dictionary, very similar to specifying parameters for GridSearchCV. Additionally, a computation budget, being the number of sampled candidates or sampling iterations, is specified using the n_iter parameter. For each parameter, either a distribution over possible values or a list of discrete choices (which will be sampled uniformly) can be specified:
# 

import scipy

params = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}


# This example uses the scipy.stats module, which contains many useful distributions for sampling parameters, such as expon, gamma, uniform or randint. In principle, any function can be passed that provides a rvs (random variate sample) method to sample a value. A call to the rvs function should provide independent random samples from possible parameter values on consecutive calls.
# 
# For continuous parameters, such as C above, it is important to specify a continuous distribution to take full advantage of the randomization. This way, increasing n_iter will always lead to a finer search.
# 

from sklearn.model_selection import RandomizedSearchCV

get_ipython().magic('pinfo RandomizedSearchCV')


clf = RandomizedSearchCV(SVC(), params, cv=5,
                       scoring='f1_macro')
clf.fit(X_train, y_train)


clf.best_params_


clf.cv_results_


y_true, y_pred = y_test, clf.predict(X_test)
print classification_report(y_true, y_pred)


for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    print param, score


# Don't forget the old _CV classes that are faster than gridsearch! And also don't forget about OOB error that can be a great proxy
# 




# # Novely and Outlier Detection
# 
# Many applications require being able to decide whether a new observation belongs to the same distribution as existing observations (it is an inlier), or should be considered as different (it is an outlier). Often, this ability is used to clean real data sets. Two important distinction must be made:
# 
# * novelty detection:
#  	The training data is not polluted by outliers, and we are interested in detecting anomalies in new observations.
# * outlier detection:
#  	The training data contains outliers, and we need to fit the central mode of the training data, ignoring the deviant observations.
# 
# The scikit-learn project provides a set of machine learning tools that can be used both for novelty or outliers detection. This strategy is implemented with objects learning in an unsupervised way from the data:
# 
# `estimator.fit(X_train)`
# 
# new observations can then be sorted as inliers or outliers with a predict method:
# 
# `estimator.predict(X_test)`
# 
# Inliers are labeled 1, while outliers are labeled -1.
# 

# ## Novelty Detection
# 
# Consider a data set of n observations from the same distribution described by p features. Consider now that we add one more observation to that data set. Is the new observation so different from the others that we can doubt it is regular? (i.e. does it come from the same distribution?) Or on the contrary, is it so similar to the other that we cannot distinguish it from the original observations? This is the question addressed by the novelty detection tools and methods.
# 
# In general, it is about to learn a rough, close frontier delimiting the contour of the initial observations distribution, plotted in embedding p-dimensional space. Then, if further observations lay within the frontier-delimited subspace, they are considered as coming from the same population than the initial observations. Otherwise, if they lay outside the frontier, we can say that they are abnormal with a given confidence in our assessment.
# 
# The One-Class SVM has been introduced by Schölkopf et al. for that purpose and implemented in the Support Vector Machines module in the svm.OneClassSVM object. It requires the choice of a kernel and a scalar parameter to define a frontier. The RBF kernel is usually chosen although there exists no exact formula or algorithm to set its bandwidth parameter. This is the default in the scikit-learn implementation. The \nu parameter, also known as the margin of the One-Class SVM, corresponds to the probability of finding a new, but regular, observation outside the frontier.
# 

from sklearn.svm import OneClassSVM

get_ipython().magic('pinfo OneClassSVM')


import numpy as np

X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))


clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


n_error_train, n_error_test, n_error_outliers


# ## Outlier Detection
# 
# Outlier detection is similar to novelty detection in the sense that the goal is to separate a core of regular observations from some polluting ones, called “outliers”. Yet, in the case of outlier detection, we don’t have a clean data set representing the population of regular observations that can be used to train any tool.
# 

# #### Isolation Forest
# 
# One efficient way of performing outlier detection in high-dimensional datasets is to use random forests. The ensemble.IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# 
# Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
# This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
# 
# Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.
# 

from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)


y_pred_outliers[y_pred_outliers == 1].size





# # Preprocessing Data
# 
# The sklearn.preprocessing package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
# 

# ## Standardization
# 
# Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.
# In practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
# 
# For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that all features are centered around zero and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
# 
# The function scale provides a quick and easy way to perform this operation on a single array-like dataset:
# 

from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X)

X_scaled                                          


print X_scaled.mean(axis=0)


print X_scaled.std(axis=0)


# But this has some drawbacks. 
# 
# The preprocessing module further provides a utility class StandardScaler that implements the Transformer API to compute the mean and standard deviation on a training set so as to be able to later reapply the same transformation on the testing set. This class is hence suitable for use in the early steps of a sklearn.pipeline.Pipeline:
# 

scaler = preprocessing.StandardScaler().fit(X)
scaler


scaler.mean_


scaler.scale_


scaler.transform(X)


scaler.transform([[-1.,  1., 0.]])    


# #### Scaling features to a range
# 
# An alternative standardization is scaling features to lie between a given minimum and maximum value, often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size. This can be achieved using MinMaxScaler or MaxAbsScaler, respectively.
# The motivation to use this scaling include robustness to very small standard deviations of features and preserving zero entries in sparse data.
# 
# Here is an example to scale a toy data matrix to the [0, 1] range:
# 

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax


# ## Normalization
# 
# Normalization is the process of scaling individual samples to have unit norm. This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
# 
# This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
# 
# The function normalize provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms:
# 

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

X_normalized                                      


# The preprocessing module further provides a utility class Normalizer that implements the same operation using the Transformer API (even though the fit method is useless in this case: the class is stateless as this operation treats samples independently).
# 
# This class is hence suitable for use in the early steps of a sklearn.pipeline.Pipeline:
# 

normalizer = preprocessing.Normalizer().fit(X) 
normalizer


normalizer.transform(X)                            


normalizer.transform([[-1.,  1., 0.]])  


# ## Binarization
# 
# Feature binarization is the process of thresholding numerical features to get boolean values. This can be useful for downstream probabilistic estimators that make assumption that the input data is distributed according to a multi-variate Bernoulli distribution. For instance, this is the case for the sklearn.neural_network.BernoulliRBM.
# 
# It is also common among the text processing community to use binary feature values (probably to simplify the probabilistic reasoning) even if normalized counts (a.k.a. term frequencies) or TF-IDF valued features often perform slightly better in practice.
# 
# As for the Normalizer, the utility class Binarizer is meant to be used in the early stages of sklearn.pipeline.Pipeline. The fit method does nothing as each sample is treated independently of others:
# 

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer


binarizer.transform(X)


binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)


# ## Encoding Categorical Features
# 
# Often features are not given as continuous values but categorical. For example a person could have features ["male", "female"], ["from Europe", "from US", "from Asia"], ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]. Such features can be efficiently coded as integers, for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3] while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].
# 
# Such integer representation can not be used directly with scikit-learn estimators, as these expect continuous input, and would interpret the categories as being ordered, which is often not desired (i.e. the set of browsers was ordered arbitrarily).
# 
# One possibility to convert categorical features to features that can be used with scikit-learn estimators is to use a one-of-K or one-hot encoding, which is implemented in OneHotEncoder. This estimator transforms each categorical feature with m possible values into m binary features, with only one active.
# 
# Continuing the example above:
# 

enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  


enc.transform([[0, 1, 3]]).toarray()


# ## Imputation of missing values
# 
# For various reasons, many real world datasets contain missing values, often encoded as blanks, NaNs or other placeholders. Such datasets however are incompatible with scikit-learn estimators which assume that all values in an array are numerical, and that all have and hold meaning. A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values, i.e., to infer them from the known part of the data.
# 
# The Imputer class provides basic strategies for imputing missing values, either using the mean, the median or the most frequent value of the row or column in which the missing values are located. This class also allows for different missing values encodings.
# 
# The following snippet demonstrates how to replace missing values, encoded as np.nan, using the mean value of the columns (axis 0) that contain the missing values:
# 

import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])


X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))  


# ## Generating Polynomial Features
# 
# Often it’s useful to add complexity to the model by considering nonlinear features of the input data. A simple and common method to use is polynomial features, which can get features’ high-order and interaction terms. It is implemented in PolynomialFeatures:
# 

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
X      


poly = PolynomialFeatures(2)
poly.fit_transform(X) 


# ## Label Binarization
# 
# LabelBinarizer is a utility class to help create a label indicator matrix from a list of multi-class labels:
# 

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])


lb.classes_


lb.transform([1,6])


lb = preprocessing.MultiLabelBinarizer()
lb.fit_transform([(1, 2), (3,)])


lb.classes_


# ## Label Encoding
# 
# LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1. This is sometimes useful for writing efficient Cython routines. LabelEncoder can be used as follows:
# 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])


le.classes_


le.transform([1, 1, 2, 6])


le.inverse_transform([0, 0, 1, 2])


# ## Custom Transformers
# 
# Often, you will want to convert an existing Python function into a transformer to assist in data cleaning or processing. You can implement a transformer from an arbitrary function with FunctionTransformer. For example, to build a transformer that applies a log transformation in a pipeline, do:
# 

import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)


get_ipython().magic('pinfo np.log1p')





# ## Model Persistence
# 
# After training a scikit-learn model, it is desirable to have a way to persist the model for future use without having to retrain. The following section gives you an example of how to persist a model with pickle. We’ll also review a few security and maintainability issues when working with pickle serialization.
# 

from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  


import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])


y[0]


# In the specific case of the scikit, it may be more interesting to use joblib’s replacement of pickle (joblib.dump & joblib.load), which is more efficient on objects that carry large numpy arrays internally as is often the case for fitted scikit-learn estimators, but can only pickle to the disk and not to a string:
# 

from sklearn.externals import joblib

joblib.dump(clf, 'model.pkl') 


clf = joblib.load('model.pkl') 





# # Density Estimation
# 
# Density estimation walks the line between unsupervised learning, feature engineering, and data modeling. Some of the most popular and useful density estimation techniques are mixture models such as Gaussian Mixtures (sklearn.mixture.GaussianMixture), and neighbor-based approaches such as the kernel density estimate (sklearn.neighbors.KernelDensity). Gaussian Mixtures are discussed more fully in the context of clustering, because the technique is also useful as an unsupervised clustering scheme.
# 
# Density estimation is a very simple concept, and most people are already familiar with one common density estimation technique: the histogram.
# 

# ## Kernel Density Estimation
# 
# Kernel density estimation in scikit-learn is implemented in the sklearn.neighbors.KernelDensity estimator, which uses the Ball Tree or KD Tree for efficient queries (see Nearest Neighbors for a discussion of these). Though the above example uses a 1D data set for simplicity, kernel density estimation can be performed in any number of dimensions, though in practice the curse of dimensionality causes its performance to degrade in high dimensions.
# 
# The kernel density estimator can be used with any of the valid distance metrics (see sklearn.neighbors.DistanceMetric for a list of available metrics), though the results are properly normalized only for the Euclidean metric. One particularly useful metric is the Haversine distance which measures the angular distance between points on a sphere.
# 

from sklearn.neighbors.kde import KernelDensity

get_ipython().magic('pinfo KernelDensity')


from sklearn.neighbors.kde import KernelDensity
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)

kde.score_samples([[32,4]])


kde.sample(1)


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)


estimators = []
for c in [0, 1, 2]:
    m = KernelDensity().fit(X[y == c])
    estimators.append(m)
    
for estimator in estimators:
    print estimator.score_samples([X[0]])








get_ipython().magic('pylab inline')

import sklearn.datasets as datasets


# # Datasets
# 
# There are three distinct kinds of dataset interfaces for different types of datasets. The simplest one is the interface for sample images, which is described below in the Sample images section.
# 
# The dataset generation functions and the svmlight loader share a simplistic interface, returning a tuple `(X, y)` consisting of a `n_samples * n_features` numpy array `X` and an array of length n_samples containing the targets `y`.
# 
# The toy datasets as well as the ‘real world’ datasets and the datasets fetched from mldata.org have more sophisticated structure. These functions return a dictionary-like object holding at least two items: an array of shape `n_samples * n_features` with key `data` (except for 20newsgroups) and a numpy array of length `n_samples`, containing the target values, with key `target`.
# 
# The datasets also contain a description in `DESCR` and some contain `feature_names` and `target_names`. See the dataset descriptions below for details.
# 

# ## Sample Images
# 
# The scikit also embed a couple of sample JPEG images published under Creative Commons license by their authors. Those image can be useful to test algorithms and pipeline on 2D data.
# 

# datasets.load_sample_images()
china = datasets.load_sample_image('china.jpg')

flower = datasets.load_sample_image('flower.jpg')


plt.imshow(china)


plt.imshow(flower)


flower.shape


# ## Sample Generators
# 
# In addition, scikit-learn includes various random sample generators that can be used to build artificial datasets of controlled size and complexity.
# 
# All of the generators are prefixed with the word `make` 
# 

get_ipython().magic('pinfo datasets.make_blobs')


X, y = datasets.make_blobs()


X.shape, y.shape


# ## Toy and Fetched Datasets
# 
# Scikit-learn comes with a few small standard datasets that do not require to download any file from some external website.
# 
# These datasets are useful to quickly illustrate the behavior of the various algorithms implemented in the scikit. They are however often too small to be representative of real world machine learning tasks. 
# 
# These datasets are prefixed with the `load` command.
# 

data = datasets.load_boston()


data.keys()


data.data.shape, data.target.shape


data.feature_names


print data.DESCR


# #### Fetched datasets
# 
# These are all somewhat unique with their own functions to fetch and load them. I'll go through a single one below. They are all prefixed with the word `fetch`
# 

faces = datasets.fetch_olivetti_faces()


faces.keys()


faces.images.shape, faces.data.shape, faces.target.shape





# # Cross-validation: evaluating estimator performance
# 
# Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally.
# 
# In scikit-learn a random split into training and test sets can be quickly computed with the train_test_split helper function. Let’s load the iris data set to fit a linear support vector machine on it:
# 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape


get_ipython().magic('pinfo train_test_split')


X_train, X_test, y_train, y_test = train_test_split(
     iris.data, iris.target, test_size=0.4, random_state=0)

X_train.shape, y_train.shape


X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)  


clf.score(X_train, y_train)


# When evaluating different settings (“hyperparameters”) for estimators, such as the C setting that must be manually set for an SVM, there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “validation set”: training proceeds on the training set, after which evaluation is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.
# 
# However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.
# 
# A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”:
# 
# * A model is trained using k-1 of the folds as training data;
# * the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).
# 
# The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data (as it is the case when fixing an arbitrary test set), which is a major advantage in problem such as inverse inference where the number of samples is very small.
# 

from sklearn.model_selection import cross_val_score

get_ipython().magic('pinfo cross_val_score')


clf = svm.SVC(kernel='linear', C=1)

scores = cross_val_score(clf, iris.data, iris.target, cv=2)

scores


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


from sklearn import metrics

scores = cross_val_score(
     clf, iris.data, iris.target, cv=5, scoring='f1_macro')

scores


from sklearn.model_selection import ShuffleSplit

get_ipython().magic('pinfo ShuffleSplit')


n_samples = iris.data.shape[0]

cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

cross_val_score(clf, iris.data, iris.target, cv=cv)


from sklearn.model_selection import cross_val_predict

get_ipython().magic('pinfo cross_val_predict')


predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)

predicted.shape


metrics.accuracy_score(iris.target, predicted) 


from sklearn.linear_model import LassoCV


# ## Cross validation iterators
# 
# The following sections list utilities to generate indices that can be used to generate dataset splits according to different cross validation strategies.
# 
# Assuming that some data is Independent Identically Distributed (i.i.d.) is making the assumption that all samples stem from the same generative process and that the generative process is assumed to have no memory of past generated samples.
# 
# The following cross-validators can be used in such cases.
# 

from sklearn.model_selection import KFold

get_ipython().magic('pinfo KFold')


kf = KFold(n_splits=4, shuffle=True)

X = ["a", "b", "c", "d"]
for train, test in kf.split(X):
    print("%s %s" % (train, test))


# #### Stratification
# 
# Some classification problems can exhibit a large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples. In such cases it is recommended to use stratified sampling as implemented in StratifiedKFold and StratifiedShuffleSplit to ensure that relative class frequencies is approximately preserved in each train and validation fold.
# 

from sklearn.model_selection import StratifiedKFold

get_ipython().magic('pinfo StratifiedKFold')


X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))


# #### Grouped Data
# 
# The i.i.d. assumption is broken if the underlying generative process yield groups of dependent samples.
# 
# Such a grouping of data is domain specific. An example would be when there is medical data collected from multiple patients, with multiple samples taken from each patient. And such data is likely to be dependent on the individual group. In our example, the patient id for each sample will be its group identifier.
# 
# In this case we would like to know if a model trained on a particular set of groups generalizes well to the unseen groups. To measure this, we need to ensure that all the samples in the validation fold come from groups that are not represented at all in the paired training fold.
# 
# The following cross-validation splitters can be used to do that. The grouping identifier for the samples is specified via the groups parameter.
# 

from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))


# #### Time Series Split
# 
# TimeSeriesSplit is a variation of k-fold which returns first k folds as train set and the (k+1) th fold as test set. Note that unlike standard cross-validation methods, successive training sets are supersets of those that come before them. Also, it adds all surplus data to the first training partition, which is always used to train the model.
# 
# This class can be used to cross-validate time series data samples that are observed at fixed time intervals.
# 

from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)  

for train, test in tscv.split(X):
    print("%s %s" % (train, test))





# # Feature Transformation
# 
# I am going to show off only two parts of the massive quantity of code in the unsupervised learning section of sklearn. And they can be put into this single bucket:
# 
# * Feature Transformation
# * Exploratory Data Analysis
# 

# ## Clustering
# 
# Clustering of unlabeled data can be performed with the module sklearn.cluster.
# Each clustering algorithm comes in two variants: a class, that implements the fit method to learn the clusters on train data, and a function, that, given train data, returns an array of integer labels corresponding to the different clusters. For the class, the labels over the training data can be found in the labels_ attribute.
# 

# #### Kmeans
# 
# The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares. This algorithm requires the number of clusters to be specified. It scales well to large number of samples and has been used across a large range of application areas in many different fields.
# 
# Let's check out how it is used
# 

from sklearn.cluster import KMeans

get_ipython().magic('pinfo KMeans')


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)


cluster = KMeans(n_clusters=3)


cluster.fit(X)


cluster.predict(X)


from sklearn.tree import DecisionTreeClassifier

m = DecisionTreeClassifier(max_depth=2)


m.fit(cluster.predict(X)[:, None], y)


m.score(cluster.predict(X)[:, None], y)


# ## Principal component analysis (PCA)
# 
# PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance. In scikit-learn, PCA is implemented as a transformer object that learns n components in its fit method, and can be used on new data to project it on these components.
# 
# The optional parameter whiten=True makes it possible to project the data onto the singular space while scaling each component to unit variance. This is often useful if the models down-stream make strong assumptions on the isotropy of the signal: this is for example the case for Support Vector Machines with the RBF kernel and the K-Means clustering algorithm.
# 

from sklearn.decomposition import PCA

get_ipython().magic('pinfo PCA')


from sklearn.svm import SVC

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

X_pca.shape


SVC().fit(X_pca, y).score(X_pca, y)





# # Regression
# 
# We won't go over every model, in fact I will stick to as few as possible models but go over how they are used and what their commonalities are.
# 
# We will first start off by importing some toy data.
# 

import sklearn.datasets as datasets

X, y = datasets.load_boston(return_X_y=True)


print y[0]


# Next we will do the training. Models have two states:
# 
# 1. Instantiated
# 2. Fit
# 
# When we instantiate the model we specify the hyperparameters of the model and nothing else. 
# 

from sklearn import linear_model

get_ipython().magic('pinfo linear_model.ElasticNet')


m = linear_model.ElasticNet(alpha=.1, l1_ratio=.9)


# The next step is fitting the model
# 

m.fit(X, y)


m.coef_


m.intercept_


m.predict([X[0]])


y[0]


m.score(X, y)


get_ipython().magic('pinfo m.score')


# ## CV models
# 
# Some of these models come with a CV model. 
# 

get_ipython().magic('pinfo linear_model.ElasticNetCV')


m = linear_model.ElasticNetCV(
    l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
    n_alphas=20)


m.fit(X, y)


m.alphas_


m.mse_path_


m.alpha_


m.l1_ratio_


m.predict([X[0]])


m.score(X, y)


# # Classification
# 
# Okay this one is quite quick. And is very much so the same as the above. So to cut to the chase, I'll train a Cross Validated Logistic Regression.
# 

X, y = datasets.load_iris(return_X_y=True)


d = datasets.load_iris()

print d.DESCR


get_ipython().magic('pinfo linear_model.LogisticRegressionCV')


# One thing you might notice here is that we have the option of parallelization!
# 

m = linear_model.LogisticRegressionCV(Cs=10, n_jobs=2)


m.fit(X, y)


m.coef_


m.predict([X[0]])


y[0]


m.predict_proba([X[0]])


m.predict_log_proba([X[0]])


m.score(X, y)


get_ipython().magic('pinfo m.score')








# # Feature Extraction
# 
# The sklearn.feature_extraction module can be used to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image.
# 

# ##  Loading features from dicts
# 
# The class DictVectorizer can be used to convert feature arrays represented as lists of standard Python dict objects to the NumPy/SciPy representation used by scikit-learn estimators.
# 
# While not particularly fast to process, Python’s dict has the advantages of being convenient to use, being sparse (absent features need not be stored) and storing feature names in addition to values.
# 
# DictVectorizer implements what is called one-of-K or “one-hot” coding for categorical (aka nominal, discrete) features. Categorical features are “attribute-value” pairs where the value is restricted to a list of discrete of possibilities without ordering (e.g. topic identifiers, types of objects, tags, names...).
# 
# In the following, “city” is a categorical attribute while “temperature” is a traditional numerical feature:
# 

measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

vec.fit_transform(measurements).toarray()




vec.get_feature_names()


# ## Text feature extraction
# 
# Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
# 
# In order to address this, scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:
# 
# * tokenizing strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
# * counting the occurrences of tokens in each document.
# * normalizing and weighting with diminishing importance tokens that occur in the majority of samples / documents.
# 
# In this scheme, features and samples are defined as follows:
# 
# * each individual token occurrence frequency (normalized or not) is treated as a feature.
# * the vector of all the token frequencies for a given document is considered a multivariate sample.
# 
# A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus.
# 
# We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.
# 

# CountVectorizer implements both tokenization and occurrence counting in a single class:
# 

from sklearn.feature_extraction.text import CountVectorizer

get_ipython().magic('pinfo CountVectorizer')


vectorizer = CountVectorizer(min_df=1)
vectorizer 


corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
X                              


X.toarray()


analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.")


vectorizer.get_feature_names()


vectorizer.vocabulary_.get('document')


vectorizer.transform(['Something completely new.']).toarray()


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
transformer   


counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]

tfidf = transformer.fit_transform(counts)
tfidf                         


tfidf.toarray()      


from sklearn.feature_extraction.text import TfidfVectorizer

get_ipython().magic('pinfo TfidfVectorizer')





