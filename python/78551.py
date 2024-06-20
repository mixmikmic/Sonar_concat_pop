# # hospital readmissions reduction
# 
# - 20% readmitted within 30 days, 34% within 90
# - $17.4 B annual cost
# - symptomatic of quality problems within hospitals, lack of coordination in follow up care, and misalighed financial incentives
# 
# - oct 2012 - HRRP links Medicare reimbursements to hospital's risk-adjusted readmission rate
# - exceeding risk-adjusted, 3-year rolling readmission rate for AMI, HF, and pneumonia(3 conditions) => penalized portion of Medicare reimbursements
# - 2012, penalty up to 1% of total
# - 2,225 hospitals subject to reduced payment penalties in 2012, worth 225 M nationwide
# - max penalty increased to 3% in 2014
# 
# - 18% of Tahoe's total revenues were Medicare reimbursements for the 3 conditions
# - fines for exceeding readmission rates in 2012 were over $750k
# 
# - estimated loss would rise to 8k per readmitted patient if readmissions rates not recuded
# 
# - CareTracker program incorporates patient education and periodic at-home monitoring
# - pilot study reduced readmission 40%; however cost was $1.2k/patient
# 
# - cost and benefit analysis for rollout to entire hospital system
# 

get_ipython().magic('matplotlib inline')

from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from seaborn import pairplot

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import mglearn

from sklearn.datasets import load_iris
import random
random.seed(15)
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns

from collections import Counter

from imblearn.over_sampling import RandomOverSampler


# admissions over last year for patients with AMI

# severity score - generic physiologic severity of illness score based on lab tests and vital signs
# comorbidity score - severity score based on patients' pre-existing diagnoses
# readmit30 - indicator for hospital readmission within 30 day

df = pd.read_csv('Final Project Data_Case.csv').dropna()
df.tail()


# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
df.tail()


# >first train a classificiation model to predict whether a patient will readmit in 30 days
# since there's class imbalance
# 
# >use SVM classifier, then rank feature importance
# 
# 

df['comorbidity score'].describe()


df['readmit30'].hist();


# check for relationships
pairplot(df);


df.plot(x='comorbidity score', y='readmit30', style='o');


# there's class imbalance, and readmits are the minority class
print(Counter(df['readmit30']))

readmits = df[df['readmit30'] == 1]
readmits.tail()


X_holdout = df.ix[:599,:6]

y_holdout = df['readmit30'][:600]
df = df.ix[599:,:]


df.shape


y_holdout.shape


X_holdout.shape


X = df.ix[:,:6]
y = df['readmit30']

# up-sample: resample the minority class to deal with class imbalance

# !pip install -U imbalanced-learn
# from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(X, y)

print(X_resampled.shape)

# default 75-25 train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


X_train[:,4:6]


X_train


# train classifiers using severity score comorbidity score

# X = np.array(X_train[['severity score', 'comorbidity score']])
X = np.array(X_train[:,4:6])
y = np.array(y_train)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    
    # visualize decision boundary found by linear model
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("severity score")
    ax.set_ylabel("comorbidity score")
axes[0].legend()

# Decision boundaries of a linear SVM and logistic regression on the dataset with the default parameters


# linear methods won't work, use nonlinear methods
# 

def decision_boundary(clf, X, Y, h=.02):
    """Inputs:
        clf - a trained classifier, with a predict method
    """
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(12, 9))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()


# X = X_train.as_matrix()
# y = y_train.as_matrix()

# severity and comorbidity
plt.scatter(X[:,0],X[:,1],c = y)


# make a mini set for illustration purposes
len(X[::25])


len(y[::25])


clf = SVC(kernel='poly')

clf.fit(X[::25], y[::25])


decision_boundary(clf, X[::25], y[::25])


print("Done")


decision_boundary(clf, X[::3], y[::3])


print("Done")


decision_boundary(clf, X[::10], y[::10])


decision_boundary(clf, X[::10], y[::10])


# grid search for parameters C and gamma

# C = 1
# gamma = [1e-1, 1, 1e1]
# classifiers = []
# for gamma in gamma:
#     clf = SVC(C=1, gamma=gamma)
#     clf.fit(X, y)
#     decision_boundary(clf,X,y)
    
# C = [1E-1,1,10,100]
# gamma = 250
# classifiers = []
# for C in C:
#     clf = SVC(C=C, gamma=gamma)
#     clf.fit(X, y)
#     decision_boundary(clf,X,y)


# default kernel is kernel='rbf', other nonlinear kernels are poly and sigmoid

# clf = SVC(kernel = 'poly')
# clf.fit(X, y)
# decision_boundary(clf,X,y)

# clf = SVC(kernel = 'sigmoid')
# clf.fit(X_small, y_small)
# decision_boundary(clf,X_small,y_small)


# now use all the features

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Returns the mean accuracy on the given test data and labels.

print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))


# now use all the features

# clf = SVC() # use default values for C and gamma
# clf.fit(X_train, y_train)

y_pred = clf.predict(X_holdout)

# Returns the mean accuracy on the given test data and labels.

print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_holdout, y_holdout)))


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))


from sklearn.ensemble import RandomForestClassifier

# now use all the features

clf = RandomForestClassifier() # use default values for C and gamma
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Returns the mean accuracy on the given test data and labels.

print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_holdout, y_holdout)))


# Plot feature importance

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(14,7))
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center', color='m')

plt.yticks(pos, np.array(list(df))[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# now use all the features

clf = SVC() # use default values for C and gamma
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Returns the mean accuracy on the given test data and labels.

print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))


# now use all the features

# clf = SVC() # use default values for C and gamma
# clf.fit(X_train, y_train)

y_pred = clf.predict(X_holdout)

# Returns the mean accuracy on the given test data and labels.

print("Training set score: {:.2f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_holdout, y_holdout)))


clf = SVC(random_state=0)

# specify parameters and distributions to sample from
# set the parameters by cross-validation
# C = [1E-1,1,10,100]
# gamma = [1e-1, 1, 1e1]

# default C is 1 and default gamma is 1/n_features = 1/6
param_dist = {"C": sp_randint(1, 100),
              "gamma": [.1, 'auto', 1],
             }
            
# run randomized search
# n_iter_search = 20
n_iter_search = 10
random_search = RandomizedSearchCV(clf,
                    param_distributions=param_dist,
                    cv=5,
                    n_iter=n_iter_search,
#                     n_jobs=-1
                   )
# 5-fold cross validation


start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))


# random_search.cv_results_

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            

# report(random_search.cv_results_)


report(random_search.cv_results_, n_top=1)


# now use all the features

clf = SVC(C=15, gamma=1) # use default values for C and gamma
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Returns the mean accuracy on the given test data and labels.

print("Training set score: {:.4f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_holdout, y_holdout)))


# personalized intervention, identify subgroups that would benefit from CareTracker, let sleeping dogs lie
# let's construct an uplift model to directly model the incremental impact of intervention
# maximizes return on investment by targeting the persuadables


# first, calculate the costs if without any CareTracker intervention
# 
# assumptions:
# 
# $8k loss in Medicare reimbursements per readmitted patient (2014), based off CFO Leila Houssein's estimated figure
# 
# so with our given dataset, we have 998 readmits, 998*8000 = 7,984,000
# 

df = pd.read_csv('Final Project Data_Case.csv').dropna()
readmits = df[df['readmit30'] == 1]
len(readmits)


# now suppose we introduce CareTracker to all 4382 patients
# 
# our cost would be 4382*1200 = 5,258,400 for CareTracker, but we'd expect to recoup $ through preventing readmission.
# 
# CareTracker has shown to reduce readmission by % in the past, that means instead of 998 we'd expect x readmits, reducing penalties by $xx.  So the net loss is x.
# 

4382*1200 + (8000*.6*998)


len(df)


4382*1200


from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

cm_train = confusion_matrix(y_train,
                            clf.predict(X_train))
cm_train


from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

cm_train = confusion_matrix(y_test,
                            clf.predict(X_test))
cm_train


from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

cm_holdout = confusion_matrix(y_holdout,
                            clf.predict(X_holdout))
cm_holdout


# Thus in binary classification, the count of true negatives is
# :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
# :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

# true negatives is quadrant II, true positibes is quadrant I
m = cm_holdout
print('true negatives: ', m[0][0])
print('false negatives: ', m[1][0])
print('true positives: ', m[1][1])
print('false positives: ', m[0][1])


import matplotlib.pyplot as plt
import numpy as np

def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    Source: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html
    """
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)
    
    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))
    
    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))
    
    plt.tight_layout()
    plt.show()
    
def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    Source: http://notmatthancock.github.io/2015/10/28/confusion-matrix.html
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))
    
    plt.tight_layout()
    plt.show()


show_confusion_matrix(cm_holdout,
                     class_labels=["no readmit", "readmit"])


cm_holdout


sum(cm_holdout)


1/600


155/600


157+42+1+1


1200*.4 + 9200*.6


1200+.6*8000


cost = np.array([[0, 1200], [8000, 6000]])
cost


cm_percentages = cm_holdout/600
cm_percentages


ppl = cm_percentages*4382


ppl


ppl*cost


sum(sum(ppl*cost))


no_care_tracker = 998*8000


quadratic_discriminant_analysis = 6037465.6934306575


4382 * 1200 + 600*8000


# admissions over last year for patients with AMI
# severity score - generic physiologic severity of illness score based on lab tests and vital signs
# comorbidity score - severity score based on patients' pre-existing diagnoses
# readmit30 - indicator for hospital readmission within 30 day

df = pd.read_csv('Final Project Data_Case.csv').dropna()
df.tail()


Counter(df['readmit30'])


X = df.ix[:,:6]
y = df['readmit30']

# default 75-25 train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=171)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# Quadratic Discriminant Analysis
# A classifier with a quadratic decision boundary, generated by fitting class conditional densities 
# to the data and using Bayes’ rule.
# The model fits a Gaussian density to each class.

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Training set score: {:.3f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))


cm_train = confusion_matrix(y_test,
                            clf.predict(X_test))
cm_train


show_confusion_matrix(cm_train,
                     class_labels=["no readmit", "readmit"])


train_population = sum(sum(cm_train))
cm_percentages = cm_train/train_population
all_patients = cm_percentages * 4381
total_cost_with_model = sum(sum(all_patients * cost))
total_cost_with_model


all_patients


# no care tracker at all
8000*998


# savings
savings = 7984000 - total_cost_with_model
savings

# depending on the random state of the train-test split, we could have 80k or 1.17M


# 4382-998

# perfect = np.array([[ 3384,  0],
#        [ 0,  998]])

# cost
# .6*998
# 998*1200 + 598.8*8000


def try_different_random_states():
    X = df.ix[:,:6]
    y = df['readmit30']

    # default 75-25 train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y)

#     print("X_train shape: {}".format(X_train.shape))
#     print("y_train shape: {}".format(y_train.shape))
#     print("X_test shape: {}".format(X_test.shape))
#     print("y_test shape: {}".format(y_test.shape))

    # Quadratic Discriminant Analysis
    # A classifier with a quadratic decision boundary, generated by fitting class conditional densities 
    # to the data and using Bayes’ rule.
    # The model fits a Gaussian density to each class.

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
#     print("Training set score: {:.3f}".format(clf.score(X_train, y_train)))
#     print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))

    cm_train = confusion_matrix(y_test,
                                clf.predict(X_test))

    train_population = sum(sum(cm_train))
    cm_percentages = cm_train/train_population
    all_patients = cm_percentages * 4381
    total_cost_with_model = sum(sum(all_patients * cost))
    total_cost_with_model

    # savings
    savings = 7984000 - total_cost_with_model
    return savings

# depending on the random state of the train-test split, we could have 80k or 1.17M

v=[]
for i in range(10000):
    v.append(try_different_random_states())
    
# print(v)


import pprint
pprint.pprint(pd.core.series.Series(v).describe())


possible_savings = pd.core.series.Series(v).describe()


min(pd.core.series.Series(v).describe())


max(pd.core.series.Series(v).describe())


np.mean(pd.core.series.Series(v).describe())


np.std(pd.core.series.Series(v).describe())


plt.hist(pd.core.series.Series(v), color = 'g')
plt.title('Estimated Savings from 10k different splits')


from sklearn.metrics import roc_curve, auc

probabilities = clf.predict_proba(X_test)[:, 1]

fpr,tpr,threshold = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
print ("ROC AUC: %0.2f" % roc_auc)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")


plt.show()


print("Test set score: {:.2f}".format(.80))





