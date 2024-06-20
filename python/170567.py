# Putting it All Together
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load the classifying models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data[:, :2]  #load the first two features of the iris data 
y = iris.target #load the target of the iris data


from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 0)


from sklearn.model_selection import cross_val_score
knn_3_clf = KNeighborsClassifier(n_neighbors = 3)
knn_5_clf = KNeighborsClassifier(n_neighbors = 5)

knn_3_scores = cross_val_score(knn_3_clf, X_train, y_train, cv=10)
knn_5_scores = cross_val_score(knn_5_clf, X_train, y_train, cv=10)


print "knn_3 mean scores: ", knn_3_scores.mean(), "knn_3 std: ",knn_3_scores.std()
print "knn_5 mean scores: ", knn_5_scores.mean(), " knn_5 std: ",knn_5_scores.std()


all_scores = []
for n_neighbors in range(3,9,1):
    knn_clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    all_scores.append((n_neighbors, cross_val_score(knn_clf, X_train, y_train, cv=10).mean()))
sorted(all_scores, key = lambda x:x[1], reverse = True) 


# In this example, the stacker was handled with cross-validation instead of a train-test-split.
# The whole training set was used in both of the two stacking phases.
# 

from __future__ import division

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

cali_housing = fetch_california_housing()

X = cali_housing.data
y = cali_housing.target

bins = np.arange(6)
 

from sklearn.model_selection import train_test_split

binned_y = np.digitize(y, bins)

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
 
from sklearn.model_selection import GridSearchCV

X_train_prin, X_test_prin, y_train_prin, y_test_prin = train_test_split(X, y,test_size=0.2,stratify=binned_y,random_state=7)

binned_y_train_prin = np.digitize(y_train_prin, bins)


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 3, random_state = 7)
skf.split(X_train_prin, binned_y_train_prin)


from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
 'max_samples': [0.5,1.0],
 'max_features' : [0.5,1.0],
 'oob_score' : [True, False],
 'base_estimator__n_neighbors': [3,5],
 'n_estimators': [100]
 }

single_estimator = KNeighborsRegressor()
ensemble_estimator = BaggingRegressor(base_estimator = single_estimator)

pre_gs_inst_bag = RandomizedSearchCV(ensemble_estimator,
                                     param_distributions = param_dist,
                                     cv = skf,
                                     n_iter = 5,
                                     n_jobs=-1,
                                    random_state=7)

pre_gs_inst_bag.fit(X_train_prin, y_train_prin)


pre_gs_inst_bag.best_params_


rs_bag = BaggingRegressor(**{'max_features': 0.5,
 'max_samples': 1.0,
 'n_estimators': 3000,
 'oob_score': True, 
 'base_estimator': KNeighborsRegressor(n_neighbors=5)})


from sklearn.model_selection import cross_val_predict

bag_predicted = cross_val_predict(rs_bag, X_train_prin, y_train_prin, cv=skf, n_jobs=-1)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

param_dist = {'max_features' : ['log2',0.4,0.5,0.6,1.0],
 'max_depth' : [2,3, 4, 5,6, 7, 10],
 'min_samples_leaf' : [1,2, 3, 4, 5, 10],
 'n_estimators': [50, 100],
 'learning_rate' : [0.01,0.05,0.1,0.25,0.275,0.3,0.325],
 'loss' : ['ls','huber']
 }
pre_gs_inst_gb = RandomizedSearchCV(GradientBoostingRegressor(warm_start=True),
                                   param_distributions = param_dist,
                                   cv=skf, 
                                   n_iter = 30, 
                                   n_jobs=-1,random_state=7)
pre_gs_inst_gb.fit(X_train_prin, y_train_prin)


pre_gs_inst_gb.best_estimator_


gbt_inst = GradientBoostingRegressor(**{'learning_rate': 0.25,
 'loss': 'huber',
 'max_depth': 6,
 'max_features': 1.0,
 'min_samples_leaf': 10,
 'n_estimators': 3000,
 'warm_start': True})


gbt_predicted = cross_val_predict(gbt_inst, X_train_prin, y_train_prin, cv=skf, n_jobs=-1)


preds_df = pd.DataFrame(X_train_prin.copy(),columns = cali_housing .feature_names )#pd.DataFrame(columns = ['bag', 'gbt'])

preds_df['bag'] = bag_predicted
preds_df['gbt'] = gbt_predicted


preds_df[['bag','gbt']].corr()


preds_df.shape


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV

param_dist = {'max_features' : ['sqrt','log2',1.0],
 'min_samples_leaf' : [1, 2, 3, 7, 11],
 'n_estimators': [50, 100],
 'oob_score': [True, False]}

pre_gs_inst_etr = RandomizedSearchCV(ExtraTreesRegressor(warm_start=True,bootstrap=True),
                                 param_distributions = param_dist,
                                 cv=skf,
                                 n_iter = 15,
                                 random_state = 7)

pre_gs_inst_etr.fit(preds_df.values, y_train_prin)


pre_gs_inst_etr.best_params_


final_etr = ExtraTreesRegressor(**{'max_features': 1.0,
 'min_samples_leaf': 11,
 'n_estimators': 2000,
 'oob_score': False})
final_etr.fit(preds_df.values, y_train_prin)


rs_bag.fit(X_train_prin, y_train_prin)


gbt_inst.fit(X_train_prin, y_train_prin)


def handle_X_set(X_set):
    X_copy = X_set.copy()
    
    y_pred_bag = rs_bag.predict(X_copy)
    y_pred_gbt = gbt_inst.predict(X_copy)
    preds_df = pd.DataFrame(X_copy, columns = cali_housing .feature_names)

    preds_df['bag'] = y_pred_bag
    preds_df['gbt'] = y_pred_gbt
 
    return preds_df.values

def predict_from_X_set(X_set):
    return final_etr.predict(handle_X_set(X_set)) 

y_pred = predict_from_X_set(X_test_prin)


def mase(y_test, y_pred):
    y_avg = y_test.mean()
    denominator = np.abs(y_test - y_avg).mean()
    numerator = y_test - y_pred
    
    return np.abs(numerator/denominator).mean()


# https://www.otexts.org/fpp/2/5 : contains SMAPE (attributed to Armstrong) and MASE (Hyndman and Koehler)
from sklearn.metrics import r2_score, mean_absolute_error

print ("R-squared",r2_score(y_test_prin, y_pred))
print ("MAE   :  ",mean_absolute_error(y_test_prin, y_pred))
print ("MAPE  :  ",(np.abs(y_test_prin- y_pred)/y_test_prin).mean())
print ("SMAPE :  ",(np.abs(y_test_prin- y_pred)/((y_test_prin + y_pred)/2)).mean())
print ("MASE  :  ",mase(y_test_prin, y_pred)) 


