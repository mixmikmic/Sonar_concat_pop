import h2o
h2o.connect()


import pandas as pd


titanic_df = pd.read_csv('/Users/avkashchauhan/learn/seattle-workshop/titanic_list.csv')


titanic_df.shape


titanic_df.count()


# Converting Pandas Frame to H2O Frame
titanic = h2o.H2OFrame(titanic_df)


titanic


# Note: You will see that the following command will not work
# Because it is a H2OFrame
titanic.count()


# The Other option to import data directly is to use H2O.
titanic_data = h2o.import_file('/Users/avkashchauhan/learn/seattle-workshop/titanic_list.csv')


titanic_data.shape


# Loading Estimators
from h2o.estimators.glm import H2OGeneralizedLinearEstimator


# set this to True if interactive (matplotlib) plots are desired
import matplotlib
interactive = False
if not interactive: matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt


titanic_data.describe()


titanic_data.col_names


response = "survived"


# Selected Columns
# pclass, survived, sex, age, sibsp, parch, fare, embarked 
#

predictors = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
# predictors = titanic_data.columns[:-1]


predictors


#Now setting factors to specific columns
titanic_data["pclass"] = titanic_data["pclass"].asfactor()
titanic_data["sex"] = titanic_data["sex"].asfactor()
titanic_data["embarked"] = titanic_data["embarked"].asfactor()


titanic_data.describe()


# Spliting the data set for training and validation
titanic_train, titanic_valid = titanic_data.split_frame(ratios=[0.9])


print(titanic_train.shape)
print(titanic_valid.shape)


# # Creating GLM Model
# 

titanic_glm = H2OGeneralizedLinearEstimator(alpha = .25)


titanic_glm.train(x = predictors, y = response, training_frame = titanic_train, validation_frame = titanic_valid)


# print the mse for the validation data
print "mse: ", titanic_glm.mse(valid=True)
print "r2 : ", titanic_glm.r2(valid=True)
print "rmse:", titanic_glm.rmse(valid=True)

# Note: Look for titanic_glm.[TAB] for the values you are interested into


# # Adding Grid Search now
# 

# grid over `alpha`
# import Grid Search
from h2o.grid.grid_search import H2OGridSearch


# select the values for `alpha` to grid over
hyper_params = {'alpha': [0, .25, .5, .75, .1]}


# this example uses cartesian grid search because the search space is small
# and we want to see the performance of all models. For a larger search space use
# random grid search instead: {'strategy': "RandomDiscrete"}
# initialize the GLM estimator
titanic_glm_hype = H2OGeneralizedLinearEstimator()


# build grid search with previously made GLM and hyperparameters
titanitc_grid = H2OGridSearch(model = titanic_glm_hype, hyper_params = hyper_params,
                     search_criteria = {'strategy': "Cartesian"})


# train using the grid
titanitc_grid.train(x = predictors, y = response, training_frame = titanic_train, validation_frame = titanic_valid)


# sort the grid models by mse
titanic_sorted_grid = titanitc_grid.get_grid(sort_by='mse', decreasing=False)
print(titanic_sorted_grid)





# If you want to sort by r2 then try this
titanic_sorted_grid = titanitc_grid.get_grid(sort_by='r2', decreasing=False)
print(titanic_sorted_grid)


# # Adding multiple hyperparameters
# 

# Now adding alpha and lambda together
hyper_params = {'alpha': [0, .25, .5, .75, .1], 'lambda': [0, .1, .01, .001, .0001]}


titanic_glm_hype = H2OGeneralizedLinearEstimator()
titanitc_grid = H2OGridSearch(model = titanic_glm_hype, hyper_params = hyper_params,
                     search_criteria = {'strategy': "Cartesian"})
titanitc_grid.train(x = predictors, y = response, training_frame = titanic_train, validation_frame = titanic_valid)


# If you want to sort by r2 then try this
titanic_sorted_grid = titanitc_grid.get_grid(sort_by='r2', decreasing=False)
print(titanic_sorted_grid)





import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske


# # Titanic Facts
# http://www.titanicfacts.net/titanic-passengers.html
# 
# Total Passangers: 1317
# 
# Details:
# 
# https://blog.socialcops.com/engineering/machine-learning-python/
# 

titanic_df = pd.read_csv('/Users/avkashchauhan/learn/seattle-workshop/titanic_list.csv')


titanic_df.describe


titanic_df.shape


titanic_df.columns


titanic_df.head()


# # DataSet details
# 
# survival: Survival (0 = no; 1 = yes)
# 
# class: Passenger class (1 = first; 2 = second; 3 = third)
# 
# name: Name
# 
# sex: Sex
# 
# age: Age
# 
# sibsp: Number of siblings/spouses aboard
# 
# parch: Number of parents/children aboard
# 
# ticket: Ticket number
# 
# fare: Passenger fare
# 
# cabin: Cabin
# 
# embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# boat: Lifeboat (if survived)
# 
# body: Body number (if did not survive and body was recovered)
# 

titanic_df['survived'].mean()


titanic_df.groupby('pclass').mean()


class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
class_sex_grouping


class_sex_grouping['survived'].plot.bar()


group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()


print "You can see the data set has lots of missing entities"
titanic_df.count()


# Fixing inconsistencies 
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")
#removing body, cabin and boat features
titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)
#removing all NA values
titanic_df = titanic_df.dropna()


print "You will see the values are consitant now"
titanic_df.count()


# We can also drop 'name','ticket','home.dest' features as it will not help
titanic_df = titanic_df.drop(['name','ticket','home.dest'], axis=1)
titanic_df.count()


titanic_df.sex = preprocessing.LabelEncoder().fit_transform(titanic_df.sex)
titanic_df.sex
# Now SEX convers to 0 and 1 instead of male or female 


titanic_df.embarked = preprocessing.LabelEncoder().fit_transform(titanic_df.embarked)
titanic_df.embarked


# Create a dataframe which has all features we will use for model building
X = titanic_df.drop(['survived'], axis=1).values


y = titanic_df['survived'].values


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)


#Decision Tree Classifier
classify_dt = tree.DecisionTreeClassifier(max_depth=10)


print " This result means the model correctly predicted survival rate of given value %"
classify_dt.fit (X_train, y_train)
scr = classify_dt.score (X_test, y_test)
print "score : " , scr
print "Model is able to correctly predict survival rate of", scr *100 , "% time.."


# Creating a vlidator data which works on 80%-20% 
shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)


def test_classifier(clf):
    scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))


test_classifier(classify_dt)
# Note: If you run shuffle_validator again and then run test classifier, you will see different accuracy


# # Random Forest
# The “Random Forest” classification algorithm will create a multitude of (generally very poor) trees for the data set using different random subsets of the input variables, and will return whichever prediction was returned by the most trees. This helps to avoid “overfitting”, a problem that occurs when a model is so tightly fitted to arbitrary correlations in the training data that it performs poorly on test data.
# 

clf_rf = ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)


# Performing Prediction

clf_rf.fit(X_train, y_train)
clf_rf.score(X_test, y_test)


# # Gradient Boosting
# 
# The “Gradient Boosting” classifier will generate many weak, shallow prediction trees and will combine, or “boost”, them into a strong model. This model performs very well on our data set, but has the drawback of being relatively slow and difficult to optimize, as the model construction happens sequentially so it cannot be parallelized.
# 

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
test_classifier(clf_gb)


# Performing Prediction

clf_gb.fit(X_train, y_train)
clf_gb.score(X_test, y_test)


# # Voting Classifier
# A “Voting” classifier can be used to apply multiple conceptually divergent classification models to the same data set and will return the majority vote from all of the classifiers. For instance, if the gradient boosting classifier predicts that a passenger will not survive, but the decision tree and random forest classifiers predict that they will live, the voting classifier will chose the latter.
# 

eclf = ske.VotingClassifier([('dt', classify_dt), ('rf', clf_rf), ('gb', clf_gb)])
test_classifier(eclf)


# Performing Prediction

eclf.fit(X_train, y_train)
eclf.score(X_test, y_test)


# # Performing Prediction
# 

# Collection 10 records from each passenger class - Create datset of 30 records
passengers_set_1 = titanic_df[titanic_df.pclass == 1].iloc[:10,:].copy()
passengers_set_2 = titanic_df[titanic_df.pclass == 2].iloc[:10,:].copy()
passengers_set_3 = titanic_df[titanic_df.pclass == 3].iloc[:10,:].copy()
passenger_set = pd.concat([passengers_set_1,passengers_set_2,passengers_set_3])
#testing_set = preprocess_titanic_df(passenger_set)


passenger_set.count()
# You must see 30 uniform records


passenger_set.survived.count()


titanic_df.count()


passenger_set_new = passenger_set.drop(['survived'], axis=1)
prediction = clf_rf.predict(passenger_set_new)


passenger_set[passenger_set.survived != prediction]





# # Using Tensorflow with H2O 
# 
# This notebook shows how to use the tensorflow backend to tackle a simple image classification problem.
# 
# We start by connecting to our h2o cluster:
# 

import h2o
h2o.init(port=54321, nthreads=-1)


# Then we make sure that the H2O cluster has the DeepWater distribution
# 

from h2o.estimators.deepwater import H2ODeepWaterEstimator
if not H2ODeepWaterEstimator.available(): exit


# Load some python utilities library 
# 

import sys, os
import os.path
import pandas as pd
import numpy as np
import random


# and finally we configure the IPython notebook to have nice visualizations
# 

get_ipython().magic('matplotlib inline')
from IPython.display import Image, display, HTML
import matplotlib.pyplot as plt


# ## Configuration
# 
# Set the path to your h2o installation
# and download the 'bigdata' dataset using `./gradlew syncBigdataLaptop` from the H2O source distribution.
# 

H2O_PATH=os.path.expanduser("~/h2o-3/")


# ## Image Classification Task
# 
# H2O DeepWater allows you to specify a list of URIs (file paths) or URLs (links) to images, together with a response column (either a class membership (enum) or regression target (numeric)).
# 
# For this example, we use a small dataset that has a few hundred images, and three classes: cat, dog and mouse.
# 

frame = h2o.import_file(H2O_PATH + "/bigdata/laptop/deepwater/imagenet/cat_dog_mouse.csv")
print(frame.dim)
print(frame.head(5))


# To build a LeNet image classification model in H2O, simply specify `network = "lenet"` and the **Tensorflow** backend to use the tensorflow lenet implementation:
# 

model = H2ODeepWaterEstimator(epochs      = 500, 
                              network     = "lenet", 
                              image_shape = [28,28],  ## provide image size
                              channels    = 3,
                              backend     = "tensorflow",
                              model_id    = "deepwater_tf_simple")

model.train(x = [0], # file path e.g. xxx/xxx/xxx.jpg
            y = 1, # label cat/dog/mouse
            training_frame = frame)

model.show()


# If you'd like to build your own Tensorflow network architecture, then this is easy as well.
# In this example script, we are using the **Tensorflow** backend. 
# Models can easily be imported/exported between H2O and Tensorflow since H2O uses Tensorflow's format for model definition.
# 

def simple_model(w, h, channels, classes):
    import json
    import tensorflow as tf    
    # always create a new graph inside ipython or
    # the default one will be used and can lead to
    # unexpected behavior
    graph = tf.Graph() 
    with graph.as_default():
        size = w * h * channels
        x = tf.placeholder(tf.float32, [None, size])
        W = tf.Variable(tf.zeros([size, classes]))
        b = tf.Variable(tf.zeros([classes]))
        y = tf.matmul(x, W) + b

        # labels
        y_ = tf.placeholder(tf.float32, [None, classes])
     
        # accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1),                                                                                                                                                                                                                                   
                                       tf.argmax(y_, 1))                       
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # train
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        
        tf.add_to_collection("train", train_step)
        # this is required by the h2o tensorflow backend
        global_step = tf.Variable(0, name="global_step", trainable=False)
        
        init = tf.initialize_all_variables()
        tf.add_to_collection("init", init)
        tf.add_to_collection("logits", y)
        saver = tf.train.Saver()
        meta = json.dumps({
                "inputs": {"batch_image_input": x.name, "categorical_labels": y_.name}, 
                "outputs": {"categorical_logits": y.name}, 
                "metrics": {"accuracy": accuracy.name, "total_loss": cross_entropy.name},
                "parameters": {"global_step": global_step.name},
        })
        print(meta)
        tf.add_to_collection("meta", meta)
        filename = "/tmp/lenet_tensorflow.meta"
        tf.train.export_meta_graph(filename, saver_def=saver.as_saver_def())
    return filename


filename = simple_model(28, 28, 3, classes=3)


model = H2ODeepWaterEstimator(epochs                  = 500, 
                              network_definition_file = filename,  ## specify the model
                              image_shape             = [28,28],  ## provide expected image size
                              channels                = 3,
                              backend                 = "tensorflow",
                              model_id                = "deepwater_tf_custom")

model.train(x = [0], # file path e.g. xxx/xxx/xxx.jpg
            y = 1, # label cat/dog/mouse
            training_frame = frame)

model.show()


# # Using H2O Machine Learning and Kalman Filters for Machine Prognostics
# 

# # Import and initialize h2o
# 

import h2o
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set()


h2o.init()


# # Import the training and test sets, look at summary statistics
# 

train = h2o.upload_file("train_FD004_processed.csv")
test  = h2o.upload_file("test_FD004_processed.csv")


# # Define the names of the features
# These are both the features in the original data file and some of the features that will be created later on.
# 

# Setup the column names of the training file
index_columns_names =  ["UnitNumber","Cycle"]

weight_column = "Cycle"

# And the name of the to be engineered target variable
dependent_var = ['RemainingUsefulLife']

independent_vars = [column_name for column_name in train.columns if re.search("CountInMode|stdized_SensorMeasure", column_name)]
independent_vars


# ## Cross validation strategy
# Use approximately 80% of the units for model building and cross validate with the other 20% of the units.  Use units as the information for the fold, not individual observations of the units.
# 

fold_column_name = "FoldColumn"
train[fold_column_name] = train["UnitNumber"] % 5


# ## Build a H2O GBM Estimator, train the estimator, and review model results
# 

from h2o.estimators.gbm import H2OGradientBoostingEstimator


gbm_regressor = H2OGradientBoostingEstimator(distribution="laplace", 
                                             score_each_iteration=True,
                                             stopping_metric="MSE", 
                                             stopping_tolerance=0.001,
                                             stopping_rounds=5,
                                             max_depth=10, ntrees=300)
gbm_regressor.train(x=independent_vars, y=dependent_var, 
                    training_frame=train, weights_column=weight_column,
                    fold_column=fold_column_name)


gbm_regressor


# ## Prediction on the test data
# Using the best model from the grid search (using MSE)
# 
# There is previous version of the work that uses the final scoring to pick the best model, this was kept simple on purpose.
# 

best_model = gbm_regressor


# # Prediction post processing
# Because these are linear dynamic systems, with the prior belief that after each operation the unit's remaining useful life decreases by one, a Kalman filter is used to post process and ensemble the data.
# 
# ### Signal processing using Kalman smoothing filter
# Kalman filters use the prior belief about the state of a system, and ensembles measurement of the next measured state and model of how the system evolves from prior state (physics!) to make an improved estimate of the current system state.  In the absense of having the 'physics' and knowledge about the sensors, here we estimate the parameters for the Kalman filter.  Kalman filters are like smoothers, but  use future state information in addition to prior state information.
# 
# A visual introduction to Kalman filters is available at http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/ .  A great hackers introduction to Kalman and Bayseian filters is at  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python .
# 
# <img src="cycle.png">
# 
# The hypothesis is that the state of the systems under consideration evolves in this time series manner and that remaining useful life decreases by 1 each operation. 
# 
# H2O now has Discrete Cosine Transforms (DCT) so those can be applied to data at scale in a distributed, in memory environment:  https://0xdata.atlassian.net/browse/PUBDEV-1865.  Also, see SciPy's signal and interpolation packages.
# 
# (image courtesy of http://www.codeproject.com/Articles/865935/Object-Tracking-Kalman-Filter-with-Ease)
# 

# ### Use the main model and each cross validated model to build an ensemble of predictions
# 

def sensor_preds(frame):
    frame["predict"] = ((frame["predict"] < 0.).ifelse(0., frame["predict"]))[0]
    return frame

models_for_pred = [best_model]+best_model.xvals
preds = [ sensor_preds(model.predict(test)) for model in models_for_pred ]
index = test[["UnitNumber","Cycle"]]
for i,pred in enumerate(preds):
    if i == 0:  # special handling for first model
        predictions = index.cbind(preds[i])
    else:
        predictions = predictions.cbind(preds[i])

predictions_df = predictions.as_data_frame(use_pandas=True)


# state is represented as [RUL, -1]
n_dim_state=2
n_dim_obs=len(preds)
a_transition_matrix = np.array([[1,1],[0,1]]) # Dynamics take state of [RUL, -1] and transition to [RUL-1, -1]
r_observation_covariance = np.diag( [ model.mse() for model in models_for_pred ] )
h_observation_matrices = np.array([[1,0] for _ in models_for_pred])


import pykalman as pyk

final_ensembled_preds = {}
pred_cols = [ name for name in predictions_df.columns if "predict" in name]
for unit in predictions_df.UnitNumber.unique():
    preds_for_unit = predictions_df[ predictions_df.UnitNumber == unit ]
    observations = preds_for_unit.as_matrix(pred_cols)
    initial_state_mean = np.array( [np.mean(observations[0]),-1] )
    kf = pyk.KalmanFilter(transition_matrices=a_transition_matrix,                          initial_state_mean=initial_state_mean,                          observation_covariance=r_observation_covariance,                          observation_matrices=h_observation_matrices,                          n_dim_state=n_dim_state, n_dim_obs=n_dim_obs)
    mean,_ = kf.filter(observations)
    final_ensembled_preds[unit] = mean


final_preds = { k:final_ensembled_preds[k][-1][0] for k in final_ensembled_preds.keys() }


final_preds_df = pd.DataFrame.from_dict(final_preds,orient='index')
final_preds_df.columns = ['predicted']


sns.tsplot(predictions_df[ predictions_df.UnitNumber == 2 ]["predict"])


sns.tsplot(final_ensembled_preds[2].T[0])


# ## Do final scoring on the test data
# 

actual_RUL = pd.read_csv("RUL_FD004.txt",header=None,names=["actual"])
actual_RUL.index = actual_RUL.index+1
actual_preds = actual_RUL.join(final_preds_df)

def score(x):
    diff = x.predicted-x.actual
    result = np.expm1(diff/-13.) if diff < 0. else np.expm1(diff/10.)
    return result

actual_preds["score"] = actual_preds.apply(score, axis=1)
sum(actual_preds.score)/len(actual_preds)


g = sns.regplot("actual", "predicted", data=actual_preds, fit_reg=False)
g.set(xlim=(0, 160), ylim=(0, 180));
g.axes.plot((0, 160), (0, 160), c=".2", ls="--");


h2o.shutdown(prompt=False)


