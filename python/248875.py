# # Predicting Remaining Useful Life
# <p style="margin:30px">
#     <img style="display:inline; margin-right:50px" width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
#     <img style="display:inline" width=15% src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" alt="NASA" />
# </p>
# 
# The general setup for the problem is a common one: we have a single table of sensor observations over time. Now that collecting information is easier than ever, most industries have already generated *time-series* type problems by the way that they store data. As such, it is crucial to be able to handle data in this form. Thankfully, built-in functionality from [Featuretools](https://www.featuretools.com) handles time varying data well. 
# 
# We'll demonstrate an end-to-end workflow using a [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan) from NASA. This notebook demonstrates a rapid way to predict the Remaining Useful Life (RUL) of an engine using an initial dataframe of time-series data. There are three sections of the notebook:
# 1. [Understand the Data](#Step-1:-Understanding-the-Data)
# 2. [Generate features](#Step-2:-DFS-and-Creating-a-Model)
# 3. [Make predictions with Machine Learning](#Step-3:-Using-the-Model)
# 
# *If you're running this notebook yourself, note that the Challenge Dataset will be downloaded into the data folder in this repository. If you'd prefer to download the data yourself, download and unzip the file from [https://ti.arc.nasa.gov/c/13/](https://ti.arc.nasa.gov/c/6/)*.
# 
# ## Highlights
# * Quickly make end-to-end workflow using time-series data
# * Find interesting automatically generated features
# 
# # Step 1: Understanding the Data
# Here we load in the train data and give the columns names according to the `description.txt` file. 
# 

import numpy as np
import pandas as pd
import featuretools as ft
import utils

utils.download_data()

data_path = 'data/train_FD004.txt'
data = utils.load_data(data_path)
data.head()


# ## NASA Run To Failure Dataset
# In this dataset we have 249 engines (`engine_no`) which are monitored over time (`time_in_cycles`). Each engine had `operational_settings` and `sensor_measurements` recorded for each cycle. The **Remaining Useful Life** (RUL) is the amount of cycles an engine has left before it needs maintenance.
# What makes this dataset special is that the engines run all the way until failure, giving us precise RUL information for every engine at every point in time.
# 
# To train a model that will predict RUL, we can can simulate real predictions on by choosing a random point in the life of the engine and only using the data from before that point. We can create features with that restriction easily by using [cutoff_times](https://docs.featuretools.com/automated_feature_engineering/handling_time.html) in Featuretools.
# 
# The function `make_cutoff_times` in [utils](utils.py) does that sampling for both the `cutoff_time` and the label. You can run the next cell several times and see differing results.
# 

cutoff_times = utils.make_cutoff_times(data)

cutoff_times.head()


# Let's walk through a row of the `cutoff_times` dataframe. In the third row, we have engine number 3. At 3:20 on January 6, the remaining useful life of engine number 3 is 213. Having a dataframe in this format tells Featuretools that the feature vector for engine number 3 should only be calculated with data from before that point in time. 
# 
# To apply Deep Feature Synthesis we need to establish an `EntitySet` structure for our data. The key insight in this step is that we're really interested in our data as collected by `engine`. We can create an `engines` entity by normalizing by the `engine_no` column in the raw data. In the next section, we'll create a feature matrix for the `engines` entity directly rather than the base dataframe of `recordings`.
# 

def make_entityset(data):
    es = ft.EntitySet('Dataset')
    es.entity_from_dataframe(dataframe=data,
                             entity_id='recordings',
                             index='index',
                             time_index='time')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='engines',
                        index='engine_no')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='cycles',
                        index='time_in_cycles')
    return es
es = make_entityset(data)
es


# # Step 2: DFS and Creating a Model
# With the work from the last section in hand, we can quickly build features using Deep Feature Synthesis (DFS). The function `ft.dfs` takes an `EntitySet` and stacks primitives like `Max`, `Min` and `Last` exhaustively across entities. Feel free to try the next step with a different primitive set to see how the results differ!
# 

from featuretools.primitives import Sum, Mean, Std, Skew, Max, Min, Last, CumSum, Diff, Trend
fm, features = ft.dfs(entityset=es, 
                      target_entity='engines',
                      agg_primitives=[Last, Max, Min],
                      trans_primitives=[],
                      cutoff_time=cutoff_times,
                      max_depth=3,
                      verbose=True)
fm.to_csv('simple_fm.csv')


# ## Machine Learning Baselines
# Before we use that feature matrix to make predictions, we should check how well guessing does on this dataset. We can use a `train_test_split` from scikit-learn to split our training data once and for all. Then, we'll check the following baselines:
# 1. Always predict the median value of `y_train`
# 2. Always predict the RUL as if every engine has the median lifespan in `X_train`
# 
# We'll check those predictions by finding the mean of the absolute value of the errors.
# 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

fm = pd.read_csv('simple_fm.csv', index_col='engine_no')
X = fm.copy().fillna(0)
y = X.pop('RUL')

X_train, X_test, y_train, y_test = train_test_split(X, y)

medianpredict1 = [np.median(y_train) for _ in y_test]
print('Baseline by median label: Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict1, y_test)))


recordings_from_train = es['recordings'].df[es['recordings'].df['engine_no'].isin(y_train.index)]
median_life = np.median(recordings_from_train.groupby(['engine_no']).apply(lambda df: df.shape[0]))

recordings_from_test = es['recordings'].df[es['recordings'].df['engine_no'].isin(y_test.index)]
life_in_test = recordings_from_test.groupby(['engine_no']).apply(lambda df: df.shape[0])-y_test

medianpredict2 = (median_life - life_in_test).apply(lambda row: max(row, 0))
print('Baseline by median life: Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict2, y_test)))


# # Step 3: Using the Model
# Now, we can use our created features to fit a `RandomForestRegressor` to our data and see if we can improve on the previous scores.
# 

reg = RandomForestRegressor()
reg.fit(X_train, y_train)
    
preds = reg.predict(X_test)
scores = mean_absolute_error(preds, y_test)
print('Mean Abs Error: {:.2f}'.format(scores))
high_imp_feats = utils.feature_importances(X, reg, feats=10)


# Next, we can apply the exact same transformations (including DFS) to our test data. For this particular case, the real answer isn't in the data so we don't need to worry about cutoff times.
# 

data2 = utils.load_data('data/test_FD004.txt')
es2 = make_entityset(data2)
fm2 = ft.calculate_feature_matrix(entityset=es2, features=features, verbose=True)
fm2.head()


X = fm2.copy().fillna(0)
y = pd.read_csv('data/RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)
preds2 = reg.predict(X)
print('Mean Abs Error: {:.2f}'.format(mean_absolute_error(preds2, y)))

medianpredict1 = [np.median(y_train) for _ in preds2]
print('Baseline by median label: Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict1, y)))

medianpredict2 = (median_life - es2['recordings'].df.groupby(['engine_no']).apply(lambda df: df.shape[0])).apply(lambda row: max(row, 0))
print('Baseline by median life: Mean Abs Error = {:.2f}'.format(
    mean_absolute_error(medianpredict2, y)))


# This is the simple version of a more advanced notebook that can be found in the [second](Advanced%20Featuretools%20RUL.ipynb) notebook. That notebook will show how to use a novel entityset structure, custom primitives, and automated hyperparameter tuning to improve the score.
# 

# # Predicting Remaining Useful Life (advanced)
# <p style="margin:30px">
#     <img style="display:inline; margin-right:50px" width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
#     <img style="display:inline" width=15% src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" alt="NASA" />
# </p>
# 
# This notebook has a more advanced workflow than [the other notebook](Simple%20Featuretools%20RUL%20Demo.ipynb) for predicting Remaining Useful Life (RUL). If you are a new to either this dataset or Featuretools, I would recommend reading the other notebook first. 
# 
# ## Highlights
# * Demonstrate how novel entityset structures improve predictive accuracy
# * Build custom primitives using time-series functions from [tsfresh](https://github.com/blue-yonder/tsfresh)
# * Improve Mean Absolute Error by tuning hyper parameters with [BTB](https://github.com/HDI-Project/BTB)
# 
# Here is a collection of mean absolute errors from both notebooks. Though we've used averages where possible (denoted by \*), the randomness in the Random Forest Regressor and how we choose labels from the train data changes the score.
# 
# |                                 | Train/Validation MAE|  Test MAE|
# |---------------------------------|--------------------------------|
# | Median Baseline                 | 65.81*              | 50.93*   |
# | Simple Featuretools             | 38.41*              | 39.56    |
# | Advanced: Custom Primitives     | 35.30*              | 32.38    |
# | Advanced: Hyperparameter Tuning | 31.10*              | 28.60    |
# 
# 
# # Step 1: Load Data
# We load in the train data using the same function we used in the previous notebook:
# 

import numpy as np
import pandas as pd
import featuretools as ft
import utils

utils.download_data()
data_path = 'data/train_FD004.txt'
data = utils.load_data(data_path)

data.head()


# We also make cutoff times by selecting a random cutoff time from the life of each engine. We're going to make 5 sets of cutoff times to use for cross validation.
# 

from tqdm import tqdm

splits = 5
cutoff_time_list = []

for i in tqdm(range(splits)):
    cutoff_time_list.append(utils.make_cutoff_times(data))

cutoff_time_list[0].head()


# We're going to do something fancy for our entityset. The values for `operational_setting` 1-3 are continuous but create an implicit relation between different engines. If two engines have a similar `operational_setting`, it could indicate that we should expect the sensor measurements to mean similar things. We make clusters of those settings using [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) from scikit-learn and make a new entity from the clusters.
# 

from sklearn.cluster import KMeans

nclusters = 50

def make_entityset(data, nclusters, kmeans=None):
    X = data[['operational_setting_1', 'operational_setting_2', 'operational_setting_3']]
    if kmeans:
        kmeans=kmeans
    else:
        kmeans = KMeans(n_clusters=nclusters).fit(X)
    data['settings_clusters'] = kmeans.predict(X)
    
    es = ft.EntitySet('Dataset')
    es.entity_from_dataframe(dataframe=data,
                             entity_id='recordings',
                             index='index',
                             time_index='time')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='engines',
                        index='engine_no')
    
    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='settings_clusters',
                        index='settings_clusters')
    
    return es, kmeans
es, kmeans = make_entityset(data, nclusters)
es


# # Step 2: DFS and Creating a Model
# In addition to changing our `EntitySet` structure, we're also going to use the [Complexity](http://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.cid_ce) time series primitive from the package [tsfresh](https://github.com/blue-yonder/tsfresh). Any function that takes in a pandas `Series` and outputs a float can be converted into an aggregation primitive using the `make_agg_primitive` function as shown below.
# 

from featuretools.primitives import Last, Max
from featuretools.primitives import make_agg_primitive
import featuretools.variable_types as vtypes

from tsfresh.feature_extraction.feature_calculators import (number_peaks, mean_abs_change, 
                                                            cid_ce, last_location_of_maximum, length)


Complexity = make_agg_primitive(lambda x: cid_ce(x, False),
                              input_types=[vtypes.Numeric],
                              return_type=vtypes.Numeric,
                              name="complexity")

fm, features = ft.dfs(entityset=es, 
                      target_entity='engines',
                      agg_primitives=[Last, Max, Complexity],
                      trans_primitives=[],
                      chunk_size=.26,
                      cutoff_time=cutoff_time_list[0],
                      max_depth=3,
                      verbose=True)

fm.to_csv('advanced_fm.csv')
fm.head()


# We build 4 more feature matrices with the same feature set but different cutoff times. That lets us test the pipeline multiple times before using it on test data.
# 

fm_list = [fm]
for i in tqdm(range(1, splits)):
    fm = ft.calculate_feature_matrix(entityset=make_entityset(data, nclusters, kmeans=kmeans)[0], 
                                     features=features, 
                                     chunk_size=.26, 
                                     cutoff_time=cutoff_time_list[i])
    fm_list.append(fm)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE
def pipeline_for_test(fm_list, hyperparams=[100, 50, 50], do_selection=False):
    scores = []
    regs = []
    selectors = []
    for fm in fm_list:
        X = fm.copy().fillna(0)
        y = X.pop('RUL')
        reg = RandomForestRegressor(n_estimators=int(hyperparams[0]), 
                                    max_features=min(int(hyperparams[1]), 
                                                     int(hyperparams[2])))
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        if do_selection:
            reg2 = RandomForestRegressor(n_jobs=3)
            selector = RFE(reg2, int(hyperparams[2]), step=25)
            selector.fit(X_train, y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            selectors.append(selector)
        reg.fit(X_train, y_train)
        regs.append(reg)
        
        preds = reg.predict(X_test)
        scores.append(mean_absolute_error(preds, y_test))
    return scores, regs, selectors    
scores, regs, selectors = pipeline_for_test(fm_list)
print([float('{:.1f}'.format(score)) for score in scores])
print('Average MAE: {:.1f}, Std: {:.2f}\n'.format(np.mean(scores), np.std(scores)))

most_imp_feats = utils.feature_importances(fm_list[0], regs[0])


data_test = utils.load_data('data/test_FD004.txt')

es_test, _ = make_entityset(data_test, nclusters, kmeans=kmeans)
fm_test = ft.calculate_feature_matrix(entityset=es_test, features=features, verbose=True, chunk_size='cutoff time')
X = fm_test.copy().fillna(0)
y = pd.read_csv('data/RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)
preds = regs[0].predict(X)
print('Mean Abs Error: {:.2f}'.format(mean_absolute_error(preds, y)))


# # Step 3: Feature Selection and Scoring
# Here, we'll use [Recursive Feature Elimination](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html). In order to set ourselves up for later optimization, we're going to write a generic `pipeline` function which takes in a set of hyperparameters and returns a score. Our pipeline will first run `RFE` and then split the remaining data for scoring by a `RandomForestRegressor`. We're going to pass in a list of hyperparameters, which we will tune later. 
# 

# Lastly, we can use that selector and regressor to score the test values.
# 

# # Step 4: Hyperparameter Tuning
# Because of the way we set up our pipeline, we can use a Gaussian Process to tune the hyperparameters. We will use [BTB](https://github.com/HDI-Project/BTB) from the [HDI Project](https://github.com/HDI-Project). This will search through the hyperparameters `n_estimators` and `max_feats` for RandomForest, and the number of features for RFE to find the hyperparameter set that has the best average score.
# 

from btb.hyper_parameter import HyperParameter
from btb.tuning import GP

def run_btb(fm_list, n=30):
    hyperparam_ranges = [
            ('n_estimators', HyperParameter('int', [10, 200])),
            ('max_feats', HyperParameter('int', [5, 50])),
            ('nfeats', HyperParameter('int', [10, 70])),
    ]
    tuner = GP(hyperparam_ranges)

    tested_parameters = np.zeros((n, len(hyperparam_ranges)), dtype=object)
    scores = []
    
    print('[n_est, max_feats, nfeats]')
    best = 45

    for i in tqdm(range(n)):
        tuner.fit(tested_parameters[:i, :], scores)
        hyperparams = tuner.propose()
        cvscores, regs, selectors = pipeline_for_test(fm_list, hyperparams=hyperparams, do_selection=True)
        bound = np.mean(cvscores)
        tested_parameters[i, :] = hyperparams
        scores.append(-np.mean(cvscores))
        if np.mean(cvscores) + np.std(cvscores) < best:
            best = np.mean(cvscores)
            best_hyperparams = [name for name in hyperparams]
            best_reg = regs[0]
            best_sel = selectors[0]
            print('{}. {} -- Average MAE: {:.1f}, Std: {:.2f}'.format(i, 
                                                                      best_hyperparams, 
                                                                      np.mean(cvscores), 
                                                                      np.std(cvscores)))
            print('Raw: {}'.format([float('{:.1f}'.format(s)) for s in cvscores]))

    return best_hyperparams, (best_sel, best_reg)

best_hyperparams, best_pipeline = run_btb(fm_list, n=30)


X = fm_test.copy().fillna(0)
y = pd.read_csv('data/RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)

preds = best_pipeline[1].predict(best_pipeline[0].transform(X))
score = mean_absolute_error(preds, y)
print('Mean Abs Error on Test: {:.2f}'.format(score))
most_imp_feats = utils.feature_importances(X.iloc[:, best_pipeline[0].support_], best_pipeline[1])


# # Appendix: Averaging old scores
# To make a fair comparison between the previous notebook and this one, we should average scores where possible. The work in this section is exactly the work in the previous notebook plus some code for taking the average in the validation step.
# 

from featuretools.primitives import Min
old_fm, features = ft.dfs(entityset=es, 
                      target_entity='engines',
                      agg_primitives=[Last, Max, Min],
                      trans_primitives=[],
                      cutoff_time=cutoff_time_list[0],
                      max_depth=3,
                      verbose=True)

old_fm_list = [old_fm]
for i in tqdm(range(1, splits)):
    old_fm = ft.calculate_feature_matrix(entityset=make_entityset(data, nclusters, kmeans=kmeans)[0], 
                                     features=features, 
                                     cutoff_time=cutoff_time_list[i])
    old_fm_list.append(fm)

old_scores = []
median_scores = []
for fm in old_fm_list:
    X = fm.copy().fillna(0)
    y = X.pop('RUL')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    old_scores.append(mean_absolute_error(preds, y_test))
    
    medianpredict = [np.median(y_train) for _ in y_test]
    median_scores.append(mean_absolute_error(medianpredict, y_test))

print([float('{:.1f}'.format(score)) for score in old_scores])
print('Average MAE: {:.2f}, Std: {:.2f}\n'.format(np.mean(old_scores), np.std(old_scores)))

print([float('{:.1f}'.format(score)) for score in median_scores])
print('Baseline by Median MAE: {:.2f}, Std: {:.2f}\n'.format(np.mean(median_scores), np.std(median_scores)))


y = pd.read_csv('data/RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)
median_scores_2 = []
for ct in cutoff_time_list:
    medianpredict2 = [np.median(ct['RUL'].values) for _ in y.values]
    median_scores_2.append(mean_absolute_error(medianpredict2, y))
print([float('{:.1f}'.format(score)) for score in median_scores_2])
print('Baseline by Median MAE: {:.2f}, Std: {:.2f}\n'.format(np.mean(median_scores_2), np.std(median_scores_2)))





