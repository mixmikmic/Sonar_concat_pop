# # iPython Notebook overview
# 
# iPython Notebook is an interactive scientific computing environment built on Python. It enables us to compose a scientific report with text (for story-telling / explanation of what is going on) and Python code for data analysis.
# 
# A notebook can be used interactively, and it can also be converted to various publication formats, including HTML (for web publication) and PDF.
# 

# # Editing text in Markdown
# 
# [**Markdown**](http://daringfireball.net/projects/markdown) is an easy text file format that allows you to write up a document and apply stylings to create highly presentable output that can be published in various ways, including as PDF and as HTML webpages. Inspect the PDF output of this file (generated by command: **`ipython nbconvert "Python iPython Notebook tutorial.ipynb" --to pdf`**) to see how the following Markdown texts are rendered in the final PDF document.
# 
# # Make a 1st-Level Header by inserting 1 hash character in front
# 
# ## Make a 2nd-Level Header by inserting 2 hash characters in front
# 
# ### Make a 3rd-Level Header by inserting 3 hash characters in front
# 
# _make italic text by putting 1 underline character at each end_
# 
# **make bold text by putting 2 star characters at each end**
# 
# **_then, obviously, this is how to make bold italic text_**
# 
# Make an numbered (ordered) list by simply putting "1."", "2."", "3.", etc., like so:
# 
# 1. First item
# 2. Second item
# 3. Third item
# 
# Make an unordered list by putting "-" and a space in front of each item, like so:
# 
# - Unordered item
# - Unordered item
# - Unordered item
# 
# Make [hyperlinked text, e.g. referring to Google.com](http://www.google.com) like this.
# 
# Images can be embedded easily, like so:
# ![Chicago Booth logo](Chicago Booth logo.jpeg)

# # Embedding Python code
# 
# Embedding Python code, apparently, is what iPython Notebook is mainly about. Usual Python code can be scripted in iPython "cells", and each cell can be run by the key combination **Ctrl** + **Enter** or through **Cell** > **Run** in the navigation bar.
# 
# _Note that plots by `matplotlib` can be generated on the fly with the use of **`%matplotlib inline`** below._
# 

# General Setups and Imports

get_ipython().magic('matplotlib inline')
from matplotlib.pyplot import plot, title, xlabel, ylabel
from numpy import linspace, sin

x = linspace(0, 20, 1000)  # 100 evenly-spaced values from 0 to 50
y = sin(x)

plot(x, y)
xlabel('this is X')
ylabel('this is Y')
title('My Plot')


# # OVERVIEW
# 
# This iPython Notebook uses the **_Boston Housing_** data set to illustrate the following:
# 
# - The **$k$-Nearest Neighbors** (**KNN**) algorithm;
# - The **Bias-Variance Trade-Off**; and
# - The use of **Cross Validation** to estimate Out-of-Sample (OOS) prediction error and determine optimal hyper-parameters, in this case the number of nearest neighbors $k$. 
# 

# # _first, some boring logistics..._
# 
# Let's first import some necessary Python packages and helper modules, and set the random number generator's seed:
# 

# enable In-Line MatPlotLib
get_ipython().magic('matplotlib inline')


# import:
from ggplot import aes, geom_line, geom_point, ggplot, ggtitle, xlab, ylab
from numpy import log, nan, sqrt
from os import system
from pandas import DataFrame, melt, read_csv
from random import seed
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor

system('pip install --upgrade git+git://GitHub.com/ChicagoBoothML/Helpy --no-dependencies')
from ChicagoBoothML_Helpy.CostFunctions import rmse

seed(99)


# # Boston Housing data set
# 
# Let's now import the Boston Housing data into a **`pandas`** data frame:
# 

# read Boston Housing data into data frame
boston_housing = read_csv(
    'https://raw.githubusercontent.com/ChicagoBoothML/DATA___BostonHousing/master/BostonHousing.csv')
boston_housing.sort(columns='lstat', inplace=True)
nb_samples = len(boston_housing)
boston_housing


# Let us then focus on the two variables of interest: **`lstat`** (our predictor variable(s) $\mathbf X$) and **`medv`** (our variable to predict $\mathbf y$). Below is a plot of them against each other:
# 

def plot_boston_housing_data(boston_housing_data,
                             x_name='lstat', y_name='medv', y_hat_name='predicted_medv',
                             title='Boston Housing: medv vs. lstat',
                             plot_predicted=True):
    g = ggplot(aes(x=x_name, y=y_name), data=boston_housing_data) +        geom_point(size=10, color='blue') +        ggtitle(title) +        xlab(x_name) + ylab(y_name)
    if plot_predicted:
        g += geom_line(aes(x=x_name, y=y_hat_name), size=2, color='darkorange')
    return g

plot_boston_housing_data(boston_housing, plot_predicted=False)


# # $k$-Nearest Neighbors algorithm and Bias-Variance Trade-Off
# 
# Let's now try fitting a KNN predictor, with $k = 5$, of _medv_ from _lstat_, using all samples:
# 

k = 5
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' %k)


# With $k = 5$ &ndash; a small number of nearest neighbors &ndash; we have a very "squiggly" predictor, which **fits the training data well** but is **over-sensitive to small changes** in the _lstat_ variable. We call this a **LOW-BIAS**, **HIGH-VARIANCE** predictor. We don't like it.
# 

# Now, with, say, $k = 200$, we have the following:
# 

k = 200
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' % k)


# _Meh..._, we're not exactly jumping around with joy with this one, either. The predictor line is **not over-sensitive**, but **too "smooth" and too simple**, **not responding sufficiently to significant changes** in _lstat_. We call this a **HIGH-BIAS, LOW-VARIANCE** predictor.
# 

# Let's try something in between, say, $k = 50$, to see if we have any better luck:
# 

k = 50
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' % k)


# Now, this looks pretty reasonable, and we'd think this predictor would **generalize well** when facing new, not yet seen, data. This is a **low-bias**, **low-variance** predictor. We love ones like this.
# 
# Hence, the key take-away is that, throughout a range of **hyper-parameter** $k$ from small to large, we have seen a spectrum of corresponding predictors from "low-bias high-variance" to "high-bias low-variance". This phenomenon is called the **BIAS-VARIANCE TRADE OFF**, a fundamental concept in Machine Learning that is applicable to not only KNN alone but to all modeling methods.
# 
# The bias-variance trade-off concerns the **generalizability of a trained predictor** in light of new data it's not seen before. If a predictor has high bias and/or high variance, it will not do well in new cases. **Good, generalizable predictors** need to have **both low bias and low variance**.
# 

# # Out-of-Sample Error and Cross-Validation
# 
# To **quantify the generalizability of a predictor**, we need to estimate its **out-of-sample (OOS) error**, i.e. a certain measure of **how well the predictor performs on data not used in its training process**.
# 
# A popular way to produce such OOS error estimates is to perform **cross validation**. Refer to lecture slides or <a href="http://en.wikipedia.org/wiki/Cross-validation_(statistics)">here</a> for discussions on cross validation.
# 
# Now, let's consider [**Root Mean Square Error** (**RMSE**)](http://en.wikipedia.org/wiki/Root-mean-square_deviation) as our predictor-goodness evaluation criterion and use **5-fold** cross validation **6 times** to pick a KNN predictor that has satisfactory RMSE.
# 

# define Root-Mean-Square-Error scoring/evaluation function
# compliant with what SciKit Learn expects in this guide:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html#sklearn.cross_validation.cross_val_score
def rmse_score(estimator, X, y):
    y_hat = estimator.predict(X)
    return rmse(y_hat, y)

NB_CROSS_VALIDATION_FOLDS = 5
NB_CROSS_VALIDATIONS = 6


k_range = range(2, 201)
cross_validations_avg_rmse_dataframe = DataFrame(dict(k=k_range, model_complexity=-log(k_range)))
cross_validations_avg_rmse_dataframe['cv_avg_rmse'] = 0.
cv_column_names = []
for v in range(NB_CROSS_VALIDATIONS):
    cv_column_name = 'cv_%i_rmse' % v
    cv_column_names.append(cv_column_name)
    cross_validations_avg_rmse_dataframe[cv_column_name] = nan
    for k in k_range:
        knn_model = KNeighborsRegressor(n_neighbors=k)
        avg_rmse_score = cross_val_score(
            knn_model,
            X=boston_housing[['lstat']],
            y=boston_housing.medv,
            cv=KFold(n=nb_samples,
                     n_folds=NB_CROSS_VALIDATION_FOLDS,
                     shuffle=True),
            scoring=rmse_score).mean()
        cross_validations_avg_rmse_dataframe.ix[
            cross_validations_avg_rmse_dataframe.k==k, cv_column_name] = avg_rmse_score
        
    cross_validations_avg_rmse_dataframe.cv_avg_rmse +=        (cross_validations_avg_rmse_dataframe[cv_column_name] -
         cross_validations_avg_rmse_dataframe.cv_avg_rmse) / (v + 1)
        
cross_validations_avg_rmse_longdataframe = melt(
    cross_validations_avg_rmse_dataframe,
    id_vars=['model_complexity', 'cv_avg_rmse'], value_vars=cv_column_names)

ggplot(aes(x='model_complexity', y='value', color='variable'),
       data=cross_validations_avg_rmse_longdataframe) +\
    geom_line(size=1, linetype='dashed') +\
    geom_line(aes(x='model_complexity', y='cv_avg_rmse'),
              data=cross_validations_avg_rmse_longdataframe,
              size=2, color='black') +\
    ggtitle('Cross Validations') +\
    xlab('Model Complexity (-log K)') + ylab('OOS RMSE')


# Best $k$ that minimizes average cross-validation RMSE:
# 

best_k_index = cross_validations_avg_rmse_dataframe.cv_avg_rmse.argmin()
best_k = k_range[best_k_index]
best_k


k = best_k
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' % k)


