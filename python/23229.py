# # ISLR- Python Ch4 Applied 12
# 

import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # emulate pretty r-style plots


# ## A. Function Printing 2**3
# 

def power():
    """ prints out 2**3 """
    print(2**3)
    
power()


# ## B-C. Function Printing x**a
# 

def power2(x,a):
    """ prints x to the power of a:float """
    print(x**a)
    

power2(3,8)
power2(10,3)
power2(8,17)
power2(131,3)


# ## D. Function Returning x**a
# 

def power3(x,a):
    """ returns x raised to float a """
    return(x**a)


# ## E. Plots using Power3
# 

# call power3 with exponent=2 on an array
x = np.arange(1,10)
y = power3(x,2)

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize=(16,4))

# Plot x vs y
ax1.plot(x,y,linestyle='-.', marker='o')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Plot log(x) vs y
ax2.semilogx(x,y, linestyle='-.', marker='o')
ax2.set_xlabel('log(x)')
ax2.set_ylabel('y')

# Plot x vs log(y)
ax3.semilogy(x,y, linestyle='-.', marker='o')
ax3.set_xlabel('x')
ax3.set_ylabel('log(y)')

# Plot log log
ax4.loglog(x,y, linestyle='-.', marker='o')
ax4.set_xlabel('log(x)')
ax4.set_ylabel('log(y)')

plt.tight_layout()


# ## F. PlotPower Function
# 

def plot_power(x,a):
    """Plots x vs x**a """
    # generate dependent
    y = x**a
    
    # create plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x,y, linestyle = '-.', marker = 'o')
    ax.set_xlabel('x',fontsize=16)
    ax.set_ylabel('x**'+str(a),fontsize=16)

plot_power(np.arange(1,11),3)


# # ISLR - Python Ch8 Applied 11
# 

# - [Import Caravan Dataset](#Import-Caravan-Dataset)
# - [Split the data](#Split-the-data)
# - [Build a Boosting Model](#Build-a-Boosting-Model)
# - [Predict with Boosting Model](#Predict-with-Boosting-Model)
# - [Build Confusion Matrix](#Build-Confusion-Matrix)
# - [Compare with KNN Model](#Compare-with-KNN-Model)
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

get_ipython().magic('matplotlib inline')


df = pd.read_csv('../../../data/Caravan.csv', index_col=0)


df.head()


# ## Split the data
# 

# Define the predictors and the response variables
predictors = df.columns.tolist()
predictors.remove('Purchase')

X = df[predictors].values
y = df['Purchase'].values

# use the first 1000 as training and the remainder for testing
X_train = X[0:1000]
X_test = X[1000:]
y_train = y[0:1000]
y_test = y[1000:]


# ## Build a Boosting Model
# 

# build and fit a boosting model to the training data
booster = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, max_depth=3, 
                                     random_state=0)
boost_est = booster.fit(X_train, y_train)


# get the variable importance
Importances = pd.DataFrame(boost_est.feature_importances_, index=predictors, 
             columns=['Importance']).sort_values(by='Importance', ascending=False)
Importances.head(8)


# The above dataframe list the top 8 predictors for classifying the response variable 'Purchase'.
# 

# ## Predict with Boosting Model
# 

y_pred = boost_est.predict_proba(X_test)
print(y_pred)


# The above gives the class probabilities for [No Yes] for each instance in the test set.
# The predicted class according to the problem are 'yes' if the 'yes' probability exceeds 20% and 'No' otherwise.
# 

# if the yes probability exceeds 0.2 then assign it as a purchase
pred_purchase = ['No'if row[1] < 0.2 else 'Yes' for row in y_pred ]


# ## Build Confusion Matrix
# 

cm = confusion_matrix(y_true = y_test, y_pred=pred_purchase, labels=['No', 'Yes'])
print(cm)


# The CM matrix is [[NN NY]
#                   [YN, YY] 
# where C_ij is equal to the number of observations known to be in group i but 
# predicted to be in group j.

#so the fraction predicted to be Yes that are actually Yes is 
cm[1,1]/(cm[1,1]+cm[0,1])


# So 16% that are predicted to be in class Yes are actually Yes. Apply a KNN model to this data for comparison.
# 

# ## Compare with KNN Model
# 

# Build KNN clasifier
knn_est = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
# make predictons
predicted_class = knn_est.predict_proba(X_test)
# if the yes probability exceeds 0.2 then assign it as a purchase
knn_pred = ['No'if row[1] < 0.2 else 'Yes' for row in predicted_class ]
# build confusion matrix
cm = confusion_matrix(y_true = y_test, y_pred=knn_pred, labels=['No', 'Yes'])
print(cm)


#so the fraction predicted to be Yes that are actually Yes is 
cm[1,1]/(cm[1,1]+cm[0,1])


# So the Boosting performs nearly 2x better than the KNN on this hard classification problem.
# 




# # ISLR- Python Ch4 Applied 10
# 

# - [Load Dataset](#Load-Datasets)
# - [A. Numerical and Graphical Summary of Data](#A.-Numerical-and-Graphical-Summary-of-Data)
# - [B. Logistic Regression of Market Direction](#B.-Logistic-Regression-of-Market-Direction)
# - [C. Examine Confusion Matrix](#C.-Examine-Confusion-Matrix)
# - [D. Split Data and Re-Examine](#D.-Split-Data-and-Re-Examine)
# - [E. Linear Discriminant Aanlysis of Market Direction](#E.-Linear-Discriminant-Analysis-of-Market-Direction)
# - [F. Quadratic Discriminant Analysis of Market Direction](#F.-Quadratic-Discriminant-Analysis-of-Market-Direction)
# - [G. K-Nearest Neighbors Analysis of Market Direction](#G.-K-Nearest-Neighbors-Analysis-of-Market-Direction)
# - [H-I. Experiment with Predictors](#H-I.-Experiment-with-Predictors)
# 

## perform imports and set-up
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from scipy import stats

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # emulate pretty r-style plots

# print numpy arrays with precision 4
np.set_printoptions(precision=4)


# ## Load Weekly Dataset
# 

df = pd.read_csv('../../../data/Weekly.csv')
print('Weekly dataframe shape =', df.shape)
df.head()


# ## A. Numerical and Graphical Summary of Data
# 

# We are interested in the relationship between each of the predictors (lags and volume) with the market direction. We can get an idea if any of these predictors are correlated with the response by looking at the correlation matrix.
# 

# Compute correlation coeffecient matrix
correlations = df.corr(method='pearson')
print(correlations)


# As with the Smarket data, the strongest correlation is between year and volume. The Today's return value is weakly correlated with lag1, lag2 and lag3. Lets make a few plots.
# 

# Plot the Trading Volume vs. Year
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,4));

ax1.scatter(df.Year.values,df.Volume.values, facecolors='none', edgecolors='b');
ax1.set_xlabel('Year');
ax1.set_ylabel('Volume in Billions');

# Plot Lag1 vs Today's return
ax2.scatter(df.Lag1.values, df.Today.values, facecolors='none', edgecolors='b' );
ax2.set_xlabel('Lag1 Percent Return');
ax2.set_ylabel('Today\'s Percent Return');

# Plot Lag1 vs Today's return
ax3.scatter(df.Lag2.values, df.Today.values, facecolors='none', edgecolors='b' );
ax3.set_xlabel('Lag2 Percent Return');
ax3.set_ylabel('Today\'s Percent Return');


# As expected given the correlation matrix, there is a strong relationship between year and trading volume (stock market volume increases over time) and a very weak relationship (not visible graphically) between Today's percentage return and the previous days returns (i.e. the lags). We should not expect our models to perform exceedingly well because the is little relationship between the predictors and response.
# 

# ## B. Logistic Regression of Market Direction
# 

# We will now build a logistic regression model for the market direction using the entire dataset using all the lags and volume to attempt to predict the direction response variable.
# 

# Construct Design Matrix #
###########################
predictors = df.columns[1:7] # the lags and volume
X = sm.add_constant(df[predictors])

# Convert the Direction to Binary #
###################################
y = np.array([1 if el=='Up' else 0 for el in df.Direction.values])

# Construct the logit model #
###########################
logit = sm.Logit(y,X)
results=logit.fit()
print(results.summary())


# Only the Lag2 variable of the predictors is significant at an alpha confidence level of 0.05. The predictors that have the lowest p-values are Lag1 and Lag2. A model based on these two variables would be reasonable to attempt.
# 

# Get the predicted results for the full dataset
y_predicted = results.predict(X)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y, bins=2)[0]
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# PRECISION: On days when the model predicts the market to increase, the probability the market does increase is 557/(430+557) of 56%. 
# 
# TYPE I ERROR: (AKA false-positive), the probability that we have predicted the market to increase and it does not increase is 430/(430+54) = 89%
# 
# TYPE II ERROR: This is the number of false negatives to all positives. It is 48/(557+48)= 8%
# 
# The sensitivity is therefore 92%. The model is very sensitive to catching all true positives, but this incurs a high false positive rate. This is the inverse relationship between Type I and Type II errors.
# 

# ## D. Split Data and Re-Examine
# 

# We are now going to split the data into a training set and a testing set and refit a logistic model using Lag2 as the only predictor. The training data will be the data from the years 1990 through 2008 and the testing data will be from 2009 through 2010.
# 

# Split Data #
##############
# get the Lag2 values for years less than =  2008
X_train = sm.add_constant(df[df.Year <= 2008].Lag2)
response_train = df[df.Year <= 2008].Direction
# convert responses to 0,1's
y_train = np.array([1 if el=='Up' else 0 for el in response_train])

# for the test set use the years > 2008
X_test = sm.add_constant(df[df.Year > 2008].Lag2)
response_test = df[df.Year > 2008].Direction
y_test = np.array([1 if el=='Up' else 0 for el in response_test])

# Construct Classifier and Fit #
################################
logit = sm.Logit(y_train, X_train)
results = logit.fit()
print(results.summary())
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = results.predict(X_test)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# The model correctly predicts (56+9)/104 = 62.5% of the market directions. We still have high number of Type I errors and low number of Type II errors as with the model with Lag1, Lag2 and volume as predictors. Notice that the p-value of the lag2 coeffecient increased without Lag1 and Volume present.
# 

# ## E. Linear Discriminant Analysis of Market Direction
# 

# We will now try to predict market direction by maiximing the Linear Discriminant function. The LDA module in sklearn will estimate the priors, the vector of class means, and the variance (remember the variance is assumed to be the same for all classes in LDA). Our classes are 'Up' and 'Down', the priors are P('Up') and P('Down') and the means are $\mu_{Up}$ and $\mu_{Down}$.
# 

# Create LDA Classifier and Fit #
#################################
clf = LDA(solver='lsqr', store_covariance=True)
# No constant needed for LDA so reset the X_train
X_train = df[df.Year <= 2008].Lag2.values
# reshape so indexed by two indices
X_train = X_train.reshape((len(X_train),1))

# also go ahead and get test set and responses
X_test = df[df.Year > 2008].Lag2.values
# reshape into so indexed by two indices
X_test = X_test.reshape((len(X_test),1))

clf.fit(X_train, y_train)
print('Priors = ', clf.priors_ )
print('Class Means = ', clf.means_[0], clf.means_[1])
print('Coeffecients = ', clf.coef_)
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = clf.predict(X_test)
#conver the predicted probabilities to class 0 or 1
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# The results of the LDA model are exactly the same as the regression model.
# 

# ## F. Quadratic Discriminant Analysis of Market Direction 
# 

# Now we will use quadratic discriminant analysis on the weekly returns data set. QDA does not assume that all the classes have the same variance and hence their is a quadratic term in the maximization of the density function P(X=x|y=k). Again, sklearn will estimate the class priors, means, and variances and compute the class that maximizes the conditional probability P(y=k|X=x).
# 

# Build Classifier and Fit #
############################
qclf = QDA(store_covariances=True)
qclf.fit(X_train,y_train)

print('Priors = ', qclf.priors_ )
print('Class Means = ', qclf.means_[0], qclf.means_[1])
print('Covariances = ', qclf.covariances_)
print('\n')

# Predict Test Set Responses #
##############################
y_predict = qclf.predict(X_test)
#conver the predicted probabilities to class 0 or 1
y_predicted= np.array(y_predict > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predict, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# For a quadratic decision boundary we need to have multiple predictors with different correlation for each class. Here we have a single predictor so the boundary cannot be quadratic. Thus the model perfoms poorly.
# 

# ## G. K-Nearest Neighbors Analysis of Market Direction
# 

# Build KNN Classifier and Fit #
################################
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

# Predict Test Set Responses #
##############################
y_predicted = clf.predict(X_test)

table = np.histogram2d(y_predicted, y_test , bins=2)[0]
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# Using only 1 Nearest Neighbor, the model performs at chance level.
# 

# Build KNN Classifier and Fit #
################################
clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)

# Predict Test Set Responses #
##############################
y_predicted = clf.predict(X_test)

table = np.histogram2d(y_predicted, y_test , bins=2)[0]
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# ## H-I. Experiment with Predictors
# 

# By varying the k-value, we can empiraically find the minimum test error rate. This occurs at ~ 20 nearest neighbors (see results above). The error rate (40 %) is still higher than in the logistic or LDA models. They perform best on this data set. 
# 
# **Model with Lag1, Lag2 and Lag3**
# 
# It is unlikely that we can make much improvement on the model by adding more uncorrelated variables to the model. Below, I first tried adding Lag3, the predictor with the next lowest p-value. The error rate increases on the test data.
# 
# ** Model with Lag1 x Lag2 Interaction **
# 
# The Lag2 variable in the full logistic model was the only variable with a significant coeffecient. The next most significant term was Lag1. We will build a model with both of these predictors and include an interaction term. We have very little motivation for this because the plots of Lag1 and Lag2 vs today's return do not indicate any strong relationships. 
# 

# ### Model with Three Lags
# 

# Split Data #
##############
predictors = df.columns[1:4]
X_train = sm.add_constant(df[df.Year <= 2008][predictors])
response_train = df[df.Year <= 2008].Direction
# convert responses to 0,1's
y_train = np.array([1 if el=='Up' else 0 for el in response_train])

# for the test set use the years > 2008
X_test = sm.add_constant(df[df.Year > 2008][predictors])
response_test = df[df.Year > 2008].Direction
y_test = np.array([1 if el=='Up' else 0 for el in response_test])

# Construct Classifier and Fit #
################################
logit = sm.Logit(y_train, X_train)
results = logit.fit()
print(results.summary())
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = results.predict(X_test)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# ### Model with Two Lags and Interaction
# 

# Add Interaction #
###################
# add the interaction term to the dataframe
df['Lag1xLag2'] = pd.Series(df.Lag1*df.Lag2, index=df.index)
predictors = ['Lag1', 'Lag2', 'Lag1xLag2']

# Split Data #
##############
X_train = sm.add_constant(df[df.Year <= 2008][predictors])
response_train = df[df.Year <= 2008].Direction
# convert responses to 0,1's
y_train = np.array([1 if el=='Up' else 0 for el in response_train])

# for the test set use the years > 2008
X_test = sm.add_constant(df[df.Year > 2008][predictors])
response_test = df[df.Year > 2008].Direction
y_test = np.array([1 if el=='Up' else 0 for el in response_test])

# Construct Classifier and Fit #
################################
logit = sm.Logit(y_train, X_train)
results = logit.fit()
print(results.summary())
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = results.predict(X_test)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))


# Again we are finding no improvement, the best model is the logistic or LDA model. In these models we used only the significant Lag2 variable and achieved and error rate ~33%. 
# 




# # ISLR-Python: Ch7 Applied 3
# 

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from patsy import cr, dmatrix
from pandas import scatter_matrix

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

np.set_printoptions(precision=4)


# ## Import Auto Dataset
# 

df = pd.read_csv('../../../data/Auto.csv', na_values='?')
df = df.dropna() # drop rows with na values
df.head()


pd.scatter_matrix(df, alpha=0.2, figsize=(12,8));


# ## Natural Splines
# 

# Lets fit a natural spline regressing the acceleration onto the mpg variable. We will use cross-validation to pick the appropriate degrees of freedom for the model and get an estimate for the MSE. To do this we are going to write a function that calls Patsy's dmatrix to construct natural spline design matrices for various degrees of freedom. There is a slight difference in the way Patsy determines degrees of freedom and the way ISLR does. For a natural cubic spline, ISLR deduced that dof = k, where k is the number of knots (see pg 275). However Patsy does not allow for less than 2 degrees of freedom (when centering is on) and 3 dofs (when centering is off). The centering which seems to always be used by ISLR does not change the results but does add one extra knot. It removes the means of each of the resulting columns of the design matrix (see http://patsy.readthedocs.io/en/latest/API-reference.html#patsy.cr )
# 

def natural_spline_cv(predictor, response, dofs=list(np.arange(2,10)), kfolds=5):
    """
    Returns an sklearn LinearRegression model object of a spline regression of predictor(pd.Series) onto response 
    (pd.Series). Uses kfold cross-validation and gan optionally return a plot .
    """
    # cross-val scores- array[dof]
    scores = np.array([])
    X_basis = np.array([])
    
    for dof in dofs:
        # natural spline dmatrix
        formula = r'1 + cr(predictor, df=%d, constraints="center")' %(dof)
        X_basis = dmatrix(formula, data={'predictor':predictor}, return_type='matrix')
     
        # model
        estimator = LinearRegression(fit_intercept=False)
        # cross-validation
        scores = np.append(scores, -np.mean(cross_val_score(estimator, X_basis, response, 
                                                            scoring='mean_squared_error', cv=kfolds)))
    # Build CV Error plot
    fig,ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(dofs, scores, lw=2, color='k', marker='o')
    ax.set_xlabel('Degrees of Freedom')
    ax.set_ylabel('CV Test MSE')
        

scores = natural_spline_cv(df.acceleration, df.mpg)


# So the CV plot suggest a 5 degrees of freedom for the natural cubic spline is the best model (though 8 dofs has a lower CV error). 
# 

scores = natural_spline_cv(df.weight, df.mpg)


# For the weight and MPG 2 degrees of freedom provide a better fit. This is likely because the data is more quadratic than cubic so the natural spline cv function is selecting the least flexible cubic possible.
# 




