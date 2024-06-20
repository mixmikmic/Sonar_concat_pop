# <h1 style="font-family: Georgia; font-size:3em;color:#2462C0; font-style:bold">
# Predicting Loan Repayment</h1><br>
# 
# <p align="center">
# <img src="images/Loans-borrow-repay.jpg"; style="height: 400px; width: 800px">
# </p>
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Introduction</h2><br>
# The two most critical questions in the lending industry are: 1) How risky is the borrower? 2) Given the borrower's risk, should we lend him/her? The answer to the first question determines the interest rate the borrower would have. Interest rate measures among other things (such as time value of money) the riskness of the borrower, i.e. the riskier the borrower, the higher the interest rate. With interest rate in mind, we can then determine if the borrower is eligible for the loan.
# 
# Investors (lenders) provide loans to borrowers in exchange for the promise of repayment with interest. That means the lender only makes profit (interest) if the borrower pays off the loan. However, if he/she doesn't repay the loan, then the lender loses money.
# 
# We'll be using publicly available data from [LendingClub.com](https://www.LendingClub.com). The data covers the 9,578 loans funded by the platform between May 2007 and February 2010. The interest rate is provided to us for each borrower. Therefore, so we'll address the second question indirectly by trying to predict if the borrower will repay the loan by its mature date or not. Through this excerise we'll illustrate three modeling concepts:
# - What to do with missing values.
# - Techniques used with imbalanced classification problems.
# - Illustrate how to build an ensemble model using two methods: blending and stacking, which most likely gives us a boost in performance.
# 
# Below is a short description of each feature in the data set:
# - **credit_policy**: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# - **purpose**: The purpose of the loan such as: credit_card, debt_consolidation, etc.
# - **int_rate**: The interest rate of the loan (proportion).
# - **installment**: The monthly installments (\$) owed by the borrower if the loan is funded.
# - **log_annual_inc**: The natural log of the annual income of the borrower.
# - **dti**: The debt-to-income ratio of the borrower.
# - **fico**: The FICO credit score of the borrower.
# - **days_with_cr_line**: The number of days the borrower has had a credit line.
# - **revol_bal**: The borrower's revolving balance.
# - **revol_util**: The borrower's revolving line utilization rate.
# - **inq_last_6mths**: The borrower's number of inquiries by creditors in the last 6 months.
# - **delinq_2yrs**: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# - **pub_rec**: The borrower's number of derogatory public records.
# - **not_fully_paid**: indicates whether the loan was not paid back in full (the borrower either defaulted or the borrower was deemed unlikely to pay it back).
# 
# Let's load the data and check:
# - Data types of each feature
# - If we have missing values
# - If we have imbalanced data
# 

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import fancyimpute
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsemble
from mlens.visualization import corrmat
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer, RobustScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             accuracy_score, roc_curve,
                             precision_recall_curve, f1_score)
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from keras import models, layers, optimizers

os.chdir("../")
from scripts.plot_roc import plot_roc_and_pr_curves
os.chdir("notebooks/")

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
sns.set_context("notebook")


# Load the data
df = pd.read_csv("../data/loans.csv")

# Check both the datatypes and if there is missing values
print(f"\033[1m\033[94mData types:\n{11 * '-'}")
print(f"\033[30m{df.dtypes}\n")
print(f"\033[1m\033[94mSum of null values in each feature:\n{35 * '-'}")
print(f"\033[30m{df.isnull().sum()}")
df.head()


# Get number of positve and negative examples
pos = df[df["not_fully_paid"] == 1].shape[0]
neg = df[df["not_fully_paid"] == 0].shape[0]
print(f"Positive examples = {pos}")
print(f"Negative examples = {neg}")
print(f"Proportion of positive to negative examples = {(pos / neg) * 100:.2f}%")
plt.figure(figsize=(8, 6))
sns.countplot(df["not_fully_paid"])
plt.xticks((0, 1), ["Paid fully", "Not paid fully"])
plt.xlabel("")
plt.ylabel("Count")
plt.title("Class counts", y=1, fontdict={"fontsize": 20});


# It looks like we have only one categorical feature ("purpose"). Also, six features have missing values (no missing values in labels). Moreover, the data set is pretty imbalanced as expected where positive examples ("not paid fully") are only 19%. We'll explain in the next section how to handle all of them after giving an overview of ensemble methods.
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Modeling</h2><br>
# **Ensemble methods** can be defined as combining several different models (base learners) into final model (meta learner) to reduce the generalization error. It relies on the assumption that each model would look at a different aspect of the data which yield to capturing part of the truth. Combining good performing models the were trained independently will capture more of the truth than a single model. Therefore, this would result in more accurate predictions and lower generalization errors.
# - Almost always ensemble model performance gets improved as we add more models.
# - Try to combine models that are as much different as possible. This will reduce the correlation between the models that will improve the performance of the ensemble model that will lead to significantly outperform the best model. In the worst case where all models are perfectly correlated, the ensemble would have the same performance as the best model and sometimes even lower if some models are very bad. As a result, pick models that are as good as possible.
# 
# Diﬀerent ensemble methods construct the ensemble of models in diﬀerent ways. Below are the most common methods:
# - Blending: Averaging the predictions of all models.
# - Bagging: Build different models on different datasets and then take the majority vote from all the models. Given the original dataset, we sample with replacement to get the same size of the original dataset. Therefore, each dataset will include, on average, 2/3 of the original data and the rest 1/3 will be duplicates. Since each model will be built on a different dataset, it can be seen as a different model. *Random Forest* improves on default bagging trees by reducing the likelihood of strong features to picked on every split. In other words, it reduces the number of features available at each split from $n$ features to, for example, $n/2$ or $log(n)$ features. This will reduce correlation --> reduce variance.
# - Boosting: Build models sequentially. That means each model learns from the residuals of the previous model. The output will be all output of each single model weighted by the learning rate ($\lambda$). It reduces the bias resulted from bagging by learning sequentially from residuals of previous trees (models). 
# - Stacking: Build k models called base learners. Then fit a model to the output of the base learners to predict the final output.
# 
# Since we'll be using Random Fores (bagging) and Gradient Boosting (boosting) classifiers as base learners in the ensemble model, we'll illustrate only averaging and stacking ensemble methods. Therefore, modeling parts would be consisted of three parts:
# - Strategies to deal with missing values.
# - Strategies to deal with imbalanced datasets.
# - Build ensemble models.
# 
# Before going further, the following data preprocessing steps will be applicable to all models:
# 1. Create dummy variables from the feature "purpose" since its nominal (not ordinal) categorical variable. It's also a good practice to drop the first one to avoid linear dependency between the resulted features since some algorithms may struggle with this issue.
# 3. Split the data into training set (70%), and test set (30%). Training set will be used to fit the model, and test set will be to evaluate the best model to get an estimation of generalization error. Instead of having validation set to tune hyperparameters and evaluate different models, we'll use 10-folds cross validation because it's more reliable estimate of generalization error.
# 2. Standardize the data. We'll be using `RobustScaler` so that the standarization will be less influenced by the outliers, i.e. more robust. It centers the data around the median and scale it using *interquartile range (IQR)*. This step will be included in the pipelines for each model as a transformer so we will not do it separately.
# 

# Create dummy variables from the feature purpose
df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
df.head()


# <h3 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Strategies to deal with missing value</h3><br>
# Almost always real world data sets have missing values. This can be due, for example, users didn't fill some part of the forms or some transformations happened while collecting and cleaning the data before they send it to you. Sometimes missing values are informative and weren't generated randomly. Therefore, it's a good practice to add binary features to check if there is missing values in each row for each feature that has missing values. In our case, six features have missing values so we would add six binary features one for each feature. For example, "log_annual_inc" feature has missing values, so we would add a feature "is_log_annual_inc_missing" that takes the values $\in \{0, 1\}$. Good thing is that the missing values are in the predictors only and not the labels. Below are some of the most common strategies for dealing with missing values:
# - Simply delete all examples that have any missing values. This is usually done if the missing values are very small compared to the size of the data set and the missing values were random. In other words, the added binary features did not improve the model. One disadvantage for this strategy is that the model will throw an error when test data has missing values at prediction.
# - Impute the missing values using the mean of each feature separately.
# - Impute the missing values using the median of each feature separately.
# - Use *Multivariate Imputation by Chained Equations (MICE)*. The main disadvantage of MICE is that we can't use it as a transformer in sklearn pipelines and it requires to use the full data set when imputing the missing values. This means that there will be a risk of data leakage since we're using both training and test sets to impute the missing values. The following steps explain how MICE works:
#    - First step: Impute the missing values using the mean of each feature separately.
#    - Second step: For each feature that has missing values, we take all other features as predictors (including the ones that had missing values) and try to predict the values for this feature using linear regression for example. The predicted values will replace the old values for that feature. We do this for all features that have missing values, i.e. each feature will be used once as a target variable to predict its values and the rest of the time as a predictor to predict other features' values. Therefore, one complete cycle (iteration) will be done once we run the model $k$ times to predict the $k$ features that have missing values. For our data set, each iteration will run the linear regression 6 times to predict the 6 features.
#    - Third step: Repeat step 2 until there is not much of change between predictions.
# - Impute the missing values using K-Nearest Neighbors. We compute distance between all examples (excluding missing values) in the data set and take the average of k-nearest neighbors of each missing value. There's no implementation for it yet in sklearn and it's pretty inefficient to compute it since we'll have to go through all examples to calculate distances. Therefore, we'll skip this strategy in this notebook.
# 
# To evaluate each strategy, we'll use *Random Forest* classifier with hyperparameters' values guided by [Data-driven Advice for Applying Machine Learning to Bioinformatics Problems](https://arxiv.org/pdf/1708.05070.pdf) as a starting point.
# 
# Let's first create binary features for missing values and then prepare the data for each strategy discussed above. Next, we'll compute the 10-folds cross validation *AUC* score for all the models using training data.
# 

# Create binary features to check if the example is has missing values for all features that have missing values
for feature in df.columns:
    if np.any(np.isnan(df[feature])):
        df["is_" + feature + "_missing"] = np.isnan(df[feature]) * 1

df.head()


# Original Data
X = df.loc[:, df.columns != "not_fully_paid"].values
y = df.loc[:, df.columns == "not_fully_paid"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)
print(f"Original data shapes: {X_train.shape, X_test.shape}")

# Drop NA and remove binary columns
train_indices_na = np.max(np.isnan(X_train), axis=1)
test_indices_na = np.max(np.isnan(X_test), axis=1)
X_train_dropna, y_train_dropna = X_train[~train_indices_na, :][:, :-6], y_train[~train_indices_na]
X_test_dropna, y_test_dropna = X_test[~test_indices_na, :][:, :-6], y_test[~test_indices_na]
print(f"After dropping NAs: {X_train_dropna.shape, X_test_dropna.shape}")

# MICE data
mice = fancyimpute.MICE(verbose=0)
X_mice = mice.complete(X)
X_train_mice, X_test_mice, y_train_mice, y_test_mice = train_test_split(
    X_mice, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)
print(f"MICE data shapes: {X_train_mice.shape, X_test_mice.shape}")


# Build random forest classifier
rf_clf = RandomForestClassifier(n_estimators=500,
                                max_features=0.25,
                                criterion="entropy",
                                class_weight="balanced")
# Build base line model -- Drop NA's
pip_baseline = make_pipeline(RobustScaler(), rf_clf)
scores = cross_val_score(pip_baseline,
                         X_train_dropna, y_train_dropna,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mBaseline model's average AUC: {scores.mean():.3f}")

# Build model with mean imputation
pip_impute_mean = make_pipeline(Imputer(strategy="mean"),
                                RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_mean,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mMean imputation model's average AUC: {scores.mean():.3f}")

# Build model with median imputation
pip_impute_median = make_pipeline(Imputer(strategy="median"),
                                  RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_median,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mMedian imputation model's average AUC: {scores.mean():.3f}")

# Build model using MICE imputation
pip_impute_mice = make_pipeline(RobustScaler(), rf_clf)
scores = cross_val_score(pip_impute_mice,
                         X_train_mice, y_train_mice,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mMICE imputation model's average AUC: {scores.mean():.3f}")


# Let's plot the feature importances to check if the added binary features added anything to the model.
# 

# fit RF to plot feature importances
rf_clf.fit(RobustScaler().fit_transform(Imputer(strategy="median").fit_transform(X_train)), y_train)

# Plot features importance
importances = rf_clf.feature_importances_
indices = np.argsort(rf_clf.feature_importances_)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(1, 25), importances[indices], align="center")
plt.xticks(range(1, 25), df.columns[df.columns != "not_fully_paid"][indices], rotation=90)
plt.title("Feature Importance", {"fontsize": 16});


# Guided by the 10-fold cross validation *AUC* scores, it looks like all strategies have comparable results and missing values were generated randomly. Also, the added six binary features showed no importance when plotting feature importances from *Random Forest* classifier. Therefore, it's safe to drop those features and use *Median Imputation* method as a transformer later on in the pipeline.
# 

# Drop generated binary features
X_train = X_train[:, :-6]
X_test = X_test[:, :-6]


# <h3 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Strategies to deal with imbalanced data</h3><br>
# Classification problems in most real world applications have imbalanced data sets. In other words, the positive examples (minority class) are a lot less than negative examples (majority class). We can see that in spam detection, ads click, loan approvals, etc. In our example, the positive examples (people who haven't fully paid) were only 19% from the total examples. Therefore, accuracy is no longer a good measure of performance for different models because if we simply predict all examples to belong to the negative class, we achieve 81% accuracy. Better metrics for imbalanced data sets are *AUC* (area under the ROC curve) and f1-score. However, that's not enough because class imbalance influences a learning algorithm during training by making the decision rule biased towards the majority class by implicitly learns a model that optimizes the predictions based on the majority class in the dataset. As a result, we'll explore different methods to overcome class imbalance problem.
# - Under-Sample: Under-sample the majority class with or w/o replacement by making the number of positive and negative examples equal. One of the drawbacks of under-sampling is that it ignores a good portion of training data that has valuable information. In our example, it would loose around 6500 examples. However, it's very fast to train.
# - Over-Sample: Over-sample the minority class with or w/o replacement by making the number of positive and negative examples equal. We'll add around 6500 samples from the training data set with this strategy. It's a lot more computationally expensive than under-sampling. Also, it's more prune to overfitting due to repeated examples.
# - EasyEnsemble: Sample several subsets from the majority class, build a classifier on top of each sampled data, and combine the output of all classifiers. More details can be found [here](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tsmcb09.pdf).
# - Synthetic Minority Oversampling Technique (SMOTE): It over-samples the minority class but using synthesized examples. It operates on feature space not the data space. Here how it works:
#    - Compute the k-nearest neighbors for all minority samples.
#    - Randomly choose number between 1-k.
#    - For each feature:
#        - Compute the difference between minority sample and its randomly chosen neighbor (from previous step).
#        - Multiply the difference by random number between 0 and 1.
#        - Add the obtained feature to the synthesized sample attributes.
#    - Repeat the above until we get the number of synthesized samples needed. More information can be found [here](https://www.jair.org/media/953/live-953-2037-jair.pdf).
# 
# There are other methods such as `EditedNearestNeighbors` and `CondensedNearestNeighbors` that we will not cover in this notebook and are rarely used in practice.
# 
# In most applications, misclassifying the minority class (false negative) is a lot more expensive than misclassifying the majority class (false positive). In the context of lending, loosing money by lending to a risky borrower who is more likely to not fully pay the loan back is a lot more costly than missing the opportunity of lending to trust-worthy borrower (less risky). As a result, we can use `class_weight` that changes the weight of misclassifying positive example in the loss function. Also, we can use different cut-offs assign examples to classes. By default, 0.5 is the cut-off; however, we see more often in applications such as lending that the cut-off is less than 0.5. Note that changing the cut-off from the default 0.5 reduce the overall accuracy but may improve the accuracy of predicting positive/negative examples.
# 
# We'll evaluate all the above methods plus the original model without resampling as a baseline model using the same *Random Forest* classifier we used in the missing values section.
# 

# Build random forest classifier (same config)
rf_clf = RandomForestClassifier(n_estimators=500,
                                max_features=0.25,
                                criterion="entropy",
                                class_weight="balanced")

# Build model with no sampling
pip_orig = make_pipeline(Imputer(strategy="mean"),
                         RobustScaler(),
                         rf_clf)
scores = cross_val_score(pip_orig,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mOriginal model's average AUC: {scores.mean():.3f}")

# Build model with undersampling
pip_undersample = imb_make_pipeline(Imputer(strategy="mean"),
                                    RobustScaler(),
                                    RandomUnderSampler(), rf_clf)
scores = cross_val_score(pip_undersample,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mUnder-sampled model's average AUC: {scores.mean():.3f}")

# Build model with oversampling
pip_oversample = imb_make_pipeline(Imputer(strategy="mean"),
                                    RobustScaler(),
                                    RandomOverSampler(), rf_clf)
scores = cross_val_score(pip_oversample,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mOver-sampled model's average AUC: {scores.mean():.3f}")

# Build model with EasyEnsemble
resampled_rf = BalancedBaggingClassifier(base_estimator=rf_clf,
                                         n_estimators=10, random_state=123)
pip_resampled = make_pipeline(Imputer(strategy="mean"),
                              RobustScaler(), resampled_rf)
                             
scores = cross_val_score(pip_resampled,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mEasyEnsemble model's average AUC: {scores.mean():.3f}")

# Build model with SMOTE
pip_smote = imb_make_pipeline(Imputer(strategy="mean"),
                              RobustScaler(),
                              SMOTE(), rf_clf)
scores = cross_val_score(pip_smote,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)
print(f"\033[1m\033[94mSMOTE model's average AUC: {scores.mean():.3f}")


# EasyEnsemble method has the highest 10-folds CV with average AUC = 0.665.
# 

# <h3 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Build Ensemble methods</h3><br>
# We'll build ensemble models using three different models as base learners:
# - Extra Gradient Boosting
# - Support Vector Classifier
# - Random Forest
# 
# The ensemble models will be built using two different methods:
# - Blending (average) ensemble model. Fits the base learners to the training data and then, at test time, average the predictions generated by all the base learners.
#    - Use VotingClassifier from sklearn that:
#        - fit all the base learners on the training data
#        - at test time, use all base learners to predict test data and then take the average of all predictions.
# - Stacked ensemble model: Fits the base learners to the training data. Next, use those trained base learners to generate predictions (meta-features) used by the meta-learner (assuming we have only one layer of base learners). There are few different ways of training stacked ensemble model:
#    - Fitting the base learners to all training data and then generate predictions using the same training data it was used to fit those learners. This method is more prune to overfitting because the meta learner will give more weights to the base learner who memorized the training data better, i.e. meta-learner won't generate well and would overfit.
#    - Split the training data into 2 to 3 different parts that will be used for training, validation, and generate predictions. It's a suboptimal method because held out sets usually have higher variance and different splits give different results as well as learning algorithms would have fewer data to train.
#    - Use k-folds cross validation where we split the data into k-folds. We fit the base learners to the (k - 1) folds and use the fitted models to generate predictions of the held out fold. We repeat the process until we generate the predictions for all the k-folds. When done, refit the base learners to the full training data. This method is more reliable and will give models that memorize the data less weight. Therefore, it generalizes better on future data.
# 
# We'll use logistic regression as the meta-learner for the stacked model. Note that we can use k-folds cross validation to validate and tune the hyperparameters of the meta learner. We will not tune the hyperparameters of any of the base learners or the meta-learner; however, we will use some of the values recommended by the [Pennsylvania Benchmarking Paper](https://arxiv.org/pdf/1708.05070.pdf). Additionally, we won't use EasyEnsemble in training because, after some experimentation, it didn't improve the AUC of the ensemble model more than 2% on average and it was computationally very expensive. In practice, we sometimes are willing to give up small improvements if the model would become a lot more complex computationally. Therefore, we will use `RandomUnderSampler`. Also, we'll impute the missing values and standardize the data beforehand so that it would shorten the code of the ensemble models and allows use to avoid using `Pipeline`. Additionally, we will plot ROC and PR curves using test data and evaluate the performance of all models.
# 

# Impute the missing data using features means
imp = Imputer()
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

# Standardize the data
std = RobustScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

# Implement RandomUnderSampler
random_undersampler = RandomUnderSampler()
X_res, y_res = random_undersampler.fit_sample(X_train, y_train)
# Shuffle the data
perms = np.random.permutation(X_res.shape[0])
X_res = X_res[perms]
y_res = y_res[perms]
X_res.shape, y_res.shape


# Define base learners
xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                            learning_rate=0.03,
                            n_estimators=500,
                            max_depth=1,
                            subsample=0.4,
                            random_state=123)

svm_clf = SVC(gamma=0.1,
                C=0.01,
                kernel="poly",
                degree=3,
                coef0=10.0,
                probability=True)

rf_clf = RandomForestClassifier(n_estimators=300,
                                max_features="sqrt",
                                criterion="gini",
                                min_samples_leaf=5,
                                class_weight="balanced")

# Define meta-learner
logreg_clf = LogisticRegression(penalty="l2",
                                C=100,
                                fit_intercept=True)

# Fitting voting clf --> average ensemble
voting_clf = VotingClassifier([("xgb", xgb_clf),
                               ("svm", svm_clf),
                               ("rf", rf_clf)],
                              voting="soft",
                              flatten_transform=True)
voting_clf.fit(X_res, y_res)
xgb_model, svm_model, rf_model = voting_clf.estimators_
models = {"xgb": xgb_model, "svm": svm_model,
          "rf": rf_model, "avg_ensemble": voting_clf}

# Build first stack of base learners
first_stack = make_pipeline(voting_clf,
                            FunctionTransformer(lambda X: X[:, 1::2]))
# Use CV to generate meta-features
meta_features = cross_val_predict(first_stack,
                                  X_res, y_res,
                                  cv=10,
                                  method="transform")
# Refit the first stack on the full training set
first_stack.fit(X_res, y_res)
# Fit the meta learner
second_stack = logreg_clf.fit(meta_features, y_res)

# Plot ROC and PR curves using all models and test data
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, model in models.items():
            model_probs = model.predict_proba(X_test)[:, 1:]
            model_auc_score = roc_auc_score(y_test, model_probs)
            fpr, tpr, _ = roc_curve(y_test, model_probs)
            precision, recall, _ = precision_recall_curve(y_test, model_probs)
            axes[0].plot(fpr, tpr, label=f"{name}, auc = {model_auc_score:.3f}")
            axes[1].plot(recall, precision, label=f"{name}")
stacked_probs = second_stack.predict_proba(first_stack.transform(X_test))[:, 1:]
stacked_auc_score = roc_auc_score(y_test, stacked_probs)
fpr, tpr, _ = roc_curve(y_test, stacked_probs)
precision, recall, _ = precision_recall_curve(y_test, stacked_probs)
axes[0].plot(fpr, tpr, label=f"stacked_ensemble, auc = {stacked_auc_score:.3f}")
axes[1].plot(recall, precision, label="stacked_ensembe")
axes[0].legend(loc="lower right")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC curve")
axes[1].legend()
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precision")
axes[1].set_title("PR curve")
plt.tight_layout()


# As we can see from the chart above, stacked ensemble model didn't improve the performance. One of the major reasons are that the base learners are considerably highly correlated especially *Random Forest* and *Gradient Boosting* (see the correlation matrix below).
# 

# Plot the correlation between base learners
probs_df = pd.DataFrame(meta_features, columns=["xgb", "svm", "rf"])
corrmat(probs_df.corr(), inflate=True);


# In addition, with classification problems where False Negatives are a lot more expensive than False Positives, we may want to have a model with a high precision rather than high recall, i.e. the probability of the model to identify positive examples from randomly selected examples. Below is the confusion matrix:
# 

second_stack_probs = second_stack.predict_proba(first_stack.transform(X_test))
second_stack_preds = second_stack.predict(first_stack.transform(X_test))
conf_mat = confusion_matrix(y_test, second_stack_preds)
# Define figure size and figure ratios
plt.figure(figsize=(16, 8))
plt.matshow(conf_mat, cmap=plt.cm.Reds, alpha=0.2)
for i in range(2):
    for j in range(2):
        plt.text(x=j, y=i, s=conf_mat[i, j], ha="center", va="center")
plt.title("Confusion matrix", y=1.1, fontdict={"fontsize": 20})
plt.xlabel("Predicted", fontdict={"fontsize": 14})
plt.ylabel("Actual", fontdict={"fontsize": 14});


# Let's finally check the partial dependence plots to see what are the most important features and their relationships with whether the borrower will most likely pay the loan in full before mature data. we will plot only the top 8 features to make it easier to read.
# 

# Plot partial dependence plots
gbrt = GradientBoostingClassifier(loss="deviance",
                                  learning_rate=0.1,
                                  n_estimators=100,
                                  max_depth=3,
                                  random_state=123)
gbrt.fit(X_res, y_res)
fig, axes = plot_partial_dependence(gbrt, X_res,
                                    np.argsort(gbrt.feature_importances_)[::-1][:8],
                                    n_cols=4,
                                    feature_names=df.columns[:-6],
                                    figsize=(14, 8))
plt.subplots_adjust(top=0.9)
plt.suptitle("Partial dependence plots of borrower not fully paid\n"
             "the loan based on top most influential features")
for ax in axes: ax.set_xticks(())
for ax in [axes[0], axes[4]]: ax.set_ylabel("Partial dependence")


# As we might expected, borrowers with lower annual income and less FICO scores are less likely to pay the loan fully; however, borrowers with lower interest rates (riskier) and smaller installments are more likely to pay the loan fully.
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Conclusion</h2><br>
# Most classification problems in the real world are imbalanced. Also, almost always data sets have missing values. In this notebook, we covered strategies to deal with both missing values and imbalanced data sets. We also explored different ways of building ensembles in sklearn. Below are some takeaway points:
# - There is no definitive guide of which algorithms to use given any situation. What may work on some data sets may not necessarily work on others. Therefore, always evaluate methods using cross validation to get a reliable estimates.
# - Sometimes we may be willing to give up some improvement to the model if that would increase the complexity much more than the percentage change in the improvement to the evaluation metrics.
# - In some classification problems, *False Negatives* are a lot more expensive than *False Positives*. Therefore, we can reduce cut-off points to reduce the False Negatives.
# - When building ensemble models, try to use good models that are as different as possible to reduce correlation between the base learners. We could've enhanced our stacked ensemble model by adding *Dense Neural Network* and some other kind of base learners as well as adding more layers to the stacked model.
# - EasyEnsemble usually performs better than any other resampling methods.
# - Missing values sometimes add more information to the model than we might expect. One way of capturing it is to add binary features for each feature that has missing values to check if each example is missing or not.
# 

# <h1 style="font-family: Georgia; font-size:3em;color:#2462C0; font-style:bold">
# Predicting Employee Turnover</h1><br>
# 

# <img src="images/employee-turnover.png"><br>
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Introduction</h2><br>
# **Employee turnover** refers to the percentage of workers who leave an organization and are replaced by new employees. It is very costly for organizations, where costs include but not limited to: separation, vacancy, recruitment, training and replacement. On average, organizations invest between four weeks and three months training new employees. This investment would be a loss for the company if the new employee decided to leave the first year. Furthermore, organizations such as consulting firms would suffer from deterioration in customer satisfaction due to regular changes in Account Reps and/or consultants that would lead to loss of businesses with clients.
# 
# In this notebook, we'll work on simulated HR data from [kaggle](https://www.kaggle.com/ludobenistant/hr-analytics) to build a classifier that helps us predict what kind of employees will be more likely to leave given some attributes. Such classifier would help an organization predict employee turnover and be pro-active in helping to solve such costly matter. We'll restrict ourselves to use the most common classifiers: Random Forest, Gradient Boosting Trees, K-Nearest Neighbors, Logistic Regression and Support Vector Machine. 
# 
# The data has 14,999 examples (samples). Below are the features and the definitions of each one:
# - satisfaction_level: Level of satisfaction {0-1}.
# - last_evaluationTime: Time since last performance evaluation (in years).
# - number_project: Number of projects completed while at work.
# - average_montly_hours: Average monthly hours at workplace.
# - time_spend_company: Number of years spent in the company.
# - Work_accident: Whether the employee had a workplace accident.
# - left: Whether the employee left the workplace or not {0, 1}.
# - promotion_last_5years: Whether the employee was promoted in the last five years.
# - sales: Department the employee works for.
# - salary: Relative level of salary {low, medium, high}.
# 
# Let's first load all the packages.
# 

import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             roc_curve,
                             confusion_matrix)
from sklearn.model_selection import (cross_val_score,
                                     GridSearchCV,
                                     RandomizedSearchCV,
                                     learning_curve,
                                     validation_curve,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from warnings import filterwarnings

os.chdir("../")
from scripts.plot_roc import plot_conf_matrix_and_roc, plot_roc

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context("notebook")
plt.style.use("fivethirtyeight")
filterwarnings("ignore")


# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Data Preprocessing</h2><br>
# Let's take a look at the data (check if there are missing values and the data type of each features):
# 

# Load the data
df = pd.read_csv("data/HR_comma_sep.csv")

# Check both the datatypes and if there is missing values
print("\033[1m" + "\033[94m" + "Data types:\n" + 11 * "-")
print("\033[30m" + "{}\n".format(df.dtypes))
print("\033[1m" + "\033[94m" + "Sum of null values in each column:\n" + 35 * "-")
print("\033[30m" + "{}".format(df.isnull().sum()))
df.head()


# Since there are no missing values, we do not have to do any imputation. However, there are some data preprocessing needed:
# 1. Change **sales** feature name to **department**.
# 2. Convert **salary** into *ordinal categorical* feature since there is intrinsic order between: low, medium and high.
# 3. Create dummy features from **department** feature and drop the first one to avoid linear dependency where some learning algorithms may struggle with.
# 

# Rename sales feature into department
df = df.rename(columns={"sales": "department"})

# Map salary into integers
salary_map = {"low": 0, "medium": 1, "high": 2}
df["salary"] = df["salary"].map(salary_map)

# Create dummy variables for department feature
df = pd.get_dummies(df, columns=["department"], drop_first=True)
df.head()


df.columns[df.columns != "left"].shape


# The data is now ready to be used for modeling. The final number of features are now 17.
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Modeling</h2><br>
# 

# Let's first take a look at the proportion of each class to see if we're dealing with balanced or imbalanced data since each one has its own set of tools to be used when fitting classifiers.
# 

# Get number of positve and negative examples
pos = df[df["left"] == 1].shape[0]
neg = df[df["left"] == 0].shape[0]
print("Positive examples = {}".format(pos))
print("Negative examples = {}".format(neg))
print("Proportion of positive to negative examples = {:.2f}%".format((pos / neg) * 100))
sns.countplot(df["left"])
plt.xticks((0, 1), ["Didn't leave", "Left"])
plt.xlabel("Left")
plt.ylabel("Count")
plt.title("Class counts");


# As the graph shows, we have an imbalanced dataset. As a result, when we fit classifiers on such datasets, we should use metrics other than accuracy when comparing models such as *f1-score* or *AUC* (area under ROC curve). Moreover, class imbalance influences a
# learning algorithm during training by making the decision rule biased towards the majority class by implicitly learns a model that optimizes the predictions based on the majority class in the dataset. There are three ways to deal with this issue:
# 1. Assign a larger penalty to wrong predictions from the minority class.
# 2. Upsampling the minority class or downsampling the majority class.
# 3. Generate synthetic training examples.
# 
# Nonetheless, there is no definitive guide or best practices to deal with such situations. Therefore, we have to try them all and see which one works better on the problem at hand. We'll restrict ourselves to use the first two, i.e assign larger penalty to wrong predictions from the minority class using `class_weight` in classifiers that allows us do that and evaluate upsampling/downsampling on the training data to see which gives higher performance.
# 
# First, split the data into training and test sets using 80/20 split; 80% of the data will be used to train the models and 20% to test the performance of the models. Second, Upsample the minority class and downsample the majority class. For this data set, positive class is the minority class and negative class is the majority class.
# 

# Convert dataframe into numpy objects and split them into
# train and test sets: 80/20
X = df.loc[:, df.columns != "left"].values
y = df.loc[:, df.columns == "left"].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)

# Upsample minority class
X_train_u, y_train_u = resample(X_train[y_train == 1],
                                y_train[y_train == 1],
                                replace=True,
                                n_samples=X_train[y_train == 0].shape[0],
                                random_state=1)
X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))

# Downsample majority class
X_train_d, y_train_d = resample(X_train[y_train == 0],
                                y_train[y_train == 0],
                                replace=True,
                                n_samples=X_train[y_train == 1].shape[0],
                                random_state=1)
X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))

print("Original shape:", X_train.shape, y_train.shape)
print("Upsampled shape:", X_train_u.shape, y_train_u.shape)
print("Downsampled shape:", X_train_d.shape, y_train_d.shape)


# I don't think we need to apply dimensionality reduction such as PCA because: 1) We want to know the importance of each feature in determining who will leave vs who won't (inference). 2) Dimension of the dataset is descent (17 features). However, it's good to see how many principal components needed to explain 90%, 95% and 99% of the variation in the data.
# 

# Build PCA using standarized trained data
pca = PCA(n_components=None, svd_solver="full")
pca.fit(StandardScaler().fit_transform(X_train))
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(12, 6))
plt.bar(range(1, 18), pca.explained_variance_ratio_, align="center",
        color='red', label="Individual explained variance")
plt.step(range(1, 18), cum_var_exp, where="mid", label="Cumulative explained variance")
plt.xticks(range(1, 18))
plt.legend(loc="best")
plt.xlabel("Principal component index", {"fontsize": 14})
plt.ylabel("Explained variance ratio", {"fontsize": 14})
plt.title("PCA on training data", {"fontsize": 16});


cum_var_exp


# Looks like it needs 14, 15 and 16 principal components to capture 90%, 95% and 99% of the variation in the data respectively. In other words, this means that the data is already in a good space since eigenvalues are very close to each other and gives further evidence that we don't need to compress the data.
# 

# The methodology that we'll follow when building the classifiers goes as follows:
# 1. Build a pipeline that handles all the steps when fitting the classifier using scikit-learn's `make_pipeline` which will have two steps:
#     1. Standardizing the data to speed up convergence and make all features on the same scale.
#     2. The classifier (`estimator`) we want to use to fit the model.
# 2. Use `GridSearchCV` to tune hyperparameters using 10-folds cross validation. We can use `RandomizedSearchCV` which is faster and may outperform `GridSearchCV` especially if we have more than two hyperparameters and the range for each one is very big; however, `GridSearchCV` will work just fine since we have only two hyperparameters and descent range.
# 3. Fit the model using training data.
# 5. Plot both confusion matrix and ROC curve for the best estimator using test data.
# 
# Repeat the above steps for *Random Forest, Gradient Boosting Trees, K-Nearest Neighbors, Logistic Regression and Support Vector Machine*. Next, pick the classifier that has the highest cross validation f1 score. Note that some of the hyperparameter ranges will be guided by the paper [Data-driven Advice for Applying Machine Learning to Bioinformatics Problems](https://arxiv.org/pdf/1708.05070.pdf).
# 

# <h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Random Forest</h2><br>
# First, we will start by fitting a Random Forest classifier using unsampled, upsampled and downsampled data. Second, we will evaluate each method using cross validation (CV) f1-score and pick the one with the highest CV f1-score. Finally, we will use that method to fit the rest of the classifiers.
# 
# The only hyperparameters we'll tune are:
# - `max_feature`: how many features to consider randomly on each split. This will help avoid having few strong features to be picked on each split and let other features have the chance to contribute. Therefore, predictions will be less correlated and the variance of each tree will decrease.
# - `min_samples_leaf`: how many examples to have for each split to be a final leaf node.
# 
# Random Forest is an ensemble model that has multiple trees (`n_estimators`), where each tree is a weak learner. The final prediction would be a weighting average or mode of the predictions from all estimators. Note: high number of trees don't cause overfitting.
# 

# Build random forest classifier
methods_data = {"Original": (X_train, y_train),
                "Upsampled": (X_train_u, y_train_u),
                "Downsampled": (X_train_d, y_train_d)}

for method in methods_data.keys():
    pip_rf = make_pipeline(StandardScaler(),
                           RandomForestClassifier(n_estimators=500,
                                                  class_weight="balanced",
                                                  random_state=123))
    
    hyperparam_grid = {
        "randomforestclassifier__n_estimators": [10, 50, 100, 500],
        "randomforestclassifier__max_features": ["sqrt", "log2", 0.4, 0.5],
        "randomforestclassifier__min_samples_leaf": [1, 3, 5],
        "randomforestclassifier__criterion": ["gini", "entropy"]}
    
    gs_rf = GridSearchCV(pip_rf,
                         hyperparam_grid,
                         scoring="f1",
                         cv=10,
                         n_jobs=-1)
    
    gs_rf.fit(methods_data[method][0], methods_data[method][1])
    
    print("\033[1m" + "\033[0m" + "The best hyperparameters for {} data:".format(method))
    for hyperparam in gs_rf.best_params_.keys():
        print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_rf.best_params_[hyperparam])
        
    print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_rf.best_score_) * 100))


# Upsampling yielded the highest CV f1-score with 99.80%. Therefore, we'll be using the upsampled data to fit the rest of the classifiers. The new data now has 18,284 examples with 50% of the examples belong to the positive class and the other 50% belong to the negative example.
# 

X_train_u[y_train_u == 0].shape, X_train_u[y_train_u == 1].shape


# Let's refit the Random Forest with Upsampled data using best hyperparameters tuned above and plot confusion matrix and ROC curve using test data.
# 

# Reassign original training data to upsampled data
X_train, y_train = np.copy(X_train_u), np.copy(y_train_u)

# Delete original and downsampled data
del X_train_u, y_train_u, X_train_d, y_train_d

# Refit RF classifier using best params
clf_rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=50,
                                              criterion="entropy",
                                              max_features=0.4,
                                              min_samples_leaf=1,
                                              class_weight="balanced",
                                              n_jobs=-1,
                                              random_state=123))


clf_rf.fit(X_train, y_train)

# Plot confusion matrix and ROC curve
plot_conf_matrix_and_roc(clf_rf, X_test, y_test)


# <h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Gradient Boosting Trees</h2><br>
# 

# Gradient Boosting trees are the same as Random Forest except for:
# - It starts with small tree and start learning from grown trees by taking into account the residual of grown trees.
# - More trees can lead to overfitting; opposite to Random Forest.
# 
# The two other hyperparameters than `max_features` and `n_estimators` that we're going to tune are:
# - `learning_rate`: rate the tree learns, the slower the better.
# - `max_depth`: number of split each time a tree is growing which limits the number of nodes in each tree.
# 
# Let's fit GB classifier and plot confusion matrix and ROC curve using test data.
# 

# Build Gradient Boosting classifier
pip_gb = make_pipeline(StandardScaler(),
                       GradientBoostingClassifier(loss="deviance",
                                                  random_state=123))

hyperparam_grid = {"gradientboostingclassifier__max_features": ["log2", 0.5],
                   "gradientboostingclassifier__n_estimators": [100, 300, 500],
                   "gradientboostingclassifier__learning_rate": [0.001, 0.01, 0.1],
                   "gradientboostingclassifier__max_depth": [1, 2, 3]}

gs_gb = GridSearchCV(pip_gb,
                      param_grid=hyperparam_grid,
                      scoring="f1",
                      cv=10,
                      n_jobs=-1)

gs_gb.fit(X_train, y_train)

print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_gb.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_gb.best_params_[hyperparam])

print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_gb.best_score_) * 100))


# Plot confusion matrix and ROC curve
plot_conf_matrix_and_roc(gs_gb, X_test, y_test)


# <h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# K-Nearest Neighbors</h2><br>
# KNN is called a lazy learning algorithm because it doesn't learn or fit any parameter. It takes `n_neighbors` points from the training data closest to the point we're interested to predict it's class and take the mode (majority vote) of the classes for the neighboring point as its class. The two hyperparameters we're going to tune are:
# - `n_neighbors`: number of neighbors to use in prediction.
# - `weights`: how much weight to assign neighbors based on:
#    - "uniform": all neighboring points have the same weight.
#    - "distance": use the inverse of euclidean distance of each neighboring point used in prediction.
#    
# Let's fit KNN classifier and plot confusion matrix and ROC curve.
# 

# Build KNN classifier
pip_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
hyperparam_range = range(1, 20)

gs_knn = GridSearchCV(pip_knn,
                      param_grid={"kneighborsclassifier__n_neighbors": hyperparam_range,
                                  "kneighborsclassifier__weights": ["uniform", "distance"]},
                      scoring="f1",
                      cv=10,
                      n_jobs=-1)

gs_knn.fit(X_train, y_train)


print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_knn.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_knn.best_params_[hyperparam])

print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_knn.best_score_) * 100))


plot_conf_matrix_and_roc(gs_knn, X_test, y_test)


# <h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Logistic Regression</h2><br>
# For logistic regression, we'll tune three hyperparameters:
# - `penalty`: type of regularization, L2 or L1 regularization.
# - `C`: the opposite of regularization of parameter $\lambda$. The higher C the less regularization. We'll use values that cover the full range between unregularized to fully regularized where model is the mode of the examples' label.
# - `fit_intercept`: whether to include intercept or not.
# 
# We won't use any non-linearities such as polynomial features.
# 

# Build logistic model classifier
pip_logmod = make_pipeline(StandardScaler(),
                           LogisticRegression(class_weight="balanced"))

hyperparam_range = np.arange(0.5, 20.1, 0.5)

hyperparam_grid = {"logisticregression__penalty": ["l1", "l2"],
                   "logisticregression__C":  hyperparam_range,
                   "logisticregression__fit_intercept": [True, False]
                  }

gs_logmodel = GridSearchCV(pip_logmod,
                           hyperparam_grid,
                           scoring="accuracy",
                           cv=2,
                           n_jobs=-1)

gs_logmodel.fit(X_train, y_train)

print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_logmodel.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_logmodel.best_params_[hyperparam])

print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_logmodel.best_score_) * 100))


plot_conf_matrix_and_roc(gs_logmodel, X_test, y_test)


# <h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Support Vector Machine (SVM)</h2><br>
# SVM is comutationally very expensive to tune it's hyperparameters for two reasons:
# 1. With big datasets, it becomes very slow.
# 2. It has good number of hyperparameters to tune that takes very long time to tune on a CPU.
# 
# Therefore, we'll use recommended hyperparameters' values from the paper we mentioned before that showed to yield the best performane on Penn Machine Learning Benchmark 165 datasets. The hyperparameters that we usually look to tune are:
# - `C`, `gamma`, `kernel`, `degree` and `coef0`
# 

# Build SVM classifier
clf_svc = make_pipeline(StandardScaler(),
                        SVC(C=0.01,
                            gamma=0.1,
                            kernel="poly",
                            degree=5,
                            coef0=10,
                            probability=True))

clf_svc.fit(X_train, y_train)

svc_cv_scores = cross_val_score(clf_svc,
                                X=X_train,
                                y=y_train,
                                scoring="f1",
                                cv=10,
                                n_jobs=-1)

# Print CV
print("\033[1m" + "\033[94m" + "The 10-folds CV f1-score is: {:.2f}%".format(
       np.mean(svc_cv_scores) * 100))


plot_conf_matrix_and_roc(clf_svc, X_test, y_test)


# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Conclusion</h2><br>
# 

# Let’s conclude by printing out the test accuracy rates for all classifiers we’ve trained so far and plot ROC curves. Finally, we’ll pick the classifier that has the highest area under ROC curve.
# 

# Plot ROC curves for all classifiers
estimators = {"RF": clf_rf,
              "LR": gs_logmodel,
              "SVC": clf_svc,
              "GBT": gs_gb,
              "KNN": gs_knn}
plot_roc(estimators, X_test, y_test, (12, 8))

# Print out accuracy score on test data
print("The accuracy rate and f1-score on test data are:")
for estimator in estimators.keys():
    print("{}: {:.2f}%, {:.2f}%.".format(estimator,
        accuracy_score(y_test, estimators[estimator].predict(X_test)) * 100,
         f1_score(y_test, estimators[estimator].predict(X_test)) * 100))


# Even though Random Forest and Gradient Boosting Trees have almost equal auc, Random Forest has higher accuracy rate and an f1-score with 99.27% and 99.44% respectively. Therefore, we safely say Random Forest outperforms the rest of the classifiers. Let's have a look of feature importances from Random Forest classifier.
# 

# Refit RF classifier
clf_rf = RandomForestClassifier(n_estimators=50,
                                criterion="entropy",
                                max_features=0.4,
                                min_samples_leaf=1,
                                class_weight="balanced",
                                n_jobs=-1,
                                random_state=123)


clf_rf.fit(StandardScaler().fit_transform(X_train), y_train)

# Plot features importance
importances = clf_rf.feature_importances_
indices = np.argsort(clf_rf.feature_importances_)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(1, 18), importances[indices], align="center")
plt.xticks(range(1, 18), df.columns[df.columns != "left"][indices], rotation=90)
plt.title("Feature Importance", {"fontsize": 16});


# Looks like the five most important features are:
# - satisfaction_level
# - time_spend_company
# - average_montly_hours
# - number_project
# - lats_evaluation
# 
# The take home message is the following:
# - When dealing with imbalanced classes, accuracy is not a good method for model evaluation. AUC and f1-score are examples of metrics we can use.
# - Upsampling/downsampling, data synthetic and using balanced class weights are good strategies to try to improve the accuracy of a classifier for imbalanced classes datasets.
# - `GridSearchCV` helps tune hyperparameters for each learning algorithm. `RandomizedSearchCV` is faster and may outperform `GridSearchCV` especially when we have more than two hyperparameters to tune.
# - Principal Component Analysis (PCA) isn't always recommended especially if the data is in a good feature space and their eigen values are very close to each other.
# - As expected, ensemble models outperforms other learning algorithms in most cases.
# 

# <h1 style="font-family: Georgia; font-size:3em;color:#2462C0; font-style:bold">
# Gradient Descent Algorithm and Its Variants
# </h1>
# <p align="center">
# <img src="posts_images/gradient_descent_variants/gradient_cover.PNG"; style="width: 600px; height: 400px"><br>
# <caption><center><u><font color="purple">**Figure 1:**</font></u> Trajectory towards local minimum</center></caption>
# </p>
# 

# **Optimization** refers to the task of minimizing/maximizing an objective function $f(x)$ parameterized by $x$. In machine/deep learning terminology, it's the task of minimizing the cost/loss function $J(w)$ parameterized by the model's parameters $w \in \mathbb{R}^d$. Optimization algorithms (in case of minimization) have one of the following goals:
# - Find the global minimum of the objective function. This is feasible if the objective function is convex, i.e. any local minimum is a global minimum.
# - Find the lowest possible value of the objective function within its neighbor. That's usually the case if the objective function is not convex as the case in most deep learning problems.
# 
# There are three kinds of optimization algorithms:
# 
# - Optimization algorithm that is not iterative and simply solves for one point.
# - Optimization algorithm that is iterative in nature and converges to acceptable solution regardless of the parameters initialization such as gradient descent applied to logistic regression.
# - Optimization algorithm that is iterative in nature and applied to a set of problems that have non-convex cost functions such as neural networks. Therefore, parameters' initialization plays a critical role in speeding up convergence and achieving lower error rates.
# 
# **Gradient Descent** is the most common optimization algorithm in *machine learning* and *deep learning*. It is a first-order optimization algorithm. This means it only takes into account the first derivative when performing the updates on the parameters. On each iteration, we update the parameters in the opposite direction of the gradient of the objective function $J(w)$ w.r.t to the parameters where the gradient gives the direction of the steepest ascent. The size of the step we take on each iteration to reach the local minimum is determined by the learning rate $\alpha$. Therefore, we follow the direction of the slope downhill until we reach a local minimum. 
# 
# In this notebook, we'll cover gradient descent algorithm and its variants: *Batch Gradient Descent, Mini-batch Gradient Descent, and Stochastic Gradient Descent*.
# 
# Let's first see how gradient descent and its associated steps works on logistic regression before going into the details of its variants. For the sake of simplicity, let's assume that the logistic regression model has only two parameters: weight $w$ and bias $b$.
# 
# 1. Initialize weight $w$ and bias $b$ to any random numbers.
# 2. Pick a value for the learning rate $\alpha$. The learning rate determines how big the step would be on each iteration.
#     * If $\alpha$ is very small, it would take long time to converge and become computationally expensive.
#     * IF $\alpha$ is large, it may fail to converge and overshoot the minimum.
#     
#     Therefore, plot the cost function against different values of $\alpha$ and pick the value of $\alpha$ that is right before the first value that didn't converge so that we would have a very fast learning algorithm that converges (see figure 2).
# <p align="center">
# <img src="posts_images/gradient_descent_variants/learning_rate.PNG"; style="width: 600px; height: 400px"><br>
# <caption><center><u><font color="purple">**Figure 2:**</font></u> Gradient descent with different learning rates. [Source](http://cs231n.github.io/neural-networks-3/)</center></caption>
# </p>
#     * The most commonly used rates are : *0.001, 0.003, 0.01, 0.03, 0.1, 0.3*.
# 3. Make sure to scale the data if it's on very different scales. If we don't scale the data, the level curves (contours) would be narrower and taller which means it would take longer time to converge (see figure 3).
# <p align="center">
# <img src="posts_images/gradient_descent_variants/normalized-vs-unnormalized.PNG"; style="width: 800px; height: 300px"><br>
# <caption><center><u><font color="purple">**Figure 3:**</font></u> Gradient descent: normalized versus unnormalized level curves</center></caption>
# </p>
#     Scale the data to have $\mu = 0$ and $\sigma = 1$. Below is the formula for scaling each example:
# $$\\{}\frac{x_i - \mu}{\sigma}\tag{1}\\{} $$
# 4. On each iteration, take the partial derivative of the cost function $J(w)$ w.r.t each parameter (gradient):
# $$\frac{\partial}{\partial w}J(w) = \nabla_w J\tag{2}\\{}$$
# $$\frac{\partial}{\partial b}J(w) = \nabla_b J\tag{3}\\{}$$
# The update equations are:
# $$w = w - \alpha \nabla_w J\tag{4}\\{}$$
# $$b = b - \alpha \nabla_b J\tag{5}\\{}$$
#     * For the sake of illustration, assume we don't have bias. If the slope of the current values of $w > 0$, this means that we are  to the right of optimal $w^*$. Therefore, the update will be negative, and will start getting close to the optimal values of $w^*$. However, if it's negative, the update will be positive and will increase the current values of $w$ to converge to the optimal values of $w^*$ (see figure 4):
# <p align="center">
# <img src="posts_images/gradient_descent_variants/gradients.PNG"; style="width: 600px; height: 400px"><br>
# <caption><center><u><font color="purple">**Figure 4:**</font></u> Gradient descent. An illustration of how gradient descent algorithm uses the first derivative of the loss function to follow downhill it's minimum.</center></caption>
# </p>
#     * Continue the process until the cost function converges. That is, until the error curve becomes flat and doesn't change.
#     * In addition, on each iteration, the step would be in the direction that gives the maximum change since it's perpendicular to level curves at each step.
#     
# Now let's discuss the three variants of gradient descent algorithm. The main difference between them is the amount of data we use when computing the gradients for each learning step. The trade-off between them is the accuracy of the gradient versus the time complexity to perform each parameter's update (learning step).
# 

# <h3 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Batch Gradient Descent
# </h3>
# 

# Batch Gradient Descent is when we sum up over all examples on each iteration when performing the updates to the parameters. Therefore, for each update, we have to sum over all examples:
# $$w = w - \alpha \nabla_w J(w)\tag{6}$$
# 
# ```
# for i in range(num_epochs):
#   grad = compute_gradient(data, params)
#   params = params - learning_rate * grad
# ```
# The main advantages:
# - We can use fixed learning rate during training without worrying about learning rate decay.
# - It has straight trajectory towards the minimum and it is guaranteed to converge in theory to the global minimum if the loss function is convex and to a local minimum if the loss function is not convex.
# - It has unbiased estimate of gradients. The more the examples, the lower the standard error.
# 
# The main disadvantages:
# - Even though we can use vectorized implementation, it may still be slow to go over all examples especially when we have large datasets.
# - Each step of learning happens after going over all examples where some examples may be redundant and don't contribute much to the update.
# 

# <h3 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Mini-Batch Gradient Descent
# </h3>
# 

# Instead of going over all examples, Mini-batch Gradient Descent sums up over lower number of examples based on batch size. Therefore, learning happens on each mini-batch of $b$ examples:
# 
# $$w = w - \alpha \nabla_w J(x^{\{i:i + b\}}, y^{\{i: i + b\}}; w)\tag{7}\\{}$$
# 
# - Shuffle the training dataset to avoid pre-existing order of examples.
# - Partition the training dataset into $b$ mini-batches based on the batch size. If the training set size is not divisible by batch size, the remaining will be its own batch.
# ```
# for i in range(num_epochs):
#     np.random.shuffle(data)
#     for batch in radom_minibatches(data, batch_size=32):
#         grad = compute_gradient(batch, params)
#         params = params - learning_rate * grad
# ```
# 
# The batch size is something we can tune. It is usually chosen as power of 2 such as 32, 64, 128, 256, 512, etc. The reason behind it is because some hardware such as GPUs achieve better runtime with common batch sizes such as power of 2.
# 
# The main advantages:
# - Faster than Batch version because it goes through a lot less examples than Batch (all examples).
# - Randomly selecting examples will help avoid redundant examples or examples that are very similar that don't contribute much to the learning.
# - With batch size < size of training set, it adds noise to the learning process that helps improving generalization error.
# - Even though with more examples the estimate would have lower standard error, the return is less than linear compared to the computational burden we incur.
# 
# The main disadvantages:
# - It won't converge. On each iteration, the learning step may go back and forth due to the noise. Therefore, it wanders around the minimum region but never converges.
# - Due to the noise, the learning steps have more oscillations (see figure 5) and requires adding learning-decay to decrease the learning rate as we become closer to the minimum.
# <p align="center">
# <img src="posts_images/gradient_descent_variants/batch-vs-minibatch.PNG"; style="width: 800px; height: 300px"><br>
# <caption><center><u><font color="purple">**Figure 5:**</font></u> Gradient descent: batch versus mini-batch loss function</center></caption>
# </p>
# 
# With large training datasets, we don't usually need more than 2-10 passes over all training examples (epochs). Note: with batch size $b = m$, we get the Batch Gradient Descent.
# 

# <h3 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Stochastic Gradient Descent
# </h3>
# 

# Instead of going through all examples, Stochastic Gradient Descent (SGD) performs the parameters update on each example $(x^i, y^i)$. Therefore, learning happens on every example:
# 
# $$w = w - \alpha \nabla_w J(x^i, y^i; w)\tag{7}$$
# - Shuffle the training dataset to avoid pre-existing order of examples.
# - Partition the training dataset into $m$ examples.
# ```
# for i in range(num_epochs):
#     np.random.shuffle(data)
#     for example in data:
#         grad = compute_gradient(example, params)
#         params = params - learning_rate * grad
# ```
# It shares most of the advantages and the disadvantages with mini-batch version. Below are the ones that are specific to SGD:
# - It adds even more noise to the learning process than mini-batch that helps improving generalization error. However, this would increase the run time.
# - We can't utilize vectorization over 1 example and becomes very slow. Also, the variance becomes large since we only use 1 example for each learning step.
# 
# Below is a graph that shows the gradient descent's variants and their direction towards the minimum:
#     
# <p align="center">
# <img src="posts_images/gradient_descent_variants/batch-vs-minibatch-vs-stochastic.PNG"; style="width: 600px; height: 300px"><br>
# <caption><center><u><font color="purple">**Figure 6:**</font></u> Gradient descent variants' trajectory towards minimum</center></caption>
# </p>
# As the figure above shows, SGD direction is very noisy compared to mini-batch.
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Challenges
# </h2>
# 

# Below are some challenges regarding gradient descent algorithm in general as well as its variants - mainly batch and mini-batch:
# - Gradient descent is a first-order optimization algorithm, which means it doesn't take into account the second derivatives of the cost function. However, the curvature of the function affects the size of each learning step. The gradient measures the steepness of the curve but the second derivative measures the curvature of the curve. Therefore, if:
#     - Second derivative = 0 $\rightarrow$ the curvature is linear. Therefore, the step size = the learning rate $\alpha$.
#     - Second derivative > 0 $\rightarrow$ the curvature is going upward. Therefore, the step size < the learning rate $\alpha$ and may lead to divergence.
#     - Second derivative < 0 $\rightarrow$ the curvature is going downward. Therefore, the step size > the learning rate $\alpha$.
#     
#     As a result, the direction that looks promising to the gradient may not be so and may lead to slow the learning process or even diverge.
# - If Hessian matrix has poor conditioning number, i.e. the direction of the most curvature has much more curvature than the direction of the lowest curvature. This will lead the cost function to be very sensitive in some directions and insensitive in other directions. As a result, it will make it harder on the gradient because the direction that looks promising for the gradient may not lead to big changes in the cost function (see figure 7).
# <p align="center">
# <img src="posts_images/gradient_descent_variants/curvature.PNG"; style="width: 400px; height: 400px"><br>
# <caption><center><u><font color="purple">**Figure 7:**</font></u> Gradient descent fails to exploit the curvature information contained in the Hessian matrix. [Source](http://www.deeplearningbook.org/contents/numerical.html)</center></caption>
# </p>
# - The norm of the gradient $g^Tg$ is supposed to decrease slowly with each learning step because the curve is getting flatter and steepness of the curve will decrease. However, we see that the norm of the gradient is increasing, because of the curvature of the curve. Nonetheless, even though the gradients' norm is increasing, we're able to achieve a very low error rates  (see figure 8).
# <p align="center">
# <img src="posts_images/gradient_descent_variants/gradient_norm.PNG"; style="width: 600px; height: 300px"><br>
# <caption><center><u><font color="purple">**Figure 8:**</font></u> Gradient norm. [Source](http://www.deeplearningbook.org/contents/optimization.html)</center></caption>
# </p>
# - In small dimensions, local minimum is common; however, in large dimensions, saddle points are more common. Saddle point is when the function curves up in some directions and curves down in other directions. In other words, saddle point looks a minimum from one direction and a maximum from other direction (see figure 9). This happens when at least one eigenvalue of the hessian matrix is negative and the rest of eigenvalues are positive.
# <p align="center">
# <img src="posts_images/gradient_descent_variants/saddle.PNG"; style="width: 600px; height: 300px"><br>
# <caption><center><u><font color="purple">**Figure 9:**</font></u> Saddle point</center></caption>
# </p>
# - As discussed previously, choosing a proper learning rate is hard. Also, for mini-batch gradient descent, we have to adjust the learning rate during the training process to make sure it converges to the local minimum and not wander around it. Figuring out the decay rate of the learning rate is also hard and changes with different datasets.
# - All parameter updates have the same learning rate; however, we may want to perform larger updates to some parameters that have their directional derivatives more inline with the trajectory towards the minimum than other parameters.
# 

# <h1 style="font-family: Georgia; font-size:3em;color:#2462C0; font-style:bold">
# Bandit Algorithms: epsilon-Greedy Algorithm</h1>
# 

# A/B testing can be defined as a randomized controlled experiment that allows us to test if there is a causal relationship between a change to a website/app and the user behavior. The change can be visible such as location of a button on the homepage or invisible such as the ranking/recommendation algorithms and backend infrastructure.
# 
# Web/Mobile developers and business stakeholders always face the following dilemma: Should we try out all ideas and explore all options continuously? Or should we exploit the best available option and stick to it?
# The answer is, as in most cases, will be a trade-off between the two extremes. If we explore all the time, we'll collect a lot of data and waste resources in testing inferior ideas and missing sales (e-commerce case). However, if we only exploit the available option and never try new ideas, we would be left behind and loose in the long-term with ever-changing markets.
# 
# In this series, we'll explore solutions offered by **Multi-armed Bandit Algorithms** that have two main advantages over traditional A/B testing:
# - Smoothly decrease exploration over time instead of sudden jumps.
# - Focus resources on better options and not keep evaluating inferior options during the life of the experiment.
# 
# What is **Bandit Algorithms**? Bandit Algorithms are algorithms that try to learn a rule of selecting a sequence of options that balance exploring available options and getting enough knowledge about each option and maximize profits by selecting the best option. Note that during the experiment, we only have knowledge about the options we tried. Therefore, every time we select an option that's not the best one, we incur an opportunity cost of not selecting the best option; however, we also gain a new knowledge (feedback) about the selected option. In other words, we need to have enough feedback about each option to learn the best option. As a result, the best strategy would be to explore more at the beginning of the experiment until we know the best option and then start exploiting that option.

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# epsilon-Greedy Algorithm</h2><br>
# In this notebook, we'll cover **epsilon-Greedy Algorithm**. Greedy Algorithm can be defined as the algorithm that picks the best currently available option without taking into consideration the long-term effect of that decision, which may happen to be a suboptimal decision. Given that, we can define epsilon-Greedy Algorithm as a Greedy Algorithm that adds some randomness when deciding between options: Instead of picking always the best available option, randomly explore other options with a probability = $\epsilon$ or pick the best option with a probability = $1 - \epsilon$. Therefore, we can add randomness to the algorithm by increasing $\epsilon$, which will make the algorithm explores other options more frequently. Additionally, $\epsilon$ is a hyper-parameter that needs to be tuned based on the experiment, i.e. there is no value that works best on all experiments.
# Let's explore how the algorithm works assuming we have two options: A and B (we can think of them as Control and Treatment groups). For each new user:
# - Assume we have a coin that has a probability of coming heads = $\epsilon$ and a probability of coming tails = $1 - \epsilon$. Therefore,
#     - If it comes heads, explore randomly the available options (exploration).
#         - The probability of selecting any option is $\frac{1}{2}$.
#     - If it comes tails, select the best option (exploitation).
#     
# As a result, the probability of selecting any option randomly if we have $N$ options is $\epsilon \frac{1}{N}$; however, the probability of selecting the best option is $1 - \epsilon$ (see figure 1).
# 

# <p align="center">
# <img src="posts_images/bandit_algorithms/epsilon_greedy.PNG" style="height: 400px; width: 800px">
# <caption><center><u><b><font color="purple">Figure 1:</font></b></u> epsilon-Greedy Algorithm</center></caption>
# </p>
# 

# Let's import the needed packages and implement the algorithm.
# 

# Import packages
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add module path to system path
sys.path.append(os.path.abspath("../"))
from utils import plot_algorithm, compare_algorithms

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
sns.set_context("notebook")


class EpsilonGreedy:
    def __init__(self, epsilon, counts=None, values=None):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        z = np.random.random()
        if z > self.epsilon:
            # Pick the best arm
            return np.argmax(self.values)
        # Randomly pick any arm with prob 1 / len(self.counts)
        return np.random.randint(0, len(self.values))

    def update(self, chosen_arm, reward):
        # Increment chosen arm's count by one
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        # Recompute the estimated value of chosen arm using new reward
        value = self.values[chosen_arm]
        new_value = value * ((n - 1) / n) + reward / n
        self.values[chosen_arm] = new_value


# Few things to note from the above implementation:
# - Initialization of values (rewards) affect the long term performance of the algorithm.
# - The larger the sample size (N), the less influential the rewards from the recent options since we are using the average of each option in the values array.
# - Values array will store the estimated values (average) of each option.
# - Counts is just an internal counter that keeps track of the number of times we selected each option.
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Monte Carlo Simulations</h2><br>
# In order to evaluate the algorithm, we will use Monte Carlo simulations. We'll use 5000 simulations to overcome the randomness generated from the random number generator. Also, we'll use Bernoulli distribution to get the reward from each option on each run. For each simulation:
# - Initialize the algorithm with no prior knowledge.
# - Loop over the time horizon:
#     - Select the option.
#     - Draw the reward for the selected option using Bernoulli distribution and the probability defined.
#     - Update the counts and estimated values of selected arm.
# 
# We'll define the % of reward (probability) of each option and test the performance of the algorithm using three different metrics:
# - Probability of selecting the best option.
# - Average rewards. This metric is a better approximation if the options are similar.
# - Cumulative rewards. The previous two metrics are not fair metrics for algorithms with large epsilon where they sacrifice by exploring more options; however, cumulative rewards is what we should care about.
# 
# Moreover, we'll evaluate the algorithm using 5 different values of $\epsilon$: $0.1, 0.2, 0.3, 0.4, 0.5$. Since in the literature they use *arm* instead of *option* for historical reasons, we'll be using *arm* and *option* interchangeably.
# 

class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def draw(self):
        z = np.random.random()
        if z > self.p:
            return 0.0
        return 1.0


def test_algorithm(algo, arms, num_simulations, horizon):
    # Initialize rewards and chosen_arms with zero 2d arrays
    chosen_arms = np.zeros((num_simulations, horizon))
    rewards = np.zeros((num_simulations, horizon))

    # Loop over all simulations
    for sim in range(num_simulations):
        # Re-initialize algorithm's counts and values arrays
        algo.initialize(len(arms))

        # Loop over all time horizon
        for t in range(horizon):
            # Select arm
            chosen_arm = algo.select_arm()
            chosen_arms[sim, t] = chosen_arm

            # Draw from Bernoulli distribution to get rewards
            reward = arms[chosen_arm].draw()
            rewards[sim, t] = reward

            # Update the algorithms' count and estimated values
            algo.update(chosen_arm, reward)

    # Average rewards across all sims and compute cumulative rewards
    average_rewards = np.mean(rewards, axis=0)
    cumulative_rewards = np.cumsum(average_rewards)

    return chosen_arms, average_rewards, cumulative_rewards


np.random.seed(1)
# Average reward by arm
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
# Shuffle the arms
np.random.shuffle(means)
# Each arm will follow and Bernoulli distribution
arms = list(map(lambda mu: BernoulliArm(mu), means))
# Get the index of the best arm to test if algorithm will be able to learn that
best_arm_index = np.argmax(means)
# Define epsilon value to check the performance of the algorithm using each one
epsilon = [0.1, 0.2, 0.3, 0.4, 0.5]

# Plot the epsilon-Greedy algorithm
plot_algorithm(alg_name="epsilon-Greedy", arms=arms, best_arm_index=best_arm_index,
               hyper_params=epsilon, num_simulations=5000, horizon=500, label="eps")


# Few thing to note from the above graphs:
# - Regardless of the epsilon values, all algorithms learned the best option.
# - The algorithm picks options randomly; therefore, it's not guaranteed to always pick the best option even if it found that option. That's the main reason why none of the algorithms achieved a probability = 1 of selecting the best option or average rewards = % rewards of the best option even after they learned the best option.
# - As $\epsilon$ increases $\rightarrow$ increase the exploration $\rightarrow$ increases the chance of picking options randomly instead of the best option.
# - Algorithms with higher epsilon learn quicker but don't use that knowledge in exploiting the best option.
# - Using accuracy in picking the best option and average rewards metrics, the algorithm $\epsilon = 0.1$ outperforms the rest; however, cumulative rewards metric shows that it takes that algorithm long time to outperform the algorithm with $\epsilon = 0.2$.
# - Depends on time planned to run the experiment, different values of epsilons may be more optimal. For example, $\epsilon = 0.2$ is the best value for almost anything at or below 400.
# 
# Let's run the experiment again to see how would the algorithm behave under the following settings:
# - Only two options.
# - 50 options.
# - 5 option that are very similar.
# 

np.random.seed(1)
# Average reward by arm
means = [0.1, 0.9]
n_arms = len(means)
# Shuffle the arms
np.random.shuffle(means)
# Each arm will follow and Bernoulli distribution
arms = list(map(lambda mu: BernoulliArm(mu), means))
# Get the index of the best arm to test if algorithm will be able to learn that
best_arm_index = np.argmax(means)
# Define epsilon value to check the performance of the algorithm using each one
epsilon = [0.1, 0.2, 0.3, 0.4, 0.5]

# Plot the epsilon-Greedy algorithm
plot_algorithm(alg_name="epsilon-Greedy", arms=arms, best_arm_index=best_arm_index,
               hyper_params=epsilon, num_simulations=5000, horizon=500, label="eps")


np.random.seed(1)
# Average reward by arm
means = [i for i in np.random.random(50)]
n_arms = len(means)
# Shuffle the arms
np.random.shuffle(means)
# Each arm will follow and Bernoulli distribution
arms = list(map(lambda mu: BernoulliArm(mu), means))
# Get the index of the best arm to test if algorithm will be able to learn that
best_arm_index = np.argmax(means)
# Define epsilon value to check the performance of the algorithm using each one
epsilon = [0.1, 0.2, 0.3, 0.4, 0.5]

# Plot the epsilon-Greedy algorithm
plot_algorithm(alg_name="epsilon-Greedy", arms=arms, best_arm_index=best_arm_index,
               hyper_params=epsilon, num_simulations=5000, horizon=250, label="eps")


np.random.seed(1)
# Average reward by arm
means = [0.2, 0.18, 0.22, 0.19, 0.21]
n_arms = len(means)
# Shuffle the arms
np.random.shuffle(means)
# Each arm will follow and Bernoulli distribution
arms = list(map(lambda mu: BernoulliArm(mu), means))
# Get the index of the best arm to test if algorithm will be able to learn that
best_arm_index = np.argmax(means)
# Define epsilon value to check the performance of the algorithm using each one
epsilon = [0.1, 0.2, 0.3, 0.4, 0.5]

# Plot the epsilon-Greedy algorithm
plot_algorithm(alg_name="epsilon-Greedy", arms=arms, best_arm_index=best_arm_index,
               hyper_params=epsilon, num_simulations=5000, horizon=500, label="eps")


# - When we had lower number of options, all algorithms were faster at learning the best option which can be seen by the steepness of all curves of the first two graphs when time < 100. As a result, all algorithms had higher cumulative rewards than when we had 5 options.
# - Having large number of options made it hard on all algorithms to learn the best option and may need a lot more time to figure it out.
# - Lastly, when options are very similar (in terms of rewards), the probability of selecting the best option by all algorithms decreases over time. Let's take the algorithm with $\epsilon = 0.1$ and see why is this the case. After some investigation, the algorithm was struggling in differentiating between the best option and the second best option since the difference between the % rewards is 1%. Therefore, the probability of selecting the best arm was around 50%.
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Annealed epsilon-Greedy Algorithm</h2><br>
# Epsilon value plays a major role in the performance of epsilon-Greedy algorithm and has to be tuned to the best of our knowledge in terms of the expectations of the estimated rewards of each option. Nonetheless, this estimation suffers from high uncertainty since most of the times either we have no clue what might work or the results would be against our intuition as user experience research has shown in multiple studies. Therefore, isn't it nice if we can avoid setting up the epsilon values and make the algorithm parameter-free? That's what **Annealed epsilon-Greedy Algorithm** does. We specify the rule of decaying epsilon with time and let the algorithm runs with no hyper-parameter configurations. The rule of we will use here is: $\epsilon = \frac{1}{log(time + 0.0000001)}$. As we can see, at the beginning of the experiment, $\epsilon$ would be close to Inf and that means a lot of exploration; however, as time goes, $\epsilon$ start approaching zero and the algorithm would exploit more and more by selecting the best option.
# 
# We will evaluate the Annealed version using the same settings as before and compare it to standard version.
# 

class AnnealingEpsilonGreedy(EpsilonGreedy):
    def __init__(self, counts=None, values=None):
        self.counts = counts
        self.values = values

    def select_arm(self):
        # Epsilon decay schedule
        t = np.sum(self.counts) + 1
        epsilon = 1 / np.log(t + 0.0000001)

        z = np.random.random()
        if z > epsilon:
            # Pick the best arm
            return np.argmax(self.values)
        # Randomly pick any arm with prob 1 / len(self.counts)
        return np.random.randint(0, len(self.values))


np.random.seed(1)
# Average reward by arm
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
# Shuffle the arms
np.random.shuffle(means)
# Each arm will follow and Bernoulli distribution
arms = list(map(lambda mu: BernoulliArm(mu), means))
# Get the index of the best arm to test if algorithm will be able to learn that
best_arm_index = np.argmax(means)

# Plot the epsilon-Greedy algorithm
plot_algorithm(alg_name="Annealing epsilon-Greedy", arms=arms, best_arm_index=best_arm_index,
               num_simulations=5000, horizon=500)


# Even though the accuracy of selecting the best option and the average rewards of the annealing epsilon-Greedy Algorithm is lower than the standard version, it has higher cumulative rewards. Also, since the real world is uncertain and we may not have any clue about the designed options, it may be preferred to use the annealing version under some scenarios.
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold\">
# Conclusion</h2><br>
# epsilon-Greedy Algorithm works by going back and forth between exploration with probability = $\epsilon$ and exploitation with probability $1 - \epsilon$. Below are some takeaways:
# - Setting the value of epsilon:
#     - If we set $\epsilon = 1$, we would only explore the available options with a probability = $\frac{1}{N}$ of selecting any option. This will enable us to explore a lot of ideas at the expense of wasting resources by evaluating inferior options.
#     - If we set $\epsilon = 0$, we would exploit the best option and never explore any new idea. This strategy would leave up behind our competitors given that the markets are so volatile.
# - Exploration should be high at the beginning of the experiment to gain the knowledge about all the available options. It should decrease as a function of time where at some point after having enough data about all options, the algorithm should focus on exploiting the best option.
# - All algorithms with different epsilon values learned the best option; however, they differ by the level of randomness of each algorithm in keep randomly exploring available options.
# - To get the best results of any Bandit algorithm, we should have a lot of data, which means to run the experiment longer in most cases.
# - For experiments that run for short period of time, traditional A/B testing may be better.
# - Initialization of estimated rewards can affect the long-term performance of the algorithm. As a result, we may need to use previous experience and intuition to guide our initial values.
# 

# <h1 style="font-family: Georgia; font-size:3em;color:#2462C0; font-style:bold">
# Coding Neural Network - Gradient Checking</h1><br>
# 

# In the previous notebook, [*Coding Neural Network - Forward Propagation and Backpropagation*](https://nbviewer.jupyter.org/github/ImadDabbura/blog-posts/blob/master/notebooks/Coding-Neural-Network-Forwad-Back-Propagation.ipynb), we implemented both forward propagation and backpropagation in `numpy`. However, implementing backpropagation from scratch is usually more prune to bugs/errors. Therefore, it's necessary before running the neural network on training data to check if our implementation of backpropagation is correct. Before we start, let's revisit what back-propagation is: We loop over the nodes in reverse topological order starting at the final node to compute the derivative of the cost with respect to each edge's node tail. In other words, we compute the derivative of cost function with respect to all parameters, i.e $\frac{\partial J}{\partial \theta}$ where $\theta$ represents the parameters of the model.
# 
# The way to test our implementation is by computing numerical gradients and compare it with gradients from backpropagation (analytical). There are two way of computing numerical gradients:
# - Right-hand form:
# $$\frac{J(\theta + \epsilon) - J(\theta)}{\epsilon}\tag{1}$$
# - Two-sided form (see figure 1):
# $$\frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2 \epsilon}\tag{2}$$
# <p align="center">
# <img src="posts_images/coding_nn_from_scratch/two_sided_gradients.png" style="height: 400px; width: 700px">
# <caption><center><u><b><font color="purple"> Figure 1:</font></b></u> Two-sided numerical gradients<center></caption>
# </p>
# Two-sided form of approximating the derivative is closer than the right-hand form. Let's illustrate that with the following example using the function $f(x) = x^2$ by taking its derivative at $x = 3$.
# - Analytical derivative:
# $$\nabla_x f(x) = 2x\ \Rightarrow\nabla_x f(3) = 6$$
# - Two-sided numerical derivative:
# $$\frac{(3 + 1e-2)^2 - (3 - 1e-2)^2}{2 * 1e-2} = 5.999999999999872$$
# - Right-hand numerical derivative:
# $$\frac{(3 + 1e-2)^2 - 3^2}{1e-2} = 6.009999999999849$$
# As we see above, the difference between analytical derivative and two-sided numerical gradient is almost zero; however, the difference between analytical derivative and right-sided derivative is 0.01. Therefore, we'll use two-sided epsilon method to compute the numerical gradients.
# 
# In addition, we'll normalize the difference between numerical. gradients and analytical gradients using the following formula:
# $$$$
# $$\frac{\|grad - grad_{approx}\|_2}{\|grad\|_2 + \|grad_{approx}\|_2}\tag{3}$$
# $$$$
# If the difference is $\leq 10^{-7}$, then our implementation is fine; otherwise, we have a mistake somewhere and have to go back and revisit backpropagation code.
# 
# Below are the steps needed to implement gradient checking:
# 1. Pick random number of examples from training data to use it when computing both numerical and analytical gradients.
#     - Don't use all examples in the training data because gradient checking is very slow.
# 2. Initialize parameters.
# 3. Compute forward propagation and the cross-entropy cost.
# 4. Compute the gradients using our back-propagation implementation.
# 5. Compute the numerical gradients using the two-sided epsilon method.
# 6. Compute the difference between numerical and analytical gradients.
# 
# We'll be using functions we wrote in *"Coding Neural Network - Forward Propagation and Backpropagation"* notebook to initialize parameters, compute forward propagation and back-propagation as well as the cross-entropy cost.
# 
# Let's first import the data.
# 

# Loading packages
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import seaborn as sns

sys.path.append("../scripts/")
from coding_neural_network_from_scratch import (initialize_parameters,
                                                L_model_forward,
                                                L_model_backward,
                                                compute_cost)

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context("notebook")
plt.style.use("fivethirtyeight")


# Import the data
train_dataset = h5py.File("../data/train_catvnoncat.h5")
X_train = np.array(train_dataset["train_set_x"]).T
y_train = np.array(train_dataset["train_set_y"]).T
X_train = X_train.reshape(-1, 209)
y_train = y_train.reshape(-1, 209)

X_train.shape, y_train.shape


# Next, we'll write helper functions that faciltate converting parameters and gradients dictionaries into vectors and then re-convert them back to dictionaries.
# 

def dictionary_to_vector(params_dict):
    """
    Roll a dictionary into a single vector.

    Arguments
    ---------
    params_dict : dict
        learned parameters.

    Returns
    -------
    params_vector : array
        vector of all parameters concatenated.
    """
    count = 0
    for key in params_dict.keys():
        new_vector = np.reshape(params_dict[key], (-1, 1))
        if count == 0:
            theta_vector = new_vector
        else:
            theta_vector = np.concatenate((theta_vector, new_vector))
        count += 1

    return theta_vector


def vector_to_dictionary(vector, layers_dims):
    """
    Unroll parameters vector to dictionary using layers dimensions.

    Arguments
    ---------
    vector : array
        parameters vector.
    layers_dims : list or array_like
        dimensions of each layer in the network.

    Returns
    -------
    parameters : dict
        dictionary storing all parameters.
    """
    L = len(layers_dims)
    parameters = {}
    k = 0

    for l in range(1, L):
        # Create temp variable to store dimension used on each layer
        w_dim = layers_dims[l] * layers_dims[l - 1]
        b_dim = layers_dims[l]

        # Create temp var to be used in slicing parameters vector
        temp_dim = k + w_dim

        # add parameters to the dictionary
        parameters["W" + str(l)] = vector[
            k:temp_dim].reshape(layers_dims[l], layers_dims[l - 1])
        parameters["b" + str(l)] = vector[
            temp_dim:temp_dim + b_dim].reshape(b_dim, 1)

        k += w_dim + b_dim

    return parameters


def gradients_to_vector(gradients):
    """
    Roll all gradients into a single vector containing only dW and db.

    Arguments
    ---------
    gradients : dict
        storing gradients of weights and biases for all layers: dA, dW, db.

    Returns
    -------
    new_grads : array
        vector of only dW and db gradients.
    """
    # Get the number of indices for the gradients to iterate over
    valid_grads = [key for key in gradients.keys()
                   if not key.startswith("dA")]
    L = len(valid_grads)// 2
    count = 0
    
    # Iterate over all gradients and append them to new_grads list
    for l in range(1, L + 1):
        if count == 0:
            new_grads = gradients["dW" + str(l)].reshape(-1, 1)
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        else:
            new_grads = np.concatenate(
                (new_grads, gradients["dW" + str(l)].reshape(-1, 1)))
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        count += 1
        
    return new_grads


# Finally, we'll write the gradient checking function that will compute the difference between the analytical and numerical gradients and tell us if our implementation of back-propagation is correct. We'll randomly choose 1 example to compute the difference.
# 

def forward_prop_cost(X, parameters, Y, hidden_layers_activation_fn="tanh"):
    """
    Implements the forward propagation and computes the cost.
    
    Arguments
    ---------
    X : 2d-array
        input data, shape: number of features x number of examples.
    parameters : dict
        parameters to use in forward prop.
    Y : array
        true "label", shape: 1 x number of examples.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".

    Returns
    -------
    cost : float
        cross-entropy cost.
    """
    # Compute forward prop
    AL, _ = L_model_forward(X, parameters, hidden_layers_activation_fn)

    # Compute cost
    cost = compute_cost(AL, Y)

    return cost


def gradient_check(
        parameters, gradients, X, Y, layers_dims, epsilon=1e-7,
        hidden_layers_activation_fn="tanh"):
    """
    Checks if back_prop computes correctly the gradient of the cost output by
    forward_prop.
    
    Arguments
    ---------
    parameters : dict
        storing all parameters to use in forward prop.
    gradients : dict
        gradients of weights and biases for all layers: dA, dW, db.
    X : 2d-array
        input data, shape: number of features x number of examples.
    Y : array
        true "label", shape: 1 x number of examples.
    epsilon : 
        tiny shift to the input to compute approximate gradient.
    layers_dims : list or array_like
        dimensions of each layer in the network.
    
    Returns
    -------
    difference : float
        difference between approx gradient and back_prop gradient
    """
    
    # Roll out parameters and gradients dictionaries
    parameters_vector = dictionary_to_vector(parameters)
    gradients_vector = gradients_to_vector(gradients)

    # Create vector of zeros to be used with epsilon
    grads_approx = np.zeros_like(parameters_vector)

    for i in range(len(parameters_vector)):
        # Compute cost of theta + epsilon
        theta_plus = np.copy(parameters_vector)
        theta_plus[i] = theta_plus[i] + epsilon
        j_plus = forward_prop_cost(
            X, vector_to_dictionary(theta_plus, layers_dims), Y,
            hidden_layers_activation_fn)

        # Compute cost of theta - epsilon
        theta_minus = np.copy(parameters_vector)
        theta_minus[i] = theta_minus[i] - epsilon
        j_minus = forward_prop_cost(
            X, vector_to_dictionary(theta_minus, layers_dims), Y,
            hidden_layers_activation_fn)

        # Compute numerical gradients
        grads_approx[i] = (j_plus - j_minus) / (2 * epsilon)

    # Compute the difference of numerical and analytical gradients
    numerator = norm(gradients_vector - grads_approx)
    denominator = norm(grads_approx) + norm(gradients_vector)
    difference = numerator / denominator

    if difference > 10e-7:
        print ("\033[31mThere is a mistake in back-propagation " +               "implementation. The difference is: {}".format(difference))
    else:
        print ("\033[32mThere implementation of back-propagation is fine! "+               "The difference is: {}".format(difference))

    return difference


# Set up neural network architecture
layers_dims = [X_train.shape[0], 5, 5, 1]

# Initialize parameters
parameters = initialize_parameters(layers_dims)

# Randomly selecting 1 example from training data
perms = np.random.permutation(X_train.shape[1])
index = perms[:1]

# Compute forward propagation
AL, caches = L_model_forward(X_train[:, index], parameters, "tanh")

# Compute analytical gradients
gradients = L_model_backward(AL, y_train[:, index], caches, "tanh")

# Compute difference of numerical and analytical gradients
difference = gradient_check(parameters, gradients, X_train[:, index], y_train[:, index], layers_dims)


# Congratulations! Our implementation is correct :)
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Conclusion
# </h2>
# 

# Below are some key takeaways:
# - Two-sided numerical gradient approximates the analytical gradients more closely than right-side form.
# - Since gradient checking is very slow:
#     - Apply it on one or few training examples.
#     - Turn it off when training neural network after making sure that backpropagation's implementation is correct.
# - Gradient checking doesn't work when applying drop-out method. Use keep-prob = 1 to check gradient checking and then change it when training neural network.
# - Epsilon = 10e-7 is a common value used for the difference between analytical gradient and numerical gradient. If the difference is less than 10e-7 then the implementation of backpropagation is correct.
# - Thanks to *Deep Learning* frameworks such as Tensorflow and Pytorch, we may find ourselves rarely implement backpropagation because such frameworks compute that for us; however, it's a good practice to understand what happens under the hood to become a good Deep Learning practitioner.
# 

# <h1 style="font-family: Georgia; font-size:3em;color:#2462C0; font-style:bold">
# Coding Neural Network - Forward Propagation and Backpropagation
# </h1><br>
# 

# **Why Neural Networks?**
# 
# According to *Universal Approximate Theorem*, Neural Networks can approximate as well as learn and represent any function given a large enough layer and desired error margin. The way neural network learns the true function is by building complex representations on top of simple ones. On each hidden layer, the neural network learns new feature space by first compute the affine (linear) transformations of the given inputs and then apply non-linear function which in turn will be the input of the next layer. This process will continue until we reach the output layer. Therefore, we can define neural network as information flows from inputs through hidden layers towards the output. For a 3-layers neural network, the learned function would be: $f(x) = f_3(f_2(f_1(x)))$ where:
# - $f_1(x)$: Function learned on first hidden layer
# - $f_2(x)$: Function learned on second hidden layer
# - $f_3(x)$: Function learned on output layer
# 
# Therefore, on each layer we learn different representation that gets more complicated with later hidden layers.Below is an example of a 3-layers neural network (we don't count input layer):
# <p align="center">
# <img src="images/neural_net.jpg"><br>
# <caption><center><u><b><font color="purple">Figure 1:</font></b></u> Neural Network with two hidden layers</center></caption>
# </p>
# 

# For example, computers can't understand images directly and don't know what to do with pixels data. However, a neural network can build a simple representation of the image in the early hidden layers that identifies edges. Given the first hidden layer output, it can learn corners and contours. Given the second hidden layer, it can learn parts such as nose. Finally, it can learn the object identity.
# 
# Since **truth is never linear** and representation is very critical to the performance of a machine learning algorithm, neural network can help us build very complex models and leave it to the algorithm to learn such representations without worrying about feature engineering that takes practitioners very long time and effort to curate a good representation. 
# 
# The notebook has two parts:
# 1. [Coding the neural network](#Coding the NN): This entails writing all the helper functions that would allow us to implement a multi-layer neural network. While doing so, I'll explain the theoretical parts whenever possible and give some advices on implementations.
# 2. [Application](#Application): We'll implement the neural network we coded in the first part on image recognition problem to see if the network we built will be able to detect if the image has a cat or a dog and see it working :)
# 

# Import packages
import os as os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context("notebook")
plt.style.use("fivethirtyeight")


# <a id="Coding the NN"></a>
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# I. Coding The Neural Network
# </h2>
# 

# <h3 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Forward Propagation
# </h3>
# 

# The input $X$ provides the initial information that then propagates to the hidden units at each layer and finally produce the output $\widehat{Y}$. The architecture of the network entails determining its depth, width, and activation functions used on each layer. **Depth** is the number of hidden layers. **Width** is the number of units (nodes) on each hidden layer since we don't control neither input layer nor output layer dimensions. There are quite a few set of activation functions such *Rectified Linear Unit, Sigmoid, Hyperbolic tangent, etc*. Research has proven that deeper networks outperform networks with more hidden units. Therefore, it's always better and won't hurt to train a deeper network (with diminishing returns).
# 
# Lets first introduce some notations that will be used throughout the notebook:
# * $W^l$: Weights matrix for the $l^{th}$ layer
# * $b^l$: Bias vector for the $l^{th}$ layer
# * $Z^l$: Linear (affine) transformations of given inputs for the $l^{th}$ layer
# * $g^l$: Activation function applied on the $l^{th}$ layer
# * $A^l$: Post-activation output for the $l^{th}$ layer
# * $dW^l$: Derivative of the cost function w.r.t  $W^l$ ($\frac{\partial J}{\partial W^l}$)
# * $db^l$: Derivative of the cost function w.r.t $b^l$ ($\frac{\partial J}{\partial b^l})$)
# * $dZ^l$: Derivative of the cost function w.r.t $Z^l$ ($\frac{\partial J}{\partial Z^l}$)
# * $dA^l$: Derivative of the cost function w.r.t $A^l$ ($\frac{\partial J}{\partial A^l}$)
# * $n^l$: Number of units (nodes) of the $l^{th}$ layer
# * $m$: Number of examples
# * $L$: Number of layers in the network (not including the input layer)
# 
# Next, we'll write down the dimensions of a multi-layer neural network in the general form to help us in matrix multiplication because one of the major challenges in implementing a neural network is getting the dimensions right.
# * $W^l,\ dW^l$: Number of units (nodes) in $l^{th}$ layer x Number of units (nodes) in $l - 1$ layer
# * $b^l,\ db^l$: Number of units (nodes) in $l^{th}$ layer x 1
# * $Z^l,\ dZ^l$: Number of units (nodes) in $l^{th}$ layer x number of examples
# * $A^l,\ dA^l$: Number of units (nodes) in $l^{th}$ layer x number of examples
# 
# The two equations we need to implement forward propagations are:
# $$Z^l = W^lA^{l - 1} + b ^l\tag1\\{}$$
# $$A^l = g^l(Z^l) = g^l(W^lA^{l - 1} + b ^l)\tag2$$
# These computations will take place on each layer.
# 

# <h3 style="font-family: Georgia; font-size:1.3em;color:purple; font-style:bold">
# Parameters Initialization
# </h3><br>
# We'll first initialize the weight matrices and the bias vectors. It's important to note that we shouldn't initialize all the parameters to zero because doing so will lead the gradients to be equal and on each iteration the output would be the same and the learning algorithm won't learn anything. Therefore, it's important to randomly initialize the parameters to values between 0 and 1. It's also recommended to multiply the random values by small scalar such as 0.01 to make the activation units active and be on the regions where activation functions' derivatives are not close to zero.
# 

# Initialize parameters
def initialize_parameters(layers_dims):
    """
    Initialize parameters dictionary.
    
    Weight matrices will be initialized to random values from uniform normal
    distribution.
    bias vectors will be initialized to zeros.

    Arguments
    ---------
    layers_dims : list or array-like
        dimensions of each layer in the network.

    Returns
    -------
    parameters : dict
        weight matrix and the bias vector for each layer.
    """
    np.random.seed(1)               
    parameters = {}
    L = len(layers_dims)            

    for l in range(1, L):           
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters["W" + str(l)].shape == (
            layers_dims[l], layers_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layers_dims[l], 1)

    return parameters


# <h3 style="font-family: Georgia; font-size:1.3em;color:purple; font-style:bold">
# Activation Functions
# </h3><br>
# There is no definitive guide for which activation function works best on specific problems. It's a trial and error process where one should try different set of functions and see which one works best on the problem at hand. We'll cover 4 of the most commonly used activation functions:
# - **Sigmoid function ($\sigma$)**: $g(z) = \frac{1}{1 + e^{-z}}$. It's recommended to be used only on the output layer so that we can easily interpret the output as probabilities since it has restricted output between 0 and 1. One of the main disadvantages for using sigmoid function on hidden layers is that the gradient is very close to zero over a large portion of its domain which makes it slow and harder for the learning algorithm to learn.
# - **Hyperbolic Tangent function**: $g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. It's superior to sigmoid function in which the mean of its output is very close to zero, which in other words center the output of the activation units around zero and make the range of values very small which means faster to learn. The disadvantage that it shares with sigmoid function is that the gradient is very small on good portion of the domain.
# - **Rectified Linear Unit (ReLU)**: $g(z) = max\{0, z\}$. The models that are close to linear are easy to optimize. Since ReLU shares a lot of the properties of linear functions, it tends to work well on most of the problems. The only issue is that the derivative is not defined at $z = 0$, which we can overcome by assigning the derivative to 0 at $z = 0$. However, this means that for $z\leq 0$ the gradient is zero and again can't learn.
# - **Leaky Rectified Linear Unit**: $g(z) = max\{\alpha*z, z\}$. It overcomes the zero gradient issue from ReLU and assigns $\alpha$ which is a small value for $z\leq 0$.
# 
# If you're not sure which activation function to choose, start with ReLU.
# 
# Next, we'll implement the above activation functions and draw a graph for each one to make it easier to see the domain and range of each function.
# 

# Define activation functions that will be used in forward propagation
def sigmoid(Z):
    """
    Computes the sigmoid of Z element-wise.

    Arguments
    ---------
    Z : array
        output of affine transformation.

    Returns
    -------
    A : array
        post activation output.
    Z : array
        output of affine transformation.
    """
    A = 1 / (1 + np.exp(-Z))

    return A, Z


def tanh(Z):
    """
    Computes the Hyperbolic Tagent of Z elemnet-wise.

    Arguments
    ---------
    Z : array
        output of affine transformation.

    Returns
    -------
    A : array
        post activation output.
    Z : array
        output of affine transformation.
    """
    A = np.tanh(Z)

    return A, Z


def relu(Z):
    """
    Computes the Rectified Linear Unit (ReLU) element-wise.

    Arguments
    ---------
    Z : array
        output of affine transformation.

    Returns
    -------
    A : array
        post activation output.
    Z : array
        output of affine transformation.
    """
    A = np.maximum(0, Z)

    return A, Z


def leaky_relu(Z):
    """
    Computes Leaky Rectified Linear Unit element-wise.

    Arguments
    ---------
    Z : array
        output of affine transformation.

    Returns
    -------
    A : array
        post activation output.
    Z : array
        output of affine transformation.
    """
    A = np.maximum(0.1 * Z, Z)

    return A, Z


# Plot the 4 activation functions
z = np.linspace(-10, 10, 100)

# Computes post-activation outputs
A_sigmoid, z = sigmoid(z)
A_tanh, z = tanh(z)
A_relu, z = relu(z)
A_leaky_relu, z = leaky_relu(z)

# Plot sigmoid
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(z, A_sigmoid, label = "Function")
plt.plot(z, A_sigmoid * (1 - A_sigmoid), label = "Derivative")
plt.legend(loc = "upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{1}{1 + e^{-z}}$")
plt.title("Sigmoid Function", fontsize = 16)
# Plot tanh
plt.subplot(2, 2, 2)
plt.plot(z, A_tanh, 'b', label = "Function")
plt.plot(z, 1 - np.square(A_tanh), 'r',label = "Derivative")
plt.legend(loc = "upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{e^z - e^{-z}}{e^z + e^{-z}}$")
plt.title("Hyperbolic Tangent Function", fontsize = 16)
# plot relu
plt.subplot(2, 2, 3)
plt.plot(z, A_relu, 'g')
plt.xlabel("z")
plt.ylabel(r"$max\{0, z\}$")
plt.title("ReLU Function", fontsize = 16)
# plot leaky relu
plt.subplot(2, 2, 4)
plt.plot(z, A_leaky_relu, 'y')
plt.xlabel("z")
plt.ylabel(r"$max\{0.1z, z\}$")
plt.title("Leaky ReLU Function", fontsize = 16)
plt.tight_layout();


# <h3 style="font-family: Georgia; font-size:1.3em;color:purple; font-style:bold">
# Feed Forward
# </h3><br>
# Given its inputs from previous layer, each unit computes affine transformation $z = W^Tx + b$ and then apply an activation function $g(z)$ such as ReLU element-wise. During the process, we'll store (cache) all variables computed and used on each layer to be used in back-propagation. We'll write first two helper functions that will be used in the L-model forward propagation to make it easier to debug. Keep in mind that on each layer, we may have different activation function.
# 

# Define helper functions that will be used in L-model forward prop
def linear_forward(A_prev, W, b):
    """
    Computes affine transformation of the input.

    Arguments
    ---------
    A_prev : 2d-array
        activations output from previous layer.
    W : 2d-array
        weight matrix, shape: size of current layer x size of previuos layer.
    b : 2d-array
        bias vector, shape: size of current layer x 1.

    Returns
    -------
    Z : 2d-array
        affine transformation output.
    cache : tuple
        stores A_prev, W, b to be used in backpropagation.
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):
    """
    Computes post-activation output using non-linear activation function.

    Arguments
    ---------
    A_prev : 2d-array
        activations output from previous layer.
    W : 2d-array
        weight matrix, shape: size of current layer x size of previuos layer.
    b : 2d-array
        bias vector, shape: size of current layer x 1.
    activation_fn : str
        non-linear activation function to be used: "sigmoid", "tanh", "relu".

    Returns
    -------
    A : 2d-array
        output of the activation function.
    cache : tuple
        stores linear_cache and activation_cache. ((A_prev, W, b), Z) to be used in backpropagation.
    """
    assert activation_fn == "sigmoid" or activation_fn == "tanh" or         activation_fn == "relu"

    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, hidden_layers_activation_fn="relu"):
    """
    Computes the output layer through looping over all units in topological
    order.

    Arguments
    ---------
    X : 2d-array
        input matrix of shape input_size x training_examples.
    parameters : dict
        contains all the weight matrices and bias vectors for all layers.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".

    Returns
    -------
    AL : 2d-array
        probability vector of shape 1 x training_examples.
    caches : list
        that contains L tuples where each layer has: A_prev, W, b, Z.
    """
    A = X                           
    caches = []                     
    L = len(parameters) // 2        

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
            activation_fn=hidden_layers_activation_fn)
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)],
        activation_fn="sigmoid")
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches


# <h3 style="font-family: Georgia; font-size:1.3em;color:purple; font-style:bold">
# Cost
# </h3><br>
# We'll use the binary **Cross-Entropy** cost. It uses the log-likelihood method to estimate its error. The cost is:
# $$J(W, b) = -\frac{1}{m}\sum_{i = 1}^m\big(y^ilog(\widehat{y^i}) + (1 - y^i)log(1 - \widehat{y^i})\big)\tag3$$
# The above cost function is convex; however, neural network usually stuck on a local minimum and is not guaranteed to find the optimal parameters. We'll use here gradient-based learning.
# 

# Compute cross-entropy cost
def compute_cost(AL, y):
    """
    Computes the binary Cross-Entropy cost.

    Arguments
    ---------
    AL : 2d-array
        probability vector of shape 1 x training_examples.
    y : 2d-array
        true "label" vector.

    Returns
    -------
    cost : float
        binary cross-entropy cost.
    """
    m = y.shape[1]              
    cost = - (1 / m) * np.sum(
        np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))

    return cost


# <h2 style="font-family: Georgia; font-size:1.5em;color:purple; font-style:bold">
# Back-Propagation
# </h2>
# 

# Backpropagation allows the information to go back from the cost backward through the network in order to compute the gradient. Therefore, loop over the nodes starting at the final node in reverse topological order to compute the derivative of the final node output with respect to each edge's node tail. Doing so will help us know who is responsible for the most error and change the parameters in that direction. The following derivatives' formulas will help us write the back-propagate functions:
# $$dA^L = \frac{A^L - Y}{A^L(1 - A^L)}\tag4\\{}$$
# $$dZ^L = A^L - Y\tag5\\{}$$
# $$dW^l = \frac{1}{m}dZ^l{A^{l - 1}}^T\tag6\\{}$$
# $$db^l = \frac{1}{m}\sum_i(dZ^l)\tag7\\{}$$
# $$dA^{l - 1} = {W^l}^TdZ^l\tag8\\{}$$
# $$dZ^{l} = dA^l*g^{'l}(Z^l)\tag9\\{}$$
# Since $b^l$ is always a vector, the sum would be across rows (since each column is an example).
# 

# Define derivative of activation functions w.r.t z that will be used in back-propagation
def sigmoid_gradient(dA, Z):
    """
    Computes the gradient of sigmoid output w.r.t input Z.

    Arguments
    ---------
    dA : 2d-array
        post-activation gradient, of any shape.
    Z : 2d-array
        input used for the activation fn on this layer.

    Returns
    -------
    dZ : 2d-array
        gradient of the cost with respect to Z.
    """
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ


def tanh_gradient(dA, Z):
    """
    Computes the gradient of hyperbolic tangent output w.r.t input Z.

    Arguments
    ---------
    dA : 2d-array
        post-activation gradient, of any shape.
    Z : 2d-array
        input used for the activation fn on this layer.

    Returns
    -------
    dZ : 2d-array
        gradient of the cost with respect to Z.
    """
    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_gradient(dA, Z):
    """
    Computes the gradient of ReLU output w.r.t input Z.

    Arguments
    ---------
    dA : 2d-array
        post-activation gradient, of any shape.
    Z : 2d-array
        input used for the activation fn on this layer.

    Returns
    -------
    dZ : 2d-array
        gradient of the cost with respect to Z.
    """
    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ


# define helper functions that will be used in L-model back-prop
def linear_backword(dZ, cache):
    """
    Computes the gradient of the output w.r.t weight, bias, and post-activation
    output of (l - 1) layers at layer l.

    Arguments
    ---------
    dZ : 2d-array
        gradient of the cost w.r.t. the linear output (of current layer l).
    cache : tuple
        values of (A_prev, W, b) coming from the forward propagation in the current layer.

    Returns
    -------
    dA_prev : 2d-array
        gradient of the cost w.r.t. the activation (of the previous layer l-1).
    dW : 2d-array
        gradient of the cost w.r.t. W (current layer l).
    db : 2d-array
        gradient of the cost w.r.t. b (current layer l).
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation_fn):
    """
    Arguments
    ---------
    dA : 2d-array
        post-activation gradient for current layer l.
    cache : tuple
        values of (linear_cache, activation_cache).
    activation : str
        activation used in this layer: "sigmoid", "tanh", or "relu".

    Returns
    -------
    dA_prev : 2d-array
        gradient of the cost w.r.t. the activation (of the previous layer l-1), same shape as A_prev.
    dW : 2d-array
        gradient of the cost w.r.t. W (current layer l), same shape as W.
    db : 2d-array
        gradient of the cost w.r.t. b (current layer l), same shape as b.
    """
    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, y, caches, hidden_layers_activation_fn="relu"):
    """
    Computes the gradient of output layer w.r.t weights, biases, etc. starting
    on the output layer in reverse topological order.

    Arguments
    ---------
    AL : 2d-array
        probability vector, output of the forward propagation (L_model_forward()).
    y : 2d-array
        true "label" vector (containing 0 if non-cat, 1 if cat).
    caches : list
        list of caches for all layers.
    hidden_layers_activation_fn :
        activation function used on hidden layers: "tanh", "relu".

    Returns
    -------
    grads : dict
        with the gradients.
    """
    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward(
            dAL, caches[L - 1], "sigmoid")

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
            "db" + str(l)] = linear_activation_backward(
                grads["dA" + str(l)], current_cache,
                hidden_layers_activation_fn)

    return grads


# define the function to update both weight matrices and bias vectors
def update_parameters(parameters, grads, learning_rate):
    """
    Update the parameters' values using gradient descent rule.

    Arguments
    ---------
    parameters : dict
        contains all the weight matrices and bias vectors for all layers.
    grads : dict
        stores all gradients (output of L_model_backward).

    Returns
    -------
    parameters : dict
        updated parameters.
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


# <a id="Application"></a>
# 

# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# II. Application
# </h2>
# 

# The dataset that we'll be working on has 209 images. Each image is 64 x 64 pixels on RGB scale. We'll build a neural network to classify if the image has a cat or not. Therefore, $y^i \in \{0, 1\}.$
# - We'll first load the images.
# - Show sample image for a cat.
# - Reshape input matrix so that each column would be one example. Also, since each image is 64 x 64 x 3, we'll end up having 12,288 features for each image. Therefore, the input matrix would be 12,288 x 209.
# - Standardize the data so that the gradients don't go out of control. Also, it will help hidden units have similar range of values. For now, we'll divide every pixel by 255 which shouldn't be an issue. However, it's better to standardize the data to have a mean of 0 and a standard deviation of 1.
# 

# Import training dataset
train_dataset = h5py.File("../data/train_catvnoncat.h5")
X_train = np.array(train_dataset["train_set_x"])
y_train = np.array(train_dataset["train_set_y"])

test_dataset = h5py.File("../data/test_catvnoncat.h5")
X_test = np.array(test_dataset["test_set_x"])
y_test = np.array(test_dataset["test_set_y"])

# print the shape of input data and label vector
print(f"""Original dimensions:\n{20 * '-'}\nTraining: {X_train.shape}, {y_train.shape}
Test: {X_test.shape}, {y_test.shape}""")

# plot cat image
plt.figure(figsize=(6, 6))
plt.imshow(X_train[50])
plt.axis("off");

# Transform input data and label vector
X_train = X_train.reshape(209, -1).T
y_train = y_train.reshape(-1, 209)

X_test = X_test.reshape(50, -1).T
y_test = y_test.reshape(-1, 50)

# standarize the data
X_train = X_train / 255
X_test = X_test / 255

print(f"""\nNew dimensions:\n{15 * '-'}\nTraining: {X_train.shape}, {y_train.shape}
Test: {X_test.shape}, {y_test.shape}""")


# Now, our dataset is ready to be used and test our neural network implementation. Let's first write **multi-layer model** function to implement gradient-based learning using predefined number of iterations and learning rate.
# 

# Define the multi-layer model using all the helper functions we wrote before


def L_layer_model(
        X, y, layers_dims, learning_rate=0.01, num_iterations=3000,
        print_cost=True, hidden_layers_activation_fn="relu"):
    """
    Implements multilayer neural network using gradient descent as the
    learning algorithm.

    Arguments
    ---------
    X : 2d-array
        data, shape: number of examples x num_px * num_px * 3.
    y : 2d-array
        true "label" vector, shape: 1 x number of examples.
    layers_dims : list
        input size and size of each layer, length: number of layers + 1.
    learning_rate : float
        learning rate of the gradient descent update rule.
    num_iterations : int
        number of iterations of the optimization loop.
    print_cost : bool
        if True, it prints the cost every 100 steps.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".

    Returns
    -------
    parameters : dict
        parameters learnt by the model. They can then be used to predict test examples.
    """
    np.random.seed(1)

    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []

    # iterate over num_iterations
    for i in range(num_iterations):
        # iterate over L-layers to get the final output and the cache
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = compute_cost(AL, y)

        # iterate over L-layers backward to get gradients
        grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # append each 100th cost to the cost list
        if (i + 1) % 100 == 0 and print_cost:
            print(f"The cost after {i + 1} iterations is: {cost:.4f}")

        if i % 100 == 0:
            cost_list.append(cost)

    # plot the cost curve
    plt.figure(figsize=(10, 6))
    plt.plot(cost_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Loss")
    plt.title(f"Loss curve for the learning rate = {learning_rate}")

    return parameters


def accuracy(X, parameters, y, activation_fn="relu"):
    """
    Computes the average accuracy rate.

    Arguments
    ---------
    X : 2d-array
        data, shape: number of examples x num_px * num_px * 3.
    parameters : dict
        learnt parameters.
    y : 2d-array
        true "label" vector, shape: 1 x number of examples.
    activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".

    Returns
    -------
    accuracy : float
        accuracy rate after applying parameters on the input data
    """
    probs, caches = L_model_forward(X, parameters, activation_fn)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100

    return f"The accuracy rate is: {accuracy:.2f}%."


# Next, we'll train two versions of the neural network where each one will use different activation function on hidden layers: One will use rectified linear unit (**ReLU**) and the second one will use hyperbolic tangent function (**tanh**). Finally we'll use the parameters we get from both neural networks to classify test examples and compute the test accuracy rates for each version to see which activation function works best on this problem.
# 

# Setting layers dims
layers_dims = [X_train.shape[0], 5, 5, 1]

# NN with tanh activation fn
parameters_tanh = L_layer_model(
    X_train, y_train, layers_dims, learning_rate=0.03, num_iterations=3000,
    hidden_layers_activation_fn="tanh")

# Print the accuracy
accuracy(X_test, parameters_tanh, y_test, activation_fn="tanh")


# NN with relu activation fn
parameters_relu = L_layer_model(
    X_train, y_train, layers_dims, learning_rate=0.03, num_iterations=3000,
    hidden_layers_activation_fn="relu")

# Print the accuracy
accuracy(X_test, parameters_relu, y_test, activation_fn="relu")


# <h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
# Conclusion
# </h2>
# 

# The purpose of this notebook is to code Deep Neural Network step-by-step and explain the important concepts while doing that. We don't really care about the accuracy rate at this moment since there are tons of things we could've done to increase the accuracy which would be the subject of following notebooks. Below are some takeaways:
# - Even if neural network can represent any function, it may fail to learn for two reasons:
#     1. The optimization algorithm may fail to find the best value for the parameters of the desired (true) function.
#         It can stuck in a local optimum.
#     2. The learning algorithm may find different functional form that is different than the intended function due to overfitting.
# - Even if neural network rarely converges and always stuck in a local minimum, it is still able to reduce the cost significantly and come up with very complex models with high test accuracy.
# - The neural network we used in this notebook is standard fully connected network. However, there are two other kinds of networks:
#     - Convolutional NN: Where not all nodes are connected. It's best in class for image recognition.
#     - Recurrent NN: There is a feedback connections where output of the model is fed back into itself. It's used mainly in sequence modeling.
# - The fully connected neural network also forgets what happened in previous steps and also doesn't know anything about the output.
# - There are number of hyperparameters that we can tune using cross validation to get the best performance of our network:
#     1. Learning rate ($\alpha$): Determines how big the step for each update of parameters.
#         - Small $\alpha$ leads to slow convergence and may become computationally very expensive.
#         - Large $\alpha$ may lead to overshooting where our learning algorithm may never converge.
#     2. Number of hidden layers (depth): The more hidden layers the better, but comes at a cost computationally.
#     3. Number of units per hidden layer (width): Research proven that huge number of hidden units per layer doesn't add to the improvement of the network.
#     4. Activation function: Which function to use on hidden layers differs among applications and domains. It's a trial and error process to try different functions and see which one works best.
#     5. Number of iterations.
# - Standardize data would help activation units have similar range of values and avoid gradients to go out of control.
# 

