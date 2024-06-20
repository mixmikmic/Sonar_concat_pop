# Import the modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier  # The decision tree Classifier from Scikit
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


filename = 'winequality-red.csv'  #Download the file from https://archive.ics.uci.edu/ml/datasets/wine+quality
df = pd.read_csv(filename, sep=';')
df.describe()


df['quality'].value_counts()


#categorize wine quality in three levels
bins = (0,3.5,5.5,10)
categories = pd.cut(df['quality'], bins, labels = ['bad','ok','good'])
df['quality'] = categories


df['quality'].value_counts()


# Preprocessing and splitting data to X and y
X = df.drop(['quality'], axis = 1)
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X)
y = df['quality']
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 323)


classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)
#Predicting the Test Set
y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt='2.0f')


print("Accuracy is {}".format(accuracy_score(y_test, y_pred)))


# ### Ensemble Learning in Decision trees
# 
# The code below implements bagging technique a way to implement enseble learning. To know more about ensemble learning refer to the section __Ensemble Learning__ in __Chapter 3__.
# 

from sklearn.ensemble import BaggingClassifier
bag_classifier = BaggingClassifier(
            DecisionTreeClassifier(), n_estimators=500, max_samples=1000,\
            bootstrap=True, n_jobs=-1)

bag_classifier.fit(X_train, y_train)

y_pred = bag_classifier.predict(X_test)


print("Accuracy is {}".format(accuracy_score(y_test, y_pred)))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt='2.0f')


## Random Forest 


