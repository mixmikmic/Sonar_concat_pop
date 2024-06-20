import sklearn
import pandas as pd
import numpy as np
from __future__ import division
import collections
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from sklearn.cross_validation import cross_val_score
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingRegressor
import math
import csv
import scipy
get_ipython().magic('matplotlib inline')
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import urllib


xTrain = np.load('PrecomputedMatrices/xTrain.npy')
yTrain = np.load('PrecomputedMatrices/yTrain.npy')


def normalizeInput(arr):
    for i in range(arr.shape[1]):
        minVal = min(arr[:,i])
        maxVal = max(arr[:,i])
        arr[:,i] =  (arr[:,i] - minVal) / (maxVal - minVal)
    return arr
#xTrain = normalizeInput(xTrain)
#yTrain = normalizeInput(yTrain)


categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
           'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','Location']
df = pd.DataFrame(xTrain, columns=categories)
df['Result'] = pd.Series(yTrain)


df.head()


allWins = df[df['Result'] == 1]


plt.hist(allWins['Wins'], 100)


# Basically, on average the winners of basketball games have the following difference between themselves and the opponenet in the below categories. 
# 

for cat in categories:
    print 'The mean of category', cat, ': ',(allWins[cat]).mean()


for cat in categories:
    print 'The Pearson correlation between the result and', cat, 'is :',scipy.stats.pearsonr(df[cat], df['Result'])


#xTrain = normalizeInput(xTrain)
xTrain = np.load('PrecomputedMatrices/xTrain.npy')


model4 = RandomForestClassifier(n_estimators=200)
model2 = linear_model.BayesianRidge()
model5 = AdaBoostClassifier(n_estimators=100)
model = GradientBoostingRegressor(n_estimators=100)
model3 = KNeighborsClassifier(n_neighbors=101)


def showFeatureImportance(my_categories):
    fx_imp = pd.Series(model.feature_importances_, index=my_categories)
    fx_imp /= fx_imp.max()
    fx_imp.sort()
    fx_imp.plot(kind='barh')


xTrain[0]
categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
           'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','Location']


#accuracy=[]
#for i in range(100):
#    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
#    dtrain = xgb.DMatrix( X_train, label=Y_train)
#    dtest = xgb.DMatrix( X_test, label=Y_test)
#    param = {'bst:max_depth':3, 'bst:eta':1, 'silent':0, 'objective':'binary:logistic', 'gamma':.1, 'eta': .3, 
#             'min_child_weight': 3, 'alpha': .1, 'lamda':.1}
#    evallist  = [(dtest,'eval')]
#    num_round = 200
#    bst = xgb.train( param, dtrain, num_round, evallist,early_stopping_rounds=10 )
#    preds = bst.predict(dtest)
#    preds[preds < .5] = 0
#    preds[preds >= .5] = 1
#    accuracy.append(np.mean(preds == Y_test))
#print "The accuracy is", sum(accuracy)/len(accuracy)


#xgb.plot_importance(bst)


xTrain.shape


categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
           'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','Location']
accuracy=[]
totals=[]
for i in range(1):
    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
    results1 = model.fit(X_train, Y_train)
    preds1 = model.predict(X_test)
    
    results2 = model2.fit(X_train, Y_train)
    preds2 = model2.predict(X_test)
    
    results3 = model3.fit(X_train, Y_train)
    preds3 = model3.predict(X_test)

    results4 = model4.fit(X_train, Y_train)
    preds4 = model4.predict(X_test)
    
    results5 = model5.fit(X_train, Y_train)
    preds5 = model5.predict(X_test)
    
    preds = (preds1 + preds2 + preds3 + preds4 + preds5)/5
    totals.append(preds)
    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    accuracy.append(np.mean(preds == Y_test))
    #accuracy.append(np.mean(predictions == Y_test))
print "The accuracy is", sum(accuracy)/len(accuracy)
showFeatureImportance(categories)


lis = [5,6,3]
for index,item in enumerate(lis):
    print item


categories=['PPG','PPGA','PowerConf','3PG','TOP', 'APG', 'Conference Champ', 'Tourney Conference Champ',
            'Seed','SOS','SRS', 'Rebounds', 'Steals', 'Tourney Appearances','National Championships','Location']
np.random.choice(categories,5, replace=False)


realCategories = ['Wins','PPG','PPGA','PowerConf','3PG', 'APG', 'Conference Champ', 'Tourney Conference Champ',
            'Seed','SOS','SRS', 'Rebounds', 'Steals', 'Tourney Appearances','National Championships','Location']
xTrain = df[categories].as_matrix()
xTrain = xTrain.reshape((113567,4,4,1))
xTrain.shape
X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test_categorical = np_utils.to_categorical(Y_test, 2)


trainingLoss = []
validationLoss = []
img_channels = 1
img_rows = 4
img_cols = 4
num_classes = 2
model = Sequential()
model.add(Convolution2D(32, 2, 2, border_mode='same', input_shape=(img_rows, img_cols, img_channels)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=32, nb_epoch=1000,shuffle=True, validation_data=(X_test, Y_test_categorical))
trainingLoss.append(hist.history['loss'])
validationLoss.append(hist.history['val_loss'])
#preds = model.predict_classes( X_test, batch_size=32, verbose=1)
#print (np.mean(preds == Y_test))


trainingLoss
yRange = range(len(trainingLoss[0]))
yRange = [x+1 for x in yRange]
line1 = plt.plot(yRange, trainingLoss[0], label='Training Loss')
line2 = plt.plot(yRange, validationLoss[0], label='Validation Loss')
plt.legend()
plt.show()


xTrain = df.as_matrix()
xTrain = xTrain[50000:, :]
xTrain.shape


trainDict={}
xTrain = df.as_matrix()
for i in range(17):   
    for p in range(10):
        modifiedCategories = np.random.choice(categories,i, replace=False)
        modifiedCategories = np.append(modifiedCategories, 'Wins')
        modCatList = modifiedCategories.tolist()
        str1 = ''.join(modCatList)
        modifiedxTrain = df[modifiedCategories].as_matrix()
        #model = tree.DecisionTreeClassifier()
        #model = tree.DecisionTreeRegressor()
        #model = linear_model.LogisticRegression()
        #model = linear_model.BayesianRidge()
        #model = linear_model.Lasso()
        #model = svm.SVC()
        #model = svm.SVR()
        #model = linear_model.Ridge(alpha = 0.5)
        #model = AdaBoostClassifier(n_estimators=100)
        #model1 = GradientBoostingClassifier(n_estimators=100)
        model = GradientBoostingRegressor(n_estimators=100)
        #model = RandomForestClassifier(n_estimators=200)
        #model = KNeighborsClassifier(n_neighbors=101)
        accuracy=[]
        for q in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(modifiedxTrain, yTrain)
            #X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
            model.fit(X_train, Y_train)
            results = model.fit(X_train, Y_train)
            preds = model.predict(X_test)
            preds[preds < .5] = 0
            preds[preds >= .5] = 1
            accuracy.append(np.mean(preds == Y_test))
        trainDict[str1] = sum(accuracy)/len(accuracy)
        print 'Iteration',(i*10 + p),'Done'
for key, value in sorted(trainDict.iteritems(), key=lambda (k,v): (v,k), reverse=True):
    print "%s: %s" % (key, value)


categories=['Wins','PPG','PPGA','PowerConf','3PG', 'Conference Champ', 'Tourney Conference Champ',
            'Seed','SOS','SRS', 'Rebounds', 'Steals', 'Tourney Appearances','National Championships','Location']
modifiedxTrain = df[categories].as_matrix()
modifiedxTrain = modifiedxTrain[50000:, :]
yTrain = yTrain[50000:]
model = GradientBoostingRegressor(n_estimators=100)
accuracy=[]
for q in range(1):
    #X_train, X_test, Y_train, Y_test = train_test_split(modifiedxTrain, yTrain)
    X_train, X_test, Y_train, Y_test = train_test_split(modifiedxTrain, yTrain)
    model.fit(X_train, Y_train)
    results = model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    accuracy.append(np.mean(preds == Y_test))
print sum(accuracy)/len(accuracy)





