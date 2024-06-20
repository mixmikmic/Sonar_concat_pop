# # Chapter 5: Supervised learning
# 
# This is the notebook companion to chapter 7. We use data from the Lending Club to develop our understanding of machine learning concepts. The Lending Club is a peer-to-peer lending company. It offers loans which are funded by other people. In this sense, the Lending Club acts as a hub connecting borrowers with investors. The client applies for a loan of a certain amount, and the company assesses the risk of the operation. If the application is accepted, it may or may not be fully covered. We will focus on the prediction of whether the loan will be fully funded, based on the scoring of and information related to the application. The data come from the following URL:
# 
# https://www.lendingclub.com/info/download-data.action
# 
# We will clean and reprocess the partial dataset from 2007-2011. Framing the problem a little bit more, based on the information supplied by the customer asking for a loan, we want to predict whether an accepted loan will be fully funded or not.
# 

import matplotlib.pylab as plt

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('font', size=12) 
plt.rc('figure', figsize = (12, 5))


import pickle
ofname = open('files/ch05/dataset_small.pkl','rb') 
(x,y) = pickle.load(ofname)


# and check the shapes
# 

dims = x.shape[1]
N = x.shape[0]
print 'dims: ' + str(dims)+', samples: '+ str(N)


from sklearn import neighbors
from sklearn import datasets
#Create an instance of K-nearest neighbor classifier
knn = neighbors.KNeighborsClassifier(n_neighbors=11)
#Train the classifier
knn.fit(x,y)
#Compute the prediction according to the model
yhat = knn.predict(x)
print 'Predicted value: ' + str(yhat[-1]), ', real target: ' + str(y[-1])


# And check its score/accuracy.
# 

knn.score(x,y)


# The distribution of the labels to predict is shown in the following pie chart.
# 

#%matplotlib inline
#import matplotlib.pyplot as plt
import numpy as np
plt.pie(np.c_[np.sum(np.where(y==1,1,0)),np.sum(np.where(y==-1,1,0))][0],
        labels=['Not fully funded','Full amount'],
        colors=['g','r'],
        shadow=False,
        autopct ='%.2f' )
plt.gcf().set_size_inches((6,6))
plt.savefig("pie.png",dpi=300, bbox_inches='tight')


# Let us compute the elements of a confusion matrix.
# 

yhat = knn.predict(x)
TP = np.sum(np.logical_and(yhat==-1,y==-1))
TN = np.sum(np.logical_and(yhat==1,y==1))
FP = np.sum(np.logical_and(yhat==-1,y==1))
FN = np.sum(np.logical_and(yhat==1,y==-1))
print 'TP: '+ str(TP), ', FP: '+ str(FP)
print 'FN: '+ str(FN), ', TN: '+ str(TN)


from sklearn import metrics
metrics.confusion_matrix(yhat,y)


#Train a classifier using .fit()
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)
yhat=knn.predict(x)

print "classification accuracy:", metrics.accuracy_score(yhat, y)
print "confusion matrix: \n" + str(metrics.confusion_matrix(yhat, y))


# Up to this point we used training data for "assessing" the performance of the method, as we will see later this is a bad practice. Let us simulate the exploitation stage by holding out a subset of the training data and assess the performance on that set. 
# 

# Simulate a real case: Randomize and split data in two subsets PRC*100% for training and 
# the rest (1-PRC)*100% for testing
import numpy as np
perm = np.random.permutation(y.size)
PRC = 0.7
split_point = int(np.ceil(y.shape[0]*PRC))

X_train = x[perm[:split_point].ravel(),:]
y_train = y[perm[:split_point].ravel()]

X_test = x[perm[split_point:].ravel(),:]
y_test = y[perm[split_point:].ravel()]

print 'Training shape: ' + str(X_train.shape), ' , training targets shape: '+str(y_train.shape)
print 'Testing shape: ' + str(X_test.shape), ' , testing targets shape: '+str(y_test.shape)


#Train a classifier on training data
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#Check on the training set and visualize performance
yhat=knn.predict(X_train)

from sklearn import metrics
print "\nTRAINING STATS:"
print "classification accuracy:", metrics.accuracy_score(yhat, y_train)
print "confusion matrix: \n"+ str(metrics.confusion_matrix(y_train, yhat))


#Check on the test set
yhat=knn.predict(X_test)
print "TESTING STATS:"
print "classification accuracy:", metrics.accuracy_score(yhat, y_test)
print "confusion matrix: \n"+ str(metrics.confusion_matrix(yhat,y_test))


# We can automatize this process with the tools provided in sklearn.
# 

#The splitting can be done using the tools provided by sklearn:
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import metrics

PRC = 0.3
acc = np.zeros((10,))
for i in xrange(10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=PRC)
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    yhat = knn.predict(X_test)
    acc[i] = metrics.accuracy_score(yhat, y_test)
acc.shape=(1,10)
print "Mean expected error: "+str(np.mean(acc[0]))


# We can use the validation process for model selection.
# 

#The splitting can be done using the tools provided by sklearn:
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import tree
from sklearn import svm
from sklearn import metrics

PRC = 0.1
acc_r=np.zeros((10,4))
for i in xrange(10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=PRC)
    nn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
    nn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    svc = svm.SVC()
    dt = tree.DecisionTreeClassifier()
    
    nn1.fit(X_train,y_train)
    nn3.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    dt.fit(X_train,y_train)
    
    yhat_nn1=nn1.predict(X_test)
    yhat_nn3=nn3.predict(X_test)
    yhat_svc=svc.predict(X_test)
    yhat_dt=dt.predict(X_test)
    
    acc_r[i][0] = metrics.accuracy_score(yhat_nn1, y_test)
    acc_r[i][1] = metrics.accuracy_score(yhat_nn3, y_test)
    acc_r[i][2] = metrics.accuracy_score(yhat_svc, y_test)
    acc_r[i][3] = metrics.accuracy_score(yhat_dt, y_test)


plt.boxplot(acc_r);
for i in xrange(4):
    xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)
    
ax = plt.gca()
ax.set_xticklabels(['1-NN','3-NN','SVM','Decission Tree'])
plt.ylabel('Accuracy')
plt.savefig("error_ms_1.png",dpi=300, bbox_inches='tight')


# ## Learning curves
# 
# Let us try to understand the behavior of machine learning algorithms when the amount of data and the "complexity" of the method change. Let us start first by varying the amount of data for a fixed complexity.
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
MAXN=700

fig = plt.figure()
fig.set_size_inches(6,5)

plt.plot(1.25*np.random.randn(MAXN,1),1.25*np.random.randn(MAXN,1),'r.',alpha = 0.3)
fig.hold('on')
plt.plot(8+1.5*np.random.randn(MAXN,2),5+1.5*np.random.randn(MAXN,2),'r.', alpha = 0.3)
plt.plot(5+1.5*np.random.randn(MAXN,1),5+1.5*np.random.randn(MAXN,1),'b.',alpha = 0.3)
plt.savefig("toy_problem.png",dpi=300, bbox_inches='tight')


import numpy as np
from sklearn import metrics
from sklearn import tree

C=5
MAXN=1000

yhat_test=np.zeros((10,299,2))
yhat_train=np.zeros((10,299,2))
#Repeat ten times to get smooth curves
for i in xrange(10):
    X = np.concatenate([1.25*np.random.randn(MAXN,2),5+1.5*np.random.randn(MAXN,2)]) 
    X = np.concatenate([X,[8,5]+1.5*np.random.randn(MAXN,2)])
    y = np.concatenate([np.ones((MAXN,1)),-np.ones((MAXN,1))])
    y = np.concatenate([y,np.ones((MAXN,1))])
    perm = np.random.permutation(y.size)
    X = X[perm,:]
    y = y[perm]

    X_test = np.concatenate([1.25*np.random.randn(MAXN,2),5+1.5*np.random.randn(MAXN,2)]) 
    X_test = np.concatenate([X_test,[8,5]+1.5*np.random.randn(MAXN,2)])
    y_test = np.concatenate([np.ones((MAXN,1)),-np.ones((MAXN,1))])
    y_test = np.concatenate([y_test,np.ones((MAXN,1))])
    j=0
    for N in xrange(10,3000,10):
        Xr=X[:N,:]
        yr=y[:N]
        #Evaluate the model
        clf = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=C)
        clf.fit(Xr,yr.ravel())
        yhat_test[i,j,0] = 1. - metrics.accuracy_score(clf.predict(X_test), y_test.ravel())
        yhat_train[i,j,0] = 1. - metrics.accuracy_score(clf.predict(Xr), yr.ravel())
        j=j+1

p1,=plt.plot(np.mean(yhat_test[:,:,0].T,axis=1),'pink')
p2,=plt.plot(np.mean(yhat_train[:,:,0].T,axis=1),'c')
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.xlabel('Number of samples x10')
plt.ylabel('Error rate')
plt.legend([p1,p2],["Test C = 5","Train C = 5"])
plt.savefig("learning_curve_1.png",dpi=300, bbox_inches='tight')


# Let us repeat the process with a simpler model.
# 

C=1
MAXN=1000

#Repeat ten times to get smooth curves
for i in xrange(10):
    X = np.concatenate([1.25*np.random.randn(MAXN,2),5+1.5*np.random.randn(MAXN,2)]) 
    X = np.concatenate([X,[8,5]+1.5*np.random.randn(MAXN,2)])
    y = np.concatenate([np.ones((MAXN,1)),-np.ones((MAXN,1))])
    y = np.concatenate([y,np.ones((MAXN,1))])
    perm = np.random.permutation(y.size)
    X = X[perm,:]
    y = y[perm]

    X_test = np.concatenate([1.25*np.random.randn(MAXN,2),5+1.5*np.random.randn(MAXN,2)]) 
    X_test = np.concatenate([X_test,[8,5]+1.5*np.random.randn(MAXN,2)])
    y_test = np.concatenate([np.ones((MAXN,1)),-np.ones((MAXN,1))])
    y_test = np.concatenate([y_test,np.ones((MAXN,1))])
    j=0
    for N in xrange(10,3000,10):
        Xr=X[:N,:]
        yr=y[:N]
        #Evaluate the model
        clf = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=C)
        clf.fit(Xr,yr.ravel())
        yhat_test[i,j,1] = 1. - metrics.accuracy_score(clf.predict(X_test), y_test.ravel())
        yhat_train[i,j,1] = 1. - metrics.accuracy_score(clf.predict(Xr), yr.ravel())
        j=j+1

p3,=plt.plot(np.mean(yhat_test[:,:,1].T,axis=1),'r')
p4,=plt.plot(np.mean(yhat_train[:,:,1].T,axis=1),'b')
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.xlabel('Number of samples x10')
plt.ylabel('Error rate')
plt.legend([p3,p4],["Test C = 1","Train C = 1"])
plt.savefig("learning_curve_2.png",dpi=300, bbox_inches='tight')


# and join both to see the differences.
# 

p1,=plt.plot(np.mean(yhat_test[:,:,0].T,axis=1),color='pink')
p2,=plt.plot(np.mean(yhat_train[:,:,0].T,axis=1),'c')
p3,=plt.plot(np.mean(yhat_test[:,:,1].T,axis=1),'r')
p4,=plt.plot(np.mean(yhat_train[:,:,1].T,axis=1),'b')
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.xlabel('Number of samples x10')
plt.ylabel('Error rate')
plt.legend([p1,p2,p3,p4],["Test C = 5","Train C = 5","Test C = 1","Train C = 1"])
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.savefig("learning_curve_3.png",dpi=300, bbox_inches='tight')


# Let us check now what happens when we fix the amount of data and change the complexity of the technique.
# 

get_ipython().magic('reset -f')
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from IPython.html.widgets import interact
from sklearn import metrics
from sklearn import tree

MAXC=20
N=1000
NTEST=4000
ITERS=3

yhat_test=np.zeros((ITERS,MAXC,2))
yhat_train=np.zeros((ITERS,MAXC,2))
#Repeat ten times to get smooth curves
for i in xrange(ITERS):
    X = np.concatenate([1.25*np.random.randn(N,2),5+1.5*np.random.randn(N,2)]) 
    X = np.concatenate([X,[8,5]+1.5*np.random.randn(N,2)])
    y = np.concatenate([np.ones((N,1)),-np.ones((N,1))])
    y = np.concatenate([y,np.ones((N,1))])
    perm = np.random.permutation(y.size)
    X = X[perm,:]
    y = y[perm]

    X_test = np.concatenate([1.25*np.random.randn(NTEST,2),5+1.5*np.random.randn(NTEST,2)]) 
    X_test = np.concatenate([X_test,[8,5]+1.5*np.random.randn(NTEST,2)])
    y_test = np.concatenate([np.ones((NTEST,1)),-np.ones((NTEST,1))])
    y_test = np.concatenate([y_test,np.ones((NTEST,1))])

    j=0
    for C in xrange(1,MAXC+1):
        #Evaluate the model
        clf = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=C)
        clf.fit(X,y.ravel())
        yhat_test[i,j,0] = 1. - metrics.accuracy_score(clf.predict(X_test), y_test.ravel())
        yhat_train[i,j,0] = 1. - metrics.accuracy_score(clf.predict(X), y.ravel())
        j=j+1

p1, = plt.plot(np.mean(yhat_test[:,:,0].T,axis=1),'r')
p2, = plt.plot(np.mean(yhat_train[:,:,0].T,axis=1),'b')
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.xlabel('Complexity')
plt.ylabel('Error rate')
plt.legend([p1, p2], ["Testing error", "Training error"])
plt.savefig("learning_curve_4.png",dpi=300, bbox_inches='tight')


# We may use all these concepts to understand and select the complexity of a model.
# 

get_ipython().magic('reset -f')
get_ipython().magic('matplotlib inline')
import pickle
ofname = open('files/ch05/dataset_small.pkl','rb') 
(X,y) = pickle.load(ofname)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn import cross_validation

#

#Create a 10-fold cross validation set
kf=cross_validation.KFold(n=y.shape[0], n_folds=10, shuffle=True, random_state=0)
      
#Search the parameter among the following
C=np.arange(2,20,)

acc = np.zeros((10,18))
i=0
for train_index, val_index in kf:
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    j=0
    for c in C:
        dt = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=c)
        dt.fit(X_train,y_train)
        yhat = dt.predict(X_val)
        acc[i][j] = metrics.accuracy_score(yhat, y_val)
        j=j+1
    i=i+1
    
plt.boxplot(acc);
for i in xrange(18):
    xderiv = (i+1)*np.ones(acc[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv,acc[:,i],'ro',alpha=0.3)

print 'Mean accuracy: ' + str(np.mean(acc,axis = 0))
print 'Selected model index: ' + str(np.argmax(np.mean(acc,axis = 0)))
print 'Complexity: ' + str(C[np.argmax(np.mean(acc,axis = 0))])
plt.ylim((0.7,1.))
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.xlabel('Complexity')
plt.ylabel('Accuracy')
plt.savefig("model_selection.png",dpi=300, bbox_inches='tight')


get_ipython().magic('reset -f')
get_ipython().magic('matplotlib inline')
import pickle
ofname = open('files/ch05/dataset_small.pkl','rb') 
(X,y) = pickle.load(ofname)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn import cross_validation

#Train_test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.20, random_state=42)

#Create a 10-fold cross validation set
kf=cross_validation.KFold(n=y_train.shape[0], n_folds=10, shuffle=True, random_state=0)     
#Search the parameter among the following
C=np.arange(2,20,)
acc = np.zeros((10,18))
i=0
for train_index, val_index in kf:
    X_t, X_val = X_train[train_index], X_train[val_index]
    y_t, y_val = y_train[train_index], y_train[val_index]
    j=0
    for c in C:
        dt = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=c)
        dt.fit(X_t,y_t)
        yhat = dt.predict(X_val)
        acc[i][j] = metrics.accuracy_score(yhat, y_val)
        j=j+1
    i=i+1

print 'Mean accuracy: ' + str(np.mean(acc,axis = 0))
print 'Selected model index: ' + str(np.argmax(np.mean(acc,axis = 0)))
print 'Complexity: ' + str(C[np.argmax(np.mean(acc,axis = 0))])


#Train the model with the complete training set with the selected complexity
dt = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=C[np.argmax(np.mean(acc,axis = 0))])
dt.fit(X_train,y_train)
#Test the model with the test set 
yhat = dt.predict(X_test)
print 'Test accuracy: ' + str(metrics.accuracy_score(yhat, y_test))

#Train the model for handling to the client
dt = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=C[np.argmax(np.mean(acc,axis = 0))])
dt.fit(X,y)

plt.boxplot(acc);
for i in xrange(18):
    xderiv = (i+1)*np.ones(acc[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
    plt.plot(xderiv,acc[:,i],'ro',alpha=0.3)


plt.ylim((0.7,1.))
fig = plt.gcf()
fig.set_size_inches(12,5)


# WARNING: The following cell takes a long time to execute.
# 

get_ipython().magic('reset -f')

import pickle
ofname = open('files/ch05/dataset_small.pkl','rb') 
(X,y) = pickle.load(ofname)

import numpy as np
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

parameters = {'C':[1e4,1e5,1e6],'gamma':[1e-5,1e-4,1e-3]}

N_folds = 5

kf=cross_validation.KFold(n=y.shape[0], n_folds=N_folds,  shuffle=True, random_state=0)

acc = np.zeros((N_folds,))
i=0
#We will build the predicted y from the partial predictions on the test of each of the folds
yhat = y.copy()
for train_index, test_index in kf:
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = svm.SVC(kernel='rbf')
    clf = grid_search.GridSearchCV(clf, parameters, cv = 3) #This line does a cross-validation on the 
    clf.fit(X_train,y_train.ravel())
    X_test = scaler.transform(X_test)
    yhat[test_index] = clf.predict(X_test)
    
print metrics.accuracy_score(yhat, y)
print metrics.confusion_matrix(yhat, y)


# WARNING: The following cell takes a long time to execute.
# 

import numpy as np
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics

dvals = [{1:0.25},{1:0.5},{1:1},{1:2},{1:4},{1:8},{1:16}]
opoint = []

for cw in dvals:

    parameters = {'C':[1e4,1e5,1e6],'gamma':[1e-5,1e-4,1e-3],'class_weight':[cw]}
  
    print parameters

    N_folds = 5

    kf=cross_validation.KFold(n=y.shape[0], n_folds=N_folds,  shuffle=True, random_state=0)

    acc = np.zeros((N_folds,))
    mat = np.zeros((2,2,N_folds))
    i=0
    yhat = y.copy()
    for train_index, test_index in kf:
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        clf = svm.SVC(kernel='rbf')
        clf = grid_search.GridSearchCV(clf, parameters, cv = 3)
        clf.fit(X_train,y_train.ravel())
        X_test = scaler.transform(X_test)
        yhat[test_index] = clf.predict(X_test)
        acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
        mat[:,:,i] = metrics.confusion_matrix(yhat[test_index], y_test)
        print str(clf.best_params_)
        i=i+1
    print 'Mean accuracy: '+ str(np.mean(acc))
    opoint.append((np.mean(acc),np.sum(mat,axis=2)))


get_ipython().magic('reset -f')

import pickle
ofname = open('files/ch05/dataset_small.pkl','rb') 
(X,y) = pickle.load(ofname)


from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn import metrics
import numpy as np

dvals = [{1:0.25},{1:0.5},{1:1},{1:2},{1:4},{1:8},{1:16}]

kf=cross_validation.KFold(n=y.shape[0], n_folds=5, shuffle=True, random_state=0)

acc = np.zeros((5,))
i=0
yhat = y.copy()
for cw in dvals:
    i=0
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = ensemble.RandomForestClassifier(n_estimators=51, class_weight=cw)
        dt.fit(X_train,y_train)
        yhat[test_index]=dt.predict(X_test)
        acc[i] = metrics.accuracy_score(yhat[test_index], y_test)
        i=i+1
    #Run here the code from the next cell for checking each of the performance plots


#You may run this code for each iteration in the former cell to get the surface pltos.
#The prediction of a configuration is given in yhat

M=metrics.confusion_matrix(yhat, y)

from matplotlib import cm
ccampaing = [10,20,30,40,50,60,70,80,90]
retention = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
TP=M[0,0]
FN=M[1,0]
FP=M[0,1]
TN=M[1,1]
campaing=TN+FN
profit=TP+FN

[xx,yy]=np.meshgrid(ccampaing,retention)
cost = 100*profit-xx*campaing + yy*TN*100

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111, projection='3d')
cost_no_campaign = 100*profit+0*xx
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

Z = np.where(cost>cost_no_campaign,cost,cost_no_campaign)
ax.plot_surface(xx,yy,Z,cmap=cm.coolwarm,alpha=0.3,linewidth=0.1,rstride=1,cstride=1)
ax.plot_wireframe(xx, yy, Z, rstride=1, cstride=1, color=[0.5,0.5,0.5],alpha=0.5)
fig.set_size_inches((12,8))
ax.set_xlabel('campaign cost',size=16)
ax.set_ylabel('retention rate',size=16)
ax.set_zlabel('profit',size=16)
fig.savefig('rf_cost.png',dpi=100,format='PNG')

print 'Max profit: ' + str(100*(np.max(Z)-np.min(Z))/np.min(Z))
print 'Max profit for retention rate: '  + str(100*(np.max(Z[5])-np.min(Z))/np.min(Z))
print 'Campaign cost:'
print 'Accuracy: ' + str((TP+TN)/(TP+FN+FP+FN*1.))
print 'Confusion: ' + str(M)





# Download data from "https://www.lendingclub.com/info/download-data.action" years 2007-2011.
# 

get_ipython().magic('reset -f')
from __future__ import division
import pandas as pd
import numpy as np

filename = '' #Write the filename of the original data set
df = pd.read_csv(filename, low_memory=False,skiprows=1)
col_names = df.columns.tolist()
print col_names
print 'Number of attributes: ' + str(len(col_names))


# Case Study: we want to predict based on the information filled by the customer asking for a loan if it will be granted or not up to a certain threshold $thr$. We are using data from the Lending Club. Not all information is relevant to our interest, thus we detail in the following the fields we will base our prediction on:
# 
# + annual_inc	The annual income provided by the borrower during registration.
# + delinq_2yrs	The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
# + dti	A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
# + earliest_cr_line	The month the borrower's earliest reported credit line was opened
# + emp_length	Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 
# + home_ownership	The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
# + installment	The monthly payment owed by the borrower if the loan originates.
# + int_rate	Interest Rate on the loan
# + is_inc_v	Indicates if income was verified by LC, not verified, or if the income source was verified
# + last_fico_range_high	The last upper boundary of range the borrower’s FICO belongs to pulled.
# + last_fico_range_low	The last lower boundary of range the borrower’s FICO belongs to pulled.
# + fico_range_high	The upper boundary of range the borrower’s FICO belongs to.
# + fico_range_low	The lower boundary of range the borrower’s FICO belongs to.
# + mths_since_last_delinq	The number of months since the borrower's last delinquency.
# + mths_since_last_major_derog	Months since most recent 90-day or worse rating
# + open_acc	The number of open credit lines in the borrower's credit file.
# + term	The number of payments on the loan. Values are in months and can be either 36 or 60.
# + total_acc	The total number of credit lines currently in the borrower's credit file
# + loan_amnt	The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
# 
# We could work different scenarios:
# 
# + Because we are given two datasets of rejected vs accepted we could work out the acceptance criterions
# + On the other hand we could try to predict successful accepted loans. A loan applicaiton is successful if the funded amount (funded_amnt) or the funded amount by investors (funded_amnt_inv) is close to the loan amount (loan_amnt) requested. In this sense we could put a threshold in which the acceptance is based on 
# $$\frac{loan - funded}{loan}\geq 0.9$$
# 
# Let us focus for simplicity on this second case. Considering the accepted loans which ones are successful and can we derive some rules for success?
# 

# Let us drop the non-useful columns
# 

drop_cols = ['id', 'member_id', 'grade', 'sub_grade','earliest_cr_line', 'emp_title', 'verification_status', 'issue_d', 'loan_status', 'pymnt_plan', 'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state', 'inq_last_6mths', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code']


df = df.drop(drop_cols,axis=1)


col_names = df.columns.tolist()
print col_names


get_ipython().magic('matplotlib inline')
loan = df['loan_amnt'].values
funded = df['funded_amnt_inv'].values
targets = np.abs(loan-funded)/loan

df['targets'] = targets
wrk_records = np.where(~np.isnan(targets))
y = targets[wrk_records]>=0.05

import matplotlib.pyplot as plt
plt.hist(targets[wrk_records],bins=30)

print 'Larger deviation: ' + str(np.sum(y))
print 'Total: ' + str(np.sum(1-y))


# We stick to the accepted loans and try to predict if it will get the full amount by investors.
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.pie(np.c_[len(y)-np.sum(y),np.sum(y)][0],labels=['Full amount','Not fully funded'],colors=['r','g'],shadow=True,autopct ='%.2f' )
fig = plt.gcf()
fig.set_size_inches(6,6)


# Check the features
# 

df.head()


# Observe the different issues in term, int_rate, emp_length, home_ownership. 
# 
# Interest rate is a string, so we may convert it to float by removing the percentage character and converting to float point values. With respect to term and emp_length we could use several strategies: we can vectorize the different results. But note that there is an order relationship. In this particular case categorical values can be directly translated to numbers that represent that order. Finally, house_ownership will be vectorized into as many features as values in the categorical variable.
# 

def clear_percent (row):
    try:
        d = float(row['int_rate'][:-1])/100.
    except:
        d = None
    return d

df['int_rate_clean'] = df.apply (lambda row: clear_percent(row),axis=1)
    


print 'Values of the variable term: ' + str(np.unique(df['term']))


def clear_term (row):
    try:
        if row['term']==' 36 months':
            d = 1
        else:
            if row['term']==' 60 months':
                d = 2
            else:
                if np.isnan(row['term']):
                    d = None
                else:
                    print 'WRONG'
                    print row['term']
    except:
        print 'EXCEPT'
        d = None
    return d

df['term_clean'] = df.apply (lambda row: clear_term(row),axis=1)
    


print 'Values for employment length: ' + str(np.unique(df['emp_length']))


#We use dictionary mapping as a switch 
def clean_emp_length(argument):
    switcher = {
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
        '< 1 year': 0,
        'n/a':None,
    }
    try:
        d = switcher[argument['emp_length']]    
    except:
        d = None
    return d

df['emp_length_clean'] = df.apply (lambda row: clean_emp_length(row),axis=1)


df.head()


np.unique(df['home_ownership'])


from sklearn.feature_extraction import DictVectorizer

comb_dict = df[['home_ownership']].to_dict(orient='records')
vec = DictVectorizer()
home = 2*vec.fit_transform(comb_dict).toarray()-1
home[:5]


df_vector = pd.DataFrame(home[:,1:])
vector_columns = vec.get_feature_names()
df_vector.columns = vector_columns[1:]
df_vector.index = df.index
df_vector.head()


#Join data
df = df.join(df_vector)
df.head()


#Drop processed columns
df = df.drop(['term','int_rate','emp_length','home_ownership'],axis=1)
df.head()


#Drop the funded ammount
df=df.drop(['funded_amnt_inv'],axis=1)


#Declare targets
y = df['targets'].values>0.05
print 'Undefined values:' + str(np.sum(np.where(np.isnan(y),1,0)))
x=df.drop(['targets'],axis=1).values
idx_rmv = np.where(np.isnan(y))[0]
y = np.delete(y,idx_rmv)
x = np.delete(x,idx_rmv,axis=0)
print y.shape,x.shape


#Check what is going on in the data NaN
nan_feats=np.sum(np.where(np.isnan(x),1,0),axis=0)
plt.bar(np.arange(len(nan_feats)),nan_feats)
fig = plt.gcf()
fig.set_size_inches((12,5))
nan_feats


#Drop feature 6, too much NaN
print col_names[6]
x=np.hstack((x[:,:6],x[:,7:]))


x.shape
#Check now
nan_feats=np.sum(np.where(np.isnan(x),1,0),axis=0)
plt.bar(np.arange(len(nan_feats)),nan_feats)
fig = plt.gcf()
fig.set_size_inches((12,5))
nan_feats


#Check records
nan_records=np.sum(np.where(np.isnan(x),1,0),axis=1)
np.histogram(nan_records)


# We have all information from almost all members. Let us check the 1112 members and see if are from the minority class. 
# 

print len(nan_records),len(y)
idx_rmv = np.where(nan_records>0)[0]
y = np.delete(y,idx_rmv)
x = np.delete(x,idx_rmv,axis=0)
print y.shape,x.shape





# # Chapter 11: Parallelization
# 
# This notebook shows IPthon parallel capabilities. We will see the commands that allow
# us to parallelize tasks, see section Introduction. If properly combined, they can lead to 
# interting parallel algorithms, see section A complete example. 
# 
# 
# ## Introduction
# 
# In order to start IPython's parallel capabilities, the simplest way of proceeding 
# it to click on the Clusters tab of the notebook dashboard, and press 
# Start with the desired number of cores. 
# This will automatically run the necessary commands to start the
# IPython cluster. We will now be able to send different tasks to
# the engines using the web interface.
# 
# The next commands allows you to connect to the cluster
# 

from IPython import parallel
engines = parallel.Client()
engines.block = True
print engines.ids


# These commands connect to the cluster and output the number of engines in it. 
# If an error is shown when running the
# commands, the cluster has not been correctly created.
# 
# ### Direct view of engines
# 
# The following commands,
# executed on the notebook (i.e., the IPython interpreter), send commands
# to the first engine:
# 

engines[0].execute('a = 2') 
engines[0].execute('b = 10') 
engines[0].execute('c = a + b')  
engines[0].pull('c')


# Observe that we do not have direct access to the command line of the first engine. Rather, we send commands to it through the client. The result is retrieved by means of the `pull` command.
# 
# Since each engine is
# an independent process, the operating system may schedule each engine
# in a different core and thus execution may be performed in parallel. Take a look at the following example.
# 

engines[0].execute('a = 2') 
engines[0].execute('b = 10') 
engines[1].execute('a = 9') 
engines[1].execute('b = 7') 
engines[0:2].execute('c = a + b')   
engines[0:2].pull('c')


# Observe that the operation `c = a + b` is executed on each egine. The latter calculation and does not show the power of parallization. For that purpose we are next perform more computational intenstive tasks. Let us now show that we are really doing computations in parallel. Let us try with something bigger!
# 
# In order to simplify the code, let us define the following variable that references the first two engines (even if there are more active engines).
# 

dview2 = engines[0:2]    


# We are next going to focus on matrix multiplication, for instance. We begin by doing serialized computations on the notebook and compute the total processing time.
# 

import time
import numpy as np

# Create four 1000x1000 matrix
A0 = np.random.rand(1000,1000)
B0 = np.random.rand(1000,1000)
A1 = np.random.rand(1000,1000)
B1 = np.random.rand(1000,1000)

t0 = time.time() 

C0 = np.dot(A0, B0)
C1 = np.dot(A1, B1)
    
print "Time in seconds (Computations): ", time.time() - t0 


# And now we will do computations in parallel.
# 

dview2.execute('import numpy as np')       # We import numpy on both engines!

t0 = time.time()
engines[0].push(dict(A=A0, B=B0))    # We send A0 and B0 to engine 0 
engines[1].push(dict(A=A1, B=B1))    # We send A1 and B1 to engine 1 

t0_computations = time.time()

dview2.execute('C = np.dot(A,B)')
    
print "Computations: ", time.time() - t0_computations

[C0, C1] = dview2.pull('C')
print "Time in seconds: ", time.time() - t0


# The total computing time should decrease thanks to the divison of the computation in two tasks. Each task is then manually executed in two different engines and each engine is scheduled by the operating system on two different processors (if the computer has at least two processors).
# 
# The previous commands show us how to execute commands on engines as if we were typing 
# them directly on the command-line. Indeed, we
# have manually sent, executed and retrieved the results of computations.
# This procedure may be useful in some cases but in many cases there
# will be no need to do so. The `map` function may be used to that purpose
# 

def mul(A, B):
    import numpy as np
    C = np.dot(A, B)
    return C

[C0, C1] = dview2.map(mul,[A0, A1],[B0, B1])


# These commands, executed on the client, perform a remote call.
# The function `mul` is defined locally. There is no need to use the `push`
# and `pull` functions explicitly to send and retrieve the results; it is done
# implicitly. Note the `import numpy as np`
# inside the `mul` function. This is a common model, to ensure that the appropriate
# toolboxes are imported to where the task is run. 
# 
# The `map` call splits the tasks between the engines associated
# with `dview2`. In the previous example, the task `mul(A0,B0)`
# is executed on one engine and `mul(A1, B1)` is executed on
# the one. Which command is executed on each engine? What happens if
# the list of arguments to map includes three or more matrices? We may
# see this with the following example:
# 

engines[0].execute('my_id = "engineA"') 
engines[1].execute('my_id = "engineB"')

def sleep_and_return_id(sec):     
    import time     
    time.sleep(sec)                      
    return my_id,sec

dview2.map(sleep_and_return_id, [3,3,3,1,1,1])


# Execute the previous code and observe the returned result that indicates us which engine executed the function. You may repeat this experment as many times as you wish, but the result will always be the same. The tasks are distributed in a uniform way among the
# engines before executing them no matter which is the delay we pass
# as argument to the function `sleep_and_return_id`. This is in fact a characteristic of the direct view interface: the tasks are distributed among the engines before executing them. 
# 
# This a good way to proceed if you expect each task to take
# the same amount of time. But if not, as is the case in the previous
# example, computation time is wasted and so we recommend to use the 
# load-balanced view instead.
# 
# ### Load-balanced view of engines
# 
# This interface is simpler and more powerful than the direct interface. We would like to point out, however, that with this interface the user has no direct access to individual engines. It is the IPython scheduler that assignes work to each engine. To create a load-balanced view we may use the following command: 
# 

engines.block = True
lview2 = engines.load_balanced_view(targets=[0,1])


# We use the blocking mode since it simplifies the code. The `lview2` is a variable
# that references the first two engines.
# 
# Our example here will be centered on the `sleep_and_return_id`
# function we have seen in the previous subsection:
# 

lview2.map(sleep_and_return_id, [3,3,3,1,1,1])


# Observe that rather than using the direct
# view interface (`dview2` variable) of the `map` function, we use the associated load-balanced view interface (`lview2` variable).
# 
# In this case, the tasks are assigned to the engines in a dynamic way.
# The `map` function of the load-balanced view begins by assigning
# one task to each engine in the order given by the parameters of the
# `map` function. By default, the load-balanced view scheduler
# then assigns a new task to an engine when it becomes free
# 

# ## A complete example: the New York taxi trips database
# 
# We next present a real application of the parallel capabilities
# of IPython and the discussion of several approaches to it. The dataset
# is a database of taxi trips in New York and it has been obtained through
# a Freedom of Information Law (FOIL) request from the New York City
# Taxi & Limousine Commission (NYCT&L) by University of Illinois at
# Urbana-Champaign (http://publish.illinois.edu/dbwork/open-data/).
# The dataset consists in $12\times2$GBytes Comma Separated Files (CSV)
# files. Each file has approximately $14$ million entries (lines) and
# is already cleaned. Thus no special preprocessing is needed to be
# able to process it. For our purposes we are interested only in the
# following information from each entry: 
# 
# + `pickup_datetime`: start time of the trip, mm-dd-yyyy hh24:mm:ss
# EDT. 
# + `pickup_longitude` and `pickup_latitude`: GPS coordinates
# at the start of the trip. 
# 
# Our objective is to perform an analysis of this data in order to answer
# the following questions: for each district, how many pickups are performed
# during week days and how many during weekends? And how many pickups
# are performed in the morning? For that issue the New York city is
# arbitrarily divided into nine districts: ChinaTown, WTC, Soho, Harlem,
# UpperTown, MidTown, DownTown, UpperEastSide, UpperWestSide and Financial. 
# 
# Implementing the previous classification is rather simple since it
# only requires checking, for each entry, the GPS coordinates of the
# start of the trip and the pickup datetime. Performing this task in
# a sequential may take a rather large amount of time since the number
# of entries, for a single CSV file, is rather large. In addition, special
# care has to be taken when reading the file since a 2GByte file may
# not fully fit into the computer's memory. 
# 
# We may take advantage of the parallelization capabilities in order
# to reduce the processing time. The idea is to divide the input data
# into chunks so that each engine takes care of classifying the entries
# of their corresponding chunks. We propose here an approach which 
# is based on implementing a producer-consumer paradigm
# in order to distribute the tasks. The producer, associated to the
# client, reads the chunks from disc and distributes them among the
# engines using a round robin technique. No explicit `map` function
# is used in this case. Rather, we simulate the behavior of the `map`
# function in order to have fine control of the parallel problem. Recall
# that each engine is an independent process. Since we assign different
# tasks to each engine, the operating system will try to execute each
# engine on a different process.
# 
# For further deails, please see corresponding chapter in the book.
# 
# ## The source code
# 
# We begin by initializing the engines:
# 

get_ipython().magic('reset -f')

from IPython import parallel
from itertools import islice
from itertools import cycle
from collections import Counter
import sys
import time

#Connect to the Ipython cluster    
engines = parallel.Client()

#Create a DirectView to all engines
dview = engines.direct_view()

print "The number of engines in the cluster is: " + str(len(engines.ids))


# We next declare the functions that will be executed on the engines. We do this thanks to the `%%px` parallel magic command.
# 

get_ipython().run_cell_magic('px', '', '\n# The %%px magic executes the code of this cell on each engine.\n\nfrom datetime import datetime\nfrom collections import Counter\n\nimport pandas as pd\nimport numpy as np\n\n# A Counter object to store engine\'s local result\nlocal_total = Counter();\n\ndef dist(p0, p1):\n    "Returns the distance**2 between two points"\n    # We compute the squared distance. Since we only want to compare\n    # distances there is no need to compute the square root (sqrt) \n    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2\n\n# Coordinates (latitude, longitude) of diferent points of the island\ndistrict_dict = { \n    \'Financial\': [40.724863, -73.994718], \n    \'Midtown\': [40.755905, -73.984997],\n    \'Chinatown\': [40.716224, -73.995925],\n    \'WTC\': [40.711724, -74.012888],\n    \'Harlem\': [40.810469, -73.943318],\n    \'Uppertown\': [40.826381, -73.943964],\n    \'Soho\': [40.723783, -74.001237],\n    \'UpperEastSide\': [40.773861, -73.956329],\n    \'UpperWestSide\': [40.787347, -73.975267]\n    }\n\n# Computes the distance to each district center and obtains the one that\n# gives minimum distance\ndef get_district(coors):\n    "Given a coordinate inn latitude and longitude, returns the district in Manhatan"   \n    #If dist^2 is bigger than 0.0005, the district is \'None\'.\n    dist_min = 0.0005\n    district = None\n    for key in district_dict.iterkeys():\n        d = dist(coors, district_dict[key])\n        if dist_min > d:\n            dist_min = d\n            district = key\n    return district\n\ndef is_morning(d):\n    "Given a datetime, returns if it was on morning or not"\n    h = datetime.strptime(d, "%Y-%m-%d %H:%M:%S").hour\n    return 0 <= h and h < 12\n\ndef is_weekend(d):\n    "Given a datetime, returns if it was on weekend or not"\n    wday = datetime.strptime(d, "%Y-%m-%d %H:%M:%S").weekday() #strptime transforms str to date\n    return 4 < wday <= 6\n\n#Function that classifies a single data\ndef classify(x):\n    "Given a tuple with a datetime, latitude and longitude, returns the group where it fits"\n    date, lat, lon = x\n    latitude = float(lat)\n    longitude = float(lon)\n    return is_weekend(date), is_morning(date), get_district([latitude, longitude])\n\n# Function that given a dictionary (data), applies classify function on each element\n# and returns an histogram in a Counter object\ndef process(b):\n    #Recives a block (list of strings) and updates result in global var local_total()\n    global local_total\n    \n    #Create an empty df. Preallocate the space we need by providing the index (number of rows)\n    df = pd.DataFrame(index=np.arange(0,len(b)), columns=(\'datetime\',\'latitude\',\'longitude\'))\n    \n    # Data is a list of lines, containing datetime at col 5 and latitude at row 11.\n    # Allocate in the dataFrame the datetime and latitude and longitude dor each line in data\n    count = 0\n    for line in b:\n        elements = line.split(",")\n        df.loc[count] = elements[5], elements[11], elements[10]\n        count += 1\n        \n    #Delete NaN values from de DF\n    df.dropna(thresh=(len(df.columns) - 1), axis=0)\n    \n    #Apply classify function to the dataFrame\n    cdf = df.apply(classify, axis=1)\n    \n    #Increment the global variable local_total\n    local_total += Counter(cdf.value_counts().to_dict())\n\n# Initialization function\ndef init():\n    #Reset total var\n    global local_total\n    local_total = Counter()')


# Next we show the code executed by the client. The next code performs the next task
# 
# + It reads a chunk of `lines_per_block` lines form the file. The chunk is assigned to an engine which performs the classification. The result of the classification is updated on a local variable on each engine. This process is repeated until all chunks have been processed by the engines.
# + Once finished, the client retrieves the local variable of each engine and computes the overall result.
# 
# This is the principle of the **MapReduce** programming model: a MapReduce program is composed of a Map() procedure that performs filtering and sorting (such as counting the number of times each word appears in a file) and a Reduce() procedure that performs a summary operation (that is, taking each of the results and computing the overall result).
# 

# This is the main code executed on the client
t0 = time.time() 

#File to be processed
filename = 'trip_data.csv'

def get_chunk(f,N):
    """ Returns blocks of nl lines from the file descriptor fd"""
    #Deletes first line on first chunk (header line)
    first = 1
    while True:
        new_chunk = list(islice(f, first, N))
        if not new_chunk:
            break
        first = 0
        yield new_chunk

# A simple counter to verify execution
chunk_n = 0

# Number of lines to be sent to each engine at a time. Use carefully!
lines_per_block = 20

# Create an emty list of async tasks. One element for each engine
async_tasks = [None] * len(engines.ids)

# Cycle Object to get an infinite iterator over the list of engines
c_engines = cycle(engines.ids)

# Initialize each engine. Observe that the execute is performed
# in a non-blocking fashion.
for i in engines.ids:
    async_tasks[i] = engines[i].execute('init()', block=False)

# The variable to store results
global_result = Counter()

# Open the file in ReadOnly mode
try:
    f = open(filename, 'r') #iterable
except IOError:
    sys.exit("Could not open input file!")

# Used to show the progress
print "Beginning to send chunks"
sys.stdout.flush()

# While the generator returns new chunk, sent them to the engines
for new_chunk in get_chunk(f,lines_per_block):
    
    #After the first loop, first_chunk is False. 
    first_chunk = False
    
    #Decide the engine to be used to classify the new chunk
    run_engine = c_engines.next()
    
    # Wait until the engine is ready
    while ( not async_tasks[run_engine].ready() ):
        time.sleep(1)
    
    #Send data to the assigned engine.
    mydict = dict(data = new_chunk)
    
    # The data is sent to the engine in blocking mode. The push function does not return
    # until the engine has received the data. 
    engines[run_engine].push(mydict,block=True)

    # We execute the classification task on the engine. Observe that the task is executed
    # in non-blocking mode. Thus the execute function reurns immediately. 
    async_tasks[run_engine] = engines[run_engine].execute('process(data)', block=False)
    
    # Increase the counter    
    chunk_n += 1

    # Update the progress
    if chunk_n % 1000 == 0:
        print "Chunks sent until this moment: " + str(chunk_n)
        sys.stdout.flush()

print "All chunks have been sent"
sys.stdout.flush()
# Get the results from each engine and accumulate in global_result
for engine in engines.ids:
    # Be sure that all async tasks are finished
    while ( not async_tasks[engine].ready() ):
        time.sleep(1)
    global_result += engines[engine].pull('local_total', block=True)

#Close the file
f.close()

print "Total number of chunks processed: " + str(chunk_n)
print "---------------------------------------------"
print "Agregated dictionary"
print "---------------------------------------------"
print dict(global_result)

print "Time in seconds: ", time.time() - t0
sys.stdout.flush()


# The results of the experiments performed with this code can be seen in the corresponding chapter of the book.
# 

# # Chapter 8: Network Analysis
# 

# Network data are currently generated and collected to an increasing extent from different fields.
# In this notebook we will introduce the basiscs of network analysis. 
# We show how network data analysis allows us to gain insight into the data that would be hard to acquire by other means.
# We work with concepts such as connected components, centrality measures and ego-networks, as well as community detection.
# We use a Python toolbox (NetworkX) to build graphs easily and analyze them.
# We deal with real problems dealing with a Facebook network and answering a set of questions. 
# 

import numpy as np
import networkx as nx


import matplotlib.pylab as plt

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex = True)
plt.rc('font', family = 'times')
plt.rc('xtick', labelsize = 10) 
plt.rc('ytick', labelsize = 10) 
plt.rc('font', size = 12) 
plt.rc('figure', figsize = (12, 5))


# ## Begining with NetworkX
# 

G = nx.Graph()

# Add three edges
G.add_edge('A', 'B');
G.add_edge('A', 'C');
G.add_edge('B', 'D');
G.add_edge('B', 'E');
G.add_edge('D', 'E');

# Draw the graph
nx.draw_networkx(G, node_size = 2000, font_size = 20)

plt.axis('off');
plt.savefig("files/ch08/graph_example.png", dpi = 300, bbox_inches = 'tight')


# To create a directed graph we use DiGraph:
G = nx.DiGraph()
G.add_edge('A', 'B');
G.add_edge('A', 'C');
G.add_edge('B', 'D');
G.add_edge('B', 'E');
G.add_edge('D', 'E');
nx.draw_networkx(G, node_size = 1000, font_size = 20)
plt.axis('off');


# This example shows an undirected graph with 5 nodes and 5 edges. 
# The degree of the node C is 1, the nodes A, E and F is 2 and node B is 3. 
# In this example, paths (C, A, B, E), (C, A, B, F, E) are the paths between nodes C and E. 
# This graph is unweighted, so the shortest path between C and E is the one crossing less number of edges, which is (C, A, B, E). 
# In the example the graph has only one connected component.
# 

# We can create another graph, called star graph:
# 

# Create a star graph:
G = nx.Graph()
G.add_edge('A', 'C');
G.add_edge('B', 'C');
G.add_edge('D', 'C');
G.add_edge('E', 'C');
G.add_edge('F', 'C');
G.add_edge('G', 'C');
G.add_edge('H', 'C');

nx.draw_networkx(G, node_size = 2000, font_size = 20)
plt.axis('off')
plt.savefig("files/ch08/star_graph.png", dpi = 300, bbox_inches = 'tight')


# ## Data description of Facebook Network
# 
# We will use data from [SNAP](https://snap.stanford.edu/data/) collection: *Social circles: Facebook* [EGO_FACEBOOK](https://snap.stanford.edu/data/egonets-Facebook.html).
# The Facbook dataset consists of a network representing friendship between users of Facebook.
# Facebook data was collected from survey participants using a Facebook app.
# Facebook data was anonymized by replacing the Facebook internal identifiers for each user with a new value.
# 

# Let's load the Facebook network into NetworkX.
# The network consists of an undirected and unweighted network that contains friendships between users of Facebook.
# The facebook dataset is defined by an edge list, so the file is a plain text file with one edge per line. The file is uncompressed and ready to load as follows.
# 

fb = nx.read_edgelist("files/ch08/facebook_combined.txt")


# First, we can extract information from the graph without visualizing it. Basic graph properties include number of nodes, of edges and average degree.
# 

fb_n, fb_k = fb.order(), fb.size()
fb_avg_deg = fb_k / fb_n
print 'Nodes: ', fb_n
print 'Edges: ', fb_k
print 'Average degree: ', fb_avg_deg


# We can also compute the **degree distribution** of the graph and plot it.
# 

degrees = fb.degree().values()
degree_hist = plt.hist(degrees, 100)
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Degree distribution')
plt.savefig("files/ch08/degree_hist_plt.png", dpi = 300, bbox_inches = 'tight')


# Networks with power-law distributions are called *scale-free networks*, because power laws have the same functional form at all scales.
# 

# Next, let us find out if the Facebook dataset contains one or more than one connected components.
# 

print '# connected components of Facebook network: ', nx.number_connected_components(fb)


# Let us prune the graph removing node '0' (arbitrarily selected) and compute the number of connected components of the pruned version of the graph:
# 

fb_prunned = nx.read_edgelist("files/ch08/facebook_combined.txt")
fb_prunned.remove_node('0')
print 'Remaining nodes:', fb_prunned.number_of_nodes()
print 'New # connected components:', nx.number_connected_components(fb_prunned)


# Let us see the sizes of the 19 connected components:
# 

fb_components = nx.connected_components(fb_prunned)
print 'Sizes of the connected components', [len(c) for c in fb_components]


# ## Centrality
# 

# **Centrality** of a node **measures its relative importance** within the graph. And there are many ways of calculating centrality, each one with a slightly different meaning.
# Four of these measures are: 
# - The degree centrality
# - The betweenness centrality
# - The closeness centrality
# - The eigenvector centrality
# 

# Let's compute the centrality of the nodes in the star graph, G, defined above.
# 

# Centrality measures for the star graph:
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
print 'Degree centrality: ', sorted(degree.items(), key = lambda x: x[1], reverse = True)
print 'Betweenness centrality: ', sorted(betweenness.items(), key = lambda x: x[1], reverse = True)


# As it can be seen, even if the order of nodes following the two centrality measures is the same, the measure values are different.
# 

# Let us compute the degree centrality of facebook graph.
# 

degree_cent_fb = nx.degree_centrality(fb)
# Once we are calculated degree centrality, we sort the results to see which nodes are more central.
print 'Facebook degree centrality: ', sorted(degree_cent_fb.items(), key = lambda x: x[1], reverse = True)[:10]


# Now, let us compute the degree centrality histogram.
# 

fig = plt.figure(figsize = (6,5))

degree_hist = plt.hist(list(degree_cent_fb.values()), 100)
plt.xlabel('Degree centrality')
plt.ylabel('Number of nodes')
plt.title('Degree centrality histogram')
plt.savefig("files/ch08/degree_centrality_hist.png", dpi = 300, bbox_inches = 'tight')


# Let us compute the degree histogram and plot it using logaithmic scale.
# 

fig = plt.figure(figsize = (6,5))
degree_hist = plt.hist(list(degree_cent_fb.values()), 100)
plt.loglog(degree_hist[1][1:], degree_hist[0], 'b', marker = 'o')
plt.ylabel('Number of nodes (log)')
plt.xlabel('Degree centrality (log)')
plt.title('Sorted nodes degree (loglog)')
plt.savefig("files/ch08/degree_centrality_hist_log.png", dpi = 300, bbox_inches = 'tight')


# There is an interesting (large) cluster which corresponds to low degrees. The representation using a logarithmic scale  is useful to distinguish the members of this cluster, which are clearly visible as a straight line at lower x-axis values (upper left-hand part).
# We can conclude that most of the nodes in the graph have low degree centrality; only a few of them have high degree centrality. These latter nodes can be properly seen as the points in the bottom right-hand part of the logarithmic plot.
# 

# Let's compute the other centrality measures: betweenness, closeness and eigenvector. We sort the results to see which nodes are more central.
# 

betweenness_fb = nx.betweenness_centrality(fb)
closeness_fb = nx.closeness_centrality(fb)
eigencentrality_fb = nx.eigenvector_centrality(fb)
print 'Facebook betweenness centrality:', sorted(betweenness_fb.items(), key = lambda x: x[1], reverse = True)[:10]
print 'Facebook closeness centrality:', sorted(closeness_fb.items(), key = lambda x: x[1], reverse = True)[:10]
print 'Facebook eigenvector centrality:', sorted(eigencentrality_fb.items(), key = lambda x: x[1], reverse = True)[:10]


# As can be seen in the results, the four measures differ in their ordering. Although the node '107' is the most central node for degree, betweenness and closeness centrality it is not for eigenvector centrality.
# The second most central node is different for closeness and eigenvector centralities; while the third most central node is different for all four centrality measures.
# 

# Another interesting measure is the *current flow betweenness centrality*, also called *random walk centrality*, of a node. It can be defined as the probability of passing through the node in question on a random walk starting and ending at some node.
# In this way, the betweenness is not computed as a function of shortest paths, but of all paths.
# This makes sense for some social networks where messages may get to where they are going not by the shortest path, but by a random path, as in the case of gossip floating through a social network for example.
# 
# Computing the current flow betweenness centrality can take a while, so we work with a trimmed Facebook network instead of the original one. 
# 

# What happen if we consider only the graph nodes with more than the average degree of the network (21)? 
# We can trim the graph using degree centrality values. To do this, in the next code, we define the function to trim the graph based on the degree centrality of the graph nodes. We set the threshold to 21 connections:
# 

def trim_degree_centrality(graph, degree = 0.01):
    g = graph.copy()
    d = nx.degree_centrality(g)
    for n in g.nodes():
        if d[n] <= degree:
            g.remove_node(n)
    return g

thr = 21.0/(fb.order() - 1.0)
print 'Degree centrality threshold:', thr

fb_trimed = trim_degree_centrality (fb , degree = thr)
print 'Remaining # nodes:', len (fb_trimed)


# The new graph is much smaller; we have removed almost half of the nodes (we have moved from 4,039 to 2,226 nodes).
# 

# The current flow betweenness centrality measure needs connected graphs, as does any betweenness centrality measure, so we should first extract a connected component from the trimmed Facebook network and then compute the measure:
# 

fb_subgraph = list(nx.connected_component_subgraphs(fb_trimed))
print 'Number of found sub graphs:', np.size(fb_subgraph)
print 'Number of nodes in the first sub graph:', len(fb_subgraph[0])


betweenness = nx.betweenness_centrality(fb_subgraph[0])
print 'Trimmed Facebook betweenness centrality: ', sorted(betweenness.items(), key = lambda x: x[1], reverse = True)[:10]


current_flow = nx.current_flow_betweenness_centrality(fb_subgraph[0])
print 'Trimmed Facebook current flow betweenness centrality:', sorted(current_flow.items(), key = lambda x: x[1], reverse = True)[:10]


# As can be seen, there are similarities in the 10 most central nodes for the betweenness and current flow betweenness centralities. In particular, seven up to ten are the same nodes, even if they are differently ordered.
# 

# ## Drawing Centrality in Graphs
# 
# Let us explore some visualizations of the Facebook Network.
# Graph visualization can help in the network data understanding and usability.
# 

# The visualization of a network with a large amount of nodes is a complex task. Different layouts can be used to try to build a proper visualization. For instance, we can draw the Facebook graph using the random layout, but this is a bad option, as can be seen below:
# 

fig = plt.figure(figsize = (6,6))

pos = nx.random_layout(fb)
nx.draw_networkx(fb, pos, with_labels = False)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Random.png', dpi = 300, bbox_inches = 'tight')


# We can try to draw the graph using default options.
# 

fig = plt.figure(figsize = (6,6))
nx.draw(fb)
plt.savefig('files/ch08/facebook_Default.png', dpi = 300, bbox_inches = 'tight')


# Let us try to define better ways to shape the graph and also to fine tune the parameters.
# We draw centrality measures on the whole Facebook Network.
# For that, we compute the position of the nodes using Spring Layout and then we draw the four measures of Centrality.
# 

# The function nx.spring_layout returns the position of the nodes using the Fruchterman–Reingold force-directed algorithm.
# This algorithm distributes the graph nodes in such a way that all the edges are more or less equally long and they cross themselves as few times as possible. Moreover, we can change the size of the nodes to that defined by their degree centrality. 
# 

pos_fb = nx.spring_layout(fb, iterations = 1000)


# In the next code, the degree centrality is normalized to values between 0 and 1, and multiplied by a constant to make the sizes appropriate for the format of the figure.
# 

fig = plt.figure(figsize = (6,6))
nsize = np.array([v for v in degree_cent_fb.values()])
cte = 500
nsize = cte*(nsize  - min(nsize))/(max(nsize)-min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize, with_labels = True)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1, with_labels = True)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Degree.png', dpi = 300, bbox_inches = 'tight')


fig = plt.figure(figsize=(6,6))

# Betweenness Centrality
nsize = np.array([v for v in betweenness_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges=nx.draw_networkx_edges(fb, pos = pos_fb,alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Betweenness.png', dpi = 300, bbox_inches = 'tight')


fig = plt.figure(figsize=(6,6))

# Eigenvector Centrality
nsize = np.array([v for v in eigencentrality_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges = nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Eigenvector.png', dpi = 300, bbox_inches = 'tight')


fig = plt.figure(figsize=(6,6))

# Closeness Centrality
nsize = np.array([v for v in closeness_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Closeness.png', dpi = 300, bbox_inches = 'tight')


# These graph visualizations allow us to understand the network better.
# Now we can distinguish several groups of nodes or "communities" more clearly in the graph.
# Moreover, the more central nodes are the larger more prominent nodes, which are highly connected.
# 

# Generally different centrality metrics will be positively correlated, but when they are not, there is probably something interesting about the network nodes.
# For instance, if you can spot nodes with high betweenness but relatively low degree, these are the nodes with few links but which are  crucial for network flow.
# We can also look for the opposite effect: nodes with high degree but relatively low betweenness. These nodes are those with redundant communication.
# 

# ### Pagerank
# Pagerank is an algorithm related to the concept of eigenvector centrality in directed graphs. It is used to rate webpages objectively and effectively measure the attention devoted to them.
# Pagerank was invented by Larry Page and Sergey Brin, and became a Google trademark in 1998.
# Assigning the importance of a webpage is a subjective task, which depends on the
# interests and knowledge of the browsers.
# However, there are ways to objectively rank the relative importance of webpages.
# Intuitively, a page has a high rank if the sum of the ranks of its incoming edges is high. This considers both cases when a page has many incoming links and when a page has a few highly ranked incoming links.
# Nowadays, a variant of the algorithm is used by Google. It does not only use information on the number of links pointing into and out of a website, but uses many more variables.
# 

# Let us compute the Pagerank vector of the Facebook network and use it to define the size of the nodes, as was done above.
# The code below outputs the graph that emphasizes some of the nodes with high Pagerank:
# 

fig = plt.figure(figsize = (6,6))

# Pagerank 
pr=nx.pagerank(fb, alpha = 0.85)
nsize=np.array([v for v in pr.values()])
cte = 500
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_pagerank.png', dpi = 300, bbox_inches = 'tight')


# ## Ego-networks of the Facebook Network
# 

# Ego-networks are subnetworks of neighbors that are centered on a certain node. In Facebook and LinkedIn, these are described as \emph{your network}. Every person in an ego-network has her/his own ego-network and can only access the nodes in it. All ego-networks interlock to form the whole social network.
# The ego-network definition depends on the network distance considered. In the basic case, a link means that person A is a friends of person B, a distance of 2 means that a person, C, is a friend of a friend of A, and a distance of 3 means that another person, D, is a friend of a friend of a friend of A.
# Knowing the size of an ego-network is important when it comes to understanding the reach of the information that a person can transmit or have access to.
# 

fig = plt.figure(figsize=(6,6))

# Example of ego network:
G = nx.Graph()
G.add_edge('A', 'C');
G.add_edge('A', 'B');
G.add_edge('A', 'D');
G.add_edge('A', 'E');
G.add_edge('A', 'F');
G.add_edge('A', 'G');
G.add_edge('A', 'H');
G.add_edge('A', 'I');
G.add_edge('D', 'C');
G.add_edge('E', 'F');
G.add_edge('G', 'H');
G.add_edge('G', 'I');
G.add_edge('H', 'C');
G.add_edge('H', 'D');
G.add_edge('B', 'I');
c=[1, 2, 2, 2, 2, 2, 2, 2, 2]
nx.draw_networkx(G,  with_labels = False, node_color = c)
plt.axis('off') 
plt.savefig("files/ch08/ego_graph.png")


# Our Facebook network is divided into a set of 10 ego-networks which are interconnected to form the fully connected graph we have been analyzing in previous sections.
# The dataset includes the information of these 10 manually defined ego-networks. In particular, we have available the list of the 10 ego nodes: '0', '107', '348', '414', '686', '1684', '1912', '3437', '3980' and their connections.
# Above we saw that node '107' is the most central node for three of the four centrality measures computed. So, let us extract the ego-networks of the popular node '107' with a distance of 1 and 2, and compute their sizes. NetworkX has a function devoted to this task:
# 

# Automatically compute ego-network
ego_107 = nx.ego_graph(fb, '107')
print '# nodes of the ego graph 107:', len(ego_107)
print '# nodes of the ego graph 107 with radius up to 2:', len(nx.ego_graph(fb, '107', radius = 2))


# Since the Facebook dataset also provides the manually labeled ego-networks, we can compute the actual size of the ego-networks.
# We can access to the ego-networks by simply importing *os.path* and reading the edge list corresponding as it is done in the next code.
# 

import os.path


ego_id = '107'
G_107 = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(ego_id)), nodetype = int)    
print 'Nodes of the ego graph 107: ', len(G_107)


# As can be seen, the size of the manually defined ego-network of node '107' is slightly different from the ego-network automatically computed using NetworX. This is due to the fact that the manual definition is not necessarily referred to the subgraph of neighbors centered on a node.
# 

# Let's compare the ego-networks by answering the following questions:
# - Which is the most densely connected ego-network?
# - Which is the largest (# nodes) ego-network? 
# - Is there intersection between ego-networks in the Facebook network?

# #### Which is the most densely connected ego-network?
# 

# To do that, we compute the number of edges in every ego-network and select the network with the maximum number:
# 

ego_ids = (0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980)
ego_sizes = np.zeros((10,1))

# Fill the 'ego_sizes' vector with the size (# edges) of the 10 ego-networks in egoids:
i=0
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)    
    ego_sizes[i] = G.size()      
    print 'size of the ego-network ', id,  ego_sizes[i] 
    i +=1


[i_max, j] = (ego_sizes == ego_sizes.max()).nonzero()
ego_max = ego_ids[i_max]
print 'The most densely connected ego-network is the one of ego:', ego_max


# Load the ego network of node 1912
G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(ego_max)), nodetype = int)
G_n = G.order()
G_k = G.size()
G_avg_deg = G_k / G_n
print 'Nodes: ', G_n
print 'Edges: ', G_k
print 'Average degree: ', G_avg_deg


# The most densely connected ego-network is that of node '1912', which has an average degree of 40.
# 

# #### Which is the largest (# nodes) ego-network?
# 

ego_sizes = np.zeros((10,1))
i = 0
# Fill the 'egosizes' vector with the size (# nodes) of the 10 ego-networks in egoids:
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)    
    ego_sizes[i] = G.order()      
    print 'size of the ego-network ', id,  ego_sizes[i] 
    i += 1


[i_max, j] = (ego_sizes == ego_sizes.max()).nonzero()
ego_max = ego_ids[i_max]
print 'The largest ego-network is the one of ego: ', ego_max


G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(ego_max)), nodetype = int)
G_n = G.order()
G_k = G.size()
G_avg_deg = G_k / G_n
print 'Nodes: ', G_n
print 'Edges: ', G_k
print 'Average degree: ', G_avg_deg


# #### Is there intersection between ego-networks in the Facebook network?
# 

# Add a field 'egonet' to the nodes of the whole facebook network. 
# Default value egonet=[], meaning that this node does not belong to any ego-netowrk
for i in fb.nodes() :
    fb.node[str(i)]['egonet'] = []


# Fill the 'egonet' field with one of the 10 ego values in ego_ids:
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)
    print id
    for n in G.nodes() :
        if (fb.node[str(n)]['egonet'] == []) :
            fb.node[str(n)]['egonet'] = [id]
        else :
            fb.node[str(n)]['egonet'].append(id)


# Compute the intersections:
S = [len(x['egonet']) for x in fb.node.values()]


print '# nodes belonging to 0 ego-network: ', sum(np.equal(S,0))
print '# nodes belonging to 1 ego-network: ', sum(np.equal(S,1))
print '# nodes belonging to 2 ego-network: ', sum(np.equal(S,2))
print '# nodes belonging to 3 ego-network: ', sum(np.equal(S,3))
print '# nodes belonging to 4 ego-network: ', sum(np.equal(S,4))
print '# nodes belonging to more than 4 ego-network: ', sum(np.greater(S,4))


# As can be seen, there is intersection between the ego-networks in the Facebook network, since some of the nodes belongs to more than 1 and up to 4 ego-networks simultaneously.
# 

# #### More drawings:
# Let's draw the ego-networks with different colors on the whole facebook network:
# 

# Add a field 'egocolor' to the nodes of the whole facebook network. 
# Default value egocolor=0, meaning that this node does not belong to any ego-netowrk

for i in fb.nodes() :
    fb.node[str(i)]['egocolor'] = 0
    
# Fill the 'egocolor' field with a different color number for each ego-network in ego_ids:
id_color = 1
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)
    for n in G.nodes() :
        fb.node[str(n)]['egocolor'] = id_color
    id_color += 1 


colors = [ x['egocolor'] for x in fb.node.values()]


fig = plt.figure(figsize = (6,6))

nsize = np.array([v for v in degree_cent_fb.values()])
nsize = 500*(nsize  - min(nsize))/(max(nsize) - min(nsize))

nodes=nx.draw_networkx_nodes(fb,pos = pos_fb, cmap = plt.get_cmap('Paired'), node_color = colors, 
                             node_size = nsize, with_labels = False)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Colors.png', dpi = 300, bbox_inches = 'tight')


# ## Community detection
# 

# Let's compute communities automaticaly and plot them with different colors on the whole facebook network.
# Before import the toolbox download and install in following the instructions in the link:
# https://pypi.python.org/pypi/python-louvain/0.3
# 
# 

import community
partition = community.best_partition(fb)


print "# found communities:", max(partition.values())


colors2 = [partition.get(node) for node in fb.nodes()]


fig = plt.figure(figsize = (6,6))

nsize = np.array([v for v in degree_cent_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))

nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, cmap = plt.get_cmap('Paired'), node_color = colors2, 
                             node_size = nsize, with_labels = False)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_AutoPartition.png', dpi = 300, bbox_inches = 'tight')


# As can be seen, the 15 communities found automatically are similar to the 10 ego-networks loaded from the dataset. However, some of the 10 ego-networks are subdivided into several communities now.
# This discrepancy is due to the fact that the ego-networks are manually annotated based on more properties of the nodes, whereas communities are extracted based only on the graph information.
# 

