import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import io


from sklearn import linear_model as lmd


InFile1          = 'LinSepC1.mat'
InFile2          = 'LinSepC2.mat'
C1Dict           = io.loadmat(InFile1)
C2Dict           = io.loadmat(InFile2)
C1               = C1Dict['LinSepC1']
C2               = C2Dict['LinSepC2']
NSampsClass    = 200
NSamps         = 2*NSampsClass


### Set Target Outputs ###
TargetOutputs                     =  np.ones((NSamps,1))
TargetOutputs[NSampsClass:NSamps] = -TargetOutputs[NSampsClass:NSamps]


AllSamps     = np.concatenate((C1,C2),axis=0)


AllSamps.shape


#import sklearn
#sklearn.__version__


get_ipython().set_next_input('LinMod = lmd.LinearRegression.fit');get_ipython().magic('pinfo lmd.LinearRegression.fit')


LinMod = lmd.LinearRegression.fit


M = lmd.LinearRegression()


print(M)


LinMod = lmd.LinearRegression.fit(M, AllSamps, TargetOutputs, sample_weight=None)


R = lmd.LinearRegression.score(LinMod, AllSamps, TargetOutputs, sample_weight=None)


print(R)


LinMod


w = LinMod.coef_
w


w0 = LinMod.intercept_
w0


### Question:  How would we compute the outputs of the regression model?


# Learn About Kernels
# 

# Do some SVM Classification
# 

from sklearn.svm import SVC


### SVC wants a 1d array, not a column vector
Targets = np.ravel(TargetOutputs)


InitSVM = SVC()
InitSVM


TrainedSVM = InitSVM.fit(AllSamps, Targets)


y = TrainedSVM.predict(AllSamps)


plt.figure(1)
plt.plot(y)
plt.show()


d = TrainedSVM.decision_function(AllSamps)


plt.figure(1)
plt.plot(d)
plt.show()


# Can try it with Outliers if we have time
# 

# Let's look at some spectra
# 

### Look at some Pine and Oak spectra from
### NEON Site D03 Ordway-Swisher Biological Station
### at UF
### Pinus palustris
### Quercus virginiana
InFile1 = 'Pines.mat'
InFile2 = 'Oaks.mat'
C1Dict  = io.loadmat(InFile1)
C2Dict  = io.loadmat(InFile2)
Pines   = C1Dict['Pines']
Oaks    = C2Dict['Oaks']


WvFile  = 'NEONWvsNBB.mat'
WvDict  = io.loadmat(WvFile)
Wv      = WvDict['NEONWvsNBB']


Pines.shape


Oaks.shape


NBands=Wv.shape[0]
print(NBands)


# Notice that these training sets are unbalanced
# 

NTrainSampsClass = 600
NTestSampsClass  = 200
Targets          = np.ones((1200,1))
Targets[range(600)] = -Targets[range(600)]
Targets             = np.ravel(Targets)
print(Targets.shape)


plt.figure(111)
plt.plot(Targets)
plt.show()


TrainPines = Pines[0:600,:]
TrainOaks  = Oaks[0:600,:]
#TrainSet   = np.concatenate?


TrainSet   = np.concatenate((TrainPines, TrainOaks), axis=0)
print(TrainSet.shape)


plt.figure(3)
### Plot Pine Training Spectra ###
plt.subplot(121)
plt.plot(Wv, TrainPines.T)
plt.ylim((0.0,0.8))
plt.xlim((Wv[1], Wv[NBands-1]))
### Plot Oak Training Spectra ###
plt.subplot(122)
plt.plot(Wv, TrainOaks.T)
plt.ylim((0.0,0.8))
plt.xlim((Wv[1], Wv[NBands-1]))
plt.show()


InitSVM= SVC()


TrainedSVM=InitSVM.fit(TrainSet, Targets)


plt.figure(4)
plt.plot(d)
plt.show()


# Does this seem to be too good to be true?

TestPines = Pines[600:800,:]
TestOaks  = Oaks[600:800,:]


TestSet = np.concatenate((TestPines, TestOaks), axis=0)
print(TestSet.shape)


dtest = TrainedSVM.decision_function(TestSet)


plt.figure(5)
plt.plot(dtest)
plt.show()


# Yeah, too good to be true...What can we do?

# Error Analysis: Identify characteristics of Errors, Try different Magic Numbers using Cross Validation, etc.
# 




