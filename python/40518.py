# # Facies classification from well logs with Convolutional Neural Networks (CNN)
# 
# ## Shiang Yong Looi
# 
# Using Keras running on top for Tensorflow, we build two CNNs : first to impute PE on two wells with missing data and then for the main task of classifying facies. 
# 

import numpy as np
import pandas
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, normalization, Convolution1D
from keras.callbacks import History
from keras.utils import np_utils
from keras.callbacks import History
from sklearn import metrics
from classification_utilities import display_cm


# Read in data
data = pandas.read_csv('./facies_vectors.csv')
Y_train = data[data['PE'].notnull()]['PE'].values


def prepare_feature_vectors(data, features, window_width):

    raw_feature_vectors = data[features]
    well_labels = data['Well Name']
    num_features = np.shape(raw_feature_vectors)[1]

    output = np.zeros((1, window_width, num_features))
    for x in well_labels.unique():
        well = raw_feature_vectors[well_labels == x].values
        well = np.concatenate((np.repeat(well[0:1], np.floor((window_width-1)/2.0), axis=0), well,
                              np.repeat(well[-1:None], np.floor(window_width/2.0), axis=0)), axis=0)

        tmp = np.zeros((np.size(well, axis=0) - window_width + 1, window_width, num_features))
        for i in np.arange(np.size(well, axis=0) - window_width + 1):
            tmp[i] = np.reshape(well[i: i + window_width], (window_width, num_features))

        output = np.append(output, tmp, axis=0)

    return output[1:]


# Window around central value and list the six features we are using
window_width = 15
feature_list = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
X_train = prepare_feature_vectors(data[data['PE'].notnull()], feature_list, window_width)
num_train_samples = np.asarray(np.shape(X_train))[0]

X_test = prepare_feature_vectors(data[data['PE'].isnull()], feature_list, window_width)
num_test_samples = np.asarray(np.shape(X_test))[0]

print('Training Samples=', num_train_samples, '   Test Samples=', num_test_samples)


# define neural network to perform regression on PE
num_filters = 12
dropout_prob = 0.6666

cnn = Sequential()
cnn.add(Convolution1D(num_filters, 1, border_mode='valid', input_shape=(window_width, len(feature_list))))
cnn.add(normalization.BatchNormalization())
cnn.add(Activation('tanh'))
cnn.add(Convolution1D(num_filters, 3, border_mode='valid'))
cnn.add(normalization.BatchNormalization())
cnn.add(Activation('tanh'))
cnn.add(Dropout(dropout_prob / 2))

cnn.add(Flatten())
cnn.add(Dense(4*num_filters))
cnn.add(normalization.BatchNormalization())
cnn.add(Activation('tanh'))
cnn.add(Dropout(dropout_prob))

cnn.add(Dense(1))
cnn.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
cnn.summary()

# save initial weights, which are random
initial_weights = cnn.get_weights()


# define training parameters and prepare arrays to store training metrics
epochs_per_fold = 1000
num_fold = 5
roll_stride = np.ceil(num_train_samples/num_fold).astype(int)

cnn_hist = History()
hist = np.zeros((4, num_fold, epochs_per_fold))
f1scores = np.zeros(num_fold)
Y_test = np.zeros((num_test_samples, num_fold))


# shuffle input data
rand_perm = np.random.permutation(num_train_samples)
X_train = X_train[rand_perm]
Y_train = Y_train[rand_perm]


# use 5-fold cross validation and train 5 neural networks, ending up with 5 sets of predictions
for i in np.arange(num_fold):
    cnn.set_weights(initial_weights)
    X_train = np.roll(X_train, i*roll_stride, axis=0)
    Y_train = np.roll(Y_train, i*roll_stride, axis=0)

    cnn.fit(X_train, Y_train, batch_size=150, nb_epoch=epochs_per_fold, verbose=0,
                validation_split=1.0/num_fold, callbacks=[cnn_hist])

    # make predictions, i.e. impute PE
    Y_test[:, i] = cnn.predict(X_test)[:, 0]

    hist[:, i, :] = [cnn_hist.history['acc'], cnn_hist.history['val_acc'],
                     cnn_hist.history['loss'], cnn_hist.history['val_loss']]
    print("Accuracy  =", np.mean(hist[1, i, -100:]))


# plot callbacks to evaluate quality of training
drop_values = 100
drop_hist = np.reshape(hist[:, :, drop_values:], (4, num_fold * (epochs_per_fold - drop_values)))
print("Mean Validation Accuracy  =", np.mean(hist[1, :, -drop_values:]))

plt.plot(drop_hist[0]); plt.plot(drop_hist[1])
plt.legend(['train', 'val'], loc='lower left')


plt.plot(drop_hist[2]); plt.plot(drop_hist[3])
plt.legend(['train', 'val'], loc='upper left')


# Update dataframe with imputed values by averaging results of 5 neural networks 
data['PE'][np.array(data['PE'].isnull())] = np.mean(Y_test, axis=1)

# Write intermediate data to file
data.to_csv('./ShiangYong/facies_vectors_imputedPE.csv', index=False)


# Plot PE of all wells (original plus imputed) 
plt.plot(data['PE'])


# At this point, we have a training dataset with imputed PE so we can proceed with the facies classification.
# 

# Read in data with imputed PE, also read in two test wells
# data = pandas.read_csv('facies_vectors.csv')
data = pandas.read_csv('./ShiangYong/facies_vectors_imputedPE.csv')
blind_wells = pandas.read_csv('./nofacies_data.csv')

# Impute missing values with average if there are NaNs in PE
if data['PE'].isnull().any():
    data['PE'] = data['PE'].fillna(value=data['PE'].mean())


# Convert facies class to one-hot-vector representation
num_classes = data['Facies'].unique().size
Y_train = np_utils.to_categorical(data['Facies'].values-1, num_classes)

# Window around central value and define the seven features we are using
window_width = 15
feature_list = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
X_train = prepare_feature_vectors(data, feature_list, window_width)
X_test = prepare_feature_vectors(blind_wells, feature_list, window_width)

num_train_samples = np.asarray(np.shape(X_train))[0]
num_test_samples = np.asarray(np.shape(X_test))[0]

print('Training Samples=', num_train_samples, '   Test Samples=', num_test_samples)


# define neural network to classify facies
num_filters = 12
dropout_prob = 0.6

convnet = Sequential()
convnet.add(Convolution1D(num_filters, 1, border_mode='valid',
                          input_shape=(window_width, len(feature_list))))
convnet.add(Activation('relu'))
convnet.add(Convolution1D(7, 1, border_mode='valid'))
convnet.add(Activation('relu'))
convnet.add(Convolution1D(num_filters, 3, border_mode='valid'))
convnet.add(Activation('relu'))
convnet.add(Dropout(dropout_prob / 2))

convnet.add(Flatten())
convnet.add(Dense(4 * num_filters))
convnet.add(normalization.BatchNormalization())
convnet.add(Activation('sigmoid'))
convnet.add(Dropout(dropout_prob))

convnet.add(Dense(num_classes, activation='softmax'))
convnet.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
convnet.summary()

# save initial weights
initial_weights = convnet.get_weights()


# define training parameters and prepare arrays to store training metrics
epochs_per_fold = 1500
num_fold = 6
roll_stride = np.ceil(num_train_samples/num_fold).astype(int)

convnet_hist = History()
hist = np.zeros((4, num_fold, epochs_per_fold))
f1scores = np.zeros(num_fold)
Y_test_ohv = np.zeros((num_test_samples, num_fold, num_classes))


# shuffle input data
rand_perm = np.random.permutation(num_train_samples)
X_train = X_train[rand_perm]
Y_train = Y_train[rand_perm]


# use 6-fold cross validation and train 6 neural networks, ending up with 6 sets of predictions
for i in np.arange(num_fold):
    convnet.set_weights(initial_weights)
    X_train = np.roll(X_train, i*roll_stride, axis=0)
    Y_train = np.roll(Y_train, i*roll_stride, axis=0)

    convnet.fit(X_train, Y_train, batch_size=200, nb_epoch=epochs_per_fold, verbose=0,
                validation_split=1.0/num_fold, callbacks=[convnet_hist])

    hist[:, i, :] = [convnet_hist.history['acc'], convnet_hist.history['val_acc'],
                     convnet_hist.history['loss'], convnet_hist.history['val_loss']]

    Y_predict = 1 + np.argmax(convnet.predict(X_train), axis=1)
    f1scores[i] = metrics.f1_score(1 + np.argmax(Y_train, axis=1), Y_predict, average='micro')
    print('F1 Score =', f1scores[i])

    Y_test_ohv[:, i, :] = convnet.predict(X_test)
    
print('Average F1 Score =', np.mean(f1scores))


# Plot callbacks
hist = np.reshape(hist, (4, num_fold * epochs_per_fold))
plt.plot(hist[0]); plt.plot(hist[1])
plt.legend(['train', 'val'], loc='lower left')


plt.plot(hist[2]); plt.plot(hist[3])
plt.legend(['train', 'val'], loc='upper left')


# Soft majority voting on 6 predictions to produce final prediction 
Y_test = 1 + np.argmax(np.sum(Y_test_ohv, axis=1), axis=1)

# Append predictions to dataframe and write to file
blind_wells['Facies'] = Y_test
blind_wells.to_csv('./ShiangYong/predicted_facies_ver04.csv', index=False)





# ## Facies classification - Sequential Feature Selection
# 

# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The code and ideas in this notebook,</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Matteo Niccoli and Mark Dahl,</span> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
# 

# The [mlxtend](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/) library used for the sequential feature selection is by [Sebastian Raschka](https://sebastianraschka.com/projects.html).
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, make_scorer


filename = 'train_engineered_features.csv'
train_data = pd.read_csv(filename)
train_data.describe()


train_data['Well Name'] = train_data['Well Name'].astype('category')
train_data['Formation'] = train_data['Formation'].astype('category')
train_data['Well Name'].unique()


y = train_data['Facies'].values
print y[25:40]
print np.shape(y)


X = train_data.drop(['Formation', 'Well Name','Facies'], axis=1)
print np.shape(X)
X.describe(percentiles=[.05, .25, .50, .75, .95])


stdscaler = preprocessing.StandardScaler().fit(X)
X = stdscaler.transform(X)


# ### Make performance scorers 
# 

Fscorer = make_scorer(f1_score, average = 'micro')


# ### Sequential Feature Selection with mlextend
# 
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
# 

from sklearn.ensemble import RandomForestClassifier


# ### The next cell will take many hours to run, skip it
# 

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
clf = RandomForestClassifier(random_state=49)

sfs = SFS(clf, 
          k_features=100, 
          forward=True, 
          floating=False, 
          scoring=Fscorer,
          cv = 8,
          n_jobs = -1)

sfs = sfs.fit(X_train, y_train)


np.save('sfs_RF_metric_dict.npy', sfs.get_metric_dict()) 


# ### Restart from here
# 

# load previously saved dictionary
read_dictionary = np.load('sfs_RF_metric_dict.npy').item()


# plot results
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# run this twice
fig = plt.figure()                                                               
ax = plot_sfs(read_dictionary, kind='std_err')
fig_size = plt.rcParams["figure.figsize"] 
fig_size[0] = 20
fig_size[1] = 18

plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.xticks( rotation='vertical')
locs, labels = plt.xticks()
plt.xticks( locs, labels)
plt.show()


# ##### The curve stabiizes at about 6 features, reaches a max at 15, then more or less flattens up to about 70 features wher it begins to tail off. We will save the top 45 and 75. 
# 

# save results to dataframe
selected_summary = pd.DataFrame.from_dict(read_dictionary).T
selected_summary['index'] = selected_summary.index
selected_summary.sort_values(by='avg_score', ascending=0)


# save dataframe
selected_summary.to_csv('SFS_RF_selected_features_summary.csv', sep=',', header=True, index = False)


# re load saved dataframe and sort by score
filename = 'SFS_RF_selected_features_summary.csv'
selected_summary = pd.read_csv(filename)
selected_summary = selected_summary.set_index(['index'])
selected_summary.sort_values(by='avg_score', ascending=0).head()


# feature selection with highest score
selected_summary.iloc[44]['feature_idx']


slct = np.array([257, 3, 4, 6, 7, 8, 10, 12, 16, 273, 146, 19, 26, 27, 284, 285, 30, 34, 163, 1, 42, 179, 155, 181, 184, 58, 315, 190, 320, 193, 194, 203, 290, 80, 210, 35, 84, 90, 97, 18, 241, 372, 119, 120, 126])
slct


# isolate and save selected features
filename = 'train_engineered_features.csv'
train_data = pd.read_csv(filename)
trainX = train_data.drop(['Formation', 'Well Name','Facies'], axis=1)
trainXs = trainX.iloc[:, slct]
trainXs = pd.concat([train_data[['Depth', 'Well Name', 'Formation', 'Facies']], trainXs], axis = 1)
print np.shape(trainXs), list(trainXs)
trainXs.to_csv('train_SFS_top45_engfeat.csv', sep=',',  index=False)


# isolate and save selected features
filename = 'test_engineered_features.csv'
test_data = pd.read_csv(filename)
testX = test_data.drop(['Formation', 'Well Name'], axis=1)
testXs = testX.iloc[:, slct]
testXs = pd.concat([test_data[['Depth', 'Well Name', 'Formation']], testXs], axis = 1)
print np.shape(testXs), list(testXs)
testXs.to_csv('test_SFS_top45_engfeat.csv', sep=',',  index=False)


# feature selection with highest score
selected_summary.iloc[74]['feature_idx']


slct = np.array([257, 3, 4, 5, 6, 7, 8, 265, 10, 12, 13, 16, 273, 18, 19, 26, 27, 284, 285, 30, 34, 35, 1, 42, 304, 309, 313, 58, 315, 319, 320, 75, 80, 338, 84, 341, 89, 90, 92, 97, 101, 102, 110, 372, 119, 120, 122, 124, 126, 127, 138, 139, 146, 155, 163, 165, 167, 171, 177, 179, 180, 181, 184, 190, 193, 194, 198, 203, 290, 210, 211, 225, 241, 249, 253])
slct


# isolate and save selected features
filename = 'train_engineered_features.csv'
train_data = pd.read_csv(filename)
trainX = train_data.drop(['Formation', 'Well Name','Facies'], axis=1)
trainXs = trainX.iloc[:, slct]
trainXs = pd.concat([train_data[['Depth', 'Well Name', 'Formation', 'Facies']], trainXs], axis = 1)
print np.shape(trainXs), list(trainXs)
trainXs.to_csv('train_SFS_top75_engfeat.csv', sep=',',  index=False)


# isolate and save selected features
filename = 'test_engineered_features.csv'
test_data = pd.read_csv(filename)
testX = test_data.drop(['Formation', 'Well Name'], axis=1)
testXs = testX.iloc[:, slct]
testXs = pd.concat([test_data[['Depth', 'Well Name', 'Formation']], testXs], axis = 1)
print np.shape(testXs), list(testXs)
testXs.to_csv('test_SFS_top75_engfeat.csv', sep=',',  index=False)





# # 'Grouped' k-fold CV
# 
# ### A quick demo by Matt
# 
# In cross-validating, we'd like to drop out one well at a time. `LeaveOneGroupOut` is good for this:
# 

import pandas as pd
training_data = pd.read_csv('../training_data.csv')


# Isolate X and y:
# 

X = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1).values
y = training_data['Facies'].values


# We want the well names to use as groups in the k-fold analysis, so we'll get those too:
# 

wells = training_data["Well Name"].values


# Now we train as normal, but `LeaveOneGroupOut` gives us the approriate indices from `X` and `y` to test against one well at a time:
# 

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()

for train, test in logo.split(X, y, groups=wells):
    well_name = wells[test[0]]
    score = SVC().fit(X[train], y[train]).score(X[test], y[test])
    print("{:>20s}  {:.3f}".format(well_name, score))





# # Facies classification using machine learning techniques
# Contact author: Clara Castellanos, CCastellanos10@slb.com
# 
# Notebook based on :
# - Notebook by <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini's</a> 
# - notebook by Alan Richardson (Ausar Geophysical) 
# - notebook by SHandPR
# 
# 
# Evaluates the effect of 
# - Feature augmentation strategies.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from scipy.signal import medfilt
seed = 123
np.random.seed(seed)


import sys, scipy, sklearn
#print('Python:  ' + sys.version.split('\n')[0])
#print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 




# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. This is the only cell that differs from the approach of Paolo Bestagini. Currently no feature engineering is used, but this should be explored in the future.
# 

reg = RandomForestRegressor(max_features='sqrt', n_estimators=250)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))
data['PE']=X[:,4]


# ## Introduce more features 
# 

def add_more_features(data,feature_names):
    data['GR025']=np.power(np.abs(data['GR']),0.25)
    data['GR2']=np.power(np.abs(data['GR']),2)
    data['PHIND025']=np.power(np.abs(data['PHIND']),0.25)
    data['PHIND2']=np.power(np.abs(data['PHIND']),2)
    data['DeltaPHIlog']=np.power(data['DeltaPHI'],2)
    data['DeltaPHI05']=np.power(data['DeltaPHI'],3)
    data['NM_M_GR']= data['NM_M']* data['GR']
    data['NM_M_PHIND']= data['NM_M']* data['PHIND']
    data['NM_M_DeltaPHI']= data['NM_M']* data['DeltaPHI']
    data['GR_PHIND']= data['GR']* data['PHIND']
    data['NM_M_ILD_log10']= data['NM_M']* data['ILD_log10']
    data['NM_M_ILD_log10_GR']= data['NM_M']* data['ILD_log10']* data['GR']
    data['NM_M_ILD_log10_GR_PHIND']= data['NM_M']* data['ILD_log10']* data['GR']* data['PHIND']
    data['ILD_log10_GR_PHIND']= data['ILD_log10']* data['GR']* data['PHIND']
    data['ILD_log10_GR_PHIND_DeltaPHI']= data['ILD_log10']* data['GR']* data['PHIND']* data['DeltaPHI']
    feature_names= feature_names+['GR025','GR2','PHIND025','PHIND2','DeltaPHIlog','DeltaPHI05','NM_M_GR','NM_M_PHIND',
                                  'NM_M_DeltaPHI','GR_PHIND','NM_M_ILD_log10','NM_M_ILD_log10_GR','NM_M_ILD_log10_GR_PHIND','ILD_log10_GR_PHIND','ILD_log10_GR_PHIND_DeltaPHI']
    # Store features and labels
    X = data[feature_names].values  # features
    y = data['Facies'].values  # labels
    # Store well labels and depths
    well = data['Well Name'].values
    depth = data['Depth'].values
    return (data,feature_names,X,y,well,depth)


data,feature_names,X,y,well,depth= add_more_features(data,feature_names)


# ## Feature augmentation
# Our guess is that facies do not abrutly change from a given depth layer to the next one. Therefore, we consider features at neighboring layers to be somehow correlated. To possibly exploit this fact, let us perform feature augmentation by:
# - Aggregating features at neighboring depths.
# - Computing feature spatial gradient.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth):
    
    # Augment features
    padded_rows = []
    X_aug = np.zeros((X.shape[0], X.shape[1]*2))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X[w_idx, :], X_aug_grad), axis=1)
        padded_rows.append(w_idx[-1])
        
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

print('No of Feats',X.shape[1])


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [100]  
MD_grid = [3]  
M_grid = [30]
LR_grid = [0.1]  
L_grid = [5]
S_grid = [25]  
param_grid = []
for N in N_grid:
    for M in MD_grid:
        for M1 in M_grid:
            for S in LR_grid: 
                for L in L_grid:
                    for S1 in S_grid:
                        param_grid.append({'N':N, 'MD':M, 'MF':M1,'LR':S,'L':L,'S1':S1})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    #clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0), n_jobs=-1)
    #clf = RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0)
    #clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='gini',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0), n_jobs=-1)
    # Train classifier  
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss='exponential',
                                                        n_estimators=param['N'], 
                                                        learning_rate=param['LR'], 
                                                        max_depth=param['MD'],
                                                        max_features= param['MF'],
                                                        min_samples_leaf=param['L'],
                                                        min_samples_split=param['S1'],
                                                        random_state=seed, 
                                                        max_leaf_nodes=None, 
                                                        verbose=1), n_jobs=-1)
    
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return (y_v_hat, clf)


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        addnoise=0
        if ( addnoise==1 ):
            X_tr=X_tr+np.random.normal(loc=np.zeros(X_tr.shape[1]), scale=0.01*np.sqrt(np.std(X_tr,axis=0)/len(X_tr)), size=X_tr.shape)
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        (y_v_hat,clf) = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


feature_names_original = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
data,feature_names,X,y,well,depth= add_more_features(data,feature_names_original)


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 


test_data['Facies']=0
# Prepare test data
test_data,feature_names,X_ts,y_ts,well_ts,depth_ts= add_more_features(test_data,feature_names_original)
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat[0]
test_data.to_csv('cc_predicted_facies_noneigh_boosting_refine8_win.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# # Facies classification using machine learning techniques
# Copy of <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini's</a> "Try 2", and submission2  by Alan Richardson (Ausar Geophysical) with an ML estimator for PE in the wells where it is missing (rather than just using the mean).
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm is based on the use of random forests combined in one-vs-one multiclass strategy. In particular, we would like to study the effect of:
# - Robust feature normalization.
# - Feature augmentation strategies.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from scipy.signal import medfilt
seed = 123
np.random.seed(seed)


import sys, scipy, sklearn
#print('Python:  ' + sys.version.split('\n')[0])
#print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 




# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. This is the only cell that differs from the approach of Paolo Bestagini. Currently no feature engineering is used, but this should be explored in the future.
# 

reg = RandomForestRegressor(max_features='auto', n_estimators=250)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))
data['PE']=X[:,4]


# ## Introduce more features 
# 

def add_more_features(data,feature_names):
    data['GR025']=np.power(np.abs(data['GR']),0.25)
    data['GR2']=np.power(np.abs(data['GR']),2)
    data['PHIND025']=np.power(np.abs(data['PHIND']),0.25)
    data['PHIND2']=np.power(np.abs(data['PHIND']),2)
    data['DeltaPHIlog']=np.power(data['DeltaPHI'],2)
    data['DeltaPHI05']=np.power(data['DeltaPHI'],3)
    data['NM_M_GR']= data['NM_M']* data['GR']
    data['NM_M_PHIND']= data['NM_M']* data['PHIND']
    data['NM_M_DeltaPHI']= data['NM_M']* data['DeltaPHI']
    data['GR_PHIND']= data['GR']* data['PHIND']
    data['NM_M_PE']= data['NM_M']* data['PE']
    data['NM_M_PE_GR']= data['NM_M']* data['PE']* data['GR']
    data['NM_M_PE_GR_PHIND']= data['NM_M']* data['PE']* data['GR']* data['PHIND']
    data['PE_GR_PHIND']= data['PE']* data['GR']* data['PHIND']
    data['PE_GR_PHIND_DeltaPHI']= data['PE']* data['GR']* data['PHIND']* data['DeltaPHI']
    feature_names= feature_names+['GR025','GR2','PHIND025','PHIND2','DeltaPHIlog','DeltaPHI05','NM_M_GR','NM_M_PHIND',
                                  'NM_M_DeltaPHI','GR_PHIND','NM_M_PE','NM_M_PE_GR','NM_M_PE_GR_PHIND','PE_GR_PHIND','PE_GR_PHIND_DeltaPHI']
    # Store features and labels
    X = data[feature_names].values  # features
    y = data['Facies'].values  # labels
    # Store well labels and depths
    well = data['Well Name'].values
    depth = data['Depth'].values
    return (data,feature_names,X,y,well,depth)


data,feature_names,X,y,well,depth= add_more_features(data,feature_names)


feature_names


data.isnull().sum()


# ## Feature augmentation
# Our guess is that facies do not abrutly change from a given depth layer to the next one. Therefore, we consider features at neighboring layers to be somehow correlated. To possibly exploit this fact, let us perform feature augmentation by:
# - Aggregating features at neighboring depths.
# - Computing feature spatial gradient.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    Norigfeat=len(feature_names)
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
       
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:Norigfeat] == np.zeros((1, Norigfeat)))[0])
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth, N_neig=0)


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

print('No of Feats',X.shape[1])


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [150]  
MD_grid = [3]  
M_grid = [30] #[25,30]
LR_grid = [0.1]  
L_grid = [5]
S_grid = [20]#[10,20]  
param_grid = []
for N in N_grid:
    for M in MD_grid:
        for M1 in M_grid:
            for S in LR_grid: 
                for L in L_grid:
                    for S1 in S_grid:
                        param_grid.append({'N':N, 'MD':M, 'MF':M1,'LR':S,'L':L,'S1':S1})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    #clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0), n_jobs=-1)
    #clf = RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0)
    #clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='gini',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0), n_jobs=-1)
    # Train classifier  
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss='exponential',
                                                        n_estimators=param['N'], 
                                                        learning_rate=param['LR'], 
                                                        max_depth=param['MD'],
                                                        max_features= param['MF'],
                                                        min_samples_leaf=param['L'],
                                                        min_samples_split=param['S1'],
                                                        random_state=seed, 
                                                        max_leaf_nodes=None, 
                                                        verbose=1), n_jobs=-1)
    
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return (y_v_hat, clf)


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        addnoise=1
        if ( addnoise==1 ):
            X_tr=X_tr+np.random.normal(loc=np.zeros(X_tr.shape[1]), scale=0.01*np.sqrt(np.std(X_tr,axis=0)/len(X_tr)), size=X_tr.shape)
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        (y_v_hat,clf) = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


feature_names_original = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
data,feature_names,X,y,well,depth= add_more_features(data,feature_names_original)


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 


test_data['Facies']=0
# Prepare test data
test_data,feature_names,X_ts,y_ts,well_ts,depth_ts= add_more_features(test_data,feature_names_original)
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat[0]
test_data.to_csv('cc_predicted_facies_noneigh_boosting_refine4_win.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# The Facies classification project for SEG A test notes is appended at bottom
# The project has three parts :
# 1. Raw Data analysis (Small data, quick statistics)
# 
# 2. Feature Engineering
#     a. Missing "PE" data : Regressional fillin is better than median and mean (https://github.com/seg/2016-ml-contest/blob/master/LA_Team/Facies_classification_LA_TEAM_05.ipynb)
#     b. How many features to include : 
#       Current tests from other groups use only pre-defined features. 
#       I found Formation has predicting power too, including Formation info give extra uplift to my model See test 8,9
#     c. Feature augmentation : https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb
#       Great works, included the depth information in a nature way
#     d. Robust model scaling 
# 
# 3. Model Selection
#    XGBOOST is superior to SVC (My benchmark)
#    A brutal gridsearch was done on XGBOOST on top of the best feature engineering  
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


## Read in data
train_raw = pd.read_csv('01_raw_data/facies_vectors.csv')
train = train_raw.copy()
cols = train.columns.values
well = train["Well Name"].values
depth = train["Depth"].values


## 01 Raw data analysis
print("No. of Wells is " + str(len(train["Well Name"].unique())))
print("No. of Formation is " + str(len(train["Formation"].unique())))
well_PE_Miss = train.loc[train["PE"].isnull(),"Well Name"].unique()
#print("Wells with Missing PE " + well_PE_Miss)
#print(train.loc[train["Well Name"] == well_PE_Miss[0],["PE","Depth"]].count())
#print(train.loc[train["Well Name"] == well_PE_Miss[1],["PE","Depth"]].count())
#print(train.loc[train["Well Name"] == well_PE_Miss[2],["PE","Depth"]].count())
(train.groupby("Well Name"))["PE"].mean()
(train.groupby("Well Name"))["PE"].median()


### 02 Feature definition and QC functions
features = ['GR', 'ILD_log10', 'DeltaPHI', 
    'PHIND','PE','NM_M', 'RELPOS']
feature_vectors = train[features]
facies_labels = train['Facies']
## 1=sandstone  2=c_siltstone   3=f_siltstone 
## 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
## 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors

facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
train.loc[:,'FaciesLabels'] = train.apply(lambda row: label_facies(row, facies_labels), axis=1)


def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(10, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    ax[5].plot(logs.NM_M, logs.Depth, '-', color='black')
    im=ax[6].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('NoMarine/Marine')
    ax[6].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([]); ax[6].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)


### 03 Feature Engineering tests (SVC and XGB were used to test this)
## a. Fill in missing PE values : Median, mean, NN regressor
## b. Feature augmentaions
## c. Additional dummy features : Formation 
## d. Featuere scaling
# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_feat = {}
## Feature Engeering 1 : With dummy variable from Formation
## Create dummy variables for Well Name, Formation, which may have geologic or geospatial information
train_dummy = pd.get_dummies(train[["Formation"]])
train_dummy.describe()
cols_dummy = train_dummy.columns.values
train[cols_dummy] = train_dummy[cols_dummy]
train_inp = train.drop(["Formation","Well Name",'FaciesLabels',"Depth"],axis =1)
X_fe1 = train_inp.drop(["Facies"],axis = 1).values
X_feat.update({"X_fe1" : X_fe1})
#
## Feature Engeering 2 : With dummy variable from Formation and feature augmentation
train["PE"] = train_raw["PE"].fillna(train_raw["PE"].median())
train_inp = train.drop(["Formation","Well Name",'FaciesLabels',"Depth"],axis =1)
X_fe1 = train_inp.drop(["Facies"],axis = 1).values
X_fe1_aug, padded_rows = augment_features(X_fe1, well, depth,N_neig = 1)
X_feat.update({"X_fe2" : X_fe1_aug})


## Feature Engeering 3 : With dummy variable from Formation and feature augmentation
## Fill Nan PE with mean
train["PE"] = train_raw["PE"].fillna(train_raw["PE"].mean())
train_inp = train.drop(["Formation","Well Name",'FaciesLabels',"Depth"],axis =1)
X_fe1 = train_inp.drop(["Facies"],axis = 1).values
X_fe1_aug, padded_rows = augment_features(X_fe1, well, depth,N_neig = 1)
X_feat.update({"X_fe3" : X_fe1_aug})


### Feature Engeering 4 : With dummy variable from Formation and feature augmentation
### Fill Nan PE with MPRRegressor
from sklearn.neural_network import MLPRegressor
reg = MLPRegressor()
DataImpAll = train_raw.drop(['Formation', 'Well Name', 'Depth'], axis=1).copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
train.loc[np.array(DataImpAll.PE.isnull()),"PE"] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))
train_inp = train.drop(["Formation","Well Name",'FaciesLabels',"Depth"],axis =1)
X_fe1 = train_inp.drop(["Facies"],axis = 1).values
X_fe1_aug, padded_rows = augment_features(X_fe1, well, depth,N_neig = 1)
X_feat.update({"X_fe4" : X_fe1_aug})


### Feature Engeering 5 : Drop dummy formation compared to Feature 4
X_fe1 = (train_inp.drop(["Facies"],axis = 1)).drop(cols_dummy,axis=1).values
X_fe1_aug, padded_rows = augment_features(X_fe1, well, depth,N_neig = 1)
X_feat.update({"X_fe5" : X_fe1_aug})

## Select which feature engineering for next model test
# Feature enginering Selection 
X_tr = X_feat["X_fe4"]
y = train["Facies"].values
## Feature Scaling
from sklearn import preprocessing
scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
X = scaler.transform(X_tr)


## Reading Test dataset and process the same way as trainning
test = pd.read_csv('01_raw_data/validation_data_nofacies.csv')
## Test data Check
print(test.count())  # Make sure no missing data in test
print("No. of Formation in test is " + str(len(test["Formation"].unique())))
## Dummy formation
test_dummy = pd.get_dummies(test[["Formation"]])
test_cols_dummy = test_dummy.columns.values
test[test_cols_dummy] = test_dummy[cols_dummy]
## Feature augmentaion
Well_test = test["Well Name"].values
Depth_test = test["Depth"].values
test_inp = test.drop(["Formation","Well Name","Depth"],axis =1)
test_fe = test_inp.values
test_aug,t_pad_row = augment_features(test_fe,Well_test,Depth_test)
## Scaling
X_test = scaler.transform(test_aug)


# Split Group
from sklearn.model_selection import LeavePGroupsOut
lpgo = LeavePGroupsOut(n_groups=2)
#split_no = lpgo.get_n_splits(X,y,wellgroups)
train_index=[]
val_index = []
for tr_i,val_i in lpgo.split(X, y, groups=train['Well Name'].values):
    hist_tr = np.histogram(y[tr_i], bins=np.arange(len(facies_labels)+1)+0.5)
    hist_val = np.histogram(y[val_i], bins=np.arange(len(facies_labels)+1)+0.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):    
        train_index.append(tr_i)
        val_index.append(val_i)
split_no = len(train_index)


from sklearn.multiclass import OneVsOneClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from scipy.signal import medfilt
seed = 123

param = {'alpha': 0.2,
 'colsamplebytree': 0.8,
 'gamma': 0.3,
 'learningrate': 0.05,
 'maxdepth': 8,
 'minchildweight': 1,
 'n_estimators': 200,
 'subsample': 0.7}


clf = XGBClassifier(
            learning_rate = param['learningrate'],
            n_estimators=param['n_estimators'],
            max_depth=param['maxdepth'],
            min_child_weight=param['minchildweight'],
            gamma = param['gamma'],
            subsample=param['subsample'],
            colsample_bytree=param['colsamplebytree'],
            reg_alpha = param['alpha'],
            nthread =4,
            seed = seed,
        ) 
svc_best = SVC(C = 10, gamma = 0.01, kernel = 'rbf')


## Always compare XGBOOST with SVC
f1 = np.zeros((split_no,1))
f1_svc = np.zeros((split_no,1))
for i in range(split_no):
    X_train = X[train_index[i]]
    Y_train = y[train_index[i]]
    X_val = X[val_index[i]]
    Y_val = y[val_index[i]]
    print(i)
    ### XGBOOST
    clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_val)
#    y_pred = medfilt(y_pred,kernel_size=5)
    f1[i] = f1_score(Y_val, y_pred, average='micro')  
    
    
    svc_best.fit(X_train,Y_train)
    Y_pred = svc_best.predict(X_val)
#    Y_pred = medfilt(Y_pred,kernel_size=5)
    f1_svc[i] = f1_score(Y_val, Y_pred, average='micro')  
    
print("XGBOOST score " + str(np.mean(f1)))
print("SVC score" + str(np.mean(f1_svc)))



## Predict for testing data
# Plot predicted labels
test_facies = clf.predict(X_test)
test["Facies"] = test_facies
test.to_csv("HoustonJ_sub2.csv")

make_facies_log_plot(
    test[test['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test[test['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
plt.show()


# HoustonJ's test notes
# 
# Facies Classification :
#  
# facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
#                  'WS', 'D','PS', 'BS']
# 
# Facies Description Label Adjacen facies
# 1 Nonmarine sandstone  SS 2
# 2 Nonmarine coarse siltstone CSiS 1,3
# 3 Nonmarine fine siltstone FSiS 2
# 4 Marine siltstone and shale SiSh 5
# 5 Mudstone MS 4,6
# 6 Wackestone WS 5,7,8
# 7 Dolomite D 6,8
# 8 Packstone-grainstone PS 6,7,9
# 9 Phylloid-algal bafflestone BS 7,8
# 
# Features : 'Facies', 'Formation', 'Well Name', 'Depth', 'GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS'
# 
# Pre-processing 1: 
# 1. Well 'ALEXANDER D', 'KIMZEY A' has missing PE, filled in with median values 
# 2. Map "Formation", "Well Name" into dummy features ; Effective features increase from 8 to 21
# 3. Robust normalization, dropping depth
# 
# 
# Pre-processing 2: 
# 1. Well 'ALEXANDER D', 'KIMZEY A' has missing PE, filled in with median values 
# 2. Map "Formation", "Well Name" into dummy features ; Effective features increase from 8 to 21
# 3. Feature augmentation
# 4. Robust normalization, dropping depth
# 
# Pre-processing 3: 
# 1. Well 'ALEXANDER D', 'KIMZEY A' has missing PE, filled in with mean values 
# 2. Map "Formation", "Well Name" into dummy features ; Effective features increase from 8 to 21
# 3. Feature augmentation
# 4. Robust normalization, dropping depth
# 
# Pre-processing 4: 
# 1. Well 'ALEXANDER D', 'KIMZEY A' has missing PE, filled in NN Regressor
# 2. Map "Formation", "Well Name" into dummy features ; Effective features increase from 8 to 21
# 3. Feature augmentation
# 4. Robust normalization, dropping depth
# 
# Pre-processing 5: 
# 1. Well 'ALEXANDER D', 'KIMZEY A' has missing PE, filled in with mean values 
# 
# 3. Feature augmentation
# 4. Robust normalization, dropping depth
# 
# Model_selection_pre:
# Split group by Well Name : 2 test 8 train
# 
# Test 1 : The pre-processing 1
# Radomly choose one slipt and run SVC vs XGBOOST
# f1 score  : SVC 0.486  <  XGBOOST 0.535
# Conclusion_pre : XGBOOST > SVC 
# 
# Feature Engineering Selection : 
# Test 2 : The pre-processing 1
# XGBOOST for all splits : 0.535
# Houmath Best : 0.563
# 
# Test3 : The feature augmentation 
# XGBOOST score 0.552620109649
# SVC score0.502307800369
# 
# Test 4 : The feature augmentation N_neig = 2
# XGBOOST score 0.544176923417
# SVC score0.489872101252
# 
# Test 5 : The pre-processing 2  
# XGBOOST score 0.557558000862
# SVC score0.499220019065
# 
# Test 6 : The pre-processing 3  
# XGBOOST score 0.557884804169
# SVC score0.500650895029
# 
# Test 7 : The pre-processing 3  y_pre = medfil size 5
# XGBOOST score 0.559944170153
# SVC score0.509190227257
# 
# Test 8 : The pre-processing 4  y_pre = medfil size 5
# XGBOOST score 0.566146182295
# SVC score0.507362308656
# 
# Test 9 : The pre-processing 5 Drop Formation dummy  y_pre = medfil size 5
# XGBOOST score 0.555870232144
# SVC score0.509423764916
# 
# Model Optimization : 
# Test 10 : The pre-processing 4  y_pre = medfil size 5
# Grid Search for XGBOOST parameters
# 
# Learning_rate 0.05, n_estimators : 200 ; Hope not overfitting
# XGBOOST score 0.563599674886
# SVC score0.510516447302
# 

# # Facies classification using machine learning techniques
# ##### Contact author: <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini</a>
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm is based on the use of random forests combined in one-vs-one multiclass strategy. In particular, we would like to study the effect of:
# - Robust feature normalization.
# - Well-based cross-validation routines.
# - Feature augmentation strategies.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier

from scipy.signal import medfilt


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../training_data.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    well_idx = data['Well Name'].values == w
    
    ax = plt.subplot(2, 4, w_idx+1)
    hist = np.histogram(y[well_idx], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# ## Feature augmentation
# Our guess is that facies do not abrutly change from a given depth layer to the next one. Therefore, we consider features at neighboring layers to be somehow correlated. To possibly exploit this fact, let us perform feature augmentation by:
# - Aggregating features at neighboring depths.
# - Computing feature spatial gradient.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# Parameters search grid
N_grid = [10, 30, 50, 100, 150, 200]
C_grid = ['gini', 'entropy']
param_grid = []
for N in N_grid:
    for C in C_grid:
        param_grid.append({'N':N, 'C':C})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion=param['C'],
                             class_weight='balanced', random_state=0), n_jobs=-1)
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('well_data_with_facies.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)


# # Facies classification using machine learning techniques
# 
# ### ISPL Team
# ##### Contact author: <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini</a>
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# This is a corrected version of [our previous submission try03](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try03.ipynb) built upon:
# - Part of the feature engineering work presented in [our previous submission](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb).
# - The gradient boosting classifier used in [SHandPR submission](https://github.com/seg/2016-ml-contest/blob/master/SHandPR/Face_classification_SHPR_GradientBoost.ipynb).
# 
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import GradientBoostingClassifier

from scipy.signal import medfilt


import sys, scipy, sklearn
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well)):
    X_well = X[well == w]
    X[well == w] = X_well[np.argsort(depth[well == w])]
    depth[well == w] = np.sort(depth[well == w])


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. Different strategies could be used. We simply substitute them with the average PE value.
# 

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# ## Feature augmentation
# In this submission, we propose a feature augmentation strategy based on:
# - Computing feature spatial gradient.
# - Computing higher order features and interaction terms.
# 

# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth):
    
    # Augment features
    padded_rows = []
    X_aug = np.zeros((X.shape[0], X.shape[1]*2))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X[w_idx, :], X_aug_grad), axis=1)
        padded_rows.append(w_idx[-1])
        
    # Find padded rows
    #padded_rows = np.where(~X_aug[:, 7:].any(axis=1))[0]
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# Add higher degree terms and interaction terms to the model
deg = 2
poly = preprocessing.PolynomialFeatures(deg, interaction_only=False)
X_aug = poly.fit_transform(X_aug)
X_aug = X_aug[:,1:]


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# # Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
# Loss_grid = ['exponential'] # ['deviance', 'exponential']
# N_grid = [100] # [100]
# M_grid = [10] # [5, 10, 15]
# S_grid = [25] # [10, 25, 50, 75]
# L_grid = [5] # [2, 3, 4, 5, 10, 25]
# R_grid = [.1] # [.05, .1, .5]
# Sub_grid = [1] # [0.5, 0.75, 1]
# MED_grid = [1] # [0, 1]
# param_grid = []
# for N in N_grid:
#     for M in M_grid:
#         for S in S_grid:
#             for L in L_grid:
#                 for R in R_grid:
#                     for Sub in Sub_grid:
#                         for MED in MED_grid:
#                             for Loss in Loss_grid:
#                                 param_grid.append({'N':N, 'M':M, 'S':S, 'L':L,
#                                                    'R':R, 'Sub':Sub, 'MED': MED, 'Loss':Loss})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, clf):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    if param['MED']:
        for w in np.unique(well_v):
            y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
# score_param = []
# for param in param_grid:
    
#     # For each data split
#     score_split = []
#     for split in split_list:
    
#         # Remove padded rows
#         split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
#         # Select training and validation data from current split
#         X_tr = X_aug[split_train_no_pad, :]
#         X_v = X_aug[split['val'], :]
#         y_tr = y[split_train_no_pad]
#         y_v = y[split['val']]
        
#         # Select well labels for validation data
#         well_v = well[split['val']]

#         # Train and test
#         y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
#         # Score
#         score = f1_score(y_v, y_v_hat, average='micro')
#         score_split.append(score)

#     # Average score for this param
#     score_param.append(np.mean(score_split))
          
# # Best set of parameters
# best_idx = np.argmax(score_param)
# param_best = param_grid[best_idx]
# score_best = score_param[best_idx]
# print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Best params from team's optimization
param = {'MED': 1, 'S': 25, 'R': 0.1, 'Sub': 1, 'Loss': 'exponential', 'M': 10, 'L': 5, 'N': 100}


# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# # Prepare training data
# X_tr = X
# y_tr = y

# # Augment features
# X_tr, padded_rows = augment_features(X_tr, well, depth)

# # Removed padded rows
# X_tr = np.delete(X_tr, padded_rows, axis=0)
# y_tr = np.delete(y_tr, padded_rows, axis=0)


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values


# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well_ts)):
    X_ts_well = X_ts[well_ts == w]
    X_ts[well_ts == w] = X_ts_well[np.argsort(depth_ts[well_ts == w])]
    depth_ts[well_ts == w] = np.sort(depth_ts[well_ts == w])

# Augment features
# X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


y_pred = []
print('.' * 100)
for seed in range(100):
    np.random.seed(seed)

    # Make training data.
    X_train, padded_rows = augment_features(X, well, depth)
    y_train = y
    X_train = np.delete(X_train, padded_rows, axis=0)
    y_train = np.delete(y_train, padded_rows, axis=0) 

    # Train classifier
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss=param['Loss'],
                                        n_estimators=param['N'],
                                        learning_rate=param['R'], 
                                        max_features=param['M'],
                                        min_samples_leaf=param['L'],
                                        min_samples_split=param['S'],
                                        random_state=seed,
                                        subsample=param['Sub'],
                                        max_leaf_nodes=None, 
                                        verbose=0), n_jobs=-1)

    # Make blind data.
    X_test, _ = augment_features(X_ts, well_ts, depth_ts)

    # Train and test.
    y_ts_hat = train_and_test(X_train, y_train, X_test, well_ts, clf)
    
    # Collect result.
    y_pred.append(y_ts_hat)
    print('|', end='')
    
np.save('ispl_100_realizations.npy', y_pred)


# Predict test labels
# y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
# test_data['Facies'] = y_ts_hat
# test_data.to_csv('well_data_with_facies_try03_v2.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)


# # Facies classification using machine learning techniques
# 
# ### ISPL Team
# ##### Contact author: <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini</a>
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm builds upon:
# - Part of the feature engineering work presented in [our previous submission](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try03_v2.ipynb).
# - The gradient boosting classifier used in [SHandPR submission](https://github.com/seg/2016-ml-contest/blob/master/SHandPR/Face_classification_SHPR_GradientBoost.ipynb).
# 
# Differently from our previous attempt, we try working with a layered classification approach.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier

from scipy.signal import medfilt

from xgboost import XGBClassifier


import sys, scipy, sklearn, xgboost
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)
print('XGBoost: ' + xgboost.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well)):
    X_well = X[well == w]
    X[well == w] = X_well[np.argsort(depth[well == w])]
    depth[well == w] = np.sort(depth[well == w])


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. Different strategies could be used. We simply substitute them with the average PE value.
# 

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# ## Feature augmentation
# In this submission, we propose a feature augmentation strategy based on:
# - Computing feature spatial gradient.
# - Computing higher order features and interaction terms.
# 

# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient in both directions
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad_aux = X_diff / d_diff    
    
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad_aux, np.zeros((1, X_grad_aux.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth):
    
    # Augment features
    padded_rows = []
    X_aug = np.zeros((X.shape[0], X.shape[1]*2))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X[w_idx, :], X_aug_grad), axis=1)
        padded_rows.append(w_idx[-1])
        
    # Find padded rows
    #padded_rows = np.where(~X_aug[:, 7:].any(axis=1))[0]
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# Add higher degree terms and interaction terms to the model
deg = 2
poly = preprocessing.PolynomialFeatures(deg, interaction_only=False)
X_aug = poly.fit_transform(X_aug)
X_aug = X_aug[:,1:]


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [100] # [50, 100, 150, 200]
D_grid = [3] # [2, 3, 5, 7]
W_grid = [8] # [6, 7, 8, 10, 12]
R_grid = [.12] # [.10, .11, .12, .13, .14]
T_grid = [.9] # [.8, .9, .10]
MED_grid = [1] # [0, 1]
param_grid = []
for N in N_grid:
    for D in D_grid:
        for W in W_grid:
            for R in R_grid:
                for T in T_grid:
                    for MED in MED_grid:
                        param_grid.append({'N':N, 'D':D, 'W':W, 'R':R, 'T':T, 'MED':MED})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Prepare layers
    idx_2a = np.where(y_tr < 4)[0]
    idx_2b = np.where(y_tr >= 4)[0]
    y_tr_1 = y_tr.copy()
    y_tr_1[idx_2a] = 0
    y_tr_1[idx_2b] = 1
    
    # Define classifiers
    clf_2a = XGBClassifier(learning_rate=param['R'], max_depth=param['D'], min_child_weight=param['W'],
                           n_estimators=param['N'], seed=0, colsample_bytree=param['T'])
    clf_2b = XGBClassifier(learning_rate=param['R'], max_depth=param['D'], min_child_weight=param['W'],
                           n_estimators=param['N'], seed=0, colsample_bytree=param['T'])

    # Train layers
    clf_2a.fit(X_tr[idx_2a], y_tr[idx_2a])
    clf_2b.fit(X_tr[idx_2b], y_tr[idx_2b])
    
    # Test classifier
    y_v_hat_1 = np.zeros((X_v.shape[0]))
    y_v_hat_1[np.where(X_v[:, 5] <= np.mean(X_v[:, 5]))] = 0
    y_v_hat_1[np.where(X_v[:, 5] > np.mean(X_v[:, 5]))] = 1
    idx_2a_hat = np.where(y_v_hat_1 == 0)[0]
    idx_2b_hat = np.where(y_v_hat_1 == 1)[0]
    y_v_hat_2a = clf_2a.predict(X_v[idx_2a_hat])
    y_v_hat_2b = clf_2b.predict(X_v[idx_2b_hat])
    y_v_hat = y_v_hat_1.copy()
    y_v_hat[idx_2a_hat] = y_v_hat_2a
    y_v_hat[idx_2b_hat] = y_v_hat_2b
    
    # Clean isolated facies for each well
    if param['MED']:
        for w in np.unique(well_v):
            y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
        # Score per classifier
        y_v_1 = y_v.copy()
        y_v_1[np.where(y_v < 4)[0]] = 0
        y_v_1[np.where(y_v >= 4)[0]] = 1
        y_v_hat_1 = y_v_hat.copy()
        y_v_hat_1[np.where(y_v_hat_1 < 4)[0]] = 0
        y_v_hat_1[np.where(y_v_hat_1 >= 4)[0]] = 1
        score_1 = f1_score(y_v_1, y_v_hat_1, average='micro')
        score_2a = f1_score(y_v[np.where(y_v < 4)[0]], y_v_hat[np.where(y_v < 4)[0]], average='micro')
        score_2b = f1_score(y_v[np.where(y_v >= 4)[0]], y_v_hat[np.where(y_v >= 4)[0]], average='micro')
        print('   F1 score split = %.3f (%.3f, %.3f, %.3f)' % (score_split[-1], score_1, score_2a, score_2b)) 
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f (std %.3f) %s' % (score_param[-1], np.std(score_split), param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0)


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well_ts)):
    X_ts_well = X_ts[well_ts == w]
    X_ts[well_ts == w] = X_ts_well[np.argsort(depth_ts[well_ts == w])]
    depth_ts[well_ts == w] = np.sort(depth_ts[well_ts == w])

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('well_data_with_facies_try04.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)


# # Facies classification using machine learning techniques
# 
# ### ISPL Team
# ##### Contact author: <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini</a>
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm builds upon:
# - Part of the feature engineering work presented in [our previous submission](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb).
# - The gradient boosting classifier used in [SHandPR submission](https://github.com/seg/2016-ml-contest/blob/master/SHandPR/Face_classification_SHPR_GradientBoost.ipynb).
# 
# Differently from our previous attempt, we inlcuded in this solution a set of higher-order features.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import GradientBoostingClassifier

from scipy.signal import medfilt


import sys, scipy, sklearn
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well)):
    X_well = X[well == w]
    X[well == w] = X_well[np.argsort(depth[well == w])]
    depth[well == w] = np.sort(depth[well == w])


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. Different strategies could be used. We simply substitute them with the average PE value.
# 

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# ## Feature augmentation
# In this submission, we propose a feature augmentation strategy based on:
# - Computing feature spatial gradient.
# - Computing higher order features and interaction terms.
# 

# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=0):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*2))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X[w_idx, :], X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# Add higher degree terms and interaction terms to the model
deg = 2
poly = preprocessing.PolynomialFeatures(deg, interaction_only=False)
X_aug = poly.fit_transform(X_aug)
X_aug = X_aug[:,1:]


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
Loss_grid = ['exponential'] # ['deviance', 'exponential']
N_grid = [100] # [100]
M_grid = [10] # [5, 10, 15]
S_grid = [25] # [10, 25, 50, 75]
L_grid = [5] # [2, 3, 4, 5, 10, 25]
R_grid = [.1] # [.05, .1, .5]
Sub_grid = [1] # [0.5, 0.75, 1]
MED_grid = [1] # [0, 1]
param_grid = []
for N in N_grid:
    for M in M_grid:
        for S in S_grid:
            for L in L_grid:
                for R in R_grid:
                    for Sub in Sub_grid:
                        for MED in MED_grid:
                            for Loss in Loss_grid:
                                param_grid.append({'N':N, 'M':M, 'S':S, 'L':L,
                                                   'R':R, 'Sub':Sub, 'MED': MED, 'Loss':Loss})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss=param['Loss'],
                                        n_estimators=param['N'],
                                        learning_rate=param['R'], 
                                        max_features=param['M'],
                                        min_samples_leaf=param['L'],
                                        min_samples_split=param['S'],
                                        random_state=0,
                                        subsample=param['Sub'],
                                        max_leaf_nodes=None, 
                                        verbose=0), n_jobs=-1)
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    if param['MED']:
        for w in np.unique(well_v):
            y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)

    # Average score for this param
    score_param.append(np.mean(score_split))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0)


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well_ts)):
    X_ts_well = X_ts[well_ts == w]
    X_ts[well_ts == w] = X_ts_well[np.argsort(depth_ts[well_ts == w])]
    depth_ts[well_ts == w] = np.sort(depth_ts[well_ts == w])

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('well_data_with_facies_try03.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)


# # Facies classification using Machine Learning #
# ## LA Team Submission 5 ## 
# ### _[Lukas Mosser](https://at.linkedin.com/in/lukas-mosser-9948b32b/en), [Alfredo De la Fuente](https://pe.linkedin.com/in/alfredodelafuenteb)_ ####
# 

# In this approach for solving the facies classfication problem ( https://github.com/seg/2016-ml-contest. ) we will explore the following statregies:
# - Features Exploration: based on [Paolo Bestagini's work](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb), we will consider imputation, normalization and augmentation routines for the initial features.
# - Model tuning: 
# 

# ## Libraries
# 
# We will need to install the following libraries and packages.
# 

get_ipython().run_cell_magic('sh', '', 'pip install pandas\npip install scikit-learn\npip install tpot')


from __future__ import print_function
import numpy as np
get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , StratifiedKFold
from classification_utilities import display_cm, display_adj_cm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import medfilt


# ## Data Preprocessing
# 

#Load Data
data = pd.read_csv('../facies_vectors.csv')

# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# Store features and labels
X = data[feature_names].values 
y = data['Facies'].values 

# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values

# Fill 'PE' missing values with mean
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# We procceed to run [Paolo Bestagini's routine](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb) to include a small window of values to acount for the spatial component in the log analysis, as well as the gradient information with respect to depth. This will be our prepared training dataset.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
    
        
def preprocess():
    
    # Preprocess data to use in model
    X_train_aux = []
    X_test_aux = []
    y_train_aux = []
    y_test_aux = []
    
    # For each data split
    split = split_list[5]
        
    # Remove padded rows
    split_train_no_pad = np.setdiff1d(split['train'], padded_rows)

    # Select training and validation data from current split
    X_tr = X_aug[split_train_no_pad, :]
    X_v = X_aug[split['val'], :]
    y_tr = y[split_train_no_pad]
    y_v = y[split['val']]

    # Select well labels for validation data
    well_v = well[split['val']]

    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
        
    X_train_aux.append( X_tr )
    X_test_aux.append( X_v )
    y_train_aux.append( y_tr )
    y_test_aux.append (  y_v )
    
    X_train = np.concatenate( X_train_aux )
    X_test = np.concatenate ( X_test_aux )
    y_train = np.concatenate ( y_train_aux )
    y_test = np.concatenate ( y_test_aux )
    
    return X_train , X_test , y_train , y_test 


# ## Data Analysis
# 
# In this section we will run a Cross Validation routine 
# 

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = preprocess()

tpot = TPOTClassifier(generations=5, population_size=20, 
                      verbosity=2,max_eval_time_mins=20,
                      max_time_mins=100,scoring='f1_micro',
                      random_state = 17)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('FinalPipeline.py')


from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    #clf = make_pipeline(make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)), XGBClassifier(learning_rate=0.73, max_depth=10, min_child_weight=10, n_estimators=500, subsample=0.27))
    #clf =  make_pipeline( KNeighborsClassifier(n_neighbors=5, weights="distance") ) 
    #clf = make_pipeline(MaxAbsScaler(),make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),ExtraTreesClassifier(criterion="entropy", max_features=0.0001, n_estimators=500))
    # * clf = make_pipeline( make_union(VotingClassifier([("est", BernoulliNB(alpha=60.0, binarize=0.26, fit_prior=True))]), FunctionTransformer(lambda X: X)),RandomForestClassifier(n_estimators=500))
    clf = make_pipeline ( XGBClassifier(learning_rate=0.12, max_depth=3, min_child_weight=10, n_estimators=150, seed = 17, colsample_bytree = 0.9) )
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# ## Prediction
# 

#Load testing data
test_data = pd.read_csv('../validation_data_nofacies.csv')

# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 

# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)

# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts)

# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('Prediction_XX_Final.csv')





# Like Ar4 submission but with KNeighborsClassifier classifier
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']=(20.0,10.0)
inline_rc = dict(mpl.rcParams)

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt

from pandas.tools.plotting import scatter_matrix

import matplotlib.colors as colors

import datetime

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from classification_utilities import display_cm, display_adj_cm
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_files

from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack

#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

seed = 123
np.random.seed(seed)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# Load data from file
data = pd.read_csv('../facies_vectors.csv')


X = data[feature_names].values
y = data['Facies'].values


well = data['Well Name'].values
depth = data['Depth'].values


# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


reg = RandomForestRegressor(max_features='sqrt', n_estimators=50)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [20]  
W_grid = ['distance']  
A_grid = ['kd_tree']
L_grid = [30]  
M_grid = ['minkowski']
P_grid = [1]  
param_grid = []
for N in N_grid:
    for W in W_grid:
        for A in A_grid:
            for L in L_grid: 
                for M in M_grid:
                    for P in P_grid:
                        param_grid.append({'N':N, 'W':W, 'A':A,'L':L,'M':M,'P':P})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier  
    clf = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=param['N'],
                                                  weights=param['W'],
                                                  algorithm=param['A'],
                                                  leaf_size=param['L'],
                                                  metric=param['M'],
                                                  p=param['P']), 
                             n_jobs=-1)

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:

    now = datetime.datetime.now().time()
    print('%s' % now)   

    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# # Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 



# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)


from mpl_toolkits.axes_grid1 import make_axes_locatable


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('SHPR_NearestNeighbour_predicted_facies.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# Like Ar4 submission but with RandomForestClassifier classifier
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']=(20.0,10.0)
inline_rc = dict(mpl.rcParams)

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt

from pandas.tools.plotting import scatter_matrix

import matplotlib.colors as colors

import datetime

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from classification_utilities import display_cm, display_adj_cm
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_files

from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack

#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.neighbors import KNeighborsClassifier

seed = 123
np.random.seed(seed)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# Load data from file
data = pd.read_csv('../facies_vectors.csv')


X = data[feature_names].values
y = data['Facies'].values


well = data['Well Name'].values
depth = data['Depth'].values


# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


reg = RandomForestRegressor(max_features='sqrt', n_estimators=50)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [200]  
CR_grid = ['gini']  
MF_grid = ['auto']
MD_grid = [None]  
MSS_grid = [7]
MSL_grid = [1]  
MWFL_grid = [0]  
MLN_grid = [None]  
MIS_grid = [1e-7]  
B_grid = [True]  
O_grid = [False]  
R_grid = [None]  
W_grid = [False]  
CW_grid = [None]  
param_grid = []
for N in N_grid:
    for CR in CR_grid:
        for MF in MF_grid:
            for MD in MD_grid: 
                for MSS in MSS_grid:
                    for MSL in MSL_grid:
                        for MWFL in MWFL_grid:
                            for MLN in MLN_grid:
                                for MIS in MIS_grid:
                                    for B in B_grid:
                                        for O in O_grid:
                                            for R in R_grid:
                                                for W in W_grid:
                                                    for CW in CW_grid:
                                                        param_grid.append({'N':N, 
                                                                           'CR':CR, 
                                                                           'MF':MF,
                                                                           'MD':MD,
                                                                           'MSS':MSS,
                                                                           'MSL':MSL,
                                                                           'MWFL':MWFL,
                                                                           'MLN':MLN,
                                                                           'MIS':MIS,
                                                                           'B':B,
                                                                           'O':O,
                                                                           'R':R,
                                                                           'W':W,
                                                                           'CW':CW})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier  
    clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'],
                                                    criterion=param['CR'],
                                                    max_features=param['MF'],
                                                    max_depth=param['MD'],
                                                    min_samples_split=param['MSS'],
                                                    min_samples_leaf=param['MSL'],
                                                    min_weight_fraction_leaf=param['MWFL'],
                                                    max_leaf_nodes=param['MLN'],
                                                    min_impurity_split=param['MIS'],
                                                    bootstrap=param['B'],
                                                    oob_score=param['O'],
                                                    random_state=param['R'],
                                                    verbose=0,
                                                    warm_start=param['W'],
                                                    class_weight=param['CW']
                                                    ),
                             n_jobs=-1)

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:

    now = datetime.datetime.now().time()
    print('%s' % now)   

    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# # Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 



# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)


from mpl_toolkits.axes_grid1 import make_axes_locatable


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('SHPR_RandomForest_predicted_facies.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# # Facies classification using machine learning techniques
# #### Contact authors: Steve Hall & Priyanka Raghavan
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm builds upon:
#    * Our previous submission
#    * Which builds on a previous submission by Alan Richardson
#    * Which builds on a previous submission by Paolo Bestagini
# 
# In this attempt we have tweaked the parameters of the GradientBoostingClassifier for what we hope is a (small) improvement.
# 

# Like Ar4 submission but with GradientBoosting classifier
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']=(20.0,10.0)
inline_rc = dict(mpl.rcParams)

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt

from pandas.tools.plotting import scatter_matrix

import matplotlib.colors as colors

import datetime

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from classification_utilities import display_cm, display_adj_cm
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_files

from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack

from sklearn.ensemble import GradientBoostingClassifier

seed = 123
np.random.seed(seed)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# Load data from file
data = pd.read_csv('../facies_vectors.csv')


X = data[feature_names].values
y = data['Facies'].values


well = data['Well Name'].values
depth = data['Depth'].values


# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


reg = RandomForestRegressor(max_features='sqrt', n_estimators=50)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
L_grid = ['exponential']  
LR_grid = [0.1]  
N_grid = [200]  
MD_grid = [2]  
MSS_grid = [25]  
MSL_grid = [5]
MF_grid = [None]
param_grid = []
for L in L_grid: 
    for LR in LR_grid: 
        for N in N_grid:
            for MD in MD_grid:
                for MSS in MSS_grid:
                    for MSL in MSL_grid:
                        for MF in MF_grid:
                            param_grid.append({'L':L, 'LR':LR, 'N':N, 'MD':MD, 'MSS':MSS, 'MSL':MSL, 'MF':MF})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier  
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss=param['L'],
                                                        learning_rate=param['LR'], 
                                                        n_estimators=param['N'], 
                                                        max_depth=param['MD'],
                                                        min_samples_split=param['MSS'],
                                                        min_samples_leaf=param['MSL'],
                                                        max_features= param['MF'],
                                                        max_leaf_nodes=None, 
                                                        random_state=seed, 
                                                        verbose=1), n_jobs=-1)

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:

    now = datetime.datetime.now().time()
    print('%s' % now)   

    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# # Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 



# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)


from mpl_toolkits.axes_grid1 import make_axes_locatable


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('SHPR_GradientBoost_predicted_facies.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# Like Ar4 submission but with LogisticRegression classifier
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']=(20.0,10.0)
inline_rc = dict(mpl.rcParams)

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt

from pandas.tools.plotting import scatter_matrix

import matplotlib.colors as colors

import datetime

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from classification_utilities import display_cm, display_adj_cm
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_files

from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack

#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

seed = 123
np.random.seed(seed)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# Load data from file
data = pd.read_csv('../facies_vectors.csv')


X = data[feature_names].values
y = data['Facies'].values


well = data['Well Name'].values
depth = data['Depth'].values


# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


reg = RandomForestRegressor(max_features='sqrt', n_estimators=50)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
P_grid = ['l2']  
D_grid = [False]  
C_grid = [1e6]  
FI_grid = [True]  
IS_grid = [1]  
CW_grid = [None]
MI_grid = [100]
S_grid = ['newton-cg']
T_grid = [1e-04]
MC_grid = ['multinomial']
W_grid = [False]
param_grid = []
for P in P_grid: 
    for D in D_grid:
        for C in C_grid:
            for FI in FI_grid:
                for IS in IS_grid:
                    for CW in CW_grid:
                        for MI in MI_grid:
                            for S in S_grid:
                                for T in T_grid:
                                    for MC in MC_grid:
                                        for W in W_grid:
                                            param_grid.append({'P':P, 'D':D, 'C':C, 'FI':FI, 'IS':IS, 'CW':CW, 'MI':MI, 'S':S, 'T':T, 'MC':MC, 'W':W})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier  
    clf = OneVsOneClassifier(LogisticRegression(penalty=param['P'],
                                                        dual=param['D'], 
                                                        C=param['C'],
                                                        fit_intercept=param['FI'],
                                                        intercept_scaling=param['IS'],
                                                        class_weight=param['CW'],
                                                        max_iter=param['MI'],
                                                        random_state=seed,
                                                        solver=param['S'], 
                                                        tol=param['T'], 
                                                        multi_class=param['MC'], 
                                                        warm_start=param['W'], 
                                                        verbose=0), 
                             n_jobs=-1)

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:

    now = datetime.datetime.now().time()
    print('%s' % now)   

    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# # Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 



# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)


from mpl_toolkits.axes_grid1 import make_axes_locatable


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('SHPR_LogisticRegression_predicted_facies.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# # 1.0 - Facies Classification Using RandomForestClassifier.
# 
# ## Chris Esprey - https://www.linkedin.com/in/christopher-esprey-beng-8aab1078?trk=nav_responsive_tab_profile
# 
# I have generated two main feature types, namely:
# 
# - The absolute difference between each feature for all feature rows.
# - The difference between each sample and the mean and standard deviation of each facies.
# 
# I then threw this at a RandomForestClassifier. 
# 
# Possible future improvements:
# - Perform Univariate feature selection to hone in on the best features
# - Try out other classifiers e.g. gradient boost, SVM etc. 
# - Use an ensemble of algorithms for classification 
# 

get_ipython().magic('matplotlib notebook')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm


filename = 'training_data.csv'
training_data = pd.read_csv(filename)


# # 2.0 - Feature Generation 
# 

## Create a difference vector for each feature e.g. x1-x2, x1-x3... x2-x3...

# order features in depth.

feature_vectors = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
feature_vectors = feature_vectors[['Depth','GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]

def difference_vector(feature_vectors):
    length = len(feature_vectors['Depth'])
    df_temp = np.zeros((25, length))
                          
    for i in range(0,int(len(feature_vectors['Depth']))):
                       
        vector_i = feature_vectors.iloc[i,:]
        vector_i = vector_i[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
        for j, value_j in enumerate(vector_i):
            for k, value_k in enumerate(vector_i): 
                differ_j_k = value_j - value_k          
                df_temp[5*j+k, i] = np.abs(differ_j_k)
                
    return df_temp

def diff_vec2frame(feature_vectors, df_temp):
    
    heads = feature_vectors.columns[1::]
    for i in range(0,5):
        string_i = heads[i]
        for j in range(0,5):
            string_j = heads[j]
            col_head = 'diff'+string_i+string_j
            
            df = pd.Series(df_temp[5*i+j, :])
            feature_vectors[col_head] = df
    return feature_vectors
            
df_diff = difference_vector(feature_vectors)    
feature_vectors = diff_vec2frame(feature_vectors, df_diff)

# Drop duplicated columns and column of zeros
feature_vectors = feature_vectors.T.drop_duplicates().T   
feature_vectors.drop('diffGRGR', axis = 1, inplace = True)


# Add Facies column back into features vector

feature_vectors['Facies'] = training_data['Facies']

# # group by facies, take statistics of each facies e.g. mean, std. Take sample difference of each row with 

def facies_stats(feature_vectors):
    facies_labels = np.sort(feature_vectors['Facies'].unique())
    frame_mean = pd.DataFrame()
    frame_std = pd.DataFrame()
    for i, value in enumerate(facies_labels):
        facies_subframe = feature_vectors[feature_vectors['Facies']==value]
        subframe_mean = facies_subframe.mean()
        subframe_std = facies_subframe.std()
        
        frame_mean[str(value)] = subframe_mean
        frame_std[str(value)] = subframe_std
    
    return frame_mean.T, frame_std.T

def feature_stat_diff(feature_vectors, frame_mean, frame_std):
    
    feature_vec_origin = feature_vectors[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
    
    for i, column in enumerate(feature_vec_origin):
        
        feature_column = feature_vec_origin[column]
        stat_column_mean = frame_mean[column]
        stat_column_std = frame_std[column]
        
        for j in range(0,9):
            
            stat_column_mean_facie = stat_column_mean[j]
            stat_column_std_facie = stat_column_std[j]
            
            feature_vectors[column + '_mean_diff_facies' + str(j)] = feature_column-stat_column_mean_facie
            feature_vectors[column + '_std_diff_facies' + str(j)] = feature_column-stat_column_std_facie
    return feature_vectors
             
frame_mean, frame_std = facies_stats(feature_vectors)  
feature_vectors = feature_stat_diff(feature_vectors, frame_mean, frame_std)


# # 3.0 - Generate plots of each feature
# 

# A = feature_vectors.sort_values(by='Facies')
# A.reset_index(drop=True).plot(subplots=True, style='b', figsize = [12, 400])


# # 4.0 - Train model using RandomForestClassifier
# 

df = feature_vectors
predictors = feature_vectors.columns
predictors = list(predictors.drop('Facies'))
correct_facies_labels = df['Facies'].values
# Scale features
df = df[predictors]

scaler = preprocessing.StandardScaler().fit(df)
scaled_features = scaler.transform(df)

# Train test split:

X_train, X_test, y_train, y_test = train_test_split(scaled_features,  correct_facies_labels, test_size=0.2, random_state=0)
alg = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=8, min_samples_leaf=3, max_features= None)
alg.fit(X_train, y_train)

predicted_random_forest = alg.predict(X_test)


facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
result = predicted_random_forest
conf = confusion_matrix(y_test, result)
display_cm(conf, facies_labels, hide_zeros=True, display_metrics = True)

def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf))
    return acc

print(accuracy(conf))

adjacent_facies = np.array([[1], [0,2], [1], [4], [3,5], [4,6,7], [5,7], [5,6,8], [6,7]])

def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))

print(accuracy_adjacent(conf, adjacent_facies))


# # 5.0 - Predict on test data
# 

# read in Test data

filename = 'validation_data_nofacies.csv'
test_data = pd.read_csv(filename)


# Reproduce feature generation

feature_vectors_test = test_data.drop(['Formation', 'Well Name'], axis=1)
feature_vectors_test = feature_vectors_test[['Depth','GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]

df_diff_test = difference_vector(feature_vectors_test)    
feature_vectors_test = diff_vec2frame(feature_vectors_test, df_diff_test)

# Drop duplicated columns and column of zeros

feature_vectors_test = feature_vectors_test.T.drop_duplicates().T   
feature_vectors_test.drop('diffGRGR', axis = 1, inplace = True)

# Create statistical feature differences using preivously caluclated mean and std values from train data.

feature_vectors_test = feature_stat_diff(feature_vectors_test, frame_mean, frame_std)
feature_vectors_test = feature_vectors_test[predictors]
scaler = preprocessing.StandardScaler().fit(feature_vectors_test)
scaled_features = scaler.transform(feature_vectors_test)

predicted_random_forest = alg.predict(scaled_features)



predicted_random_forest
test_data['Facies'] = predicted_random_forest
test_data.to_csv('test_data_prediction_CE.csv')





# ## Houmath's excellent notebook + median filtering
# 

from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']=(20.0,10.0)
inline_rc = dict(mpl.rcParams)

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt

from pandas.tools.plotting import scatter_matrix

import matplotlib.colors as colors

import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from classification_utilities import display_cm, display_adj_cm
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_files

from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack
from sklearn import metrics
from scipy.spatial.distance import correlation
from scipy.signal import medfilt, gaussian
seed = 123
np.random.seed(seed)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# Load data from file
data = pd.read_csv('facies_vectors.csv')


X = data[feature_names].values
y = data['Facies'].values


well = data['Well Name'].values
depth = data['Depth'].values


# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


reg = RandomForestRegressor(max_features='sqrt', n_estimators=50)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad

# Feature gradient computation function
def augment_features_median(X, dist=3):
    #print(X)
    X_grad = None
    X_out = np.zeros((X.shape[0], 7))
    for i in range(X.shape[1]):
        X_out[:, i] = medfilt(X[:, i], dist)
    # Compensate for last missing value
    #X_out = np.array(X_out)
    #X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    print(X_out.shape)
    return X_out


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    print(X.shape)
    # Augment features
    X_aug = np.zeros((X.shape[0], 42))#X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        
        X_aug_cov = augment_features_median(X[w_idx, :])
        X_aug_grad_med = augment_features_median(X_aug_grad, dist=7)
        print(X_aug_win.shape, X_aug_grad.shape, X_aug_cov.shape)
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad, X_aug_cov, X_aug_grad_med), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
md_grid = [3]
mcw_grid = [1]
gamma_grid = [0.3]  
ss_grid = [0.7] 
csb_grid = [0.8]
alpha_grid =[0.2]
lr_grid = [0.05]
ne_grid = [200]
param_grid = []
for N in md_grid:
    for M in mcw_grid:
        for S in gamma_grid:
            for L in ss_grid:
                for K in csb_grid:
                    for P in alpha_grid:
                        for R in lr_grid:
                            for E in ne_grid:
                                param_grid.append({'maxdepth':N, 
                                                   'minchildweight':M, 
                                                   'gamma':S, 
                                                   'subsample':L,
                                                   'colsamplebytree':K,
                                                   'alpha':P,
                                                   'learningrate':R,
                                                   'n_estimators':E})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v,well_v):
    
    #Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)

    print("lel")
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    clf = OneVsOneClassifier(XGBClassifier(
            learning_rate = param['learningrate'],
            n_estimators=param['n_estimators'],
            max_depth=param['maxdepth'],
            min_child_weight=param['minchildweight'],
            gamma = param['gamma'],
            subsample=param['subsample'],
            colsample_bytree=param['colsamplebytree'],
            reg_alpha = param['alpha'],
            nthread =1,
            seed = seed, silent=True
        ) , n_jobs=1)
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v)
        
        score = f1_score(y_v, y_v_hat, average='micro')
        print("cur score", score)
        score_split.append(score)
        print("running mean", np.mean(score_split))
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


print('Min Max', np.min(score_split), np.max(score_split))


for param in param_grid:
    
    clf = OneVsOneClassifier(XGBClassifier(
            learning_rate = param['learningrate'],
            n_estimators=param['n_estimators'],
            max_depth=param['maxdepth'],
            min_child_weight=param['minchildweight'],
            gamma = param['gamma'],
            subsample=param['subsample'],
            colsample_bytree=param['colsamplebytree'],
            reg_alpha = param['alpha'],
            nthread =4,
            seed = seed,
        ) , n_jobs=-1)


clf


# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('Prediction_houmath_edit.csv')


test_data





def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)


from mpl_toolkits.axes_grid1 import make_axes_locatable


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# #### Contest entry by Wouter Kimman 
# 
# 
# Strategy: 
# ----------------------------------------------
# 
# 

from numpy.fft import rfft
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py


import pandas as pd
import timeit
from sqlalchemy.sql import text
from sklearn import tree
from sklearn.model_selection import LeavePGroupsOut

#from sklearn import cross_validation
#from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
#import sherlock.filesystem as sfs
#import sherlock.database as sdb

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from scipy import stats


#filename = 'training_data.csv'
filename = 'facies_vectors.csv'
training_data0 = pd.read_csv(filename)



def magic(df):
    df1=df.copy()
    b, a = signal.butter(2, 0.2, btype='high', analog=False)
    feats0=['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
    #feats01=['GR','ILD_log10','DeltaPHI','PHIND']
    #feats01=['DeltaPHI']
    #feats01=['GR','DeltaPHI','PHIND']
    feats01=['GR',]
    feats02=['PHIND']
    #feats02=[]
    for ii in feats0:
        df1[ii]=df[ii]
        name1=ii + '_1'
        name2=ii + '_2'
        name3=ii + '_3'
        name4=ii + '_4'
        name5=ii + '_5'
        name6=ii + '_6'
        name7=ii + '_7'
        name8=ii + '_8'
        name9=ii + '_9'
        xx1 = list(df[ii])
        xx_mf= signal.medfilt(xx1,9)
        x_min1=np.roll(xx_mf, 1)
        x_min2=np.roll(xx_mf, -1)
        x_min3=np.roll(xx_mf, 3)
        x_min4=np.roll(xx_mf, 4)
        xx1a=xx1-np.mean(xx1)
        xx_fil = signal.filtfilt(b, a, xx1)        
        xx_grad=np.gradient(xx1a) 
        x_min5=np.roll(xx_grad, 3)
        #df1[name4]=xx_mf
        if ii in feats01: 
            df1[name1]=x_min3
            df1[name2]=xx_fil
            df1[name3]=xx_grad
            df1[name4]=xx_mf 
            df1[name5]=x_min1
            df1[name6]=x_min2
            df1[name7]=x_min4
            #df1[name8]=x_min5
            #df1[name9]=x_min2
        if ii in feats02:
            df1[name1]=x_min3
            df1[name2]=xx_fil
            df1[name3]=xx_grad
            #df1[name4]=xx_mf 
            df1[name5]=x_min1
            #df1[name6]=x_min2 
            #df1[name7]=x_min4
    return df1

        


        
        


all_wells=training_data0['Well Name'].unique()
print all_wells


# what to do with the naans
training_data1=training_data0.copy()
me_tot=training_data1['PE'].median()
print me_tot
for well in all_wells:
    df=training_data0[training_data0['Well Name'] == well] 
    print well
    print len(df)
    df0=df.dropna()
    #print len(df0)
    if len(df0) > 0:
        print "using median of local"
        me=df['PE'].median()
        df=df.fillna(value=me)
    else:
        print "using median of total"
        df=df.fillna(value=me_tot)
    training_data1[training_data0['Well Name'] == well] =df
    

print len(training_data1)
df0=training_data1.dropna()
print len(df0)


#remove outliers
df=training_data1.copy()
print len(df)
df0=df.dropna()
print len(df0)
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
#df=pd.DataFrame(np.random.randn(20,3))
#df.iloc[3,2]=5
print len(df1)
df2=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
print len(df2)


def run_test(remove_well, df_train):
    
    df_test=training_data2
    blind = df_test[df_test['Well Name'] == remove_well]      
    training_data = df_train[df_train['Well Name'] != remove_well]  
    
    correct_facies_labels_train = training_data['Facies'].values
    feature_vectors = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    #rf = RandomForestClassifier(max_depth = 15, n_estimators=600) 
    #rf = RandomForestClassifier(max_depth = 7, n_estimators=600)  
    rf = RandomForestClassifier(max_depth = 5, n_estimators=300,min_samples_leaf=15)
    rf.fit(feature_vectors, correct_facies_labels_train)

    correct_facies_labels = blind['Facies'].values
    features_blind = blind.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    scaler = preprocessing.StandardScaler().fit(feature_vectors)
    scaled_features =feature_vectors
    predicted_random_forest = rf.predict(features_blind)

    out_f1=metrics.f1_score(correct_facies_labels, predicted_random_forest,average = 'micro')
    return out_f1

    
    



training_data2=magic(training_data1)
df_train=training_data2


wells=['CHURCHMAN BIBLE','SHANKLE','NOLAN','NEWBY','Recruit F9' ,'CROSS H CATTLE','LUKE G U','SHRIMPLIN']
av_all=[]
for remove_well in wells:
    all=[]
    print("well : %s, f1 for different runs:" % (remove_well))
    for ii in range(5):
        out_f1=run_test(remove_well,df_train)   
        if remove_well is not 'Recruit F9':
            all.append(out_f1)        
    av1=np.mean(all) 
    av_all.append(av1)
    print("average f1 is %f, 2*std is %f" % (av1, 2*np.std(all)) )
print("overall average f1 is %f" % (np.mean(av_all)))


# Train for the test data
# ---------------------------------------------------
# 

filename = 'validation_data_nofacies.csv'
test_data = pd.read_csv(filename)


test_data1=magic(test_data)


#test_well='STUART'
test_well='CRAWFORD'


blind = test_data1[test_data1['Well Name'] == test_well]      
training_data = training_data2

correct_facies_labels_train = training_data['Facies'].values
feature_vectors = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
rf = RandomForestClassifier(max_depth = 14, n_estimators=2500,min_samples_leaf=15) 
rf.fit(feature_vectors, correct_facies_labels_train)

features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)
predicted_random_forest = rf.predict(features_blind)



predicted_stu=predicted_random_forest
predicted_stu


predicted_craw=predicted_random_forest
predicted_craw








import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


train = pd.read_csv('01_raw_data/facies_vectors.csv')


cols = train.columns.values


# #There are 4149 elements, and PE has a significant amount of missing values
# 

well_PE_Miss = train.loc[train["PE"].isnull(),"Well Name"].unique()
well_PE_Miss


train.loc[train["Well Name"] == well_PE_Miss[0]].count()
train.loc[train["Well Name"] == well_PE_Miss[1]].count()


# The two wells have all PE missed
# 

(train.groupby("Well Name"))["PE"].mean()
(train.groupby("Well Name"))["PE"].median()


train["PE"] = train["PE"].fillna(train["PE"].median())
print(train.loc[train["Well Name"] == "CHURCHMAN BIBLE","PE"].mean())
print(train.loc[train["Well Name"] == "CHURCHMAN BIBLE","PE"].median())
print((train.groupby("Well Name"))["PE"].median()) ## QC for the fill in
print(train.loc[train["Well Name"] == "CHURCHMAN BIBLE","PE"].mean())
print(train.loc[train["Well Name"] == "CHURCHMAN BIBLE","PE"].median())
plt.show()


# The PE of all wells have no strong variance; For now, fillin the Missing value of median
# 

# Fancy visualization from forum
# 

features = ['GR', 'ILD_log10', 'DeltaPHI', 
    'PHIND','PE','NM_M', 'RELPOS']
feature_vectors = train[features]
facies_labels = train['Facies']
## 1=sandstone  2=c_siltstone   3=f_siltstone 
## 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
## 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
train.loc[:,'FaciesLabels'] = train.apply(lambda row: label_facies(row, facies_labels), axis=1)
#

def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)

make_facies_log_plot(
    train[train['Well Name'] == 'SHRIMPLIN'],
    facies_colors)
plt.show()


## Investigate the dependencies of the depth feature and Facies
wells = train["Well Name"].unique()
#train.plot(x = "Depth", y = "Facies")
#plt.show()
pi = 0
for well in wells:
    pi = pi + 1 # Plot index
    ax = plt.subplot(3, 4, pi)
    depthi = train.loc[train["Well Name"] == well, "Depth"].values
    faci = train.loc[train["Well Name"] == well, "Facies"].values
    plt.plot(faci,depthi)
    ax.set_title(well)


## Create dummy variables for Well Name, Formation, which may have geologic or geospatial information
train_dummy = pd.get_dummies(train[["Formation"]])
train_dummy.describe()
cols_dummy = train_dummy.columns.values
train[cols_dummy] = train_dummy[cols_dummy]
print(len(cols_dummy))


## For trainning drop Formation, FaciesLabels Leave Well Name for Later group splitting
wellgroups = train["Well Name"].values
train_inp = train.drop(["Formation","Well Name",'FaciesLabels'],axis =1)
train_inp.info()


# ### Build up Initial Test Loop for model and feature engineering : Test 1 SVC
# 

from sklearn.model_selection import LeavePGroupsOut
X = train_inp.drop(["Facies","Depth"],axis = 1).values
y = train_inp["Facies"].values
lpgo = LeavePGroupsOut(n_groups=2)
split_no = lpgo.get_n_splits(X,y,wellgroups)


# ## Bad indicator of model performance. It means no accurate prediction was found in one class  
# 
# 
# #/home/computer/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
# #'precision', 'predicted', average, warn_for)
# 

svc_b1 = SVC(C =1, gamma = 0.001, kernel = 'rbf')
svc_b1.fit(X,y)


test = pd.read_csv('01_raw_data/validation_data_nofacies.csv')
test.count()


test["Formation"].unique()


test_dummy = pd.get_dummies(test[["Formation"]])
test_cols_dummy = test_dummy.columns.values
test[test_cols_dummy] = test_dummy[cols_dummy]
test_inp = test.drop(["Formation","Well Name"],axis =1)
X_test = test_inp.drop(["Depth"],axis = 1).values


svc_b1.predict(X_test)
test = test.drop(test_cols_dummy,axis = 1)
test["Facies"] = svc_b1.predict(X_test)


test.to_csv("Houston_J_sub_1.csv")


# ## Facies classification using Random forest and engineered features
# 
# 
# #### Contest entry by: <a href="https://github.com/mycarta">Matteo Niccoli</a>,  <a href="https://github.com/dahlmb">Mark Dahl</a>, with a contribution by Daniel Kittridge.
# 
# ####  [Original contest notebook](https://github.com/seg/2016-ml-contest/blob/master/Facies_classification.ipynb) by Brendon Hall, [Enthought](https://www.enthought.com/)
# 
# 

# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The code and ideas in this notebook,</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Matteo Niccoli and Mark Dahl, </span> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
# 

# ### Loading the dataset with selected set of top 70 engineered features.
# 
# - We first created a large set of moments and GLCM features. The workflow is described in the 03_Facies_classification_MandMs_feature_engineering_commented.ipynb notebook (with huge thanks go to Daniel Kittridge for his critically needed Pandas magic, and useful suggestions). 
# - We then selected 70 using a Sequential (Forward) Feature Selector form Sebastian Raschka's [mlxtend](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/) library. Details in the 03_Facies_classification-MandMs_SFS_feature_selection.ipynb notebook.
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
from scipy.stats import randint as sp_randint
from scipy.signal import argrelextrema
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import LeaveOneGroupOut, validation_curve


filename = 'SFS_top70_selected_engineered_features.csv'
training_data = pd.read_csv(filename)
training_data.describe()


training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()


# Now we extract just the feature variables we need to perform the classification.  The predictor variables are the five log values and two geologic constraining variables, **and we are also using depth**. We also get a vector of the facies labels that correspond to each feature vector.
# 

y = training_data['Facies'].values
print y[25:40]
print np.shape(y)


X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
print np.shape(X)
X.describe(percentiles=[.05, .25, .50, .75, .95])


# ### Preprocessing data with standard scaler
# 

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)


# ### Make F1 performance scorers
# 

Fscorer = make_scorer(f1_score, average = 'micro')


# ### Parameter tuning ( maximum number of features and number of estimators): validation curves combined with leave one well out cross validation
# 

wells = training_data["Well Name"].values
logo = LeaveOneGroupOut()


# ###  Random forest classifier
# 
# In Random Forest classifiers serveral decision trees (often hundreds - a forest of trees) are created and trained on a random subsets of samples (drawn with replacement) and features (drawn without replacement); the decision trees work together to make a more accurate classification (description from Randal Olson's <a href="http://nbviewer.jupyter.org/github/rhiever/Data-Analysis-and-Machine-Learning-Projects/blob/master/example-data-science-notebook/Example%20Machine%20Learning%20Notebook.ipynb"> excellent notebook</a>).
# 

from sklearn.ensemble import RandomForestClassifier
RF_clf100 = RandomForestClassifier (n_estimators=100, n_jobs=-1, random_state = 49)
RF_clf200 = RandomForestClassifier (n_estimators=200, n_jobs=-1, random_state = 49)
RF_clf300 = RandomForestClassifier (n_estimators=300, n_jobs=-1, random_state = 49)
RF_clf400 = RandomForestClassifier (n_estimators=400, n_jobs=-1, random_state = 49)
RF_clf500 = RandomForestClassifier (n_estimators=500, n_jobs=-1, random_state = 49)
RF_clf600 = RandomForestClassifier (n_estimators=600, n_jobs=-1, random_state = 49)

param_name = "max_features"
#param_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
param_range = [9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51]

plt.figure()
plt.suptitle('n_estimators = 100', fontsize=14, fontweight='bold')
_, test_scores = validation_curve(RF_clf100, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 200', fontsize=14, fontweight='bold')
_, test_scores = validation_curve(RF_clf200, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 300', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf300, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show() 
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 400', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf400, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 500', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf500, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 600', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf600, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 


# ### Average test F1 score with leave one well out
# 

RF_clf_f1 = RandomForestClassifier (n_estimators=600, max_features = 21,
                                  n_jobs=-1, random_state = 49)

f1_RF = []

for train, test in logo.split(X, y, groups=wells):
    well_name = wells[test[0]]
    RF_clf_f1.fit(X[train], y[train])
    pred = RF_clf_f1.predict(X[test])
    sc = f1_score(y[test], pred, labels = np.arange(10), average = 'micro')
    print("{:>20s}  {:.3f}".format(well_name, sc))
    f1_RF.append(sc)
    
print "-Average leave-one-well-out F1 Score: %6f" % (sum(f1_RF)/(1.0*(len(f1_RF))))


# ### Predicting and saving facies for blind wells
# 

RF_clf_b = RandomForestClassifier (n_estimators=600, max_features = 21,
                                  n_jobs=-1, random_state = 49)


blind = pd.read_csv('engineered_features_validation_set_top70.csv') 
X_blind = np.array(blind.drop(['Formation', 'Well Name'], axis=1)) 
scaler1 = preprocessing.StandardScaler().fit(X_blind)
X_blind = scaler1.transform(X_blind) 
y_pred = RF_clf_b.fit(X, y).predict(X_blind) 
#blind['Facies'] = y_pred


np.save('ypred_RF_SFS_VC.npy', y_pred)


# ## Facies classification - Sequential Feature Selection
# 

# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The code and ideas in this notebook,</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Matteo Niccoli and Mark Dahl,</span> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
# 

# The [mlxtend](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/) library used for the sequential feature selection is by [Sebastian Raschka](https://sebastianraschka.com/projects.html).
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, make_scorer


filename = 'engineered_features.csv'
training_data = pd.read_csv(filename)
training_data.describe()


training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()


y = training_data['Facies'].values
print y[25:40]
print np.shape(y)


X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
print np.shape(X)
X.describe(percentiles=[.05, .25, .50, .75, .95])


scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)


# ### Make performance scorers 
# 

Fscorer = make_scorer(f1_score, average = 'micro')


# ### Sequential Feature Selection with mlextend
# 
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
# 

from sklearn.ensemble import RandomForestClassifier


# ### The next cell will take many hours to run, skip it
# 

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
clf = RandomForestClassifier(random_state=49)

sfs = SFS(clf, 
          k_features=100, 
          forward=True, 
          floating=False, 
          scoring=Fscorer,
          cv = 8,
          n_jobs = -1)

sfs = sfs.fit(X, y)


np.save('sfs_RF_metric_dict.npy', sfs.get_metric_dict()) 


# ### Restart from here
# 

# load previously saved dictionary
read_dictionary = np.load('sfs_RF_metric_dict.npy').item()


# plot results
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# run this twice
fig = plt.figure()                                                               
ax = plot_sfs(read_dictionary, kind='std_err')
fig_size = plt.rcParams["figure.figsize"] 
fig_size[0] = 22
fig_size[1] = 18

plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.xticks( rotation='vertical')
locs, labels = plt.xticks()
plt.xticks( locs, labels)
plt.show()


# ##### It looks like the score stabilizes after about 6 features, reaches a max at 16, then begins to taper off after about 70 features. We will save the top 45 and the top 75. 
# 

# save results to dataframe
selected_summary = pd.DataFrame.from_dict(read_dictionary).T
selected_summary['index'] = selected_summary.index
selected_summary.sort_values(by='avg_score', ascending=0)


# save dataframe
selected_summary.to_csv('SFS_RF_selected_features_summary.csv', sep=',', header=True, index = False)


# re load saved dataframe and sort by score
filename = 'SFS_RF_selected_features_summary.csv'
selected_summary = pd.read_csv(filename)
selected_summary = selected_summary.set_index(['index'])
selected_summary.sort_values(by='avg_score', ascending=0).head()


# feature selection with highest score
selected_summary.iloc[44]['feature_idx']


slct = np.array([257, 3, 4, 6, 7, 8, 10, 12, 16, 273, 146, 19, 26, 27, 284, 285, 30, 34, 163, 1, 42, 179, 155, 181, 184, 58, 315, 190, 320, 193, 194, 203, 290, 80, 210, 35, 84, 90, 97, 18, 241, 372, 119, 120, 126])
slct


# isolate and save selected features
filename = 'engineered_features_validation_set2.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation']], Xs], axis = 1)
print np.shape(Xs), list(Xs)


Xs.to_csv('SFS_top45_selected_engineered_features_validation_set.csv', sep=',',  index=False)


# feature selection with highest score
selected_summary.iloc[74]['feature_idx']


slct = np.array([257, 3, 4, 5, 6, 7, 8, 265, 10, 12, 13, 16, 273, 18, 19, 26, 27, 284, 285, 30, 34, 35, 1, 42, 304, 309, 313, 58, 315, 319, 320, 75, 80, 338, 84, 341, 89, 90, 92, 97, 101, 102, 110, 372, 119, 120, 122, 124, 126, 127, 138, 139, 146, 155, 163, 165, 167, 171, 177, 179, 180, 181, 184, 190, 193, 194, 198, 203, 290, 210, 211, 225, 241, 249, 253])
slct


# isolate and save selected features
filename = 'engineered_features_validation_set2.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation']], Xs], axis = 1)
print np.shape(Xs), list(Xs)


Xs.to_csv('SFS_top75_selected_engineered_features_validation_set.csv', sep=',',  index=False)





# ## Facies classification - Sequential Feature Selection
# 

# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The code and ideas in this notebook,</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Matteo Niccoli and Mark Dahl,</span> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
# 

# The [mlxtend](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/) library used for the sequential feature selection is by [Sebastian Raschka](https://sebastianraschka.com/projects.html).
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, make_scorer


filename = 'engineered_features.csv'
training_data = pd.read_csv(filename)
training_data.describe()


training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()


y = training_data['Facies'].values
print y[25:40]
print np.shape(y)


X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
print np.shape(X)
X.describe(percentiles=[.05, .25, .50, .75, .95])


scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)


# ### Make performance scorers 
# 

Fscorer = make_scorer(f1_score, average = 'micro')


# ### Sequential Feature Selection with mlextend
# 
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
# 

from sklearn.ensemble import RandomForestClassifier


# ### The next cell will take many hours to run, skip it
# 

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
clf = RandomForestClassifier(random_state=49)

sfs = SFS(clf, 
          k_features=100, 
          forward=True, 
          floating=False, 
          scoring=Fscorer,
          cv = 8,
          n_jobs = -1)

sfs = sfs.fit(X, y)


np.save('sfs_RF_metric_dict.npy', sfs.get_metric_dict()) 


# ### Restart from here
# 

# load previously saved dictionary
read_dictionary = np.load('sfs_RF_metric_dict.npy').item()


# plot results
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# run this twice
fig = plt.figure()                                                               
ax = plot_sfs(read_dictionary, kind='std_err')
fig_size = plt.rcParams["figure.figsize"] 
fig_size[0] = 22
fig_size[1] = 18

plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.xticks( rotation='vertical')
locs, labels = plt.xticks()
plt.xticks( locs, labels)
plt.show()


# ##### It looks like the score stabilizes after about 6 features, reaches a max at 16, then begins to taper off after about 70 features. We will save the top 45 and the top 75. 
# 

# save results to dataframe
selected_summary = pd.DataFrame.from_dict(read_dictionary).T
selected_summary['index'] = selected_summary.index
selected_summary.sort_values(by='avg_score', ascending=0)


# save dataframe
selected_summary.to_csv('SFS_RF_selected_features_summary.csv', sep=',', header=True, index = False)


# re load saved dataframe and sort by score
filename = 'SFS_RF_selected_features_summary.csv'
selected_summary = pd.read_csv(filename)
selected_summary = selected_summary.set_index(['index'])
selected_summary.sort_values(by='avg_score', ascending=0).head()


# feature selection with highest score
selected_summary.iloc[39]['feature_idx']


slct = np.array([256, 257, 3, 6, 1, 264, 137, 23, 280, 281, 288, 289, 113, 168, 7, 304, 305, 312, 193, 328, 
                 329, 224, 80, 81, 83, 122, 95, 352, 353, 232, 233, 295, 208, 109, 336, 360, 118, 248, 250, 255])
slct


# isolate and save selected features
filename = 'engineered_features.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation', 'Facies']], Xs], axis = 1)
print np.shape(Xs), list(Xs)


Xs.to_csv('SFS_top40_selected_engineered_features.csv', sep=',',  index=False)


# feature selection with highest score
selected_summary.iloc[69]['feature_idx']


slct = np.array([256, 257, 3, 4, 6, 1, 264, 9, 17, 277, 23, 280, 281, 283, 288, 289, 295, 40, 7, 304, 305, 308, 265, 
                 312, 317, 360, 97, 328, 329, 331, 79, 80, 81, 83, 89, 350, 95, 352, 353, 99, 104, 364, 109, 113, 
                 118, 120, 122, 128, 137, 149, 151, 153, 168, 169, 171, 174, 193, 196, 207, 208, 224, 336, 226, 
                 227, 232, 233, 25, 248, 250, 255])
slct


# isolate and save selected features
filename = 'engineered_features.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation', 'Facies']], Xs], axis = 1)
print np.shape(Xs), list(Xs)


Xs.to_csv('SFS_top70_selected_engineered_features.csv', sep=',',  index=False)





# # 03 - Facies Determination with Regression
# 
# As with the prior entries, this is a combination of brute-force feature creation and an ExtraTrees Regressor method. The aim of this is to capture more of the inter-dependancy of samples.
# I will freely admit that this is stretching my ML knowledge, I've spent quite a lot of time trying to ascertain whether this is a sensible thing to be doing at all. Comments and thoughts very welcome!
# 

import pandas as pd
import bokeh.plotting as bk
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier, TPOTRegressor

import sys
sys.path.append(r'C:\Users\george.crowther\Documents\Python\Projects\2016-ml-contest-master')

import classification_utilities

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

bk.output_notebook()


# Input file paths
train_path = r'..\training_data.csv'

# Read training data to dataframe
train = pd.read_csv(train_path)

# TPOT library requires that the target class is renamed to 'class'
train.rename(columns={'Facies': 'class'}, inplace=True)

well_names = train['Well Name']


# Set string features to integers

for i, value in enumerate(train['Formation'].unique()):
    train.loc[train['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(train['Well Name'].unique()):
    train.loc[train['Well Name'] == value, 'Well Name'] = i


# The first thing that will be done is to upsample and interpolate the training data,
# the objective here is to provide significantly more samples to train the regressor on and
# also to capture more of the sample interdependancy.
upsampled_arrays = []
train['orig_index'] = train.index

for well, group in train.groupby('Well Name'):
    # This is a definite, but helpful, mis-use of the pandas resample timeseries
    # functionality.
    group.index = pd.to_datetime(group['Depth'] * 10)
    # Upsampled by a factor of 5 and interpolate
    us_group = group.resample('1ns').mean().interpolate(how='time')
    # Revert to integer
    us_group.index = us_group.index.asi8 / 10
    us_group['Well Name'] = well
    
    upsampled_arrays.append(us_group)


upsampled_arrays[0].head()


resample_factors = [2, 5, 10, 50, 100, 200]

initial_columns = ['Formation', 'Well Name', 'Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']

upsampled_frame = pd.concat(upsampled_arrays, axis = 0)


# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []

for well, group in upsampled_frame.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS', 'Well Name']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = f)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = f)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = f)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = f)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = f)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['class'] = group['class']
    well_frame['Well Name'] = group['Well Name']
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    mean_frames.append(well_frame)


upsampled_frame.index = upsampled_frame['orig_index']
upsampled_frame.drop(['orig_index', 'class', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
upsampled_frame = pd.concat((upsampled_frame, rolling_frame), axis = 1)

# Features is the column set used for training the model
features = upsampled_frame.columns[:-4]
print(features)


# Define model

from sklearn.ensemble import ExtraTreesRegressor, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

exported_pipeline = make_pipeline(
    ExtraTreesRegressor(max_features=0.27, n_estimators=500)
)


# Fit model to data
exported_pipeline.fit(upsampled_frame[features], upsampled_frame['class'])


# Now load and process the test data set, then predict using the 'exported_pipeline' model.
# 

test_path = r'..\validation_data_nofacies.csv'

# Read training data to dataframe
test = pd.read_csv(test_path)

# TPOT library requires that the target class is renamed to 'class'
test.rename(columns={'Facies': 'class'}, inplace=True)

# Set string features to integers

for i, value in enumerate(test['Formation'].unique()):
    test.loc[train['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(test['Well Name'].unique()):
    test.loc[test['Well Name'] == value, 'Well Name'] = i

# The first thing that will be done is to upsample and interpolate the training data,
# the objective here is to provide significantly more samples to train the regressor on and
# also to capture more of the sample interdependancy.
upsampled_arrays = []
test['orig_index'] = test.index

for well, group in test.groupby('Well Name'):
    # This is a definite, but helpful, mis-use of the pandas resample timeseries
    # functionality.
    group.index = pd.to_datetime(group['Depth'] * 10)
    # Upsampled by a factor of 5 and interpolate
    us_group = group.resample('1ns').mean().interpolate(how='time')
    # Revert to integer
    us_group.index = us_group.index.asi8 / 10
    us_group['Well Name'] = well
    
    upsampled_arrays.append(us_group)
    
upsampled_frame = pd.concat(upsampled_arrays, axis = 0)

# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []

for well, group in upsampled_frame.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS', 'Well Name']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = f)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = f)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = f)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = f)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = f)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['Well Name'] = group['Well Name']
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    mean_frames.append(well_frame)
    
upsampled_frame.index = upsampled_frame['orig_index']
upsampled_frame.drop(['orig_index', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
upsampled_frame = pd.concat((upsampled_frame, rolling_frame), axis = 1)

tfeatures = upsampled_frame.columns[:-3]
print(tfeatures)


# Predict result on full sample set
result = exported_pipeline.predict(upsampled_frame[tfeatures])
# Round result to nearest int
upsampled_frame['Facies'] = [round(n) for n in result]
# Extract results against test index
result_frame = upsampled_frame.loc[test.index, :]
# Output to csv
result_frame.to_csv('regressor_results.csv')





# ## Stochastic validations
# 
# We'd like to score the teams based on 100 realizations of their models — most of which are stochastic and take a random seed parameter. Please see the notebooks with `_VALIDATION` suffixes to see how the realizations we generated. Each one of those notebooks generartes a file called `<team>_100_realizations.npy`, which is what we are consuming here.
# 
# This notebook is super hacky I'm afraid.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.metrics import confusion_matrix, f1_score
from utils import accuracy, accuracy_adjacent, display_cm, facies_labels


# Using globals. I am a miserable person.
# 

PRED = pd.read_csv('prediction_depths.csv')
PRED.set_index(["Well Name", "Depth"], inplace=True)
PRED.head()


TRUE = pd.read_csv('blind_stuart_crawford_core_facies.csv')
TRUE.rename(columns={'Depth.ft': 'Depth'}, inplace=True)
TRUE.rename(columns={'WellName': 'Well Name'}, inplace=True)
TRUE.set_index(["Well Name", "Depth"], inplace=True)
TRUE.head()


def get_accuracies(y_preds):
    """
    Get the F1 scores from all the y_preds.
    y_blind is a 1D array. y_preds is a 2D array.
    """
    accs = []
    for y_pred in y_preds:
        PRED['Facies'] = y_pred
        all_data = PRED.join(TRUE, how='inner')
        y_blind = all_data['LithCode'].values
        y_pred = all_data['Facies'].values
        y_pred = y_pred[y_blind!=11]
        y_blind = y_blind[y_blind!=11]
        cv_conf = confusion_matrix(y_blind, y_pred)
        accs.append(accuracy(cv_conf))
    return np.array(accs)


from glob import glob
from os import path
import operator

scores, medians = {}, {}
for f in glob('./*/*_100_realizations.npy'):
    team = path.basename(f).split('_')[0]
    y_preds = np.load(f)
    scores[team] = get_accuracies(y_preds)
    medians[team] = np.median(scores[team])
    plt.hist(pd.Series(scores[team]), alpha=0.5)
    
for t, m in sorted(medians.items(), key=operator.itemgetter(1), reverse=True):
    print("{:20s}{:.4f}".format(t, m))


# ## Look more closely at LA Team
# 

s = pd.Series(scores['LA-Team'])
plt.hist(s)


s.describe()





# # Facies classification using Machine Learning #
# ## LA Team Submission 5 ## 
# ### _[Lukas Mosser](https://at.linkedin.com/in/lukas-mosser-9948b32b/en), [Alfredo De la Fuente](https://pe.linkedin.com/in/alfredodelafuenteb)_ ####
# 

# In this approach for solving the facies classfication problem ( https://github.com/seg/2016-ml-contest. ) we will explore the following statregies:
# - Features Exploration: based on [Paolo Bestagini's work](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb), we will consider imputation, normalization and augmentation routines for the initial features.
# - Model tuning: 
# 

# ## Libraries
# 
# We will need to install the following libraries and packages.
# 

get_ipython().run_cell_magic('sh', '', 'pip install pandas\npip install scikit-learn\npip install tpot')


from __future__ import print_function
import numpy as np
get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , StratifiedKFold
from classification_utilities import display_cm, display_adj_cm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import medfilt


# ## Data Preprocessing
# 

#Load Data
data = pd.read_csv('../facies_vectors.csv')

# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

print(data.head())
data['PE'] = data.groupby("Facies").PE.transform(lambda x: x.fillna(x.mean()))


# We procceed to run [Paolo Bestagini's routine](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb) to include a small window of values to acount for the spatial component in the log analysis, as well as the gradient information with respect to depth. This will be our prepared training dataset.
# 

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, train_size=0.75, test_size=0.25)

# Store features and labels
X = data[feature_names].values 
y = data['Facies'].values 

# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
    
        
def preprocess():
    
    # Preprocess data to use in model
    X_train_aux = []
    X_test_aux = []
    y_train_aux = []
    y_test_aux = []
    
    # For each data split
    split = split_list[5]
        
    # Remove padded rows
    split_train_no_pad = np.setdiff1d(split['train'], padded_rows)

    # Select training and validation data from current split
    X_tr = X_aug[split_train_no_pad, :]
    X_v = X_aug[split['val'], :]
    y_tr = y[split_train_no_pad]
    y_v = y[split['val']]

    # Select well labels for validation data
    well_v = well[split['val']]

    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
        
    X_train_aux.append( X_tr )
    X_test_aux.append( X_v )
    y_train_aux.append( y_tr )
    y_test_aux.append (  y_v )
    
    X_train = np.concatenate( X_train_aux )
    X_test = np.concatenate ( X_test_aux )
    y_train = np.concatenate ( y_train_aux )
    y_test = np.concatenate ( y_test_aux )
    
    return X_train , X_test , y_train , y_test 


# ## Data Analysis
# 
# In this section we will run a Cross Validation routine 
# 

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = preprocess()

tpot = TPOTClassifier(generations=5, population_size=20, 
                      verbosity=2,max_eval_time_mins=20,
                      max_time_mins=100,scoring='f1_micro',
                      random_state = 17)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('FinalPipeline_LM_mean_per_facies.py')


from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    #clf = make_pipeline(make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)), XGBClassifier(learning_rate=0.73, max_depth=10, min_child_weight=10, n_estimators=500, subsample=0.27))
    #clf =  make_pipeline( KNeighborsClassifier(n_neighbors=5, weights="distance") ) 
    #clf = make_pipeline(MaxAbsScaler(),make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),ExtraTreesClassifier(criterion="entropy", max_features=0.0001, n_estimators=500))
    # * clf = make_pipeline( make_union(VotingClassifier([("est", BernoulliNB(alpha=60.0, binarize=0.26, fit_prior=True))]), FunctionTransformer(lambda X: X)),RandomForestClassifier(n_estimators=500))
    clf = make_pipeline ( XGBClassifier(learning_rate=0.12, max_depth=3, min_child_weight=10, n_estimators=150, seed = 17, colsample_bytree = 0.9) )
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# ## Prediction
# 

#Load testing data
test_data = pd.read_csv('../validation_data_nofacies.csv')

# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 

# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)

# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts)

# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('Prediction_XX_Final.csv')





# # Facies classification using Machine Learning
# ## Joshua Lowe
# ### https://uk.linkedin.com/in/jlowegeo
# 

# This notebook contains my submission to the SEG Machine Learning contest 2016/17.
# I have implemented code to train a Neural Network and predict facies in a well from a variety or wireline logs.
# 
# I have used bits of code from the original tutorial by Brendon Hall and from PA_Team, where I have used the 'blind well test' implemented by using leaveonegroupout. 
# 
# Thanks for all the different teams submissions as I have been able to learn a lot of skills around implementing machine learning algorithms in Python.
# 

import numpy as np
np.random.seed(1000)

import warnings
warnings.filterwarnings("ignore")

import time as tm
import pandas as pd
from scipy.signal import medfilt

from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
from keras.utils import np_utils

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import preprocessing

#Cross Val of final model
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier


training_data = pd.read_csv('../training_data.csv')
blind_data = pd.read_csv('../nofacies_data.csv')


def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf))
    return acc

adjacent_facies = np.array([[1], [0, 2], [1], [4], [3, 5], [4, 6, 7], [5, 7], [5, 6, 8], [6, 7]])


def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))


# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
training_data.loc[:,'FaciesLabels'] = training_data.apply(lambda row: label_facies(row, facies_labels), axis=1)


# ### Sorting the data and dropping unwanted columns from the training and test data
# 

# Leave the depth in as a predictor - can the NN recognise depth trends? - Other teams gone much further and have taken into account a predictors relationship/change with depth.
# 

X = training_data.drop(['Formation', 'Well Name', 'Facies', 'FaciesLabels'], axis=1).values
y = training_data['Facies'].values - 1
X_blind = blind_data.drop(['Formation', 'Well Name'], axis=1).values
wells = training_data["Well Name"].values


# Scaling predictors in the data.
# 

scaler = preprocessing.RobustScaler().fit(X)
X_scaled = scaler.transform(X)


# ### Defining the neural network model
# 

def DNN():
    # Model
    model = Sequential()
    model.add(Dense(205, input_dim=8, activation='relu',W_constraint=maxnorm(5)))
    model.add(Dropout(0.1))
    model.add(Dense(69, activation='relu',W_constraint=maxnorm(5)))
    model.add(Dropout(0.1))
    model.add(Dense(69, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    # Compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ### Cross Validation using a 'Blind Well Test'. Code adapted from PA_Team submission 
# 

logo = LeaveOneGroupOut()
t0 = tm.time()
f1s_ls = []
acc_ls = []
adj_ls = []

for train, test in logo.split(X_scaled, y, groups=wells):
    well_name = wells[test[0]]
    X_tr = X_scaled[train]
    X_te = X_scaled[test]
   
    #convert y array into categories matrix
    classes = 9
    y_tr = np_utils.to_categorical(y[train], classes)
    
    # Method initialization
    NN = DNN()
    
    # Training
    NN.fit(X_tr, y_tr, nb_epoch=15, batch_size=5, verbose=0) 
    
    # Predict
    y_hat = NN.predict_classes(X_te, verbose=0)
    y_hat = medfilt(y_hat, kernel_size=7)
    
    try:
        f1s = f1_score(y[test], y_hat, average="weighted", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    except:
        f1s = 0

    try:
        conf = confusion_matrix(y[test], y_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        acc = accuracy(conf) # similar to f1 micro
    except:
        acc = 0

    try:
        acc_adj = accuracy_adjacent(conf, adjacent_facies)
    except:
        acc_adj = 0

    f1s_ls += [f1s]
    acc_ls += [acc]
    adj_ls += [acc_adj]
    print("{:>20s} f1_weigthted:{:.3f} | acc:{:.3f} | acc_adj:{:.3f}".format(well_name, f1s, acc, acc_adj))

t1 = tm.time()
print("Avg F1", np.average(f1s_ls)*100, "Avg Acc", np.average(acc_ls)*100, "Avg Adj", np.average(adj_ls)*100)
print("Blind Well Test Run Time:",'{:f}'.format((t1-t0)), "seconds")


# ### Cross Validation using stratified K-fold
# 

#Another robustness test of the model using statified K fold
X_train = X_scaled
Y_train = np_utils.to_categorical(y, classes)
t2 = tm.time()
estimator = KerasClassifier(build_fn=DNN, nb_epoch=15, batch_size=5, verbose=0)
skf = StratifiedKFold(n_splits=5, shuffle=True)
results_dnn = cross_val_score(estimator, X_train, Y_train, cv= skf.get_n_splits(X_train, Y_train))
print (results_dnn)
t3 = tm.time()
print("Cross Validation Run Time:",'{:f}'.format((t3-t2)), "seconds")


# ### Final Model which uses all the training data 
# 

# By using all the training data I may be potentially increasing the variance of the model but I believe it’s best to use all the data in the model as the data available is limited.
# 

NN = DNN()
NN.fit(X_train, Y_train, nb_epoch=15, batch_size=5, verbose=0)

y_predicted = NN.predict_classes(X_train, verbose=0)
y_predicted = medfilt(y_predicted, kernel_size=7)

f1s = f1_score(y, y_predicted, average="weighted")
Avgf1s = np.average(f1s_ls)*100
print ("f1 training error: ", '{:f}'.format(f1s))
print ("f1 test error: ", '{:f}'.format(Avgf1s))


# My variance is high and my bias is too low.
# 
# I haven’t found the optimum bias-variance trade off. --> Back to the drawing board.
# 

# ### Predicting the lithologies in the unknown test wells 
# 

x_blind = scaler.transform(X_blind)
y_blind = NN.predict_classes(x_blind, verbose=0)
y_blind = medfilt(y_blind, kernel_size=7)
blind_data["Facies"] = y_blind + 1  # return the original value (1-9)


blind_data.to_csv("J_Lowe_Submission.csv")


# ### *The Leading Edge*
# 
# # Machine learning contest 2016
# 
# **Welcome to an experiment!**
# 
# You mission, should you choose to accept it, is to make the best lithology prediction you can. We want you to try to beat the accuracy score Brendon Hall achieved in his Geophyscial Tutorial (TLE, October 2016). 
# 
# First, read the [open access](https://en.wikipedia.org/wiki/Open_access) tutorial by Brendon in [the October issue of *The Leading Edge*](http://library.seg.org/toc/leedff/35/10). 
# 
# Here's the text of that box again:
# 
# > I hope you enjoyed this month's tutorial. It picks up on a recent wave of interest in artificial intelligence approaches to prediction. I love that Brendon shows how approachable the techniques are — the core part of the process only amounts to a few dozen lines of fairly readable Python code. All the tools are free and open source, it's just a matter of playing with them and learning a bit about data science.
# 
# > In the blind test, Brendon's model achieves an accuracy of 43% with exact facies. We think the readers of this column can beat this — and we invite you to have a go. The repository at [github.com/seg/2016-ml-contest](http://github.com/seg/2016-ml-contest) contains everything you need to get started, including the data and Brendon's code. We invite you to find a friend or two or more, and have a go!
# 
# > To participate, fork that repo, and add a directory for your own solution, naming it after your team. You can make pull requests with your contributions, which must be written in Python, R, or Julia. We'll run them against the blind well — the same one Brendon used in the article — and update the leaderboard. You can submit solutions as often as you like. We'll close the contest at **23:59 UT on 31 January 2017**. There will be a goody bag of completely awesome and highly desirable prizes for whoever is at the top of the leaderboard when the dust settles. The full rules are in the repo.
# 
# > Have fun with it, and good luck!
# 
# ## Now for the code
# 
# All the code and data to reproduce *everything* in that article is right here in this repository. You can read the code in a [Jupyter Notebook](http://jupyter.org/) here...
# 
# <div style="width:50%; margin: 12px 0px 6px 20px; padding: 8px; border: 2px solid darkblue; border-radius: 6px; font-size: 125%; background: #EEEEFF;">
# [**Facies_classification.ipynb**](Facies_classification.ipynb)
# </div>
# 
# See [the February issue of *The Leading Edge*](http://library.seg.org/doi/abs/10.1190/tle35020190.1) for Matt Hall's user guide to the tutorials; it explains how to run a Jupyter Notebook.
# 
# See **Running the notebook live** (below) for information on running that noteobook live right now this minute in your web browser.
# 
# 
# ## Entering the contest
# 
# - Find a friend or two or ten (optional) and form a team.
# - To get a copy of the repo that you can make pull requests from (that is, notify us that you want to send us an entry to the contest), you need to [fork the repo](https://help.github.com/articles/fork-a-repo/)
# - Use Jupyter Notebook (to make our life easy!) with Python, R, or Julia kernels, or write scripts and put them in the repo in a directory named after your team. 
# - When you have a good result, send it to us by [making a pull request](https://help.github.com/articles/about-pull-requests/).
# - Everyone can see your entry. If you're not familiar with open source software, this might feel like a bug. It's not, it's a feature. If it's good, your contribution will improve others' results. Welcome to reproducible science!
# 
# 
# ## Running the notebook live
# 
# To make it even easier to try machine learning for yourself, you can launch this notebook on [**mybinder.org**](http://www.mybinder.org/) and run it there. You can load the data, change the code, and do everything... except enter the contest. Everything on your mybinder.org machine is **temporary**. If you make something awesome, be sure to use **File > Download as... > Notebook (.ipynb)** to save it locally. Then you can fork the repo in GitHub, add your new notebook, and make your pull request.
# 

# ## Rules
# 
# We've never done anything like this before, so there's a good chance these rules will become clearer as we go. We aim to be fair at all times, and reserve the right to make judgment calls for dealing with unforeseen circumstances.
# 
# - You must submit your result as code and we must be able to run your code.
# - The result we get with your code is the one that counts as your result.
# - To make it more likely that we can run it, your code must be written in Python or R or Julia.
# - The contest is over at 23:59:59 UT (i.e. midnight in London, UK) on 31 January 2017. Pull requests made aftetr that time won't be eligible for the contest.
# - If you can do even better with code you don't wish to share fully, that's really cool, nice work! But you can't enter it for the contest. We invite you to share your result through your blog or other channels... maybe a paper in *The Leading Edge*.
# - This document and documents it links to will be the channel for communication of the leading solution and everything else about the contest.
# - This document contains the rules. Our decision is final. No purchase necessary. Please exploit artificial intelligence responsibly. 
# 

# <hr />
# 
# &copy; 2016 SEG, licensed CC-BY, please share this!
# 

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from pandas import set_option
set_option("display.max_rows", 10)


# ### Load and pre-process data
# 

from sklearn import preprocessing

filename = '../facies_vectors.csv'
train = pd.read_csv(filename)

# encode well name and formation features
le = preprocessing.LabelEncoder()
train["Well Name"] = le.fit_transform(train["Well Name"])
train["Formation"] = le.fit_transform(train["Formation"])

data_loaded = train.copy()

# cleanup memory
del train

data_loaded


# ### Impute PE
# 
# First, I will impute PE by replacing missing values with the mean PE. Second, I will impute PE using a random forest regressor. I will compare the results by looking at the average RMSE's by performing the method across all wells with PE data (leaving each well out as a test set).
# 
# #### Impute PE through mean substitution
# 
# To evaluate - I will build a model for each well (the data for that well being the test data). Then I'll compute the RMSE for each model where we know the outcomes (the actual PE) to give us an idea of how good the model is.
# 

from sklearn import preprocessing

data = data_loaded.copy()

impPE_features = ['Facies', 'Formation', 'Well Name', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
rmse = []

for w in data["Well Name"].unique():
    wTrain = data[(data["PE"].notnull()) & (data["Well Name"] != w)]
    wTest = data[(data["PE"].notnull()) & (data["Well Name"] == w)]
    
    if wTest.shape[0] > 0:
        yTest = wTest["PE"].values
        
        meanPE = wTrain["PE"].mean()
        wTest["predictedPE"] = meanPE
        
        rmse.append((((yTest - wTest["predictedPE"])**2).mean())**0.5)
        
print(rmse)
print("Average RMSE:" + str(sum(rmse)/len(rmse)))

# cleanup memory
del data


# #### Impute PE through random forest regression
# 
# Using mean substitution as a method for PE imputing has an expected RMSE of just over 1.00. Let's see if I can do better using a random forest regressor.
# 

from sklearn.ensemble import RandomForestRegressor

data = data_loaded.copy()

impPE_features = ['Facies', 'Formation', 'Well Name', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
rf = RandomForestRegressor(max_features='sqrt', n_estimators=100, random_state=1)
rmse = []

for w in data["Well Name"].unique():
    wTrain = data[(data["PE"].isnull() == False) & (data["Well Name"] != w)]
    wTest = data[(data["PE"].isnull() == False) & (data["Well Name"] == w)]
    
    if wTest.shape[0] > 0:
        XTrain = wTrain[impPE_features].values
        yTrain = wTrain["PE"].values
        XTest = wTest[impPE_features].values
        yTest = wTest["PE"].values
        
        w_rf = rf.fit(XTrain, yTrain)
        
        predictedPE = w_rf.predict(XTest)
        rmse.append((((yTest - predictedPE)**2).mean())**0.5)
    
print(rmse)
print("Average RMSE:" + str(sum(rmse)/len(rmse)))

# cleanup memory
del data


# This approach gives us an expected RMSE of about 0.575 - now let's impute the missing data using this approach!
# 

data = data_loaded.copy()

rf_train = data[data['PE'].notnull()]
rf_test = data[data['PE'].isnull()]

xTrain = rf_train[impPE_features].values
yTrain = rf_train["PE"].values
xTest = rf_test[impPE_features].values

rf_fit = rf.fit(xTrain, yTrain)
predictedPE = rf_fit.predict(xTest)
data["PE"][data["PE"].isnull()] = predictedPE

data_imputed = data.copy()

# cleanup memory
del data

# output
data_imputed


# Now we have a full data set with no missing values!
# 
# ### Feature engineering
# 
# I'm going to now calculate the average value of each log feature on a **by facies** basis. For instance, I will calculate the *distance* of an observation's **GR** reading from the **MS** GR average. The idea being that true **MS**'s will be close to that average! I will be squaring the observation deviations from these averages to make it more of a data-distance proxy.
# 

facies_labels = ['SS','CSiS','FSiS','SiSh','MS','WS','D','PS','BS']

data = data_imputed.copy()

features = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
for f in features:
    facies_mean = data[f].groupby(data["Facies"]).mean()
    
    for i in range(0, len(facies_mean)):
        data[f + "_" + facies_labels[i] + "_SqDev"] = (data[f] - facies_mean.values[i])**2

data_fe = data.copy()

del data
data_fe


# I proceed to run [Paolo Bestagini's routines](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb) to include a small window of values to account for the spatial component in the log analysis, as well as gradient information with respect to depth.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug

# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad

# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


# Now I'll apply the Paolo routines to the data - augmenting the features!
# 

data = data_fe.copy()

remFeatures = ["Facies", "Well Name", "Depth"]
x = list(data)
features = [f for f in x if f not in remFeatures]

X = data[features].values
y = data["Facies"].values

# Store well labels and depths
well = data['Well Name']
depth = data['Depth'].values

X_aug, padded_rows = augment_features(X, well.values, depth)


# ### Tuning and Cross-Validation
# 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
#from classification_utilities import display_cm, display_adj_cm

# 1) loops through wells - splitting data (current well held out as CV/test)
# 2) trains model (using all wells excluding current)
# 3) evaluates predictions against known values and adds f1-score to array
# 4) returns average f1-score (expected f1-score)
def cvTrain(X, y, well, params):
    rf = RandomForestClassifier(max_features=params['M'], n_estimators=params['N'], criterion='entropy', 
                                min_samples_split=params['S'], min_samples_leaf=params['L'], random_state=1)
    f1 = []
    
    for w in well.unique():
        Xtrain_w = X[well.values != w]
        ytrain_w = y[well.values != w]
        Xtest_w = X[well.values == w]
        ytest_w = y[well.values == w]
        
        w_rf = rf.fit(Xtrain_w, ytrain_w)
        predictedFacies = w_rf.predict(Xtest_w)
        f1.append(f1_score(ytest_w, predictedFacies, average='micro'))
        
    f1 = (sum(f1)/len(f1))
    return f1


# Apply tuning to search for optimal hyperparameters.
# 

# parameters search grid (uncomment for full grid search - will take a long time)
N_grid = [250]    #[50, 250, 500]        # n_estimators
M_grid = [75]     #[25, 50, 75]          # max_features
S_grid = [5]      #[5, 10]               # min_samples_split
L_grid = [2]      #[2, 3, 5]             # min_samples_leaf

# build grid of hyperparameters
param_grid = []
for N in N_grid:
    for M in M_grid:
        for S in S_grid:
            for L in L_grid:
                param_grid.append({'N':N, 'M':M, 'S':S, 'L':L})
                
# loop through parameters and cross-validate models for each
for params in param_grid:
    print(str(params) + ' Average F1-score: ' + str(cvTrain(X_aug, y, well, params)))


# Through tuning we observe optimal hyperparameters to be 250 (number of estimators), 2 (minimum number of samples per leaf), 75 (maximum number of features to consider when looking for the optimal split), and 5 (minimum number of samples required to split a node). These values yielded an average F1-score of 0.584 through cross-validation.
# 
# ### Prediction
# 
# Before applying out algorithm to the test data, I must apply the feature engineering to the test data. This involves calculating the data deviations from the facies averages and applying [Paolo Bestagini's routines](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb).
# 

from sklearn import preprocessing

filename = '../validation_data_nofacies.csv'
test = pd.read_csv(filename)

# encode well name and formation features
le = preprocessing.LabelEncoder()
test["Well Name"] = le.fit_transform(test["Well Name"])
test["Formation"] = le.fit_transform(test["Formation"])
test_loaded = test.copy()

facies_labels = ['SS','CSiS','FSiS','SiSh','MS','WS','D','PS','BS']

train = data_imputed.copy()

features = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
for f in features:
    facies_mean = train[f].groupby(train["Facies"]).mean()
    
    for i in range(0, len(facies_mean)):
        test[f + "_" + facies_labels[i] + "_SqDev"] = (test[f] - facies_mean.values[i])**2

test_fe = test.copy()

del test

test_fe


# Now I will apply [Paolo Bestagini's routines](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb).
# 

test = test_fe.copy()

remFeatures = ["Well Name", "Depth"]
x = list(test)
features = [f for f in x if f not in remFeatures]

Xtest = test[features].values

# Store well labels and depths
welltest = test['Well Name']
depthtest = test['Depth'].values

Xtest_aug, test_padded_rows = augment_features(Xtest, welltest.values, depthtest)


from sklearn.ensemble import RandomForestClassifier

test = test_loaded.copy()

rf = RandomForestClassifier(max_features=75, n_estimators=250, criterion='entropy', 
                                min_samples_split=5, min_samples_leaf=2, random_state=1)
fit = rf.fit(X_aug, y)
predictedFacies = fit.predict(Xtest_aug)

test["Facies"] = predictedFacies
test.to_csv('jpoirier011_submission001.csv')


# # Facies classification using machine learning techniques
# ##### Contact author: <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini</a>
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm is based on the use of random forests combined in one-vs-one multiclass strategy. In particular, we would like to study the effect of:
# - Robust feature normalization.
# - Feature imputation for missing feature values.
# - Well-based cross-validation routines.
# - Feature augmentation strategies.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier

from scipy.signal import medfilt


import sys, scipy, sklearn
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. Different strategies could be used. We simply substitute them with the average PE value.
# 

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# ## Feature augmentation
# Our guess is that facies do not abrutly change from a given depth layer to the next one. Therefore, we consider features at neighboring layers to be somehow correlated. To possibly exploit this fact, let us perform feature augmentation by:
# - Aggregating features at neighboring depths.
# - Computing feature spatial gradient.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [100]  # [50, 100, 150]
M_grid = [10]  # [5, 10, 15]
S_grid = [25]  # [10, 25, 50, 75]
L_grid = [5] # [2, 3, 4, 5, 10, 25]
param_grid = []
for N in N_grid:
    for M in M_grid:
        for S in S_grid:
            for L in L_grid:
                param_grid.append({'N':N, 'M':M, 'S':S, 'L':L})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
                             max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
                             class_weight='balanced', random_state=0), n_jobs=-1)

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('well_data_with_facies_try02.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)


# # Facies classification using machine learning techniques
# 
# ### ISPL Team
# ##### Contact author: <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini</a>
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# This is a corrected version of [our previous submission try03](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try03.ipynb) built upon:
# - Part of the feature engineering work presented in [our previous submission](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb).
# - The gradient boosting classifier used in [SHandPR submission](https://github.com/seg/2016-ml-contest/blob/master/SHandPR/Face_classification_SHPR_GradientBoost.ipynb).
# 
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import GradientBoostingClassifier

from scipy.signal import medfilt


import sys, scipy, sklearn
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well)):
    X_well = X[well == w]
    X[well == w] = X_well[np.argsort(depth[well == w])]
    depth[well == w] = np.sort(depth[well == w])


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. Different strategies could be used. We simply substitute them with the average PE value.
# 

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# ## Feature augmentation
# In this submission, we propose a feature augmentation strategy based on:
# - Computing feature spatial gradient.
# - Computing higher order features and interaction terms.
# 

# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth):
    
    # Augment features
    padded_rows = []
    X_aug = np.zeros((X.shape[0], X.shape[1]*2))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X[w_idx, :], X_aug_grad), axis=1)
        padded_rows.append(w_idx[-1])
        
    # Find padded rows
    #padded_rows = np.where(~X_aug[:, 7:].any(axis=1))[0]
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# Add higher degree terms and interaction terms to the model
deg = 2
poly = preprocessing.PolynomialFeatures(deg, interaction_only=False)
X_aug = poly.fit_transform(X_aug)
X_aug = X_aug[:,1:]


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
Loss_grid = ['exponential'] # ['deviance', 'exponential']
N_grid = [100] # [100]
M_grid = [10] # [5, 10, 15]
S_grid = [25] # [10, 25, 50, 75]
L_grid = [5] # [2, 3, 4, 5, 10, 25]
R_grid = [.1] # [.05, .1, .5]
Sub_grid = [1] # [0.5, 0.75, 1]
MED_grid = [1] # [0, 1]
param_grid = []
for N in N_grid:
    for M in M_grid:
        for S in S_grid:
            for L in L_grid:
                for R in R_grid:
                    for Sub in Sub_grid:
                        for MED in MED_grid:
                            for Loss in Loss_grid:
                                param_grid.append({'N':N, 'M':M, 'S':S, 'L':L,
                                                   'R':R, 'Sub':Sub, 'MED': MED, 'Loss':Loss})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss=param['Loss'],
                                        n_estimators=param['N'],
                                        learning_rate=param['R'], 
                                        max_features=param['M'],
                                        min_samples_leaf=param['L'],
                                        min_samples_split=param['S'],
                                        random_state=0,
                                        subsample=param['Sub'],
                                        max_leaf_nodes=None, 
                                        verbose=0), n_jobs=-1)
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    if param['MED']:
        for w in np.unique(well_v):
            y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)

    # Average score for this param
    score_param.append(np.mean(score_split))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0)


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well_ts)):
    X_ts_well = X_ts[well_ts == w]
    X_ts[well_ts == w] = X_ts_well[np.argsort(depth_ts[well_ts == w])]
    depth_ts[well_ts == w] = np.sort(depth_ts[well_ts == w])

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('well_data_with_facies_try03_v2.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)


# # Facies classification using Machine Learning #
# ## LA Team Submission 6 ## 
# ### _[Lukas Mosser](https://at.linkedin.com/in/lukas-mosser-9948b32b/en), [Alfredo De la Fuente](https://pe.linkedin.com/in/alfredodelafuenteb)_ ####
# 

# In this approach for solving the facies classfication problem ( https://github.com/seg/2016-ml-contest. ) we will explore the following statregies:
# - Features Exploration: based on [Paolo Bestagini's work](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb), we will consider imputation, normalization and augmentation routines for the initial features.
# - Model tuning: we use TPOT to come up with a good enough pipeline, and then tune the hyperparameters of the model obtained using HYPEROPT.
# 

# ## Packages and Libraries
# 

get_ipython().run_cell_magic('sh', '', 'pip install pandas\npip install scikit-learn\npip install tpot\npip install hyperopt\npip install xgboost')


from __future__ import print_function
import numpy as np
get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , StratifiedKFold
from classification_utilities import display_cm, display_adj_cm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.multiclass import OneVsOneClassifier
from scipy.signal import medfilt


# ## Data Preprocessing 
# 
# We procceed to run [Paolo Bestagini's routine](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb) to include a small window of values to acount for the spatial component in the log analysis, as well as the gradient information with respect to depth. This will be our prepared training dataset.
# 

#Load Data
data = pd.read_csv('../facies_vectors.csv')

# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# Store features and labels
X = data[feature_names].values 
y = data['Facies'].values 

# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values

# Fill 'PE' missing values with mean
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
    
        
def preprocess():
    
    # Preprocess data to use in model
    X_train_aux = []
    X_test_aux = []
    y_train_aux = []
    y_test_aux = []
    
    # For each data split
    for split in split_list:
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)

        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]

        # Select well labels for validation data
        well_v = well[split['val']]

        # Feature normalization
        scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_v = scaler.transform(X_v)

        X_train_aux.append( X_tr )
        X_test_aux.append( X_v )
        y_train_aux.append( y_tr )
        y_test_aux.append (  y_v )

        X_train = np.concatenate( X_train_aux )
        X_test = np.concatenate ( X_test_aux )
        y_train = np.concatenate ( y_train_aux )
        y_test = np.concatenate ( y_test_aux )
    
    return X_train , X_test , y_train , y_test 

X_train, X_test, y_train, y_test = preprocess()
y_train = y_train - 1 
y_test = y_test - 1 


# ## Data Analysis
# 
# In this section we will run a Cross Validation routine 
# 

import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.pipeline import make_pipeline


SEED = 314159265
VALID_SIZE = 0.2
TARGET = 'outcome'

# Scoring and optimization functions

def score(params):
    print("Training with params: ")
    print(params)
    #clf = xgb.XGBClassifier(**params) 
    #clf.fit(X_train, y_train)
    #y_predictions = clf.predict(X_test)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True)
    y_predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    
    score = f1_score (y_test, y_predictions , average ='micro')
    print("\tScore {0}\n\n".format(score))
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


def optimize(random_state=SEED):
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 150, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'mlogloss',
        'objective': 'multi:softmax',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'num_class' : 9,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,  max_evals=5)
    return best


best_hyperparams = optimize()
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    #clf = make_pipeline(make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)), XGBClassifier(learning_rate=0.73, max_depth=10, min_child_weight=10, n_estimators=500, subsample=0.27))
    #clf =  make_pipeline( KNeighborsClassifier(n_neighbors=5, weights="distance") ) 
    #clf = make_pipeline(MaxAbsScaler(),make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),ExtraTreesClassifier(criterion="entropy", max_features=0.0001, n_estimators=500))
    # * clf = make_pipeline( make_union(VotingClassifier([("est", BernoulliNB(alpha=60.0, binarize=0.26, fit_prior=True))]), FunctionTransformer(lambda X: X)),RandomForestClassifier(n_estimators=500))
    # ** clf = make_pipeline ( XGBClassifier(learning_rate=0.12, max_depth=3, min_child_weight=10, n_estimators=150, seed = 17, colsample_bytree = 0.9) )
    clf = clf = make_pipeline ( XGBClassifier(learning_rate=0.15, max_depth=8, min_child_weight=4, n_estimators=148, seed = SEED, colsample_bytree = 0.85, subsample = 0.9 , gamma = 0.75) )
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# ## Prediction
# 


#Load testing data
test_data = pd.read_csv('../validation_data_nofacies.csv')

# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) - 1

# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)

# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts)

# Save predicted labels
test_data['Facies'] = y_ts_hat + 1
test_data.to_csv('Prediction_XXX_Final.csv')








# # Facies classification using Machine Learning #
# ## LA Team Submission 5 ## 
# ### _[Lukas Mosser](https://at.linkedin.com/in/lukas-mosser-9948b32b/en), [Alfredo De la Fuente](https://pe.linkedin.com/in/alfredodelafuenteb)_ ####
# 

# In this approach for solving the facies classfication problem ( https://github.com/seg/2016-ml-contest. ) we will explore the following statregies:
# - Features Exploration: based on [Paolo Bestagini's work](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb), we will consider imputation, normalization and augmentation routines for the initial features.
# - Model tuning: 
# 

# ## Libraries
# 
# We will need to install the following libraries and packages.
# 

# %%sh
# pip install pandas
# pip install scikit-learn
# pip install tpot


from __future__ import print_function
import numpy as np
get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , StratifiedKFold
from classification_utilities import display_cm, display_adj_cm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import medfilt


# ## Data Preprocessing
# 

#Load Data
data = pd.read_csv('../facies_vectors.csv')

# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# Store features and labels
X = data[feature_names].values 
y = data['Facies'].values 

# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values

# Fill 'PE' missing values with mean
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# We procceed to run [Paolo Bestagini's routine](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb) to include a small window of values to acount for the spatial component in the log analysis, as well as the gradient information with respect to depth. This will be our prepared training dataset.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_aug, padded_rows = augment_features(X, well, depth)


# # Initialize model selection methods
# lpgo = LeavePGroupsOut(2)

# # Generate splits
# split_list = []
# for train, val in lpgo.split(X, y, groups=data['Well Name']):
#     hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
#     hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
#     if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
#         split_list.append({'train':train, 'val':val})


def preprocess():
    
    # Preprocess data to use in model
    X_train_aux = []
    X_test_aux = []
    y_train_aux = []
    y_test_aux = []
    
    # For each data split
    split = split_list[5]
        
    # Remove padded rows
    split_train_no_pad = np.setdiff1d(split['train'], padded_rows)

    # Select training and validation data from current split
    X_tr = X_aug[split_train_no_pad, :]
    X_v = X_aug[split['val'], :]
    y_tr = y[split_train_no_pad]
    y_v = y[split['val']]

    # Select well labels for validation data
    well_v = well[split['val']]

    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
        
    X_train_aux.append( X_tr )
    X_test_aux.append( X_v )
    y_train_aux.append( y_tr )
    y_test_aux.append (  y_v )
    
    X_train = np.concatenate( X_train_aux )
    X_test = np.concatenate ( X_test_aux )
    y_train = np.concatenate ( y_train_aux )
    y_test = np.concatenate ( y_test_aux )
    
    return X_train , X_test , y_train , y_test 


# ## Data Analysis
# 
# In this section we will run a Cross Validation routine 
# 

# from tpot import TPOTClassifier
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = preprocess()

# tpot = TPOTClassifier(generations=5, population_size=20, 
#                       verbosity=2,max_eval_time_mins=20,
#                       max_time_mins=100,scoring='f1_micro',
#                       random_state = 17)
# tpot.fit(X_train, y_train)
# print(tpot.score(X_test, y_test))
# tpot.export('FinalPipeline.py')


from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier


# Train and test a classifier

# Pass in the classifier so we can iterate over many seed later.
def train_and_test(X_tr, y_tr, X_v, well_v, clf):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# ## Prediction
# 

#Load testing data
test_data = pd.read_csv('../validation_data_nofacies.csv')

    # Train classifier
    #clf = make_pipeline(make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)), XGBClassifier(learning_rate=0.73, max_depth=10, min_child_weight=10, n_estimators=500, subsample=0.27))
    #clf =  make_pipeline( KNeighborsClassifier(n_neighbors=5, weights="distance") ) 
    #clf = make_pipeline(MaxAbsScaler(),make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),ExtraTreesClassifier(criterion="entropy", max_features=0.0001, n_estimators=500))
    # * clf = make_pipeline( make_union(VotingClassifier([("est", BernoulliNB(alpha=60.0, binarize=0.26, fit_prior=True))]), FunctionTransformer(lambda X: X)),RandomForestClassifier(n_estimators=500))

# # Prepare training data
# X_tr = X
# y_tr = y

# # Augment features
# X_tr, padded_rows = augment_features(X_tr, well, depth)

# # Removed padded rows
# X_tr = np.delete(X_tr, padded_rows, axis=0)
# y_tr = np.delete(y_tr, padded_rows, axis=0) 

# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values


    
y_pred = []
print('.' * 100)
for seed in range(100):
    np.random.seed(seed)

    # Make training data.
    X_train, padded_rows = augment_features(X, well, depth)
    y_train = y
    X_train = np.delete(X_train, padded_rows, axis=0)
    y_train = np.delete(y_train, padded_rows, axis=0) 

    # Train classifier  
    clf = make_pipeline(XGBClassifier(learning_rate=0.12,
                                      max_depth=3,
                                      min_child_weight=10,
                                      n_estimators=150,
                                      seed=seed,
                                      colsample_bytree=0.9))

    # Make blind data.
    X_test, _ = augment_features(X_ts, well_ts, depth_ts)

    # Train and test.
    y_ts_hat = train_and_test(X_train, y_train, X_test, well_ts, clf)
    
    # Collect result.
    y_pred.append(y_ts_hat)
    print('|', end='')
    
np.save('LA_Team_100_realizations.npy', y_pred)


# # Augment features
# X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)

# # Predict test labels
# y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts)

# # Save predicted labels
# test_data['Facies'] = y_ts_hat
# test_data.to_csv('Prediction_XX_Final.csv')





# # Facies classification using Machine Learning #
# ## LA Team Submission 6 ## 
# ### _[Lukas Mosser](https://at.linkedin.com/in/lukas-mosser-9948b32b/en), [Alfredo De la Fuente](https://pe.linkedin.com/in/alfredodelafuenteb)_ ####
# 

# In this approach for solving the facies classfication problem ( https://github.com/seg/2016-ml-contest. ) we will explore the following statregies:
# - Features Exploration: based on [Paolo Bestagini's work](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb), we will consider imputation, normalization and augmentation routines for the initial features.
# - Model tuning: 
# 

# ## Libraries
# 
# We will need to install the following libraries and packages.
# 

get_ipython().run_cell_magic('sh', '', 'pip install pandas\npip install scikit-learn\npip install tpot')


from __future__ import print_function
import numpy as np
get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , StratifiedKFold
from classification_utilities import display_cm, display_adj_cm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import medfilt

from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier


# ## Data Preprocessing
# 

#Load Data
data = pd.read_csv('../facies_vectors.csv')

# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# Store features and labels
X = data[feature_names].values 
y = data['Facies'].values 

# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# Fill 'PE' missing values with mean
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# We procceed to run [Paolo Bestagini's routine](https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try02.ipynb) to include a small window of values to acount for the spatial component in the log analysis, as well as the gradient information with respect to depth. This will be our prepared training dataset.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
    
        
def preprocess():
    
    # Preprocess data to use in model
    X_train_aux = []
    X_test_aux = []
    y_train_aux = []
    y_test_aux = []
    
    # For each data split
    split = split_list[5]
        
    # Remove padded rows
    split_train_no_pad = np.setdiff1d(split['train'], padded_rows)

    # Select training and validation data from current split
    X_tr = X_aug[split_train_no_pad, :]
    X_v = X_aug[split['val'], :]
    y_tr = y[split_train_no_pad]
    y_v = y[split['val']]

    # Select well labels for validation data
    well_v = well[split['val']]

    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
        
    X_train_aux.append( X_tr )
    X_test_aux.append( X_v )
    y_train_aux.append( y_tr )
    y_test_aux.append (  y_v )
    
    X_train = np.concatenate( X_train_aux )
    X_test = np.concatenate ( X_test_aux )
    y_train = np.concatenate ( y_train_aux )
    y_test = np.concatenate ( y_test_aux )
    
    return X_train , X_test , y_train , y_test 


# ## Data Analysis
# 
# In this section we will run a Cross Validation routine 
# 

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = preprocess()

tpot = TPOTClassifier(generations=5, population_size=100, 
                      verbosity=2, max_eval_time_mins=30,
                      max_time_mins=6*60, scoring='f1_micro',
                      random_state = 17)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
#tpot.export('FinalPipeline_LM_long_2.py')


from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    #clf = make_pipeline(make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)), XGBClassifier(learning_rate=0.73, max_depth=10, min_child_weight=10, n_estimators=500, subsample=0.27))
    #clf =  make_pipeline( KNeighborsClassifier(n_neighbors=5, weights="distance") ) 
    #clf = make_pipeline(MaxAbsScaler(),make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),ExtraTreesClassifier(criterion="entropy", max_features=0.0001, n_estimators=500))
    # * clf = make_pipeline( make_union(VotingClassifier([("est", BernoulliNB(alpha=60.0, binarize=0.26, fit_prior=True))]), FunctionTransformer(lambda X: X)),RandomForestClassifier(n_estimators=500))
    clf = make_pipeline(
        make_union(VotingClassifier([("est", BernoulliNB(alpha=0.41000000000000003, binarize=0.43, fit_prior=True))]), FunctionTransformer(lambda X: X)),
        StandardScaler(),
        RandomForestClassifier(n_estimators=500)
    )

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# ## Prediction
# 

#Load testing data
test_data = pd.read_csv('../validation_data_nofacies.csv')

# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 

# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)

# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts)

# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('Prediction_XXI_LM_Final.csv')





# # 05 - Facies Classifier
# 
# George Crowther
# 
# This is an extension / amalgamation of prior entries. The workflow remains not dissimilar to those completed previously, this is:
# - Load and set strings to integers
# - Cursory data examination, this workbook does not attempt to detail the full data analysis
# - Group data by well and brute force feature creation
#     - Feature creation focuses on bringing results from adjacent samples into features
#     - Look at some ratios between features
# - Used TPOT to train a classifier (exported_pipeline)
# - Feature creation and extraction on test dataset
# - Result prediction
# 

import pandas as pd
import bokeh.plotting as bk
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier, TPOTRegressor

import sys
sys.path.append('~/home/slygeorge/Documents/Python/SEG ML Competition')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

bk.output_notebook()


# Input file paths
train_path = '../training_data.csv'

# Read training data to dataframe
train = pd.read_csv(train_path)

# TPOT library requires that the target class is renamed to 'class'
train.rename(columns={'Facies': 'class'}, inplace=True)

well_names = train['Well Name']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']


train.head()


train.dropna().describe()


# Some quick-look plots, PE has been highlighted, as this appears to be missing from the alternative version of the training dataset
plots = []
for well, group in train.groupby('Well Name'):
    group = group.sort_values(by = 'Depth')
    plots.append(bk.figure(height = 500, width = 150))
    plots[-1].line(group['PE'], group['Depth'], color = 'blue')
    plots[-1].line(group['DeltaPHI'], group['Depth'], color = 'red')
    plots[-1].title.text = well
    
grid = bk.gridplot([plots])
bk.show(grid)


# Set string features to integers

for i, value in enumerate(train['Formation'].unique()):
    train.loc[train['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(train['Well Name'].unique()):
    train.loc[train['Well Name'] == value, 'Well Name'] = i


# Used to reassign index, initally after attempting to upsample results

train['orig_index'] = train.index


# Define resample factors
resample_factors = [2, 5, 10, 25, 50]

initial_columns = ['Formation', 'Well Name', 'Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
div_columns = ['Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']


# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []
above = []
below = []

for well, group in train.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = None)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = None)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = None)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = None)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = None)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        skew_frame = working_frame.rolling(window = f, center = True).skew().interpolate(method = 'index', limit_direction = 'both', limit = None)
        skew_frame.columns = ['Skew_{0}_{1}'.format(f, column) for column in skew_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['class'] = group['class']
    well_frame['Well Name'] = well
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    df = group.sort_values('Depth')
    u = df.shift(-1).fillna(method = 'ffill')
    b = df.shift(1).fillna(method = 'bfill')
    above.append(u[div_columns])
    below.append(b[div_columns])
    
    mean_frames.append(well_frame.fillna(method = 'bfill').fillna(method = 'ffill'))


# Concatenate all sub-frames together into single 'upsampled_frane'
frame = train
frame.index = frame['orig_index']
frame.drop(['orig_index', 'class', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
above_frame = pd.concat(above)
above_frame.columns = ['above_'+ column for column in above_frame.columns]
below_frame = pd.concat(below)
below_frame.columns = ['below_'+ column for column in below_frame.columns]
upsampled_frame = pd.concat((frame, rolling_frame, above_frame, below_frame), axis = 1)


# Features is the column set used for training the model
features = [feature for feature in upsampled_frame.columns if 'class' not in feature]


# Normalise dataset
std_scaler = preprocessing.StandardScaler().fit(upsampled_frame[features])

train_std = std_scaler.transform(upsampled_frame[features])

train_std_frame = upsampled_frame
for i, column in enumerate(features):
    train_std_frame.loc[:, column] = train_std[:, i]

upsampled_frame_std = train_std_frame


# Create ratios between features
div_columns = ['Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']

for feature in div_columns:
    for f in div_columns:
        if f == feature:
            continue
        upsampled_frame['{0}_{1}'.format(feature, f)] = upsampled_frame[f] / upsampled_frame[feature]


features = []
[features.append(column) for column in upsampled_frame.columns if 'class' not in column]
print(features)


train_f, test_f = train_test_split(upsampled_frame_std, test_size = 0.2, 
                                   random_state = 72)


# --------------------------
# TPOT Generated Model
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="entropy", max_features=0.36, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    DecisionTreeClassifier()
)

exported_pipeline.fit(train_f[features], train_f['class'])


exported_pipeline.score(test_f[features], test_f['class'])


result = exported_pipeline.predict(test_f[features])

from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm

conf = confusion_matrix(test_f['class'], result)
display_cm(conf, facies_labels, hide_zeros = True, display_metrics = True)

def accuracy(conf):
    total_correct = 0
    nb_classes = conf.shape[0]
    for i in np.arange(0, nb_classes):
        total_correct += conf[i][i]
    acc = total_correct / sum(sum(conf))
    return acc

print (accuracy(conf))

adjacent_facies = np.array([[1], [0, 2], [1], [4], [3, 5], [4, 6, 7], [5, 7], [5, 6, 8], [6, 7]])

def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0
    for i in np.arange(0, nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))

print(accuracy_adjacent(conf, adjacent_facies))            


# Now load and process the test data set, then predict using the 'exported_pipeline' model.
# 

test_path = '../validation_data_nofacies.csv'

# Read training data to dataframe
test = pd.read_csv(test_path)

# Set string features to integers

for i, value in enumerate(test['Formation'].unique()):
    test.loc[test['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(test['Well Name'].unique()):
    test.loc[test['Well Name'] == value, 'Well Name'] = i

# The first thing that will be done is to upsample and interpolate the training data,
# the objective here is to provide significantly more samples to train the regressor on and
# also to capture more of the sample interdependancy.
upsampled_arrays = []
test['orig_index'] = test.index

# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []
above = []
below = []

for well, group in test.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = None)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = None)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = None)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = None)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = None)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        skew_frame = working_frame.rolling(window = f, center = True).skew().interpolate(method = 'index', limit_direction = 'both', limit = None)
        skew_frame.columns = ['Skew_{0}_{1}'.format(f, column) for column in skew_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['Well Name'] = well
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    df = group.sort_values('Depth')
    u = df.shift(-1).fillna(method = 'ffill')
    b = df.shift(1).fillna(method = 'bfill')
    above.append(u[div_columns])
    below.append(b[div_columns])
    
    mean_frames.append(well_frame.fillna(method = 'bfill').fillna(method = 'ffill'))
    
frame = test
frame.index = frame['orig_index']
frame.drop(['orig_index', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
above_frame = pd.concat(above)
above_frame.columns = ['above_'+ column for column in above_frame.columns]
below_frame = pd.concat(below)
below_frame.columns = ['below_'+ column for column in below_frame.columns]
upsampled_frame = pd.concat((frame, rolling_frame, above_frame, below_frame), axis = 1)

features = [feature for feature in upsampled_frame.columns if 'class' not in feature]

std_scaler = preprocessing.StandardScaler().fit(upsampled_frame[features])
train_std = std_scaler.transform(upsampled_frame[features])

train_std_frame = upsampled_frame
for i, column in enumerate(features):
    train_std_frame.loc[:, column] = train_std[:, i]

upsampled_frame_std = train_std_frame

for feature in div_columns:
    for f in div_columns:
        if f == feature:
            continue
        upsampled_frame['{0}_{1}'.format(feature, f)] = upsampled_frame[f] / upsampled_frame[feature]
        
features = [feature for feature in upsampled_frame.columns if 'class' not in feature]


# Predict result on full sample set
result = exported_pipeline.predict(upsampled_frame[features])
# Add result to test set
upsampled_frame['Facies'] = result
# Output to csv
upsampled_frame.to_csv('05 - Well Facies Prediction - Test Data Set.csv')





# # 06 - Facies Classifier
# 
# George Crowther
# 
# This is an extension / amalgamation of prior entries. The workflow remains not dissimilar to those completed previously, this is:
# - Load and set strings to integers
# - Cursory data examination, this workbook does not attempt to detail the full data analysis
# - Group data by well and brute force feature creation
#     - Feature creation focuses on bringing results from adjacent samples into features
#     - Look at some ratios between features
# - Leaving out two wells at a time, use TPOT to generate a pipeline for prediction.
# - Modal vote on fitted model predicting on the test data set.
# 

import pandas as pd
import bokeh.plotting as bk
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier, TPOTRegressor

import sys
sys.path.append('~/home/slygeorge/Documents/Python/SEG ML Competition')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

bk.output_notebook()


from scipy.stats import mode
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFwe, SelectKBest, f_classif, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, Binarizer, Normalizer, StandardScaler
from xgboost import XGBClassifier


models = [
    make_pipeline(
    MinMaxScaler(),
    XGBClassifier(learning_rate=0.02, max_depth=5, min_child_weight=20, n_estimators=500, subsample=0.19)
),
    make_pipeline(
    make_union(VotingClassifier([("est", LogisticRegression(C=0.13, dual=False, penalty="l1"))]), FunctionTransformer(lambda X: X)),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    Binarizer(threshold=0.72),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    make_union(VotingClassifier([("est", GradientBoostingClassifier(learning_rate=1.0, max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    BernoulliNB(alpha=28.0, binarize=0.85, fit_prior=True)
),
    make_pipeline(
    Normalizer(norm="l1"),
    make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),
    SelectKBest(k=47, score_func=f_classif),
    SelectFwe(alpha=0.05, score_func=f_classif),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    make_union(VotingClassifier([("est", LinearSVC(C=0.26, dual=False, penalty="l2"))]), FunctionTransformer(lambda X: X)),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    Normalizer(norm="l2"),
    make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="entropy", max_features=0.3, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    GaussianNB()
),
    make_pipeline(
    make_union(VotingClassifier([("est", BernoulliNB(alpha=49.0, binarize=0.06, fit_prior=True))]), FunctionTransformer(lambda X: X)),
    StandardScaler(),
    make_union(VotingClassifier([("est", GradientBoostingClassifier(learning_rate=0.87, max_features=0.87, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    ExtraTreesClassifier(criterion="entropy", max_features=0.001, n_estimators=500)
),
    make_pipeline(
    make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),
    BernoulliNB(alpha=1e-06, binarize=0.09, fit_prior=True)
),
    make_pipeline(
    Normalizer(norm="max"),
    MinMaxScaler(),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    SelectPercentile(percentile=18, score_func=f_classif),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    SelectKBest(k=50, score_func=f_classif),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    XGBClassifier(learning_rate=0.51, max_depth=10, min_child_weight=20, n_estimators=500, subsample=1.0)
),
    make_pipeline(
    make_union(VotingClassifier([("est", KNeighborsClassifier(n_neighbors=5, weights="uniform"))]), FunctionTransformer(lambda X: X)),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    StandardScaler(),
    SelectPercentile(percentile=19, score_func=f_classif),
    LinearSVC(C=0.02, dual=False, penalty="l1")
),
    make_pipeline(
    XGBClassifier(learning_rate=0.01, max_depth=10, min_child_weight=20, n_estimators=500, subsample=0.36)
)]


train_path = '../training_data.csv'
test_path = '../validation_data_nofacies.csv'

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']


def feature_extraction(file_path, retain_class = True):

    # Read training data to dataframe
    test = pd.read_csv(file_path)
    
    if 'Facies' in test.columns:
        test.rename(columns={'Facies': 'class'}, inplace=True)

    # Set string features to integers

    for i, value in enumerate(test['Formation'].unique()):
        test.loc[test['Formation'] == value, 'Formation'] = i

    for i, value in enumerate(test['Well Name'].unique()):
        test.loc[test['Well Name'] == value, 'Well Name'] = i

    # The first thing that will be done is to upsample and interpolate the training data,
    # the objective here is to provide significantly more samples to train the regressor on and
    # also to capture more of the sample interdependancy.
    upsampled_arrays = []
    test['orig_index'] = test.index

    # Use rolling windows through upsampled frame, grouping by well name.

    # Empty list to hold frames
    mean_frames = []
    above = []
    below = []

    for well, group in test.groupby('Well Name'):
        # Empty list to hold rolling frames
        constructor_list = []
        for f in resample_factors:

            working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
           'RELPOS']]

            mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = None)
            mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
            max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = None)
            max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
            min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = None)
            min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
            std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = None)
            std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
            var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = None)
            var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
            diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None)
            diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
            rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None).sort_index()
            rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
            skew_frame = working_frame.rolling(window = f, center = True).skew().interpolate(method = 'index', limit_direction = 'both', limit = None)
            skew_frame.columns = ['Skew_{0}_{1}'.format(f, column) for column in skew_frame.columns]

            f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)

            constructor_list.append(f_frame)

        well_frame = pd.concat(constructor_list, axis = 1)
        well_frame['Well Name'] = well
        # orig index is holding the original index locations, to make extracting the results trivial
        well_frame['orig_index'] = group['orig_index']
        df = group.sort_values('Depth')
        u = df.shift(-1).fillna(method = 'ffill')
        b = df.shift(1).fillna(method = 'bfill')
        above.append(u[div_columns])
        below.append(b[div_columns])

        mean_frames.append(well_frame.fillna(method = 'bfill').fillna(method = 'ffill'))

    frame = test
    frame.index = frame['orig_index']
    frame.drop(['orig_index', 'Well Name'], axis = 1, inplace = True)

    for f in mean_frames:
        f.index = f['orig_index']

    rolling_frame = pd.concat(mean_frames, axis = 0)
    above_frame = pd.concat(above)
    above_frame.columns = ['above_'+ column for column in above_frame.columns]
    below_frame = pd.concat(below)
    below_frame.columns = ['below_'+ column for column in below_frame.columns]
    upsampled_frame = pd.concat((frame, rolling_frame, above_frame, below_frame), axis = 1)

    features = [feature for feature in upsampled_frame.columns if 'class' not in feature]

    std_scaler = preprocessing.StandardScaler().fit(upsampled_frame[features])
    train_std = std_scaler.transform(upsampled_frame[features])

    train_std_frame = upsampled_frame
    for i, column in enumerate(features):
        train_std_frame.loc[:, column] = train_std[:, i]

    upsampled_frame_std = train_std_frame

    for feature in div_columns:
        for f in div_columns:
            if f == feature:
                continue
            upsampled_frame['{0}_{1}'.format(feature, f)] = upsampled_frame[f] / upsampled_frame[feature]
 
    return upsampled_frame_std, features

train_data_set, features = feature_extraction(train_path)
test_data_set, test_features = feature_extraction(test_path)


train_data_set.head()


lpgo = LeavePGroupsOut(2)

split_list = []
fitted_models = []

for train, val in lpgo.split(train_data_set[features], 
                             train_data_set['class'], 
                             groups = train_data_set['Well Name']):
    hist_tr = np.histogram(train_data_set.loc[train, 'class'], 
                           bins = np.arange(len(facies_labels) + 1) + 0.5)
    hist_val = np.histogram(train_data_set.loc[val, 'class'],
                           bins = np.arange(len(facies_labels) + 1) + 0.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train': train, 'val': val})
        
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (train_data_set['Well Name'].loc[split['train']].unique()))
    print('    validation: %s' % (train_data_set['Well Name'].loc[split['val']].unique()))


fitted_models = []
r = []

for i, split in enumerate(split_list):

    # Select training and validation data from current split
    X_tr = train_data_set.loc[split['train'], features]
    X_v = train_data_set.loc[split['val'], features]
    y_tr = train_data_set.loc[split['train'], 'class']
    y_v = train_data_set.loc[split['val'], 'class']

    # Fit model from split
    fitted_models.append(models[i].fit(X_tr, y_tr))
    
    # Predict for model
    r.append(fitted_models[-1].predict(test_data_set[test_features]))
    
results = mode(np.vstack(r))[0][0]

test_data_set['Facies'] = results


test_data_set.iloc[:, ::-1].head()


test_data_set.iloc[:, ::-1].to_csv('06 - Combined Models.csv')





# # Facies classification utilizing an Adaptive Boosted Random Forest
# 

# [Ryan Thielke](http://www.linkedin.com/in/ryan-thielke-b987012a)
# 
# 
# In the following, we provide a possible solution to the facies classification problem described in https://github.com/seg/2016-ml-contest.
# 

# ## Exploring the data
# 

import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..")

#Import standard pydata libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


filename = '../facies_vectors.csv'
training_data = pd.read_csv(filename)
#training_data['Well Name'] = training_data['Well Name'].astype('category')
#training_data['Formation'] = training_data['Formation'].astype('category')
training_data['train'] = 1
training_data.describe()


validation_data = pd.read_csv("../validation_data_nofacies.csv")
#validation_data['Well Name'] = validation_data['Well Name'].astype('category')
#validation_data['Formation'] = validation_data['Formation'].astype('category')
validation_data['train'] = 0
validation_data.describe()


all_data = training_data.append(validation_data)
all_data.describe()


#Visualize the distribution of facies for each well
wells = training_data['Well Name'].unique()

fig, ax = plt.subplots(5,2, figsize=(20,20))
for i, well in enumerate(wells):
    row = i % ax.shape[0]
    column = i // ax.shape[0]
    counts = training_data[training_data['Well Name']==well].Facies.value_counts()
    data_for_well = [counts[j] if j in counts.index else 0 for j in range(1,10)]
    ax[row, column].bar(range(1,10), data_for_well, align='center')
    ax[row, column].set_title("{well}".format(well=well))
    ax[row, column].set_ylabel("Counts")
    ax[row, column].set_xticks(range(1,10))

plt.show()
    


plt.figure(figsize=(10,10))
sns.heatmap(training_data.drop(['Formation', 'Well Name'], axis=1).corr())


# # Feature Engineering
# 
# Here we will do a couple things to clean the data and attempt to create new features for our model to consume.
# 
# First, we will smooth the PE and GR features.
# Second, we replace missing PE values with the mean of the entire dataset (might want to investigate other methods)
# Last, we will encode the formations into integer values
# 

avg_PE_facies = training_data[['Facies', 'PE']].groupby('Facies').mean()
avg_PE_facies = avg_PE_facies.to_dict()
all_data['PE2'] = all_data.Facies.map(avg_PE_facies['PE'])


dfs = []
for well in all_data['Well Name'].unique():
    df = all_data[all_data['Well Name']==well].copy(deep=True)
    df.sort_values('Depth', inplace=True)
    for col in ['PE', 'GR']:
        smooth_col = 'smooth_'+col
        df[smooth_col] = pd.rolling_mean(df[col], window=10)
        df[smooth_col].fillna(method='ffill', inplace=True)
        df[smooth_col].fillna(method='bfill', inplace=True)
    dfs.append(df)
all_data = pd.concat(dfs)
all_data['PE'] = all_data.PE.fillna(all_data.PE2)
all_data['smooth_PE'] = all_data.smooth_PE.fillna(all_data.PE2)
formation_encoder = dict(zip(all_data.Formation.unique(), range(len(all_data.Formation.unique()))))
all_data['enc_formation'] = all_data.Formation.map(formation_encoder)


def to_binary_vec(value, vec_length):
    vec = np.zeros(vec_length)
    vec[value] = 1
    return vec


dfs = list()
for well in all_data['Well Name'].unique():
    tmp_df = all_data[all_data['Well Name'] == well].copy(deep=True)
    tmp_df.sort_values('Depth', inplace=True)
    for feature in ['Depth', 'ILD_log10', 'DeltaPHI', 'PHIND', 'smooth_PE', 'smooth_GR']:
        tmp_df['3prev_'+feature] = tmp_df[feature] / tmp_df[feature].shift(4)
        #tmp_df['2prev_'+feature] = tmp_df[feature] / tmp_df[feature].shift(-1)
        
        tmp_df['3prev_'+feature].fillna(method='bfill', inplace=True)
        #tmp_df['2prev_'+feature].fillna(method='ffill', inplace=True)
    
        tmp_df['3prev_'+feature].replace([np.inf, -np.inf], 0, inplace=True)
        #tmp_df['2prev_'+feature].replace([np.inf, -np.inf], 0, inplace=True)
        
    tmp_df['3prev_enc'] = tmp_df['enc_formation'].shift(3).fillna(method='bfill')
    tmp_df['2prev_enc'] = tmp_df['enc_formation'].shift(2).fillna(method='bfill')
    dfs.append(tmp_df)
all_data = pd.concat(dfs)


all_data.columns


#Let's build a model
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, cross_validation 
from classification_utilities import display_cm


#We will take a look at an F1 score for each well
estimators=200
learning_rate=.01
random_state=0
facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']

title_length = 20 

training_data = all_data[all_data.train==1]
scores = list()

wells = training_data['Well Name'].unique()
for well in wells:
    blind = training_data[training_data['Well Name']==well]
    train = training_data[(training_data['Well Name']!=well)]
    
    train_X = train.drop(['Formation', 'Well Name', 'Facies', 'Depth', 'PE2', 'train'], axis=1)
    train_Y = train.Facies.values
    test_X = blind.drop(['Formation', 'Well Name', 'Facies', 'Depth', 'PE2', 'train'], axis=1)
    test_Y = blind.Facies.values
    
    clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=200), n_estimators=200, learning_rate=learning_rate, random_state=random_state, algorithm='SAMME.R')
    
    clf.fit(train_X,train_Y)
    print(clf.feature_importances_)
    pred_Y = clf.predict(test_X)
    f1 = metrics.f1_score(test_Y, pred_Y, average='micro')
    scores.append(f1)
    print("*"*title_length)
    print("{well}={f1:.4f}".format(well=well,f1=f1))
    print("*"*title_length)
print("Avg F1: {score}".format(score=sum(scores)/len(scores)))


train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(training_data.drop(['Formation', 'Well Name','Facies', 'Depth', 'PE2', 'train'], axis=1), training_data.Facies.values, test_size=.2)


print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=estimators), n_estimators=estimators, random_state=0,learning_rate=learning_rate, algorithm='SAMME.R')
clf.fit(train_X, train_Y)
pred_Y = clf.predict(test_X)
cm = metrics.confusion_matrix(y_true=test_Y, y_pred=pred_Y)
display_cm(cm, facies_labels, display_metrics=True)


validation_data = all_data[all_data.train==0]


validation_data.describe()


X = training_data.drop(['Formation', 'Well Name', 'Depth','Facies', 'train', 'PE2'], axis=1)
Y = training_data.Facies.values
test_X = validation_data.drop(['Formation', 'Well Name', 'Depth', 'train', 'PE2', 'Facies'], axis=1)

clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=estimators), n_estimators=estimators, learning_rate=learning_rate, random_state=0)
clf.fit(X,Y)
predicted_facies = clf.predict(test_X)
validation_data['Facies'] = predicted_facies


validation_data.to_csv("Kr1m_SEG_ML_Attempt2.csv", index=False)





# # Facies classification utilizing an Adaptive Boosted Random Forest
# 

# [Ryan Thielke](http://www.linkedin.com/in/ryan-thielke-b987012a)
# 
# 
# In the following, we provide a possible solution to the facies classification problem described in https://github.com/seg/2016-ml-contest.
# 

# ## Exploring the data
# 

import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..")

#Import standard pydata libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


filename = '../facies_vectors.csv'
training_data = pd.read_csv(filename)
#training_data['Well Name'] = training_data['Well Name'].astype('category')
#training_data['Formation'] = training_data['Formation'].astype('category')
training_data['train'] = 1
training_data.describe()


validation_data = pd.read_csv("../validation_data_nofacies.csv")
#validation_data['Well Name'] = validation_data['Well Name'].astype('category')
#validation_data['Formation'] = validation_data['Formation'].astype('category')
validation_data['train'] = 0
validation_data.describe()


all_data = training_data.append(validation_data)
all_data.describe()


#Visualize the distribution of facies for each well
wells = training_data['Well Name'].unique()

fig, ax = plt.subplots(5,2, figsize=(20,20))
for i, well in enumerate(wells):
    row = i % ax.shape[0]
    column = i // ax.shape[0]
    counts = training_data[training_data['Well Name']==well].Facies.value_counts()
    data_for_well = [counts[j] if j in counts.index else 0 for j in range(1,10)]
    ax[row, column].bar(range(1,10), data_for_well, align='center')
    ax[row, column].set_title("{well}".format(well=well))
    ax[row, column].set_ylabel("Counts")
    ax[row, column].set_xticks(range(1,10))

plt.show()
    


plt.figure(figsize=(10,10))
sns.heatmap(training_data.drop(['Formation', 'Well Name'], axis=1).corr())


# # Feature Engineering
# 
# Here we will do a couple things to clean the data and attempt to create new features for our model to consume.
# 
# First, we will smooth the PE and GR features.
# Second, we replace missing PE values with the mean of the entire dataset (might want to investigate other methods)
# Last, we will encode the formations into integer values
# 


dfs = []
for well in all_data['Well Name'].unique():
    df = all_data[all_data['Well Name']==well].copy(deep=True)
    df.sort_values('Depth', inplace=True)
    for col in ['PE']:
        smooth_col = 'smooth_'+col
        df[smooth_col] = pd.rolling_mean(df[col], window=25)
        df[smooth_col].fillna(method='ffill', inplace=True)
        df[smooth_col].fillna(method='bfill', inplace=True)
    dfs.append(df)
all_data = pd.concat(dfs)

all_data['PE'] = all_data.PE.fillna(all_data.PE.mean())
all_data['smooth_PE'] = all_data.smooth_PE.fillna(all_data.smooth_PE.mean())
formation_encoder = dict(zip(all_data.Formation.unique(), range(len(all_data.Formation.unique()))))
all_data['enc_formation'] = all_data.Formation.map(formation_encoder)


all_data.columns


from sklearn import preprocessing

feature_names = all_data.drop(['Well Name', 'train', 'Depth', 'Formation', 'enc_formation', 'Facies'], axis=1).columns
train_labels = all_data.train.tolist()
facies_labels = all_data.Facies.tolist()
well_names = all_data['Well Name'].tolist()
depths = all_data.Depth.tolist()

scaler = preprocessing.StandardScaler().fit(all_data.drop(['Well Name', 'train', 'Depth', 'Formation', 'enc_formation', 'Facies'], axis=1))
scaled_features = scaler.transform(all_data.drop(['Well Name', 'train', 'Depth', 'Formation', 'enc_formation', 'Facies'], axis=1))

scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
scaled_df['train'] = train_labels
scaled_df['Facies'] = facies_labels
scaled_df['Well Name'] = well_names
scaled_df['Depth'] = depths


def to_binary_vec(value, vec_length):
    vec = np.zeros(vec_length+1)
    vec[value] = 1
    return vec

catagorical_vars = []

for i in all_data.enc_formation:
    vec = to_binary_vec(i, all_data.enc_formation.max())
    catagorical_vars.append(vec)
    
catagorical_vars = np.array(catagorical_vars)

for i in range(catagorical_vars.shape[1]):
    scaled_df['f'+str(i)] = catagorical_vars[:,i]



'''
dfs = list()
for well in all_data['Well Name'].unique():
    tmp_df = all_data[all_data['Well Name'] == well].copy(deep=True)
    tmp_df.sort_values('Depth', inplace=True)
    for feature in ['PE', 'GR']:
        tmp_df['3'+feature] = tmp_df[feature] / tmp_df[feature].shift(1)
        
        tmp_df['3'+feature].fillna(0, inplace=True)
        
    dfs.append(tmp_df)
scaled_df = pd.concat(dfs)
'''


#Let's build a model
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, cross_validation 
from classification_utilities import display_cm
import xgboost as xgb


scaled_df


import xgboost as xgb
#We will take a look at an F1 score for each well
estimators=200
learning_rate=.01
random_state=0
facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']

title_length = 20 

training_data = scaled_df[scaled_df.train==1]
scores = list()

wells = training_data['Well Name'].unique()
for well in wells:
    blind = training_data[training_data['Well Name']==well]
    train = training_data[(training_data['Well Name']!=well)]
    
    train_X = train.drop(['Well Name', 'Facies', 'Depth', 'train'], axis=1)
    train_Y = train.Facies.values
    test_X = blind.drop(['Well Name', 'Facies', 'Depth', 'train'], axis=1)
    test_Y = blind.Facies.values
    
    gcf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.01)
    gcf.fit(train_X,train_Y)
    pred_Y = gcf.predict(test_X)
    f1 = metrics.f1_score(test_Y, pred_Y, average='micro')
    scores.append(f1)
    print("*"*title_length)
    print("{well}={f1:.4f}".format(well=well,f1=f1))
    print("*"*title_length)
print("Avg F1: {score}".format(score=sum(scores)/len(scores)))


train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(training_data.drop(['Well Name', 'Facies', 'Depth', 'train'], axis=1), training_data.Facies.values, test_size=.2)


print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


gcf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05)
gcf.fit(train_X,train_Y)
pred_Y = gcf.predict(test_X)
cm = metrics.confusion_matrix(y_true=test_Y, y_pred=pred_Y)
display_cm(cm, facies_labels, display_metrics=True)


validation_data = scaled_df[scaled_df.train==0]


validation_data.describe()


X = training_data.drop(['Well Name', 'Facies', 'Depth', 'train'], axis=1)
Y = training_data.Facies.values
test_X = validation_data.drop(['Well Name', 'Facies', 'Depth', 'train'], axis=1)

gcf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.01)
gcf.fit(X,Y)
pred_Y = gcf.predict(test_X)

validation_data['Facies'] = pred_Y


validation_data.to_csv("Kr1m_SEG_ML_Attempt4.csv", index=False)


validation_data.describe()





# # Facies classification utilizing an Adaptive Boosted Random Forest
# 

# [Ryan Thielke](http://www.linkedin.com/in/ryan-thielke-b987012a)
# 
# 
# In the following, we provide a possible solution to the facies classification problem described in https://github.com/seg/2016-ml-contest.
# 

# ## Exploring the data
# 

import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..")

#Import standard pydata libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


filename = '../facies_vectors.csv'
training_data = pd.read_csv(filename)
training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')                            
training_data.describe()


#Visualize the distribution of facies for each well
wells = training_data['Well Name'].unique()

fig, ax = plt.subplots(5,2, figsize=(20,20))
for i, well in enumerate(wells):
    row = i % ax.shape[0]
    column = i // ax.shape[0]
    counts = training_data[training_data['Well Name']==well].Facies.value_counts()
    data_for_well = [counts[j] if j in counts.index else 0 for j in range(1,10)]
    ax[row, column].bar(range(1,10), data_for_well, align='center')
    ax[row, column].set_title("{well}".format(well=well))
    ax[row, column].set_ylabel("Counts")
    ax[row, column].set_xticks(range(1,10))

plt.show()
    


plt.figure(figsize=(10,10))
sns.heatmap(training_data.drop(['Formation', 'Well Name'], axis=1).corr())


# # Feature Engineering
# 
# Here we will do a couple things to clean the data and attempt to create new features for our model to consume.
# 
# First, we will smooth the PE and GR features.
# Second, we replace missing PE values with the mean of the entire dataset (might want to investigate other methods)
# Last, we will encode the formations into integer values
# 

dfs = []
for well in training_data['Well Name'].unique():
    df = training_data[training_data['Well Name']==well].copy(deep=True)
    df.sort_values('Depth', inplace=True)
    for col in ['PE', 'GR']:
        smooth_col = 'smooth_'+col
        df[smooth_col] = pd.rolling_mean(df[col], window=25)
        df[smooth_col].fillna(method='ffill', inplace=True)
        df[smooth_col].fillna(method='bfill', inplace=True)
    dfs.append(df)
training_data = pd.concat(dfs)
pe_mean = training_data.PE.mean()
sm_pe_mean = training_data.smooth_PE.mean()
training_data['PE'] = training_data.PE.replace({np.nan:pe_mean})
training_data['smooth_PE'] = training_data['smooth_PE'].replace({np.nan:sm_pe_mean})
formation_encoder = dict(zip(training_data.Formation.unique(), range(len(training_data.Formation.unique()))))
training_data['enc_formation'] = training_data.Formation.map(formation_encoder)


training_data.describe()


# ## Building the model and parameter tuning
# 
# In the section below we will create a Adaptive Boosted Random Forest Classifier from the Scikit-Learn ML Library
# 

#Let's build a model
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, cross_validation 
from classification_utilities import display_cm


#We will take a look at an F1 score for each well
n_estimators=100
learning_rate=.01
random_state=0
facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']

title_length = 20 

wells = training_data['Well Name'].unique()
for well in wells:
    blind = training_data[training_data['Well Name']==well]
    train = training_data[(training_data['Well Name']!=well)]
    
    train_X = train.drop(['Formation', 'Well Name', 'Depth', 'Facies'], axis=1)
    train_Y = train.Facies.values
    test_X = blind.drop(['Formation', 'Well Name', 'Facies', 'Depth'], axis=1)
    test_Y = blind.Facies.values
    
    clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=200, learning_rate=learning_rate, random_state=random_state, algorithm='SAMME.R')
    clf.fit(X=train_X, y=train_Y)
    pred_Y = clf.predict(test_X)
    f1 = metrics.f1_score(test_Y, pred_Y, average='micro')
    print("*"*title_length)
    print("{well}={f1:.4f}".format(well=well,f1=f1))
    print("*"*title_length)


train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(training_data.drop(['Formation', 'Well Name','Facies', 'Depth'], axis=1), training_data.Facies.values, test_size=.2)


print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=200, learning_rate=learning_rate, random_state=0, algorithm='SAMME.R')

clf.fit(train_X, train_Y)
pred_Y = clf.predict(test_X)
cm = metrics.confusion_matrix(y_true=test_Y, y_pred=pred_Y)
display_cm(cm, facies_labels, display_metrics=True)


validation_data = pd.read_csv("../validation_data_nofacies.csv")


dfs = []
for well in validation_data['Well Name'].unique():
    df = validation_data[validation_data['Well Name']==well].copy(deep=True)
    df.sort_values('Depth', inplace=True)
    for col in ['PE', 'GR']:
        smooth_col = 'smooth_'+col
        df[smooth_col] = pd.rolling_mean(df[col], window=25)
        df[smooth_col].fillna(method='ffill', inplace=True)
        df[smooth_col].fillna(method='bfill', inplace=True)
    dfs.append(df)
validation_data = pd.concat(dfs)

validation_data['enc_formation'] = validation_data.Formation.map(formation_encoder)
validation_data.describe()


X = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
Y = training_data.Facies.values
test_X = validation_data.drop(['Formation', 'Well Name', 'Depth'], axis=1)

clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=200, learning_rate=learning_rate, random_state=0)
clf.fit(X,Y)
predicted_facies = clf.predict(test_X)
validation_data['Facies'] = predicted_facies


validation_data.to_csv("Kr1m_SEG_ML_Attempt1.csv")





# # Facies classification utilizing an Adaptive Boosted Random Forest
# 

# [Ryan Thielke](http://www.linkedin.com/in/ryan-thielke-b987012a)
# 
# 
# In the following, we provide a possible solution to the facies classification problem described in https://github.com/seg/2016-ml-contest.
# 

# ## Exploring the data
# 

import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..")

#Import standard pydata libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


filename = '../facies_vectors.csv'
training_data = pd.read_csv(filename)
#training_data['Well Name'] = training_data['Well Name'].astype('category')
#training_data['Formation'] = training_data['Formation'].astype('category')
training_data['train'] = 1
training_data.describe()


validation_data = pd.read_csv("../validation_data_nofacies.csv")
#validation_data['Well Name'] = validation_data['Well Name'].astype('category')
#validation_data['Formation'] = validation_data['Formation'].astype('category')
validation_data['train'] = 0
validation_data.describe()


all_data = training_data.append(validation_data)
all_data.describe()


#Visualize the distribution of facies for each well
wells = training_data['Well Name'].unique()

fig, ax = plt.subplots(5,2, figsize=(20,20))
for i, well in enumerate(wells):
    row = i % ax.shape[0]
    column = i // ax.shape[0]
    counts = training_data[training_data['Well Name']==well].Facies.value_counts()
    data_for_well = [counts[j] if j in counts.index else 0 for j in range(1,10)]
    ax[row, column].bar(range(1,10), data_for_well, align='center')
    ax[row, column].set_title("{well}".format(well=well))
    ax[row, column].set_ylabel("Counts")
    ax[row, column].set_xticks(range(1,10))

plt.show()
    


plt.figure(figsize=(10,10))
sns.heatmap(training_data.drop(['Formation', 'Well Name'], axis=1).corr())


# # Feature Engineering
# 
# Here we will do a couple things to clean the data and attempt to create new features for our model to consume.
# 
# First, we will smooth the PE and GR features.
# Second, we replace missing PE values with the mean of the entire dataset (might want to investigate other methods)
# Last, we will encode the formations into integer values
# 


dfs = []
for well in all_data['Well Name'].unique():
    df = all_data[all_data['Well Name']==well].copy(deep=True)
    df.sort_values('Depth', inplace=True)
    for col in ['PE']:
        smooth_col = 'smooth_'+col
        df[smooth_col] = pd.rolling_mean(df[col], window=25)
        df[smooth_col].fillna(method='ffill', inplace=True)
        df[smooth_col].fillna(method='bfill', inplace=True)
    dfs.append(df)
all_data = pd.concat(dfs)

all_data['PE'] = all_data.PE.fillna(all_data.PE.mean())
all_data['smooth_PE'] = all_data.smooth_PE.fillna(all_data.smooth_PE.mean())
formation_encoder = dict(zip(all_data.Formation.unique(), range(len(all_data.Formation.unique()))))
all_data['enc_formation'] = all_data.Formation.map(formation_encoder)


all_data.columns


from sklearn import preprocessing

feature_names = all_data.drop(['Well Name', 'train', 'Depth', 'Formation', 'enc_formation', 'Facies'], axis=1).columns
train_labels = all_data.train.tolist()
facies_labels = all_data.Facies.tolist()
well_names = all_data['Well Name'].tolist()
depths = all_data.Depth.tolist()

scaler = preprocessing.StandardScaler().fit(all_data.drop(['Well Name', 'train', 'Depth', 'Formation', 'enc_formation', 'Facies'], axis=1))
scaled_features = scaler.transform(all_data.drop(['Well Name', 'train', 'Depth', 'Formation', 'enc_formation', 'Facies'], axis=1))

scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
scaled_df['train'] = train_labels
scaled_df['Facies'] = facies_labels
scaled_df['Well Name'] = well_names
scaled_df['Depth'] = depths


def to_binary_vec(value, vec_length):
    vec = np.zeros(vec_length+1)
    vec[value] = 1
    return vec

catagorical_vars = []

for i in all_data.enc_formation:
    vec = to_binary_vec(i, all_data.enc_formation.max())
    catagorical_vars.append(vec)
    
catagorical_vars = np.array(catagorical_vars)

for i in range(catagorical_vars.shape[1]):
    scaled_df['f'+str(i)] = catagorical_vars[:,i]



dfs = list()
for well in all_data['Well Name'].unique():
    tmp_df = all_data[all_data['Well Name'] == well].copy(deep=True)
    tmp_df.sort_values('Depth', inplace=True)
    for feature in ['PE', 'GR']:
        tmp_df['3'+feature] = tmp_df[feature] / tmp_df[feature].shift(1)
        
        tmp_df['3'+feature].fillna(0, inplace=True)
        
    dfs.append(tmp_df)
scaled_df = pd.concat(dfs)


#Let's build a model
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, cross_validation 
from classification_utilities import display_cm
import xgboost as xgb


import xgboost as xgb
#We will take a look at an F1 score for each well
estimators=200
learning_rate=.01
random_state=0
facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']

title_length = 20 

training_data = scaled_df[scaled_df.train==1]
scores = list()

wells = training_data['Well Name'].unique()
for well in wells:
    blind = training_data[training_data['Well Name']==well]
    train = training_data[(training_data['Well Name']!=well)]
    
    train_X = train.drop(['Well Name', 'Facies', 'Depth', 'train', 'Formation'], axis=1)
    train_Y = train.Facies.values
    test_X = blind.drop(['Well Name', 'Facies', 'Depth', 'train', 'Formation'], axis=1)
    test_Y = blind.Facies.values
    
    gcf = xgb.XGBClassifier(n_estimators=2000, learning_rate=learning_rate)
    gcf.fit(train_X,train_Y)
    pred_Y = gcf.predict(test_X)
    f1 = metrics.f1_score(test_Y, pred_Y, average='micro')
    scores.append(f1)
    print("*"*title_length)
    print("{well}={f1:.4f}".format(well=well,f1=f1))
    print("*"*title_length)
print("Avg F1: {score}".format(score=sum(scores)/len(scores)))


train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(training_data.drop(['Well Name', 'Facies', 'Depth', 'train', 'Formation'], axis=1), training_data.Facies.values, test_size=.2)


print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


gcf = xgb.XGBClassifier(n_estimators=2000, learning_rate=learning_rate)
gcf.fit(train_X,train_Y)
pred_Y = gcf.predict(test_X)
cm = metrics.confusion_matrix(y_true=test_Y, y_pred=pred_Y)
display_cm(cm, facies_labels, display_metrics=True)


validation_data = scaled_df[scaled_df.train==0]


validation_data.describe()


X = training_data.drop(['Well Name', 'Facies', 'Depth', 'train', 'Formation'], axis=1)
Y = training_data.Facies.values
test_X = validation_data.drop(['Well Name', 'Facies', 'Depth', 'train', 'Formation'], axis=1)

gcf = xgb.XGBClassifier(n_estimators=2000, learning_rate=learning_rate)
gcf.fit(X,Y)
pred_Y = gcf.predict(test_X)

validation_data['Facies'] = pred_Y


validation_data.to_csv("Kr1m_SEG_ML_Attempt3.csv", index=False)





# #### Contest entry by Wouter Kimman 
# 
# 
# Strategy: 
# ----------------------------------------------
# stacking 2 layers of random forests
# 
# downsampling and permutation of targeted prediction for some difficult facies
# 
# I have also added Paolo Bestagini's pre-preproccessing routine, but kept mine as well.
# 

from numpy.fft import rfft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import pandas as pd
import timeit
from sqlalchemy.sql import text
from sklearn import tree
#from sklearn.model_selection import LeavePGroupsOut
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
#import sherlock.filesystem as sfs
#import sherlock.database as sdb
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter, OrderedDict
import csv



def permute_facies_nr(predicted_super, predicted0, faciesnr):
    predicted=predicted0.copy()
    N=len(predicted)
    for ii in range(N):
        if predicted_super[ii]==1:
            predicted[ii]=faciesnr  
    return predicted


def binarify(dataset0, facies_nr):
    dataset=dataset0.copy()
    mask=dataset != facies_nr
    dataset[mask]=0
    mask=dataset == facies_nr
    dataset[mask]=1    
    return dataset



def make_balanced_binary(df_in, faciesnr, factor):
    df=df_in.copy()
    y=df['Facies'].values
    y0=binarify(y, faciesnr)
    df['Facies']=y0

    df1=df[df['Facies']==1]
    X_part1=df1.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    y_part1=df1['Facies'].values
    N1=len(df1)

    df2=df[df['Facies']==0]
    X_part0=df2.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    y_part0=df2['Facies'].values
    N2=len(df2)
    print "ratio now:"
    print float(N2)/float(N1)
    ratio_to_keep=factor*float(N1)/float(N2)
    print "ratio after:"
    print float(N2)/(factor*float(N1))
    dum1, X_part2, dum2, y_part2 = train_test_split(X_part0, y_part0, test_size=ratio_to_keep, random_state=42)

    tmp=[X_part1, X_part2]  
    X = pd.concat(tmp, axis=0)
    y = np.concatenate((y_part1, y_part2))
    return X, y




def phaseI_model(regime_train, correctA, go_B, clf, pred_array, pred_blind, features_blind):      
    clf.fit(regime_train,correctA)     
    predicted_B = clf.predict(go_B)
    pred_array = np.vstack((predicted_B, pred_array))   
    predicted_blind1 = clf.predict(features_blind)
    pred_blind = np.vstack((predicted_blind1, pred_blind))    
    return pred_array, pred_blind

def phaseI_model_scaled(regime_train, correctA, go_B, clf, pred_array, pred_blind, features_blind):   
    regime_train=StandardScaler().fit_transform(regime_train)
    go_B=StandardScaler().fit_transform(go_B)
    features_blind=StandardScaler().fit_transform(features_blind)
    clf.fit(regime_train,correctA)     
    predicted_B = clf.predict(go_B)
    pred_array = np.vstack((predicted_B, pred_array))
    predicted_blind1 = clf.predict(features_blind)
    pred_blind = np.vstack((predicted_blind1, pred_blind))
    return pred_array, pred_blind


def create_structure_for_regimes(df):
    allfeats=['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
    data_all = []
    for feat in allfeats:
        dff=df.groupby('Well Name').describe(percentiles=[0.1, 0.25, .5, 0.75, 0.9]).reset_index().pivot(index='Well Name', values=feat, columns='level_1')
        dff = dff.drop(['count'], axis=1)
        cols=dff.columns
        cols_new=[]
        for ii in cols:
            strin=feat + "_" + str(ii)
            cols_new.append(strin)
        dff.columns=cols_new 
        dff1=dff.reset_index()
        if feat=='GR':
            data_all.append(dff1)
        else:
            data_all.append(dff1.iloc[:,1:])
    data_all = pd.concat(data_all,axis=1)
    return data_all 



def magic(df):
    df1=df.copy()
    b, a = signal.butter(2, 0.2, btype='high', analog=False)
    feats0=['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
    #feats01=['GR','ILD_log10','DeltaPHI','PHIND']
    #feats01=['DeltaPHI']
    #feats01=['GR','DeltaPHI','PHIND']
    feats01=['GR',]
    feats02=['PHIND']
    #feats02=[]
    for ii in feats0:
        df1[ii]=df[ii]
        name1=ii + '_1'
        name2=ii + '_2'
        name3=ii + '_3'
        name4=ii + '_4'
        name5=ii + '_5'
        name6=ii + '_6'
        name7=ii + '_7'
        name8=ii + '_8'
        name9=ii + '_9'
        xx1 = list(df[ii])
        xx_mf= signal.medfilt(xx1,9)
        x_min1=np.roll(xx_mf, 1)
        x_min2=np.roll(xx_mf, -1)
        x_min3=np.roll(xx_mf, 3)
        x_min4=np.roll(xx_mf, 4)
        xx1a=xx1-np.mean(xx1)
        xx_fil = signal.filtfilt(b, a, xx1)        
        xx_grad=np.gradient(xx1a) 
        x_min5=np.roll(xx_grad, 3)
        #df1[name4]=xx_mf
        if ii in feats01: 
            df1[name1]=x_min3
            df1[name2]=xx_fil
            df1[name3]=xx_grad
            df1[name4]=xx_mf 
            df1[name5]=x_min1
            df1[name6]=x_min2
            df1[name7]=x_min4
            #df1[name8]=x_min5
            #df1[name9]=x_min2
        if ii in feats02:
            df1[name1]=x_min3
            df1[name2]=xx_fil
            df1[name3]=xx_grad
            #df1[name4]=xx_mf 
            df1[name5]=x_min1
            #df1[name6]=x_min2 
            #df1[name7]=x_min4
    return df1

        


        
        


#As others have done, this is Paolo Bestagini's pre-preoccessing routine 
# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

#X_aug, padded_rows = augment_features(X, well, depth)


#filename = 'training_data.csv'
filename = 'facies_vectors.csv'
training_data0 = pd.read_csv(filename)
filename = 'validation_data_nofacies.csv'
test_data = pd.read_csv(filename)

#blindwell='CHURCHMAN BIBLE'
#blindwell='LUKE G U'
blindwell='CRAWFORD'


all_wells=training_data0['Well Name'].unique()
print all_wells


# what to do with the naans
training_data1=training_data0.copy()
me_tot=training_data1['PE'].median()
print me_tot
for well in all_wells:
    df=training_data0[training_data0['Well Name'] == well] 
    print well
    print len(df)
    df0=df.dropna()
    #print len(df0)
    if len(df0) > 0:
        print "using median of local"
        me=df['PE'].median()
        df=df.fillna(value=me)
    else:
        print "using median of total"
        df=df.fillna(value=me_tot)
    training_data1[training_data0['Well Name'] == well] =df
    

print len(training_data1)
df0=training_data1.dropna()
print len(df0)


#remove outliers
df=training_data1.copy()
print len(df)
df0=df.dropna()
print len(df0)
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
#df=pd.DataFrame(np.random.randn(20,3))
#df.iloc[3,2]=5
print len(df1)
df2=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
print len(df2)


df2a=df2[df2['Well Name'] != 'Recruit F9'] 


data_all=create_structure_for_regimes(df2a)
data_test=create_structure_for_regimes(test_data)


# Collating the Data:
# -----------------------------------------------------------
# based on the regimes we determined 
# 

# based on kmeans clustering
data=[]
df = training_data0[training_data0['Well Name'] == 'ALEXANDER D'] 
data.append(df)
df = training_data0[training_data0['Well Name'] == 'LUKE G U']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'CROSS H CATTLE']  
data.append(df)
Regime_1 = pd.concat(data, axis=0)
print len(Regime_1)

data=[]
df = training_data0[training_data0['Well Name'] == 'KIMZEY A']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'NOLAN']
data.append(df)
df = training_data0[training_data0['Well Name'] == 'CHURCHMAN BIBLE']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'SHANKLE'] 
data.append(df)
Regime_2 = pd.concat(data, axis=0)
print len(Regime_2)

data=[]

df = training_data0[training_data0['Well Name'] == 'SHRIMPLIN']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'NEWBY']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'Recruit F9']  
data.append(df)
Regime_3 = pd.concat(data, axis=0)
print len(Regime_3)



# **Split the data into 2 parts:**
# 
# from A We will make initial predictions
# 
# from B we will make the final prediction(s)
# 

# Phase 0:
# ---------------------------------
# -Create predictions specifically for the most difficult facies
# 
# -For this stage we focus on TP and FP only: We want only a few predictions that are
# likely to be correct to edge the f1 prediction up slightly at the end
# 
# -For each facies i consider the samples binary 0 or 1 and downsample the zeros to 
# get a more even distribution. However, I found the results change quite a bit depending 
# on the degree of downsampling. As a type of dumb-men's L-curve analysis I varied this to the 
# point where the nr of predictions (1) doesn't change much more
# 
# -also, based on the similarity to the other wells we have an 'indiction' on how much of the different facies we can expect 
# 

# ___________________________________________
# **training for facies 9 specifically**
# ___________________________________________
# 

df0 = test_data[test_data['Well Name'] == blindwell] 
df1 = df0.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#df0 = training_data0[training_data0['Well Name'] == blindwell]  
#df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)

df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
blind=magic(df1a)

#features_blind = blind.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)


#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)
#X, y = make_balanced_binary(all1, 9,6)
for kk in range(3,4):
    X, y = make_balanced_binary(all1, 9,kk)
#============================================================
    correct_train=y

    #clf = RandomForestClassifier(max_depth = 6, n_estimators=1600)
    clf = RandomForestClassifier(max_depth = 6, n_estimators=800)
    clf.fit(X,correct_train)

    predicted_blind1 = clf.predict(features_blind)

    predicted_regime9=predicted_blind1.copy()
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime9)))
    print "----------------------------------"


# ___________________________________________
# **training for facies 1 specifically**
# ________________________
# 
# 
# 
# 


#features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)

for kk in range(4,6):
#for kk in range(1,6): 
    X, y = make_balanced_binary(all1, 1,kk)
    #============================================================

    #=============================================
    go_A=StandardScaler().fit_transform(X)
    go_blind=StandardScaler().fit_transform(features_blind)
    correct_train_A=binarify(y, 1)
                                        

    clf = linear_model.LogisticRegression()
    clf.fit(go_A,correct_train_A)
    predicted_blind1 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(go_A,correct_train_A)                                                  
    predicted_blind2 = clf.predict(go_blind)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(go_A,correct_train_A)   
    predicted_blind3 = clf.predict(go_blind)

    clf = svm.LinearSVC()
    clf.fit(go_A,correct_train_A)   
    predicted_blind4 = clf.predict(go_blind)



    #####################################
    predicted_blind=predicted_blind1+predicted_blind2+predicted_blind3+predicted_blind4
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] > 3:
            predicted_blind[ii]=1
        else:
            predicted_blind[ii]=0 
        
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] == 1 and predicted_blind[ii-1] == 0 and predicted_blind[ii+1] == 0:
            predicted_blind[ii]=0
        if predicted_blind[ii] == 1 and predicted_blind[ii-1] == 0 and predicted_blind[ii+2] == 0:
            predicted_blind[ii]=0        
        if predicted_blind[ii] == 1 and predicted_blind[ii-2] == 0 and predicted_blind[ii+1] == 0:
            predicted_blind[ii]=0     
    #####################################    

    print "-------"
    predicted_regime1=predicted_blind.copy()

    #print("%c is my %s letter and my number %d number is %.5f" % ('X', 'favorite', 1, .14))
 
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime1)))
    print "----------------------------------"


# **training for facies 5 specifically**
# 

#features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)
for kk in range(1,6):
#for kk in range(2,4): 
    X, y = make_balanced_binary(all1, 5,kk)
    #X, y = make_balanced_binary(all1, 5,13)
    #============================================================

    go_A=StandardScaler().fit_transform(X)
    go_blind=StandardScaler().fit_transform(features_blind)
    correct_train_A=binarify(y, 1)
    #=============================================                                        

    clf = KNeighborsClassifier(n_neighbors=4,algorithm='brute')
    clf.fit(go_A,correct_train_A)
    predicted_blind1 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5,leaf_size=10)
    clf.fit(go_A,correct_train_A)                                                  
    predicted_blind2 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(go_A,correct_train_A)   
    predicted_blind3 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind4 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind5 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)    
    predicted_blind6 = clf.predict(go_blind)


    #####################################
    predicted_blind=predicted_blind1+predicted_blind2+predicted_blind3+predicted_blind4+predicted_blind5+predicted_blind6
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] > 4:
            predicted_blind[ii]=1
        else:
            predicted_blind[ii]=0 

    print "-------"
    predicted_regime5=predicted_blind.copy()
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime5)))
    print "----------------------------------"


# **training for facies 7 specifically**
# 

#features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)
for kk in range(2,5):
    X, y = make_balanced_binary(all1, 7,kk)
    #X, y = make_balanced_binary(all1, 7,13)
    #============================================================

    go_A=StandardScaler().fit_transform(X)
    go_blind=StandardScaler().fit_transform(features_blind)
    correct_train_A=binarify(y, 1)
    #=============================================                                        

    clf = KNeighborsClassifier(n_neighbors=4,algorithm='brute')
    clf.fit(go_A,correct_train_A)
    predicted_blind1 = clf.predict(go_blind)


    clf = KNeighborsClassifier(n_neighbors=5,leaf_size=10)
    clf.fit(go_A,correct_train_A)                                                  
    predicted_blind2 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(go_A,correct_train_A)   
    predicted_blind3 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind4 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind5 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)    
    predicted_blind6 = clf.predict(go_blind)


    #####################################
    predicted_blind=predicted_blind1+predicted_blind2+predicted_blind3+predicted_blind4+predicted_blind5+predicted_blind6
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] > 5:
            predicted_blind[ii]=1
        else:
            predicted_blind[ii]=0 


    #####################################    
    print "-------"
    predicted_regime7=predicted_blind.copy()
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime7)))
    print "----------------------------------"


# 
# PHASE Ib 
# ======================================
# **PREPARE THE DATA**
# 

def prepare_data(Regime_1, Regime_2, Regime_3, test_data, w1, w2,w3):
    df0=Regime_1.dropna()
    df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    feature_names0 = ['GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'PHIND_1', 'PHIND_2']
    X0 = df2a[feature_names0].values
    df2a=(df1a)
    y=df2a['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = df2a[feature_names].values
    well = df2a['Well Name'].values
    depth = df2a['Depth'].values
    X2, padded_rows = augment_features(X1, well, depth)
    Xtot_train=np.column_stack((X0,X2))
    regime1A_train, regime1B_train, regime1A_test, regime1B_test = train_test_split(Xtot_train, y, test_size=w1, random_state=42)

    df0=Regime_2.dropna()
    df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    feature_names0 = ['GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'PHIND_1', 'PHIND_2']
    X0 = df2a[feature_names0].values
    df2a=(df1a)
    y=df2a['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = df2a[feature_names].values
    well = df2a['Well Name'].values
    depth = df2a['Depth'].values
    X2, padded_rows = augment_features(X1, well, depth)
    Xtot_train=np.column_stack((X0,X2))
    regime2A_train, regime2B_train, regime2A_test, regime2B_test = train_test_split(Xtot_train, y, test_size=w2, random_state=42)


    df0=Regime_3.dropna()
    df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    feature_names0 = ['GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'PHIND_1', 'PHIND_2']
    X0 = df2a[feature_names0].values
    df2a=(df1a)
    y=df2a['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = df2a[feature_names].values
    well = df2a['Well Name'].values
    depth = df2a['Depth'].values
    X2, padded_rows = augment_features(X1, well, depth)
    Xtot_train=np.column_stack((X0,X2))
    regime3A_train, regime3B_train, regime3A_test, regime3B_test = train_test_split(Xtot_train, y, test_size=w3, random_state=42)


    #df0 = training_data0[training_data0['Well Name'] == blindwell]
    #df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df0 = test_data[test_data['Well Name'] == blindwell] 
    df1 = df0.drop(['Formation', 'Well Name', 'Depth'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    #df2a=df1a
    X0blind = df2a[feature_names0].values

    blind=df1a
    #correct_facies_labels = blind['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = blind[feature_names].values
    well = blind['Well Name'].values
    depth = blind['Depth'].values
    X2blind,  padded_rows = augment_features(X1, well, depth)

    features_blind=np.column_stack((X0blind,X2blind))
#=======================================================
    main_regime=regime2A_train
    other1=regime1A_train
    other2=regime3A_train

    main_test=regime2A_test
    other1_test=regime1A_test
    other2_test=regime3A_test

    go_B=np.concatenate((regime1B_train, regime2B_train, regime3B_train))
    correctB=np.concatenate((regime1B_test, regime2B_test, regime3B_test))
#     #===================================================
    train1= np.concatenate((main_regime, other1, other2))
    correctA1=np.concatenate((main_test, other1_test, other2_test))
#     #=================================================== 
#     train2= np.concatenate((main_regime, other2))
#     correctA2=np.concatenate((main_test, other2_test))
#     #===================================================

    #===================================================
    #train1=main_regime
    #correctA1=main_test
    train2=other1
    correctA2=other1_test   
    train3=other2
    correctA3=other2_test   

    return train1, train2, train3, correctA1, correctA2, correctA3, correctB, go_B, features_blind


# **PREPARE THE DATA FOR SERIAL MODELLING**
# 
# 

# **Create several predictions, varying the dataset and the technique**
# 

def run_phaseI(train1,train2,train3,correctA1,correctA2,correctA3,correctB, go_B, features_blind):    
    pred_array=0*correctB
    pred_blind=np.zeros(len(features_blind))

    print "rf1"
    clf = RandomForestClassifier(max_depth = 5, n_estimators=2600, random_state=1)
    pred_array, pred_blind=phaseI_model(train1, correctA1, go_B, clf, pred_array, pred_blind, features_blind)
    clf = RandomForestClassifier(max_depth = 15, n_estimators=3000)
    pred_array, pred_blind=phaseI_model(train1, correctA1, go_B, clf, pred_array, pred_blind, features_blind)    
#     pred_array, pred_blind=phaseI_model(train2, correctA2, go_B, clf, pred_array, pred_blind, features_blind)
#     pred_array, pred_blind=phaseI_model(train3, correctA3, go_B, clf, pred_array, pred_blind, features_blind)
    clf = RandomForestClassifier(n_estimators=1200, max_depth = 15, criterion='entropy',
                                 max_features=10, min_samples_split=25, min_samples_leaf=5,
                                 class_weight='balanced', random_state=1)
    pred_array, pred_blind=phaseI_model(train1, correctA1, go_B, clf, pred_array, pred_blind, features_blind)
    #pred_array, pred_blind=phaseI_model(train2, correctA2, go_B, clf, pred_array, pred_blind, features_blind)
    #pred_array, pred_blind=phaseI_model(train3, correctA3, go_B, clf, pred_array, pred_blind, features_blind)
    return pred_array, pred_blind


# Phase II:
# ---------------------------------------------
# Stacking the predictions from phase Ib. 
# New predictions from data B
# 
# ------------------------------------------------
# 

# **First prediction of B data without Phase I input:**
# 

# **Add the initial predictions as features:**
# 

# **Make a new prediction, with the best model on the full dataset B:**
# 

w1=0.05
w2=0.05
w3=0.05
print "preparing data:"
#train1, train2, train3, correctA1, correctA2, correctA3, correctB, go_B, features_blind=prepare_data(Regime_1, Regime_2, Regime_3, training_data0, w1, w2,w3)
train1, train2, train3, correctA1, correctA2, correctA3, correctB, go_B, features_blind=prepare_data(Regime_1, Regime_2, Regime_3, test_data, w1, w2,w3)
print(len(correctB))
print "running phase I:"
pred_array, pred_blind = run_phaseI(train1,train2,train3,correctA1,correctA2, correctA3, correctB, go_B, features_blind)
print "prediction phase II:"
clf = RandomForestClassifier(max_depth = 8, n_estimators=3000, max_features=10, criterion='entropy',class_weight='balanced')
#clf = RandomForestClassifier(max_depth = 5, n_estimators=300, max_features=10, criterion='entropy',class_weight='balanced')
#clf = RandomForestClassifier(n_estimators=1200, max_depth = 15, criterion='entropy',
#                             max_features=10, min_samples_split=25, min_samples_leaf=5,
#                             class_weight='balanced', random_state=1)
#clf = RandomForestClassifier(n_estimators=1200, max_depth = 5, criterion='entropy',
#                             max_features=10, min_samples_split=25, min_samples_leaf=5,
#                             class_weight='balanced', random_state=1)
clf.fit(go_B,correctB)
predicted_blind_PHASE_I = clf.predict(features_blind)

print "prediction phase II-stacked:"
pa=pred_array[:len(pred_array)-1]
go_B_PHASE_II=np.concatenate((pa, go_B.transpose())).transpose()
pa1=np.median(pa,axis=0)
go_B_PHASE_II=np.column_stack((go_B_PHASE_II,pa1))
print go_B_PHASE_II.shape
feat=pred_blind[:len(pred_blind)-1]
features_blind_PHASE_II=np.concatenate((feat, features_blind.transpose())).transpose()
feat1=np.median(feat,axis=0)
features_blind_PHASE_II=np.column_stack((features_blind_PHASE_II,feat1))

#second pred
clf.fit(go_B_PHASE_II,correctB)
predicted_blind_PHASE_II = clf.predict(features_blind_PHASE_II)

#print "finished"
#out_f1=metrics.f1_score(correct_facies_labels, predicted_blind_PHASE_I, average = 'micro')
#print " f1 score on the prediction of blind:"
#print out_f1
#out_f1=metrics.f1_score(correct_facies_labels, predicted_blind_PHASE_II, average = 'micro')
#print " f1 score on the prediction of blind:"
#print out_f1
#print "finished"
#print "-----------------------------"   


# **Permute facies based on earlier predictions**:
# 

print(sum(predicted_regime5))
predicted_blind_PHASE_IIa=permute_facies_nr(predicted_regime5, predicted_blind_PHASE_II, 5)
print(sum(predicted_regime7))
predicted_blind_PHASE_IIb=permute_facies_nr(predicted_regime7, predicted_blind_PHASE_IIa, 7)
print(sum(predicted_regime1))
predicted_blind_PHASE_IIc=permute_facies_nr(predicted_regime1, predicted_blind_PHASE_IIb, 1)
print(sum(predicted_regime9))
predicted_blind_PHASE_III=permute_facies_nr(predicted_regime9, predicted_blind_PHASE_IIc, 9)


print "values changed:"

print len(predicted_blind_PHASE_II)-np.count_nonzero(predicted_blind_PHASE_III==predicted_blind_PHASE_II)


predicted_blind_CRAWFORD=predicted_blind_PHASE_III
predicted_blind_CRAWFORD


x=Counter(predicted_blind_PHASE_I)
y = OrderedDict(x)
y


x=Counter(predicted_blind_PHASE_II)
y = OrderedDict(x)
y


x=Counter(predicted_blind_PHASE_III)
y = OrderedDict(x)
y





# ## Facies classification using Machine Learning
# 
# ### aaML Submission
# 
# ### By:
# 
# [Alexsandro G. Cerqueira](https://github.com/alexleogc),    
# [Alã de C. Damasceno](https://github.com/aladamasceno)
# 
# There are tow main notebooks:
# 
# - Data Analysis and edition
# - Submission
# 
# 

from libtools import *


# ### Loading the data training data without Shankle well
# 

training = pd.read_csv('data-test.csv')


training.head()


training.describe()


training = training.fillna(-99999)


# ### Loading the SHANKLE well
# 

blind = pd.read_csv('blind.csv')


blind.head()


blind.describe()


training_SH = divisao_sh(training)
training_LM = divisao_lm(training)

blind_SH = divisao_sh(blind)
blind_LM = divisao_lm(blind)


training_SH.head()


training_LM.head()


blind_SH.head()


blind_LM.head()


X_SH = training_SH.drop(['Facies'],axis=1)
y_SH = training_SH['Facies']

X_LM = training_LM.drop(['Facies'],axis=1)
y_LM = training_LM['Facies']

X_SH_blind = blind_SH.drop(['Facies'],axis=1)
y_SH_blind = blind_SH['Facies']

X_LM_blind = blind_LM.drop(['Facies'],axis=1)
y_LM_blind = blind_LM['Facies']


from sklearn.model_selection import train_test_split

X_train_SH, X_test_SH, y_train_SH, y_test_SH = train_test_split(X_SH, y_SH, test_size=0.1)

X_train_LM, X_test_LM, y_train_LM, y_test_LM = train_test_split(X_LM, y_LM, test_size=0.1)


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix


ETC_SH = ExtraTreesClassifier(n_estimators=500, bootstrap=True)
ETC_LM = ExtraTreesClassifier(n_estimators=500)

ETC_SH.fit(X_train_SH, y_train_SH)
ETC_LM.fit(X_train_LM, y_train_LM)


pred_SH = ETC_SH.predict(X_test_SH)
print(confusion_matrix(y_test_SH,pred_SH))
print(classification_report(y_test_SH,pred_SH))


pred_LM = ETC_LM.predict(X_test_LM)
print(confusion_matrix(y_test_LM,pred_LM))
print(classification_report(y_test_LM,pred_LM))


blind_pred_SH = ETC_SH.predict(X_SH_blind)
print(confusion_matrix(y_SH_blind, blind_pred_SH))
print(classification_report(y_SH_blind, blind_pred_SH))


blind_pred_LM = ETC_LM.predict(X_LM_blind)
print(confusion_matrix(y_LM_blind, blind_pred_LM))
print(classification_report(y_LM_blind, blind_pred_LM))


blind_pred_SH = pd.DataFrame(blind_pred_SH, index=X_SH_blind.index)
blind_pred_LM = pd.DataFrame(blind_pred_LM, index=X_LM_blind.index)
pred_blind = pd.concat([blind_pred_SH,blind_pred_LM])
pred_blind = pred_blind.sort_index()


y_blind = blind['Facies']


print(confusion_matrix(y_blind, pred_blind))
print(classification_report(y_blind, pred_blind))


# ### Using the complete training data
# 

training_data = pd.read_csv('training.csv')


training_data.head()


training_data.describe()


training_data_SH = divisao_sh(training_data)
training_data_LM = divisao_lm(training_data)


training_data_SH.describe()


training_data_LM.describe()


X_SH = training_data_SH.drop(['Facies'],axis=1)
y_SH = training_data_SH['Facies']

X_LM = training_data_LM.drop(['Facies'],axis=1)
y_LM = training_data_LM['Facies']


X_SH.describe()


X_LM.describe()


from sklearn.model_selection import train_test_split

X_train_SH, X_test_SH, y_train_SH, y_test_SH = train_test_split(X_SH, y_SH, test_size=0.1)

X_train_LM, X_test_LM, y_train_LM, y_test_LM = train_test_split(X_LM, y_LM, test_size=0.1)


# #  Applying  ExtraTreeClassifier
# 

ETC_SH = ExtraTreesClassifier(n_estimators=500, bootstrap=True)
ETC_LM = ExtraTreesClassifier(n_estimators=500)

ETC_SH.fit(X_train_SH, y_train_SH)
ETC_LM.fit(X_train_LM, y_train_LM)


pred_SH = ETC_SH.predict(X_test_SH)
print(confusion_matrix(y_test_SH,pred_SH))
print(classification_report(y_test_SH,pred_SH))


pred_LM = ETC_LM.predict(X_test_LM)
print(confusion_matrix(y_test_LM,pred_LM))
print(classification_report(y_test_LM,pred_LM))


validation = pd.read_csv('validation_data_nofacies.csv')


validation.head()


validation.describe()


# ### Making the division between SH and LM
# 

validation['Label_Form_SH_LM'] = validation.Formation.apply((label_two_groups_formation))


validation.head()


validation_SH = divisao_sh(validation)
validation_LM = divisao_lm(validation)


validation_SH.head()


validation_LM.head()


# ### Removing the colums: Formation, Well Name, Depth
# 

X_val_SH = validation_SH.drop(['Formation','Well Name','Depth','NM_M'], axis=1)
X_val_LM = validation_LM.drop(['Formation','Well Name','Depth','NM_M'], axis=1)


X_val_SH.head()


X_val_LM.head()


pred_val_SH = ETC_SH.predict(X_val_SH)


pred_val_LM =ETC_LM.predict(X_val_LM)


pred_val_SH = pd.DataFrame(pred_val_SH, index=X_val_SH.index)
pred_val_LM = pd.DataFrame(pred_val_LM, index=X_val_LM.index)
pred_val = pd.concat([pred_val_SH,pred_val_LM])
pred_val = pred_val.sort_index()


pred_val.describe()


validation['Facies Pred'] = pred_val


validation=validation.drop(['Label_Form_SH_LM'],axis=1)


validation.head()


validation.to_csv('Prediction.csv')





# Like Ar4 submission but with GradientBoosting classifier
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize']=(20.0,10.0)
inline_rc = dict(mpl.rcParams)

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt

from pandas.tools.plotting import scatter_matrix

import matplotlib.colors as colors


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from classification_utilities import display_cm, display_adj_cm
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_files

from xgboost.sklearn import XGBClassifier
from scipy.sparse import vstack

from sklearn.ensemble import GradientBoostingClassifier

seed = 123
np.random.seed(seed)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# Load data from file
data = pd.read_csv('../facies_vectors.csv')


X = data[feature_names].values
y = data['Facies'].values


well = data['Well Name'].values
depth = data['Depth'].values


# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


reg = RandomForestRegressor(max_features='sqrt', n_estimators=50)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


X_aug, padded_rows = augment_features(X, well, depth)


# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [100]  
MD_grid = [3]  
M_grid = [10]
LR_grid = [0.1]  
L_grid = [5]
S_grid = [25]  
param_grid = []
for N in N_grid:
    for M in MD_grid:
        for M1 in M_grid:
            for S in LR_grid: 
                for L in L_grid:
                    for S1 in S_grid:
                        param_grid.append({'N':N, 'MD':M, 'MF':M1,'LR':S,'L':L,'S1':S1})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier  
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss='exponential',
                                                        n_estimators=param['N'], 
                                                        learning_rate=param['LR'], 
                                                        max_depth=param['MD'],
                                                        max_features= param['MF'],
                                                        min_samples_leaf=param['L'],
                                                        min_samples_split=param['S1'],
                                                        random_state=seed, 
                                                        max_leaf_nodes=None, 
                                                        verbose=1), n_jobs=-1)

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# # Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 



# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)


from mpl_toolkits.axes_grid1 import make_axes_locatable


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('SH_predicted_facies_submission001.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)





# # Facies classification using machine learning techniques
# Copy of <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini's</a> "Try 2", augmented, by Alan Richardson (Ausar Geophysical), with an ML estimator for PE in the wells where it is missing (rather than just using the mean).
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm is based on the use of random forests combined in one-vs-one multiclass strategy. In particular, we would like to study the effect of:
# - Robust feature normalization.
# - Feature imputation for missing feature values.
# - Well-based cross-validation routines.
# - Feature augmentation strategies.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt


import sys, scipy, sklearn
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. This is the only cell that differs from the approach of Paolo Bestagini. Currently no feature engineering is used, but this should be explored in the future.
# 

reg = RandomForestRegressor(max_features='sqrt', n_estimators=50)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))


# ## Feature augmentation
# Our guess is that facies do not abrutly change from a given depth layer to the next one. Therefore, we consider features at neighboring layers to be somehow correlated. To possibly exploit this fact, let us perform feature augmentation by:
# - Aggregating features at neighboring depths.
# - Computing feature spatial gradient.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [100]  # [50, 100, 150]
M_grid = [10]  # [5, 10, 15]
S_grid = [25]  # [10, 25, 50, 75]
L_grid = [5] # [2, 3, 4, 5, 10, 25]
param_grid = []
for N in N_grid:
    for M in M_grid:
        for S in S_grid:
            for L in L_grid:
                param_grid.append({'N':N, 'M':M, 'S':S, 'L':L})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
                             max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
                             class_weight='balanced', random_state=0), n_jobs=-1)

    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)


# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)


# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('ar4_predicted_facies_submission002.csv')


# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)


# # 2016 SEG ML Contest entry by Alan Richardson (Ausar Geophysical)
# 
# This notebook discusses some of the ideas contained in my submission. The majority of the code in my submission is in the [`ar4_submission3.py`](https://github.com/seg/2016-ml-contest/blob/master/ar4/ar4_submission3.py) script.
# 

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import scipy.interpolate

from ar4_submission3 import run, get_numwells, get_wellnames

run_ml = False # run the ML estimators, if false just loads results from file
solve_rgt = False # run the RGT solver - takes about 30mins, run_ml must be True

if run_ml:
    data = run_all(solve_rgt)
else:
    data = pd.read_csv('ar4_submission3.csv')


matplotlib.style.use('ggplot')
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')

def plotwellsim(data,f,y=None,title=None):
    wells = data['Well Name'].unique()
    nwells = len(wells)
    dfg = data.groupby('Well Name',sort=False)
    fig, ax = plt.subplots(nrows=1, ncols=nwells, figsize=(12, 9), sharey=True)
    if (title):
        plt.suptitle(title)
    if (y):
        miny = data[y].min()
        maxy = data[y].max()
        ax[0].set_ylabel(y)
    else:
        miny = 0
        maxy = dfg.size().max()
        ax[0].set_ylabel('Depth from top of well')
    vmin=data[f].min()
    vmax=data[f].max()
    if (f=='Facies') | (f=='NeighbFacies'):
        cmap = cmap_facies
    else:
        cmap = 'viridis'
    for wellidx,(name,group) in enumerate(dfg):
        if y:
            welldinterp = scipy.interpolate.interp1d(group[y], group[f], bounds_error=False)
            nf = len(data[y].unique())
        else:
            welldinterp = scipy.interpolate.interp1d(np.arange(0,len(group[f])), group[f], bounds_error=False)
            nf = (maxy-miny)/0.5
        fnew = np.linspace(miny, maxy, nf) 
        ynew = welldinterp(fnew)
        ynew = ynew[:, np.newaxis]
        ax[wellidx].set_xticks([])
        ax[wellidx].set_yticks([])
        ax[wellidx].grid(False)
        ax[wellidx].imshow(ynew, aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
        ax[wellidx].set_xlabel(name,rotation='vertical')


# ## Introduction
# 
# Many contestants have experimented with different estimators. At the time of writing, ensemble tree methods are clearly dominating the top of the leaderboard. This is likely to be because they are (reportedly) less likely to suffer from overfitting than other approaches. As the training set is small, and at least one of the validation wells shows notable differences compared to the training wells (discussed below), overfitting the training dataset is likely to be severely detrimental to validation performance.
# 
# Some feature engineering has also proved useful to other entrants. The most successful entries are currently (at the time of writing) all based on the submission 2 of Paolo Bestagini. One of the notable features of Paolo's submission is that it not does contain very sophisticated feature engineering: primarily a simple augmentation of each sample's features by those of the sample above and below in depth. Other entrants, such as Bird Team and geoLEARN, have invested great effort into developing new features, but it has currently not been as successful as Paolo's simple approach. This is again likely to be due to detailed feature sets causing overfitting.
# 
# My submission thus only considers ensemble tree estimators. Although a significant number of new features are created through feature engineering, I remained conscious of the risk of overfitting and attempted to mitigate this by using cautious estimator parameters.
# 

# ## Feature engineering
# 
# One of the features provided with the data is the depth (probably height above sea level) at which each sample was measured. This information is discarded by many contestants, as in its current form it does not contain much predictive value. This is due to the differing amounts of uplift at each well causing formations to be at different depths in different wells. This is demonstrated in the figure below. This figure also shows that one of the validation wells (Crawford) has experienced more uplift than any of the training wells. This indicates that it may not be close to the training wells, and so they may not be good predictors of this validation well.
# 

well_width = 100
mind = data['Depth'].min()
maxd = data['Depth'].max()
fim = np.nan*np.ones([int((maxd-mind)*2)+1,get_numwells(data)*well_width])
dfg = data.groupby('Well Name',sort=False)
plt.figure(figsize=(12, 9))
plt.title('Wells cover different depths')
ax=plt.subplot(111)
ax.grid(False)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('Depth')
plt.tick_params(axis="both", which="both", bottom="off", top="off", 
                labelbottom="off", left="off", right="off", labelleft="off")
ax.text(well_width*7,1000,'Colours represent\nformation class')
for i,(name,group) in enumerate(dfg):
    if (maxd-group['Depth'].max())*2 > 600:
        ty = (maxd-group['Depth'].max())*2-50
        tva='bottom'
    else:
        ty = (maxd-group['Depth'].min())*2+50
        tva='top'
    ax.text(well_width*(i+0.5),ty,name,va=tva,rotation='vertical')
    for j in range(len(group)):
        fim[-int((group.loc[group.index[j],'Depth']-maxd)*2),i*well_width:(i+1)*well_width]=group.loc[group.index[j],'FormationClass']
    
plt.imshow(fim,cmap='viridis',aspect='auto')


# The figure also demonstrates that most of the wells follow a somewhat consistent pattern with depth in the well. One new feature that I create is thus the depth below the surface, by subtracting the surface depth from the depth measurement at each sample. This creates a feature that captures the element of consistency between the wells with depth below the surface.
# 
# One important feature that was not provided with the data is the location of each well. It is plausible that this feature would have some predictive power, as nearby wells are more likely to be similar. The amount of uplift may be an indicator for location, if we assume that the formations can be approximated as being planar. I performed inversions to find 1D and 2D positions for each well using this idea to extract this information, but in the end decided to simply create a feature for each well containing the top depth of the third formation from the surface (the first that is present in all wells without interference from the top of the well)).
# 

ax=data.groupby('Well Name',sort=False)['Formation3Depth'].first().plot.bar(title='Top of Formation 3 Depth')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.grid(axis='x')


# The wells are too far apart to allow matching of facies between wells. It is thus not possible to build a 3D lithofacies model with which one could interpolate facies in wells without core samples. Nevertheless, one can still try to extract some information by matching wells. To achieve this, I use an approach similar to that described by Wheeler and Hale ([CWP Report](http://cwp.mines.edu/Documents/cwpreports/cwp826.pdf)). Dynamic warping (usually referred to as dynamic time warping in the signal processing community) attempts to match two sequences by shifting and stretching/compressing them to get the best fit. The amount of stretching/compressing is variable with position along the sequence. This idea seems well suited to matching well logs as varying amounts of uplift and compaction are likely to result in a well log being shifted and stretched/compressed relative to corresponding portions of another well log.
# 
# To implement this dynamic warping strategy, I use the FastDTW Python package to calculate the dynamic warping distance and best path between each pair of wells. I use all available features (so PE is not used for matching for the wells that lack it, and Facies is not used for matching the validation wells). I constrain the warping to only match portions of the wells that are in the same formation. As described by Wheeler and Hale, it is now necessary to form and solve a system of equations to find the best warping that satisfies all pairs. The position of each point after warping will be referred to as its relative geologic time (RGT). Points with the same RGT in different wells should thus correspond. The distance measure reported by the dynamic time warping tool is supposed to represent how closely pairs of wells match. Following Wheeler and Hale, I use this in the system of equations to weight more strongly rows corresponding to wells that match closely. In order to find a solution that results in the logs continuing to be monotonically increasing with depth, I solve for the difference in RGT between adjacent depths. I use SciPy's constrained linear least squares tool to find a solution where these RGT differences are always at least 1.
# 
# The results of dynamic warping are demonstrated by the matching of RELPOS and GR values between wells. The Facies values for the training set are also shown.
# 

plotwellsim(data,'RELPOS',title='RELPOS by depth')
plotwellsim(data,'RELPOS','RGT',title='RELPOS by RGT')
plotwellsim(data,'GR',title='GR by depth')
plotwellsim(data,'GR','RGT',title='GR by RGT')
plotwellsim(data,'Facies',title='Facies by depth')
plotwellsim(data,'Facies','RGT',title='Facies by RGT')


# I also used the dynamic warping distance to create another feature estimating the location of each well, by inverting for this using a least squares inversion. The results, shown below, do not appear to be very consistent with the Formation 3 Depth plot shown above. This may be due to inaccuracies in the approximation that the wells are in a line, or in the approximation that the formations are planar. The X1D values that I find are almost monotonic with the order of wells in the provided data.
# 

ax=plt.subplot(111)
plt.axis('off')
plt.ylim(-1,1)
plt.title('Well Position (X1D)')
for i,wellname in enumerate(get_wellnames(data)):
    x=data.loc[data['Well Name']==wellname, 'X1D'].values[0]
    plt.scatter(x=-x,y=0)
    if i%2==0:
        tva='top'
        ty=-0.1
    else:
        tva='bottom'
        ty=0.1
    plt.text(-x,ty,wellname,rotation='vertical',ha='center',va=tva)


# The similarity of wells logs at the same RGT means that I can estimate the facies using a nearest neighbour estimator. The results are shown below, with the true facies for the training wells also shown for comparison.
# 

plotwellsim(data,'NeighbFacies','RGT','Facies found by KNN classifier')
plotwellsim(data,'Facies','RGT','True facies (except validation wells)')


# I created what I refer to as 'interval' features. These capture information about the current interval, such as the distance from the beginning of it, size, and what fraction of the way through the interval the sample was taken. These features are made for the intervals of Formation, NM_M (non-marine/marine), and, when predicting PE, Facies. The features are calculated using the metrics Depth and RGT. Another interesting interval feature I created is compaction, which I calculate by dividing the size of the interval in RGT by its size in depth, capturing how much time has been compressed into that depth.
# 

plt.figure(figsize=(12, 9))
plt.tick_params(axis="x", which="both", bottom="off", top="off", 
                labelbottom="off", left="off", right="off", labelleft="off")
wellnames = get_wellnames(data).tolist()
num_wells = get_numwells(data)
formations = data['FormationClass'].unique().tolist()
num_formations = len(formations)
colors=plt.cm.viridis(np.linspace(0,1,len(wellnames)))
dfg=data.groupby(['Well Name','FormationClass'], sort=False)
for i,(name,group) in enumerate(dfg):
    widx = wellnames.index(name[0])
    fidx = formations.index(name[1])
    plt.bar(fidx*num_wells+widx,group['FormationClassCompaction'].values[0],color=colors[widx])
plt.ylim(0,0.1)
plt.title('Formation compaction')
plt.text(70,0.09,'Color represents well, each cycle along the x axis is one formation')


# My 'measurement' features consist of a median filter, derivative, and second derivative, of each type of log measurement. I also have a sharpened version of each measurement constructed by subtracting the second derivative from the measurement. The motivation for the sharpening filter is a crude attempt at deconvolving the smoothing effect of the measurement tools.
# 
# The final class of new features I introduce are 'interval measurement' features. These capture information about the well log measurements over an interval. The features are the mean, difference from the mean, standard deviation, and fraction of the standard deviation from the mean. A motivation for creating these features is the idea that the well log measurements may not be well calibrated between the wells, so using the raw values may not be very predictive. What fraction of the standard deviation over the interval the measurement is away from the mean over that interval, should be less affected by the calibration, however. The intervals I use are the Formation, the NM_M interval, each whole well, and a local Gaussian window around the sample depth.
# 

# ## Prediction
# 
# Two of the training wells do not have PE measurement values. Scikit-Learn does not allow estimators to be created when the training data contains missing values. The missing PE values must thus be filled-in ("imputed") so that PE can be used in the facies prediction. A popular strategy for this is to use the mean or median. As well as its simplicity, this approach may be less susceptible to overfitting than another approach; using an estimator to predict the missing values. Despite the risk of overfitting, the estimator approach has the attraction of using all of the data, and so has the potential to yield more accurate results.
# 
# I perform the PE prediction using two steps. The first is to make an initial prediction of the facies values in the two validation wells using all of the data except the PE values. This allows all of the data, including that in the validation wells, to be used for the prediction of PE.
# 
# Once the PE data is complete, all of the data, now including PE, can be used for a second, and hopefully more accurate, prediction of the facies in the validation wells.
# 

v_rows = (data['Well Name'] == 'STUART') | (data['Well Name'] == 'CRAWFORD')
plotwellsim(data.loc[v_rows,:],'Facies',title='Predicted facies')


# # Facies classification using machine learning techniques
# Copy of <a href="https://home.deib.polimi.it/bestagini/">Paolo Bestagini's</a> "Try 2", augmented, by Alan Richardson (Ausar Geophysical), with an ML estimator for PE in the wells where it is missing (rather than just using the mean).
# 
# In the following, we provide a possible solution to the facies classification problem described at https://github.com/seg/2016-ml-contest.
# 
# The proposed algorithm is based on the use of random forests combined in one-vs-one multiclass strategy. In particular, we would like to study the effect of:
# - Robust feature normalization.
# - Feature imputation for missing feature values.
# - Well-based cross-validation routines.
# - Feature augmentation strategies.
# 
# ## Script initialization
# Let us import the used packages and define some parameters (e.g., colors, labels, etc.).
# 

# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
#import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scipy.signal import medfilt


import sys, scipy, sklearn
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)


# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']


# ## Load data
# Let us load training data and store features, labels and other data into numpy arrays.
# 

# Load data from file
data = pd.read_csv('../facies_vectors.csv')


# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels


# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values


# ## Data inspection
# Let us inspect the features we are working with. This step is useful to understand how to normalize them and how to devise a correct cross-validation strategy. Specifically, it is possible to observe that:
# - Some features seem to be affected by a few outlier measurements.
# - Only a few wells contain samples from all classes.
# - PE measurements are available only for some wells.
# 

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


# Feature distribution
# plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
# mpl.rcParams.update(inline_rc)


# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)


# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)


# ## Feature imputation
# Let us fill missing PE values. This is the only cell that differs from the approach of Paolo Bestagini. Currently no feature engineering is used, but this should be explored in the future.
# 

def make_pe(X, seed):
    reg = RandomForestRegressor(max_features='sqrt', n_estimators=50, random_state=seed)
    DataImpAll = data[feature_names].copy()
    DataImp = DataImpAll.dropna(axis = 0, inplace=False)
    Ximp=DataImp.loc[:, DataImp.columns != 'PE']
    Yimp=DataImp.loc[:, 'PE']
    reg.fit(Ximp, Yimp)
    X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))
    return X


# ## Feature augmentation
# Our guess is that facies do not abrutly change from a given depth layer to the next one. Therefore, we consider features at neighboring layers to be somehow correlated. To possibly exploit this fact, let us perform feature augmentation by:
# - Aggregating features at neighboring depths.
# - Computing feature spatial gradient.
# 

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, seed=None, pe=True, N_neig=1):
    seed = seed or None
    
    if pe:
        X = make_pe(X, seed)
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows


# Augment features
X_aug, padded_rows = augment_features(X, well, depth)


# ## Generate training, validation and test data splits
# The choice of training and validation data is paramount in order to avoid overfitting and find a solution that generalizes well on new data. For this reason, we generate a set of training-validation splits so that:
# - Features from each well belongs to training or validation set.
# - Training and validation sets contain at least one sample for each class.
# 

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))


# ## Classification parameters optimization
# Let us perform the following steps for each set of parameters:
# - Select a data split.
# - Normalize features using a robust scaler.
# - Train the classifier on training data.
# - Test the trained classifier on validation data.
# - Repeat for all splits and average the F1 scores.
# 
# At the end of the loop, we select the classifier that maximizes the average F1 score on the validation set. Hopefully, this classifier should be able to generalize well on new data.
# 

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [100]  # [50, 100, 150]
M_grid = [10]  # [5, 10, 15]
S_grid = [25]  # [10, 25, 50, 75]
L_grid = [5] # [2, 3, 4, 5, 10, 25]
param_grid = []
for N in N_grid:
    for M in M_grid:
        for S in S_grid:
            for L in L_grid:
                param_grid.append({'N':N, 'M':M, 'S':S, 'L':L})


# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, clf):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat


# For each set of parameters
# score_param = []
# for param in param_grid:

#     # For each data split
#     score_split = []
#     for split in split_list:

#         # Remove padded rows
#         split_train_no_pad = np.setdiff1d(split['train'], padded_rows)

#         # Select training and validation data from current split
#         X_tr = X_aug[split_train_no_pad, :]
#         X_v = X_aug[split['val'], :]
#         y_tr = y[split_train_no_pad]
#         y_v = y[split['val']]

#         # Select well labels for validation data
#         well_v = well[split['val']]

#         # Train and test
#         y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)

#         # Score
#         score = f1_score(y_v, y_v_hat, average='micro')
#         score_split.append(score)

#     # Average score for this param
#     score_param.append(np.mean(score_split))
#     print('F1 score = %.3f %s' % (score_param[-1], param))

# # Best set of parameters
# best_idx = np.argmax(score_param)
# param_best = param_grid[best_idx]
# score_best = score_param[best_idx]
# print('\nBest F1 score = %.3f %s' % (score_best, param_best))


# ## Predict labels on test data
# Let us now apply the selected classification technique to test data.
# 

param_best = {'S': 25, 'M': 10, 'L': 5, 'N': 100}


# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')


# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values


y_pred = []
print('o' * 100)
for seed in range(100):
    np.random.seed(seed)

    # Make training data.
    X_train, padded_rows = augment_features(X, well, depth, seed=seed)
    y_train = y
    X_train = np.delete(X_train, padded_rows, axis=0)
    y_train = np.delete(y_train, padded_rows, axis=0) 
    param = param_best
    clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
                             max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
                             class_weight='balanced', random_state=seed), n_jobs=-1)
    
    # Make blind data.
    X_test, _ = augment_features(X_ts, well_ts, depth_ts, seed=seed, pe=False)

    # Train and test.
    y_ts_hat = train_and_test(X_train, y_train, X_test, well_ts, clf)
    
    # Collect result.
    y_pred.append(y_ts_hat)
    print('.', end='')
    
np.save('100_realizations.npy', y_pred)


