# In this notebook, we will use a multi-layer perceptron to develop time series forecasting models.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


"""
Rows having NaN values in column pm2.5 are dropped.
"""
df.dropna(subset=['pm2.5'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['pm2.5'])
g.set_title('Box plot of pm2.5')
plt.savefig('plots/ch5/B07887_05_06.png', format='png', dpi=300)


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'])
g.set_title('Time series of pm2.5')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')
plt.savefig('plots/ch5/B07887_05_10.png', format='png', dpi=300)


#Let's plot the series for six months to check if any pattern apparently exists.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=6,day=30)], color='g')
g.set_title('pm2.5 during 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')
plt.savefig('plots/ch5/B07887_05_07.png', format='png', dpi=300)

#Let's zoom in on one month.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=1,day=31)], color='g')
g.set_title('pm2.5 during Jan 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')
plt.savefig('plots/ch5/B07887_05_08.png', format='png', dpi=300)


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The pm2.5 variable is mixmax scaled to bound the tranformed variable within [0,1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_pm2.5'] = scaler.fit_transform(np.array(df['pm2.5']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of scaled pm2.5 is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_pm2.5'], color='b')
g.set_title('Time series of scaled pm2.5 in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_pm2.5'], color='r')
g.set_title('Time series of scaled pm2.5 in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm log_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_pm2.5'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_pm2.5'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,), dtype='float32')


#Dense layers are defined with linear activation
dense1 = Dense(32, activation='tanh')(input_layer)
dense2 = Dense(16, activation='tanh')(dense1)
dense3 = Dense(16, activation='tanh')(dense2)


# Multiple hidden layers and large number of neurons in each hidden layer gives neural networks the ability to model complex non-linearity of the underlying relations between regressors and target. However, deep neural networks can also overfit train data and give poor results on validation or test set. Dropout has been effectively used to regularize deep neural networks. In this example, a Dropout layer is added before the output layer. Dropout randomly sets p fraction of input neurons to zero before passing to the next layer. Randomly dropping inputs essentially acts as a bootstrap aggregating or bagging type of model ensembling. Random forest uses bagging by building trees on random subsets of input features. We use p=0.2 to dropout 20% of randomly selected input features.
# 

dropout_layer = Dropout(0.2)(dense3)


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. The box plot of pm2.5 shows the presence of outliers. Hence, mean absolute error (MAE) is used as absolute deviations suffer less fluctuations compared to squared deviations.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
# is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
# used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
# completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
# at which the loss function has been minimum.
# 

save_weights_at = os.path.join('keras_models', 'PRSA_data_PM2.5_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the pm2.5 from the best saved model. The model's predictions, which are on the scaled pm2.5, are inverse transformed to get predictions of original pm2.5.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_PM2.5_MLP_weights.01-0.0117.hdf5'))
preds = best_model.predict(X_val)
pred_pm25 = scaler.inverse_transform(preds)
pred_pm25 = np.squeeze(pred_pm25)


from sklearn.metrics import mean_absolute_error


mae = mean_absolute_error(df_val['pm2.5'].loc[7:], pred_pm25)
print('MAE for the validation set:', round(mae, 4))


#Let's plot the first 50 actual and predicted values of pm2.5.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['pm2.5'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_pm25[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')
plt.savefig('plots/ch5/B07887_05_09.png', format='png', dpi=300)


# In this notebook, we will use Gated Recurrent Unit RNN to develop a time series forecasting model.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['PRES'])
g.set_title('Box plot of Air Pressure')


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['PRES'])
g.set_title('Time series of Air Pressure')
g.set_xlabel('Index')
g.set_ylabel('Air Pressure readings in hPa')


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. The PRES variable is minmax scaled as shown in the following cell in order to bound the scaled features within [0, 1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_PRES'] = scaler.fit_transform(np.array(df['PRES']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of standardized PRES is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_PRES'], color='b')
g.set_title('Time series of scaled Air Pressure in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_PRES'], color='r')
g.set_title('Time series of scaled Air Pressure in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm scaled_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_PRES'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_PRES'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# The input to RNN layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only pm2.5 hence number of features per timestep is one. Number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays.
# 

X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of 3D arrays:', X_train.shape, X_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')


#LSTM layer is defined for seven timesteps
gru_layer = GRU(64, input_shape=(7,1), return_sequences=False)(input_layer)


dropout_layer = Dropout(0.2)(gru_layer)


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. Mean square error (mse) is used as the loss function.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mse', optimizer='adam')
ts_model.summary()


"""
The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
at which the loss function has been minimum.
"""
save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_GRU_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the air pressure from the best saved model. The model's predictions, which are on the minmax scaled  air-pressure, are inverse transformed to get predictions on original air pressure. The goodness-of-fit, R-squared is also calculated for the predictions on the original variable.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_GRU_weights.17-0.0001.hdf5'))
preds = best_model.predict(X_val)
pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)


from sklearn.metrics import r2_score
r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)
print('R-squared on validation set of the original air pressure:', r2)


#Let's plot the first 50 actual and predicted values of air pressure.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['PRES'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Air Pressure')
plt.ylabel('Air Pressure')
plt.xlabel('Index')





# In this notebook, we will use a multi-layer perceptron to develop time series forecasting models.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


"""
Rows having NaN values in column pm2.5 are dropped.
"""
df.dropna(subset=['pm2.5'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['pm2.5'])
g.set_title('Box plot of pm2.5')


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'])
g.set_title('Time series of pm2.5')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


#Let's plot the series for six months to check if any pattern apparently exists.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=6,day=30)], color='g')
g.set_title('pm2.5 during 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')

#Let's zoom in on one month.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=1,day=31)], color='g')
g.set_title('pm2.5 during Jan 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The pm2.5 variable is mixmax scaled to bound the tranformed variable within [0,1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_pm2.5'] = scaler.fit_transform(np.array(df['pm2.5']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of scaled pm2.5 is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_pm2.5'], color='b')
g.set_title('Time series of scaled pm2.5 in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_pm2.5'], color='r')
g.set_title('Time series of scaled pm2.5 in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm standardized_pm2.5 in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_pm2.5'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_pm2.5'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# The input to convolution layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only pm2.5 hence number of features per timestep is one. Number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays.
# 

#X_train and X_val are reshaped to 3D arrays
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),                 X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of arrays after reshaping:', X_train.shape, X_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')


# ZeroPadding1D layer is added next to add zeros at the begining and end of each series. Zeropadding ensure that the downstream convolution layer does not reduce the dimension of the output sequences. Pooling layer, added after the convolution layer is used to downsampling the input.
# 

#Add zero padding
zeropadding_layer = ZeroPadding1D(padding=1)(input_layer)


# The first argument of Conv1D is the number of filters, which determine the number of features in the output. Second argument indicates length of the 1D convolution window. The third argument is strides and represent the number of places to shift the convolution window. Lastly, setting use_bias as True, add a bias value during computation of an output feature. Here, the 1D convolution can be thought of as generating local AR models over rolling window of three time units.
# 

#Add 1D convolution layers
conv1D_layer1 = Conv1D(64, 3, strides=1, use_bias=True)(zeropadding_layer)
conv1D_layer2 = Conv1D(32, 3, strides=1, use_bias=True)(conv1D_layer1)


# AveragePooling1D is added next to downsample the input by taking average over pool size of three with stride of one timesteps. The average pooling in this case can be thought of as taking moving averages over a rolling window of three time units. We have used average pooling instead of max pooling to generate the moving averages.
# 

#Add AveragePooling1D layer
avgpooling_layer = AveragePooling1D(pool_size=3, strides=1)(conv1D_layer2)


# The preceeding pooling layer returns 3D output. Hence before passing to the output layer, a Flatten layer is added. The Flatten layer reshapes the input to (number of samples, number of timesteps*number of features per timestep), which is then fed to the output layer
# 

#Add Flatten layer
flatten_layer = Flatten()(avgpooling_layer)


#A couple of Dense layers are also added
dense_layer1 = Dense(32)(avgpooling_layer)
dense_layer2 = Dense(16)(dense_layer1)


dropout_layer = Dropout(0.2)(flatten_layer)


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. The box plot of pm2.5 shows the presence of outliers. Hence, mean absolute error (MAE) is used as absolute deviations suffer less fluctuations compared to squared deviations.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')#SGD(lr=0.001, decay=1e-5))
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
# is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
# used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
# completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
# at which the loss function has been minimum.
# 

save_weights_at = os.path.join('keras_models', 'PRSA_data_PM2.5_1DConv_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the pm2.5 from the best saved model. The model's predictions, which are on the standardized  pm2.5, are inverse transformed to get predictions of original pm2.5.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_PM2.5_1DConv_weights.18-0.0128.hdf5'))
preds = best_model.predict(X_val)
pred_pm25 = scaler.inverse_transform(preds)
pred_pm25 = np.squeeze(pred_pm25)


from sklearn.metrics import mean_absolute_error


mae = mean_absolute_error(df_val['pm2.5'].loc[7:], pred_pm25)
print('MAE for the validation set:', round(mae, 4))


#Let's plot the first 50 actual and predicted values of pm2.5.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['pm2.5'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_pm25[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')


# In this notebook, we will use Long Short Term Memory RNN to develop a time series forecasting model.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['PRES'])
g.set_title('Box plot of Air Pressure')


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['PRES'])
g.set_title('Time series of Air Pressure')
g.set_xlabel('Index')
g.set_ylabel('Air Pressure readings in hPa')


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The PRES variable is mixmax scaled to bound the tranformed variable within [0,1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_PRES'] = scaler.fit_transform(np.array(df['PRES']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of standardized PRES is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_PRES'], color='b')
g.set_title('Time series of scaled Air Pressure in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_PRES'], color='r')
g.set_title('Time series of scaled Air Pressure in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm log_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_PRES'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_PRES'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# The input to RNN layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only pm2.5 hence number of features per timestep is one. Number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays.
# 

X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),                 X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of 3D arrays:', X_train.shape, X_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')


#LSTM layer is defined for seven timesteps
lstm_layer = LSTM(64, input_shape=(7,1), return_sequences=False)(input_layer)


dropout_layer = Dropout(0.2)(lstm_layer)


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. Mean square error (mse) is used as the loss function.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mae', optimizer='adam')
ts_model.summary()


"""
The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
at which the loss function has been minimum.
"""
save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the air pressure from the best saved model. The model's predictions, which are on the scaled  air-pressure, are inverse transformed to get predictions on original air pressure. The goodness-of-fit, R-squared is also calculated for the predictions on the original variable.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_LSTM_weights.06-0.0087.hdf5'))
preds = best_model.predict(X_val)
pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)


from sklearn.metrics import r2_score
r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)
print('R-squared on validation set of the original air pressure:', r2)


#Let's plot the first 50 actual and predicted values of air pressure.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['PRES'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Air Pressure')
plt.ylabel('Air Pressure')
plt.xlabel('Index')
plt.savefig('plots/ch5/B07887_05_11.png', format='png', dpi=300)


# In this notebook, we will use Gated Recurrent Units to develop time series forecasting models.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


"""
Rows having NaN values in column pm2.5 are dropped.
"""
df.dropna(subset=['pm2.5'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['pm2.5'])
g.set_title('Box plot of pm2.5')


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'])
g.set_title('Time series of pm2.5')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


#Let's plot the series for six months to check if any pattern apparently exists.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=6,day=30)], color='g')
g.set_title('pm2.5 during 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')

#Let's zoom in on one month.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=1,day=31)], color='g')
g.set_title('pm2.5 during Jan 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The pm2.5 variable is mixmax scaled to bound the tranformed variable within [0,1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_pm2.5'] = scaler.fit_transform(np.array(df['pm2.5']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of scaled_pm2.5 is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_pm2.5'], color='b')
g.set_title('Time series of scaled pm2.5 in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_pm2.5'], color='r')
g.set_title('Time series of scaled pm2.5 in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm standardized_pm2.5 in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_pm2.5'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_pm2.5'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# The input to RNN layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only pm2.5 hence number of features per timestep is one. Number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays.
# 

#X_train and X_val are reshaped to 3D arrays
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of arrays after reshaping:', X_train.shape, X_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')


# The GRU layers are defined for seven timesteps. In this example, two GRU layers are stacked. The first GRU returns the output from each all seven timesteps. This output is a sequence and is fed to the second GRU which returns output only from the last step. The first GRU has sixty four hidden neurons in each timestep. Hence the sequence returned by the first GRU has sixty four features.
# 

gru_layer1 = GRU(64, input_shape=(7,1), return_sequences=True)(input_layer)
gru_layer2 = GRU(32, input_shape=(7,64), return_sequences=False)(gru_layer1)


dropout_layer = Dropout(0.2)(gru_layer2)


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. The box plot of pm2.5 shows the presence of outliers. Hence, mean absolute error (MAE) is used as absolute deviations suffer less fluctuations compared to squared deviations.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
# is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
# used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
# completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
# at which the loss function has been minimum.
# 

save_weights_at = os.path.join('keras_models', 'PRSA_data_PM2.5_GRU_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the pm2.5 from the best saved model. The model's predictions, which are on the standardized  pm2.5, are inverse transformed to get predictions of original pm2.5.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_PM2.5_GRU_weights.15-0.0116.hdf5'))
preds = best_model.predict(X_val)
pred_pm25 = scaler.inverse_transform(preds)
pred_pm25 = np.squeeze(pred_pm25)


from sklearn.metrics import mean_absolute_error


mae = mean_absolute_error(df_val['pm2.5'].loc[7:], pred_pm25)
print('MAE for the validation set:', round(mae, 4))


#Let's plot the first 50 actual and predicted values of pm2.5.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['pm2.5'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_pm25[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')
plt.savefig('plots/ch5/B07887_05_12.png', format='png', dpi=300)


# In this notebook, we will use a multi-layer perceptron to develop time series forecasting models.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temperature, dew point and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['PRES'])
g.set_title('Box plot of Air Pressure')
plt.savefig('plots/ch5/B07887_05_01.png', format='png', dpi=300)


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['PRES'])
g.set_title('Time series of Air Pressure')
g.set_xlabel('Index')
g.set_ylabel('Air Pressure readings in hPa')
plt.savefig('plots/ch5/B07887_05_02.png', format='png', dpi=300)


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The PRES variable is mixmax scaled to bound the tranformed variable within [0,1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_PRES'] = scaler.fit_transform(np.array(df['PRES']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not necessarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period is from
Jan 1st, 2010 to Dec 31st, 2014. The first four years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of standardized PRES are also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_PRES'], color='b')
g.set_title('Time series of scaled Air Pressure in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')
plt.savefig('plots/ch5/B07887_05_03.png', format='png', dpi=300)

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_PRES'], color='r')
g.set_title('Time series of scaled Air Pressure in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')
plt.savefig('plots/ch5/B07887_05_04.png', format='png', dpi=300)


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressors and 1-D array of target is created from the original 1-D array of column standardized_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_PRES'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_PRES'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,), dtype='float32')


#Dense layers are defined with linear activation
dense1 = Dense(32, activation='linear')(input_layer)
dense2 = Dense(16, activation='linear')(dense1)
dense3 = Dense(16, activation='linear')(dense2)


# Multiple hidden layers and large number of neurons in each hidden layer gives neural networks the ability to model complex non-linearity of the underlying relations between regressors and target. However, deep neural networks can also overfit train data and give poor results on validation or test set. Dropout has been effectively used to regularize deep neural networks. In this example, a Dropout layer is added before the output layer. Dropout randomly sets p fraction of input neurons to zero before passing to the next layer. Randomly dropping inputs essentially acts as a bootstrap aggregating or bagging type of model ensembling. Random forest uses bagging by building trees on random subsets of input features. We use p=0.2 to dropout 20% of randomly selected input features.
# 

dropout_layer = Dropout(0.2)(dense3)


#Finally, the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. Mean square error (MSE) is used as the loss function.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, Adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_squared_error', optimizer='adam')
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be used for a instance of back propagation. The validation dataset is also passed to evaluate the model after every epoch completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch, at which the loss function has been minimum.
# 

save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the air pressure from the best saved model. The model's predictions, which are on the scaled  air-pressure, are inverse transformed to get predictions on original air pressure. The goodness-of-fit or R squared is also calculated.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_MLP_weights.13-0.0001.hdf5'))
preds = best_model.predict(X_val)
pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)


from sklearn.metrics import r2_score


r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)
print('R-squared for the validation set:', round(r2,4))


#Let's plot the first 50 actual and predicted values of air pressure.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['PRES'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Air Pressure')
plt.ylabel('Air Pressure')
plt.xlabel('Index')
plt.savefig('plots/ch5/B07887_05_05.png', format='png', dpi=300)


# In this notebook, we will use LSTM layers to develop time series forecasting models.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


"""
Rows having NaN values in column pm2.5 are dropped.
"""
df.dropna(subset=['pm2.5'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['pm2.5'])
g.set_title('Box plot of pm2.5')


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'])
g.set_title('Time series of pm2.5')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


#Let's plot the series for six months to check if any pattern apparently exists.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=6,day=30)], color='g')
g.set_title('pm2.5 during 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')

#Let's zoom in on one month.
plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=1,day=31)], color='g')
g.set_title('pm2.5 during Jan 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The pm2.5 variable is mixmax scaled to bound the tranformed variable within [0,1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_pm2.5'] = scaler.fit_transform(np.array(df['pm2.5']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of scaled_pm2.5 is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_pm2.5'], color='b')
g.set_title('Time series of scaled pm2.5 in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_pm2.5'], color='r')
g.set_title('Time series of scaled pm2.5 in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm log_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_pm2.5'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_pm2.5'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


#X_train and X_val are reshaped to 3D arrays
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of arrays after reshaping:', X_train.shape, X_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense, Input, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')


# The LSTM layers are defined for seven timesteps. In this example, two LSTM layers are stacked. The first LSTM returns the output from each all seven timesteps. This output is a sequence and is fed to the second LSTM which returns output only from the last step. The first LSTM has sixty four hidden neurons in each timestep. Hence the sequence returned by the first LSTM has sixty four features.
# 

lstm_layer1 = LSTM(64, input_shape=(7,1), return_sequences=True)(input_layer)
lstm_layer2 = LSTM(32, input_shape=(7,64), return_sequences=False)(lstm_layer1)


dropout_layer = Dropout(0.2)(lstm_layer2)


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. The box plot of pm2.5 shows the presence of outliers. Hence, mean absolute error (MAE) is used as absolute deviations suffer less fluctuations compared to squared deviations.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')#SGD(lr=0.001, decay=1e-5))
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
# is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
# used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
# completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
# at which the loss function has been minimum.
# 

save_weights_at = os.path.join('keras_models', 'PRSA_data_PM2.5_LSTM_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the pm2.5 from the best saved model. The model's predictions, which are on the standardized  pm2.5, are inverse transformed to get predictions of original pm2.5.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_PM2.5_LSTM_weights.09-0.0117.hdf5'))
preds = best_model.predict(X_val)
pred_pm25 = scaler.inverse_transform(preds)
pred_pm25 = np.squeeze(pred_pm25)


from sklearn.metrics import mean_absolute_error


mae = mean_absolute_error(df_val['pm2.5'].loc[7:], pred_pm25)
print('MAE for the validation set:', round(mae, 4))


#Let's plot the first 50 actual and predicted values of pm2.5.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['pm2.5'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_pm25[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')


# In this notebook, we will use a multi-layer perceptron to develop time series forecasting models.
# The dataset used for the examples of this notebook is on air pollution measured by concentration of
# particulate matter (PM) of diameter less than or equal to 2.5 micrometers. There are other variables
# such as air pressure, air temparature, dewpoint and so on.
# Two time series models are developed - one on air pressure and the other on pm2.5.
# The dataset has been downloaded from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


#set current working directory
os.chdir('D:/Practical Time Series')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('datasets/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)


#Let's see the first five rows of the DataFrame
df.head()


# To make sure that the rows are in the right order of date and time of observations,
# a new column datetime is created from the date and time related columns of the DataFrame.
# The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order
# over this column.
# 

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'],
                                                                                          hour=row['hour']), axis=1)
df.sort_values('datetime', ascending=True, inplace=True)


#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['PRES'])
g.set_title('Box plot of PRES')


plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df['PRES'])
g.set_title('Time series of PRES')
g.set_xlabel('Index')
g.set_ylabel('PRES readings')


# Gradient descent algorithms perform better (for example converge faster) if the variables are wihtin range [-1, 1]. Many sources relax the boundary to even [-3, 3]. The PRES variable is mixmax scaled to bound the tranformed variable within [0,1].
# 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_PRES'] = scaler.fit_transform(np.array(df['PRES']).reshape(-1, 1))


# Before training the model, the dataset is split in two parts - train set and validation set.
# The neural network is trained on the train set. This means computation of the loss function, back propagation
# and weights updated by a gradient descent algorithm is done on the train set. The validation set is
# used to evaluate the model and to determine the number of epochs in model training. Increasing the number of 
# epochs will further decrease the loss function on the train set but might not neccesarily have the same effect
# for the validation set due to overfitting on the train set.Hence, the number of epochs is controlled by keeping
# a tap on the loss function computed for the validation set. We use Keras with Tensorflow backend to define and train
# the model. All the steps involved in model training and validation is done by calling appropriate functions
# of the Keras API.
# 

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)


#First five rows of train
df_train.head()


#First five rows of validation
df_val.head()


#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True)


"""
The train and validation time series of scaled PRES is also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_train['scaled_PRES'], color='b')
g.set_title('Time series of scaled PRES in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled PRES readings')

plt.figure(figsize=(5.5, 5.5))
g = sns.tsplot(df_val['scaled_PRES'], color='r')
g.set_title('Time series of scaled PRES in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled PRES readings')


# Now we need to generate regressors (X) and target variable (y) for train and validation. 2-D array of regressor and 1-D array of target is created from the original 1-D array of columm standardized_PRES in the DataFrames. For the time series forecasting model, Past seven days of observations are used to predict for the next day. This is equivalent to a AR(7) model. We define a function which takes the original time series and the number of timesteps in regressors as input to generate the arrays of X and y.
# 

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y


X_train, y_train = makeXy(df_train['scaled_PRES'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)


X_val, y_val = makeXy(df_val['scaled_PRES'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)


# The input to convolution layers must be of shape (number of samples, number of timesteps, number of features per timestep). In this case we are modeling only PRES hence number of features per timestep is one. Number of timesteps is seven and number of samples is same as the number of samples in X_train and X_val, which are reshaped to 3D arrays.
# 

#X_train and X_val are reshaped to 3D arrays
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),                 X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of arrays after reshaping:', X_train.shape, X_val.shape)


# Now we define the MLP using the Keras Functional API. In this approach a layer can be declared as the input of the following layer at the time of defining the next layer. 
# 

from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')


# ZeroPadding1D layer is added next to add zeros at the beginning and end of each series. Zeropadding ensure that the downstream convolution layer does not reduce the dimension of the output sequences. Pooling layer, added after the convolution layer is used to downsampling the input.
# 

#Add zero padding
zeropadding_layer = ZeroPadding1D(padding=1)(input_layer)


# The first argument of Conv1D is the number of filters, which determine the number of features in the output. Second argument indicates length of the 1D convolution window. The third argument is strides and represent the number of places to shift the convolution window. Lastly, setting use_bias as True, add a bias value during computation of an output feature. Here, the 1D convolution can be thought of as generating local AR models over rolling window of three time units.
# 

#Add 1D convolution layer
conv1D_layer = Conv1D(64, 3, strides=1, use_bias=True)(zeropadding_layer)


# AveragePooling1D is added next to downsample the input by taking average over pool size of three with stride of one timesteps. The average pooling in this case can be thought of as taking moving averages over a rolling window of three time units. We have used average pooling instead of max pooling to generate the moving averages.
# 

#Add AveragePooling1D layer
avgpooling_layer = AveragePooling1D(pool_size=3, strides=1)(conv1D_layer)


# The preceeding pooling layer returns 3D output. Hence before passing to the output layer, a Flatten layer is added. The Flatten layer reshapes the input to (number of samples, number of timesteps*number of features per timestep), which is then fed to the output layer
# 

#Add Flatten layer
flatten_layer = Flatten()(avgpooling_layer)


dropout_layer = Dropout(0.2)(flatten_layer)


#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making
# predictions. Mean squared error (MSE) is used as the loss function.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation
# and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses
# different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.
# 

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')#SGD(lr=0.001, decay=1e-5))
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training 
# is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be
# used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch
# completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch,
# at which the loss function has been minimum.
# 

save_weights_at = os.path.join('keras_models', 'PRSA_data_Air_Pressure_1DConv_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)


# Prediction are made for the PRES from the best saved model. The model's predictions, which are on the standardized  PRES, are inverse transformed to get predictions of original PRES.
# 

best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_1DConv_weights.16-0.0097.hdf5'))
preds = best_model.predict(X_val)
pred_PRES = np.squeeze(scaler.inverse_transform(preds))


from sklearn.metrics import r2_score


r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)
print('R-squared for the validation set:', round(r2, 4))


#Let's plot the first 50 actual and predicted values of PRES.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['PRES'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted PRES')
plt.ylabel('PRES')
plt.xlabel('Index')


