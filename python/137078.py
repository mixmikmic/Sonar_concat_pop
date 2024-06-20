#%load_ext autoreload
#%autoreload 2
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# basic functionalities
import re
import os
import sys
import datetime
import itertools
import math 


# data transforamtion and manipulation
import pandas as pd
import pandas_datareader.data as web
import numpy as np
# prevent crazy long pandas prints
pd.options.display.max_columns = 16
pd.options.display.max_rows = 16
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)


# remove warnings
import warnings
warnings.filterwarnings('ignore')


# plotting and plot stying
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
plt.style.use('seaborn')
#sns.set_style("whitegrid", {'axes.grid' : False})
#set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"


# deep learning
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model 
from keras.layers import LSTM

# sklearn functionalities
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# jupyter wdgets
from ipywidgets import interactive, widgets, RadioButtons, ToggleButtons, Select, FloatSlider, FloatProgress
from IPython.display import set_matplotlib_formats, Image, IFrame


# # Artificial Neural Networks for Time Series Forecasting
# 
# <strong>Artificial neural networks (ANNs)</strong> are computing systems inspired by the biological neural networks that constitute bilological brains. Such systems learn (progressively improve performance) to do tasks by considering examples, generally without task-specific programming (if-then clauses).
# 
# An ANN is based on a collection of connected units called <strong>Perceptrons</strong> analogous to neurons in a biological brain. Each connection (synapse) between neurons can transmit a signal to another neuron. The receiving (postsynaptic) neuron can process the signal(s) and then signal downstream neurons connected to it. - <a href="https://en.wikipedia.org/wiki/Artificial_neural_network">Wikipedia</a>
# 
# https://www.youtube.com/watch?v=MVyauNNinC0
# 
# 
# ## Neural Network Basics
# 
# The basic constitutes of a ANN are **Perceptron**. A perceptron takes an input matrix $X$ and applies weights $W$ to each resulting in a weighted sum. This summed is transformed using an **Activation Function** and passed as the output $y$, also called **activation**. An activation function is a decision making function that determines the presence / importance of particular neural feature. It is (typically) mapped between 0 and 1, where zero mean the feature is not there, while one means the feature is present.
# 

IFrame("./img/perceptron.pdf", width=1000, height=800)


# The purpose of the activation function is to introduce non-linearity into the network. Non-linear means that the output cannot be reproduced from a linear combination of the inputs. Since the perceptron is a **linear classifier**, i.e. it can only distinguish between two forms of output (e.g. 0 and 1), it is not capable of solving XOR problems like this
# 

# initialize figur and axes
fig, axes = plt.subplots(1, 2, sharey=False, sharex=False);
fig.suptitle('XOR Problem', fontsize=24, fontweight='bold')

# classifiable plot
axes[0].plot([0,0,1], [0,1,0], 'o', color='grey')
axes[0].plot([1], [1], 'X', color='red')
axes[0].plot([0.25, 1.25], [1.25, 0.25], color='black')
axes[0].set_xlim((-0.25, 1.25))
axes[0].set_ylim((-0.25, 1.25))

# unclassifiable plot
axes[1].plot([0,1], [1,0], 'o', color='grey')
axes[1].plot([0, 1], [0, 1], 'X', color='red')
axes[1].set_xlim((-0.25, 1.25))
axes[1].set_ylim((-0.25, 1.25))

plt.show()


# The limitations of the perceptron to estimate non-linear relationships is solved by chaining together multiple perceptrons to form a network, an artificial neural network. The same input pass through principle as for a single perceptron applies for all network nodes which will eventually result in the output estimate $\hat{y}$.
# 

IFrame("./img/network.pdf", width=1000, height=800)


# The estimate $\hat{y}$ is then compared to the actual observation $y$ using a <strong>Cost Function</strong> $E$ which computes the sum of squared errors. 
# 

IFrame("./img/error_computation.pdf", width=1000, height=800)


# If the cost function is minimized with respect to all weight parameters in $W$ the neural network learns the most effective (cheapest) representation of the data that leads to the desired result $y$. According to each weight's contribution to the total error $\delta E / \delta w_{ij}$ this error is propagated back through the network to the corresponding which which are than adjusted by a **learning rate**. This process is called **Back Propagation** since the errors are propagated back through the network. The optimal parameter constellati0on is than found by the **Gradient Descent Algorithm**
# 
# 1. initialize with weights w_{ij}^{(0)}
# 2. for s = 1 to S do:
# 3.     compute $E[w_{ij}^{(s-1)}]$
# 4.     compute $w_{ij}^{(s)} = w_{ij}^{(s-1)} - \eta \frac{\partial E}{\partial w_{ij}^{(s)}}$   
# 
# until $E$ is sufficient small
# 

IFrame("./img/gradient_descent.pdf", width=1000, height=800)


# ## A Supervised Learning Task
# 
# Time series prediction problems are inherently different from supervised learning problems in that obervastions posses a timely ordering and no observeable input. Thus, the time series prediction problem has to be converted into a regression problem.
# 
# A supervised learing problem requires data of the form $(X, y)$ where $y$ is the observeable output and $X$ is a matrix of input data which is assumed to cause the observed output. Since the output variable $y$ is continuous this problem is called a supervised regression problem.
# 

np.random.seed(7)


df = pd.read_csv('./data/passengers.csv', sep=';', parse_dates=True, index_col=0)
data = df.values

# using keras often requires the data type float32
data = data.astype('float32')

# slice the data
train = data[0:120, :]
test = data[120:, :]

print(len(train), len(test))


data


# The <code>prepare_data</code> function will be used to transform the time series into a regression problem. The <code>lags</code> argument takes an integer which corresponds the the number of previous time steps to use as input variables to predict the next time period. The default value is one but will be changed in a next iteration.
# 
# With the default setting a data set will be created where $X$ is the number of passengers at time $t$ and $y$ is the number of passengers at time $t+1$.
# 

len(train)


def prepare_data(data, lags=1):
    """
    Create lagged data from an input time series
    """
    X, y = [], []
    for row in range(len(data) - lags - 1):
        a = data[row:(row + lags), 0]
        X.append(a)
        y.append(data[row + lags, 0])
    return np.array(X), np.array(y)


lags = 1
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
y_true = y_test     # due to naming convention


y_train


X_train


y_train


# The data set now has the following form
# <pre>
# X       y
# 112     118
# 118     132
# 132     129
# 129     121
# 121     135
# </pre>
# That is, the function has successfully shifted the data for one time step and saved this new shifted series to an array.
# 

# plot the created data
plt.plot(y_test, label='y or t+1')
plt.plot(X_test, label='X or t', color='red')
plt.legend(loc='upper left')
plt.show()


# ## Multilayer Perceptron Network
# 
# As previously, the data is sliced up into a <code>train</code> and <code>test</code> set to evaluate the performance of a two-year-ahead forecast.
# 
# The first simple network will have one input (size of the <code>lags</code> variable), one hidden layer with 8 neurons and an output layer. The model is fitted using the MSE criterion and rectified linear units as activation function
# 

# create and fit Multilayer Perceptron model
mdl = Sequential()
mdl.add(Dense(8, input_dim=lags, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2)


# estimate model performance
train_score = mdl.evaluate(X_train, y_train, verbose=0)
print('Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, math.sqrt(train_score)))
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))


# generate predictions for training
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)


# shift train predictions for plotting
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[lags: len(train_predict) + lags, :] = train_predict

# shift test predictions for plotting
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1, :] = test_predict

# plot baseline and predictions
plt.plot(data, label='Observed');
plt.plot(train_predict_plot, label='Prediction for train', color='orange');
plt.plot(test_predict_plot, label='Prediction for test', color='darkorange');
plt.legend(loc='best');
plt.show()


mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='orange')
plt.legend(loc='best');
plt.show()


# Since the neural network has only been fed by the last observation, it did not have much choice but to learn to apply observation $t$ for the prediction of $t+1$.
# 

# ## Multilayer perceptron with window
# 

# reshape dataset
lags = 3
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)


# plot the created data
plt.plot(y_test, label='y or t+1')
plt.plot(X_test, label='X or t', color='red')
plt.legend(loc='best')
plt.show()


# create and fit Multilayer Perceptron model
mdl = Sequential()
mdl.add(Dense(12, input_dim=lags, activation='relu'))
mdl.add(Dense(8, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=400, batch_size=2, verbose=2)


# Estimate model performance
train_score = mdl.evaluate(X_train, y_train, verbose=0)
print('Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, math.sqrt(train_score)))
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))


# generate predictions for training
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)

# shift train predictions for plotting
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[lags: len(train_predict) + lags, :] = train_predict

# shift test predictions for plotting
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(lags * 2)+1:len(data)-1, :] = test_predict

# plot observation and predictions
plt.plot(data, label='Observed');
plt.plot(train_predict_plot, label='Prediction for train', color='orange');
plt.plot(test_predict_plot, label='Prediction for test', color='darkorange');
plt.legend(loc='best');
plt.show()


y_test


mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='orange')
plt.legend(loc='best');
plt.show()


# # LSTM Recurrent Neural Network
# 
# Long short-term memory (LSTM) is a recurrent neural network (RNN) architecture that remembers values over arbitrary intervals. Stored values are not modified as learning proceeds. RNNs allow forward and backward connections between neurons. An LSTM is well-suited to classify, process and predict time series given time lags of unknown size and duration between important events. - <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">Wikipedia</a>
# 
# ## Architecture
# 
# 
# 
# LSTM blocks contain three or four "gates" that control information flow. These gates are implemented using the logistic function to compute a value between 0 and 1. Multiplication is applied with this value to partially allow or deny information to flow into or out of the memory. For example, an "input" gate controls the extent to which a new value flows into the memory. A "forget" gate controls the extent to which a value remains in memory. An "output" gate controls the extent to which the value in memory is used to compute the output activation of the block. (In some implementations, the input and forget gates are merged into a single gate. The motivation for combining them is that the time to forget is when a new value worth remembering becomes available.)
# 

# fix random seed for reproducibility
np.random.seed(1)

# load the dataset
df = pd.read_csv('./data/passengers.csv', sep=';', parse_dates=True, index_col=0)
data = df.values
data = data.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

# split into train and test sets
train = dataset[0:120, :]
test = dataset[120:, :]

# reshape into X=t and Y=t+1
lags = 3
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(Dense(3, input_shape=(1, lags), activation='relu'))
model.add(LSTM(4))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)


# make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# invert transformation
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])


# calculate root mean squared error
train_score = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
print('Train Score: {:.2f} RMSE'.format(train_score))
test_score = math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print('Test Score: {:.2f} RMSE'.format(test_score))


# shift train predictions for plotting
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[lags:len(train_predict)+lags, :] = train_predict

# shift test predictions for plotting
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (lags * 2)+1:len(data)-1, :] = test_predict

# plot observation and predictions
plt.plot(data, label='Observed');
plt.plot(train_predict_plot, label='Prediction for train', color='orange');
plt.plot(test_predict_plot, label='Prediction for test', color='darkorange');
plt.legend(loc='best');
plt.show()


mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='orange')
plt.legend(loc='best');
plt.show()











#%load_ext autoreload
#%autoreload 2
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# basic functionalities
import re
import os
import sys
import datetime
import itertools
import math 


# data transforamtion and manipulation
import pandas as pd
import pandas_datareader.data as web
import numpy as np
# prevent crazy long pandas prints
pd.options.display.max_columns = 16
pd.options.display.max_rows = 16
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)


# remove warnings
import warnings
warnings.filterwarnings('ignore')


# plotting and plot styling
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
plt.style.use('seaborn')
#sns.set_style("whitegrid", {'axes.grid' : False})
#set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"


# jupyter wdgets
from ipywidgets import interactive, widgets, RadioButtons, ToggleButtons, Select, FloatSlider, FloatProgress
from IPython.display import set_matplotlib_formats, Image


# # Time Series Forecasting wih Prophet
# 
# Prophet is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with <strong>daily periodicity data</strong> with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers. - <a href="https://facebookincubator.github.io/prophet/">Prophet</a>
# 
# ## The Model
# 
# Forecasting in the domain of Prophet is a curve-fitting task. The underlying model has an additive form
# 
# $$
# y(t) = d(t) + s(t) + h(t) + \varepsilon_t
# $$
# 
# where $d(t)$ denotes a trend function modeling non-periodic changes, $s(t)$ denotes seasonality modeling periodic changes and $h(t)$ representing the effects of holidays. This model assumes time as its only regressor, however, linear and non-linear transformations are included if it increases the models fit. Hence, 
# 

from fbprophet import Prophet


# The input to Prophet is always a dataframe with two columns: <code>ds</code> and <code>y</code>. The <code>ds</code> (datestamp) column must contain a date or datetime (either is fine). The <code>y</code> column must be numeric, and represents the measurement to forecast.
# 

df = pd.read_csv('./data/passengers.csv', sep=';', header=0, parse_dates=True)

# create new coumns, specific headers needed for Prophet
df['ds'] = df['month']
df['y'] = pd.DataFrame(df['n_passengers'])
df.pop('month')
df.pop('n_passengers')


# <code>Prophet</code> assume an additive  time series model and thus, the data is transformed using the log operator of <code>numpy</code>.
# 

df['y'] = pd.DataFrame(np.log(df['y']))
df.head()


ax = df.set_index('ds').plot();
ax.set_ylabel('Passengers');
ax.set_xlabel('Date');

plt.show()


# train test split
df_train = df[:120]
df_test = df[120:]


mdl = Prophet(interval_width=0.95)


# fit the model on the training data
mdl.fit(df_train)


# define future time frame
future = mdl.make_future_dataframe(periods=24, freq='MS')
future.tail()


# The <code>predict</code> method assigns each row in future a predicted value <code>yhat</code>.
# 

# generate the forecast
forecast = mdl.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# <code>mdl</code> is a <code>Prophet</code> object and and can be plotted with predefined settings
# 

mdl.plot(forecast);
plt.show()


# Similar to the <code>statsmodels</code> module a time series decomposition method is available.
# 

# plot time series components
mdl.plot_components(forecast);
plt.show()


# retransform using e
y_hat = np.exp(forecast['yhat'][120:])
y_true = np.exp(df_test['y'])

# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))


# prepare data for plotting
#.reindex(pd.date_range(start='1959-01-01', end='1960-12-01', freq='MS'))
#.reindex(pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS'))
y_hat_plot = pd.DataFrame(y_hat)
y_true_plot = pd.DataFrame(np.exp(df['y']))


len(pd.date_range(start='1959-01-01', end='1960-12-01', freq='MS'))


y_true_plot


plt.plot(y_true_plot, label='Original');
plt.plot(y_hat_plot, color='orange', label='Forecast');
ax.set_ylabel('Passengers');
ax.set_xlabel('Date');

plt.show()








