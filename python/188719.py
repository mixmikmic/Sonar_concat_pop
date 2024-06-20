import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)


tesla = pd.read_csv('TSLA.csv')
tesla = tesla[['Date','Open','High','Low','Close']]
print(tesla.shape)
tesla.head()


tesla_2011 = pd.read_csv('TSLA-2011.csv')
tesla_2011 = tesla_2011[['Date','Open','High','Low','Close']]
print(tesla_2011.shape)
tesla_2011.head()


import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
from datetime import date
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

df_cp = tesla.copy()
df_cp.Date = date2num(pd.to_datetime(tesla.Date).dt.to_pydatetime())
ax1 = plt.subplot2grid((1,1), (0,0))
candlestick_ohlc(ax1,df_cp.values, width=0.4, colorup='#77d879', colordown='#db3f3f',alpha=2)
x_range = np.arange(df_cp.shape[0])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


ax1 = plt.subplot2grid((1,1), (0,0))
ret=candlestick_ohlc(ax1,df_cp.iloc[:100,:].values, width=0.4, colorup='#77d879', colordown='#db3f3f',alpha=2)
x_range = np.arange(df_cp.shape[0])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


tesla.Close.plot()


tesla_2011.Close.plot()


tesla.plot(kind = "line", y = ['Open', 'High', 'Low','Close'])


tesla_2011.plot(kind = "line", y = ['Open', 'High', 'Low','Close'])


tesla_2011['months'] = pd.DatetimeIndex(tesla_2011['Date']).month
tesla_2011['year'] = pd.DatetimeIndex(tesla_2011['Date']).year
tesla_2011.head()


teslaPivot = pd.pivot_table(tesla_2011, values = "Close", columns = "year", index = "months")


teslaPivot.head()


teslaPivot.plot()


teslaPivot.plot(subplots = True, figsize=(15, 15), layout=(4,4), sharey=True)


tesla.Close.plot(kind = "hist", bins = 30)


tesla['Closelog'] = np.log(tesla.Close)
tesla.head()


tesla.Closelog.plot(kind = "hist", bins = 30)


tesla.Closelog.plot()


model_mean_pred = tesla.Closelog.mean()
# reverse log e
tesla["Closemean"] = np.exp(model_mean_pred)
tesla.plot(kind="line", x="Date", y = ["Close", "Closemean"])


from sklearn import linear_model
x = np.arange(tesla.shape[0]).reshape((-1,1))
y = tesla.Close.values.reshape((-1,1))
reg = linear_model.LinearRegression()
pred = reg.fit(x, y).predict(x)


tesla['linear'] = pred
tesla.plot(kind="line", x="Date", y = ["Close", "Closemean", "linear"])


tesla.Date = pd.DatetimeIndex(tesla.Date)
tesla.index = pd.PeriodIndex(tesla.Date, freq='D')
tesla = tesla.sort_values(by = "Date")
tesla.head()


tesla['timeIndex']= tesla.Date - tesla.Date.min()
tesla["timeIndex"] =tesla["timeIndex"] / np.timedelta64(1, 'D')
tesla.head()


tesla["timeIndex"] = tesla["timeIndex"].round(0).astype(int)
tesla.tail()


import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller


model_linear = smf.ols('Closelog ~ timeIndex', data = tesla).fit()
model_linear.summary()


model_linear.params


model_linear_pred = model_linear.predict()
model_linear_pred.shape


tesla['linear_stats'] = model_linear_pred
tesla.head()


tesla.plot(kind="line", x="timeIndex", y = ["Closelog", 'linear_stats'])


model_linear.resid.plot(kind = "bar").get_xaxis().set_visible(False)


model_linear_forecast_auto = model_linear.predict(exog = pd.DataFrame(dict(timeIndex=252), index=[0]))
model_linear_forecast_auto


tesla['pricelinear'] = np.exp(model_linear_pred)
tesla.head()


tesla.plot(kind="line", x="timeIndex", y = ["Close", "Closemean", "pricelinear"])


tesla["CloselogShift1"] = tesla.Closelog.shift()
tesla.head()


tesla.plot(kind= "scatter", y = "Closelog", x = "CloselogShift1", s = 50)


tesla["CloselogDiff"] = tesla.Closelog - tesla.CloselogShift1
tesla.CloselogDiff.plot()


tesla["CloseRandom"] = np.exp(tesla.CloselogShift1)
tesla.head()


def adf(ts):
    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    orig = plt.plot(ts.values, color='blue',label='Original')
    mean = plt.plot(rolmean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolstd.values, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    adftest = adfuller(ts, autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',
                                              'Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    return adfoutput


tesla['CloselogMA12'] = pd.rolling_mean(tesla.Closelog, window = 12)
tesla.plot(kind ="line", y=["CloselogMA12", "Closelog"])


ts = tesla.Closelog - tesla.CloselogMA12
ts.dropna(inplace = True)
adf(ts)


# if test statistic < critical value (any), we can assume this data is stationary.
# 

half_life = 12
tesla['CloselogExp12'] = pd.ewma(tesla.Closelog, halflife=half_life)
1 - np.exp(np.log(0.5)/half_life)


tesla.plot(kind ="line", y=["CloselogExp12", "Closelog"])


tesla["CloseExp12"] = np.exp(tesla.CloselogExp12)
tesla.tail()


tesla.plot(kind="line", x="timeIndex", y = ["Close", "Closemean", "pricelinear", 
                                             "CloseRandom", "CloseExp12"])


ts = tesla.Closelog - tesla.CloselogExp12
ts.dropna(inplace = True)
adf(ts)


from statsmodels.tsa.seasonal import seasonal_decompose
tesla.index = tesla.index.to_datetime()


decomposition = seasonal_decompose(tesla.Closelog,freq=31)


decomposition.plot()


ts = tesla.Closelog
ts_diff = tesla.CloselogDiff
ts_diff.dropna(inplace = True)


from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)


ACF = pd.Series(lag_acf)


ACF.plot(kind = "bar")


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import model
import time
from datetime import datetime
from datetime import timedelta
sns.set()


df = pd.read_csv('GOOG-year.csv')
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
df.head()


minmax = MinMaxScaler().fit(df.iloc[:, 3].values.reshape((-1,1)))
close_normalize = minmax.transform(df.iloc[:, 3].values.reshape((-1,1))).reshape((-1))


close_normalize.shape


class encoder:
    def __init__(self, input_, dimension = 2, learning_rate = 0.01, hidden_layer = 256, epoch = 20):
        input_size = input_.shape[1]
        self.X = tf.placeholder("float", [None, input_.shape[1]])
    
        weights = {
        'encoder_h1': tf.Variable(tf.random_normal([input_size, hidden_layer])),
        'encoder_h2': tf.Variable(tf.random_normal([hidden_layer, dimension])),
        'decoder_h1': tf.Variable(tf.random_normal([dimension, hidden_layer])),
        'decoder_h2': tf.Variable(tf.random_normal([hidden_layer, input_size])),
        }
    
        biases = {
        'encoder_b1': tf.Variable(tf.random_normal([hidden_layer])),
        'encoder_b2': tf.Variable(tf.random_normal([dimension])),
        'decoder_b1': tf.Variable(tf.random_normal([hidden_layer])),
        'decoder_b2': tf.Variable(tf.random_normal([input_size])),
        }
    
        first_layer_encoder = tf.nn.sigmoid(tf.add(tf.matmul(self.X, weights['encoder_h1']), biases['encoder_b1']))
        self.second_layer_encoder = tf.nn.sigmoid(tf.add(tf.matmul(first_layer_encoder, weights['encoder_h2']), biases['encoder_b2']))
        first_layer_decoder = tf.nn.sigmoid(tf.add(tf.matmul(self.second_layer_encoder, weights['decoder_h1']), biases['decoder_b1']))
        second_layer_decoder = tf.nn.sigmoid(tf.add(tf.matmul(first_layer_decoder, weights['decoder_h2']), biases['decoder_b2']))
        self.cost = tf.reduce_mean(tf.pow(self.X - second_layer_decoder, 2))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        for i in range(epoch):
            last_time = time.time()
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: input_})
            if (i + 1) % 10 == 0:
                print('epoch:', i + 1, 'loss:', loss, 'time:', time.time() - last_time)
    
    def encode(self, input_):
        return self.sess.run(self.second_layer_encoder, feed_dict={self.X: input_})


tf.reset_default_graph()
Encoder=encoder(close_normalize.reshape((-1,1)), 32, 0.01, 128, 100)
thought_vector = Encoder.encode(close_normalize.reshape((-1,1)))
thought_vector.shape


from sklearn.ensemble import *
ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
bagging = BaggingRegressor(n_estimators=500)
et = ExtraTreesRegressor(n_estimators=500)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=500)


ada.fit(thought_vector[:-1, :], close_normalize[1:])
bagging.fit(thought_vector[:-1, :], close_normalize[1:])
et.fit(thought_vector[:-1, :], close_normalize[1:])
gb.fit(thought_vector[:-1, :], close_normalize[1:])
rf.fit(thought_vector[:-1, :], close_normalize[1:])


plt.bar(np.arange(32), ada.feature_importances_)
plt.title('ada boost important feature')
plt.show()


plt.bar(np.arange(32), et.feature_importances_)
plt.title('et important feature')
plt.show()


plt.bar(np.arange(32), gb.feature_importances_)
plt.title('gb important feature')
plt.show()


plt.bar(np.arange(32), rf.feature_importances_)
plt.title('rf important feature')
plt.show()


ada_pred=ada.predict(thought_vector)
bagging_pred=bagging.predict(thought_vector)
et_pred=et.predict(thought_vector)
gb_pred=gb.predict(thought_vector)
rf_pred=rf.predict(thought_vector)


ada_actual = np.hstack([close_normalize[0],ada_pred[:-1]])
bagging_actual = np.hstack([close_normalize[0],bagging_pred[:-1]])
et_actual = np.hstack([close_normalize[0],et_pred[:-1]])
gb_actual = np.hstack([close_normalize[0],gb_pred[:-1]])
rf_actual = np.hstack([close_normalize[0],rf_pred[:-1]])
stack_predict = np.vstack([ada_actual,bagging_actual,et_actual,gb_actual,rf_actual,close_normalize]).T
corr_df = pd.DataFrame(stack_predict)


sns.heatmap(corr_df.corr(), annot=True)
plt.show()


# # Wow, I do not expect this heatmap. Totally a heat!
# 

import xgboost as xgb
params_xgd = {
    'max_depth': 7,
    'objective': 'reg:logistic',
    'learning_rate': 0.05,
    'n_estimators': 10000
    }
train_Y = close_normalize[1:]
clf = xgb.XGBRegressor(**params_xgd)
clf.fit(stack_predict[:-1,:],train_Y, eval_set=[(stack_predict[:-1,:],train_Y)], 
        eval_metric='rmse', early_stopping_rounds=20, verbose=False)


xgb_pred = clf.predict(stack_predict)
xgb_actual = np.hstack([close_normalize[0],xgb_pred[:-1]])
date_original=pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()


def reverse_close(array):
    return minmax.inverse_transform(array.reshape((-1,1))).reshape((-1))


plt.figure(figsize = (15,6))
x_range = np.arange(df.Close.shape[0])
plt.plot(x_range, df.Close, label = 'Real Close')
plt.plot(x_range, reverse_close(ada_pred), label = 'ada Close')
plt.plot(x_range, reverse_close(bagging_pred), label = 'bagging Close')
plt.plot(x_range, reverse_close(et_pred), label = 'et Close')
plt.plot(x_range, reverse_close(gb_pred), label = 'gb Close')
plt.plot(x_range, reverse_close(rf_pred), label = 'rf Close')
plt.plot(x_range, reverse_close(xgb_pred), label = 'xgb stacked Close')
plt.legend()
plt.xticks(x_range[::50], date_original[::50])
plt.title('stacked')
plt.show()


ada_list = ada_pred.tolist()
bagging_list = bagging_pred.tolist()
et_list = et_pred.tolist()
gb_list = gb_pred.tolist()
rf_list = rf_pred.tolist()
xgb_list = xgb_pred.tolist()
def predict(count, history = 5):
    for i in range(count):
        roll = np.array(xgb_list[-history:])
        thought_vector = Encoder.encode(roll.reshape((-1,1)))
        ada_pred=ada.predict(thought_vector)
        bagging_pred=bagging.predict(thought_vector)
        et_pred=et.predict(thought_vector)
        gb_pred=gb.predict(thought_vector)
        rf_pred=rf.predict(thought_vector)
        ada_list.append(ada_pred[-1])
        bagging_list.append(bagging_pred[-1])
        et_list.append(et_pred[-1])
        gb_list.append(gb_pred[-1])
        rf_list.append(rf_pred[-1])
        ada_actual = np.hstack([xgb_list[-history],ada_pred[:-1]])
        bagging_actual = np.hstack([xgb_list[-history],bagging_pred[:-1]])
        et_actual = np.hstack([xgb_list[-history],et_pred[:-1]])
        gb_actual = np.hstack([xgb_list[-history],gb_pred[:-1]])
        rf_actual = np.hstack([xgb_list[-history],rf_pred[:-1]])
        stack_predict = np.vstack([ada_actual,bagging_actual,et_actual,gb_actual,rf_actual,xgb_list[-history:]]).T
        xgb_pred = clf.predict(stack_predict)
        xgb_list.append(xgb_pred[-1])
        date_ori.append(date_ori[-1]+timedelta(days=1))


predict(30, history = 5)


plt.figure(figsize = (15,6))
x_range = np.arange(df.Close.shape[0])
x_range_future = np.arange(len(xgb_list))
plt.plot(x_range, df.Close, label = 'Real Close')
plt.plot(x_range_future, reverse_close(np.array(ada_list)), label = 'ada Close')
plt.plot(x_range_future, reverse_close(np.array(bagging_list)), label = 'bagging Close')
plt.plot(x_range_future, reverse_close(np.array(et_list)), label = 'et Close')
plt.plot(x_range_future, reverse_close(np.array(gb_list)), label = 'gb Close')
plt.plot(x_range_future, reverse_close(np.array(rf_list)), label = 'rf Close')
plt.plot(x_range_future, reverse_close(np.array(xgb_list)), label = 'xgb stacked Close')
plt.legend()
plt.xticks(x_range_future[::50], pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()[::50])
plt.title('stacked')
plt.show()





