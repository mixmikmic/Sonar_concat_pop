# # Quantiacs Toolbox Sample: Support Vector Machine(Momentum) 
# This tutorial will show you how to use svm and momentum to predict the trend using the Quantiacs toolbox.  
# We use the 20-day closing price momentum for the last week (5 days) as features and trend of the next day as value.  
# For each prediction, we lookback one year (252 days).  
# 

import quantiacsToolbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
get_ipython().magic('matplotlib inline')


get_ipython().run_cell_magic('html', '', '<style>\ntable {float:left}\n</style>')


# For developing and testing a strategy, we will use the raw data in the tickerData folder that has been downloaded via the Toolbox's loadData() function.
#   
# This is just a simple sample to show how svm works.  
# Extract the closing price of the Australian Dollar future (F_AD) for the past year:
# 

F_AD = pd.read_csv('./tickerData/F_AD.txt')
CLOSE = np.array(F_AD.loc[:252-1, [' CLOSE']])
plt.plot(CLOSE)


# Momentum is generally defined as the return between two points in time separated by a fixed interval:  
# **(p2-p1)/p1**  
# Momentum is an indicator of the average speed of price on a time scale defined by the interval.  
# The most used intervals by investors are 1, 3, 6 and 12 months, or their equivalent in trading days.  
# 
# Calculate 20-day momentum:
# 

momentum = (CLOSE[20:] - CLOSE[:-20]) / CLOSE[:-20]
plt.plot(momentum)


# ## Now we can create samples.  
# Use the last 5 days' momentum as features.  
# We will use a binary trend: y = 1 if price goes up, y = -1 if price goes down  
#   
# For example, given close price, momentum at 19900114:  
# 

# | DATE | CLOSE | MOMENTUM |     
# | :--- | ----- | -------- |
# | 19900110 | 77580.0 | -0.01778759 |
# | 19900111 | 77980.0 | -0.00599427 |
# | 19900112 | 78050.0 | -0.01574397 |
# | 19900113 | 77920.0 | -0.00402702 |
# | 19900114 | 77770.0 | -0.01696891 |
# | 19900115 | 78060.0 | -0.01824298 |
# 

# Corresponding sample should be  
# x = (-0.01778759, -0.00599427, -0.01574397, -0.00402702, -0.01696891)  
# y = 1  
# 

X = np.concatenate([momentum[i:i+5] for i in range(252-20-5)], axis=1).T
y = np.sign((CLOSE[20+5:] - CLOSE[20+5-1: -1]).T[0])


# #### Use svm learn and predict:
# 

clf = svm.SVC()
clf.fit(X, y)
clf.predict(momentum[-5:].T)


# #### 1 shows that the close price will go up tomorrow.  
# #### What is the real result?
# 

F_AD.loc[251:252, ['DATE', ' CLOSE']]


# #### Hooray! Our strategy successfully predict the trend.  
# #### Now we can use Quantiac's Toolbox to run our strategy.
# 

class myStrategy(object):
    
    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):

        def predict(momentum, CLOSE, lookback, gap, dimension):
            X = np.concatenate([momentum[i:i + dimension] for i in range(lookback - gap - dimension)], axis=1).T
            y = np.sign((CLOSE[dimension+gap:] - CLOSE[dimension+gap-1:-1]).T[0])
            y[y==0] = 1

            clf = svm.SVC()
            clf.fit(X, y)

            return clf.predict(momentum[-dimension:].T)

        nMarkets = len(settings['markets'])
        lookback = settings['lookback']
        dimension = settings['dimension']
        gap = settings['gap']

        pos = np.zeros((1, nMarkets), dtype=np.float)

        momentum = (CLOSE[gap:, :] - CLOSE[:-gap, :]) / CLOSE[:-gap, :]

        for market in range(nMarkets):
            try:
                pos[0, market] = predict(momentum[:, market].reshape(-1, 1),
                                         CLOSE[:, market].reshape(-1, 1),
                                         lookback,
                                         gap,
                                         dimension)
            except ValueError:
                pos[0, market] = .0
        return pos, settings


    def mySettings(self):
        """ Define your trading system settings here """

        settings = {}

        # Futures Contracts
        settings['markets'] = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
                               'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC',
                               'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
                               'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
                               'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX',
                               'F_YM']

        settings['lookback'] = 252
        settings['budget'] = 10 ** 6
        settings['slippage'] = 0.05

        settings['gap'] = 20
        settings['dimension'] = 5

        return settings


result = quantiacsToolbox.runts(myStrategy)


# ** Congrats! You just finished your first svm(momentum) strategy.**  
# 
# Try to optimize it!  
# 
# **Quantiacs https://www.quantiacs.com/**
# 

# # Quantiacs Toolbox Sample: Linear regression
# This tutorial will show you how to use linear regression with the Quantiacs Toolbox to predict the return of a portfolio.
# A simple linear regression can be extended by constructing polynomial features from the coefficients.
# We will use a polynomial regression to fit the daily closing price of the past year to predict trend of next day.
# 

import quantiacsToolbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
get_ipython().magic('matplotlib inline')


# For developing and testing a strategy, we will use the raw data in the tickerData folder that has been downloaded with the Toolbox's loadData function.
# 
# This is just a simple sample to show how linear regression works.  
# Here we extract the closing price of the Australian Dollar futures (F_AD) for the last year:
# 

F_AD = pd.read_csv('./tickerData/F_AD.txt')
CLOSE = np.array(F_AD.loc[:252-1, [' CLOSE']])
plt.plot(CLOSE)


# Create samples:
# 

poly = PolynomialFeatures(degree=5)
X = poly.fit_transform(np.arange(252).reshape(-1, 1))
y = CLOSE[:]


# Use linear regression learn and predict:
# 

reg = linear_model.LinearRegression()
reg.fit(X, y)
plt.plot(y)
plt.plot(reg.predict(X))


# As we can see the model can roughly fit the price.  
# Now we can predict the next day's close price:
# 

reg.predict(poly.fit_transform(np.array([[252]])))


# Compare with real price:
# 

F_AD.loc[252, [' CLOSE']]


# #### Hooray! Our strategy successfully predicted the price.
# #### Now we use the Quantiacs Toolbox to run our strategy.
# 

class myStrategy(object):

    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
        """ This system uses linear regression to allocate capital into the desired equities"""

        # Get parameters from setting
        nMarkets = len(settings['markets'])
        lookback = settings['lookback']
        dimension = settings['dimension']
        threshold = settings['threshold']

        pos = np.zeros(nMarkets, dtype=np.float)

        poly = PolynomialFeatures(degree=dimension)
        for market in range(nMarkets):
            reg = linear_model.LinearRegression()
            try:
                reg.fit(poly.fit_transform(np.arange(lookback).reshape(-1, 1)), CLOSE[:, market])
                trend = (reg.predict(poly.fit_transform(np.array([[lookback]]))) - CLOSE[-1, market]) / CLOSE[-1, market]

                if abs(trend[0]) < threshold:
                    trend[0] = 0

                pos[market] = np.sign(trend)

            # for NaN data set position to 0
            except ValueError:
                pos[market] = .0

        return pos, settings


    def mySettings(self):
        """ Define your trading system settings here """

        settings = {}

        # Futures Contracts
        settings['markets'] = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
                               'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC',
                               'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
                               'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
                               'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX',
                               'F_YM']

        settings['lookback'] = 252
        settings['budget'] = 10 ** 6
        settings['slippage'] = 0.05

        settings['threshold'] = 0.2
        settings['dimension'] = 3

        return settings


result = quantiacsToolbox.runts(myStrategy)


# ** Congrats! You just finished your first linear regression strategy.**  
# 
# Try to optimize it!  
# 
# **Quantiacs https://www.quantiacs.com/**
# 

# # Quantiacs Toolbox Sample: Bollinger bands
# This tutorial will show you how to use Bollinger Bands with the Quantiacs Toolbox.  
# 

import quantiacsToolbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')


# This is just a simple sample to show how Bollinger Bands work.
# For developing and testing a strategy, we will use the raw data in the tickerData folder that has been downloaded with the Toolbox's loadData function.
# 
# Here we extract the closing price of the Australian Dollar futures (F_AD) for the last year:
# 

F_AD = pd.read_csv('./tickerData/F_AD.txt')
CLOSE = np.array(F_AD.loc[:252-1, [' CLOSE']])
plt.plot(CLOSE)


# ### What is a 'Bollinger Band®'
# A Bollinger Band®, developed by the famous technical trader John Bollinger, is plotted two standard deviations away from a simple moving average.  
# 
# You may read more here: Bollinger Band® http://www.investopedia.com/terms/b/bollingerbands.asp#ixzz4joGECFt7 
# 

# We can create a function calculating Bollinger bands based on their definition.
# 

def bollingerBands(a, n=20):
    sma = np.nansum(a[-n:]) / n
    std = np.std(a[-n:])
    return sma, sma + 2 * std, sma - 2 * std


upperBand, lowerBand = np.zeros(252 - 20 + 1), np.zeros(252 - 20 + 1)
for i in range(252 - 20 + 1):
    _, upperBand[i], lowerBand[i] = bollingerBands(CLOSE[i:i+20])
plt.plot(upperBand)
plt.plot(lowerBand)
plt.plot(CLOSE[20:])


# In this example of Bollinger Bands®, the price of the stock is bracketed by an upper and lower band along with a 20-day simple moving average. Because standard deviation is a measure of volatility, when the markets become more volatile, the bands widen; during less volatile periods, the bands contract.  
#   
# Bollinger Bands® are a highly popular technical analysis technique. Many traders believe the closer the prices move to the upper band, the more overbought the market, and the closer the prices move to the lower band, the more oversold the market. John Bollinger has a set of 22 rules to follow when using the bands as a trading system.
# 
# Now lets take a look to see if Bollinger Bands could work in futures markets by using the Quantiacs Toolbox!
# 

class myStrategy(object):
    
    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):

        def bollingerBands(a, n=20):
            sma = np.nansum(a[-n:]) / n
            std = np.std(a[-n:], ddof=1)
            return sma, sma + 2 * std, sma - 2 * std

        nMarkets = len(settings['markets'])
        threshold = settings['threshold']
        pos = np.zeros((1, nMarkets), dtype=np.float)

        for market in range(nMarkets):
            sma, upperBand, lowerBand = bollingerBands(CLOSE[:, market])
            currentPrice = CLOSE[-1, market]

            if currentPrice >= upperBand + (upperBand - lowerBand) * threshold:
                pos[0, market] = -1
            elif currentPrice <= lowerBand - (upperBand - lowerBand) * threshold:
                pos[0, market] = 1

        return pos, settings


    def mySettings(self):
        """ Define your trading system settings here """

        settings = {}

        # Futures Contracts
        settings['markets'] = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
                               'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC',
                               'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
                               'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
                               'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX',
                               'F_YM']

        settings['beginInSample'] = '19900101'
        settings['endInSample'] = '20170522'
        settings['lookback'] = 20
        settings['budget'] = 10 ** 6
        settings['slippage'] = 0.05

        settings['threshold'] = 0.4

        return settings


result = quantiacsToolbox.runts(myStrategy)


# ** Congrats! You just finished your first Bollinger Bands strategy.**  
# 
# John Bollinger suggests using them with two or three other non-correlated indicators that provide more direct market signals. He believes it is crucial to use indicators based on different types of data. Some of his favored technical techniques are moving average divergence/convergence (MACD), on-balance volume (OBV), and relative strength index (RSI).  
# 
# Try to optimize it!  
# 
# **Quantiacs https://www.quantiacs.com/**
# 

