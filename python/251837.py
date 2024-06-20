# # How to Forecast a Time Series with Python
# 
# Wouldn't it be nice to know the future? This is the notebook that relates to the blog post on medium. Please check the blog for visualizations and explanations, this notebook is really just for the code :)
# 
# 
# ## Processing the Data
# 
# Let's explore the Industrial production of electric and gas utilities in the United States, from the years 1985-2018, with our frequency being Monthly production output.
# 
# You can access this data here: https://fred.stlouisfed.org/series/IPG2211A2N
# 
# This data measures the real output of all relevant establishments located in the United States, regardless of their ownership, but not those located in U.S. territories.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
data = pd.read_csv("Electric_Production.csv",index_col=0)
data.head()


# Right now our index is actually just a list of strings that look like a date, we'll want to adjust these to be timestamps, that way our forecasting analysis will be able to interpret these values:
# 

data.index


data.index = pd.to_datetime(data.index)


data.head()


data.index


# Let's first make sure that the data doesn't have any missing data points:
# 

data[pd.isnull(data['IPG2211A2N'])]


# Let's also rename this column since its hard to remember what "IPG2211A2N" code stands for:
# 

data.columns = ['Energy Production']


data.head()


import plotly
# plotly.tools.set_credentials_file()


from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
plot_mpl(fig)


import plotly.plotly as ply
import cufflinks as cf
# Check the docs on setting up offline plotting


data.iplot(title="Energy Production Jan 1985--Jan 2018", theme='pearl')





from pyramid.arima import auto_arima


# **he AIC measures how well a model fits the data while taking into account the overall complexity of the model. A model that fits the data very well while using lots of features will be assigned a larger AIC score than a model that uses fewer features to achieve the same goodness-of-fit. Therefore, we are interested in finding the model that yields the lowest AIC value.
# 

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True) 


stepwise_model.aic()


# ## Train Test Split
# 

data.head()


data.info()


# We'll train on 20 years of data, from the years 1985-2015 and test our forcast on the years after that and compare it to the real data.
# 

train = data.loc['1985-01-01':'2016-12-01']


train.tail()


test = data.loc['2015-01-01':]


test.head()


test.tail()


len(test)


stepwise_model.fit(train)


future_forecast = stepwise_model.predict(n_periods=37)


future_forecast


future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])


future_forecast.head()


test.head()


pd.concat([test,future_forecast],axis=1).iplot()


future_forecast2 = future_forcast


pd.concat([data,future_forecast2],axis=1).iplot()





