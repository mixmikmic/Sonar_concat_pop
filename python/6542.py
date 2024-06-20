# <h1 style="color:blue">Reshape dataframe using stack/unstack</h1>
# 

import pandas as pd
df = pd.read_excel("stocks.xlsx",header=[0,1])
df


df.stack()


df.stack(level=0)


df_stacked=df.stack()
df_stacked


df_stacked.unstack()


# <h1 style="color:blue">3 levels of column headers</h1>
# 

df2 = pd.read_excel("stocks_3_levels.xlsx",header=[0,1,2])
df2


df2.stack()


df2.stack(level=0)


df2.stack(level=1)


# <img src="http://pandas.pydata.org/_static/pandas_logo.png" style="width:400px;height:80px;">
# # <font color='red'>Pandas Tutorial On Stock Price Analysis</font> 
# 

# This tutorial will cover how to retrieve stock price from google finance using pandas data reader. The analysis of stock is done by plotting its high, low, close, volumne values in table and a chart. Charts are of two types,
# 
# 1. Line Chart
# 2. Bar Chart
# 
# If you don't know what is stock then first **watch this video to gain understanding on fundamentals of stocks and investing**,
# 

get_ipython().run_cell_magic('HTML', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/XRO6lEu9-5w" frameborder="0" allowfullscreen></iframe>')


import pandas.io.data as web


df = web.DataReader('AAPL', 'google', '2016/1/1', '2017/1/1')


df.head()


get_ipython().magic('matplotlib inline')
df.plot(y='Close', color="Green")


df.plot.bar(y='Volume')





get_ipython().magic('pinfo df.plot.bar')


get_ipython().magic('lsmagic')


get_ipython().magic('time for i in range(100000): i*i')


get_ipython().magic('system pwd')


get_ipython().system('ls')


# <img src="http://pandas.pydata.org/_static/pandas_logo.png" style="width:400px;height:80px;">
# # <font color='red'>Pandas Tutorial On Stock Price Analysis</font> 
# 

# *This tutorial will cover how to retrieve stock price from google finance using pandas data reader. The analysis of stock is done by plotting its high, low, close, volumne values in table and a chart. Charts are of two types,*
# 
# 1. Line Chart
# 2. Bar Chart
# 
# If you don't know what is stock then first **watch this video to gain understanding on fundamentals of stocks and investing**,
# 

get_ipython().run_cell_magic('HTML', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/XRO6lEu9-5w" frameborder="0" allowfullscreen></iframe>')


import pandas.io.data as web
df = web.DataReader('AAPL', 'google', '2016/1/1', '2017/1/1')
df.head()


get_ipython().magic('matplotlib inline')
df.plot(y="Close", color="Green")


df.plot.bar(y="Volume")





get_ipython().magic('system ls')


get_ipython().system('pwd')

















# # <font color="purple"><h3 align="center">Different Ways Of Creating Dataframe</h3></font>
# 

# ## <font color="green">Using csv</h3></font>
# 

df = pd.read_csv("weather_data.csv")
df


# ## <font color="green">Using excel</h3></font>
# 

df=pd.read_excel("weather_data.xlsx","Sheet1")
df


# ## <font color="green">Using dictionary</h3></font>
# 

import pandas as pd
weather_data = {
    'day': ['1/1/2017','1/2/2017','1/3/2017'],
    'temperature': [32,35,28],
    'windspeed': [6,7,2],
    'event': ['Rain', 'Sunny', 'Snow']
}
df = pd.DataFrame(weather_data)
df


# ## <font color="green">Using tuples list</h3></font>
# 

weather_data = [
    ('1/1/2017',32,6,'Rain'),
    ('1/2/2017',35,7,'Sunny'),
    ('1/3/2017',28,2,'Snow')
]
df = pd.DataFrame(data=weather_data, columns=['day','temperature','windspeed','event'])
df


# ## <font color="green">Using list of dictionaries</h3></font>
# 

weather_data = [
    {'day': '1/1/2017', 'temperature': 32, 'windspeed': 6, 'event': 'Rain'},
    {'day': '1/2/2017', 'temperature': 35, 'windspeed': 7, 'event': 'Sunny'},
    {'day': '1/3/2017', 'temperature': 28, 'windspeed': 2, 'event': 'Snow'},
    
]
df = pd.DataFrame(data=weather_data, columns=['day','temperature','windspeed','event'])
df


# ## <font color="maroon"><h4 align="center">Pandas Group By</font>
# 

# **In this tutorial we are going to look at weather data from various cities and see how group by can be used to run some analytics.** 
# 

import pandas as pd
df = pd.read_csv("weather_by_cities.csv")
df


# ### For this dataset, get following answers,
# #### 1. What was the maximum temperature in each of these 3 cities?
# #### 2. What was the average windspeed in each of these 3 cities?
# 

g = df.groupby("city")
g


# **DataFrameGroupBy object looks something like below,**
# 

# <img src="group_by_cities.png">
# 

for city, data in g:
    print("city:",city)
    print("\n")
    print("data:",data)    


# **This is similar to SQL,**
# 
# **SELECT * from weather_data GROUP BY city**
# 

g.get_group('mumbai')


g.max()


g.mean()


# **This method of splitting your dataset in smaller groups and then applying an operation 
# (such as min or max) to get aggregate result is called Split-Apply-Combine. It is illustrated in a diagram below**
# 

# <img src="split_apply_combine.png">
# 

g.min()


g.describe()


g.size()


g.count()


get_ipython().magic('matplotlib inline')
g.plot()


# <h1 style="color:blue">Pivot basics</h1>
# 

import pandas as pd
df = pd.read_csv("weather.csv")
df


df.pivot(index='city',columns='date')


df.pivot(index='city',columns='date',values="humidity")


df.pivot(index='date',columns='city')


df.pivot(index='humidity',columns='city')


# <h1 style="color:blue">Pivot Table</h1>
# 

df = pd.read_csv("weather2.csv")
df


df.pivot_table(index="city",columns="date")


# <h2 style="color:brown">Margins</h2>
# 

df.pivot_table(index="city",columns="date", margins=True,aggfunc=np.sum)


# <h2 style="color:brown">Grouper</h2>
# 

df = pd.read_csv("weather3.csv")
df


df['date'] = pd.to_datetime(df['date'])


df.pivot_table(index=pd.Grouper(freq='M',key='date'),columns='city')


# # <font color="purple"><h3 align="center">Reshape pandas dataframe using melt</h3></font>
# 

import pandas as pd
df = pd.read_csv("weather.csv")
df


melted = pd.melt(df, id_vars=["day"], var_name='city', value_name='temperature')
melted


