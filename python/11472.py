# Initial exploration of 311 data
# 

from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# import matplotlib as mpl
import seaborn as sns


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")


# # Download data
# 
# https://data.sfgov.org/City-Infrastructure/Case-Data-from-San-Francisco-311-SF311-/vw6y-z8j6 (export csv)
# 
# ## And rename with timestamp
# 
# ```bash
# $ mv Case_Data_from_San_Francisco_311__SF311.csv Case_Data_from_San_Francisco_311__SF311_2016-02-10.csv
# ```
# 

df = pd.read_csv('Case_Data_from_San_Francisco_311__SF311_2016-02-10.csv', header=0, index_col=0, parse_dates=[1, 2, 3])


df.shape


df.head()


df['Category'].value_counts()


df['Opened_ymd'] = df['Opened'].map(lambda x: x.strftime('%Y-%m-%d'))


fix, ax = plt.subplots(figsize=(16, 12))
colormap='Set1'
df.groupby(by=['Opened_ymd', 'Category'])['Opened'].count().unstack().plot(ax=ax, colormap=colormap, alpha=.75)


fix, ax = plt.subplots(figsize=(16, 12))
colormap='Set1'
df[(df['Opened'] > '2015-08-08') & (df['Opened'] < '2015-08-13')].groupby(by=['Opened_ymd', 'Category'])['Opened'].count().unstack().plot(ax=ax, colormap=colormap)


df[(df['Category'] == 'Street and Sidewalk Cleaning')]['Neighborhood'].value_counts().head(10)


df[(df['Category'] == 'Noise Report')]['Neighborhood'].value_counts().head(10)


df[(df['Opened_ymd'] == '2015-08-11')]['Category'].value_counts().head(10)


fig, ax = plt.subplots(figsize=(12, 8))
df.groupby(by=['Category', 'Neighborhood'])['Opened'].count().unstack()['Mission'].T.plot(ax=ax, kind='bar')


