from __future__ import absolute_import, division, print_function


# # Interactive Notebook Possibilities
# 
# http://mpld3.github.io/examples/linked_brush.html
# 




# uncomment the bottom line in this cell, change the final line of 
# the loaded script to `mpld3.display()` (instead of show).


# %load http://mpld3.github.io/_downloads/linked_brush.py


# ## Linked Brushing Example
# 
# This example uses the standard Iris dataset and plots it with a linked brushing
# tool for dynamically exploring the data. The paintbrush button at the bottom
# left can be used to enable and disable the behavior.
# 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import mpld3
from mpld3 import plugins, utils


data = load_iris()
X = data.data
y = data.target


# dither the data for clearer plotting
X += 0.1 * np.random.random(X.shape)


fig, ax = plt.subplots(4, 4, sharex="col", sharey="row", figsize=(8, 8))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                    hspace=0.1, wspace=0.1)

for i in range(4):
    for j in range(4):
        points = ax[3 - i, j].scatter(X[:, j], X[:, i],
                                      c=y, s=40, alpha=0.6)

# remove tick labels
for axi in ax.flat:
    for axis in [axi.xaxis, axi.yaxis]:
        axis.set_major_formatter(plt.NullFormatter())

# Here we connect the linked brush plugin
plugins.connect(fig, plugins.LinkedBrush(points))

# mpld3.show()
mpld3.display()


























get_ipython().magic('matplotlib inline')

import mpld3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context('poster')
# sns.set_style('whitegrid') 
sns.set_style('darkgrid') 
plt.rcParams['figure.figsize'] = 12, 8  # plotsize 


def sinplot(flip=1, ax=None):
    """Demo plot from seaborn."""
    x = np.linspace(0, 14, 500)
    for i in range(1, 7):
        ax.plot(x, np.sin(-1.60 + x + i * .5) * (7 - i) * flip, label=str(i))


mpld3.enable_notebook()


fig, ax = plt.subplots(figsize=(12, 8))
sinplot(ax=ax)
ax.set_ylabel("y-label")
ax.set_xlabel("x-label")
fig.tight_layout()


mpld3.disable_notebook()





# # Clean data
# 
# Coal mining data from [eia.gov](http://www.eia.gov/coal/data.cfm#prices)
# 
# Combining and cleaning the raw csv files into a cleaned data set and coherent database. 
# 
# Generally a good idea to have a separate data folder with the raw data.
# 
# When you clean the raw data, leave the raw in place, and create cleaned version with the steps included (ideal situation for Notebook).
# 

import numpy as np
import pandas as pd


get_ipython().system('pwd')


# The cleaned data file is saved here:
output_file = "../data/coal_prod_cleaned.csv"


df7 = pd.read_csv("../data/coal_prod_2008.csv", index_col="MSHA_ID")
df8 = pd.read_csv("../data/coal_prod_2009.csv", index_col="MSHA_ID")
df9 = pd.read_csv("../data/coal_prod_2010.csv", index_col="MSHA_ID")
df10 = pd.read_csv("../data/coal_prod_2011.csv", index_col="MSHA_ID")
df11 = pd.read_csv("../data/coal_prod_2012.csv", index_col="MSHA_ID")


dframe = pd.concat((df7, df8, df9, df10, df11))


# Noticed a probable typo in the data set: 
dframe['Company_Type'].unique()


# Correcting the Company_Type
dframe.loc[dframe['Company_Type'] == 'Indepedent Producer Operator', 'Company_Type'] = 'Independent Producer Operator'
dframe.head()


dframe[dframe.Year == 2008].head()


# # Final Cleaned Data Product
# 

dframe.to_csv(output_file, )





