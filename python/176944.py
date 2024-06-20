


# ## Working with text plotting
# Introduction:
# 
# Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shell, the jupyter notebook, web application servers, and four graphical user interface toolkits.
# 
# ### Here are the main steps we will go through
# * How to add text to graph?
# 
# This is Just a little illustration.
# 
# <img style="float:left;" src="https://matplotlib.org/1.3.0/_images/annotate_text_arrow.png"></img>

# import matplotlib, numpy
import matplotlib.pyplot as plt
import numpy as np


# set plt.figure()
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_title('axes title')

ax.set_ylabel('Frequency')
ax.set_xlabel('Data')

ax.text(4, 7, 'Some text', style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

ax.plot(np.linspace(0, 1, 5), np.linspace(0, 5, 5))
ax.plot([4], [1], 's')
ax.annotate('Here is the point', xy=(4, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='blue', shrink=0.04))

ax.axis([0, 10, 0, 10])

plt.show()


# more advance example
fig = plt.figure()
ax = fig.add_subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax.plot(t, s, lw=2)

ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='green', shrink=0.05),
            )

ax.set_ylim(-2,2)
plt.show()





# ## Creating and Combining DataFrame
# <b>class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)</b>
# 
# Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. 
# 
# <b>class pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)</b>
# One-dimensional ndarray with axis labels (including time series).
# 
# Labels need not be unique but must be a hashable type. The object supports both integer- and label-based indexing and provides a host of methods for performing operations involving the index. Statistical methods from ndarray have been overridden to automatically exclude missing data (currently represented as NaN).
# 
# ### Here are the main steps we will go through
# * How to create dataframe using pandas?
# * How to combine two data set using pandas?
# 
# This is Just a little illustration.
# 
# <img style="float: left;" src="https://www.tutorialspoint.com/python_pandas/images/structure_table.jpg"></img>

import pandas as pd
import numpy as np


# #### How to create dataframe using pandas?
# 

# working with series
#create a series
s = pd.Series(np.random.randn(5))
#create a dataframe column
df = pd.DataFrame(s, columns=['Column_1'])
df 


#sorting 
df.sort_values(by='Column_1')


#boolean indexing
#It returns all rows in column_name,
#that are less than 10
df[df['Column_1'] <= 1]


# creating simple series
obj2 = pd.Series(np.random.randn(5), index=['d', 'b', 'a', 'c', 'e'])
obj2


obj2.index


# returns the value in e
obj2['e']


# returns all values that are greater than -2
obj2[obj2 > -2]


# we can do multiplication on dataframe
obj2 * 2


# we can do boolean expression
'b' in obj2


# returns false, because 'g' is not defined in our data
'g' in obj2


#Let's see we have this data
sdata = {'Cat': 24, 'Dog': 11, 'Fox': 18, 'Horse': 1000}
obj3 = pd.Series(sdata)
obj3


# defined list, and assign series to it
sindex = ['Lion', 'Dog', 'Cat', 'Horse']
obj4 = pd.Series(sdata, index=sindex)
obj4


# checking if our data contains null
obj4.isnull()


#we can add two dataframe together
obj3 + obj4


# we can create series calling Series function on pandas
programming = pd.Series([89,78,90,100,98])
programming


# assign index to names
programming.index = ['C++', 'C', 'R', 'Python', 'Java']
programming


# let's create simple data
data = {'Programming': ['C++', 'C', 'R', 'Python', 'Java'],
        'Year': [1998, 1972, 1993, 1980, 1991],
        'Popular': [90, 79, 75, 99, 97]}
frame = pd.DataFrame(data)
frame


# set our index 
pd.DataFrame(data, columns=['Popular', 'Programming', 'Year'])


data2 = pd.DataFrame(data, columns=['Year', 'Programming', 'Popular', 'Users'],
                    index=[1,2,3,4,5])
data2


data2['Programming']


data2.Popular


data2.Users = np.random.random(5)*104
data2 = np.round(data2)
data2


# #### How to combine two data set using pandas?
# 

# we will do merging two dataset together 
merg1 = {'Edit': 24, 'View': 11, 'Inser': 18, 'Cell': 40}
merg1 = pd.Series(merg1)
merg1 = pd.DataFrame(merg1, columns=['Merge1'])

merg2 = {'Kernel':50, 'Navigate':27, 'Widgets':29,'Help':43}
merg2 = pd.Series(merg2)
merg2 = pd.DataFrame(merg2, columns=['Merge2'])


merg1


merg2


#join matching rows from bdf to adf
#pd.merge(merg1, merg2, left_index=True, right_index=True)
join = merg1.join(merg2)
join


#replace all NA/null data with value
join = join.fillna(12)
join


#compute and append one or more new columns
join = join.assign(Area=lambda df: join.Merge1*join.Merge2)
join


#add single column
join['Volume'] = join.Merge1*join.Merge2*join.Area
join


join.head(2)


join.tail(2)


#randomly select fraction of rows
join.sample(frac=0.5)


#order rows by values of a column (low to high)
join.sort_values('Volume')


#order row by values of a column (high to low)
join.sort_values('Volume', ascending=False)


#return the columns of a dataframe - by renaming
join = join.rename(columns={'Merge1':'X','Merge2':'Y'})


join


#count number of rows with each unique value of variable
join['Y'].value_counts()


#number of rows in dataframe
len(join)


#descriptive statistics
join.describe()


# ### Thank you, more to come soon!
# 




# ## Model fitting
# * Supervised Learning
# * Unsupervised Learning
# 

# ### supervised learning model fitting
# 

from sklearn.datasets import load_iris
holder = load_iris()
X, y = holder.data, holder.target


#supervised learning model fitting with linearRegression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state =0)


#supervised learning model fitting with Kneighbor classifier
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


#supervised learning model fitting with support vector classifier
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)


# ### unsupervised learning model fitting
# 

#unsupervised learning model fitting with kmeans
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X_train)


#unsupervised learning model fitting with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
pca_model = pca.fit_transform(X_train)





# ## Model Prediction
# * Supervised Learning
# * Unsupervised Learning
# 

from sklearn.datasets import load_iris
holder = load_iris()
X, y = holder.data, holder.target


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state =0)


# ### supervised learning model prediction
# 

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train,y_train)
# here we call the predict methond
y_pred = lr.predict(X_test)


from sklearn.svm import SVC
svc = SVC(kernel='linear').fit(X, y)
# here we call the predict methond
y_pred = svc.predict(X_test)


from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# here we call the predict methond
y_pred = knn.predict_proba(X_test)


# ### unsupervised learning model prediction
# 

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0).fit(X_train, y_train)
# here we call the predict methond
y_pred = k_means.predict(X_test)





# ## Imputing missing values in sklearn
# 
# Mean imputation replaces missing values with the mean value of that feature/variable. Mean imputation is one of the most 'naive' imputation methods because unlike more complex methods like k-nearest neighbors imputation, it does not use the information we have about an observation to estimate a value for it.
# 

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


# Create an empty dataset
df = pd.DataFrame()

# Create two variables called x0 and x1. Make the first value of x1 a missing value
df['x0'] = [0.3051,0.4949,0.6974,np.nan,0.2231,np.nan,0.4436,0.5897,0.6308,0.5]
df['x1'] = [np.nan,0.2654,0.2615,0.5846,0.4615,np.nan,0.4962,0.3269,np.nan,0.6731]

# View the dataset
df


# Create an imputer object that looks for 'Nan' values
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# Train the imputor on the df dataset
mean_imputer = mean_imputer.fit(df)


# Apply the imputer to the df dataset
imputed_df = mean_imputer.transform(df.values)








# # Bar plots in Matplotlib
# 

get_ipython().run_line_magic('pylab', '')
get_ipython().run_line_magic('matplotlib', 'inline')


# input data
mean_values = [3.3, 1, 2, 3]
variance = [0.3,0.2, 0.4, 0.5]
bar_labels = ['bar 1', 'bar 2', 'bar 3','bar 4']

# plot bars
x_pos = list(range(len(bar_labels)))
bar(x_pos, mean_values)

# set height of the y-axis
max_y = max(zip(mean_values, variance)) # returns a tuple, here: (3, 5)
ylim([0, (max_y[0] + max_y[1]) * 1.1])

# set axes labels and title
ylabel('Y lable')
xticks(x_pos, bar_labels)
title('Bar plot')


# input data
figure(figsize=(10,5))
mean_values = [3.3, 1, 2, 3]
variance = [0.3,0.2, 0.4, 0.5]
bar_labels = ['bar 1', 'bar 2', 'bar 3','bar 4']

# plot bars
x_pos = list(range(len(bar_labels)))
barh(x_pos, mean_values,align='center', alpha=0.4, color='g')

# set height of the y-axis
max_y = max(zip(mean_values, variance)) 
ylim([len(max_y)+0.5])
# set axes labels and title
ylabel('Y lable')
xticks(x_pos, bar_labels)
title('Bar plot')


# Input data
green_data = [1, 3, 5]
blue_data = [4, 2, 3.4]
red_data = [3, 4, 6]
labels = ['Green', 'Blue', 'Red']

# Setting the positions and width for the bars
pos = list(range(len(green_data))) 
width = 0.2 
    
# Plotting the bars
fig, ax = subplots(figsize=(8,6))

bar(pos, green_data, width, alpha=0.5, color='g', label=labels[0])
bar([p + width for p in pos], blue_data, width, alpha=0.5, color='b', label=labels[1])
bar([p + width*2 for p in pos], red_data, width, alpha=0.5, color='r', label=labels[2])

# Setting axis labels and ticks
ax.set_ylabel('y-value')
ax.set_title('Grouped bar plot')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(labels)

# Setting the x-axis and y-axis limits
xlim(min(pos)-width, max(pos)+width*4)
ylim([0, max(green_data + blue_data + red_data) * 1.5])

# Adding the legend and showing the plot
legend(['Green', 'Blue', 'Red'], loc='upper left')


data = range(200, 225, 5)

bar_labels = ['a', 'b', 'c', 'd', 'e']

fig = figure(figsize=(10,8))

# plot bars
y_pos = np.arange(len(data))
yticks(y_pos, bar_labels, fontsize=12)
bars = barh(y_pos, data, align='center', alpha=0.5, color='g')

# annotation and labels
for b,d in zip(bars, data):
    text(b.get_width() + b.get_width()*0.05, b.get_y() + b.get_height()/2,
        '{0:.0%}'.format(d/min(data)), 
        ha='center', va='bottom', fontsize=12, color='k')

xlabel('X axis label', fontsize=12)
ylabel('Y axis label', fontsize=12)
t = title('Bar plot with plot labels/text', fontsize=14)
ylim([-1,len(data)+0.5])
vlines(min(data), -1, len(data)+0.5, linestyles='--', color='r')





# ## Creating, Adding, Split and Removing arrays
# Numpy, short for Numerical Python, is the fundamental package required for hight performance scientific computing and its best library to learn and apply on data science career.
# ### Here are the main steps we will go through
# * How to create array in numpy?
# * How to add two arrays together?
# * How to delete array in numpy?
# * Inserts values into arr given index
# * Split array in given index
# 
# This is just little illustration.
# <img src="http://community.datacamp.com.s3.amazonaws.com/community/production/ckeditor_assets/pictures/332/content_arrays-axes.png">

# #### How to create array in numpy?
# <b>arange([start,] stop[, step,][, dtype])</b> : 
# 
# arange() will create arrays with regularly incrementing values. Check the docstring for complete information on the various ways it can be used. A few examples will be given here:
# 

import numpy as np
# we can use the first argument [start]
arr = np.arange(5)
arr


# we can pass [start] stop[step]
arr2 = np.arange(1, 10)
print("array containing [start(1) - end(10)]: ",arr2)

#apply step
arr3 = np.arange(1, 10, 2)
print("array containing [start(1) - end(10) - step(2)]: ", arr3)


# we can print shape of the array and as well as dtype
shp = np.arange(1,10)
print("Shape of array: ",shp.shape )
# dtype
dty = np.arange(1,20)
print("Dtype: ", dty.shape)


# ##### creating 2D array
# <b>np.array</b>
# 
# NumPy’s array class is called ndarray. It is also known by the alias array. Note that numpy.array is not the same as the Standard Python Library class array.array, which only handles one-dimensional arrays and offers less functionality. Check the docstring for complete information on the various ways it can be used.
# 

# we can create 2-dimention array
d_2 = np.array([[1,2,3],[4,5,6]])
d_2
print("2D shape: ", d_2.shape)
# we can use random function
rnd = np.random.random(9).reshape(3,3)
rnd
print("random array: ", rnd.shape)


# #### How to add two arrays together?
# <b>numpy.concatenate((a1, a2, ...), axis=0)</b>
# 
# Join a sequence of arrays along an existing axis. Check the docstring for complete information on the various ways it can be used.
# 

array_1 = np.array([[1, 2], [3, 4]])
array_2 = np.array([[5, 6], [7, 8]])
array_1


array_2 


# we can add array_2 as rows to the end of array_1
# axis 0 = rows
np.concatenate((array_1, array_2), axis=0)


# we can add array_2 as columns to end of array_1
# axis 1 = columns
np.concatenate((array_1, array_2), axis=1)


# #### How to delete array in numpy?
# <b>numpy.delete(arr, obj, axis=None)</b>
# 
# Return a new array with sub-arrays along an axis deleted. Check the docstring for complete information on the various ways it can be used.
# 

del_arry = np.array([[1,2,3],[4,5,6]])
del_arry


# column 2: [3 and 6]
# we can delete columm on index 2 of array
del_arry = np.delete(del_arry, 2, axis=1)
del_arry


# row 1: [4, 5, 6]
# we can delete row on index 1 of the array
del_arry = np.delete(del_arry, 1, axis=0)
del_arry


# #### Inserts values into arr given index?
# <b>numpy.insert(arr, obj, values, axis=None)</b>
# 
# Insert values along the given axis before the given indices. Check the docstring for complete information on the various ways it can be used.
# 

insert_array = np.array([[1,2,3],[4,5,6]])
# we can insert values into array index 6 - at the end
insert_array = np.insert(insert_array, 6, 10)
# we can also insert at the begining 
insert_array = np.insert(insert_array, 0, 100)
insert_array


# we can fill up the whole column given value
insert_2 = np.arange(0,9).reshape(3,3)
print("original array:")
print(insert_2)

# we can insert 0s in second column
insert_2 = np.insert(insert_2, 1, 0, axis=1)
print("\nafter inserting 0's on the first column:")
print(insert_2)


# we can also insert list as well
list_array = np.arange(0,9).reshape(3,3)
list_array = np.insert(list_array, [1], [[10],[10],[10]], axis=1)
list_array = np.insert(list_array, [1], 10, axis=0)
list_array


# #### Split array in given index?
# <b>numpy.split(ary, indices_or_sections, axis=0)</b>
# 
# Split an array into multiple sub-arrays. Check the docstring for complete information on the various ways it can be used.
# 

or_array = np.array([[1,2,3],[4,5,6]])
print("Orignal array:\n ",or_array)
#splits arr into 3 sub-arrays 
split_array = np.split(or_array, 2)
print("\nwe have our array splitted into two arrays")
split_array


copy_array = np.arange(16.0).reshape(4, 4)
#splits arr horizontally on the 5th index
print("copy array:\n",copy_array)

# we splits our array into horizontal on the given index
h_split = np.hsplit(copy_array, 2)
h_split


# we can also split array into vertical on the given index
h_split = np.vsplit(copy_array, 2)
h_split











# ## Importing, Exporting, Basic Slicing and Indexing.
# In terms of the importing and exporting files I would not go depth on it. You can refer the docstring for complete information on the various ways it can be used. A few examples will be given here in regard to this. I would spent sometime on the slicing and indexing arrays.
# ### Here are the main steps we will go through
# * How to use loadtxt, genfromtxt, and savetxt?
# * How to slice and index array?
# 
# This is just little illustration.
# <img src="http://www.bogotobogo.com/python/images/python_strings/string_diagram.png">

# #### How to use loadtxt, genfromtxt, and savetxt??
# * <b>numpy.loadtxt()</b> : Load data from a text file. This function aims to be a fast reader for simply formatted files. 
# * <b>numpy.genfromtxt()</b>: Load data from a text file, with missing values handled as specified. When spaces are used as delimiters, or when no delimiter has been given as input, there should not be any missing data between two fields. When the variables are named (either by a flexible dtype or with names, there must not be any header in the file (else a ValueError exception is raised). Individual values are not stripped of spaces by default. When using a custom converter, make sure the function does remove spaces.
# * <b>numpy.savetxt()</b>: Save an array to a text file. Further explanation of the fmt parameter (%[flag]width[.precision]specifier):
# 
# ##### Example 
# Here is the general idea, I'll come back to it.
# 

import numpy as np 
# using numpy you can load text file
np.loadtxt('file_name.txt')
# load csv file
np.genfromtxt('file_name.csv', delimiter=',')
# you can write to a text file and save it
np.savetxt('file_name.txt', arr, delimiter=' ')
# you can write to a csv file and save it
np.savetxt('file_name.csv', arr, delimiter=',')


# #### How to slice and index array?
# <b>ndarrays</b> can be indexed using the standard Python x[obj] syntax, where x is the array and obj the selection. There are three kinds of indexing available: field access, basic slicing, advanced indexing. Which one occurs depends on obj.
# 
# The basic slice syntax is i:j:k where i is the starting index, j is the stopping index, and k is the step (k\neq0). This selects the m elements (in the corresponding dimension) with index values i, i + k, ..., i + (m - 1) k where m = q + (r\neq0) and q and r are the quotient and remainder obtained by dividing j - i by k: j - i = q k + r, so that i + (m - 1) k < j.
# 
# Check the docstring for complete information on the various ways it can be used. A few examples will be given here:
# 

# slicing 1 to 7 gives us: [1 through 6]
slice_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
slice_array[1:7]


# if we do this, we are giving k, which is the step function. in this case step by 2
slice_array[1:7:2]


slice_arrays = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])+1
#return the element at index 5
slice_arrays[5]


#returns the 2D array element on index value of 2 and 5
slice_arrays[[2,5]]


#assign array element on index 1 the value 4
slice_arrays[1] = 100
#assign array element on index [1][3] the value 10
slice_arrays[[1,3]] = 100
slice_arrays


#return the elements at indices 0,1,2 on a 2D array:
slice_arrays[0:3]


#returns the elements at indices 1,100
slice_arrays[:2]


slice_2d = np.arange(16).reshape(4,4)
slice_2d


#returns the elements on rows 0,1,2, at column 4
slice_2d[0:3, :4]


#returns the elements at index 1 on all columns
slice_2d[:, 1]


# return the last two rows
slice_2d[-2:10]
# returns the last three rows
slice_2d[1:]


# reverse all the array backword
slice_2d[::-1]


#returns an array with boolean values
slice_2d < 5


#inverts a boolearn array, if its positive arr - convert to negative, vice versa
~slice_2d


#returns array elements smaller than 5
slice_2d[slice_2d < 5]





# ## Evaluating Classification Metrics
# First, we'll try to build a model without cross-validation apply to it. Second, we'll use the same model with applying cross-validation and see if the accuracy score changed or not.
# 
# ### Main contents:
# ### Training and testing on the same data
#   * Rewards overly complex models that "overfit" the training data and won't necessarily generalize
# #### Train/test split
#   * Split the dataset into two pieces, so that the model can be trained and tested on different data
#   * Better estimate of out-of-sample performance, but still a "high variance" estimate
# #### K-fold cross-validation
#   * Systematically create "K" train/test splits and average the results together
#   * Even better estimate of out-of-sample performance
#   * Runs "K" times slower than train/test split
#   
# ### Model evaluation metrics
# ##### Classification problems: Classification accuracy
#   * There are many more metrics, here we'll cover on classification metrics
# 
# <img style="float:left;" src="https://image.slidesharecdn.com/finalcustlingprofiling-160226163538/95/customer-linguistic-profiling-10-638.jpg?cb=1456504658" width=600 height=300>
# 

import warnings
warnings.filterwarnings('ignore')


# libraries we need
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC


# load iris dataset
data_holder = load_iris()
print(data_holder.data.shape)
print(data_holder.target.shape)


# set our X and y to data and target values
X , y = data_holder.data, data_holder.target


# let's split into 70/30: train=70% and test=30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .4, random_state = 0)


print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print()
print("y train shape: ", y_train.shape)
print("y test shape: ", y_test.shape)


# we'll set it to some parameters, but we'll go through depth on parameter tuning later
model = SVC(kernel='linear', C=1)
# fit our training data
model.fit(X_train, y_train)
#let's predict 
pred = model.predict(X_test)


# ### Accuracy Score
# Calling the accuracy_score class we can get the score of our model
# 

#accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# let's get our classification report


# ### Classification Metrics
# Build a text report showing the main classification metrics
# 

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))


# ### Confusion Matrix
# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.
# 

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))








# ## Data to from String in Numpy
# Numpy, short for Numerical Python, is the fundamental package required for hight performance scientific computing and its best library to learn and apply on data science career.
# 
# This is just little illustration.
# 
# <img style="float: left;" src="http://community.datacamp.com.s3.amazonaws.com/community/production/ckeditor_assets/pictures/332/content_arrays-axes.png" width=600 height=400>
# 

import numpy as np


a = np.array([[1,2],
           [3,4]], 
          dtype = np.uint8)


a.tostring()


a.tostring(order='F')


s = a.tostring()
a = np.fromstring(s, dtype=np.uint8)
a


a.shape = 2,2
a





# ## Groupby
# <b>DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, **kwargs)</b>
# 
# Group series using mapper (dict or key function, apply given function to group, return result as series) or by a series of columns.
# 
# ### Any groupby operation involves one of the following operations on the original object. They are −
# 
# * Splitting the Object
# 
# * Applying a function
# 
# * Combining the results
# 
# <img style="float: left;" src="https://i.stack.imgur.com/sgCn1.jpg"></img>
# 

# import library
import pandas as pd


data = {'Students': ['S1', 'S2', 'S3', 'S3', 'S1',
         'S4', 'S4', 'S3', 'S2', 'S2', 'S4', 'S3'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Grade':[87,79,83,73,74,81,56,78,94,70,80,69]}
df = pd.DataFrame(data)
df


# ### Split Data into Groups
# Pandas object can be split into any of their objects. There are multiple ways to split an object like −
# 
# * obj.groupby('key')
# * obj.groupby(['key1','key2'])
# * obj.groupby(key,axis=1)
# 
# Let us now see how the grouping objects can be applied to the DataFrame object
# 

# let's groupby students
df.groupby('Students')


# to view groups 
df.groupby('Students').groups


# you can group by with multiple columns 
df.groupby(['Students','Year']).groups


# iterating through groups
grouped = df.groupby('Students')
for student, group_name in grouped:
    print(student)
    print(group_name)


# select group by value
grouped = df.groupby('Year')
print(grouped.get_group(2014))


# find the mean of grouped by data
import numpy as np
grouped = df.groupby('Year')
print(grouped['Grade'].agg(np.mean))


# find the average for all students
grouped = df.groupby('Students')
print(grouped['Grade'].agg(np.mean).round())


# count 
grouped = df.groupby('Year')
print(grouped['Grade'].value_counts())


#Filtration filters the data on a defined criteria and returns the subset of data. 
#The filter() function is used to filter the data.
# we are going to find the top 3 students
df.groupby('Students').filter(lambda x: len(x) >= 3)


# ### I'll be updating this notebook soon!
# using real dataset!!
# 




# ## Cross Validation
# First, we'll try to build a model without cross-validation apply to it. Second, we'll use the same model with applying cross-validation and see if the accuracy score changed or not.
# 
# ### Main contents:
# * Build Support-Vector-Machine without cross-validation
# * Apply the same model with cross-validation
# 
# Note! We'll cover the metrics scores later
# 
# <img style="float:left;" src="https://cdn-images-1.medium.com/max/1600/1*J2B_bcbd1-s1kpWOu_FZrg.png" width=700 height=300>
# 

# libraries we need
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC


# load iris dataset
data_holder = load_iris()
print(data_holder.data.shape)
print(data_holder.target.shape)


# set our X and y to data and target values
X , y = data_holder.data, data_holder.target


# split our data into train and test sets
# let's split into 70/30: train=70% and test=30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .4, random_state = 0)


print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print()
print("y train shape: ", y_train.shape)
print("y test shape: ", y_test.shape)


# let's fit into our model, svc
# we'll set it to some parameters, but we'll go through depth on parameter tuning later
model = SVC(kernel='linear', C=1)
# fit our training data
model.fit(X_train, y_train)
# print how our model is doing
print("Score: ", model.score(X_test, y_test))


# As you can see our model scores 96% on our training data, we'll try to boost that accuracy score higher.
# 
# ### Computing cross-validated 
# The simplest way to use cross-validation is to call the cross_val_score helper function on the estimator and the dataset.
# 

# call cross-validation library
from sklearn.model_selection import cross_val_score
model = SVC(kernel='linear', C=1)

# let's try it using cv
scores = cross_val_score(model, X, y, cv=5)


# #### Evaluate Model
# Here is the output of our 5 KFold cross validation. Each value is the accuracy score of the support vector classifier when leaving out a different fold. There are three values because there are three folds. A higher accuracy score, the better.
# 

scores


# To get an good measure of the model's accuracy, we calculate the mean of the three scores. This is our measure of model accuracy.
# 

# print mean score
print("Accuracy using CV: ", scores.mean())


# ## I'll be updating more on this section!
# 




# ## Sorting arrays in Numpy
# Numpy, short for Numerical Python, is the fundamental package required for hight performance scientific computing and its best library to learn and apply on data science career.
# 
# 

import numpy as np


names = np.array(['F', 'C', 'A', 'G'])
weights = np.array([20.8, 93.2, 53.4, 61.8])

sort(weights)


#argsort
ordered_indices = np.argsort(weights)
ordered_indices


weights[ordered_indices]


names[ordered_indices]


data = np.array([20.8,  93.2,  53.4,  61.8])
data.argsort()


# sort data
data.sort()
data


# 2d array
a = np.array([
        [.2, .1, .5], 
        [.4, .8, .3],
        [.9, .6, .7]
    ])
a


sort(a)


# sort by column
sort(a, axis = 0)


# search sort
sorted_array = linspace(0,1,5)
values = array([.1,.8,.3,.12,.5,.25])


np.searchsorted(sorted_array, values)











# ## Structured Arrays in Numpy
# Numpy, short for Numerical Python, is the fundamental package required for hight performance scientific computing and its best library to learn and apply on data science career.
# 
# This is just little illustration.
# 
# <img style="float: left;" src="http://slideplayer.com/6419067/22/images/5/NumPy+Array.jpg" width=600 height=400>
# 

import numpy as np


a = np.array([1.0,2.0,3.0,4.0], np.float32)


# called the function view on our data
a.view(np.complex64)


# assign our to data to dtype
my_dtype = np.dtype([('mass', 'float32'), ('vol', 'float32')])


a.view(my_dtype)


my_data = np.array([(1,1), (1,2), (2,1), (1,3)], my_dtype)
print(my_data)


my_data[0]


my_data[0]['vol']


my_data['mass']


my_data.sort(order=('vol', 'mass'))
print(my_data)


person_dtype = np.dtype([('name', 'S10'), ('age', 'int'), ('weight', 'float')])


person_dtype.itemsize


people = np.empty((3,4), person_dtype)


people['age'] = [[33, 25, 47, 54],
                 [29, 61, 32, 27],
                 [19, 33, 18, 54]]


people['weight'] = [[135., 105., 255., 140.],
                    [154., 202., 137., 187.],
                    [188., 135., 88., 145.]]


print(people)


people[-1,-1]





# ## Matrix Object in Numpy
# Numpy, short for Numerical Python, is the fundamental package required for hight performance scientific computing and its best library to learn and apply on data science career.
# 
# This is just little illustration.
# 
# <img style="float: left;" src="https://www.safaribooksonline.com/library/view/python-for-data/9781449323592/httpatomoreillycomsourceoreillyimages1346880.png" width=400 height=200>
# 

import numpy as np


a = np.array([[1,2,4],
              [2,5,3], 
              [7,8,9]])
A = np.mat(a)
A


A = np.mat('1,2,4;2,5,3;7,8,9')
A


a = np.array([[ 1, 2],
              [ 3, 4]])
b = np.array([[10,20], 
              [30,40]])

np.bmat('a,b;b,a')


x = np.array([[4], [2], [3]])
x


A * x


print(A * A.I)


print(A ** 3)





# ## LabelEncoder
# Encode labels with value between 0 and n_classes-1.
# 

import warnings
warnings.filterwarnings('ignore')


from sklearn import preprocessing
# call our labelEncoder class
le = preprocessing.LabelEncoder()
# fit our data
le.fit([1, 2, 2, 6])
# print classes
le.classes_
# transform
le.transform([1, 1, 2, 6]) 
#print inverse data
le.inverse_transform([0, 0, 1, 2])


le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])

list(le.classes_)

le.transform(["tokyo", "tokyo", "paris"]) 

#list(le.inverse_transform([2, 2, 1]))





import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


data = pd.read_excel('data/Governance.xlsx', sheetname=0)
melt_data = data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name="Year")
melt_data = melt_data[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Year', 'value']]


melt_data.head(5)


copy_melt = melt_data.copy()


copy_melt = pd.pivot_table(copy_melt, values = 'value', index=['Country Name', 'Country Code','Year'], columns = 'Indicator Name').reset_index()
#copy_melt.index = piv_data['Year']
copy_melt.head(5)


#copy_melt.index = copy_melt.Year


copy_melt.head()


control_corruption = ['Year','Country Name','Control of Corruption: Estimate', 'Control of Corruption: Number of Sources',
                      'Control of Corruption: Percentile Rank', 
                      'Control of Corruption: Percentile Rank, Lower Bound of 90% Confidence Interval',
                      'Control of Corruption: Percentile Rank, Upper Bound of 90% Confidence Interval',
                      'Control of Corruption: Standard Error']
control_data = copy_melt[control_corruption]


government_effectivesness = ['Year','Country Name','Government Effectiveness: Estimate', 'Government Effectiveness: Number of Sources',
                             'Government Effectiveness: Percentile Rank', 
                             'Government Effectiveness: Percentile Rank, Lower Bound of 90% Confidence Interval',
                             'Government Effectiveness: Percentile Rank, Upper Bound of 90% Confidence Interval',
                             'Government Effectiveness: Standard Error']
government_data = copy_melt[government_effectivesness]
government_data.head(5)


political_stab = ['Year','Country Name','Political Stability and Absence of Violence/Terrorism: Estimate',
                  'Political Stability and Absence of Violence/Terrorism: Number of Sources',
                  'Political Stability and Absence of Violence/Terrorism: Percentile Rank',
                  'Political Stability and Absence of Violence/Terrorism: Percentile Rank, Lower Bound of 90% Confidence Interval',
                  'Political Stability and Absence of Violence/Terrorism: Percentile Rank, Upper Bound of 90% Confidence Interval',
                  'Political Stability and Absence of Violence/Terrorism: Standard Error']
political_data = copy_melt[political_stab]


regulatory_quality = ['Year','Country Name','Regulatory Quality: Estimate', 'Regulatory Quality: Number of Sources',
                      'Regulatory Quality: Percentile Rank', 
                      'Regulatory Quality: Percentile Rank, Lower Bound of 90% Confidence Interval',
                      'Regulatory Quality: Percentile Rank, Upper Bound of 90% Confidence Interval',
                      'Regulatory Quality: Standard Error']
regulatory_data = copy_melt[regulatory_quality]


rule_law = ['Year','Country Name','Rule of Law: Estimate', 'Rule of Law: Number of Sources', 'Rule of Law: Percentile Rank',
            'Rule of Law: Percentile Rank, Lower Bound of 90% Confidence Interval',
            'Rule of Law: Percentile Rank, Upper Bound of 90% Confidence Interval',
            'Rule of Law: Standard Error']
rule_data = copy_melt[rule_law]


voice_and_account = ['Year','Country Name','Voice and Accountability: Estimate', 'Voice and Accountability: Number of Sources',
                     'Voice and Accountability: Percentile Rank',
                     'Voice and Accountability: Percentile Rank, Lower Bound of 90% Confidence Interval',
                     'Voice and Accountability: Percentile Rank, Upper Bound of 90% Confidence Interval',
                     'Voice and Accountability: Standard Error']
voice_data = copy_melt[voice_and_account]


# ## Saving to CSV 
# 

voice_data.to_csv('data/voice_accountability.csv',encoding='utf-8', index=False)
rule_data.to_csv('data/rule_of_law.csv',encoding='utf-8', index=False)
regulatory_data.to_csv('data/regulatory_quality.csv',encoding='utf-8', index=False)
political_data.to_csv('data/political_stability.csv',encoding='utf-8', index=False)
control_data.to_csv('data/control_corruption.csv',encoding='utf-8', index=False)
government_data.to_csv('data/government_effectiveness.csv',encoding='utf-8', index=False)


voice_data.head()


vvv = pd.read_csv('data/voice_accountability.csv')
vvv.head()








voice_data.groupby(lambda x: pd.to_datetime(x))
voice_data.sort_values('Year').head()


SSA = ["Angola", "Gabon", "Nigeria", "Benin", "Gambia, The", "Rwanda", "Guinea-Bissau","Botswana", 
       "Ghana", "São Tomé and Principe", "Burkina Faso", "Guinea", "Senegal", "Burundi", "Seychelles", 
       "Cabo Verde", "Kenya", "Sierra Leone", "Cameroon", "Lesotho", "Somalia", "Central African Republic", 
       "Liberia", "South Africa", "Chad", "Madagascar", "Comoros", "Malawi", "Sudan", "Congo, Dem. Rep.", 
       "Mali", "Swaziland", "Congo, Rep", "Mauritania", "Tanzania", "Côte d'Ivoire", "Mauritius", "Togo", 
       "Equatorial Guinea", "Mozambique", "Uganda", "Eritrea" "Namibia", "Zambia", "Ethiopia", "Niger", "Zimbabwe"]
ssa_melt = voice_data[voice_data['Country Name'].isin(SSA)]


ssa_melt['Country Name'].nunique()


ssa_melt.head()


est_voice = ssa_melt[ssa_melt['Country Name'] == 'Somalia'].groupby('Voice and Accountability: Estimate').size().head(10).to_frame(name = 'count').reset_index()


ssa_melt.groupby('Country Name')['Voice and Accountability: Number of Sources'].mean()








ax = ssa_melt.plot(x='Year', y=["Country Name","Voice and Accountability: Number of Sources"])




















table = pivot_table(df, values='D', index=['Somalia'], columns=['C'], aggfunc=np.sum)


get_ipython().run_line_magic('pinfo', 'pd.pivot_table')


estimate = voice_data.groupby('Country Name')['Voice and Accountability: Estimate'].sum()


np.argsort(estimate)








# ## Working with image plotting
# Introduction:
# 
# Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shell, the jupyter notebook, web application servers, and four graphical user interface toolkits.
# 
# ### Here are the main steps we will go through
# * How to add text to graph?
# 
# This is Just a little illustration.
# 
# <img style="float:left;" src="https://i.imgur.com/bFsdlJy.png"></img>

import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


img = mpimg.imread('Beauty-Black-White-Wallpaper.jpg')


img.shape


imgplot = plt.imshow(img)


lum_img = img[:,:,0]
imgplot = plt.imshow(lum_img)


imgplot = plt.imshow(lum_img)
imgplot.set_cmap('hot')


imgplot = plt.imshow(lum_img)
imgplot.set_cmap('spectral')


imgplot = plt.imshow(lum_img)
imgplot.set_cmap('spectral')
plt.colorbar()
plt.show()


imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.0,0.7)





# ## Apply function 
# Subset rows or columns of dataframe according to labels in the specified index.
# 
# Note that this routine does not filter a dataframe on its contents. The filter is applied to the labels of the index.
# 

import pandas as pd
import numpy as np


data = pd.read_csv('data/train.csv')


data.head(4)


# let's use apply function to get the length of names
data["Name_length"] = data.Name.apply(len)


data.loc[0:5, ["Name", "Name_length"]]


# let's get the mean price on fare column
data["Fare_mean"] = data.Fare.apply(np.mean)


data.loc[0:5, ["Fare", "Fare_mean"]]


data.Name.str.split('.')[0][0].split(',')[1]


# let's get the name perfix, like Mr. Mrs. Mss. Ms...
data['prefix'] = data.Name.str.split('.').apply(lambda x: x[0].split(',')[1])


data.loc[0:10, ['Name', 'prefix']]


del data['dummy_prefix']


data.tail()


# let's get the unique prefix
data['prefix'].unique()


# let's use apply function to combined prefixes, 
# male = Mr Master, Don, rev, Dr, sir, col, capt, == 0
# female = Mrs miss, Mme, Ms, Lady, Mlle, the countess,Jonkheer  == 1


dummy_pre = data.groupby('prefix')


#list(data.groupby('prefix'))


dummy_pre.count()


get_dummy = data.prefix


pd.get_dummies(data['prefix'])


data.head()





