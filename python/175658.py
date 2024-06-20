# # Discover emerging trends
# In this notebook, we analyze the trends of Stack Overflow questions. In particular, we are finding the percentage of new questions tagged from four popular Python data science libraries: matplotlib, numpy, scikit-learn, and pandas.
# 
# The data was retrieved from Stack Overflow's handy data explorer [using this query](http://data.stackexchange.com/stackoverflow/query/767327/select-all-posts-from-a-single-tag).
# 
# We use pandas stacked area plot, by quarter, the percentage of new questions added to stack overflow from those four libraries.
# 

import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')


import glob


dfs = [pd.read_csv(file_name, parse_dates=['creationdate']) 
           for file_name in glob.glob('../data/stackoverflow/*.csv')]
df = pd.concat(dfs)


df.head()


df.groupby([pd.Grouper(key='creationdate', freq='QS'), 'tagname'])   .size()   .unstack('tagname', fill_value=0)   .pipe(lambda x: x.div(x.sum(1), axis=0))   .plot(kind='area', figsize=(12,6), title='Percentage of New Stack Overflow Questions')   .legend(loc='upper left')


# Pandas has grown the fastest
# 




# # Dataset Descriptions
# This notebook contains the descriptions of all the datasets used during the tutorials found within this [Learning Pandas repository](https://github.com/tdpetrou/Learn-Pandas).
# 
# ### Datasets
# * [Employee](#Employee-Data)
# * [Stack Overflow](#Stack-Overflow-Data)
# * [Food Inspections](#Food-Inspections-Data)
# 

# ## Helper Function
# The below **`create_description_table`** function can help create datasets descriptions for any DataFrame. You must first import each DataFrame in as normal and then pass it to the function. You must also pass it a list of the **`descriptions`** as strings.
# 

def create_description_table(df, descriptions, round_num=2):
    df_desc = df.dtypes.to_frame(name='Data Type')
    df_desc['Description'] = descriptions
    df_desc['Missing Values'] = df.isnull().sum()
    df_desc['Mean'] = df.select_dtypes('number').mean().round(round_num)
    df_desc['Most Common'] = df.apply(lambda x: x.value_counts().index[0])
    df_desc['Most Common Ct'] = df.apply(lambda x: x.value_counts().iloc[0])
    df_desc['Unique Values'] = df.nunique()
    return df_desc


# # Employee Data
# 
# ### Brief Overview
# The city of Houston provides information on all its employees to the public. This is a random sample of 2,000 employees with a selection of the more interesting columns. For more on [open Houston data visit their website](http://data.houstontx.gov/). Data was pulled in December, 2016.
# 

import pandas as pd


employee = pd.read_csv('../../data/employee.csv', parse_dates=['HIRE_DATE', 'JOB_DATE'])
employee.head()


employee.shape


descriptions = ['Position', 'Department', 'Base salary', 'Race', 
                'Full time/Part time/Temporary, etc...', 'Gender', 
                'Date hired', 'Date current job began']


create_description_table(employee, descriptions)


# # Stack Overflow Data
# This data was gathered from the [Stack Exchange data explorer](https://data.stackexchange.com/), an excellent tool to get almost any data you want from any of the Stack Exchange sites.
# 
# This particular dataset was collected December 7, 2017 with [this query](http://data.stackexchange.com/stackoverflow/query/768430/get-all-questions-and-answerers-from-tag). You'll have to run the query twice to get all the data because the query exceeds 50,000, the maximum allowable number of rows. Switch the inequality on the **`creationdate`** in the `where` clause to do so.
# 

so = pd.read_csv('../../data/stackoverflow_qa.csv')
so.head()


so.shape


descriptions = ['Question ID', 'Creation date', '# of question upvotes', 'View count',
                'Question Title', 'Number of Answers', 'Number of comments for Question',
                'Number of favorites for Question', 'User name of question author',
                'Reputation of question author', 'User name of selected answer author',
                'Reputation of selected answer author']


create_description_table(so, descriptions)


# # Food Inspections Data
# 

food_inspections = pd.read_csv('../../data/food_inspections.csv', parse_dates=['Inspection Date'])
food_inspections.head()


food_inspections.shape


descriptions = ['Doing business as Name', 'Restaurant, Grocery store, School, Bakery, etc...',
                'High/Medium/Low', 'Address', 'Zip Code', 'Inspection Date',
                'Inspection Type', 'Pass/Fail/Out of business, etc...',
                'Detailed description of violations']


create_description_table(food_inspections, descriptions)





