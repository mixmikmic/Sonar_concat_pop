# # Chapter 6: Index Alignment
# ## Recipes
# * [Examining the Index object](#Examining-the-index)
# * [Producing Cartesian products](#Producing-Cartesian-products)
# * [Exploding indexes](#Exploding-Indexes)
# * [Filling values with unequal indexes](#Filling-values-with-unequal-indexes)
# * [Appending columns from different DataFrames](#Appending-columns-from-different-DataFrames)
# * [Highlighting the maximum value from each column](#Highlighting-maximum-value-from-each-column)
# * [Replicating idxmax with method chaining](#Replicating-idxmax-with-method-chaining)
# * [Finding the most common maximum](#Finding-the-most-common-maximum)
# 

import pandas as pd
import numpy as np


# # Examining the index
# 

college = pd.read_csv('data/college.csv')
columns = college.columns
columns


columns.values


columns[5]


columns[[1,8,10]]


columns[-7:-4]


columns.min(), columns.max(), columns.isnull().sum()


columns + '_A'


columns > 'G'


columns[1] = 'city'


c1 = columns[:4]
c1


c2 = columns[2:5]
c2


c1.union(c2)


c1 | c2


c1.symmetric_difference(c2)


c1 ^ c2


# # Producing Cartesian products
# 

s1 = pd.Series(index=list('aaab'), data=np.arange(4))
s1


s2 = pd.Series(index=list('cababb'), data=np.arange(6))
s2


s1 + s2


# ## There's more
# 

s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('aaabb'), data=np.arange(5))
s1 + s2


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('bbaaa'), data=np.arange(5))
s1 + s2


# # Exploding Indexes
# 

employee = pd.read_csv('data/employee.csv', index_col='RACE')
employee.head()


salary1 = employee['BASE_SALARY']
salary2 = employee['BASE_SALARY']
salary1 is salary2


salary1 = employee['BASE_SALARY'].copy()
salary2 = employee['BASE_SALARY'].copy()
salary1 is salary2


salary1 = salary1.sort_index()
salary1.head()


salary2.head()


salary_add = salary1 + salary2


salary_add.head()


salary_add1 = salary1 + salary1
len(salary1), len(salary2), len(salary_add), len(salary_add1)


# ## There's more...
# 

index_vc = salary1.index.value_counts(dropna=False)
index_vc


index_vc.pow(2).sum()


# # Filling values with unequal indexes
# 

baseball_14 = pd.read_csv('data/baseball14.csv', index_col='playerID')
baseball_15 = pd.read_csv('data/baseball15.csv', index_col='playerID')
baseball_16 = pd.read_csv('data/baseball16.csv', index_col='playerID')
baseball_14.head()


baseball_14.index.difference(baseball_15.index)


baseball_14.index.difference(baseball_15.index)


hits_14 = baseball_14['H']
hits_15 = baseball_15['H']
hits_16 = baseball_16['H']
hits_14.head()


(hits_14 + hits_15).head()


hits_14.add(hits_15, fill_value=0).head()


hits_total = hits_14.add(hits_15, fill_value=0).add(hits_16, fill_value=0)
hits_total.head()


hits_total.hasnans


# ## How it works...
# 

s = pd.Series(index=['a', 'b', 'c', 'd'], data=[np.nan, 3, np.nan, 1])
s


s1 = pd.Series(index=['a', 'b', 'c'], data=[np.nan, 6, 10])
s1


s.add(s1, fill_value=5)


s1.add(s, fill_value=5)


# ## There's more
# 

df_14 = baseball_14[['G','AB', 'R', 'H']]
df_14.head()


df_15 = baseball_15[['AB', 'R', 'H', 'HR']]
df_15.head()


(df_14 + df_15).head(10).style.highlight_null('yellow')


df_14.add(df_15, fill_value=0).head(10).style.highlight_null('yellow')


# # Appending columns from different DataFrames
# 

employee = pd.read_csv('data/employee.csv')
dept_sal = employee[['DEPARTMENT', 'BASE_SALARY']]


dept_sal = dept_sal.sort_values(['DEPARTMENT', 'BASE_SALARY'],
                                ascending=[True, False])


max_dept_sal = dept_sal.drop_duplicates(subset='DEPARTMENT')
max_dept_sal.head()


max_dept_sal = max_dept_sal.set_index('DEPARTMENT')
employee = employee.set_index('DEPARTMENT')


employee['MAX_DEPT_SALARY'] = max_dept_sal['BASE_SALARY']


pd.options.display.max_columns = 6


employee.head()


employee.query('BASE_SALARY > MAX_DEPT_SALARY')


# ## How it works...
# 

np.random.seed(1234)
random_salary = dept_sal.sample(n=10).set_index('DEPARTMENT')
random_salary


employee['RANDOM_SALARY'] = random_salary['BASE_SALARY']


# ## There's more...
# 

employee['MAX_SALARY2'] = max_dept_sal['BASE_SALARY'].head(3)


employee.MAX_SALARY2.value_counts()


employee.MAX_SALARY2.isnull().mean()


# # Highlighting maximum value from each column
# 

pd.options.display.max_rows = 8


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.dtypes


college.MD_EARN_WNE_P10.iloc[0]


college.GRAD_DEBT_MDN_SUPP.iloc[0]


college.MD_EARN_WNE_P10.sort_values(ascending=False).head()


cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')

college.dtypes.loc[cols]


college_n = college.select_dtypes(include=[np.number])
college_n.head() # only numeric columns


criteria = college_n.nunique() == 2
criteria.head()


binary_cols = college_n.columns[criteria].tolist()
binary_cols


college_n2 = college_n.drop(labels=binary_cols, axis='columns')
college_n2.head()


max_cols = college_n2.idxmax()
max_cols


unique_max_cols = max_cols.unique()
unique_max_cols[:5]


college_n2.loc[unique_max_cols].style.highlight_max()


# ## There's more...
# 

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_').head()
college_ugds.style.highlight_max(axis='columns')


pd.Timedelta(1, unit='Y')


# # Replicating idxmax with method chaining
# 

college = pd.read_csv('data/college.csv', index_col='INSTNM')

cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')

college_n = college.select_dtypes(include=[np.number])
criteria = college_n.nunique() == 2
binary_cols = college_n.columns[criteria].tolist()
college_n = college_n.drop(labels=binary_cols, axis='columns')


college_n.max().head()


college_n.eq(college_n.max()).head()


has_row_max = college_n.eq(college_n.max()).any(axis='columns')
has_row_max.head()


college_n.shape


has_row_max.sum()


pd.options.display.max_rows=6


college_n.eq(college_n.max()).cumsum().cumsum()


has_row_max2 = college_n.eq(college_n.max())                        .cumsum()                        .cumsum()                        .eq(1)                        .any(axis='columns')
has_row_max2.head()


has_row_max2.sum()


idxmax_cols = has_row_max2[has_row_max2].index
idxmax_cols


set(college_n.idxmax().unique()) == set(idxmax_cols)


# ## There's more...
# 

get_ipython().run_line_magic('timeit', 'college_n.idxmax().values')


get_ipython().run_line_magic('timeit', "college_n.eq(college_n.max())                              .cumsum()                              .cumsum()                              .eq(1)                              .any(axis='columns')                              [lambda x: x].index")


# # Finding the most common maximum
# 

pd.options.display.max_rows= 40


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


highest_percentage_race = college_ugds.idxmax(axis='columns')
highest_percentage_race.head()


highest_percentage_race.value_counts(normalize=True)


# # There's more...
# 

college_black = college_ugds[highest_percentage_race == 'UGDS_BLACK']
college_black = college_black.drop('UGDS_BLACK', axis='columns')
college_black.idxmax(axis='columns').value_counts(normalize=True)





# # Chapter 9: Combining Pandas Objects
# ## Recipes
# * [Appending new rows to DataFrames](#Appending-new-rows-to-DataFrames)
# * [Concatenating multiple DataFrames together](#Concatenating-multiple-DataFrames-together)
# * [Comparing President Trump's and Obama's approval ratings](#Comparing-President-Trump's-and-Obama's-approval-ratings)
# * [Understanding the differences between concat, join, and merge](#Understanding-the-differences-between-concat,-join,-and-merge)
# * [Connecting to SQL databases](#Connecting-to-SQL-Databases)
# 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Appending new rows to DataFrames
# 

names = pd.read_csv('data/names.csv')
names


new_data_list = ['Aria', 1]
names.loc[4] = new_data_list
names


names.loc['five'] = ['Zach', 3]
names


names.loc[len(names)] = {'Name':'Zayd', 'Age':2}
names


names


names.loc[len(names)] = pd.Series({'Age':32, 'Name':'Dean'})
names


# Use append with fresh copy of names
names = pd.read_csv('data/names.csv')
names.append({'Name':'Aria', 'Age':1})


names.append({'Name':'Aria', 'Age':1}, ignore_index=True)


names.index = ['Canada', 'Canada', 'USA', 'USA']
names


names.append({'Name':'Aria', 'Age':1}, ignore_index=True)


s = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s


names.append(s)


s1 = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s2 = pd.Series({'Name': 'Zayd', 'Age': 2}, name='USA')
names.append([s1, s2])


bball_16 = pd.read_csv('data/baseball16.csv')
bball_16.head()


data_dict = bball_16.iloc[0].to_dict()
print(data_dict)


new_data_dict = {k: '' if isinstance(v, str) else np.nan for k, v in data_dict.items()}
print(new_data_dict)


# ## There's more...
# 

random_data = []
for i in range(1000):
    d = dict()
    for k, v in data_dict.items():
        if isinstance(v, str):
            d[k] = np.random.choice(list('abcde'))
        else:
            d[k] = np.random.randint(10)
    random_data.append(pd.Series(d, name=i + len(bball_16)))
    
random_data[0].head()


get_ipython().run_cell_magic('timeit', '', 'bball_16_copy = bball_16.copy()\nfor row in random_data:\n    bball_16_copy = bball_16_copy.append(row)')


get_ipython().run_cell_magic('timeit', '', 'bball_16_copy = bball_16.copy()\nbball_16_copy = bball_16_copy.append(random_data)')


# # Concatenating multiple DataFrames together
# 

stocks_2016 = pd.read_csv('data/stocks_2016.csv', index_col='Symbol')
stocks_2017 = pd.read_csv('data/stocks_2017.csv', index_col='Symbol')


stocks_2016


stocks_2017


s_list = [stocks_2016, stocks_2017]
pd.concat(s_list)


pd.concat(s_list, keys=['2016', '2017'], names=['Year', 'Symbol'])


pd.concat(s_list, keys=['2016', '2017'], axis='columns', names=['Year', None])


pd.concat(s_list, join='inner', keys=['2016', '2017'], axis='columns', names=['Year', None])


# ## There's more...
# 

stocks_2016.append(stocks_2017)


stocks_2015 = stocks_2016.copy()


stocks_2017


# possibly add rule for no duplicate index
# 

# # Comparing President Trump's and Obama's approval ratings
# 

base_url = 'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
trump_url = base_url.format(45)

df_list = pd.read_html(trump_url)
len(df_list)


df0 = df_list[0]
df0.shape


df0.head(7)


df_list = pd.read_html(trump_url, match='Start Date')
len(df_list)


df_list = pd.read_html(trump_url, match='Start Date', attrs={'align':'center'})
len(df_list)


trump = df_list[0]
trump.shape


trump.head(8)


df_list = pd.read_html(trump_url, match='Start Date', attrs={'align':'center'}, 
                       header=0, skiprows=[0,1,2,3,5], parse_dates=['Start Date', 'End Date'])
trump = df_list[0]
trump.head()


trump = trump.dropna(axis=1, how='all')
trump.head()


trump.isnull().sum()


trump = trump.ffill()
trump.head()


trump.dtypes


def get_pres_appr(pres_num):
    base_url = 'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
    pres_url = base_url.format(pres_num)
    df_list = pd.read_html(pres_url, match='Start Date', attrs={'align':'center'}, 
                       header=0, skiprows=[0,1,2,3,5], parse_dates=['Start Date', 'End Date'])
    pres = df_list[0].copy()
    pres = pres.dropna(axis=1, how='all')
    pres['President'] = pres['President'].ffill()
    return pres.sort_values('End Date').reset_index(drop=True)


obama = get_pres_appr(44)
obama.head()


pres_41_45 = pd.concat([get_pres_appr(x) for x in range(41,46)], ignore_index=True)
pres_41_45.groupby('President').head(3)


pres_41_45['End Date'].value_counts().head(8)


pres_41_45 = pres_41_45.drop_duplicates(subset='End Date')


pres_41_45.shape


pres_41_45['President'].value_counts()


pres_41_45.groupby('President', sort=False).median().round(1)


from matplotlib import cm
fig, ax = plt.subplots(figsize=(16,6))

styles = ['-.', '-', ':', '-', ':']
colors = [.9, .3, .7, .3, .9]
groups = pres_41_45.groupby('President', sort=False)

for style, color, (pres, df) in zip(styles, colors, groups):
    df.plot('End Date', 'Approving', ax=ax, label=pres, style=style, color=cm.Greys(color), 
            title='Presedential Approval Rating')


days_func = lambda x: x - x.iloc[0]
pres_41_45['Days in Office'] = pres_41_45.groupby('President')                                              ['End Date']                                              .transform(days_func)


pres_41_45['Days in Office'] = pres_41_45.groupby('President')['End Date'].transform(lambda x: x - x.iloc[0])
pres_41_45.groupby('President').head(3)


pres_41_45.dtypes


pres_41_45['Days in Office'] = pres_41_45['Days in Office'].dt.days
pres_41_45['Days in Office'].head()


pres_pivot = pres_41_45.pivot(index='Days in Office', columns='President', values='Approving')
pres_pivot.head()


plot_kwargs = dict(figsize=(16,6), color=cm.gray([.3, .7]), style=['-', '--'], title='Approval Rating')
pres_pivot.loc[:250, ['Donald J. Trump', 'Barack Obama']].ffill().plot(**plot_kwargs)


# ## There's more...
# 

pres_rm = pres_41_45.groupby('President', sort=False)                     .rolling('90D', on='End Date')['Approving']                     .mean()
pres_rm.head()


styles = ['-.', '-', ':', '-', ':']
colors = [.9, .3, .7, .3, .9]
color = cm.Greys(colors)
title='90 Day Approval Rating Rolling Average'
plot_kwargs = dict(figsize=(16,6), style=styles, color = color, title=title)
correct_col_order = pres_41_45.President.unique()
pres_rm.unstack('President')[correct_col_order].plot(**plot_kwargs)


# # Understanding the differences between concat, join, and merge
# 

from IPython.display import display_html

years = 2016, 2017, 2018
stock_tables = [pd.read_csv('data/stocks_{}.csv'.format(year), index_col='Symbol') 
                for year in years]

def display_frames(frames, num_spaces=0):
    t_style = '<table style="display: inline;"'
    tables_html = [df.to_html().replace('<table', t_style) for df in frames]

    space = '&nbsp;' * num_spaces
    display_html(space.join(tables_html), raw=True)

display_frames(stock_tables, 30)
stocks_2016, stocks_2017, stocks_2018 = stock_tables


pd.concat(stock_tables, keys=[2016, 2017, 2018])


pd.concat(dict(zip(years,stock_tables)), axis='columns')


stocks_2016.join(stocks_2017, lsuffix='_2016', rsuffix='_2017', how='outer')


stocks_2016


other = [stocks_2017.add_suffix('_2017'), stocks_2018.add_suffix('_2018')]
stocks_2016.add_suffix('_2016').join(other, how='outer')


stock_join = stocks_2016.add_suffix('_2016').join(other, how='outer')
stock_concat = pd.concat(dict(zip(years,stock_tables)), axis='columns')


stock_concat.columns = stock_concat.columns.get_level_values(1) + '_' +                             stock_concat.columns.get_level_values(0).astype(str)


stock_concat


step1 = stocks_2016.merge(stocks_2017, left_index=True, right_index=True, 
                          how='outer', suffixes=('_2016', '_2017'))
stock_merge = step1.merge(stocks_2018.add_suffix('_2018'), 
                          left_index=True, right_index=True, how='outer')

stock_concat.equals(stock_merge)


names = ['prices', 'transactions']
food_tables = [pd.read_csv('data/food_{}.csv'.format(name)) for name in names]
food_prices, food_transactions = food_tables
display_frames(food_tables, 30)


food_transactions.merge(food_prices, on=['item', 'store'])


food_transactions.merge(food_prices.query('Date == 2017'), how='left')


food_prices_join = food_prices.query('Date == 2017').set_index(['item', 'store'])
food_prices_join


food_transactions.join(food_prices_join, on=['item', 'store'])


pd.concat([food_transactions.set_index(['item', 'store']), 
           food_prices.set_index(['item', 'store'])], axis='columns')


import glob

df_list = []
for filename in glob.glob('data/gas prices/*.csv'):
    df_list.append(pd.read_csv(filename, index_col='Week', parse_dates=['Week']))

gas = pd.concat(df_list, axis='columns')
gas.head()


# # Connecting to SQL Databases
# 

from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')


tracks = pd.read_sql_table('tracks', engine)
tracks.head()


genres = pd.read_sql_table('genres', engine)
genres.head()


genre_track = genres.merge(tracks[['GenreId', 'Milliseconds']], 
                           on='GenreId', how='left') \
                     .drop('GenreId', axis='columns')
genre_track.head()


genre_time = genre_track.groupby('Name')['Milliseconds'].mean()
pd.to_timedelta(genre_time, unit='ms').dt.floor('s').sort_values()


cust = pd.read_sql_table('customers', engine, 
                         columns=['CustomerId', 'FirstName', 'LastName'])
invoice = pd.read_sql_table('invoices', engine, 
                            columns=['InvoiceId','CustomerId'])
ii = pd.read_sql_table('invoice_items', engine, 
                       columns=['InvoiceId', 'UnitPrice', 'Quantity'])


cust_inv = cust.merge(invoice, on='CustomerId')                .merge(ii, on='InvoiceId')
cust_inv.head()


total = cust_inv['Quantity'] * cust_inv['UnitPrice']
cols = ['CustomerId', 'FirstName', 'LastName']
cust_inv.assign(Total = total).groupby(cols)['Total']                                   .sum()                                   .sort_values(ascending=False).head()


# ## There's more...
# 

pd.read_sql_query('select * from tracks limit 5', engine)


sql_string1 = '''
select 
    Name, 
    time(avg(Milliseconds) / 1000, 'unixepoch') as avg_time
from (
        select 
            g.Name, 
            t.Milliseconds
        from 
            genres as g 
        join
            tracks as t
            on 
                g.genreid == t.genreid
    )
group by 
    Name
order by 
    avg_time
'''
pd.read_sql_query(sql_string1, engine)


sql_string2 = '''
select 
      c.customerid, 
      c.FirstName, 
      c.LastName, 
      sum(ii.quantity *  ii.unitprice) as Total
from
     customers as c
join
     invoices as i
          on c.customerid = i.customerid
join
    invoice_items as ii
          on i.invoiceid = ii.invoiceid
group by
    c.customerid, c.FirstName, c.LastName
order by
    Total desc
'''
pd.read_sql_query(sql_string2, engine)





# # Chapter 3: Beginning Data Analysis
# 
# ## Recipes
# * [Developing a data analysis routine](#Developing-a-data-analysis-routine)
# * [Reducing memory by changing data types](#Reducing-memory-by-changing-data-types)
# * [Selecting the smallest of the largest](#Selecting-the-smallest-of-the-largest)
# * [Selecting the largest of each group by sorting](#Selecting-the-largest-of-each-group-by-sorting)
# * [Replicating nlargest with sort_values](#Replicating-nlargest-with-sort_values)
# * [Calculating a trailing stop order price](#Calculating-a-trailing-stop-order-price)
# 

import pandas as pd
import numpy as np
from IPython.display import display
pd.options.display.max_columns = 50


# # Developing a data analysis routine
# 

college = pd.read_csv('data/college.csv')


college.head()


college.shape


with pd.option_context('display.max_rows', 8):
    display(college.describe(include=[np.number]).T)


college.describe(include=[np.object, pd.Categorical]).T


college.info()


college.describe(include=[np.number]).T


college.describe(include=[np.object, pd.Categorical]).T


# ## There's more...
# 

with pd.option_context('display.max_rows', 5):
    display(college.describe(include=[np.number], 
                 percentiles=[.01, .05, .10, .25, .5, .75, .9, .95, .99]).T)


college_dd = pd.read_csv('data/college_data_dictionary.csv')


with pd.option_context('display.max_rows', 8):
    display(college_dd)


# # Reducing memory by changing data types
# 

college = pd.read_csv('data/college.csv')
different_cols = ['RELAFFIL', 'SATMTMID', 'CURROPER', 'INSTNM', 'STABBR']
col2 = college.loc[:, different_cols]
col2.head()


col2.dtypes


original_mem = col2.memory_usage(deep=True)
original_mem


col2['RELAFFIL'] = col2['RELAFFIL'].astype(np.int8)


col2.dtypes


col2.select_dtypes(include=['object']).nunique()


col2['STABBR'] = col2['STABBR'].astype('category')
col2.dtypes


new_mem = col2.memory_usage(deep=True)
new_mem


new_mem / original_mem


# ## There's more...
# 

college = pd.read_csv('data/college.csv')


college[['CURROPER', 'INSTNM']].memory_usage(deep=True)


college.loc[0, 'CURROPER'] = 10000000
college.loc[0, 'INSTNM'] = college.loc[0, 'INSTNM'] + 'a'
# college.loc[1, 'INSTNM'] = college.loc[1, 'INSTNM'] + 'a'
college[['CURROPER', 'INSTNM']].memory_usage(deep=True)


college['MENONLY'].dtype


college['MENONLY'].astype('int8') # ValueError: Cannot convert non-finite values (NA or inf) to integer


college.describe(include=['int64', 'float64']).T


college.describe(include=[np.int64, np.float64]).T


college['RELAFFIL'] = college['RELAFFIL'].astype(np.int8)


college.describe(include=['int', 'float']).T  # defaults to 64 bit int/floats


college.describe(include=['number']).T  # also works as the default int/float are 64 bits


college['MENONLY'] = college['MENONLY'].astype('float16')
college['RELAFFIL'] = college['RELAFFIL'].astype('int8')


college.index = pd.Int64Index(college.index)
college.index.memory_usage()


# # Selecting the smallest of the largest
# 

movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie2.head()


movie2.nlargest(100, 'imdb_score').head()


movie2.nlargest(100, 'imdb_score').nsmallest(5, 'budget')


# # Selecting the largest of each group by sorting
# 

movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'title_year', 'imdb_score']]


movie2.sort_values('title_year', ascending=False).head()


movie3 = movie2.sort_values(['title_year','imdb_score'], ascending=False)
movie3.head()


movie_top_year = movie3.drop_duplicates(subset='title_year')
movie_top_year.head()


movie4 = movie[['movie_title', 'title_year', 'content_rating', 'budget']]
movie4_sorted = movie4.sort_values(['title_year', 'content_rating', 'budget'], 
                                   ascending=[False, False, True])
movie4_sorted.drop_duplicates(subset=['title_year', 'content_rating']).head(10)


# # Replicating nlargest with sort_values
# 

movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie_smallest_largest = movie2.nlargest(100, 'imdb_score').nsmallest(5, 'budget')
movie_smallest_largest


movie2.sort_values('imdb_score', ascending=False).head(100).head()


movie2.sort_values('imdb_score', ascending=False).head(100).sort_values('budget').head()


movie2.nlargest(100, 'imdb_score').tail()


movie2.sort_values('imdb_score', ascending=False).head(100).tail()


# # Calculating a trailing stop order price
# 

import pandas_datareader as pdr


# ### Note: pandas_datareader issues
# pandas_datareader can have issues when the source is 'google'. It can also read from Yahoo! finance. Try switching it to 'yahoo'
# 

tsla = pdr.DataReader('tsla', data_source='yahoo',start='2017-1-1')
tsla.head(8)


tsla_close = tsla['Close']


tsla_cummax = tsla_close.cummax()
tsla_cummax.head(8)


tsla_trailing_stop = tsla_cummax * .9
tsla_trailing_stop.head(8)


# ## There's more...
# 

def set_trailing_loss(symbol, purchase_date, perc):
    close = pdr.DataReader(symbol, 'yahoo', start=purchase_date)['Close']
    return close.cummax() * perc


msft_trailing_stop = set_trailing_loss('msft', '2017-6-1', .85)
msft_trailing_stop.head()





# # Chapter 5: Boolean Indexing
# ## Recipes
# * [Calculating boolean statistics](#Calculating-boolean-statistics)
# * [Constructing multiple boolean conditions](#Constructing-multiple-boolean-conditions)
# * [Filtering with boolean indexing](#Filtering-with-boolean-indexing)
# * [Replicating boolean indexing with index selection](#Replicating-boolean-indexing-with-index-selection)
# * [Selecting with unique and sorted indexes](#Selecting-with-unique-and-sorted-indexes)
# * [Gaining perspective on stock prices](#Gaining-perspective-on-stock-prices)
# * [Translating SQL WHERE clauses](#Translating-SQL-WHERE-clauses)
# * [Determining the normality of stock market returns](#Determining-the-normality-of-stock-market-returns)
# * [Improving readability of boolean indexing with the query method](#Improving-readability-of-boolean-indexing-with-the-query-method)
# * [Preserving Series with the where method](#Preserving-Series-with-the-where-method)
# * [Masking DataFrame rows](#Masking-DataFrame-rows)
# * [Selecting with booleans, integer location, and labels](#Selecting-with-booleans,-integer-location-and-labels)
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Calculating boolean statistics
# 

pd.options.display.max_columns = 50


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie.head()


movie_2_hours = movie['duration'] > 120
movie_2_hours.head(10)


movie_2_hours.sum()


movie_2_hours.mean()


movie_2_hours.describe()


movie['duration'].dropna().gt(120).mean()


# ## How it works...
# 

movie_2_hours.value_counts(normalize=True)


# ## There's more...
# 

actors = movie[['actor_1_facebook_likes', 'actor_2_facebook_likes']].dropna()
(actors['actor_1_facebook_likes'] > actors['actor_2_facebook_likes']).mean()


# # Constructing multiple boolean conditions
# 

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie.head()


criteria1 = movie.imdb_score > 8
criteria2 = movie.content_rating == 'PG-13'
criteria3 = (movie.title_year < 2000) | (movie.title_year >= 2010)

criteria2.head()


criteria_final = criteria1 & criteria2 & criteria3
criteria_final.head()


# # There's more...
# 

movie.title_year < 2000 | movie.title_year > 2009


# # Filtering with boolean indexing
# 

movie = pd.read_csv('data/movie.csv', index_col='movie_title')

crit_a1 = movie.imdb_score > 8
crit_a2 = movie.content_rating == 'PG-13'
crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
final_crit_a = crit_a1 & crit_a2 & crit_a3


crit_b1 = movie.imdb_score < 5
crit_b2 = movie.content_rating == 'R'
crit_b3 = (movie.title_year >= 2000) & (movie.title_year <= 2010)
final_crit_b = crit_b1 & crit_b2 & crit_b3


final_crit_all = final_crit_a | final_crit_b
final_crit_all.head()


movie[final_crit_all].head()


cols = ['imdb_score', 'content_rating', 'title_year']
movie_filtered = movie.loc[final_crit_all, cols]
movie_filtered.head(10)


# # There's more...
# 

final_crit_a2 = (movie.imdb_score > 8) &                 (movie.content_rating == 'PG-13') &                 ((movie.title_year < 2000) | (movie.title_year > 2009))
final_crit_a2.equals(final_crit_a)


# # Replicating boolean indexing with index selection
# 

college = pd.read_csv('data/college.csv')
college[college['STABBR'] == 'TX'].head()


college2 = college.set_index('STABBR')
college2.loc['TX'].head()


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


get_ipython().run_line_magic('timeit', "college2 = college.set_index('STABBR')")


# ## There's more...
# 

states =['TX', 'CA', 'NY']
college[college['STABBR'].isin(states)]
college2.loc[states].head()


# # Selecting with unique and sorted indexes
# 

college = pd.read_csv('data/college.csv')
college2 = college.set_index('STABBR')


college2.index.is_monotonic


college3 = college2.sort_index()
college3.index.is_monotonic


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


get_ipython().run_line_magic('timeit', "college3.loc['TX']")


college_unique = college.set_index('INSTNM')
college_unique.index.is_unique


college[college['INSTNM'] == 'Stanford University']


college_unique.loc['Stanford University']


get_ipython().run_line_magic('timeit', "college[college['INSTNM'] == 'Stanford University']")


get_ipython().run_line_magic('timeit', "college_unique.loc['Stanford University']")


# ## There's more...
# 

college.index = college['CITY'] + ', ' + college['STABBR']
college = college.sort_index()
college.head()


college.loc['Miami, FL'].head()


get_ipython().run_cell_magic('timeit', '', "crit1 = college['CITY'] == 'Miami' \ncrit2 = college['STABBR'] == 'FL'\ncollege[crit1 & crit2]")


get_ipython().run_line_magic('timeit', "college.loc['Miami, FL']")


college[(college['CITY'] == 'Miami') & (college['STABBR'] == 'FL')].equals(college.loc['Miami, FL'])


# # Gaining perspective on stock prices
# 

slb = pd.read_csv('data/slb_stock.csv', index_col='Date', parse_dates=['Date'])
slb.head()


slb_close = slb['Close']
slb_summary = slb_close.describe(percentiles=[.1, .9])
slb_summary


upper_10 = slb_summary.loc['90%']
lower_10 = slb_summary.loc['10%']
criteria = (slb_close < lower_10) | (slb_close > upper_10)
slb_top_bottom_10 = slb_close[criteria]


slb_close.plot(color='black', figsize=(12,6))
slb_top_bottom_10.plot(marker='o', style=' ', ms=4, color='lightgray')

xmin = criteria.index[0]
xmax = criteria.index[-1]
plt.hlines(y=[lower_10, upper_10], xmin=xmin, xmax=xmax,color='black')


# ## There's more...
# 

slb_close.plot(color='black', figsize=(12,6))
plt.hlines(y=[lower_10, upper_10], 
           xmin=xmin, xmax=xmax,color='lightgray')
plt.fill_between(x=criteria.index, y1=lower_10,
                 y2=slb_close.values, color='black')
plt.fill_between(x=criteria.index,y1=lower_10,
                 y2=slb_close.values, where=slb_close < lower_10,
                 color='lightgray')
plt.fill_between(x=criteria.index, y1=upper_10, 
                 y2=slb_close.values, where=slb_close > upper_10,
                 color='lightgray')


# # Translating SQL WHERE clauses
# 

employee = pd.read_csv('data/employee.csv')


employee.DEPARTMENT.value_counts().head()


employee.GENDER.value_counts()


employee.BASE_SALARY.describe().astype(int)


depts = ['Houston Police Department-HPD', 
             'Houston Fire Department (HFD)']
criteria_dept = employee.DEPARTMENT.isin(depts)
criteria_gender = employee.GENDER == 'Female'
criteria_sal = (employee.BASE_SALARY >= 80000) &                (employee.BASE_SALARY <= 120000)


criteria_final = criteria_dept & criteria_gender & criteria_sal


select_columns = ['UNIQUE_ID', 'DEPARTMENT', 'GENDER', 'BASE_SALARY']
employee.loc[criteria_final, select_columns].head()


# ## There's more...
# 

criteria_sal = employee.BASE_SALARY.between(80000, 120000)


top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
criteria = ~employee.DEPARTMENT.isin(top_5_depts)
employee[criteria].head()


# # Determining the normality of stock market returns
# 

amzn = pd.read_csv('data/amzn_stock.csv', index_col='Date', parse_dates=['Date'])
amzn.head()


amzn_daily_return = amzn.Close.pct_change()
amzn_daily_return.head()


amzn_daily_return = amzn_daily_return.dropna()
amzn_daily_return.hist(bins=20)


mean = amzn_daily_return.mean()  
std = amzn_daily_return.std()


abs_z_score = amzn_daily_return.sub(mean).abs().div(std)


pcts = [abs_z_score.lt(i).mean() for i in range(1,4)]
print('{:.3f} fall within 1 standard deviation. '
      '{:.3f} within 2 and {:.3f} within 3'.format(*pcts))


def test_return_normality(stock_data):
    close = stock_data['Close']
    daily_return = close.pct_change().dropna()
    daily_return.hist(bins=20)
    mean = daily_return.mean() 
    std = daily_return.std()
    
    abs_z_score = abs(daily_return - mean) / std
    pcts = [abs_z_score.lt(i).mean() for i in range(1,4)]

    print('{:.3f} fall within 1 standard deviation. '
          '{:.3f} within 2 and {:.3f} within 3'.format(*pcts))


slb = pd.read_csv('data/slb_stock.csv', 
                  index_col='Date', parse_dates=['Date'])
test_return_normality(slb)


# # Improving readability of boolean indexing with the query method
# 

employee = pd.read_csv('data/employee.csv')
depts = ['Houston Police Department-HPD', 'Houston Fire Department (HFD)']
select_columns = ['UNIQUE_ID', 'DEPARTMENT', 'GENDER', 'BASE_SALARY']


qs = "DEPARTMENT in @depts "          "and GENDER == 'Female' "          "and 80000 <= BASE_SALARY <= 120000"
        
emp_filtered = employee.query(qs)
emp_filtered[select_columns].head()


# # There's more...
# 

top10_depts = employee.DEPARTMENT.value_counts().index[:10].tolist()
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2[['DEPARTMENT', 'GENDER']].head()


# # Preserving Series with the where method
# 

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()


fb_likes.describe(percentiles=[.1, .25, .5, .75, .9]).astype(int)


fb_likes.describe(percentiles=[.1,.25,.5,.75,.9])


fb_likes.hist()


criteria_high = fb_likes < 20000
criteria_high.mean().round(2)


fb_likes.where(criteria_high).head()


fb_likes.where(criteria_high, other=20000).head()


criteria_low = fb_likes > 300
fb_likes_cap = fb_likes.where(criteria_high, other=20000)                       .where(criteria_low, 300)
fb_likes_cap.head()


len(fb_likes), len(fb_likes_cap)


fb_likes_cap.hist()


fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
fb_likes_cap2.equals(fb_likes_cap)


# # Masking DataFrame rows
# 

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['title_year'] >= 2010
c2 = movie['title_year'].isnull()
criteria = c1 | c2


movie.mask(criteria).head()


movie_mask = movie.mask(criteria).dropna(how='all')
movie_mask.head()


movie_boolean = movie[movie['title_year'] < 2010]
movie_boolean.head()


movie_mask.equals(movie_boolean)


movie_mask.shape == movie_boolean.shape


movie_mask.dtypes == movie_boolean.dtypes


from pandas.testing import assert_frame_equal
assert_frame_equal(movie_boolean, movie_mask, check_dtype=False)


get_ipython().run_line_magic('timeit', "movie.mask(criteria).dropna(how='all')")


get_ipython().run_line_magic('timeit', "movie[movie['title_year'] < 2010]")


# # Selecting with booleans, integer location and labels
# 

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['content_rating'] == 'G'
c2 = movie['imdb_score'] < 4
criteria = c1 & c2


movie_loc = movie.loc[criteria]
movie_loc.head()


movie_loc.equals(movie[criteria])


movie_iloc = movie.iloc[criteria]


movie_iloc = movie.iloc[criteria.values]


movie_iloc.equals(movie_loc)


movie.loc[criteria.values]


criteria_col = movie.dtypes == np.int64
criteria_col.head()


movie.loc[:, criteria_col].head()


movie.iloc[:, criteria_col.values].head()


cols = ['content_rating', 'imdb_score', 'title_year', 'gross']
movie.loc[criteria, cols].sort_values('imdb_score')


col_index = [movie.columns.get_loc(col) for col in cols]
col_index


movie.iloc[criteria.values, col_index].sort_values('imdb_score')


# ## How it works
# 

a = criteria.values
a[:5]


len(a), len(criteria)


# # There's more...
# 

movie.loc[[True, False, True], [True, False, False, True]]





# # Chapter 10: Time Series Analysis
# ## Recipes
# 
# * [Understanding the difference between Python and pandas date tools](#Understanding-the-difference-between-Python-and-pandas-date-tools)
# * [Slicing time series intelligently](#Slicing-time-series-intelligently)
# * [Using methods that only work with a DatetimeIndex](#Using-methods-that-only-work-with-a-DatetimeIndex)
# * [Counting the number of weekly crimes](#Counting-the-number-of-weekly-crimes)
# * [Aggregating weekly crime and traffic accidents separately](#Aggregating-weekly-crime-and-traffic-separately)
# * [Measuring crime by weekday and year](#Measuring-crime-by-weekday-and-year)
# * [Grouping with anonymous functions with a DatetimeIndex](#Grouping-with-anonymous-functions-with-a-DatetimeIndex)
# * [Grouping by a Timestamp and another column](#Grouping-by-a-Timestamp-and-another-column)
# * [Finding the last time crime was 20% lower with merge_asof](#Finding-the-last-time-crime-was-20%-lower-with-merge_asof)
# 

import pandas as pd
import numpy as np
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')


# # Understanding the difference between Python and pandas date tools
# 

date = datetime.date(year=2013, month=6, day=7)
time = datetime.time(hour=12, minute=30, second=19, microsecond=463198)
dt = datetime.datetime(year=2013, month=6, day=7, 
                       hour=12, minute=30, second=19, microsecond=463198)

print("date is ", date)
print("time is", time)
print("datetime is", dt)


td = datetime.timedelta(weeks=2, days=5, hours=10, minutes=20, 
                        seconds=6.73, milliseconds=99, microseconds=8)
print(td)


print('new date is', date + td)
print('new datetime is', dt + td)


time + td


pd.Timestamp(year=2012, month=12, day=21, hour=5, minute=10, second=8, microsecond=99)


pd.Timestamp('2016/1/10')


pd.Timestamp('2014-5/10')


pd.Timestamp('Jan 3, 2019 20:45.56')


pd.Timestamp('2016-01-05T05:34:43.123456789')


pd.Timestamp(500)


pd.Timestamp(5000, unit='D')


pd.to_datetime('2015-5-13')


pd.to_datetime('2015-13-5', dayfirst=True)


pd.Timestamp('Saturday September 30th, 2017')


pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm', format='Start Date: %b %d, %Y Start Time: %I:%M %p')


pd.to_datetime(100, unit='D', origin='2013-1-1')


s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit='D')


s = pd.Series(['12-5-2015', '14-1-2013', '20/12/2017', '40/23/2017'])
pd.to_datetime(s, dayfirst=True, errors='coerce')


pd.to_datetime(['Aug 3 1999 3:45:56', '10/31/2017'])


pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')


pd.Timedelta(days=5, minutes=7.34)


pd.Timedelta(100, unit='W')


pd.to_timedelta('5 dayz', errors='ignore')


pd.to_timedelta('67:15:45.454')


s = pd.Series([10, 100])
pd.to_timedelta(s, unit='s')


time_strings = ['2 days 24 minutes 89.67 seconds', '00:45:23.6']
pd.to_timedelta(time_strings)


pd.Timedelta('12 days 5 hours 3 minutes') * 2


pd.Timestamp('1/1/2017') + pd.Timedelta('12 days 5 hours 3 minutes') * 2


td1 = pd.to_timedelta([10, 100], unit='s')
td2 = pd.to_timedelta(['3 hours', '4 hours'])
td1 + td2


pd.Timedelta('12 days') / pd.Timedelta('3 days')


ts = pd.Timestamp('2016-10-1 4:23:23.9')


ts.ceil('h')


ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second


ts.dayofweek, ts.dayofyear, ts.daysinmonth


ts.to_pydatetime()


td = pd.Timedelta(125.8723, unit='h')
td


td.round('min')


td.components


td.total_seconds()


# ## There's more...
# 

date_string_list = ['Sep 30 1984'] * 10000


get_ipython().run_line_magic('timeit', "pd.to_datetime(date_string_list, format='%b %d %Y')")


get_ipython().run_line_magic('timeit', 'pd.to_datetime(date_string_list)')


# # Slicing time series intelligently
# 

crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


crime = crime.set_index('REPORTED_DATE')
crime.head()


pd.options.display.max_rows = 4


crime.loc['2016-05-12 16:45:00']


crime.loc['2016-05-12']


crime.loc['2016-05'].shape


crime.loc['2016'].shape


crime.loc['2016-05-12 03'].shape


crime.loc['Dec 2015'].sort_index()


crime.loc['2016 Sep, 15'].shape


crime.loc['21st October 2014 05'].shape


crime.loc['2015-3-4':'2016-1-1'].sort_index()


crime.loc['2015-3-4 22':'2016-1-1 23:45:00'].sort_index()


# ## How it works...
# 

mem_cat = crime.memory_usage().sum()
mem_obj = crime.astype({'OFFENSE_TYPE_ID':'object', 
                        'OFFENSE_CATEGORY_ID':'object', 
                        'NEIGHBORHOOD_ID':'object'}).memory_usage(deep=True)\
                                                    .sum()
mb = 2 ** 20
round(mem_cat / mb, 1), round(mem_obj / mb, 1)


crime.index[:2]


# ## There's more...
# 

get_ipython().run_line_magic('timeit', "crime.loc['2015-3-4':'2016-1-1']")


crime_sort = crime.sort_index()


get_ipython().run_line_magic('timeit', "crime_sort.loc['2015-3-4':'2016-1-1']")


pd.options.display.max_rows = 60


# # Using methods that only work with a DatetimeIndex
# 

crime = pd.read_hdf('data/crime.h5', 'crime').set_index('REPORTED_DATE')
print(type(crime.index))


crime.between_time('2:00', '5:00', include_end=False).head()


crime.at_time('5:47').head()


crime_sort = crime.sort_index()


pd.options.display.max_rows = 6


crime_sort.first(pd.offsets.MonthBegin(6))


crime_sort.first(pd.offsets.MonthEnd(6))


crime_sort.first(pd.offsets.MonthBegin(6, normalize=True))


crime_sort.loc[:'2012-06']


crime_sort.first('5D')


crime_sort.first('5B')


crime_sort.first('7W')


crime_sort.first('3QS')


# ## How it works...
# 

import datetime
crime.between_time(datetime.time(2,0), datetime.time(5,0), include_end=False)


first_date = crime_sort.index[0]
first_date


first_date + pd.offsets.MonthBegin(6)


first_date + pd.offsets.MonthEnd(6)


# ## There's more...
# 

dt = pd.Timestamp('2012-1-16 13:40')
dt + pd.DateOffset(months=1)


do = pd.DateOffset(years=2, months=5, days=3, hours=8, seconds=10)
pd.Timestamp('2012-1-22 03:22') + do


pd.options.display.max_rows=60


# # Counting the number of weekly crimes
# 

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()


crime_sort.resample('W')


weekly_crimes = crime_sort.resample('W').size()
weekly_crimes.head()


len(crime_sort.loc[:'2012-1-8'])


len(crime_sort.loc['2012-1-9':'2012-1-15'])


crime_sort.resample('W-THU').size().head()


weekly_crimes_gby = crime_sort.groupby(pd.Grouper(freq='W')).size()
weekly_crimes_gby.head()


weekly_crimes.equals(weekly_crimes_gby)


# ## How it works...
# 

r = crime_sort.resample('W')
resample_methods = [attr for attr in dir(r) if attr[0].islower()]
print(resample_methods)


# ## There's more...
# 

crime = pd.read_hdf('data/crime.h5', 'crime')
weekly_crimes2 = crime.resample('W', on='REPORTED_DATE').size()
weekly_crimes2.equals(weekly_crimes)


weekly_crimes_gby2 = crime.groupby(pd.Grouper(key='REPORTED_DATE', freq='W')).size()
weekly_crimes_gby2.equals(weekly_crimes_gby)


weekly_crimes.plot(figsize=(16,4), title='All Denver Crimes')


# # Aggregating weekly crime and traffic separately
# 

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()


crime_quarterly = crime_sort.resample('Q')['IS_CRIME', 'IS_TRAFFIC'].sum()
crime_quarterly.head()


crime_sort.resample('QS')['IS_CRIME', 'IS_TRAFFIC'].sum().head()


crime_sort.loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']].sum()


crime_quarterly2 = crime_sort.groupby(pd.Grouper(freq='Q'))['IS_CRIME', 'IS_TRAFFIC'].sum()
crime_quarterly2.equals(crime_quarterly)


plot_kwargs = dict(figsize=(16,4), 
                   color=['black', 'lightgrey'], 
                   title='Denver Crimes and Traffic Accidents')
crime_quarterly.plot(**plot_kwargs)


# ## How it works...
# 

crime_sort.resample('Q').sum().head()


crime_sort.resample('QS-MAR')['IS_CRIME', 'IS_TRAFFIC'].sum().head()


# ## There's more...
# 

crime_begin = crime_quarterly.iloc[0]
crime_begin


crime_quarterly.div(crime_begin)                .sub(1)                .round(2)                .plot(**plot_kwargs)


# # Measuring crime by weekday and year
# 

crime = pd.read_hdf('data/crime.h5', 'crime')
crime.head()


wd_counts = crime['REPORTED_DATE'].dt.weekday_name.value_counts()
wd_counts


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
        'Friday', 'Saturday', 'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'
wd_counts.reindex(days).plot(kind='barh', title=title)


title = 'Denver Crimes and Traffic Accidents per Year' 
crime['REPORTED_DATE'].dt.year.value_counts()                               .sort_index()                               .plot(kind='barh', title=title)


weekday = crime['REPORTED_DATE'].dt.weekday_name
year = crime['REPORTED_DATE'].dt.year

crime_wd_y = crime.groupby([year, weekday]).size()
crime_wd_y.head(10)


crime_table = crime_wd_y.rename_axis(['Year', 'Weekday']).unstack('Weekday')
crime_table


criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()


round(272 / 365, 3)


crime_pct = crime['REPORTED_DATE'].dt.dayofyear.le(272)                                   .groupby(year)                                   .mean()                                   .round(3)
crime_pct


crime_pct.loc[2012:2016].median()


crime_table.loc[2017] = crime_table.loc[2017].div(.748).astype('int')
crime_table = crime_table.reindex(columns=days)
crime_table


import seaborn as sns
sns.heatmap(crime_table, cmap='Greys')


denver_pop = pd.read_csv('data/denver_pop.csv', index_col='Year')
denver_pop


den_100k = denver_pop.div(100000).squeeze()
crime_table2 = crime_table.div(den_100k, axis='index').astype('int')
crime_table2


sns.heatmap(crime_table2, cmap='Greys')


# ## How it works...
# 

wd_counts.loc[days]


crime_table / den_100k


# ## There's more...
# 

ADJ_2017 = .748

def count_crime(df, offense_cat): 
    df = df[df['OFFENSE_CATEGORY_ID'] == offense_cat]
    weekday = df['REPORTED_DATE'].dt.weekday_name
    year = df['REPORTED_DATE'].dt.year
    
    ct = df.groupby([year, weekday]).size().unstack()
    ct.loc[2017] = ct.loc[2017].div(ADJ_2017).astype('int')
    
    pop = pd.read_csv('data/denver_pop.csv', index_col='Year')
    pop = pop.squeeze().div(100000)
    
    ct = ct.div(pop, axis=0).astype('int')
    ct = ct.reindex(columns=days)
    sns.heatmap(ct, cmap='Greys')
    return ct


count_crime(crime, 'auto-theft')


# # Grouping with anonymous functions with a DatetimeIndex
# 

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()


common_attrs = set(dir(crime_sort.index)) & set(dir(pd.Timestamp))
print([attr for attr in common_attrs if attr[0] != '_'])


crime_sort.index.weekday_name.value_counts()


crime_sort.groupby(lambda x: x.weekday_name)['IS_CRIME', 'IS_TRAFFIC'].sum()


funcs = [lambda x: x.round('2h').hour, lambda x: x.year]
cr_group = crime_sort.groupby(funcs)['IS_CRIME', 'IS_TRAFFIC'].sum()
cr_final = cr_group.unstack()
cr_final.style.highlight_max(color='lightgrey')


# ## There's more...
# 

cr_final.xs('IS_TRAFFIC', axis='columns', level=0).head()


cr_final.xs(2016, axis='columns', level=1).head()


# # Grouping by a Timestamp and another column
# 

employee = pd.read_csv('data/employee.csv', 
                       parse_dates=['JOB_DATE', 'HIRE_DATE'], 
                       index_col='HIRE_DATE')
employee.head()


employee.groupby('GENDER')['BASE_SALARY'].mean().round(-2)


employee.resample('10AS')['BASE_SALARY'].mean().round(-2)


sal_avg = employee.groupby('GENDER').resample('10AS')['BASE_SALARY'].mean().round(-2)
sal_avg


sal_avg.unstack('GENDER')


employee[employee['GENDER'] == 'Male'].index.min()


employee[employee['GENDER'] == 'Female'].index.min()


sal_avg2 = employee.groupby(['GENDER', pd.Grouper(freq='10AS')])['BASE_SALARY'].mean().round(-2)
sal_avg2


sal_final = sal_avg2.unstack('GENDER')
sal_final


# ## How it works...
# 

'resample' in dir(employee.groupby('GENDER'))


'groupby' in dir(employee.resample('10AS'))


# ## There's more...
# 

years = sal_final.index.year
years_right = years + 9
sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final


cuts = pd.cut(employee.index.year, bins=5, precision=0)
cuts.categories.values


employee.groupby([cuts, 'GENDER'])['BASE_SALARY'].mean().unstack('GENDER').round(-2)


# # Finding the last time crime was 20% lower with merge_asof
# 

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()


crime_sort.index.max()


crime_sort = crime_sort[:'2017-8']
crime_sort.index.max()


all_data = crime_sort.groupby([pd.Grouper(freq='M'), 'OFFENSE_CATEGORY_ID']).size()
all_data.head()


all_data = all_data.sort_values().reset_index(name='Total')
all_data.head()


goal = all_data[all_data['REPORTED_DATE'] == '2017-8-31'].reset_index(drop=True)
goal['Total_Goal'] = goal['Total'].mul(.8).astype(int)
goal.head()


pd.merge_asof(goal, all_data, left_on='Total_Goal', right_on='Total', 
              by='OFFENSE_CATEGORY_ID', suffixes=('_Current', '_Last'))


# ## There's more...
# 

pd.Period(year=2012, month=5, day=17, hour=14, minute=20, freq='T')


crime_sort.index.to_period('M')


ad_period = crime_sort.groupby([lambda x: x.to_period('M'), 
                                'OFFENSE_CATEGORY_ID']).size()
ad_period = ad_period.sort_values()                      .reset_index(name='Total')                      .rename(columns={'level_0':'REPORTED_DATE'})
ad_period.head()


cols = ['OFFENSE_CATEGORY_ID', 'Total']
all_data[cols].equals(ad_period[cols])


aug_2018 = pd.Period('2017-8', freq='M')
goal_period = ad_period[ad_period['REPORTED_DATE'] == aug_2018].reset_index(drop=True)
goal_period['Total_Goal'] = goal_period['Total'].mul(.8).astype(int)

pd.merge_asof(goal_period, ad_period, left_on='Total_Goal', right_on='Total', 
                  by='OFFENSE_CATEGORY_ID', suffixes=('_Current', '_Last')).head()





# # Chapter 2: Essential DataFrame Operations
# 
# ## Recipes
# * [Selecting multiple DataFrame columns](#Selecting-multiple-DataFrame-columns)
# * [Selecting columns with methods](#Selecting-columns-with-methods)
# * [Ordering column names sensibly](#Ordering-column-names-sensibly)
# * [Operating on the entire DataFrame](#Operating-on-the-entire-DataFrame)
# * [Chaining DataFrame methods together](#Chaining-DataFrame-methods-together)
# * [Working with operators on a DataFrame](#Working-with-operators-on-a-DataFrame)
# * [Comparing missing values](#Comparing-missing-values)
# * [Transposing the direction of a DataFrame operation](#Transposing-the-direction-of-a-DataFrame-operation)
# * [Determining college campus diversity](#Determining-college-campus-diversity)
# 

import pandas as pd
import numpy as np
pd.options.display.max_columns = 40


# # Selecting multiple DataFrame columns
# 

movie = pd.read_csv('data/movie.csv')
movie_actor_director = movie[['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']]
movie_actor_director.head()


movie[['director_name']].head()


movie['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']


# ## There's more...
# 

cols =['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
movie_actor_director = movie[cols]


# # Selecting columns with methods
# 

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie.get_dtype_counts()


movie.select_dtypes(include=['int']).head()


movie.select_dtypes(include=['number']).head()


movie.filter(like='facebook').head()


movie.filter(regex='\d').head()


movie.filter(items=['actor_1_name', 'asdf']).head()


# # Ordering column names sensibly
# 

movie = pd.read_csv('data/movie.csv')


movie.head()


movie.columns


disc_core = ['movie_title','title_year', 'content_rating','genres']
disc_people = ['director_name','actor_1_name', 'actor_2_name','actor_3_name']
disc_other = ['color','country','language','plot_keywords','movie_imdb_link']
cont_fb = ['director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes',
           'actor_3_facebook_likes', 'cast_total_facebook_likes', 'movie_facebook_likes']
cont_finance = ['budget','gross']
cont_num_reviews = ['num_voted_users','num_user_for_reviews', 'num_critic_for_reviews']
cont_other = ['imdb_score','duration', 'aspect_ratio', 'facenumber_in_poster']


new_col_order = disc_core + disc_people + disc_other +                     cont_fb + cont_finance + cont_num_reviews + cont_other
set(movie.columns) == set(new_col_order)


movie2 = movie[new_col_order]
movie2.head()


# # Operating on the entire DataFrame
# 

pd.options.display.max_rows = 8
movie = pd.read_csv('data/movie.csv')
movie.shape


movie.size


movie.ndim


len(movie)


movie.count()


movie.min()


movie.describe()


pd.options.display.max_rows = 10


movie.describe(percentiles=[.01, .3, .99])


pd.options.display.max_rows = 8


movie.isnull().sum()


# ## There's more...
# 

movie.min(skipna=False)


# # Chaining DataFrame methods together
# 

movie = pd.read_csv('data/movie.csv')
movie.isnull().head()


movie.isnull().sum().head()


movie.isnull().sum().sum()


movie.isnull().any().any()


# ## How it works...
# 

movie.isnull().get_dtype_counts()


# ## There's more...
# 

movie[['color', 'movie_title', 'color']].max()


movie.select_dtypes(['object']).fillna('').max()


# # Working with operators on a DataFrame
# 

# ## Getting ready...
# 

college = pd.read_csv('data/college.csv')
college + 5


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')


college == 'asdf'


college_ugds_.head()


college_ugds_.head() + .00501


(college_ugds_.head() + .00501) // .01


college_ugds_op_round = (college_ugds_ + .00501) // .01 / 100
college_ugds_op_round.head()


college_ugds_round = (college_ugds_ + .00001).round(2)
college_ugds_round.head()


.045 + .005


college_ugds_op_round.equals(college_ugds_round)


# ## There's more...
# 

college_ugds_op_round_methods = college_ugds_.add(.00501).floordiv(.01).div(100)


# # Comparing missing values
# 

np.nan == np.nan


None == None


5 > np.nan


np.nan > 5


5 != np.nan


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')


college_ugds_.head() == .0019


college_self_compare = college_ugds_ == college_ugds_
college_self_compare.head()


college_self_compare.all()


(college_ugds_ == np.nan).sum()


college_ugds_.isnull().sum()


from pandas.testing import assert_frame_equal


assert_frame_equal(college_ugds_, college_ugds_)


# ## There's more...
# 

college_ugds_.eq(.0019).head()


# # Transposing the direction of a DataFrame operation
# 

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')
college_ugds_.head()


college_ugds_.count()


college_ugds_.count(axis=0)


college_ugds_.count(axis='index')


college_ugds_.count(axis='columns').head()


college_ugds_.sum(axis='columns').head()


college_ugds_.median(axis='index')


# ## There's more
# 

college_ugds_cumsum = college_ugds_.cumsum(axis=1)
college_ugds_cumsum.head()


college_ugds_cumsum.sort_values('UGDS_HISP', ascending=False)


# # Determining college campus diversity
# 

pd.read_csv('data/college_diversity.csv', index_col='School')


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')
college_ugds_.head()


college_ugds_.isnull().sum(axis=1).sort_values(ascending=False).head()


college_ugds_ = college_ugds_.dropna(how='all')


college_ugds_.isnull().sum()


college_ugds_.ge(.15).head()


diversity_metric = college_ugds_.ge(.15).sum(axis='columns')
diversity_metric.head()


diversity_metric.value_counts()


diversity_metric.sort_values(ascending=False).head()


college_ugds_.loc[['Regency Beauty Institute-Austin', 
                          'Central Texas Beauty College-Temple']]


us_news_top = ['Rutgers University-Newark', 
               'Andrews University', 
               'Stanford University', 
               'University of Houston',
               'University of Nevada-Las Vegas']


diversity_metric.loc[us_news_top]


# ## There's more...
# 

college_ugds_.max(axis=1).sort_values(ascending=False).head(10)


college_ugds_.loc['Talmudical Seminary Oholei Torah']


(college_ugds_ > .01).all(axis=1).any()


