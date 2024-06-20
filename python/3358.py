# ## Introduction to Pandas - Pyladies Berlin
# 

import pandas as pd


import numpy as np


from csv import reader


# ### Campaign Finance
# 
# Today we will take a quick look at the 2016 election in the states and compare some campaign finance data on it. We'll use this as a way to get started with [Pandas](http://pandas.pydata.org/), a powerful data analysis library in Python.
# 
# The data files we will use are in the [data folder in this repository](https://github.com/kjam/random_hackery) or available for download on the [FEC site](http://www.fec.gov/finance/disclosure/ftpdet.shtml#a2015_2016). 
# 
# To begin, we need to extract the headers for the file, as they are not included in the data dumps.
# 

cand_header = [r for r in reader(open('data/cn_header_file.csv', 'r'))]


cand_header


# We can then pass this first list item along with the data to the [read_csv method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html). We are using the sep keyword argument to define how our file is separated between fields.
# 
# This will create a [Pandas Dataframe](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).
# 

candidates = pd.read_csv('data/cn.txt', names=cand_header[0], sep='|')


candidates.head()


# To take a look at columns, we can use the dataframe like a dictionary and pass the column name as the key. In return we get what is called a [Pandas Series](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html).
# 

candidates['CAND_NAME']


# We can also use slicing and selecting to review either a Series or a DataFrame. There are a few different methods available. 
# 
# We can use a boolean selector which returns a truth Series or DataFrame.
# 

candidates['CAND_ELECTION_YR'] == 2016


candidates[candidates['CAND_ELECTION_YR'] == 2016]


# We can also simply select with indexing or slices.
# 
# Both [loc](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html) and [iloc](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html) methods can be very useful for selecting rows.
# 

candidates[candidates['CAND_ELECTION_YR'] == 2016][['CAND_ID', 'CAND_NAME']].head()


candidates.shape


candidates.loc[6940:]


candidates.iloc[2]


# Because Pandas DataFrames are built upon NumPy datatypes, it's always a good idea to test your datatypes and see what they are. I usually do this early on so I can fix any bad imports.
# 

candidates.dtypes


# Remember: objects should really only represent strings, arrays or dicts. Everything else should be an integer or float or boolean or datetime.
# 
# These look okay since we actually have a lot of string data in this set.
# 
# ## How might I find Donald Trump's data?
# 
# 
# 
# 
# 
# 

candidates[candidates['CAND_NAME'] == 'TRUMP, DONALD']


# There are some cool string methods avialable, let's try one of those
# 

candidates[candidates['CAND_NAME'].str.contains('TRUMP')]


# Hm, that's odd. We must have some missing candidate names. Lucky for us, we can use pandas [notnull](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.notnull.html) to skip over the rows that have null values. This will allow our string method to run on only rows where a candidate name exists.
# 
# Remember: with Python the first False statement means the second statement won't need to run :)
# 

candidates[candidates['CAND_NAME'].notnull() & candidates['CAND_NAME'].str.contains('TRUMP')]


# And we've found him. As well as some of his critics. 
# 
# Now that we know we can search and slice with Pandas, let's try merging this dataset with some actual campaign finance data. Onward!! :)
# 

donations_header = [r for r in reader(open('data/indiv_header_file.csv', 'r'))]


donations_header[0]


donations = pd.read_csv('data/itcont.txt', names=donations_header[0], sep='|')


donations.head()


donations.dtypes


# These look about right. We can also use [describe](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html) to see some statistical representations of our data.
# 

donations.describe()


# That's not a super helpful list of numbers, let's take a look at the transaction amount column more specifically. We can run many different statistical functions just on the Series itself.
# 

donations['TRANSACTION_AMT'].mean()


donations['TRANSACTION_AMT'].min()


donations['TRANSACTION_AMT'].max()


donations['TRANSACTION_AMT'].median()


# We can also load [matplotlib](http://matplotlib.org/) in our session by using IPython magic command %pylab inline
# 
# Then plotting will be available within our notebook.
# 

get_ipython().magic('pylab inline')


hist(donations['TRANSACTION_AMT'])


# So we likely aren't going to see good distribution until we remove outliers. As we can see from the histogram the *vast* majority of donations are in a small section, but the outliers (both negative and positive) are making our histogram unreadable.
# 
# Since we might just want to look at one or two candidates and see the distribution there, let's first combine our dataframes and then look for outliers. This helps us tell more of the story as well, if we find that an overwhelming number of outliers for one candidate exist.
# 
# First, let's see how many of our candidates have a major political committee listed. We can use [shape](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) to take a look at how many rows we have in the resulting DataFrame.
# 

candidates[candidates['CAND_PCC'].notnull()].shape


# Not too bad! So let's now merge that in with the individual dataframe, and see what results we get when using a few different joins.
# 

donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'))


# So, not a great sign. It looks like our committee IDs don't properly match. Let's try joining on the candidates table.
# 

donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'), how='right')


# This is better, but now maybe I'm just curious about the candidates with donations, not the other ones without. We can use an inner join to do so.
# 

donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'), how='inner')


# This is looking better, let's save the output to a new combined dataframe.
# 

cand_donations = donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'), how='inner')


cand_donations.describe()


hist(cand_donations['TRANSACTION_AMT'])


cand_donations['TRANSACTION_AMT'].max()


# Wow! Let's just check candidates with donations over a million USD.
# 
# We can utilize [value_counts](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html) to give us a nice stacked chart.
# 

cand_donations[cand_donations['TRANSACTION_AMT'] > 1000000]


cand_donations[cand_donations['TRANSACTION_AMT'] > 1000000]['CAND_NAME'].value_counts()


# And what about smaller donations?

cand_donations[cand_donations['TRANSACTION_AMT'] < 200]['CAND_NAME'].value_counts()


# Since we have quite a lot of candidate data in here, I want to whittle it down to this year's election, maybe to those who have a significant amount of donors. Let's first just get it down to this year. I can't remember the column name of the year, but I can check with the columns attribute, which will return the Column index.
# 

cand_donations.columns


cand_donations = cand_donations[cand_donations['CAND_ELECTION_YR'] == 2016]


# Next, we can start grouping by candidate.
# 
# A Pandas [groupby object](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) operates differently than DataFrames. You can easily run aggregate groupings on them, of varying complexity. The results are a dataframe with the grouping as the index.
# 

grouped = cand_donations.groupby('CAND_NAME')


grouped.sum()


grouped.agg({'TRANSACTION_AMT': [np.sum, np.mean], 'NAME': lambda x: len(set(x))})


# You can also use [transform](http://pandas.pydata.org/pandas-docs/stable/groupby.html#transformation) to modify a dataframe based on a groupby (you can even do this in place!)
# 

cand_donations['unique_donors'] = cand_donations.groupby('CAND_NAME')['NAME'].transform(lambda x: 
                                                                                        len(set(x)))


cand_donations['unique_donors'].mean()


cand_donations['unique_donors'].median()


sign_cand_donations = cand_donations[cand_donations['unique_donors'] > cand_donations['unique_donors'].mean()]


sign_cand_donations.shape


sign_cand_donations.groupby('CAND_NAME').sum()


# Wait! What happened to Trump? Does this mean Trump has less than the mean value for candidates just in terms of numbers of donors? Let's see how he compares. 
# 

cand_donations[cand_donations['CAND_NAME'].str.contains('TRUMP')]['unique_donors']


cand_donations[cand_donations['CAND_NAME'].str.contains('TRUMP')].describe()


# Let's add him back in here, since we likely want to compare the main contenders.
# 

sign_cand_donations = sign_cand_donations.append(cand_donations[cand_donations['CAND_NAME'].str.contains('TRUMP')])


sign_cand_donations.groupby('CAND_NAME').sum()['TRANSACTION_AMT']


sign_cand_donations.groupby('CAND_NAME').min()['unique_donors'].sort_values()


# So we can already see some trends emerging. Bernie Sanders has the most donations, Hillary Clinton has the most money from individual donors and Donald Trump is significantly lacking in both in comparison. In fact, if you use [candidate committe details](http://www.fec.gov/fecviewer/CandidateCommitteeDetail.do) you can see he has loaned himself more than $43M to run his campaign so far.
# 

# ## Now it's your turn!
# 
# 
# Possible Next Tasks:
# ----------------------
# 
# - Make a histogram of each candidates donations, do you see any trends?
# - Use [standard deviations](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.std.html) to remove or focus on outliers from the dataset.
# - Which states have the most donors?
# - Which occupations or employers are most represented by which campaigns?
# - Use below committee data to find other committees for the main candidates (i.e. what other committees support Bernie, Trump and Clinton)? or even, how many people donated to which committees this year? (like the NRA or Exxon or AT&T?)




# Bonus: Import the committee data and see if you can join with our original donations file, can you find what other commmittees have large and small donation groupings? Here's the import to get started.
# 

comm_header = [r for r in reader(open('data/cm_header_file.csv', 'r'))]


committees = pd.read_csv('data/cm.txt', names=comm_header[0], sep='|')


committees.head()





from dask import array, dataframe, bag


# ### Arrays: Parallelized NumPy
# 

da = array.random.binomial(100, .3, 1000, chunks=(100))


da.size


da[:2]


da[:2].compute()


da.npartitions


da[:400].chunks


da.min().visualize()


da.min().compute()


# ### Dataframe: Parallelized Pandas
# 

df = dataframe.read_csv('../data/speakers.csv')


df.head()


df.dtypes


df.npartitions


df.twitter.loc[4]


df.twitter.loc[4].compute()


df.groupby(['nationality', 'curr_pol_group_abbr'])


df.groupby(['nationality', 'curr_pol_group_abbr']).sum().visualize()


df2 = df.set_index('speaker_id')


df2.npartitions


df.speaker_id.compute().values


partitions = sorted(df.speaker_id.compute().values)[::50]


partitions


df3 = df.set_partition('speaker_id', partitions)


df3.npartitions


df3.divisions


df3.groupby(['nationality', 'curr_pol_group_abbr']).sum().visualize()


df3.groupby(['nationality', 'curr_pol_group_abbr']).sum().compute()


# ### Dask Bags: Parallelized objects
# 

db = bag.read_text('../data/europarl_speech_text.txt')


db.take(2)


db.filter(lambda x: 'Czech' in x)


db.filter(lambda x: 'Czech' in x).take(3)


db.filter(lambda x: 'Czech' in x).str.split().concat().frequencies().topk(100)


db.filter(lambda x: 'Czech' in x).str.split().concat().frequencies().topk(100).compute()





from distributed import Client
from time import sleep
import random
import math


cl = Client()


cl


# ### Now you should have a scheduler here: http://localhost:8787/status
# 

def nsleep(num_secs):
    sleep(num_secs)
    return num_secs


def get_rand_secs(max_int):
    return random.randint(0, max_int)


def do_some_math_with_errors(number):
    if number > 30:
        return math.log(number)
    elif number > 15:
        return round(number / 3.0)
    elif number >= 5:
        return math.floor(number / 1.5)
    elif number <= 2:
        return number / 0
    return number ** 2


def do_some_math(number):
    if number > 30:
        return math.log(number)
    elif number > 15:
        return round(number / 3.0)
    elif number >= 5:
        return math.floor(number / 1.5)
    elif number <= 2:
        return number 
    return number ** 2


random_secs = cl.map(get_rand_secs, range(200))


random_secs


cl.gather(random_secs)[:10]


random_math = cl.map(do_some_math_with_errors, random_secs)


excs = [(e.traceback(), e.exception()) for e in random_math if e.exception()]


excs[0]


random_math = cl.map(do_some_math, random_secs)


random_sleeps = cl.map(nsleep, random_math)


random_sleeps


sum_sleep = cl.submit(sum, random_sleeps)


sum_sleep


sum_sleep.result()





