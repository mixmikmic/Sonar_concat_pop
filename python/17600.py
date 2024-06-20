# Python Class used in harvesting and processing data collected and retrived from `Twitter`, `Github` and `Meetup`.
# 

from pymongo import MongoClient as MCli





class IO_Mongo(object):
    """Connect to the mongo server on localhost at port 27017."""
    conn={'host':'localhost', 'ip':'27017'}


    # Initialize the class with client connection, the database (i.e. twtr_db), and the collection (i.e. twtr_coll)
    def __init__(self, db='twtr_db', coll='twtr_coll', **conn):
        """Connect to the MonfoDB server"""
        self.client = MCli(**conn)
        self.db = self.client[db]
        self.coll = self.db[coll]


    # The `save` method inserts new records in the pre_initialized collection and database
    def save(self, data):
        """ Insert data to collection in db. """
        return self.coll.insert(data)
    
    
    # The `load` method allows the retrieval of specific records
    def load(self, return_cursor=False, criteria=None, projection=None):
        """ The `load` method allows the retrieval of specific records according to criteria and projection. 
            In case of large amount of data, it returns a cursor.
        """
        if criteria is None:
            criteria = {}
        
        # Find record according to some criteria.
        if projection is None:
            cursor = self.coll.find(criteria)
        else:
            cursor = self.coll.find(criteria, projection)
        
        # Return a cursor for large amount of data
        if return_cursor:
            return cursor
        else:
            return [item for item in cursor]
        
        


f = IO_Mongo()
f.load()





# # Connect and harvest data from Twitter
# 




import twitter
import urlparse
import pandas as pd

# parallel print
from pprint import pprint as pp


# Load the twitter API keys
twitter_tokens = pd.read_csv("../twitter_tokens.csv")
twitter_tokens.keys()



class TwitterAPI(object):
    """
        TwitterAPI class allows the Connection to Twitter via OAuth
        once you have registered with Twitter and receive the
        necessary credentials.
    """
    # Initialize key variables and get the twitter credentials
    def __init__(self):
        consumer_key = twitter_tokens.values.flatten()[0]
        consumer_secret = twitter_tokens.values.flatten()[1]
        access_token = twitter_tokens.values.flatten()[2]
        access_secret = twitter_tokens.values.flatten()[3]
        
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        
    # Authenticate credentials with Twitter using OAuth
        self.auth = twitter.oauth.OAuth(access_token, access_secret, 
                                        consumer_key, consumer_secret)
        
        
    # Create registered Twitter API
        self.api = twitter.Twitter(auth=self.auth)
        
        
    # Search Twitter with query q (i.e "ApacheSpark") and max result
    def searchTwitter(self, q, max_res=10, **kwargs):
        search_results = self.api.search.tweets(q=q, count=10, **kwargs)
        statuses = search_results['statuses']
        max_results = min(1000, max_res)
        
        for _ in range(10):
            try:
                next_results = search_results['search_metadata']['next_results']
            except KeyError as e:
                break
            
            next_results = urlparse.parse_qsl(next_results[1:])
            kwargs = dict(next_results)
            
            search_results = self.api.search.tweets(**kwargs)
            statuses += search_results['statuses']
            
            if len(statuses) > max_results:
                break
            
        return statuses
    
    
    
    # Parse tweets as it is collected to extract ID, creation date, userID, tweet text
    def parseTweets(self, statuses):
        tweetx = [(status['id'],
                   status['created_at'],
                   status['user']['id'],
                   status['user']['name'],
                   url['expanded_url'],
                   status['text']) 
                    for status in statuses 
                      for url in status['entities']['urls']
                 ]
        return tweetx
    


# Instantiate the class with the required authentication
obj = TwitterAPI()


# Run a query on the search tern
twtx = obj.searchTwitter("ApacheSpark")

# Parse the tweets
parsed_tweetx = obj.parseTweets(twtx)


# Display output of parsed tweets
print("Lenth of parsed tweets: {} \n\n".format(len(parsed_tweetx)))

# Serialize the data into CSV
csv_fields = ['id', 'created_at', 'user_id', 'user_name', 'tweet_text', 'url']
tweets_2frames = pd.DataFrame(parsed_tweetx, columns=csv_fields)
tweets_2frames.to_csv("tweets.csv", encoding='utf-8')

# Display first 3 rows
tweets_2frames.ix[:2]



tweets_2frames.url[2]





import pandas as pd
from pprint import pprint
from github import Github


# * Example Call: `Show USER PROFILE in URL`
# > `https://api.github.com/users/RichardAfolabi` (in browser) <br>
# > `curl -i https://api.github.com/users/RichardAfolabi` (in shell)
# > All API calls: `https://api.github.com` (in browser) or `curl -i https://api.github.com` (in shell)
# 
# Load Github token (Created at https://github.com/settings/tokens)
# 

# Load Github token 
github_token = pd.read_csv("../github_token.csv")
ACCESS_TOKEN = github_token.values.flatten()[0]


# ### Dig deeper into `User = apache` and `Repo = spark`
# 

# Set user and repo of interest
USER = 'apache'
REPO = 'spark'

githubx = Github(ACCESS_TOKEN, per_page=100)
user = githubx.get_user(USER)
repo = user.get_repo(REPO)


# Retrieve key facts from the user - Apache. 
# repos_apache = [repo.name for repo in githubx.get_user('apache').get_repos()]


# Retrieve key facts from the user - Apache.
# apache_repos = [repo.name for repo in repos]
repos_apache = [rp.name for rp in githubx.get_user('apache').get_repos()]
print("\n User '{}' has {} repos \n".format(USER, len(apache_repos)))


# Check if project Spark exists
'spark' in apache_repos


# ### Retrieve key facts from the Spark repository
# 

print("Programming Languages used under the `Spark` project are: \n")
pp(rp.get_languages())


stargazers = [s for s in repo.get_stargazers()]
print("Number of Stargazers is {}".format(len(stargazers)))


# Retrieve a few key participants of the wide Spark GitHub repository network.
# The first stargazer is Matei Zaharia, the cofounder of the Spark project when he was doing his PhD in Berkeley.

[stargazers[i].login for i in range(0,20)]





import json
import mimeparse
import requests
import urllib
import pandas as pd
from pprint import pprint as pp


MEETUP_API_HOST = 'https://api.meetup.com'
EVENTS_URL = MEETUP_API_HOST + '/2/events.json'
MEMBERS_URL = MEETUP_API_HOST + '/2/members.json'
GROUPS_URL = MEETUP_API_HOST + '/2/groups.json'
RSVPS_URL = MEETUP_API_HOST + '/2/rsvps.json'
PHOTOS_URL = MEETUP_API_HOST + '/2/photos.json'
GROUP_URLNAME = 'London-Machine-Learning-Meetup'

# GROUP_URLNAME = 'Data-Science-London'


# Load Meetup API Key
meetup_api_key = pd.read_csv("../meetup_token.csv")


class MeetupAPI(object):
    """ Retreives information about meetup.com
    """
    def __init__(self, api_key, num_past_events=10, http_timeout=1, http_retries=2):
        """ Create new instance of meetup """
        self._api_key = api_key
        self._http_timeout = http_timeout
        self._http_retries = http_retries
        self._num_past_events = num_past_events

    
    def get_past_events(self):
        """ Get past meetup events for a given meetup group """
        params = {'key': self._api_key,
                  'group_urlname': GROUP_URLNAME,
                  'status': 'past',
                  'desc': 'true'}
        if self._num_past_events:
            params['page'] = str(self._num_past_events)
            
        query = urllib.urlencode(params)
        url = '{0}?{1}'.format(EVENTS_URL, query)
        response = requests.get(url, timeout=self._http_timeout)
        data = response.json()['results']
        return data
    
    

    def get_members(self):
        """ Get meetup members for a given meetup group """
        params = {'key': self._api_key,
                  'group_urlname': GROUP_URLNAME,
                  'offset': '0',
                  'format': 'json',
                  'page': '100',
                  'order':'name'}
        query = urllib.urlencode(params)
        url = '{0}?{1}'.format(MEMBERS_URL, query)
        response = requests.get(url, timeout=self._http_timeout)
        data = response.json()['results']
        return data
    
    
    def get_groups_by_member(self, member_id='38680722'):
        """Get meetup groups for a given meetup member """
        params = {'key': self._api_key,
                  'member_id': member_id,
                  'offset': '0',
                  'format':'json',
                  'page':'100',
                  'order':'id'}
        query = urllib.urlencode(params)
        url = '{0}?{1}'.format(GROUPS_URL, query)
        response = requests.get(url, timeout=self._http_timeout)
        data = response.json()['results']
        return data
    
    


m = MeetupAPI(api_key=meetup_api_key.values.flatten()[0])


last_meetups = m.get_past_events()
pp(last_meetups[1])


# ### Get information about the MeetUp members
# 

members = m.get_members()
members





# # Spark Tutorials - Learning Apache Sparks
# 
# **`Apache Spark`**, is a framework for large-scale data processing. Many traditional frameworks were designed to be run on a single computer. However, many datasets today are too large to be stored on a single computer, and even when a dataset can be stored on one computer, it can often be processed much more quickly using multiple computers.
# 

# ## `Transformation`
# `map(), mapPartitions(), mapPartitionsWithIndex(), filter(), flatMap(), reduceByKey(), groupByKey()`
# 

# Create RDD and subtract 1 from each number then find max
dataRDD = sc.parallelize(xrange(1,21))

# Let's see how many partitions the RDD will be split into using the getNumPartitions()
dataRDD.getNumPartitions()


dataRDD.map(lambda x: x - 1).max()
dataRDD.toDebugString()


# Find the even numbers
print(dataRDD.getNumPartitions())

# Find even numbers
evenRDD = dataRDD.filter(lambda x: x % 2 == 0)

# Reduce by adding up all values in the RDD
print(evenRDD.reduce(lambda x, y: x + y))


# Use Python add function to sum
from operator import add
print(evenRDD.reduce(add))





# # `Action`
# `first(), take(), takeSample(), takeOrdered(), collect(), count(), countByValue(), reduce(), top()`
# 

# Take first n values
evenRDD.take(10)


# Count distinct values in RDD and return dictionary of values and counts
evenRDD.countByValue()


# ## `reduceByKey()`, `combineByKey()` and `foldByKey()` are better than `groupByKey()`!
# 

pairRDD = sc.parallelize([('a', 1), ('a', 2), ('b', 1)])

# mapValues only used to improve format for printing
print pairRDD.groupByKey().mapValues(lambda x: list(x)).collect()

# Different ways to sum by key
print pairRDD.groupByKey().map(lambda (k, v): (k, sum(v))).collect()

# Using mapValues, which is recommended when the key doesn't change
print pairRDD.groupByKey().mapValues(lambda x: sum(x)).collect()

# reduceByKey is more efficient / scalable
print pairRDD.reduceByKey(add).collect()


# ## `mapPartitions()`
# The mapPartitions() transformation uses a function that takes in an iterator (to the items in that specific partition) and returns an iterator. The function is applied on a partition by partition basis.
# 

# mapPartitions takes a function that takes an iterator and returns an iterator
print wordsRDD.collect()

itemsRDD = wordsRDD.mapPartitions(lambda iterator: [','.join(iterator)])

print itemsRDD.collect()


# ## `mapPartitionsWithIndex()`
# The mapPartitionsWithIndex() transformation uses a function that takes in a partition index (think of this like the partition number) and an iterator (to the items in that specific partition). For every partition (index, iterator) pair, the function returns a tuple of the same partition index number and an iterator of the transformed items in that partition.
# 

itemsByPartRDD = wordsRDD.mapPartitionsWithIndex(lambda index, iterator: [(index, list(iterator))])

# We can see that three of the (partitions) workers have one element and the fourth worker has two
# elements, although things may not bode well for the rat...
print itemsByPartRDD.collect()

# Rerun without returning a list (acts more like flatMap)
itemsByPartRDD = wordsRDD.mapPartitionsWithIndex(lambda index, iterator: (index, list(iterator)))

print itemsByPartRDD.collect()


# # `Others`
# `cache(), unpersist(), id(), setName()`
# 

def brokenTen(value):
    """Incorrect implementation of the ten function.

    Note:
        The `if` statement checks an undefined variable `val` instead of `value`.

    Args:
        value (int): A number.

    Returns:
        bool: Whether `value` is less than ten.

    Raises:
        NameError: The function references `val`, which is not available in the local or global
            namespace, so a `NameError` is raised.
    """
#     if (val < 10):
    if (value < 10):
        return True
    else:
        return False

brokenRDD = dataRDD.filter(brokenTen)


brokenRDD.collect()


# # Word Count & Related Processing Lab
# 

# ### Word Count
# 

wordslist = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordslist)
print(type(wordsRDD))


# ### Pluralize words
# 

# Pluralize each word
pluralwords = wordsRDD.map(lambda w: w +'s').collect()
pluralwords


# ### Length of words
# 

# Find length of each word
pluralRDD = sc.parallelize(pluralwords)
pluralRDD.map(len).collect()


# ### Word Count, Key-Value Pair
# 

# CountByValue()
pluralRDD.countByValue().items()


# Lambda with CountByKey()
pluralRDD.map(lambda d: (d,1)).countByKey().items()


# Count with Lambda with reduceByKey()
newPluralRDD = pluralRDD.map(lambda d: (d,1))
print(newPluralRDD.collect())

newPluralRDD.reduceByKey(add).collect()


# ### Unique Words in RDD
# 

print(pluralRDD.collect())
pluralRDD.distinct().collect()


# ### Find `mean()`
# 

total = pluralRDD.count()
size = float(pluralRDD.distinct().count())
round(total/size, 2)


# ### Apply WordCount to a File 
# 


def wordCount(wordsListRDD):
    """ Inputs word List RDD and outputs Key value pair count of the words."""
    return wordsListRDD.countByValue().items()

wordCount(wordsRDD)



import re
f = open("alice.txt", 'r')

pattern = re.compile(r"[.,\[\]\"'*!?`_\s();-]+")
wordsList = [re.sub(pattern, '', sents).lower() for sents in f.read().split()]
wordsList = filter(None, wordsList)



wordsFileRDD = sc.parallelize(wordsList)
p_wordfile = wordCount(wordsFileRDD)

# Print top 30 words
sorted(p_wordfile, key=lambda tup:tup[1], reverse=True)[:30]





import datetime
import os, re, sys
import pandas as pd
from pyspark.sql import Row
# from test_helper import Test


# Use Pandas to read sample and inspect
logDataFile = "../_datasets_downloads/NASA_access_log_Aug95.gz"
logfile = pd.read_table(logDataFile, header=None, encoding='utf-8')
# test_log = logfile[0][0]
# test_log
logfile


APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)" (\d{3}) (\S+)'





# We create a regular expression pattern to extract the nine fields of the log line using the Python regular expression search function. The function returns a pair consisting of a Row object and 1. If the log line fails to match the regular expression, the function returns a pair consisting of the log line string and 0. A '-' value in the content size field is cleaned up by substituting it with 0.
# 

# Search pattern and extract
match = re.search(APACHE_ACCESS_LOG_PATTERN, test_log)
print(match.group(9))

match.group(6)


#  Convert sample `logfile` entry to date time format

month_map = {'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
    'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12}


def parse_apache_time(s):
    """ Convert date entry in log file to datetime format"""
    return datetime.datetime(int(s[7:11]),
                             month_map[s[3:6]],
                             int(s[0:2]),
                             int(s[12:14]),
                             int(s[15:17]),
                             int(s[18:20]))


# Test
parse_apache_time(match.groups()[3])



def parseApacheLogLine(logline):
    """ Parse each line entry of the log file """
    match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)
    
    # If no match is found, return entry and zero
    if match is None:
        return(logline, 0)
    
    # If field_size is empty, initialize to zero
    field_size = match.group(9) 
    if field_size == '_':
        size = long(0)
    else:
        size = long(match.group(9))
    return (Row(
            host = match.group(1),
            client_identd = match.group(2),
            user_id = match.group(3),
            date_time = parse_apache_time(match.group(4)),
            method = match.group(5),
            endpoint = match.group(6),
            protocol = match.group(7),
            response_code = int(match.group(8)),
            content_size = size
            ), 1)


parseApacheLogLine(test_log)


# ## Configuration and Initial RDD Creation
# 

# We first load the text file using sc.textFile(filename) 
# to convert each line of the file into an element in an RDD.

log_fileRDD = sc.textFile("../_datasets_downloads/NASA_access_log_Jul95")
log_fileRDD.take(2)


# ### Test function `parseApacheLogLine` - takes `str` input
# 

# Next, we use map(parseApacheLogLine) to apply the parse function to each element 
# (that is, a line from the log file) in the RDD and turn each line into a pair Row object.
log_fileRDD = log_fileRDD.map(parseApacheLogLine)
log_fileRDD.cache()


def parseLogs():
    """ Read and parse log file. """
    parsed_logs = (sc.textFile(logDataFile).map(parseApacheLogLine).cache())
    
    access_logs = (parsed_logs.filter(lambda s: s[1] == 1).map(lambda s: s[0]).cache())
    
    failed_logs = (parsed_logs.filter(lambda s: s[1] == 0).map(lambda s: s[0]))
    
    failed_logs_count = failed_logs.count()
    
    if failed_logs_count > 0:
        print("Number of invalud logline" % failed_logs_count())
        for line in failed_logs.take(20):
            print("Invalid login: %s" %line)
    
    print("Lines red : %d, \n Parsed successfully: %d \n Failed parse: %d"% (parsed_logs.count(),
                                                                            access_logs.count(),
                                                                            failed_logs.count()))
    return parsed_logs, access_logs, failed_logs



parsed_logs, access_logs, failed_logs = parseLogs()





# Odo migrates between many formats. `CSV <=> JSON <=> HDFS <=> SQL <=> etc`
# 

from odo import odo
import pandas as pd

csv_dataset = "/Users/RichardAfolabi/myGitHub/Data_Science_Harvard/2014_data/countries.csv"


# Convert from CSV to Dataframe
df = odo(csv_dataset, pd.DataFrame)
df.head()


# Convert from CSV to List
odo(csv_dataset, list)[:5]


# Convert from dataframe to JSON and dump in directory
# odo(df, 'json_dataset.json')


# Use Spark SQL Context to read json file
sp_df = sqlContext.read.json('json_dataset.json').take(5)
sp_df


# Create Spark SchemaRDD using Odo
data = [('Alice', 300.0), ('Bob', 200.0), ('Donatello', -100.0)]
# odo(data, sqlContext.sql, dshape='var * {Country: string, Region: string}')





# # Analyzing Orange Telecoms Customer Churn Dataset
# 
# Predicting which customer is likely to cancel subscription to a service. It has become a common practise across banks, ISPs, insurance firms and credit card companies.
# 
# ### Run `Apache pyspark` from the terminal:
# Load `pyspark` with the [Spark-CSV](http://spark-packages.org/package/databricks/spark-csv) package.
# 
# > IPYTHON_OPTS="notebook" ~/path_to_pyspark --packages com.databricks:spark-csv_2.11:1.4.0
# 

get_ipython().magic('matplotlib inline')

import pandas as pd
import seaborn as sns
import plotly.plotly as py
import cufflinks as cf


# ## Fetching and Importing Churn Dataset
# For this tutorial, we'll be using the Orange Telecoms Churn Dataset. It consists of cleaned customer activity data (features), along with a churn label specifying whether the customer canceled their subscription or not. The data can be fetched from BigML's S3 bucket, [churn-80](https://bml-data.s3.amazonaws.com/churn-bigml-80.csv) and [churn-20](https://bml-data.s3.amazonaws.com/churn-bigml-20.csv). The two sets are from the same batch, but have been split by an 80/20 ratio. We'll use the larger set for training and cross-validation purposes, and the smaller set for final testing and model performance evaluation. The two data sets have been included in this repository for convenience.
# 

# Import dataset with spark CSV package
orange_sprk_df = sqlContext.read.load("../_datasets_downloads/churn-bigml-80.csv",
                                format='com.databricks.spark.csv',
                                header='true',
                                inferschema='true')

orange_final_dataset = sqlContext.read.load("../_datasets_downloads/churn-bigml-20.csv",
                                format='com.databricks.spark.csv',
                                header='true',
                                inferschema='true')

# Print Dataframe Schema. That's DataFrame = Dataset[Row]
orange_sprk_df.cache()
orange_sprk_df.printSchema()


# Display first 5 Rows or Spark Dataset
orange_sprk_df.toPandas().head()


# ## Summary Statistics
# Show the summary statistic for the dataset. The describe function compute statistics ONLY for columns having numeric data types. So we can extract the numeric indexes from the resulting pandas dataframe.
# 

num_set = orange_sprk_df.describe().toPandas().transpose()
num_set.head()


# Display the numeric index
num_set.index.values


# Drop the `summary` and `Area code` columns and slice the dataframe using the numeric index.
new_df = orange_sprk_df.toPandas()
new_df = new_df[num_set.index.drop(['summary','Area code'])]
new_df.head()


axs = pd.scatter_matrix(new_df, figsize=(18,18))

# Rotate axis labels and remove axis ticks
n = len(new_df.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
    


# Several of the numerical data are very correlated. (**`Total day minutes` and `Total day charge`**), (**`Total eve minutes` and `Total eve charge`**), (**`Total night minutes` and `Total night charge`**) and lastly (**`Total intl minutes` and `Total intl charge`**) are alo correlated. We only have to select one of them
# 

binary_map = {'Yes':1.0, 'No':0.0, 'True':1.0, 'False':0.0}


# Remove correlated and unneccessary columns
col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 'Total night charge','Total intl charge']
orange_df = orange_sprk_df.toPandas().drop(col_to_drop, axis=1)

# Change categorical data to Numeric for the traininfg set 80%
orange_df[['International plan', 'Voice mail plan']] = orange_df[['International plan', 'Voice mail plan']].replace(binary_map)
orange_df['Churn'] = orange_df['Churn'].apply(lambda d: d.astype(float))


# Perform same function for the 20% test data
orange_train_df = orange_final_dataset.toPandas().drop(col_to_drop, axis=1)
orange_train_df[['International plan', 'Voice mail plan']] = orange_train_df[['International plan', 'Voice mail plan']].replace(binary_map)
orange_train_df['Churn'] = orange_train_df['Churn'].apply(lambda d: d.astype(float))

# Print sample
orange_df.head()


orange_2sparkdf = sqlContext.createDataFrame(orange_df)
orange_2sparkdf.take(2)



# # Spark Machine Learning Package `MLLib`
# 
# MLlib classifiers and regressors require data sets in a format of rows of type LabeledPoint, which separates row labels and feature lists, and names them accordingly. We split it further into training and testing sets. A decision tree classifier model is then generated using the training data, using a maxDepth of 2, to build a "shallow" tree. The tree depth can be regarded as an indicator of model complexity.
# 

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics


# Separate the features and the target variable

def labelData(data):
    """
        LabeledPoint(self, label, features)
        Label: row[end] i.e. 'Churn'
        Features: row[0 : end-1] i.e. other columns beside last column
    """
    return data.map(lambda row: LabeledPoint(row[-1], row[:-1]))

# Example: Target Variable and Feature Variables.
labelData(orange_2sparkdf).takeSample(True, 5)


# Divide into training data and test data
training_data, testing_data = labelData(orange_2sparkdf).randomSplit([0.8, 0.2])

# Design the model
# Map categorical variables to number of into categories.
# index 1 `International plan` has 2 variables (Yes/No) and Index 2 'Voice mail plan' has 2 variables (Yes/No)
model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                     categoricalFeaturesInfo={1:2, 2:2}, 
                                     maxDepth=3, impurity='gini', maxBins=32
                                    )
print(model.toDebugString())


# The toDebugString() function provides a print of the tree's decision nodes and final prediction outcomes at the end leafs. We can see that features 12 and 4 are used for decision making and should thus be considered as having high predictive power to determine a customer's likeliness to churn. It's not surprising that these feature numbers map to the fields Customer service calls and Total day minutes. Decision trees are often used for feature selection because they provide an automated mechanism for determining the most important features (those closest to the tree root).
# 

print(orange_df.columns[4])
print(orange_df.columns[12])
print(orange_df.columns[6])
print(orange_df.columns[1])


# ## Model Evaluation
# 
# Predictions of the testing data's churn outcome are made with the model's `predict()` function and grouped together with the actual churn label of each customer data using `getPredictionsLabels()`.
# 
# We'll use MLlib's `MulticlassMetrics()` for the model evaluation, which takes rows of *(prediction, label)* tuples as input. It provides metrics such as `precision, recall, F1 score and confusion matrix`, which have been bundled for printing with the custom `printMetrics()` function.
# 

def getPredictionLabels(model, testing_data):
    predictions = model.predict(testing_data.map(lambda r: r.features))
    return predictions.zip(testing_data.map(lambda r: r.label))


def printMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print('Precision of True ', metrics.precision(1))
    print('Precision of False', metrics.precision(0))
    print('Recall of True / False Positive  ', metrics.recall(1))
    print('Recall of False / False Negative  ', metrics.recall(0))
    print('F-1 Score        \n\n ', metrics.fMeasure())
    print(pd.DataFrame([['True Positive','False Negative'],['False Positive','True Negative']]))
    print('\nConfusion Matrix \n\n {}'.format(metrics.confusionMatrix().toArray()))
    


predictions_and_labels = getPredictionLabels(model, testing_data)
predictions_and_labels.take(5)


printMetrics(predictions_and_labels)


# The `F-1 Score` is $90.7%$ however the `recall measures` or `sensitivity` have high discrepancy. Perhaps the model's sensitivity bias toward `Churn=False` samples is due to a skewed distribution of the two types of samples. Let's try grouping the `orange_2sparkdf` DataFrame by the Churn field and counting the number of instances in each group.
# 

orange_2sparkdf.groupBy('Churn').count().toPandas()


# ## Stratified sampling 
# A probability sampling technique where the entire data/population is divided into different subgroups or strata, then randomly selects the final subjects proportionally from the different strata. The `False` samples are almost **7 times** larger than the `True` samples so the distribution is skewed. We can build a new model using an evenly distributed data set.
# 

# Sample all the 1s (100% of ones) amd 20% of zeros.
strat_orange_2sparkdf = orange_2sparkdf.sampleBy('Churn', fractions={0:0.2, 1:1.0})
strat_orange_2sparkdf.groupBy('Churn').count().toPandas()


training_data, testing_data = labelData(strat_orange_2sparkdf).randomSplit([0.8, 0.2])

model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                     categoricalFeaturesInfo={1:2, 2:2}, 
                                     maxDepth=3, impurity='gini', maxBins=32 )

predictions_and_labels = getPredictionLabels(model, testing_data)
printMetrics(predictions_and_labels)


# With these new recall values, we see that the stratified data built a less biased model, which will ultimately provide more generalized and robust predictions. 
# 




