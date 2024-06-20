# # Nested Attributes & Functions Operating on Nested Types in PySpark
# 
# In this notebook we will be working with spotify songs Dataset from Kaggle. Specifically we will work with nested data types where the columns are to type arrays or maps.
# 

import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
from pyspark.sql.window import Window

import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col


pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 400)


# setting random seed for notebook reproducability
rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed


# ## 1. Create the Spark Session
# 

spark = SparkSession.builder.master("local[*]").appName("nested_attributes").getOrCreate()


spark


sc = spark.sparkContext
sc


sqlContext = SQLContext(spark.sparkContext)
sqlContext


import re

# Utility function to emulate stripMargin in Scala string.
def strip_margin(text):
    nomargin = re.sub('\n[ \t]*\|', ' ', text)
    trimmed = re.sub('\s+', ' ', nomargin)
    return trimmed


# ## 2. Load Spotify Songs Dataset
# 

spotify_df = spark.read.csv(path='data/spotify-songs.csv', inferSchema=True, header=True).cache()


spotify_df.limit(10).toPandas()


# ## 3. Data Wrangling
# 
# ### 3.1 Create Nested Types
# 
# + Combine the columns ['key', 'mode', 'target'] into an array using the `array` function of PySpark. 
# + Transform the acoustic qualities {'acousticness', 'tempo', 'liveness', 'instrumentalness', 'energy', 'danceability', 'speechiness', 'loudness'} of a song from individual columns into a map (key being acoustic quality). Although `create_map` function is meant to create map between a pair of columns but here we use the F.lit(...) function to generate the string key name for the acoustic quality.
# http://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html#pyspark.sql.functions.create_map
# 

map_df = (spotify_df
          .select('id', 'song_title', 'artist', 'duration_ms',
                  F.array('key', 'mode', 'target').alias('audience'), 
                  F.create_map(
                      F.lit('acousticness'), 'acousticness', 
                      F.lit('danceability'), 'acousticness',
                      F.lit('energy'), 'energy',
                      F.lit('instrumentalness'), 'instrumentalness',
                      F.lit('liveness'), 'liveness',
                      F.lit('loudness'), 'loudness',
                      F.lit('speechiness'), 'speechiness',
                      F.lit('tempo'), 'tempo'
                  ).alias('qualities'),
                 'time_signature',
                 'valence')
        .cache())


map_df.limit(10).toPandas()


# Let's check the schema of the new DataFrame
map_df.printSchema()


# **Write the DataFrame to a json file:**
# 

map_df.write.json(path='data/spotify-songs', mode="overwrite")


# ### 3.2 Reload the above restructured DataFrame now using a more complex schema with Nested Data Types
# 

nested_schema = StructType([
    StructField('id', IntegerType(), nullable=False),
    StructField('song_title', StringType(), nullable=False),
    StructField('artist', StringType(), nullable=False),
    StructField('duration_ms', IntegerType(), nullable=False),
    StructField('audience', ArrayType(elementType=IntegerType()), nullable=False),
    StructField('qualities', MapType(keyType=StringType(), valueType=DoubleType(), valueContainsNull=False), nullable=True),
    StructField('time_signature', IntegerType(), nullable=False),
    StructField('valence', DoubleType(), nullable=False),
  ])


spotify_reloaded_df = spark.read.json(path='data/spotify-songs', schema=nested_schema).cache()


spotify_reloaded_df.limit(10).toPandas()


spotify_reloaded_df.printSchema()


# ### 3.3 Extract Individual Nested/Complex Atributes as a Column
# 
# We can extract out each nested attribute within an array or map into a column of its own.
# 
# **Extract out array elements:**  
# The audience column is a combination of three attributes 'key', 'mode' and 'target'. Extract out each array element into a column of its own.
# 

(spotify_reloaded_df
 .select(spotify_reloaded_df.audience.getItem(0).alias('key'), 
         spotify_reloaded_df.audience.getItem(1).alias('mode'),
         spotify_reloaded_df.audience.getItem(2).alias('target'))
 .limit(10)
 .toPandas())


# **Extract out Map attributes:**  
# The acoustic column is a map created from attributes 'acousticness', 'tempo', 'liveness', 'instrumentalness', etc. of a song. Extract out those qualities into individual columns.
# 

(spotify_reloaded_df
 .select(
     spotify_reloaded_df.qualities.getItem('acousticness').alias('acousticness'),
     spotify_reloaded_df.qualities.getItem('speechiness').alias('speechiness')
 )
 .limit(10)
 .toPandas())


# **Reconstruct the original Table:**
# 

(spotify_reloaded_df
 .select('id', 'song_title', 'artist',
     spotify_reloaded_df.qualities.getItem('acousticness').alias('acousticness'),
     spotify_reloaded_df.qualities.getItem('danceability').alias('danceability'),
     'duration_ms',
     spotify_reloaded_df.qualities.getItem('energy').alias('energy'),
     spotify_reloaded_df.qualities.getItem('instrumentalness').alias('instrumentalness'),
     spotify_reloaded_df.audience.getItem(0).alias('key'),
     spotify_reloaded_df.qualities.getItem('liveness').alias('liveness'),
     spotify_reloaded_df.qualities.getItem('loudness').alias('loudness'),
     spotify_reloaded_df.audience.getItem(1).alias('mode'),
     spotify_reloaded_df.qualities.getItem('speechiness').alias('speechiness'),
     spotify_reloaded_df.qualities.getItem('tempo').alias('tempo'),
     'time_signature',
     'valence',
     spotify_reloaded_df.audience.getItem(2).alias('target')
 )
 .limit(10)
 .toPandas())


# ### 3.4 Explode Individual Nested/Complex into a row of its own
# 

# Using `posexplode` function we can extract array element into a new row for each element with position in the given array.
# 

(spotify_reloaded_df
 .select(F.posexplode(spotify_reloaded_df.audience))
 .limit(10)
 .toPandas())


# Using `explode` function we can extract a new row for each element in the given array or map.
# 

(spotify_reloaded_df
 .select(F.explode(spotify_reloaded_df.qualities).alias("qualities", "value"))
 .limit(10)
 .toPandas())


spark.stop()


# # Analyzing Customer-Music Data using Apache Spark
# 
# The original Tableau based tutorial is at https://mapr.com/blog/real-time-user-profiles-spark-drill-and-mapr-db/. I have coverted them to Python and also configured to run the codes on Jupter Notebooks with Pyspark 2.2 and used the Jupyter Notebook to render visualization and added more stuffs.
# 
# Users are continuously connecting to the service and listening to tracks that they like -- this generates our main data set. The behaviors captured in these events, over time, represent the highest level of detail about actual behaviors of customers as they consume the service by listening to music. In addition to the events of listening to individual tracks, we have a few other data sets representing all the information we might normally have in such a service. In this post we will make use of the following two data sets, and in my next post we will bring in an additional data set relating to click events.
# 

# **How to Configure Jupter Notebook to run Spark 2.2:**
# ```
# 1. Create an Anaconda environment 'pyspark'
# 2. !pip install pyspark (this package will directly communicate with Spark)
# 3. Install Spark in your machine (from here: https://spark.apache.org/downloads.html). I installed spark-2.2.0-bin-hadoop2.6.tgz
# 4. In your .bash_profile set the following
#     export SPARK_HOME=/Users/anindyas/work/spark-2.2.0-bin-hadoop2.6
#     PATH=$SPARK_HOME/bin:$PATH
#     export PATH
# 
#     export PATH="/Users/anindyas/anaconda/envs/pyspark/bin:$PATH"
#     
#     There is further better way to set the environment variables for anaconda but for now it is fine.
# 5. Switch to pyspark anaconda environment and start the jupyter notebook from that environment.
# ```
# 

# ## 1. Understanding the Data Set
# 
# **Individual customers listening to individual tracks: (tracks.csv)** - a collection of events, one per line, where each event is a client listening to a track.
# 
# This data is approximately 1M lines and contains simulated listener events over several months.
# 
# <table>
#   <tr>
#     <th><strong>Field Name</strong></th>
#     <th>Event ID</th>
#     <th>Customer ID</th>
#     <th>Track ID</th>
#     <th>Datetime</th>
#     <th>Mobile</th>
#     <th>Listening Zip</th>
#   </tr>
#   <tr>
#     <td><strong>Type</strong></td>
#     <td>Integer</td>
#     <td>Integer</td>
#     <td>Integer</td>
#     <td>String</td>
#     <td>Integer</td>
#     <td>Integer</td>
#   </tr>
#   <tr>
#     <td><strong>Example Value</strong></td>
#     <td>9999767</td>
#     <td>2597</td>
#     <td>788</td>
#     <td>2014-12-01 09:54:09</td>
#     <td>0</td>
#     <td>11003</td>
#   </tr>
# </table>
# 
# The event, customer and track IDs tell us what occurred (a customer listened to a certain track), while the other fields tell us some associated information, like whether the customer was listening on a mobile device and a guess about their location while they were listening. With many customers listening to many tracks, this data can get very large and will be the input into our Spark job.
# 
# **Customer information:** - information about individual customers.
# 
# <table>
#   <tr>
#     <th><strong>Field Name</strong></th>
#     <th>Customer ID</strong></th>
#     <th>Name</th>
#     <th>Gender</th>
#     <th>Address</th>
#     <th>ZIP</th>
#     <th>Sign Date</th>    
#     <th>Status</th>
#     <th>Level</th>
#     <th>Campaign</th>
#     <th>Linked with Apps?</th>
#   </tr>
#   <tr>
#     <td><strong>Type</strong></td>
#     <td>Integer</td>
#     <td>String</td>
#     <td>Integer</td>
#     <td>String</td>
#     <td>Integer</td>
#     <td>String</td>
#     <td>Integer</td>
#     <td>Integer</td>
#     <td>Integer</td>
#     <td>Integer</td>    
#   </tr>
#   <tr>
#     <td><strong>Example Value</strong></td>
#     <td>10</td>
#     <td>Joshua Threadgill</td>
#     <td>0</td>
#     <td>10084 Easy Gate Bend</td>
#     <td>66216</td>
#     <td>01/13/2013</td>
#     <td>0</td>
#     <td>1</td>
#     <td>1</td>
#     <td>1</td>
#   </tr>
# </table>
# 
# 
# The fields are defined as follows:
# ```
# Customer ID: a unique identifier for that customer
# Name, gender, address, zip: the customer’s associated information
# Sign date: the date of addition to the service
# Status: indicates whether or not the account is active (0 = closed, 1 = active)
# Level: indicates what level of service -- 0, 1, 2 for Free, Silver and Gold, respectively
# Campaign: indicates the campaign under which the user joined, defined as the following (fictional) campaigns driven by our (also fictional) marketing team:
# NONE - no campaign
# 30DAYFREE - a ‘30 days free’ trial offer
# SUPERBOWL - a Superbowl-related program
# RETAILSTORE - an offer originating in brick-and-mortar retail stores
# WEBOFFER - an offer for web-originated customers
# ```
# 

import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col


# Visualization
import seaborn as sns
import matplotlib.pyplot as plt


# Visualization
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (6,4)})
rcParams['figure.figsize'] = 6,4

# this allows plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# setting random seed for notebook reproducability
rnd_seed=42
np.random.seed=rnd_seed
np.random.set_state=rnd_seed


# ## 2. Creating the Spark Session
# 

os.environ['SPARK_HOME']


spark = (SparkSession
         .builder
         .master("local[*]")
         .appName("music-customer-analysis")
         .getOrCreate())


spark


sc = spark.sparkContext
sc


sqlContext = SQLContext(spark.sparkContext)
sqlContext


# ## 3. Load the data from files into DataFrames
# 

MUSIC_TRACKS_DATA = 'data/tracks.csv'
CUSTOMER_DATA =     'data/cust.csv'


# define the schema, corresponding to a line in the csv data file for music
music_schema = StructType([
    StructField('event_id', IntegerType(), nullable=True),
    StructField('customer_id', IntegerType(), nullable=True),
    StructField('track_id', StringType(), nullable=True),
    StructField('datetime', StringType(), nullable=True),
    StructField('is_mobile', IntegerType(), nullable=True),
    StructField('zip', IntegerType(), nullable=True)]
  )


# define the schema, corresponding to a line in the csv data file for customer
cust_schema = StructType([
    StructField('customer_id', IntegerType(), nullable=True),
    StructField('name', StringType(), nullable=True),
    StructField('gender', IntegerType(), nullable=True),
    StructField('address', StringType(), nullable=True),
    StructField('zip', IntegerType(), nullable=True),
    StructField('sign_date', StringType(), nullable=True),
    StructField('status', IntegerType(), nullable=True),
    StructField('level', IntegerType(), nullable=True),
    StructField('campaign', IntegerType(), nullable=True),
    StructField('lnkd_with_apps', IntegerType(), nullable=True)]
  )


# Load data
music_df = spark.read.csv(path=MUSIC_TRACKS_DATA, schema=music_schema).cache()
cust_df = spark.read.csv(path=CUSTOMER_DATA, schema=cust_schema, header=True).cache()


# How many music data rows
music_df.count()


music_df.show(5)


# How many customer data rows
cust_df.count()


cust_df.show(5)


# ## 4. Data Exploration
# 

# ### 4.1 Add a new Column Hour to the Music data
# 

music_df = music_df.withColumn('hour', F.hour('datetime')).cache()


music_df.show(5)


# **Divide the entire day into four time zones based on the hour:**
# 

music_df = (music_df
    .withColumn('night', F.when((col('hour') < 5) | (col('hour') == 23), 1).otherwise(0))
    .withColumn('morn', F.when((col('hour') >= 5) & (col('hour') < 12), 1).otherwise(0))
    .withColumn('aft', F.when((col('hour') >= 12) & (col('hour') < 17), 1).otherwise(0))
    .withColumn('eve', F.when((col('hour') >= 17) & (col('hour') < 22), 1).otherwise(0)))


music_df.show(5)


# ### 4.2 Compute Summary profile of each customer
# 
# Now we’re ready to compute a summary profile for each user. By passing a function we’ll write to mapValues, we compute some high-level data:
# 
# + Average number of tracks listened during each period of the day: morning, afternoon, evening, and night. We arbitrarily define the time ranges in the code.
# + Total unique tracks listened by that user, i.e. the set of unique track IDs.
# + Total mobile tracks listened by that user, i.e. the count of tracks that were listened that had their mobile flag set.
# 

cust_profile_df = (music_df.select(['customer_id', 'track_id', 'night', 'morn', 'aft', 'eve', 'is_mobile'])
     .groupBy('customer_id')
     .agg(F.countDistinct('track_id'), F.sum('night'), F.sum('morn'), F.sum('aft'), F.sum('eve'), F.sum('is_mobile'))).cache()


cust_profile_df.show(10)


# **Summary Statistics:**
# 
# Since we have the summary data readily available we compute some basic statistics on it.
# 

cust_profile_df.select([c for c in cust_profile_df.columns if c not in ['customer_id']]).describe().show()


# Interpreting the summary statistics:
# > People Listen to highest number of songs in the Night!
# 

# ### 4.3 Average number of tracks listened by Customers of Different Levels during Different Time of the Day:
# 

cust_df.show(5)


# Map from level number to actual level string
level_map = {0:"Free", 1:"Silver", 2:"Gold"}

# Define a udf
udfIndexTolevel = udf(lambda x: level_map[x], StringType())


result_df = (cust_df.join(cust_profile_df, on='customer_id', how='inner')
                     .select([udfIndexTolevel('level').alias('level'), 'sum(night)', 'sum(morn)', 'sum(aft)', 'sum(eve)'])
                     .groupBy('level')
                     .agg(F.avg('sum(aft)').alias('Afternoon'), 
                          F.avg('sum(eve)').alias('Evening'), 
                          F.avg('sum(morn)').alias('Morning'), 
                          F.avg('sum(night)').alias("Night")))


result_df.cache().show()


result_df.toPandas().plot.bar(x='level', figsize=(12, 4));


result_df.unpersist()


# ### 4.4 Distribution of Customers By Level:
# 

result_df = (cust_df.select(['level', (F.when(col('gender') == 0, "Male").otherwise("Female")).alias('gender')])
                 .groupBy('level')
                 .pivot('gender')
                 .count()
                 .orderBy('level', ascending=False))


result_df.cache().show()


result_df.toPandas().set_index('level').plot.barh(stacked=True);


result_df.unpersist()


# ### 4.4 Top 10 Zip Codes: Which regions consume most from this service:
# 

result_df = cust_df.groupBy('zip').count().orderBy('count', ascending=False).limit(10)


result_df.cache().show()


result_df.toPandas().plot.barh(x='zip');


result_df.unpersist()


# ### 4.5 Distribution of Customers By SignUp Campaign:
# 

# Map from campaign number to actual campaign string
campaign_map = {0:"None", 1:"30DaysFree", 2:"SuperBowl",  3:"RetailStore", 4:"WebOffer"}

# Define a udf
udfIndexToCampaign = udf(lambda x: campaign_map[x], StringType())


result_df = (cust_df.select(udfIndexToCampaign("campaign").alias("campaign"))
                 .groupBy('campaign')
                 .count()
                 .orderBy('count', ascending=True))


result_df.cache().show()


result_df.toPandas().plot.barh(x='campaign');


result_df.unpersist()


# ### 4.6 Average Unique Track Count By Customer Level:
# 

result_df = (music_df.select(['customer_id', 'track_id'])
                            .groupBy('customer_id')
                            .agg(F.countDistinct('track_id').alias('unique_track_count'))
                            .join(cust_df, on='customer_id', how='inner')
                            .select([udfIndexTolevel('level').alias('level'), 'unique_track_count'])
                            .groupBy('level')
                            .agg(F.avg('unique_track_count').alias('avg_unique_track_count')))


result_df.cache().show()


result_df.toPandas().sort_values(by='avg_unique_track_count', ascending=False).plot.barh(x='level');


result_df.unpersist()


# ### 4.7 Mobile Tracks Count By Customer Level:
# 

result_df = (music_df.select(['customer_id', 'track_id'])
                            .filter(col('is_mobile') == 1)
                            .groupBy('customer_id')
                            .count()
                            .withColumnRenamed('count', 'mobile_track_count')
                            .join(cust_df, on='customer_id', how='inner')
                            .select([udfIndexTolevel('level').alias('level'), 'mobile_track_count'])
                            .groupBy('level')
                            .agg(F.avg('mobile_track_count').alias('avg_mobile_track_count'))
                            .orderBy('avg_mobile_track_count'))


result_df.cache().show()


result_df.toPandas().sort_values(by='avg_mobile_track_count', ascending=False).plot.barh(x='level');


result_df.unpersist()


music_df.unpersist()
cust_df.unpersist()


spark.stop()





