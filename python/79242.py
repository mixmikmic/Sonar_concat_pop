# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/clusters.png" width="400px"></div>
# 
# # Clustering: Picking the 'K' hyperparameter
# The unsupervised machine learning technique of clustering data into similar groups can be useful and fairly efficient in most cases. The big trick is often how you pick the number of clusters to make (the K hyperparameter). 
# The number of clusters may vary dramatically depending on the characteristics of the data, the different types of variables (numeric or categorical), how the data is normalized/encoded and the distance metric used.
# 
# <div style="float: left; margin: 20px 50px 20px 20px"><img src="images/picking.png" width="100px"></div>
# 
# **For this notebook we're going to focus specifically on the following:**
# - Optimizing the number of clusters (K hyperparameter) using Silhouette Scoring
# - Utilizing an algorithm (DBSCAN) that automatically determines the number of clusters
# 
# 
# ### Software
# - Bro Analysis Tools (BAT): https://github.com/Kitware/bat
# - Pandas: https://github.com/pandas-dev/pandas
# - Scikit-Learn: http://scikit-learn.org/stable/index.html
# 
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/scikit.png" width="200px"></div>
# ### Techniques
# - One Hot Encoding: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
# - t-SNE: https://distill.pub/2016/misread-tsne/
# - Kmeans: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# - Silhouette Score: https://en.wikipedia.org/wiki/Silhouette_(clustering)
# - DBSCAN: https://en.wikipedia.org/wiki/DBSCAN
# 

# Third Party Imports
import pandas as pd
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, DBSCAN

# Local imports
import bat
from bat.log_to_dataframe import LogToDataFrame
from bat.dataframe_to_matrix import DataFrameToMatrix

# Good to print out versions of stuff
print('BAT: {:s}'.format(bat.__version__))
print('Pandas: {:s}'.format(pd.__version__))
print('Scikit Learn Version:', sklearn.__version__)


# Create a Pandas dataframe from the Bro log
http_df = LogToDataFrame('../data/http.log')

# Print out the head of the dataframe
http_df.head()


# # Our HTTP features are a mix of numeric and categorical data
# When we look at the http records some of the data is numerical and some of it is categorical so we'll need a way of handling both data types in a generalized way. We have a DataFrameToMatrix class that handles a lot of the details and mechanics of combining numerical and categorical data, we'll use below.
# 

# <div style="float: right; margin: 10px 40px 10px 40px"><img src="images/transformers.png" width="200px"></div>
# ## Transformers
# **We'll now use the Scikit-Learn tranformer class to convert the Pandas DataFrame to a numpy ndarray (matrix). The transformer class takes care of many low-level details**
# * Applies 'one-hot' encoding for the Categorical fields
# * Normalizes the Numeric fields
# * The class can be serialized for use in training and evaluation
#   * The categorical mappings are saved during training and applied at evaluation
#   * The normalized field ranges are stored during training and applied at evaluation
# 

# We're going to pick some features that might be interesting
# some of the features are numerical and some are categorical
features = ['id.resp_p', 'method', 'resp_mime_types', 'request_body_len']

# Use the DataframeToMatrix class (handles categorical data)
# You can see below it uses a heuristic to detect category data. When doing
# this for real we should explicitly convert before sending to the transformer.
to_matrix = DataFrameToMatrix()
http_feature_matrix = to_matrix.fit_transform(http_df[features], normalize=True)

print('\nNOTE: The resulting numpy matrix has 12 dimensions based on one-hot encoding')
print(http_feature_matrix.shape)
http_feature_matrix[:1]


# Plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12.0
plt.rcParams['figure.figsize'] = 14.0, 7.0


# <div style="float: right; margin: 10px 40px 10px 40px"><img src="images/silhouette.jpg" width="250px"></div>
# # Silhouette Scoring
# "The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters."
# - https://en.wikipedia.org/wiki/Silhouette_(clustering)
# 

from sklearn.metrics import silhouette_score

scores = []
clusters = range(2,20)
for K in clusters:
    
    clusterer = KMeans(n_clusters=K)
    cluster_labels = clusterer.fit_predict(http_feature_matrix)
    score = silhouette_score(http_feature_matrix, cluster_labels)
    scores.append(score)

# Plot it out
pd.DataFrame({'Num Clusters':clusters, 'score':scores}).plot(x='Num Clusters', y='score')


# ## Silhouette graphs shows that 10 is the 'optimal' number of clusters
# - 'Optimal': Human intuition and clustering involves interpretation/pattern finding and is often partially subjective :)
# - For large datasets running an exhaustive search can be time consuming
# - For large datasets you can often get a large K using max score, so pick the 'knee' of the graph as your K
# 

# So we know that the highest (closest to 1) silhouette score is at 10 clusters
kmeans = KMeans(n_clusters=10).fit_predict(http_feature_matrix)

# TSNE is a great projection algorithm. In this case we're going from 12 dimensions to 2
projection = TSNE().fit_transform(http_feature_matrix)

# Now we can put our ML results back onto our dataframe!
http_df['cluster'] = kmeans
http_df['x'] = projection[:, 0] # Projection X Column
http_df['y'] = projection[:, 1] # Projection Y Column


# Now use dataframe group by cluster
cluster_groups = http_df.groupby('cluster')

# Plot the Machine Learning results
colors = {-1:'black', 0:'green', 1:'blue', 2:'red', 3:'orange', 4:'purple', 5:'brown', 6:'pink', 7:'lightblue', 8:'grey', 9:'yellow'}
fig, ax = plt.subplots()
for key, group in cluster_groups:
    group.plot(ax=ax, kind='scatter', x='x', y='y', alpha=0.5, s=250,
               label='Cluster: {:d}'.format(key), color=colors[key])


# Now print out the details for each cluster
pd.set_option('display.width', 1000)
for key, group in cluster_groups:
    print('\nCluster {:d}: {:d} observations'.format(key, len(group)))
    print(group[features].head(3))


# <div style="float: right; margin: 30px 20px 20px 20px"><img src="images/no_hands.jpg" width="250px"></div>
# # Look Ma... No K!
# ### DBSCAN
# Density-based spatial clustering is a data clustering algorithm that given a set of points in space, groups points that are closely packed together and marking low-density regions as outliers.
# 
# - You don't have to pick K
# - There are other hyperparameters (eps and min_samples) but defaults often work well
# - https://en.wikipedia.org/wiki/DBSCAN
# - Hierarchical version: https://github.com/scikit-learn-contrib/hdbscan
# 

# Now try DBScan
http_df['cluster_db'] = DBSCAN().fit_predict(http_feature_matrix)
print('Number of Clusters: {:d}'.format(http_df['cluster_db'].nunique()))


# Now use dataframe group by cluster
cluster_groups = http_df.groupby('cluster_db')

# Plot the Machine Learning results
fig, ax = plt.subplots()
for key, group in cluster_groups:
    group.plot(ax=ax, kind='scatter', x='x', y='y', alpha=0.5, s=250,
               label='Cluster: {:d}'.format(key), color=colors[key])


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/magic.jpg" width="300px"></div>
# # DBSCAN automagically determined 10 clusters!
# So obviously we got a bit lucky here and for different datasets with different feature distributions DBSCAN may not give you the optimal number of clusters right off the bat. There are two hyperparameters that can be tweeked but like we said the defaults often work well. See the DBSCAN and Hierarchical DBSCAN links for more information.
# 
# - https://en.wikipedia.org/wiki/DBSCAN
# - Hierarchical version: https://github.com/scikit-learn-contrib/hdbscan
# 

# <div style="float: right; margin: 50px 0px 0px 0px"><img src="https://www.kitware.com/img/small_logo_over.png"></div>
# ## Wrap Up
# Well that's it for this notebook, given the usefulness and relatively efficiency of clustering it a good technique to include in your toolset. Understanding the K hyperparameter and how to determine optimal K (or not if you're using DBSCAN) is a good trick to know.
# 
# If you liked this notebook please visit the [bat](https://github.com/Kitware/bat) project for more notebooks and examples.
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/bro.png" width="100px"></div>
# 
# # Bro Network Data to Plotting
# Yes, this is provincial, but going from a Bro log to a visual data plot in a few lines of code might be really handy sometimes. So without further ado here's a very small bit of code :)
# 
# <div style="float: right; margin: 30px -100px 0px 0px"><img src="images/matplotlib.png" width="300px"></div>
# 
# ### Software
# - Bro Analysis Tools (BAT): https://github.com/Kitware/bat
# - Pandas: https://github.com/pandas-dev/pandas
# - Matplotlib: https://matplotlib.org
# 

# ## Quickly go from Bro log to Pandas DataFrame
# 

from bat.log_to_dataframe import LogToDataFrame
from bat.utils import plot_utils

# Just some plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plot_utils.plot_defaults()

# Convert it to a Pandas DataFrame
http_df = LogToDataFrame('../data/http.log')
http_df.head()


# <div style="float: left; margin: 20px 20px 20px 20px"><img src="images/eyeball.jpeg" width="100px"></div>
# 
# ## Lets look at our data
# Above we used a BAT utility method to set up nice plotting defaults and here we simply use the plotting provided by Pandas.
# 

http_df[['request_body_len','response_body_len']].hist()


# <div style="float: left; margin: 20px 20px 20px 20px"><img src="images/eyeball.jpeg" width="100px"></div>
# 
# ## Lets look at our data again
# Since BAT automatically makes the timestamp the index, we can plot volume over time super easy.
# 

http_df['uid'].resample('1S').count().plot()
plt.xlabel('HTTP Requests per Second')


# <div style="float: right; margin: 50px 0px 0px 0px"><img src="https://www.kitware.com/img/small_logo_over.png"></div>
# ## Wrap Up
# Well that's it for this notebook, it was kinda simple but sometime you just want to know how to plot the darn thing. :)
# 
# If you liked this notebook please visit the [BAT](https://github.com/Kitware/bat) project for more notebooks and examples.
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/bro.png" width="130px"></div>
# 
# # Bro to Kafka to Spark
# This notebook covers how to stream Bro data into Spark using Kafka as a message queue. The setup takes a bit of work but the result will be a nice scalable, robust way to process and analyze streaming data from Bro.
# 
# For a super **EASY** way to get started with Spark (local data without Kafka) you can view this notebook:
# - https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Spark_Cheesy.ipynb
# 
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/kafka.png" width="180px"></div>
# 
# ### Software
# - Bro Network Monitor: https://www.bro.org
# - Kafka: https://kafka.apache.org
# - Spark: https://spark.apache.org
# 
# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/spark.png" width="200px"></div> 
# 
# <div style="float: left; margin: 20px 20px 20px 20px"><img src="https://www.kitware.com/img/small_logo_over.png"></div>
# 

# <div style="float: right; margin: 20px 0px 0px 0px"><img src="images/confused.jpg" width="300px"></div>
# 
# # Getting Everything Setup
# Get a nice cup of coffee and some snacks, we estimate 45-60 minutes for setting up a local Spark server and configuring Bro with the Kafka plugin. 
# 
# ## Install local Spark Server
# For this notebook we're going to be using a local spark server, obviously we would want to setup a cluster for running Spark for a real system.
# ```
# $ pip install pyspark
# ```
# Yep, that's it. The latest version of PySpark will setup a local server with a simple pip install.
# 
# # Setting up Bro with the Kafka Plugin
# So this is the most complicated part of the setup, once you've setup the Bro Kafka plugin you're basically done.
# 
# ## Shout Out/Thanks:
# - **Mike Sconzo** (for the initial setup instructions and general helpful awesomeness)
# - Jon Zeolla,  Nick Allen, and Casey Stella for the Kafka Plugin (Apache Metron project)
# - Kafka Plugin currently exists here:https://github.com/apache/metron-bro-plugin-kafka
# - The official Bro Package, when done, will be here: https://github.com/apache/metron-bro-plugin-kafka
# 
# ## Install Kafka
# ```
# $ brew install kafka
# $ brew Install librdkafka
# $ brew install zookeeper
# ```
# **Note: For Ubuntu 16.04 instructions see: https://hevo.io/blog/how-to-set-up-kafka-on-ubuntu-16-04/**
# 
# ## Add Kafka plugin to Bro
# **Note**: This will be much easier when the Kafka Plugin is a Bro 'Package' (coming soon :)
# 
# **Get and Compile Bro (you have do this for now)**
# ```
# $ git clone --recursive https://github.com/bro/bro.git
# $ cd bro 
# $ ./configure
# $ make install
# ```
# 
# **Get the Kafka Bro plugin**
# ```
# $ git clone https://github.com/apache/metron-bro-plugin-kafka
# $ cd metron-bro-plugin-kafka
# $ ./configure --bro-dist=$BRO_SRC_PATH
# $ make install
# ```
# 
# ## Test the Bro Kafka Plugin
# ```
# $ bro -N Bro::Kafka
# > Bro::Kafka - Writes logs to Kafka (dynamic, version 0.1)
# ```
# 
# ## Setup plugin in local.bro
# Okay, so the logic below will output each log to a different Kafka topic. So the dns.log events will be sent to the 'dns' topic and the http.log events will be sent to the 'http' topic.. etc. If you'd like to send all the events to one topic or other configurations, please see https://github.com/apache/metron-bro-plugin-kafka
# 
#     @load Bro/Kafka/logs-to-kafka.bro
#     redef Kafka::topic_name = "";
#     redef Kafka::logs_to_send = set(Conn::LOG, HTTP::LOG, DNS::LOG, SMTP::LOG);
#     redef Kafka::kafka_conf = table(["metadata.broker.list"] = "localhost:9092");
# 
# ## Start Kafka
# ```
# $ zkserver start
# $ kafka-server-start
# ```
# 
# ## Run Bro
# ```
# $ bro -i en0 <path to>/local.bro
# or 
# $ broctl deploy
# ```
# 
# ## Verify messages are in the queue
# ```
# $ kafka-console-consumer --bootstrap-server localhost:9092 --topic dns
# ```
# **After a second or two.. you should start seeing DNS requests/replies coming out.. hit Ctrl-C after you see some.**
# ```
# {"ts":1503513688.232274,"uid":"CdA64S2Z6Xh555","id.orig_h":"192.168.1.7","id.orig_p":58528,"id.resp_h":"192.168.1.1","id.resp_p":53,"proto":"udp","trans_id":43933,"rtt":0.02226,"query":"brian.wylie.is.awesome.tk","qclass":1,"qclass_name":"C_INTERNET","qtype":1,"qtype_name":"A","rcode":0,"rcode_name":"NOERROR","AA":false,"TC":false,"RD":true,"RA":true,"Z":0,"answers":["17.188.137.55","17.188.142.54","17.188.138.55","17.188.141.184","17.188.129.50","17.188.128.178","17.188.129.178","17.188.141.56"],"TTLs":[25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0],"rejected":false}
# ```
# # If you made it this far you are done!
# <div style="float: left; margin: 20px 20px 20px 20px"><img src="images/whew.jpg" width="300px"></div>
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark.png" width="200px"></div>
# 
# # Structured Streaming in Spark
# Structured Streaming is the new hotness with Spark. Michael Armbrust from DataBricks gave a great talk at Spark Summit 2017 on Structured Streaming:
# - https://www.youtube.com/watch?v=8o-cyjMRJWg
# 
# There's also a good example on the DataBricks blog:
# - https://databricks.com/blog/2017/04/26/processing-data-in-apache-kafka-with-structured-streaming-in-apache-spark-2-2.html
# 

import pyspark
from pyspark.sql import SparkSession

# Always good to print out versions of libraries
print('PySpark: {:s}'.format(pyspark.__version__))


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark.png" width="200px"></div>
# 
# # Spark It!
# ### Spin up Spark with 4 Parallel Executors
# Here we're spinning up a local spark server with 4 parallel executors, although this might seem a bit silly since we're probably running this on a laptop, there are a couple of important observations:
# 
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark_jobs.png" width="400px"></div>
# 
# - If you have 4/8 cores use them!
# - It's the exact same code logic as if we were running on a distributed cluster.
# - We run the same code on **DataBricks** (www.databricks.com) which is awesome BTW.
# 

# Spin up a local Spark Session
# Please note: the config line is an important bit,
# readStream.format('kafka') won't work without it
spark = SparkSession.builder.master('local[4]').appName('my_awesome')        .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.11:2.2.0')        .getOrCreate()


# <div style="float: right; margin: -20px -20px -20px -20px"><img src="images/arrow.png" width="350px"></div>
# 
# # Sidebar: Apache Arrow is going to be Awesome
# For all kinds of reasons, multi-core pipelines, cross language storage, basically it will improve and enable flexible/performant data analysis and machine learning pipelines.
# - Apache Arros: https://arrow.apache.org
# - Spark to Pandas: http://arrow.apache.org/blog/2017/07/26/spark-arrow
# - JupyterCon 2017 Wes McKinney: https://www.youtube.com/watch?v=wdmf1msbtVs
# 

# Optimize the conversion to Pandas
spark.conf.set("spark.sql.execution.arrow.enable", "true")


# # Streaming data pipeline
# Our streaming data pipeline looks conceptually like this.
# <div style="margin: 20px 20px 20px 20px"><img src="images/pipeline.png" width="750px"></div>
# - Kafka Plugin for Bro
# - ** Publish (provides a nice decoupled architecture) **
# - Pull/Subscribe to whatever feed you want (http, dns, conn, x509...)
# - ETL (Extract Transform Load) on the raw message data (parsed data with types)
# - Perform Filtering/Aggregation
# - Data Analysis and Machine Learning
# 

# SUBSCRIBE: Setup connection to Kafka Stream 
raw_data = spark.readStream.format('kafka')   .option('kafka.bootstrap.servers', 'localhost:9092')   .option('subscribe', 'dns')   .option('startingOffsets', 'latest')   .load()


# ETL: Hardcoded Schema for DNS records (do this better later)
from pyspark.sql.types import StructType, StringType, BooleanType, IntegerType
from pyspark.sql.functions import from_json, to_json, col, struct

dns_schema = StructType()     .add('ts', StringType())     .add('uid', StringType())     .add('id.orig_h', StringType())     .add('id.orig_p', IntegerType())     .add('id.resp_h', StringType())     .add('id.resp_p', IntegerType())     .add('proto', StringType())     .add('trans_id', IntegerType())     .add('query', StringType())     .add('qclass', IntegerType())     .add('qclass_name', StringType())     .add('qtype', IntegerType())     .add('qtype_name', StringType())     .add('rcode', IntegerType())     .add('rcode_name', StringType())     .add('AA', BooleanType())     .add('TC', BooleanType())     .add('RD', BooleanType())     .add('RA', BooleanType())     .add('Z', IntegerType())     .add('answers', StringType())     .add('TTLs', StringType())     .add('rejected', BooleanType())


# ETL: Convert raw data into parsed and proper typed data
parsed_data = raw_data   .select(from_json(col("value").cast("string"), dns_schema).alias('data'))   .select('data.*')


# FILTER/AGGREGATE: In this case a simple groupby operation
group_data = parsed_data.groupBy('`id.orig_h`', 'qtype_name').count()


# At any point in the pipeline you can see what you're getting out
group_data.printSchema()


# # Streaming pipeline output to an in-memory table
# Now, for demonstration and discussion purposes, we're going to pull the end of the pipeline  back into memory to inspect the output. A couple of things to note explicitly here:
# 
# - Writing a stream to memory is dangerous and should be done only on small data. Since this is aggregated output we know it's going to be small.
# 
# - The queryName param used below will be the name of the in-memory table.
# 

# Take the end of our pipeline and pull it into memory
dns_count_memory_table = group_data.writeStream.format('memory')   .queryName('dns_counts')   .outputMode('complete')   .start()


dns_count_memory_table


# <div style="float: left; margin: 20px 20px 20px 20px"><img src="images/dynamic.jpg" width="350px"></div>
# 
# # Streaming Query/Table: Looking Deeper
# Note: The in-memory table above is **dynamic**. So as the streaming data pipeline continues to process data the table contents will change. Below we make two of the **same** queries and as more data streams in the results will change.
# 

# Create a Pandas Dataframe by querying the in memory table and converting
dns_counts_df = spark.sql("select * from dns_counts").toPandas()
print('DNS Query Counts = {:d}'.format(len(dns_counts_df)))
dns_counts_df.sort_values(ascending=False, by='count')


# <div style="float: left; margin: 0px 20px 0px 0px"><img src="images/eyeball.jpeg" width="100px"></div>
# 
# # Same Query with Updated Results
# Now we run the same query as above and since the streaming pipeline continues to process new incoming data the in-memory table will **dynamically** update.
# 

# Create a Pandas Dataframe by querying the in memory table and converting
dns_counts_df = spark.sql("select * from dns_counts").toPandas()
print('DNS Query Counts = {:d}'.format(len(dns_counts_df)))
dns_counts_df.sort_values(ascending=False, by='count')


# We should stop our streaming pipeline when we're done
dns_count_memory_table.stop()


# ## Wrap Up
# Well that's it for this notebook, we know this ended before we got to the **exciting** part of the streaming data pipeline. For this notebook we showed everything in the pipeline up to aggregation. In future notebooks we'll dive into the deep end of our pipeline and cover the data analysis and machine learning aspects of Spark.
# <div style="margin: 20px 20px 20px 20px"><img src="images/pipeline.png" width="750px"></div>
# 
# <div style="float: right; margin: 0px 0px 0px 0px"><img src="https://www.kitware.com/img/small_logo_over.png" width="200px"></div>
# If you liked this notebook please visit the [bat](https://github.com/Kitware/bat) project for more notebooks and examples.
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/bro.png" width="100px"></div>
# 
# # Bro to Spark: Clustering
# In this notebook we will pull Bro data into Spark then do some analysis and clustering. The first step is to convert your Bro log data into a Parquet file, for instructions on how to do this (just a few lines of Python code using the BAT package) please see this notebook:
# 
# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/parquet.png" width="300px"></div>
# 
# ### Bro logs to Parquet Notebook
# - [Bro to Parquet to Spark](https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Parquet_to_Spark.ipynb)
# 
# Apache Parquet is a columnar storage format focused on performance. Parquet data is often used within the Hadoop ecosystem and we will specifically be using it for loading data into Spark.
# 
# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/mllib.png" width="200px"></div>
# <div style="float: right; margin: 30px 0px 0px 0px"><img src="images/spark.png" width="200px"></div>
# 
# ### Software
# - Bro Analysis Tools (BAT): https://github.com/Kitware/bat
# - Parquet: https://parquet.apache.org
# - Spark: https://spark.apache.org
# - Spark MLLib: https://spark.apache.org/mllib/
# 
# ### Data
# - Sec Repo: http://www.secrepo.com (no headers on these)
# - Kitware: [data.kitware.com](https://data.kitware.com/#collection/58d564478d777f0aef5d893a) (with headers)
# 

# Third Party Imports
import pyspark
from pyspark.sql import SparkSession
import pyarrow

# Local imports
import bat
from bat.log_to_parquet import log_to_parquet

# Good to print out versions of stuff
print('BAT: {:s}'.format(bat.__version__))
print('PySpark: {:s}'.format(pyspark.__version__))
print('PyArrow: {:s}'.format(pyarrow.__version__))


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark.png" width="200px"></div>
# 
# # Spark It!
# ### Spin up Spark with 4 Parallel Executors
# Here we're spinning up a local spark server with 4 parallel executors, although this might seem a bit silly since we're probably running this on a laptop, there are a couple of important observations:
# 
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark_jobs.png" width="400px"></div>
# 
# - If you have 4/8 cores use them!
# - It's the exact same code logic as if we were running on a distributed cluster.
# - We run the same code on **DataBricks** (www.databricks.com) which is awesome BTW.
# 
# 

# Spin up a local Spark Session (with 4 executors)
spark = SparkSession.builder.master("local[4]").appName('my_awesome').getOrCreate()


# <div style="float: right; margin: 50px 20px 20px -20px"><img src="images/parquet.png" width="350px"></div>
# ##  Read in our Parquet File
# Here we're loading in a Bro HTTP log with ~2 million rows to demonstrate the functionality and do some analysis and clustering on the data. For more information on converting Bro logs to Parquet files please see our Bro to Parquet notebook:
# 
# #### Bro logs to Parquet Notebook
# - [Bro to Parquet to Spark](https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Parquet_to_Spark.ipynb)
# 

# Have Spark read in the Parquet File
spark_df = spark.read.parquet("dns.parquet")


# <div style="float: left; margin: 20px 20px 20px 20px"><img src="images/eyeball.jpeg" width="150px"></div>
# # Lets look at our data
# We should always inspect out data when it comes in. Look at both the data values and the data types to make sure you're getting exactly what you should be.
# 

# Get information about the Spark DataFrame
num_rows = spark_df.count()
print("Number of Rows: {:d}".format(num_rows))
columns = spark_df.columns
print("Columns: {:s}".format(','.join(columns)))


spark_df.groupby('qtype_name','proto').count().sort('count', ascending=False).show()


# <div style="float: right; margin: 50px 0px 0px 20px"><img src="images/deep_dive.jpeg" width="350px"></div>
# 
# # Data looks good, lets take a deeper dive
# Spark has a powerful SQL engine as well as a Machine Learning library. So now that we've loaded our Bro data we're going to utilize the Spark SQL commands to do some investigation of our data including clustering from the MLLib.
# 
# <div style="float: left; margin: 20px 0px 0px 0px"><img src="images/spark_sql.jpg" width="180px"></div>
# <div style="float: left; margin: 0px 50px 0px 0px"><img src="images/mllib.png" width="180px"></div>
# 

# Add a column with the string length of the DNS query
from pyspark.sql.functions import col, length

# Create new dataframe that includes two new column
spark_df = spark_df.withColumn('query_length', length(col('query')))
spark_df = spark_df.withColumn('answer_length', length(col('answers')))


# Plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from bat.utils import plot_utils
plot_utils.plot_defaults()


# Show histogram of the Spark DF request body lengths
bins, counts = spark_df.select('query_length').rdd.flatMap(lambda x: x).histogram(50)

# This is a bit awkward but I believe this is the correct way to do it
plt.hist(bins[:-1], bins=bins, weights=counts, log=True)
plt.grid(True)
plt.xlabel('DNS Query Lengths')
plt.ylabel('Counts')


# Show histogram of the Spark DF request body lengths
bins, counts = spark_df.select('answer_length').rdd.flatMap(lambda x: x).histogram(50)

# This is a bit awkward but I believe this is the correct way to do it
plt.hist(bins[:-1], bins=bins, weights=counts, log=True)
plt.grid(True)
plt.xlabel('DNS Answer Lengths')
plt.ylabel('Counts')


from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

categoricalColumns = ['qtype_name', 'proto']
stages = [] 

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, 
        outputCol=categoricalCol+"Index")
    encoder = OneHotEncoder(inputCol=categoricalCol+"Index", 
        outputCol=categoricalCol+"classVec")
    stages += [stringIndexer, encoder]

numericCols = ['query_length', 'answer_length', 'Z', 'rejected']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(spark_df)
spark_df = pipelineModel.transform(spark_df)


spark_df.select('features').show()


from pyspark.ml.clustering import KMeans

# Trains a k-means model.
kmeans = KMeans().setK(70)
model = kmeans.fit(spark_df)


# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(spark_df)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


features = ['qtype_name', 'proto', 'query_length', 'answer_length', 'Z', 'rejected']
transformed = model.transform(spark_df).select(features + ['prediction'])
transformed.collect()
transformed.show()


transformed.groupby(features + ['prediction']).count().sort('prediction').show(50)


# # More Coming...
# 

# <div style="float: right; margin: 50px 0px 0px -20px"><img src="https://www.kitware.com/img/small_logo_over.png" width="250px"></div>
# ## Wrap Up
# Well that's it for this notebook, we pulled in Bro data from a Parquet file, then did some digging with high speed, parallel SQL operations and we clustered our data to organize the restuls.
# 
# If you liked this notebook please visit the [BAT](https://github.com/Kitware/bat) project for more notebooks and examples.
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/bro.png" width="100px"></div>
# 
# # Bro to Parquet to Spark
# Apache Parquet is a columnar storage format focused on performance. Parquet data is often used within the Hadoop ecosystem and we will specifically be using it for loading data into both Pandas and Spark.
# 
# <div style="float: right; margin: 30px -100px 0px 0px"><img src="images/parquet.png" width="300px"></div>
# 
# ### Software
# - Bro Analysis Tools (BAT): https://github.com/Kitware/bat
# - Pandas: https://github.com/pandas-dev/pandas
# - Parquet: https://parquet.apache.org
# - Spark: https://spark.apache.org
# 
# <div style="float: right; margin: 30px 0px 0px 0px"><img src="images/spark.png" width="200px"></div>
# 
# ### Data
# - Sec Repo: http://www.secrepo.com (there's no Bro headers on these)
# - Kitware: https://data.kitware.com/#collection/58d564478d777f0aef5d893a (with headers)
# 
# <div style="float: left; margin: 80px 20px 50px 20px"><img src="images/bleeding.jpg" width="250px"></div>
# ### Bleeding Edge Warning:
# You know you're on the bleeding edge when you link PRs that are still open/in-progess. There are **two open issues** with saving Parquet Files right now.
# 
# - Timestamps in Spark: https://issues.apache.org/jira/browse/ARROW-1499
# - TimeDelta Support: https://issues.apache.org/jira/browse/ARROW-835
# 
# For Spark timestamps, the BAT Parquet writer used below will output INT96 timestamps for now (we'll change over later when ARROW-1499 is complete). 
# 
# For the TimeDelta support we'll just have to wait until that gets pushed into the main branch and released.
# 

# Third Party Imports
import pyspark
from pyspark.sql import SparkSession
import pyarrow

# Local imports
import bat
from bat.log_to_parquet import log_to_parquet

# Good to print out versions of stuff
print('BAT: {:s}'.format(bat.__version__))
print('PySpark: {:s}'.format(pyspark.__version__))
print('PyArrow: {:s}'.format(pyarrow.__version__))


# ## Bro log to Parquet File
# Here we're loading in a Bro HTTP log with ~2 million rows to demonstrate the functionality and do some simple spark processing on the data.
# - log_to_parquet is iterative so it can handle large files
# - 'row_group_size' defaults to 1 Million rows but can be set manually
# 

# Create a Parquet file from a Bro Log with a super nice BAT method.
log_to_parquet('/Users/briford/data/bro/sec_repo/http.log', 'http.parquet')


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/compressed.jpeg" width="300px"></div>
# 
# # Parquet files are compressed
# Here we see the first benefit of Parquet which stores data with compressed columnar format. There are several compression options available (including uncompressed).
# 
# ## Original http.log = 1.3 GB 
# ## http.parquet = 106 MB
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark.png" width="200px"></div>
# 
# # Spark It!
# ### Spin up Spark with 4 Parallel Executors
# Here we're spinning up a local spark server with 4 parallel executors, although this might seem a bit silly since we're probably running this on a laptop, there are a couple of important observations:
# 
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark_jobs.png" width="400px"></div>
# 
# - If you have 4/8 cores use them!
# - It's the exact same code logic as if we were running on a distributed cluster.
# - We run the same code on **DataBricks** (www.databricks.com) which is awesome BTW.
# 
# 

# Spin up a local Spark Session (with 4 executors)
spark = SparkSession.builder.master('local[4]').appName('my_awesome').getOrCreate()


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/fast.jpg" width="350px"></div>
# 
# # Parquet files are fast
# We see from the below timer output that the Parquet file only takes a few seconds to read into Spark.
# 

# Have Spark read in the Parquet File
get_ipython().magic('time spark_df = spark.read.parquet("http.parquet")')


# <div style="float: right; margin: 0px 0px 0px -80px"><img src="images/spark_distributed.png" width="500px"></div>
# 
# # Parquet files are Parallel
# We see that, in this case, the number of data partitions in our dataframe(rdd) equals the number of executors/workers. If we had 8 workers there would be 8 partitions (at least, often there are more partitions based on how big the data is and how the files were writen, etc). 
# 
# 
# **Image Credit:** Jacek Laskowski, please see his excellent book - Mastering Apache Spark  https://jaceklaskowski.gitbooks.io/mastering-apache-spark
# 

spark_df.rdd.getNumPartitions()


# <div style="float: left; margin: 20px 20px 20px 20px"><img src="images/eyeball.jpeg" width="150px"></div>
# # Lets look at our data
# We should always inspect out data when it comes in. Look at both the data values and the data types to make sure you're getting exactly what you should be.
# 

# Get information about the Spark DataFrame
num_rows = spark_df.count()
print("Number of Rows: {:d}".format(num_rows))
columns = spark_df.columns
print("Columns: {:s}".format(','.join(columns)))


spark_df.select(['`id.orig_h`', 'host', 'uri', 'status_code', 'user_agent']).show(5)


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/fast.jpg" width="350px"></div>
# 
# # Did we mention fast?
# The query below was executed on 4 workers. The data contains over 2 million HTTP requests/responses and the time to complete was **less than 1 second**. All this code is running on a 2016 Mac Laptop :)
# 

get_ipython().magic("time spark_df.groupby('method','status_code').count().sort('count', ascending=False).show()")


# <div style="float: right; margin: 50px 0px 0px 20px"><img src="images/deep_dive.jpeg" width="350px"></div>
# 
# # Data looks good, lets take a deeper dive
# Spark has a powerful SQL engine as well as a Machine Learning library. So now that we've got the data loaded into Parquet we're going to utilize the Spark SQL commands to do some investigation and clustering using the Spark MLLib. For this deeper dive we're going to go to another notebook :)
# 
# ### Spark Clustering Notebook
# - [Bro Spark Clustering](https://github.com/Kitware/bat/blob/master/notebooks/Spark_Clustering.ipynb)
# 
# <div style="float: left; margin: 0px 0px 0px 0px"><img src="images/spark_sql.jpg" width="150px"></div>
# <div style="float: left; margin: -20px 50px 0px 0px"><img src="images/mllib.png" width="150px"></div>
# 

# <div style="float: right; margin: 50px 0px 0px -20px"><img src="https://www.kitware.com/img/small_logo_over.png" width="250px"></div>
# ## Wrap Up
# Well that's it for this notebook, we went from a Bro log to a high performance Parquet file and then did some digging with high speed, parallel SQL and groupby operations.
# 
# If you liked this notebook please visit the [BAT](https://github.com/Kitware/bat) project for more notebooks and examples.
# 

# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/fish.jpg" width="280px"></div>
# 
# ## Anomaly Exploration (understanding 'Odd')
# In this notebook we're going to be using the bat Python module for processing, transformation and anomaly detection on Bro network data. We're going to look at 'normal' http traffic and demonstrate the use of Isolation Forests for anomaly detection. We'll then explore those anomalies with clustering and PCA.
# 
# **Software**
# - bat: https://github.com/Kitware/bat
# - Pandas: https://github.com/pandas-dev/pandas
# - Scikit-Learn: http://scikit-learn.org/stable/index.html
# 
# **Techniques**
# - One Hot Encoding: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
# - Isolation Forest: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
# - PCA: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# 
# **Related Notebooks**
# - Bro to Scikit-Learn: https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Scikit_Learn.ipynb
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="https://www.kitware.com/img/small_logo_over.png"></div>
# 
# **Note:** A previous version of this notebook used a large http log (1 million rows) but we wanted people to be able to run the notebook themselves, so we've changed it to run on the local example http.log.
# 

import bat
from bat import log_to_dataframe
from bat import dataframe_to_matrix
print('bat: {:s}'.format(bat.__version__))
import pandas as pd
print('Pandas: {:s}'.format(pd.__version__))
import numpy as np
print('Numpy: {:s}'.format(np.__version__))
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
print('Scikit Learn Version:', sklearn.__version__)


# Create a Pandas dataframe from the Bro HTTP log
bro_df = log_to_dataframe.LogToDataFrame('../data/http.log')
print('Read in {:d} Rows...'.format(len(bro_df)))
bro_df.head()


# <div style="float: left; margin: 20px -10px 0px -40px"><img src="images/confused.jpg" width="200px"></div>
# <div style="float: right; margin: 20px -10px 0px -10px"><img src="images/pandas.png" width="300px"></div>
# ## So... what just happened?
# **Yep it was quick... the two little lines of code above turned a Bro log (any log) into a Pandas DataFrame. The bat package also supports streaming data from dynamic/active logs, handles log rotations and in general tries to make your life a bit easier when doing data analysis and machine learning on Bro data.**
# 
# **Now that we have the data in a dataframe there are a million wonderful things we could do for data munging, processing and analysis but that will have to wait for another time/notebook.**
# 

# We're going to pick some features that might be interesting
# some of the features are numerical and some are categorical
features = ['id.resp_p', 'method', 'resp_mime_types', 'request_body_len']


# ## Our HTTP features are a mix of numeric and categorical data
# When we look at the http records some of the data is numerical and some of it is categorical so we'll need a way of handling both data types in a generalized way. bat has a DataFrameToMatrix class that handles a lot of the details and mechanics of combining numerical and categorical data, we'll use below.
# 

# Show the dataframe with mixed feature types
bro_df[features].head()


# <div style="float: right; margin: -10px 40px -10px 40px"><img src="images/transformers.png" width="200px"></div>
# ## Transformers
# **We'll now use a scikit-learn tranformer class to convert the Pandas DataFrame to a numpy ndarray (matrix). Yes it's awesome... I'm not sure it's Optimus Prime awesome.. but it's still pretty nice.**
# 

# Use the bat DataframeToMatrix class (handles categorical data)
# You can see below it uses a heuristic to detect category data. When doing
# this for real we should explicitly convert before sending to the transformer.
to_matrix = dataframe_to_matrix.DataFrameToMatrix()
bro_matrix = to_matrix.fit_transform(bro_df[features], normalize=True)
print(bro_matrix.shape)
bro_matrix[:1]


# Train/fit and Predict anomalous instances using the Isolation Forest model
odd_clf = IsolationForest(contamination=0.20) # Marking 20% odd
odd_clf.fit(bro_matrix)


# Now we create a new dataframe using the prediction from our classifier
odd_df = bro_df[features][odd_clf.predict(bro_matrix) == -1]
print(odd_df.shape)
odd_df.head()


# Now we're going to explore our odd dataframe with help from KMeans and PCA algorithms
odd_matrix = to_matrix.fit_transform(odd_df)


# Just some simple stuff for this example, KMeans and PCA
kmeans = KMeans(n_clusters=4).fit_predict(odd_matrix)  # Change this to 3/5 for fun
pca = PCA(n_components=3).fit_transform(odd_matrix)

# Now we can put our ML results back onto our dataframe!
odd_df['x'] = pca[:, 0] # PCA X Column
odd_df['y'] = pca[:, 1] # PCA Y Column
odd_df['cluster'] = kmeans
odd_df.head()


# Plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14.0
plt.rcParams['figure.figsize'] = 15.0, 6.0

# Helper method for scatter/beeswarm plot
def jitter(arr):
    stdev = .02*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


# Jitter so we can see instances that are projected coincident in 2D
odd_df['jx'] = jitter(odd_df['x'])
odd_df['jy'] = jitter(odd_df['y'])

# Now use dataframe group by cluster
cluster_groups = odd_df.groupby('cluster')

# Plot the Machine Learning results
colors = {0:'green', 1:'blue', 2:'red', 3:'orange', 4:'purple', 5:'brown'}
fig, ax = plt.subplots()
for key, group in cluster_groups:
    group.plot(ax=ax, kind='scatter', x='jx', y='jy', alpha=0.5, s=250,
               label='Cluster: {:d}'.format(key), color=colors[key])


# Now print out the details for each cluster
pd.set_option('display.width', 1000)
for key, group in cluster_groups:
    print('\nCluster {:d}: {:d} observations'.format(key, len(group)))
    print(group[features].head())


# #### <div style="float: right; margin: 10px 10px 10px 10px"><img src="images/deep_dive.jpeg" width="250px"></div>
# ## Categorical variables that are anomalous
# - Cluster 0: http method of OPTIONS (instead of normal GET/POST)
# - Cluster 1: application/x-dosexec mime_types
# - Cluster 3: response port of 8080 (instead of 80)
# 
# ## Numerical variable outliers
# - Cluster 2: The request_body_len values are outliers (for this demo dataset)
# 
# **The important thing here is that both categorical and numerical variables were properly handled and the machine learning algorithm 'did the right thing' when marking outliers (for categorical and numerical fields)**
# 

# Distribution of the request body length
bro_df[['request_body_len']].hist()
print('\nFor this small demo dataset almost all request_body_len are 0\nCluster 2 represents outliers')


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/why_normal.jpg" width="200px"></div>
# 
# ## The anomalies identified by the model might be fine/expected
# Looking at the anomalous clusters for this small demo http log reveals four clusters that may be perfectly fine.  So
# here we're not equating anomalous with 'bad'. The use of an anomaly detection algorithm can bring latent issues to the attention of threat hunters and system administrations. The results might be expected or a misconfigured appliance or something more nefarious that needs attention from security.
# 
# 
# If you liked this notebook please visit the [bat](https://github.com/Kitware/bat) project for more notebooks and examples.
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="https://www.kitware.com/img/small_logo_over.png"></div>
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/bro.png" width="150px"></div>
# 
# # Bro Network Data to Scikit-Learn
# In this notebook we're going to be using the bat Python module and explore the functionality that enables us to easily go from Bro data to Pandas to Scikit-Learn. Once we get our data in a form that is usable by Scikit-Learn we have a wide array of data analysis and machine learning algorithms at our disposal.
# 
# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/pandas.png" width="300px"></div>
# 
# ### Software
# - bat: https://github.com/Kitware/bat
# - Pandas: https://github.com/pandas-dev/pandas
# - Scikit-Learn: http://scikit-learn.org/stable/index.html
# 
# ### Techniques
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/scikit.png" width="220px"></div>
# 
# - One Hot Encoding: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
# - t-SNE: https://distill.pub/2016/misread-tsne/
# - Kmeans: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# 
# ### Thanks
# - DummyEncoder class, used internally as part of DataFrameToMatrix(), is based on a great talk from Tom Augspurger's at PyData Chicago 2016: https://youtu.be/KLPtEBokqQ0
# 
# ### Code Availability
# All this code in this notebook is from the examples/bro_to_scikit.py file in the bat repository (https://github.com/Kitware/bat). If you have any questions/problems please don't hesitate to open up an Issue in GitHub or even better submit a PR. :) 
# 
# <div style="float: left; margin: 20px 20px 20px 20px"><img src="https://www.kitware.com/img/small_logo_over.png"></div>
# 

# Third Party Imports
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Local imports
import bat
from bat import log_to_dataframe
from bat import dataframe_to_matrix

# Good to print out versions of stuff
print('bat: {:s}'.format(bat.__version__))
print('Pandas: {:s}'.format(pd.__version__))
print('Numpy: {:s}'.format(np.__version__))
print('Scikit Learn Version:', sklearn.__version__)


# ## Quickly go from Bro log to Pandas DataFrame
# 

# Create a Pandas dataframe from a Bro log
bro_df = log_to_dataframe.LogToDataFrame('../data/dns.log')

# Print out the head of the dataframe
bro_df.head()


# <div style="float: left; margin: 20px -10px 0px -40px"><img src="images/confused.jpg" width="200px"></div>
# <div style="float: right; margin: 20px -10px 0px -10px"><img src="images/pandas.png" width="300px"></div>
# ## So... what just happened?
# **Yep it was quick... the two little lines of code above turned a Bro log (any log) into a Pandas DataFrame. The bat package also supports streaming data from dynamic/active logs, handles log rotations and in general tries to make your life a bit easier when doing data analysis and machine learning on Bro data.**
# 
# **Now that we have the data in a dataframe there are a million wonderful things we could do for data munging, processing and analysis but that will have to wait for another time/notebook.**
# 

# Using Pandas we can easily and efficiently compute additional data metrics
# Here we use the vectorized operations of Pandas/Numpy to compute query length
bro_df['query_length'] = bro_df['query'].str.len()


# ## DNS records are a mix of numeric and categorical data
# When we look at the dns records some of the data is numerical and some of it is categorical so we'll need a way of handling both data types in a generalized way. bat has a DataFrameToMatrix class that handles a lot of the details and mechanics of combining numerical and categorical data, we'll use below.
# 

# These are the features we want (note some of these are categorical :)
features = ['AA', 'RA', 'RD', 'TC', 'Z', 'rejected', 'proto', 'query', 
            'qclass_name', 'qtype_name', 'rcode_name', 'query_length']
feature_df = bro_df[features]
feature_df.head()


# <div style="float: right; margin: -10px 40px -10px 40px"><img src="images/transformers.png" width="200px"></div>
# ## Transformers
# **We'll now use a bat scikit-learn tranformer class to convert the Pandas DataFrame to a numpy ndarray (matrix). Yes it's awesome... I'm not sure it's Optimus Prime awesome.. but it's still pretty nice.**
# 

# Use the bat DataframeToMatrix class (handles categorical data)
# You can see below it uses a heuristic to detect category data. When doing
# this for real we should explicitly convert before sending to the transformer.
to_matrix = dataframe_to_matrix.DataFrameToMatrix()
bro_matrix = to_matrix.fit_transform(feature_df)


# Just showing that the class is tracking categoricals and normalization maps
print(to_matrix.cat_columns)
print(to_matrix.norm_map)


# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/rock.gif" width="150px"></div>
# <div style="float: left; margin: 10px 20px 10px 10px"><img src="images/scikit.png" width="200px"></div>
# 
# ## Scikit-Learn
# **Now that we have a numpy ndarray(matrix) we ready to rock with scikit-learn...**
# 

# Now we're ready for scikit-learn!
# Just some simple stuff for this example, KMeans and TSNE projection
kmeans = KMeans(n_clusters=5).fit_predict(bro_matrix)
projection = TSNE().fit_transform(bro_matrix)

# Now we can put our ML results back onto our dataframe!
bro_df['x'] = projection[:, 0] # Projection X Column
bro_df['y'] = projection[:, 1] # Projection Y Column
bro_df['cluster'] = kmeans
bro_df[['query', 'proto', 'x', 'y', 'cluster']].head()  # Showing the scikit-learn results in our dataframe


# Plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18.0
plt.rcParams['figure.figsize'] = 15.0, 7.0


# Now use dataframe group by cluster
cluster_groups = bro_df.groupby('cluster')

# Plot the Machine Learning results
fig, ax = plt.subplots()
colors = {0:'green', 1:'blue', 2:'red', 3:'orange', 4:'purple'}
for key, group in cluster_groups:
    group.plot(ax=ax, kind='scatter', x='x', y='y', alpha=0.5, s=250,
               label='Cluster: {:d}'.format(key), color=colors[key])


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/eyeball.jpeg" width="200px"></div>
# 
# ## Lets Investigate the 5 clusters of DNS data
# **We cramed a bunch of features into the clustering algorithm. The features were both numerical and categorical. So did the clustering 'do the right thing'? Well first some caveats and disclaimers:** 
# - We're obviously working with a small amount of Bro DNS data
# - This is an example to show how the tranformations work (from Bro to Pandas to Scikit)
# - The DNS data is real data but for this example and others we obviously pulled in 'weird' stuff on purpose
# - We knew that the K in KMeans should be 5 :)
# 
# **Okay will all those caveats lets look at how the clustering did on both numeric and categorical data combined**
# ### Cluster details
# - Cluster 0: (42 observations) Looks like 'normal' DNS requests
# - Cluster 1: (11 observations) All the queries are '-' (Bro for NA/not found/etc)
# - Cluster 2: ( 6 observations) The protocol is TCP instead of the normal UDP
# - Cluster 3: ( 4 observations) The reserved Z bit is set to 1 (required to be 0)
# - Cluster 4: ( 4 observations) All the DNS queries are exceptionally long
# 
# ## Numerical + Categorical = AOK
# With our example data we've successfully gone from Bro logs to Pandas to scikit-learn. The clusters appear to make sense and certainly from an investigative and threat hunting perspective being able to cluster the data and use PCA for dimensionality reduction might come in handy depending on your use case 
# 

# Now print out the details for each cluster
pd.set_option('display.width', 1000)
show_fields = ['query', 'Z', 'proto', 'qtype_name', 'cluster']
for key, group in cluster_groups:
    print('\nCluster {:d}: {:d} observations'.format(key, len(group)))
    print(group[show_fields].head())


# <div style="float: right; margin: 50px 0px 0px 0px"><img src="https://www.kitware.com/img/small_logo_over.png"></div>
# ## Wrap Up
# Well that's it for this notebook, we'll have an upcoming notebook that addresses some of the issues that we overlooked in this simple example. We use Isolation Forest for anomaly detection which works well for high dimensional data. The notebook will cover both training and streaming evalution against the model.
# 
# If you liked this notebook please visit the [bat](https://github.com/Kitware/bat) project for more notebooks and examples.
# 

# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/cheesy.jpg" width="250px"></div>
# 
# # Bro to Spark: Cheesy/Easy Way
# ** NOTE:** This is NOT the correct way to go from Bro to Spark. We're going to be using local data and a local Spark kernel which obviously won't scale at all. But if you just want to explore Spark with some smaller datasets this is a super **EASY** way to get started. 
# 
# All you need to install for this notebook/approach is:
# 
#     $ pip install bat pyspark 
# 
# For the correct (but more complicated) way please see our Bro to Spark notebooks:
# - https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Parquet_to_Spark.ipynb
# - https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Kafka_to_Spark.ipynb
# 
# <div style="float: right; margin: 0px 0px 0px 0px"><img src="images/bro.png" width="100px"></div>
# 
# You can test whether spark is installed correctly by starting up the spark shell.
#     
#     $ spark-shell
# 
# There are some warnings and stuff but if you get this you have successfully installed spark.
# You can quit the shell by typing ':quit' and the scala> prompt
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark.png" width="250px"></div>
# <div style="margin: 20px 20px 20px 20px"><img align="left" src="images/spark_shell.png" width="400px"></div>
# 

from pyspark.sql import SparkSession
from bat import log_to_dataframe
import pandas as pd


# Convert Bro log to Pandas DataFrame
dns_df = log_to_dataframe.LogToDataFrame('../data/dns.log')
dns_df.head()


# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark.png" width="200px"></div>
# 
# # Spark It!
# ### Spin up Spark with 4 Parallel Executors
# Here we're spinning up a local spark server with 4 parallel executors, although this might seem a bit silly since we're probably running this on a laptop, there are a couple of important observations:
# 
# <div style="float: right; margin: 20px 20px 20px 20px"><img src="images/spark_jobs.png" width="400px"></div>
# 
# - If you have 4/8 cores use them!
# - It's the exact same code logic as if we were running on a distributed cluster.
# - We run the same code on **DataBricks** (www.databricks.com) which is awesome BTW.
# 

# Spin up a local Spark Session (with 4 executors)
spark = SparkSession.builder.master('local[4]').appName('my_awesome').getOrCreate()


# Convert to Spark DF
spark_df = spark.createDataFrame(dns_df)


# Some simple spark operations
num_rows = spark_df.count()
print("Number of Spark DataFrame rows: {:d}".format(num_rows))
columns = spark_df.columns
print("Columns: {:s}".format(','.join(columns)))


# Some simple spark operations
spark_df.groupBy('proto').count().show()


# <div style="float: right; margin: 0px 0px 0px -30px"><img src="images/confused.jpg" width="150px"></div>
# ### Note: Spark/PySpark does not like column names with a '.' in them
# So for the fields like 'id.orig_h' we have to put the backticks around them ( \`id.orig_h\` )
# 

# Some simple spark operations
spark_df.groupBy('`id.orig_h`', '`id.resp_h`').count().show()


# Add a column with the string length of the DNS query
from pyspark.sql.functions import col, length

# Create new dataframe that includes new column
spark_df = spark_df.withColumn('query_length', length(col('query')))
spark_df[['query', 'query_length']].show()


# Plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from bat.utils import plot_utils
plot_utils.plot_defaults()


# Show histogram of the Spark DF query lengths
bins, counts = spark_df.select('query_length').rdd.flatMap(lambda x: x).histogram(20)

# This is a bit awkward but I believe this is the correct way to do it
plt.hist(bins[:-1], bins=bins, weights=counts)
plt.grid(True)
plt.xlabel('Query Lengths')
plt.ylabel('Counts')


# Compare the computation of query_length and resulting histogram with Pandas
dns_df['query_length'] = dns_df['query'].str.len()
dns_df['query_length'].hist(bins=20)
plt.xlabel('Query Lengths')
plt.ylabel('Counts')


# <div style="float: right; margin: 20px 0px 0px 0px"><img src="images/spark.png" width="150px"></div>
# 
# # That was easy.. is this the same as 'real' Spark?
# Yep, if you've gotten this far you are running a local instance of the Spark server with all the exact same functionality as any Spark cluster (minus the scalability that comes with lots of nodes obviously).
# 
# Check out your Spark jobs by simply going to http://localhost:4040
# 
# <div style="margin: 20px 0px 0px 0px"><img src="images/spark_jobs.png" width="600px"></div>
# 

# ## Wrap Up
# Well that's it for this notebook. With a few simple pip installs you are ready to try out Spark on your Bro Logs. Yes it will only work on smaller data but it gets you **'in the saddle'** quickly. You can try some stuff out, get familiar with Spark and then dive into setting it up the right way:
# <div style="float: right; margin: 0px 0px 0px 0px"><img src="https://www.kitware.com/img/small_logo_over.png" width="200px"></div>
# - https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Parquet_to_Spark.ipynb
# - https://github.com/Kitware/bat/blob/master/notebooks/Bro_to_Kafka_to_Spark.ipynb
# 
# If you liked this notebook please visit the [bat](https://github.com/Kitware/bat) project for more notebooks and examples.
# 

