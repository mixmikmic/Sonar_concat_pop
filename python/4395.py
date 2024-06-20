# ## Twitter + Watson Tone Analyzer Sample Notebook
# In this sample notebook, we show how to load and analyze data from the Twitter + Watson Tone Analyzer Spark sample application (code can be found here https://github.com/ibm-watson-data-lab/spark.samples/tree/master/streaming-twitter). The tweets data has been enriched with scores from various Sentiment Tone (e.g Anger, Cheerfulness, etc...).
# 

# Import SQLContext and data types
from pyspark.sql import SQLContext
from pyspark.sql.types import *


# ## Load the data
# In this section, we load the data from a parquet file that has been saved from a scala notebook (see tutorial here...) and create a SparkSQL DataFrame that contains all the data.
# 

parquetFile = sqlContext.read.parquet("swift://notebooks.spark/tweetsFull.parquet")
print parquetFile


parquetFile.registerTempTable("tweets");
sqlContext.cacheTable("tweets")
tweets = sqlContext.sql("SELECT * FROM tweets")
print tweets.count()
tweets.cache()


# ## Compute the distribution of tweets by sentiments > 60%
# In this section, we demonstrate how to use SparkSQL queries to compute for each tone that number of tweets that are greater than 60%
# 

#create an array that will hold the count for each sentiment
sentimentDistribution=[0] * 13
#For each sentiment, run a sql query that counts the number of tweets for which the sentiment score is greater than 60%
#Store the data in the array
for i, sentiment in enumerate(tweets.columns[-13:]):
    sentimentDistribution[i]=sqlContext.sql("SELECT count(*) as sentCount FROM tweets where " + sentiment + " > 60")        .collect()[0].sentCount


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ind=np.arange(13)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Tweet count')
plt.xlabel('Tone')
plt.title('Distribution of tweets by sentiments > 60%')
plt.xticks(ind+width, tweets.columns[-13:])
plt.legend()

plt.show()


from operator import add
import re
tagsRDD = tweets.flatMap( lambda t: re.split("\s", t.text))    .filter( lambda word: word.startswith("#") )    .map( lambda word : (word, 1 ))    .reduceByKey(add, 10).map(lambda (a,b): (b,a)).sortByKey(False).map(lambda (a,b):(b,a))
top10tags = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

print(top10tags)
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2, plSize[1]*2) )

labels = [i[0] for i in top10tags]
sizes = [int(i[1]) for i in top10tags]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', "beige", "paleturquoise", "pink", "lightyellow", "coral"]

plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.show()


# ## Breakdown of the top 5 hashtags by sentiment scores
# In this section, we demonstrate how to build a more complex analytic which decompose the top 5 hashtags by sentiment scores. The code below computes the mean of all the sentiment scores and visualize them in a multi-series bar chart
# 

cols = tweets.columns[-13:]
def expand( t ):
    ret = []
    for s in [i[0] for i in top10tags]:
        if ( s in t.text ):
            for tone in cols:
                ret += [s.replace(':','').replace('-','') + u"-" + unicode(tone) + ":" + unicode(getattr(t, tone))]
    return ret 
def makeList(l):
    return l if isinstance(l, list) else [l]

#Create RDD from tweets dataframe
tagsRDD = tweets.map(lambda t: t )

#Filter to only keep the entries that are in top10tags
tagsRDD = tagsRDD.filter( lambda t: any(s in t.text for s in [i[0] for i in top10tags] ) )

#Create a flatMap using the expand function defined above, this will be used to collect all the scores 
#for a particular tag with the following format: Tag-Tone-ToneScore
tagsRDD = tagsRDD.flatMap( expand )

#Create a map indexed by Tag-Tone keys 
tagsRDD = tagsRDD.map( lambda fullTag : (fullTag.split(":")[0], float( fullTag.split(":")[1]) ))

#Call combineByKey to format the data as follow
#Key=Tag-Tone
#Value=(count, sum_of_all_score_for_this_tone)
tagsRDD = tagsRDD.combineByKey((lambda x: (x,1)),
                  (lambda x, y: (x[0] + y, x[1] + 1)),
                  (lambda x, y: (x[0] + y[0], x[1] + y[1])))

#ReIndex the map to have the key be the Tag and value be (Tone, Average_score) tuple
#Key=Tag
#Value=(Tone, average_score)
tagsRDD = tagsRDD.map(lambda (key, ab): (key.split("-")[0], (key.split("-")[1], round(ab[0]/ab[1], 2))))

#Reduce the map on the Tag key, value becomes a list of (Tone,average_score) tuples
tagsRDD = tagsRDD.reduceByKey( lambda x, y : makeList(x) + makeList(y) )

#Sort the (Tone,average_score) tuples alphabetically by Tone
tagsRDD = tagsRDD.mapValues( lambda x : sorted(x) )

#Format the data as expected by the plotting code in the next cell. 
#map the Values to a tuple as follow: ([list of tone], [list of average score])
#e.g. #someTag:([u'Agreeableness', u'Analytical', u'Anger', u'Cheerfulness', u'Confident', u'Conscientiousness', u'Negative', u'Openness', u'Tentative'], [1.0, 0.0, 0.0, 1.0, 0.0, 0.48, 0.0, 0.02, 0.0])
tagsRDD = tagsRDD.mapValues( lambda x : ([elt[0] for elt in x],[elt[1] for elt in x])  )

#Use custom sort function to sort the entries by order of appearance in top10tags
def customCompare( key ):
    for (k,v) in top10tags:
        if k == key:
            return v
    return 0
tagsRDD = tagsRDD.sortByKey(ascending=False, numPartitions=None, keyfunc = customCompare)

#Take the mean tone scores for the top 10 tags
top10tagsMeanScores = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*3, plSize[1]*2) )

top5tagsMeanScores = top10tagsMeanScores[:5]
width = 0
ind=np.arange(13)
(a,b) = top5tagsMeanScores[0]
labels=b[0]
colors = ["beige", "paleturquoise", "pink", "lightyellow", "coral", "lightgreen", "gainsboro", "aquamarine","c"]
idx=0
for key, value in top5tagsMeanScores:
    plt.bar(ind + width, value[1], 0.15, color=colors[idx], label=key)
    width += 0.15
    idx += 1
plt.xticks(ind+0.3, labels)
plt.ylabel('AVERAGE SCORE')
plt.xlabel('TONES')
plt.title('Breakdown of top hashtags by sentiment tones')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',ncol=5, mode="expand", borderaxespad=0.)

plt.show()





# # Twitter Sentiment analysis with Watson Tone Analyzer and Watson Personality Insights
# 
# <img style="max-width: 800px; padding: 25px 0px;" src="https://ibm-watson-data-lab.github.io/spark.samples/Twitter%20Sentiment%20with%20Watson%20TA%20and%20PI%20architecture%20diagram.png"/>
# 
# In this notebook, we perform the following steps:  
# 1. Install python-twitter and watson-developer-cloud modules
# 2. Install the streaming Twitter jar using PixieDust packageManager
# 3. Invoke the streaming Twitter app using the PixieDust Scala Bridge to get a DataFrame containing all the tweets enriched with Watson Tone Analyzer scores
# 4. Create a new RDD that groups the tweets by author and concatenates all the associated tweets into one blob
# 5. For each author and aggregated text, invoke the Watson Personality Insights to get the scores
# 6. Visualize results using PixieDust display  
# 
# ## Learn more 
# * [Watson Tone Analyzer](http://www.ibm.com/watson/developercloud/tone-analyzer.html)  
# * [Watson Personality Insights](http://www.ibm.com/watson/developercloud/personality-insights.html)  
# * [python-twitter](https://github.com/bear/python-twitter)  
# * [watson-developer-cloud](https://github.com/watson-developer-cloud)  
# * [PixieDust](https://github.com/ibm-watson-data-lab/pixiedust)
# * [Realtime Sentiment Analysis of Twitter Hashtags with Spark](https://developer.ibm.com/clouddataservices/2016/01/15/real-time-sentiment-analysis-of-twitter-hashtags-with-spark)
# 

# # Install python-twitter and watson-developer-cloud
# If you haven't already installed the following modules, run these 2 cells:
# 

get_ipython().system('pip install --user python-twitter')


get_ipython().system('pip install --user watson-developer-cloud')


# # Install latest pixiedust
# Make sure you are running the latest pixiedust version. After upgrading restart the kernel before continuing to the next cells.
# 

get_ipython().system('pip install --upgrade --user pixiedust')


# ## Install the streaming Twitter jar in the notebook from the Github repo
# This jar file contains the Spark Streaming application (written in Scala) that connects to Twitter to fetch the tweets and send them to Watson Tone Analyzer for analysis. The resulting scores are then added to the tweets dataframe as separate columns.
# 

import pixiedust
jarPath = "https://github.com/ibm-watson-data-lab/spark.samples/raw/master/dist/streaming-twitter-assembly-1.6.jar"
pixiedust.installPackage(jarPath)
print("done")


# <h3>If PixieDust or the streaming Twitter jar were just installed or upgraded, <span style="color: red">restart the kernel</span> before continuing.</h3>
# 

# ## Use Scala Bridge to run the command line version of the app
# Insert your credentials for Twitter, Watson Tone Analyzer, and Watson Personality Insights. Then run the following cell. 
# [Read how to provision these services and get credentials](https://github.com/ibm-watson-data-lab/spark.samples/blob/master/notebook/Get%20Service%20Credentials%20for%20Twitter%20Sentiment%20with%20Watson%20TA%20and%20PI.md). 
# 

import pixiedust

sqlContext=SQLContext(sc)

#Set up the twitter credentials, they will be used both in scala and python cells below
consumerKey = "XXXX"
consumerSecret = "XXXX"
accessToken = "XXXX"
accessTokenSecret = "XXXX"

#Set up the Watson Personality insight credentials
piUserName = "XXXX"
piPassword = "XXXX"

#Set up the Watson Tone Analyzer credentials
taUserName = "XXXX"
taPassword = "XXXX"


get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.StreamingTwitter\ndemo.setConfig("twitter4j.oauth.consumerKey",consumerKey)\ndemo.setConfig("twitter4j.oauth.consumerSecret",consumerSecret)\ndemo.setConfig("twitter4j.oauth.accessToken",accessToken)\ndemo.setConfig("twitter4j.oauth.accessTokenSecret",accessTokenSecret)\ndemo.setConfig("watson.tone.url","https://gateway.watsonplatform.net/tone-analyzer/api")\ndemo.setConfig("watson.tone.password",taPassword)\ndemo.setConfig("watson.tone.username",taUserName)\n\nimport org.apache.spark.streaming._\ndemo.startTwitterStreaming(sc, Seconds(30))  //Run the application for a limited time')


# # Create a tweets dataframe from the data fetched above and transfer it to Python
# Notice the __ prefix for each variable which is used to signal PixieDust that the variable needs to be transfered back to Python
# 

get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.StreamingTwitter\nval (__sqlContext, __df) = demo.createTwitterDataFrames(sc)')


# ## Group the tweets by author and userid
# This will be used later to fetch the last 200 tweets for each author
# 

import pyspark.sql.functions as F
usersDF = __df.groupby("author", "userid").agg(F.avg("Anger").alias("Anger"), F.avg("Disgust").alias("Disgust"))
usersDF.show()


# # Set up the Twitter API from python-twitter module
# 

import twitter
api = twitter.Api(consumer_key=consumerKey,
                  consumer_secret=consumerSecret,
                  access_token_key=accessToken,
                  access_token_secret=accessTokenSecret)

#print(api.VerifyCredentials())


# # For each author, fetch the last 200 tweets
# use flatMap to return a new RDD that contains a list of tuples composed of userid and tweets text: (userid, tweetText)
# 

def getTweets(screenName):
    statuses = api.GetUserTimeline(screen_name=screenName,
                        since_id=None,
                        max_id=None,
                        count=200,
                        include_rts=False,
                        trim_user=False,
                        exclude_replies=True)
    return statuses

usersWithTweetsRDD = usersDF.flatMap(lambda s: [(s.user.screen_name, s.text.encode('ascii', 'ignore')) for s in getTweets(s['userid'])])
print(usersWithTweetsRDD.count())


# # Concatenate all the tweets for each user so we have enough words to send to Watson Personality Insights
# * Use map to create an RDD of key, value pair composed of userId and tweets 
# * Use reduceByKey to group all record with same author and concatenate the tweets
# 

import re
usersWithTweetsRDD2 = usersWithTweetsRDD.map(lambda s: (s[0], s[1])).reduceByKey(lambda s,t: s + '\n' + t)    .filter(lambda s: len(re.findall(r'\w+', s[1])) > 100 )
print(usersWithTweetsRDD2.count())
#usersWithTweetsRDD2.take(2)


# # Call Watson Personality Insights on the text for each author
# Watson Personality Insights requires at least 100 words from its lexicon to be available, which may not exist for each user. This is why the getPersonlityInsight helper function guards against exceptions from calling Watson PI. If an exception occurs, then an empty array is returned. Each record with empty array is filtered out of the resulting RDD.
# 
# Note also that we use broadcast variables to propagate the userName and password to the cluster
# 

from pyspark.sql.types import *
from watson_developer_cloud import PersonalityInsightsV3
broadCastPIUsername = sc.broadcast(piUserName)
broadCastPIPassword = sc.broadcast(piPassword)
def getPersonalityInsight(text, schema=False):
    personality_insights = PersonalityInsightsV3(
          version='2016-10-20',
          username=broadCastPIUsername.value,
          password=broadCastPIPassword.value)
    try:
        p = personality_insights.profile(
            text, content_type='text/plain',
            raw_scores=True, consumption_preferences=True)

        if schema:
            return                 [StructField(t['name'], FloatType()) for t in p["needs"]] +                 [StructField(t['name'], FloatType()) for t in p["values"]] +                 [StructField(t['name'], FloatType()) for t in p['personality' ]]
        else:
            return                 [t['raw_score'] for t in p["needs"]] +                 [t['raw_score'] for t in p["values"]] +                 [t['raw_score'] for t in p['personality']]   
    except:
        return []

usersWithPIRDD = usersWithTweetsRDD2.map(lambda s: [s[0]] + getPersonalityInsight(s[1])).filter(lambda s: len(s)>1)
print(usersWithPIRDD.count())
#usersWithPIRDD.take(2)


# # Convert the RDD back to a DataFrame and call PixieDust display to visualize the results
# The schema is automatically created from introspecting a sample payload result from Watson Personality Insights
# 

#convert to dataframe
schema = StructType(
    [StructField('userid',StringType())] + getPersonalityInsight(usersWithTweetsRDD2.take(1)[0][1], schema=True)
)

usersWithPIDF = sqlContext.createDataFrame(
    usersWithPIRDD, schema
)

usersWithPIDF.cache()
display(usersWithPIDF)


# # Compare Twitter users Personality Insights scores with this year presidential candidates
# 
# For a quick look on the difference in Personality Insights scores Spark provides a describe() function that computes stddev and mean values off the dataframe. Compare differences in the scores of twitter users and presidential candidates.
# 

candidates = "realDonaldTrump HillaryClinton".split(" ")
candidatesRDD = sc.parallelize(candidates)    .flatMap(lambda s: [(t.user.screen_name, t.text.encode('ascii', 'ignore')) for t in getTweets(s)])    .map(lambda s: (s[0], s[1]))    .reduceByKey(lambda s,t: s + '\n' + t)    .filter(lambda s: len(re.findall(r'\w+', s[1])) > 100 )    .map(lambda s: [s[0]] + getPersonalityInsight(s[1]))

candidatesPIDF = sqlContext.createDataFrame(
   candidatesRDD, schema
)


c = candidatesPIDF.collect()
broadCastTrumpPI = sc.broadcast(c[0][1:])
broadCastHillaryPI = sc.broadcast(c[1][1:])


display(candidatesPIDF)


candidatesPIDF.select('userid','Emotional range','Agreeableness', 'Extraversion','Conscientiousness', 'Openness').show()

usersWithPIDF.describe(['Emotional range']).show()
usersWithPIDF.describe(['Agreeableness']).show()
usersWithPIDF.describe(['Extraversion']).show()
usersWithPIDF.describe(['Conscientiousness']).show()
usersWithPIDF.describe(['Openness']).show()


# # Calculate Euclidean distance (norm) between each Twitter user and the presidential candidates using the Personality Insights scores
# 
# Add the distances into 2 extra columns and display the results
# 

import numpy as np
from pyspark.sql.types import Row
def addEuclideanDistance(s):
    dict = s.asDict()
    def getEuclideanDistance(a,b):
        return np.linalg.norm(np.array(a) - np.array(b)).item()
    dict["distDonaldTrump"]=getEuclideanDistance(s[1:], broadCastTrumpPI.value)
    dict["distHillary"]=getEuclideanDistance(s[1:], broadCastHillaryPI.value)
    dict["closerHillary"] = "Yes" if dict["distHillary"] < dict["distDonaldTrump"] else "No"
    return Row(**dict)

#add euclidean distances to Trump and Hillary
euclideanDF = sqlContext.createDataFrame(usersWithPIDF.map(lambda s: addEuclideanDistance(s)))

#Reorder columns to have userid and distances first
cols = euclideanDF.columns
reorderCols = ["userid","distHillary","distDonaldTrump", "closerHillary"]
euclideanDF = euclideanDF.select(reorderCols + [x for x in cols if x not in reorderCols])

#PixieDust display. 
#To visualize the distribution, select the bar chart display, use closerHillary as key and value and aggregation=count
display(euclideanDF)


# # Optional: do some extra data science on the tweets
# 

tweets=__df
tweets.count()
display(tweets)


# # Compute the sentiment distributions for tweets with scores greater than 60% and create matplotlib chart visualization
# 

#create an array that will hold the count for each sentiment
sentimentDistribution=[0] * 13
#For each sentiment, run a sql query that counts the number of tweets for which the sentiment score is greater than 60%
#Store the data in the array
for i, sentiment in enumerate(tweets.columns[-13:]):
    sentimentDistribution[i]=__sqlContext.sql("SELECT count(*) as sentCount FROM tweets where " + sentiment + " > 60")        .collect()[0].sentCount


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ind=np.arange(13)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Tweet count')
plt.xlabel('Tone')
plt.title('Distribution of tweets by sentiments > 60%')
plt.xticks(ind+width, tweets.columns[-13:])
plt.legend()

plt.show()


# # Compute the top hashtags used in each tweet
# 

from operator import add
import re
tagsRDD = tweets.flatMap( lambda t: re.split("\s", t.text))    .filter( lambda word: word.startswith("#") )    .map( lambda word : (word, 1 ))    .reduceByKey(add, 10).map(lambda (a,b): (b,a)).sortByKey(False).map(lambda (a,b):(b,a))
top10tags = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2, plSize[1]*2) )

labels = [i[0] for i in top10tags]
sizes = [int(i[1]) for i in top10tags]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', "beige", "paleturquoise", "pink", "lightyellow", "coral"]

plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


# # Compute the aggregate sentiment distribution for all the tweets that contain the top hashtags
# 

cols = tweets.columns[-13:]
def expand( t ):
    ret = []
    for s in [i[0] for i in top10tags]:
        if ( s in t.text ):
            for tone in cols:
                ret += [s.replace(':','').replace('-','') + u"-" + unicode(tone) + ":" + unicode(getattr(t, tone))]
    return ret 
def makeList(l):
    return l if isinstance(l, list) else [l]

#Create RDD from tweets dataframe
tagsRDD = tweets.map(lambda t: t )

#Filter to only keep the entries that are in top10tags
tagsRDD = tagsRDD.filter( lambda t: any(s in t.text for s in [i[0] for i in top10tags] ) )

#Create a flatMap using the expand function defined above, this will be used to collect all the scores 
#for a particular tag with the following format: Tag-Tone-ToneScore
tagsRDD = tagsRDD.flatMap( expand )

#Create a map indexed by Tag-Tone keys 
tagsRDD = tagsRDD.map( lambda fullTag : (fullTag.split(":")[0], float( fullTag.split(":")[1]) ))

#Call combineByKey to format the data as follow
#Key=Tag-Tone
#Value=(count, sum_of_all_score_for_this_tone)
tagsRDD = tagsRDD.combineByKey((lambda x: (x,1)),
                  (lambda x, y: (x[0] + y, x[1] + 1)),
                  (lambda x, y: (x[0] + y[0], x[1] + y[1])))

#ReIndex the map to have the key be the Tag and value be (Tone, Average_score) tuple
#Key=Tag
#Value=(Tone, average_score)
tagsRDD = tagsRDD.map(lambda (key, ab): (key.split("-")[0], (key.split("-")[1], round(ab[0]/ab[1], 2))))

#Reduce the map on the Tag key, value becomes a list of (Tone,average_score) tuples
tagsRDD = tagsRDD.reduceByKey( lambda x, y : makeList(x) + makeList(y) )

#Sort the (Tone,average_score) tuples alphabetically by Tone
tagsRDD = tagsRDD.mapValues( lambda x : sorted(x) )

#Format the data as expected by the plotting code in the next cell. 
#map the Values to a tuple as follow: ([list of tone], [list of average score])
#e.g. #someTag:([u'Agreeableness', u'Analytical', u'Anger', u'Cheerfulness', u'Confident', u'Conscientiousness', u'Negative', u'Openness', u'Tentative'], [1.0, 0.0, 0.0, 1.0, 0.0, 0.48, 0.0, 0.02, 0.0])
tagsRDD = tagsRDD.mapValues( lambda x : ([elt[0] for elt in x],[elt[1] for elt in x])  )

#Use custom sort function to sort the entries by order of appearance in top10tags
def customCompare( key ):
    for (k,v) in top10tags:
        if k == key:
            return v
    return 0
tagsRDD = tagsRDD.sortByKey(ascending=False, numPartitions=None, keyfunc = customCompare)

#Take the mean tone scores for the top 10 tags
top10tagsMeanScores = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*3, plSize[1]*2) )

top5tagsMeanScores = top10tagsMeanScores[:5]
width = 0
ind=np.arange(13)
(a,b) = top5tagsMeanScores[0]
labels=b[0]
colors = ["beige", "paleturquoise", "pink", "lightyellow", "coral", "lightgreen", "gainsboro", "aquamarine","c"]
idx=0
for key, value in top5tagsMeanScores:
    plt.bar(ind + width, value[1], 0.15, color=colors[idx], label=key)
    width += 0.15
    idx += 1
plt.xticks(ind+0.3, labels)
plt.ylabel('AVERAGE SCORE')
plt.xlabel('TONES')
plt.title('Breakdown of top hashtags by sentiment tones')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',ncol=5, mode="expand", borderaxespad=0.)

plt.show()


# # Optional: Use Twitter demo embedded app to run the same app with a UI
# 

get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.PixiedustStreamingTwitter\ndemo.setConfig("twitter4j.oauth.consumerKey",consumerKey)\ndemo.setConfig("twitter4j.oauth.consumerSecret",consumerSecret)\ndemo.setConfig("twitter4j.oauth.accessToken",accessToken)\ndemo.setConfig("twitter4j.oauth.accessTokenSecret",accessTokenSecret)\ndemo.setConfig("watson.tone.url","https://gateway.watsonplatform.net/tone-analyzer/api")\ndemo.setConfig("watson.tone.password",taPassword)\ndemo.setConfig("watson.tone.username",taUserName)\ndemo.setConfig("checkpointDir", System.getProperty("user.home") + "/pixiedust/ssc")')


get_ipython().system('pip install --upgrade --user pixiedust-twitterdemo')


from pixiedust_twitterdemo import *
twitterDemo()


# ## The embedded app has generated a DataFrame called __tweets. Let's use it to do some data science
# 

display(__tweets)


from pyspark.sql import Row
from pyspark.sql.types import *
emotions=__tweets.columns[-13:]
distrib = __tweets.flatMap(lambda t: [(x,t[x]) for x in emotions]).filter(lambda t: t[1]>60)    .toDF(StructType([StructField('emotion',StringType()),StructField('score',DoubleType())]))
display(distrib)


__tweets.registerTempTable("pixiedust_tweets")
#create an array that will hold the count for each sentiment
sentimentDistribution=[0] * 13
#For each sentiment, run a sql query that counts the number of tweets for which the sentiment score is greater than 60%
#Store the data in the array
for i, sentiment in enumerate(__tweets.columns[-13:]):
    sentimentDistribution[i]=sqlContext.sql("SELECT count(*) as sentCount FROM pixiedust_tweets where " + sentiment + " > 60")        .collect()[0].sentCount


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ind=np.arange(13)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Tweet count')
plt.xlabel('Tone')
plt.title('Distribution of tweets by sentiments > 60%')
plt.xticks(ind+width, __tweets.columns[-13:])
plt.legend()

plt.show()


from operator import add
import re
tagsRDD = __tweets.flatMap( lambda t: re.split("\s", t.text))    .filter( lambda word: word.startswith("#") )    .map( lambda word : (word, 1 ))    .reduceByKey(add, 10).map(lambda (a,b): (b,a)).sortByKey(False).map(lambda (a,b):(b,a))
top10tags = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2, plSize[1]*2) )

labels = [i[0] for i in top10tags]
sizes = [int(i[1]) for i in top10tags]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', "beige", "paleturquoise", "pink", "lightyellow", "coral"]

plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


cols = __tweets.columns[-13:]
def expand( t ):
    ret = []
    for s in [i[0] for i in top10tags]:
        if ( s in t.text ):
            for tone in cols:
                ret += [s.replace(':','').replace('-','') + u"-" + unicode(tone) + ":" + unicode(getattr(t, tone))]
    return ret 
def makeList(l):
    return l if isinstance(l, list) else [l]

#Create RDD from tweets dataframe
tagsRDD = __tweets.map(lambda t: t )

#Filter to only keep the entries that are in top10tags
tagsRDD = tagsRDD.filter( lambda t: any(s in t.text for s in [i[0] for i in top10tags] ) )

#Create a flatMap using the expand function defined above, this will be used to collect all the scores 
#for a particular tag with the following format: Tag-Tone-ToneScore
tagsRDD = tagsRDD.flatMap( expand )

#Create a map indexed by Tag-Tone keys 
tagsRDD = tagsRDD.map( lambda fullTag : (fullTag.split(":")[0], float( fullTag.split(":")[1]) ))

#Call combineByKey to format the data as follow
#Key=Tag-Tone
#Value=(count, sum_of_all_score_for_this_tone)
tagsRDD = tagsRDD.combineByKey((lambda x: (x,1)),
                  (lambda x, y: (x[0] + y, x[1] + 1)),
                  (lambda x, y: (x[0] + y[0], x[1] + y[1])))

#ReIndex the map to have the key be the Tag and value be (Tone, Average_score) tuple
#Key=Tag
#Value=(Tone, average_score)
tagsRDD = tagsRDD.map(lambda (key, ab): (key.split("-")[0], (key.split("-")[1], round(ab[0]/ab[1], 2))))

#Reduce the map on the Tag key, value becomes a list of (Tone,average_score) tuples
tagsRDD = tagsRDD.reduceByKey( lambda x, y : makeList(x) + makeList(y) )

#Sort the (Tone,average_score) tuples alphabetically by Tone
tagsRDD = tagsRDD.mapValues( lambda x : sorted(x) )

#Format the data as expected by the plotting code in the next cell. 
#map the Values to a tuple as follow: ([list of tone], [list of average score])
#e.g. #someTag:([u'Agreeableness', u'Analytical', u'Anger', u'Cheerfulness', u'Confident', u'Conscientiousness', u'Negative', u'Openness', u'Tentative'], [1.0, 0.0, 0.0, 1.0, 0.0, 0.48, 0.0, 0.02, 0.0])
tagsRDD = tagsRDD.mapValues( lambda x : ([elt[0] for elt in x],[elt[1] for elt in x])  )

#Use custom sort function to sort the entries by order of appearance in top10tags
def customCompare( key ):
    for (k,v) in top10tags:
        if k == key:
            return v
    return 0
tagsRDD = tagsRDD.sortByKey(ascending=False, numPartitions=None, keyfunc = customCompare)

#Take the mean tone scores for the top 10 tags
top10tagsMeanScores = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*3, plSize[1]*2) )

top5tagsMeanScores = top10tagsMeanScores[:5]
width = 0
ind=np.arange(13)
(a,b) = top5tagsMeanScores[0]
labels=b[0]
colors = ["beige", "paleturquoise", "pink", "lightyellow", "coral", "lightgreen", "gainsboro", "aquamarine","c"]
idx=0
for key, value in top5tagsMeanScores:
    plt.bar(ind + width, value[1], 0.15, color=colors[idx], label=key)
    width += 0.15
    idx += 1
plt.xticks(ind+0.3, labels)
plt.ylabel('AVERAGE SCORE')
plt.xlabel('TONES')
plt.title('Breakdown of top hashtags by sentiment tones')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',ncol=5, mode="expand", borderaxespad=0.)

plt.show()


# ## install the streaming twitter jar in the notebook from the Github repo
# 

import pixiedust
jarPath = "https://github.com/ibm-watson-data-lab/spark.samples/raw/master/dist/streaming-twitter-assembly-1.6.jar"
pixiedust.installPackage(jarPath)


# ## Use Scala Bridge to run the command line version of the app
# For instruction on how to set up the twitter and Tone Analyzer credentials, please refer to https://developer.ibm.com/clouddataservices/2016/01/15/real-time-sentiment-analysis-of-twitter-hashtags-with-spark/ 
# 

twitterConsumerKey = "SOOo6EmsqAbfVMEidXy12DvRZ"
twitterConsumerSecret = "Ni1OIqqkei0aq60vC8wrei2WPTpCCX4j0EXEBd80PPebOgUZKk"
twitterAccessToken = "404118511-z4rf7f1Qm85oWQncf7Y59yc1oKHQjhRFOdRhN2Wm"
twitterAccessTokenSecret = "QqUSWaJr7GCak1P75PheBstQJjbZyrRZSRfzqFMyjvjEP"
toneAnalyzerPassword = "UMioqyrFAaNi"
toneAnalyzerUserName = "a3e0dd21-ebe9-4475-a6ab-eb2f6382db27"


get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.StreamingTwitter\ndemo.setConfig("twitter4j.oauth.consumerKey",twitterConsumerKey)\ndemo.setConfig("twitter4j.oauth.consumerSecret",twitterConsumerSecret)\ndemo.setConfig("twitter4j.oauth.accessToken",twitterAccessToken)\ndemo.setConfig("twitter4j.oauth.accessTokenSecret",twitterAccessTokenSecret)\ndemo.setConfig("watson.tone.url","https://gateway.watsonplatform.net/tone-analyzer/api")\ndemo.setConfig("watson.tone.password",toneAnalyzerPassword)\ndemo.setConfig("watson.tone.username",toneAnalyzerUserName)\n\nimport org.apache.spark.streaming._\ndemo.startTwitterStreaming(sc, Seconds(30))')


get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.StreamingTwitter\nval (__sqlContext, __df) = demo.createTwitterDataFrames(sc)')


# ## Do some data science with the DataFrame __df obtained from the scala code above
# 

tweets=__df
tweets.count()
display(tweets)


#create an array that will hold the count for each sentiment
sentimentDistribution=[0] * 13
#For each sentiment, run a sql query that counts the number of tweets for which the sentiment score is greater than 60%
#Store the data in the array
for i, sentiment in enumerate(tweets.columns[-13:]):
    sentimentDistribution[i]=__sqlContext.sql("SELECT count(*) as sentCount FROM tweets where " + sentiment + " > 60")        .collect()[0].sentCount


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ind=np.arange(13)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Tweet count')
plt.xlabel('Tone')
plt.title('Distribution of tweets by sentiments > 60%')
plt.xticks(ind+width, tweets.columns[-13:])
plt.legend()

plt.show()


from operator import add
import re
tagsRDD = tweets.flatMap( lambda t: re.split("\s", t.text))    .filter( lambda word: word.startswith("#") )    .map( lambda word : (word, 1 ))    .reduceByKey(add, 10).map(lambda (a,b): (b,a)).sortByKey(False).map(lambda (a,b):(b,a))
top10tags = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2, plSize[1]*2) )

labels = [i[0] for i in top10tags]
sizes = [int(i[1]) for i in top10tags]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', "beige", "paleturquoise", "pink", "lightyellow", "coral"]

plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


cols = tweets.columns[-13:]
def expand( t ):
    ret = []
    for s in [i[0] for i in top10tags]:
        if ( s in t.text ):
            for tone in cols:
                ret += [s.replace(':','').replace('-','') + u"-" + unicode(tone) + ":" + unicode(getattr(t, tone))]
    return ret 
def makeList(l):
    return l if isinstance(l, list) else [l]

#Create RDD from tweets dataframe
tagsRDD = tweets.map(lambda t: t )

#Filter to only keep the entries that are in top10tags
tagsRDD = tagsRDD.filter( lambda t: any(s in t.text for s in [i[0] for i in top10tags] ) )

#Create a flatMap using the expand function defined above, this will be used to collect all the scores 
#for a particular tag with the following format: Tag-Tone-ToneScore
tagsRDD = tagsRDD.flatMap( expand )

#Create a map indexed by Tag-Tone keys 
tagsRDD = tagsRDD.map( lambda fullTag : (fullTag.split(":")[0], float( fullTag.split(":")[1]) ))

#Call combineByKey to format the data as follow
#Key=Tag-Tone
#Value=(count, sum_of_all_score_for_this_tone)
tagsRDD = tagsRDD.combineByKey((lambda x: (x,1)),
                  (lambda x, y: (x[0] + y, x[1] + 1)),
                  (lambda x, y: (x[0] + y[0], x[1] + y[1])))

#ReIndex the map to have the key be the Tag and value be (Tone, Average_score) tuple
#Key=Tag
#Value=(Tone, average_score)
tagsRDD = tagsRDD.map(lambda (key, ab): (key.split("-")[0], (key.split("-")[1], round(ab[0]/ab[1], 2))))

#Reduce the map on the Tag key, value becomes a list of (Tone,average_score) tuples
tagsRDD = tagsRDD.reduceByKey( lambda x, y : makeList(x) + makeList(y) )

#Sort the (Tone,average_score) tuples alphabetically by Tone
tagsRDD = tagsRDD.mapValues( lambda x : sorted(x) )

#Format the data as expected by the plotting code in the next cell. 
#map the Values to a tuple as follow: ([list of tone], [list of average score])
#e.g. #someTag:([u'Agreeableness', u'Analytical', u'Anger', u'Cheerfulness', u'Confident', u'Conscientiousness', u'Negative', u'Openness', u'Tentative'], [1.0, 0.0, 0.0, 1.0, 0.0, 0.48, 0.0, 0.02, 0.0])
tagsRDD = tagsRDD.mapValues( lambda x : ([elt[0] for elt in x],[elt[1] for elt in x])  )

#Use custom sort function to sort the entries by order of appearance in top10tags
def customCompare( key ):
    for (k,v) in top10tags:
        if k == key:
            return v
    return 0
tagsRDD = tagsRDD.sortByKey(ascending=False, numPartitions=None, keyfunc = customCompare)

#Take the mean tone scores for the top 10 tags
top10tagsMeanScores = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*3, plSize[1]*2) )

top5tagsMeanScores = top10tagsMeanScores[:5]
width = 0
ind=np.arange(13)
(a,b) = top5tagsMeanScores[0]
labels=b[0]
colors = ["beige", "paleturquoise", "pink", "lightyellow", "coral", "lightgreen", "gainsboro", "aquamarine","c"]
idx=0
for key, value in top5tagsMeanScores:
    plt.bar(ind + width, value[1], 0.15, color=colors[idx], label=key)
    width += 0.15
    idx += 1
plt.xticks(ind+0.3, labels)
plt.ylabel('AVERAGE SCORE')
plt.xlabel('TONES')
plt.title('Breakdown of top hashtags by sentiment tones')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',ncol=5, mode="expand", borderaxespad=0.)

plt.show()


# ## Use Twitter demo embedded app to run the same app with a UI
# 

get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.PixiedustStreamingTwitter\ndemo.setConfig("twitter4j.oauth.consumerKey",twitterConsumerKey)\ndemo.setConfig("twitter4j.oauth.consumerSecret",twitterConsumerSecret)\ndemo.setConfig("twitter4j.oauth.accessToken",twitterAccessToken)\ndemo.setConfig("twitter4j.oauth.accessTokenSecret",twitterAccessTokenSecret)\ndemo.setConfig("watson.tone.url","https://gateway.watsonplatform.net/tone-analyzer/api")\ndemo.setConfig("watson.tone.password",toneAnalyzerPassword)\ndemo.setConfig("watson.tone.username",toneAnalyzerUserName)\ndemo.setConfig("checkpointDir", System.getProperty("user.home") + "/pixiedust/ssc")')


from pixiedust_twitterdemo import *
twitterDemo()


# ## The embedded app has generated a DataFrame called __tweets. Let's use it to do some data science
# 

display(__tweets)


from pyspark.sql import Row
from pyspark.sql.types import *
emotions=__tweets.columns[-13:]
distrib = __tweets.flatMap(lambda t: [(x,t[x]) for x in emotions]).filter(lambda t: t[1]>60)    .toDF(StructType([StructField('emotion',StringType()),StructField('score',DoubleType())]))
display(distrib)


__tweets.registerTempTable("pixiedust_tweets")
#create an array that will hold the count for each sentiment
sentimentDistribution=[0] * 13
#For each sentiment, run a sql query that counts the number of tweets for which the sentiment score is greater than 60%
#Store the data in the array
for i, sentiment in enumerate(__tweets.columns[-13:]):
    sentimentDistribution[i]=sqlContext.sql("SELECT count(*) as sentCount FROM pixiedust_tweets where " + sentiment + " > 60")        .collect()[0].sentCount


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ind=np.arange(13)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Tweet count')
plt.xlabel('Tone')
plt.title('Distribution of tweets by sentiments > 60%')
plt.xticks(ind+width, __tweets.columns[-13:])
plt.legend()

plt.show()


from operator import add
import re
tagsRDD = __tweets.flatMap( lambda t: re.split("\s", t.text))    .filter( lambda word: word.startswith("#") )    .map( lambda word : (word, 1 ))    .reduceByKey(add, 10).map(lambda (a,b): (b,a)).sortByKey(False).map(lambda (a,b):(b,a))
top10tags = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2, plSize[1]*2) )

labels = [i[0] for i in top10tags]
sizes = [int(i[1]) for i in top10tags]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', "beige", "paleturquoise", "pink", "lightyellow", "coral"]

plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


cols = __tweets.columns[-13:]
def expand( t ):
    ret = []
    for s in [i[0] for i in top10tags]:
        if ( s in t.text ):
            for tone in cols:
                ret += [s.replace(':','').replace('-','') + u"-" + unicode(tone) + ":" + unicode(getattr(t, tone))]
    return ret 
def makeList(l):
    return l if isinstance(l, list) else [l]

#Create RDD from tweets dataframe
tagsRDD = __tweets.map(lambda t: t )

#Filter to only keep the entries that are in top10tags
tagsRDD = tagsRDD.filter( lambda t: any(s in t.text for s in [i[0] for i in top10tags] ) )

#Create a flatMap using the expand function defined above, this will be used to collect all the scores 
#for a particular tag with the following format: Tag-Tone-ToneScore
tagsRDD = tagsRDD.flatMap( expand )

#Create a map indexed by Tag-Tone keys 
tagsRDD = tagsRDD.map( lambda fullTag : (fullTag.split(":")[0], float( fullTag.split(":")[1]) ))

#Call combineByKey to format the data as follow
#Key=Tag-Tone
#Value=(count, sum_of_all_score_for_this_tone)
tagsRDD = tagsRDD.combineByKey((lambda x: (x,1)),
                  (lambda x, y: (x[0] + y, x[1] + 1)),
                  (lambda x, y: (x[0] + y[0], x[1] + y[1])))

#ReIndex the map to have the key be the Tag and value be (Tone, Average_score) tuple
#Key=Tag
#Value=(Tone, average_score)
tagsRDD = tagsRDD.map(lambda (key, ab): (key.split("-")[0], (key.split("-")[1], round(ab[0]/ab[1], 2))))

#Reduce the map on the Tag key, value becomes a list of (Tone,average_score) tuples
tagsRDD = tagsRDD.reduceByKey( lambda x, y : makeList(x) + makeList(y) )

#Sort the (Tone,average_score) tuples alphabetically by Tone
tagsRDD = tagsRDD.mapValues( lambda x : sorted(x) )

#Format the data as expected by the plotting code in the next cell. 
#map the Values to a tuple as follow: ([list of tone], [list of average score])
#e.g. #someTag:([u'Agreeableness', u'Analytical', u'Anger', u'Cheerfulness', u'Confident', u'Conscientiousness', u'Negative', u'Openness', u'Tentative'], [1.0, 0.0, 0.0, 1.0, 0.0, 0.48, 0.0, 0.02, 0.0])
tagsRDD = tagsRDD.mapValues( lambda x : ([elt[0] for elt in x],[elt[1] for elt in x])  )

#Use custom sort function to sort the entries by order of appearance in top10tags
def customCompare( key ):
    for (k,v) in top10tags:
        if k == key:
            return v
    return 0
tagsRDD = tagsRDD.sortByKey(ascending=False, numPartitions=None, keyfunc = customCompare)

#Take the mean tone scores for the top 10 tags
top10tagsMeanScores = tagsRDD.take(10)


get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*3, plSize[1]*2) )

top5tagsMeanScores = top10tagsMeanScores[:5]
width = 0
ind=np.arange(13)
(a,b) = top5tagsMeanScores[0]
labels=b[0]
colors = ["beige", "paleturquoise", "pink", "lightyellow", "coral", "lightgreen", "gainsboro", "aquamarine","c"]
idx=0
for key, value in top5tagsMeanScores:
    plt.bar(ind + width, value[1], 0.15, color=colors[idx], label=key)
    width += 0.15
    idx += 1
plt.xticks(ind+0.3, labels)
plt.ylabel('AVERAGE SCORE')
plt.xlabel('TONES')
plt.title('Breakdown of top hashtags by sentiment tones')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',ncol=5, mode="expand", borderaxespad=0.)

plt.show()


