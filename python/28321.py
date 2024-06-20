# ## Display Random Streaming data with bokeh line chart and scatter plot
# 

from pixiedust.display.app import *
from pixiedust.display.streaming import *
from pixiedust.display.streaming.bokeh import *

N = 100
class RandomStreamingAdapter(StreamingDataAdapter):
    def __init__(self):
        self.x = np.random.random(size=N) * 100
        self.y = np.random.random(size=N) * 100
    
    def doGetNextData(self):
        rand = 2 * (np.random.random(size=N) - 0.5)
        d = np.sqrt((self.x-50)**2 + (self.y-50)**2)/100
        self.x = self.x + 2 * np.sin(d) * rand
        self.y = self.y + np.cos(d**2) * rand
        #return (self.x, self.y)
        return self.y

@PixieApp
class StreamingApp():    
    def setup(self):
        self.streamingData = RandomStreamingAdapter()
        self.scatter = False
        self.streamingDisplay = None

    def newDisplayHandler(self, options, entity):
        if self.streamingDisplay is None:
            self.streamingDisplay = ScatterPlotStreamingDisplay(options, entity) if self.scatter else LineChartStreamingDisplay(options, entity)
        return self.streamingDisplay
        
    @route()
    def main(self):
        return"""
<button type="button" class="btn btn-default">Toggle
    <pd_script>
self.scatter = not self.scatter
self.streamingDisplay = None
</pd_script>
</button>
            <div pd_entity="streamingData" pd_refresh_rate="1000">
            </div>
        """

#run the app
a = StreamingApp()
a.run(runInDialog='false')


# # Stashing Your Data
# 
# With PixieDust, you also have the option to export the data from your Notebook to external sources.
# The output of the **`display`** API includes a toolbar which contains the **Download** button.
# 
# <img style="margin:10px 0" src="https://ibm-watson-data-lab.github.io/pixiedust/_images/downloadfile.png">
# 
# 

# ***
# ### Stash to Cloudant
# 
# One export option is to save the data directly into a [Cloudant](https://cloudant.com/) or [CouchDB](https://couchdb.apache.org/) database.
# 
# 
# #### Prerequisites
# 
# Collect your database connection information: the database host, user name, and password.  
#   
# > If your Cloudant instance was provisioned in [Bluemix](https://console.ng.bluemix.net/catalog/services/cloudant-nosql-db/) you can find the connectivity information in the Service Credentials tab.
# 
# 
# #### Steps  
# 
# * From the toolbar in the **`display`** output, click the **Download** button  
# 
# * Choose **Stash to Cloudant** from the dropdown menu  
#   
# > If you get an error that a library is missing, you may need to [install the cloudant-spark](https://github.com/ibm-watson-data-lab/pixiedust/blob/plotting-tools/docsrc/source/install.rst#stash-to-cloudant) library.
# 
# * Click the dropdown to see the list of available connections  
# 
# * Select an existing connection or add a new connection  
# 
#     * Click the **`+`** plus button to add a new connection
#     * Enter your Cloudant database credentials in JSON format  
#   
#     > If you are stashing to CouchDB be sure include the protocol. See the [sample credentials format](#Sample-Credentials-Format) below.
# 
#     * Click **OK**
#     * Select the new connection
#     
# * Click **Submit**
# 
# 
# #### Sample Credentials Format  
# 
# * **CouchDB**
# ```
# {
#     "name": "local-couchdb-connection",
#     "credentials": {
#         "username": "couchdbuser",
#         "password": "password",
#         "protocol": "http",
#         "host": "127.0.0.1:5984",
#         "port": 5984,
#         "url": "http://couchdbuser:password@127.0.0.1:5984"
#     }
# }
# ```
# 
# * **Cloudant**
# ```
# {
#     "name": "remote-cloudant-connection",
#     "credentials": {
#         "username": "username-bluemix",
#         "password": "password",
#         "host": "host-bluemix.cloudant.com",
#         "port": 443,
#         "url": "https://username-bluemix:password@host-bluemix.cloudant.com"
#     }
# }
# ```
# 

# ***
# ### Download as File
# 
# Alternatively, you may choose to save the data set to a various file formats (e.g., CSV, JSON, XML, etc.)
# 
# #### Steps  
# 
# * From the toolbar in the **`display`** output, click the **Download** button
# * Choose **Download as File**
# * Choose the desired format
# * Specify the number of records to download
# <img style="margin:10px 0" src="https://ibm-watson-data-lab.github.io/pixiedust/_images/save_as.png">
# * Click **OK**
# 

#!pip install -e /Users/rajrsingh/workspace/lib/pixiedust
import pixiedust
# import pandas as pd


homesdf = pixiedust.sampleData(6)
# homesdf = pd.read_csv("https://openobjectstore.mybluemix.net/misc/milliondollarhomes.csv")


roadslines = {
    "type": "FeatureCollection", 
    "features": [
        {
            "type": "Feature", 
            "properties": {}, 
            "geometry": {
                "type": "LineString", 
                "coordinates": [
                  [-71.0771656036377,42.364537198664884],
                  [-71.07780933380127,42.36133451106724],
                  [-71.07562065124512,42.359812384483625],
                  [-71.07557773590088,42.35610204645879]
                ]
            }
        }, 
        {
            "type": "Feature", 
            "properties": {
                "name": "Highway to the Danger Zone"
            }, 
            "geometry": {
                "type": "LineString", 
                "coordinates": [
                  [-71.09179973602294,42.35848049347556],
                  [-71.08287334442139,42.356419177928906],
                  [-71.07184410095215,42.35794138670829],
                  [-71.06772422790527,42.35686315929846]
                ]
            }
        }
    ]
}
roadslayer = {
    "id": "Roads",
    "maptype": "mapbox", 
    "order": 2,
    "type": "line",
    "source": {
        "type": "geojson",
        "data": roadslines
    },
    "paint": {
        "line-color": "rgba(128,0,128,0.65)",
        "line-width": 6, 
        "line-blur": 2, 
        "line-opacity": 0.75
    },
    "layout": {
        "line-join": "round"        
    }
}


dangerzones = {
    "type": "FeatureCollection", 
    "features": [
        {
            "type": "Feature", 
            "properties": {
                "name": "Danger Zone"
            }, 
            "geometry": {
                "type": "Polygon", 
                "coordinates": [
                    [[-71.08828067779541, 42.360890561289295],
                    [-71.08802318572998, 42.35032997408756],
                    [-71.07295989990234, 42.35591176680853],
                    [-71.07583522796631, 42.3609539828782],
                    [-71.08828067779541, 42.360890561289295]]
                ]
            }
        }
    ]
}
dglayer = {
    "id": "Danger Zone",
    "maptype": "mapbox", 
    "order": 3,
    "type": "fill",
    "source": {
        "type": "geojson",
        "data": dangerzones
    },
    "paint": {
        "fill-antialias": True, 
        "fill-color": "rgba(248,64,0,1.0)",
        "fill-outline-color": "#ff0000"
    },
    "layout": {}
}


custompt = {
    "type": "FeatureCollection", 
    "features": [
        {
            "type": "Feature", 
            "properties": {}, 
            "geometry": {
                "type": "Point", 
                "coordinates": [-71.0771, 42.3599]
            }
        }, 
        {
            "type": "Feature", 
            "properties": {}, 
            "geometry": {
                "type": "Point", 
                "coordinates": [-71.0771, 42.3610]
            }
        }
    ]
}
customLayer = {
    "id": "specialdata",
    "maptype": "mapbox", 
    "order": 1,
    "type": "circle",
    "source": {
        "type": "geojson",
        "data": custompt
    },
    "paint": {
        "circle-color": "rgba(0,0,255,0.85)", 
        "circle-radius": 20
    },
    "layout": {}
}


display(homesdf)














from pixiedust.display.app import *
from pixiedust.utils.shellAccess import ShellAccess
import geojson


@PixieApp
class MapboxUserLayers:
    
    @route()
    def main(self):
        self.USERLAYERS = []
        for key in ShellAccess:
            v = ShellAccess[key]
            if isinstance(v, dict) and "source" in v and "type" in v["source"] and v["source"]["type"] == "geojson" and "id" in v and "paint" in v and "layout" in v and "data" in v["source"]:
#                 gj = geojson.loads(v["source"]["data"])
#                 isvalid = geojson.is_valid(gj)
#                 if isvalid["valid"] == "yes":
                self.USERLAYERS.append(v)
#                 else:
#                     print("Invalid GeoJSON: {0}".format(str(v["source"]["data"])))

        return """<pre>{% for layer in this.USERLAYERS %}
        var layertype = "circle";
        {% if layer["type"] %}
        layertype = "{{layer["type"]}}";
        {%endif%}

        var layerpaint = "{}";
        {% if layer["paint"] %}
        layerpaint = "{{layer["paint"]}}";
        {%endif%}

        var layerlayout = "{}";
        {% if layer["layout"] %}
        layerlayout = "{{layer["layout"]}}";
        {%endif%}

        map.addLayer({
            "id": "{{layer["id"]}}", 
            "type": layertype, 
            "source": {{layer["source"]}},
            "paint": layerpaint, 
            "layout": layerlayout
        });
        {% endfor %}</pre>
"""


mbl = MapboxUserLayers()
mbl.run()


# ## Notes
# 

get_ipython().magic('pixiedustLog -l debug')





# ## install the streaming twitter jar in the notebook from the Github repo
# 

import pixiedust
jarPath = "https://github.com/ibm-watson-data-lab/spark.samples/raw/master/dist/streaming-twitter-assembly-1.6.jar"
pixiedust.installPackage(jarPath)
print("done")


# ## Use Scala Bridge to run the command line version of the app
# For instruction on how to set up the twitter and Tone Analyzer credentials, please refer to https://developer.ibm.com/clouddataservices/2016/01/15/real-time-sentiment-analysis-of-twitter-hashtags-with-spark/ 
# 

get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.StreamingTwitter\ndemo.setConfig("twitter4j.oauth.consumerKey","XXXX")\ndemo.setConfig("twitter4j.oauth.consumerSecret","XXXX")\ndemo.setConfig("twitter4j.oauth.accessToken","XXXX)\ndemo.setConfig("twitter4j.oauth.accessTokenSecret","XXXX")\ndemo.setConfig("watson.tone.url","https://gateway.watsonplatform.net/tone-analyzer/api")\ndemo.setConfig("watson.tone.password","XXXX")\ndemo.setConfig("watson.tone.username","XXXX")\n\nimport org.apache.spark.streaming._\ndemo.startTwitterStreaming(sc, Seconds(30))')


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

get_ipython().run_cell_magic('scala', '', 'val demo = com.ibm.cds.spark.samples.PixiedustStreamingTwitter\ndemo.setConfig("twitter4j.oauth.consumerKey","XXXX")\ndemo.setConfig("twitter4j.oauth.consumerSecret","XXXX")\ndemo.setConfig("twitter4j.oauth.accessToken","XXXX")\ndemo.setConfig("twitter4j.oauth.accessTokenSecret","XXX")\ndemo.setConfig("watson.tone.url","https://gateway.watsonplatform.net/tone-analyzer/api")\ndemo.setConfig("watson.tone.password","XXXX")\ndemo.setConfig("watson.tone.username","XXXX")\ndemo.setConfig("checkpointDir", System.getProperty("user.home") + "/pixiedust/ssc")')


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


# # Load the airport and flight data from Cloudant
# 

cloudantHost='dtaieb.cloudant.com'
cloudantUserName='weenesserliffircedinvers'
cloudantPassword='72a5c4f939a9e2578698029d2bb041d775d088b5'


airports = sqlContext.read.format("com.cloudant.spark").option("cloudant.host",cloudantHost)    .option("cloudant.username",cloudantUserName).option("cloudant.password",cloudantPassword)    .option("schemaSampleSize", "-1").load("flight-metadata")
airports.cache()
airports.registerTempTable("airports")


import pixiedust

# Display the airports data
display(airports)


flights = sqlContext.read.format("com.cloudant.spark").option("cloudant.host",cloudantHost)    .option("cloudant.username",cloudantUserName).option("cloudant.password",cloudantPassword)    .option("schemaSampleSize", "-1").load("pycon_flightpredict_training_set")
flights.cache()
flights.registerTempTable("training")


# Display the flights data
display(flights)


# # Build the vertices and edges dataframe from the data
# 

from pyspark.sql import functions as f
from pyspark.sql.types import *

rdd = flights.rdd.flatMap(lambda s: [s.arrivalAirportFsCode, s.departureAirportFsCode]).distinct()    .map(lambda row:[row])
vertices = airports.join(
      sqlContext.createDataFrame(rdd, StructType([StructField("fs",StringType())])), "fs"
    ).dropDuplicates(["fs"]).withColumnRenamed("fs","id")

print(vertices.count())


edges = flights.withColumnRenamed("arrivalAirportFsCode","dst")    .withColumnRenamed("departureAirportFsCode","src")    .drop("departureWeather").drop("arrivalWeather").drop("pt_type").drop("_id").drop("_rev")

print(edges.count())


# # Install GraphFrames package using PixieDust packageManager  
# 
# The [GraphFrames package](https://mvnrepository.com/artifact/graphframes/graphframes) to install depends on the environment.
# 
# **Spark 1.6**
# 
# - `graphframes:graphframes:0.5.0-spark1.6-s_2.11`
# 
# **Spark 2.x**
# 
# - `graphframes:graphframes:0.5.0-spark2.1-s_2.11`
# 
# In addition, recent versions of graphframes have dependencies on other packages which will need to also be installed:
# 
# - `com.typesafe.scala-logging:scala-logging-api_2.11:2.1.2`
# - `com.typesafe.scala-logging:scala-logging-slf4j_2.11:2.1.2`
# 
# > Note: After installing packages, the kernel will need to be restarted and all the previous cells re-run (including the install package cell).
# 

import pixiedust

if sc.version.startswith('1.6.'):  # Spark 1.6
    pixiedust.installPackage("graphframes:graphframes:0.5.0-spark1.6-s_2.11")
elif sc.version.startswith('2.'):  # Spark 2.1, 2.0
    pixiedust.installPackage("graphframes:graphframes:0.5.0-spark2.1-s_2.11")


pixiedust.installPackage("com.typesafe.scala-logging:scala-logging-api_2.11:2.1.2")
pixiedust.installPackage("com.typesafe.scala-logging:scala-logging-slf4j_2.11:2.1.2")

print("done")


# # Create the GraphFrame from the Vertices and Edges Dataframes
# 

from graphframes import GraphFrame
g = GraphFrame(vertices, edges)
display(g)


# ### Compute the degree for each vertex in the graph
# The degree of a vertex is the number of edges incident to the vertex. In a directed graph, in-degree is the number of edges where vertex is the destination and out-degree is the number of edges where the vertex is the source. With GraphFrames, there is a degrees, outDegrees and inDegrees property that return a DataFrame containing the id of the vertext and the number of edges. We then sort then in descending order
# 

from pyspark.sql.functions import *
degrees = g.degrees.sort(desc("degree"))
display( degrees )


# ### Compute a list of shortest paths for each vertex to a specified list of landmarks
# For this we use the `shortestPaths` api that returns DataFrame containing the properties for each vertex plus an extra column called distances that contains the number of hops to each landmark.
# In the following code, we use BOS and LAX as the landmarks
# 

r = g.shortestPaths(landmarks=["BOS", "LAX"]).select("id", "distances")
display(r)


# ### Compute the pageRank for each vertex in the graph
# [PageRank](https://en.wikipedia.org/wiki/PageRank) is a famous algorithm used by Google Search to rank vertices in a graph by order of importance. To compute pageRank, we'll use the `pageRank` api that returns a new graph in which the vertices have a new `pagerank` column representing the pagerank score for the vertex and the edges have a new `weight` column representing the edge weight that contributed to the pageRank score. We'll then display the vertice ids and associated pageranks sorted descending: 
# 

from pyspark.sql.functions import *

ranks = g.pageRank(resetProbability=0.20, maxIter=5)

rankedVertices = ranks.vertices.select("id","pagerank").orderBy(desc("pagerank"))
rankedEdges = ranks.edges.select("src", "dst", "weight").orderBy(desc("weight") )

ranks = GraphFrame(rankedVertices, rankedEdges)
display(ranks)


# ### Search routes between 2 airports with specific criteria
# In this section, we want to find all the routes between Boston and San Francisco operated by United Airlines with at most 2 hops. To accomplish this, we use the `bfs` ([Breath First Search](https://en.wikipedia.org/wiki/Breadth-first_search)) api that returns a DataFrame containing the shortest path between matching vertices. For clarity will only keep the edge when displaying the results
# 

paths = g.bfs(fromExpr="id='BOS'",toExpr="id = 'SFO'",edgeFilter="carrierFsCode='UA'", maxPathLength = 2)    .drop("from").drop("to")
paths.cache()
display(paths)


# ### Find all airports that do not have direct flights between each other
# In this section, we'll use a very powerful graphFrames search feature that uses a pattern called [motif](http://graphframes.github.io/user-guide.html#motif-finding) to find nodes. The pattern we'll use the following pattern `"(a)-[]->(b);(b)-[]->(c);!(a)-[]->(c)"` which searches for all nodes a, b and c that have a path to (a,b) and a path to (b,c) but not a path to (a,c). 
# Also, because the search is computationally expensive, we reduce the number of edges by grouping the flights that have the same src and dst.
# 

from pyspark.sql.functions import *

h = GraphFrame(g.vertices, g.edges.select("src","dst")   .groupBy("src","dst").agg(count("src").alias("count")))

query = h.find("(a)-[]->(b);(b)-[]->(c);!(a)-[]->(c)").drop("b")
query.cache()
display(query)


# ### Compute the strongly connected components for this graph
# [Strongly Connected Components](https://en.wikipedia.org/wiki/Strongly_connected_component) are components for which each vertex is reachable from every other vertex. To compute them, we'll use the `stronglyConnectedComponents` api that returns a DataFrame containing all the vertices with the addition of a `component` column that has the component id in which the vertex belongs to. We then group all the rows by components and aggregate the sum of all the member vertices. This gives us a good idea of the components distribution in the graph
# 

from pyspark.sql.functions import *
components = g.stronglyConnectedComponents(maxIter=10).select("id","component")    .groupBy("component").agg(count("id").alias("count")).orderBy(desc("count"))
display(components)


# ### Detect communities in the graph using Label Propagation algorithm
# [Label Propagation algorithm](https://en.wikipedia.org/wiki/Label_Propagation_Algorithm) is a popular algorithm for finding communities within a graph. It has the advantage to be computationally inexpensive and thus works well with large graphs. To compute the communities, we'll use the `labelPropagation` api that returns a DataFrame containing all the vertices with the addition of a `label` column that has the label id for the communities in which the vertex belongs to. Similar to the strongly connected components, we'll then group all the rows by label and aggregate the sum of all the member vertices.
# 

from pyspark.sql.functions import *
communities = g.labelPropagation(maxIter=5).select("id", "label")    .groupBy("label").agg(count("id").alias("count")).orderBy(desc("count"))
display(communities)


# ## Use AggregateMessages to compute the average flight delays by originating airport
# 
# AggregateMessages api is not currently available in Python, so we use PixieDust Scala bridge to call out the Scala API
# Note: Notice that PixieDust is automatically rebinding the python GraphFrame variable g into a scala GraphFrame with same name
# 

get_ipython().run_cell_magic('scala', '', 'import org.graphframes.lib.AggregateMessages\nimport org.apache.spark.sql.functions.{avg,desc,floor}\n\n// For each airport, average the delays of the departing flights\nval msgToSrc = AggregateMessages.edge("deltaDeparture")\nval __agg = g.aggregateMessages\n  .sendToSrc(msgToSrc)  // send each flight delay to source\n  .agg(floor(avg(AggregateMessages.msg)).as("averageDelays"))  // average up all delays\n  .orderBy(desc("averageDelays"))\n  .limit(10)\n__agg.cache()\n__agg.show()')


display(__agg)





# ## Preview Streaming data from MessageHub
# 

from pixiedust.display.app import *
from pixiedust.display.streaming.data import *
from pixiedust.display.streaming.bokeh import *
import requests

@PixieApp
class MessageHubStreamingApp():
    def setup(self):
        self.streamingDisplay = None
        self.streamingData = None
        self.contents = []
        self.schemaX = None      

    def newDisplayHandler(self, options, entity):
        if self.streamingDisplay is None:
            self.streamingDisplay = LineChartStreamingDisplay(options, entity)
        return self.streamingDisplay
    
    def getTopics(self):
        rest_endpoint = "https://kafka-rest-prod01.messagehub.services.us-south.bluemix.net:443"
        headers = {
            'X-Auth-Token': self.credentials["api_key"],
            'Content-Type': 'application/json'
        }        
        return requests.get('{}/topics'.format(rest_endpoint),headers = headers).json()
    
    @route()
    def mainScreen(self):
        return """
<div class="well" style="text-align:center">
    <div style="font-size:x-large">MessageHub Streaming Browser.</div>
    <div style="font-size:large">Click on a topic to start</div>
</div>

{%for topic in this.getTopics()%}
    {%if loop.first or ((loop.index % 4) == 1)%}
<div class="row">
    <div class="col-sm-2"/>
    {%endif%}
    <div pd_options="topic=$val(topic{{loop.index}}{{prefix}})" class="col-sm-2" style="border: 1px solid lightblue;margin: 10px;border-radius: 25px;cursor:pointer;
        min-height: 150px;background-color:#e6eeff;display: flex;align-items: center;justify-content:center">
        <span id="topic{{loop.index}}{{prefix}}">{{topic}}</span>
    </div>
    {%if loop.last or ((loop.index % 4) == 0)%}
    <div class="col-sm-2"/>
</div>
    {%endif%}
{%endfor%}
        """
    
    def displayNextTopics(self):
        payload = self.streamingData.getNextData()
        if payload is not None and len(payload)>0:
            self.contents = self.contents + payload
            self.contents = self.contents[-10:]                
            html = ""
            for event in self.contents:
                html += "{}<br/>".format(json.dumps(event))
            print(html)
            
    def newDisplayHandler(self, options, entity):
        if self.streamingDisplay is None:
            self.streamingDisplay = LineChartStreamingDisplay(options, entity)
        return self.streamingDisplay
            
    @route(topic="*",streampreview="*",schemaX="*")
    def showChart(self, schemaX):
        self.schemaX = schemaX
        self.avgChannelData = self.streamingData.getStreamingChannel(self.computeAverages)
        return """
<div class="well" style="text-align:center">
    <div style="font-size:x-large">Real-time chart for {{this.schemaX}}(average).</div>
</div>
<style>
.bk-root{
display:flex;
justify-content:center;
}
</style>
<div pd_refresh_rate="1000" pd_entity="avgChannelData"></div>
        """
    
    def computeAverages(self, avg, newData):
        newValue = []
        for jsonValue in newData:
            if self.schemaX in jsonValue:
                thisValue = float(jsonValue[self.schemaX])
                avg = thisValue if avg is None else (avg + thisValue)/2
                newValue.append(avg)
        return newValue, avg
        
    
    @route(topic="*",streampreview="*")
    def createStreamWidget(self, streampreview):
        if streampreview=="realtimeChart":
            return """
<div>
    {%for key in this.streamingData.schema%}
    {%if loop.first%}
    <div class="well" style="text-align:center">
        <div style="font-size:x-large">Create a real-time chart by selecting a field.</div>
    </div>
    {%endif%}
    <div class="radio" style="margin-left:20%">
      <label>
          <input type="radio" pd_options="streampreview=""" + streampreview + """;schemaX=$val(schemaX{{loop.index}}{{prefix}})" 
              id="schemaX{{loop.index}}{{prefix}}" pd_target="realtimeChartStreaming{{prefix}}" 
              name="schemaX" value="{{key}}">{{key}}
      </label>
    </div>
    {%endfor%}
</div>"""
        return """<div pd_refresh_rate="1000" pd_script="self.displayNextTopics()"></div>"""
        
    @route(topic="*")
    def previewTopic(self, topic):
        self.topic = topic
        if self.streamingData is not None:
            self.streamingData.close()
        self.streamingData = MessagehubStreamingAdapter( self.topic, self.credentials["username"], self.credentials["password"] )
        return """
<div class="row">
    <div class="col-sm-12" id="targetstreaming{{prefix}}">
        <div pd_refresh_rate="1000" style="white-space:nowrap;overflow-x:auto;border:aliceblue 2px solid;height:17em;line-height:1.5em">
            <pd_script>self.displayNextTopics()</pd_script>
            <div style="width:100px;height:60px;left:47%;position:relative">
                <i class="fa fa-circle-o-notch fa-spin" style="font-size:48px"></i>
            </div>
            <div style="text-align:center">Waiting for data from MessageHub</div>
        </div>
    </div>
</div>
<div class="row" id="realtimeChartStreaming{{prefix}}">
    <div pd_refresh_rate="4000" pd_options="streampreview=realtimeChart">
    </div>
</div>
        """
        
    
MessageHubStreamingApp().run(credentials={
    "username": "XXXX",
    "password": "XXXX",
    "api_key" : "XXXX"
})





# # Hello PixieDust!
# This sample notebook provides you with an introduction to many features included in PixieDust. You can find more information about PixieDust at https://ibm-watson-data-lab.github.io/pixiedust/. To ensure you are running the latest version of PixieDust uncomment and run the following cell. Do not run this cell if you installed PixieDust locally from source and want to continue to run PixieDust from source.
# 

#!pip install --user --upgrade pixiedust


# # Import PixieDust
# Run the following cell to import the PixieDust library. You may need to restart your kernel after importing. Follow the instructions, if any, after running the cell. Note: You must import PixieDust every time you restart your kernel.
# 

import pixiedust


# # Enable the Spark Progress Monitor
# PixieDust includes a Spark Progress Monitor bar that lets you track the status of your Spark jobs. You can find more info at https://ibm-watson-data-lab.github.io/pixiedust/sparkmonitor.html. Run the following cell to enable the Spark Progress Monitor:
# 

pixiedust.enableJobMonitor();


# # Example use of the PackageManager
# You can use the PackageManager component of Pixiedust to install and uninstall maven packages into your notebook kernel without editing configuration files. This component is essential when you run notebooks from a hosted cloud environment and do not have access to the configuration files. You can find more info at https://ibm-watson-data-lab.github.io/pixiedust/packagemanager.html. Run the following cell to install the GraphFrame package. You may need to restart your kernel after installing new packages. Follow the instructions, if any, after running the cell. 
# 

pixiedust.installPackage("graphframes:graphframes:0.1.0-spark1.6")
print("done")


# Run the following cell to print out all installed packages:
# 

pixiedust.printAllPackages()


# # Example use of the display() API
# PixieDust lets you visualize your data in just a few clicks using the display() API. You can find more info at https://ibm-watson-data-lab.github.io/pixiedust/displayapi.html. The following cell creates a DataFrame and uses the display() API to create a bar chart:
# 

sqlContext=SQLContext(sc)
d1 = sqlContext.createDataFrame(
[(2010, 'Camping Equipment', 3),
 (2010, 'Golf Equipment', 1),
 (2010, 'Mountaineering Equipment', 1),
 (2010, 'Outdoor Protection', 2),
 (2010, 'Personal Accessories', 2),
 (2011, 'Camping Equipment', 4),
 (2011, 'Golf Equipment', 5),
 (2011, 'Mountaineering Equipment',2),
 (2011, 'Outdoor Protection', 4),
 (2011, 'Personal Accessories', 2),
 (2012, 'Camping Equipment', 5),
 (2012, 'Golf Equipment', 5),
 (2012, 'Mountaineering Equipment', 3),
 (2012, 'Outdoor Protection', 5),
 (2012, 'Personal Accessories', 3),
 (2013, 'Camping Equipment', 8),
 (2013, 'Golf Equipment', 5),
 (2013, 'Mountaineering Equipment', 3),
 (2013, 'Outdoor Protection', 8),
 (2013, 'Personal Accessories', 4)],
["year","zone","unique_customers"])

display(d1)


# # Example use of the Scala bridge
# Data scientists working with Spark may occasionaly need to call out to one of the hundreds of libraries available on spark-packages.org which are written in Scala or Java. PixieDust provides a solution to this problem by letting users directly write and run scala code in its own cell. It also lets variables be shared between Python and Scala and vice-versa. You can find more info at https://ibm-watson-data-lab.github.io/pixiedust/scalabridge.html.
# 

# Start by creating a python variable that we'll use in scala:
# 

python_var = "Hello From Python"
python_num = 10


# Create scala code that use the python_var and create a new variable that we'll use in Python:
# 

get_ipython().run_cell_magic('scala', '', 'println(python_var)\nprintln(python_num+10)\nval __scala_var = "Hello From Scala"')


# Use the __scala_var from python:
# 

print(__scala_var)


# # Sample Data
# PixieDust includes a number of sample data sets. You can use these sample data sets to start playing with the display() API and other PixieDust features. You can find more info at https://ibm-watson-data-lab.github.io/pixiedust/loaddata.html. Run the following cell to view the available data sets:
# 

pixiedust.sampleData()


# # Example use of sample data
# To use sample data locally run the following cell to install required packages. You may need to restart your kernel after running this cell.
# 

pixiedust.installPackage("com.databricks:spark-csv_2.10:1.5.0")
pixiedust.installPackage("org.apache.commons:commons-csv:0")


# Run the following cell to get the first data set from the list. This will return a DataFrame and assign it to the variable d2:
# 

d2 = pixiedust.sampleData(1)


# Pass the sample data set (d2) into the display() API:
# 

display(d2)


# You can also download data from a CSV file into a DataFrame which you can use with the display() API:
# 

d3 = pixiedust.sampleData("https://openobjectstore.mybluemix.net/misc/milliondollarhomes.csv")


# # PixieDust Log
# PixieDust comes complete with logging to help you troubleshoot issues. You can find more info at https://ibm-watson-data-lab.github.io/pixiedust/logging.html. To access the log run the following cell:
# 

get_ipython().magic('pixiedustLog -l debug')


# # Environment Info.
# The following cells will print out information related to your notebook environment.
# 

get_ipython().run_cell_magic('scala', '', 'val __scala_version = util.Properties.versionNumberString')


import platform
print('PYTHON VERSON = ' + platform.python_version())
print('SPARK VERSON = ' + sc.version)
print('SCALA VERSON = ' + __scala_version)


# # More Info.
# For more information about PixieDust check out the following:
# #### PixieDust Documentation: https://ibm-watson-data-lab.github.io/pixiedust/index.html
# #### PixieDust GitHub Repo: https://github.com/ibm-watson-data-lab/pixiedust
# 

import pixiedust


pixiedust.sampleData()


popdf = pixiedust.sampleData(3)


display(popdf)


# # Million dollar home sales
# 
# Home sales of $1million and above in Northeastern Massachusetts for the 3 months prior to Jan. 27, 2017. Downloaded from Redfin.com on Jan. 27, 2017.
# 

homesdf = pixiedust.sampleData(6)


# ## Map display
# The first time you run the main pixiedust visualization command, display(), you get the default visualization, which is a table. Then, from the drop-down menu select "Map", and populate the options dialog as follows:
# - *Keys:* put your latitude and longitude fields here. They must be floating values. These fields must be named latitude, lat or y and longitude, lon or x.
# - *Values:* the field you want to use to thematically color the map. Only one field can be used.
# - *Mapbox Access Token:* The token you get from your Mapbox account here: https://www.mapbox.com/studio/signup/
# 

display(homesdf)


# # Welcome to PixieDust
# 
# This notebook features an introduction to PixieDust, the Python library that makes data visualization easy. 
# 
# ## Get started
# 
# This notebook is pretty simple and self-explanatory, but it wouldn't hurt to load up the [PixieDust documentation](https://ibm-watson-data-lab.github.io/pixiedust/) so you have it. 
# 
# New to notebooks? Don't worry, all you need to know to use this notebook is that to run code cells, put your cursor in the cell and press **Shift + Enter**.
# 

# Make sure you have the latest version of PixieDust installed on your system
# Only run this cell if you did _not_ install PixieDust from source
# To confirm you have the latest, uncomment the next line and run this cell
#!pip install --user --upgrade pixiedust


# Now that you have PixieDust installed and up-to-date on your system, you need to import it into this notebook. This is the last dependency before you can play with PixieDust.
# 

# Run this cell
import pixiedust


# Once you see the success message output from running `import pixiedust`, you're all set.
# 

# ## Behold, display()
# 
# In the next cell, build a very simple dataset and store it in a variable. 
# 

# Run this cell to
# a) build a SQL context for a Spark dataframe 
sqlContext=SQLContext(sc) 
# b) create Spark dataframe, and assign it to a variable
df = sqlContext.createDataFrame(
[("Green", 75),
 ("Blue", 25)],
["Colors","%"])


# The data in the variable we just created is ready to be displayed, without any code other than the call to `display()`.
# 

# Run this cell to display the dataframe above as a pie chart
display(df)


# After running the cell above, you should have seen a Spark dataframe displayed as a **pie chart**, along with some controls to tweak the display. All that came from passing the dataframe variable to `display()`.
# 
# In the next cell, we'll pass more interesting data to `display()`, which will also offer more advanced controls.
# 

# create another dataframe, in a new variable
df2 = sqlContext.createDataFrame(
[(2010, 'Camping Equipment', 3),
 (2010, 'Golf Equipment', 1),
 (2010, 'Mountaineering Equipment', 1),
 (2010, 'Outdoor Protection', 2),
 (2010, 'Personal Accessories', 2),
 (2011, 'Camping Equipment', 4),
 (2011, 'Golf Equipment', 5),
 (2011, 'Mountaineering Equipment',2),
 (2011, 'Outdoor Protection', 4),
 (2011, 'Personal Accessories', 2),
 (2012, 'Camping Equipment', 5),
 (2012, 'Golf Equipment', 5),
 (2012, 'Mountaineering Equipment', 3),
 (2012, 'Outdoor Protection', 5),
 (2012, 'Personal Accessories', 3),
 (2013, 'Camping Equipment', 8),
 (2013, 'Golf Equipment', 5),
 (2013, 'Mountaineering Equipment', 3),
 (2013, 'Outdoor Protection', 8),
 (2013, 'Personal Accessories', 4)],
["year","category","unique_customers"])

# This time, we've combined the dataframe and display() call in the same cell
# Run this cell 
display(df2)


# ## display() controls
# 
# ### Renderers
# This chart like the first one is rendered by matplotlib. With PixieDust, you have other options. To toggle between renderers, use the `Renderers` control at top right of the display output:
# 1. [Bokeh](http://bokeh.pydata.org/en/0.10.0/index.html) is interactive; play with the controls along the top of the chart, e.g., zoom, save
# 1. [Matplotlib](http://matplotlib.org/) is static; you can save the image as a PNG
# 
# ### Chart options
# 
# 1. **Chart types**: At top left, you should see an option to display the dataframe as a table. You should also see a dropdown menu with other chart options, including bar charts, pie charts, scatter plots, and so on.
# 1. **Options**: Click the `Options` button to explore other display configurations; e.g., clustering
# 
# To know more : https://ibm-watson-data-lab.github.io/pixiedust/displayapi.html
# 

# ## Loading External Data
# So far, we've worked with data hard-coded into our notebook. Now, let's load external data (CSV) from an addressable `URL`.
# 

# load a CSV with pixiedust.sampledata()
df3 = pixiedust.sampleData("https://github.com/ibm-watson-data-lab/open-data/raw/master/cars/cars.csv")
display(df3)


# You should see a scatterplot above, rendered again by matplotlib. Look at the `Renderer` menu at top right. You should see options for **Bokeh** and now, **Seaborn**. If you don't see Seaborn, it's not installed on your system. No problem, just install it by running the next cell.
# 

# To install Seaborn, uncomment the next line, and then run this cell
#!pip install --user seaborn


# *If you installed Seaborn, you'll need to also restart your notebook kernel, and run the cell to `import pixiedust` again. Find **Restart** in the **Kernel** menu above.*
# 

# ### Loading data from DashDB 
# 
# You can load data from [DashDB](https://console.ng.bluemix.net/catalog/services/dashdb) tables and views.
# 
# #### Prerequisites
# 
# * Collect your database connection information: 
#  * JDBC URL
#  * user name
#  * password 
#  * source table or view name
# 
#   <div class="alert alert-block alert-info">
# If your DashDB service instance was provisioned in Bluemix you can find the connectivity information in the _Service Credentials_ tab or in the _Connect_ tab the dashDB web console.
# </div>
# 
# * Import PixieDust and enable the Spark Job monitor
# 

import pixiedust
pixiedust.enableJobMonitor()


# #### Configure database connectivity
# 
# Customize this cell with your DashDB connection information.
# 

# @hidden_cell
# Enter your DashDB JDBC URL (e.g. 'jdbc:db2://dashdb-entry-yp-dal00-00.services.dal.bluemix.net:50000/BLUDB')
jdbcurl = 'jdbc:db2://...'
# Enter your DashDB user name (e.g. 'dash0815')
user = '...'
# Enter your DashDB password (e.g. 'myvoiceismypassword')
password = '...'
# Enter your source table or view name (e.g. 'mytable')
table = '...'


# #### Load data from table
# 
# Load the table or view into a Spark DataFrame.
# 

# no changes are required to this cell
# obtain Spark SQL Context
sqlContext = SQLContext(sc)
# load data
props = {}
props['user'] = user
props['password'] = password
dashdb_data = sqlContext.read.jdbc(jdbcurl, table, properties=props)


# #### Explore the loaded data using PixieDust
# 

display(dashdb_data)


# <div class="alert alert-block alert-info">
# For information on how to load data from other sources refer to [these code snippets](https://apsportal.ibm.com/docs/content/analyze-data/python_load.html).
# </div>
# 

# ## Add Spark packages and run inside your notebook
# 
# PixieDust PackageManager lets you install spark packages inside your notebook. This is especailly useful when you're working in a hosted cloud environment without access to configuration files. Use PixieDust Package Manager to install:
# 
# - a spark package from spark-packages.org
# - from maven search repository
# - a jar file directly from URL
# 
# > **Note:** After you install a package, you must restart the kernel.
# 

# ### View list of packages
# To see the packages installed on your system, run the following command:
# 
# 

import pixiedust
pixiedust.printAllPackages()


# ### Add a package from spark-packages.org
# 
# Run the following cell to install GraphFrames.
# 

pixiedust.installPackage("graphframes:graphframes:0")


# #### Restart your kernel
# 
# From the menu at the top of this notebook, choose **Kernel > Restart**, then run the next cell.
# 

# ### View updated list of packages
# 
# Run printAllPackages again to see that GraphFrames is now in your list:
# 

pixiedust.printAllPackages()


# ### Display a GraphFrames data sample
# 
# GraphGrames comes with sample data sets. Even if GraphFrames is already installed, running the install command loads the Python that comes along with the package and enables features like the one you're about to see. Run the following cell and PixieDust displays a sample graph data set called **friends**. On the upper left of the display, click the table dropdown and switch between views of nodes and edges. 
# 

#import the Graphs example
from graphframes.examples import Graphs
#create the friends example graph
g=Graphs(sqlContext).friends()
#use the pixiedust display
display(g)


# ### Install from maven
# To install a package from [Maven](https://maven.apache.org/), visist the project and find its `groupId` and `artifactId`, then enter it in the following install command.  [Read more](https://ibm-watson-data-lab.github.io/pixiedust/packagemanager.html#install-from-maven-search-repository). For example, the following cell installs Apache Commons: 
# 

pixiedust.installPackage("org.apache.commons:commons-csv:0")


# ### Install a jar file directly from a URL 
#     
# To install a jar file that is not packaged in a maven repository, provide its URL. 
# 

pixiedust.installPackage("https://github.com/ibm-watson-data-lab/spark.samples/raw/master/dist/streaming-twitter-assembly-1.6.jar")


# #### Follow the tutorial
# 
# To understand what you can do with this jar file, read David Taieb's latest [Realtime Sentiment Analysis of Twitter Hashtags with Spark](https://medium.com/ibm-watson-data-lab/real-time-sentiment-analysis-of-twitter-hashtags-with-spark-7ee6ca5c1585#.2iblfu58c) tutorial.
# 

# ### Uninstall a package
# 
# It's just as easy to get rid of a package you installed. Just run the command `pixiedust.uninstallPackage("<<mypackage>>")`. For example, you can uninstall Apache Commons:
# 

pixiedust.uninstallPackage("org.apache.commons:commons-csv:0")


