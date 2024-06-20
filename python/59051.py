# # Investigate Breitbart dbpedia tags
# 
# This notebook runs the "lead" fields from Brietbart news articles through a dbpedia spotlight endpoint (check out http://dbpedia-spotlight.github.io) and collects the linked tags associated with each article.  Towards the end of the notebook I do two things with the tag, first I just plot a heatmap showing co-occurence of the tags and second I collect tags per author.  Since each article is associated with some tags, you can pivot this data on any article metadata you want (publication date, authors etc.).  Could be a lot of interesting analysis that comes out of this, especially if it's tied into other data sets.
# 

import requests
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

metrics = defaultdict(float)  # perf measurements

# use this function to run dbpedia spotlight on some text and get back dbpedia URIs
def dbpedia_spotlight(text, confidence, blacklist_types=[], sim_threshold=0):
    t0 = time.time()
    url = "http://spotlight.sztaki.hu:2222/rest/annotate"
    try:
        r = requests.get(url, params={ 
                'text': text,
                'support': 20,
                'confidence': confidence
            }, headers={'Accept': 'application/json'}, timeout=5)
    except requests.exceptions.Timeout:
        metrics['request_get'] += time.time() - t0
        print "timed out"
        return []
    
    metrics['request_get'] += time.time() - t0
    try:
        result = r.json()
    except:
        print "no response for text: %s" % text
        return []
    dbpedia_resources = []
    if 'Resources' not in result: return []
    for resource in result['Resources']:
        if float(resource['@similarityScore']) < sim_threshold: continue
        resource_types = [ r.lower() for r in resource['@types'].split(',') ]
        if len(set(blacklist_types).intersection(set(resource_types))) == 0:
            dbpedia_resources.append(resource)
    return dbpedia_resources


file_path = "/mnt/sda1/Datasets/D4D/assemble/breitbart/hist_natsec.json"  # any of the breitbart docs will work
posts = []
with open(file_path, 'r') as fin:
    posts = json.load(fin)
    
print len(posts)
posts_df = pd.DataFrame(posts)


# this dictionary is keyed by tags and has a list of post indices as values
direct_tags = defaultdict(list)
# this dictionary is keyed by post index and has a list of tags
content_resources = defaultdict(list)

# WARNING: This takes time for a full run!  Each of the 32k posts will be sent to a server for annotation.  
# If you'd like, you can run dbpedia spotlight locally (there's a docker container as well).
# The results below are from a run on the first 100 articles only
for idx, title in enumerate(posts_df['lead'][:100]):
    dbpedia_resources = dbpedia_spotlight(title, 0.35)
    for resource in dbpedia_resources:
        content_resources[idx].append(resource)
        direct_tags[resource['@URI']].append(idx)


# sort by number of articles the tag appears in
sorted_tags = sorted(direct_tags.keys(), key=lambda x: len(direct_tags[x]), reverse=True)


# check out the top 50 tags by article count
for t in sorted_tags[:50]:
    print t, len(direct_tags[t])

# build co-occurence matrix of top 50 tags
co_occuring_tags = []
idx = 0
for t1 in sorted_tags[:50]:
    row = []
    # filtering out bad tags ("The Times" was being tagged as New York Times which was rarely correct)
    if t1 in ["http://dbpedia.org/resource/The_New_York_Times", 
              "http://dbpedia.org/resource/-elect"] : continue
    for t2 in sorted_tags[:50]:
        if t2 in ["http://dbpedia.org/resource/The_New_York_Times", 
              "http://dbpedia.org/resource/-elect"] : continue
        s1 = set(direct_tags[t1])
        s2 = set(direct_tags[t2])
        if t1 == t2: 
            row.append(0.)
        else:
            # appending (2*intersection of set1 and set2) / (size of set1 + size of set2)
            row.append(float(2*len(s1.intersection(s2))) / float(len(s1)+len(s2)))
    co_occuring_tags.append(row)


np.set_printoptions(suppress=True)
plt.rcParams['figure.figsize'] = (150.0, 150.0)

get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize'] = (50.0, 50.0)
ax = plt.imshow(np.array(co_occuring_tags), interpolation='nearest', cmap='cool', vmin=0, vmax=1).axes

m = len(co_occuring_tags)
n = m
_ = ax.set_xticks(np.linspace(0, n-1, n))
_ = ax.set_xticklabels([ x.split('/')[-1] for x in sorted_tags[:50]], fontsize=40, rotation=-90)
_ = ax.set_yticks(np.linspace(0, m-1, m))
_ = ax.set_yticklabels([ x.split('/')[-1] for x in sorted_tags[:50]], fontsize=40)

ax.grid('on')
ax.xaxis.tick_top()


def map_tags_to_field(tags_dict, data_frame, field):
    """
    Pivot function.  Returns a dict keyed by the field values
    with values equal to the tags mentioned.
    """
    pivot_dict = defaultdict(list)
    for tag, indices in tags_dict.iteritems():
        for idx in indices:
            field_value = data_frame.loc[idx][field]
            pivot_dict[field_value].append(tag)            
    return pivot_dict

def filter_dict(d, intersect):
    """
    Filter all values of the dict by intersecting with intersect.  Assumes d is a dict of lists.
    """
    for k, v in d.iteritems():
        d[k] = list(set(v).intersection(set(intersect)))
    return d


# pivot the tag data using the data frame and build a dictionary keyed by authors with lists of tags as values
author_tags = map_tags_to_field(direct_tags, posts_df, 'authors')


# filter the tag list for each author to only show tags that are in the top 100 most frequent tags
author_tags = filter_dict(author_tags, sorted_tags[:100])


author_tags


# # Basic Eventador Report
# 
# Goal: to set up a simple eventador report with basic indicators on tweets. Should be the base for the later, more complicated analytics pipeline.
# 
# ## Example Eventador NB from @asragab
# 

from kafka import KafkaConsumer
import uuid
import json


consumer = KafkaConsumer(bootstrap_servers='', 
                         value_deserializer=lambda s: json.loads(s, encoding='utf-8'), 
                         auto_offset_reset='smallest', 
                         group_id=uuid.uuid4()) 


consumer.subscribe(['tweets'])


limit = 500
consumer.poll(max_records=limit)
count = 0
data = []
for msg in consumer:
    data.append(msg.value)
    count += 1
    if count >= limit:
        break


len(data)


# ## Simple graph analytics for the Twitter stream
# For this first step we want:
# - top 10 retweeted users
# - top 10 PageRanked users
# - basic matplotlib viz
# 

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import networkx as nx


# ### Building the directed graph
# We build the retweet graph, where an edge is from the original tweeter to the retweeter. We add node weights corresponding to how much each node was retweeted
# 

graph = nx.DiGraph()


for tweet in data:
    if tweet.get('retweet') == 'Y':
        name = tweet.get('name')
        original_name = tweet.get('original_name')
        followers = tweet.get('followers')
        if name not in graph: graph.add_node(name, retweets = 0)
        if original_name not in graph: 
            graph.add_node(original_name, retweets = 1)
        else:
            graph.node[original_name]['retweets'] = graph.node[original_name]['retweets'] +1
        graph.add_edge(original_name, name)    


# ### Most retweeted users
# 

top10_retweets = sorted([(node,graph.node[node]['retweets']) for node in graph.nodes()], key = lambda x: -x[1])[0:10]
top10_retweets


# ### Top 10 Pageranked users
# Note - these are the 'archetypal retweeters' of the graph (well, not exactly. see https://en.wikipedia.org/wiki/PageRank)
# 

pr = nx.pagerank(graph)
colors = [pr[node] for node in graph.nodes()]
top10_pr = sorted([(k,v) for k,v in pr.items()], key = lambda x: x[1])[0:10]
label_dict = dict([(k[0],k[0]) for k in top10_pr])
top10_pr


# ### Basic network viz
# - size of nodes is number of retweets
# - color of nodes is pagerank
# - we only label the top 10 pageranked users
# 

plt.figure(figsize=(11,11))
plt.axis('off')
weights = [10*(graph.node[node]['retweets'] + 1) for node in graph.nodes()]
nx.draw_networkx(graph, node_size = weights,  width = .1, linewidths = .1, with_labels=True,
                 node_color = colors, cmap = 'RdYlBu', 
                 labels = label_dict)


consumer.close()





# ## 4chan Sample Thread Exploration
# 
# This notebook contains the cleaning and exploration of the chan_example csv which is hosted on the far-right s3 bucket.  It contains cleaning out the html links from the text of the messages with beautiful soup, grouping the messages into their threads, and an exploratory sentiment analysis.
# 
# Further work could be to get the topic modelling for messages working and perhaps look at sentiment regarding different topics.
# 

import boto3
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


session = boto3.Session(profile_name='default')
s3 = session.resource('s3')
bucket = s3.Bucket("far-right")
session.available_profiles


# print all objects in bucket
for obj in bucket.objects.all():
    if "chan" in obj.key:
        #print(obj.key)
        pass


bucket.download_file('fourchan/chan_example.csv', 'chan_example.csv')


chan = pd.read_csv("chan_example.csv")
# remove the newline tags.  They're not useful for our analysis and just clutter the text.
chan.com = chan.com.astype(str).apply(lambda x: x.replace("<br>", " "))


bucket.download_file('info-source/daily/20170228/fourchan/fourchan_1204.json', '2017-02-28-1204.json')
chan2 = pd.read_json("2017-02-28-1204.json")


soup = BeautifulSoup(chan.com[19], "lxml")
quotes = soup.find("span")
for quote in quotes.contents:
    print(quote.replace(">>", ""))
parent = soup.find("a")
print(parent.contents[0].replace(">>", ""))


print(chan.com[19])


# If there's a quote and then the text, this would work. 
print(chan.com[19].split("</span>")[-1])


def split_comment(comment):
    """Splits up a comment into parent, quotes, and text"""
    
    # I used lxml to 
    soup = BeautifulSoup(comment, "lxml")
    quotes, quotelink, text = None, None, None
    try:
        quotes = soup.find("span")
        quotes = [quote.replace(">>", "") for quote in quotes.contents]
    except:
        pass
    try:
        quotelink = soup.find("a").contents[0].replace(">>", "")
    except: 
        pass
    # no quote or parent
    if quotes is None and quotelink is None:
        text = comment
    # Parent but no quote
    if quotelink is not None and quotes is None:
        text = comment.split("a>")[-1]
    # There is a quote
    if quotes is not None:
        text = comment.split("</span>")[-1]
    return {'quotes':quotes, 'quotelink': quotelink, 'text': text}


df = pd.DataFrame({'quotes':[], 'quotelink':[], 'text':[]})
for comment in chan['com']:
    df = df.append(split_comment(comment), ignore_index = True)
    
full = pd.merge(chan, df, left_index = True, right_index = True)


quotes = pd.Series()
quotelinks = pd.Series()
texts = pd.Series()
for comment in chan['com']:
    parse = split_comment(comment)
    quotes.append(pd.Series(parse['quotes']))
    quotelinks.append(pd.Series(parse['quotelink']))
    texts.append(pd.Series(parse['text']))
chan['quotes'] = quotes
chan['quotelinks'] = quotelinks
chan['text'] = texts


# # Message Threads
# 
# Forchan messages are all part of a message thread, which can be reassembled by following the parents for each post and chaining them back together. This code creates a thread ID and maps that thread ID to the corresponding messages.  
# 
# I don't know currently whether or not messages are linear, or if they can be a tree structure.  This section of code simply tries to find which messages belong to which threads
# 
# ## Looks like a thread is all just grouped by the parent comment.  Doh
# 
# Here i'll group the threads into a paragraph like structure and store it in a dictionary with the key being the parent chan_id.
# 

threads = full['parent'].unique()
full_text = {}
for thread in threads:
    full_text[int(thread)] = ". ".join(full[full['parent'] == thread]['text'])


# Now we can do some topic modeling on the different threads
# 
# Following along with the topic modelling tweet exploration, we're going to tokenize our messages and then build a corpus from it.  We'll then use the gensim library to run our topic model over the tokenized messages
# 

import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis


tokenized_messages = []
for msg in nlp.pipe(full['text'], n_threads = 100, batch_size = 100):
    ents = msg.ents
    msg = [token.lemma_ for token in msg if token.is_alpha and not token.is_stop]
    tokenized_messages.append(msg)

# Build the corpus using gensim     
dictionary = gensim.corpora.Dictionary(tokenized_messages)
msg_corpus = [dictionary.doc2bow(x) for x in tokenized_messages]
msg_dictionary = gensim.corpora.Dictionary([])
          
# gensim.corpora.MmCorpus.serialize(tweets_corpus_filepath, tweets_corpus)


# 
# ## Creating an Emotion Sentiment Classifier
# 
# Labeled dataset provided by @crowdflower hosted on data.world. Dataset contains 40,000 tweets which are labeled as one of 13 emotions. Here I looked at the top 5 emotions, since the bottom few had very tweets by comparison, so it would be hard to get a properly split dataset on for train/testing. Probably the one i'd want to include that wasn't included yet is anger, but neutral, worry, happinness, sadness, love are pretty good starting point for emotion classification regarding news tweets.
# https://data.world/crowdflower/sentiment-analysis-in-text
# 

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk import WordNetLemmatizer
lemma = nltk.WordNetLemmatizer()
df = pd.read_csv('https://query.data.world/s/8c7bwy8c55zx1t0c4yyrnjyax')


emotions = list(df.groupby("sentiment").agg("count").sort_values(by = "content", ascending = False).head(6).index)
print(emotions)
emotion_subset = df[df['sentiment'].isin(emotions)]

def format_sentence(sent):
    ex = [i.lower() for i in sent.split()]
    lemmas = [lemma.lemmatize(i) for i in ex]
    
    return {word: True for word in nltk.word_tokenize(" ".join(lemmas))}


def create_train_vector(row):
    """
    Formats a row when used in df.apply to create a train vector to be used by a 
    Naive Bayes Classifier from the nltk library.
    """
    sentiment = row[1]
    text = row[3]
    return [format_sentence(text), sentiment]

train = emotion_subset.apply(create_train_vector, axis = 1)
# Split off 10% of our train vector to be for test.

test = train[:int(0.1*len(train))]
train = train[int(0.9)*len(train):]

emotion_classifier = NaiveBayesClassifier.train(train)

print(accuracy(emotion_classifier, test))


# 64% test accuracy on the test is nothing to phone home about.  It's also likely to be a lot less accurate on our data from the 4chan messages, since those will be using much different *language* than the messages in our training set.
# 

emotion_classifier.show_most_informative_features()


for comment in full['text'].head(10):
    print(emotion_classifier.classify(format_sentence(comment)), ": ", comment)


# Looking at this sample of 10 posts, I'm not convinced in the accuracy of this classifier on the far-right data, but out of curiosity, what did it classifer the 
# 

full['emotion'] = full['text'].apply(lambda x: emotion_classifier.classify(format_sentence(x)))


grouped_emotion_messages = full.groupby('emotion').count()[[2]]
grouped_emotion_messages.columns = ["count"]
grouped_emotion_messages


grouped_emotion_messages.plot.bar()


# ## Considering the dataset is extremely out of sample in regards to training data, there's no way this emotion classifier is accurate.
# 
# These results do seem semi logical though, based on some knowledge of the group.  Online trolls are well known for their anger and rudeness, which could seemingly be classified as surprise and worry on a more standard data set.
# 

# # Parsing and cleaning tweets
# This notebook is a slight modification of @wwymak's word2vec notebook, with different tokenization, and a way to iterate over tweets linked to their named user 
# 

# ### WWmyak's iterator and helper functions
# 

import gensim
import os
import numpy as np
import itertools
import json
import re
import pymoji
import importlib
from nltk.tokenize import TweetTokenizer
from gensim import corpora
import string
from nltk.corpus import stopwords
from six import iteritems
import csv

tokenizer = TweetTokenizer()

def keep_retweets(tweets_objs_arr):
    return [x["text"] for x in tweets_objs_arr if x['retweet'] != 'N'], [x["name"] for x in tweets_objs_arr if x['retweet'] != 'N'], [x["followers"] for x in tweets_objs_arr if x['retweet'] != 'N']

def convert_emojis(tweets_arr):
    return [pymoji.replaceEmojiAlt(x, trailingSpaces=1) for x in tweets_arr]

def tokenize_tweets(tweets_arr):
    result = []
    for x in tweets_arr:
        try:
            tokenized = tokenizer.tokenize(x)
            result.append([x.lower() for x in tokenized if x not in string.punctuation])
        except:
            pass
#             print(x)
    return result

class Tweets(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, directories, filenames in os.walk(self.dirname):
            for filename in filenames:
                if(filename.endswith('json')):
                    print(root + filename)
                    with open(os.path.join(root,filename), 'r') as f:
                        data = json.load(f)
                        data_parsed_step1, user_names, followers = keep_retweets(data)
                        data_parsed_step2 = convert_emojis(data_parsed_step1)
                        data_parsed_step3 = tokenize_tweets(data_parsed_step2)
                        for data, name, follower in zip(data_parsed_step3, user_names, followers):
                            yield name, data, follower


#model = gensim.models.Word2Vec(sentences, workers=2, window=5, sg = 1, size = 100, max_vocab_size = 2 * 10000000)
#model.save('tweets_word2vec_2017_1_size100_window5')
#print('done')
#print(time.time() - start_time)


# ### My gensim tinkering
# Tasks:
# - build the gensim dictionary
# - build the bow matrix using this dictionary (sparse matrix so memory friendly)
# - save the names and the dicitionary for later use
# 

# building the dictionary first, from the iterator
sentences = Tweets('/media/henripal/hd1/data/2017/1/') # a memory-friendly iterator
dictionary = corpora.Dictionary((tweet for _, tweet, _ in sentences))


# here we use the downloaded  stopwords from nltk and create the list
# of stop ids using the hash defined above
stop = set(stopwords.words('english'))
stop_ids = [dictionary.token2id[stopword] for stopword in stop if stopword in dictionary.token2id]

# and this is the items we don't want - that appear less than 20 times
# hardcoded numbers FTW
low_freq_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq  <1500]


# finally we filter the dictionary and compactify
dictionary.filter_tokens(stop_ids + low_freq_ids)
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)


# reinitializing the iterator to get more stuff
sentences = Tweets('/media/henripal/hd1/data/2017/1/')
corpus = []
name_to_follower = {}
names = []

for name, tweet, follower in sentences:
    corpus.append(tweet) 
    names.append(name)
    name_to_follower[name] = follower


# And now we save everything for later analysis
# 

with open('/media/henripal/hd1/data/name_to_follower.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in name_to_follower.items():
        writer.writerow([key, value])

with open('/media/henripal/hd1/dta/corpus_names.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(names)


# now we save the sparse bow corpus matrix using matrix market format
corpora.MmCorpus.serialize('/media/henripal/hd1/data/corp.mm', corpus)

# and we save the dictionary as a text file
dictionary.save('/media/henripal/hd1/data/dict')


import pandas as pd
import re
import math
from nltk.classify import NaiveBayesClassifier
import collections
import nltk


# Let's pull in the tweets from the extracted ICE tweets
df = pd.read_csv('https://s3.amazonaws.com/d4d-public/public/ice_extract.csv')


# Let's take a look at it
df.head()


# OK, now that we have that, let's save it to a text file. This will help get a quick way to look at the text
df['text'].to_csv('training_data/test.txt', index=False)


# Now we're going to completely switch gears and build a model to determine whether a statement is positive or negative. We're going to use quotes from movies review, which is a bit of a stretch, but it's a place to start. This method is from Andy Bromberg's webpage. My goal is to build on it, but for now let's just try to get it working
# 

#I have two files, one of positive statements (from movie reviews) and the other with negative. To run this
#download the files from Andy Bromberg's GitHub page: https://github.com/abromberg/sentiment_analysis_python
positive_statements = 'C:/Users/HMGSYS/Google Drive/JupyterNotebooks/Data4Democracy/training_data/pos.txt'
negative_statements = 'C:/Users/HMGSYS/Google Drive/JupyterNotebooks/Data4Democracy/training_data/neg.txt'
#And I also have our file we just created with ICERaids tweets
test_statements = 'C:/Users/HMGSYS/Google Drive/JupyterNotebooks/Data4Democracy/training_data/test.txt'


#creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])


# Let's open the files and create lists with all the words in them
posFeatures = []
negFeatures = []
mytestFeatures = []
with open(positive_statements, 'r') as posSentences:
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords = [make_full_dict(posWords), 'pos']
        posFeatures.append(posWords)
with open(negative_statements, 'r') as negSentences:
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords = [make_full_dict(negWords), 'neg']
        negFeatures.append(negWords)
# Now let's do the same with our test data
with open(test_statements, 'r') as mytestSentences:
    for i in mytestSentences:
        mytestWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        # We're going to label them as positive so we can check the accuracy
        mytestWords = [make_full_dict(mytestWords), 'pos']
        mytestFeatures.append(mytestWords)


# Let's take a quick look at our result
posFeatures[0:2]


# Now we have two big lists: posFeatures and negFeatures. These are lists of lists, where each internal list is a collection of all the words that are in a positive movie review. Inside those lists are two things: a dictionary and a string. The dictionary is a mapping of every word in the review to a boolean (True). The string is either 'pos' or 'neg' depending on which corpus it came from.
# 

#selects 3/4 of the features to be used for training and 1/4 to be used for testing
posCutoff = int(math.floor(len(posFeatures)*3/4))
negCutoff = int(math.floor(len(negFeatures)*3/4))
mytestCutoff = int(math.floor(len(mytestFeatures)*3/4))
#Now this is a bit tricky because we have testFeatures and mytestFeatures. testFeatures is from the Bromberg model
#mytestFeatures is me throwing our test (ICERaids) tweets into the same process
trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
#This last one doesn't change
mytestFeatures = mytestFeatures


# We'll start with a Naive Bayes Classifier. There's a lot more we could do here but it's a start
classifier = NaiveBayesClassifier.train(trainFeatures)

#initiates referenceSets and testSets
referenceSets = collections.defaultdict(set)
testSets = collections.defaultdict(set)


# puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(testFeatures):
    referenceSets[label].add(i)
    predicted = classifier.classify(features)
    testSets[predicted].add(i)


#prints metrics to show how well the feature selection did
print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
print ('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
print ('pos precision:', nltk.scores.precision(referenceSets['pos'], testSets['pos']))
print ('pos recall:', nltk.scores.recall(referenceSets['pos'], testSets['pos']))
print ('neg precision:', nltk.scores.precision(referenceSets['neg'], testSets['neg']))
print ('neg recall:', nltk.scores.recall(referenceSets['neg'], testSets['neg']))


# Now we have a Naive Bayes Classifier that looks at words in movie reviews and predicts whether that review is positive or negative. As it stands, the accuracy is only 77%, which means that we're on the right path (better than just guessing) but it's not very impressive. Still, we have a bunch of words that correlate with a positive or negative review. Let's take a look at some of the most predictive words and see what we've got
# 

classifier.show_most_informative_features(10)


# Those words aren't great for determining someone's thoughts about ICE raids. "Engrossing" is definintely a movie word. It's also interesting that "flaws" is such a positive word.
# 

# OK, now for the moment of truth - let's see what percentage of tweets about the ICE raids this model classifies as positive. Remember, it only got 77% right for movie reviews, so this could be wildly inaccurate.
# 

print ('This model predicts that {:.1%} of tweets about the ICE raids have been positive'
       .format(nltk.classify.util.accuracy(classifier, mytestFeatures)))


# Well, we got significantly over 50%. Can we conclude that the majority of tweets about this have been positive? Perhaps. We could also check it every month or so to see how this number changes over time.
# 




get_ipython().magic('matplotlib inline')
import pandas as pd
import datetime
import ast
import tldextract


# You will need access to D4D data.world organization. Check in slack for more info
# 150mb / 240k rows. Give it time, has to download over internet
df = pd.read_csv('https://query.data.world/s/bbokc1f08to11j19j5axvkrcv', sep='\t', parse_dates=['date'])


df.set_index('date', inplace=True)
df.count()


# ### Articles by year (2 months of 2012 missing)
# 

by_year=df.groupby([pd.TimeGrouper('A')]).count()['title']
by_year


by_year.plot()


# ## Category publications by year
# 

df.groupby([pd.TimeGrouper('A'),'category']).count()['title']


# ### Top 25 authors
# 

df.groupby(['author']).count()['title'].sort_values(ascending=0).head(25)


# ### Hacky attempt to explore most common top level domains linked in articles
# 

from collections import Counter
tld_counter = Counter()


def get_tld(hrefs):
    
    # Quick and dirty, not thorough yet
    for link in ast.literal_eval(hrefs):
        top_level = tldextract.extract(link)
        top_level = top_level.domain
        tld_counter[top_level] += 1


_ = df[['hrefs']].applymap(get_tld)


tld_counter.most_common(25)


# Experimenting with Gensim/Word2Vec on tweets collected by the folks at the discursive project. Also making use of the gensim model built from ~ 400 million Twitter posts (built by Fréderic Godin , available at http://www.fredericgodin.com/software/)
# 

import gensim
import pymongo
import json
import numpy as np
import pandas as pd
from pymongo import MongoClient


import requests


from gensim import corpora, models, similarities


mongoClient = MongoClient()
db = mongoClient.data4democracy
tweets_collection = db.tweets


from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import smart_open, simple_preprocess
def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]


tweets_model = Word2Vec.load_word2vec_format('../../../../Volumes/SDExternal2/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')


#now calculate word simiarities on twitter data e.g.  
tweets_model.most_similar('jewish')


#to remind myself what a tweet is like:
r = requests.get('https://s3-us-west-2.amazonaws.com/discursive/2017/1/10/18/tweets-25.json')


tweets_collection = r.json()
print(tweets_collection[0])
#for text analysis, the 'text' field is the one of interest


#the tweets text are in the 'text' field
print(tweets_collection[0]['text'])


# The following is a bit of experimentation/learning with gensim -- following along some tutuorials on the gensim site to vectorize text, find tfidf etc
# 

tweets_text_documents = [x['text'] for x in tweets_collection]


#quick check that the mapping was done correctly
tweets_text_documents[0]


#quick check of the tokenize function -- remove stopwords included 
tokenize(tweets_text_documents[0])


tokenized_tweets = [[word for word in tokenize(x) if word != 'rt'] for x in tweets_text_documents]


tokenized_tweets[0]


#construct a dictoinary of the words in the tweets using gensim
# the dictionary is a mapping between words and their ids
tweets_dictionary = corpora.Dictionary(tokenized_tweets)


#save gyhe dict for future reference
tweets_dictionary.save('temp/tweets_dictionary.dict')


#just a quick view of words and ids
dict(list(tweets_dictionary.token2id.items())[0:20])


#convert tokenized documents to vectors
# compile corpus (vectors number of times each elements appears)
tweet_corpus = [tweets_dictionary.doc2bow(x) for x in tokenized_tweets]
corpora.MmCorpus.serialize('temp/tweets_corpus.mm', tweet_corpus) # save for future ref


tweets_tfidf_model = gensim.models.TfidfModel(tweet_corpus, id2word = tweets_dictionary)


tweets_tfidf_model[tweet_corpus]


#Create similarity matrix of all tweets
'''note from gensim docs: The class similarities.MatrixSimilarity is only appropriate when 
   the whole set of vectors fits into memory. For example, a corpus of one million documents 
   would require 2GB of RAM in a 256-dimensional LSI space, when used with this class.
   Without 2GB of free RAM, you would need to use the similarities.Similarity class.
   This class operates in fixed memory, by splitting the index across multiple files on disk, 
   called shards. It uses similarities.MatrixSimilarity and similarities.SparseMatrixSimilarity internally,
   so it is still fast, although slightly more complex.'''
index = similarities.MatrixSimilarity(tweets_tfidf_model[tweet_corpus]) 
index.save('temp/tweetsSimilarity.index')


#get similarity matrix between docs: https://groups.google.com/forum/#!topic/gensim/itYEaOYnlEA
#and check that the similarity matrix is what you expect
tweets_similarity_matrix = np.array(index)
print(tweets_similarity_matrix.shape)


#save the similarity matrix and associated tweets to json
#work in progress-- use tSNE to visualise the tweets to see if there's any clustering
outputDict = {'tweets' : [{'text': x['text'], 'id': x['id_str'], 'user': x['original_name']} for x in tweets_collection], 'matrix': tweets_similarity_matrix.tolist()}
with open('temp/tweetSimilarity.json', 'w') as f:
    json.dump(outputDict, f)


#back to the word2vec idea, use min_count=1 since corpus is tiny
tweets_collected_model = gensim.models.Word2Vec(tokenized_tweets, min_count=1)


#looking again at the term jewish in our small tweet collection...
tweets_collected_model.most_similar('jewish')


# next step is to loop through the data on s3 and build up a bigger corpus of tweets from the 
# 

# # Muslim ban analysis
# 
# Preliminary analysis of tweets set from January 28
# 

get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tweets = pd.read_json("https://s3.amazonaws.com/far-right/twitter/mb_protests.json")
tweets


from pprint import pprint 
print('Top Tweeters')
user_tweet_counts = tweets['user_name'].value_counts()
pprint(user_tweet_counts[:20])

hi_tweeters = user_tweet_counts[user_tweet_counts > 20]
plt.title('High volume tweeters tweet counts')
plt.hist(hi_tweeters, bins = 20)
plt.show()

lo_tweeters = user_tweet_counts[user_tweet_counts <= 20]
plt.title('Lo volume tweeters tweet counts')
plt.hist(lo_tweeters, bins = 20)
plt.show()


def total_tweet_count(tweets):
    return len(tweets)

def unique_tweets(tweets):
    return tweets['text'].unique()

def unique_tweet_count(tweets):
    return len(unique_tweets(tweets))

def unique_users(tweets):
    return tweets['user_name'].unique()

def unique_users_count(tweets):
    return len(unique_users(tweets))

print(str(total_tweet_count(tweets)) + " total tweets")

print(str(unique_tweet_count(tweets)) + " unique tweets")

print(str(unique_users_count(tweets)) + " unique user_names")


def hashtags_table(tweets):
    hashtags= {}
    for row in tweets['hashtags'].unique():
        row = eval(row)
        for tag in row:
            tag = tag.lower()
            if tag not in hashtags:
                hashtags[tag] = 1
            else:
                hashtags[tag] = hashtags[tag] + 1
    hashtagspd = pd.DataFrame(list(hashtags.items()), columns=['hashtag', 'count']).sort_values('count', ascending=False)
    return hashtagspd
hashtags = hashtags_table(tweets)
print("Top 20 hashtags")
pprint(hashtags[:20])





# Here are the imports we'll use
import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer


# Let's grab all the tweets from https://data.world/data4democracy/far-right/file/sample_tweets.json. Here is the URL:
df = pd.read_json('https://query.data.world/s/bsbt4eb4g8sm4dsgi7w2ecbkt')


# Let's take a look at it
df.head()
# Does anyone know the difference between message and text?


print('Now we have all the tweets inside a {}'.format(type(df)))
print('There are a total of {} tweets in our dataset'.format(len(df)))


# Let's see what different columns we have
print('Here are the columns we have: \n   {}'.format(df.columns))


# It looks like the topics section is the same for all tweets. These were the search terms used to collect the data
# 

# Let's start by tokenizing all the words
df['tokenized'] = df['text'].apply (lambda row: nltk.word_tokenize(row))


# Let's add part of speech tags. This function can take a bit of time if it's a large dataset
df['tags']=df['tokenized'].apply(lambda row: nltk.pos_tag(row))


# Now let's remove stop words (e.g. and, to, an, etc.)
# We'll build a little function for that
def remove_stop_words(text):
    filtered = [word for word in text if word not in nltk.corpus.stopwords.words('english')]
    return filtered

df['no_stop'] = df['tokenized'].apply(lambda row: remove_stop_words(row))


# Now we can stem the remaining words
stemmer = SnowballStemmer("english")
df['stems'] = df['no_stop'].apply(lambda words: 
                                    [stemmer.stem(word) for word in words])


# OK, let's take another look at the dataframe
df.head()


# Let's see what variables we have so we know what to store for other datasets
get_ipython().magic('who')


# Since we add everything to the dataframe that's the only thing that appears to be worth storing
get_ipython().magic('store df')


# To access this dataframe from another notebook, simply run this notebook from your other notebook with this command:
# %run ./CleanText.ipynb
# 
# Then, to load a variable use this command:
# %store -r df
# 
# You can look at the Hashtag Analysis notebook for an example.

# ## This note book contains code for pulling and aggregating the bi-daily scapes from 4chan into one dataframe which can be easily used for further analysis
# 
# If you don't have access to s3, here's an excellent guide on obtaining access. Keys can be obtained by asking moderaters in the slack channel for far-right https://github.com/Data4Democracy/tutorials/blob/master/aws/AWS_Boto3_s3_intro.ipynb
# 
# From this notebook anyone should be able to work on some more analysis, and the text of the messages from these scrapes looks very clean.  
# 
# If you only want to pull a certain set of dates, just adjust the regex in match_string, or add some more
# conditionals to the loop which grabs the list of files to be read.
# 
# You can also read fewer files (if you have a slow connection or not that much memory) by shortening the files list before the second for loop.
# 

import boto
import boto3
import pandas as pd
import re
from IPython.display import clear_output


session = boto3.Session(profile_name='default')
s3 = session.resource('s3')
bucket = s3.Bucket("far-right")
session.available_profiles


base_url = 's3:far-right/'
match_string = "info-source/daily/[0-9]+/fourchan/fourchan"

files = []
print("Getting bucket and files info")
for obj in bucket.objects.all():
    if bool(re.search(match_string, obj.key)):
        files.append(obj.key)
        
df = pd.DataFrame()
for i, file in enumerate(files):
    clear_output()
    print("Loading file: " + str(i + 1) + " out of " + str(len(files)))
    if df.empty:
        df = pd.read_json(base_url + file)        
    else:
        df = pd.concat([df, pd.read_json(base_url + file)])
    
clear_output()
print("Completed Loading Files")


df.shape


df.head()





get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime


df = pd.read_json("https://s3.amazonaws.com/far-right/twitter/mb_protests.json")


df.columns


print("Total number of tweets = {}".format(len(df)))


# ##### How many tweets are about the 'wall'?
# 

# Lowercase the hashtags and tweet body
df['hashtags'] = df['hashtags'].str.lower()
df['text'] = df['text'].str.lower()


print("Total number of tweets containing hashtag 'wall' = {}".format(len(df[df['hashtags'].str.contains('wall')])))


print("Total number of tweets whose body contains 'wall' = {}".format(len(df[df['text'].str.contains('wall')])))


wall_tweets = df[(df['hashtags'].str.contains('wall')) | (df['text'].str.contains('wall'))].copy()


print("Total number of tweets about the 'wall' = {}".format(len(wall_tweets)))


# ##### What is the average twitter tenure of people who tweeted about the wall?
# 

def months_between(end, start):
    return (end.year - start.year)*12 + end.month - start.month


wall_tweets['created'] = pd.to_datetime(wall_tweets['created'])
wall_tweets['user_created'] = pd.to_datetime(wall_tweets['user_created'])


wall_tweets['user_tenure'] = wall_tweets[['created',                             'user_created']].apply(lambda row: months_between(row[0], row[1]), axis=1)


tenure_grouping = wall_tweets.groupby('user_tenure').size() / len(wall_tweets) * 100

fig, ax = plt.subplots()

ax.plot(tenure_grouping.index, tenure_grouping.values)

ax.set_ylabel("% of tweets")
ax.set_xlabel("Acct tenure in months")

plt.show()


# ##### There are a couple of users tweeting multiple times, but most tweets come from distinct twitter handles 
# 

tweets_per_user = wall_tweets.groupby('user_name').size().sort_values(ascending=False)

fig, ax = plt.subplots()

ax.plot(tweets_per_user.values)

plt.show()


# ##### Who are the 'top tweeters' + descriptions?
# 

wall_tweets.groupby(['user_name', 'user_description']).size().sort_values(ascending=False).head(20).to_frame()


# ##### What is the reach of these tweets in terms of followers?
# 

plt.boxplot(wall_tweets['friends_count'].values, vert=False)
plt.show()


wall_tweets['friends_count'].describe()


# ##### Location of the tweets?
# 

wall_tweets.groupby('user_location').size().sort_values(ascending=False)





