# # A graph based approach to opinion mining
# 
# Based on the work done by Kavita Ganesan [Opinosis](http://kavita-ganesan.com/opinosis) for mining aspects from product reviews, this notebook fetches product reviews from the Best Buy API and builds a word adjacency graph of the review corpus in Neo4j. Opinions are then minded from the graph using a simplied version of the Opinosis algorithm.
# 

from py2neo import Graph
import json
import requests
import re, string
from py2neo.packages.httpstream import http
http.socket_timeout = 9999


API_KEY = "BEST_BUY_API_KEY"
# SKU = "9439005" # Kindle
# SKU = "4642026" # Bose headphones
# SKU = "6422016" # Samsung TV
# SKU = "3656051" # Samsung washing machine
# SKU = "2498029" # Dyson vacuum

REQUEST_URL = "https://api.bestbuy.com/v1/reviews(sku={sku})?apiKey={API_KEY}&show=comment,id,rating,reviewer.name,sku,submissionTime,title&pageSize=100&page={page}&sort=comment.asc&format=json"


# ### Connect to Neo4j instance and define Cypher queries
# 

graph = Graph()

# Build a word adjacency graph for a comment string
INSERT_QUERY = '''
WITH split(tolower({comment}), " ") AS words
WITH [w in words WHERE NOT w IN ["the","and","i", "it", "to"]] AS text
UNWIND range(0,size(text)-2) AS i
MERGE (w1:Word {name: text[i]})
ON CREATE SET w1.count = 1 ON MATCH SET w1.count = w1.count + 1
MERGE (w2:Word {name: text[i+1]})
ON CREATE SET w2.count = 1 ON MATCH SET w2.count = w2.count + 1
MERGE (w1)-[r:NEXT]->(w2)
  ON CREATE SET r.count = 1
  ON MATCH SET r.count = r.count + 1;
'''

OPINION_QUERY = '''
MATCH p=(:Word)-[r:NEXT*1..4]->(:Word) WITH p
WITH reduce(s = 0, x IN relationships(p) | s + x.count) AS total, p
WITH nodes(p) AS text, 1.0*total/size(nodes(p)) AS weight
RETURN extract(x IN text | x.name) AS phrase, weight ORDER BY weight DESC LIMIT 10
'''


# define a regular expression to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
# exclude = set(string.punctuation)


# ### Fetch comments from Best Buy API and build word adjaceny graph
# 

def load_graph(product_sku):
    for i in range(1,6):
        r = requests.get(REQUEST_URL.format(sku=product_sku, API_KEY=API_KEY, page=str(i)))
        data = r.json()
        for comment in data["reviews"]:
            comments = comment["comment"].split(".")
            for sentence in comments:
                sentence = sentence.strip()
                sentence = regex.sub("", sentence)
                graph.cypher.execute(INSERT_QUERY, parameters={'comment': sentence})


# #### Query the graph for opinions
# Find word paths of 3-5 words with highest number of occurences
# 

def summarize_opinions():
    results = graph.cypher.execute(OPINION_QUERY)
    for result in results:
        print(str(result.phrase) + " " + str(result.weight))


# ### Bose headphones
# <img src="img/bose.png" align="right" style="width:20%;">
# * "They are great sound quality"
# * "Comfortable and the great sound"
# * "These headphones great sound quality"
# 

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("4642026")
summarize_opinions()


# ### Samsung TV
# <img src="img/tv.png" align="left" style="width:20%;">
# * "Bought this smart TV for the price"
# 

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("6422016")
summarize_opinions()


# ### Amazon Kindle
# <img src="img/kindle.png" align="right" style="width:20%;">
# * "Easy to read"
# * "Easy to read in the light"
# 

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("9439005")
summarize_opinions()


# ### Samsung washer
# <img src="img/washer.png" align="left" style="width:20%;">
# * "I love this washer"
# 

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("3656051")
summarize_opinions()


# ### Dyson vacuum
# <img src="img/dyson.png" align="right" style="width:20%;">
# * "Easy to use this vacuum"
# 

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("2498029")
summarize_opinions()








# ## Graph based content recommendation using keyword extraction
# 
# This notebook uses a small sample of articles to demonstrate how a content based recommendation system can be implemented using the Neo4j graph database. A small sample of articles of interest are used to seed the graph. Keyword extraction is performed using the [newspaper python library](https://github.com/codelucas/newspaper). Candidate articles for recommendation are then scraped and inserted into the graph model. Content based graph queries are then used to generate article recommendations based on the user's interest.
# 
# ![](https://dl.dropboxusercontent.com/u/67572426/Screenshot%202016-01-15%2018.09.47.png)

import requests
import newspaper
from newspaper import Article
from xml.etree import ElementTree
from py2neo import Graph


graph = Graph()


INSERT_ARTICLE_QUERY = '''
    MERGE (u:URL {url: {url}})
    SET u.title = {title}
    FOREACH (keyword IN {keywords} | MERGE (k:Keyword {text: keyword}) CREATE UNIQUE (k)<-[:IS_ABOUT]-(u) )
    FOREACH (img IN {images} | MERGE (i:Image {url: img})<-[:WITH_IMAGE]-(u) )
    FOREACH (vid IN {videos} | MERGE (v:Video {url: vid})<-[:WITH_VIDEO]-(u) )
    FOREACH (author IN {authors} | MERGE (a:Author {name: author})<-[:AUTHORED_BY]-(u) )    
'''

INSERT_LIKED_QUERY = '''
    MERGE (u:User {name: {username}})
    MERGE (a:URL {url: {url}})
    CREATE UNIQUE (u)-[:LIKED]->(a)
'''


# insert liked articles
for u in liked_articles:
    insertLikedArticle("lyonwj", u)
    article = newspaper_article(u)
    writeToGraph(article)
    


# insert newspaper articles
for url in newspapers:
    p = newspaper.build(url)
    for article in p.articles:
        parsed_a = newspaper_article(article.url)
        writeToGraph(parsed_a)


# articles from the read later queue
liked_articles = [
    'http://paulgraham.com/ineq.html',
    'https://codewords.recurse.com/issues/five/what-restful-actually-means',
    'http://priceonomics.com/the-history-of-the-black-scholes-formula/',
    'https://buildingrecommenders.wordpress.com/2015/11/16/overview-of-recommender-algorithms-part-1/',
    'http://blog.crew.co/makers-and-managers/',
    'http://www.lrb.co.uk/v37/n22/jacqueline-rose/bantu-in-the-bathroom',
    'http://www.techrepublic.com/article/how-the-paypal-mafia-redefined-success-in-silicon-valley/',
    'http://www.bloomberg.com/bw/articles/2012-07-10/how-the-mormons-make-money',
    'https://jasonrogena.github.io/2015/10/09/matatus-route-planning-using-neo4j.html',
    'http://efavdb.com/principal-component-analysis/',
    'http://www.tsartsaris.gr/How-to-write-faster-from-Python-to-Neo4j-with-OpenMpi',
    'http://burakkanber.com/blog/machine-learning-full-text-search-in-javascript-relevance-scoring/',
    'https://www.pubnub.com/blog/2015-10-22-turning-neo4j-realtime-database/',
    'http://www.greatfallstribune.com/story/news/local/2016/01/12/montana-coal-mine-deal-includes-secret-side-settlement/78697796/',
    'http://billingsgazette.com/news/opinion/editorial/gazette-opinion/a-big-win-for-montana-businesses-taxpayers/article_ffa8c111-ce4b-508f-8813-8337b6d9a4b2.html',
    'http://billingsgazette.com/news/state-and-regional/montana/appeals-court-says-one-time-billionaire-will-stay-in-montana/article_90e41f92-60a5-5685-90ba-ad63721715c7.html',
    'http://missoulian.com/news/state-and-regional/missoula-man-seeks-a-fortune-in-anaconda-slag/article_c1fa2a2a-3468-56fe-a794-814f83a8eb6a.html',
    'http://www.theverge.com/2015/9/30/9416579/spotify-discover-weekly-online-music-curation-interview',
    'https://theintercept.com/2015/09/09/makers-zero-dark-thirty-seduced-cia-tequila-fake-earrings/',
    'https://www.quantamagazine.org/20150903-the-road-less-traveled/',
    'https://medium.com/@bolerio/scheduling-tasks-and-drawing-graphs-the-coffman-graham-algorithm-3c85eb975ab#.xm0lpx2l3',
    'http://www.datastax.com/dev/blog/tales-from-the-tinkerpop',
    'http://open.blogs.nytimes.com/2015/08/11/building-the-next-new-york-times-recommendation-engine/?_r=0',
    'http://www.economist.com/news/americas/21660149-voters-are-about-start-choosing-next-president-scion-and-heir?fsrc=scn/tw/te/pe/ed/TheScionAndTheHeir',
    'https://lareviewofbooks.org/essay/why-your-rent-is-so-high-and-your-pay-is-so-low-tom-streithorst',
    'http://www.economist.com/news/asia/21660551-propaganda-socialist-theme-park-relentless-so-march-money-bread-and-circuses?fsrc=scn/tw/te/pe/ed/BreadAndCircuses',
    'http://www.markhneedham.com/blog/2015/08/10/neo4j-2-2-3-unmanaged-extensions-creating-gzipped-streamed-responses-with-jetty/?utm_source=NoSQL+Weekly+Newsletter&utm_campaign=5836be97da-NoSQL_Weekly_Issue_246_August_13_2015&utm_medium=email&utm_term=0_2f0470315b-5836be97da-328632629',
    'https://medium.com/@dtauerbach/software-engineers-will-be-obsolete-by-2060-2a214fdf9737#.lac4umwmq',
    'http://www.nytimes.com/2015/08/16/opinion/sunday/how-california-is-winning-the-drought.html?action=click&pgtype=Homepage&module=opinion-c-col-right-region&region=opinion-c-col-right-region&WT.nav=opinion-c-col-right-region&_r=1'
]

# source for potential articles to recommend
newspapers = [
    'http://cnn.com',
    'http://news.ycombinator.com',
    'http://nytimes.com',
    'http://missoulian.com',
    'http://www.washingtonpost.com',
    'http://www.reuters.com/',
    'http://sfgate.com',
    'http://datatau.com',
    'http://economist.com',
    'http://medium.com',
    'http://theverge.com'
]





def insertLikedArticle(username, url):
    graph.cypher.execute(INSERT_LIKED_QUERY, {"username": username, "url": url})


def writeToGraph(article):

    #TODO: better data model, remove unnecessary data from data model
    insert_tx = graph.cypher.begin()
    insert_tx.append(INSERT_ARTICLE_QUERY, article)
    insert_tx.commit()





def newspaper_article(url):
    
    article = Article(url)
    article.download()
    article.parse()

    try:
        html_string = ElementTree.tostring(article.clean_top_node)
    except:
        html_string = "Error converting HTML to string"

    try:
        article.nlp()
    except:
        pass

    return {
        'url': url,
        'authors': article.authors,
        'title': article.title,
        'top_image': article.top_image,
        'videos': article.movies,
        'keywords': article.keywords,
        'images': filter_images(list(article.images))
    }





def filter_images(images):
    imgs = []
    for img in images:
        if img.startswith('http'):
            imgs.append(img)
    return imgs


# TODO: generate recommendations








# ## Natural Language Processing with Neo4j: Mining Paradigmatic Relations
# 
# This IPython notebook is the companion for this [blog post](http://lyonwj.com/2015/06/16/nlp-with-neo4j/) about getting started with Natural Language Processing using Neo4j. This notebook mostly covers data insertion.
# 

from py2neo import Graph
import re, string


# connect to Neo4j instance using py2neo - default running locally
graphdb = Graph('http://neo4j:neo4j@localhost:7474/db/data')


# define some parameterized Cypher queries

# For data insertion
INSERT_QUERY = '''
    FOREACH (t IN {wordPairs} | 
        MERGE (w0:Word {word: t[0]})
        MERGE (w1:Word {word: t[1]})
        CREATE (w0)-[:NEXT_WORD]->(w1)
        )
'''

# get the set of words that appear to the left of a specified word in the text corpus
LEFT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)-[:NEXT_WORD]->(s)
    RETURN w.word as word
'''

# get the set of words that appear to the right of a specified word in the text corpus
RIGHT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)<-[:NEXT_WORD]-(s)
    RETURN w.word as word
'''


# ### Data insertion
# #### Normalizing the data (lowercase, remove punctuation)
# 

# define a regular expression to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
exclude = set(string.punctuation)


# convert a sentence string into a list of lists of adjacent word pairs
# arrifySentence("Hi there, Bob!) = [["hi", "there"], ["there", "bob"]]
def arrifySentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = regex.sub('', sentence)
    wordArray = sentence.split()
    tupleList = []
    for i, word in enumerate(wordArray):
        if i+1 == len(wordArray):
            break
        tupleList.append([word, wordArray[i+1]])
    return tupleList


# #### Load the data
# The text corpus used here is the [CEEAUS](http://earthlab.uoi.gr/theste/index.php/theste/article/viewFile/55/37) [corpus](http://language.sakura.ne.jp/s/eng.html) which is distributed with the [MeTA NLP library](https://meta-toolkit.org/).
# 

def loadFile():
    tx = graphdb.cypher.begin()
    with open('data/ceeaus.dat', encoding='ISO-8859-1') as f:
        count = 0
        for l in f:
            params = {'wordPairs': arrifySentence(l)}
            tx.append(INSERT_QUERY, params)
            tx.process()
            count += 1
            if count > 300:
                tx.commit()
                tx = graphdb.cypher.begin()
                count = 0
    f.close()
    tx.commit()


loadFile()


# ### Calculating Paradigmatic relations
# 
# We first define two functions (`left1` and `right1`) to allow us to represent each word by its context (as a set of words). We then define a function to compute the [Jaccard similarity coefficient](https://en.wikipedia.org/wiki/Jaccard_index) given two arbitrary sets which will allow us to compute a measure of Paradigmatic similarity
# 

# return a set of all words that appear to the left of `word`
def left1(word):
    params = {
        'word': word.lower()
    }
    tx = graphdb.cypher.begin()
    tx.append(LEFT1_QUERY, params)
    results = tx.commit()
    words = []
    for result in results:
        for line in result:
            words.append(line.word)
    return set(words)


# return a set of all words that appear to the right of `word`
def right1(word):
    params = {
        'word': word.lower()
    }
    tx = graphdb.cypher.begin()
    tx.append(RIGHT1_QUERY, params)
    results = tx.commit()
    words = []
    for result in results:
        for line in result:
            words.append(line.word)
    return set(words)


# compute Jaccard coefficient
def jaccard(a,b):
    intSize = len(a.intersection(b))
    unionSize = len(a.union(b))
    return intSize / unionSize


# we define paradigmatic similarity as the average of the Jaccard coefficents of the `left1` and `right1` sets
def paradigSimilarity(w1, w2):
    return (jaccard(left1(w1), left1(w2)) + jaccard(right1(w1), right1(w2))) / 2.0





# What is the measure of paradigmatic similarity between "school" and "university" in the corpus?
paradigSimilarity("school", "university")





