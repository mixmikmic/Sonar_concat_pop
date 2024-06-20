# # Bayesian Optimization of Neural Network Learning Rate
# 
# Here a single node neural network is used to evaluate the correlation between the Google Trends score of "bitcoin" and the price of BTC on the Bitfinex exchange over time. 
# 
# <img src="bitcoin.png">
# 
# <img src="Google-Trends.png">
# 
# 
# The learning rate of this neural network will be optimized using bayesian optimization.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# ## Data Preprocessing
# 
# I collected two CSVs for this investigation. The first details the daily prices of Bitcoin in USD on Bitfinex, a major bitcoin exchange. The second states the Google Trends score for the "bitcoin" search term on a weekly basis. 
# 
# I have matched each week in the Google Trends dataset with a price from that week. This doesn't give us a perfect averge of the BTC/USD price for each week but it does give us an accurate idea of what the price was like for the weekly time period. 
# 

bitfinex_prices = pd.read_csv("./bitfinex-USDprice.csv")


bitfinex_prices.head()


bitfinex_prices.tail()


bitcoin = pd.read_csv('./googletrends-bitcoin.csv')


bitcoin.head()


weekly_price = []

for week in bitcoin['Week']:
    w = bitfinex_prices.loc[bitfinex_prices['Date'] == week]
    price = w['High'].values
    if len(price) == 0:
        weekly_price.append(np.nan)
    else:
        weekly_price.append(price[0])
     


bitcoin['Price'] = weekly_price


bitcoin


# Drop weeks for which there are missing values 
bitcoin = bitcoin.dropna(axis=0, how='any')


data = bitcoin[['bitcoin','Price']].values


plt.plot(data[:,1],data[:,0], "bo")
plt.xlabel('price (USD)')
plt.ylabel('Google Trends Score')
plt.show()


# Normalize the data
X = (data[:,1] - np.mean(data[:,1]))/np.std(data[:, 1])
Y = (data[:,0] - np.mean(data[:,0]))/np.std(data[:, 0])


plt.plot(X,Y, "bo")
plt.xlabel('price (USD)')
plt.ylabel('Google Trends Score')
plt.show()


def simplest_neural_net(x, y, epochs, learning_rate):
    weights = np.array([0, 0])
    bias = 1.
    for e in range(epochs):
        gradient = np.array([0., 0.])
        for i in range(len(x)):
            xi = x[i]
            xi = np.array([bias, xi])
            yi = y[i]
            
            h = np.dot(weights, xi) - yi
            gradient += 2*xi*h
        
        weights = weights - learning_rate*gradient
    return weights[1], weights[0]


# Here the ideal values for slope and y-intercept are converged upon
m, b = simplest_neural_net(X,Y,100, 1e-3)
target_m = m
target_b = b


x_points = np.linspace(np.min(X), np.max(X), 10)
line = b + m*x_points


plt.plot(X,Y, 'bo', x_points, line, 'r-')
plt.show()


# Cool, there seems to be a correlation here based on the line of best fit produced by the neural net's emulation of linear regression. Maybe you should check Google Trends next time you are thinking about buying bitcoin.
# 
# But deciding if we should buy bitcoin or not isn't why we are here...
# 
# # Learning Rate Optimization
# 
# To perform a bayesian optimization of the learning rate I will take the value for the *slope* and *y intercept* that I converged upon over many epochs above and use those values as a model in an evalutation function. In the evaluation function below I drastically reduced the amount of epochs the nerual net is being given to fit the data. Given it doesn't have enough epochs to converge on the ideal *slope* and *y intercept* we will be able to see which learning rate gets the neural net closest to the ideal values in a short amount of epochs.
# 

from sklearn.metrics import mean_squared_error

def evaluation_fxn(x, y, learn_rate, ideal_m, ideal_b):
    x_points = np.linspace(np.min(x), np.max(x), 10)
    ideal_line = ideal_m*x_points + ideal_b
    m, b = simplest_neural_net(x,y,5,learn_rate)
    test_line = m*x_points + b
    
    return 1 - mean_squared_error(ideal_line, test_line)
    


# Make some inital guesses about the learning rate and evaluate them
# The Gaussian Process will be fit to this data initially.
guesses = [6e-3,1e-3,1e-4]

outcomes = [evaluation_fxn(X,Y,guess, target_m, target_b)
                     for guess in guesses]


from sklearn.gaussian_process import GaussianProcess
import warnings
warnings.filterwarnings('ignore')


plt.plot(guesses,outcomes,'ro')
plt.xlabel('learning rate guesses')
plt.ylabel('score out of 1')
plt.show()


def hyperparam_selection(guesses, outcomes):
    guesses = np.array(guesses)
    outcomes = np.array(outcomes)
    gp = GaussianProcess(corr='squared_exponential',
                         theta0=1e-1, thetaL=1e-3, thetaU=1)
    
    gp.fit(guesses.reshape((-1,1)), outcomes)
    
    x = np.linspace(np.min(guesses), np.max(guesses), 10)
    
    mean, var = gp.predict(x.reshape((-1,1)), eval_MSE=True)
    std = np.sqrt(var)
    
    expected_improv_lower = mean - 1.96 * std
    expected_improv_upper = mean + 1.96 * std
    
    acquisition_curve = expected_improv_upper - expected_improv_lower
    
    
    idx = acquisition_curve.argmax()
    
    next_param = x[idx]
    
    plt.plot(guesses,outcomes,'ro', label='observations')
    plt.plot(x,mean, 'b--', label='posterior mean')
    plt.plot(x, expected_improv_lower, 'g--', label='variance')
    plt.plot(x, expected_improv_upper, 'g--')
    plt.plot(x, acquisition_curve, 'k--', label='acquisition fxn')
    plt.plot(x[idx],acquisition_curve[idx], 'yX', label='next guess')
    plt.xlabel('learning rate')
    plt.ylabel('score out of 1')
    plt.legend(loc='best')
    plt.show()
    
    return next_param


for _ in range(10):
    
    try:
        new_learning_rate = hyperparam_selection(guesses,outcomes) 
    except:
        print("optimal learning rate found")
        break
    
    guesses.append(new_learning_rate)
    score = evaluation_fxn(X,Y,new_learning_rate, target_m, target_b)
    print("Suggested learning rate: ",new_learning_rate)
    outcomes.append(score)


# Here we found an optimizated learning rate by using an acquisition function. This acquisition function identified the next guess for the hyperparameter by taking the posterior mean plus a constant times the variance. 
# 

# # Spotting Trending Topics in Scientific Research with Latent Dirichlet Allocation
# 
# For this week's challenge I decided to spend some time finding a novel text dataset and dabbling in a little data mining. After bumming around the internet for a while I found my target: **Nature.com**
# 
# As far as I know nature.com does not provide any API service for programmatically accessing their content. While they have been nice enough to make some papers "open access", which means they are free to download as a PDF or view in browser as html, there's no way I am gonna point click drag copy paste through a hundred webpages to get the volume of data I would like for this notebook. 
# 
# Fortunately, python is good for more than data analysis. 
# 
# And with a url as straight forward as this
# 
# `https://www.nature.com/search?article_type=protocols,research,reviews&subject=biotechnology&page=3`
# 
# who needs an api???
# 
# There are two python files in this directory that scraped research papers off nature.com for use in the this notebook's dataset. 
# 
# The [first script](https://github.com/NoahLidell/math-of-intelligence/blob/master/generative_models/collect-article-html.py), `collect-article-html.py`, plugs different [keywords](https://www.nature.com/subjects) in the `&subject=` placeholder in the url and goes through the first eight pages of search results. Each search result page's html is loaded into an html parser (lxml) and the link and title for all 25 articles on the results page is accessed through the xpath for the respective html elements. The article links gathered are followed and the raw page html for the research paper document is saved in a database (mysql) along with the articles' title, date, etc. 
# 
# I ran this `collect-article-html.py` repeatedly across 32 different searchable keywords and in a few hours pulled down over 3000 research papers.
# 
# The [second script](https://github.com/NoahLidell/math-of-intelligence/blob/master/generative_models/process-html.py), `process-html.py`, pulls the raw html from the db for every article downloaded. Filtering the research paper text out of the html document proved easier than I expected. I used the BeautifulSoup4 library to removal all the html tags and then with just the page text leftover it was as easy as telling python to only keep the text after the "Abstract" substring and before the "References" substring. Additional text preprocessing was done in this script, removing all special characters, numbers, and excess whitespace.  
# 
# I dumped all of the article text data out of mysql and into a sqlite database file for this notebook to pull its data from. The sqlite db is over 100MB so I couldn't upload it directly to github. If you trying to run this notebook, you'll need to unzip the `article_db.zip` file in this directory.
# 
# I was motived to compile this dataset since I believe the machine learning isn't the only discipline where interesting things are happening right now. What about CRISPR? What about quantum computing? What about nano technology? I don't know anything about those topics, but they seem interesting... So this notebook is my attempt to gather and explore data on current research across a variety of fields, using LDA as a method for identifying keywords and topics within larger fields such as biotechnology and physics. 
# 

import sqlite3
import pandas as pd
from gensim import corpora, models, similarities
import nltk
from collections import Counter


# ### Load the DB
# The table where the articles are stored is called `articles`.
# 
# The columns are:
# 
# 
# id | title | text | url | topic | journal | date | type | wc
# --- | --- | --- | --- | --- | --- | --- | --- | ---
# int | mediumtext | longtext | mediumtext | varchar(245) | varchar(245) | varchar(245) | varchar(245) | int
# 

conn = sqlite3.connect('./database/nature_articles.db')
cursor = conn.cursor()
num_articles = cursor.execute('SELECT count(distinct title) FROM articles WHERE wc > 1500;').fetchall()[0][0]
print('Number of unquie articles in dataset: ', num_articles)

df = pd.read_sql_query("SELECT distinct(title), text, url, journal, date FROM articles WHERE wc > 1500 ORDER BY random();",
                       conn)
df.head()


# ### Here is a sample article from the dataset
# 

title, subject, article = cursor.execute("SELECT title, topic, text FROM articles ORDER BY random() LIMIT 1;").fetchall()[0]
print("\n", title)
print("\nSubject:", subject)
print("\n\t", article)


subjects = cursor.execute("SELECT distinct topic FROM articles;").fetchall()
print("Subjects in dataset:\n")
for s in subjects:
    print('\t',s[0])


def render_topics(subjects, num_topics=3, stem=False, filter_n_most_common_words=500, num_words=30):
    if isinstance(subjects, str):
        df = pd.read_sql_query("SELECT distinct(title), text FROM articles WHERE wc > 1500 and topic = '{}';".format(subjects),
                               conn)
        
    
    else:
        df = pd.read_sql_query("SELECT distinct(title), text FROM articles WHERE wc > 1500 and topic IN {};".format(subjects),
                               conn)
    
    docs = df['text'].values
    split_docs = [doc.split(' ') for doc in docs]
    doc_words = [words for doc in split_docs for words in doc]
    wcount = Counter()
    wcount.update(doc_words)
    stopwords = nltk.corpus.stopwords.words('english') + ['introduction','conclusion'] # filter out terms used as section titles in most research papers
    for w, _ in wcount.most_common(filter_n_most_common_words):
        stopwords.append(w)
        
    if stem == True:
        docs = [stem_and_stopword_filter(doc, stopwords) for doc in docs]
    else:
        docs = [stopword_filter(doc, stopwords) for doc in docs]
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda_model = models.LdaMulticore(corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda_model.show_topics(formatted=False, num_words=num_words)
    
    print(subjects)
    
    for t in range(len(topics)):
        print("\nTopic {}, top {} words:".format(t+1, num_words))
        print(" ".join([w[0] for w in topics[t][1]]))
        
    
        
        
def stem_and_stopword_filter(text, filter_list):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    return [stemmer.stem(word) for word in text.split() if word not in filter_list and len(word) > 2]

def stopword_filter(text, filter_list):
    return [word for word in text.split() if word not in filter_list and len(word) > 2]


# specific subjects to analyze for topics as a tuple of strings
# ie subjects = ('philosophy', 'nanoscience-and-technology', 'biotechnology')
subjects = ('philosophy')

render_topics(subjects, num_topics=9, stem=False, filter_n_most_common_words=500)


# ### Discussion
# For the all the subjects that I pulled from nature.com, the philosophy articles seemed to present the clearest themes in the topics generated by LDA. I think this is because in philosophy you have different areas (game theory, ethics, theology, etc) which have established jargon that is specific to that sub field of philosophy but used widely within that sub field (ie, people who study ethics all have some take on Kant). 
# 
# Contrast this with the topics generated by LDA for the scientific disciplines. Even within a narrow subfield, such as nano technology, papers seem to have very specific subject matter. Instead of words representing subjects of study and inquery reocurring across texts within a scientific subfield, you have terms related to the execution and process of science occurring prominently (terms like 'datasets', 'index', 'amount'). 
# 

render_topics(('mathematics-and-computing','computational-biology-and-bioinformatics','nanoscience-and-technology'),
               num_topics=9, stem=False, filter_n_most_common_words=500)


subjects = cursor.execute('SELECT distinct topic FROM articles;').fetchall()

for s in subjects:
    render_topics(s[0], num_topics=9, stem=False, filter_n_most_common_words=500) 
    print('==================================================================================================')


