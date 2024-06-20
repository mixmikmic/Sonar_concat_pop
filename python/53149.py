# # New Term Topics Methods and Document Coloring
# 

from gensim.corpora import Dictionary
from gensim.models import ldamodel
import numpy
get_ipython().magic('matplotlib inline')


# We're setting up our corpus now. We want to show off the new `get_term_topics` and `get_document_topics` functionalities, and a good way to do so is to play around with words which might have different meanings in different context.
# 
# The word `bank` is a good candidate here, where it can mean either the financial institution or a river bank.
# In the toy corpus presented, there are 11 documents, 5 `river` related and 6 `finance` related. 
# 

texts = [['bank','river','shore','water'],
        ['river','water','flow','fast','tree'],
        ['bank','water','fall','flow'],
        ['bank','bank','water','rain','river'],
        ['river','water','mud','tree'],
        ['money','transaction','bank','finance'],
        ['bank','borrow','money'], 
        ['bank','finance'],
        ['finance','money','sell','bank'],
        ['borrow','sell'],
        ['bank','loan','sell']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# We set up the LDA model in the corpus. We set the number of topics to be 2, and expect to see one which is to do with river banks, and one to do with financial banks. 
# 

numpy.random.seed(1) # setting random seed to get the same results each time.
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)


model.show_topics()


# And like we expected, the LDA model has given us near perfect results. Bank is the most influential word in both the topics, as we can see. The other words help define what kind of bank we are talking about. Let's now see where our new methods fit in.
# 

# ### get_term_topics
# 

# The function `get_term_topics` returns the odds of that particular word belonging to a particular topic. 
# A few examples:
# 

model.get_term_topics('water')


# Makes sense, the value for it belonging to `topic_0` is a lot more.
# 

model.get_term_topics('finance')


# This also works out well, the word finance is more likely to be in topic_1 to do with financial banks.
# 

model.get_term_topics('bank')


# And this is particularly interesting. Since the word bank is likely to be in both the topics, the values returned are also very similar.
# 

# ### get_document_topics and Document Word-Topic Coloring
# 

# `get_document_topics` is an already existing gensim functionality which uses the `inference` function to get the sufficient statistics and figure out the topic distribution of the document.
# 
# The addition to this is the ability for us to now know the topic distribution for each word in the document. 
# Let us test this with two different documents which have the word bank in it, one in the finance context and one in the river context.
# 
# The `get_document_topics` method returns (along with the standard document topic proprtion) the word_type followed by a list sorted with the most likely topic ids, when `per_word_topics` is set as true.
# 

bow_water = ['bank','water','bank']
bow_finance = ['bank','finance','bank']


bow = model.id2word.doc2bow(bow_water) # convert to bag of words format first
doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)

word_topics


# Now what does that output mean? It means that like `word_type 1`, our `word_type` `3`, which is the word `bank`, is more likely to be in `topic_0` than `topic_1`.
# 

# You must have noticed that while we unpacked into `doc_topics` and `word_topics`, there is another variable - `phi_values`. Like the name suggests, phi_values contains the phi values for each topic for that particular word, scaled by feature length. Phi is essentially the probability of that word in that document belonging to a particular topic. The next few lines should illustrate this. 
# 

phi_values


# This means that `word_type` 0 has the following phi_values for each of the topics. 
# What is intresting to note is `word_type` 3 - because it has 2 occurences (i.e, the word `bank` appears twice in the bow), we can see that the scaling by feature length is very evident. The sum of the phi_values is 2, and not 1.
# 

# Now that we know exactly what `get_document_topics` does, let us now do the same with our second document, `bow_finance`.
# 

bow = model.id2word.doc2bow(bow_finance) # convert to bag of words format first
doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)

word_topics


# And lo and behold, because the word bank is now used in the financial context, it immedietly swaps to being more likely associated with `topic_1`.
# 
# We've seen quite clearly that based on the context, the most likely topic associated with a word can change. 
# This differs from our previous method, `get_term_topics`, where it is a 'static' topic distribution. 
# 
# It must also be noted that because the gensim implementation of LDA uses Variational Bayes sampling, a `word_type` in a document is only given one topic distribution. For example, the sentence 'the bank by the river bank' is likely to be assigned to `topic_0`, and each of the bank word instances have the same distribution.
# 

# #### get_document_topics for entire corpus
# 
# You can get `doc_topics`, `word_topics` and `phi_values` for all the documents in the corpus in the following manner :
# 

all_topics = model.get_document_topics(corpus, per_word_topics=True)

for doc_topics, word_topics, phi_values in all_topics:
    print('New Document \n')
    print 'Document topics:', doc_topics
    print 'Word topics:', word_topics
    print 'Phi values:', phi_values
    print(" ")
    print('-------------- \n')


# In case you want to store `doc_topics`, `word_topics` and `phi_values` for all the documents in the corpus in a variable and later access details of a particular document using its index, it can be done in the following manner:
# 

topics = model.get_document_topics(corpus, per_word_topics=True)
all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in topics]


# Now, I can access details of a particular document, say Document #3, as follows: 
# 

doc_topic, word_topics, phi_values = all_topics[2]
print 'Document topic:', doc_topics, "\n"
print 'Word topic:', word_topics, "\n"
print 'Phi value:', phi_values


# We can print details for all the documents (as shown above), in the following manner:
# 

for doc in all_topics:
    print('New Document \n')
    print 'Document topic:', doc[0]
    print 'Word topic:', doc[1]
    print 'Phi value:', doc[2]
    print(" ")
    print('-------------- \n')


# ## Coloring topic-terms
# 

# These methods can come in handy when we want to color the words in a corpus or a document. If we wish to color the words in a corpus (i.e, color all the words in the dictionary of the corpus), then `get_term_topics` would be a better choice. If not, `get_document_topics` would do the trick.
# 

# We'll now attempt to color these words and plot it using `matplotlib`. 
# This is just one way to go about plotting words - there are more and better ways.
# 
# [WordCloud](https://github.com/amueller/word_cloud) is such a python package which also does this.
# 
# For our simple illustration, let's keep `topic_1` as red, and `topic_0` as blue.
# 

# this is a sample method to color words. Like mentioned before, there are many ways to do this.

def color_words(model, doc):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # make into bag of words
    doc = model.id2word.doc2bow(doc)
    # get word_topics
    doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True)

    # color-topic matching
    topic_colors = { 1:'red', 0:'blue'}
    
    # set up fig to plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    # a sort of hack to make sure the words are well spaced out.
    word_pos = 1/len(doc)
    
    # use matplotlib to plot words
    for word, topics in word_topics:
        ax.text(word_pos, 0.8, model.id2word[word],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color=topic_colors[topics[0]],  # choose just the most likely topic
                transform=ax.transAxes)
        word_pos += 0.2 # to move the word for the next iter

    ax.set_axis_off()
    plt.show()


# Let us revisit our old examples to show some examples of document coloring
# 

# our river bank document

bow_water = ['bank','water','bank']
color_words(model, bow_water)


bow_finance = ['bank','finance','bank']
color_words(model, bow_finance)


# What is fun to note here is that while bank was colored blue in our first example, it is now red because of the financial context - something which the numbers proved to us before.
# 

# sample doc with a somewhat even distribution of words among the likely topics

doc = ['bank', 'water', 'bank', 'finance', 'money','sell','river','fast','tree']
color_words(model, doc)


# We see that the document word coloring is done just the way we expected. :)
# 
# ## Word-coloring a dictionary
# 
# We can do the same for the entire vocabulary, statically. The only difference would be in using `get_term_topics`, and iterating over the dictionary.
# 
# We will use a modified version of the coloring code when passing an entire dictionary.
# 

def color_words_dict(model, dictionary):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    word_topics = []
    for word_id in dictionary:
        word = str(dictionary[word_id])
        # get_term_topics returns static topics, as mentioned before
        probs = model.get_term_topics(word)
        # we are creating word_topics which is similar to the one created by get_document_topics
        try:
            if probs[0][1] >= probs[1][1]:
                word_topics.append((word_id, [0, 1]))
            else:
                word_topics.append((word_id, [1, 0]))
        # this in the case only one topic is returned
        except IndexError:
            word_topics.append((word_id, [probs[0][0]]))
            
    # color-topic matching
    topic_colors = { 1:'red', 0:'blue'}
    
    # set up fig to plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    # a sort of hack to make sure the words are well spaced out.
    word_pos = 1/len(doc)
         
    # use matplotlib to plot words
    for word, topics in word_topics:
        ax.text(word_pos, 0.8, model.id2word[word],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color=topic_colors[topics[0]],  # choose just the most likely topic
                transform=ax.transAxes)
        word_pos += 0.2 # to move the word for the next iter

    ax.set_axis_off()
    plt.show()


color_words_dict(model, dictionary)


# As we can see, the red words are to do with finance, and the blue ones are to do with water. 
# 
# You can also notice that some words, like mud, shore and borrow seem to be incorrectly colored - however, they are correctly colored according to the LDA model used for coloring. A small corpus means that the LDA algorithm might not assign 'ideal' topic proportions to each word. Fine tuning the model and having a larger corpus would improve the model, and improve the results of the word coloring.
# 

# ## COBRA Visualisations
# 
# This notebook will cover the visulaisation and plotting of COBRA. 
# 

get_ipython().magic('matplotlib inline')
import numpy as np
from pycobra.cobra import cobra
from pycobra.visualisation import visualisation
from pycobra.diagnostics import diagnostics


# setting up our random data-set
rng = np.random.RandomState(42)

# D1 = train machines; D2 = create COBRA; D3 = calibrate epsilon, alpha; D4 = testing
n_features = 2
D1, D2, D3, D4 = 200, 200, 200, 200
D = D1 + D2 + D3 + D4
X = rng.uniform(-1, 1, D * n_features).reshape(D, n_features)
# Y = np.power(X[:,1], 2) + np.power(X[:,3], 3) + np.exp(X[:,10]) 
Y = np.power(X[:,0], 2) + np.power(X[:,1], 3)

# training data-set
X_train = X[:D1 + D2]
X_test = X[D1 + D2 + D3:D1 + D2 + D3 + D4]
X_eps = X[D1 + D2:D1 + D2 + D3]
# for testing
Y_train = Y[:D1 + D2]
Y_test = Y[D1 + D2 + D3:D1 + D2 + D3 + D4]
Y_eps = Y[D1 + D2:D1 + D2 + D3]

# set up our COBRA machine with the data
COBRA = cobra(X_train, Y_train, epsilon = 0.5)


# ### Plotting COBRA
# 
# We use the visualisation class to plot our results, and for various visualisations.
# 

cobra_vis = visualisation(COBRA, X_test, Y_test)


# to plot our machines, we need a linspace as input. This is the 'scale' to plot and should be the range of the results
# since our data ranges from -1 to 1 it is such - and we space it out to a hundred points
cobra_vis.plot_machines(machines=["COBRA"])


cobra_vis.plot_machines(y_test=Y_test)


# ### Plots and Visualisations of Results
# 
# QQ, Box Plots and Sorted Plots
# 

cobra_vis.QQ(Y_test)


cobra_vis.boxplot()


# ### Plotting colors!
# 
# Going to experiment with plotting colors and data.
# After we get information about which indices are used by which machines the best for a fixed epsilon (or not, we can toggle this option), we can plot the distribution of machines. We first present a plot where the machine colors are mixed depending on which machines were selected; after which we plot one machine at a time.
# 

indices, MSE = cobra_vis.indice_info(X_eps[0:50], Y_eps[0:50], epsilon=0.50)


cobra_vis.color_cobra(X_eps[0:50], indice_info=indices, single=True)


cobra_vis.color_cobra(X_eps[0:50], indice_info=indices)


# ### Voronoi Tesselation
# 
# We present a variety of Voronoi Tesselation based plots - the purpose of this is to help in visualising the pattern of points which tend to be picked up.
# 

cobra_vis.voronoi(X_eps[0:50], indice_info=indices, single=True)


cobra_vis.voronoi(X_eps[0:50], indice_info=indices)


# ### Gradient-Colored Based Voronoi
# 
# 

cobra_vis.voronoi(X_eps[0:50], indice_info=indices, MSE=MSE, gradient=True)


# ### Topic Modelling - and more - with Gensim!
# 
# This tutorial will attempt to walk you through the entire process of analysing your text - from pre-processing to creating your topic models and visualising them. 
# 
# python offers a very rich suite of NLP and CL tools, and we will illustrate these to the best of our capabilities.
# Let's start by setting up our imports.
# 
# We will be needing: 
# ```
# - Gensim
# - matplotlib
# - spaCy
# - pyLDAVis
# ```
# 

import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim

import os, re, operator, warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
get_ipython().magic('matplotlib inline')


# For this tutorial, we will be using the Lee corpus which is a shortened version of the [Lee Background Corpus](http://www.socsci.uci.edu/~mdlee/lee_pincombe_welsh_document.PDF). The shortened version consists of 300 documents selected from the Australian Broadcasting Corporation's news mail service. It consists of texts of headline stories from around the year 2000-2001. 
# 
# We should keep in mind we can use pretty much any textual data-set and go ahead with what we will be doing.
# 

# since we're working in python 2.7 in this tutorial, we need to make sure to clean our data to make it unicode consistent
def clean(text):
    return unicode(''.join([i if ord(i) < 128 else ' ' for i in text]))

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
text = open(lee_train_file).read()


# ### Pre-processing data!
# 
# It's been often said in Machine Learning and NLP algorithms - garbage in, garbage out. We can't have state-of-the-art results without data which is aa good. Let's spend this section working on cleaning and understanding our data set.
# NTLK is usually a popular choice for pre-processing - but is a rather [outdated](https://explosion.ai/blog/dead-code-should-be-buried) and we will be checking out spaCy, an industry grade text-processing package. 
# 

from spacy.en import English
nlp = spacy.load("en")


# For safe measure, let's add some stopwords. It's a newspaper corpus, so it is likely we will be coming across variations of 'said' and 'Mister' which will not really add any value to the topic models.
# 

my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


doc = nlp(clean(text))


# Voila! With the `English` pipeline, all the heavy lifting has been done. Let's see what went on under the hood.
# 

doc


# It seems like nothing, right? But spaCy's internal data structure has done all the work for us. Let's see how we can create our corpus. You can check out what a gensim corpus looks like [here](google.com).
# 

# we add some words to the stop word list
texts, article = [], []
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    if w.text == '\n':
        texts.append(article)
        article = []


# And this is the magic of spaCy - just like that, we've managed to get rid of stopwords, punctauation markers, and added the lemmatized word. There's lot more we can do with spaCy which I would really recommend checking out.
# 
# Sometimes topic models make more sense when 'New' and 'York' are treated as 'New_York' - we can do this by creating a bigram model and modifying our corpus accordingly.
# 

bigram = gensim.models.Phrases(texts)


texts = [bigram[line] for line in texts]


dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# We're now done with a very important part of any text analysis - the data cleaning and setting up of corpus. It must be kept in mind that we created the corpus the way we did because that's how gensim requires it - most algorithms still require one to clean the data set the way we did, by removing stop words and numbers, adding the lemmatized form of the word, and using bigrams. 
# 

# ### LSI
# 
# LSI stands for Latent Semantic Indeixing - it is a popular information retreival method which works by decomposing the original matrix of words to maintain key topics. Gensim's implementation uses an SVD.
# 

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)


lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics


# ### HDP
# 
# HDP, the Hierarchical Dirichlet process is an unsupervised topic model which figures out the number of topics on it's own.
# 

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)


hdpmodel.show_topics()


# ### LDA
# 
# LDA, or Latent Dirichlet Allocation is arguably the most famous topic modelling algorithm out there. Out here we create a simple topic model with 10 topics.
# 

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)


ldamodel.show_topics()


# ### pyLDAvis 
# 
# Thanks to pyLDAvis, we can visualise our topic models in a really handy way. All we need to do is enable our notebook and prepare the object.
# 

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)


# ### Round-up
# 
# Okay - so what have we learned so far? 
# By using spaCy, we cleaned up our data super fast. It's worth noting that by running our doc through the pipeline we also know about every single words POS-tag and NER-tag. This is useful information and we can do some funky things with it! I would highly recommend going through [this](https://github.com/explosion/spacy-notebooks) repository to see examples of hands-on spaCy usage.
# 
# As for gensim and topic modelling, it's pretty easy to see how well we could create our topic models. Now the obvious next question is - how do we use these topic models? The [news classification notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb) in the Gensim [notebooks](https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks) directory is a good example of how we can use topic models in a practical scenario.
# 
# We will continue this tutorial by demonstrating a newer topic modelling features of gensim - in particular, Topic Coherence. 
# 
# ### Topic Coherence
# 
# Topic Coherence is a new gensim functionality where we can identify which topic model is 'better'. 
# By returning a score, we can compare between different topic models of the same. We use the same example from the news classification notebook to plot a graph between the topic models we have created.
# 

lsitopics = [[word for word, prob in topic] for topicid, topic in lsimodel.show_topics(formatted=False)]

hdptopics = [[word for word, prob in topic] for topicid, topic in hdpmodel.show_topics(formatted=False)]

ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]


lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10).get_coherence()


def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')


evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence],
                   ['LSI', 'HDP', 'LDA'])


# We can see that topic coherence helped us get past manually inspecting our topic models - we can now keep fine tuning our models and compare between them to see which has the best performance. 
# 
# This also brings us to the end of the runnable part of this tutorial - we will continue however by briefly going over two more Jupyter notebooks I have previously worked on - mainly, [Dynamic Topic Modelling](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/ldaseqmodel.ipynb) and [Document Word Coloring](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb).
# 

# ### Topic Modelling - and more - with Gensim!
# 
# This tutorial will attempt to walk you through the entire process of analysing your text - from pre-processing to creating your topic models and visualising them. 
# 
# python offers a very rich suite of NLP and CL tools, and we will illustrate these to the best of our capabilities.
# Let's start by setting up our imports.
# 
# We will be needing: 
# ```
# - Gensim
# - matplotlib
# - spaCy
# - pyLDAVis
# ```
# 

import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim

import os, re, operator, warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
get_ipython().magic('matplotlib inline')


# For this tutorial, we will be using the Lee corpus which is a shortened version of the [Lee Background Corpus](http://www.socsci.uci.edu/~mdlee/lee_pincombe_welsh_document.PDF). The shortened version consists of 300 documents selected from the Australian Broadcasting Corporation's news mail service. It consists of texts of headline stories from around the year 2000-2001. 
# 
# We should keep in mind we can use pretty much any textual data-set and go ahead with what we will be doing.
# 

# since we're working in python 2.7 in this tutorial, we need to make sure to clean our data to make it unicode consistent
def clean(text):
    return unicode(''.join([i if ord(i) < 128 else ' ' for i in text]))

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
text = open(lee_train_file).read()


# ### Pre-processing data!
# 
# It's been often said in Machine Learning and NLP algorithms - garbage in, garbage out. We can't have state-of-the-art results without data which is aa good. Let's spend this section working on cleaning and understanding our data set.
# NTLK is usually a popular choice for pre-processing - but is a rather [outdated](https://explosion.ai/blog/dead-code-should-be-buried) and we will be checking out spaCy, an industry grade text-processing package. 
# 

from spacy.en import English
nlp = spacy.load("en")


# For safe measure, let's add some stopwords. It's a newspaper corpus, so it is likely we will be coming across variations of 'said' and 'Mister' which will not really add any value to the topic models.
# 

my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


doc = nlp(clean(text))


# Voila! With the `English` pipeline, all the heavy lifting has been done. Let's see what went on under the hood.
# 

doc


# It seems like nothing, right? But spaCy's internal data structure has done all the work for us. Let's see how we can create our corpus. You can check out what a gensim corpus looks like [here](google.com).
# 

# we add some words to the stop word list
texts, article = [], []
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    if w.text == '\n':
        texts.append(article)
        article = []


# And this is the magic of spaCy - just like that, we've managed to get rid of stopwords, punctauation markers, and added the lemmatized word. There's lot more we can do with spaCy which I would really recommend checking out.
# 
# Sometimes topic models make more sense when 'New' and 'York' are treated as 'New_York' - we can do this by creating a bigram model and modifying our corpus accordingly.
# 

bigram = gensim.models.Phrases(texts)


texts = [bigram[line] for line in texts]


dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# We're now done with a very important part of any text analysis - the data cleaning and setting up of corpus. It must be kept in mind that we created the corpus the way we did because that's how gensim requires it - most algorithms still require one to clean the data set the way we did, by removing stop words and numbers, adding the lemmatized form of the word, and using bigrams. 
# 

# ### LSI
# 
# LSI stands for Latent Semantic Indeixing - it is a popular information retreival method which works by decomposing the original matrix of words to maintain key topics. Gensim's implementation uses an SVD.
# 

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)


lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics


# ### HDP
# 
# HDP, the Hierarchical Dirichlet process is an unsupervised topic model which figures out the number of topics on it's own.
# 

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)


hdpmodel.show_topics()


# ### LDA
# 
# LDA, or Latent Dirichlet Allocation is arguably the most famous topic modelling algorithm out there. Out here we create a simple topic model with 10 topics.
# 

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)


ldamodel.show_topics()


# ### pyLDAvis 
# 
# Thanks to pyLDAvis, we can visualise our topic models in a really handy way. All we need to do is enable our notebook and prepare the object.
# 

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)


# ### Round-up
# 
# Okay - so what have we learned so far? 
# By using spaCy, we cleaned up our data super fast. It's worth noting that by running our doc through the pipeline we also know about every single words POS-tag and NER-tag. This is useful information and we can do some funky things with it! I would highly recommend going through [this](https://github.com/explosion/spacy-notebooks) repository to see examples of hands-on spaCy usage.
# 
# As for gensim and topic modelling, it's pretty easy to see how well we could create our topic models. Now the obvious next question is - how do we use these topic models? The [news classification notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb) in the Gensim [notebooks](https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks) directory is a good example of how we can use topic models in a practical scenario.
# 
# We will continue this tutorial by demonstrating a newer topic modelling features of gensim - in particular, Topic Coherence. 
# 
# ### Topic Coherence
# 
# Topic Coherence is a new gensim functionality where we can identify which topic model is 'better'. 
# By returning a score, we can compare between different topic models of the same. We use the same example from the news classification notebook to plot a graph between the topic models we have created.
# 

lsitopics = [[word for word, prob in topic] for topicid, topic in lsimodel.show_topics(formatted=False)]

hdptopics = [[word for word, prob in topic] for topicid, topic in hdpmodel.show_topics(formatted=False)]

ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]


lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10).get_coherence()


def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')


evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence],
                   ['LSI', 'HDP', 'LDA'])


# We can see that topic coherence helped us get past manually inspecting our topic models - we can now keep fine tuning our models and compare between them to see which has the best performance. 
# 
# This also brings us to the end of the runnable part of this tutorial - we will continue however by briefly going over two more Jupyter notebooks I have previously worked on - mainly, [Dynamic Topic Modelling](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/ldaseqmodel.ipynb) and [Document Word Coloring](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb).
# 

