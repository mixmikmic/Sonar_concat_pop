# __Note__: This is best viewed on [NBViewer](http://nbviewer.ipython.org/github/tdhopper/notes-on-dirichlet-processes/blob/master/2015-10-07-econtalk-topics.ipynb). It is part of a series on [Dirichlet Processes and Nonparametric Bayes](https://github.com/tdhopper/notes-on-dirichlet-processes).
# 

# # Nonparametric Latent Dirichlet Allocation
# 
# ## Analysis of the topics of [Econtalk](http://www.econtalk.org/)
# 
# In 2003, a groundbreaking statistical model called "[Latent Dirichlet Allocation](https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf)" was presented by David Blei, Andrew Ng, and Michael Jordan.
# 
# LDA provides a method for summarizing the topics discussed in a document. LDA defines topics to be discrete probability distrbutions over words. For an introduction to LDA, see [Edwin Chen's post](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/).
# 
# The original LDA model requires the number of topics in the document to be specfied as a known parameter of the model. In 2005, Yee Whye Teh and others published [a "nonparametric" version of this model](http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf) that doesn't require the number of topics to be specified. This model uses a prior distribution over the topics called a hierarchical Dirichlet process. [I wrote an introduction to this HDP-LDA model](https://github.com/tdhopper/notes-on-dirichlet-processes/blob/master/2015-08-03-nonparametric-latent-dirichlet-allocation.ipynb) earlier this year.
# 
# For the last six months, I have been developing a Python-based Gibbs sampler for the HDP-LDA model. This is part of a larger library of "robust, validated Bayesian nonparametric models for discovering structure in data" known as [Data Microscopes](http://datamicroscopes.github.io).
# 
# This notebook demonstrates the functionality of this implementation.
# 
# The Data Microscopes library is available on [anaconda.org](https://anaconda.org/datamicroscopes/) for Linux and OS X. `microscopes-lda` can be installed with:
# 
#     $ conda install -c datamicroscopes -c distributions microscopes-lda 
# 

get_ipython().magic('matplotlib inline')
import pyLDAvis
import json
import sys
import cPickle

from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import initialize
from microscopes.lda import utils
from microscopes.lda import model, runner

from numpy import genfromtxt 
from numpy import linalg
from numpy import array


# `dtm.csv` contains a document-term matrix representation of the words used in Econtalk transcripts. The columns of the matrix correspond to the words in `vocab.txt`. The rows in the matrix correspond to the show urls in `urls.txt`.
# 
# Our LDA implementation takes input data as a list of lists of hashable objects (typically words). We can use a utility function to convert the document-term matrix to the list of tokenized documents. 
# 

vocab = genfromtxt('./econtalk-data/vocab.txt', delimiter=",", dtype='str').tolist()
dtm = genfromtxt('./econtalk-data/dtm.csv', delimiter=",", dtype=int)
docs = utils.docs_from_document_term_matrix(dtm, vocab=vocab)
urls = [s.strip() for s in open('./econtalk-data/urls.txt').readlines()]


dtm.shape[1] == len(vocab)


dtm.shape[0] == len(urls)


# Here's a utility method to get the title of a webpage that we'll use later.
# 

def get_title(url):
    """Scrape webpage title
    """
    import lxml.html
    t = lxml.html.parse(url)
    return t.find(".//title").text.split("|")[0].strip()


# Let's set up our model. First we created a model definition describing the basic structure of our data. Next we initialize an MCMC state object using the model definition, documents, random number generator, and hyper-parameters.
# 

N, V = len(docs), len(vocab)
defn = model_definition(N, V)
prng = rng(12345)
state = initialize(defn, docs, prng,
                        vocab_hp=1,
                        dish_hps={"alpha": .6, "gamma": 2})
r = runner.runner(defn, docs, state, )


# When we first create a state object, the words are randomly assigned to topics. Thus, our perplexity (model score) is quite high. After we start to run the MCMC, the score will drop quickly.
# 

print "randomly initialized model:"
print " number of documents", defn.n
print " vocabulary size", defn.v
print " perplexity:", state.perplexity(), "num topics:", state.ntopics()


# Run one iteration of the MCMC to make sure everything is working.
# 

get_ipython().run_cell_magic('time', '', 'r.run(prng, 1)')


# Now lets run 1000 generations of the MCMC.
# 
# Unfortunately, MCMC is slow going.
# 

get_ipython().run_cell_magic('time', '', 'r.run(prng, 500)')


with open('./econtalk-data/2015-10-07-state.pkl', 'w') as f:
    cPickle.dump(state, f)


get_ipython().run_cell_magic('time', '', 'r.run(prng, 500)')


with open('./econtalk-data/2015-10-07-state.pkl', 'w') as f:
    cPickle.dump(state, f)


# Now that we've run the MCMC, the perplexity has dropped significantly. 
# 

print "after 1000 iterations:"
print " perplexity:", state.perplexity(), "num topics:", state.ntopics()


# [pyLDAvis](https://github.com/bmabey/pyLDAvis) projects the topics into two dimensions using techniques described by [Carson Sievert](http://stat-graphics.org/movies/ldavis.html).
# 

vis = pyLDAvis.prepare(**state.pyldavis_data())
pyLDAvis.display(vis)


# We can extract the term relevance (shown in the right hand side of the visualization) right from our state object. Here are the 10 most relevant words for each topic:
# 

relevance = state.term_relevance_by_topic()
for i, topic in enumerate(relevance):
    print "topic", i, ":",
    for term, _ in topic[:10]:
        print term, 
    print 


# We could assign titles to each of these topics. For example, _Topic 5_ appears to be about the _foundations of classical liberalism_. _Topic 6_ is obviously _Bitcoin and Software_.  _Topic 0_ is the _financial system and monetary policy_. _Topic 4_ seems to be _generic words used in most episodes_; unfortunately, the prevalence of "don" is a result of my preprocessing which splits up the contraction "don't".
# 

# We can also get the topic distributions for each document. 
# 

topic_distributions = state.topic_distribution_by_document()


# Topic 5 appears to be about the theory of classical liberalism. Let's find the 20 episodes which have the highest proportion of words from that topic.
# 

austrian_topic = 5
foundations_episodes = sorted([(dist[austrian_topic], url) for url, dist in zip(urls, topic_distributions)], reverse=True)
for url in [url for _, url in foundations_episodes][:20]:
    print get_title(url), url


# We could also find the episodes that have notable discussion of both politics AND the financial system.
# 

topic_a = 0
topic_b = 1
joint_episodes = [url for url, dist in zip(urls, topic_distributions) if dist[0] > 0.18 and dist[1] > 0.18]
for url in joint_episodes:
    print get_title(url), url


# We can look at the topic distributions as projections of the documents into a much lower dimension (16). 
# We can try to find shows that are similar by comparing the topic distributions of the documents. 
# 

def find_similar(url, max_distance=0.2):
    """Find episodes most similar to input url.
    """
    index = urls.index(url)
    for td, url in zip(topic_distributions, urls):
        if linalg.norm(array(topic_distributions[index]) - array(td)) < max_distance:
            print get_title(url), url


# Which Econtalk episodes are most similar, in content, to "Mike Munger on the Division of Labor"?
# 

find_similar('http://www.econtalk.org/archives/2007/04/mike_munger_on.html')


# How about episodes similar to "Kling on Freddie and Fannie and the Recent History of the U.S. Housing Market"?
# 

find_similar('http://www.econtalk.org/archives/2008/09/kling_on_freddi.html')


# The model also gives us distributions over words for each topic.
# 

word_dists = state.word_distribution_by_topic()


# We can use this to find the topics a word is most likely to occur in.
# 

def bars(x, scale_factor=10000):
    return int(x * scale_factor) * "="

def topics_related_to_word(word, n=10):
    for wd, rel in zip(word_dists, relevance):
        score = wd[word]
        rel_words = ' '.join([w for w, _ in rel][:n]) 
        if bars(score):
            print bars(score), rel_words


# What topics are most likely to contain the word "Munger" (as in [Mike Munger](http://www.michaelmunger.com/)). The number of equal signs indicates the probability the word is generated by the topic. If a topic isn't shown, it's extremely unlikley to generate the word.
# 

topics_related_to_word('munger')


# Where does Munger come up? In discussing the moral foundations of classical liberalism and microeconomics!
# 

# 
# 
# How about the word "lovely"? Russ Roberts uses it often when talking about the _Theory of Moral Sentiments_. It looks like it also comes up when talking about schools.
# 

topics_related_to_word('lovely')


# If you have feedback on this implementation of HDP-LDA, you can reach me on [Twitter](http://twitter.com/tdhopper) or open an [issue on Github](https://github.com/datamicroscopes/lda/issues).
# 

# __Note__: This is best viewed on [NBViewer](http://nbviewer.ipython.org/github/tdhopper/stigler-diet/blob/master/content/articles/2015-08-03-nonparametric-latent-dirichlet-allocation.ipynb). It is part of a series on [Dirichlet Processes and Nonparametric Bayes](https://github.com/tdhopper/notes-on-dirichlet-processes).
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('precision 2')


# # Nonparametric Latent Dirichlet Allocation
# 
# _Latent Dirichlet Allocation_ is a [generative](https://en.wikipedia.org/wiki/Generative_model) model for topic modeling. Given a collection of documents, an LDA inference algorithm attempts to determined (in an unsupervised manner) the topics discussed in the documents. It makes the assumption that each document is generated by a probability model, and, when doing inference, we try to find the parameters that best fit the model (as well as unseen/latent variables generated by the model). If you are unfamiliar with LDA, Edwin Chen has a [friendly introduction](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/) you should read.
# 
# 
# Because LDA is a _generative_ model, we can simulate the construction of documents by forward-sampling from the model. The generative algorithm is as follows (following [Heinrich](http://www.arbylon.net/publications/text-est.pdf)):
# 
# * for each topic $k\in [1,K]$ do
#     * sample term distribution for topic $\overrightarrow \phi_k \sim \text{Dir}(\overrightarrow \beta)$
# * for each document $m\in [1, M]$ do
#     * sample topic distribution for document $\overrightarrow\theta_m\sim \text{Dir}(\overrightarrow\alpha)$
#     * sample document length $N_m\sim\text{Pois}(\xi)$
#     * for all words $n\in [1, N_m]$ in document $m$ do
#         * sample topic index $z_{m,n}\sim\text{Mult}(\overrightarrow\theta_m)$
#         * sample term for word $w_{m,n}\sim\text{Mult}(\overrightarrow\phi_{z_{m,n}})$
#         
# You can implement this with [a little bit of code](https://gist.github.com/tdhopper/521006b60e1311d45509) and start to simulate documents.
# 
# In LDA, we assume each word in the document is generated by a two-step process:
# 
# 1. Sample a topic from the topic distribution for the document.
# 2. Sample a word from the term distribution from the topic. 
# 
# When we fit the LDA model to a given text corpus with an inference algorithm, our primary objective is to find the set of topic distributions $\underline \Theta$, term distributions $\underline \Phi$ that generated the documents, and latent topic indices $z_{m,n}$ for each word.
# 
# To run the generative model, we need to specify each of these parameters:
# 

vocabulary = ['see', 'spot', 'run']
num_terms = len(vocabulary)
num_topics = 2 # K
num_documents = 5 # M
mean_document_length = 5 # xi
term_dirichlet_parameter = 1 # beta
topic_dirichlet_parameter = 1 # alpha


# The term distribution vector $\underline\Phi$ is a collection of samples from a Dirichlet distribution. This describes how our 3 terms are distributed across each of the two topics.
# 

from scipy.stats import dirichlet, poisson
from numpy import round
from collections import defaultdict
from random import choice as stl_choice


term_dirichlet_vector = num_terms * [term_dirichlet_parameter]
term_distributions = dirichlet(term_dirichlet_vector, 2).rvs(size=num_topics)
print term_distributions


# Each document corresponds to a categorical distribution across this distribution of topics (in this case, a 2-dimensional categorical distribution). This categorical distribution is a _distribution of distributions_; we could look at it as a Dirichlet process!
# 
# The base base distribution of our Dirichlet process is a uniform distribution of topics (remember, topics are term distributions). 
# 

base_distribution = lambda: stl_choice(term_distributions)
# A sample from base_distribution is a distribution over terms
# Each of our two topics has equal probability
from collections import Counter
for topic, count in Counter([tuple(base_distribution()) for _ in range(10000)]).most_common():
    print "count:", count, "topic:", [round(prob, 2) for prob in topic]


# Recall that a sample from a Dirichlet process is a distribution that approximates (but varies from) the base distribution. In this case, a sample from the Dirichlet process will be a distribution over topics that varies from the uniform distribution we provided as a base. If we use the stick-breaking metaphor, we are effectively breaking a stick one time and the size of each portion corresponds to the proportion of a topic in the document.
# 
# To construct a sample from the DP, we need to [again define our DP class](http://stiglerdiet.com/blog/2015/Jul/28/dirichlet-distribution-and-dirichlet-process/):
# 

from scipy.stats import beta
from numpy.random import choice

class DirichletProcessSample():
    def __init__(self, base_measure, alpha):
        self.base_measure = base_measure
        self.alpha = alpha
        
        self.cache = []
        self.weights = []
        self.total_stick_used = 0.

    def __call__(self):
        remaining = 1.0 - self.total_stick_used
        i = DirichletProcessSample.roll_die(self.weights + [remaining])
        if i is not None and i < len(self.weights) :
            return self.cache[i]
        else:
            stick_piece = beta(1, self.alpha).rvs() * remaining
            self.total_stick_used += stick_piece
            self.weights.append(stick_piece)
            new_value = self.base_measure()
            self.cache.append(new_value)
            return new_value
      
    @staticmethod 
    def roll_die(weights):
        if weights:
            return choice(range(len(weights)), p=weights)
        else:
            return None


# For each document, we will draw a topic distribution from the Dirichlet process:
# 

topic_distribution = DirichletProcessSample(base_measure=base_distribution, 
                                            alpha=topic_dirichlet_parameter)


# A sample from this _topic_ distribution is a _distribution over terms_. However, unlike our base distribution which returns each term distribution with equal probability, the topics will be unevenly weighted.
# 

for topic, count in Counter([tuple(topic_distribution()) for _ in range(10000)]).most_common():
    print "count:", count, "topic:", [round(prob, 2) for prob in topic]


# To generate each word in the document, we draw a sample topic from the topic distribution, and then a term from the term distribution (topic). 
# 

topic_index = defaultdict(list)
documents = defaultdict(list)

for doc in range(num_documents):
    topic_distribution_rvs = DirichletProcessSample(base_measure=base_distribution, 
                                                    alpha=topic_dirichlet_parameter)
    document_length = poisson(mean_document_length).rvs()
    for word in range(document_length):
        topic_distribution = topic_distribution_rvs()
        topic_index[doc].append(tuple(topic_distribution))
        documents[doc].append(choice(vocabulary, p=topic_distribution))


# Here are the documents we generated:
# 

for doc in documents.values():
    print doc


# We can see how each topic (term-distribution) is distributed across the documents:
# 

for i, doc in enumerate(Counter(term_dist).most_common() for term_dist in topic_index.values()):
    print "Doc:", i
    for topic, count in doc:
        print  5*" ", "count:", count, "topic:", [round(prob, 2) for prob in topic]


# To recap: for each document we draw a _sample_ from a Dirichlet _Process_. The base distribution for the Dirichlet process is a categorical distribution over term distributions; we can think of the base distribution as an $n$-sided die where $n$ is the number of topics and each side of the die is a distribution over terms for that topic. By sampling from the Dirichlet process, we are effectively reweighting the sides of the die (changing the distribution of the topics).
# 
# For each word in the document, we draw a _sample_ (a term distribution) from the distribution (over term distributions) _sampled_ from the Dirichlet process (with a distribution over term distributions as its base measure). Each term distribution uniquely identifies the topic for the word. We can sample from this term distribution to get the word.
# 
# Given this formulation, we might ask if we can roll an _infinite_ sided die to draw from an unbounded number of topics (term distributions). We can do exactly this with a _Hierarchical_ Dirichlet process. Instead of the base distribution of our Dirichlet process being a _finite_ distribution over topics (term distributions) we will instead make it an infinite Distribution over topics (term distributions) by using yet another Dirichlet process! This base Dirichlet process will have as its base distribution a Dirichlet _distribution_ over terms. 
# 
# We will again draw a _sample_ from a Dirichlet _Process_ for each document. The base distribution for the Dirichlet process is itself a Dirichlet process whose base distribution is a Dirichlet distribution over terms. (Try saying that 5-times fast.) We can think of this as a countably infinite die each side of the die is a distribution over terms for that topic. The sample we draw is a topic (distribution over terms).
# 
# For each word in the document, we will draw a _sample_ (a term distribution) from the distribution (over term distributions) _sampled_ from the Dirichlet process (with a distribution over term distributions as its base measure). Each term distribution uniquely identifies the topic for the word. We can sample from this term distribution to get the word.
# 
# These last few paragraphs are confusing! Let's illustrate with code.
# 

term_dirichlet_vector = num_terms * [term_dirichlet_parameter]
base_distribution = lambda: dirichlet(term_dirichlet_vector).rvs(size=1)[0]

base_dp_parameter = 10
base_dp = DirichletProcessSample(base_distribution, alpha=base_dp_parameter)


# This sample from the base Dirichlet process is our infinite sided die. It is a probability distribution over a countable infinite number of topics. 
# 
# The fact that our die is countably infinite is important. The sampler `base_distribution` draws topics (term-distributions) from an uncountable set. If we used this as the base distribution of the Dirichlet process below each document would be constructed from a _completely unique set of topics_. By feeding `base_distribution` into a Dirichlet Process (stochastic memoizer), we allow the topics to be shared across documents. 
# 
# In other words, `base_distribution` will never return the same topic twice; however, every topic sampled from `base_dp` would be sampled an infinite number of times (if we sampled from `base_dp` forever). At the same time, `base_dp` will also return an _infinite number_ of topics. In our formulation of the the LDA sampler above, our base distribution only ever returned a finite number of topics (`num_topics`); there is no `num_topics` parameter here.
# 
# Given this setup, we can generate documents from the _hierarchical Dirichlet process_ with an algorithm that is essentially identical to that of the original _latent Dirichlet allocation_ generative sampler:
# 

nested_dp_parameter = 10

topic_index = defaultdict(list)
documents = defaultdict(list)

for doc in range(num_documents):
    topic_distribution_rvs = DirichletProcessSample(base_measure=base_dp, 
                                                    alpha=nested_dp_parameter)
    document_length = poisson(mean_document_length).rvs()
    for word in range(document_length):
        topic_distribution = topic_distribution_rvs()
        topic_index[doc].append(tuple(topic_distribution))
        documents[doc].append(choice(vocabulary, p=topic_distribution))


# Here are the documents we generated:
# 

for doc in documents.values():
    print doc


# And here are the latent topics used:
# 

for i, doc in enumerate(Counter(term_dist).most_common() for term_dist in topic_index.values()):
    print "Doc:", i
    for topic, count in doc:
        print  5*" ", "count:", count, "topic:", [round(prob, 2) for prob in topic]


# Our documents were generated by an unspecified number of topics, and yet the topics were shared across the 5 documents. This is the power of the hierarchical Dirichlet process!
# 
# This non-parametric formulation of Latent Dirichlet Allocation was first published by [Yee Whye Teh et al](http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf). 
# 
# Unfortunately, forward sampling is the easy part. Fitting the model on data requires [complex MCMC](http://psiexp.ss.uci.edu/research/papers/sciencetopics.pdf) or [variational inference](http://www.cs.princeton.edu/~chongw/papers/WangPaisleyBlei2011.pdf). There are a [limited](http://www.stats.ox.ac.uk/~teh/software.html) [number](https://github.com/shuyo/iir/blob/master/lda/hdplda2.py) of [implementations](https://github.com/renaud/hdp-faster) [of HDP-LDA](http://www.arbylon.net/resources.html) available, and none of them are great. 
# 

