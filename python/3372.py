# Late night 1 hour hack of the freshly released dataset on train time tables by IRCTC.
# Source: https://data.gov.in/catalog/indian-railways-train-time-table-0#web_catalog_tabs_block_10
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Load the data into a dataframe
df = pd.read_csv("data/isl_wise_train_detail_03082015_v1.csv")


sns.set_context("poster")
# Show some rows
df.head()


df.columns


# Convert time columns to datetime objects
df[u'Arrival time'] = pd.to_datetime(df[u'Arrival time'])
df[u'Departure time'] = pd.to_datetime(df[u'Departure time'])


df.head()


# ##Distribution of Arrival and Departure Times
# Lets analyze the arrival and departure time distributions. As we can see from the plots below, both the times follow as similar distribution. What is interesting is that a majority of the trains arrive during the night (which is good as Indians love to travel during night).
# 

fig, ax = plt.subplots(1,2, sharey=True)
df[u'Arrival time'].map(lambda x: x.hour).hist(ax=ax[0], bins=24)
df[u'Departure time'].map(lambda x: x.hour).hist(ax=ax[1], bins=24)
ax[0].set_xlabel("Arrival Time")
ax[1].set_xlabel("Departure Time")


# It would also be interesting to find out the distribution of the stoppage time at a station. 
# $Stoppage\_time = Departure\_time - Arrival\_time$
# 

df["Stoppage"] = (df[u'Departure time'] - df[u'Arrival time']).astype('timedelta64[m]') # Find stoppage time in minutes
# Plot distribution of stoppage time
df["Stoppage"].hist()
plt.xlabel("Stoppage Time")


# This looks wierd. Stoppage time cannot be negative or more than 500 minutes (~8 hours). Let us remove these outlires and plot our distributions again. 
# 

df["Stoppage"][(df["Stoppage"]> 0) & (df["Stoppage"] < 61)].hist() # Let us take that max stoppage time can be an hour. 
plt.xlabel("Stoppage Time")


# This is better but still appears that most stoppage times are less than 30 minutes. So let us plot again in that range. 
# 

df["Stoppage"][(df["Stoppage"]> 0) & (df["Stoppage"] < 31)].hist(bins=30) # Let us take that max stoppage time can be an hour. 
plt.xlabel("Stoppage Time")


# This is more informative. We see that most stoppage times are either 1 or 2 minutes or a multiple of 5 minutes. Makes a lot of sense. Now let us look filter the data to make it consist of the stoppage time in this range. 
# 

df_stoppage_30 = df[(df["Stoppage"]> 0) & (df["Stoppage"] < 31)] # Filter data between nice stoppage times
# Plot data for this stoppage time range.
fig, ax = plt.subplots(1,2, sharey=True)
df_stoppage_30[u'Arrival time'].map(lambda x: x.hour).hist(ax=ax[0], bins=24)
df_stoppage_30[u'Departure time'].map(lambda x: x.hour).hist(ax=ax[1], bins=24)
ax[0].set_xlabel("Arrival Time")
ax[1].set_xlabel("Departure Time")


# Aah, it looks like less trains arrive and depart during lunch hours around 1200-1500 Hours. Looks wierd but can also point to the fact that many trains run at night and travel short distances. This makes me think that we should look closely at the total distance per train. 
# 
# ## Distance analysis
# 
# Lets now analyze the total distance travelled by a train. This can be easily found by using the last value for each train. 
# 

# Total Number of stations of the train, last arrival time, first departure time, last distance, first station and last station.

df_train_dist = df[[u'Train No.', u'station Code', u'Arrival time', u'Departure time',
                    u'Distance', u'Source Station Code', u'Destination station Code']]\
.groupby(u'Train No.').agg({u'station Code': "count", u'Arrival time': "last",
                                                               u'Departure time': "first", u'Distance': "last",
                                                               u'Source Station Code': "first", u'Destination station Code': "last"})


df_train_dist.head()


# Let us plot the distribution of the distances as well as station codes, as well as arrival and departure times
fig, ax = plt.subplots(2,2)
df_train_dist[u'station Code'].hist(ax=ax[0][0], bins=range(df_train_dist[u'station Code'].max() + 1))
df_train_dist[u'Distance'].hist(ax=ax[0][1], bins=50)
ax[1][0].set_xlabel("Total Stations stopped")
ax[1][1].set_xlabel("Total Distance covered")

df_train_dist[u'Arrival time'].map(lambda x: x.hour).hist(ax=ax[1][0], bins=range(24))
df_train_dist[u'Departure time'].map(lambda x: x.hour).hist(ax=ax[1][1], bins=range(24))
ax[1][0].set_xlabel("Arrival Time")
ax[1][1].set_xlabel("Departure Time")


# ## Train specific analysis
# Ok this is insteresting. 
# 
# * We observe that majority of the trains cover 15-25 stations. 
# * We also see that many trains are short distance trains travelling only 500-700 Kilometers. 
# * Arrival time for many trains at their last stop is mostly during morning 0500 to afternoon 1300 hours and also a lot around midnight. 
# * Departure time for a majority of the trains is actually mostly during night. 
# 
# Now the question is: Do trains on average having more stops run longer distance or not ? Let us try to answer this question.  
# 
# 

sns.lmplot(x=u'station Code', y=u'Distance', data=df_train_dist, x_estimator=np.mean)


# The regression plot shows that we cannot draw any conclusion regarding the relation between number of stopns and distance. We do see that low stops mean small distances but for larger distances we observe that this condition doesn't hold true. This can be attributed to the availability of both express as well as passenger trains for longer distances.
# 

# Lets us see what are some general statistics of the distances and the number of stops. 
df_train_dist.describe()


# We observe that 50% of the trains travel less than 810 Km as well as have less than 20 stops. Maximum distance travelled by a train is 4273 Km and maximum stoppages are 128, both of which are very high numbers. 
# 

# ## Analysis of Stations
# 
# Let us look at which stations are popular. 
# 

df[[u'Train No.', u'Station Name']].groupby(u'Station Name').count().sort(u'Train No.', ascending=False).head(20)


# Looks like Vijaywada is the station where maximum trains have a stoppage. I am upset not to see my place Allahabad in the top 20 list. Neverthless, let us plot the distribution of these stoppages. 
# 

df[[u'Train No.', u'Station Name']].groupby(u'Station Name').count().hist(bins=range(1,320,2), log=True)
plt.xlabel("Number of trains stopping")
plt.ylabel("Number of stations")


# Looks like very few stations have a high volume of trains stopping. Most stations see close to 5 trains. 
# Let us now look at some train statistics like:
#     
# * Trains with maximum stops, I would personally avoid these trains. 
# * Trains which travel maximum distance, if they take less stops I would prefer these.
# 

df_train_dist.sort(u'station Code', ascending=False).head(10) # Top 10 trains with maximum number of stops


df_train_dist.sort(u'Distance', ascending=False).head(10) # Top 10 trains with maximum distance


fig, ax = plt.subplots(1,2)
sns.regplot(x=df_train_dist[u'Arrival time'].map(lambda x: x.hour), y=df_train_dist[u'Distance'], x_estimator=np.mean, ax=ax[0])
sns.regplot(x=df_train_dist[u'Departure time'].map(lambda x: x.hour), y=df_train_dist[u'Distance'], x_estimator=np.mean, ax=ax[1])


# We see that departure and arrival time of a lot of long distance trains is during night around 0000 Hours, many long route trains arrive during late afternoons around 1500 hours and many long route trains leave early morning around 1000 Hours as well. Most medium distance trains arrive during the day
# 




# # Measuring importance of coefficients of OLS
# 
# $s^2 = \frac{y^TMy}{n -p} = \frac{y^T(y - X\hat{\beta})}{n -p}$
# 
# $s.e.(\hat{\beta_{j}}) = \sqrt{s^2(X^TX)^{-1}_{jj}}$
# 
# $t = \frac{\hat{\beta}}{s.e.(\hat{\beta})}$
# 
# $p = SF(|t|, n-p) * 2$
# 
# $c.i. = PPF((1 + confidence)/2, n-p)$
# 

import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy import stats
pd.options.display.float_format = '{:,.4f}'.format

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


N = 20000
X = sm.add_constant(np.random.randn(N,3)*10)
w = np.array([0.25, 0.5, 0.3, -0.5])
y = np.dot(X,w) + np.random.randn(X.shape[0])
print X.shape, y.shape, w.shape
plt.plot(X[:, 1], y, "o", X[:, 2], y, "s", X[:, 3], y, "d")


model = sm.OLS(y, X)
res = model.fit()
print res.summary2()


def R2(y, X, coeffs):
    y_hat = np.dot(X, coeffs)
    y_mean = np.mean(y)
    SST = np.sum((y-y_mean)**2)
    SSR = np.sum((y_hat - y_mean)**2)
    SSE = np.sum((y_hat - y)**2)
    #R_squared = SSR / SST
    R_squared = SSE / SST
    return 1- R_squared


R2(y, X, res.params)


def se_coeff(y, X, coeffs):
    # Reference: https://en.wikipedia.org/wiki/Ordinary_least_squares#Finite_sample_properties
    s2 = np.dot(y, y - np.dot(X, coeffs)) / (X.shape[0] - X.shape[1]) # Calculate S-squared
    XX = np.diag(np.linalg.inv(np.dot(X.T, X))) # Calculate 
    return np.sqrt(s2*XX)


coeffs = res.params
N, K = X.shape
se = se_coeff(y, X, coeffs)
t = coeffs / se
p = stats.t.sf(np.abs(t), N - K)*2
ci = stats.t.ppf((1 + 0.95)/2, N-K)*se
pd.DataFrame(np.vstack((coeffs, se, t, p, coeffs - ci, coeffs + ci)).T, columns=["coeff", "S.E.", "t", "p-value", "ci-", "ci+"])


# # Coefficient Significant for Logistic Regression
# 

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


plt.clf()
y_sc = sigmoid(np.dot(X,w))
idx = np.argsort(y_sc)
plt.plot(y_sc[idx], "-b", label="logit")
y = stats.bernoulli.rvs(y_sc, size=y_sc.shape[0])
plt.plot(y[idx], "or", label="label")
plt.legend()


model = sm.Logit(y, X)
res = model.fit()
print res.summary2()
print w


plt.hist(y)
plt.hist(y_sc)


plt.plot(X[idx, 1], y_sc[idx], "o", X[idx, 2], y_sc[idx], "s", X[idx, 3], y_sc[idx], "d")


def se_coeff_logit(y, X, coeffs):
    # Reference: https://en.wikipedia.org/wiki/Ordinary_least_squares#Finite_sample_properties
    s2 = np.dot(y, y - sigmoid(np.dot(X, coeffs))) / (X.shape[0] - X.shape[1]) # Calculate S-squared
    XX = np.diag(np.linalg.inv(np.dot(X.T, X))) # Calculate 
    return np.sqrt(s2*XX)


coeffs = res.params
N, K = X.shape
se = se_coeff_logit(y, X, coeffs)
t = coeffs / se
p = stats.t.sf(np.abs(t), N - K)*2
ci = stats.t.ppf((1 + 0.95)/2, N-K)*se
pd.DataFrame(np.vstack((coeffs, se, t, p, coeffs - ci, coeffs + ci)).T, columns=["coeff", "S.E.", "t", "p-value", "ci-", "ci+"])





get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import cross_validation
from sklearn.decomposition import PCA


iris = load_iris()


# # Data Preperation
# 
# We will work with the Iris data set which is a standard toy dataset for machine learning model evaluation. 
# 
# ## One Hot Encoding
# Iris data set consists of 3 class of Iris flowers. The data comes in the format where each element of `y` corresponds to one integer which uniquely identifies the flower class. Since, the numerical values of y are of no use to us as y is a categorical variable hence it will be more approapriate to convert it into one hot vectors as followes:
# 
# | Class | One Hot Vector |
# | ------------- |:-------------:|
# | 0 | [1, 0, 0] |
# | 1 | [0, 1, 0] |
# | 2 | [0, 0, 1] |
# 
# ## Feature Scaling
# 
# Feature scaling is a very important step when training neural network models. Feature scaling ensures that no feature dominates the prediction process because of the high range of values which it acquires for every instance. A common way of using feature scaling is by processing the features so that they have mean = 0.0 and standard deviation = 1.0. Without this step the neural network in most cases will not converge. 
# 
# **PS: I tried the neural network without feature scaling and was getting very bad accuracy sometimes as close to 0.**
# 

X_org, y = iris.data, iris.target
print "Classes present in IRIS", iris.target_names

# Convert y to one hot vector for each category
enc = OneHotEncoder()
y= enc.fit_transform(y[:, np.newaxis]).toarray()

# **VERY IMPORTANT STEP** Scale the values so that mean is 0 and variance is 1.
# If this step is not performed the Neural network will not converge. The logistic regression model might converge.
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


X.shape, y.shape


# Implement Model in Keras
model = Sequential()
model.add(Dense(X.shape[1], 2, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, 2, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, y.shape[1], init='uniform', activation='softmax'))

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD()


# Compile the model using theano
model.compile(loss='categorical_crossentropy', optimizer="rmsprop")


"""
Use this cell only if you want to reset the weights of the model. 
"""
"""
#print model.get_weights()
model.set_weights(np.array([np.random.uniform(size=k.shape) for k in model.get_weights()]))
print model.to_yaml()
model.optimizer.lr = 0.01
model.optimizer.decay = 0.
model.optimizer.momentum = 0.
model.optimizer.nesterov = False
"""
# Done


# Perform cross validated training
cond = (y[:,0] == 1) | (y[:,1] == 1) | (y[:,2] == 1)
kf = cross_validation.KFold(X[cond].shape[0], n_folds=10, shuffle=True)
scores = []
for train_index, test_index in kf:
    model.fit(X[cond][train_index], y[cond][train_index], nb_epoch=10, batch_size=200, verbose=0)
    scores.append(model.evaluate(X[cond][test_index], y[cond][test_index], show_accuracy=1))
model.fit(X, y, nb_epoch=100, batch_size=200, verbose=0)
scores.append(model.evaluate(X, y, show_accuracy=1))
print scores
print np.mean(np.array(scores), axis=0)


print model.predict_classes(X[cond][test_index]), np.argmax(y[cond][test_index], axis=1)
print set(model.predict_classes(X))


logit = Sequential()
logit.add(Dense(X.shape[1], y.shape[1], init='uniform', activation='softmax'))
logit_sgd = SGD()


logit.compile(loss='categorical_crossentropy', optimizer=logit_sgd)


scores = []
kf = cross_validation.KFold(X.shape[0], n_folds=10)
for train_index, test_index in kf:
    logit.fit(X[train_index], y[train_index], nb_epoch=100, batch_size=200, verbose=0)
    scores.append(logit.evaluate(X[test_index], y[test_index], show_accuracy=1))
print scores
print np.mean(np.array(scores), axis=0)


# # Plotting decision boundaries
# 
# We would like to see what decision boundaries is the model learning. However, one issue with plotting our data is that `X` consists of $4$ dimensional feature vectors. Hence, we transform each feature vector in `X` to a $2d$ vector using Principal Component Analysis (PCA). The vectors obtained from **PCA** are then used for showing the points in a $2d$ plane and the decision boundaries of each classifier are shown as well. 
# 

pca = PCA(n_components=2)
X_t = pca.fit_transform(X)

h = 0.1
x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, ax = plt.subplots(1,2, figsize=(20,10))
for i, v in enumerate({"Neural Net": model, "Logistic": logit}.items()):
    # here "model" is your model's prediction (classification) function
    Z = v[1].predict_classes(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])) 

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax[i].contourf(xx, yy, Z, cmap=plt.cm.Paired)
    #ax[i].set_axis('off')
    # Plot also the training points
    ax[i].scatter(X_t[:, 0], X_t[:, 1], c=np.argmax(y, axis=1), cmap=plt.cm.Paired)
    ax[i].set_title(v[0])


# # Final Analysis
# 
# From our analysis the logistic regression classifier performs slightly better than the neural network classifier. 
# 
# | Classifier | Accuracy |
# |:----------:|:--------:|
# | Logistic Regression | $96\%$ |
# | Neural Network | $95.33\%$ |
# 
#  * One of the major reason for logistic regression classifier performing better than neural net can be attributed to the fact that the number of training examples are very few (150). 
#  * The decision boundaries learned by both classifiers are linear in the PCA space. However, logistic regression classifier learns better decision boundary than the neural network. Hence, we can say that the neural network is trying to overfit.
#  * A good approach can be to try training the network with different regularization than Dropout i.e. $L1$ and $L2$ regularization. 
# 




# In this notebook I will try to implement and analyze LDA algorithm using PyMC package. The implementation is replication of the code provided at: http://stats.stackexchange.com/questions/104771/latent-dirichlet-allocation-in-pymc
# 
# The LDA model is described below:
# ![LDA Model](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Latent_Dirichlet_allocation.svg/250px-Latent_Dirichlet_allocation.svg.png)

import numpy as np
import pymc as pm
#K, V, D = 2, 4, 3 # number of topics, words, documents
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
get_ipython().magic('matplotlib inline')


K, V, D = 5, 10, 20 # number of topics, words, documents

#data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

data_temp = np.random.randint(0,10,size=(D,V))


class LDA(object):
    
    def __init__(self, data, topics=K, vocab=V):
        """
        Takes the data variable and outputs a model
        """
        self.data = data
        self.topics = topics
        self.vocab = vocab+1
        self.docs = len(self.data)
        self.alpha = np.ones(self.topics)
        self.beta = np.ones(self.vocab)

        self.theta = pm.Container([pm.CompletedDirichlet("theta_%s" % i,                                                         pm.Dirichlet("ptheta_%s" % i, theta=self.alpha))
                                   for i in range(self.docs)])
        self.phi = pm.Container([pm.CompletedDirichlet("phi_%s" % i,                                                       pm.Dirichlet("pphi_%s" % i, theta=self.beta))
                                 for i in range(self.topics)])
        self.Wd = [len(doc) for doc in self.data]

        self.Z = pm.Container([pm.Categorical("z_%s" % d, 
                                              p=self.theta[d],
                                              size=self.Wd[d],
                                              value=np.random.randint(self.topics,size=self.Wd[d]))
                               for d in range(self.docs)])
        self.W = pm.Container([pm.Categorical("w_%s,%s" % (d,i),
                                              p=pm.Lambda("phi_z_%s_%s" % (d,i), 
                                                          lambda z=self.Z[d][i], phi=self.phi: phi[z]),
                                              value=self.data[d][i],
                                              observed=True)
                               for d in range(self.docs) for i in range(self.Wd[d])])

        self.model = pm.Model([self.theta, self.phi, self.Z, self.W])
        self.mcmc = pm.MCMC(self.model)
    
    def fit(self, iterations=1000, burn_in=10):
        # Fit the model by sampling from the data iterations times with burn in of burn_in.
        self.mcmc.sample(iterations, burn=burn_in)
        
    def show_topics(self):
        # Show distribution of topics over words
        return self.phi.value
    
    def show_words(self):
        # Show distribution of words in documents over topics
        return self.W.value
    
    def KLDiv(self, p,q):
        return np.sum(p*np.log10(p/q))
    
    def cosine_sim(self, x,y):
        return np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y))
    
    def sorted_docs_sim(self):
        kldivs_docs = [(i, j, self.KLDiv(self.theta[i].value,self.theta[j].value),
                        self.cosine_sim(self.data[i], self.data[j]))
                       for i in range(len(self.theta)) for j in range(len(self.theta))
                       if i != j]
        return sorted(kldivs_docs, key=lambda x: x[3], reverse=True)
    
    def show_topic_words(self, idwords, n=10):
        for i, t in enumerate(self.phi.value):
            print "Topic %i : " % i, ", ".join(idwords[w_] for w_ in np.argsort(t[0])[-10:] if w_ < (self.vocab-1-1))
    
    def plot_data(self):
        plt.clf()
        plt.matshow(data, fignum=1000, cmap=plt.cm.Reds)
        plt.gca().set_aspect('auto')
        plt.xlabel("Words")
        plt.ylabel("Documents")
    
    def plot_words_per_topic(self, ax=None):
        if ax is None:
            plt.clf()
            fig, ax = plt.subplots(1,1)
        words = self.Z.value
        topic_dist = dict()
        for k_i in words:
            for k in k_i:
                if k not in topic_dist:
                    topic_dist[k] = 0
                topic_dist[k] += 1
        ax.bar(topic_dist.keys(), topic_dist.values())
        ax.set_xlabel("Topics")
        ax.set_ylabel("Counts")
        ax.set_title("Document words per topics")
        plt.show()
        
    def plot_word_dist(self, ax=None):
        topics = self.phi.value
        if ax is None:
            plt.clf()
            fig, ax = plt.subplots((len(topics)+1)/2, 2, figsize=(10,10))
        for i, t in enumerate(topics):
            ax[i/2][i%2].bar(range(len(t[0])), t[0])
            ax[i/2][i%2].set_title("Topic %s" % i)
        plt.suptitle("Vocab word proportions per topic")
        fig.subplots_adjust(hspace=0.5, wspace=0.5)


lda = LDA(data_temp)
lda.fit()
lda.plot_words_per_topic()


lda.plot_word_dist()


kld_sorted = lda.sorted_docs_sim()
kld_sorted[:10]


# # REAL DATA
# Now lets move to trying on some real world data. I will use the reuters corpus.
# 

from nltk.corpus import inaugural
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


# Create a vocabulary from the corpus
vocab = dict()
for fileid in inaugural.fileids():
    for word in inaugural.words(fileid):
        word = word.lower()
        if word not in stops and word.isalpha():
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1


"""
Sort the vocab keep only words which occur more than 50 times
Then Create word to id and id to word dictionaries
"""
vocab_sorted = filter(lambda x: x[1] > 50, sorted(vocab.items(), key=lambda x: x[1], reverse=True))
wordids = {v[0]: i for i, v in enumerate(vocab_sorted)}
idwords = {i: v[0] for i, v in enumerate(vocab_sorted)}
vocab_size = len(wordids)
print vocab_size


# Generate corpus document vectors
data = []
for fileid in inaugural.fileids():
    data.append([0]*vocab_size)
    for word in inaugural.words(fileid):
        word = word.lower()
        if word in wordids:
            data[-1][wordids[word]] += 1

len(data)


# Plot the document word matrix
print data[0][:10]
data = np.array(data)
plt.clf()
plt.matshow(data, fignum=1000, cmap=plt.cm.Reds)
plt.gca().set_aspect('auto')
plt.xlabel("Words")
plt.ylabel("Documents")


inaugural_lda = LDA(data, topics=10, vocab=vocab_size)


inaugural_lda.fit()
inaugural_lda.plot_words_per_topic()


inaugural_lda.plot_word_dist()


# Above diagram is wrong for Document words per topics

plt.clf()
fig, ax = plt.subplots(1,1)
words = inaugural_lda.Z.value
topic_dist = dict()
for k_i in words:
    for k in k_i:
        if k not in topic_dist:
            topic_dist[k] = 0
        topic_dist[k] += 1
ax.bar(topic_dist.keys(), topic_dist.values())
ax.set_xlabel("Topics")
ax.set_ylabel("Counts")
ax.set_title("Document words per topics")
plt.show()


for i, t in enumerate(inaugural_lda.phi.value):
    print "Topic %i : " % i, ", ".join(idwords[w_] for w_ in np.argsort(t[0])[-10:] if w_ < vocab_size -1)





# # Reinforcement Learning
# 
# This is some code replicated from http://outlace.com/Reinforcement-Learning-Part-1/ for understanding how reinforcement learning works. Please read the above link for details. 
# 
# The problem explained in the code is of playing the n-armed bandits (n slot machines) in a casino in order to maximize ones rewards. Maximum payoff is \$10 and each armed bandit has a varing probability of payoff. 
# 

import numpy as np
from scipy import stats

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")


# For replication
np.random.seed(1337)


n = 10
bandit_payoff_probs = np.random.rand(n)
print bandit_payoff_probs
print "Bandit with best payoff: %s" % np.argmax(bandit_payoff_probs)


def rewards(p, max_cost=5):
    # Return the total reward equal to the times random number < p
    return np.sum(np.random.rand(max_cost) < p)
    
rewards(0.3)


fig, ax = plt.subplots(3,3, figsize=(20,20))

for i, (axi, p) in enumerate(zip(ax.flatten(), [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])):
    axi.hist([rewards(p) for k in xrange(10000)], bins=range(6))
    axi.set_title("p=%s" % p)


def best_arm(mean_rewards):
    return np.argmax(mean_rewards)
    
best_arm([0.1,0.5, 0.3])


eps = 0.05 # epsilon for randomly using a bandit
num_plays = 500
running_mean_reward = 0
mean_rewards = np.zeros(n)
count_arms = np.zeros(n)
print bandit_payoff_probs

plt.clf()
fig, ax = plt.subplots(3,3, figsize=(30,30))
for i, eps in enumerate([0.05, 0.1, 0.2]):
    mean_rewards = np.zeros(n)
    count_arms = np.zeros(n)
    ax[i,0].set_xlabel("Plays")
    ax[i,0].set_ylabel("Mean Reward (eps = %s)" % eps)
    for j in xrange(1,num_plays+1):
        if np.random.rand() > eps:
            choice = best_arm(mean_rewards)
        else:
            choice = np.random.randint(n)
        curr_reward = rewards(bandit_payoff_probs[choice])
        count_arms[choice] += 1
        mean_rewards[choice] += (curr_reward - mean_rewards[choice]) * 1. / count_arms[choice]
        running_mean_reward += (curr_reward - running_mean_reward) * 1. / j
        ax[i,0].scatter(j,running_mean_reward)

    width = 0.4
    ax[i,1].bar(np.arange(n), count_arms * 1. / num_plays, width, color="r", label="Selected")
    ax[i,1].bar(np.arange(n) + width, bandit_payoff_probs, width, color="b", label="Payoff")
    ax[i,1].set_xlabel("Bandit")
    ax[i,1].set_ylabel("p(selected) and p(payoff)")
    ax[i,1].legend(loc="upper right")
    ax[i,2].bar(np.arange(n), mean_rewards)
    ax[i,2].set_xlabel("Bandit")
    ax[i,2].set_ylabel("Mean Reward")



# ## Using softmax function for random selection of arms
# 

def best_arm(mean_rewards, tau=1.0):
    exp_r = np.exp(mean_rewards/tau)
    exp_r = exp_r / exp_r.sum()
    return np.random.choice(range(n), p=exp_r, size=1)[0]

[best_arm(mean_rewards) for k in xrange(10)]


plt.hist(np.random.choice([1,2,3], p=[0.5,0.1,0.4], size=100))


num_plays = 500
running_mean_reward = 0

plt.clf()
fig, ax = plt.subplots(3,3, figsize=(30,30))
for i, tau in enumerate([0.9, 1., 1.1]):
    mean_rewards = np.zeros(n)
    count_arms = np.zeros(n)
    ax[i,0].set_xlabel("Plays")
    ax[i,0].set_ylabel("Mean Reward (tau = %s)" % tau)
    for j in xrange(1,num_plays+1):
        choice = best_arm(mean_rewards, tau=tau)
        curr_reward = rewards(bandit_payoff_probs[choice])
        count_arms[choice] += 1
        mean_rewards[choice] += (curr_reward - mean_rewards[choice]) * 1. / count_arms[choice]
        running_mean_reward += (curr_reward - running_mean_reward) * 1. / j
        ax[i,0].scatter(j,running_mean_reward)

    width = 0.4
    ax[i,1].bar(np.arange(n), count_arms * 1. / num_plays, width, color="r", label="Selected")
    ax[i,1].bar(np.arange(n) + width, bandit_payoff_probs, width, color="b", label="Payoff")
    ax[i,1].set_xlabel("Bandit")
    ax[i,1].set_ylabel("p(selected) and p(payoff)")
    ax[i,1].legend(loc="upper right")
    ax[i,2].bar(np.arange(n), mean_rewards)
    ax[i,2].set_xlabel("Bandit")
    ax[i,2].set_ylabel("Mean Reward")






# # Dynamic Programming
# 
# Source: https://www.topcoder.com/community/data-science/data-science-tutorials/dynamic-programming-from-novice-to-advanced/
# 

# ## Sum of coins
# Given coins of value $V_1, V_2, ... V_n$ find min coins required to create a sum $S$
# 

def wrapper(S, coins):
    states = [(10000, set()) for k in range(S+1)]
    states = [(10000, []) for k in range(S+1)]
    return n_coins(S, coins, states)

def n_coins(S, coins, states):
    if S < 1:
        return (10000, [])
    if S in coins:
        return (1, [S])
    if S < min(coins):
        return (10000, [])
    if states[S][0] < 10000:
        return states[S]
    for c in coins:
        print S, states[S]
        if c > S:
            continue
        new_s = S - c
        new_state = n_coins(new_s, coins, states)
        new_state = (new_state[0]+1, new_state[1] + [c])
        if new_state[0] < states[S][0]:
            states[S] = new_state
    return states[S]

def n_coins_iter(S, coins):
    states = [(10000, []) for k in range(S+1)]
    states[0] = (0, [])
    for s in range(1,S+1):
        for c in coins:
            if c <= s and states[s-c][0] < states[s][0]:
                states[s] = (states[s-c][0] + 1, states[s-c][1] + [c])
    print states, states[S]


S = int(raw_input("S: "))
coins = set(int(k) for k in raw_input("Coins(comma seperated): ").split(','))
if min(coins) < 1:
    raise Exception("Coins should be positive values >= 1")
print S, coins
print wrapper(S, coins)
print n_coins_iter(S, coins)


# ## Longest sequence problem
# Given a sequence of $N$ numbers – $A[1] , A[2] , …, A[N]$ . Find the length of the longest non-decreasing sequence.
# 
# ### Approach
# $len_{LSS}(A, 0, N) = min(\{len_{LSS}(A, i, N-i) | i \in [1,N]\})$
# 

def wrapper(arr):
    states = [1]*len(arr)
    return longest_sub(arr, len(arr)-1, states)

def longest_sub(arr, i, states):
    if i <= 0:
        return 1
    if states[i] > 1:
        return states[i]
    for j in range(i):
        lj = longest_sub(arr, j, states)
        if arr[j] <= arr[i]:
            print j, i, states
            states[i] = lj + 1
        else:
            states[i] = lj
    return states[i]


arr = [int(k) for k in raw_input("Array(comma seperated): ").split(',')]
print arr
print wrapper(arr)


# ## Apples on a table
# A table composed of N x M cells, each having a certain quantity of apples, is given. You start from the upper-left corner. At each step you can go down or right one cell. Find the maximum number of apples you can collect.
# 

def wrapper(mat, N, M):
    states = [[0 for c in range(M)] for r in range(N)]
    return apples(mat,0,0,N,M, states)


def apples(mat,i,j,N,M, states):
    if i >= N or j >= M:
        return 0
    if states[i][j] > 0:
        return states[i][j]
    states[i][j] = mat[i][j] + max([apples(mat,i+1,j,N,M, states), apples(mat,i,j+1,N,M, states)])
    return states[i][j]


arr = [int(k) for k in raw_input("Array(comma seperated): ").split(',')]
N, M = [int(k) for k in raw_input("N, M (comma seperated): ").split(',')]
mat = [arr[i*N:i*N+M] for i in range(N)]
print mat
wrapper(mat, N, M)





# Understanding Logisitic Regression
# =================
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")


# ## Understanding probability, odds and logodds
# 

p = np.logspace(-10,0,100)
odds = p/(1.0-p)
logodds = np.log10(odds)
fig, ax = plt.subplots(1,3)

ax[0].plot(p, odds)
ax[0].set_xlabel("$p$")
ax[0].set_ylabel("odds = $p/(1-p)$")

ax[1].plot(p, logodds)
ax[1].set_xlabel("$p$")
ax[1].set_ylabel("logodds = $log(p/(1-p))$")


ax[2].plot(odds, logodds)
ax[2].set_xlabel("$odds$")
ax[2].set_ylabel("logodds = $log(p/(1-p))$")


from sklearn import linear_model, datasets, cross_validation


diabetes = datasets.load_diabetes()


X = diabetes.data[:]
y = np.vectorize(lambda x: 0 if x< 100 else 1)(diabetes.target)
logit = linear_model.LogisticRegression()
acc = cross_validation.cross_val_score(logit, X, y, n_jobs=1)
print acc


logit.fit(X, y)


logit.coef_


X.shape


np.vectorize(lambda x: 0 if x< 100 else 1)(y)


np.unique(y)


df = pd.DataFrame(X, columns=["x%s" %k for k in range(X.shape[1])])
df["y_lbl"] = y


df.head()


df.plot(kind="hist")





# Unattacked Queens
# ===
# The properties of chess pieces play a part in many challenges, including in a group of problems about unattacked queens. Imagine three white queens and five black queens on a 5 × 5 chessboard. Can you arrange them so that no queen of one color can attack a queen of the other color? There is only one solution, excluding reflections and rotations.
# http://www.scientificamerican.com/article/martin-gardner-fans-try-these-mathematical-games/
# 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from copy import deepcopy

get_ipython().magic('matplotlib inline')


fig, ax = plt.subplots()

min_val, max_val, diff = 0., 10., 1.

#imshow portion
N_points = (max_val - min_val) / diff
imshow_data = np.random.rand(N_points, N_points)
ax.imshow(imshow_data, interpolation='nearest')

#text portion
ind_array = np.arange(min_val, max_val, diff)
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = 'x' if (x_val + y_val)%2 else 'o'
    ax.text(x_val, y_val, c, va='center', ha='center')

#set tick marks for grid
ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(min_val-diff/2, max_val-diff/2)
ax.set_ylim(min_val-diff/2, max_val-diff/2)
ax.grid()
plt.show()


N = 5
#board = [[(i*N+j)%2 for j in range(N)] for i in range(N)]
board = [[0 for j in range(N)] for i in range(N)]
temp_board = deepcopy(board)
plt.matshow(board, interpolation='nearest', cmap=plt.get_cmap("gray"))
plt.grid()
plt.show()


def addQueen(board, pos, val=-2):
    x, y = pos
    board[x][y] = val
    val = (abs(val)+1)*val/abs(val)
    for i in range(1,N):
        board[(x+i)%N][y] = val
        board[x][(y+i)%N] = val
        for j in [1, -1]:
            for k in [1,-1]:
                if (x+i*j) >= 0 and (x+i*j) < N and (y+i*k) >= 0 and (y+i*k) < N:
                    board[(x+i*j)][(y+i*k)] = val
    plt.matshow(board, interpolation='nearest', cmap=plt.get_cmap("Set3"), vmin=-3,vmax=3)
    plt.grid()
    plt.show()
    return board


board = deepcopy(temp_board)
addQueen(board, (0,0))
addQueen(board, (N-2,N-1), val = 2)


plt.imshow(temp_board, interpolation='nearest', cmap=plt.get_cmap("gray"))





