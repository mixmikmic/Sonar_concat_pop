# ## Missing Value Support in pomegranate

get_ipython().run_line_magic('pylab', 'inline')
from pomegranate import *
import seaborn
seaborn.set_style('whitegrid')
numpy.random.seed(0)


# The majority of machine learning algorithms assume that they are operating on a fully observed data set. In contast, a great deal of data sets in the real world are missing some values. Sometimes, this missingness is missing at random (MAR), which means that there is no important pattern to the missingness, and sometimes the missingness itself can be interpreted as a feature. For example, in the Titanic data set, males were more likely to have missing records than females were, and those without children were more likely to have missing records. 
# 
# A common approach to bridging this gap is to impute the missing values and then treat the entire data set as observed. For continuous features this is commonly done by replacing the missing values with the mean or median of the column. For categorical variables it is commonly done by replacing the missing values with the most common category observed in that column. While these techniques are simple and allow for almost any ML algorithm to be run, they are frequently suboptimal. Consider the follow simple example of continuous data that is bimodally distributed:

X = numpy.concatenate([numpy.random.normal(0, 1, size=(1000)), numpy.random.normal(6, 1, size=(1250))])

plt.title("Bimodal Distribution", fontsize=14)
plt.hist(X, bins=numpy.arange(-3, 9, 0.1), alpha=0.6)
plt.ylabel("Count", fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14)
plt.yticks(fontsize=12)
plt.vlines(numpy.mean(X), 0, 80, color='r', label="Mean")
plt.vlines(numpy.median(X), 0, 80, color='b', label="Median")
plt.legend(fontsize=14)
plt.show()


# The data peaks around 0 and around 6. Replacing the missing values with ~3 will be inserting values into the data set that mostly don't exist in the observed values. The median is slightly better, but will still cause the imputed values to be in one of the two clusters. This has the effect of essentially increasing the variance of the appropriate distribution. Let's take a look at what the distribution looks like if we add 500 missing values and then impute them using the mean of the observed values.

X = numpy.concatenate([X, [numpy.nan]*500])
X_imp = X.copy()
X_imp[numpy.isnan(X_imp)] = numpy.mean(X_imp[~numpy.isnan(X_imp)])

plt.title("Bimodal Distribution", fontsize=14)
plt.hist(X_imp, bins=numpy.arange(-3, 9, 0.1), alpha=0.6)
plt.ylabel("Count", fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14)
plt.yticks(fontsize=12)
plt.vlines(numpy.mean(X), 0, 80, color='r', label="Mean")
plt.vlines(numpy.median(X), 0, 80, color='b', label="Median")
plt.legend(fontsize=14)
plt.show()


# It doesn't appear to be that great. We can see the issue with increased variance by trying to fit a Gaussian mixture model to the data with the imputed values, versus fitting it to the data and ignoring missing values.

x = numpy.arange(-3, 9, 0.1)
model1 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X_imp.reshape(X_imp.shape[0], 1))
model2 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X.reshape(X.shape[0], 1))
p1 = model1.probability(x.reshape(x.shape[0], 1))
p2 = model2.probability(x.reshape(x.shape[0], 1))

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.title("Mean Impute Missing Values", fontsize=14)
plt.hist(X_imp, bins=x, alpha=0.6, density=True)
plt.plot(x, p1, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)

plt.subplot(122)
plt.title("Ignore Missing Values", fontsize=14)
plt.hist(X[~numpy.isnan(X)], bins=x, alpha=0.6, density=True)
plt.plot(x, p2, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.show()


# When we impute the missing values, it seems that one component is fit properly and one has drastically increased variance. In contrast, when we ignore the missing values, we fit a model that represents the underlying data much more faithfully.
# 
# At this point, you may think that as long as the data comes from a single distribution it shouldn't matter if you do a mean imputation of the data. However, this has the effect of shrinking the variance inappropriately. Let's take a look quickly at data drawn from a single Gaussian.

X = numpy.concatenate([numpy.random.normal(0, 1, size=(750)), [numpy.nan]*250])
X_imp = X.copy()
X_imp[numpy.isnan(X_imp)] = numpy.mean(X_imp[~numpy.isnan(X_imp)])

x = numpy.arange(-3, 3, 0.1)
d1 = NormalDistribution.from_samples(X_imp)
d2 = NormalDistribution.from_samples(X)
p1 = d1.probability(x.reshape(x.shape[0], 1))
p2 = d2.probability(x.reshape(x.shape[0], 1))

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.title("Mean Impute Missing Values", fontsize=14)
plt.hist(X_imp, bins=x, alpha=0.6, density=True, label="$\sigma$ = {:4.4}".format(d1.parameters[1]))
plt.plot(x, p1, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.legend(fontsize=14)

plt.subplot(122)
plt.title("Ignore Missing Values", fontsize=14)
plt.hist(X[~numpy.isnan(X)], bins=x, alpha=0.6, density=True, label="$\sigma$ = {:4.4}".format(d2.parameters[1]))
plt.plot(x, p2, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.show()


# Even when the data is all drawn from a single, Gaussian, distribution, it is not a great idea to do mean imputation. We can see that the standard deviation of the learned distribution is significantly smaller than the true standard deviation (of 1), whereas if the missing data is ignored the value is closer.

# This might all be intuitive for a single variable. However, the concept of only collecting sufficient statistics from values that are present in the data and ignoring the missing values can be used in much more complicated, and/or multivariate models. Let's take a look at how well one can estimate the covariance matrix of a multivariate Gaussian distribution using these two strategies. 

n, d, steps = 1000, 10, 50
diffs1 = numpy.zeros(int(steps*0.86))
diffs2 = numpy.zeros(int(steps*0.86))

X = numpy.random.normal(6, 3, size=(n, d))

for k, size in enumerate(range(0, int(n*d*0.86), n*d / steps)):
    idxs = numpy.random.choice(numpy.arange(n*d), replace=False, size=size)
    i, j = idxs / d, idxs % d

    cov_true = numpy.cov(X, rowvar=False, bias=True)
    X_nan = X.copy()
    X_nan[i, j] = numpy.nan

    X_mean = X_nan.copy()
    for col in range(d):
        mask = numpy.isnan(X_mean[:,col])
        X_mean[mask, col] = X_mean[~mask, col].mean()

    diff = numpy.abs(numpy.cov(X_mean, rowvar=False, bias=True) - cov_true).sum()
    diffs1[k] = diff

    dist = MultivariateGaussianDistribution.from_samples(X_nan)
    diff = numpy.abs(numpy.array(dist.parameters[1]) - cov_true).sum()
    diffs2[k] = diff


plt.title("Error in Multivariate Gaussian Covariance Matrix", fontsize=16)
plt.plot(diffs1, label="Mean")
plt.plot(diffs2, label="Ignore")

plt.xlabel("Percentage Missing", fontsize=14)
plt.ylabel("L1 Errors", fontsize=14)
plt.xticks(range(0, 51, 10), numpy.arange(0, 5001, 1000) / 5000.)
plt.xlim(0, 50)
plt.legend(fontsize=14)
plt.show()


# In even the simplest case of Gaussian distributed data with a diagonal covariance matrix, it is more accurate to use the ignoring strategy rather than imputing the mean. When the data set is mostly unobserved the mean imputation strategy tends to do better in this case, but only because there is so little data for the ignoring strategy to actually train on. The deflation of the variance benefits the mean imputation strategy because all of the off-diagonal elements should be 0, but are likely to be artificially high when there are only few examples of the pairs of the variables co-existing in the dataset. This weakness in the ignoring strategy also makes it more likely to encounter linear algebra errors, such as a non-invertable covariance matrix.

# This long introduction is a way of saying that pomegranate uses a strategy of ignoring missing values instead of attempting to impute them, followed by fitting to the newly complete data set. There are other imputation strategies, such as those based on EM, that would be a natural fit with the types of probabilistic models implemented in pomegranate. While those have not yet been added, they would be a good addition that I hope to get to this year.
# 
# Let's now take a look at how to use missing values in some pomegranate models!

# ### 1. Distributions
# 
# We've seen some examples of fitting distributions to missing data. For univariate distributions, the missing values are simply ignored when fitting to the data.

X = numpy.random.randn(100)
X_nan = numpy.concatenate([X, [numpy.nan]*100])

print "Fitting only to observed values:"
print NormalDistribution.from_samples(X)
print 
print "Fitting to observed and missing values:"
print NormalDistribution.from_samples(X_nan)


# This may seem to be an obvious thing to do. However, it suggests a way for dealing with multivariate data being modeled with an IndependentComponentsDistribution when some of the features are missing. Specifically, treat each column independently, and update based on the observed values, regardless of if there is an unobserved value in the same sample but another column. For example:

X = numpy.random.normal(0, 1, size=(500, 3))
idxs = numpy.random.choice(1500, replace=False, size=500)
i, j = idxs // 3, idxs % 3
X[i, j] = numpy.nan

d = IndependentComponentsDistribution.from_samples(X, distributions=[NormalDistribution]*3)
d


# Easy. As we saw above, we can do the same to learn a multivariate Gaussian distribution in the presence of missing data. Again, we don't need to change anything about how we interact with the data, and there are no flags to toggle.
# 
# The last aspect is that the probability of missing data under any univariate distribution is 1, for the purposes of downstream algorithms.

NormalDistribution(1, 2).probability(numpy.nan)


# In an IndependentComponentsDistribution, this just means that when multiplying together the probabilities of each feature to get the total probability, that some dimensions don't factor into the calculation.

d.probability((numpy.nan, 2, 3))


d.distributions[1].probability(2) * d.distributions[2].probability(3)


# ### 2. K-Means Clustering
# 
# K-means clustering mostly serves a helper role in initializing mixture models and hidden Markov models. However, it can still be used by itself if desired. In addition to having the same parallelization and out-of-core capabilities of the main models, it also supports missing values now.

X = numpy.concatenate([numpy.random.normal(0, 1, size=(50, 2)), numpy.random.normal(3, 1, size=(75, 2))])
X_nan = X.copy()

idxs = numpy.random.choice(250, replace=False, size=50)
i, j = idxs // 2, idxs % 2
X_nan[i, j] = numpy.nan


# Just like the other models, you don't need to change the method calls in order to handle missing data. You can fit a K-means model to data sets with missing values and make predictions on samples with missing values in the same way you would without missing values. The prediction step will assign samples to the nearest centroid in the dimensions that are observed, ignoring the missing values.

model1 = Kmeans.from_samples(2, X)
model2 = Kmeans.from_samples(2, X_nan)

y1 = model1.predict(X)
y2 = model2.predict(X_nan)

plt.figure(figsize=(14, 6))
plt.subplot(121)
plt.title("Fit w/o Missing Values", fontsize=16)
plt.scatter(X[y1 == 0,0], X[y1 == 0,1], color='b')
plt.scatter(X[y1 == 1,0], X[y1 == 1,1], color='r')

plt.subplot(122)
plt.title("Fit w/ Missing Values", fontsize=16)
plt.scatter(X[y2 == 0,0], X[y2 == 0,1], color='b')
plt.scatter(X[y2 == 1,0], X[y2 == 1,1], color='r')
plt.show()


# We can see that there are some blue points in the red cluster on the right plot because those samples are entirely NaN. Any sample that is entirely NaN is assigned to cluster 0. Otherwise, it's still able to identify the two clusters even there there are many missing values.

# ### 3. General Mixture Models
# 
# Missing value support for mixture models follows that of k-means fairly closely. Essentially, one passes in a data set containing missing values denoted as `numpy.nan` and they get used appropriately for the fit and predict steps. All methods automatically handle missing values appropriately without any additional flags or methods. 
# 
# Since training is an iterative procedure that involves calculating the probabilities of samples given each component, multivariate Gaussian mixtures will be much slower when handling missing values than they would be when using only observed values. This is because an inverse covariance matrix has to be calculated by subsetting the covariance matrix and inverting it based only on the observed dimensions for each sample. Each sample, then, needs its own matrix inversion. Since there is no single inverse covariance matrix, one also cannot use BLAS or a GPU to accelerate this step. 

X = numpy.concatenate([numpy.random.normal(0, 1, size=(1000, 10)), numpy.random.normal(2, 1, size=(1250, 10))])

idxs = numpy.random.choice(22500, replace=False, size=5000)
i, j = idxs // 10, idxs % 10

X_nan = X.copy()
X_nan[i, j] = numpy.nan

get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X, max_iterations=10)')
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_nan, max_iterations=10)')


# However, if one was modeling each dimension independently, there should be no hit at all!

get_ipython().run_line_magic('timeit', '-n 100 GeneralMixtureModel.from_samples([NormalDistribution]*2, 2, X, max_iterations=10)')
get_ipython().run_line_magic('timeit', '-n 100 GeneralMixtureModel.from_samples([NormalDistribution]*2, 2, X_nan, max_iterations=10)')


# ### 4. Naive Bayes / Bayes Classifiers

# Support for these models mirrors what's been seen before. However, since the fitting step does not involve calculating probabilities of samples, it should be no slower to train them on data sets involving missing values than to train them on dense data sets.

y = numpy.concatenate([numpy.zeros(1000), numpy.ones(1250)])

get_ipython().run_line_magic('timeit', '-n 100 BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)')
get_ipython().run_line_magic('timeit', '-n 100 BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y)')


# Since pomegranate also has semi-supervised learning built-in, this means that one can now fit Bayes classifiers on data sets with missingness in both the labels and in the values! Since semi-supervised learning does rely on EM, it will be slower to train multivariate Gaussian models with missing values than not to.

idx = numpy.random.choice(2250, replace=False, size=750)
y_nan = y.copy()
y_nan[idx] = -1

model = BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y_nan, verbose=True)


get_ipython().run_line_magic('timeit', 'BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y_nan)')
get_ipython().run_line_magic('timeit', 'BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y_nan)')


# ### 5. Hidden Markov Models
# 
# Hidden Markov models are slightly different from the others, in that they operate over sequences. This adds another level of complication to the model because the forward and backward algorithms are needed in order to identify the best component for each observation. Typically this involves calculating the probability of each observation given each state, and taking the sum of all paths through the model, multiplying the transition probability of each edge crossed by the emission probability of the state you transition to emitting the next character. If this is a univariate model and the character is missing, you ignore the emission probability and just multiply by the transition probability. This is easily done by having the probability of missing values be 1 under all univariate models.

d1 = DiscreteDistribution({'A': 0.25, 'B': 0.75})
d2 = DiscreteDistribution({'A': 0.67, 'B': 0.33})

s1 = State(d1, name="s1")
s2 = State(d2, name="s2")

model = HiddenMarkovModel()
model.add_states(s1, s2)
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.5)
model.add_transition(s1, s2, 0.5)
model.add_transition(s2, s2, 0.5)
model.add_transition(s2, s1, 0.5)
model.bake()


# Now let's run the forward algorithm on a simple sequence.

numpy.exp(model.forward(['A', 'B', 'A', 'A']))


# Let's see what happens when we remove one of the characters.

numpy.exp(model.forward(['A', 'nan', 'A', 'A']))


# We can see that initially the first character is aligned to s1 because there is a 100% chance of going from the start state to s1. The value is 0.25 because it is equal to the transition probability (1.0) multiplied by the emission probability (0.25). In the next step, you can see that the probability is equally diffused between two options, staying in the current state (transition probability of 0.5) and moving to s2 (also transition probability of 0.5). Since the character is missing, there is no emission probability to multiply by. 
# 
# If we want to decode the sequence we can call the same methods as before.

model.predict(['A', 'A', 'B', 'B', 'A', 'A'])


model.predict(['A', 'nan', 'B', 'B', 'nan', 'A'])


# Fitting is pretty much the same story as the previous models. Like the Bayes classifiers, one can now train a hidden Markov model in a supervised manner, having some observations in the sequence missing, but also labels on each observation. Labeled missing data can still be used to train the transition parameters.

# ### 6. Bayesian Networks
# 
# Bayesian networks could previously handle missing data in the `predict` and `predict_proba` methods. In fact, handling missing data was the area in which they excelled. However, one couldn't calculate the probability of a sample containing missing values, fit a Bayesian network to incomplete data sets, or learn the structure of a network on incomplete data sets. The new missing value support allows all of these things (except Chow-Liu tree building on missing data sets)! To be clear, there is a very common iterative technique for fitting/learning models on incomplete data sets using an iterative EM based approach. This does not do that. This only ignores sufficient statistics from missing samples, and so is not an iterative approach.
# 
# Learning the structure of a network on an incomplete data set should take a similar amount of time as learning it on a complete data set. If you are indicating missing values in numeric data sets, you will have to convert your data set to floats, whereas previously integers would be fine. If your data set is complete, there is no need.

X = numpy.random.randint(3, size=(500, 10)).astype('float64')

idxs = numpy.random.choice(5000, replace=False, size=2000)
i, j = idxs // 10, idxs % 10
X_nan = X.copy()
X_nan[i, j] = numpy.nan

get_ipython().run_line_magic('timeit', "-n 100 BayesianNetwork.from_samples(X, algorithm='exact')")
get_ipython().run_line_magic('timeit', "-n 100 BayesianNetwork.from_samples(X_nan, algorithm='exact')")


# ## Conclusions
# 
# Missing value support has been added to pomegranate as of v0.9.0! A lot of care was taken to make the interface as simple to the end user while not compromising on speed. While multivariate Gaussian distributions, and the compositional models that are built on top of them, may be slower on incomplete data sets than on complete ones, everything else should take a similar amount of time!
# 
# The implemented missing value support focuses on ignoring data that is missing. Another approach that works well is to use an EM based algorithm to impute the missing values based on the observed values and the model and fit to that complete data set. This works well in the framework of probabilistic modeling and is a natural addition to pomegranate that I hope to add in the coming year.
# 
# As always, feedback and questions are always welcome!

# ## Multi-threaded Parallelism and GPU Support
# 
# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com
# 
# pomegranate supports parallelization through a set of built in functions based off of joblib. All computationally intensive functions in pomegranate are implemented in cython with the global interpreter lock (GIL) released, allowing for multithreading to be used for efficient parallel processing. 
# 
# These functions can all be simply parallelized by passing in `n_jobs=X` to the method calls. This tutorial will demonstrate how to use those calls. First we'll look at a simple multivariate Gaussian mixture model, and compare its performance to sklearn. Then we'll look at a hidden Markov model with Gaussian emissions, and lastly we'll look at a mixture of Gaussian HMMs. These can all utilize the build-in parallelization that pomegranate has.
# 
# Let's dive right in!

get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pandas
import random
import numpy
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import itertools

from pomegranate import *

random.seed(0)
numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


get_ipython().run_line_magic('pylab', 'inline')
from sklearn.mixture import GaussianMixture
from pomegranate import *
import seaborn, time
seaborn.set_style('whitegrid')


def create_dataset(n_samples, n_dim, n_classes, alpha=1):
    """Create a random dataset with n_samples in each class."""
    
    X = numpy.concatenate([numpy.random.normal(i*alpha, 1, size=(n_samples, n_dim)) for i in range(n_classes)])
    y = numpy.concatenate([numpy.zeros(n_samples) + i for i in range(n_classes)])
    idx = numpy.arange(X.shape[0])
    numpy.random.shuffle(idx)
    return X[idx], y[idx]


# ## 1. General Mixture Models
# 
# pomegranate has a very efficient implementation of mixture models, particularly Gaussian mixture models. Lets take a look at how fast pomegranate is versus sklearn, and then see how much faster parallelization can get it to be.

n, d, k = 1000000, 5, 3
X, y = create_dataset(n, d, k)

print "sklearn GMM"
get_ipython().run_line_magic('timeit', "GaussianMixture(n_components=k, covariance_type='full', max_iter=15, tol=1e-10).fit(X)")
print 
print "pomegranate GMM"
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=15, stop_threshold=1e-10)')
print
print "pomegranate GMM (4 jobs)"
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, n_jobs=4, max_iterations=15, stop_threshold=1e-10)')


# It looks like on a large dataset not only is pomegranate faster than sklearn at performing 15 iterations of EM on 3 million 5 dimensional datapoints with 3 clusters, but the parallelization is able to help in speeding things up. 
# 
# Lets now take a look at the time it takes to make predictions using GMMs. Lets fit the model to a small amount of data, and then predict a larger amount of data drawn from the same underlying distributions.

d, k = 25, 2
X, y = create_dataset(1000, d, k)
a = GaussianMixture(k, n_init=1, max_iter=25).fit(X)
b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=25)


del X, y
n = 1000000
X, y = create_dataset(n, d, k)

print "sklearn GMM"
get_ipython().run_line_magic('timeit', '-n 1 a.predict_proba(X)')
print
print "pomegranate GMM"
get_ipython().run_line_magic('timeit', '-n 1 b.predict_proba(X)')
print
print "pomegranate GMM (4 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 b.predict_proba(X, n_jobs=4)')


# It looks like pomegranate can be slightly slower than sklearn when using a single processor, but that it can be parallelized to get faster performance. At the same time, predictions at this level happen so quickly (millions per second) that this may not be the most reliable test for parallelization.
# 
# To ensure that we're getting the exact same results just faster, lets subtract the predictions from each other and make sure that the sum is equal to 0.

print (b.predict_proba(X) - b.predict_proba(X, n_jobs=4)).sum()


# Great, no difference between the two.
# 
# Lets now make sure that pomegranate and sklearn are learning basically the same thing. Lets fit both models to some 2 dimensional 2 component data and make sure that they both extract the underlying clusters by plotting them.

d, k = 2, 2
X, y = create_dataset(1000, d, k, alpha=2)
a = GaussianMixture(k, n_init=1, max_iter=25).fit(X)
b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=25)

y1, y2 = a.predict(X), b.predict(X)

plt.figure(figsize=(16,6))
plt.subplot(121)
plt.title("sklearn clusters", fontsize=14)
plt.scatter(X[y1==0, 0], X[y1==0, 1], color='m', edgecolor='m')
plt.scatter(X[y1==1, 0], X[y1==1, 1], color='c', edgecolor='c')

plt.subplot(122)
plt.title("pomegranate clusters", fontsize=14)
plt.scatter(X[y2==0, 0], X[y2==0, 1], color='m', edgecolor='m')
plt.scatter(X[y2==1, 0], X[y2==1, 1], color='c', edgecolor='c')


# It looks like we're getting the same basic results for the two. The two algorithms are initialized a bit differently, and so it can be difficult to directly compare the results between them, but it looks like they're getting roughly the same results.

# ## 3. Multivariate Gaussian HMM
# 
# Now let's move on to training a hidden Markov model with multivariate Gaussian emissions with a diagonal covariance matrix. We'll randomly generate some Gaussian distributed numbers and use pomegranate with either one or four threads to fit our model to the data.

X = numpy.random.randn(1000, 500, 50)

print "pomegranate Gaussian HMM (1 job)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5, n_jobs=2)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5, n_jobs=4)')


# All we had to do was pass in the n_jobs parameter to the fit function in order to get a speed improvement. It looks like we're getting a really good speed improvement, as well! This is mostly because the HMM algorithms perform a lot more operations than the other models, and so spend the vast majority of time with the GIL released. You may not notice as strong speedups when using a MultivariateGaussianDistribution because BLAS uses multithreaded operations already internally, even when only one job is specified.
# 
# Now lets look at the prediction function to make sure the we're getting speedups there as well. You'll have to use a wrapper function to parallelize the predictions for a HMM because it returns an annotated sequence rather than a single value like a classic machine learning model might.

model = HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=2, verbose=False)

print "pomegranate Gaussian HMM (1 job)"
get_ipython().run_line_magic('timeit', 'predict_proba(model, X)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', 'predict_proba(model, X, n_jobs=2)')


# Great, we're getting a really good speedup on that as well! Looks like the parallel processing is more efficient with a bigger, more complex model, than with a simple one. This can make sense, because all inference/training is more complex, and so there is more time with the GIL released compared to with the simpler operations.

# ## 4. Mixture of Hidden Markov Models
# 
# Let's stack another layer onto this model by making it a mixture of these hidden Markov models, instead of a single one. At this point we're sticking a multivariate Gaussian HMM into a mixture and we're going to train this big thing in parallel.

def create_model(mus):
    n = mus.shape[0]
    
    starts = numpy.zeros(n)
    starts[0] = 1.
    
    ends = numpy.zeros(n)
    ends[-1] = 0.5
    
    transition_matrix = numpy.zeros((n, n))
    distributions = []
    
    for i in range(n):
        transition_matrix[i, i] = 0.5
        
        if i < n - 1:
            transition_matrix[i, i+1] = 0.5
    
        distribution = IndependentComponentsDistribution([NormalDistribution(mu, 1) for mu in mus[i]])
        distributions.append(distribution)
    
    model = HiddenMarkovModel.from_matrix(transition_matrix, distributions, starts, ends)
    return model
    

def create_mixture(mus):
    hmms = [create_model(mu) for mu in mus]
    return GeneralMixtureModel(hmms)

n, d = 50, 10
mus = [(numpy.random.randn(d, n)*0.2 + numpy.random.randn(n)*2).T for i in range(2)]


model = create_mixture(mus)
X = numpy.random.randn(400, 150, d)

print "pomegranate Mixture of Gaussian HMMs (1 job)"
get_ipython().run_line_magic('timeit', 'model.fit(X, max_iterations=5)')
print

model = create_mixture(mus)
print "pomegranate Mixture of Gaussian HMMs (2 jobs)"
get_ipython().run_line_magic('timeit', 'model.fit(X, max_iterations=5, n_jobs=2)')


# Looks like we're getting a really nice speed improvement when training this complex model. Let's take a look now at the time it takes to do inference with it.

model = create_mixture(mus)

print "pomegranate Mixture of Gaussian HMMs (1 job)"
get_ipython().run_line_magic('timeit', 'model.predict_proba(X)')
print

model = create_mixture(mus)
print "pomegranate Mixture of Gaussian HMMs (2 jobs)"
get_ipython().run_line_magic('timeit', 'model.predict_proba(X, n_jobs=2)')


# We're getting a good speed improvement here too through parallelization.

# ## Conclusions
# 
# Hopefully you'll find pomegranate useful in your work! Parallelization should allow you to train complex models faster than before. Keep in mind though that there is an overhead to using parallel processing, and so it's possible that on some smaller examples it does not work as well. In general, the bigger the dataset, the closer to a linear speedup you'll get with pomegranate.
# 
# If you have any interesting examples of how you've used pomegranate in your work, I'd love to hear about them. In addition I'd like to hear any feedback you may have on features you'd like to see. Please shoot me an email. Good luck!

# ## Semi-supervised Learning
# 
# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com
# 
# Most classical machine learning algorithms either assume that an entire dataset is either labeled (supervised learning) or that there are no labels (unsupervised learning). However, frequently it is the case that some labeled data is present but there is a great deal of unlabeled data as well. A great example of this is that of computer vision where the internet is filled of pictures (mostly of cats) that could be useful, but you don't have the time or money to label them all in accordance with your specific task. Typically what ends up happening is that either the unlabeled data is discarded in favor of training a model solely on the labeled data, or that an unsupervised model is initialized with the labeled data and then set free on the unlabeled data. Neither method uses both sets of data in the optimization process.
# 
# Semi-supervised learning is a method to incorporate both labeled and unlabeled data into the training task, typically yield better performing estimators than using the labeled data alone. There are many methods one could use for semisupervised learning, and <a href="http://scikit-learn.org/stable/modules/label_propagation.html">scikit-learn has a good write-up on some of these techniques</a>.
# 
# pomegranate natively implements semi-supervised learning through the a merger of maximum-likelihood and expectation-maximization. As an overview, the models are initialized by first fitting to the labeled data directly using maximum-likelihood estimates. The models are then refined by running expectation-maximization (EM) on the unlabeled datasets and adding the sufficient statistics to those acquired from maximum-likelihood estimates on the labeled data. Under the hood both a supervised model and an unsupervised mixture model are created using the same underlying distribution objects. The summarize method is first called using the supervised method on the labeled data, and then the summarize method is called again using the unsupervised method on the unlabeled data. This causes the sufficient statistics to be updated appropriately given the results of first maximum-likelihood and then EM. This process continues until convergence in the EM step.
# 
# Let's take a look!

get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pandas
import random
import numpy
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import itertools

from pomegranate import *

random.seed(0)
numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


get_ipython().run_line_magic('pylab', 'inline')
from pomegranate import *
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import make_blobs
import seaborn, time
seaborn.set_style('whitegrid')
numpy.random.seed(1)


# Let's first generate some data in the form of blobs that are close together. Generally one tends to have far more unlabeled data than labeled data, so let's say that a person only has 50 samples of labeled training data and 4950 unlabeled samples. In pomegranate you a sample can be specified as lacking a label by providing the integer -1 as the label, just like in scikit-learn. Let's also say there there is a bit of bias in the labeled samples to inject some noise into the problem, as otherwise Gaussian blobs are trivially modeled with even a few samples.

X, y = make_blobs(10000, 2, 3, cluster_std=2)
x_min, x_max = X[:,0].min()-2, X[:,0].max()+2
y_min, y_max = X[:,1].min()-2, X[:,1].max()+2

X_train = X[:5000]
y_train = y[:5000]

# Set the majority of samples to unlabeled.
y_train[numpy.random.choice(5000, size=4950, replace=False)] = -1

# Inject noise into the problem
X_train[y_train != -1] += 2.5

X_test = X[5000:]
y_test = y[5000:]


# Now let's take a look at the data when we plot it.

plt.figure(figsize=(8, 8))
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color='0.6')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='c')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='m')
plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='r')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


# The clusters of unlabeled data seem clear to us, and it doesn't seem like the labeled data is perfectly faithful to these clusters. This can typically happen in a semisupervised setting as well, as the data that is labeled is sometimes biased either because the labeled data was chosen as it was easy to label, or the data was chosen to be labeled in a biased maner.
# 
# Now let's try fitting a simple naive Bayes classifier to this data and compare the results when using only the labeled data to when using both the labeled and unlabeled data together.

model_a = NaiveBayes.from_samples(NormalDistribution, X_train[y_train != -1], y_train[y_train != -1])
print "Supervised Learning Accuracy: {}".format((model_a.predict(X_test) == y_test).mean())

model_b = NaiveBayes.from_samples(NormalDistribution, X_train, y_train)
print "Semisupervised Learning Accuracy: {}".format((model_b.predict(X_test) == y_test).mean())


# It seems like we get a big bump in test set accuracy when we use semi-supervised learning. Let's visualize the data to get a better sense of what is happening here.

def plot_contour(X, y, Z):
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='0.2', alpha=0.5, s=20)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', s=20)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', s=20)
    plt.scatter(X[y == 2, 0], X[y == 2, 1], color='r', s=20)
    plt.contour(xx, yy, Z)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.1), numpy.arange(y_min, y_max, 0.1))
Z1 = model_a.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(16, 16))
plt.subplot(221)
plt.title("Training Data, Supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z1)

plt.subplot(223)
plt.title("Training Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z2)

plt.subplot(222)
plt.title("Test Data, Supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z1)

plt.subplot(224)
plt.title("Test Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z2)
plt.show()


# The contours plot the decision boundaries between the different classes with the left figures corresponding to the partially labeled training set and the right figures corresponding to the test set. We can see that the boundaries learning using only the labeled data look a bit weird when considering the unlabeled data, particularly in that it doesn't cleanly separate the cyan cluster from the other two. In addition, it seems like the boundary between the magenta and red clusters is a bit curved in an unrealistic way. We would not expect points that fell around (-18, -7) to actually come from the red class. Training the model in a semi-supervised manner cleaned up both of these concerns by learning better boundaries that are also flatter and more generalizable.
# 
# Let's next compare the training times to see how much slower it is to do semi-supervised learning than it is to do simple supervised learning.

print "Supervised Learning: "
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X_train[y_train != -1], y_train[y_train != -1])')
print
print "Semi-supervised Learning: "
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X_train, y_train)')
print
print "Label Propagation (sklearn): "
get_ipython().run_line_magic('timeit', 'LabelPropagation().fit(X_train, y_train)')


# It is quite a bit slower to do semi-supervised learning than simple supervised learning in this example. This is expected as the simple supervised update for naive Bayes is a trivial MLE across each dimension whereas the semi-supervised case requires EM to converge to complete. However, it is still faster to do semi-supervised learning this setting to learn a naive Bayes classifier than it is to fit the label propagation estimator from sklearn. 
# 
# However, though it is widely used, the naive Bayes classifier is still a fairly simple model. One can construct a more complicated model that does not assume feature independence called a Bayes classifier that can also be trained using semi-supervised learning in pretty much the same manner. You can read more about the Bayes classifier in its tutorial in the tutorial folder. Let's move on to more complicated data and try to fit a mixture model Bayes classifier, comparing the performance between using only labeled data and using all data.
# 
# First let's generate some more complicated, noisier data.

X = numpy.empty(shape=(0, 2))
X = numpy.concatenate((X, numpy.random.normal(4, 1, size=(3000, 2)).dot([[-2, 0.5], [2, 0.5]])))
X = numpy.concatenate((X, numpy.random.normal(3, 1, size=(6500, 2)).dot([[-1, 2], [1, 0.8]])))
X = numpy.concatenate((X, numpy.random.normal(7, 1, size=(8000, 2)).dot([[-0.75, 0.8], [0.9, 1.5]])))
X = numpy.concatenate((X, numpy.random.normal(6, 1, size=(2200, 2)).dot([[-1.5, 1.2], [0.6, 1.2]])))
X = numpy.concatenate((X, numpy.random.normal(8, 1, size=(3500, 2)).dot([[-0.2, 0.8], [0.7, 0.8]])))
X = numpy.concatenate((X, numpy.random.normal(9, 1, size=(6500, 2)).dot([[-0.0, 0.8], [0.5, 1.2]])))
x_min, x_max = X[:,0].min()-2, X[:,0].max()+2
y_min, y_max = X[:,1].min()-2, X[:,1].max()+2

y = numpy.concatenate((numpy.zeros(9500), numpy.ones(10200), numpy.ones(10000)*2))
idxs = numpy.arange(29700)
numpy.random.shuffle(idxs)

X = X[idxs]
y = y[idxs]

X_train, X_test = X[:25000], X[25000:]
y_train, y_test = y[:25000], y[25000:]
y_train[numpy.random.choice(25000, size=24920, replace=False)] = -1

plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color='0.6', s=1)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='c', s=10)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='m', s=10)
plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='r', s=10)
plt.show()


# Now let's take a look at the accuracies that we get when training a model using just the labeled examples versus all of the examples in a semi-supervised manner.

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model_a = BayesClassifier([d1, d2, d3]).fit(X_train[y_train != -1], y_train[y_train != -1])
print "Supervised Learning Accuracy: {}".format((model_a.predict(X_test) == y_test).mean())

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model_b = BayesClassifier([d1, d2, d3])
model_b.fit(X_train, y_train)
print "Semisupervised Learning Accuracy: {}".format((model_b.predict(X_test) == y_test).mean())


# As expected, the semi-supervised method performs better. Let's visualize the landscape in the same manner as before in order to see why this is the case.

xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.1), numpy.arange(y_min, y_max, 0.1))
Z1 = model_a.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(16, 16))
plt.subplot(221)
plt.title("Training Data, Supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z1)

plt.subplot(223)
plt.title("Training Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z2)

plt.subplot(222)
plt.title("Test Data, Supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z1)

plt.subplot(224)
plt.title("Test Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z2)
plt.show()


# Immediately, one would notice that the decision boundaries when using semi-supervised learning are smoother than those when using only a few samples. This can be explained mostly because having more data can generally lead to smoother decision boundaries as the model does not overfit to spurious examples in the dataset. It appears that the majority of the correctly classified samples come from having a more accurate decision boundary for the magenta samples in the left cluster. When using only the labeled samples many of the magenta samples in this region get classified incorrectly as cyan samples. In contrast, when using all of the data these points are all classified correctly.
# 
# Lastly, let's take a look at a time comparison in this more complicated example.

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model = BayesClassifier([d1, d2, d3])

print "Supervised Learning: "
get_ipython().run_line_magic('timeit', 'model.fit(X_train[y_train != -1], y_train[y_train != -1])')
print

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model = BayesClassifier([d1, d2, d3])

print "Semi-supervised Learning: "
get_ipython().run_line_magic('timeit', 'model.fit(X_train, y_train)')

print
print "Label Propagation (sklearn): "
get_ipython().run_line_magic('timeit', 'LabelPropagation().fit(X_train, y_train)')


# It looks like the difference, while still large, is not as large as in the previous example, being only a ~40x difference instead of a ~1000x difference. This is likely because even without the unlabeled data the supervised model is performing EM to train each of the mixtures that are the components of the Bayes classifier. Again, it is faster to do semi-supervised learning in this manner for generative models than it is to perform LabelPropagation.

# ## Summary
# 
# In the real world (ack) there are frequently situations where only a small fraction of the available data has useful labels. Semi-supervised learning provides a framework for leveraging both the labeled and unlabeled aspects of a dataset to learn a sophisticated estimator. In this case, semi-supervised learning plays well with probabilistic models as normal maximum likelihood estimates can be done on the labeled data and expectation-maximization can be run on the unlabeled data using the same distributions.
# 
# This notebook has covered how to implement semi-supervised learning in pomegranate using both naive Bayes and a Bayes classifier. All one has to do is set the labels of unlabeled samples to -1 and pomegranate will take care of the rest. This can be particularly useful when encountering complex, noisy, data in the real world that aren't neat Gaussian blobs.

# ## Missing Value Support
# 
# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com

# The majority of machine learning algorithms assume that they are operating on a fully observed data set. In contast, a great deal of data sets in the real world are missing some values. Sometimes, this missingness is missing at random (MAR), which means that there is no important pattern to the missingness, and sometimes the missingness itself can be interpreted as a feature. For example, in the Titanic data set, males were more likely to have missing records than females were, and those without children were more likely to have missing records. 
# 
# A common approach to bridging this gap is to impute the missing values and then treat the entire data set as observed. For continuous features this is commonly done by replacing the missing values with the mean or median of the column. For categorical variables it is commonly done by replacing the missing values with the most common category observed in that column. While these techniques are simple and allow for almost any ML algorithm to be run, they are frequently suboptimal. Consider the follow simple example of continuous data that is bimodally distributed:

get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pandas
import random
import numpy
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import itertools

from pomegranate import *

random.seed(0)
numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


get_ipython().run_line_magic('pylab', 'inline')
from pomegranate import *
import seaborn
seaborn.set_style('whitegrid')
numpy.random.seed(0)


X = numpy.concatenate([numpy.random.normal(0, 1, size=(1000)), numpy.random.normal(6, 1, size=(1250))])

plt.title("Bimodal Distribution", fontsize=14)
plt.hist(X, bins=numpy.arange(-3, 9, 0.1), alpha=0.6)
plt.ylabel("Count", fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14)
plt.yticks(fontsize=12)
plt.vlines(numpy.mean(X), 0, 80, color='r', label="Mean")
plt.vlines(numpy.median(X), 0, 80, color='b', label="Median")
plt.legend(fontsize=14)
plt.show()


# The data peaks around 0 and around 6. Replacing the missing values with ~3 will be inserting values into the data set that mostly don't exist in the observed values. The median is slightly better, but will still cause the imputed values to be in one of the two clusters. This has the effect of essentially increasing the variance of the appropriate distribution. Let's take a look at what the distribution looks like if we add 500 missing values and then impute them using the mean of the observed values.

X = numpy.concatenate([X, [numpy.nan]*500])
X_imp = X.copy()
X_imp[numpy.isnan(X_imp)] = numpy.mean(X_imp[~numpy.isnan(X_imp)])

plt.title("Bimodal Distribution", fontsize=14)
plt.hist(X_imp, bins=numpy.arange(-3, 9, 0.1), alpha=0.6)
plt.ylabel("Count", fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14)
plt.yticks(fontsize=12)
plt.vlines(numpy.mean(X), 0, 80, color='r', label="Mean")
plt.vlines(numpy.median(X), 0, 80, color='b', label="Median")
plt.legend(fontsize=14)
plt.show()


# It doesn't appear to be that great. We can see the issue with increased variance by trying to fit a Gaussian mixture model to the data with the imputed values, versus fitting it to the data and ignoring missing values.

x = numpy.arange(-3, 9, 0.1)
model1 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X_imp.reshape(X_imp.shape[0], 1))
model2 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X.reshape(X.shape[0], 1))
p1 = model1.probability(x.reshape(x.shape[0], 1))
p2 = model2.probability(x.reshape(x.shape[0], 1))

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.title("Mean Impute Missing Values", fontsize=14)
plt.hist(X_imp, bins=x, alpha=0.6, density=True)
plt.plot(x, p1, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)

plt.subplot(122)
plt.title("Ignore Missing Values", fontsize=14)
plt.hist(X[~numpy.isnan(X)], bins=x, alpha=0.6, density=True)
plt.plot(x, p2, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.show()


# When we impute the missing values, it seems that one component is fit properly and one has drastically increased variance. In contrast, when we ignore the missing values, we fit a model that represents the underlying data much more faithfully.
# 
# At this point, you may think that as long as the data comes from a single distribution it shouldn't matter if you do a mean imputation of the data. However, this has the effect of shrinking the variance inappropriately. Let's take a look quickly at data drawn from a single Gaussian.

X = numpy.concatenate([numpy.random.normal(0, 1, size=(750)), [numpy.nan]*250])
X_imp = X.copy()
X_imp[numpy.isnan(X_imp)] = numpy.mean(X_imp[~numpy.isnan(X_imp)])

x = numpy.arange(-3, 3, 0.1)
d1 = NormalDistribution.from_samples(X_imp)
d2 = NormalDistribution.from_samples(X)
p1 = d1.probability(x.reshape(x.shape[0], 1))
p2 = d2.probability(x.reshape(x.shape[0], 1))

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.title("Mean Impute Missing Values", fontsize=14)
plt.hist(X_imp, bins=x, alpha=0.6, density=True, label="$\sigma$ = {:4.4}".format(d1.parameters[1]))
plt.plot(x, p1, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.legend(fontsize=14)

plt.subplot(122)
plt.title("Ignore Missing Values", fontsize=14)
plt.hist(X[~numpy.isnan(X)], bins=x, alpha=0.6, density=True, label="$\sigma$ = {:4.4}".format(d2.parameters[1]))
plt.plot(x, p2, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.show()


# Even when the data is all drawn from a single, Gaussian, distribution, it is not a great idea to do mean imputation. We can see that the standard deviation of the learned distribution is significantly smaller than the true standard deviation (of 1), whereas if the missing data is ignored the value is closer.

# This might all be intuitive for a single variable. However, the concept of only collecting sufficient statistics from values that are present in the data and ignoring the missing values can be used in much more complicated, and/or multivariate models. Let's take a look at how well one can estimate the covariance matrix of a multivariate Gaussian distribution using these two strategies. 

n, d, steps = 1000, 10, 50
diffs1 = numpy.zeros(int(steps*0.86))
diffs2 = numpy.zeros(int(steps*0.86))

X = numpy.random.normal(6, 3, size=(n, d))

for k, size in enumerate(range(0, int(n*d*0.86), n*d / steps)):
    idxs = numpy.random.choice(numpy.arange(n*d), replace=False, size=size)
    i, j = idxs / d, idxs % d

    cov_true = numpy.cov(X, rowvar=False, bias=True)
    X_nan = X.copy()
    X_nan[i, j] = numpy.nan

    X_mean = X_nan.copy()
    for col in range(d):
        mask = numpy.isnan(X_mean[:,col])
        X_mean[mask, col] = X_mean[~mask, col].mean()

    diff = numpy.abs(numpy.cov(X_mean, rowvar=False, bias=True) - cov_true).sum()
    diffs1[k] = diff

    dist = MultivariateGaussianDistribution.from_samples(X_nan)
    diff = numpy.abs(numpy.array(dist.parameters[1]) - cov_true).sum()
    diffs2[k] = diff


plt.title("Error in Multivariate Gaussian Covariance Matrix", fontsize=16)
plt.plot(diffs1, label="Mean")
plt.plot(diffs2, label="Ignore")

plt.xlabel("Percentage Missing", fontsize=14)
plt.ylabel("L1 Errors", fontsize=14)
plt.xticks(range(0, 51, 10), numpy.arange(0, 5001, 1000) / 5000.)
plt.xlim(0, 50)
plt.legend(fontsize=14)
plt.show()


# In even the simplest case of Gaussian distributed data with a diagonal covariance matrix, it is more accurate to use the ignoring strategy rather than imputing the mean. When the data set is mostly unobserved the mean imputation strategy tends to do better in this case, but only because there is so little data for the ignoring strategy to actually train on. The deflation of the variance benefits the mean imputation strategy because all of the off-diagonal elements should be 0, but are likely to be artificially high when there are only few examples of the pairs of the variables co-existing in the dataset. This weakness in the ignoring strategy also makes it more likely to encounter linear algebra errors, such as a non-invertable covariance matrix.

# This long introduction is a way of saying that pomegranate uses a strategy of ignoring missing values instead of attempting to impute them, followed by fitting to the newly complete data set. There are other imputation strategies, such as those based on EM, that would be a natural fit with the types of probabilistic models implemented in pomegranate. While those have not yet been added, they would be a good addition that I hope to get to this year.
# 
# Let's now take a look at how to use missing values in some pomegranate models!

# ### 1. Distributions
# 
# We've seen some examples of fitting distributions to missing data. For univariate distributions, the missing values are simply ignored when fitting to the data.

X = numpy.random.randn(100)
X_nan = numpy.concatenate([X, [numpy.nan]*100])

print "Fitting only to observed values:"
print NormalDistribution.from_samples(X)
print 
print "Fitting to observed and missing values:"
print NormalDistribution.from_samples(X_nan)


# This may seem to be an obvious thing to do. However, it suggests a way for dealing with multivariate data being modeled with an IndependentComponentsDistribution when some of the features are missing. Specifically, treat each column independently, and update based on the observed values, regardless of if there is an unobserved value in the same sample but another column. For example:

X = numpy.random.normal(0, 1, size=(500, 3))
idxs = numpy.random.choice(1500, replace=False, size=500)
i, j = idxs // 3, idxs % 3
X[i, j] = numpy.nan

d = IndependentComponentsDistribution.from_samples(X, distributions=[NormalDistribution]*3)
d


# Easy. As we saw above, we can do the same to learn a multivariate Gaussian distribution in the presence of missing data. Again, we don't need to change anything about how we interact with the data, and there are no flags to toggle.
# 
# The last aspect is that the probability of missing data under any univariate distribution is 1, for the purposes of downstream algorithms.

NormalDistribution(1, 2).probability(numpy.nan)


# In an IndependentComponentsDistribution, this just means that when multiplying together the probabilities of each feature to get the total probability, that some dimensions don't factor into the calculation.

d.probability((numpy.nan, 2, 3))


d.distributions[1].probability(2) * d.distributions[2].probability(3)


# ### 2. K-Means Clustering
# 
# K-means clustering mostly serves a helper role in initializing mixture models and hidden Markov models. However, it can still be used by itself if desired. In addition to having the same parallelization and out-of-core capabilities of the main models, it also supports missing values now.

X = numpy.concatenate([numpy.random.normal(0, 1, size=(50, 2)), numpy.random.normal(3, 1, size=(75, 2))])
X_nan = X.copy()

idxs = numpy.random.choice(250, replace=False, size=50)
i, j = idxs // 2, idxs % 2
X_nan[i, j] = numpy.nan


# Just like the other models, you don't need to change the method calls in order to handle missing data. You can fit a K-means model to data sets with missing values and make predictions on samples with missing values in the same way you would without missing values. The prediction step will assign samples to the nearest centroid in the dimensions that are observed, ignoring the missing values.

model1 = Kmeans.from_samples(2, X)
model2 = Kmeans.from_samples(2, X_nan)

y1 = model1.predict(X)
y2 = model2.predict(X_nan)

plt.figure(figsize=(14, 6))
plt.subplot(121)
plt.title("Fit w/o Missing Values", fontsize=16)
plt.scatter(X[y1 == 0,0], X[y1 == 0,1], color='b')
plt.scatter(X[y1 == 1,0], X[y1 == 1,1], color='r')

plt.subplot(122)
plt.title("Fit w/ Missing Values", fontsize=16)
plt.scatter(X[y2 == 0,0], X[y2 == 0,1], color='b')
plt.scatter(X[y2 == 1,0], X[y2 == 1,1], color='r')
plt.show()


# We can see that there are some blue points in the red cluster on the right plot because those samples are entirely NaN. Any sample that is entirely NaN is assigned to cluster 0. Otherwise, it's still able to identify the two clusters even there there are many missing values.

# ### 3. General Mixture Models
# 
# Missing value support for mixture models follows that of k-means fairly closely. Essentially, one passes in a data set containing missing values denoted as `numpy.nan` and they get used appropriately for the fit and predict steps. All methods automatically handle missing values appropriately without any additional flags or methods. 
# 
# Since training is an iterative procedure that involves calculating the probabilities of samples given each component, multivariate Gaussian mixtures will be much slower when handling missing values than they would be when using only observed values. This is because an inverse covariance matrix has to be calculated by subsetting the covariance matrix and inverting it based only on the observed dimensions for each sample. Each sample, then, needs its own matrix inversion. Since there is no single inverse covariance matrix, one also cannot use BLAS or a GPU to accelerate this step. 

X = numpy.concatenate([numpy.random.normal(0, 1, size=(1000, 10)), numpy.random.normal(2, 1, size=(1250, 10))])

idxs = numpy.random.choice(22500, replace=False, size=5000)
i, j = idxs // 10, idxs % 10

X_nan = X.copy()
X_nan[i, j] = numpy.nan

get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X, max_iterations=10)')
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_nan, max_iterations=10)')


# However, if one was modeling each dimension independently, there should be no hit at all!

get_ipython().run_line_magic('timeit', '-n 100 GeneralMixtureModel.from_samples([NormalDistribution]*2, 2, X, max_iterations=10)')
get_ipython().run_line_magic('timeit', '-n 100 GeneralMixtureModel.from_samples([NormalDistribution]*2, 2, X_nan, max_iterations=10)')


# ### 4. Naive Bayes / Bayes Classifiers

# Support for these models mirrors what's been seen before. However, since the fitting step does not involve calculating probabilities of samples, it should be no slower to train them on data sets involving missing values than to train them on dense data sets.

y = numpy.concatenate([numpy.zeros(1000), numpy.ones(1250)])

get_ipython().run_line_magic('timeit', '-n 100 BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)')
get_ipython().run_line_magic('timeit', '-n 100 BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y)')


# Since pomegranate also has semi-supervised learning built-in, this means that one can now fit Bayes classifiers on data sets with missingness in both the labels and in the values! Since semi-supervised learning does rely on EM, it will be slower to train multivariate Gaussian models with missing values than not to.

idx = numpy.random.choice(2250, replace=False, size=750)
y_nan = y.copy()
y_nan[idx] = -1

model = BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y_nan, verbose=True)


get_ipython().run_line_magic('timeit', 'BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y_nan)')
get_ipython().run_line_magic('timeit', 'BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y_nan)')


# ### 5. Hidden Markov Models
# 
# Hidden Markov models are slightly different from the others, in that they operate over sequences. This adds another level of complication to the model because the forward and backward algorithms are needed in order to identify the best component for each observation. Typically this involves calculating the probability of each observation given each state, and taking the sum of all paths through the model, multiplying the transition probability of each edge crossed by the emission probability of the state you transition to emitting the next character. If this is a univariate model and the character is missing, you ignore the emission probability and just multiply by the transition probability. This is easily done by having the probability of missing values be 1 under all univariate models.

d1 = DiscreteDistribution({'A': 0.25, 'B': 0.75})
d2 = DiscreteDistribution({'A': 0.67, 'B': 0.33})

s1 = State(d1, name="s1")
s2 = State(d2, name="s2")

model = HiddenMarkovModel()
model.add_states(s1, s2)
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.5)
model.add_transition(s1, s2, 0.5)
model.add_transition(s2, s2, 0.5)
model.add_transition(s2, s1, 0.5)
model.bake()


# Now let's run the forward algorithm on a simple sequence.

numpy.exp(model.forward(['A', 'B', 'A', 'A']))


# Let's see what happens when we remove one of the characters.

numpy.exp(model.forward(['A', 'nan', 'A', 'A']))


# We can see that initially the first character is aligned to s1 because there is a 100% chance of going from the start state to s1. The value is 0.25 because it is equal to the transition probability (1.0) multiplied by the emission probability (0.25). In the next step, you can see that the probability is equally diffused between two options, staying in the current state (transition probability of 0.5) and moving to s2 (also transition probability of 0.5). Since the character is missing, there is no emission probability to multiply by. 
# 
# If we want to decode the sequence we can call the same methods as before.

model.predict(['A', 'A', 'B', 'B', 'A', 'A'])


model.predict(['A', 'nan', 'B', 'B', 'nan', 'A'])


# Fitting is pretty much the same story as the previous models. Like the Bayes classifiers, one can now train a hidden Markov model in a supervised manner, having some observations in the sequence missing, but also labels on each observation. Labeled missing data can still be used to train the transition parameters.

# ### 6. Bayesian Networks
# 
# Bayesian networks could previously handle missing data in the `predict` and `predict_proba` methods. In fact, handling missing data was the area in which they excelled. However, one couldn't calculate the probability of a sample containing missing values, fit a Bayesian network to incomplete data sets, or learn the structure of a network on incomplete data sets. The new missing value support allows all of these things (except Chow-Liu tree building on missing data sets)! To be clear, there is a very common iterative technique for fitting/learning models on incomplete data sets using an iterative EM based approach. This does not do that. This only ignores sufficient statistics from missing samples, and so is not an iterative approach.
# 
# Learning the structure of a network on an incomplete data set should take a similar amount of time as learning it on a complete data set. If you are indicating missing values in numeric data sets, you will have to convert your data set to floats, whereas previously integers would be fine. If your data set is complete, there is no need.

X = numpy.random.randint(3, size=(500, 10)).astype('float64')

idxs = numpy.random.choice(5000, replace=False, size=2000)
i, j = idxs // 10, idxs % 10
X_nan = X.copy()
X_nan[i, j] = numpy.nan

get_ipython().run_line_magic('timeit', "-n 100 BayesianNetwork.from_samples(X, algorithm='exact')")
get_ipython().run_line_magic('timeit', "-n 100 BayesianNetwork.from_samples(X_nan, algorithm='exact')")


# ## Conclusions
# 
# Missing value support has been added to pomegranate as of v0.9.0! A lot of care was taken to make the interface as simple to the end user while not compromising on speed. While multivariate Gaussian distributions, and the compositional models that are built on top of them, may be slower on incomplete data sets than on complete ones, everything else should take a similar amount of time!
# 
# The implemented missing value support focuses on ignoring data that is missing. Another approach that works well is to use an EM based algorithm to impute the missing values based on the observed values and the model and fit to that complete data set. This works well in the framework of probabilistic modeling and is a natural addition to pomegranate that I hope to add in the coming year.
# 
# As always, feedback and questions are always welcome!

from pomegranate import *
import seaborn
get_ipython().run_line_magic('pylab', 'inline')
seaborn.set_style('whitegrid')
numpy.set_printoptions(suppress=True)


# # Naive Bayes and Bayes Classifiers: A Tutorial
# 
# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com
# 
# Bayes classifiers are some of the simplest machine learning models that exist, due to their intuitive probabilistic interpretation and simple fitting step. Each class is modeled as a probability distribution, and the data is interpreted as samples drawn from these underlying distributions. Fitting the model to data is as simple as calculating maximum likelihood parameters for the data that falls under each class, and making predictions is as simple as using Bayes' rule to determine which class is most likely given the distributions. Bayes' Rule is the following: 
# 
# \begin{equation}
# P(M|D) = \frac{P(D|M)P(M)}{P(D)}
# \end{equation}
# 
# where M stands for the model and D stands for the data. $P(M)$ is known as the <i>prior</i>, because it is the probability that a sample is of a certain class before you even know what the sample is. This is generally just the frequency of each class. Intuitively, it makes sense that you would want to model this, because if one class occurs 10x more than another class, it is more likely that a given sample will belong to that distribution. $P(D|M)$ is the likelihood, or the probability, of the data under a given model. Lastly, $P(M|D)$ is the posterior, which is the probability of each component of the model, or class, being the component which generated the data. It is called the posterior because the prior corresponds to probabilities before seeing data, and the posterior corresponds to probabilities after observing the data. In cases where the prior is uniform, the posterior is just equal to the normalized likelihoods. This equation forms the basis of most probabilistic modeling, with interesting priors allowing the user to inject sophisticated expert knowledge into the problem directly.
# 
# Let's take a look at some single dimensional data in order to introduce these concepts more thoroughly.

X = numpy.concatenate((numpy.random.normal(3, 1, 200), numpy.random.normal(10, 2, 1000)))
y = numpy.concatenate((numpy.zeros(200), numpy.ones(1000)))

x1 = X[:200]
x2 = X[200:]

plt.figure(figsize=(16, 5))
plt.hist(x1, bins=25, color='m', edgecolor='m', label="Class A")
plt.hist(x2, bins=25, color='c', edgecolor='c', label="Class B")
plt.xlabel("Value", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# The data seems like it comes from two normal distributions, with the cyan class being more prevalent than the magenta class. A natural way to model this data would be to create a normal distribution for the cyan data, and another for the magenta distribution.
# 
# Let's take a look at doing that. All we need to do is use the `from_samples` class method of the `NormalDistribution` class.

d1 = NormalDistribution.from_samples(x1)
d2 = NormalDistribution.from_samples(x2)
idxs = numpy.arange(0, 15, 0.1)

p1 = map(d1.probability, idxs)
p2 = map(d2.probability, idxs)

plt.figure(figsize=(16, 5))
plt.plot(idxs, p1, color='m'); plt.fill_between(idxs, 0, p1, facecolor='m', alpha=0.2)
plt.plot(idxs, p2, color='c'); plt.fill_between(idxs, 0, p2, facecolor='c', alpha=0.2)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# It looks like some aspects of the data are captured well by doing things this way-- specifically the mean and variance of the normal distributions. This allows us to easily calculate $P(D|M)$ as the probability of a sample under either the cyan or magenta distributions using the normal (or Gaussian) probability density equation:
# 
# \begin{align}
# P(D|M) &= P(x|\mu, \sigma) \\
#        &= \frac{1}{\sqrt{2\pi\sigma^{2}}} exp \left(-\frac{(x-u)^{2}}{2\sigma^{2}} \right) 
# \end{align}
# 
# However, if we look at the original data, we see that the cyan distributions is both much wider than the purple distribution and much taller, as there were more samples from that class in general. If we reduce that data down to these two distributions, we lose the class imbalance. We want our prior to model this class imbalance, with the reasoning being that if we randomly draw a sample from the samples observed thus far, it is far more likely to be a cyan than a magenta sample. Let's take a look at this class imbalance exactly.

magenta_prior = 1. * len(x1) / len(X)
cyan_prior = 1. * len(x2) / len(X)

plt.figure(figsize=(4, 6))
plt.title("Prior Probabilities P(M)", fontsize=14)
plt.bar(0, magenta_prior, facecolor='m', edgecolor='m')
plt.bar(1, cyan_prior, facecolor='c', edgecolor='c')
plt.xticks([0, 1], ['P(Magenta)', 'P(Cyan)'], fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# The prior $P(M)$ is a vector of probabilities over the classes that the model can predict, also known as components. In this case, if we draw a sample randomly from the data that we have, there is a ~83% chance that it will come from the cyan class and a ~17% chance that it will come from the magenta class.
# 
# Let's multiply the probability densities we got before by this imbalance.

d1 = NormalDistribution.from_samples(x1)
d2 = NormalDistribution.from_samples(x2)
idxs = numpy.arange(0, 15, 0.1)

p_magenta = numpy.array(map(d1.probability, idxs)) * magenta_prior
p_cyan = numpy.array(map(d2.probability, idxs)) * cyan_prior

plt.figure(figsize=(16, 5))
plt.plot(idxs, p_magenta, color='m'); plt.fill_between(idxs, 0, p_magenta, facecolor='m', alpha=0.2)
plt.plot(idxs, p_cyan, color='c'); plt.fill_between(idxs, 0, p_cyan, facecolor='c', alpha=0.2)
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M)P(D|M)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# This looks a lot more faithful to the original data, and actually corresponds to $P(M)P(D|M)$, the prior multiplied by the likelihood. However, these aren't actually probability distributions anymore, as they no longer integrate to 1. This is why the $P(M)P(D|M)$ term has to be normalized by the $P(D)$ term in Bayes' rule in order to get a probability distribution over the components. However, $P(D)$ is difficult to determine exactly-- what is the probability of the data? Well, we can sum over the classes to get that value, since $P(D) = \sum_{i=1}^{c} P(D|M)P(M)$ for a problem with c classes. This translates into $P(D) = P(M=Cyan)P(D|M=Cyan) + P(M=Magenta)P(D|M=Magenta)$ for this specific problem, and those values can just be pulled from the unnormalized plots above.
# 
# This gives us the full Bayes' rule, with the posterior $P(M|D)$ being the proportion of density of the above plot coming from each of the two distributions at any point on the line. Let's take a look at the posterior probabilities of the two classes on the same line.

magenta_posterior = p_magenta / (p_magenta + p_cyan)
cyan_posterior = p_cyan / (p_magenta + p_cyan)

plt.figure(figsize=(16, 5))
plt.subplot(211)
plt.plot(idxs, p_magenta, color='m'); plt.fill_between(idxs, 0, p_magenta, facecolor='m', alpha=0.2)
plt.plot(idxs, p_cyan, color='c'); plt.fill_between(idxs, 0, p_cyan, facecolor='c', alpha=0.2)
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M)P(D|M)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(212)
plt.plot(idxs, magenta_posterior, color='m')
plt.plot(idxs, cyan_posterior, color='c')
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M|D)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# The top plot shows the same densities as before, while the bottom plot shows the proportion of the density belonging to either class at that point. This proportion is known as the posterior $P(M|D)$, and can be interpreted as the probability of that point belonging to each class. This is one of the native benefits of probabilistic models, that instead of providing a hard class label for each sample, they can provide a soft label in the form of the probability of belonging to each class.
# 
# We can implement all of this simply in pomegranate using the `NaiveBayes` class.

idxs = idxs.reshape(idxs.shape[0], 1)
X = X.reshape(X.shape[0], 1)

model = NaiveBayes.from_samples(NormalDistribution, X, y)
posteriors = model.predict_proba(idxs)

plt.figure(figsize=(14, 4))
plt.plot(idxs, posteriors[:,0], color='m')
plt.plot(idxs, posteriors[:,1], color='c')
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M|D)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Looks like we're getting the same plots for the posteriors just through fitting the naive Bayes model directly to data. The predictions made will come directly from the posteriors in this plot, with cyan predictions happening whenever the cyan posterior is greater than the magenta posterior, and vice-versa.

# ## Naive Bayes
# 
# In the univariate setting, naive Bayes is identical to a general Bayes classifier. The divergence occurs in the multivariate setting, the naive Bayes model assumes independence of all features, while a Bayes classifier is more general and can support more complicated interactions or covariances between features. Let's take a look at what this means in terms of Bayes' rule. 
# 
# \begin{align}
# P(M|D) &= \frac{P(M)P(D|M)}{P(D)} \\
#        &= \frac{P(M)\prod_{i=1}^{d}P(D_{i}|M_{i})}{P(D)}
# \end{align}
# 
# This looks fairly simple to compute, as we just need to pass each dimension into the appropriate distribution and then multiply the returned probabilities together. This simplicity is one of the reasons why naive Bayes is so widely used. Let's look closer at using this in pomegranate, starting off by generating two blobs of data that overlap a bit and inspecting them.

X = numpy.concatenate([numpy.random.normal(3, 2, size=(150, 2)), numpy.random.normal(7, 1, size=(250, 2))])
y = numpy.concatenate([numpy.zeros(150), numpy.ones(250)])

plt.figure(figsize=(8, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m')
plt.xlim(-2, 10)
plt.ylim(-4, 12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Now, let's fit our naive Bayes model to this data using pomegranate. We can use the `from_samples` class method, pass in the distribution that we want to model each dimension, and then the data. We choose to use `NormalDistribution` in this particular case, but any supported distribution would work equally well, such as `BernoulliDistribution` or `ExponentialDistribution`. To ensure we get the correct decision boundary, let's also plot the boundary recovered by sklearn.

from sklearn.naive_bayes import GaussianNB

model = NaiveBayes.from_samples(NormalDistribution, X, y)
clf = GaussianNB().fit(X, y)

xx, yy = np.meshgrid(np.arange(-2, 10, 0.02), np.arange(-4, 12, 0.02))
Z1 = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("pomegranate naive Bayes", fontsize=16)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m')
plt.contour(xx, yy, Z1)
plt.xlim(-2, 10)
plt.ylim(-4, 12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(122)
plt.title("sklearn naive Bayes", fontsize=16)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m')
plt.contour(xx, yy, Z2)
plt.xlim(-2, 10)
plt.ylim(-4, 12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Drawing the decision boundary helps to verify that we've produced a good result by cleanly splitting the two blobs from each other.
# 
# Bayes' rule provides a great deal of flexibility in terms of what the actually likelihood functions are. For example, when considering a multivariate distribution, there is no need for each dimension to be modeled by the same distribution. In fact, each dimension can be modeled by a different distribution, as long as we can multiply the $P(D|M)$ terms together. 
# 
# Let's consider the example of some noisy signals that have been segmented. We know that they come from two underlying phenomena, the cyan phenomena and the magenta phenomena, and want to classify future segments. To do this, we have three features-- the mean signal of the segment, the standard deviation, and the duration.  

def plot_signal(X, n):
    plt.figure(figsize=(16, 6))
    t_current = 0
    for i in range(n):
        mu, std, t = X[i]
        chunk = numpy.random.normal(mu, std, int(t))
        plt.plot(numpy.arange(t_current, t_current+t), chunk, c='cm'[i % 2])
        t_current += t
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Signal", fontsize=14)
    plt.ylim(20, 40)
    plt.show()

def create_signal(n):
    X, y = [], []
    for i in range(n):
        mu = numpy.random.normal(30.0, 0.4)
        std = numpy.random.lognormal(-0.1, 0.4)
        t = int(numpy.random.exponential(50)) + 1
        X.append([mu, std, int(t)])
        y.append(0)

        mu = numpy.random.normal(30.5, 0.8)
        std = numpy.random.lognormal(-0.3, 0.6)
        t = int(numpy.random.exponential(200)) + 1
        X.append([mu, std, int(t)])
        y.append(1)
    
    return numpy.array(X), numpy.array(y)

X_train, y_train = create_signal(1000)
X_test, y_test = create_signal(250)
plot_signal(X_train, 20)


# We can start by modeling each variable as Gaussians, like before, and see what accuracy we get.

model = NaiveBayes.from_samples(NormalDistribution, X_train, y_train)
print "Gaussian Naive Bayes: ", (model.predict(X_test) == y_test).mean()

clf = GaussianNB().fit(X_train, y_train)
print "sklearn Gaussian Naive Bayes: ", (clf.predict(X_test) == y_test).mean()


# We get identical values for sklearn and for pomegranate, which is good. However, let's take a look at the data itself to see whether a Gaussian distribution is the appropriate distribution for the data.

plt.figure(figsize=(14, 4))
plt.subplot(131)
plt.title("Mean")
plt.hist(X_train[y_train == 0, 0], color='c', alpha=0.5, bins=25)
plt.hist(X_train[y_train == 1, 0], color='m', alpha=0.5, bins=25)

plt.subplot(132)
plt.title("Standard Deviation")
plt.hist(X_train[y_train == 0, 1], color='c', alpha=0.5, bins=25)
plt.hist(X_train[y_train == 1, 1], color='m', alpha=0.5, bins=25)

plt.subplot(133)
plt.title("Duration")
plt.hist(X_train[y_train == 0, 2], color='c', alpha=0.5, bins=25)
plt.hist(X_train[y_train == 1, 2], color='m', alpha=0.5, bins=25)
plt.show()


# So, unsurprisingly (since you can see that I used non-Gaussian distributions to generate the data originally), it looks like only the mean follows a normal distribution, whereas the standard deviation seems to follow either a gamma or a log-normal distribution. We can take advantage of that by explicitly using these distributions instead of approximating them as normal distributions. pomegranate is flexible enough to allow for this, whereas sklearn currently is not.

model = NaiveBayes.from_samples(NormalDistribution, X_train, y_train)
print "Gaussian Naive Bayes: ", (model.predict(X_test) == y_test).mean()

clf = GaussianNB().fit(X_train, y_train)
print "sklearn Gaussian Naive Bayes: ", (clf.predict(X_test) == y_test).mean()

model = NaiveBayes.from_samples([NormalDistribution, LogNormalDistribution, ExponentialDistribution], X_train, y_train)
print "Heterogeneous Naive Bayes: ", (model.predict(X_test) == y_test).mean()


# It looks like we're able to get a small improvement in accuracy just by using appropriate distributions for the features, without any type of data transformation or filtering. This certainly seems worthwhile if you can determine what the appropriate underlying distribution is.
# 
# Next, there's obviously the issue of speed. Let's compare the speed of the pomegranate implementation and the sklearn implementation.

get_ipython().run_line_magic('timeit', 'GaussianNB().fit(X_train, y_train)')
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X_train, y_train)')
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples([NormalDistribution, LogNormalDistribution, ExponentialDistribution], X_train, y_train)')


# Looks as if on this small dataset they're all taking approximately the same time. This is pretty much expected, as the fitting step is fairly simple and both implementations use C-level numerics for the calculations. We can give a more thorough treatment of the speed comparison on larger datasets. Let's look at the average time it takes to fit a model to data of increasing dimensionality across 25 runs.

pom_time, skl_time = [], []

n1, n2 = 15000, 60000,
for d in range(1, 101, 5): 
    X = numpy.concatenate([numpy.random.normal(3, 2, size=(n1, d)), numpy.random.normal(7, 1, size=(n2, d))])
    y = numpy.concatenate([numpy.zeros(n1), numpy.ones(n2)])

    tic = time.time()
    for i in range(25):
        GaussianNB().fit(X, y)
    skl_time.append((time.time() - tic) / 25)
    
    tic = time.time()
    for i in range(25):
        NaiveBayes.from_samples(NormalDistribution, X, y)
    pom_time.append((time.time() - tic) / 25)


plt.figure(figsize=(14, 6))
plt.plot(range(1, 101, 5), pom_time, color='c', label="pomegranate")
plt.plot(range(1, 101, 5), skl_time, color='m', label="sklearn")
plt.xticks(fontsize=14)
plt.xlabel("Number of Dimensions", fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Time (s)")
plt.legend(fontsize=14)
plt.show()


# It appears as if the two implementations are basically the same speed. This is unsurprising given the simplicity of the calculations, and as mentioned before, the low level implementation.

# ## Bayes Classifiers
# 
# The natural generalization of the naive Bayes classifier is to allow any multivariate function take the place of $P(D|M)$ instead of it being the product of several univariate probability distributions. One immediate difference is that now instead of creating a Gaussian model with effectively a diagonal covariance matrix, you can now create one with a full covariance matrix. Let's see an example of that at work.

tilt_a = [[-2, 0.5], [5, 2]]
tilt_b = [[-1, 1.5], [3, 3]]

X = numpy.concatenate((numpy.random.normal(4, 1, size=(250, 2)).dot(tilt_a), numpy.random.normal(3, 1, size=(800, 2)).dot(tilt_b)))
y = numpy.concatenate((numpy.zeros(250), numpy.ones(800)))

model_a = NaiveBayes.from_samples(NormalDistribution, X, y)
model_b = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)

xx, yy = np.meshgrid(np.arange(-5, 30, 0.02), np.arange(0, 25, 0.02))
Z1 = model_a.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(18, 8))
plt.subplot(121)
plt.contour(xx, yy, Z1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.xlim(-5, 30)
plt.ylim(0, 25)

plt.subplot(122)
plt.contour(xx, yy, Z2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.xlim(-5, 30)
plt.ylim(0, 25)
plt.show()


# It looks like we are able to get a better boundary between the two blobs of data. The primary for this is because the data don't form spherical clusters, like you assume when you force a diagonal covariance matrix, but are tilted ellipsoids, that can be better modeled by a full covariance matrix. We can quantify this quickly by looking at performance on the training data.

print "naive training accuracy: {:4.4}".format((model_a.predict(X) == y).mean())
print "bayes classifier training accuracy: {:4.4}".format((model_b.predict(X) == y).mean())


# Looks like there is a significant boost. Naturally you'd want to evaluate the performance of the model on separate validation data, but for the purposes of demonstrating the effect of a full covariance matrix this should be sufficient.
# 
# While using a full covariance matrix is certainly more complicated than using only the diagonal, there is no reason that the $P(D|M)$ has to even be a single simple distribution versus a full probabilistic model. After all, all probabilistic models, including general mixtures, hidden Markov models, and Bayesian networks, can calculate $P(D|M)$. Let's take a look at an example of using a mixture model instead of a single gaussian distribution.

X = numpy.empty(shape=(0, 2))
X = numpy.concatenate((X, numpy.random.normal(4, 1, size=(200, 2)).dot([[-2, 0.5], [2, 0.5]])))
X = numpy.concatenate((X, numpy.random.normal(3, 1, size=(350, 2)).dot([[-1, 2], [1, 0.8]])))
X = numpy.concatenate((X, numpy.random.normal(7, 1, size=(700, 2)).dot([[-0.75, 0.8], [0.9, 1.5]])))
X = numpy.concatenate((X, numpy.random.normal(6, 1, size=(120, 2)).dot([[-1.5, 1.2], [0.6, 1.2]])))
y = numpy.concatenate((numpy.zeros(550), numpy.ones(820)))

model_a = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)

gmm_a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 0])
gmm_b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 1])
model_b = BayesClassifier([gmm_a, gmm_b], weights=numpy.array([1-y.mean(), y.mean()]))

xx, yy = np.meshgrid(np.arange(-10, 10, 0.02), np.arange(0, 25, 0.02))
Z1 = model_a.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
centroids1 = numpy.array([distribution.mu for distribution in model_a.distributions])
centroids2 = numpy.concatenate([[distribution.mu for distribution in component.distributions] for component in model_b.distributions])

plt.figure(figsize=(18, 8))
plt.subplot(121)
plt.contour(xx, yy, Z1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.scatter(centroids1[:,0], centroids1[:,1], color='k', s=100)

plt.subplot(122)
plt.contour(xx, yy, Z2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.scatter(centroids2[:,0], centroids2[:,1], color='k', s=100)
plt.show()


# Frequently in the real world you will end up with data that looks like the ones in the above plots, not neatly falling into any single simple distribution. Using a mixture here allowed us to model the various components separately and get a more sophisticated decision boundary that doesn't seem to be extremely overfit.
# 
# If one wanted to use hidden Markov models to model sequences, all that needs to change is passing in `HiddenMarkovModel` objects instead of `GeneralMixtureModel` ones. Likewise, Bayesian networks are similarly supported and can be passed in just as easily. In fact, the most stacking one could reasonably do is to create a Bayes classifier that distinguishes mixtures of hidden Markov models with mixture emissions (GMM-HMM-GMMs) from each other!

# ## pomegranate: fast and flexible probabilistic modelling
# 
# Author: Jacob Schreiber <br>
# Contact: <jmschreiber91@gmail.com>

get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pandas
import random
import numpy
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import itertools

from pomegranate import *

random.seed(0)
numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


# pomegranate is a probabilistic modeling library for Python that aims to be easy to use, flexible, and fast. It is easy to use because it maintains a consistent and minimal API that mirrors the scikit-learn API when possible. It is flexible because it allows complicated probabilistic models to be built from simpler ones by easily stacking components on top of each other. It is fast due to its Cython backend which allows for fast numerics and multi-threading without cluttering the interface. pomegranate was developed at the University of Washington with the generous support from the following organizations:
# 
# <a href="https://escience.washington.edu/">
# <img src="https://escience.washington.edu/wp-content/uploads/2015/10/eScience_Logo_HR.png" width="50%">
# </a>
# 
# <a href="http://msdse.org/">
# <img src="http://msdse.org/images/msdse.jpg" width="50%">
# </a>
# 
# <a href="https://numfocus.org/">
# <img src="https://numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png" width="30%">
# </a>

# ### The API
# 
# One of the core tenant of pomegranate is that everything is fundamentally a probability distribution and should be treated the same way. Most people are familiar with simple probability distributions such as the normal or the uniform distribution. However, a mixture of many simple distributions is still a probability distribution because the density under it still adds up to 1. A hidden Markov model is still a probability distribution over sequences because the probability of all potential sequences must add up to 1. A Bayesian network is literally a probability distribution that has been factorized along a graphical structure, with a discrete Bayesian network factorizing a very large joint probability table and a linear Gaussian Bayesian network factorizing a multivariate Gaussian distribution that has many 0's in the inverse covariance matrix.
# 
# A benefit of considering all methods as probability distributions is that it simplifies and unifies the API across all of the models. The common API is the following methods. Let's specify some distribution and take a look.

model = NormalDistribution(5, 1)


# ***model.probability(X)*** and ***model.log_probability(X)***
# 
# These methods return the probability of a single sample or a vector of probabilities if provided a vector of samples. This is equal to P(D|M) where D is the data and M is the model.

print(model.probability([4., 6., 7.]))
print(model.log_probability([4., 6., 7.]))


# ***model.sample(n=1)***
# 
# This method will return a random sample from the probability distribution or a vector of random samples if n is set to be greater than 1.

model.sample(n=10)


# ***model.fit(X, weights=None, inertia=0.0, other arguments)***
# 
# This method will fit the distribution to the data. For simple distributions and Bayesian networks this corresponds to weighted maximum likelihood estimates. For other compositional distributions such as hidden Markov models this corresponds to an iterative expectation-maximization allgorithm. Optionally, weights can be provided for each of the samples. Different models will have different arguments that can be specified as well.

X = numpy.random.normal(7, 2, size=(100,))

model.fit(X)
model


# ***model.summary(X, weights=None)***
# 
# This method implements the first part of the fitting process, which is summarizing a batch of data down to its sufficient statistics and storing them. These sufficient statistics are additive and will be updated for each successive batch that's seen. The sufficient statistics can be reset either through a call to `from_summaries` or a call to `clear_summaries`. 

X = numpy.random.normal(8, 1.5, size=(100,))

model.summarize(X)
model


# ***model.from_summaries(inertia=0.0)***
# 
# This method implements the second part of the fitting process, which is using the stored sufficient statistics in order to update the model parameters.

model.from_summaries()
model


# ***model.clear_summaries()***
# 
# This method resets the sufficient statistics stored to the model.

X = numpy.random.normal(3, 0.2, size=(100,))

model.summarize(X)
model.clear_summaries()
model.from_summaries()
model


# ***Model.from_samples(distributions, n_components, X, weights=None)*** or ***Model.from_samples(X, weights=None)***
# 
# This class method will initialize and then fit the parameters of a model to some data. This differs from the `fit` function in that the `fit` function will update the parameters of a pre-defined model whereas the `from_samples` method will initialize a model to data and return the best parameters given the data.

X = numpy.random.normal(6, 1, size=(250, 1))

model = NormalDistribution.from_samples(X)
model


model = GeneralMixtureModel.from_samples(NormalDistribution, 3, X)
model


# ***model.to_json(separators=(',', ' :'), indent=4)***
# 
# This method returns the JSON serialization of the distribution as a string.

print(model.to_json())


# ***Model.from_json(s)***
# 
# This class method returns the deserialization of the JSON string back to the model object. s can either be the JSON string or a filename ending in `.json` to read from.

model = NormalDistribution(5, 2)

model2 = Distribution.from_json(model.to_json())
model2


# Compositional models, i.e., those that are not simple distributions, have three additional methods whose named are inspired by scikit-learn. These methods relate to the posterior probabilities P(M|D) of each of the components of the model given some data.

d1 = ExponentialDistribution(5.0)
d2 = ExponentialDistribution(0.3)

model = GeneralMixtureModel([d1, d2])
model


# ***model.predict(X)***
# 
# This method returns the most likely component for each sample. In the case of a mixture model it returns the component that is most likely, in the case of a hidden Markov model it returns the most likely component for each observation in the sequence.

X = numpy.random.exponential(3, size=(10,1))

model.predict(X)


# ***model.predict_proba(X)***
# 
# This method returns the probability of each component for each sample. It is similar to the `predict` method except that it returns the probabilities instead of simply the most likely component.

model.predict_proba(X)


# ***model.predict_log_proba(X)***
# 
# Like predict_proba except that it returns the log probabilities instead of the probabilities.

model.predict_log_proba(X)


# ### Flexibility
# 
# #### Modeling different features as different distributions
# 
# A second benefit of treating all models as probability distributions is that it greatly increases the flexibility that pomegranate provides. When people build naive Bayes classifiers, they typically will use a Gaussian distribution. However, there's no reason that one has to use a Gaussian distribution, you can drop in any type of distribution that you'd like. For example:

X = numpy.random.normal(5, 1, size=(100, 2))
X[50:] += 1

y = numpy.zeros(100)
y[50:] = 1

model1 = NaiveBayes.from_samples(NormalDistribution, X, y)
model2 = NaiveBayes.from_samples(LogNormalDistribution, X, y)


# It is easy to drop in whatever probability distribution you'd like because it should be easy. Mathematically, the naive Bayes model relies on Bayes' rule, which says:
# 
# \begin{equation}
# P(M|D) = \frac{P(D|M)P(M)}{P(D)}
# \end{align}
# 
# Because the "naive" part of a "naive" Bayes model means that the model treats each feature independently, we can rewrite the $P(D|M)$ aspect as the product of these probabilities over all $d$ features:
# 
# \begin{equation}
# P(M|D) = \frac{P(M) \prod\limits_{i=1}^{d} P(D_{i}|M)}{P(D)}
# \end{equation}
# 
# Now, because each feature is independent, they can be modeled by different probability distributions.

mu = numpy.random.normal(7, 2, size=1000)
std = numpy.random.lognormal(-0.8, 0.8, size=1000)
dur = numpy.random.exponential(50, size=1000)

data = numpy.concatenate([numpy.random.normal(mu_, std_, int(t)) for mu_, std_, t in zip(mu, std, dur)])

plt.figure(figsize=(14, 4))
plt.title("Randomly Generated Signal", fontsize=16)
plt.plot(data)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Signal", fontsize=14)
plt.xlim(0, 3000)
plt.show()


# If someone was trying to model this signal, they could potentially try to segment it and then extract properties of those segments, such as the mean, the variance, and the duration. We can plot what thos would look like:

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("mu", fontsize=14)
plt.hist(mu, bins=numpy.arange(0, 15))

plt.subplot(132)
plt.title("sigma", fontsize=14)
plt.hist(std, bins=numpy.arange(0.00, 1.75, 0.05))

plt.subplot(133)
plt.title("Duration", fontsize=14)
plt.hist(dur, bins=numpy.arange(0, 150, 10))
plt.show()


# We can see that the mean of the segments does have a Gaussian distribution, but that neither the standard deviation or the duration do. It would be suboptimal to assume all features were Gaussian, as a simple Gaussian naive Bayes model would. pomegranate allows you to define a different distribution for each feature, like the following:

X1 = numpy.array([numpy.random.normal(7, 2, size=400),
                  numpy.random.lognormal(-0.8, 0.8, size=400),
                  numpy.random.exponential(50, size=400)]).T

X2 = numpy.array([numpy.random.normal(8, 2, size=600),
                  numpy.random.lognormal(-1.2, 0.6, size=600),
                  numpy.random.exponential(100, size=600)]).T

X = numpy.concatenate([X1, X2])
y = numpy.zeros(1000)
y[400:] = 1

NaiveBayes.from_samples([NormalDistribution, LogNormalDistribution, ExponentialDistribution], X, y)


# Modeling each feature independently is allowed for naive Bayes, mixtures, and hidden Markov models. This is very useful when trying to capture different dynamics in different features.
# 
# #### Stacking models on top of each other
# 
# Next, another feature that emerges when you treat all models as probability distributions is that they can be easily stacked within each other. For example, if a mixture is just a probability distribution, then naturally you should be able to make a mixture Bayes classifier by dropping a `GeneralMixtureModel` into a `BayesClassifier` just as easily as a Gaussian Bayes classifier by dropping a `MultivariateGaussianDistribution` into the `BayesClassifier`.

X = numpy.concatenate([numpy.random.normal((5, 1), 1, size=(200, 2)),
                       numpy.random.normal((6, 4), 1, size=(200, 2)),
                       numpy.random.normal((3, 5), 1, size=(350, 2)),
                       numpy.random.normal((7, 6), 1, size=(250, 2))])

y = numpy.zeros(1000)
y[400:] = 1

model = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)
print model.log_probability(X).sum()


d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 0])
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 1])
model2 = BayesClassifier([d1, d2], [0.4, 0.6])
print model2.log_probability(X).sum()


# In this case we build data that intentionally has 4 clusters, of which each class is composed of two of the clusters. A simple normal distribution is unlikely to capture this well, but a mixture model is built specifically to model distributions that are composed of multiple parts. We can see that the correspond log probability is much higher when we incorporate the mixture model in. There is an implementation detail that doesn't allow you yet to pass in a compositional model into the `from_samples` method quite yet, but in this case it is fairly simple to break up the data such that we build the model ourselves.

# ### Speed
# 
# pomegranate uses a Cython backend for the computationally intensive aspects of the calculation, both dramatically speeding up calculations and allowing for multithreading to be utilized.
# 
# #### numpy
# 
# pomegranate used to be consistently faster than numpy at fitting probability distributions to data, when compared to performing the same operations in numpy. However, due to consistent improvements in numpy, it is now a more complicated picture. For example, let's look at fitting a normal distribution to 1,000 samples.

X = numpy.random.normal(4, 1, size=1000)

get_ipython().run_line_magic('timeit', '-n 1 -r 1 numpy.mean(X), numpy.std(X)')
get_ipython().run_line_magic('timeit', '-n 1 -r 1 NormalDistribution.from_samples(X)')


# pomegranate appears to be a bit faster, but the total amount of time is negligble one way or another. If we increase the size of the data we're using to 10 million samples:

X = numpy.random.normal(4, 1, size=10000000)

get_ipython().run_line_magic('timeit', 'numpy.mean(X), numpy.std(X)')
get_ipython().run_line_magic('timeit', 'NormalDistribution.from_samples(X)')


# It looks like pomegranate is more comparable to numpy.
# 
# Let's now look at fitting a multivariate Gaussian distribution. The calculation of a covariance matrix requires a dot product, which numpy accelerates using BLAS. pomegranate uses a Cython wrapper of BLAS and so can take advantage of those speed improvements. Let's look at fitting to one million data points with 3 dimensions.

X = numpy.random.normal(4, 1, size=(1000000, 3))

get_ipython().run_line_magic('timeit', 'numpy.mean(X, axis=0), numpy.cov(X, rowvar=False, ddof=0)')
get_ipython().run_line_magic('timeit', 'MultivariateGaussianDistribution.from_samples(X)')


# It looks like pomegranate is around the same speed here. However, if we fit to a 1000 dimensional sample rather than a 3 dimensional one:

X = numpy.random.normal(4, 1, size=(100000, 1000))

get_ipython().run_line_magic('timeit', 'numpy.mean(X, axis=0), numpy.cov(X, rowvar=False, ddof=0)')
get_ipython().run_line_magic('timeit', 'MultivariateGaussianDistribution.from_samples(X)')


# Now it looks like pomegranate is slower than numpy.
# 
# The main take-away when compared to numpy is that pomegranate and numpy appear to perform similarly. As the samples become larger numpy becomes faster than pomegranate, but they're both within an order of magnitude of each other one way or another. If you're trying to do basic operations on a large amount of data, you may want to stick with using numpy.

# #### scipy
# 
# scipy can be used to calculate probabilities of samples given a distribution. This corresponds to the `probability` and `log_probability` functions in pomegranate. Let's see how long it takes to calculate these probabilities in the two packages given a normal distrubution.

from scipy.stats import norm

d = NormalDistribution(0, 1)
x = numpy.random.normal(0, 1, size=(10000000,))

get_ipython().run_line_magic('timeit', 'norm.logpdf(x, 0, 1)')
get_ipython().run_line_magic('timeit', 'NormalDistribution(0, 1).log_probability(x)')

print "\nlogp difference: {}".format((norm.logpdf(x, 0, 1) - NormalDistribution(0, 1).log_probability(x)).sum())


# Looks like it can be significantly faster. Let's also look at a large multivariate normal distribution with 2500 dimensions.

from scipy.stats import multivariate_normal

dim = 2500
n = 1000

mu = numpy.random.normal(6, 1, size=dim)
cov = numpy.eye(dim)

X = numpy.random.normal(8, 1, size=(n, dim))

d = MultivariateGaussianDistribution(mu, cov)

get_ipython().run_line_magic('timeit', 'multivariate_normal.logpdf(X, mu, cov)')
get_ipython().run_line_magic('timeit', 'MultivariateGaussianDistribution(mu, cov).log_probability(X)')
get_ipython().run_line_magic('timeit', 'd.log_probability(X)')

print "\nlogp difference: {}".format((multivariate_normal.logpdf(X, mu, cov) - d.log_probability(X)).sum())


# One of the reasons which pomegranate can be so fast at calculating log probabilities is that it is able to cache parts of the logpdf equation so that it doesn't need to do all of the calculations each time. For example, let's look at the Normal distribution pdf equation:
# 
# \begin{equation}
# P(X|\mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} exp \left( -\frac{(x - \mu)^{2}}{2\sigma^{2}} \right) \\
# \end{equation}
# 
# We can take the log of this to simplify it.
# 
# \begin{equation}
# logP(X|\mu, \sigma) = -\log \left(\sqrt{2\pi}\sigma \right) - \frac{(x-\mu)^{2}}{2\sigma^{2}}
# \end{equation}
# 
# pomegranate speeds up this calculation by caching $-\log(\sqrt{2\pi}\sigma)$ and $2\sigma^{2}$ when the object is created. This means that the equation is simplified to the following:
# 
# \begin{equation}
# logP(X|\mu, \sigma) = \alpha - \frac{(x - \mu)^{2}}{\beta}
# \end{equation}
# 
# We don't need to calculate any logs or exponentials here, just a difference, a multiplication, a division, and a subtraction.

# #### scikit-learn
# 
# scikit-learn and pomegranate overlap when it comes to naive Bayes classifiers and mixture models. In pomegranate, both of these model types can be used with any distribution, allowing mixtures of exponentials or log-normals to be made just as easily as mixtures of Gaussians, whereas in scikit-learn only Gaussians or multinomials are allowed. Let's compare speed of the overlap.

from sklearn.mixture import GaussianMixture

X = numpy.random.normal(8, 1, size=(10000, 100))

get_ipython().run_line_magic('timeit', 'model1 = GaussianMixture(5, max_iter=10).fit(X)')
get_ipython().run_line_magic('timeit', 'model2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 5, X, max_iterations=10)')


# It looks like the two are approximately the same speed on a single thread. Let's look at naive Bayes now.

from sklearn.naive_bayes import GaussianNB

X = numpy.random.normal(8, 1, size=(100000, 500))
X[:50000] += 1

y = numpy.zeros(100000)
y[:50000] = 1

get_ipython().run_line_magic('timeit', 'GaussianNB().fit(X, y)')
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X, y)')


# It looks like scikit-learn is a bit faster when it comes to training a Gaussian naive Bayes model. 
# 
# Much like with the comparisons to numpy, if scikit-learn has something easily implemented, it's likely better to try to use their implementation. The goal of pomegranate is not to step on the feet of these other packages, but rather to provide similarly-fast implementations that extend to other probability distributions. For example, you can train a log normal naive Bayes in the following manner:

model = NaiveBayes.from_samples(LogNormalDistribution, X, y)


# No such functionality yet exists in scikit-learn to train naive Bayes models for arbitrary distributions.

# # Markov Chains
# 
# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com
# 
# Markov Chains are a simple model based on conditional probability, where a sequence is modelled as the product of conditional probabilities. A n-th order Markov chain looks back n emissions to base its conditional probability on. For example, a 3rd order Markov chain models $P(X_{t} | X_{t-1}, X_{t-2}, X_{t-3})$.
# 
# However, a full Markov model needs to model the first observations, and the first n-1 observations. The first observation can't really be modelled well using $P(X_{t} | X_{t-1}, X_{t-2}, X_{t-3})$, but can be modelled by $P(X_{t})$. The second observation has to be modelled by $P(X_{t} | X_{t-1} )$. This means that these distributions have to be passed into the Markov chain as well. 
# 
# We can initialize a Markov chain easily enough by passing in a list of the distributions.

from pomegranate import *
get_ipython().run_line_magic('pylab', 'inline')


d1 = DiscreteDistribution({'A': 0.10, 'C': 0.40, 'G': 0.40, 'T': 0.10})
d2 = ConditionalProbabilityTable([['A', 'A', 0.10],
                                ['A', 'C', 0.50],
                                ['A', 'G', 0.30],
                                ['A', 'T', 0.10],
                                ['C', 'A', 0.10],
                                ['C', 'C', 0.40],
                                ['C', 'T', 0.40],
                                ['C', 'G', 0.10],
                                ['G', 'A', 0.05],
                                ['G', 'C', 0.45],
                                ['G', 'G', 0.45],
                                ['G', 'T', 0.05],
                                ['T', 'A', 0.20],
                                ['T', 'C', 0.30],
                                ['T', 'G', 0.30],
                                ['T', 'T', 0.20]], [d1])

clf = MarkovChain([d1, d2])


# Markov chains have log probability, fit, summarize, and from summaries methods implemented. They do not have classification capabilities by themselves, but when combined with a Naive Bayes classifier can be used to do discrimination between multiple models (see the Naive Bayes tutorial notebook).
# 
# Lets see the log probability of some data.

clf.log_probability( list('CAGCATCAGT') ) 


clf.log_probability( list('C') )


clf.log_probability( list('CACATCACGACTAATGATAAT') )


# We can fit the model to sequences which we pass in, and as expected, get better performance on sequences which we train on. 

clf.fit( map( list, ('CAGCATCAGT', 'C', 'ATATAGAGATAAGCT', 'GCGCAAGT', 'GCATTGC', 'CACATCACGACTAATGATAAT') ) )
print clf.log_probability( list('CAGCATCAGT') ) 
print clf.log_probability( list('C') )
print clf.log_probability( list('CACATCACGACTAATGATAAT') )


print clf.distributions[0] 


print clf.distributions[1]


# # pomegranate and parallelization
# 
# pomegranate supports parallelization through a set of built in functions based off of joblib. All computationally intensive functions in pomegranate are implemented in cython with the global interpreter lock (GIL) released, allowing for multithreading to be used for efficient parallel processing. The following functions can be called for parallelization:
# 
# 1. fit
# 2. summarize
# 3. predict
# 4. predict_proba
# 5. predict_log_proba
# 6. log_probability
# 7. probability
# 
# These functions can all be simply parallelized by passing in `n_jobs=X` to the method calls. This tutorial will demonstrate how to use those calls. First we'll look at a simple multivariate Gaussian mixture model, and compare its performance to sklearn. Then we'll look at a hidden Markov model with Gaussian emissions, and lastly we'll look at a mixture of Gaussian HMMs. These can all utilize the build-in parallelization that pomegranate has.
# 
# Let's dive right in!

get_ipython().run_line_magic('pylab', 'inline')
from sklearn.mixture import GaussianMixture
from pomegranate import *
import seaborn, time
seaborn.set_style('whitegrid')


def create_dataset(n_samples, n_dim, n_classes, alpha=1):
    """Create a random dataset with n_samples in each class."""
    
    X = numpy.concatenate([numpy.random.normal(i*alpha, 1, size=(n_samples, n_dim)) for i in range(n_classes)])
    y = numpy.concatenate([numpy.zeros(n_samples) + i for i in range(n_classes)])
    idx = numpy.arange(X.shape[0])
    numpy.random.shuffle(idx)
    return X[idx], y[idx]


# ## 1. General Mixture Models
# 
# pomegranate has a very efficient implementation of mixture models, particularly Gaussian mixture models. Lets take a look at how fast pomegranate is versus sklearn, and then see how much faster parallelization can get it to be.

n, d, k = 1000000, 5, 3
X, y = create_dataset(n, d, k)

print "sklearn GMM"
get_ipython().run_line_magic('timeit', "GaussianMixture(n_components=k, covariance_type='full', max_iter=15, tol=1e-10).fit(X)")
print 
print "pomegranate GMM"
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=15, stop_threshold=1e-10)')
print
print "pomegranate GMM (4 jobs)"
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, n_jobs=4, max_iterations=15, stop_threshold=1e-10)')


# It looks like on a large dataset not only is pomegranate faster than sklearn at performing 15 iterations of EM on 3 million 5 dimensional datapoints with 3 clusters, but the parallelization is able to help in speeding things up. 
# 
# Lets now take a look at the time it takes to make predictions using GMMs. Lets fit the model to a small amount of data, and then predict a larger amount of data drawn from the same underlying distributions.

d, k = 25, 2
X, y = create_dataset(1000, d, k)
a = GaussianMixture(k, n_init=1, max_iter=25).fit(X)
b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=25)


del X, y
n = 1000000
X, y = create_dataset(n, d, k)

print "sklearn GMM"
get_ipython().run_line_magic('timeit', '-n 1 a.predict_proba(X)')
print
print "pomegranate GMM"
get_ipython().run_line_magic('timeit', '-n 1 b.predict_proba(X)')
print
print "pomegranate GMM (4 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 b.predict_proba(X, n_jobs=4)')


# It looks like pomegranate can be slightly slower than sklearn when using a single processor, but that it can be parallelized to get faster performance. At the same time, predictions at this level happen so quickly (millions per second) that this may not be the most reliable test for parallelization.
# 
# To ensure that we're getting the exact same results just faster, lets subtract the predictions from each other and make sure that the sum is equal to 0.

print (b.predict_proba(X) - b.predict_proba(X, n_jobs=4)).sum()


# Great, no difference between the two.
# 
# Lets now make sure that pomegranate and sklearn are learning basically the same thing. Lets fit both models to some 2 dimensional 2 component data and make sure that they both extract the underlying clusters by plotting them.

d, k = 2, 2
X, y = create_dataset(1000, d, k, alpha=2)
a = GaussianMixture(k, n_init=1, max_iter=25).fit(X)
b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=25)

y1, y2 = a.predict(X), b.predict(X)

plt.figure(figsize=(16,6))
plt.subplot(121)
plt.title("sklearn clusters", fontsize=14)
plt.scatter(X[y1==0, 0], X[y1==0, 1], color='m', edgecolor='m')
plt.scatter(X[y1==1, 0], X[y1==1, 1], color='c', edgecolor='c')

plt.subplot(122)
plt.title("pomegranate clusters", fontsize=14)
plt.scatter(X[y2==0, 0], X[y2==0, 1], color='m', edgecolor='m')
plt.scatter(X[y2==1, 0], X[y2==1, 1], color='c', edgecolor='c')


# It looks like we're getting the same basic results for the two. The two algorithms are initialized a bit differently, and so it can be difficult to directly compare the results between them, but it looks like they're getting roughly the same results.

# ## 3. Multivariate Gaussian HMM
# 
# Now let's move on to training a hidden Markov model with multivariate Gaussian emissions with a diagonal covariance matrix. We'll randomly generate some Gaussian distributed numbers and use pomegranate with either one or four threads to fit our model to the data.

X = numpy.random.randn(1000, 500, 50)

print "pomegranate Gaussian HMM (1 job)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5, n_jobs=2)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5, n_jobs=4)')


# All we had to do was pass in the n_jobs parameter to the fit function in order to get a speed improvement. It looks like we're getting a really good speed improvement, as well! This is mostly because the HMM algorithms perform a lot more operations than the other models, and so spend the vast majority of time with the GIL released. You may not notice as strong speedups when using a MultivariateGaussianDistribution because BLAS uses multithreaded operations already internally, even when only one job is specified.
# 
# Now lets look at the prediction function to make sure the we're getting speedups there as well. You'll have to use a wrapper function to parallelize the predictions for a HMM because it returns an annotated sequence rather than a single value like a classic machine learning model might.

model = HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=2, verbose=False)

print "pomegranate Gaussian HMM (1 job)"
get_ipython().run_line_magic('timeit', 'predict_proba(model, X)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', 'predict_proba(model, X, n_jobs=2)')


# Great, we're getting a really good speedup on that as well! Looks like the parallel processing is more efficient with a bigger, more complex model, than with a simple one. This can make sense, because all inference/training is more complex, and so there is more time with the GIL released compared to with the simpler operations.

# ## 4. Mixture of Hidden Markov Models
# 
# Let's stack another layer onto this model by making it a mixture of these hidden Markov models, instead of a single one. At this point we're sticking a multivariate Gaussian HMM into a mixture and we're going to train this big thing in parallel.

def create_model(mus):
    n = mus.shape[0]
    
    starts = numpy.zeros(n)
    starts[0] = 1.
    
    ends = numpy.zeros(n)
    ends[-1] = 0.5
    
    transition_matrix = numpy.zeros((n, n))
    distributions = []
    
    for i in range(n):
        transition_matrix[i, i] = 0.5
        
        if i < n - 1:
            transition_matrix[i, i+1] = 0.5
    
        distribution = IndependentComponentsDistribution([NormalDistribution(mu, 1) for mu in mus[i]])
        distributions.append(distribution)
    
    model = HiddenMarkovModel.from_matrix(transition_matrix, distributions, starts, ends)
    return model
    

def create_mixture(mus):
    hmms = [create_model(mu) for mu in mus]
    return GeneralMixtureModel(hmms)

n, d = 50, 10
mus = [(numpy.random.randn(d, n)*0.2 + numpy.random.randn(n)*2).T for i in range(2)]


model = create_mixture(mus)
X = numpy.random.randn(400, 150, d)

print "pomegranate Mixture of Gaussian HMMs (1 job)"
get_ipython().run_line_magic('timeit', 'model.fit(X, max_iterations=5)')
print

model = create_mixture(mus)
print "pomegranate Mixture of Gaussian HMMs (2 jobs)"
get_ipython().run_line_magic('timeit', 'model.fit(X, max_iterations=5, n_jobs=2)')


# Looks like we're getting a really nice speed improvement when training this complex model. Let's take a look now at the time it takes to do inference with it.

model = create_mixture(mus)

print "pomegranate Mixture of Gaussian HMMs (1 job)"
get_ipython().run_line_magic('timeit', 'model.predict_proba(X)')
print

model = create_mixture(mus)
print "pomegranate Mixture of Gaussian HMMs (2 jobs)"
get_ipython().run_line_magic('timeit', 'model.predict_proba(X, n_jobs=2)')


# We're getting a good speed improvement here too through parallelization.

# ## Conclusions
# 
# Hopefully you'll find pomegranate useful in your work! Parallelization should allow you to train complex models faster than before. Keep in mind though that there is an overhead to using parallel processing, and so it's possible that on some smaller examples it does not work as well. In general, the bigger the dataset, the closer to a linear speedup you'll get with pomegranate.
# 
# If you have any interesting examples of how you've used pomegranate in your work, I'd love to hear about them. In addition I'd like to hear any feedback you may have on features you'd like to see. Please shoot me an email. Good luck!

# ## Markov Chains
# 
# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com
# 
# Markov Chains are a simple model based on conditional probability, where a sequence is modelled as the product of conditional probabilities. A n-th order Markov chain looks back n emissions to base its conditional probability on. For example, a 3rd order Markov chain models $P(X_{t} | X_{t-1}, X_{t-2}, X_{t-3})$.
# 
# However, a full Markov model needs to model the first observations, and the first n-1 observations. The first observation can't really be modelled well using $P(X_{t} | X_{t-1}, X_{t-2}, X_{t-3})$, but can be modelled by $P(X_{t})$. The second observation has to be modelled by $P(X_{t} | X_{t-1} )$. This means that these distributions have to be passed into the Markov chain as well. 
# 
# We can initialize a Markov chain easily enough by passing in a list of the distributions.

get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pandas
import random
import numpy
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import itertools

from pomegranate import *

random.seed(0)
numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


from pomegranate import *
get_ipython().run_line_magic('pylab', 'inline')


d1 = DiscreteDistribution({'A': 0.10, 'C': 0.40, 'G': 0.40, 'T': 0.10})
d2 = ConditionalProbabilityTable([['A', 'A', 0.10],
                                ['A', 'C', 0.50],
                                ['A', 'G', 0.30],
                                ['A', 'T', 0.10],
                                ['C', 'A', 0.10],
                                ['C', 'C', 0.40],
                                ['C', 'T', 0.40],
                                ['C', 'G', 0.10],
                                ['G', 'A', 0.05],
                                ['G', 'C', 0.45],
                                ['G', 'G', 0.45],
                                ['G', 'T', 0.05],
                                ['T', 'A', 0.20],
                                ['T', 'C', 0.30],
                                ['T', 'G', 0.30],
                                ['T', 'T', 0.20]], [d1])

clf = MarkovChain([d1, d2])


# Markov chains have log probability, fit, summarize, and from summaries methods implemented. They do not have classification capabilities by themselves, but when combined with a Naive Bayes classifier can be used to do discrimination between multiple models (see the Naive Bayes tutorial notebook).
# 
# Lets see the log probability of some data.

clf.log_probability( list('CAGCATCAGT') ) 


clf.log_probability( list('C') )


clf.log_probability( list('CACATCACGACTAATGATAAT') )


# We can fit the model to sequences which we pass in, and as expected, get better performance on sequences which we train on. 

clf.fit( map( list, ('CAGCATCAGT', 'C', 'ATATAGAGATAAGCT', 'GCGCAAGT', 'GCATTGC', 'CACATCACGACTAATGATAAT') ) )
print clf.log_probability( list('CAGCATCAGT') ) 
print clf.log_probability( list('C') )
print clf.log_probability( list('CACATCACGACTAATGATAAT') )


print clf.distributions[0] 


print clf.distributions[1]


# # Bayesian Networks

# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com

# Bayesian networks are a powerful inference tool, in which a set of variables are represented as nodes, and the lack of an edge represents a conditional independence statement between the two variables, and an edge represents a dependence between the two variables. One of the powerful components of a Bayesian network is the ability to infer the values of certain variables, given observed values for another set of variables. These are referred to as the 'hidden' and 'observed' variables respectively, and need not be set at the time the network is created. The same network can have a different set of variables be hidden or observed between two data points. The more values which are observed, the closer the inferred values will be to the truth.
# 
# While Bayesian networks can have extremely complex emission probabilities, usually Gaussian or conditional Gaussian distributions, pomegranate currently supports only discrete Bayesian networks. Bayesian networks are explicitly turned into Factor Graphs when inference is done, wherein the Bayesian network is turned into a bipartite graph with all variables having marginal nodes on one side, and joint tables on the other.
# 
# If you didn't understand that, it's okay! Lets get down to a simple example, the Monty Hall example. The Monty Hall problem arose from the gameshow Let's Make a Deal, where a guest had to choose which one of three doors had a prize behind it. The twist was that after the guest chose, the host, originally Monty Hall, would then open one of the doors the guest did not pick and ask if the guest wanted to switch which door they had picked. Initial inspection may lead you to believe that if there are only two doors left, there is a 50-50 chance of you picking the right one, and so there is no advantage one way or the other. However, it has been proven both through simulations and analytically that there is in fact a 66% chance of getting the prize if the guest switches their door, regardless of the door they initially went with.
# 
# We can reproduce this result using Bayesian networks with three nodes, one for the guest, one for the prize, and one for the door Monty chooses to open. The door the guest initially chooses and the door the prize is behind are completely random processes across the three doors, but the door which Monty opens is dependent on both the door the guest chooses (it cannot be the door the guest chooses), and the door the prize is behind (it cannot be the door with the prize behind it).
# 
# ## Defining a Bayesian Network in pomegranate
# 
# To create the Bayesian network in pomegranate, we first create the distributions which live in each node in the graph. For a discrete (aka categorical) bayesian network we use DiscreteDistribution objects for the root nodes and ConditionalProbabilityTable objects for the inner and leaf nodes. The columns in a ConditionalProbabilityTable correspond to the order in which the parents (the second argument) are specified, and the last column is the value the ConditionalProbabilityTable itself takes. In the case below, the first column corresponds to the value 'guest' takes, then the value 'prize' takes, and then the value that 'monty' takes. 'B', 'C', 'A' refers then to the probability that Monty reveals door 'A' given that the guest has chosen door 'B' and that the prize is actually behind door 'C', or P(Monty='A'|Guest='B', Prize='C').
# 
# Next, we pass these distributions into state objects along with the name for the node and add them to the network. In the future, all matrices of data should have their columns organized in the same order that the states are added to the network. The way the states are added to the network makes no difference to it, and so you should add the states in the same order your data has.
# 
# Next, we need to add edges to the model. These represent which states are parents of which other states. This is currently a bit redundant with the parents step for ConditionalProbabilityTable objects and will be removed soon. For now edges are added from parent -> child by calling `model.add_transition(parent, child)`.
# 
# Lastly, the model must be baked to finalize the internals. Since Bayesian networks use factor graphs for inference, an explicit factor graph is produced from the Bayesian network during the bake step.

from pomegranate import *

# The guests initial door selection is completely random
guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

# The door the prize is behind is also completely random
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )

    # Monty is dependent on both the guest and the prize. 
monty = ConditionalProbabilityTable(
        [[ 'A', 'A', 'A', 0.0 ],
         [ 'A', 'A', 'B', 0.5 ],
         [ 'A', 'A', 'C', 0.5 ],
         [ 'A', 'B', 'A', 0.0 ],
         [ 'A', 'B', 'B', 0.0 ],
         [ 'A', 'B', 'C', 1.0 ],
         [ 'A', 'C', 'A', 0.0 ],
         [ 'A', 'C', 'B', 1.0 ],
         [ 'A', 'C', 'C', 0.0 ],
         [ 'B', 'A', 'A', 0.0 ],
         [ 'B', 'A', 'B', 0.0 ],
         [ 'B', 'A', 'C', 1.0 ],
         [ 'B', 'B', 'A', 0.5 ],
         [ 'B', 'B', 'B', 0.0 ],
         [ 'B', 'B', 'C', 0.5 ],
         [ 'B', 'C', 'A', 1.0 ],
         [ 'B', 'C', 'B', 0.0 ],
         [ 'B', 'C', 'C', 0.0 ],
         [ 'C', 'A', 'A', 0.0 ],
         [ 'C', 'A', 'B', 1.0 ],
         [ 'C', 'A', 'C', 0.0 ],
         [ 'C', 'B', 'A', 1.0 ],
         [ 'C', 'B', 'B', 0.0 ],
         [ 'C', 'B', 'C', 0.0 ],
         [ 'C', 'C', 'A', 0.5 ],
         [ 'C', 'C', 'B', 0.5 ],
         [ 'C', 'C', 'C', 0.0 ]], [guest, prize] )  

# State objects hold both the distribution, and a high level name.
s1 = State( guest, name="guest" )
s2 = State( prize, name="prize" )
s3 = State( monty, name="monty" )

# Create the Bayesian network object with a useful name
model = BayesianNetwork( "Monty Hall Problem" )

# Add the three states to the network 
model.add_states(s1, s2, s3)

# Add transitions which represent conditional dependencies, where the second node is conditionally dependent on the first node (Monty is dependent on both guest and prize)
model.add_transition(s1, s3)
model.add_transition(s2, s3)
model.bake()


# ## Probability
# 
# We can calculate the probability or log probability of a point under the Bayesian network using the appropriately named `probability` and `log_probability` methods. The Bayesian network can give us a more intricate understanding of the connection between variables, and so in many cases is more sophisticated than a simple multivariate distribution.

print model.probability(['A', 'B', 'C'])
print model.probability(['B', 'B', 'B'])
print
print model.log_probability(['C', 'A', 'B'])
print model.log_probability(['B', 'A', 'A'])


# ## Inference
# 
# pomegranate uses the loopy belief propagation algorithm to do inference. This is an approximate algorithm which can yield exact results in tree-like graphs, and in most other cases still yields good results. Inference on a Bayesian network consists of taking in observations for a subset of the variables and using that to infer the values that the other variables take. The most variables which are observed, the closer the inferred values will be to truth. One of the powers of Bayesian networks is that the set of observed and 'hidden' (or unobserved) variables does not need to be specified beforehand, and can change from sample to sample.
# 
# 
# We can run inference using the `predict_proba` method and passing in a dictionary of values, where the key is the name of the state and the value is the observed value for that state. If we don't supply any values, we get the marginal of the graph, which is just the frequency of each value for each variable over an infinite number of randomly drawn samples from the graph.
# 
# Lets see what happens when we look at the marginal of the Monty hall network.

print model.predict_proba({})


# We are returned three `DiscreteDistribution` objects, each representing the marginal distribution for each variable, in the same order they were put into the model. In this case, they represent the guest, prize, and monty variables respectively. We see that everything is equally likely. If we want to access these distributions, we can do the following:

marginals = model.predict_proba({})
print marginals[0].parameters[0]


# The first element of `marginals` is a DiscreteDistribution, with all the same operations as a normal DiscreteDistribution objects. This means that parameters[0] will return the underlying dictionary used by the distribution, which we return here.
# 
# Now lets do something different, and say that the guest has chosen door 'A'. We do this by passing a dictionary to `predict_proba` with key pairs consisting of the name of the state (in the state object), and the value which that variable has taken.

model.predict_proba({'guest': 'A'})


# We can see that now Monty will not open door 'A', because the guest has chosen it. At the same time, the distribution over the prize has not changed, it is still equally likely that the prize is behind each door.
# 
# Now, lets say that Monty opens door 'C' and see what happens.

model.predict_proba({'guest': 'A', 'monty': 'C'})


# Suddenly, we see that the distribution over prizes has changed. It is now twice as likely that the car is behind the door labeled 'B'. This illustrates the somewhat famous Monty Hall problem.
# 
# ## Imputation
# 
# Bayesian networks also have an `predict` method which can be used to fill in missing values with their most likely value. If you have a bayesian network and the values for some variables in a sample, you can impute the most likely values for the remaining variables. You can indicate which values are missing by using `None`. Loopy belief propagation is then run to find the distribution of values for each variable like above, and the most likely variable is used. If all values are equally likely, it will randomly choose a value, which may be sub-optimal. For an example:

model.predict([['B', 'A', None],
               ['C', 'A', None],
               ['B', 'C', None],
               ['A', 'B', None]])


# ## Parameter Fitting
# 
# Networks can be trained by passing in a matrix with observation position corresponding to the state in the model. Currently it only works on datasets which do not contain any missing values, such as the following:

model.fit([['A', 'B', 'C'],
           ['A', 'C', 'B'],
           ['A', 'A', 'C'],
           ['B', 'B', 'C'], 
           ['B', 'C', 'A']])


print model.predict_proba({})


# ## Structure Learning
# 
# The most difficult task involving Bayesian networks is the learning of structure. Naively this algorithm will take super-exponential time with the number of nodes, because there are a super-exponential number of DAGs to consider. Given the complicated nature of this task, researchers have proposed many different ways to tackle the structure learning problem, of which there are two main approaches: constraint based algorithms, and scoring based methods. The constraint based methods attempt to find conditional independence statements in the data and learn a Bayesian network using those, while scoring based methods attempt optimize some score to match the posterior well to the data.
# 
# Currently pomegranate contains two scoring function based structure learning algorithms, the Chow-Liu tree and an exact network with an efficient dynamic programming backend. The Chow-Liu tree algorithm attempts to find the best single tree to approximate the underlying distribution, and can be extremely fast, but it limits each node to having a single parent. The exact algorithm can be very fast compared to more naive scoring techniques, but gets the problem down to "only exponential" in time with the number of nodes. Due to some programmatic efficiencies, in practice exactly optimal Bayesian networks can be found for 20-25 variables assuming that there is enough memory. 
# 
# Lets take a look at both of these learning algorithms on randomly generated data. Since the algorithms take the same amount of time given completely random data as structured data, we're going to use completely random data since it is easier to generate. We can use the `from_samples` method to do structure learning for Bayesian networks, just like we used it to initialize other models directly from data as well.
# 
# Lets first take a look at the time it takes to calculate an exact Bayesian network using 10000 samples and between 2 and 17 variables.

get_ipython().run_line_magic('pylab', 'inline')
import time

times = []
for i in range(2, 18):
    tic = time.time()
    X = numpy.random.randint(2, size=(10000, i))
    model = BayesianNetwork.from_samples(X, algorithm='exact')
    times.append( time.time() - tic )


import seaborn
seaborn.set_style('whitegrid')

plt.figure(figsize=(14, 6))
plt.title('Time To Learn Bayesian Network', fontsize=18)
plt.xlabel("Number of Variables", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(range(2, 18), times, linewidth=3, color='c')
plt.yscale('log')


# The algorithm looks like it's running in exponential time, which is a major feat for the prospect of Bayesian network structure learning. Combined with an efficient implementation of the algorithm, this seems like calculating a graph with mid-20 variables would be possible on a normal computer.
# 
# Lets take a look at the speed of the time it takes to calculate the Chow-Liu tree. Since the algorithm is much faster than finding the exact graph, lets check out how long it takes to learn for a larger range of variables (up to 100).

times = []
for i in range(2, 253, 10):
    tic = time.time()
    X = numpy.random.randint(2, size=(10000, i))
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
    times.append( time.time() - tic )


import seaborn
seaborn.set_style('whitegrid')

plt.figure(figsize=(14, 6))
plt.title('Time To Learn Bayesian Network', fontsize=18)
plt.xlabel("Number of Variables", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot( range(2, 253, 10), times, linewidth=3, color='c')
plt.yscale('log')


# It looks like this is a quadratic time algorithm, and can be used to efficiently find tree-like graph  approximations for datasets with a large number of variables. 

# # pomegranate / sklearn GMM comparison
# 
# authors: <br>
# Nicholas Farn (nicholasfarn@gmail.com) <br>
# Jacob Schreiber (jmschreiber91@gmail.com)
# 
# <a href="https://github.com/scikit-learn/scikit-learn">sklearn</a> is a very popular machine learning package for Python which implements a wide variety of classical machine learning algorithms. In this notebook we benchmark the GMM implementations in pomegranate and compare it to the implementation in sklearn. In sklearn, GMM refers exclusively to Gaussian mixture models, while in pomegranate it refers to General mixture models, as it is flexible enough to allow any combination of distributions or models to be used as components.
# 
# However, a simpler version of the GMM is kmeans clustering. Both pomegranate and sklearn implement these, so lets take a look at those first.

get_ipython().run_line_magic('pylab', 'inline')
import seaborn, time
seaborn.set_style('whitegrid')

from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from pomegranate import *


# Lets create some functions to evaluate models efficiently. We can start with fitting and predicting against an increasing data size. The data we compare against are Gaussian blobs which are 3 standard deviations away from each other to allow for some, but not a lot of, overlap.

def create_dataset(n_samples, n_dim, n_classes):
    """Create a random dataset with n_samples in each class."""
    
    X = numpy.concatenate([numpy.random.randn(n_samples, n_dim) + i*3 for i in range(n_classes)])
    y = numpy.concatenate([numpy.zeros(n_samples) + i for i in range(n_classes)])
    return X, y

def plot( fit, predict, skl_error, pom_error, sizes, xlabel ):
    """Plot the results."""
    
    idx = numpy.arange(fit.shape[1])
    
    plt.figure( figsize=(14, 4))
    plt.plot( fit.mean(axis=0), c='c', label="Fitting")
    plt.plot( predict.mean(axis=0), c='m', label="Prediction")
    plt.plot( [0, fit.shape[1]], [1, 1], c='k', label="Baseline" )
    
    plt.fill_between( idx, fit.min(axis=0), fit.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, predict.min(axis=0), predict.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks(idx, sizes, rotation=65, fontsize=14)
    plt.xlabel('{}'.format(xlabel), fontsize=14)
    plt.ylabel('pomegranate is x times faster', fontsize=14)
    plt.legend(fontsize=12, loc=4)
    plt.show()
    
    
    plt.figure( figsize=(14, 4))
    plt.plot( 1 - skl_error.mean(axis=0), alpha=0.5, c='c', label="sklearn accuracy" )
    plt.plot( 1 - pom_error.mean(axis=0), alpha=0.5, c='m', label="pomegranate accuracy" )
    
    plt.fill_between( idx, 1-skl_error.min(axis=0), 1-skl_error.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, 1-pom_error.min(axis=0), 1-pom_error.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks( idx, sizes, rotation=65, fontsize=14)
    plt.xlabel( '{}'.format(xlabel), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14) 
    plt.show()
    
def evaluate_kmeans():
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))

    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 1, 2 )

            pom = Kmeans(2)
            skl = KMeans(2, max_iter=1, n_init=1, precompute_distances=True, init=X[:2].copy())

            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic

            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "samples per component")


# Below we see the fit and predict speeds for the two algorithms on increasing dataset sizes. pomegranate is always faster than sklearn for both prediction and fitting steps. These numbers fluctuate a bit every run and so running it multiple times (20 in this case) and reporting mean and standard deviations is ideal. Since these are unsupervised algorithms, accuracy is kind of a weird metric, but it refers to assigning cluster labels correctly corresponding to the underlying distribution which generated the data.

evaluate_kmeans()


# The lines show the mean of the plotted value, and the bounds show the minimum and maximum value. We can see that kmeans looks liks it's between 5-10x faster per fitting iteration than sklearn is, and around 3-4x faster when it comes to prediction! To confirm that the two are getting the same results, we plot the unsupervised 'accuracy' for the two, and see that they are identical.  
# 
# Lets now look at Gaussian datasets with two components when running Gaussian Mixture Models. We'll look at how many times faster pomegranate is, which means that values > 1 show pomegranate is faster and < 1 show pomegranate is slower. Lets also look at the accuracy of both algorithms. Accuracy should be roughly the same, but different, since both have different random initialization points. The measured the time is the time it takes to do a single iteration of EM.

def evaluate_models():
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))

    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 1, 2 )

            pom = GeneralMixtureModel( NormalDistribution, n_components=2 )
            skl = GMM( n_components=3, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic

            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    plot(fit, predict, skl_error, pom_error, sizes, "samples per component")

evaluate_models()


# Lets also see how well it scales as we add more components where there are 10,000 total data points. 

def evaluate_models():
    sizes = numpy.arange(2, 21, dtype='int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))

    for i in range(m):
        for j, size in enumerate(sizes):        
            X, y = create_dataset(10000 / size, 1, size )

            pom = GeneralMixtureModel( NormalDistribution, n_components=size )
            skl = GMM( n_components=size, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic

            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    plot(fit, predict, skl_error, pom_error, sizes, "number of components")

evaluate_models()


# Looks like pomegranate can be much faster than sklearn at this task--around 10x faster for the fitting step. The reason that the accuracies vary is because a different random initialization is used for each model.
# 
# Now lets take a look at Multivariate Gaussian models. For sklearn the initialization is exactly the same, but for pomegranate the MultivariateGaussianDistribution object must be passed in instead of the NormalDistribution object.

def evaluate_models():
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 5, 2 )
            
            pom = GeneralMixtureModel(MultivariateGaussianDistribution, n_components=2)
            skl = GMM( n_components=2, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    plot(fit, predict, skl_error, pom_error, sizes, "samples per component")

evaluate_models()


# Looks like pomegranate is faster in the fitting step for multivariate Gaussian mixture models and roughly the same speed for making predictions. The accuracy seems to be pretty much overlapping for these at near perfect.
# 
# Lets see how the two models scale when changing the dimensionality of the data between 2 and 20 dimensions.

numpy.set_printoptions(suppress=True, linewidth=200)

def evaluate_models():
    sizes = numpy.arange(2, 21).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset(50000, size, 2)
            
            pom = GeneralMixtureModel(MultivariateGaussianDistribution, n_components=2)
            skl = GMM( n_components=2, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    plot(fit, predict, skl_error, pom_error, sizes, "dimensions")

evaluate_models()


# It seems as though both fitting and prediction using pomegranate scale less well than sklearn to more dimensions, but that it is stil faster. 
# 
# This notebook tested a main portion of the overlap between sklearn and pomegranate, but both offer clustering options which aren't displayed here. pomegranate allows any distribution or mixture of distributions, univariate or multivariate, and even some more complex models to be used as a component in the mixture model. The out of core API which pomegranate offers also extends to GMMs, allowing them to be trained using exact EM updates on data which can't fit in memory. In contrast, sklearn offers suppoort for both dirichlet process GMMs (DPGMMs) and variational Bayes GMMs (VBGMMs) as well, which are robust clustering models.
# 
# We hope this has been useful to you! If you're interested in using pomegranate, you can get it using `pip install pomegranate` or by checking out the <a href="https://github.com/jmschrei/pomegranate">github repo.</a>

# # pomegranate / libpgm comparison
# 
# authors: Jacob Schreiber (jmschreiber91@gmail.com)
# 
# <a href="https://github.com/CyberPoint/libpgm">libpgm</a> is a python package for creating and using Bayesian networks. I was unable to figure out how to use libpgm to do inference properly without raising errors, but I was able to get structure learning working. libpgm uses constraints for structure learning, a process which is not probabilistic, but can be asymptoptically more efficient (between O(n^2) and O(n^3) as opposed to exponential). To my knowledge, they do not have exact structure learning implemented, likely due to the super-exponential nature of the naive algorithm.
# 
# pomegranate has both the exact structure learning problem, and the Chow-Liu tree approximation, implemented. The exact structure learning problem uses an efficient dynamic programming solution to reduce the complexity from super-exponential to exponential in time with the number of variables. The Chow-Liu tree approximation finds the best tree which spans all variables.
# 
# Lets compare the structure learning task in pomegranate versus the structure learning task in libpgm for different numbers of variables to compare these speed of the two packages.

get_ipython().run_line_magic('pylab', 'inline')
import seaborn, time
seaborn.set_style('whitegrid')


# Lets first compare the two packages based on number of variables.

from pomegranate import BayesianNetwork
from libpgm.pgmlearner import PGMLearner

libpgm_time = []
pomegranate_time = []
pomegranate_cl_time = []

for i in range(2, 15):
    tic = time.time()
    X = numpy.random.randint(2, size=(10000, i))
    model = BayesianNetwork.from_samples(X, algorithm='exact')
    pomegranate_time.append(time.time() - tic)

    tic = time.time()
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
    pomegranate_cl_time.append(time.time() - tic)

    X = [{j : X[i, j] for j in range(X.shape[1])} for i in range(X.shape[0])]
    learner = PGMLearner()

    tic = time.time()
    model = learner.discrete_constraint_estimatestruct(X)
    libpgm_time.append(time.time() - tic)


plt.figure(figsize=(14, 6))
plt.title("Bayesian Network Structure Learning Time", fontsize=16)
plt.xlabel("Number of Variables", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.plot(range(2, 15), libpgm_time, c='c', label="libpgm")
plt.plot(range(2, 15), pomegranate_time, c='m', label="pomegranate exact")
plt.plot(range(2, 15), pomegranate_cl_time, c='r', label="pomegranate chow liu")
plt.legend(loc=2, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# We can see many expected results from this graph. libpgm implements a quadratic time algorithm, and it appears that the growth is roughly quadratic. The exact algorithm in pomegranate is an exponential time algorithm, and so while it does have an efficient implementation that causes the it to be much faster than libpgm's algorithm for small numbers of variables, eventually it will become slower. Lastly, the chow-liu tree algorithm is far faster than both other algorithms because it is finding the best tree approximation. While it is a quadratic time algorithm, it is also a simpler one.
# 
# Lets now compare the speed on different numbers of samples.

libpgm_time = []
pomegranate_time = []
pomegranate_cl_time = []

x = 10, 25, 100, 250, 1000, 2500, 10000, 25000, 100000, 250000, 1000000
for i in x:
    tic = time.time()
    X = numpy.random.randint(2, size=(i, 10))
    model = BayesianNetwork.from_samples(X, algorithm='exact')
    pomegranate_time.append(time.time() - tic)

    tic = time.time()
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
    pomegranate_cl_time.append(time.time() - tic)

    X = [{j : X[i, j] for j in range(X.shape[1])} for i in range(X.shape[0])]
    learner = PGMLearner()

    tic = time.time()
    model = learner.discrete_constraint_estimatestruct(X)
    libpgm_time.append(time.time() - tic)


plt.figure(figsize=(14, 6))
plt.title("Bayesian Network Structure Learning Time", fontsize=16)
plt.xlabel("Number of Samples", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.plot(x, libpgm_time, c='c', label="libpgm")
plt.plot(x, pomegranate_time, c='m', label="pomegranate exact")
plt.plot(x, pomegranate_cl_time, c='r', label="pomegranate chow liu")
plt.legend(loc=2, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# It looks like when fitting for the same number of variables that pomegranate is able to scale much better than libpgm is. This could mean that for large datasets, pomegranate is much faster than libpgm even though it has a worse theoretical notation.
# 
# However, it's also important to note that this isn't exactly a fair comparison because they are two different algorithms. libpgm is implementing the constraint based structure learning algorithm which is quadratic in time, and pomegranate is using an exact structure learning algorithm which is exponential in time. However, this benchmark should lay out some practical speed differences when deciding which package to use.

# # pomegranate / hmmlearn comparison
# 
# <a href="https://github.com/hmmlearn/hmmlearn">hmmlearn</a> is a Python module for hidden markov models with a scikit-learn like API. It was originally present in scikit-learn until its removal due to structural learning not meshing well with the API of many other classical machine learning algorithms. Here is a table highlighting some of the similarities and differences between the two packages.
# 
# <table>
# <tr>
# <th>Feature</th>
# <th>pomegranate</th>
# <th>hmmlearn</th>
# </tr>
# <tr>
# <th>Graph Structure</th>
# <th></th>
# <th></th>
# </tr>
# <tr>
# <td>Silent States</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Optional Explicit End State</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Sparse Implementation</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Arbitrary Emissions Allowed on States</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Discrete/Gaussian/GMM Emissions</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Large Library of Other Emissions</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Build Model from Matrices</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Build Model Node-by-Node</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Serialize to JSON</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Serialize using Pickle/Joblib</td>
# <td></td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <th>Algorithms</th>
# <th></th>
# <th></th>
# </tr>
# <tr>
# <td>Priors</td>
# <td></td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Sampling</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Log Probability Scoring</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Forward-Backward Emissions</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Forward-Backward Transitions</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Viterbi Decoding</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>MAP Decoding</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Baum-Welch Training</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Viterbi Training</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Labeled Training</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Tied Emissions</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Tied Transitions</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Emission Inertia</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Transition Inertia</td>
# <td>&#10003;</td>
# <td></td>
# </tr>
# <tr>
# <td>Emission Freezing</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Transition Freezing</td>
# <td>&#10003;</td>
# <td>&#10003;</td>
# </tr>
# <tr>
# <td>Multi-threaded Training</td>
# <td>&#10003;</td>
# <td>Coming Soon</td>
# </tr>
# 
# </table>
# </p>
# 
# Just because the two features are implemented doesn't speak to how fast they are. Below we investigate how fast the two packages are in different settings the two have implemented. 
# 
# ## Fully Connected Graphs with Multivariate Gaussian Emissions
# 
# Lets look at the sample scoring method, viterbi, and Baum-Welch training for fully connected graphs with multivariate Gaussian emisisons. A fully connected graph is one where all states have connections to all other states. This is a case which pomegranate is expected to do poorly due to its sparse implementation, and hmmlearn should shine due to its vectorized implementations.

get_ipython().run_line_magic('pylab', 'inline')
import hmmlearn, pomegranate, time, seaborn
from hmmlearn.hmm import *
from pomegranate import *
seaborn.set_style('whitegrid')


# Both hmmlearn and pomegranate are under active development. Here are the current versions of the two packages.

print "hmmlearn version {}".format(hmmlearn.__version__)
print "pomegranate version {}".format(pomegranate.__version__)


# We first should have a function which will randomly generate transition matrices and emissions for the hidden markov model, and randomly generate sequences which fit the model.

def initialize_components(n_components, n_dims, n_seqs):
    """
    Initialize a transition matrix for a model with a fixed number of components,
    for Gaussian emissions with a certain number of dimensions, and a data set
    with a certain number of sequences.
    """
    
    transmat = numpy.abs(numpy.random.randn(n_components, n_components))
    transmat = (transmat.T / transmat.sum( axis=1 )).T

    start_probs = numpy.abs( numpy.random.randn(n_components) )
    start_probs /= start_probs.sum()

    means = numpy.random.randn(n_components, n_dims)
    covars = numpy.ones((n_components, n_dims))
    
    seqs = numpy.zeros((n_seqs, n_components, n_dims))
    for i in range(n_seqs):
        seqs[i] = means + numpy.random.randn(n_components, n_dims)
        
    return transmat, start_probs, means, covars, seqs


# Lets create the model in hmmlearn. It's fairly straight forward, only some attributes need to be overridden with the known structure and emissions.

def hmmlearn_model(transmat, start_probs, means, covars):
    """Return a hmmlearn model."""

    model = GaussianHMM(n_components=transmat.shape[0], covariance_type='diag', n_iter=1, tol=1e-8)
    model.startprob_ = start_probs
    model.transmat_ = transmat
    model.means_ = means
    model._covars_ = covars
    return model


# Now lets create the model in pomegranate. Also fairly straightforward. The biggest difference is creating explicit distribution objects rather than passing in vectors, and passing everything into a function instead of overriding attributes. This is done because each state in the graph can be a different distribution and many distributions are supported.

def pomegranate_model(transmat, start_probs, means, covars):
    """Return a pomegranate model."""
    
    states = [ MultivariateGaussianDistribution( means[i], numpy.eye(means.shape[1]) ) for i in range(transmat.shape[0]) ]
    model = HiddenMarkovModel.from_matrix(transmat, states, start_probs, merge='None')
    return model


# Lets now compare some algorithm times.

def evaluate_models(n_dims, n_seqs):
    hllp, plp = [], []
    hlv, pv = [], []
    hlm, pm = [], []
    hls, ps = [], []
    hlt, pt = [], []

    for i in range(10, 112, 10):
        transmat, start_probs, means, covars, seqs = initialize_components(i, n_dims, n_seqs)
        model = hmmlearn_model(transmat, start_probs, means, covars)

        tic = time.time()
        for seq in seqs:
            model.score(seq)
        hllp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        hlv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        hlm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs.reshape(n_seqs*i, n_dims), lengths=[i]*n_seqs)
        hlt.append( time.time() - tic )

        model = pomegranate_model(transmat, start_probs, means, covars)

        tic = time.time()
        for seq in seqs:
            model.log_probability(seq)
        plp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        pv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        pm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs, max_iterations=1, verbose=False)
        pt.append( time.time() - tic )

    plt.figure( figsize=(12, 8))
    plt.xlabel("# Components", fontsize=12 )
    plt.ylabel("pomegranate is x times faster", fontsize=12 )
    plt.plot( numpy.array(hllp) / numpy.array(plp), label="Log Probability")
    plt.plot( numpy.array(hlv) / numpy.array(pv), label="Viterbi")
    plt.plot( numpy.array(hlm) / numpy.array(pm), label="Maximum A Posteriori")
    plt.plot( numpy.array(hlt) / numpy.array(pt), label="Training")
    plt.xticks( xrange(11), xrange(10, 112, 10), fontsize=12 )
    plt.yticks( fontsize=12 )
    plt.legend( fontsize=12 )


evaluate_models(10, 50)


# It looks like in this case pomegranate and hmmlearn are approximately the same for large (>30 components) dense graphs for the forward algorithm (log probability), MAP, and training. However, hmmlearn is significantly faster in terms of calculating the Viterbi path, while pomegranate is faster for smaller (<30 components) graphs.

# ## Sparse Graphs with Multivariate Gaussian Emissions
# 
# pomegranate is based off of a sparse implementations and so excels in graphs which are sparse. Lets try a model architecture where each hidden state only has transitions to itself and the next state, but running the same algorithms as last time.

def initialize_components(n_components, n_dims, n_seqs):
    """
    Initialize a transition matrix for a model with a fixed number of components,
    for Gaussian emissions with a certain number of dimensions, and a data set
    with a certain number of sequences.
    """
    
    transmat = numpy.zeros((n_components, n_components))
    transmat[-1, -1] = 1
    for i in range(n_components-1):
        transmat[i, i] = 1
        transmat[i, i+1] = 1
    transmat[ transmat < 0 ] = 0
    transmat = (transmat.T / transmat.sum( axis=1 )).T

    start_probs = numpy.abs( numpy.random.randn(n_components) )
    start_probs /= start_probs.sum()

    means = numpy.random.randn(n_components, n_dims)
    covars = numpy.ones((n_components, n_dims))
    
    seqs = numpy.zeros((n_seqs, n_components, n_dims))
    for i in range(n_seqs):
        seqs[i] = means + numpy.random.randn(n_components, n_dims)
        
    return transmat, start_probs, means, covars, seqs


evaluate_models(10, 50)


# ## Sparse Graph with Discrete Emissions
# 
# Lets also compare MultinomialHMM to a pomegranate HMM with discrete emisisons for completeness.

def initialize_components(n_components, n_seqs):
    """
    Initialize a transition matrix for a model with a fixed number of components,
    for Gaussian emissions with a certain number of dimensions, and a data set
    with a certain number of sequences.
    """
    
    transmat = numpy.zeros((n_components, n_components))
    transmat[-1, -1] = 1
    for i in range(n_components-1):
        transmat[i, i] = 1
        transmat[i, i+1] = 1
    transmat[ transmat < 0 ] = 0
    transmat = (transmat.T / transmat.sum( axis=1 )).T
    
    start_probs = numpy.abs( numpy.random.randn(n_components) )
    start_probs /= start_probs.sum()

    dists = numpy.abs(numpy.random.randn(n_components, 4))
    dists = (dists.T / dists.T.sum(axis=0)).T
    
    seqs = numpy.random.randint(0, 4, (n_seqs, n_components*2, 1))
    return transmat, start_probs, dists, seqs

def hmmlearn_model(transmat, start_probs, dists):
    """Return a hmmlearn model."""

    model = MultinomialHMM(n_components=transmat.shape[0], n_iter=1, tol=1e-8)
    model.startprob_ = start_probs
    model.transmat_ = transmat
    model.emissionprob_ = dists
    return model

def pomegranate_model(transmat, start_probs, dists):
    """Return a pomegranate model."""
    
    states = [ DiscreteDistribution({ 'A': d[0],
                                      'C': d[1],
                                      'G': d[2], 
                                      'T': d[3] }) for d in dists ]
    model = HiddenMarkovModel.from_matrix(transmat, states, start_probs, merge='None')
    return model

def evaluate_models(n_seqs):
    hllp, plp = [], []
    hlv, pv = [], []
    hlm, pm = [], []
    hls, ps = [], []
    hlt, pt = [], []

    dna = 'ACGT'
    
    for i in range(10, 112, 10):
        transmat, start_probs, dists, seqs = initialize_components(i, n_seqs)
        model = hmmlearn_model(transmat, start_probs, dists)

        tic = time.time()
        for seq in seqs:
            model.score(seq)
        hllp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        hlv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        hlm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs.reshape(n_seqs*i*2, 1), lengths=[i*2]*n_seqs)
        hlt.append( time.time() - tic )

        model = pomegranate_model(transmat, start_probs, dists)
        seqs = [[dna[i] for i in seq] for seq in seqs]

        tic = time.time()
        for seq in seqs:
            model.log_probability(seq)
        plp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        pv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        pm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs, max_iterations=1, verbose=False)
        pt.append( time.time() - tic )

    plt.figure( figsize=(12, 8))
    plt.xlabel("# Components", fontsize=12 )
    plt.ylabel("pomegranate is x times faster", fontsize=12 )
    plt.plot( numpy.array(hllp) / numpy.array(plp), label="Log Probability")
    plt.plot( numpy.array(hlv) / numpy.array(pv), label="Viterbi")
    plt.plot( numpy.array(hlm) / numpy.array(pm), label="Maximum A Posteriori")
    plt.plot( numpy.array(hlt) / numpy.array(pt), label="Training")
    plt.xticks( xrange(11), xrange(10, 112, 10), fontsize=12 )
    plt.yticks( fontsize=12 )
    plt.legend( fontsize=12 )


evaluate_models(50)


# # pomegranate / sklearn Naive Bayes comparison
# 
# authors: <br>
# Nicholas Farn (nicholasfarn@gmail.com) <br>
# Jacob Schreiber (jmschreiber91@gmail.com)
# 
# <a href="https://github.com/scikit-learn/scikit-learn">sklearn</a> is a very popular machine learning package for Python which implements a wide variety of classical machine learning algorithms. In this notebook we benchmark the Naive Bayes implementations in pomegranate and compare it to the implementation in sklearn.

get_ipython().run_line_magic('pylab', 'inline')
import seaborn, time
seaborn.set_style('whitegrid')

from sklearn.naive_bayes import GaussianNB
from pomegranate import *


# Lets first define a function which will create a dataset to train on. We want to be able to test a range of datasets, from very small to very large, to see which implementation is faster. We also want a function which will take in the models and evaluate them. Lets define both of those now.

def create_dataset(n_samples, n_dim, n_classes):
    """Create a random dataset with n_samples in each class."""
    
    X = numpy.concatenate([numpy.random.randn(n_samples, n_dim) + i for i in range(n_classes)])
    y = numpy.concatenate([numpy.zeros(n_samples) + i for i in range(n_classes)])
    return X, y

def plot( fit, predict, skl_error, pom_error, sizes, xlabel ):
    """Plot the results."""
    
    idx = numpy.arange(fit.shape[1])
    
    plt.figure( figsize=(14, 4))
    plt.plot( fit.mean(axis=0), c='c', label="Fitting")
    plt.plot( predict.mean(axis=0), c='m', label="Prediction")
    plt.plot( [0, fit.shape[1]], [1, 1], c='k', label="Baseline" )
    
    plt.fill_between( idx, fit.min(axis=0), fit.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, predict.min(axis=0), predict.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks(idx, sizes, rotation=65, fontsize=14)
    plt.xlabel('{}'.format(xlabel), fontsize=14)
    plt.ylabel('pomegranate is x times faster', fontsize=14)
    plt.legend(fontsize=12, loc=4)
    plt.show()
    
    
    plt.figure( figsize=(14, 4))
    plt.plot( 1 - skl_error.mean(axis=0), alpha=0.5, c='c', label="sklearn accuracy" )
    plt.plot( 1 - pom_error.mean(axis=0), alpha=0.5, c='m', label="pomegranate accuracy" )
    
    plt.fill_between( idx, 1-skl_error.min(axis=0), 1-skl_error.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, 1-pom_error.min(axis=0), 1-pom_error.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks( idx, sizes, rotation=65, fontsize=14)
    plt.xlabel( '{}'.format(xlabel), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14) 
    plt.show()
    
def evaluate_models( skl, pom ):
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 1, 2 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "samples per component")


# Lets look first at single dimension Gaussian datasets. We'll look at how many times faster pomegranate is, which means that values > 1 show pomegranate is faster and < 1 show pomegranate is slower. Lets also look at the accuracy of both algorithms. They should have the same accuracy since they implement the same algorithm.

skl = GaussianNB()
pom = NaiveBayes( NormalDistribution )
evaluate_models( skl, pom )


# It looks as if pomegranate has a significant speed improvement for small models, likely through having a smaller initialization cost. Even on large datasets it can be faster to fit models. While it is significantly slower at making predictions these times are usually so small that they are insignificant anyway.
# 
# 
# Now lets look if they scale as more classes are added, instead of just binary classification, with a total of 50,000 samples across all classes.

def evaluate_models( skl, pom ):
    sizes = numpy.arange(2, 21).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( 50000 / size, 1, size )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "number of classes")


skl = GaussianNB()
pom = NaiveBayes( NormalDistribution )
evaluate_models( skl, pom )


# It looks like pomegranate initially starts off as faster fitting the model with a small number of classes, but that this speed increase becomes insignificant as most classes are added, converging to roughly sklearn's speed. However, sklearn remains faster at making predictions. Lets take a look at the raw time it takes to make predictions though.

X, y = create_dataset( 50000, 1, 2 )
skl = GaussianNB()
skl.fit(X, y)

pom = NaiveBayes( NormalDistribution )
pom.fit(X, y)

get_ipython().run_line_magic('timeit', 'skl.predict(X)')
get_ipython().run_line_magic('timeit', 'pom.predict(X)')


# This does show that sklearn is significantly faster at this step. However, predicting 100,000 points in 45ms is sufficient for most applications. 
# 
# More commonly multivariate Gaussian emissions are used. scikit-learn supports this with the same estimator and pomegranate supports this with plugging in a different distribution. Lets look at increasing the number of samples again with a default of 5 dimensions.

def evaluate_models( skl, pom ):
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 5, 2 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "samples per component")


skl = GaussianNB()
pom = NaiveBayes( MultivariateGaussianDistribution )
evaluate_models( skl, pom )


# It looks like pomegranate can be several times faster at fitting multivariate Gaussian Naive Bayes models than sklearn is, with a speed that doesn't decay too much as the dataset gets larger. We are getting slightly different accuracies between the two models, but this can be explained because sklearn learns a diagonal covariance matrix while pomegranate learns the full covariance matrix. Given this, the times aren't exactly comparable either because learning the full covariance matrix is slightly more time intensive than just the diagonal. 
# 
# Finally lets show an increasing number of dimensions with a fixed set of 10 classes and 50,000 samples per class.

def evaluate_models( skl, pom ):
    sizes = numpy.arange(3, 21).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( 50000, size, 2 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "dimensions")

skl = GaussianNB()
pom = NaiveBayes( MultivariateGaussianDistribution )
evaluate_models( skl, pom )


# Looks like pomegranate does better in the low dimensional setting but eventually converges to the same as sklearn given large dimensionality and a large number of samples. Their accuracies remain identical indicating that the two are learning the same model.

# ## Out of Core Training
# 
# Lastly, both pomegranate and sklearn allow for out of core training by fitting on chunks of a dataset. pomegranate does this by calculating summary statistics on the dataset which are enough to allow for exact parameter updates to be done. sklearn implements this using the `model.partial_fit(X, y)` API call, whereas pomegranate uses `model.summarize(X, y)` followed by `model.from_summaries()` to update the internal parameters.  
# 
# Lets compare how long each method takes to train on 25 batches of increasing sizes and the accuracy of both methods.

def evaluate_models( skl, pom ):
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_time, pom_time = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            for l in range(5):
                X, y = create_dataset( size, 5, 2 )

                tic = time.time()
                skl.partial_fit( X, y, classes=[0, 1] )
                skl_time[i, j] += time.time() - tic

                tic = time.time()
                pom.summarize( X, y )
                pom_time[i, j] += time.time() - tic

            tic = time.time()
            pom.from_summaries()
            pom_time[i, j] += time.time() - tic

            skl_predictions = skl.predict( X )
            pom_predictions = pom.predict( X )

            skl_error[i, j] = ( y != skl_predictions ).mean()
            pom_error[i, j] = ( y != pom_predictions ).mean()
    
    fit = skl_time / pom_time
    idx = numpy.arange(fit.shape[1])
    
    plt.figure( figsize=(14, 4))
    plt.plot( fit.mean(axis=0), c='c', label="Fitting")
    plt.plot( [0, fit.shape[1]], [1, 1], c='k', label="Baseline" )
    plt.fill_between( idx, fit.min(axis=0), fit.max(axis=0), color='c', alpha=0.3 )
    
    plt.xticks(idx, sizes, rotation=65, fontsize=14)
    plt.xlabel('{}'.format(xlabel), fontsize=14)
    plt.ylabel('pomegranate is x times faster', fontsize=14)
    plt.legend(fontsize=12, loc=4)
    plt.show()
    
    
    plt.figure( figsize=(14, 4))
    plt.plot( 1 - skl_error.mean(axis=0), alpha=0.5, c='c', label="sklearn accuracy" )
    plt.plot( 1 - pom_error.mean(axis=0), alpha=0.5, c='m', label="pomegranate accuracy" )
    
    plt.fill_between( idx, 1-skl_error.min(axis=0), 1-skl_error.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, 1-pom_error.min(axis=0), 1-pom_error.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks( idx, sizes, rotation=65, fontsize=14)
    plt.xlabel( '{}'.format(xlabel), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14) 
    plt.show()


skl = GaussianNB()
pom = NaiveBayes( MultivariateGaussianDistribution )
evaluate_models( skl, pom )


# pomegranate seems to be faster doing out-of-core training. The out of core API of calculating sufficient statistics using `summarize` and then updating the model parameters using `from_summaries` extends to all models in pomegranate. 
# 
# In this notebook we compared an intersection of the features that pomegranate and sklearn offer. pomegranate allows you to use Naive Bayes with any distribution or model object which has an exposed `log_probability` and `fit` method. This allows you to do things such as compare hidden Markov models to each other, or compare a hidden Markov model to a Markov Chain to see which one models the data better. 
# 
# We hope this has been useful to you! If you're interested in using pomegranate, you can get it using `pip install pomegranate` or by checking out the <a href="https://github.com/jmschrei/pomegranate">github repo.</a>

# ## Bayesian Networks

# author: Jacob Schreiber <br>
# contact: jmschreiber91@gmail.com

# Bayesian networks are a powerful inference tool, in which a set of variables are represented as nodes, and the lack of an edge represents a conditional independence statement between the two variables, and an edge represents a dependence between the two variables. One of the powerful components of a Bayesian network is the ability to infer the values of certain variables, given observed values for another set of variables. These are referred to as the 'hidden' and 'observed' variables respectively, and need not be set at the time the network is created. The same network can have a different set of variables be hidden or observed between two data points. The more values which are observed, the closer the inferred values will be to the truth.
# 
# While Bayesian networks can have extremely complex emission probabilities, usually Gaussian or conditional Gaussian distributions, pomegranate currently supports only discrete Bayesian networks. Bayesian networks are explicitly turned into Factor Graphs when inference is done, wherein the Bayesian network is turned into a bipartite graph with all variables having marginal nodes on one side, and joint tables on the other.

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import numpy

from pomegranate import *

numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


# ### The Monty Hall Gameshow
# 
# The Monty Hall problem arose from the gameshow <i>Let's Make a Deal</i>, where a guest had to choose which one of three doors had a prize behind it. The twist was that after the guest chose, the host, originally Monty Hall, would then open one of the doors the guest did not pick and ask if the guest wanted to switch which door they had picked. Initial inspection may lead you to believe that if there are only two doors left, there is a 50-50 chance of you picking the right one, and so there is no advantage one way or the other. However, it has been proven both through simulations and analytically that there is in fact a 66% chance of getting the prize if the guest switches their door, regardless of the door they initially went with.
# 
# We can reproduce this result using Bayesian networks with three nodes, one for the guest, one for the prize, and one for the door Monty chooses to open. The door the guest initially chooses and the door the prize is behind are completely random processes across the three doors, but the door which Monty opens is dependent on both the door the guest chooses (it cannot be the door the guest chooses), and the door the prize is behind (it cannot be the door with the prize behind it).
# 
# To create the Bayesian network in pomegranate, we first create the distributions which live in each node in the graph. For a discrete (aka categorical) bayesian network we use DiscreteDistribution objects for the root nodes and ConditionalProbabilityTable objects for the inner and leaf nodes. The columns in a ConditionalProbabilityTable correspond to the order in which the parents (the second argument) are specified, and the last column is the value the ConditionalProbabilityTable itself takes. In the case below, the first column corresponds to the value 'guest' takes, then the value 'prize' takes, and then the value that 'monty' takes. 'B', 'C', 'A' refers then to the probability that Monty reveals door 'A' given that the guest has chosen door 'B' and that the prize is actually behind door 'C', or P(Monty='A'|Guest='B', Prize='C').

# The guests initial door selection is completely random
guest = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})

# The door the prize is behind is also completely random
prize = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})

    # Monty is dependent on both the guest and the prize. 
monty = ConditionalProbabilityTable(
        [[ 'A', 'A', 'A', 0.0 ],
         [ 'A', 'A', 'B', 0.5 ],
         [ 'A', 'A', 'C', 0.5 ],
         [ 'A', 'B', 'A', 0.0 ],
         [ 'A', 'B', 'B', 0.0 ],
         [ 'A', 'B', 'C', 1.0 ],
         [ 'A', 'C', 'A', 0.0 ],
         [ 'A', 'C', 'B', 1.0 ],
         [ 'A', 'C', 'C', 0.0 ],
         [ 'B', 'A', 'A', 0.0 ],
         [ 'B', 'A', 'B', 0.0 ],
         [ 'B', 'A', 'C', 1.0 ],
         [ 'B', 'B', 'A', 0.5 ],
         [ 'B', 'B', 'B', 0.0 ],
         [ 'B', 'B', 'C', 0.5 ],
         [ 'B', 'C', 'A', 1.0 ],
         [ 'B', 'C', 'B', 0.0 ],
         [ 'B', 'C', 'C', 0.0 ],
         [ 'C', 'A', 'A', 0.0 ],
         [ 'C', 'A', 'B', 1.0 ],
         [ 'C', 'A', 'C', 0.0 ],
         [ 'C', 'B', 'A', 1.0 ],
         [ 'C', 'B', 'B', 0.0 ],
         [ 'C', 'B', 'C', 0.0 ],
         [ 'C', 'C', 'A', 0.5 ],
         [ 'C', 'C', 'B', 0.5 ],
         [ 'C', 'C', 'C', 0.0 ]], [guest, prize])  


# Next, we pass these distributions into state objects along with the name for the node.

# State objects hold both the distribution, and a high level name.
s1 = State(guest, name="guest")
s2 = State(prize, name="prize")
s3 = State(monty, name="monty")


# Then we add the states to the network, exactly like we did when making a HMM. In the future, all matrices of data should have their columns organized in the same order that the states are added to the network. The way the states are added to the network makes no difference to it, and so you should add the states according to how the columns are organized in your data.

# Create the Bayesian network object with a useful name
model = BayesianNetwork("Monty Hall Problem")

# Add the three states to the network 
model.add_states(s1, s2, s3)


# Then we need to add edges to the model. The edges represent which states are parents of which other states. This is currently a bit redundant with passing in the distribution objects that are parents for each ConditionalProbabilityTable. For now edges are added from parent -> child by calling `model.add_edge(parent, child)`.

# Add edges which represent conditional dependencies, where the second node is 
# conditionally dependent on the first node (Monty is dependent on both guest and prize)
model.add_edge(s1, s3)
model.add_edge(s2, s3)


# Lastly, the model must be baked to finalize the internals. Since Bayesian networks use factor graphs for inference, an explicit factor graph is produced from the Bayesian network during the bake step.

model.bake()


# #### Predicting Probabilities
# 
# We can calculate probabilities of a sample under the Bayesian network in the same way that we can calculate probabilities under other models. In this case, let's calculate the probability that you initially said door A, that Monty then opened door B, but that the actual car was behind door C.

model.probability(['A', 'B', 'C'])


# That seems in line with what we know, that there is a 1/9th probability of that happening. 
# 
# Next, let's look at an impossible situation. What is the probability of initially saying door A, that Monty opened door B, and that the car was actually behind door B.

model.probability(['A', 'B', 'B'])


# The reason that situation is impossible is because Monty can't open a door that has the car behind it.

# #### Performing Inference
# 
# Perhaps the most powerful aspect of Bayesian networks is their ability to perform inference. Given any set of observed variables, including no observations, Bayesian networks can make predictions for all other variables. Obviously, the more variables that are observed, the more accurate the predictions will get of the remaining variables. 
# 
# pomegranate uses the loopy belief propagation algorithm to do inference. This is an approximate algorithm which can yield exact results in tree-like graphs, and in most other cases still yields good results. Inference on a Bayesian network consists of taking in observations for a subset of the variables and using that to infer the values that the other variables take. The most variables which are observed, the closer the inferred values will be to truth. One of the powers of Bayesian networks is that the set of observed and 'hidden' (or unobserved) variables does not need to be specified beforehand, and can change from sample to sample.
# 
# We can run inference using the `predict_proba` method and passing in a dictionary of values, where the key is the name of the state and the value is the observed value for that state. If we don't supply any values, we get the marginal of the graph, which is just the frequency of each value for each variable over an infinite number of randomly drawn samples from the graph.
# 
# Lets see what happens when we look at the marginal of the Monty hall network.

model.predict_proba({})


# We are returned three `DiscreteDistribution` objects, each representing the marginal distribution for each variable, in the same order they were put into the model. In this case, they represent the guest, prize, and monty variables respectively. We see that everything is equally likely.
# 
# We can also pass in an array where `None` (or `np.nan`) correspond to the unobserved values.

model.predict_proba([None, None, None])


# All of the variables that were observed will be the observed value, and all of the variables that were unobserved will be a `DiscreteDistribution` object. This means that `parameters[0]` will return the underlying dictionary used by the distribution.
# 
# Now lets do something different, and say that the guest has chosen door 'A'. We do this by passing a dictionary to `predict_proba` with key pairs consisting of the name of the state (in the state object), and the value which that variable has taken, or by passing in a list where the first index is our observation.

model.predict_proba(['A', None, None])


# We can see that now Monty will not open door 'A', because the guest has chosen it. At the same time, the distribution over the prize has not changed, it is still equally likely that the prize is behind each door.
# 
# Now, lets say that Monty opens door 'C' and see what happens. Here we use a dictionary rather than a list simply to show how one can use both input forms depending on what is more convenient.

model.predict_proba({'guest': 'A', 'monty': 'C'})


# Suddenly, we see that the distribution over prizes has changed. It is now twice as likely that the car is behind the door labeled 'B'. This demonstrates that when on the game show, it is always better to change your initial guess after being shown an open door. Now you could go and win tons of cars, except that the game show got cancelled.

# ### Imputation Given Structured Constraints
# 
# The task of filling in an incomplete matrix can be called imputation, and there are many approaches for doing so. One of the most well known is that of matrix factorization, where a latent representation is learned for each of the columns and each of the rows such that the dot product between the two can reconstruct values in the matrix. Due to the manner that these latent representations are learned, the matrix does not need to be complete, and the dot product can then be used to fill in all of the missing values.
# 
# One weakness of the matrix factorization approach is that constraints and known relationships can't be enforced between the features. A simple example is that the rule "when column 1 is 'A' and column 2 is 'B', column 3 must be 'C'" can potentially be learned in the representation, but can't be simply hard-coded like a conditional probability table would. A more complex example would say that a pixel in an image can be predicted from its neighbors, whereas the notion of neighbors is more difficult to specify for a factorization approach.
# 
# The process for imputing data given a Bayesian network is to either first learn the structure of the network from the given data, or have a known structure, and then use loopy-belief propogation to predict the best values for the missing features.
# 
# Let's see an example of this on the digits data set, binarizing the data based on the median value. We'll only use the first two rows because learning large, unconstrained Bayesian networks is difficult. 

from sklearn.datasets import load_digits

data = load_digits()
X, _ = data.data, data.target

plt.imshow(X[0].reshape(8, 8))
plt.grid(False)
plt.show()

X = X[:,:16]
X = (X > numpy.median(X)).astype('float64')


# Now let's remove a large portion of the pixels randomly from each of the images. We can do that with numpy arrays by setting missing values to `np.nan`.

numpy.random.seed(111)

i = numpy.random.randint(X.shape[0], size=10000)
j = numpy.random.randint(X.shape[1], size=10000)

X_missing = X.copy()
X_missing[i, j] = numpy.nan
X_missing


# We can set up a baseline for how good an imputation is by using the average pixel value as a replacement. Because this is binary data, we can use the mean absolute error to measure how good these approaches are on imputing the pixels that are not observed.

from fancyimpute import SimpleFill

y_pred = SimpleFill().fit_transform(X_missing)[i, j]
numpy.abs(y_pred - X[i, j]).mean()


# Next, we can see how good an IterativeSVD approach is, which is similar to a matrix factorization.

from fancyimpute import IterativeSVD

y_pred = IterativeSVD(verbose=False).fit_transform(X_missing)[i, j]
numpy.abs(y_pred - X[i, j]).mean()


# Now, we can try building a Bayesian network using the Chow-Liu algorithm and then use the resulting network to fill in the matrix.

y_hat = BayesianNetwork.from_samples(X_missing, max_parents=1).predict(X_missing)
numpy.abs(numpy.array(y_hat)[i, j] - X[i, j]).mean()


# We can compare this to a better imputation approach, that of K-nearest neighbors, and see how good using a Bayesian network is.

from fancyimpute import KNN

y_pred = KNN(verbose=False).fit_transform(X_missing)[i, j]
numpy.abs(y_pred - X[i, j]).mean()


# Looks like in this case the Bayesian network is better than KNN for imputing the pixels. It is, however, slower than the fancyimpute methods.

# ## The API
# 
# ### Initialization
# 
# While the methods are similar across all models in pomegranate, Bayesian networks are more closely related to hidden Markov models than the other models. This makes sense, because both of them rely on graphical structures.
# 
# The first way to initialize Bayesian networks is to pass in one distribution and state at a time, and then pass in edges. This is similar to hidden Markov models.

d1 = DiscreteDistribution({True: 0.2, False: 0.8})
d2 = DiscreteDistribution({True: 0.6, False: 0.4})
d3 = ConditionalProbabilityTable(
        [[True,  True,  True,  0.2],
         [True,  True,  False, 0.8],
         [True,  False, True,  0.3],
         [True,  False, False, 0.7],
         [False, True,  True,  0.9],
         [False, True,  False, 0.1],
         [False, False, True,  0.4],
         [False, False, False, 0.6]], [d1, d2])

s1 = State(d1, name="s1")
s2 = State(d2, name="s2")
s3 = State(d3, name="s3")

model = BayesianNetwork()
model.add_states(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.bake()


# The other way is to use the `from_samples` method if given a data set.

numpy.random.seed(111)

X = numpy.random.randint(2, size=(15, 15))
X[:,5] = X[:,4] = X[:,3]
X[:,11] = X[:,12] = X[:,13]

model = BayesianNetwork.from_samples(X)
model.plot()


# The structure learning algorithms are covered more in depth in the accompanying notebook.
# 
# We can look at the structure of the network by using the `structure` attribute. Each tuple is a node, and the integers in the tuple correspond to the parents of the node.

model.structure


# ### Prediction
# 
# The prediction method is similar to the other models. Inference is done using loopy belief propogation, which is an approximate version of the forward-backward algorithm that can be significantly faster while still precise. 

model.predict([[False, False, False, False, None, None, False, None, False, None, True, None, None, True, False]])


# The predict method will simply return the argmax of all the distributions after running the `predict_proba` method. 

model.predict_proba([[False, False, False, False, None, None, False, None, False, None, 
                      True, None, None, True, False]])


# # Semi-supervised Learning in pomegranate
# 
# Most classical machine learning algorithms either assume that an entire dataset is either labeled (supervised learning) or that there are no labels (unsupervised learning). However, frequently it is the case that some labeled data is present but there is a great deal of unlabeled data as well. A great example of this is that of computer vision where the internet is filled of pictures (mostly of cats) that could be useful, but you don't have the time or money to label them all in accordance with your specific task. Typically what ends up happening is that either the unlabeled data is discarded in favor of training a model solely on the labeled data, or that an unsupervised model is initialized with the labeled data and then set free on the unlabeled data. Neither method uses both sets of data in the optimization process.
# 
# Semi-supervised learning is a method to incorporate both labeled and unlabeled data into the training task, typically yield better performing estimators than using the labeled data alone. There are many methods one could use for semisupervised learning, and <a href="http://scikit-learn.org/stable/modules/label_propagation.html">scikit-learn has a good write-up on some of these techniques</a>.
# 
# pomegranate natively implements semi-supervised learning through the a merger of maximum-likelihood and expectation-maximization. As an overview, the models are initialized by first fitting to the labeled data directly using maximum-likelihood estimates. The models are then refined by running expectation-maximization (EM) on the unlabeled datasets and adding the sufficient statistics to those acquired from maximum-likelihood estimates on the labeled data. Under the hood both a supervised model and an unsupervised mixture model are created using the same underlying distribution objects. The summarize method is first called using the supervised method on the labeled data, and then the summarize method is called again using the unsupervised method on the unlabeled data. This causes the sufficient statistics to be updated appropriately given the results of first maximum-likelihood and then EM. This process continues until convergence in the EM step.
# 
# Let's take a look!

get_ipython().run_line_magic('pylab', 'inline')
from pomegranate import *
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import make_blobs
import seaborn, time
seaborn.set_style('whitegrid')
numpy.random.seed(1)


# Let's first generate some data in the form of blobs that are close together. Generally one tends to have far more unlabeled data than labeled data, so let's say that a person only has 50 samples of labeled training data and 4950 unlabeled samples. In pomegranate you a sample can be specified as lacking a label by providing the integer -1 as the label, just like in scikit-learn. Let's also say there there is a bit of bias in the labeled samples to inject some noise into the problem, as otherwise Gaussian blobs are trivially modeled with even a few samples.

X, y = make_blobs(10000, 2, 3, cluster_std=2)
x_min, x_max = X[:,0].min()-2, X[:,0].max()+2
y_min, y_max = X[:,1].min()-2, X[:,1].max()+2

X_train = X[:5000]
y_train = y[:5000]

# Set the majority of samples to unlabeled.
y_train[numpy.random.choice(5000, size=4950, replace=False)] = -1

# Inject noise into the problem
X_train[y_train != -1] += 2.5

X_test = X[5000:]
y_test = y[5000:]


# Now let's take a look at the data when we plot it.

plt.figure(figsize=(8, 8))
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color='0.6')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='c')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='m')
plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='r')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


# The clusters of unlabeled data seem clear to us, and it doesn't seem like the labeled data is perfectly faithful to these clusters. This can typically happen in a semisupervised setting as well, as the data that is labeled is sometimes biased either because the labeled data was chosen as it was easy to label, or the data was chosen to be labeled in a biased maner.
# 
# Now let's try fitting a simple naive Bayes classifier to this data and compare the results when using only the labeled data to when using both the labeled and unlabeled data together.

model_a = NaiveBayes.from_samples(NormalDistribution, X_train[y_train != -1], y_train[y_train != -1])
print "Supervised Learning Accuracy: {}".format((model_a.predict(X_test) == y_test).mean())

model_b = NaiveBayes.from_samples(NormalDistribution, X_train, y_train)
print "Semisupervised Learning Accuracy: {}".format((model_b.predict(X_test) == y_test).mean())


# It seems like we get a big bump in test set accuracy when we use semi-supervised learning. Let's visualize the data to get a better sense of what is happening here.

def plot_contour(X, y, Z):
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='0.2', alpha=0.5, s=20)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', s=20)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', s=20)
    plt.scatter(X[y == 2, 0], X[y == 2, 1], color='r', s=20)
    plt.contour(xx, yy, Z)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.1), numpy.arange(y_min, y_max, 0.1))
Z1 = model_a.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(16, 16))
plt.subplot(221)
plt.title("Training Data, Supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z1)

plt.subplot(223)
plt.title("Training Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z2)

plt.subplot(222)
plt.title("Test Data, Supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z1)

plt.subplot(224)
plt.title("Test Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z2)
plt.show()


# The contours plot the decision boundaries between the different classes with the left figures corresponding to the partially labeled training set and the right figures corresponding to the test set. We can see that the boundaries learning using only the labeled data look a bit weird when considering the unlabeled data, particularly in that it doesn't cleanly separate the cyan cluster from the other two. In addition, it seems like the boundary between the magenta and red clusters is a bit curved in an unrealistic way. We would not expect points that fell around (-18, -7) to actually come from the red class. Training the model in a semi-supervised manner cleaned up both of these concerns by learning better boundaries that are also flatter and more generalizable.
# 
# Let's next compare the training times to see how much slower it is to do semi-supervised learning than it is to do simple supervised learning.

print "Supervised Learning: "
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X_train[y_train != -1], y_train[y_train != -1])')
print
print "Semi-supervised Learning: "
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X_train, y_train)')
print
print "Label Propagation (sklearn): "
get_ipython().run_line_magic('timeit', 'LabelPropagation().fit(X_train, y_train)')


# It is quite a bit slower to do semi-supervised learning than simple supervised learning in this example. This is expected as the simple supervised update for naive Bayes is a trivial MLE across each dimension whereas the semi-supervised case requires EM to converge to complete. However, it is still faster to do semi-supervised learning this setting to learn a naive Bayes classifier than it is to fit the label propagation estimator from sklearn. 
# 
# However, though it is widely used, the naive Bayes classifier is still a fairly simple model. One can construct a more complicated model that does not assume feature independence called a Bayes classifier that can also be trained using semi-supervised learning in pretty much the same manner. You can read more about the Bayes classifier in its tutorial in the tutorial folder. Let's move on to more complicated data and try to fit a mixture model Bayes classifier, comparing the performance between using only labeled data and using all data.
# 
# First let's generate some more complicated, noisier data.

X = numpy.empty(shape=(0, 2))
X = numpy.concatenate((X, numpy.random.normal(4, 1, size=(3000, 2)).dot([[-2, 0.5], [2, 0.5]])))
X = numpy.concatenate((X, numpy.random.normal(3, 1, size=(6500, 2)).dot([[-1, 2], [1, 0.8]])))
X = numpy.concatenate((X, numpy.random.normal(7, 1, size=(8000, 2)).dot([[-0.75, 0.8], [0.9, 1.5]])))
X = numpy.concatenate((X, numpy.random.normal(6, 1, size=(2200, 2)).dot([[-1.5, 1.2], [0.6, 1.2]])))
X = numpy.concatenate((X, numpy.random.normal(8, 1, size=(3500, 2)).dot([[-0.2, 0.8], [0.7, 0.8]])))
X = numpy.concatenate((X, numpy.random.normal(9, 1, size=(6500, 2)).dot([[-0.0, 0.8], [0.5, 1.2]])))
x_min, x_max = X[:,0].min()-2, X[:,0].max()+2
y_min, y_max = X[:,1].min()-2, X[:,1].max()+2

y = numpy.concatenate((numpy.zeros(9500), numpy.ones(10200), numpy.ones(10000)*2))
idxs = numpy.arange(29700)
numpy.random.shuffle(idxs)

X = X[idxs]
y = y[idxs]

X_train, X_test = X[:25000], X[25000:]
y_train, y_test = y[:25000], y[25000:]
y_train[numpy.random.choice(25000, size=24920, replace=False)] = -1

plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color='0.6', s=1)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='c', s=10)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='m', s=10)
plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='r', s=10)
plt.show()


# Now let's take a look at the accuracies that we get when training a model using just the labeled examples versus all of the examples in a semi-supervised manner.

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model_a = BayesClassifier([d1, d2, d3]).fit(X_train[y_train != -1], y_train[y_train != -1])
print "Supervised Learning Accuracy: {}".format((model_a.predict(X_test) == y_test).mean())

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model_b = BayesClassifier([d1, d2, d3])
model_b.fit(X_train, y_train)
print "Semisupervised Learning Accuracy: {}".format((model_b.predict(X_test) == y_test).mean())


# As expected, the semi-supervised method performs better. Let's visualize the landscape in the same manner as before in order to see why this is the case.

xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.1), numpy.arange(y_min, y_max, 0.1))
Z1 = model_a.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(16, 16))
plt.subplot(221)
plt.title("Training Data, Supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z1)

plt.subplot(223)
plt.title("Training Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_train, y_train, Z2)

plt.subplot(222)
plt.title("Test Data, Supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z1)

plt.subplot(224)
plt.title("Test Data, Semi-supervised Boundaries", fontsize=16)
plot_contour(X_test, y_test, Z2)
plt.show()


# Immediately, one would notice that the decision boundaries when using semi-supervised learning are smoother than those when using only a few samples. This can be explained mostly because having more data can generally lead to smoother decision boundaries as the model does not overfit to spurious examples in the dataset. It appears that the majority of the correctly classified samples come from having a more accurate decision boundary for the magenta samples in the left cluster. When using only the labeled samples many of the magenta samples in this region get classified incorrectly as cyan samples. In contrast, when using all of the data these points are all classified correctly.
# 
# Lastly, let's take a look at a time comparison in this more complicated example.

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model = BayesClassifier([d1, d2, d3])

print "Supervised Learning: "
get_ipython().run_line_magic('timeit', 'model.fit(X_train[y_train != -1], y_train[y_train != -1])')
print

d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 0], max_iterations=1)
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 1], max_iterations=1)
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_train[y_train == 2], max_iterations=1)
model = BayesClassifier([d1, d2, d3])

print "Semi-supervised Learning: "
get_ipython().run_line_magic('timeit', 'model.fit(X_train, y_train)')

print
print "Label Propagation (sklearn): "
get_ipython().run_line_magic('timeit', 'LabelPropagation().fit(X_train, y_train)')


# It looks like the difference, while still large, is not as large as in the previous example, being only a ~40x difference instead of a ~1000x difference. This is likely because even without the unlabeled data the supervised model is performing EM to train each of the mixtures that are the components of the Bayes classifier. Again, it is faster to do semi-supervised learning in this manner for generative models than it is to perform LabelPropagation.

# ## Summary
# 
# In the real world (ack) there are frequently situations where only a small fraction of the available data has useful labels. Semi-supervised learning provides a framework for leveraging both the labeled and unlabeled aspects of a dataset to learn a sophisticated estimator. In this case, semi-supervised learning plays well with probabilistic models as normal maximum likelihood estimates can be done on the labeled data and expectation-maximization can be run on the unlabeled data using the same distributions.
# 
# This notebook has covered how to implement semi-supervised learning in pomegranate using both naive Bayes and a Bayes classifier. All one has to do is set the labels of unlabeled samples to -1 and pomegranate will take care of the rest. This can be particularly useful when encountering complex, noisy, data in the real world that aren't neat Gaussian blobs.

