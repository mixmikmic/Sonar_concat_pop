# # Batch Training
# 
# Running algorithms which require the full data set for each update
# can be expensive when the data is large. In order to scale inferences,
# we can do _batch training_. This trains the model using
# only a subsample of data at a time.
# 
# In this tutorial, we extend the
# [supervised learning tutorial](http://edwardlib.org/tutorials/supervised-regression), 
# where the task is to infer hidden structure from
# labeled examples $\{(x_n, y_n)\}$.
# A webpage version is available at
# http://edwardlib.org/tutorials/batch-training.
# 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


# ## Data
# 
# Simulate $N$ training examples and a fixed number of test examples.
# Each example is a pair of inputs $\mathbf{x}_n\in\mathbb{R}^{10}$ and
# outputs $y_n\in\mathbb{R}$. They have a linear dependence with
# normally distributed noise.
# 
# We also define a helper function to select the next batch of data
# points from the full set of examples. It keeps track of the current
# batch index and returns the next batch using the function 
# ``next()``. We will generate batches from `data` during inference.
# 

def build_toy_dataset(N, w):
  D = len(w)
  x = np.random.normal(0.0, 2.0, size=(N, D))
  y = np.dot(x, w) + np.random.normal(0.0, 0.05, size=N)
  return x, y


def generator(arrays, batch_size):
  """Generate batches, one with respect to each array's first axis."""
  starts = [0] * len(arrays)  # pointers to where we are in iteration
  while True:
    batches = []
    for i, array in enumerate(arrays):
      start = starts[i]
      stop = start + batch_size
      diff = stop - array.shape[0]
      if diff <= 0:
        batch = array[start:stop]
        starts[i] += batch_size
      else:
        batch = np.concatenate((array[start:], array[:diff]))
        starts[i] = diff
      batches.append(batch)
    yield batches


ed.set_seed(42)

N = 10000  # size of training data
M = 128    # batch size during training
D = 10     # number of features

w_true = np.ones(D) * 5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(235, w_true)

data = generator([X_train, y_train], M)


# ## Model
# 
# Posit the model as Bayesian linear regression (Murphy, 2012).
# For a set of $N$ data points $(\mathbf{X},\mathbf{y})=\{(\mathbf{x}_n, y_n)\}$,
# the model posits the following distributions:
# 
# \begin{align*}
#   p(\mathbf{w})
#   &=
#   \text{Normal}(\mathbf{w} \mid \mathbf{0}, \sigma_w^2\mathbf{I}),
#   \\[1.5ex]
#   p(b)
#   &=
#   \text{Normal}(b \mid 0, \sigma_b^2),
#   \  p(\mathbf{y} \mid \mathbf{w}, b, \mathbf{X})
#   &=
#   \prod_{n=1}^N
#   \text{Normal}(y_n \mid \mathbf{x}_n^\top\mathbf{w} + b, \sigma_y^2).
# \end{align*}
# 
# The latent variables are the linear model's weights $\mathbf{w}$ and
# intercept $b$, also known as the bias.
# Assume $\sigma_w^2,\sigma_b^2$ are known prior variances and $\sigma_y^2$ is a
# known likelihood variance. The mean of the likelihood is given by a
# linear transformation of the inputs $\mathbf{x}_n$.
# 
# Let's build the model in Edward, fixing $\sigma_w,\sigma_b,\sigma_y=1$. 
# 

X = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=1.0)


# Here, we define a placeholder `X`. During inference, we pass in
# the value for this placeholder according to batches of data.
# To enable training with batches of varying size, 
# we don't fix the number of rows for `X` and `y`. (Alternatively,
# we could fix it to be the batch size if training and testing 
# with a fixed size.)
# 

# ## Inference
# 
# We now turn to inferring the posterior using variational inference.
# Define the variational model to be a fully factorized normal across
# the weights.
# 

qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


# Run variational inference with the Kullback-Leibler divergence.
# We use $5$ latent variable samples for computing
# black box stochastic gradients in the algorithm.
# (For more details, see the
# [$\text{KL}(q\|p)$ tutorial](http://edwardlib.org/tutorials/klqp).)
# 
# For batch training, we will iterate over the number of batches and
# feed them to the respective placeholder. We set the number of
# iterations to be equal to the number of batches times the number of
# epochs (full passes over the data set).
# 

n_batch = int(N / M)
n_epoch = 5

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})
inference.initialize(n_iter=n_batch * n_epoch, n_samples=5, scale={y: N / M})
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  X_batch, y_batch = next(data)
  info_dict = inference.update({X: X_batch, y_ph: y_batch})
  inference.print_progress(info_dict)


# When initializing inference, note we scale $y$ by $N/M$, so it is as if the
# algorithm had seen $N/M$ as many data points per iteration.
# Algorithmically, this will scale all computation regarding $y$ by
# $N/M$ such as scaling the log-likelihood in a variational method's
# objective. (Statistically, this avoids inference being dominated by the prior.)
# 
# The loop construction makes training very flexible. For example, we
# can also try running many updates for each batch.
# 

n_batch = int(N / M)
n_epoch = 1

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})
inference.initialize(n_iter=n_batch * n_epoch * 10, n_samples=5, scale={y: N / M})
tf.global_variables_initializer().run()

for _ in range(inference.n_iter // 10):
  X_batch, y_batch = next(data)
  for _ in range(10):
    info_dict = inference.update({X: X_batch, y_ph: y_batch})

  inference.print_progress(info_dict)


# In general, make sure that the total number of training iterations is 
# specified correctly when initializing `inference`. Otherwise an incorrect
# number of training iterations can have unintended consequences; for example,
# `ed.KLqp` uses an internal counter to appropriately decay its optimizer's 
# learning rate step size.
# 
# Note also that the reported `loss` value as we run the
# algorithm corresponds to the computed objective given the current
# batch and not the total data set. We can instead have it report
# the loss over the total data set by summing `info_dict['loss']`
# for each epoch.
# 

# ## Criticism
# 
# A standard evaluation for regression is to compare prediction accuracy on
# held-out "testing" data. We do this by first forming the posterior predictive
# distribution.
# 

y_post = ed.copy(y, {w: qw, b: qb})
# This is equivalent to
# y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(N))


# With this we can evaluate various quantities using predictions from
# the model (posterior predictive).
# 

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))


# ## Footnotes
# 
# Only certain algorithms support batch training such as
# `MAP`, `KLqp`, and `SGLD`. Also, above we
# illustrated batch training for models with only global latent variables,
# which are variables are shared across all data points.
# For more complex strategies, see the
# [inference data subsampling API](http://edwardlib.org/api/inference-data-subsampling).
# 

# # Your first Edward program
# 
# Probabilistic modeling in Edward uses a simple language of random variables. Here we will show a Bayesian neural network. It is a neural network with a prior distribution on its weights.
# 
# A webpage version is available at 
# http://edwardlib.org/getting-started.
# 

get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

plt.style.use('ggplot')


def build_toy_dataset(N=50, noise_std=0.1):
  x = np.linspace(-3, 3, num=N)
  y = np.cos(x) + np.random.normal(0, noise_std, size=N)
  x = x.astype(np.float32).reshape((N, 1))
  y = y.astype(np.float32)
  return x, y


def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.matmul(h, W_1) + b_1
  return tf.reshape(h, [-1])


# First, simulate a toy dataset of 50 observations with a cosine relationship.
# 

ed.set_seed(42)

N = 50  # number of data ponts
D = 1   # number of features

x_train, y_train = build_toy_dataset(N)


# Next, define a two-layer Bayesian neural network. Here, we define the neural network manually with `tanh` nonlinearities.
# 

W_0 = Normal(loc=tf.zeros([D, 2]), scale=tf.ones([D, 2]))
W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

x = x_train
y = Normal(loc=neural_network(x, W_0, W_1, b_0, b_1),
           scale=0.1 * tf.ones(N))


# Next, make inferences about the model from data. We will use variational inference. Specify a normal approximation over the weights and biases.
# 

qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 2])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, 2]))))
qW_1 = Normal(loc=tf.Variable(tf.random_normal([2, 1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([2, 1]))))
qb_0 = Normal(loc=tf.Variable(tf.random_normal([2])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([2]))))
qb_1 = Normal(loc=tf.Variable(tf.random_normal([1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


# Defining `tf.Variable` allows the variational factors’ parameters to vary. They are initialized randomly. The standard deviation parameters are constrained to be greater than zero according to a [softplus](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) transformation.
# 

# Sample functions from variational model to visualize fits.
rs = np.random.RandomState(0)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(inputs, 1)
mus = tf.stack(
    [neural_network(x, qW_0.sample(), qW_1.sample(),
                    qb_0.sample(), qb_1.sample())
     for _ in range(10)])


# FIRST VISUALIZATION (prior)

sess = ed.get_session()
tf.global_variables_initializer().run()
outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()


# Now, run variational inference with the [Kullback-Leibler](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) divergence in order to infer the model’s latent variables with the given data. We specify `1000` iterations.
# 

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data={y: y_train})
inference.run(n_iter=1000, n_samples=5)


# Finally, criticize the model fit. Bayesian neural networks define a distribution over neural networks, so we can perform a graphical check. Draw neural networks from the inferred model and visualize how well it fits the data.
# 

# SECOND VISUALIZATION (posterior)

outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 1000")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()


# The model has captured the cosine relationship between $x$ and $y$ in the observed domain.
# 
# 
# To learn more about Edward, [delve in](http://edwardlib.org/api)!
# 
# If you prefer to learn via examples, then check out some
# [tutorials](http://edwardlib.org/tutorials/).
# 

# # Supervised Learning (Regression)
# 
# In supervised learning, the task is to infer hidden structure from
# labeled data, comprised of training examples $\{(x_n, y_n)\}$.
# Regression typically means the output $y$ takes continuous values.
# 
# We demonstrate with an example in Edward. A webpage version is available at
# http://edwardlib.org/tutorials/supervised-regression.
# 

get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

plt.style.use('ggplot')


# ## Data
# 
# Simulate training and test sets of $40$ data points. They comprise of
# pairs of inputs $\mathbf{x}_n\in\mathbb{R}^{10}$ and outputs
# $y_n\in\mathbb{R}$. They have a linear dependence with normally
# distributed noise.
# 

def build_toy_dataset(N, w):
  D = len(w)
  x = np.random.normal(0.0, 2.0, size=(N, D))
  y = np.dot(x, w) + np.random.normal(0.0, 0.01, size=N)
  return x, y


ed.set_seed(42)

N = 40  # number of data points
D = 10  # number of features

w_true = np.random.randn(D) * 0.5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)


# ## Model
# 
# Posit the model as Bayesian linear regression (Murphy, 2012).
# It assumes a linear relationship between the inputs
# $\mathbf{x}\in\mathbb{R}^D$ and the outputs $y\in\mathbb{R}$.
# 
# For a set of $N$ data points $(\mathbf{X},\mathbf{y})=\{(\mathbf{x}_n, y_n)\}$,
# the model posits the following distributions:
# 
# \begin{align*}
#   p(\mathbf{w})
#   &=
#   \text{Normal}(\mathbf{w} \mid \mathbf{0}, \sigma_w^2\mathbf{I}),
#   \\[1.5ex]
#   p(b)
#   &=
#   \text{Normal}(b \mid 0, \sigma_b^2),
#   \  p(\mathbf{y} \mid \mathbf{w}, b, \mathbf{X})
#   &=
#   \prod_{n=1}^N
#   \text{Normal}(y_n \mid \mathbf{x}_n^\top\mathbf{w} + b, \sigma_y^2).
# \end{align*}
# 
# The latent variables are the linear model's weights $\mathbf{w}$ and
# intercept $b$, also known as the bias.
# Assume $\sigma_w^2,\sigma_b^2$ are known prior variances and $\sigma_y^2$ is a
# known likelihood variance. The mean of the likelihood is given by a
# linear transformation of the inputs $\mathbf{x}_n$.
# 
# Let's build the model in Edward, fixing $\sigma_w,\sigma_b,\sigma_y=1$.
# 

X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))


# Here, we define a placeholder `X`. During inference, we pass in
# the value for this placeholder according to data.
# 

# ## Inference
# 
# We now turn to inferring the posterior using variational inference.
# Define the variational model to be a fully factorized normal across
# the weights.
# 

qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


# Run variational inference with the Kullback-Leibler divergence, using 
# $250$ iterations and $5$ latent variable samples in the algorithm.
# 

inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run(n_samples=5, n_iter=250)


# In this case `KLqp` defaults to minimizing the
# $\text{KL}(q\|p)$ divergence measure using the reparameterization
# gradient.
# For more details on inference, see the [$\text{KL}(q\|p)$ tutorial](http://edwardlib.org/tutorials/klqp).
# 

# ## Criticism
# 
# A standard evaluation for regression is to compare prediction accuracy on
# held-out "testing" data. We do this by first forming the posterior predictive
# distribution.
# 

y_post = ed.copy(y, {w: qw, b: qb})
# This is equivalent to
# y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(N))


# With this we can evaluate various quantities using predictions from
# the model (posterior predictive).
# 

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))


# The trained model makes predictions with low error
# (relative to the magnitude of the output).
# 
# We can also visualize the fit by comparing data generated with the
# prior to data generated with the posterior (on the first feature
# dimension).
# 

def visualise(X_data, y_data, w, b, n_samples=10):
  w_samples = w.sample(n_samples)[:, 0].eval()
  b_samples = b.sample(n_samples).eval()
  plt.scatter(X_data[:, 0], y_data)
  plt.ylim([-10, 10])
  inputs = np.linspace(-8, 8, num=400)
  for ns in range(n_samples):
    output = inputs * w_samples[ns] + b_samples[ns]
    plt.plot(inputs, output)


# Visualize samples from the prior.
visualise(X_train, y_train, w, b, n_samples=10)


# Visualize samples from the posterior.
visualise(X_train, y_train, qw, qb, n_samples=10)


# The model has learned a linear relationship between the
# first dimension of $\mathbf{x}\in\mathbb{R}^D$ and the outputs
# $y\in\mathbb{R}$.
# 

# # Supervised Learning (Classification)
# 
# In supervised learning, the task is to infer hidden structure from
# labeled data, comprised of training examples $\{(x_n, y_n)\}$.
# Classification means the output $y$ takes discrete values.
# 
# We demonstrate with an example in Edward. A webpage version is available at
# http://edwardlib.org/tutorials/supervised-classification.
# 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, MultivariateNormalTriL, Normal
from edward.util import rbf


# ## Data
# 
# Use the
# [crabs data set](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/crabs.html),
# which consists of morphological measurements on a crab species. We
# are interested in predicting whether a given crab has the color form
# blue or orange.
# 

ed.set_seed(42)

data = np.loadtxt('data/crabs_train.txt', delimiter=',')
data[data[:, 0] == -1, 0] = 0  # replace -1 label with 0 label

N = data.shape[0]  # number of data points
D = data.shape[1] - 1  # number of features

X_train = data[:, 1:]
y_train = data[:, 0]

print("Number of data points: {}".format(N))
print("Number of features: {}".format(D))


# ## Model
# 
# A Gaussian process is a powerful object for modeling nonlinear
# relationships between pairs of random variables. It defines a distribution over
# (possibly nonlinear) functions, which can be applied for representing
# our uncertainty around the true functional relationship.
# Here we define a Gaussian process model for classification
# (Rasumussen & Williams, 2006).
# 
# Formally, a distribution over functions $f:\mathbb{R}^D\to\mathbb{R}$ can be specified
# by a Gaussian process
# $$
# \begin{align*}
#   p(f)
#   &=
#   \mathcal{GP}(f\mid \mathbf{0}, k(\mathbf{x}, \mathbf{x}^\prime)),
# \end{align*}
# $$
# whose mean function is the zero function, and whose covariance
# function is some kernel which describes dependence between
# any set of inputs to the function.
# 
# Given a set of input-output pairs
# $\{\mathbf{x}_n\in\mathbb{R}^D,y_n\in\mathbb{R}\}$,
# the likelihood can be written as a multivariate normal
# 
# \begin{align*}
#   p(\mathbf{y})
#   &=
#   \text{Normal}(\mathbf{y} \mid \mathbf{0}, \mathbf{K})
# \end{align*}
# 
# where $\mathbf{K}$ is a covariance matrix given by evaluating
# $k(\mathbf{x}_n, \mathbf{x}_m)$ for each pair of inputs in the data
# set.
# 
# The above applies directly for regression where $\mathbb{y}$ is a
# real-valued response, but not for (binary) classification, where $\mathbb{y}$
# is a label in $\{0,1\}$. To deal with classification, we interpret the
# response as latent variables which is squashed into $[0,1]$. We then
# draw from a Bernoulli to determine the label, with probability given
# by the squashed value.
# 
# Define the likelihood of an observation $(\mathbf{x}_n, y_n)$ as
# 
# \begin{align*}
#   p(y_n \mid \mathbf{z}, x_n)
#   &=
#   \text{Bernoulli}(y_n \mid \text{logit}^{-1}(\mathbf{x}_n^\top \mathbf{z})).
# \end{align*}
# 
# Define the prior to be a multivariate normal
# 
# \begin{align*}
#   p(\mathbf{z})
#   &=
#   \text{Normal}(\mathbf{z} \mid \mathbf{0}, \mathbf{K}),
# \end{align*}
# 
# with covariance matrix given as previously stated.
# 
# Let's build the model in Edward. We use a radial basis function (RBF)
# kernel, also known as the squared exponential or exponentiated
# quadratic. It returns the kernel matrix evaluated over all pairs of
# data points; we then Cholesky decompose the matrix to parameterize the
# multivariate normal distribution.
# 

X = tf.placeholder(tf.float32, [N, D])
f = MultivariateNormalTriL(loc=tf.zeros(N), scale_tril=tf.cholesky(rbf(X)))
y = Bernoulli(logits=f)


# Here, we define a placeholder `X`. During inference, we pass in
# the value for this placeholder according to data.
# 

# ## Inference
# 
# Perform variational inference.
# Define the variational model to be a fully factorized normal.
# 

qf = Normal(loc=tf.Variable(tf.random_normal([N])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))


# Run variational inference for `500` iterations.
# 

inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_iter=5000)


# In this case
# `KLqp` defaults to minimizing the
# $\text{KL}(q\|p)$ divergence measure using the reparameterization
# gradient.
# For more details on inference, see the [$\text{KL}(q\|p)$ tutorial](/tutorials/klqp).
# (This example happens to be slow because evaluating and inverting full
# covariances in Gaussian processes happens to be slow.)
# 

# # Latent Space Models for Neural Data
# 
# Many scientific fields involve the study of network data, including
# social networks, networks in statistical physics, biological
# networks, and information networks
# (Goldenberg, Zheng, Fienberg, & Airoldi, 2010; Newman, 2010).
# 
# What we can learn about nodes in a network from their connectivity patterns?
# We can begin to study this using a latent space model (Hoff, Raftery, & Handcock, 2002).
# Latent space models embed nodes in the network in a latent space,
# where the likelihood of forming an edge between two nodes depends on
# their distance in the latent space.
# 
# We will analyze network data from neuroscience.
# A webpage version is available at
# http://edwardlib.org/tutorials/latent-space-models.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Poisson


# ## Data
# 
# The data comes from [Mark Newman's repository](http://www-personal.umich.edu/~mejn/netdata/).
# It is a weighted, directed network representing the neural network of
# the nematode
# [C. Elegans](https://en.wikipedia.org/wiki/Caenorhabditis_elegans)
# compiled by Watts & Strogatz (1998) using experimental data
# by White, Southgate, Thomson, & Brenner (1986).
# 
# The neural network consists of around $300$ neurons. Each connection
# between neurons
# is associated with a weight (positive integer) capturing the strength
# of the connection.
# 
# First, we load the data.
# 

x_train = np.load('data/celegans_brain.npy')


# ## Model
# 
# What can we learn about the neurons from their connectivity patterns? Using
# a latent space model (Hoff et al., 2002), we will learn a latent
# embedding for each neuron to capture the similarities between them.
# 
# Each neuron $n$ is a node in the network and is associated with a latent
# position $z_n\in\mathbb{R}^K$.
# We place a Gaussian prior on each of the latent positions.
# 
# The log-odds of an edge between node $i$ and
# $j$ is proportional to the Euclidean distance between the latent
# representations of the nodes $|z_i- z_j|$. Here, we
# model the weights ($Y_{ij}$) of the edges with a Poisson likelihood.
# The rate is the reciprocal of the distance in latent space. The
# generative process is as follows:
# 
# 1. 
# For each node $n=1,\ldots,N$,
# \begin{align}
# z_n \sim N(0,I).
# \end{align}
# 2. 
# For each edge $(i,j)\in\{1,\ldots,N\}\times\{1,\ldots,N\}$,
# \begin{align}
# Y_{ij} \sim \text{Poisson}\Bigg(\frac{1}{|z_i - z_j|}\Bigg).
# \end{align}
# 
# In Edward, we write the model as follows.
# 

N = x_train.shape[0]  # number of data points
K = 3  # latent dimensionality

z = Normal(loc=tf.zeros([N, K]), scale=tf.ones([N, K]))

# Calculate N x N distance matrix.
# 1. Create a vector, [||z_1||^2, ||z_2||^2, ..., ||z_N||^2], and tile
# it to create N identical rows.
xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keep_dims=True), [1, N])
# 2. Create a N x N matrix where entry (i, j) is ||z_i||^2 + ||z_j||^2
# - 2 z_i^T z_j.
xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)
# 3. Invert the pairwise distances and make rate along diagonals to
# be close to zero.
xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(N) + 1e3))

x = Poisson(rate=xp)


# ## Inference
# 
# Maximum a posteriori (MAP) estimation is simple in Edward. Two lines are
# required: Instantiating inference and running it.
# 

inference = ed.MAP([z], data={x: x_train})


# See this extended tutorial about
# [MAP estimation in Edward](http://edwardlib.org/tutorials/map).
# 
# One could instead run variational inference. This requires specifying
# a variational model and instantiating `KLqp`.
# 

# Alternatively, run
# qz = Normal(loc=tf.Variable(tf.random_normal([N * K])),
#             scale=tf.nn.softplus(tf.Variable(tf.random_normal([N * K]))))
# inference = ed.KLqp({z: qz}, data={x: x_train})


# See this extended tutorial about
# [variational inference in Edward](http://edwardlib.org/tutorials/variational-inference).
# 
# Finally, the following line runs the inference procedure for 2500
# iterations.
# 

inference.run(n_iter=2500)


# ## Acknowledgments
# 
# We thank Maja Rudolph for writing the initial version of this
# tutorial.
# 

# # Generative Adversarial Networks
# 
# Generative adversarial networks (GANs) are a powerful approach for
# probabilistic modeling (I. Goodfellow et al., 2014; I. Goodfellow, 2016).
# They posit a deep generative model and they enable fast and accurate
# inferences.
# 
# We demonstrate with an example in Edward. A webpage version is available at
# http://edwardlib.org/tutorials/gan.
# 

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf

from edward.models import Uniform
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data


def plot(samples):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig


ed.set_seed(42)

M = 128  # batch size during training
d = 100  # latent dimension

DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)


# ## Data
# 
# We use training data from MNIST, which consists of 55,000 $28\times
# 28$ pixel images (LeCun, Bottou, Bengio, & Haffner, 1998). Each image is represented
# as a flattened vector of 784 elements, and each element is a pixel
# intensity between 0 and 1.
# 
# ![GAN Fig 0](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/gan-fig0.png)
# 
# 
# The goal is to build and infer a model that can generate high quality
# images of handwritten digits.
# 
# During training we will feed batches of MNIST digits. We instantiate a
# TensorFlow placeholder with a fixed batch size of $M$ images.

mnist = input_data.read_data_sets(DATA_DIR)
x_ph = tf.placeholder(tf.float32, [M, 784])


# ## Model
# 
# GANs posit generative models using an implicit mechanism. Given some
# random noise, the data is assumed to be generated by a deterministic
# function of that noise.
# 
# Formally, the generative process is
# 
# \begin{align*}
# \mathbf{\epsilon} &\sim p(\mathbf{\epsilon}), \\mathbf{x} &= G(\mathbf{\epsilon}; \theta),
# \end{align*}
# 
# where $G(\cdot; \theta)$ is a neural network that takes the samples
# $\mathbf{\epsilon}$ as input. The distribution
# $p(\mathbf{\epsilon})$ is interpreted as random noise injected to
# produce stochasticity in a physical system; it is typically a fixed
# uniform or normal distribution with some latent dimensionality.
# 
# In Edward, we build the model as follows, using TensorFlow Slim to
# specify the neural network. It defines a 2-layer fully connected neural
# network and outputs a vector of length $28\times28$ with values in
# $[0,1]$.
# 

def generative_network(eps):
  h1 = slim.fully_connected(eps, 128, activation_fn=tf.nn.relu)
  x = slim.fully_connected(h1, 784, activation_fn=tf.sigmoid)
  return x

with tf.variable_scope("Gen"):
  eps = Uniform(tf.zeros([M, d]) - 1.0, tf.ones([M, d]))
  x = generative_network(eps)


# We aim to estimate parameters of the generative network such
# that the model best captures the data. (Note in GANs, we are
# interested only in parameter estimation and not inference about any
# latent variables.)
# 
# Unfortunately, probability models described above do not admit a tractable
# likelihood. This poses a problem for most inference algorithms, as
# they usually require taking the model's density.  Thus we are
# motivated to use "likelihood-free" algorithms
# (Marin, Pudlo, Robert, & Ryder, 2012), a class of methods which assume one
# can only sample from the model.
# 

# ## Inference
# 
# A key idea in likelihood-free methods is to learn by
# comparison (e.g., Rubin (1984; Gretton, Borgwardt, Rasch, Schölkopf, & Smola, 2012)): by
# analyzing the discrepancy between samples from the model and samples
# from the true data distribution, we have information on where the
# model can be improved in order to generate better samples.
# 
# In GANs, a neural network $D(\cdot;\phi)$ makes this comparison,
# known as the discriminator.
# $D(\cdot;\phi)$ takes data $\mathbf{x}$ as input (either
# generations from the model or data points from the data set), and it
# calculates the probability that $\mathbf{x}$ came from the true data.
# 
# In Edward, we use the following discriminative network. It is simply a
# feedforward network with one ReLU hidden layer. It returns the
# probability in the logit (unconstrained) scale.
# 

def discriminative_network(x):
  """Outputs probability in logits."""
  h1 = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
  logit = slim.fully_connected(h1, 1, activation_fn=None)
  return logit


# Let $p^*(\mathbf{x})$ represent the true data distribution.
# The optimization problem used in GANs is
# 
# \begin{equation*}
# \min_\theta \max_\phi~
# \mathbb{E}_{p^*(\mathbf{x})} [ \log D(\mathbf{x}; \phi) ]
# + \mathbb{E}_{p(\mathbf{x}; \theta)} [ \log (1 - D(\mathbf{x}; \phi)) ].
# \end{equation*}
# 
# This optimization problem is bilevel: it requires a minima solution
# with respect to generative parameters and a maxima solution with
# respect to discriminative parameters.
# In practice, the algorithm proceeds by iterating gradient updates on
# each. An
# additional heuristic also modifies the objective function for the
# generative model in order to avoid saturation of gradients
# (I. J. Goodfellow, 2014).
# 
# Many sources of intuition exist behind GAN-style training. One, which
# is the original motivation, is based on idea that the two neural
# networks are playing a game. The discriminator tries to best
# distinguish samples away from the generator. The generator tries
# to produce samples that are indistinguishable by the discriminator.
# The goal of training is to reach a Nash equilibrium.
# 
# Another source is the idea of casting unsupervised learning as
# supervised learning
# (M. U. Gutmann, Dutta, Kaski, & Corander, 2014; M. Gutmann & Hyvärinen, 2010).
# This allows one to leverage the power of classification—a problem that
# in recent years is (relatively speaking) very easy.
# 
# A third comes from classical statistics, where the discriminator is
# interpreted as a proxy of the density ratio between the true data
# distribution and the model
#  (Mohamed & Lakshminarayanan, 2016; Sugiyama, Suzuki, & Kanamori, 2012). By augmenting an
# original problem that may require the model's density with a
# discriminator (such as maximum likelihood), one can recover the
# original problem when the discriminator is optimal. Furthermore, this
# approximation is very fast, and it justifies GANs from the perspective
# of approximate inference.
# 
# In Edward, the GAN algorithm (`GANInference`) simply takes the
# implicit density model on `x` as input, binded to its
# realizations `x_ph`. In addition, a parameterized function
# `discriminator` is provided to distinguish their
# samples.
# 

inference = ed.GANInference(
    data={x: x_ph}, discriminator=discriminative_network)


# We'll use ADAM as optimizers for both the generator and discriminator.
# We'll run the algorithm for 15,000 iterations and print progress every
# 1,000 iterations.
# 

optimizer = tf.train.AdamOptimizer()
optimizer_d = tf.train.AdamOptimizer()

inference = ed.GANInference(
    data={x: x_ph}, discriminator=discriminative_network)
inference.initialize(
    optimizer=optimizer, optimizer_d=optimizer_d,
    n_iter=15000, n_print=1000)


# We now form the main loop which trains the GAN. At each iteration, it
# takes a minibatch and updates the parameters according to the
# algorithm. At every 1000 iterations, it will print progress and also
# saves a figure of generated samples from the model.
# 

sess = ed.get_session()
tf.global_variables_initializer().run()

idx = np.random.randint(M, size=16)
i = 0
for t in range(inference.n_iter):
  if t % inference.n_print == 0:
    samples = sess.run(x)
    samples = samples[idx, ]

    fig = plot(samples)
    plt.savefig(os.path.join(IMG_DIR, '{}.png').format(
        str(i).zfill(3)), bbox_inches='tight')
    plt.close(fig)
    i += 1

  x_batch, _ = mnist.train.next_batch(M)
  info_dict = inference.update(feed_dict={x_ph: x_batch})
  inference.print_progress(info_dict)


# Examining convergence of the GAN objective can be meaningless in
# practice. The algorithm is usually run until some other criterion is
# satisfied, such as if the samples look visually okay, or if the GAN
# can capture meaningful parts of the data.
# 

# ## Criticism
# 
# Evaluation of GANs remains an open problem---both in criticizing their
# fit to data and in assessing convergence.
# Recent advances have considered alternative objectives and
# heuristics to stabilize training (see also Soumith Chintala's
# [GAN hacks repo](https://github.com/soumith/ganhacks)).
# 
# As one approach to criticize the model, we simply look at generated
# images during training. Below we show generations after 14,000
# iterations (that is, 14,000 gradient updates of both the generator and
# the discriminator).
# 
# ![GAN Fig 1](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/gan-fig1.png)
#                                   
# The images are meaningful albeit a little blurry. Suggestions for
# further improvements would be to tune the hyperparameters in the
# optimization, to improve the capacity of the discriminative and
# generative networks, and to leverage more prior information (such as
# convolutional architectures).

# # Linear Mixed Effects Models
# 
# With linear mixed effects models, we wish to model a linear
# relationship for data points with inputs of varying type, categorized
# into subgroups, and associated to a real-valued output.
# 
# We demonstrate with an example in Edward. A webpage version is available 
# [here](http://edwardlib.org/tutorials/linear-mixed-effects-models).
# 

get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from edward.models import Normal

plt.style.use('ggplot')
ed.set_seed(42)


# ## Data
# 
# We use the `InstEval` data set from the popular
# [lme4 R package](http://lme4.r-forge.r-project.org) (Bates, Mächler, Bolker, & Walker, 2015), located
# [here](https://github.com/blei-lab/edward/blob/master/examples/data/insteval.csv).
# It is a data set of instructor evaluation ratings, where the inputs
# (covariates) include categories such as `students` and
# `departments`, and our response variable of interest is the instructor
# evaluation rating.
# 

# s - students - 1:2972
# d - instructors - codes that need to be remapped
# dept also needs to be remapped
data = pd.read_csv('../examples/data/insteval.csv')
data['dcodes'] = data['d'].astype('category').cat.codes
data['deptcodes'] = data['dept'].astype('category').cat.codes
data['s'] = data['s'] - 1

train = data.sample(frac=0.8)
test = data.drop(train.index)

train.head()


# In the code, we denote:
# + `students` as `s`
# + `instructors` as `d`
# + `departments` as `dept`
# + `service` as `service`
# 

s_train = train['s'].values.astype(int)
d_train = train['dcodes'].values.astype(int)
dept_train = train['deptcodes'].values.astype(int)
y_train = train['y'].values.astype(float)
service_train = train['service'].values.astype(int)
n_obs_train = train.shape[0]

s_test = test['s'].values.astype(int)
d_test = test['dcodes'].values.astype(int)
dept_test = test['deptcodes'].values.astype(int)
y_test = test['y'].values.astype(float)
service_test = test['service'].values.astype(int)
n_obs_test = test.shape[0]


n_s = 2972  # number of students
n_d = 1128  # number of instructors
n_dept = 14  # number of departments
n_obs = train.shape[0]  # number of observations


# ## Model
# 
# With linear regression, one makes an independence assumption where
# each data point regresses with a constant slope among
# each other. In our setting, the observations come from
# groups which may have varying slopes and intercepts. Thus we'd like to
# build a model that can capture this behavior (Gelman & Hill, 2006).
# 
# For examples of this phenomena:
# + The observations from a single student are not independent of
# each other. Rather, some students may systematically give low (or
# high) lecture ratings.
# + The observations from a single teacher are not independent of
# each other. We expect good teachers to get generally good ratings and
# bad teachers to get generally bad ratings.
# + The observations from a single department are not independent of
# each other. One department may generally have dry material and thus be
# rated lower than others.
# 
# 
# Typical linear regression takes the form
# 
# \begin{equation*}
# \mathbf{y} = \mathbf{X}\beta + \epsilon,
# \end{equation*}
# 
# where $\mathbf{X}$ corresponds to fixed effects with coefficients
# $\beta$ and $\epsilon$ corresponds to random noise,
# $\epsilon\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$.
# 
# In a linear mixed effects model, we add an additional term
# $\mathbf{Z}\eta$, where $\mathbf{Z}$ corresponds to random effects
# with coefficients $\eta$. The model takes the form
# 
# \begin{align*}
# \eta &\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}), \\mathbf{y} &= \mathbf{X}\beta + \mathbf{Z}\eta + \epsilon.
# \end{align*}
# 
# Given data, the goal is to infer $\beta$, $\eta$, and $\sigma^2$,
# where $\beta$ are model parameters ("fixed effects"), $\eta$ are
# latent variables ("random effects"), and $\sigma^2$ is a variance
# component parameter.
# 
# Because the random effects have mean 0, the data's mean is captured by
# $\mathbf{X}\beta$. The random effects component $\mathbf{Z}\eta$
# captures variations in the data (e.g.  Instructor \#54 is rated 1.4
# points higher than the mean).
# 
# A natural question is the difference between fixed and random effects.
# A fixed effect is an effect that is constant for a given population. A
# random effect is an effect that varies for a given population (i.e.,
# it may be constant within subpopulations but varies within the overall
# population). We illustrate below in our example:
# 
# + Select `service` as the fixed effect. It is a binary covariate
# corresponding to whether the lecture belongs to the lecturer's main
# department. No matter how much additional data we collect, it
# can only take on the values in $0$ and $1$.
# + Select the categorical values of `students`, `teachers`,
# and `departments` as the random effects. Given more
# observations from the population of instructor evaluation ratings, we
# may be looking at new students, teachers, or departments.
# 
# In the syntax of R's lme4 package (Bates et al., 2015), the model
# can be summarized as
# 
# ```
# y ~ 1 + (1|students) + (1|instructor) + (1|dept) + service
# ```
# where `1` denotes an intercept term,`(1|x)` denotes a
# random effect for `x`, and `x` denotes a fixed effect.
# 

# Set up placeholders for the data inputs.
s_ph = tf.placeholder(tf.int32, [None])
d_ph = tf.placeholder(tf.int32, [None])
dept_ph = tf.placeholder(tf.int32, [None])
service_ph = tf.placeholder(tf.float32, [None])

# Set up fixed effects.
mu = tf.Variable(tf.random_normal([]))
service = tf.Variable(tf.random_normal([]))

sigma_s = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
sigma_d = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
sigma_dept = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))

# Set up random effects.
eta_s = Normal(loc=tf.zeros(n_s), scale=sigma_s * tf.ones(n_s))
eta_d = Normal(loc=tf.zeros(n_d), scale=sigma_d * tf.ones(n_d))
eta_dept = Normal(loc=tf.zeros(n_dept), scale=sigma_dept * tf.ones(n_dept))

yhat = tf.gather(eta_s, s_ph) +     tf.gather(eta_d, d_ph) +     tf.gather(eta_dept, dept_ph) +     mu + service * service_ph
y = Normal(loc=yhat, scale=tf.ones(n_obs))


# ## Inference
# 
# Given data, we aim to infer the model's fixed and random effects.
# In this analysis, we use variational inference with the
# $\text{KL}(q\|p)$ divergence measure. We specify fully factorized
# normal approximations for the random effects and pass in all training
# data for inference. Under the algorithm, the fixed effects will be
# estimated under a variational EM scheme.
# 

q_eta_s = Normal(
    loc=tf.Variable(tf.random_normal([n_s])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_s]))))
q_eta_d = Normal(
    loc=tf.Variable(tf.random_normal([n_d])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_d]))))
q_eta_dept = Normal(
    loc=tf.Variable(tf.random_normal([n_dept])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_dept]))))

latent_vars = {
    eta_s: q_eta_s,
    eta_d: q_eta_d,
    eta_dept: q_eta_dept}
data = {
    y: y_train, 
    s_ph: s_train,
    d_ph: d_train,
    dept_ph: dept_train,
    service_ph: service_train}
inference = ed.KLqp(latent_vars, data)


# One way to critique the fitted model is a residual plot, i.e., a
# plot of the difference between the predicted value and the observed
# value for each data point. Below we manually run inference,
# initializing the algorithm and performing individual updates within a
# loop. We form residual plots as the algorithm progresses. This helps
# us examine how the algorithm proceeds to infer the random and fixed
# effects from data.
# 
# To form residuals, we first make predictions on test data. We do this
# by copying `yhat` defined in the model and replacing its
# dependence on random effects with their inferred means. During the
# algorithm, we evaluate the predictions, feeding in test inputs.
# 
# We have also fit the same model (`y ~ service + (1|dept) + (1|s) + (1|d)`, 
# fit on the entire `InstEval` dataset, specifically) in `lme4`.  We 
# have saved the random effect estimates and will compare them to our 
# learned parameters.
# 

yhat_test = ed.copy(yhat, {
    eta_s: q_eta_s.mean(),
    eta_d: q_eta_d.mean(),
    eta_dept: q_eta_dept.mean()})


inference.initialize(n_print=2000, n_iter=10000)
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  # Update and print progress of algorithm.
  info_dict = inference.update()
  inference.print_progress(info_dict)

  t = info_dict['t']
  if t == 1 or t % inference.n_print == 0:
    # Make predictions on test data.
    yhat_vals = yhat_test.eval(feed_dict={
        s_ph: s_test,
        d_ph: d_test,
        dept_ph: dept_test,
        service_ph: service_test})

    # Form residual plot.
    plt.title("Residuals for Predicted Ratings on Test Set")
    plt.xlim(-4, 4)
    plt.ylim(0, 800)
    plt.hist(yhat_vals - y_test, 75)
    plt.show()


# ## Criticism
# 
# Above, we described a method for diagnosing the fit of the model via
# residual plots. See the residual plot at the end of the algorithm.
# 
# The residuals appear normally distributed with mean 0. This is a good
# sanity check for the model.
# 
# We can also compare our learned parameters to those estimated by R's
# `lme4`.  
# 

student_effects_lme4 = pd.read_csv('../examples/data/insteval_student_ranefs_r.csv')
instructor_effects_lme4 = pd.read_csv('../examples/data/insteval_instructor_ranefs_r.csv')
dept_effects_lme4 = pd.read_csv('../examples/data/insteval_dept_ranefs_r.csv')


student_effects_edward = q_eta_s.mean().eval()
instructor_effects_edward = q_eta_d.mean().eval()
dept_effects_edward = q_eta_dept.mean().eval()


plt.title("Student Effects Comparison")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Student Effects from lme4")
plt.ylabel("Student Effects from edward")
plt.scatter(student_effects_lme4["(Intercept)"], 
            student_effects_edward, 
            alpha = 0.25)
plt.show()


plt.title("Instructor Effects Comparison")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel("Instructor Effects from lme4")
plt.ylabel("Instructor Effects from edward")
plt.scatter(instructor_effects_lme4["(Intercept)"], 
            instructor_effects_edward, 
            alpha = 0.25)
plt.show()


# Great!  Our estimates for both student and instructor effects seem to
# match those from `lme4` closely.  We have set up a slightly different 
# model here (for example, our overall mean is regularized, as are our
# variances for student, department, and instructor effects, which is not
# true of `lme4`s model), and we have a different inference method, so we 
# should not expect to find exactly the same parameters as `lme4`.  But 
# it is reassuring that they match up closely!
# 

#  Add in the intercept from R and edward
dept_effects_and_intercept_lme4 = 3.28259 + dept_effects_lme4["(Intercept)"]
dept_effects_and_intercept_edward = mu.eval() + dept_effects_edward


plt.title("Departmental Effects Comparison")
plt.xlim(3.0, 3.5)
plt.ylim(3.0, 3.5)
plt.xlabel("Department Effects from lme4")
plt.ylabel("Department Effects from edward")
plt.scatter(dept_effects_and_intercept_lme4, 
            dept_effects_and_intercept_edward,
            s = 0.01*train.dept.value_counts())
plt.show()


# Our department effects do not match up nearly as well with those from `lme4`.  
# There are likely several reasons for this:
#   *  We regularize the overal mean, while `lme4` doesn't, which causes the
#   edward model to put some of the intercept into the department effects, 
#   which are allowed to vary more widely since we learn a variance
#   *  We are using 80% of the data to train the edward model, while our `lme4`
#   estimate uses the whole `InstEval` data set
#   *  The department effects are the weakest in the model and difficult to 
#   estimate.
# 

# ## Acknowledgments
# 
# We thank Mayank Agrawal for writing the initial version of this
# tutorial.
# 




# # TensorBoard
# 
# TensorBoard provides a suite of visualization tools to make it easier
# to understand, debug, and optimize Edward programs. You can use it
# "to visualize your TensorFlow graph, plot quantitative metrics about
# the execution of your graph, and show additional data like images that
# pass through it"
# ([tensorflow.org](https://www.tensorflow.org/get_started/summaries_and_tensorboard)).
# 
# A webpage version of this tutorial is available at
# http://edwardlib.org/tutorials/tensorboard.
# 
# ![](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/tensorboard-scalars.png)
# 
# To use TensorBoard, we first need to specify a directory for storing
# logs during inference. For example, if manually controlling inference,
# call
# ```python
# inference.initialize(logdir='log')
# ```
# If you're using the catch-all `inference.run()`, include 
# `logdir` as an argument. As inference runs, files are 
# outputted to `log/` within the working directory. In 
# commandline, we run TensorBoard and point to that directory.
# ```bash
# tensorboard --logdir=log/
# ```
# The command will provide a web address to access TensorBoard. By 
# default, it is http://localhost:6006.  If working correctly, you 
# should see something like the above picture.
# 
# You're set up!
# 
# Additional steps need to be taken in order to clean up TensorBoard's
# naming. Specifically, we might configure names for random variables
# and tensors in the computational graph. To provide a concrete example,
# we extend the
# [supervised learning tutorial](http://edwardlib.org/tutorials/supervised-regression), 
# where the task is to infer hidden structure from labeled examples
# $\{(x_n, y_n)\}$.

get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

plt.style.use('ggplot')


# ## Data
# 
# Simulate training and test sets of $40$ data points. They comprise of
# pairs of inputs $\mathbf{x}_n\in\mathbb{R}^{5}$ and outputs
# $y_n\in\mathbb{R}$. They have a linear dependence with normally
# distributed noise.
# 

def build_toy_dataset(N, w):
  D = len(w)
  x = np.random.normal(0.0, 2.0, size=(N, D))
  y = np.dot(x, w) + np.random.normal(0.0, 0.01, size=N)
  return x, y

ed.set_seed(42)

N = 40  # number of data points
D = 5  # number of features

w_true = np.random.randn(D) * 0.5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)


# ## Model
# 
# Posit the model as Bayesian linear regression (Murphy, 2012).
# For a set of $N$ data points $(\mathbf{X},\mathbf{y})=\{(\mathbf{x}_n, y_n)\}$,
# the model posits the following distributions:
# 
# \begin{align*}
#   p(\mathbf{w})
#   &=
#   \text{Normal}(\mathbf{w} \mid \mathbf{0}, \sigma_w^2\mathbf{I}),
#   \\[1.5ex]
#   p(b)
#   &=
#   \text{Normal}(b \mid 0, \sigma_b^2),
#   \  p(\mathbf{y} \mid \mathbf{w}, b, \mathbf{X})
#   &=
#   \prod_{n=1}^N
#   \text{Normal}(y_n \mid \mathbf{x}_n^\top\mathbf{w} + b, \sigma_y^2).
# \end{align*}
# 
# The latent variables are the linear model's weights $\mathbf{w}$ and
# intercept $b$, also known as the bias.
# Assume $\sigma_w^2,\sigma_b^2$ are known prior variances and $\sigma_y^2$ is a
# known likelihood variance. The mean of the likelihood is given by a
# linear transformation of the inputs $\mathbf{x}_n$.
# 
# Let's build the model in Edward, fixing $\sigma_w,\sigma_b,\sigma_y=1$.
# 

with tf.name_scope('model'): 
  X = tf.placeholder(tf.float32, [N, D], name="X")
  w = Normal(loc=tf.zeros(D, name="weights/loc"), scale=tf.ones(D, name="weights/loc"), name="weights")
  b = Normal(loc=tf.zeros(1, name="bias/loc"), scale=tf.ones(1, name="bias/scale"), name="bias")
  y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N, name="y/scale"), name="y")


# Here, we define a placeholder `X`. During inference, we pass in
# the value for this placeholder according to batches of data.
# We also use a name scope. This adds the scope's name as a prefix
# (`"model/"`) to all tensors in the `with` context.
# Similarly, we name the parameters in each random variable under a
# grouped naming system.
# 

# ## Inference
# 
# We now turn to inferring the posterior using variational inference.
# Define the variational model to be a fully factorized normal across
# the weights. We add another scope to group naming in the variational
# family.
# 

with tf.name_scope("posterior"):
  qw = Normal(loc=tf.Variable(tf.random_normal([D]), name="qw/loc"),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]), name="qw/unconstrained_scale")), 
              name="qw")
  qb = Normal(loc=tf.Variable(tf.random_normal([1]), name="qb/loc"),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]), name="qb/unconstrained_scale")), 
              name="qb")


# Run variational inference with the Kullback-Leibler divergence.
# We use $5$ latent variable samples for computing
# black box stochastic gradients in the algorithm.
# (For more details, see the
# [$\text{KL}(q\|p)$ tutorial](http://edwardlib.org/tutorials/klqp).)
# 

inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run(n_samples=5, n_iter=250, logdir='log/n_samples_5')


# Optionally, we might include an `"inference"` name scope.  
# If it is absent, the charts are partitioned naturally
# and not automatically grouped under the monolithic `"inference"`. 
# If it is added, the TensorBoard graph is slightly more organized.
# 

# ## Criticism
# 
# We can use TensorBoard to explore learning and diagnose any problems.
# After running TensorBoard with the command above, we can navigate the
# tabs.
# 
# Below we assume the above code is run twice with different
# configurations
# of the `n_samples` hyperparameter.
# We specified the log directory to be `log/n_samples_*`.
# By default, Edward also includes a timestamped subdirectory so that
# multiple runs of the same experiment have properly organized logs for
# TensorBoard. You can turn it off by specifying
# `log_timestamp=False` during inference.
# 
# __TensorBoard Scalars.__
# ![](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/tensorboard-scalars.png)
# 
# Scalars provides scalar-valued information across iterations of the
# algorithm, wall time, and relative wall time. In Edward, the tab
# includes the value of scalar TensorFlow variables in the model or
# approximating family.
# 
# With variational inference, we also include information such as the
# loss function and its decomposition into individual terms. This
# particular example shows that `n_samples=1` tends to have higher
# variance than `n_samples=5` but still converges to the same solution.
# 
# __TensorBoard Distributions.__
# ![](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/tensorboard-distributions.png)
# 
# Distributions display the distribution of each non-scalar TensorFlow
# variable in the model and approximating family across iterations.
# 
# __TensorBoard Histograms.__
# ![](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/tensorboard-histograms.png)
# 
# Histograms displays the same information as Distributions but as a 3-D
# histogram changing aross iteration.
# 
# __TensorBoard Graphs.__
# ![](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/tensorboard-graphs-0.png)
# ![](https://raw.githubusercontent.com/blei-lab/edward/master/docs/images/tensorboard-graphs-1.png)
# 
# Graphs displays the computational graph underlying the model,
# approximating family, and inference. Boxes denote tensors grouped
# under the same name scope. Cleaning up names in the graph makes it
# easy to better understand and optimize your code.

# ## Acknowledgments
# 
# We thank Sean Kruzel for writing the initial version of this
# tutorial.
# 
# A TensorFlow tutorial to TensorBoard can be found 
# [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
# 

