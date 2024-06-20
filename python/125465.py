# ## Factorization Machines with tensorflow tutorial
# 
# In this tutial we demonstrate how to create a FMs model with tensorflow step-by-step.
# 
# ### References:
# 
# Blog post by Gabriele Modena: [Factorization Machines with Tensorflow](http://nowave.it/factorization-machines-with-tensorflow.html)
# 
# Factorization Machines paper: [Factorization Machines with LibFm (pdf)](http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf)
# 
# 
# ### Relevant repository:
# 
# [tffm library](https://github.com/geffy/tffm)
# 

# ## Utility function to convert list to sparse matrix
# 
# Here we created a utility function to create a sparse matrix (that is needed by factorization machines) from a list of user/item ids.
# 
# Check [this gist](https://gist.github.com/babakx/7a3fc9739b7778f6673a458605e18963) for more details about this utitly function.
# 

from itertools import count
from collections import defaultdict
from scipy.sparse import csr

def vectorize_dic(dic, ix=None, p=None):
    """ 
    Creates a scipy csr matrix from a list of lists (each inner list is a set of values corresponding to a feature) 
    
    parameters:
    -----------
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    if (ix == None):
        ix = defaultdict(count(0).next)
        
    n = len(dic.values()[0]) # num samples
    g = len(dic.keys()) # num groups
    nz = n * g # number of non-zeros

    col_ix = np.empty(nz, dtype=int)
    
    i = 0
    for k, lis in dic.iteritems():
        # append index el with k in order to prevet mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1
        
    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)
    
    if (p == None):
        p = len(ix)
        
    ixx = np.where(col_ix < p)

    return csr.csr_matrix((data[ixx],(row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix


# ## Loading data
# 
# In this tutorial we use the [MovieLens100k Dataset](https://grouplens.org/datasets/movielens/100k/). Here we convert data to scipy csr (sparse) matrix format. 
# 

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# laod data with pandas
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

# vectorize data and convert them to csr matrix
X_train, ix = vectorize_dic({'users': train.user.values, 'items': train.item.values})
X_test, ix = vectorize_dic({'users': test.user.values, 'items': test.item.values}, ix, X_train.shape[1])
y_train = train.rating.values
y_test= test.rating.values


# ## Densifying the input matrices
# Here we convert the two matrices of `X_train` and `X_test` to dense format to be able to feed them to the tf model. For large datasets this trick is not recommended. You can use `tf.SparseTensor` for large sparse datasets. Check [this file from tffm library](https://github.com/geffy/tffm/blob/a98c786917f5ca74a249748ddef8b694b7f823c9/tffm/core.py#L127) to see how a sparse tensor can be defined.
# 

X_train = X_train.todense()
X_test = X_test.todense()

# print shape of data
print X_train.shape
print X_test.shape


# ## Define FM Model with tensorflow
# 
# We first initialize the parameters of the model as follows:
# 

import tensorflow as tf

n, p = X_train.shape

# number of latent factors
k = 10

# design matrix
X = tf.placeholder('float', shape=[None, p])
# target vector
y = tf.placeholder('float', shape=[None, 1])

# bias and weights
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

# interaction factors, randomly initialized 
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

# estimate of y, initialized to 0.
y_hat = tf.Variable(tf.zeros([n, 1]))


# ### Now we define how the output values y should be calculated
# Using the trick in Rendle's paper, the output of a give feature vector `x` can be calculated using the following equation. The next cell implements that with tensorflow operations.
# 

from IPython.display import display, Math, Latex
display(Math(r'\hat{y}(\mathbf{x}) = w_0 + \sum_{j=1}^{p}w_jx_j + \frac{1}{2} \sum_{f=1}^{k} ((\sum_{j=1}^{p}v_{j,f}x_j)^2-\sum_{j=1}^{p}v_{j,f}^2 x_j^2)'))


# Calculate output with FM equation
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))
pair_interactions = (tf.multiply(0.5,
                    tf.reduce_sum(
                        tf.subtract(
                            tf.pow( tf.matmul(X, tf.transpose(V)), 2),
                            tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                        1, keep_dims=True)))
y_hat = tf.add(linear_terms, pair_interactions)


# ## Loss function
# 
# Here we implement FM point-wise loss function with tensorflow operations. The loss is defined as:
# 

display(Math(r'L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_w ||W||^2 + \lambda_v ||V||^2'))


# L2 regularized sum of squares loss function over W and V
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

l2_norm = (tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w, tf.pow(W, 2)),
                tf.multiply(lambda_v, tf.pow(V, 2)))))

error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
loss = tf.add(error, l2_norm)


# ## Optimization
# 
# Given a loss function, tensorflow can automatically calculate the derivatives of the loss function and find the optimal values for the 'variables' of the loss function. Under the hood, gradient descent optimizer update model parameters iteratively with the following update rule:
# 

display(Math(r'\Theta_{i+1} = \Theta_{i} - \eta \frac{\delta L}{\delta \Theta}'))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


# ## Preparing Batches
# 
# With SGD or Adam optimization methods, you can update model parameters in mini-batches. The larges the size of mini-batches are, the faster is the optimization method but it can become harder to find the optimized parameters. Size of mini-batches is a trade-off between accuracy and complexity. Here we implement a method to generate mini-batches from the input data. Check this great weblog for an [overview of gradient descent optimization methods](http://ruder.io/optimizing-gradient-descent/index.html).
# 

def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


# ## Lanching tensorflow graph and training the model
# Finally we can strat the tensorflow session to initialize the variabeles and optimize the model prameters. Training consists of running the `optimizer` operation and feeding the mini-batches to the optimizer.
# 

from tqdm import tqdm_notebook as tqdm

epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for epoch in tqdm(range(epochs), unit='epoch'):
    perm = np.random.permutation(X_train.shape[0])
    # iterate over batches
    for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):
        sess.run(optimizer, feed_dict={X: bX.reshape(-1, p), y: bY.reshape(-1, 1)})


# ## Evaluating the model
# We can now evaluate the trained model on out test set. We use RMSE to measure the error of predictions. Note that here we need to run the `error` operation.
# 

errors = []
for bX, bY in batcher(X_test, y_test):
    errors.append(sess.run(error, feed_dict={X: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))

RMSE = np.sqrt(np.array(errors).mean())
print RMSE


# ## Closing tensorflow session
# After finishing your experiment make sure you close the tf session that you crated to free the memory it uses.
# 

sess.close()


