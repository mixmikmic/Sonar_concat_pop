# ## Different Activation Functions in Tensorflow
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# Defining a simple data set
T, F = 1., -1.
X_in = [
    [T, T, T],
    [T, T, F],
    [T, F, T],
    [T, F, F],
    [F, T, T],
    [F, T, F],
    [F, F, T],
    [F, F, F],
]


# Threshold Activation function
def threshold (x):
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype = x.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


# Using Threshold Activation to calculate output for the given data

b = tf.Variable(tf.random_normal([1,1], stddev=2))
w = tf.Variable(tf.random_normal([3,1], stddev=2))
h = tf.matmul(X_in, w) + b
threshold_neuron = threshold(h)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    out = sess.run(threshold_neuron)

print(out)


# Plotting Threshold Activation Function
h = np.linspace(-1,1,50)
out = threshold(h)

with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    
plt.xlabel('Activity of Neuron')
plt.ylabel('Output of Neuron')
plt.title('Threshold Activation Function')
plt.plot(h, y)


# Plotting Sigmoidal Activation function
h = np.linspace(-10,10,50)
out = tf.sigmoid(h)
with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    
plt.xlabel('Activity of Neuron')
plt.ylabel('Output of Neuron')
plt.title('Sigmoidal Activation Function')
plt.plot(h, y)


# Plotting Hyperbolic Tangent Activation function
h = np.linspace(-10,10,50)
out = tf.tanh(h)
with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    
plt.xlabel('Activity of Neuron')
plt.ylabel('Output of Neuron')
plt.title('Hyperbolic Tangent Activation Function')
plt.plot(h, y)


# Linear Activation Function
b = tf.Variable(tf.random_normal([1,1], stddev=2))
w = tf.Variable(tf.random_normal([3,1], stddev=2))
linear_out = tf.matmul(X_in, w) + b
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    out = sess.run(linear_out)

print(out)


# Plotting ReLU Activation function
h = np.linspace(-10,10,50)
out = tf.nn.relu(h)
with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    
plt.xlabel('Activity of Neuron')
plt.ylabel('Output of Neuron')
plt.title('ReLU Activation Function')
plt.plot(h, y)


# Plotting Softmax Activation function
h = np.linspace(-5,5,50)
out = tf.nn.softmax(h)
with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    
plt.xlabel('Activity of Neuron')
plt.ylabel('Output of Neuron')
plt.title('Softmax Activation Function')
plt.plot(h, y)





