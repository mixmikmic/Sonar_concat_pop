# This notebook is created based on the Tensorflow tutorial: https://www.tensorflow.org/get_started/get_started
# 

import tensorflow as tf


# Create a linear model
# $$ y = W * x + b $$
# 

W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# Create a session, init function and run the init function
# 

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# Evaluate the linear model for several values of x
# 

sess.run(linear_model, {x: [1, 2, 3, 4]})


# Create a sum of squares loss function
# 

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)


# Evalute the loss function
# 

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
sess.run(loss, {x: x_train, y: y_train})


# The values of $W$ and $b$ that produce the smallest loss are -1 and 1. Assign the values for $W$ and $b$
# 

fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])
sess.run([fixW, fixb])
sess.run(loss, {x: x_train, y: y_train})


# Train the model using the gradient descent optimizer to compute the values of $W$ and $b$ for the smallest loss
# 

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})


# Display the computed model parameters
# 

sess.run([W, b, loss], {x: x_train, y: y_train})


# Create a linear model and save the output
# 

logs_path = '/home/ubuntu/tensorflow-logs'
# make the directory if it does not exist
get_ipython().system('mkdir -p $logs_path')


# Run the entire linear model at once
# 

tf.reset_default_graph()

# Model parameters
W = tf.Variable([.3], tf.float32, name='W')
b = tf.Variable([-.3], tf.float32, name='b')
# Model input and output
x = tf.placeholder(tf.float32, name='x')
linear_model = W * x + b
y = tf.placeholder(tf.float32, name='y')
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run(
    [W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


# Create a graph of the nodes
# 

summary_writer = tf.summary.FileWriter(
    logs_path, graph=sess.graph)
summary_writer.close()


# Using tf.contrib.learn to create a linear regression model
# 

import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1, 2, 3, 4])
y = np.array([0, -1, -2, -3])
input_fn = tf.contrib.learn.io.numpy_input_fn(
    {'x': x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
estimator.evaluate(input_fn=input_fn)


# Create a custom model with tf.contrib.learn
# 

def model(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", shape=[1], dtype=tf.float64)
    b = tf.get_variable("b", shape=[1], dtype=tf.float64)
    y = W * features['x'] + b
    
    loss = tf.reduce_sum(tf.square(y - labels))
    
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))
    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y, loss=loss, train_op=train)


# Create the estimator with the custom model
# 

estimator = tf.contrib.learn.Estimator(model_fn=model)
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn(
    {'x': x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
estimator.evaluate(input_fn=input_fn, steps=10)





# This notebook is created from the Tensorflow tutorial:
# https://www.tensorflow.org/get_started/get_started
# 

import tensorflow as tf


# Create two constant nodes
# 

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)  # tf.float32 implicitly
print(node1, node2)


# Create a session and run the computational graph
# 

sess = tf.Session()
print(sess.run([node1, node2]))


logs_path = '/home/ubuntu/tensorflow-logs'
# make the directory if it does not exist
get_ipython().system('mkdir -p $logs_path')


# tensorboard --purge_orphaned_data --logdir /home/ubuntu/tensorf low-logs


summary_writer = tf.summary.FileWriter(
    logs_path, graph=tf.get_default_graph())


# Add an operation node
# 

node3 = tf.add(node1, node2)
print('node3: ', node3)
print('sess.run(node3): ', sess.run(node3))


# Add a placeholder node to which a value can be supplied
# 

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b


# Supply values to the placeholders
# 

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


# Add an operation node
# 

add_and_triple = adder_node * 3.0
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


summary_writer.close()


sess.close()


# This notebook is from created from the TensorFlow tutorial:
# http://tensorflow.org/tutorials/mnist/beginners/index.md
# 

# Import the MNIST Data
# 

# import input_data
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Get size of training data
# 

mnist.train.num_examples


# Get size of test data
# 

mnist.test.num_examples


# Display size of training images tensor
# 

mnist.train.images.shape


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().magic('pylab inline')


first_image_array = mnist.train.images[0, ]
image_length = int(np.sqrt(first_image_array.size))
first_image = np.reshape(first_image_array, (-1, image_length))


first_image.shape


plt.imshow(first_image, cmap = cm.Greys_r)
plt.show()








