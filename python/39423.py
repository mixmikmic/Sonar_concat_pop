# Optimizers are the core part of any Deep learning project. No matter, How well you have engineered your network 
# and what kind of activation functions you have used, if your gradients are not computed properly, everything will be a waste of time.
# 
# I have observed some behaviours in the way optimizers are used. Most of them straight away jump to Adam or RMSprop, seeing the trend that they perform well
# 
# In this blog post, we will see how different optimizers perform on 
# - MNIST
# and understand how the accuracy is improving with time. 
# 
# For faster experimentation, lets build a class based framework so that the experimentation is really quick.
# 

import numpy as np
import tensorflow as tf
from mlp import BO
import os 

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


optimizer = ["sgd", "momentum", "nestrov_momentum", "adagrad", "adadelta", "rmsprop", "adam"]
learning_rate = [0.0001, 0.001, 0.01, 0.1]


x_train = mnist.train.images
y_train = mnist.train.labels
x_valid = mnist.validation.images
y_valid = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels
print (x_test.shape)
print (y_test.shape)


model = BO(x_train, y_train, x_valid, y_valid, x_test, y_test)
print ("[Model Initialized]")


model.build_graph()


model.compile_graph(optimize = optimizer[1], learning_rate = learning_rate[0])


model.train(summary_dir = "/tmp/mnist/"+optimizer[1]+"_"+str(learning_rate[0]))


# ## Visualizing Convolution features using MNIST Dataset
# 
# Used inputs from: 
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
# https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4#.2snsuxqts
# 

#Load the required libraries
from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10


# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()


# Initializing the variables
init = tf.initialize_all_variables()


# ### Save the model whenever we get high train accuarcy. save these checkpoints.
# 

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    model_acc = []
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc))
            model_acc.append(acc)
            if max(model_acc) == acc:
                saver.save(sess,"mnist_model"+"_"+str(step*batch_size))
        step += 1
    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))


# ### rebuilding the same graph as above but with different names. We can access each layer with its variable name
# 

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

x1 = tf.reshape(x, shape=[-1, 28, 28, 1])

# Convolution Layer
conv1 = conv2d(x1, weights['wc1'], biases['bc1'])
# Max Pooling (down-sampling)
conv1_m = maxpool2d(conv1, k=2)

# Convolution Layer
conv2 = conv2d(conv1_m, weights['wc2'], biases['bc2'])
# Max Pooling (down-sampling)
conv2_m = maxpool2d(conv2, k=2)

# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
fc1 = tf.reshape(conv2_m, [-1, weights['wd1'].get_shape().as_list()[0]])
fc2 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
fc3 = tf.nn.relu(fc2)
# Apply Dropout
fc4 = tf.nn.dropout(fc3, dropout)
# Output, class prediction
out = tf.add(tf.matmul(fc4, weights['out']), biases['out'])

correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.initialize_all_variables()


import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')

def maps(layername,feed_input,model_name):
    """
    returns feature maps (graphs) of layers 

    Args:
    layername: The required layer need to be visualized. Usually a numpy array
    feed_input: A dictionary of type {x:,y:,keep_prob:} # check with the network characteristics
    model_name: The model name 

    Returns:
    images of each feature in the layers
    """
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,model_name)
        units = sess.run(layername,feed_dict=feed_input)
        units = np.asarray(units)
        #visualizations 
        tot_features = units.shape[3]
        for i in range(int(tot_features/4)):
            images = units[0,:,:,i*4:(i+1)*4]
            images = np.rollaxis(images,-1)
            fig,ax = plt.subplots(nrows=1,ncols=4,sharex="col",sharey="row",figsize=(20,3))
            for j,img in enumerate(images):
                ax[j].imshow(img,interpolation="nearest", cmap="gray")


maps(conv1,{x: mnist.test.images[:1],y: mnist.test.labels[:1],keep_prob: 1.},"mnist_model_15360")


maps(conv1,{x: mnist.test.images[:1],y: mnist.test.labels[:1],keep_prob: 1.},"mnist_model_74240")


maps(conv2,{x: mnist.test.images[:1],y: mnist.test.labels[:1],keep_prob: 1.},"mnist_model_15360")


maps(conv2,{x: mnist.test.images[:1],y: mnist.test.labels[:1],keep_prob: 1.},"mnist_model_74240")


# ## In this way you can visualize all the layers and see how CNN's are learning over time.
# 

