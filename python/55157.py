# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- Convolutional General Adversarial Networks
# 

# Implementation of General Adversarial Nets (GAN) where both the discriminator and generator have convolutional and deconvolutional layers, respectively. In this example, the GAN generator was trained to generate MNIST images.
# 
# Uses
# 
# - samples from a random normal distribution (range [-1, 1])
# - dropout
# - leaky relus
# - batch normalization
# - separate batches for "fake" and "real" images (where the labels are 1 = real images, 0 = fake images)
# - MNIST images normalized to [-1, 1] range
# - generator with tanh output
# 

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle as pkl

tf.test.gpu_device_name()


### Abbreviatiuons
# dis_*: discriminator network
# gen_*: generator network

########################
### Helper functions
########################

def leaky_relu(x, alpha=0.0001):
    return tf.maximum(alpha * x, x)


########################
### DATASET
########################

mnist = input_data.read_data_sets('MNIST_data')


#########################
### SETTINGS
#########################

# Hyperparameters
learning_rate = 0.001
training_epochs = 50
batch_size = 64
dropout_rate = 0.5

# Architecture
dis_input_size = 784
gen_input_size = 100

# Other settings
print_interval = 200

#########################
### GRAPH DEFINITION
#########################

g = tf.Graph()
with g.as_default():
    
    # Placeholders for settings
    dropout = tf.placeholder(tf.float32, shape=None, name='dropout')
    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')
    
    # Input data
    dis_x = tf.placeholder(tf.float32, shape=[None, dis_input_size],
                           name='discriminator_inputs')     
    gen_x = tf.placeholder(tf.float32, [None, gen_input_size],
                           name='generator_inputs')


    ##################
    # Generator Model
    ##################

    with tf.variable_scope('generator'):
        
        # 100 => 784 => 7x7x64
        gen_fc = tf.layers.dense(inputs=gen_x, units=3136,
                                 bias_initializer=None, # no bias required when using batch_norm
                                 activation=None)
        gen_fc = tf.layers.batch_normalization(gen_fc, training=is_training)
        gen_fc = leaky_relu(gen_fc)
        gen_fc = tf.reshape(gen_fc, (-1, 7, 7, 64))
        
        # 7x7x64 => 14x14x32
        deconv1 = tf.layers.conv2d_transpose(gen_fc, filters=32, 
                                             kernel_size=(3, 3), strides=(2, 2), 
                                             padding='same',
                                             bias_initializer=None,
                                             activation=None)
        deconv1 = tf.layers.batch_normalization(deconv1, training=is_training)
        deconv1 = leaky_relu(deconv1)     
        deconv1 = tf.layers.dropout(deconv1, rate=dropout_rate)
        
        # 14x14x32 => 28x28x16
        deconv2 = tf.layers.conv2d_transpose(deconv1, filters=16, 
                                             kernel_size=(3, 3), strides=(2, 2), 
                                             padding='same',
                                             bias_initializer=None,
                                             activation=None)
        deconv2 = tf.layers.batch_normalization(deconv2, training=is_training)
        deconv2 = leaky_relu(deconv2)     
        deconv2 = tf.layers.dropout(deconv2, rate=dropout_rate)
        
        # 28x28x16 => 28x28x8
        deconv3 = tf.layers.conv2d_transpose(deconv2, filters=8, 
                                             kernel_size=(3, 3), strides=(1, 1), 
                                             padding='same',
                                             bias_initializer=None,
                                             activation=None)
        deconv3 = tf.layers.batch_normalization(deconv3, training=is_training)
        deconv3 = leaky_relu(deconv3)     
        deconv3 = tf.layers.dropout(deconv3, rate=dropout_rate)
        
        # 28x28x8 => 28x28x1
        gen_logits = tf.layers.conv2d_transpose(deconv3, filters=1, 
                                                kernel_size=(3, 3), strides=(1, 1), 
                                                padding='same',
                                                bias_initializer=None,
                                                activation=None)
        gen_out = tf.tanh(gen_logits, 'generator_outputs')


    ######################
    # Discriminator Model
    ######################
    
    def build_discriminator_graph(input_x, reuse=None):

        with tf.variable_scope('discriminator', reuse=reuse):
            
            # 28x28x1 => 14x14x8
            conv_input = tf.reshape(input_x, (-1, 28, 28, 1))
            conv1 = tf.layers.conv2d(conv_input, filters=8, kernel_size=(3, 3),
                                     strides=(2, 2), padding='same',
                                     bias_initializer=None,
                                     activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = leaky_relu(conv1)
            conv1 = tf.layers.dropout(conv1, rate=dropout_rate)
            
            # 14x14x8 => 7x7x32
            conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=(3, 3),
                                     strides=(2, 2), padding='same',
                                     bias_initializer=None,
                                     activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = leaky_relu(conv2)
            conv2 = tf.layers.dropout(conv2, rate=dropout_rate)

            # fully connected layer
            fc_input = tf.reshape(conv2, (-1, 7*7*32))
            logits = tf.layers.dense(inputs=fc_input, units=1, activation=None)
            out = tf.sigmoid(logits)
            
        return logits, out    

    # Create a discriminator for real data and a discriminator for fake data
    dis_real_logits, dis_real_out = build_discriminator_graph(dis_x, reuse=False)
    dis_fake_logits, dis_fake_out = build_discriminator_graph(gen_out, reuse=True)


    #####################################
    # Generator and Discriminator Losses
    #####################################
    
    # Two discriminator cost components: loss on real data + loss on fake data
    # Real data has class label 0, fake data has class label 1
    dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logits, 
                                                            labels=tf.zeros_like(dis_real_logits))
    dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits, 
                                                            labels=tf.ones_like(dis_fake_logits))
    dis_cost = tf.add(tf.reduce_mean(dis_fake_loss), 
                      tf.reduce_mean(dis_real_loss), 
                      name='discriminator_cost')
 
    # Generator cost: difference between dis. prediction and label "0" for real images
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits,
                                                       labels=tf.zeros_like(dis_fake_logits))
    gen_cost = tf.reduce_mean(gen_loss, name='generator_cost')
    
    
    #########################################
    # Generator and Discriminator Optimizers
    #########################################
      
    dis_optimizer = tf.train.AdamOptimizer(learning_rate)
    dis_train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    dis_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    
    with tf.control_dependencies(dis_update_ops): # required to upd. batch_norm params
        dis_train = dis_optimizer.minimize(dis_cost, var_list=dis_train_vars,
                                           name='train_discriminator')
    
    gen_optimizer = tf.train.AdamOptimizer(learning_rate)
    gen_train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    
    with tf.control_dependencies(gen_update_ops): # required to upd. batch_norm params
        gen_train = gen_optimizer.minimize(gen_cost, var_list=gen_train_vars,
                                           name='train_generator')
    
    # Saver to save session for reuse
    saver = tf.train.Saver()


##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    avg_costs = {'discriminator': [], 'generator': []}

    for epoch in range(training_epochs):
        dis_avg_cost, gen_avg_cost = 0., 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x*2 - 1 # normalize
            batch_randsample = np.random.uniform(-1, 1, size=(batch_size, gen_input_size))
            
            # Train
            
            _, dc = sess.run(['train_discriminator', 'discriminator_cost:0'],
                             feed_dict={'discriminator_inputs:0': batch_x, 
                                        'generator_inputs:0': batch_randsample,
                                        'dropout:0': dropout_rate,
                                        'is_training:0': True})
            
            _, gc = sess.run(['train_generator', 'generator_cost:0'],
                             feed_dict={'generator_inputs:0': batch_randsample,
                                        'dropout:0': dropout_rate,
                                        'is_training:0': True})
            
            dis_avg_cost += dc
            gen_avg_cost += gc

            if not i % print_interval:
                print("Minibatch: %04d | Dis/Gen Cost:    %.3f/%.3f" % (i + 1, dc, gc))
                

        print("Epoch:     %04d | Dis/Gen AvgCost: %.3f/%.3f" % 
              (epoch + 1, dis_avg_cost / total_batch, gen_avg_cost / total_batch))
        
        avg_costs['discriminator'].append(dis_avg_cost / total_batch)
        avg_costs['generator'].append(gen_avg_cost / total_batch)
    
    
    saver.save(sess, save_path='./gan-conv.ckpt')


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.plot(range(len(avg_costs['discriminator'])), 
         avg_costs['discriminator'], label='discriminator')
plt.plot(range(len(avg_costs['generator'])),
         avg_costs['generator'], label='generator')
plt.legend()
plt.show()


####################################
### RELOAD & GENERATE SAMPLE IMAGES
####################################


n_examples = 25

with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./gan-conv.ckpt')

    batch_randsample = np.random.uniform(-1, 1, size=(n_examples, gen_input_size))
    new_examples = sess.run('generator/generator_outputs:0',
                            feed_dict={'generator_inputs:0': batch_randsample,
                                       'dropout:0': 0.0,
                                       'is_training:0': False})

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8),
                         sharey=True, sharex=True)

for image, ax in zip(new_examples, axes.flatten()):
    ax.imshow(image.reshape((dis_input_size // 28, dis_input_size // 28)), cmap='binary')

plt.show()


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- Convolutional Neural Network (VGG16)
# 

# The VGG-16 Convolutional Neural Network Architecture [1] implemented in TensorFlow and trained on Cifar-10 [2, 3] images.
# 
# References:
# 
# - [1] Simonyan, K., & Zisserman, A. (2015). [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556). International Conference on Learning Representations (ICRL), 1–14. https://doi.org/10.1016/j.infsof.2008.09.005
# - [2] Krizhevsky, A. (2009). [Learning Multiple Layers of Features from Tiny Images](https://doi.org/10.1.1.222.9220 http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf). Science Department, University of Toronto.
# - [3] https://www.cs.toronto.edu/~kriz/cifar.html
# 

##########################
### DATASET
##########################

from helper import download_and_extract_cifar
from helper import Cifar10Loader

dest = download_and_extract_cifar('./cifar-10')
cifar = Cifar10Loader(dest, normalize=True, 
                      zero_center=True,
                      channel_mean_center=False)
cifar.num_train

X, y = cifar.load_test()
half = cifar.num_test // 2
X_test, X_valid = X[:half], X[half:]
y_test, y_valid = y[:half], y[half:]

del X, y


import tensorflow as tf
import numpy as np

tf.test.gpu_device_name()


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.001
training_epochs = 30
batch_size = 32

# Other
print_interval = 200

# Architecture
image_width, image_height, image_depth = 32, 32, 3
n_classes = 10


##########################
### WRAPPER FUNCTIONS
##########################

def conv_layer(input, input_channels, output_channels, 
               kernel_size, strides, scope, padding='SAME'):
    with tf.name_scope(scope):
        weights_shape = kernel_size + [input_channels, output_channels]
        weights = tf.Variable(tf.truncated_normal(shape=weights_shape,
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32),
                                                  name='weights')
        biases = tf.Variable(tf.zeros(shape=[output_channels]),
                             name='biases')
        conv = tf.nn.conv2d(input=input,
                            filter=weights,
                            strides=strides,
                            padding=padding,
                            name='convolution')
        out = tf.nn.bias_add(conv, biases, name='logits')
        out = tf.nn.relu(out, name='activation')
        return out


def fc_layer(input, output_nodes, scope,
             activation=None, seed=None):
    with tf.name_scope(scope):
        shape = int(np.prod(input.get_shape()[1:]))
        flat_input = tf.reshape(input, [-1, shape])
        weights = tf.Variable(tf.truncated_normal(shape=[shape,
                                                         output_nodes],
                                                  mean=0.0,
                                                  stddev=0.1,
                                                  dtype=tf.float32,
                                                  seed=seed),
                                                  name='weights')
        biases = tf.Variable(tf.zeros(shape=[output_nodes]),
                             name='biases')
        act = tf.nn.bias_add(tf.matmul(flat_input, weights), biases, 
                             name='logits')

        if activation is not None:
            act = activation(act, name='activation')

        return act


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, image_width, image_height, image_depth], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')
     
    ##########################
    ### VGG16 Model
    ##########################

    # =========
    # BLOCK 1
    # =========
    conv_layer_1 = conv_layer(input=tf_x,
                              input_channels=3,
                              output_channels=64,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv1')
    
    conv_layer_2 = conv_layer(input=conv_layer_1,
                              input_channels=64,
                              output_channels=64,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv2')    
    
    pool_layer_1 = tf.nn.max_pool(conv_layer_2,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool1') 
    # =========
    # BLOCK 2
    # =========
    conv_layer_3 = conv_layer(input=pool_layer_1,
                              input_channels=64,
                              output_channels=128,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv3')    
    
    conv_layer_4 = conv_layer(input=conv_layer_3,
                              input_channels=128,
                              output_channels=128,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv4')    
    
    pool_layer_2 = tf.nn.max_pool(conv_layer_4,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool2') 
    # =========
    # BLOCK 3
    # =========
    conv_layer_5 = conv_layer(input=pool_layer_2,
                              input_channels=128,
                              output_channels=256,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv5')        
    
    conv_layer_6 = conv_layer(input=conv_layer_5,
                              input_channels=256,
                              output_channels=256,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv6')      
    
    conv_layer_7 = conv_layer(input=conv_layer_6,
                              input_channels=256,
                              output_channels=256,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv7')
    
    pool_layer_3 = tf.nn.max_pool(conv_layer_7,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool3') 
    # =========
    # BLOCK 4
    # =========
    conv_layer_8 = conv_layer(input=pool_layer_3,
                              input_channels=256,
                              output_channels=512,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv8')      
    
    conv_layer_9 = conv_layer(input=conv_layer_8,
                              input_channels=512,
                              output_channels=512,
                              kernel_size=[3, 3],
                              strides=[1, 1, 1, 1],
                              scope='conv9')     
    
    conv_layer_10 = conv_layer(input=conv_layer_9,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv10')   
    
    pool_layer_4 = tf.nn.max_pool(conv_layer_10,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool4') 
    # =========
    # BLOCK 5
    # =========
    conv_layer_11 = conv_layer(input=pool_layer_4,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv11')   
    
    conv_layer_12 = conv_layer(input=conv_layer_11,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv12')   

    conv_layer_13 = conv_layer(input=conv_layer_12,
                               input_channels=512,
                               output_channels=512,
                               kernel_size=[3, 3],
                               strides=[1, 1, 1, 1],
                               scope='conv13') 
    
    pool_layer_5 = tf.nn.max_pool(conv_layer_12,
                                  ksize=[1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool5')     
    # ===========
    # CLASSIFIER
    # ===========
    
    fc_layer_1 = fc_layer(input=pool_layer_5, 
                          output_nodes=4096,
                          activation=tf.nn.relu,
                          scope='fc1')
    
    fc_layer_2 = fc_layer(input=fc_layer_1, 
                          output_nodes=4096,
                          activation=tf.nn.relu,
                          scope='fc2')

    out_layer = fc_layer(input=fc_layer_2, 
                         output_nodes=n_classes,
                         activation=None,
                         scope='output_layer')
    
    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1), 
                                  name='correct_predictions')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # Saver to save session for reuse
    saver = tf.train.Saver()

    
##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        
        avg_cost = 0.
        mbatch_cnt = 0
        for batch_x, batch_y in cifar.load_train_epoch(shuffle=True, batch_size=batch_size):
            
            mbatch_cnt += 1
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y})
            avg_cost += c

            if not mbatch_cnt % print_interval:
                print("Minibatch: %04d | Cost: %.3f" % (mbatch_cnt, c))
                

        # ===================
        # Training Accuracy
        # ===================
        n_predictions, n_correct = 0, 0
        for batch_x, batch_y in cifar.load_train_epoch(batch_size=batch_size):
        
            p = sess.run('correct_predictions:0', 
                         feed_dict={'features:0': batch_x,
                                    'targets:0':  batch_y})
            n_correct += np.sum(p)
            n_predictions += p.shape[0]
        train_acc = n_correct / n_predictions
        
        
        # ===================
        # Validation Accuracy
        # ===================
        #valid_acc = sess.run('accuracy:0', feed_dict={'features:0': X_valid,
        #                                              'targets:0': y_valid})
        # ---------------------------------------
        # workaround for GPUs with <= 4 Gb memory
        n_predictions, n_correct = 0, 0
        indices = np.arange(y_valid.shape[0])
        chunksize = 500
        for start_idx in range(0, indices.shape[0] - chunksize + 1, chunksize):
            index_slice = indices[start_idx:start_idx + chunksize]
            p = sess.run('correct_predictions:0', 
                         feed_dict={'features:0': X_valid[index_slice],
                                    'targets:0': y_valid[index_slice]})
            n_correct += np.sum(p)
            n_predictions += p.shape[0]
        valid_acc = n_correct / n_predictions
        # ---------------------------------------
                                                
        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / mbatch_cnt), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
    
    saver.save(sess, save_path='./convnet-vgg16.ckpt')


##########################
### RELOAD & TEST
##########################

with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./convnet-vgg16.ckpt')
    
    # test_acc = sess.run('accuracy:0', feed_dict={'features:0': X_test,
    #                                              'targets:0': y_test})
    # ---------------------------------------
    # workaround for GPUs with <= 4 Gb memory
    n_predictions, n_correct = 0, 0
    indices = np.arange(y_test.shape[0])
    chunksize = 500
    for start_idx in range(0, indices.shape[0] - chunksize + 1, chunksize):
        index_slice = indices[start_idx:start_idx + chunksize]
        p = sess.run('correct_predictions:0', 
                     feed_dict={'features:0': X_test[index_slice],
                                'targets:0': y_test[index_slice]})
        n_correct += np.sum(p)
        n_predictions += p.shape[0]
    test_acc = n_correct / n_predictions
    # ---------------------------------------

    print('Test ACC: %.3f' % test_acc)


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- Softmax Regression
# 

# Implementation of softmax regression (multinomial logistic regression).
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", one_hot=True)


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.5
training_epochs = 30
batch_size = 256

# Architecture
n_features = 784
n_classes = 10


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_features])
    tf_y = tf.placeholder(tf.float32, [None, n_classes])

    # Model parameters
    params = {
        'weights': tf.Variable(tf.zeros(shape=[n_features, n_classes],
                                               dtype=tf.float32), name='weights'),
        'bias': tf.Variable([[n_classes]], dtype=tf.float32, name='bias')}

    # Softmax regression
    linear = tf.matmul(tf_x, params['weights']) + params['bias']
    pred_proba = tf.nn.softmax(linear, name='predict_probas')
    
    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=linear, labels=tf_y), name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Class prediction
    pred_labels = tf.argmax(pred_proba, 1, name='predict_labels')
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), pred_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    
##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={tf_x: batch_x,
                                                            tf_y: batch_y})
            avg_cost += c
        
        train_acc = sess.run('accuracy:0', feed_dict={tf_x: mnist.train.images,
                                                      tf_y: mnist.train.labels})
        valid_acc = sess.run('accuracy:0', feed_dict={tf_x: mnist.validation.images,
                                                      tf_y: mnist.validation.labels})  
        
        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
        
    test_acc = sess.run(accuracy, feed_dict={tf_x: mnist.test.images,
                                             tf_y: mnist.test.labels})
    print('Test ACC: %.3f' % test_acc)


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- Convolutional Autoencoder with Deconvolutions
# 

# A convolutional autoencoder using nearest neighbor upscaling layers that compresses 768-pixel MNIST images down to a 7x7x4 (196 pixel) representation.
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", validation_size=0)


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.001
training_epochs = 5
batch_size = 128

# Architecture
hidden_size = 16
input_size = 784
image_width = 28

# Other
print_interval = 200
random_seed = 123


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, input_size], name='inputs')
    input_layer = tf.reshape(tf_x, shape=[-1, image_width, image_width, 1])

    ###########
    # Encoder
    ###########
    
    # 28x28x1 => 28x28x8
    conv1 = tf.layers.conv2d(input_layer, filters=8, kernel_size=(3, 3),
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    # 28x28x8 => 14x14x8
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), 
                                       strides=(2, 2), padding='same')
    
    # 14x14x8 => 14x14x4
    conv2 = tf.layers.conv2d(maxpool1, filters=4, kernel_size=(3, 3), 
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    
    # 14x14x4 => 7x7x4
    encode = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), 
                                     strides=(2, 2), padding='same', 
                                     name='encoding')

    ###########
    # Decoder
    ###########
    
    # 7x7x4 => 14x14x8
    deconv1 = tf.layers.conv2d_transpose(encode, filters=8, 
                                         kernel_size=(3, 3), strides=(2, 2), 
                                         padding='same',
                                         activation=tf.nn.relu)
    
    
    # 14x14x8 => 28x28x8
    deconv2 = tf.layers.conv2d_transpose(deconv1, filters=8, 
                                         kernel_size=(3, 3), strides=(2, 2), 
                                         padding='same',
                                         activation=tf.nn.relu)
    
    # 28x28x8 => 28x28x1
    logits = tf.layers.conv2d(deconv2, filters=1, kernel_size=(3,3), 
                              strides=(1, 1), padding='same', 
                              activation=None)
    
    decode = tf.nn.sigmoid(logits, name='decoding')

    ##################
    # Loss & Optimizer
    ##################
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_layer,
                                                   logits=logits)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(cost, name='train')    

    # Saver to save session for reuse
    saver = tf.train.Saver()


import numpy as np

##########################
### TRAINING & EVALUATION
##########################
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed) # random seed for mnist iterator
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'inputs:0': batch_x})

            avg_cost += c

            if not i % print_interval:
                print("Minibatch: %03d | Cost:    %.3f" % (i + 1, c))

        print("Epoch:     %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)))
    
    saver.save(sess, save_path='./autoencoder.ckpt')


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

##########################
### VISUALIZATION
##########################

n_images = 15

fig, axes = plt.subplots(nrows=2, ncols=n_images, sharex=True, 
                         sharey=True, figsize=(20, 2.5))
test_images = mnist.test.images[:n_images]

with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./autoencoder.ckpt')
    decoded = sess.run('decoding:0', feed_dict={'inputs:0': test_images})

for i in range(n_images):
    for ax, img in zip(axes, [test_images, decoded]):
        ax[i].imshow(img[i].reshape((image_width, image_width)), cmap='binary')


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- Multilayer Perceptron with Batch Normalization
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", one_hot=True)


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 10
batch_size = 64

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
n_input = 784
n_classes = 10

# Other
random_seed = 123


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)
    
    # Batchnorm settings
    training_phase = tf.placeholder(tf.bool, None, name='training_phase')

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # Multilayer perceptron
    layer_1 = tf.layers.dense(tf_x, n_hidden_1, 
                              activation=None, # Batchnorm comes before nonlinear activation
                              use_bias=False, # Note that no bias unit is used in batchnorm
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    layer_1 = tf.layers.batch_normalization(layer_1, training=training_phase)
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, 
                              activation=None,
                              use_bias=False,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer_2 = tf.layers.batch_normalization(layer_2, training=training_phase)
    layer_2 = tf.nn.relu(layer_2)
    
    out_layer = tf.layers.dense(layer_2, n_classes, activation=None, name='logits')

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    
    # control dependency to ensure that batchnorm parameters are also updated
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


import numpy as np

##########################
### TRAINING & EVALUATION
##########################
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed) # random seed for mnist iterator
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y,
                                                            'training_phase:0': True})
            avg_cost += c
        
        train_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels,
                                                      'training_phase:0': False})
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels,
                                                      'training_phase:0': False})  
        
        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
        
    test_acc = sess.run('accuracy:0', feed_dict={'features:0': mnist.test.images,
                                                 'targets:0': mnist.test.labels,
                                                 'training_phase:0': False})
    print('Test ACC: %.3f' % test_acc)





# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow,numpy")


# # Chunking an Image Dataset for Minibatch Training using NumPy NPZ Archives
# 

# This notebook provides an example for how to organize a large dataset of images into chunks for quick access during minibatch learning. This approach uses NumPy .npz archive files and only requires having NumPy as a dependency so that this approach should be compatible with different Python-based machine learning and deep learning libraries and packages for further image (pre)processing and augmentation. 
# 
# While this approach performs reasonably well (sufficiently well for my applications), you may also be interested in TensorFlow's "[Reading Data](https://www.tensorflow.org/programmers_guide/reading_data)" guide to work with `TfRecords` and file queues.
# 

# ## 0. The Dataset
# 

# Let's pretend we have a directory of images containing two subdirectories with images for training, validation, and testing. The following function will create such a dataset of images in PNG format locally for demonstration purposes.
# 

# Note that executing the following code 
# cell will download the MNIST dataset
# and save all the 60,000 images as separate JPEG
# files. This might take a few minutes depending
# on your machine.

import numpy as np
from helper import mnist_export_to_jpg

np.random.seed(123)
mnist_export_to_jpg(path='./')


import os

for i in ('train', 'valid', 'test'):
    print('mnist_%s subdirectories' % i, os.listdir('mnist_%s' % i))


# Note that the names of the subdirectories correspond directly to the class label of the images that are stored under it.
# 

# To make sure that the images look okay, the snippet below plots an example image from the subdirectory `mnist_train/9/`:
# 

get_ipython().magic('matplotlib inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

some_img = os.path.join('./mnist_train/9/', os.listdir('./mnist_train/9/')[0])

img = mpimg.imread(some_img)
print(img.shape)
plt.imshow(img, cmap='binary');


# Note: The JPEG format introduces a few artifacts that we can see in the image above. In this case, we use JPEG instead of PNG. Here, JPEG is used for demonstration purposes since that's still format many image datasets are stored in.
# 

# ## 1. Chunking Images into NumPy NPZ Archive Files
# 

# The following wrapper function creates .npz archive files training, testing, and validation. It will group images together into integer arrays that are then saved as .npz archive files. The number of rows (images) in each .npz archive will be equal to the `archive_size` argument. 
# 

import numpy as np
import glob


def images_to_pickles(data_stempath='./mnist_', which_set='train', 
                      archive_size=5000, width=28, height=28, channels=1,
                      shuffle=False, seed=None):
    
    if not os.path.exists('%snpz' % data_stempath):
        os.mkdir('%snpz' % data_stempath)
        
    img_paths = [p for p in glob.iglob('%s%s/**/*.jpg' % 
                                   (data_stempath, which_set), recursive=True)]
    if shuffle:
        rgen = np.random.RandomState(seed)
        paths = rgen.shuffle(img_paths)
    
    idx, file_idx = 0, 1
    data = np.zeros((archive_size, height, width, channels), dtype=np.uint8)
    labels = np.zeros(archive_size, dtype=np.uint8)
    for path in img_paths:
        if idx >= archive_size - 1:
            idx = 0
            savepath = os.path.join('%snpz' % data_stempath, '%s_%d.npz' % 
                                    (which_set, file_idx))
            file_idx += 1
            np.savez(savepath, data=data, labels=labels)

        label = int(os.path.basename(os.path.dirname(path)))
        image = mpimg.imread(path)
        
        if len(image.shape) == 2:
            data[idx] = image[:, :, np.newaxis]
        labels[idx] = label
        idx += 1


images_to_pickles(which_set='train', shuffle=True, seed=1)
images_to_pickles(which_set='valid', shuffle=True, seed=1)
images_to_pickles(which_set='test', shuffle=True, seed=1)


# The .npz files we created are stored under a new directory, `mnist_npz`:
# 

os.listdir('mnist_npz')


# To check that the archiving worked correctly, we will now load one of those .npz archives. Note that we can now access each archive just like a python dictionary. Here the `'data'` key contains the image data and the `'labels'` key stores an array containing the corresponding class labels:
# 

data = np.load('mnist_npz/test_1.npz')
print(data['data'].shape)
print(data['labels'].shape)


plt.imshow(data['data'][0][:, :, -1], cmap='binary');
print('Class label:', data['labels'][0])


# ## 2. Loading Minibatches
# 

# The following cell implements a class for iterating over the MNIST images, based on the .npz archives, conveniently. 
# Via the `normalize` parameter we additionally scale the image pixels to [0, 1] range, which typically helps with gradient-based optimization in practice.
# 
# The key functions (here: generators) are
# 
# - load_train_epoch
# - load_valid_epoch
# - load_test_epoch
# 
# These let us iterate over small chunks (determined via `minibatch_size`). Each of these functions will load the images from a particular .npz archive into memory (here: 5000 images) and yield minibatches of smaller or equal size (for example, 50 images at a time). Via the two shuffle parameters, we can further control if the images within each .npz archive should be shuffled, and if the order the .npz files are loaded should shuffled after each epoch. By setting `onehot=True`, the labels are converted into a onehot representation for convenience.
# 

class BatchLoader():
    def __init__(self, minibatches_path, 
                 normalize=True):
        
        self.normalize = normalize

        self.train_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if 'train' in f]
        self.valid_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if 'valid' in f]
        self.test_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if 'train' in f]

        self.num_train = 45000
        self.num_valid = 5000
        self.num_test = 10000
        self.n_classes = 10


    def load_train_epoch(self, batch_size=50, onehot=False,
                         shuffle_within=False, shuffle_paths=False,
                         seed=None):
        for batch_x, batch_y in self._load_epoch(which='train',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y

    def load_test_epoch(self, batch_size=50, onehot=False,
                        shuffle_within=False, shuffle_paths=False, 
                        seed=None):
        for batch_x, batch_y in self._load_epoch(which='test',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y
            
    def load_validation_epoch(self, batch_size=50, onehot=False,
                         shuffle_within=False, shuffle_paths=False, 
                         seed=None):
        for batch_x, batch_y in self._load_epoch(which='valid',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y

    def _load_epoch(self, which='train', batch_size=50, onehot=False,
                    shuffle_within=True, shuffle_paths=True, seed=None):

        if which == 'train':
            paths = self.train_batchpaths
        elif which == 'valid':
            paths = self.valid_batchpaths
        elif which == 'test':
            paths = self.test_batchpaths
        else:
            raise ValueError('`which` must be "train" or "test". Got %s.' %
                             which)
            
        rgen = np.random.RandomState(seed)
        if shuffle_paths:
            paths = rgen.shuffle(paths)

        for batch in paths:

            dct = np.load(batch)

            if onehot:
                labels = (np.arange(self.n_classes) == 
                          dct['labels'][:, None]).astype(np.uint8)
            else:
                labels = dct['labels']

            if self.normalize:
                # normalize to [0, 1] range
                data = dct['data'].astype(np.float32) / 255.
            else:
                data = dct['data']

            arrays = [data, labels]
            del dct
            indices = np.arange(arrays[0].shape[0])

            if shuffle_within:
                rgen.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - batch_size + 1,
                                   batch_size):
                index_slice = indices[start_idx:start_idx + batch_size]
                yield (ary[index_slice] for ary in arrays)


# The following for loop will iterate over the 45,000 training examples in our MNIST training set, yielding 50 images and labels at a time (note that we previously set aside 5000 training example as our validation datast).
# 

batch_loader = BatchLoader(minibatches_path='./mnist_npz/', 
                           normalize=True)

for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
    print(batch_x.shape)
    print(batch_y.shape)
    break


cnt = 0
for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
    cnt += batch_x.shape[0]
    
print('One training epoch contains %d images' % cnt)


def one_epoch():
    for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
        pass
    
get_ipython().magic('timeit one_epoch()')


# As we can see from the benchmark above, an iteration over one training epoch (45k images) is relatively fast.
# 

# Similarly, we could iterate over validation and test data via 
# 
# - batch_loader.load_validation_epoch
# - batch_loader.load_test_epoch
# 

# ## 3. Training a Model using TensorFlow's `feed_dict`
# 

# The following code demonstrate how we can feed our minibatches into a TensorFlow graph using a TensorFlow session's `feed_dict`.
# 

# ### Multilayer Perceptron Graph
# 

import tensorflow as tf

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 15
batch_size = 100

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
height, width = 28, 28
n_classes = 10


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(123)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, height, width, 1], name='features')
    tf_x_flat = tf.reshape(tf_x, shape=[-1, height*width])
    tf_y = tf.placeholder(tf.int32, [None, n_classes], name='targets')

    # Model parameters
    weights = {
        'h1': tf.Variable(tf.truncated_normal([width*height, n_hidden_1], stddev=0.1)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.1))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # Multilayer perceptron
    layer_1 = tf.add(tf.matmul(tf_x_flat, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


# ### Training the Neural Network with Minibatches
# 

##########################
### TRAINING & EVALUATION
##########################

batch_loader = BatchLoader(minibatches_path='./mnist_npz/', 
                           normalize=True)

# preload small validation set
# by unpacking the generator
[valid_data] = batch_loader.load_validation_epoch(batch_size=5000, 
                                                   onehot=True)
valid_x, valid_y = valid_data[0], valid_data[1]
del valid_data

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.

        n_batches = 0
        for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=batch_size, 
                                                              onehot=True, 
                                                              seed=epoch):
            n_batches += 1
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y.astype(np.int)})
            avg_cost += c
        
        train_acc = sess.run('accuracy:0', feed_dict={'features:0': batch_x,
                                                      'targets:0': batch_y})
        
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': valid_x,
                                                      'targets:0': valid_y})  
        
        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / n_batches), end="")
        print(" | MbTrain/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
        
        
    # imagine test set is too large to fit into memory:
    test_acc, cnt = 0., 0
    for test_x, test_y in batch_loader.load_test_epoch(batch_size=100, 
                                                       onehot=True):   
        cnt += 1
        acc = sess.run(accuracy, feed_dict={'features:0': test_x,
                                            'targets:0': test_y})
        test_acc += acc
    print('Test ACC: %.3f' % (test_acc / cnt))


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- Convolutional Autoencoder with Nearest-neighbor Interpolation
# 

# A convolutional autoencoder using nearest neighbor upscaling layers that compresses 768-pixel MNIST images down to a 7x7x4 (196 pixel) representation.
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


##########################
### DATASET
##########################

mnist = input_data.read_data_sets("./", validation_size=0)


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.001
training_epochs = 5
batch_size = 128

# Architecture
input_size = 784
image_width = 28

# Other
print_interval = 200
random_seed = 123


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, input_size], name='inputs')
    input_layer = tf.reshape(tf_x, shape=[-1, image_width, image_width, 1])

    ###########
    # Encoder
    ###########
    
    # 28x28x1 => 28x28x8
    conv1 = tf.layers.conv2d(input_layer, filters=8, kernel_size=(3, 3),
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    
    # 28x28x8 => 14x14x8
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), 
                                       strides=(2, 2), padding='same')
    
    # 14x14x8 => 14x14x4
    conv2 = tf.layers.conv2d(maxpool1, filters=4, kernel_size=(3, 3), 
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    
    # 14x14x4 => 7x7x4
    encode = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), 
                                     strides=(2, 2), padding='same', 
                                     name='encoding')

    ###########
    # Decoder
    ###########
    
    # 7x7x4 => 14x14x4
    deconv1 = tf.image.resize_nearest_neighbor(encode, 
                                               size=(14, 14))
    # 14x14x4 => 14x14x8
    conv3 = tf.layers.conv2d(deconv1, filters=8, kernel_size=(3, 3), 
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    
    # 14x14x8 => 28x28x8
    deconv2 = tf.image.resize_nearest_neighbor(conv3, 
                                               size=(28, 28))
    # 28x28x8 => 28x28x8
    conv4 = tf.layers.conv2d(deconv2, filters=8, kernel_size=(3, 3), 
                             strides=(1, 1), padding='same', 
                             activation=tf.nn.relu)
    # 28x28x8 => 28x28x1
    logits = tf.layers.conv2d(conv4, filters=1, kernel_size=(3,3), 
                              strides=(1, 1), padding='same', 
                              activation=None)
    
    decode = tf.nn.sigmoid(logits, name='decoding')

    ##################
    # Loss & Optimizer
    ##################
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_layer,
                                                   logits=logits)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(cost, name='train')    

    # Saver to save session for reuse
    saver = tf.train.Saver()


import numpy as np

##########################
### TRAINING & EVALUATION
##########################
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed) # random seed for mnist iterator
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run(['train', 'cost:0'], feed_dict={'inputs:0': batch_x})
            avg_cost += c

            if not i % print_interval:
                print("Minibatch: %03d | Cost:    %.3f" % (i + 1, c))

        print("Epoch:     %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)))
    
    saver.save(sess, save_path='./autoencoder.ckpt')


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

##########################
### VISUALIZATION
##########################

n_images = 15

fig, axes = plt.subplots(nrows=2, ncols=n_images, sharex=True, 
                         sharey=True, figsize=(20, 2.5))
test_images = mnist.test.images[:n_images]

with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./autoencoder.ckpt')
    decoded = sess.run('decoding:0', feed_dict={'inputs:0': test_images})

for i in range(n_images):
    for ax, img in zip(axes, [test_images, decoded]):
        ax[i].imshow(img[i].reshape((image_width, image_width)), cmap='binary')


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow,numpy,h5py")


# # Storing an Image Dataset for Minibatch Training using HDF5
# 

# This notebook provides an example for how to save a large dataset of images as Hierarchical Data Format (HDF) for quick access during minibatch learning. This approach uses the common [HDF5](https://support.hdfgroup.org/HDF5/) format and should be accessible to any programming language or tool with an HDF5 API.
# 
# While this approach performs reasonably well (sufficiently well for my applications), you may also be interested in TensorFlow's "[Reading Data](https://www.tensorflow.org/programmers_guide/reading_data)" guide to work with `TfRecords` and file queues.
# 

# ## 0. The Dataset
# 

# Let's pretend we have a directory of images containing two subdirectories with images for training, validation, and testing. The following function will create such a dataset of images in PNG format locally for demonstration purposes.
# 

# Note that executing the following code 
# cell will download the MNIST dataset
# and save all the 60,000 images as separate JPEG
# files. This might take a few minutes depending
# on your machine.

import numpy as np
from helper import mnist_export_to_jpg

np.random.seed(123)
mnist_export_to_jpg(path='./')


import os

for i in ('train', 'valid', 'test'):
    print('mnist_%s subdirectories' % i, os.listdir('mnist_%s' % i))


# Note that the names of the subdirectories correspond directly to the class label of the images that are stored under it.
# 

# To make sure that the images look okay, the snippet below plots an example image from the subdirectory `mnist_train/9/`:
# 

get_ipython().magic('matplotlib inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

some_img = os.path.join('./mnist_train/9/', os.listdir('./mnist_train/9/')[0])

img = mpimg.imread(some_img)
print(img.shape)
plt.imshow(img, cmap='binary');


# Note: The JPEG format introduces a few artifacts that we can see in the image above. In this case, we use JPEG instead of PNG. Here, JPEG is used for demonstration purposes since that's still format many image datasets are stored in.
# 

# ## 1. Saving images as HDF5
# 

# The following wrapper function creates .h5 file containing training, testing, and validation datasets. It will group images together into larger integer arrays that are then saved as subgroups in the HDF5 file. For instance, the training images will be saved as `'train/images'` and the corresponding labels as `'train/labels'` subgroup.
# 

import numpy as np
import h5py
import glob


def images_to_h5(data_stempath='./mnist_',
                 width=28, height=28, channels=1,
                 shuffle=False, random_seed=None):
    
    with h5py.File('mnist_batches.h5', 'w') as h5f:
    
        for s in ['train', 'valid', 'test']:
            img_paths = [p for p in glob.iglob('%s%s/**/*.jpg' % 
                                       (data_stempath, s), 
                                        recursive=True)]

            dset1 = h5f.create_dataset('%s/images' % s, 
                                       shape=[len(img_paths), 
                                              width, height, channels], 
                                       compression=None,
                                       dtype='uint8')
            dset2 = h5f.create_dataset('%s/labels' % s, 
                                       shape=[len(img_paths)], 
                                       compression=None,
                                       dtype='uint8')
            dset3 = h5f.create_dataset('%s/file_ids' % s, 
                                       shape=[len(img_paths)], 
                                       compression=None,
                                       dtype='S5')
            
            rand_indices = np.arange(len(img_paths))
            
            if shuffle:
                rng = np.random.RandomState(random_seed)
                rng.shuffle(rand_indices)

            for idx, path in enumerate(img_paths):

                rand_idx = rand_indices[idx]
                label = int(os.path.basename(os.path.dirname(path)))
                image = mpimg.imread(path)
                dset1[rand_idx] = image.reshape(width, height, channels)
                dset2[rand_idx] = label
                dset3[rand_idx] = np.array([os.path.basename(path)], dtype='S6')


# Note that we didn't specify any compression format. The reason is that non-compressed HDF5 datasets are much faster to read, which is an important factor for training deep learning systems. In this case, the dataset is about ~47 Mb in size. However, we are working with larger datasets, compressing the HDF5 dataset might be one easy way to deal with hardware storage limitations.
# 

images_to_h5(shuffle=True, random_seed=123)


# To check that the archiving worked correctly, we will now load the training images and print the array shape. Note that we can now access each archive similar to a python dictionary. Here the `'data'` key contains the image data and the `'labels'` key stores an array containing the corresponding class labels:
# 

with h5py.File('mnist_batches.h5', 'r') as h5f:
    print(h5f['train/images'].shape)
    print(h5f['train/labels'].shape)
    print(h5f['train/file_ids'].shape)


with h5py.File('mnist_batches.h5', 'r') as h5f:

    plt.imshow(h5f['train/images'][0][:, :, -1], cmap='binary');
    print('Class label:', h5f['train/labels'][0])
    print('File ID:', h5f['train/file_ids'][0])


# ## 2. Loading Minibatches
# 

# The following cell implements a class for iterating over the MNIST images, based on the .h5 file, conveniently. 
# Via the `normalize` parameter we additionally scale the image pixels to [0, 1] range, which typically helps with gradient-based optimization in practice.
# 
# The key functions (here: generators) are
# 
# - load_train_epoch
# - load_valid_epoch
# - load_test_epoch
# 
# These let us iterate over small chunks (determined via `minibatch_size`) and yield minibatches via memory-efficient Python generators. Via the two shuffle parameters, we can further control if the images within each batch to be shuffled. By setting `onehot=True`, the labels are converted into a onehot representation for convenience.
# 

class BatchLoader():
    def __init__(self, minibatches_path, 
                 normalize=True):
        
        self.minibatches_path = minibatches_path
        self.normalize = normalize
        self.num_train = 45000
        self.num_valid = 5000
        self.num_test = 10000
        self.n_classes = 10


    def load_train_epoch(self, batch_size=50, onehot=False,
                         shuffle_batch=False, prefetch_batches=1, seed=None):
        for batch_x, batch_y in self._load_epoch(which='train',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_batch=shuffle_batch,
                                                 prefetch_batches=prefetch_batches, 
                                                 seed=seed):
            yield batch_x, batch_y

    def load_test_epoch(self, batch_size=50, onehot=False,
                        shuffle_batch=False, prefetch_batches=1, seed=None):
        for batch_x, batch_y in self._load_epoch(which='test',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_batch=shuffle_batch,
                                                 prefetch_batches=prefetch_batches,
                                                 seed=seed):
            yield batch_x, batch_y
            
    def load_validation_epoch(self, batch_size=50, onehot=False,
                              shuffle_batch=False, prefetch_batches=1, seed=None):
        for batch_x, batch_y in self._load_epoch(which='valid',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_batch=shuffle_batch,
                                                 prefetch_batches=prefetch_batches, 
                                                 seed=seed):
            yield batch_x, batch_y

    def _load_epoch(self, which='train', batch_size=50, onehot=False,
                    shuffle_batch=False, prefetch_batches=1, seed=None):
        
        prefetch_size = prefetch_batches * batch_size
        
        if shuffle_batch:
            rgen = np.random.RandomState(seed)

        with h5py.File(self.minibatches_path, 'r') as h5f:
            indices = np.arange(h5f['%s/images' % which].shape[0])
            
            for start_idx in range(0, indices.shape[0] - prefetch_size + 1,
                                   prefetch_size):           
            

                x_batch = h5f['%s/images' % which][start_idx:start_idx + prefetch_size]
                x_batch = x_batch.astype(np.float32)
                y_batch = h5f['%s/labels' % which][start_idx:start_idx + prefetch_size]

                if onehot:
                    y_batch = (np.arange(self.n_classes) == 
                               y_batch[:, None]).astype(np.uint8)

                if self.normalize:
                    # normalize to [0, 1] range
                    x_batch = x_batch.astype(np.float32) / 255.

                if shuffle_batch:
                    rand_indices = np.arange(prefetch_size)
                    rgen.shuffle(rand_indices)
                    x_batch = x_batch[rand_indices]
                    y_batch = y_batch[rand_indices]

                for batch_idx in range(0, x_batch.shape[0] - batch_size + 1,
                                       batch_size):
                    
                    yield (x_batch[batch_idx:batch_idx + batch_size], 
                           y_batch[batch_idx:batch_idx + batch_size])


# The following for loop will iterate over the 45,000 training examples in our MNIST training set, yielding 50 images and labels at a time (note that we previously set aside 5000 training example as our validation datast).
# 

batch_loader = BatchLoader(minibatches_path='./mnist_batches.h5', 
                           normalize=True)

for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
    print(batch_x.shape)
    print(batch_y.shape)
    break


cnt = 0
for batch_x, batch_y in batch_loader.load_train_epoch(
        batch_size=100, onehot=True):
    cnt += batch_x.shape[0]
    
print('One training epoch contains %d images' % cnt)


def one_epoch():
    for batch_x, batch_y in batch_loader.load_train_epoch(
            batch_size=100, onehot=True):
        pass
    
get_ipython().magic('timeit one_epoch()')


# As we can see from the benchmark above, an iteration over one training epoch (45k images) is relatively fast.
# 

# Similarly, we could iterate over validation and test data via 
# 
# - batch_loader.load_validation_epoch
# - batch_loader.load_test_epoch
# 
# Note that increasing the `batch_size` can substantially improve the computationally efficiency loading an epoch, since it would lower the number of iterations. Further, we used two nested for loops in `_load_epoch`, where the inner one yields the actual batches. The purpose of the outer loop in this function is to prefetch multiple batches for shuffling -- otherwise, the shuffling won't have any effect on the gradients.
# 

def one_epoch():
    for batch_x, batch_y in batch_loader.load_train_epoch(
            batch_size=100, shuffle_batch=True, prefetch_batches=4, 
            seed=123, onehot=True):
        pass
    
get_ipython().magic('timeit one_epoch()')


# Also, as we can see from the benchmark, prefetching multiple batches from the HDF5 database can speed up the loading of an epoch. Note that this could not always practicable (for example, when we are working with high-resolution images) due to memory constraints.
# 

# ## 3. Training a Model using TensorFlow's `feed_dict`
# 

# The following code demonstrate how we can feed our minibatches into a TensorFlow graph using a TensorFlow session's `feed_dict`.
# 

# ### Multilayer Perceptron Graph
# 

import tensorflow as tf

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 15
batch_size = 100

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
height, width = 28, 28
n_classes = 10


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(123)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, height, width, 1], name='features')
    tf_x_flat = tf.reshape(tf_x, shape=[-1, height*width])
    tf_y = tf.placeholder(tf.int32, [None, n_classes], name='targets')

    # Model parameters
    weights = {
        'h1': tf.Variable(tf.truncated_normal([width*height, n_hidden_1], stddev=0.1)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.1))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # Multilayer perceptron
    layer_1 = tf.add(tf.matmul(tf_x_flat, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


# ### Training the Neural Network with Minibatches
# 

##########################
### TRAINING & EVALUATION
##########################

batch_loader = BatchLoader(minibatches_path='./mnist_batches.h5', 
                           normalize=True)

# preload small validation set
# by unpacking the generator
[valid_data] = batch_loader.load_validation_epoch(batch_size=5000, 
                                                   onehot=True)
valid_x, valid_y = valid_data[0], valid_data[1]
del valid_data

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.

        n_batches = 0
        for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=batch_size, 
                                                              onehot=True,
                                                              shuffle_batch=True,
                                                              prefetch_batches=10,
                                                              seed=epoch):
            n_batches += 1
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y.astype(np.int)})
            avg_cost += c
        
        train_acc = sess.run('accuracy:0', feed_dict={'features:0': batch_x,
                                                      'targets:0': batch_y})
        
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': valid_x,
                                                      'targets:0': valid_y})  
        
        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / n_batches), end="")
        print(" | MbTrain/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
        
        
    # imagine test set is too large to fit into memory:
    test_acc, cnt = 0., 0
    for test_x, test_y in batch_loader.load_test_epoch(batch_size=100, 
                                                       onehot=True):   
        cnt += 1
        acc = sess.run(accuracy, feed_dict={'features:0': test_x,
                                            'targets:0': test_y})
        test_acc += acc
    print('Test ACC: %.3f' % (test_acc / cnt))


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- General Adversarial Networks
# 

# Implementation of General Adversarial Nets (GAN) where both the discriminator and generator are multi-layer perceptrons with one hidden layer only. In this example, the GAN generator was trained to generate MNIST images.
# 
# Uses
# 
# - samples from a random normal distribution (range [-1, 1])
# - dropout
# - leaky relus
# - ~~batch normalization~~ [performs worse here]
# - separate batches for "fake" and "real" images (where the labels are 1 = real images, 0 = fake images)
# - MNIST images normalized to [-1, 1] range
# - generator with tanh output
# 

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle as pkl

tf.test.gpu_device_name()


### Abbreviatiuons
# dis_*: discriminator network
# gen_*: generator network

########################
### Helper functions
########################

def leaky_relu(x, alpha=0.0001):
    return tf.maximum(alpha * x, x)


########################
### DATASET
########################

mnist = input_data.read_data_sets('MNIST_data')


#########################
### SETTINGS
#########################

# Hyperparameters
learning_rate = 0.001
training_epochs = 100
batch_size = 64
dropout_rate = 0.5

# Other settings
print_interval = 200

# Architecture
dis_input_size = 784
gen_input_size = 100

dis_hidden_size = 128
gen_hidden_size = 128


#########################
### GRAPH DEFINITION
#########################

g = tf.Graph()
with g.as_default():
    
    # Placeholders for settings
    dropout = tf.placeholder(tf.float32, shape=None, name='dropout')
    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')
    
    # Input data
    dis_x = tf.placeholder(tf.float32, shape=[None, dis_input_size], name='discriminator_input') 
    gen_x = tf.placeholder(tf.float32, [None, gen_input_size], name='generator_input')


    ##################
    # Generator Model
    ##################

    with tf.variable_scope('generator'):
        # linear -> ~~batch norm~~ -> leaky relu -> dropout -> tanh output
        gen_hidden = tf.layers.dense(inputs=gen_x, units=gen_hidden_size,
                                      activation=None)
        #gen_hidden = tf.layers.batch_normalization(gen_hidden, training=is_training)
        gen_hidden = leaky_relu(gen_hidden)
        gen_hidden = tf.layers.dropout(gen_hidden, rate=dropout_rate)
        gen_logits = tf.layers.dense(inputs=gen_hidden, units=dis_input_size, 
                                     activation=None)
        gen_out = tf.tanh(gen_logits, 'generator_output')


    ######################
    # Discriminator Model
    ######################
    
    def build_discriminator_graph(input_x, reuse=None):
        # linear -> ~~batch norm~~ -> leaky relu -> dropout -> sigmoid output
        with tf.variable_scope('discriminator', reuse=reuse):
            hidden = tf.layers.dense(inputs=input_x, units=dis_hidden_size, 
                                     activation=None)
            #hidden = tf.layers.batch_normalization(hidden, training=is_training)
            hidden = leaky_relu(hidden)
            hidden = tf.layers.dropout(hidden, rate=dropout_rate)
            logits = tf.layers.dense(inputs=hidden, units=1, activation=None)
            out = tf.sigmoid(logits)
        return logits, out    

    # Create a discriminator for real data and a discriminator for fake data
    dis_real_logits, dis_real_out = build_discriminator_graph(dis_x, reuse=False)
    dis_fake_logits, dis_fake_out = build_discriminator_graph(gen_out, reuse=True)


    #####################################
    # Generator and Discriminator Losses
    #####################################
    
    # Two discriminator cost components: loss on real data + loss on fake data
    # Real data has class label 0, fake data has class label 1
    dis_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logits, 
                                                            labels=tf.zeros_like(dis_real_logits))
    dis_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits, 
                                                            labels=tf.ones_like(dis_fake_logits))
    dis_cost = tf.add(tf.reduce_mean(dis_fake_loss), 
                      tf.reduce_mean(dis_real_loss), 
                      name='discriminator_cost')
 
    # Generator cost: difference between dis. prediction and label "0" for real images
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits,
                                                       labels=tf.zeros_like(dis_fake_logits))
    gen_cost = tf.reduce_mean(gen_loss, name='generator_cost')
    
    
    #########################################
    # Generator and Discriminator Optimizers
    #########################################
      
    dis_optimizer = tf.train.AdamOptimizer(learning_rate)
    dis_train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    dis_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    
    with tf.control_dependencies(dis_update_ops): # required to upd. batch_norm params
        dis_train = dis_optimizer.minimize(dis_cost, var_list=dis_train_vars,
                                           name='train_discriminator')
    
    gen_optimizer = tf.train.AdamOptimizer(learning_rate)
    gen_train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    
    with tf.control_dependencies(gen_update_ops): # required to upd. batch_norm params
        gen_train = gen_optimizer.minimize(gen_cost, var_list=gen_train_vars,
                                           name='train_generator')
    
    # Saver to save session for reuse
    saver = tf.train.Saver()


##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    avg_costs = {'discriminator': [], 'generator': []}

    for epoch in range(training_epochs):
        dis_avg_cost, gen_avg_cost = 0., 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x*2 - 1 # normalize
            batch_randsample = np.random.uniform(-1, 1, size=(batch_size, gen_input_size))
            
            # Train
            _, dc = sess.run(['train_discriminator', 'discriminator_cost:0'],
                             feed_dict={'discriminator_input:0': batch_x, 
                                        'generator_input:0': batch_randsample,
                                        'dropout:0': dropout_rate,
                                        'is_training:0': True})
            _, gc = sess.run(['train_generator', 'generator_cost:0'],
                             feed_dict={'generator_input:0': batch_randsample,
                                        'dropout:0': dropout_rate,
                                        'is_training:0': True})
            
            dis_avg_cost += dc
            gen_avg_cost += gc

            if not i % print_interval:
                print("Minibatch: %03d | Dis/Gen Cost:    %.3f/%.3f" % (i + 1, dc, gc))
                

        print("Epoch:     %03d | Dis/Gen AvgCost: %.3f/%.3f" % 
              (epoch + 1, dis_avg_cost / total_batch, gen_avg_cost / total_batch))
        
        avg_costs['discriminator'].append(dis_avg_cost / total_batch)
        avg_costs['generator'].append(gen_avg_cost / total_batch)
    
    
    saver.save(sess, save_path='./gan.ckpt')


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.plot(range(len(avg_costs['discriminator'])), 
         avg_costs['discriminator'], label='discriminator')
plt.plot(range(len(avg_costs['generator'])),
         avg_costs['generator'], label='generator')
plt.legend()
plt.show()


####################################
### RELOAD & GENERATE SAMPLE IMAGES
####################################


n_examples = 25

with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./gan.ckpt')

    batch_randsample = np.random.uniform(-1, 1, size=(n_examples, gen_input_size))
    new_examples = sess.run('generator/generator_output:0',
                            feed_dict={'generator_input:0': batch_randsample,
                                       'dropout:0': 0.0,
                                       'is_training:0': False})

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8),
                         sharey=True, sharex=True)

for image, ax in zip(new_examples, axes.flatten()):
    ax.imshow(image.reshape((dis_input_size // 28, dis_input_size // 28)), cmap='binary')

plt.show()


# *Accompanying code examples of the book "Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python" by [Sebastian Raschka](https://sebastianraschka.com). All code examples are released under the [MIT license](https://github.com/rasbt/deep-learning-book/blob/master/LICENSE). If you find this content useful, please consider supporting the work by buying a [copy of the book](https://leanpub.com/ann-and-deeplearning).*
#   
# Other code examples and content are available on [GitHub](https://github.com/rasbt/deep-learning-book). The PDF and ebook versions of the book are available through [Leanpub](https://leanpub.com/ann-and-deeplearning).
# 

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")


# # Model Zoo -- Siamese Network with Multilayer Perceptrons
# 

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


##########################
### SETTINGS
##########################

# General settings

random_seed = 0

# Hyperparameters
learning_rate = 0.001
training_epochs = 5
batch_size = 100
margin = 1.0

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 1 # for 'true' and 'false' matches


def fully_connected(inputs, output_nodes, activation=None, seed=None):

    input_nodes = inputs.get_shape().as_list()[1]
    weights = tf.get_variable(name='weights', 
                              shape=(input_nodes, output_nodes),
                              initializer=tf.truncated_normal_initializer(
                                  mean=0.0,
                                  stddev=0.001,
                                  dtype=tf.float32,
                                  seed=seed))

    biases = tf.get_variable(name='biases', 
                             shape=(output_nodes,),
                             initializer=tf.constant_initializer(
                                 value=0.0, 
                                 dtype=tf.float32))
                              
    act = tf.matmul(inputs, weights) + biases
    if activation is not None:
        act = activation(act)
    return act


def euclidean_distance(x_1, x_2):
    return tf.sqrt(tf.maximum(tf.sum(
        tf.square(x - y), axis=1, keepdims=True), 1e-06))

def contrastive_loss(x_1, x_2, margin=1.0):
    return (x_1 * tf.square(x_2) +
            (1.0 - x_1) * tf.square(tf.maximum(margin - x_2, 0.)))


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)

    # Input data
    tf_x_1 = tf.placeholder(tf.float32, [None, n_input], name='inputs_1')
    tf_x_2 = tf.placeholder(tf.float32, [None, n_input], name='inputs_2')
    tf_y = tf.placeholder(tf.float32, [None], 
                          name='targets') # here: 'true' or 'false' valuess

    # Siamese Network
    def build_mlp(inputs):
        with tf.variable_scope('fc_1'):
            layer_1 = fully_connected(inputs, n_hidden_1, 
                                      activation=tf.nn.relu)
        with tf.variable_scope('fc_2'):
            layer_2 = fully_connected(layer_1, n_hidden_2, 
                                      activation=tf.nn.relu)
        with tf.variable_scope('fc_3'):
            out_layer = fully_connected(layer_2, n_classes, 
                                        activation=tf.nn.relu)

        return out_layer
    
    
    with tf.variable_scope('siamese_net', reuse=False):
        pred_left = build_mlp(tf_x_1)
    with tf.variable_scope('siamese_net', reuse=True):
        pred_right = build_mlp(tf_x_2)
    
    # Loss and optimizer
    loss = contrastive_loss(pred_left, pred_right)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')
    
##########################
### TRAINING & EVALUATION
##########################

np.random.seed(random_seed) # set seed for mnist shuffling
mnist = input_data.read_data_sets("./", one_hot=False)

with tf.Session(graph=g) as sess:
    
    print('Initializing variables:')
    sess.run(tf.global_variables_initializer())
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope='siamese_net'):
        print(i)

    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = mnist.train.num_examples // batch_size // 2

        for i in range(total_batch):
            
            batch_x_1, batch_y_1 = mnist.train.next_batch(batch_size)
            batch_x_2, batch_y_2 = mnist.train.next_batch(batch_size)
            batch_y = (batch_y_1 == batch_y_2).astype('float32')
            
            _, c = sess.run(['train', 'cost:0'], feed_dict={'inputs_1:0': batch_x_1,
                                                            'inputs_2:0': batch_x_2,
                                                            'targets:0': batch_y})
            avg_cost += c

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)))


# - Todo: add embedding visualization
# 

