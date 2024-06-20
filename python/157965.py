# # Using Long Short Term Memory (LSTM) networks for stock price prediction
# 
# In this project, we are using tensorflow to predict 7-day returns on the S&P 500 index (^GSPC). 
# You can download the data needed to run this notebook from yahoo finance at https://finance.yahoo.com/quote/%5EGSPC/history?p=^GSPC. 
# 

import pandas as pd
import tensorflow as tf
import numpy as np
import time 
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Normalization of input and output data
# Input data to the network will be normalized using z-score normalization. The normalization will be run separately at every input sequence so as to account for numbers differing widely between years (they are much higher today than 30 years ago).
# 

def normalize_df(df):
    '''
    z-score normalization for a single time window
    '''
    return (df - df.mean(axis=0)) / df.std(axis = 0)

def normalize_return(v):
    '''
    Normalize returns to range [-1, 1]
    '''
    return v / max_abs_return

def restore_return(v):
    return v * max_abs_return


# ## Preparing the inputs
# Data will be processed by feeding sequences of 60 days to the network. Per batch, 64 of these sequences will be processed. The newest data will be used for testing the model.
# 

gspc = pd.read_csv("gspc.csv") # get these from yahoo finance

gspc = gspc[-365*7:]


in_advance = 7
sequence_size = 60
batch_size = 64

n_layers = 3
n_cells = 32

returns = (gspc.iloc[in_advance:]["Adj Close"].values / gspc.iloc[:-in_advance]["Adj Close"] - 1).values
gspc = gspc[:-in_advance]


n_testbatches = 2
gspc_train = gspc.iloc[:-batch_size*n_testbatches-sequence_size]
y_train = returns[:-batch_size*n_testbatches-sequence_size]
gspc_test  = gspc.iloc[-batch_size*n_testbatches-sequence_size:]
y_test  = returns[-batch_size*n_testbatches-sequence_size:]


max_abs_return = np.max(np.abs(returns))


# ## Defining methods for accessing normalized training and testing batches
# 

def get_sequence(n=sequence_size):
    start = np.random.randint(0, len(gspc_train) - sequence_size - 1)
    x = gspc_train.iloc[start:start+n][['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    y = y_train[start + n]
    return np.asarray(normalize_df(x).values), normalize_return(y)

def get_test_sequence(n=sequence_size, start=0):
    x = gspc_test.iloc[start:start+n][['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    y = y_test[start + n]
    return np.asarray(normalize_df(x).values), normalize_return(y)
    

def get_batch(b=batch_size, n=sequence_size):
    return [np.asarray(get_sequence(n)) for _ in range(b)]

def get_test_batch(i=0, b=batch_size, n=sequence_size):
    return [np.asarray(get_test_sequence(n, b * i + bk)) for bk in range(b)]
    


# ## Creating the LSTM model
# The input is being fed to the network in the form of batches holding 64 sequences of data for days following on each other directly. For each day, there exist 5 values (open, high, low, adj. close, volume). The LSTM needs the input to have a different form; the variable `Xs` thus is a list of dimensions sequence_size x batch_size x n_features. We take the last output of our LSTM and apply a matrix multiplication to get our predicted values.
# 

tf.reset_default_graph()

X = tf.placeholder(dtype=tf.float32, shape=(batch_size, sequence_size, 5), name="X")
Xs = [tf.squeeze(part) for part in tf.split(X, sequence_size, axis=1)]
Y = tf.placeholder(dtype=tf.float32, shape=batch_size, name="y")

cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_cells) for _ in range(n_layers)])
initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
outputs, state = tf.contrib.rnn.static_rnn(cells, Xs, initial_state=initial_state)
last_output = outputs[-1]

outmatrix = tf.Variable(tf.truncated_normal([n_cells, 1]), dtype=tf.float32)
outbias   = tf.Variable(tf.constant(0.001))


result = tf.squeeze(tf.matmul(last_output, outmatrix) + outbias)


#result = tf.squeeze(tf.layers.dense(last_output, units=1, activation=tf.nn.tanh))
loss = tf.reduce_mean(tf.losses.absolute_difference(result, Y))


# ## Creating an optimizer and applying gradient clipping
# 

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# ## Training the model
# Losses converge relatively fast.
# 

for i in range(10000):
    batch = get_batch()
    
    
    x = np.asarray([b[0] for b in batch])
    y = np.asarray([b[1] for b in batch])
    
    sess.run(train_op, feed_dict={X: x, Y: y})
    
    if i % 10 == 0:
        ls, res = sess.run([loss, result], feed_dict={X: x, Y: y})
        print(i, ls)
        


# ## Comparing predictions and real data on the testing set
# 

testbatch = get_test_batch(0)
x = np.asarray([b[0] for b in testbatch])
y = np.asarray([b[1] for b in testbatch])

testbatch2 = get_test_batch(1)
x2 = np.asarray([b[0] for b in testbatch2])
y2 = np.asarray([b[1] for b in testbatch2])


ls, res = sess.run([loss, result], feed_dict={X: x, Y: y})
res2 =    sess.run(result, feed_dict={X: x2})

res = np.squeeze(np.append(res, res2))

restored = restore_return(res)
restored_true = np.append(restore_return(y), restore_return(y2))


plt.figure()
plt.title("7-day returns on S&P 500")
plt.plot(restored, label='Predicted')
plt.plot(restored_true, label='True')

mn = [np.mean(returns) for _ in range(len(restored))]
plt.plot(mn, label='Mean (complete data)', ls="--")
plt.legend()

print(np.mean(np.absolute(mn - restored_true)))
print(np.mean(np.absolute(restored - restored_true)))


# ## Conclusion
# The prediction accuracy on the trained model is higher than the mean return baseline. However, the provided data seems not to be sufficient for reliably solving this task. Stock prediction is considered to be one of the most difficult time series prediction problems. Possible adjustments could include
# * Providing the data of multiple securities
# * Adding fundamental data--the model currently is completely relying on technical analysis of a single index
# * Adding less obvious data sources like sentiment analysis etc.
# 




# # Training a DCGAN to draw human faces
# 
# This notebook does not contain much documentation text. If you are wondering about the DCGAN code shown below, please take a look at the code of a DCGAN for MNIST creation. The architecture of this network is basically the same.
# 
# ## Examples created
# See the _examples_ directory, the _lfw_ images have been created by this network.
# 
# 
# ## What to consider
# If you want to train this model yourself, please make sure you have a decent GPU--the example images were created after running the model on a Tesla K80 for several hours.
# 
# 
# 
# ## Downloading the LFW (Labeled Faces in the Wild) data
# 

url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
filename = "lfw.tgz"
directory = "imgs"
new_dir = "new_imgs"
import urllib
import tarfile
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.misc import imresize, imsave
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Saving the LFW files to a directory
# 

if not os.path.isdir(directory):
    if not os.path.isfile(filename):
        urllib.urlretrieve (url, filename)
    tar = tarfile.open(filename, "r:gz")
    tar.extractall(path=directory)
    tar.close()


# ## Modifying the images (reducing their size)
# 

filepaths = []
for dir_, _, files in os.walk(directory):
    for fileName in files:
        relDir = os.path.relpath(dir_, directory)
        relFile = os.path.join(relDir, fileName)
        filepaths.append(directory + "/" + relFile)
        
for i, fp in enumerate(filepaths):
    img = imread(fp) #/ 255.0
    img = imresize(img, (40, 40))
    imsave(new_dir + "/" + str(i) + ".png", img)        


filepaths_new = []
for dir_, _, files in os.walk(new_dir):
    for fileName in files:
        if not fileName.endswith(".png"):
            continue
        relDir = os.path.relpath(dir_, directory)
        relFile = os.path.join(relDir, fileName)
        filepaths_new.append(directory + "/" + relFile)


# ## Definition of a method to access 40 x 40 x 3 face images
# 

def next_batch(num=64, data=filepaths_new):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [imread(data[i]) for i in idx]

    shuffled = np.asarray(data_shuffle)
    
    return np.asarray(data_shuffle)


# ## Code for creating montages (by Parag Mital)
# 

# Code by Parag Mital (https://github.com/pkmital/CADL/)
def montage(images):    
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


# ## Definition of the neural network
# 

tf.reset_default_graph()
batch_size = 64
n_noise = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 40, 40, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 40, 40, 3])
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x
    
def generator(z, keep_prob=keep_prob, is_training=is_training):
    activation = lrelu
    momentum = 0.9
    with tf.variable_scope("generator", reuse=None):
        x = z
        
        d1 = 4#3
        d2 = 3
        
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[10, 10])
        
        
        
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x    


# ## Losses and optimizers
# 

g = generator(noise, keep_prob, is_training)
print(g)
d_real = discriminator(X_in)
d_fake = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))

loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss_d + d_reg, var_list=vars_d)
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(loss_g + g_reg, var_list=vars_g)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# ## Training the network
# 

for i in range(60000):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5
    
    
    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)   
    batch = [b for b in next_batch(num=batch_size)]  
    
    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})
    
    d_fake_ls_init = d_fake_ls
    
    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls
        
    if g_ls * 1.35 < d_ls:
        train_g = False
        pass
    if d_ls * 1.35 < g_ls:
        train_d = False
        pass
    
    if train_d:
        sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
        
        
    if train_g:
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})
        
        
    if not i % 10:
        print (i, d_ls, g_ls)
        if not train_g:
            print("not training generator")
        if not train_d:
            print("not training discriminator")
        gen_imgs = sess.run(g, feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        imgs = [img[:,:,:] for img in gen_imgs]
        m = montage(imgs)
        #m = imgs[0]
        plt.axis('off')
        plt.imshow(m, cmap='gray')
        plt.show()


# # Teaching a Deep Convolutional Generative Adversarial Network (DCGAN) to draw MNIST characters
# 
# In the last tutorial, we learnt using tensorflow for designing a Variational Autoencoder (VAE) that could draw MNIST characters. Most of the created digits looked nice. There was only one drawback -- some of the created images looked a bit fuzzy. The VAE was trained with the _mean squared error_ loss function. However, it's quite difficult to encode exact character edge locations, which leads to the network being unsure about those edges. And does it really matter if the edge of a character starts a few pixels more to the left or right? Not really.
# In this article, we will see how we can train a network that does not depend on the mean squared error or any related loss functions--instead, it will learn all by itself what a real image should look like.
# 
# ## Deep Convolutional Generative Adversarial Networks
# Another network architecture for learning to generate new content is the DCGAN. Like the VAE, our DCGAN consists of two parts:
# * The _discriminator_ learns how to distinguish fake from real objects of the type we'd like to create
# * The _generator_ creates new content and tries to fool the discriminator
# 
# There is a HackerNoon article by Chanchana Sornsoontorn that explains the concept in more detail and describes some creative projects DCGANs have been applied to. One of these projects is the generation of MNIST characters. Let's try to use python and tensorflow for the same purpose.
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Code by Parag Mital (github.com/pkmital/CADL)
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


# ## Setting up the basics
# Like in the last tutorial, we use tensorflow's own method for accessing batches of MNIST characters. We set our batch size to be 64. Our generator will take noise as input. The number of these inputs is being set to 100. Batch normalization considerably improved the training of this network. For tensorflow to apply batch normalization, we need to let it know whether we are in training mode. The variable _keep_prob_ will be used by our dropout layers, which we introduce for more stable learning outcomes.
# _lrelu_ defines the popular leaky ReLU, that hopefully will be supported by future versions of tensorflow! I firstly tried to apply standard ReLUs to this network, but this lead to the well-known _dying ReLU problem_, and I received generated images that looked like artwork by Kazimir Malevich--I just got black squares. 
# 
# Then, we define a function _binary_cross_entropy_, which we will use later, when computing losses.
# 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')
tf.reset_default_graph()
batch_size = 64
n_noise = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


# ## The discriminator
# Now, we can define the discriminator. It looks similar to the encoder part of our VAE. As input, it takes real or fake MNIST digits (28 x 28 pixel grayscale images) and applies a series of convolutions. Finally, we use a sigmoid to make sure our output can be interpreted as the probability to that the input image is a real MNIST character.
# 

def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x


# ## The generator
# The generator--just like the decoder part in our VAE--takes noise and tries to learn how to transform this noise into digits. To this end, it applies several transpose convolutions. At first, I didn't apply batch normalization to the generator, and its learning seemed to be really unefficient. After applying batch normalization layers, learning improved considerably. Also, I firstly had a much larger dense layer accepting the generator input. This led to the generator creating the same output always, no matter what the input noise was. Tuning the generator honestly took quite some effort!
# 

def generator(z, keep_prob=keep_prob, is_training=is_training):
    activation = lrelu
    momentum = 0.99
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x    


# ## Loss functions and optimizers
# Now, we wire both parts together, like we did for the encoder and the decoder of our VAE in the last tutorial.
# However, we have to create two objects of our discriminator
# * The first object receives the real images
# * The second object receives the fake images
# 
# _reuse_ of the second object is set to _True_ so both objects share their variables. We need both instances for computing two types of losses:
# * when receiving real images, the discriminator should learn to compute high values (near _1_), meaning that it is confident the input images are real
# * when receiving fake images, it should compute low values (near _0_), meaning it is confident the input images are not real
# 
# To accomplish this, we use the _binary cross entropy_ function defined earlier. The generator tries to achieve the opposite goal, it tries to make the discriminator assign high values to fake images.
# 
# Now, we also apply some regularization. We create two distinct optimizers, one for the discriminator, one for the generator. We have to define which variables we allow these optimizers to modify, otherwise the generator's optimizer could just mess up the discriminator's variables and vice-versa.
# 
# We have to provide the __update_ops__ to our optimizers when applying batch normalization--take a look at the tensorflow documentation for more information on this topic.
# 

g = generator(noise, keep_prob, is_training)
d_real = discriminator(X_in)
d_fake = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))
loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d + d_reg, var_list=vars_d)
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g + g_reg, var_list=vars_g)
    
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# ## Training the DCGAN
# Finally, the fun part begins--let's train our network! 
# We feed random values to our generator, which will learn to create digits out of this noise. We also take care that neither the generator nor the discriminator becomes too strong--otherwise, this would inhibit the learning of the other part and could even stop the network from learning anything at all (I unfortunately have made this experience).
# 

for i in range(60000):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5
    
    
    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)   
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]  
    
    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})
    
    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls
    
    if g_ls * 1.5 < d_ls:
        train_g = False
        pass
    if d_ls * 2 < g_ls:
        train_d = False
        pass
    
    if train_d:
        sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
        
        
    if train_g:
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})
        
        
    if not i % 50:
        print (i, d_ls, g_ls, d_real_ls, d_fake_ls)
        if not train_g:
            print("not training generator")
        if not train_d:
            print("not training discriminator")
        gen_img = sess.run(g, feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        imgs = [img[:,:,0] for img in gen_img]
        m = montage(imgs)
        gen_img = m
        plt.axis('off')
        plt.imshow(gen_img, cmap='gray')
        plt.show()


# ## Results
# Take a look at the pictures drawn by our generator--they look more realistic than the pictures drawn by the VAE, which looked more fuzzy at their edges. Training however took much longer than training the other model.
# 
# In conclusion, training the DCGAN took me much longer than training the VAE. Maybe fine-tuning the architecture could speed up the network's learning. Nonetheless, it's a real advantage that we are not dependent on loss functions based on pixel positions, making the results look less fuzzy. This is especially important when creating more complex data--e.g. pictures of human faces. So, just be a little patient--then everything is possible in the world of deep learning!
# 




# # Teaching a Variational Autoencoder (VAE) to draw MNIST characters
# 
# Autoencoders are a type of neural network that can be used to learn efficient codings of input data. 
# Given some inputs, the network firstly applies a series of transformations that map the input data into a lower dimensional space. This part of the network is called the _encoder_. Then, the network uses the encoded data to try and recreate the inputs. This part of the network is the _decoder_. Using the encoder, we can later compress data of the type that is understood by the network. However, autoencoders are rarely used for this purpose, as usually there exist hand-crafted algorithms (like _jpg_-compression) that are more efficient. Instead, autoencoders have repeatedly been applied to perform denoising tasks. Then, the encoder receives pictures that have been tampered with noise, and it learns how to reconstruct the original images.
# 
# 
# ## Variational Autoencoders put simply
# But there exists a much more interesting application for autoencoders. This application is called the _variational autoencoder_. Using variational autoencoders, it's not only possible to compress data -- it's also possible to generate new objects of the type the autoencoder has seen before.
# 
# Using a general autoencoder, we don't know anything about the coding that's been generated by our network. We could take a look at and compare different encoded objects, but it's unlikely that we'll be able to understand what's going on. This means that we won't be able to use our decoder for creating new objects -- we simply don't know what the inputs should look like.
# 
# Using a variational autoencoder, we take the opposite approach instead. We will not try to make guesses concerning the distribution that's being followed by the latent vectors. We simply tell our network what we want this distribution to look like. Usually, we will constrain the network to produce latent vectors having entries that follow the unit normal distribution. Then, when trying to generate data, we can simply sample some values from this distribution, feed them to the decoder, and the decoder will return us completely new objects that appear just like the objects our network has been trained with.
# 
# Let's see how this can be done using python and tensorflow. We are going to teach our network how to draw MNIST characters.
# 

# ## First steps -- Loading the training data
# Firstly, we perform some basic imports. Tensorflow has a quite handy function that allows us to easily access the MNIST data set.
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


# ## Defining our input and output data
# MNIST images have a dimension of 28 * 28 pixels with one color channel. Our inputs _X_in_ will be batches of MNIST characters, while our network will learn to reconstruct them and output them in a placeholder _Y_, which thus has the same dimensions. _Y_flat_ will be used later, when computing losses. _keep_prob_ will be used when applying dropouts as a means of regularization. During training, it will have a value of 0.8. When generating new data, we won't apply dropout, so the value will be 1. The function _lrelu_ is being defined as tensorflow unfortunately doesn't come up with a predefined leaky ReLU.
# 

tf.reset_default_graph()

batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


# ## Defining the encoder
# As our inputs are images, it's most reasonable to apply some convolutional transformations to them. What's most noteworthy is the fact that we are creating two vectors in our encoder, as the encoder is supposed to create objects following a Gaussian Distribution:
# * A vector of means
# * A vector of standard deviations
# 
# You will see later how we "force" the encoder to make sure it really creates values following a Normal Distribution. The returned values that will be fed to the decoder are the _z_-values. We will need the mean and standard deviation of our distributions later, when computing losses. 
# 

def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd


# ## Defining the decoder
# The decoder does not care about whether the input values are sampled from some specific distribution that has been defined by us. It simply will try to reconstruct the input images. To this end, we use a series of transpose convolutions.
# 

def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img


# Now, we'll wire together both parts:
# 

sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)


# ## Computing losses and enforcing a Gaussian latent distribution
# For computing the image reconstruction loss, we simply use squared difference (which could lead to images sometimes looking a bit fuzzy). This loss is combined with the _Kullback-Leibler divergence_, which makes sure our latent values will be sampled from a normal distribution. For more on this topic, please take a look a Jaan Altosaar's great article on VAEs. 
# 

unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# ## Training the network
# Now, we can finally train our VAE! Every 200 steps, we'll take a look at what the current reconstructions look like. After having processed about 2000 batches, most reconstructions will look reasonable.
# 

for i in range(30000):
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
        
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, dst_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))


# ## Generating new data
# The most awesome part is that we are now able to create new characters. To this end, we simply sample values from a unit normal distribution and feed them to our decoder. Most of the created characters look just like they've been written by humans.  
# 

randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')


# ## Conclusion
# Now, this obviously is a relatively simple example of an application of VAEs. But just think about what could be possible! Neural networks could learn to compose music. They could automatically create illustrations for books, games etc. With a bit of creativity, VAEs will open up space for some awesome projects 
# 

