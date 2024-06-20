# # 1.3 Performing AM in Code Space
# 

# ## Introduction
# 
# **This section corresponds to Section 3.3 of the original paper.**
# 
# Learning density models $p(x)$ that directly approximate the data distribution can be difficult, or even nearly impossible for complex datasets. Generative models do not explicitly provide the density function but are able to sample from it through the following steps:
# 
# 1. Sample from a simple distribution $q(z) \sim \mathcal{N}(0,I)$ defined in some abstract code space $\mathcal{Z}$.
# 2. Apply to the sample a decoding function $g : \mathcal{Z} \rightarrow \mathcal{X}$ that maps it back to the original input domain.
# 
# One such model that have gained popularity over the recent years is the Generative Adversarial Network (GAN). It learns a decoding function $g$ such that the generated images are theoretically impossible to distinguish from real images. The decoding function (generator) and the discriminant (discriminator) are typically neural networks. Here are some [great](https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/) [blogs](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/) on GANs, if you are not familiar with them.
# 
# [Nguyen et al.](https://arxiv.org/abs/1605.09304) proposed a method of building prototype images $x^{*}$ for each labels $\omega_c$ by incorporating a pretrained generative model into the activation maximization framework. The optimization objective is redefined as (check tutorial 1.1 if you don't remember the original objective):
# 
# \begin{equation}
# \max_{z \, \in \, \mathcal{Z}} \log p(\omega_c \, | \, g(z)) - \lambda \lVert z\rVert^2
# \end{equation}
# 
# Now, instead of optimizing an image, we are optimizing the code $z$ such that the generated image $g(z)$ will maximize the activation for a particular class $\omega_c$. Once the solution $z^{*}$ to the optimization problem is found, the prototype for $\omega_c$ is generated by passing $z^{*}$ through the generator. That is, $x^{*} = g(z^{*})$. Hopefully, the prototype will turn out to be more realistic than vanilla AM, as the generator knows how to generate natural-looking images.
# 

# ## Training Details
# 
# Remember how I said we incorporate a *pretrained* generative model into the AM framework? Here I will explain how we train the generator. The paper for this particular GAN framework by Nguyen et al. can be found [here](https://arxiv.org/abs/1602.02644). First take a look at the overall schematic for the GAN framework.
# 
# ![title](./assets/1_3_AM_Code/schematic.png)
# 
# The training process involves four networks:
# 
# 1. A pretrained encoder network $E$ to be inverted.
# 2. A generator network $G$.
# 3. A pretrained classifier $C$.
# 4. A discriminator $D$.
# 
# Note that we only have three networks in the above schematic while we require four. This is because the pretrained classifier serves as both $E$ and $C$. The activation of the hidden layer of the pretrained network acts as an encoding of the input as well as a means of comparing prominent features of two images. Now that we have figured out the components of this framework, let's see how the loss functions for $G$ and $D$ are defined. There are three parts.
# 
# #### Loss in Feature Space
# 
# Given a classifier $C: \mathbb{R}^{W \times H \times C} \rightarrow \mathbb{R}^{F}$, we define
# 
# \begin{equation}
# L_{feat} = \sum_i \lVert C(G(x_i)) - C(y_i) \rVert^2
# \end{equation}
# 
# where $y_i$ is the real image and $x_i$ is the encoding of $y_i$. With this loss, we are trying to encourage the generator to produce images whose features are similar to those of real images.
# 
# #### Adversarial Loss
# 
# The discriminator loss $L_{discr}$ and the generator loss $L_{adv}$ is given as follows:
# 
# \begin{align}
# L_{discr} = - \sum_i \log (D(y_i)) + \log (1 - D(G(x_i)))
# L_{adv} = - \sum_i \log (D(G(x_i))
# \end{align}
# 
# This is just the GAN objective that constrains the generator to produce natural-looking images.
# 
# #### Loss in Image Space
# 
# Adding a pixel-wise loss stabilizes training.
# 
# \begin{equation}
# L_{img} = \sum_i \lVert G(x_i) -  y_i \rVert^2
# \end{equation}
# 
# Now that we have a pretrained generator, we can simply plug $G$ into the AM framework. As I mentioned above, we are **not** training the generator nor the classifier. We are optimizing the code $z$ that goes into the generator such that the generated image maximizes the activation for the class $\omega_c$.
# 
# ![title](./assets/1_3_AM_Code/schematic2.png)

# ## Tensorflow Walkthrough
# 

# ### 1. Import Dependencies
# 
# We import the classifier (DNN), generator and the discriminator-the three components necessary for this AM framework.
# 

import os

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from models.models_1_3 import MNIST_DNN, MNIST_G, MNIST_D
from utils import plot

get_ipython().magic('matplotlib inline')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

logdir = './tf_logs/1_3_AM_Code/'
ckptdir = logdir + 'model'

if not os.path.exists(logdir):
    os.mkdir(logdir)


# ### 2. Building DNN Graph
# 
# In this step, we initialize a DNN classifier and attach necessary nodes for model training onto the computation graph.
# 

with tf.name_scope('Classifier'):

    # Initialize neural network
    DNN = MNIST_DNN('DNN')

    # Setup training process
    lmda = tf.placeholder_with_default(0.01, shape=[], name='lambda')
    X = tf.placeholder(tf.float32, [None, 784], name='X')
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    tf.add_to_collection('placeholders', lmda)
    tf.add_to_collection('placeholders', X)
    tf.add_to_collection('placeholders', Y)

    code, logits = DNN(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_summary = tf.summary.scalar('Cost', cost)
accuray_summary = tf.summary.scalar('Accuracy', accuracy)
summary = tf.summary.merge_all()


# ### 3. Building GAN Subgraph
# 
# We now build the GAN part of the computation graph. If you see the right hand side of the $G$ cost, you can see that it is comprised of three terms. They are $L_{adv}$, $L_{img}$ and $L_{feat}$ from left to right.
# 

with tf.name_scope('GAN'):

    G = MNIST_G(z_dim=100, name='Generator')
    D = MNIST_D(name='Discriminator')

    X_fake = G(code)
    D_real = D(X)
    D_fake = D(X_fake, reuse=True)
    code_fake, logits_fake = DNN(X_fake, reuse=True)

    D_cost = -tf.reduce_mean(tf.log(D_real + 1e-7) + tf.log(1 - D_fake + 1e-7))
    G_cost = -tf.reduce_mean(tf.log(D_fake + 1e-7)) + tf.nn.l2_loss(X_fake - X) + tf.nn.l2_loss(code_fake - code)

    D_optimizer = tf.train.AdamOptimizer().minimize(D_cost, var_list=D.vars)
    G_optimizer = tf.train.AdamOptimizer().minimize(G_cost, var_list=G.vars)


# ### 4. Building Subgraph for Generating Prototypes
# 
# Before training the network, a subgraph for generating prototypes is added onto the graph. This subgraph will be used after training the model.
# 

with tf.name_scope('Prototype'):
    
    code_mean = tf.placeholder(tf.float32, [10, 100], name='code_mean')
    code_prototype = tf.get_variable('code_prototype', shape=[10, 100], initializer=tf.random_normal_initializer())

    X_prototype = G(code_prototype, reuse=True)
    Y_prototype = tf.one_hot(tf.cast(tf.lin_space(0., 9., 10), tf.int32), depth=10)
    _, logits_prototype = DNN(X_prototype, reuse=True)

    # Objective function definition
    cost_prototype = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_prototype, labels=Y_prototype))                      + lmda * tf.nn.l2_loss(code_prototype - code_mean)

    optimizer_prototype = tf.train.AdamOptimizer().minimize(cost_prototype, var_list=[code_prototype])

# Add the subgraph nodes to a collection so that they can be used after training of the network
tf.add_to_collection('prototype', code)
tf.add_to_collection('prototype', code_mean)
tf.add_to_collection('prototype', code_prototype)
tf.add_to_collection('prototype', X_prototype)
tf.add_to_collection('prototype', Y_prototype)
tf.add_to_collection('prototype', logits_prototype)
tf.add_to_collection('prototype', cost_prototype)
tf.add_to_collection('prototype', optimizer_prototype)


# This is the general structure of the computation graph visualized using tensorboard.
# 
# ![title](./assets/1_3_AM_Code/graph.png)

# ### 5. Training Network
# 
# This is the step where the DNN is trained to classify the 10 digits of the MNIST images. Summaries are written into the logdir and you can visualize the statistics using tensorboard by typing this command: `tensorboard --lodir=./tf_logs`
# 

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Hyper parameters
training_epochs = 15
batch_size = 100

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0
    avg_acc = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c, a, summary_str = sess.run([optimizer, cost, accuracy, summary], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch
        avg_acc += a / total_batch
        
        file_writer.add_summary(summary_str, epoch * total_batch + i)
    
    print('Epoch: {:04d} cost = {:.9f} accuracy = {:.9f}'.format(epoch + 1, avg_cost, avg_acc))
    
    saver.save(sess, ckptdir)

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


# ### 6. Training GAN
# 
# We now train $G$ (and $D$) according to the description above.
# 

# Hyper parameters
training_epochs = 25
batch_size = 100
img_epoch = 1

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_D_cost = 0
    avg_G_cost = 0
    
    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs}

        _, D_c = sess.run([D_optimizer, D_cost], feed_dict=feed_dict)
        _, G_c = sess.run([G_optimizer, G_cost], feed_dict=feed_dict)

        avg_D_cost += D_c / total_batch
        avg_G_cost += G_c / total_batch
        
    print('Epoch: {:04d} G cost = {:.9f} D cost = {:.9f}'.format(epoch + 1, avg_G_cost, avg_D_cost))

# Uncomment this code if you want to see the generated images.
#
#     if (epoch + 1) % img_epoch == 0:
#         samples = sess.run(X_fake, feed_dict={X: mnist.test.images[:16, :]})
#         fig = plot(samples, 784, 1)
#         plt.savefig('./assets/1_3_AM_Code/G_{:04d}.png'.format(epoch), bbox_inches='tight')
#         plt.close(fig)
    
    saver.save(sess, ckptdir)

sess.close()


# This is a visualization of the GAN training process. The image quality initially improves although it somewhat plateaus out in the later epochs.
# 
# ![title](./assets/1_3_AM_Code/train.gif)

# ### 7. Restoring Subgraph
# 
# Here we first rebuild the DNN graph from metagraph, restore DNN parameters from the checkpoint and then gather the necessary nodes for prototype generation using the `tf.get_collection()` function (recall prototype subgraph nodes were added onto the 'prototype' collection at step 4).
# 

tf.reset_default_graph()

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

# Get necessary placeholders
placeholders = tf.get_collection('placeholders')
lmda = placeholders[0]
X = placeholders[1]

# Get prototype nodes
prototype = tf.get_collection('prototype')
code = prototype[0]
code_mean = prototype[1]
X_prototype = prototype[3]
cost_prototype = prototype[6]
optimizer_prototype = prototype[7]


# ### 8. Generating Prototype Images
# 
# Before performing gradient ascent, we calculate the image means $\overline{z}$ that will be used to regularize the prototype images. Then, we generate prototype images that maximize $\log p(\omega_c \, | \, g(z)) - \lambda \lVert z - \overline{z}\rVert^2$. I used 0.1 for lambda (lmda), but fine tuning may produce better prototype images.
# 

images = mnist.train.images
labels = mnist.train.labels

code_means = []
for i in range(10):
    imgs = images[np.argmax(labels, axis=1) == i]
    img_codes = sess.run(code, feed_dict={X: imgs})
    code_means.append(np.mean(img_codes, axis=0))

for epoch in range(15000):
    _, c = sess.run([optimizer_prototype, cost_prototype], feed_dict={lmda: 0.1, code_mean: code_means})
    
    if epoch % 500 == 0:
        print('Epoch: {:05d} Cost = {:.9f}'.format(epoch, c))
    
X_prototypes = sess.run(X_prototype)

sess.close()


# ### 9. Displaying Images
# 
# By incorporating $G$ into the AM framework, we are now able to produce realistic images. Recall that just using AM resulted in blurry prototype images.
# 

plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(np.reshape(X_prototypes[2 * i], [28, 28]), cmap='gray', interpolation='none')
    plt.title('Digit: {}'.format(2 * i))
    plt.colorbar()
    
    plt.subplot(5, 2, 2 * i + 2)
    plt.imshow(np.reshape(X_prototypes[2 * i + 1], [28, 28]), cmap='gray', interpolation='none')
    plt.title('Digit: {}'.format(2 * i + 1))
    plt.colorbar()

plt.tight_layout()





# # 1.1 Activation Maximization
# 

# ## Introduction
# 
# **This section corresponds to Section 3.1 of the original paper.**
# 
# When we are asked to describe a certain concept, say, a chair, it is impossible to come up with a precise description because there are so many types of chairs. However, there are certain characteristics unique to chairs that let us recognize chairs if we see one, such as a platform that we can sit on or legs that support the platform. With such features of chair in mind, we may be able draw a picture of a typical chair. In this section, we ask a deep neural network (DNN) to give us a prototype image $x^{*}$ that describes characteristics common to the set of objects it was trained to recognize.
# 
# A deep neural network classifier mapping a set of data points or images $x$ to a set of classes $(\omega_c)_c$ models the conditional probability distribution $p(\omega_c | x)$. Therefore, a prototype $x^{*}$ can be found by optimizing the following objective:
# 
# \begin{equation}
# \max_x \log p(\omega_c \, | \, x) - \lambda \lVert x\rVert^2
# \end{equation}
# 
# This can be achieved by first training a deep neural network and then performing gradient ascent on images with randomly initialized pixel values. The rightmost term of the objective function ($\lambda \lVert x\rVert^2$) is an $l_2$-norm regularizer that prevents prototype images from deviating largely from the origin. As we will see later, activation maximization is crude compared to other interpretation techniques in that it is unable to produce natural-looking prototype images.
# 

# ## Tensorflow Walkthrough
# 

# ### 1. Import Dependencies
# 
# We import a pre-built DNN that we will train and then perform activation maximization on. You can check out `models_1_1.py` in the models directory for more network details.
# 

import os

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from models.models_1_1 import MNIST_DNN

get_ipython().magic('matplotlib inline')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

logdir = './tf_logs/1_1_AM/'
ckptdir = logdir + 'model'

if not os.path.exists(logdir):
    os.mkdir(logdir)


# ### 2. Building Graph
# 
# In this step, we initialize a DNN classifier and attach necessary nodes for model training onto the computation graph.
# 

with tf.name_scope('Classifier'):

    # Initialize neural network
    DNN = MNIST_DNN('DNN')

    # Setup training process
    lmda = tf.placeholder_with_default(0.01, shape=[], name='lambda')
    X = tf.placeholder(tf.float32, [None, 784], name='X')
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    tf.add_to_collection('placeholders', lmda)

    logits = DNN(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_summary = tf.summary.scalar('Cost', cost)
accuray_summary = tf.summary.scalar('Accuracy', accuracy)
summary = tf.summary.merge_all()


# ### 3. Building Subgraph for Generating Prototypes
# 
# Before training the network, we attach a subgraph for generating prototypes onto the computation graph.
# 

with tf.name_scope('Prototype'):

    X_mean = tf.placeholder(tf.float32, [10, 784], name='X_mean')
    X_prototype = tf.get_variable('X_prototype', shape=[10, 784], initializer=tf.constant_initializer(0.))
    Y_prototype = tf.one_hot(tf.cast(tf.lin_space(0., 9., 10), tf.int32), depth=10)

    logits_prototype = DNN(X_prototype, reuse=True)

    # Objective function definition
    cost_prototype = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_prototype, labels=Y_prototype))                      + lmda * tf.nn.l2_loss(X_prototype - X_mean)

    optimizer_prototype = tf.train.AdamOptimizer().minimize(cost_prototype, var_list=[X_prototype])

# Add the subgraph nodes to a collection so that they can be used after training of the network
tf.add_to_collection('prototype', X_mean)
tf.add_to_collection('prototype', X_prototype)
tf.add_to_collection('prototype', Y_prototype)
tf.add_to_collection('prototype', logits_prototype)
tf.add_to_collection('prototype', cost_prototype)
tf.add_to_collection('prototype', optimizer_prototype)


# Here's the general structure of the computation graph visualized using tensorboard.
# 
# ![title](./assets/1_1_Activation_Maximization/graph.png)

# ### 4. Training Network
# 
# This is the step where the DNN is trained to classify the 10 digits of the MNIST images. Summaries are written into the logdir and you can visualize the statistics using tensorboard by typing this command: `tensorboard --lodir=./tf_logs`
# 

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Hyper parameters
training_epochs = 15
batch_size = 100

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0
    avg_acc = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c, a, summary_str = sess.run([optimizer, cost, accuracy, summary], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch
        avg_acc += a / total_batch
        
        file_writer.add_summary(summary_str, epoch * total_batch + i)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(avg_acc))
    
    saver.save(sess, ckptdir)

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

sess.close()


# ### 5. Restoring Subgraph
# 
# Here we first rebuild the DNN graph from metagraph, restore the DNN parameters from the checkpoint and then gather the necessary nodes for prototype generation using the `tf.get_collection()` function (recall prototype subgraph nodes were added onto the 'prototype' collection at step 3).
# 

tf.reset_default_graph()

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

# Get necessary placeholders
lmda = tf.get_collection('placeholders')[0]

# Get prototype nodes
prototype = tf.get_collection('prototype')
X_mean = prototype[0]
X_prototype = prototype[1]
cost_prototype = prototype[4]
optimizer_prototype = prototype[5]


# ### 6. Generating Prototype Images
# 
# Before performing gradient ascent, we calculate the image means $\overline{x}$ that will be used to regularize the prototype images. Then, we generate prototype images that maximize $\log p(\omega_c | x) - \lambda \lVert x - \overline{x}\rVert^2$. I used 0.1 for lambda (lmda), but fine tuning may produce better prototype images.
# 

images = mnist.train.images
labels = mnist.train.labels

img_means = []
for i in range(10):
    img_means.append(np.mean(images[np.argmax(labels, axis=1) == i], axis=0))

for epoch in range(5000):
    _, c = sess.run([optimizer_prototype, cost_prototype], feed_dict={lmda: 0.1, X_mean: img_means})
    
    if epoch % 500 == 0:
        print('Epoch: {:05d} Cost = {:.9f}'.format(epoch, c))
    
X_prototypes = sess.run(X_prototype)

sess.close()


# ### 7. Displaying Images
# 
# As I mentioned in the introduction, the resulting prototype images are rather blurry.
# 

plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(np.reshape(X_prototypes[2 * i], [28, 28]), cmap='gray', interpolation='none')
    plt.title('Digit: {}'.format(2 * i))
    plt.colorbar()
    
    plt.subplot(5, 2, 2 * i + 2)
    plt.imshow(np.reshape(X_prototypes[2 * i + 1], [28, 28]), cmap='gray', interpolation='none')
    plt.title('Digit: {}'.format(2 * i + 1))
    plt.colorbar()

plt.tight_layout()





# # 2.3 Layer-wise Relevance Propagation Part 2.
# 

# ## Tensorflow Walkthrough
# 

# ### 1. Import Dependencies
# 
# I made a custom `LRP` class for Layer-wise Relevance Propagation. If you are interested in the details, check out `models_3_1.py` in the models directory.
# 

import os

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from models.models_2_3 import MNIST_NN, MNIST_DNN, LRP
from utils import pixel_range

get_ipython().magic('matplotlib inline')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

logdir = './tf_logs/2_3_LRP/'
ckptdir = logdir + 'model'

if not os.path.exists(logdir):
    os.mkdir(logdir)


# ### 2. Building Graph
# 

with tf.name_scope('Classifier'):

    # Initialize neural network
    DNN = MNIST_DNN('DNN')

    # Setup training process
    X = tf.placeholder(tf.float32, [None, 784], name='X')
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    activations, logits = DNN(X)
    
    tf.add_to_collection('LRP', X)
    
    for activation in activations:
        tf.add_to_collection('LRP', activation)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_summary = tf.summary.scalar('Cost', cost)
accuray_summary = tf.summary.scalar('Accuracy', accuracy)
summary = tf.summary.merge_all()


# ### 3. Training Network
# 
# This is the step where the DNN is trained to classify the 10 digits of the MNIST images. Summaries are written into the logdir and you can visualize the statistics using tensorboard by typing this command: `tensorboard --lodir=./tf_logs`
# 

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Hyper parameters
training_epochs = 15
batch_size = 100

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0
    avg_acc = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c, a, summary_str = sess.run([optimizer, cost, accuracy, summary], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch
        avg_acc += a / total_batch
        
        file_writer.add_summary(summary_str, epoch * total_batch + i)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(avg_acc))
    
    saver.save(sess, ckptdir)

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

sess.close()


# ### 4. Restoring Subgraph
# 
# Here we first rebuild the DNN graph from metagraph, restore DNN parameters from the checkpoint and then gather the necessary weights and biases for LRP using the `tf.get_collection()` function.
# 

tf.reset_default_graph()

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*kernel.*')
biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*bias.*')
activations = tf.get_collection('LRP')
X = activations[0]


# ### 5. Attaching Subgraph for Calculating Relevance Scores
# 

conv_ksize = [1, 3, 3, 1]
pool_ksize = [1, 2, 2, 1]
conv_strides = [1, 1, 1, 1]
pool_strides = [1, 2, 2, 1]

weights.reverse()
biases.reverse()
activations.reverse()

# LRP-alpha1-beta0
lrp10 = LRP(1, activations, weights, biases, conv_ksize, pool_ksize, conv_strides, pool_strides, 'LRP10')

# LRP-alpha2-beta1
lrp21 = LRP(2, activations, weights, biases, conv_ksize, pool_ksize, conv_strides, pool_strides, 'LRP21')

Rs10 = [lrp10(i) for i in range(10)]
Rs21 = [lrp21(i) for i in range(10)]


# ### 6. Calculating Relevance Scores $R(x_i)$
# 

sample_imgs = []
for i in range(10):
    sample_imgs.append(images[np.argmax(labels, axis=1) == i][3])

imgs10 = []
imgs21 = []
for i in range(10):
    imgs10.append(sess.run(Rs10[i], feed_dict={X: sample_imgs[i][None,:]}))
    imgs21.append(sess.run(Rs21[i], feed_dict={X: sample_imgs[i][None,:]}))

sess.close()


# ### 7. Displaying Images for LRP-$\alpha_1 \beta_0$
# 
# The relevance scores are visualized as heat maps. You can see which features/data points influenced the DNN most its decision making.
# 

plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    vmin, vmax = pixel_range(imgs10[2 * i])
    plt.imshow(np.reshape(imgs10[2 * i], [28, 28]), vmin=-vmax, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i))
    plt.colorbar()
    
    plt.subplot(5, 2, 2 * i + 2)
    vmin, vmax = pixel_range(imgs10[2 * i + 1])
    plt.imshow(np.reshape(imgs10[2 * i + 1], [28, 28]), vmin=-vmax, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i + 1))
    plt.colorbar()

plt.tight_layout()


# ### 8. Displaying Images for LRP-$\alpha_2 \beta_1$
# 
# You can see that for LRP-$\alpha_2 \beta_1$, there are also spots with negative relevance scores.
# 

plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    vmin, vmax = pixel_range(imgs21[2 * i])
    plt.imshow(np.reshape(imgs21[2 * i], [28, 28]), vmin=vmin, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i))
    plt.colorbar()
    
    plt.subplot(5, 2, 2 * i + 2)
    vmin, vmax = pixel_range(imgs21[2 * i + 1])
    plt.imshow(np.reshape(imgs21[2 * i + 1], [28, 28]), vmin=vmin, vmax=vmax, cmap='bwr')
    plt.title('Digit: {}'.format(2 * i + 1))
    plt.colorbar()

plt.tight_layout()





# # 2.3 Layer-wise Relevance Propagation Part 1.
# 

# ## Introduction
# 
# **This section corresponds to Sections 4.3 and 5.1 of the original paper.**
# 
# Recall how in Section 2.1 we defined a relevance function $R$ that takes in an ith pixel $x_i$ of image $x$ as an argument and returns a scalar value $R(x_i)$ which indicates how much positive or negative contribution $x_i$ has had on the final decision $f(x)$. Then as we progressed throughout Sections 2.1 and 2.2, we added several constraints to $R(x_i)$ such that it better described the rationale behind the DNN's decision. I will restate the constraints below so we can refer back to it later.
# 
# ##### Constraint 1. $R(x_i) > 0$ should indicate positive contribution and $R(x_i) < 0$ negative contribution. If the range of $R$ is limited to nonnegative numbers, $R(x_i) = 0$ indicates no contribution and $R(x_i) > 0$ indicates positive contribution (Section 2.1).
# 
# ##### Constraint 2. Relevance Conservation Constraint (Section 2.2): 
# 
# \begin{equation}
# f(x) = \sum^V_{i=1} R(x_i)
# \end{equation}
# 
# By adding *Constraint (1)* in the Sensitivity Analysis framework, we quantified the amount of influence individual pixel $x_i$ has on $f(x)$. However, sensitivity analysis did not provide an straightforward explanation of the function value  $f(x)$, but rather suggestions. By introducing *Contraint (2)* in the Simple Taylor Decomposition framework, we created an explicit relationship between $R(x_i)$ and $f(x)$. We aim to improve $R$ by attaching a third constraint that exploits the feed-forward graph structure of the DNN.
# 
# **Notation Warning: **
# so far, we defined the relevance score $R(x_i)$ as a function that takes in pixels only. However, we are now going to generalize the concept of relevance score such that it can be applied to all layers of a DNN. The $l$th layer of a DNN is modeled as a vector of pre-activations $z = (z^{(l)}_d)^{V(l)}_{d = 1}$ with dimensionality $V(l)$. That is, $z = (z^{(l)}_1, z^{(l)}_2, z^{(l)}_3, ..., z^{(l)}_{V(l)})$ where $V(l)$ is the number of nodes/neurons in the $l$th layer. Then we can define a relevance score $R^{(l)}_d$ for each pre-activation $z^{(l)}_d$ of the vector $z$ at layer $l$. See that we have now incorporated both the layer index and the neuron index into the notation of the relevance score. With this new notation, we can indicate the relevance of any neuron at any layer, even including the input layer! For example, the previous notation $R(x_i)$ is now equivalent to $R_i^{(1)}$ for $i = 1, 2, 3, ..., V(1)$. They both indicate the relevance of $i$th neuron at the first layer.
# 
# ## Layer-wise Relevance Propagation
# 
# As its name implies, LRP makes explicit use of the feed-forward graph structure of a DNN. The first layer are the inputs, the pixels of the image, and the last layer is the real-valued prediction output of the classifier $f$. LRP assumes that we have a relevance score $R^{(l+1)}_d$ for each pre-activation $z^{(l+1)}_d$ of the vector $z$ at layer $l + 1$. The idea is to find a relevance score $R^{(l)}_d$ for each dimension $z^{(l+1)}_d$ of the vector $z$ at the next layer $l$ which is close to the input layer such that the following holds:
# 
# \begin{equation}
# f(x) = \cdots = \sum_{d = 1}^{V(l+1)} R^{(l+1)}_d = \sum_{d = 1}^{V(l)} R^{(l)}_d = \cdots = \sum_{i = 1}^{V(1)} R^{(1)}_d
# \end{equation}
# 
# Iterating this equation from the last layer which is the classifier output $f(x)$ down to the input layer $x$ consisting of image pixels naturally leads to *Constraint (2)*. With this, we can refine *Relevance Conservation Constraint* into *Layer-wise Relevance Conservation Constraint*.
# 
# ##### Constraint 3. Layer-wise Relevance Conservation Constraint: total relevance must be preserved throughout layers.
# 
# Given a DNN where $j$ and $k$ are indices for neurons at two successive layers $l$ and $(l+1)$. We can define $R_{j \leftarrow k}^{(l,l+1)}$ as the portion of relevance that flows from neuron $k$ to neuron $j$. The portion is determined by the amount of contribution of neuron $j$ to $R_{k}^{(l+1)}$, subject to the *Layer-wise Relevance Conservation Constraint*:
# 
# \begin{equation}
# \sum_j R_{j \leftarrow k}^{(l,l+1)} = R_{k}^{(l+1)}
# \end{equation}
# 
# Same can be said for $R_{j}^{(l)}$:
# 
# \begin{equation}
# \sum_k R_{j \leftarrow k}^{(l,l+1)} = R_{j}^{(l)}
# \end{equation}
# 
# ## Propagation Rules for DNNs
# 
# Let the neurons of the DNN be described by the equation
# 
# \begin{equation}
# a_k = \sigma \left( \sum_j a_j w_{jk} + b_k \right)
# \end{equation}
# 
# with $a_k$ the neuron activation, $a_j$ the activations from the previous layer, $w_{jk}$ the weight and $b_k$ the bias parameters of the neuron. The activation function $\sigma$ is a positive and monotonically increasing activation function (e.g. tanh, ReLU).
# 
# One propagation rule that fulfills the three constraints mentioned above is the $\alpha \beta$-rule. Let $z_{jk}^{+} = a_j w_{jk}^{+}$ and $z_{k}^{+} = \sum_j a_j w_{jk}^{+} + b_{k}^{+} = \sum_j z_{jk}^{+} + b_{k}^{+}$. $()^{+}$ and $()^{-}$ denote the positive and negative parts respectively. Same applies to $z_{jk}^{-}$ and $z_{k}^{-}$. Then the $\alpha \beta$-rule is given by
# 
# \begin{equation}
# R_{j \leftarrow k}^{(l,l+1)} = R_{k}^{(l+1)} \cdot \left(\alpha \cdot \frac{z_{jk}^{+}}{z_{k}^{+}} - \beta \cdot \frac{z_{jk}^{-}}{z_{k}^{-}} \right)
# \end{equation}
# 
# where the parameters $\alpha$ and $\beta$ satisfy the constraint $\alpha - \beta = 1$ and $\beta \geq 0$. Then, the *Layer-wise Relevance Conservation Constraint* becomes:
# 
# \begin{align}
# \sum_j R_{j \leftarrow k}^{(l,l+1)} & = \sum_j R_{k}^{(l+1)} \cdot \left( \alpha \cdot \frac{z_{jk}^{+}}{z_{k}^{+}} - \beta \cdot \frac{z_{jk}^{-}}{z_{k}^{-}} \right) \& = R_{k}^{(l+1)} \cdot \left( \alpha \cdot \frac{\sum_j z_{jk}^{+}}{z_{k}^{+}} - \beta \cdot \frac{\sum_j z_{jk}^{-}}{z_{k}^{-}} \right) \& = R_{k}^{(l+1)} \cdot \left( \alpha \cdot \frac{z_{k}^{+} - b_{k}^{+}}{z_{k}^{+}} - \beta \cdot \frac{z_{k}^{-} - b_{k}^{-}}{z_{k}^{-}} \right) \& = R_{k}^{(l+1)} \cdot \left( (\alpha - \beta) - \alpha \cdot \frac{b_{k}^{+}}{z_{k}^{+}} + \beta \cdot \frac{b_{k}^{-}}{z_{k}^{-}} \right) \& = R_{k}^{(l+1)} \cdot \left( 1 - \alpha \cdot \frac{b_{k}^{+}}{z_{k}^{+}} + \beta \cdot \frac{b_{k}^{-}}{z_{k}^{-}} \right) \\end{align}
# 
# For different combinations of $\alpha$ and $\beta$, we name the corresponding propagation rule by subscripting $\alpha$ and $\beta$ with corresponding values. For example, choosing the parameters $\alpha = 2$ and $\beta = 1$ will give us the LRP-$\alpha_2 \beta_1$ rule.
# 
# Choosing LRP-$\alpha_1 \beta_0$ simplifies the propagation rule to and the *Layer-wise Relevance Conservation Constraint* to:
# 
# \begin{equation}
# R_{j \leftarrow k}^{(l,l+1)} = \frac{z_{jk}^{+}}{z_{k}^{+}} \cdot R_{k}^{(l+1)}
# \end{equation}
# 
# \begin{equation}
# \sum_j R_{j \leftarrow k}^{(l,l+1)} = R_{k}^{(l+1)} \cdot \left( 1 - \frac{b_{k}^{+}}{z_{k}^{+}} \right)
# \end{equation}
# 
# In addition, if all the biases are constrained or set to be $0$, the propagation rule and the *Layer-wise Relevance Conservation Constraint* can be further simplified into:
# 
# \begin{equation}
# R_{j \leftarrow k}^{(l,l+1)} = R_{k}^{(l+1)} \cdot \frac{a_j w_{jk}^{+}}{\sum_j a_j w_{jk}^{+}}
# \end{equation}
# 
# \begin{equation}
# \sum_j R_{j \leftarrow k}^{(l,l+1)} = R_{k}^{(l+1)}
# \end{equation}
# 
# In this tutorial, we are going to apply the above rule to a ReLU network without bias (which is basically equivalent to a network with $0$ bias).
# 

# ## Tensorflow Implementation Details
# 
# Consider the LRP-$\alpha_1 \beta_0$ propagation rule of *Eq. (12)*:
# 
# \begin{equation}
# R_{j}^{(l)} = a_j \sum_k \frac{w_{jk}^{+}}{\sum_j a_j w_{jk}^{+} + b_{k}^{+}} R_{k}^{(l+1)}
# \end{equation}
# 
# This rule can be written as four elementary computations, all of which can also be expressed in vector form:
# 
# ##### Element-wise
# 
# \begin{align*}
# z_k & \leftarrow \sum_j a_j w_{jk}^{+} \s_k & \leftarrow R_k / z_k \c_j & \leftarrow \sum_k w_{jk}^{+} s_k \R_j & \leftarrow a_j c_j
# \end{align*}
# 
# ##### Vector Form
# 
# \begin{align*}
# \mathbf{z} & \leftarrow W_{+}^{\top} \cdot \mathbf{a} \\mathbf{s} & \leftarrow \mathbf{R} \oslash \mathbf{z} \\mathbf{c} & \leftarrow W_{+} \cdot \mathbf{s} \\mathbf{R} & \leftarrow \mathbf{a} \odot \mathbf{c}
# \end{align*}
# 
# By applying the same procedure to negative parts, we obtain the LRP implementation for all cases of $\alpha$ and $\beta$. In fully-connected dense layers, LRP can be implemented by the following sequence of Tensorflow operations:
# 

def backprop_dense(activation, kernel, bias, relevance):
    W_p = tf.maximum(0., kernel)
    b_p = tf.maximum(0., bias)
    z_p = tf.matmul(activation, W_p) + b_p
    s_p = relevance / z_p
    c_p = tf.matmul(s_p, tf.transpose(W_p))

    W_n = tf.maximum(0., kernel)
    b_n = tf.maximum(0., bias)
    z_n = tf.matmul(activation, W_n) + b_n
    s_n = relevance / z_n
    c_n = tf.matmul(s_n, tf.transpose(W_n))

    return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)


# In convolution layers, the matrix-vector multiplications can be more efficiently implemented by `backprop` methods used for gradient propagation.
# 

def backprop_conv(self, activation, kernel, bias, relevance, strides, padding='SAME'):
    W_p = tf.maximum(0., kernel)
    b_p = tf.maximum(0., bias)
    z_p = nn_ops.conv2d(activation, W_p, strides, padding) + b_p
    s_p = relevance / z_p
    c_p = nn_ops.conv2d_backprop_input(tf.shape(activation), W_p, s_p, strides, padding)

    W_n = tf.minimum(0., kernel)
    b_n = tf.minimum(0., bias)
    z_n = nn_ops.conv2d(activation, W_n, strides, padding) + b_n
    s_n = relevance / z_n
    c_n = nn_ops.conv2d_backprop_input(tf.shape(activation), W_n, s_n, strides, padding)

    return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)


# In max-pooling layers, the original paper by [Bach et al.](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) uses a winner-take-all redistribution policy where all relevance goes to the most activated neuron in the pool. On the other hand, the paper by [Montavon et al.](https://www.sciencedirect.com/science/article/pii/S0031320316303582) uses the proportional redistribution rule where the redistribution is proportional to neuron activations in the pool:
# 
# \begin{equation}
# R_{j}^{(l)} = \frac{x_j}{\sum_j x_j} R_{k}^{(l+1)}
# \end{equation}
# 
# Like the convolution layers, redistribution in pooling layers can also be efficiently implemend by `backprop` methods.
# 

# Bach et al.'s redistribution rule
def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):
    z = nn_ops.max_pool(activation, ksize, strides, padding) + 1e-10
    s = relevance / z
    c = gen_nn_ops._max_pool_grad(activation, z, s, ksize, strides, padding)
    return activation * c


# Montavon et al.'s redistribution rule
def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):
    z = nn_ops.avg_pool(activation, ksize, strides, padding) + 1e-10
    s = relevance / z
    c = gen_nn_ops._avg_pool_grad(tf.shape(activation), s, ksize, strides, padding)
    return activation * c


# Now that we have all the tools for LRP-$\alpha \beta$, we will see its application to a DNN trained on MNIST in the next part of the tutorial.
# 



