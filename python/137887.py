# ## Convolutional Neural Network Example
# 
# Recipe of a simple ConvNet to classify the MNIST data set
# The convnet architecture is as follows:
# 
# Image -> [CONV -> RELU -> POOL] \*2 -> [FC -> RELU] \*1 -> OUT
# 
# **Test Accuracy: 98.83%**
# 

### Imports
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


### 1. Get the dataset
mnist = input_data.read_data_sets("mnist_dataset/", one_hot=True, reshape=False)


### 2. Pre-process the dataset

### 3. Define the hyper params
learning_rate = 5e-3
num_steps = 500
batch_size = 128
test_valid_size = 256


### 4. Define the architecture of your CNN
n_labels = 10

images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
labels = tf.placeholder(tf.float32, shape=[None, n_labels])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.truncated_normal([1024, n_labels]))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bd1': tf.Variable(tf.truncated_normal([1024])),
    'out': tf.Variable(tf.truncated_normal([n_labels]))
}

def conv2d(input_vol, W, b, stride=1):
    conv_layer = tf.nn.conv2d(input_vol, W, strides=[1, stride, stride, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, b)
    return tf.nn.relu(conv_layer)

def maxpool2d(input_vol, k=2, stride=2):
    return tf.nn.max_pool(input_vol, ksize=[1, k, k, 1],
                         strides=[1, stride, stride, 1],
                         padding='SAME')

def convnet(image, weights, biases, keep_prob):
    # CONV1 [28x28x1] * 32[5x5x1] -> [28x28x32]
    conv1 = conv2d(images, weights['wc1'], biases['bc1'])
    # POOL1 [28x28x32] -> [14x14x32]
    conv1 = maxpool2d(conv1)
    
    # CONV2 [14x14x32] * [5x5x64] -> [14x14x64]
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # POOL2 [14x14x64] -> [7x7x64]
    conv2 = maxpool2d(conv2)
    
    # FC1 [7x7x64] -> [1024]
    fc1 = tf.reshape(conv2, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, weights['wd1']) + biases['bd1']
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
    
    # OUT
    out = tf.matmul(fc1, weights['out']) + biases['out']
    return out


### 5. Run the training session and track the loss & validation accuracy
logits = convnet(images, weights, biases, keep_prob)

# Define the loss and the optimisation function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Accuracy
correct_predictions = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Run the graph
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    for step_i in range(num_steps):
        batch = mnist.train.next_batch(batch_size)
        session.run(optimiser,
                   feed_dict={
                       images: batch[0],
                       labels: batch[1],
                       keep_prob: 0.75
                   })

        # Calculate the batch loss and validation accuracy
        if step_i % 10 == 0: 
            loss = session.run(cost, feed_dict={
                images: batch[0],
                labels: batch[1],
                keep_prob: 1.
            })

            acc = session.run(accuracy, feed_dict={
                images: mnist.validation.images[:test_valid_size],
                labels: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.
            })

            print('Step: {:>2} '
                  'Loss: {:>10.4f} Val Acc {:.6f}'.format(
                step_i, loss, acc))
            
    # 6. Calculate the test accuracy
    test_acc = session.run(accuracy, feed_dict={
        images: mnist.test.images[:test_valid_size],
        labels: mnist.test.labels[:test_valid_size],
        keep_prob: 1.
    })
    print('Testing Accuracy: {:.4f}'.format(test_acc))





# ## Minimal Neural Network Case Study
# 

# ### Train a Softmax Linear Classifier
# 

# A bit of setup
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Generate the spiral classification data set
np.random.seed(0)
N = 100
D = 2
K = 3
X = np.zeros((N * K, D))
y = np.zeros(N * K, dtype=np.int8)

for j in range(K):
    ix = range(j*N, (j+1)*N)
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2 # theta
    X[ix] = np.c_[r * np.sin(t), r*np.cos(t)]
    y[ix] = j

# Visualise the data set
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


# Init the hyperparams
learn_rate = 1e-0
reg = 1e-3
epochs = 200

# Define the Softmax classifier
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

n_examples = X.shape[0]
# Gradient descent loop
for epoch_i in range(epochs):
    
    # 1. Perform the forward pass
    # Compute the class scores and loss
    logits = np.dot(X, W) + b

    softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    cross_entropy = -np.log(softmax[range(n_examples), y])
    data_loss = np.mean(cross_entropy)
    reg_loss = 0.5* reg * np.sum(W * W)
    loss = data_loss + reg_loss
    
    if epoch_i % 10 == 0:
        print("Epochs: {0}/{1} Loss: {2}".format(epoch_i, epochs, loss))

    # 2. Backpropate the gradient to the parameters (W,b)
    dlogits = softmax
    dlogits[range(n_examples),y] -= 1
    dlogits /= n_examples

    dW = np.dot(X.T, dlogits)
    db = np.sum(dlogits, axis=0, keepdims=True)
    dW += reg*W # regularization gradient

    # 3. Perform the param update process
    W += -learn_rate * dW
    b += -learn_rate * db

# Evaluate the training accuracy
logits = np.dot(X, W) + b
predictions = np.argmax(logits, axis=1)
print("Training Accuracy: {0:.2f}".format(np.mean(predictions == y)))


# Plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()


# ### Train a 2 layer NN
# 

# Define the hyper params
learn_rate = 1e-0
reg = 1e-3
epochs = 10000 

# Init the NN params
n_hidden = 100 # size of the hidden layer
W1 = 0.01 * np.random.randn(D, n_hidden)
b1 = np.zeros((1, n_hidden))

W2 = 0.01 * np.random.rand(n_hidden, K)
b2 = np.zeros((1, K))

n_examples = X.shape[0]

# Gradient Descent loop
for epoch_i in range(epochs):
    # 1. Forward pass
    # Compute the class scores and loss
    hidden_layer = np.dot(X, W1) + b1
    hidden_layer = np.maximum(0, hidden_layer) # ReLU Activiation function

    scores = np.dot(hidden_layer, W2) + b2
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    cross_entropy = -np.log(softmax[range(n_examples), y])
    data_loss = np.mean(cross_entropy)
    reg_loss = (0.5* reg * np.sum(W1 * W1)) +                 (0.5 * reg * np.sum(W2 * W2))
    loss = data_loss + reg_loss
    
    if epoch_i % 1000 == 0:
        print("Epochs: {0}/{1} Loss: {2}".format(epoch_i, epochs, loss))

    # 2. Backpropate the gradient to the parameters (W2, b2, W1, b1)
    # 2nd Layer
    dscores = softmax
    dscores[range(n_examples),y] -= 1
    dscores /= n_examples

    dW2 = np.dot(hidden_layer.T, dscores) + reg * W2
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dhidden_layer = np.dot(dscores, W2.T)
    dhidden_layer[hidden_layer <= 0] = 0 # ReLU gradient

    # 1st Layer
    dW1 = np.dot(X.T, dhidden_layer) + reg * W1
    db1 = np.sum(dhidden_layer, axis=0, keepdims=True)
    
    # 3. Perform the param update process
    W2 += -learn_rate * dW2
    b2 += -learn_rate * db2
    W1 += -learn_rate * dW1
    b1 += -learn_rate * db1
    
# Training accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predictions = np.argmax(scores, axis=1)
print("Training accuracy: {0:.2f}".format(np.mean(predictions == y)))


# Visualise the 2 Layer NN
step = 0.02
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(xmin, xmax, step),
                     np.arange(ymin, ymax, step))

hidden_layer = np.dot(np.c_[np.ravel(xx), np.ravel(yy)], W1) + b1
hidden_layer = np.maximum(0, hidden_layer)
scores = np.dot(hidden_layer, W2) + b2
predictions = np.argmax(scores, axis=1)
predictions = predictions.reshape(xx.shape)

plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)


