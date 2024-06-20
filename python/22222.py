# Copyright 2016 Google Inc. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# --------------------------------------
# 
# This notebook is similar to [this python script](https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/mnist_series/mnist_onehlayer.py), and uses [this README](https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/mnist_series/03_README_mnist_layers.md).
# It implements a "one hidden layer" version of MNIST. 
# As part of this lab, you'll add another hidden layer (and make a few other small modifications).  You can make your edits in the notebook if you like, or edit the corresponding python script.
# 
# Start with some imports and variable definitions:
# 

import argparse
import math
import os
import time

from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# Define some constants.
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# Batch size. Must be evenly dividable by dataset sizes.
BATCH_SIZE = 100
EVAL_BATCH_SIZE = 3
# Number of units in hidden layers.
HIDDEN1_UNITS = 128

NUM_STEPS = 25000
DATA_DIR = "MNIST_data"
MODEL_DIR = default=os.path.join(
    "/tmp/tfmodels/mnist_onehlayer",
    str(int(time.time())))


# Next, we'll define a function that builds the core of the model graph.  We'll see below, when this function is called, that the 'images' arg passed to this function is the images input placeholder.
# 

# Build inference graph.
def mnist_inference(images, hidden1_units):
    """Build the MNIST model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        hidden1_units: Size of the first hidden layer.
    Returns:
        logits: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden1, weights) + biases
    return logits


# Next, we'll define a function that builds on the inference graph above in order to create a training graph. We define a loss function, create an optimizer, and create a 'train_op' that tells the optimizer to apply the gradients that minimize the loss.
# 

# Build training graph.
def mnist_training(logits, labels, learning_rate):
    """Build the training graph.

    Args:
        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
        labels: Labels tensor, int32 - [BATCH_SIZE], with values in the
          range [0, NUM_CLASSES).
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, loss


# Now we'll put it all together. We process the input data and generate the necessary placeholders.
# We'll also add ops to the graph for calculating accuracy.  
# We'll also add support for generating "summary" info used by TensorBoard.
# 
# Then, we create a session in which to run a training loop. 
# After training, the model is checkpointed to a file.
# 
# Finally, we show that we can load a checkpointed model file into a new session, and run a prediction based on the checkpointed model.
# 

def model_train_and_eval():
    """Build the full graph for feeding inputs, training, and
    saving checkpoints.  Run the training. Then, load the saved graph and
    run some predictions."""

    # Get input data: get the sets of images and labels for training,
    # validation, and test on MNIST.
    data_sets = read_data_sets(DATA_DIR, False)

    mnist_graph = tf.Graph()
    with mnist_graph.as_default():
        # Generate placeholders for the images and labels.
        images_placeholder = tf.placeholder(tf.float32)
        labels_placeholder = tf.placeholder(tf.int32)
        tf.add_to_collection("images", images_placeholder)  # Remember this Op.
        tf.add_to_collection("labels", labels_placeholder)  # Remember this Op.

        # Build a Graph that computes predictions from the inference model.
        logits = mnist_inference(images_placeholder,
                                 HIDDEN1_UNITS)
        tf.add_to_collection("logits", logits)  # Remember this Op.

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, loss = mnist_training(
            logits, labels_placeholder, 0.01)

        # prediction accuracy
        _, indices_op = tf.nn.top_k(logits)
        flattened = tf.reshape(indices_op, [-1])
        correct_prediction = tf.cast(
            tf.equal(labels_placeholder, flattened), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Define info to be used by the SummaryWriter. This will let
        # TensorBoard plot values during the training process.
        loss_summary = tf.scalar_summary("loss", loss)
        train_summary_op = tf.merge_summary([loss_summary])

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a summary writer.
        print("Writing Summaries to %s" % MODEL_DIR)
        train_summary_writer = tf.train.SummaryWriter(MODEL_DIR)

    # Run training for MAX_STEPS and save checkpoint at the end.
    with tf.Session(graph=mnist_graph) as sess:
        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        print("Starting training...")
        for step in xrange(NUM_STEPS):
            # Read a batch of images and labels.
            images_feed, labels_feed = data_sets.train.next_batch(BATCH_SIZE)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value, tsummary, acc = sess.run(
                [train_op, loss, train_summary_op, accuracy],
                feed_dict={images_placeholder: images_feed,
                           labels_placeholder: labels_feed})
            if step % 100 == 0:
                # Write summary info
                train_summary_writer.add_summary(tsummary, step)
            if step % 1000 == 0:
                # Print loss/accuracy info
                print('----Step %d: loss = %.4f' % (step, loss_value))
                print("accuracy: %s" % acc)

        print("\nFinished training. Writing checkpoint file.")
        checkpoint_file = os.path.join(MODEL_DIR, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        _, loss_value = sess.run(
            [train_op, loss],
            feed_dict={images_placeholder: data_sets.test.images,
                       labels_placeholder: data_sets.test.labels})
        print("Test set loss: %s" % loss_value)

    # Run evaluation based on the saved checkpoint.
    with tf.Session(graph=tf.Graph()) as sess:
        checkpoint_file = tf.train.latest_checkpoint(MODEL_DIR)
        print("\nRunning predictions based on saved checkpoint.")
        print("checkpoint file: {}".format(checkpoint_file))
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Retrieve the Ops we 'remembered'.
        logits = tf.get_collection("logits")[0]
        images_placeholder = tf.get_collection("images")[0]
        labels_placeholder = tf.get_collection("labels")[0]

        # Add an Op that chooses the top k predictions.
        eval_op = tf.nn.top_k(logits)

        # Run evaluation.
        images_feed, labels_feed = data_sets.validation.next_batch(
            EVAL_BATCH_SIZE)
        prediction = sess.run(eval_op,
                              feed_dict={images_placeholder: images_feed,
                                         labels_placeholder: labels_feed})
        for i in range(len(labels_feed)):
            print("Ground truth: %d\nPrediction: %d" %
                  (labels_feed[i], prediction.indices[i][0]))


# The payoff for all those function definitions... now we'll run `model_train_and_eval()`!
# 

model_train_and_eval()


# This model did okay... but not great.  Next we'll try adding another hidden layer.
# 
# Contrast this code with that in the [previous lab](https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/mnist_series/02_README_mnist_tflearn.md), which used TensorFlow's high-level APIs. You can probably see how it's much easier to use those high-level "pre-canned" Estimators where appropriate.
# 

# This notebook trains a Word2Vec skip-gram model over [Text8](http://mattmahoney.net/dc/textdata) data. It's based on the code [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec), with the additional use of the SummaryWriter, so that we can track training progress in TensorBoard.
# See [this example](https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/intro_word2vec/word2vec_basic_summaries.py) for its script counterpart.
# 

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import time
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from sklearn.manifold import TSNE


# Download the data from the source website if necessary.
# 

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a string.
# 

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
  
words = read_data(filename)
print('Data size %d' % len(words))


# Build the dictionary and replace rare words with UNK token.
# 

vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
del words  # Hint to reduce memory.


# Function to generate a training batch for the skip-gram model.
# 

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


# Build and train a skip-gram model.
# 

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Define info to be used by the SummaryWriter. This will let TensorBoard
  # plot loss values during the training process.
  loss_summary = tf.scalar_summary("loss", loss)
  train_summary_op = tf.merge_summary([loss_summary])

  # Add variable initializer.
  init = tf.initialize_all_variables()
  print("finished building graph.")


# Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  # Directory in which to write summary information.
  # You can point TensorBoard to this directory via:
  # $ tensorboard --logdir=/tmp/word2vec_basic/summaries
  # Tensorflow assumes this directory already exists, so we need to create it.
  timestamp = str(int(time.time()))
  if not os.path.exists(os.path.join("/tmp/word2vec_basic",
                                     "summaries", timestamp)):
    os.makedirs(os.path.join("/tmp/word2vec_basic", "summaries", timestamp))
  # Create the SummaryWriter
  train_summary_writer = tf.train.SummaryWriter(
      os.path.join(
          "/tmp/word2vec_basic", "summaries", timestamp), session.graph)

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # Also evaluate the training summary op.
    _, loss_val, tsummary = session.run(
        [optimizer, loss, train_summary_op],
        feed_dict=feed_dict)
    average_loss += loss_val
    # Write the evaluated summary info to the SummaryWriter. This info will
    # then show up in the TensorBoard events.
    train_summary_writer.add_summary(tsummary, step)

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
  print("finished training.")


# Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)


try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")





