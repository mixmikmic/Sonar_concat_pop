# # Ch `02`: Concept `08`
# 

# ## Using TensorBoard
# 

# TensorBoard is a great way to visualize what's happening behind the code. 
# 
# In this example, we'll loop through some numbers to improve our guess of the average value. Then we can visualize the results on TensorBoard. 
# 
# Let's just set ourselves up with some data to work with:
# 

import tensorflow as tf
import numpy as np

raw_data = np.random.normal(10, 1, 100)


# The moving average is defined as follows:
# 

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)

update_avg = alpha * curr_value + (1 - alpha) * prev_avg


# Here's what we care to visualize:
# 

avg_hist = tf.summary.scalar("running_average", update_avg)
value_hist = tf.summary.scalar("incoming_values", curr_value)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")


# Time to compute the moving averages. We'll also run the `merged` op to track how the values change:
# 

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: raw_data[i]})
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        writer.add_summary(summary_str, i)


# Check out the visualization by running TensorBoard from the terminal:
# 
#     $ tensorboard --logdir=path/to/logs
# 

# # Ch `08`: Concept `01`
# 

# ## Reinforcement learning
# 

# The **states** are previous history of stock prices, current budget, and current number of shares of a stock.
# 
# The **actions** are buy, sell, or hold (i.e. do nothing).
# 
# The stock market data comes from the Yahoo Finance library, `pip install yahoo-finance`.
# 

get_ipython().magic('matplotlib inline')
from yahoo_finance import Share
from matplotlib import pyplot as plt
import numpy as np
import random
import tensorflow as tf
import random


# Define an abstract class called `DecisionPolicy`:
# 

class DecisionPolicy:
    def select_action(self, current_state, step):
        pass

    def update_q(self, state, action, reward, next_state):
        pass


# Here's one way we could implement the decision policy, called a random decision policy:
# 

class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        action = random.choice(self.actions)
        return action


# That's a good baseline. Now let's use a smarter approach using a neural network:
# 

class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim):
        self.epsilon = 0.9
        self.gamma = 0.001
        self.actions = actions
        output_dim = len(actions)
        h1_dim = 200

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])
        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def select_action(self, current_state, step):
        threshold = min(self.epsilon, step / 1000.)
        if random.random() < threshold:
            # Exploit best option with probability epsilon
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)  # TODO: replace w/ tensorflow's argmax
            action = self.actions[action_idx]
        else:
            # Explore random option with probability 1 - epsilon
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})


# Define a function to run a simulation of buying and selling stocks from a market:
# 

def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist, debug=False):
    budget = initial_budget
    num_stocks = initial_num_stocks
    share_value = 0
    transitions = list()
    for i in range(len(prices) - hist - 1):
        if i % 100 == 0:
            print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1)))
        current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))
        current_portfolio = budget + num_stocks * share_value
        action = policy.select_action(current_state, i)
        share_value = float(prices[i + hist + 1])
        if action == 'Buy' and budget >= share_value:
            budget -= share_value
            num_stocks += 1
        elif action == 'Sell' and num_stocks > 0:
            budget += share_value
            num_stocks -= 1
        else:
            action = 'Hold'
        new_portfolio = budget + num_stocks * share_value
        reward = new_portfolio - current_portfolio
        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))
        transitions.append((current_state, action, reward, next_state))
        policy.update_q(current_state, action, reward, next_state)

    portfolio = budget + num_stocks * share_value
    if debug:
        print('${}\t{} shares'.format(budget, num_stocks))
    return portfolio


# We want to run simulations multiple times and average out the performances:
# 

def run_simulations(policy, budget, num_stocks, prices, hist):
    num_tries = 10
    final_portfolios = list()
    for i in range(num_tries):
        final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist)
        final_portfolios.append(final_portfolio)
    avg, std = np.mean(final_portfolios), np.std(final_portfolios)
    return avg, std


# Call the following function to use the Yahoo Finance library and obtain useful stockmarket data. 
# 

def get_prices(share_symbol, start_date, end_date, cache_filename='stock_prices.npy'):
    try:
        stock_prices = np.load(cache_filename)
    except IOError:
        share = Share(share_symbol)
        stock_hist = share.get_historical(start_date, end_date)
        stock_prices = [stock_price['Open'] for stock_price in stock_hist]
        np.save(cache_filename, stock_prices)

    return stock_prices


# Who wants to deal with stock market data without looking a pretty plots? No one. So we need this out of law:
# 

def plot_prices(prices):
    plt.title('Opening stock prices')
    plt.xlabel('day')
    plt.ylabel('price ($)')
    plt.plot(prices)
    plt.savefig('prices.png')


# Train a reinforcement learning policy:
# 

if __name__ == '__main__':
    prices = get_prices('MSFT', '1992-07-22', '2016-07-22')
    plot_prices(prices)
    actions = ['Buy', 'Sell', 'Hold']
    hist = 200
    # policy = RandomDecisionPolicy(actions)
    policy = QLearningDecisionPolicy(actions, hist + 2)
    budget = 1000.0
    num_stocks = 0
    avg, std = run_simulations(policy, budget, num_stocks, prices, hist)
    print(avg, std)


# # Ch `04`: Concept `01`
# 

# ## Linear regression for classification (just for demonstrative purposes)
# 

# Import the usual libraries:
# 

get_ipython().magic('matplotlib inline')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Let's say we have numbers that we want to classify. They'll just be 1-dimensional values. Numbers close to 5 will be given the label `[0]`, and numbers close to 2 will be given the label `[1]`, as designed here:
# 

x_label0 = np.random.normal(5, 1, 10)
x_label1 = np.random.normal(2, 1, 10)
xs = np.append(x_label0, x_label1)
labels = [0.] * len(x_label0) + [1.] * len(x_label1)

plt.scatter(xs, labels)


# Define the hyper-parameters, placeholders, and variables:
# 

learning_rate = 0.001
training_epochs = 1000

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable([0., 0.], name="parameters")


# Define the model:
# 

def model(X, w):
    return tf.add(tf.multiply(w[1], tf.pow(X, 1)),
                  tf.multiply(w[0], tf.pow(X, 0)))


# Given a model, define the cost function:
# 

y_model = model(X, w)
cost = tf.reduce_sum(tf.square(Y-y_model))


# Set up the training op, and also introduce a couple ops to calculate some metrics, such as accuracy:
# 

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(Y, tf.to_float(tf.greater(y_model, 0.5)))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))


# Prepare the session:
# 

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# Run the training op multiple times on the input data:
# 

for epoch in range(training_epochs):
    sess.run(train_op, feed_dict={X: xs, Y: labels})
    current_cost = sess.run(cost, feed_dict={X: xs, Y: labels})
    if epoch % 100 == 0:
        print(epoch, current_cost)


# Show some final metrics/results:
# 

w_val = sess.run(w)
print('learned parameters', w_val)

print('accuracy', sess.run(accuracy, feed_dict={X: xs, Y: labels}))

sess.close()


# Plot the learned function
# 

all_xs = np.linspace(0, 10, 100)
plt.plot(all_xs, all_xs*w_val[1] + w_val[0])
plt.scatter(xs, labels)
plt.show()


# # Ch `10`: Concept `02`
# 

# ## Recurrent Neural Network
# 

# Import the relevant libraries:
# 

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# Define the RNN model:
# 

class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim=10):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)
            save_path = self.saver.save(sess, 'model.ckpt')
            print('Model saved to {}'.format(save_path))

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            return output




# Now, we'll train a series predictor. Let's say we have a sequence of numbers `[a, b, c, d]` that we want to transform into `[a, a+b, b+c, c+d]`. We'll give the RNN a couple examples in the training data. Let's see how well it learns this intended transformation:
# 

if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
    train_x = [[[1], [2], [5], [6]],
               [[5], [7], [7], [8]],
               [[3], [4], [5], [7]]]
    train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],
               [3, 7, 9, 12]]
    predictor.train(train_x, train_y)

    test_x = [[[1], [2], [3], [4]],  # 1, 3, 5, 7
              [[4], [5], [6], [7]]]  # 4, 9, 11, 13
    actual_y = [[[1], [3], [5], [7]],
                [[4], [9], [11], [13]]]
    pred_y = predictor.test(test_x)
    
    print("\nLets run some tests!\n")
    
    for i, x in enumerate(test_x):
        print("When the input is {}".format(x))
        print("The ground truth output should be {}".format(actual_y[i]))
        print("And the model thinks it is {}\n".format(pred_y[i]))


# # Ch `10`: Concept `03`
# 

# ## Recurrent Neural Network on real data
# 




get_ipython().magic('matplotlib inline')
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import data_loader
import matplotlib.pyplot as plt

class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            max_patience = 3
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            while patience > 0:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1
            save_path = self.saver.save(sess, 'model.ckpt')
            print('Model saved to {}'.format(save_path))

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model.ckpt')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output





def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()





if __name__ == '__main__':
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=seq_size, hidden_dim=100)
    data = data_loader.load_series('international-airline-passengers.csv')
    train_data, actual_vals = data_loader.split_data(data)

    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])

    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i+seq_size], axis=1).tolist())
        test_y.append(actual_vals[i+1:i+seq_size+1])

    predictor.train(train_x, train_y, test_x, test_y)

    with tf.Session() as sess:
        predicted_vals = predictor.test(sess, test_x)[:,0]
        print('predicted_vals', np.shape(predicted_vals))
        plot_results(train_data, predicted_vals, actual_vals, 'predictions.png')

        prev_seq = train_x[-1]
        predicted_vals = []
        for i in range(20):
            next_seq = predictor.test(sess, [prev_seq])
            predicted_vals.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        plot_results(train_data, predicted_vals, actual_vals, 'hallucinations.png')











# # Ch `11`: Concept `01`
# 
# ## Embedding Lookup
# 

# Import TensorFlow, and begin an interactive session
# 

import tensorflow as tf
sess = tf.InteractiveSession()


# Let's say we only have 4 words in our vocabulary: *"the"*, *"fight"*, *"wind"*, and *"like"*.
# 
# Maybe each word is associated with numbers.
# 
# | Word   | Number | 
# | ------ |:------:|
# | *'the'*    | 17     |
# | *'fight'*  | 22     |
# | *'wind'*   | 35     |  
# | *'like'*   | 51     |
# 

embeddings_0d = tf.constant([17,22,35,51])



# Or maybe, they're associated with one-hot vectors.
# 
# | Word   | Vector | 
# | ------ |:------:|
# | *'the '*   | [1, 0, 0, 0]     |
# | *'fight'*  | [0, 1, 0, 0]     |
# | *'wind'*   | [0, 0, 1, 0]     |  
# | *'like'*   | [0, 0, 0, 1]     |
# 

embeddings_4d = tf.constant([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])


# This may sound over the top, but you can have any tensor you want, not just numbers or vectors.
# 
# | Word   | Tensor | 
# | ------ |:------:|
# | *'the '*   | [[1, 0] , [0, 0]]    |
# | *'fight'*  | [[0, 1] , [0, 0]]     |
# | *'wind'*   | [[0, 0] , [1, 0]]     |  
# | *'like'*   | [[0, 0] , [0, 1]]     |
# 

embeddings_2x2d = tf.constant([[[1, 0], [0, 0]],
                               [[0, 1], [0, 0]],
                               [[0, 0], [1, 0]],
                               [[0, 0], [0, 1]]])


# Let's say we want to find the embeddings for the sentence, "fight the wind".
# 

ids = tf.constant([1, 0, 2])


# We can use the `embedding_lookup` function provided by TensorFlow:
# 

lookup_0d = sess.run(tf.nn.embedding_lookup(embeddings_0d, ids))
print(lookup_0d)


lookup_4d = sess.run(tf.nn.embedding_lookup(embeddings_4d, ids))
print(lookup_4d)


lookup_2x2d = sess.run(tf.nn.embedding_lookup(embeddings_2x2d, ids))
print(lookup_2x2d)


# # Ch `09`: Concept `01`
# 

# ## Using the CIFAR-10 dataset
# 

# Download the CIFAR-10 dataset for Python here https://www.cs.toronto.edu/~kriz/cifar.html.
# 
# The documentation comes with the following helper function:
# 

import pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


# Each image is a flat vector of colored pixel values. We'll convert it to a matrix of pixels instead to make it easer to reason about the image. 
# 
# First, let's grayscale the image so that we can deal with a more natural representation.
# 
# Then, we'll crop the edges of the image away to further similify the number of dimensions in the input data.
# 
# Finally, we'll normalize the input by subtracting the mean pixel intensity and dividing by the standard deviation.
# 

import numpy as np

def clean(data):
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    grayscale_imgs = imgs.mean(1)
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    img_size = np.shape(img_data)[1]
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1.0 / np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds
    return normalized


# Here's a helper function to load and clean all the images:
# 

def read_data(directory):
    names = unpickle('{}/batches.meta'.format(directory))['label_names']
    print('names', names)

    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(directory, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels']))
        else:
            data = batch_data['data']
            labels = batch_data['labels']

    print(np.shape(data), np.shape(labels))

    data = clean(data)
    data = data.astype(np.float32)
    return names, data, labels


# Let's display some images from the dataset:
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import random
random.seed(1)

names, data, labels = read_data('./cifar-10-batches-py')

def show_some_examples(names, data, labels):
    plt.figure()
    rows, cols = 4, 4
    random_idxs = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_idxs[i]
        plt.title(names[labels[j]])
        img = np.reshape(data[j, :], (24, 24))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cifar_examples.png')

show_some_examples(names, data, labels)


# # Ch `01`: Concept `10`
# 

# ## Queue of text from file
# 

# The previous notebook (Concept 09) introduced queues. Let's see how we can use them with real world data.
# 

# Download a corpus of text.
# 
#     $ wget http://norvig.com/big.txt
# 

# Create a queue of filenames:
# 

import tensorflow as tf

filename_queue = tf.train.string_input_producer(["big.txt"])


# Define an op to dequeue a line from file:
# 

reader = tf.TextLineReader()
key_op, value_op = reader.read(filename_queue)


# Start all queue runners collected in the graph:
# 

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)


# Try reading lines from the file by dequeuing: 
# 

for i in range(100):
    key, value = sess.run([key_op, value_op])
    print(key, value)


# ## Ch `07`: Concept 02
# 

# ## Autoencoder with images
# 

# Import the autoencoder class we wrote earlier:
# 

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import pickle
import numpy as np
from autoencoder import Autoencoder


# Define some helper function to load and preprocess the data:
# 

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


# Download the CIFAR-10 dataset in Python from https://www.cs.toronto.edu/~kriz/cifar.html. Then we can load the data using the following code:
# 

names = unpickle('./cifar-10-batches-py/batches.meta')['label_names']
data, labels = [], []
for i in range(1, 6):
    filename = './cifar-10-batches-py/data_batch_' + str(i)
    batch_data = unpickle(filename)
    if len(data) > 0:
        data = np.vstack((data, batch_data['data']))
        labels = np.hstack((labels, batch_data['labels']))
    else:
        data = batch_data['data']
        labels = batch_data['labels']

data = grayscale(data)
x = np.matrix(data)
y = np.array(labels)


# Train the autoencoder on images of horses:
# 

horse_indices = np.where(y == 7)[0]
horse_x = x[horse_indices]
print(np.shape(horse_x))  # (5000, 3072)

print('Some examples of horse images we will feed to the autoencoder for training')
plt.rcParams['figure.figsize'] = (10, 10)
num_examples = 5
for i in range(num_examples):
    horse_img = np.reshape(horse_x[i, :], (32, 32))
    plt.subplot(1, num_examples, i+1)
    plt.imshow(horse_img, cmap='Greys_r')
plt.show()


input_dim = np.shape(horse_x)[1]
hidden_dim = 100
ae = Autoencoder(input_dim, hidden_dim)
ae.train(horse_x)


# Test the autoencoder on other images:
# 

test_data = unpickle('./cifar-10-batches-py/test_batch')
test_x = grayscale(test_data['data'])
test_labels = np.array(test_data['labels'])
encodings = ae.classify(test_x, test_labels)


plt.rcParams['figure.figsize'] = (100, 100)
plt.figure()
for i in range(20):
    plt.subplot(20, 2, i*2 + 1)
    original_img = np.reshape(test_x[i, :], (32, 32))
    plt.imshow(original_img, cmap='Greys_r')
    
    plt.subplot(20, 2, i*2 + 2)
    reconstructed_img = ae.decode([encodings[i]])
    plt.imshow(reconstructed_img, cmap='Greys_r')

plt.show()





# # Ch `09`: Concept `02`
# 

# ## Convolutions and maxpool
# 

# Load the CIFAR-10 images, and define helper functions to visualize the convolutions and weights:
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import cifar_tools
import tensorflow as tf

names, data, labels =     cifar_tools.read_data('./cifar-10-batches-py')


def show_conv_results(data, filename=None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def show_weights(W, filename=None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(W)[3]):
        img = W[:, :, 0, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


# Let's start with a an example image:
# 

raw_data = data[4, :]
raw_img = np.reshape(raw_data, (24, 24))
plt.figure()
plt.imshow(raw_img, cmap='Greys_r')
plt.show()


# Define the TensorFlow ops:
# 

x = tf.reshape(raw_data, shape=[-1, 24, 24, 1])
W = tf.Variable(tf.random_normal([5, 5, 1, 32]))
b = tf.Variable(tf.random_normal([32]))

conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
conv_with_b = tf.nn.bias_add(conv, b)
conv_out = tf.nn.relu(conv_with_b)

k = 2
maxpool = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Visualize the effects of running the convolution and maxpool ops:
# 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W_val = sess.run(W)
    print('weights:')
    show_weights(W_val)

    conv_val = sess.run(conv)
    print('convolution results:')
    print(np.shape(conv_val))
    show_conv_results(conv_val)
    

    conv_out_val = sess.run(conv_out)
    print('convolution with bias and relu:')
    print(np.shape(conv_out_val))
    show_conv_results(conv_out_val)


    maxpool_val = sess.run(maxpool)
    print('maxpool after all the convolutions:')
    print(np.shape(maxpool_val))
    show_conv_results(maxpool_val)


# # Ch `02`: Concept `03`
# 

# ## Interactive session
# 

# Interactive sessions are another way to use a session. Go ahead and define one:
# 

import tensorflow as tf
sess = tf.InteractiveSession()


# We have a matrix we want to invert:
# 

x = tf.constant([[1., 2.]])
neg_op = tf.negative(x)


# Since we're using an interactive session, we can just call the `eval()` method on the op.
# 

result = neg_op.eval()

print(result)


# That code's a little cleaner when using Jupyter notebooks (like this one).
# 
# Don't forget to close the session:
# 

sess.close()


