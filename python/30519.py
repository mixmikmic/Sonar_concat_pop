# # Train convolutional network for sentiment analysis. 
# 
# Based on
# "Convolutional Neural Networks for Sentence Classification" by Yoon Kim
# http://arxiv.org/pdf/1408.5882v2.pdf
# 
# For `CNN-non-static` gets to 82.1% after 61 epochs with following settings:
# embedding_dim = 20          
# filter_sizes = (3, 4)
# num_filters = 3
# dropout_prob = (0.7, 0.8)
# hidden_dims = 100
# 
# For `CNN-rand` gets to 78-79% after 7-8 epochs with following settings:
# embedding_dim = 20          
# filter_sizes = (3, 4)
# num_filters = 150
# dropout_prob = (0.25, 0.5)
# hidden_dims = 150
# 
# For `CNN-static` gets to 75.4% after 7 epochs with following settings:
# embedding_dim = 100          
# filter_sizes = (3, 4)
# num_filters = 150
# dropout_prob = (0.25, 0.5)
# hidden_dims = 150
# 
# * it turns out that such a small data set as "Movie reviews with one
# sentence per review"  (Pang and Lee, 2005) requires much smaller network
# than the one introduced in the original article:
# - embedding dimension is only 20 (instead of 300; 'CNN-static' still requires ~100)
# - 2 filter sizes (instead of 3)
# - higher dropout probabilities and
# - 3 filters per filter size is enough for 'CNN-non-static' (instead of 100)
# - embedding initialization does not require prebuilt Google Word2Vec data.
# Training Word2Vec on the same "Movie reviews" data set is enough to 
# achieve performance reported in the article (81.6%)
# 
# Another distinct difference is sliding MaxPooling window of length=2
# instead of MaxPooling over whole feature map as in the article
# 

import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D

from sklearn.cross_validation import train_test_split

np.random.seed(2)


model_variation = 'CNN-rand'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)


# Model Hyperparameters
sequence_length = 56
embedding_dim = 20          
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150


# Training parameters
batch_size = 32
num_epochs = 2


# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 10        # Context window size    


print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()



if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')    


data = np.append(x,y,axis = 1)


train, test = train_test_split(data, test_size = 0.15,random_state = 0)


X_test = test[:,:56]
Y_test = test[:,56:58]


X_train = train[:,:56]
Y_train = train[:,56:58]
train_rows = np.random.randint(0,X_train.shape[0],2500)
X_train = X_train[train_rows]
Y_train = Y_train[train_rows]


print("Vocabulary Size: {:d}".format(len(vocabulary)))


def initialize():
    
    global graph_in
    global convs
    
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []


#Buliding the first layer (Convolution Layer) of the network
def build_layer_1(filter_length):
    
   
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    return conv


#Adding a max pooling layer to the model(network)
def add_max_pooling(conv):
    
    pool = MaxPooling1D(pool_length=2)(conv)
    return pool


#Adding a flattening layer to the model(network), before adding a dense layer
def add_flatten(conv_or_pool):
    
    flatten = Flatten()(conv_or_pool)
    return flatten


def add_sequential(graph):
    
    #main sequential model
    model = Sequential()
    if not model_variation=='CNN-static':
        model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    
    return model


#1.Convolution 2.Flatten
def one_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    flatten = add_flatten(conv)
    
    convs.append(flatten)
    out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=1, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


#1.Convolution 2.Max Pooling 3.Flatten
def two_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=num_epochs, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Flatten
def three_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    
    conv = build_layer_1(4)
    flatten = add_flatten(conv)
    
    convs.append(flatten)
    
    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=1, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Max Pooling 6.Flatten
def four_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    
    conv = build_layer_1(4)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    
    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=num_epochs, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Flatten\none_layer_convolution()')


get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Max Pooling 3.Flatten\ntwo_layer_convolution()')


get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Flatten\nthree_layer_convolution()')


get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Max Pooling 6.Flatten\nfour_layer_convolution()')


# A basic sequence-to-sequence model, as introduced in Cho et al., 2014 (pdf), consists of two recurrent neural networks (RNNs): an encoder that processes the input and a decoder that generates the output. 
# 
# Every Seq2seq model has 2 primary layers : the encoder and the decoder.  Generally, the encoder encodes the input sequence to an internal representation called 'context vector'  which is used by the decoder to generate the output sequence. 
# 
# The lengths of input and output sequences can be different, as there is no explicit one on one relation between the input and output sequences. 
# 
# #Source : https://github.com/farizrahman4u/seq2seq
# 

# 
# # An implementation of sequence to sequence learning for performing addition
# 
# Input: "535+61"
# Output: "596"
# Padding is handled by using a repeated sentinel character (space)
# Input may optionally be inverted, shown to increase performance in many tasks in:
# "Learning to Execute"
# http://arxiv.org/abs/1410.4615
# and
# "Sequence to Sequence Learning with Neural Networks"
# http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
# Theoretically it introduces shorter term dependencies between source and target.
# Two digits inverted:
# + One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs
# Three digits inverted:
# + One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs
# Four digits inverted:
# + One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs
# Five digits inverted:
# + One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
# 

from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

chars = '0123456789+ '
ctable = CharacterTable(chars, MAXLEN)

questions = []
expected = []
seen = set()


get_ipython().run_cell_magic('time', '', "\n#Generating random  numbers to perofrm addition on\nprint('Generating data...')\nwhile len(questions) < TRAINING_SIZE:\n    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))\n    a, b = f(), f()\n    # Skip any addition questions we've already seen\n    # Also skip any such that X+Y == Y+X (hence the sorting)\n    key = tuple(sorted((a, b)))\n    if key in seen:\n        continue\n    seen.add(key)\n    # Pad the data with spaces such that it is always MAXLEN\n    q = '{}+{}'.format(a, b)\n    query = q + ' ' * (MAXLEN - len(q))\n    ans = str(a + b)\n    # Answers can be of maximum size DIGITS + 1\n    ans += ' ' * (DIGITS + 1 - len(ans))\n    if INVERT:\n        query = query[::-1]\n    questions.append(query)\n    expected.append(ans)\nprint('Total addition questions:', len(questions))\n\n#We now have 50000 examples of addition, each exaple contains the addition between two numbers\n#Each example contains the first number followed by '+' operand followed by the second number \n#examples - 85+96, 353+551, 6+936\n#The answers to the additon operation are stored in expected")


#Look into the training data









get_ipython().run_cell_magic('time', '', "\n#The above questions and answers are going to be one hot encoded, \n#before training.\n#The encoded values will be used to train the model\n#The maximum length of a question can be 7 \n#(3 digits followed by '+' followed by 3 digits)\n#The maximum length of an answer can be 4 \n#(Since the addition of 3 digits yields either a 3 digit number or a 4\n#4 digit number)\n\n#Now for training each number or operand is going to be one hot encode below\n#In one hot encode there are 12 possibilities '0123456789+ ' (The last one is a space)\n#Since we assume a maximum of 3 digit numbers, a two digit number is taken as space with two digts, or \n#a single digit number as two spaces with a number\n\n#So for questions we get 7 rows since the max possible length is 7, and each row has a length of 12 because it will\n#be one hot encoded with True and False, depending on the character(any one of the number, '+' operand, or space)\n#will be stored  in X_train and X_val\n#The 4th position in(1,2,3,4,5,6,7) will indicate the one hot encoding of the '+' operand\n\n##So for questions we get 4 rows since the max possible length is 4, and each row has a length of 12 because it will\n#be one hot encoded with True and False, depending on the character(any one of the number, '+' operand, or space)\n#will be stored  in y_train and y_val\n\n\nprint('Vectorization...')\nX = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)\ny = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)\nfor i, sentence in enumerate(questions):\n    X[i] = ctable.encode(sentence, maxlen=MAXLEN)\nfor i, sentence in enumerate(expected):\n    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)\n\n# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits\nindices = np.arange(len(y))\nnp.random.shuffle(indices)\nX = X[indices]\ny = y[indices]\n\n# Explicitly set apart 10% for validation data that we never train over\nsplit_at = len(X) - len(X) / 10\n(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))\n(y_train, y_val) = (y[:split_at], y[split_at:])")


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


get_ipython().run_cell_magic('time', '', '\n#Training the model with the encoded inputs\nprint(\'Build model...\')\nmodel = Sequential()\n# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE\n# note: in a situation where your input sequences have a variable length,\n# use input_shape=(None, nb_feature).\nmodel.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))\n# For the decoder\'s input, we repeat the encoded input for each time step\nmodel.add(RepeatVector(DIGITS + 1))\n# The decoder RNN could be multiple layers stacked or a single layer\nfor _ in range(LAYERS):\n    model.add(RNN(HIDDEN_SIZE, return_sequences=True))\n\n# For each of step of the output sequence, decide which character should be chosen\nmodel.add(TimeDistributed(Dense(len(chars))))\nmodel.add(Activation(\'softmax\'))')


get_ipython().run_cell_magic('time', '', "\nmodel.compile(loss='categorical_crossentropy',\n              optimizer='adam',\n              metrics=['accuracy'])\n\n# Train the model each generation and show predictions against the validation dataset\nfor iteration in range(1, 2):\n    print()\n    print('-' * 50)\n    print('Iteration', iteration)\n    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,\n              validation_data=(X_val, y_val))\n    \n    score = model.evaluate(X_val, y_val, verbose=0)\n    print('\\n')\n    print('Test score:', score[0])\n    print('Test accuracy:', score[1])\n    print('\\n')")


get_ipython().run_cell_magic('time', '', "\n#For predicting the outputs, the predict method will return \n#an one hot encoded ouput, we decode the one hot encoded \n#ouptut to get our final output\n\n# Select 10 samples from the validation set at random so we can visualize errors\nfor i in range(10):\n    ind = np.random.randint(0, len(X_val))\n    rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]\n    preds = model.predict_classes(rowX, verbose=0)\n    q = ctable.decode(rowX[0])\n    correct = ctable.decode(rowy[0])\n    guess = ctable.decode(preds[0], calc_argmax=False)\n    print('Q', q[::-1] if INVERT else q)\n    print('T', correct)\n    print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)\n    print('---')")





# # Extracting twitter data
# 
# Source: https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/
# 

import tweepy
from tweepy import OAuthHandler
 
consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)


from tweepy import Stream
from tweepy.streaming import StreamListener
 
class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('euro_python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
 
twitter_stream = Stream(auth, MyListener())
#Looking for the tweets to scrape, will scrape tweets with '#europython','#EuroPython','#ep2016','#EP2016'
twitter_stream.filter(track=['#europython','#EuroPython','#ep2016','#EP2016'])





