# # Image Captioning with LSTM
# 
# This is a partial implementation of "Show and Tell: A Neural Image Caption Generator" (http://arxiv.org/abs/1411.4555), borrowing heavily from Andrej Karpathy's NeuralTalk (https://github.com/karpathy/neuraltalk)
# 
# This example consists of three parts:
# 1. COCO Preprocessing - prepare the dataset by precomputing image representations using GoogLeNet
# 2. COCO RNN Training - train a network to predict image captions
# 3. COCO Caption Generation - use the trained network to caption new images
# 

# ### Output
# This notebook samples from the trained network to generate captions given an input image.
# 
# 
# ### Prerequisites
# 
# To run this notebook, you'll need the trained GoogLeNet model, as well as the trained RNN model produced by the previous notebook, `lstm_coco_trained.pkl`. This can also be downloaded from https://s3.amazonaws.com/emolson/pydata/lstm_coco_trained.pkl
# 

get_ipython().system('wget -N https://s3.amazonaws.com/emolson/pydata/lstm_coco_trained.pkl')


import sklearn
import numpy as np
import lasagne
import skimage.transform

from lasagne.utils import floatX

import theano
import theano.tensor as T

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import json
import pickle


import googlenet


cnn_layers = googlenet.build_model()
cnn_input_var = cnn_layers['input'].input_var
cnn_feature_layer = cnn_layers['loss3/classifier']
cnn_output_layer = cnn_layers['prob']

get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))


model_param_values = pickle.load(open('blvc_googlenet.pkl'))['param values']
lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)


MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (224, w*224/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*224/w, 224), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])


# Grab a random photo (not from ImageNet or MSCOCO as far as I know)
# 

get_ipython().system('wget -N http://akhopecenter.org/wp-content/uploads/2013/05/Dog-and-Cat-Wallpaper-teddybear64-16834786-1280-800-1024x640.jpg')


im = plt.imread('Dog-and-Cat-Wallpaper-teddybear64-16834786-1280-800-1024x640.jpg')


plt.imshow(im)


rawim, cnn_im = prep_image(im)


plt.imshow(rawim)


p = get_cnn_features(cnn_im)
CLASSES = pickle.load(open('blvc_googlenet.pkl'))['synset words']
print(CLASSES[p.argmax()])


SEQUENCE_LENGTH = 32
MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3 # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE = 1
CNN_FEATURE_SIZE = 1000
EMBEDDING_SIZE = 256

d = pickle.load(open('lstm_coco_trained.pkl'))
vocab = d['vocab']
word_to_index = d['word_to_index']
index_to_word = d['index_to_word']


l_input_sentence = lasagne.layers.InputLayer((BATCH_SIZE, SEQUENCE_LENGTH - 1))
l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence,
                                                     input_size=len(vocab),
                                                     output_size=EMBEDDING_SIZE,
                                                    )

l_input_cnn = lasagne.layers.InputLayer((BATCH_SIZE, CNN_FEATURE_SIZE))
l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=EMBEDDING_SIZE,
                                            nonlinearity=lasagne.nonlinearities.identity)

l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))

l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding])
l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                  num_units=EMBEDDING_SIZE,
                                  unroll_scan=True,
                                  grad_clipping=5.)
l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, EMBEDDING_SIZE))
l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=len(vocab), nonlinearity=lasagne.nonlinearities.softmax)

l_out = lasagne.layers.ReshapeLayer(l_decoder, (BATCH_SIZE, SEQUENCE_LENGTH, len(vocab)))


lasagne.layers.set_all_param_values(l_out, d['param values'])


x_cnn_sym = T.matrix()
x_sentence_sym = T.imatrix()

output = lasagne.layers.get_output(l_out, {
                l_input_sentence: x_sentence_sym,
                l_input_cnn: x_cnn_sym
                })

f = theano.function([x_cnn_sym, x_sentence_sym], output)


def predict(x_cnn):
    x_sentence = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH - 1), dtype='int32')
    words = []
    i = 0
    while True:
        i += 1
        p0 = f(x_cnn, x_sentence)
        pa = p0.argmax(-1)
        tok = pa[0][i]
        word = index_to_word[tok]
        if word == '#END#' or i >= SEQUENCE_LENGTH - 1:
            return ' '.join(words)
        else:
            x_sentence[0][i] = tok
            if word != '#START#':
                words.append(word)


x_cnn = get_cnn_features(cnn_im)


# Sample some predictions
for _ in range(5):
    print(predict(x_cnn))





# Custom `Layer` Classes
# ============
# 
# Lasagne is intended to be simple to extend. If you need to do something that isn't provided by one or a combination of the existing `Layer` classes, it is easy to create your own.
# 
# The procedure:
# - Subclass `lasagne.layers.base.Layer`
# - Implement `get_output_for` which take a Theano expression and returns a new expression.
# - Implement `get_output_shape_for` which takes a shape tuple and returns a new tuple (only needed if your operation changes the shape).
# 
# More details: https://lasagne.readthedocs.org/en/latest/user/custom_layers.html
# 

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer

_srng = T.shared_randomstreams.RandomStreams()


def theano_shuffled(input):
    n = input.shape[0]

    shuffled = T.permute_row_elements(input.T, _srng.permutation(n=n)).T
    return shuffled

class FractionalPool2DLayer(Layer):
    """
    Fractional pooling as described in http://arxiv.org/abs/1412.6071
    Only the random overlapping mode is currently implemented.
    """
    def __init__(self, incoming, ds, pool_function=T.max, **kwargs):
        super(FractionalPool2DLayer, self).__init__(incoming, **kwargs)
        if type(ds) is not tuple:
            raise ValueError("ds must be a tuple")
        if (not 1 <= ds[0] <= 2) or (not 1 <= ds[1] <= 2):
            raise ValueError("ds must be between 1 and 2")
        self.ds = ds  # a tuple
        if len(self.input_shape) != 4:
            raise ValueError("Only bc01 currently supported")
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape) # copy / convert to mutable list
        output_shape[2] = int(np.ceil(float(output_shape[2]) / self.ds[0]))
        output_shape[3] = int(np.ceil(float(output_shape[3]) / self.ds[1]))

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        _, _, n_in0, n_in1 = self.input_shape
        _, _, n_out0, n_out1 = self.output_shape

        # Variable stride across the input creates fractional reduction
        a = theano.shared(
            np.array([2] * (n_in0 - n_out0) + [1] * (2 * n_out0 - n_in0)))
        b = theano.shared(
            np.array([2] * (n_in1 - n_out1) + [1] * (2 * n_out1 - n_in1)))

        # Randomize the input strides
        a = theano_shuffled(a)
        b = theano_shuffled(b)

        # Convert to input positions, starting at 0
        a = T.concatenate(([0], a[:-1]))
        b = T.concatenate(([0], b[:-1]))
        a = T.cumsum(a)
        b = T.cumsum(b)

        # Positions of the other corners
        c = T.clip(a + 1, 0, n_in0 - 1)
        d = T.clip(b + 1, 0, n_in1 - 1)

        # Index the four positions in the pooling window and stack them
        temp = T.stack(input[:, :, a, :][:, :, :, b],
                       input[:, :, c, :][:, :, :, b],
                       input[:, :, a, :][:, :, :, d],
                       input[:, :, c, :][:, :, :, d])

        return self.pool_function(temp, axis=0)


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Seed for reproducibility
np.random.seed(42)


# Get test image
get_ipython().system('wget -N "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Rubik\'s_cube_scrambled.svg/64px-Rubik\'s_cube_scrambled.svg.png"')
im = plt.imread("64px-Rubik's_cube_scrambled.svg.png")
im = im[:, :, :3]
im = np.rollaxis(im, 2)[np.newaxis]


im.shape


l_in = lasagne.layers.InputLayer((1, 3, 64, 64))
l_fracpool = FractionalPool2DLayer(l_in, ds=(1.5, 1.5))


l_fracpool.output_shape


output = lasagne.layers.get_output(l_fracpool)


# Evaluate output - each time will be slightly different due to the stochastic pooling
outim = output.eval({l_in.input_var: im})
outim = outim[0]
outim = np.rollaxis(np.rollaxis(outim, 2), 2)
plt.imshow(outim, interpolation='nearest')


outim = output.eval({l_in.input_var: im})
outim = outim[0]
outim = np.rollaxis(np.rollaxis(outim, 2), 2)
plt.imshow(outim, interpolation='nearest')


# Finetuning a pretrained network
# =================
# 
# We can take a network which was trained on the ImageNet dataset and adapt it to our own image classification problem. This can be a useful technique when training data is too limited to train a model from scratch.
# 
# Here we try to classify images as either pancakes or waffles.
# 

import numpy as np
import theano
import theano.tensor as T
import lasagne

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import skimage.transform
import sklearn.cross_validation
import pickle
import os


# Seed for reproducibility
np.random.seed(42)


CLASSES = ['pancakes', 'waffles']
LABELS = {cls: i for i, cls in enumerate(CLASSES)}


# Dataset
# --------
# 
# Images were downloaded from Google Image Search, and placed in the directories `./images/pancakes' and './images/waffles'.
# 
# There are approximately 1300 images with a roughly even split.
# 

# Download and unpack dataset
get_ipython().system('wget -N https://s3.amazonaws.com/emolson/pydata/images.tgz   ')
get_ipython().system('tar -xf images.tgz')


# Read a few images and display
im = plt.imread('./images/pancakes/images?q=tbn:ANd9GcQ1Jtg2V7Me2uybx1rqxDMV58Ow17JamorQ3GCrW5TUyT1tcr8EMg')
plt.imshow(im)


im = plt.imread('./images/waffles/images?q=tbn:ANd9GcQ-0-8U4TAw6fn4wDpj8V34AwbhkpK9SNKwobolotFjNcgspX8wmA')
plt.imshow(im)


# Model definition for VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8

# More pretrained models are available from
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


# Download a pickle containing the pretrained weights
get_ipython().system('wget -N https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')


# Load model weights and metadata
d = pickle.load(open('vgg16.pkl'))


# Build the network and fill with pretrained weights
net = build_model()
lasagne.layers.set_all_param_values(net['prob'], d['param values'])


# The network expects input in a particular format and size.
# We define a preprocessing function to load a file and apply the necessary transformations
IMAGE_MEAN = d['mean value'][:, np.newaxis, np.newaxis]

def prep_image(fn, ext='jpg'):
    im = plt.imread(fn, ext)

    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # discard alpha channel if present
    im = im[:3]

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - IMAGE_MEAN
    return rawim, floatX(im[np.newaxis])


# Test preprocesing and show the cropped input
rawim, im = prep_image('./images/waffles/images?q=tbn:ANd9GcQ-0-8U4TAw6fn4wDpj8V34AwbhkpK9SNKwobolotFjNcgspX8wmA')
plt.imshow(rawim)


# Load and preprocess the entire dataset into numpy arrays
X = []
y = []

for cls in CLASSES:
    for fn in os.listdir('./images/{}'.format(cls)):
        _, im = prep_image('./images/{}/{}'.format(cls, fn))
        X.append(im)
        y.append(LABELS[cls])
        
X = np.concatenate(X)
y = np.array(y).astype('int32')


# Split into train, validation and test sets
train_ix, test_ix = sklearn.cross_validation.train_test_split(range(len(y)))
train_ix, val_ix = sklearn.cross_validation.train_test_split(range(len(train_ix)))

X_tr = X[train_ix]
y_tr = y[train_ix]

X_val = X[val_ix]
y_val = y[val_ix]

X_te = X[test_ix]
y_te = y[test_ix]


# We'll connect our output classifier to the last fully connected layer of the network
output_layer = DenseLayer(net['fc7'], num_units=len(CLASSES), nonlinearity=softmax)


# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.ivector()

prediction = lasagne.layers.get_output(output_layer, X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
                      dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)


# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
val_fn = theano.function([X_sym, y_sym], [loss, acc])
pred_fn = theano.function([X_sym], prediction)


# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == N:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# We need a fairly small batch size to fit a large network like this in GPU memory
BATCH_SIZE = 16


def train_batch():
    ix = range(len(y_tr))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return train_fn(X_tr[ix], y_tr[ix])

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])


for epoch in range(5):
    for batch in range(25):
        loss = train_batch()

    ix = range(len(y_val))
    np.random.shuffle(ix)

    loss_tot = 0.
    acc_tot = 0.
    for chunk in batches(ix, BATCH_SIZE):
        loss, acc = val_fn(X_val[chunk], y_val[chunk])
        loss_tot += loss * len(chunk)
        acc_tot += acc * len(chunk)

    loss_tot /= len(ix)
    acc_tot /= len(ix)
    print(epoch, loss_tot, acc_tot * 100)


def deprocess(im):
    im = im[::-1, :, :]
    im = np.swapaxes(np.swapaxes(im, 0, 1), 1, 2)
    im = (im - im.min())
    im = im / im.max()
    return im


# Plot some results from the validation set
p_y = pred_fn(X_val[:25]).argmax(-1)

plt.figure(figsize=(12, 12))
for i in range(0, 25):
    plt.subplot(5, 5, i+1)
    plt.imshow(deprocess(X_val[i]))
    true = y_val[i]
    pred = p_y[i]
    color = 'green' if true == pred else 'red'
    plt.text(0, 0, true, color='black', bbox=dict(facecolor='white', alpha=1))
    plt.text(0, 32, pred, color=color, bbox=dict(facecolor='white', alpha=1))

    plt.axis('off')





