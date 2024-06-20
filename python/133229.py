# # High-level Keras (Theano) Example
# 

import os
import sys
import numpy as np
os.environ['KERAS_BACKEND'] = "theano"
import keras as K
import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from common.params import *
from common.utils import *


# channels_first is faster
K.backend.set_image_data_format('channels_first')


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("Theano: ", theano.__version__)
print(K.backend.backend())
# Should be channels-first, otherwise slow
print(K.backend.image_data_format())


#CuDNN auto-tune
theano.config.dnn.conv.algo_fwd = "time_once"
theano.config.dnn.conv.algo_bwd_filter = "time_once"
theano.config.dnn.conv.algo_bwd_data = "time_once"


def create_symbol():
    model = Sequential()
    
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(3, 32, 32)))
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))
    return model


def init_model(m):
    m.compile(
        loss = "categorical_crossentropy",
        optimizer = K.optimizers.SGD(LR, MOMENTUM),
        metrics = ['accuracy'])
    return m


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True, one_hot=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


model.summary()


get_ipython().run_cell_magic('time', '', '# Train model\nmodel.fit(x_train,\n          y_train,\n          batch_size=BATCHSIZE,\n          epochs=EPOCHS,\n          verbose=1)')


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(x_test, batch_size=BATCHSIZE)\ny_guess = np.argmax(y_guess, axis=-1)\ny_truth = np.argmax(y_test, axis=-1)')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level TF Example
# 

import numpy as np
import os
import sys
import tensorflow as tf
from common.params import *
from common.utils import *


os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = "1"


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("GPU: ", get_gpu_name())


def create_symbol(training):
    """ TF pooling requires a boolean flag for dropout, faster when using
    'channels_first' for data_format """
    conv1 = tf.layers.conv2d(X, filters=50, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    relu1 = tf.nn.relu(conv1)
    conv2 = tf.layers.conv2d(relu1, filters=50, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), 
                                    padding='valid', data_format='channels_first')
    relu2 = tf.nn.relu(pool1)
    drop1 = tf.layers.dropout(relu2, 0.25, training=training)
    
    conv3 = tf.layers.conv2d(drop1, filters=100, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    relu3 = tf.nn.relu(conv3)
    conv4 = tf.layers.conv2d(relu3, filters=100, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), 
                                    padding='valid', data_format='channels_first')
    relu4 = tf.nn.relu(pool2)
    drop2 = tf.layers.dropout(relu4, 0.25, training=training)   
    
    flatten = tf.reshape(drop2, shape=[-1, 100*8*8])
    fc1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
    drop3 = tf.layers.dropout(fc1, 0.5, training=training)
    logits = tf.layers.dense(drop3, N_CLASSES, name='output')
    return logits


def init_model(m):
    # Single-class labels, don't need dense one-hot
    # Expects unscaled logits, not output of tf.nn.softmax
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=m, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=MOMENTUM)
    training_op = optimizer.minimize(loss)
    return training_op


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Place-holders\nX = tf.placeholder(tf.float32, shape=[None, 3, 32, 32])\ny = tf.placeholder(tf.int32, shape=[None])\ntraining = tf.placeholder(tf.bool)  # Indicator for dropout layer\n# Initialise model\nsym = create_symbol(training)')


get_ipython().run_cell_magic('time', '', 'model = init_model(sym)\ninit = tf.global_variables_initializer()\nsess = tf.Session()\nsess.run(init)\n# Accuracy logging\ncorrect = tf.nn.in_top_k(sym, y, 1)\naccuracy = tf.reduce_mean(tf.cast(correct, tf.float32))')


get_ipython().run_cell_magic('time', '', 'for j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        sess.run(model, feed_dict={X: data, y: label, training: True})\n    # Log\n    acc_train = sess.run(accuracy, feed_dict={X: data, y: label, training: True})\n    print(j, "Train accuracy:", acc_train)')


get_ipython().run_cell_magic('time', '', 'n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    pred = tf.argmax(sym,1)\n    output = sess.run(pred, feed_dict={X: data, training: False})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level RNN Keras (TF) Example
# 

import os
import sys
import numpy as np
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, CuDNNGRU
from common.params_lstm import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print(K.backend.backend())


def create_symbol(CUDNN=True):
    model = Sequential()
    model.add(Embedding(MAXFEATURES, EMBEDSIZE, input_length=MAXLEN))
    # Only return last output
    if not CUDNN:
        model.add(GRU(NUMHIDDEN, return_sequences=False, return_state=False))
    else:
        model.add(CuDNNGRU(NUMHIDDEN, return_sequences=False, return_state=False))
    model.add(Dense(2, activation='softmax'))
    return model


def init_model(m):
    m.compile(
        loss = "categorical_crossentropy",
        optimizer = K.optimizers.Adam(LR, BETA_1, BETA_2, EPS),
        metrics = ['accuracy'])
    return m


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, one_hot=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


model.summary()


get_ipython().run_cell_magic('time', '', '# Train model\nmodel.fit(x_train,\n          y_train,\n          batch_size=BATCHSIZE,\n          epochs=EPOCHS,\n          verbose=1)')


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(x_test, batch_size=BATCHSIZE)\ny_guess = np.argmax(y_guess, axis=-1)\ny_truth = np.argmax(y_test, axis=-1)')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level Theano + Lasagne Example
# 

import numpy as np
import os
import sys
import theano.tensor as T
import theano
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as nl
import lasagne.objectives as obj
import lasagne.updates as upd
from common.params import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Theano: ", theano.__version__)
print("Lasagne: ", lasagne.__version__)
print("GPU: ", get_gpu_name())


#CuDNN auto-tune
theano.config.dnn.conv.algo_fwd = "time_once"
theano.config.dnn.conv.algo_bwd_filter = "time_once"
theano.config.dnn.conv.algo_bwd_data = "time_once"


def create_symbol():
    conv1 = L.Conv2DLayer(X, num_filters=50, filter_size=(3, 3), pad='same')
    conv2 = L.Conv2DLayer(conv1, num_filters=50, filter_size=(3, 3), pad='same')
    pool1 = L.MaxPool2DLayer(conv2, pool_size=(2, 2), stride=(2, 2))
    drop1 = L.DropoutLayer(pool1, 0.25)
    
    conv3 = L.Conv2DLayer(drop1, num_filters=100, filter_size=(3, 3), pad='same')
    conv4 = L.Conv2DLayer(conv3, num_filters=100, filter_size=(3, 3), pad='same')
    pool2 = L.MaxPool2DLayer(conv4, pool_size=(2, 2), stride=(2, 2))
    drop2 = L.DropoutLayer(pool2, 0.25)
    
    flatten = L.FlattenLayer(drop2)
    fc1 = L.DenseLayer(flatten, 512)
    drop4 = L.DropoutLayer(fc1, 0.5)
    pred = L.DenseLayer(drop4, N_CLASSES, name="output", nonlinearity=nl.softmax)
    
    return pred


def init_model(net):
    pred = L.get_output(net)
    params = L.get_all_params(net)
    xentropy = obj.categorical_crossentropy(pred, y)
    loss = T.mean(xentropy)
    # The tensorflow LR, MOMENTUM are slightly different
    updates = upd.momentum(loss, params, learning_rate=LR, momentum=MOMENTUM)
    return pred, loss, updates


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Place-holders\nX = L.InputLayer(shape=(None, 3, 32, 32))\ny = T.ivector("y")\n# Initialise model\nnet = create_symbol()')


get_ipython().run_cell_magic('time', '', 'pred, loss, updates = init_model(net)\n# Accuracy for logging\naccuracy = obj.categorical_accuracy(pred, y)\naccuracy = T.mean(T.cast(accuracy, theano.config.floatX))')


get_ipython().run_cell_magic('time', '', '# Compile functions\ntrain_func = theano.function([X.input_var, y], [loss, accuracy], updates=updates)\npred = L.get_output(net, deterministic=True)\npred_func = theano.function([X.input_var], T.argmax(pred, axis=1))')


get_ipython().run_cell_magic('time', '', 'for j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        loss, acc_train = train_func(data, label)\n    # Log\n    print(j, "Train accuracy:", acc_train)')


get_ipython().run_cell_magic('time', '', 'n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    output = pred_func(data)\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level PyTorch Example
# 

import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.init as init
from torch.autograd import Variable
from common.params import *
from common.utils import *


# Big impact on training-time (from 350 to 165s)
torch.backends.cudnn.benchmark=True # enables cudnn's auto-tuner


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())


class SymbolModule(nn.Module):
    def __init__(self):
        super(SymbolModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        # feature map size is 8*8 by pooling
        self.fc1 = nn.Linear(100*8*8, 512)
        self.fc2 = nn.Linear(512, N_CLASSES)

    def forward(self, x):
        """ PyTorch requires a flag for training in dropout """
        x = self.conv2(F.relu(self.conv1(x)))
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        x = F.dropout(x, 0.25, training=self.training)

        x = self.conv4(F.relu(self.conv3(x)))
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        x = F.dropout(x, 0.25, training=self.training)

        x = x.view(-1, 100*8*8)   # reshape Variable
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        # nn.CrossEntropyLoss() contains softmax, don't apply twice
        #return F.log_softmax(x)
        return self.fc2(x)


def init_model(m):
    # Implementation of momentum:
    # v = \rho * v + g \\
    # p = p - lr * v
    opt = optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    return opt, criterion


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\n# Torch-specific\ny_train = y_train.astype(np.int64)\ny_test = y_test.astype(np.int64)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', 'sym = SymbolModule()\nsym.cuda() # CUDA!')


get_ipython().run_cell_magic('time', '', 'optimizer, criterion = init_model(sym)')


get_ipython().run_cell_magic('time', '', '# Sets training = True\nsym.train()  \nfor j in range(EPOCHS):\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = Variable(torch.FloatTensor(data).cuda())\n        target = Variable(torch.LongTensor(target).cuda())\n        # Init\n        optimizer.zero_grad()\n        # Forwards\n        output = sym(data)\n        # Loss\n        loss = criterion(output, target)\n        # Back-prop\n        loss.backward()\n        optimizer.step()\n    # Log\n    print(j)')


get_ipython().run_cell_magic('time', '', '# Test model\n# Sets training = False\nsym.eval()\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, target in yield_mb(x_test, y_test, BATCHSIZE):\n    # Get samples\n    data = Variable(torch.FloatTensor(data).cuda())\n    # Forwards\n    output = sym(data)\n    pred = output.data.max(1)[1].cpu().numpy().squeeze()\n    # Collect results\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level Keras (TF) Example
# 

import os
import sys
import numpy as np
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras as K
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from common.params import *
from common.utils import *


# channels_last is faster
K.backend.set_image_data_format('channels_last')


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tensorflow.__version__)
print(K.backend.backend())
# Channels should be last (otherwise slow)
print(K.backend.image_data_format())
print("GPU: ", get_gpu_name())


os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def create_symbol():
    model = Sequential()
    
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))
    return model


def init_model(m):
    m.compile(
        loss = "categorical_crossentropy",
        optimizer = K.optimizers.SGD(LR, MOMENTUM),
        metrics = ['accuracy'])
    return m


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=False, one_hot=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


model.summary()


get_ipython().run_cell_magic('time', '', '# Train model\nmodel.fit(x_train,\n          y_train,\n          batch_size=BATCHSIZE,\n          epochs=EPOCHS,\n          verbose=1)')


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(x_test, batch_size=BATCHSIZE)\ny_guess = np.argmax(y_guess, axis=-1)\ny_truth = np.argmax(y_test, axis=-1)')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level CNTK Example
# 

import numpy as np
import os
import sys
import cntk
from cntk.layers import Convolution2D, MaxPooling, Dense, Dropout
from common.params import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("CNTK: ", cntk.__version__)
print("GPU: ", get_gpu_name())


def create_symbol():
    # Weight initialiser from uniform distribution
    # Activation (unless states) is None
    with cntk.layers.default_options(init = cntk.glorot_uniform(), activation = cntk.relu):
        x = Convolution2D(filter_shape=(3, 3), num_filters=50, pad=True)(features)
        x = Convolution2D(filter_shape=(3, 3), num_filters=50, pad=True)(x)
        x = MaxPooling((2, 2), strides=(2, 2), pad=False)(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(filter_shape=(3, 3), num_filters=100, pad=True)(x)
        x = Convolution2D(filter_shape=(3, 3), num_filters=100, pad=True)(x)
        x = MaxPooling((2, 2), strides=(2, 2), pad=False)(x)
        x = Dropout(0.25)(x)    
        
        x = Dense(512)(x)
        x = Dropout(0.5)(x)
        x = Dense(N_CLASSES, activation=None)(x)
        return x


def init_model(m):
    # Loss (dense labels); check if support for sparse labels
    loss = cntk.cross_entropy_with_softmax(m, labels)  
    # Momentum SGD
    # https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_use_learners.ipynb
    # unit_gain=False: momentum_direction = momentum*old_momentum_direction + gradient
    # if unit_gain=True then ...(1-momentum)*gradient
    learner = cntk.momentum_sgd(m.parameters,
                                lr=cntk.learning_rate_schedule(LR, cntk.UnitType.minibatch) ,
                                momentum=cntk.momentum_schedule(MOMENTUM), 
                                unit_gain=False)
    trainer = cntk.Trainer(m, (loss, cntk.classification_error(m, labels)), [learner])
    return trainer


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True, one_hot=True)\n# CNTK format\ny_train = y_train.astype(np.float32)\ny_test = y_test.astype(np.float32)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Placeholders\nfeatures = cntk.input_variable((3, 32, 32), np.float32)\nlabels = cntk.input_variable(N_CLASSES, np.float32)\n# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', 'trainer = init_model(sym)')


get_ipython().run_cell_magic('time', '', '# Train model\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        trainer.train_minibatch({features: data, labels: label})\n    # Log (this is just last batch in epoch, not average of batches)\n    eval_error = trainer.previous_minibatch_evaluation_average\n    print("Epoch %d  |  Accuracy: %.6f" % (j+1, (1-eval_error)))')


get_ipython().run_cell_magic('time', '', '# Predict and then score accuracy\n# Apply softmax since that is only applied at training\n# with cross-entropy loss\nz = cntk.softmax(sym)\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = np.argmax(y_test[:n_samples], axis=-1)\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    predicted_label_probs = z.eval({features : data})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = np.argmax(predicted_label_probs, axis=-1)\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level RNN CNTK Example
# 

import numpy as np
import os
import sys
import cntk
from cntk.layers import Embedding, LSTM, GRU, Dense, Recurrence
from cntk import sequence
from common.params_lstm import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("CNTK: ", cntk.__version__)
print("GPU: ", get_gpu_name())


def create_symbol(CUDNN=True):
    # Weight initialiser from uniform distribution
    # Activation (unless states) is None
    with cntk.layers.default_options(init = cntk.glorot_uniform()):
        x = Embedding(EMBEDSIZE)(features) # output: list of len=BATCHSIZE of arrays with shape=(MAXLEN, EMBEDSIZE)
        
        # Since we have a vanilla RNN, instead of using the more flexible Recurrence(GRU) unit, which allows for
        # example LayerNormalisation to be added to the network, we can use optimized_rnnstack which quickly
        # goes down to the CuDNN level. This is another reason not to read much into the speed comparison because
        # it becomes a measure of which framework has the fastest way to go down to CuDNN.
        if not CUDNN:
            x = Recurrence(GRU(NUMHIDDEN))(x) # output: list of len=BATCHSIZE of arrays with shape=(MAXLEN, NUMHIDDEN)
        else:
            W = cntk.parameter((cntk.InferredDimension, 4))
            x = cntk.ops.optimized_rnnstack(x, W, NUMHIDDEN, num_layers=1, bidirectional=False, recurrent_op='gru')
        
        x = sequence.last(x) #o utput: array with shape=(BATCHSIZE, NUMHIDDEN)
        x = Dense(2)(x) # output: array with shape=(BATCHSIZE, 2)
        return x


def init_model(m):
    # Loss (dense labels); check if support for sparse labels
    loss = cntk.cross_entropy_with_softmax(m, labels)  
    # ADAM, set unit_gain to False to match others
    learner = cntk.adam(m.parameters,
                        lr=cntk.learning_rate_schedule(LR, cntk.UnitType.minibatch) ,
                        momentum=cntk.momentum_schedule(BETA_1), 
                        variance_momentum=cntk.momentum_schedule(BETA_2),
                        epsilon=EPS,
                        unit_gain=False)
    trainer = cntk.Trainer(m, (loss, cntk.classification_error(m, labels)), [learner])
    return trainer


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, one_hot=True)# CNTK format\ny_train = y_train.astype(np.float32)\ny_test = y_test.astype(np.float32)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Placeholders\nfeatures = sequence.input_variable(shape=MAXFEATURES, is_sparse=True)\nlabels = cntk.input_variable(2)\n# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', 'trainer = init_model(sym)')


get_ipython().run_cell_magic('time', '', '# Train model\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        data_1hot = cntk.Value.one_hot(data, MAXFEATURES) #TODO: do this externally and generate batches of 1hot\n        trainer.train_minibatch({features: data_1hot, labels: label})\n    # Log (this is just last batch in epoch, not average of batches)\n    eval_error = trainer.previous_minibatch_evaluation_average\n    print("Epoch %d  |  Accuracy: %.6f" % (j+1, (1-eval_error)))')


get_ipython().run_cell_magic('time', '', '# Predict and then score accuracy\n# Apply softmax since that is only applied at training\n# with cross-entropy loss\nz = cntk.softmax(sym)\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = np.argmax(y_test[:n_samples], axis=-1)\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    data = cntk.Value.one_hot(data, MAXFEATURES)\n    predicted_label_probs = z.eval({features : data})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = np.argmax(predicted_label_probs, axis=-1)\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level Gluon Example
# 

import os
import sys
import numpy as np
import math
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from common.params import *
from common.utils import *


ctx = mx.gpu()


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("MXNet: ", mx.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())


def SymbolModule():
    sym = gluon.nn.Sequential()
    with sym.name_scope():
        sym.add(gluon.nn.Conv2D(channels=50, kernel_size=3, padding=1, activation='relu'))
        sym.add(gluon.nn.Conv2D(channels=50, kernel_size=3, padding=1))
        sym.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        sym.add(gluon.nn.Activation('relu'))
        # Equiv to gluon.nn.LeakyReLU(0)
        sym.add(gluon.nn.Dropout(0.25))
        sym.add(gluon.nn.Conv2D(channels=100, kernel_size=3, padding=1, activation='relu'))
        sym.add(gluon.nn.Conv2D(channels=100, kernel_size=3, padding=1))
        sym.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        sym.add(gluon.nn.Activation('relu'))
        sym.add(gluon.nn.Dropout(0.25))
        sym.add(gluon.nn.Flatten())
        sym.add(gluon.nn.Dense(512, activation='relu'))
        sym.add(gluon.nn.Dropout(0.25))
        sym.add(gluon.nn.Dense(N_CLASSES))
    return sym


def init_model(m):
    trainer = gluon.Trainer(m.collect_params(), 'sgd',
                            {'learning_rate': LR, 'momentum':MOMENTUM})
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    return trainer, criterion


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\n\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', 'sym = SymbolModule()\nsym.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)')


get_ipython().run_cell_magic('time', '', 'trainer, criterion = init_model(sym)')


get_ipython().run_cell_magic('time', '', "# Sets training = True \nfor j in range(EPOCHS):\n    train_loss = 0.0\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = nd.array(data).as_in_context(ctx)\n        target = nd.array(target).as_in_context(ctx)\n        with autograd.record():\n            # Forwards\n            output = sym(data)\n            # Loss\n            loss = criterion(output, target)\n        # Back-prop\n        loss.backward()\n        trainer.step(data.shape[0])\n        train_loss += nd.sum(loss).asscalar()\n    # Log\n    print('Epoch %3d: loss: %5.4f'%(j, train_loss/len(x_train)))")


get_ipython().run_cell_magic('time', '', '# Test model\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, target in yield_mb(x_test, y_test, BATCHSIZE):\n    # Get samples\n    data = nd.array(data).as_in_context(ctx)\n    # Forwards\n    output = sym(data)\n    pred = nd.argmax(output, axis=1)\n    # Collect results\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred.asnumpy()\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level CNTK Example
# 

import numpy as np
import os
import sys
import cntk
from cntk.layers import Convolution2D, MaxPooling, Dense, Dropout
from common.params import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("CNTK: ", cntk.__version__)
print("GPU: ", get_gpu_name())


def create_symbol():
    # Weight initialiser from uniform distribution
    # Activation (unless states) is None
    with cntk.layers.default_options(init = cntk.glorot_uniform(), activation = cntk.relu):
        x = Convolution2D(filter_shape=(3, 3), num_filters=50, pad=True)(features)
        x = Convolution2D(filter_shape=(3, 3), num_filters=50, pad=True)(x)
        x = MaxPooling((2, 2), strides=(2, 2), pad=False)(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(filter_shape=(3, 3), num_filters=100, pad=True)(x)
        x = Convolution2D(filter_shape=(3, 3), num_filters=100, pad=True)(x)
        x = MaxPooling((2, 2), strides=(2, 2), pad=False)(x)
        x = Dropout(0.25)(x)    
        
        x = Dense(512)(x)
        x = Dropout(0.5)(x)
        x = Dense(N_CLASSES, activation=None)(x)
        return x


def init_model(m):
    # Loss (dense labels); check if support for sparse labels
    loss = cntk.cross_entropy_with_softmax(m, labels)
    # Momentum SGD
    # https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_use_learners.ipynb
    # unit_gain=False: momentum_direction = momentum*old_momentum_direction + gradient
    # if unit_gain=True then ...(1-momentum)*gradient
    learner = cntk.momentum_sgd(m.parameters, 
                                lr=cntk.learning_rate_schedule(LR, cntk.UnitType.minibatch) , 
                                momentum=cntk.momentum_schedule(MOMENTUM),
                                unit_gain=False)
    return loss, learner


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True, one_hot=True)\n# CNTK format\ny_train = y_train.astype(np.float32)\ny_test = y_test.astype(np.float32)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Placeholders\nfeatures = cntk.input_variable((3, 32, 32), np.float32)\nlabels = cntk.input_variable(N_CLASSES, np.float32)\n# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', 'loss, learner = init_model(sym)')


get_ipython().run_cell_magic('time', '', 'loss.train((x_train, y_train), \n           minibatch_size=BATCHSIZE, \n           max_epochs=EPOCHS,\n           parameter_learners=[learner])')


get_ipython().run_cell_magic('time', '', '# Predict and then score accuracy\n# Apply softmax since that is only applied at training\n# with cross-entropy loss\nz = cntk.softmax(sym)\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = np.argmax(y_test[:n_samples], axis=-1)\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    predicted_label_probs = z.eval({features : data})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = np.argmax(predicted_label_probs, axis=-1)\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level MXNet Example
# 

import os
import sys
import numpy as np
import mxnet as mx
from common.params import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("MXNet: ", mx.__version__)
print("GPU: ", get_gpu_name())


def create_symbol():
    data = mx.symbol.Variable('data')
    # size = [(old-size - kernel + 2*padding)/stride]+1
    # if kernel = 3, pad with 1 either side
    conv1 = mx.symbol.Convolution(data=data, num_filter=50, pad=(1,1), kernel=(3,3))
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(data=relu1, num_filter=50, pad=(1,1), kernel=(3,3))
    pool1 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2,2), stride=(2,2))
    relu2 = mx.symbol.Activation(data=pool1, act_type="relu")
    drop1 = mx.symbol.Dropout(data=relu2, p=0.25)
    
    conv3 = mx.symbol.Convolution(data=drop1, num_filter=100, pad=(1,1), kernel=(3,3))
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, num_filter=100, pad=(1,1), kernel=(3,3))
    pool2 = mx.symbol.Pooling(data=conv4, pool_type="max", kernel=(2,2), stride=(2,2))
    relu4 = mx.symbol.Activation(data=pool2, act_type="relu")
    drop2 = mx.symbol.Dropout(data=relu4, p=0.25)
           
    flat1 = mx.symbol.Flatten(data=drop2)
    fc1 = mx.symbol.FullyConnected(data=flat1, num_hidden=512)
    relu7 = mx.symbol.Activation(data=fc1, act_type="relu")
    drop4 = mx.symbol.Dropout(data=relu7, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop4, num_hidden=N_CLASSES) 
    
    input_y = mx.symbol.Variable('softmax_label')  
    m = mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")
    return m


def init_model(m):
    if GPU:
        ctx = [mx.gpu(0)]
    else:
        ctx = mx.cpu()
    mod = mx.mod.Module(context=ctx, symbol=m)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    # Glorot-uniform initializer
    mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
    mod.init_optimizer(optimizer='sgd', 
                       optimizer_params=(('learning_rate', LR), ('momentum', MOMENTUM), ))
    return mod


get_ipython().run_cell_magic('time', '', '# Data into format for library\n#x_train, x_test, y_train, y_test = mnist_for_library(channel_first=True)\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\n# Load data-iterator\ntrain_iter = mx.io.NDArrayIter(x_train, y_train, BATCHSIZE, shuffle=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


get_ipython().run_cell_magic('time', '', "# Train and log accuracy\nmetric = mx.metric.create('acc')\nfor j in range(EPOCHS):\n    train_iter.reset()\n    metric.reset()\n    for batch in train_iter:\n        model.forward(batch, is_train=True) \n        model.update_metric(metric, batch.label)\n        model.backward()              \n        model.update()\n    print('Epoch %d, Training %s' % (j, metric.get()))")


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(mx.io.NDArrayIter(x_test, batch_size=BATCHSIZE, shuffle=False))\ny_guess = np.argmax(y_guess.asnumpy(), axis=-1)')


print("Accuracy: ", sum(y_guess == y_test)/len(y_guess))


# # High-level RNN MXNet Example
# 

import os
import sys
import numpy as np
import mxnet as mx
from common.params_lstm import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("MXNet: ", mx.__version__)
print("GPU: ", get_gpu_name())


def create_symbol(CUDNN=True):
    # https://mxnet.incubator.apache.org/api/python/rnn.html
    data = mx.symbol.Variable('data')
    embedded_step = mx.symbol.Embedding(data=data, input_dim=MAXFEATURES, output_dim=EMBEDSIZE)
    
    # Fusing RNN layers across time step into one kernel
    # Improves speed but is less flexible
    # Currently only supported if using cuDNN on GPU
    if not CUDNN:
        gru_cell = mx.rnn.GRUCell(num_hidden=NUMHIDDEN)
    else:
        gru_cell = mx.rnn.FusedRNNCell(num_hidden=NUMHIDDEN, num_layers=1, mode='gru')
    
    begin_state = gru_cell.begin_state()
    # Call the cell to get the output of one time step for a batch.
    # TODO: TNC layout (sequence length, batch size, and feature dimensions) is faster for RNN
    outputs, states = gru_cell.unroll(length=MAXLEN, inputs=embedded_step, merge_outputs=False)
    
    fc1 = mx.symbol.FullyConnected(data=outputs[-1], num_hidden=2) 
    input_y = mx.symbol.Variable('softmax_label')  
    m = mx.symbol.SoftmaxOutput(data=fc1, label=input_y, name="softmax")
    return m


def init_model(m):
    if GPU:
        ctx = [mx.gpu(0)]
    else:
        ctx = mx.cpu()
    mod = mx.mod.Module(context=ctx, symbol=m)
    mod.bind(data_shapes=[('data', (BATCHSIZE, MAXLEN))],
             label_shapes=[('softmax_label', (BATCHSIZE, ))])
    # Glorot-uniform initializer
    mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
    mod.init_optimizer(optimizer='Adam', 
                       optimizer_params=(('learning_rate', LR),
                                         ('beta1', BETA_1),
                                         ('beta2', BETA_2),
                                         ('epsilon', EPS)))
    return mod


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)\n\n# Use custom iterator instead of mx.io.NDArrayIter() for consistency\n# Wrap as DataBatch class\nwrapper_db = lambda args: mx.io.DataBatch(data=[mx.nd.array(args[0])], label=[mx.nd.array(args[1])])\n\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


get_ipython().run_cell_magic('time', '', "# Train and log accuracy\nmetric = mx.metric.create('acc')\nfor j in range(EPOCHS):\n    #train_iter.reset()\n    metric.reset()\n    #for batch in train_iter:\n    for batch in map(wrapper_db, yield_mb(x_train, y_train, BATCHSIZE, shuffle=True)):\n        model.forward(batch, is_train=True) \n        model.update_metric(metric, batch.label)\n        model.backward()              \n        model.update()\n    print('Epoch %d, Training %s' % (j, metric.get()))")


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(mx.io.NDArrayIter(x_test, batch_size=BATCHSIZE, shuffle=False))\ny_guess = np.argmax(y_guess.asnumpy(), axis=-1)')


print("Accuracy: ", sum(y_guess == y_test)/len(y_guess))


# # High-level Caffe2 Example
# 

import os
import sys
import caffe2
import numpy as np
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, optimizer, utils
from caffe2.proto import caffe2_pb2
from common.params import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())


if GPU:
    device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)  # Run on GPU
else:
    device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)  # Run on CPU


def create_model(m, device_opts) :
    with core.DeviceScope(device_opts):
        conv1 = brew.conv(m, 'data', 'conv1', dim_in=3, dim_out=50, kernel=3, pad=1, no_gradient_to_input=1)
        relu1 = brew.relu(m, conv1, 'relu1')
        conv2 = brew.conv(m, relu1, 'conv2', dim_in=50, dim_out=50, kernel=3, pad=1)
        pool1 = brew.max_pool(m, conv2, 'pool1', kernel=2, stride=2)
        relu2 = brew.relu(m, pool1, 'relu2')
        drop1 = brew.dropout(m, relu2, 'drop1', ratio=0.25)

        conv3 = brew.conv(m, drop1, 'conv3', dim_in=50, dim_out=100, kernel=3, pad=1)
        relu3 = brew.relu(m, conv3, 'relu3')
        conv4 = brew.conv(m, relu3, 'conv4', dim_in=100, dim_out=100, kernel=3, pad=1)
        pool2 = brew.max_pool(m, conv4, 'pool2', kernel=2, stride=2)   
        relu4 = brew.relu(m, pool2, 'relu4')
        drop2 = brew.dropout(m, relu4, 'drop2', ratio=0.25)
        
        fc1 = brew.fc(m, drop2, 'fc1', dim_in=100 * 8 * 8, dim_out=512)
        relu5 = brew.relu(m, fc1, 'relu5')
        drop3 = brew.dropout(m, relu5, 'drop3', ratio=0.5)
        
        fc2 = brew.fc(m, drop3, 'fc2', dim_in=512, dim_out=N_CLASSES)
        softmax = brew.softmax(m, fc2, 'softmax')
        return softmax


def add_training_operators(softmax, m, device_opts) :
    with core.DeviceScope(device_opts):
        xent = m.LabelCrossEntropy([softmax, "label"], 'xent')
        loss = m.AveragedLoss(xent, "loss")
        #brew.accuracy(m, [softmax, "label"], "accuracy")
        m.AddGradientOperators([loss])
        opt = optimizer.build_sgd(
            m,
            base_learning_rate=LR, 
            policy='fixed',
            momentum=MOMENTUM)


def init_model():
    # Create Place-holder for data
    workspace.FeedBlob("data", x_train[:BATCHSIZE], device_option=device_opts)
    workspace.FeedBlob("label", y_train[:BATCHSIZE], device_option=device_opts)
    
    # Initialise model
    train_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': True,
        'ws_nbytes_limit': (64 * 1024 * 1024),
    }
    train_model = model_helper.ModelHelper(
        name="train_net", arg_scope=train_arg_scope
    )
    softmax = create_model(train_model, device_opts=device_opts)
    add_training_operators(softmax, train_model, device_opts=device_opts)

    # Initialise workspace
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)
    return train_model


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model()')


get_ipython().run_cell_magic('time', '', '# Train model\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Run one mini-batch at time\n        workspace.FeedBlob("data", data, device_option=device_opts)\n        workspace.FeedBlob("label", label, device_option=device_opts)\n        workspace.RunNet(model.net)       \n    print("Finished epoch: ", j)\n    print(str(j) + \': \' + str(workspace.FetchBlob("loss")))')


get_ipython().run_cell_magic('time', '', '# Init test model\ntest_arg_scope = {\n    \'order\': \'NCHW\',\n    \'use_cudnn\': True,\n    \'cudnn_exhaustive_search\': True,\n    \'ws_nbytes_limit\': (64 * 1024 * 1024),\n    \'is_test\': True,\n}\ntest_model= model_helper.ModelHelper(name="test_net", init_params=False, arg_scope=test_arg_scope)\ncreate_model(test_model, device_opts=device_opts)\nworkspace.RunNetOnce(test_model.param_init_net)\nworkspace.CreateNet(test_model.net, overwrite=True)\n\n# Run test\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    workspace.FeedBlob("data", data, device_option=device_opts)\n    workspace.RunNet(test_model.net)\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = (np.argmax(workspace.FetchBlob("softmax"), axis=-1))\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/float(len(y_guess)))


# # High-level RNN PyTorch Example
# 

import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.init as init
from torch import autograd
from torch.autograd import Variable
from common.params_lstm import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())


class SymbolModule(nn.Module):
    def __init__(self):
        super(SymbolModule, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=MAXFEATURES,
                                      embedding_dim=EMBEDSIZE)
        # If batch-first then input and output 
        # provided as (batch, seq, features)
        # Cudnn used by default if possible
        self.gru = nn.GRU(input_size=EMBEDSIZE, 
                          hidden_size=NUMHIDDEN, 
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)   
        self.l_out = nn.Linear(in_features=NUMHIDDEN*1,
                               out_features=2)

    def forward(self, x):
        x = self.embedding(x)
        h0 = Variable(torch.zeros(1, BATCHSIZE, NUMHIDDEN)).cuda()
        x, h = self.gru(x, h0)  # outputs, states
        # just get the last output state
        x = x[:,-1,:].squeeze()
        x = self.l_out(x)
        return x


def init_model(m):
    opt = optim.Adam(m.parameters(), lr=LR, betas=(BETA_1, BETA_2), eps=EPS)
    criterion = nn.CrossEntropyLoss()
    return opt, criterion


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)\n# Torch-specific\nx_train = x_train.astype(np.int64)\nx_test = x_test.astype(np.int64)\ny_train = y_train.astype(np.int64)\ny_test = y_test.astype(np.int64)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', 'sym = SymbolModule()\nsym.cuda() # CUDA!')


get_ipython().run_cell_magic('time', '', 'optimizer, criterion = init_model(sym)')


get_ipython().run_cell_magic('time', '', '# Sets training = True\nsym.train()  \nfor j in range(EPOCHS):\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = Variable(torch.LongTensor(data).cuda())\n        target = Variable(torch.LongTensor(target).cuda())\n        # Init\n        optimizer.zero_grad()\n        # Forwards\n        output = sym(data)\n        # Loss\n        loss = criterion(output, target)\n        # Back-prop\n        loss.backward()\n        optimizer.step()\n    # Log\n    print(j)')


get_ipython().run_cell_magic('time', '', '# Test model\n# Sets training = False\nsym.eval()\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, target in yield_mb(x_test, y_test, BATCHSIZE):\n    # Get samples\n    data = Variable(torch.LongTensor(data).cuda())\n    # Forwards\n    output = sym(data)\n    pred = output.data.max(1)[1].cpu().numpy().squeeze()\n    # Collect results\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level MXNet Example
# 
# **In the interest of comparison; a common (custom) data-generator (called yield_mb(X, y, batchsize=64, shuffle=False)) was originally used for all other frameworks - but not for MXNet. I have reproduced the MXNet example using this same generator (wrapping the results in the mx.io.DataBatch class) to test if MXNet is faster than other frameworks just because I was using its own data-generator. This does not appear to be the case. **
# 

import os
import sys
import numpy as np
import mxnet as mx
from common.params import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("MXNet: ", mx.__version__)
print("GPU: ", get_gpu_name())


def create_symbol():
    data = mx.symbol.Variable('data')
    # size = [(old-size - kernel + 2*padding)/stride]+1
    # if kernel = 3, pad with 1 either side
    conv1 = mx.symbol.Convolution(data=data, num_filter=50, pad=(1,1), kernel=(3,3))
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(data=relu1, num_filter=50, pad=(1,1), kernel=(3,3))
    pool1 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2,2), stride=(2,2))
    relu2 = mx.symbol.Activation(data=pool1, act_type="relu")
    drop1 = mx.symbol.Dropout(data=relu2, p=0.25)
    
    conv3 = mx.symbol.Convolution(data=drop1, num_filter=100, pad=(1,1), kernel=(3,3))
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, num_filter=100, pad=(1,1), kernel=(3,3))
    pool2 = mx.symbol.Pooling(data=conv4, pool_type="max", kernel=(2,2), stride=(2,2))
    relu4 = mx.symbol.Activation(data=pool2, act_type="relu")
    drop2 = mx.symbol.Dropout(data=relu4, p=0.25)
           
    flat1 = mx.symbol.Flatten(data=drop2)
    fc1 = mx.symbol.FullyConnected(data=flat1, num_hidden=512)
    relu7 = mx.symbol.Activation(data=fc1, act_type="relu")
    drop4 = mx.symbol.Dropout(data=relu7, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop4, num_hidden=N_CLASSES) 
    
    input_y = mx.symbol.Variable('softmax_label')  
    m = mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")
    return m


def init_model(m):
    if GPU:
        ctx = [mx.gpu(0)]
    else:
        ctx = mx.cpu()
    
    mod = mx.mod.Module(context=ctx, symbol=m)
    mod.bind(data_shapes=[('data', (BATCHSIZE, 3, 32, 32))],
             label_shapes=[('softmax_label', (BATCHSIZE,))])

    # Glorot-uniform initializer
    mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
    mod.init_optimizer(optimizer='sgd', 
                       optimizer_params=(('learning_rate', LR), ('momentum', MOMENTUM), ))
    return mod


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\n\n# Load data-iterator\n#train_iter = mx.io.NDArrayIter(x_train, y_train, BATCHSIZE, shuffle=True)\n# Use custom iterator instead of mx.io.NDArrayIter() for consistency\n# Wrap as DataBatch class\nwrapper_db = lambda args: mx.io.DataBatch(data=[mx.nd.array(args[0])], label=[mx.nd.array(args[1])])\n\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


get_ipython().run_cell_magic('time', '', "# Train and log accuracy\nmetric = mx.metric.create('acc')\nfor j in range(EPOCHS):\n    #train_iter.reset()\n    metric.reset()\n    #for batch in train_iter:\n    for batch in map(wrapper_db, yield_mb(x_train, y_train, BATCHSIZE, shuffle=True)):\n        model.forward(batch, is_train=True) \n        model.update_metric(metric, batch.label)\n        model.backward()              \n        model.update()\n    print('Epoch %d, Training %s' % (j, metric.get()))")


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(mx.io.NDArrayIter(x_test, batch_size=BATCHSIZE, shuffle=False))\ny_guess = np.argmax(y_guess.asnumpy(), axis=-1)')


print("Accuracy: ", sum(y_guess == y_test)/len(y_guess))


# # High-level Chainer Example
# 

import os
os.environ['CHAINER_TYPE_CHECK'] = '0'

import sys
import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from common.params import *
from common.utils import *


cuda.set_max_workspace_size(512 * 1024 * 1024)


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Chainer: ", chainer.__version__)
print("CuPy: ", chainer.cuda.cupy.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())


class SymbolModule(chainer.Chain):
    def __init__(self):
        super(SymbolModule, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 50, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(50, 50, ksize=3, pad=1)
            self.conv3 = L.Convolution2D(50, 100, ksize=3, pad=1)
            self.conv4 = L.Convolution2D(100, 100, ksize=3, pad=1)
            # feature map size is 8*8 by pooling
            self.fc1 = L.Linear(100*8*8, 512)
            self.fc2 = L.Linear(512, N_CLASSES)
    
    def __call__(self, x):
        h = F.relu(self.conv2(F.relu(self.conv1(x))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, 0.25)
        
        h = F.relu(self.conv4(F.relu(self.conv3(h))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, 0.25)       
        
        h = F.dropout(F.relu(self.fc1(h)), 0.5)
        return self.fc2(h)


def init_model(m):
    optimizer = optimizers.MomentumSGD(lr=LR, momentum=MOMENTUM)
    optimizer.setup(m)
    return optimizer


get_ipython().run_cell_magic('time', '', '# Data into format for library\n#x_train, x_test, y_train, y_test = mnist_for_library(channel_first=True)\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Create symbol\nsym = SymbolModule()\nif GPU:\n    chainer.cuda.get_device(0).use()  # Make a specified GPU current\n    sym.to_gpu()  # Copy the model to the GPU')


get_ipython().run_cell_magic('time', '', 'optimizer = init_model(sym)')


get_ipython().run_cell_magic('time', '', 'for j in range(EPOCHS):\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = cuda.to_gpu(data)\n        target = cuda.to_gpu(target)\n        output = sym(data)\n        loss = F.softmax_cross_entropy(output, target)\n        sym.cleargrads()\n        loss.backward()\n        optimizer.update()\n    # Log\n    print(j)')


get_ipython().run_cell_magic('time', '', "n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\n\nwith chainer.using_config('train', False), chainer.using_config('enable_backprop', False):\n    for data, target in yield_mb(x_test, y_test, BATCHSIZE):\n        # Forwards\n        pred = cuda.to_cpu(sym(cuda.to_gpu(data)).data.argmax(-1))\n        # Collect results\n        y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred\n        c += 1")


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level RNN TF Example
# 

import numpy as np
import os
import sys
import tensorflow as tf
from common.params_lstm import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("GPU: ", get_gpu_name())


def create_symbol(CUDNN=True):
    word_vectors = tf.contrib.layers.embed_sequence(X, vocab_size=MAXFEATURES, embed_dim=EMBEDSIZE)
    word_list = tf.unstack(word_vectors, axis=1)
    
    if not CUDNN:
        cell = tf.contrib.rnn.GRUCell(NUMHIDDEN)
        outputs, states = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
    else:
        # Using cuDNN since vanilla RNN
        cudnn_cell = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, 
                                                   num_units=NUMHIDDEN, 
                                                   input_size=EMBEDSIZE)
        params_size_t = cudnn_cell.params_size()
        params = tf.Variable(tf.random_uniform([params_size_t], -0.1, 0.1), validate_shape=False)   
        input_h = tf.Variable(tf.zeros([1, BATCHSIZE, NUMHIDDEN]))
        outputs, states = cudnn_cell(input_data=word_list,
                                     input_h=input_h,
                                     params=params)
        logits = tf.layers.dense(outputs[-1], 2, activation=None, name='output')
    return logits


def init_model(m):
    # Single-class labels, don't need dense one-hot
    # Expects unscaled logits, not output of tf.nn.softmax
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=m, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(LR, BETA_1, BETA_2, EPS)
    training_op = optimizer.minimize(loss)
    return training_op


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Place-holders\nX = tf.placeholder(tf.int32, shape=[None, MAXLEN])\ny = tf.placeholder(tf.int32, shape=[None])\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', 'model = init_model(sym)\ninit = tf.global_variables_initializer()\nsess = tf.Session()\nsess.run(init)')


get_ipython().run_cell_magic('time', '', '# Accuracy logging\ncorrect = tf.nn.in_top_k(sym, y, 1)\naccuracy = tf.reduce_mean(tf.cast(correct, tf.float32))\n\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        sess.run(model, feed_dict={X: data, y: label})\n    # Log\n    acc_train = sess.run(accuracy, feed_dict={X: data, y: label})\n    print(j, "Train accuracy:", acc_train)')


get_ipython().run_cell_magic('time', '', 'n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    pred = tf.argmax(sym, 1)\n    output = sess.run(pred, feed_dict={X: data})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output\n    c += 1')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level RNN Keras (CNTK) Example
# 

import os
import sys
import numpy as np
os.environ['KERAS_BACKEND'] = "cntk"
import keras as K
import cntk
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, CuDNNGRU
from common.params_lstm import *
from common.utils import *


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("CNTK: ", cntk.__version__)
print(K.backend.backend())


def create_symbol(CUDNN=True):
    model = Sequential()
    model.add(Embedding(MAXFEATURES, EMBEDSIZE, input_length=MAXLEN))
    # Only return last output
    if not CUDNN:
        model.add(GRU(NUMHIDDEN, return_sequences=False, return_state=False))
    else:
        model.add(CuDNNGRU(NUMHIDDEN, return_sequences=False, return_state=False))
    model.add(Dense(2, activation='softmax'))
    return model


def init_model(m):
    m.compile(
        loss = "categorical_crossentropy",
        optimizer = K.optimizers.Adam(LR, BETA_1, BETA_2, EPS),
        metrics = ['accuracy'])
    return m


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, one_hot=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\n# CuDNN RNNs are only available with the TensorFlow backend.\nsym = create_symbol(CUDNN=False)')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


model.summary()


get_ipython().run_cell_magic('time', '', '# Train model\nmodel.fit(x_train,\n          y_train,\n          batch_size=BATCHSIZE,\n          epochs=EPOCHS,\n          verbose=1)')


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(x_test, batch_size=BATCHSIZE)\ny_guess = np.argmax(y_guess, axis=-1)\ny_truth = np.argmax(y_test, axis=-1)')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


# # High-level Keras (CNTK) Example
# 

import os
import sys
import numpy as np
os.environ['KERAS_BACKEND'] = "cntk"
import keras as K
import cntk
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from common.params import *
from common.utils import *


# channels_first is faster
K.backend.set_image_data_format('channels_first')


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("CNTK: ", cntk.__version__)
print(K.backend.backend())
# Should be channels-first, otherwise slow
print(K.backend.image_data_format())
print("GPU: ", get_gpu_name())


def create_symbol():
    model = Sequential()
    
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(3, 32, 32)))
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))
    return model


def init_model(m):
    m.compile(
        loss = "categorical_crossentropy",
        optimizer = K.optimizers.SGD(LR, MOMENTUM),
        metrics = ['accuracy'])
    return m


get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True, one_hot=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')


get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')


get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')


model.summary()


get_ipython().run_cell_magic('time', '', '# Train model\nmodel.fit(x_train,\n          y_train,\n          batch_size=BATCHSIZE,\n          epochs=EPOCHS,\n          verbose=1)')


get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(x_test, batch_size=BATCHSIZE)\ny_guess = np.argmax(y_guess, axis=-1)\ny_truth = np.argmax(y_test, axis=-1)')


print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))


