# Code from the keras samples, with TensorFlow imports added. https://keras.io
# 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras.python.keras.preprocessing import image
from tensorflow.contrib.keras.python.keras.applications.resnet50 import *


print (tf.__version__) # Must be v1.1+


model = ResNet50(weights='imagenet')

img = image.load_img('deer.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=5)[0])


# Code from the keras samples, with TensorFlow imports added. https://github.com/fchollet/keras/tree/master/examples
# 
# Trains a simple deep NN on the MNIST dataset.
# Gets to 98.40% test accuracy after 20 epochs
# (there is *a lot* of margin for parameter tuning).
# 2 seconds per epoch on a K520 GPU.
# 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


print (tf.__version__) # Must be v1.1+


from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop


batch_size = 128
num_classes = 10
epochs = 1

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





# This notebook demonstrates creating a custom sprite image for the TensorFlow [Embedding Visualizer](https://www.tensorflow.org/get_started/embedding_viz) using the MNIST dataset.
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from PIL import Image, ImageOps


mnist = input_data.read_data_sets('/tmp/data', one_hot=True)


rows = 32
cols = rows
im_size = 28

sprite = np.zeros((rows*im_size, cols*im_size))

idx = -1
for i in range(rows):
    for j in range(cols):
        idx +=1
        image = mnist.test.images[idx].reshape((28,28))
        row_coord = i * 28
        col_coord = j * 28
        sprite[row_coord:row_coord + 28, col_coord:col_coord + 28] = image
        
im = Image.fromarray(sprite * 255)
im = im.convert('RGB')
im = ImageOps.invert(im)

def get_color(lbl):
    if lbl == 0: return (255, 102, 102)
    if lbl == 1: return (255, 178, 102)
    if lbl == 2: return (255, 255, 102)
    if lbl == 3: return (178, 255, 102)
    if lbl == 4: return (102, 255, 102)
    if lbl == 5: return (102, 255, 178)
    if lbl == 6: return (102, 255, 255)
    if lbl == 7: return (102, 178, 255)
    if lbl == 8: return (102, 102, 255)
    if lbl == 9: return (178, 102, 255)

labels_file = open("labels.tsv", "w")
    
# colorize
orig_color = (255,255,255)
data = np.array(im)

idx = -1
for i in range(rows):
    for j in range(cols):
        idx +=1
        row_coord = i * 28
        col_coord = j * 28
        label = np.argmax(mnist.test.labels[idx])
        labels_file.write(str(label) + "\n")
        replacement_color = get_color(label)
        r = data[row_coord:row_coord + 28, col_coord:col_coord + 28]
        r[(r == orig_color).all(axis = -1)] = replacement_color

im = Image.fromarray(data, mode='RGB')
im.save("sprite.png")
im.show()

labels_file.close()


# Pre-Download Data
# ===
# 
# To save you the hassle of repeated downloads, it's easier so save the files in a shared folder.
# 
# To run all cells, select **Cell** > **Run All**. You can also run cells one at a time, select **Cell** > **Run cells** in the menu above.
# 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Some of these are hard to distinguish.
# Check https://quickdraw.withgoogle.com/data for examples
zoo = ['frog', 'horse', 'lion', 'monkey', 'octopus', 'owl', 'rhinoceros', 
       'snail', 'tiger', 'zebra']

# Mapping between category names and ids
animal2id = dict((c,i) for i,c in enumerate(zoo))
id2animal = dict((i,c) for i,c in enumerate(zoo))
for i, animal in id2animal.items():
    print("Class {}: {}".format(i, animal))


from six.moves.urllib.request import urlretrieve
import os

DATA_DIR = 'data/'

def maybe_download(url, data_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(data_dir, filename)

    # Check if the file already exists.
    if not os.path.exists(file_path):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print("Downloading {} to {}".format(url, file_path))
        file_path, _ = urlretrieve(url=url, filename=file_path)
    else:
        print("Using previously downloaded file: {}".format(file_path))
    return file_path

def load_data(file_path, max_examples=2000, example_name=''):
    d = np.load(open(file_path, 'r'))
    d = d[:max_examples,:] # limit number of instances to save memory
    print("Loaded {} {} examples of dimension {} from {}".format(
            d.shape[0], example_name, d.shape[1], file_path))
    return d

data= []
labels =[]

for animal in zoo:
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(animal)
    file_path = maybe_download(url, DATA_DIR)
    data.append(load_data(file_path, max_examples = 1000, example_name = animal))
    labels.extend([animal2id[animal]]*data[-1].shape[0])

data = np.concatenate(data, axis=0)
labels = np.array(labels)
print("Final shape of data: {}".format(data.shape))


# The data is fun to look at. Compared to MNIST the classes seem much harder to distinguish
# 

import matplotlib.pyplot as plt

n_samples = 10
random_indices = np.random.permutation(data.shape[0])

for i in random_indices[:n_samples]:
    print(i, labels[i])
    print("Category {}: {}".format(labels[i], id2animal[labels[i]]))

    # We'll show the image and its pixel value histogram side-by-side.

    # To interpret the values as a 28x28 image, we need to reshape
    # the numpy array, which is one dimensional.
    image = data[i, :]

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image.reshape(28, 28), cmap=plt.cm.Greys, interpolation='nearest')
    ax2.hist(image, bins=20)
    ax1.grid(False)
    plt.show()


if data.dtype == 'uint8':  # avoid doing this twice
    data = data.astype(np.float32)
    data = (data - (255 / 2.0)) / 255


# Our labels are 0,1,2,..,10 right now. We convert to a one-hot representation
# 

random_indices = np.random.permutation(labels.shape[0])

print("Labels before:")
print(labels[random_indices[:5]])

def one_hot(labels, n_classes):
    n_labels = len(labels)
    one_hot_labels = np.zeros((n_labels, n_classes))
    one_hot_labels[np.arange(n_labels), labels] = 1
    return one_hot_labels

labels_one_hot = one_hot(labels, len(zoo))

print("Labels after:")
print(labels_one_hot[random_indices[:5]])


# Finally, let's split the data into random train and test partitions
# 

n_test_examples = 1000

random_indices = np.random.permutation(data.shape[0])
test_data = data[random_indices[:n_test_examples],:]
test_labels = labels_one_hot[random_indices[:n_test_examples],:]
train_data = data[random_indices[n_test_examples:],:]
train_labels = labels_one_hot[random_indices[n_test_examples:],:]
print("Data shapes: ", test_data.shape, test_labels.shape, train_data.shape, train_labels.shape)


# Save data for other experiments
# 

outfile_name = os.path.join(DATA_DIR, "zoo.npz")
with open(outfile_name, 'w') as outfile:
    np.savez(outfile, train_data, train_labels, test_data, test_labels)
print ("Saved train/test data to {}".format(outfile_name))


