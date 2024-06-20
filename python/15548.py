import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from sklearn.cross_validation import train_test_split

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils


get_ipython().magic('matplotlib inline')


PUG_IMG_DIR = "../data/pugs_cropped"
GOLDEN_RETRVR_IMG_DIR = "../data/golden_retrievers_cropped"
IMG_ROWS, IMG_COLS = 224, 224
IMG_CHANNELS = 3


# ## Read Image Files and Generate a Pickled Dataset...
# 

n_pug_images = len(os.listdir(PUG_IMG_DIR))
pug_images = np.empty((n_pug_images, IMG_CHANNELS, IMG_ROWS, IMG_COLS), dtype="uint8")

n_golden_retrvr_images = len(os.listdir(GOLDEN_RETRVR_IMG_DIR))
golden_retrvr_images = np.empty((n_golden_retrvr_images, IMG_CHANNELS, IMG_ROWS, IMG_COLS), dtype="uint8")


for n, image in enumerate(os.listdir(PUG_IMG_DIR)):
    pug_images[n] = io.imread(PUG_IMG_DIR+"/"+image).transpose()

for n, image in enumerate(os.listdir(GOLDEN_RETRVR_IMG_DIR)):
    golden_retrvr_images[n] = io.imread(GOLDEN_RETRVR_IMG_DIR+"/"+image).transpose()


print(pug_images.shape)
print(golden_retrvr_images.shape)


pug_labels = np.ones(n_pug_images)
golden_retrvr_labels = np.zeros(n_golden_retrvr_images)


plt.axis('off')
plt.imshow(pug_images[921].transpose())


plt.axis('off')
plt.imshow(golden_retrvr_images[921].transpose())


X = np.concatenate([pug_images, golden_retrvr_images])
y = np.concatenate([pug_labels, golden_retrvr_labels])


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y)


with open("../data/pugs_vs_golden_retrvrs_data.pkl.gz", "wb") as pickle_file:
    pickle.dump((X_train, X_test, y_train, y_test), pickle_file)


# ## ...Or Load the Pickled Dataset Directly
# 

with open("../data/pugs_vs_golden_retrvrs_data.pkl.gz", "rb") as pickle_file:
    X_train, X_test, y_train, y_test = pickle.load(pickle_file)


# ## Define and Train the Neural Network
# 

# we're going to use a pre-trained deep network and chop off the
# last dense layer; we'll freeze the weights in the early layers
# and then train just the final set of dense weights
# see https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS), trainable=False))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


batch_size = 32
n_classes = 2
n_epochs = 10

# load our pre-trained model
model = VGG_16('./vgg16_weights.h5')

# chop off the final layer
model.layers = model.layers[:-1]

# and add in a new one appropriate to our task
model.add(Dense(n_classes, activation='softmax'))


Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)


sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=n_epochs, show_accuracy=True,
          validation_data=(X_test, Y_test), shuffle=True)


# ## Save the Neural Network
# 

json_string = model.to_json()
open('./cnn_pug_model_architecture.json', 'w').write(json_string)
model.save_weights('cnn_pug_model_weights.h5')





