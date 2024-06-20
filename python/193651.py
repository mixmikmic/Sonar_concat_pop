# ### Reference
# 
# - http://machinethink.net/blog/coreml-custom-layers/
# 

import sys
sys.path.append('../code/')

from inception_resnet_v1 import *
model = InceptionResNetV1(weights_path='../model/keras/weights/facenet_keras_weights.h5')


from coremltools.proto import NeuralNetwork_pb2

# The conversion function for Lambda layers.
def convert_lambda(layer):
    if layer.function == scaling:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "scaling"
        return params
    else:
        return None


import coremltools

coreml_model = coremltools.converters.keras.convert(
    model,
    input_names="image",
    image_input_names="image",
    output_names="output",
    add_custom_layers=True,
    custom_conversion_functions={ "Lambda": convert_lambda })





# # TensorFlow to Keras
# 

# ### Reference
# - https://github.com/myutwo150/keras-inception-resnet-v2
# 

# ### Pre-trained tensorflow models
# 
# - https://github.com/davidsandberg/facenet
# 

import os
import re
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../code/')
from inception_resnet_v1 import *


tf_model_dir = '../model/tf/20170512-110547/'
npy_weights_dir = '../model/keras/npy_weights/'
weights_dir = '../model/keras/weights/'
model_dir = '../model/keras/model/'

weights_filename = 'facenet_keras_weights.h5'
model_filename = 'facenet_keras.h5'


os.makedirs(npy_weights_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# regex for renaming the tensors to their corresponding Keras counterpart
re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')

def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder):
    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)


extract_tensors_from_checkpoint_file(tf_model_dir+'model-20170512-110547.ckpt-250000', npy_weights_dir)


model = InceptionResNetV1()
# model.summary()


print('Loading numpy weights from', npy_weights_dir)
for layer in model.layers:
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))
            weights.append(weight_arr)
        layer.set_weights(weights)

print('Saving weights...')
model.save_weights(os.path.join(weights_dir, weights_filename))
print('Saving model...')
model.save(os.path.join(model_dir, model_filename))





# ### Reference
# - https://medium.com/@neotheicebird/webcam-based-image-processing-in-ipython-notebooks-47c75a022514
# 

get_ipython().magic('matplotlib inline')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import signal
from IPython import display

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from keras.models import load_model


cascade_path = '../model/cv2/haarcascade_frontalface_alt2.xml'


model_path = '../model/keras/model/facenet_keras.h5'
model = load_model(model_path)


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def calc_embs(imgs, margin, batch_size):
    aligned_images = prewhiten(imgs)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs


class FaceDemo(object):
    def __init__(self, cascade_path):
        self.vc = None
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.margin = 10
        self.batch_size = 1
        self.n_img_per_person = 10
        self.is_interrupted = False
        self.data = {}
        self.le = None
        self.clf = None
        
    def _signal_handler(self, signal, frame):
        self.is_interrupted = True
        
    def capture_images(self, name='Unknown'):
        vc = cv2.VideoCapture(0)
        self.vc = vc
        if vc.isOpened():
            is_capturing, _ = vc.read()
        else:
            is_capturing = False

        imgs = []
        signal.signal(signal.SIGINT, self._signal_handler)
        self.is_interrupted = False
        while is_capturing:
            is_capturing, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.cascade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(100, 100))
            if len(faces) != 0:
                face = faces[0]
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                img = resize(frame[bottom:top, left:right, :],
                             (160, 160), mode='reflect')
                imgs.append(img)
                cv2.rectangle(frame,
                              (left-1, bottom-1),
                              (right+1, top+1),
                              (255, 0, 0), thickness=2)

            plt.imshow(frame)
            plt.title('{}/{}'.format(len(imgs), self.n_img_per_person))
            plt.xticks([])
            plt.yticks([])
            display.clear_output(wait=True)
            if len(imgs) == self.n_img_per_person:
                vc.release()
                self.data[name] = np.array(imgs)
                break
            try:
                plt.pause(0.1)
            except Exception:
                pass
            if self.is_interrupted:
                vc.release()
                break
                
    def train(self):
        labels = []
        embs = []
        names = self.data.keys()
        for name, imgs in self.data.items():
            embs_ = calc_embs(imgs, self.margin, self.batch_size)    
            labels.extend([name] * len(embs_))
            embs.append(embs_)

        embs = np.concatenate(embs)
        le = LabelEncoder().fit(labels)
        y = le.transform(labels)
        clf = SVC(kernel='linear', probability=True).fit(embs, y)
        
        self.le = le
        self.clf = clf
        
    def infer(self):
        vc = cv2.VideoCapture(0)
        self.vc = vc
        if vc.isOpened():
            is_capturing, _ = vc.read()
        else:
            is_capturing = False

        signal.signal(signal.SIGINT, self._signal_handler)
        self.is_interrupted = False
        while is_capturing:
            is_capturing, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.cascade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(100, 100))
            pred = None
            if len(faces) != 0:
                face = faces[0]
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                img = resize(frame[bottom:top, left:right, :],
                             (160, 160), mode='reflect')
                embs = calc_embs(img[np.newaxis], self.margin, 1)
                pred = self.le.inverse_transform(self.clf.predict(embs))
                cv2.rectangle(frame,
                              (left-1, bottom-1),
                              (right+1, top+1),
                              (255, 0, 0), thickness=2)
            plt.imshow(frame)
            plt.title(pred)
            plt.xticks([])
            plt.yticks([])
            display.clear_output(wait=True)
            try:
                plt.pause(0.1)
            except Exception:
                pass
            if self.is_interrupted:
                vc.release()
                break


f = FaceDemo(cascade_path)


# Train with two or more people
f.capture_images('nyoki-mtl')


f.train()


f.infer()





