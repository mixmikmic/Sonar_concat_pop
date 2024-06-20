import coremltools
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Load pre-trained keras model
# 

from keras.models import load_model
# model = load_model('./model/nn4.small2.lrn.h5')

import tensorflow as tf
from keras.utils import CustomObjectScope
with CustomObjectScope({'tf': tf}):
    model = load_model('./model/nn4.small2.lrn.h5' )


import coremltools
coreml_model = coremltools.converters.keras.convert(
  model, input_names='data', image_input_names='data', image_scale=1/255.0, output_names='output')


print coreml_model


# Read a sample image as input to test the model
import cv2
import numpy as np
img = cv2.imread('./data/dlib-affine-sz/Aaron_Eckhart/Aaron_Eckhart_0001.png', 1)
img = img[...,::-1]
img = np.around(np.transpose(img, (2, 1, 0))/255.0, decimals=12)

# x_train = np.array(img)
# y = coreml_model.predict({'image': x_train})

img = np.transpose(img, (1, 2, 0))
x_train = np.array([img])
y = model.predict_on_batch(x_train)

print y


# LFW TEST
import lfw
import os
import numpy as np
import math
import facenet
import time
import tensorflow as tf

lfw_pairs='data/pairs.txt'
lfw_dir='data/dlib-affine-sz'
lfw_file_ext='png'
lfw_nrof_folds=10
image_size=96
batch_size=100

# Read the file containing the pairs used for testing
pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

# Get the paths for the corresponding images
paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)

embedding_size=128
nrof_images = len(paths)
nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
emb_array = np.zeros((nrof_images, embedding_size))

# print paths

for i in range(nrof_batches):
  start_index = i*batch_size
  end_index = min((i+1)*batch_size, nrof_images)
  paths_batch = paths[start_index:end_index]
  images = facenet.load_data(paths_batch, False, False, image_size)
  images = np.transpose(images, (0,3,1,2))
  
  t0 = time.time()
  y = []
  for img in images:
    tmp = coreml_model.predict({'input1': img})
#     print tmp.output1
    y.append(tmp['output1'])
#   y = model.predict_on_batch(images)
  emb_array[start_index:end_index,:] = y
#   print('y', y)
#   print('emb', emb_array[start_index:end_index,:])
  t1 = time.time()
  
  print('batch: ', i, ' time: ', t1-t0)

from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, 
                actual_issame, nrof_folds=lfw_nrof_folds)

print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
auc = metrics.auc(fpr, tpr)
print('Area Under Curve (AUC): %1.3f' % auc)

eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
print('Equal Error Rate (EER): %1.3f' % eer)


coreml_model.save('./model/OpenFace.mlmodel')





