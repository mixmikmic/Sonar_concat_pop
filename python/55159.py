get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import math
import copy

#import skimage.io as io
from scipy.misc import bytescale


from keras.models import Sequential, Model
from keras.layers import Input, Permute
from keras.layers import Convolution2D, Deconvolution2D, Cropping2D
from keras.layers import merge


from utils import fcn32_blank, fcn_32s_to_8s, prediction


# ## Build model architecture
# 

# ### Paper 1 : Conditional Random Fields as Recurrent Neural Networks
# ##### Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su
# ##### Dalong Du, Chang Huang, Philip H. S. Torr
# 
# http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf
# 
# ### Paper 2 : Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
# ##### Philipp Krahenbuhl, Vladlen Koltun
# 
# https://arxiv.org/pdf/1210.5644.pdf
# 
# This paper specifies the CRF kernels and the Mean Field Approximation of the CRF energy function
# 
# 
# ### WARNING :
# #### In v1 of this script we will only implement the FCN-8s subcomponent of the CRF-RNN network
# 
# #### Quotes from MatConvNet page (http://www.vlfeat.org/matconvnet/pretrained/#semantic-segmentation) :
# *These networks are trained on the PASCAL VOC 2011 training and (in part) validation data, using Berekely's extended annotations, as well as Microsoft COCO.*
# 
# *While the CRF component is missing (it may come later to MatConvNet), this model still outperforms the FCN-8s network above, partially because it is trained with additional data from COCO.*
# 
# *The model was obtained by first fine-tuning the plain FCN-32s network (without the CRF-RNN part) on COCO data, then building built an FCN-8s network with the learnt weights, and finally training the CRF-RNN network end-to-end using VOC 2012 training data only. The model available here is the FCN-8s part of this network (without CRF-RNN, while trained with 10 iterations CRF-RNN).*
# 

image_size = 64*8


fcn32model = fcn32_blank(image_size)


#print(dir(fcn32model.layers[-1]))
print(fcn32model.layers[-1].output_shape)


#fcn32model.summary() # visual inspection of model architecture


# WARNING : check dim weights against .mat file to check deconvolution setting
print fcn32model.layers[-2].get_weights()[0].shape


fcn8model = fcn_32s_to_8s(fcn32model)


# INFO : dummy image array to test the model passes
imarr = np.ones((3,image_size,image_size))
imarr = np.expand_dims(imarr, axis=0)

#testmdl = Model(fcn32model.input, fcn32model.layers[10].output) # works fine
testmdl = fcn8model # works fine
testmdl.predict(imarr).shape


if (testmdl.predict(imarr).shape != (1,21,image_size,image_size)):
    print('WARNING: size mismatch will impact some test cases')


fcn8model.summary() # visual inspection of model architecture


# ## Load VGG weigths from .mat file
# 
# #### https://www.vlfeat.org/matconvnet/pretrained/#semantic-segmentation
# ##### Download from console with :
# wget http://www.vlfeat.org/matconvnet/models/pascal-fcn8s-tvg-dag.mat
# 

from scipy.io import loadmat


USETVG = True
if USETVG:
    data = loadmat('pascal-fcn8s-tvg-dag.mat', matlab_compatible=False, struct_as_record=False)
    l = data['layers']
    p = data['params']
    description = data['meta'][0,0].classes[0,0].description
else:
    data = loadmat('pascal-fcn8s-dag.mat', matlab_compatible=False, struct_as_record=False)
    l = data['layers']
    p = data['params']
    description = data['meta'][0,0].classes[0,0].description
    print(data.keys())


l.shape, p.shape, description.shape


class2index = {}
for i, clname in enumerate(description[0,:]):
    class2index[str(clname[0])] = i
    
print(sorted(class2index.keys()))


if False: # inspection of data structure
    print(dir(l[0,31].block[0,0]))
    print(dir(l[0,44].block[0,0]))

if False:
    print l[0,36].block[0,0].upsample, l[0,36].block[0,0].size
    print l[0,40].block[0,0].upsample, l[0,40].block[0,0].size
    print l[0,44].block[0,0].upsample, l[0,44].block[0,0].size, l[0,44].block[0,0].crop


for i in range(0, p.shape[1]-1-2*2, 2): # weights #36 to #37 are not all paired
    print(i,
          str(p[0,i].name[0]), p[0,i].value.shape,
          str(p[0,i+1].name[0]), p[0,i+1].value.shape)
print '------------------------------------------------------'
for i in range(p.shape[1]-1-2*2+1, p.shape[1]): # weights #36 to #37 are not all paired
    print(i,
          str(p[0,i].name[0]), p[0,i].value.shape)


for i in range(l.shape[1]):
    print(i,
          str(l[0,i].name[0]), str(l[0,i].type[0]),
          [str(n[0]) for n in l[0,i].inputs[0,:]],
          [str(n[0]) for n in l[0,i].outputs[0,:]])


def copy_mat_to_keras(kmodel, verbose=True):
    
    kerasnames = [lr.name for lr in kmodel.layers]

    prmt = (3,2,0,1) # WARNING : important setting as 2 of the 4 axis have same size dimension
    
    for i in range(0, p.shape[1]):
        
        if USETVG:
            matname = p[0,i].name[0][0:-1]
            matname_type = p[0,i].name[0][-1] # "f" for filter weights or "b" for bias
        else:
            matname = p[0,i].name[0].replace('_filter','').replace('_bias','')
            matname_type = p[0,i].name[0].split('_')[-1] # "filter" or "bias"
        
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            if verbose:
                print 'found : ', (str(matname), str(matname_type), kindex)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            if  matname_type in ['f','filter']:
                l_weights = p[0,i].value
                f_l_weights = l_weights.transpose(prmt)
                f_l_weights = np.flip(f_l_weights, 2)
                f_l_weights = np.flip(f_l_weights, 3)
                assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
                current_b = kmodel.layers[kindex].get_weights()[1]
                kmodel.layers[kindex].set_weights([f_l_weights, current_b])
            elif matname_type in ['b','bias']:
                l_bias = p[0,i].value
                assert (l_bias.shape[1] == 1)
                assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
                current_f = kmodel.layers[kindex].get_weights()[0]
                kmodel.layers[kindex].set_weights([current_f, l_bias[:,0]])
        else:
            print 'not found : ', str(matname)


#copy_mat_to_keras(fcn32model)
copy_mat_to_keras(fcn8model, False)


# ## Tests
# 

im = Image.open('rgb.jpg') # http://www.robots.ox.ac.uk/~szheng/crfasrnndemo/static/rgb.jpg
im = im.crop((0,0,319,319)) # WARNING : manual square cropping
im = im.resize((image_size,image_size))


plt.imshow(np.asarray(im))


# WARNING : we do not deal with cropping here, this image is already fit
preds = prediction(fcn8model, im, transform=True)


#imperson = preds[0,class2index['person'],:,:]
imclass = np.argmax(preds, axis=1)[0,:,:]

plt.figure(figsize = (15, 7))
plt.subplot(1,3,1)
plt.imshow( np.asarray(im) )
plt.subplot(1,3,2)
plt.imshow( imclass )
plt.subplot(1,3,3)
plt.imshow( np.asarray(im) )
masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#plt.imshow( imclass, alpha=0.5 )
plt.imshow( masked_imclass, alpha=0.5 )


# List of dominant classes found in the image
for c in np.unique(imclass):
    print c, str(description[0,c][0])


bspreds = bytescale(preds, low=0, high=255)

plt.figure(figsize = (15, 7))
plt.subplot(2,3,1)
plt.imshow(np.asarray(im))
plt.subplot(2,3,3+1)
plt.imshow(bspreds[0,class2index['background'],:,:], cmap='seismic')
plt.subplot(2,3,3+2)
plt.imshow(bspreds[0,class2index['person'],:,:], cmap='seismic')
plt.subplot(2,3,3+3)
plt.imshow(bspreds[0,class2index['bicycle'],:,:], cmap='seismic')





from theano.sandbox import cuda


get_ipython().magic('matplotlib inline')
import utils_modified; reload(utils_modified)
from utils_modified import *
from __future__ import division, print_function


import numpy as np
import random
import sys


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Activation, merge, Flatten, Dropout, Lambda
from keras.layers import LSTM, SimpleRNN
from keras.models import Model, Sequential
from keras.engine.topology import Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import *
from keras.utils.data_utils import get_file


import quandl # pip install quandl
import pandas as pd


# https://keras.io/getting-started/sequential-model-guide/


nbassets = 9


def builder():
    # data array : 20days x 15stocks
    # note that we can name any layer by passing it a "name" argument.
    #main_input = Input(shape=(20,9), dtype='float32', name='main_input')
    
    model = Sequential()
    
    model.add( Dense(output_dim=100, input_shape=(20,nbassets), activation='tanh') )
    
    #model.add( BatchNormalization() )

    # a LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    model.add( SimpleRNN(30,
                         return_sequences=False, stateful=False,
                         activation='relu', inner_init='identity') )
    
    #model.add( Dropout(0.5) )

    model.add( Dense(30, activation='tanh') )
    
    model.add( Dense(1, activation='relu') )
    
    model.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
    
    return model


model1 = builder() # will be trained on simulated data


model1.summary()


if False:
    X = np.random.random((50,20,nbassets))
    Y = np.random.random((50,1))
else:
    AllXs = []
    AllYs = []
    for i in range(1000):
        t = (np.random.rand()-0.5)*6*5 # an offset for the sinus model
        Xs = []
        for stp in range(20):
            vol = 2 + math.sin(t+stp*0.2) # a market volatility
            Xs.append( np.random.randn(1,1,nbassets)*vol ) # one slice of stock returns
        futurevol = 2 + math.sin(t+(20-1+5)*0.2)
        #print(vol, nextvol)
        AllXs.append( np.concatenate(Xs, axis=1) )
        AllYs.append( np.array([futurevol]).reshape((1,1)) )
    X = np.concatenate(AllXs, axis=0)
    Y = np.concatenate(AllYs, axis=0)


model1.fit(X, Y, batch_size=50, nb_epoch=80, validation_split=0.2)


P = model1.predict(X)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.scatter(Y, P)
#plt.plot(Y)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['without BN','with BN'], loc='upper right')
plt.show()


def qData(tick='XLU'):
    # GOOG/NYSE_XLU.4
    # WIKI/MSFT.4
    qtck = "GOOG/NYSE_"+tick+".4"
    return quandl.get(qtck,
                      start_date="2003-01-01",
                      end_date="2016-12-31",
                      collapse="daily")


'''TICKERS = ['MSFT','JPM','INTC','DOW','KO',
             'MCD','CAT','WMT','MMM','AXP',
             'BA','GE','XOM','PG','JNJ']'''
TICKERS = ['XLU','XLF','XLK','XLY','XLV','XLB','XLE','XLP','XLI']


try:
    D.keys()
except:
    print('create empty Quandl cache')
    D = {}

for tckr in TICKERS:
    if not(tckr in D.keys()):
        print(tckr)
        qdt = qData(tckr)
        qdt.rename(columns={'Close': tckr}, inplace = True)
        D[tckr] = qdt
        
for tck in D.keys():
    assert(D[tck].keys() == [tck])


for tck in D.keys():
    print(D[tck].shape)


J = D[TICKERS[0]].join(D[TICKERS[1]])
for tck in TICKERS[2:]:
    J = J.join(D[tck])


J.head(5)


J.isnull().sum()


J2 = J.fillna(method='ffill')
#J2[J['WMT'].isnull()]


LogDiffJ = J2.apply(np.log).diff(periods=1, axis=0)
LogDiffJ.drop(LogDiffJ.index[0:1], inplace=True)
LogDiffJ.shape


MktData = LogDiffJ.as_matrix(columns=None) # as numpy.array
MktData.shape


model2 = builder() # will be trained on market data


if True:
    AllXs = []
    AllYs = []
    for i in range(500):
        t = np.random.randint(50, MktData.shape[0]-100) # an offset for whole historics
        Xs = []
        for stp in range(20):
            Xs.append( MktData[t+stp,:].reshape(1,1,-1)*100 ) # one slice of stock returns
        futurevol = math.sqrt(np.sum(MktData[t+20:t+20+10,:]*MktData[t+20:t+20+10,:]))*100
        #print(futurevol)
        AllXs.append( np.concatenate(Xs, axis=1) )
        AllYs.append( np.array([futurevol]).reshape((1,1)) )
    X = np.concatenate(AllXs, axis=0)
    Y = np.concatenate(AllYs, axis=0)


X.shape, Y.shape


print(np.min(X), np.mean(X), np.max(X))
print(np.min(Y), np.mean(Y), np.max(Y))


Yc = np.clip(Y, 0, 3*np.mean(Y))
np.mean(Y), np.mean(Yc)


# WARNING : need to scale data to speed up training !!!
factorX = 0.20
factorY = 0.10


# ## WARNING : **VALIDATION** should be posterior only
# 

# seems like 150 epochs are required to converge : BUG ?
model2.optimizer.lr = 1e-4
model2.fit(factorX*X, factorY*Yc, batch_size=50, nb_epoch=25, validation_split=0.2)


P = model2.predict(factorX*X)/factorY

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.scatter(Yc, P)
#plt.plot(Y)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
plt.legend(['clipped real vol','model vol'], loc='lower right')
plt.show()





from __future__ import print_function # for python 2.7 users


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D


# ## This notebook as been tested with :
# * Python 3.5
# * Keras 2
# * TensorFlow
# 

from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering


# ## Build model architecture
# 

def convblock(cdim, nb, bits=3):
    L = []
    
    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        #L.append( Convolution2D(cdim, 3, 3, border_mode='same', activation='relu', name=convname) ) # Keras 1
        L.append( Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname) ) # Keras 2
    
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    
    return L


def vgg_face_blank():
    
    withDO = True # no effect during evaluation but usefull for fine-tuning
    
    if True:
        mdl = Sequential()
        
        # First layer is a dummy-permutation = Identity to specify input shape
        mdl.add( Permute((1,2,3), input_shape=(224,224,3)) ) # WARNING : 0 is the sample dim

        for l in convblock(64, 1, bits=2):
            mdl.add(l)

        for l in convblock(128, 2, bits=2):
            mdl.add(l)
        
        for l in convblock(256, 3, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 4, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 5, bits=3):
            mdl.add(l)
        
        #mdl.add( Convolution2D(4096, 7, 7, activation='relu', name='fc6') ) # Keras 1
        mdl.add( Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6') ) # Keras 2
        if withDO:
            mdl.add( Dropout(0.5) )
        #mdl.add( Convolution2D(4096, 1, 1, activation='relu', name='fc7') ) # Keras 1
        mdl.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7') ) # Keras 2
        if withDO:
            mdl.add( Dropout(0.5) )
        #mdl.add( Convolution2D(2622, 1, 1, name='fc8') ) # Keras 1
        mdl.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') ) # Keras 2
        mdl.add( Flatten() )
        mdl.add( Activation('softmax') )
        
        return mdl
    
    else:
        # See following link for a version based on Keras functional API :
        # gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
        raise ValueError('not implemented')


# Reference : https://github.com/rcmalli/keras-vggface
# Reference : gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9


facemodel = vgg_face_blank()


facemodel.summary() # visual inspection of model architecture


# ## Load VGG weigths from .mat file
# 
# #### http://www.vlfeat.org/matconvnet/pretrained/#face-recognition
# ##### Download from console with :
# wget http://www.vlfeat.org/matconvnet/models/vgg-face.mat
# 
# ##### Alternatively :
# wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_matconvnet.tar.gz
# 

from scipy.io import loadmat


if False: # INFO : use this if you downloaded weights from vlfeat.org
    data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
    l = data['layers']
    description = data['meta'][0,0].classes[0,0].description
else: # INFO : use this if you downloaded weights from robots.ox.ac.uk
    data = loadmat('vgg_face_matconvnet/data/vgg_face.mat', matlab_compatible=False, struct_as_record=False)
    net = data['net'][0,0]
    l = net.layers
    description = net.classes[0,0].description


l.shape, description.shape


l[0,10][0,0].type[0], l[0,10][0,0].name[0]


l[0,10][0,0].weights[0,0].shape, l[0,10][0,0].weights[0,1].shape


def weight_compare(kmodel):
    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    #prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0,1,2,3) # INFO : for 'channels_last' setting of 'image_data_format'

    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        mattype = l[0,i][0,0].type[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print(matname, mattype)
            print(l[0,i][0,0].weights[0,0].transpose(prmt).shape, l[0,i][0,0].weights[0,1].shape)
            print(kmodel.layers[kindex].get_weights()[0].shape, kmodel.layers[kindex].get_weights()[1].shape)
            print('------------------------------------------')
        else:
            print('MISSING : ', matname, mattype)
            print('------------------------------------------')


#weight_compare(facemodel)


def copy_mat_to_keras(kmodel):

    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    #prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0,1,2,3) # INFO : for 'channels_last' setting of 'image_data_format'

    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            #print matname
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            #f_l_weights = np.flip(f_l_weights, 2) # INFO : for 'th' setting in dim_ordering
            #f_l_weights = np.flip(f_l_weights, 3) # INFO : for 'th' setting in dim_ordering
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
            #print '------------------------------------------'


copy_mat_to_keras(facemodel)


# ### Test on squared and well centered image
# 

im = Image.open('ak.png') # WARNING : this image is well centered and square
im = im.resize((224,224))


plt.imshow(np.asarray(im))


def pred(kmodel, crpimg, transform=False):
    
    # transform=True seems more robust but I think the RGB channels are not in right order
    
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:,:,0] -= 129.1863
        imarr[:,:,1] -= 104.7624
        imarr[:,:,2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        aux = copy.copy(imarr)
        #imarr[:, :, 0] = aux[:, :, 2]
        #imarr[:, :, 2] = aux[:, :, 0]

        #imarr[:,:,0] -= 129.1863
        #imarr[:,:,1] -= 104.7624
        #imarr[:,:,2] -= 93.5940

    #imarr = imarr.transpose((2,0,1)) # INFO : for 'th' setting of 'dim_ordering'
    imarr = np.expand_dims(imarr, axis=0)

    out = kmodel.predict(imarr)

    best_index = np.argmax(out, axis=1)[0]
    best_name = description[best_index,0]
    print(best_index, best_name[0], out[0,best_index], [np.min(out), np.max(out)])


crpim = im # WARNING : we deal with cropping in a latter section, this image is already fit

pred(facemodel, crpim, transform=False)
pred(facemodel, crpim, transform=True)


[(i, s[0]) for i, s in enumerate(description[:,0]) if ('laurie'.lower() in s[0].lower())]


description[100,0][0]


# ## Face Feature Vector : drop the last layer
# 

featuremodel = Model(inputs=facemodel.layers[0].input, outputs=facemodel.layers[-2].output)


def features(featmodel, crpimg, transform=False):
    
    # transform=True seems more robust but I think the RGB channels are not in right order
    
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:,:,0] -= 129.1863
        imarr[:,:,1] -= 104.7624
        imarr[:,:,2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        aux = copy.copy(imarr)
        #imarr[:, :, 0] = aux[:, :, 2]
        #imarr[:, :, 2] = aux[:, :, 0]

        #imarr[:,:,0] -= 129.1863
        #imarr[:,:,1] -= 104.7624
        #imarr[:,:,2] -= 93.5940

    #imarr = imarr.transpose((2,0,1))
    imarr = np.expand_dims(imarr, axis=0)

    fvec = featmodel.predict(imarr)[0,:]
    # normalize
    normfvec = math.sqrt(fvec.dot(fvec))
    return fvec/normfvec


f = features(featuremodel, crpim, transform=True)


f.shape, f.dot(f)


# ## Face extraction + Face identification
# #### This requires OpenCV :
# https://pypi.python.org/pypi/opencv-python
# #### See this tutorial on face CascadeClassifier :
# https://realpython.com/blog/python/face-recognition-with-python/
# 

import cv2


imagePath = 'Aamir_Khan.jpg'
#imagePath = 'mzaradzki.jpg'
#imagePath = 'hugh_laurie.jpg'
#imagePath = 'Colin_Firth.jpg'
#imagePath = 'someguy.jpg'


# WARNING : cascade XML file from this repo : https://github.com/shantnu/FaceDetect.git
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
faces = faceCascade.detectMultiScale(gray, 1.2, 5)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(image)


im = Image.open(imagePath)

(x, y, w, h) = faces[0]
center_x = x+w/2
center_y = y+h/2
b_dim = min(max(w,h)*1.2,im.width, im.height) # WARNING : this formula in incorrect
#box = (x, y, x+w, y+h)
box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)
# Crop Image
crpim = im.crop(box).resize((224,224))
plt.imshow(np.asarray(crpim))

pred(facemodel, crpim, transform=False)
pred(facemodel, crpim, transform=True)





get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import math
import copy

from scipy.misc import bytescale


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import merge


from fcn_keras2 import fcn32_blank, fcn_32s_to_16s, prediction


# ## Build model architecture
# 

# ### Fully Convolutional Networks for Semantic Segmentation
# ##### Jonathan Long, Evan Shelhamer, Trevor Darrell
# 
# www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
# 
# Extract from the article relating to the model architecture.
# 
# The model is derived from VGG16.
# 
# **remark** : deconvolution and conv-transpose are synonyms, they perform up-sampling
# 
# #### 4.1. From classifier to dense FCN
# 
# We decapitate each net by discarding the final classifier layer [**code comment** : *this is why fc8 is not included*], and convert all fully connected layers to convolutions.
# 
# We append a 1x1 convolution with channel dimension 21 [**code comment** : *layer named score_fr*] to predict scores for each of the PASCAL classes (including background) at each of the coarse output locations, followed by a deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs as described in Section 3.3.
# 
# 
# #### 4.2. Combining what and where
# We define a new fully convolutional net (FCN) for segmentation that combines layers of the feature hierarchy and
# refines the spatial precision of the output.
# While fully convolutionalized classifiers can be fine-tuned to segmentation as shown in 4.1, and even score highly on the standard metric, their output is dissatisfyingly coarse.
# The 32 pixel stride at the final prediction layer limits the scale of detail in the upsampled output.
# 
# We address this by adding skips that combine the final prediction layer with lower layers with finer strides.
# This turns a line topology into a DAG [**code comment** : *this is why some latter stage layers have 2 inputs*], with edges that skip ahead from lower layers to higher ones.
# As they see fewer pixels, the finer scale predictions should need fewer layers, so it makes sense to make them from shallower net outputs.
# Combining fine layers and coarse layers lets the model make local predictions that respect global structure.
# 
# We first divide the output stride in half by predicting from a 16 pixel stride layer.
# We add a 1x1 convolution layer on top of pool4 [**code comment** : *the score_pool4_filter layer*] to produce additional class predictions.
# We fuse this output with the predictions computed on top of conv7 (convolutionalized fc7) at stride 32 by adding a 2x upsampling layer and summing [**code comment** : *layer named sum*] both predictions [**code warning** : *requires first layer crop to insure the same size*].
# 
# Finally, the stride 16 predictions are upsampled back to the image [**code comment** : *layer named upsample_new*].
# 
# We call this net FCN-16s.
# 
# ### Remark :
# **The original paper mention that FCN-8s (slightly more complex architecture) does not provide much improvement so we stopped at FCN-16s**
# 

image_size = 64*8 # INFO: initially tested with 256, 448, 512


fcn32model = fcn32_blank(image_size)


#fcn32model.summary() # visual inspection of model architecture


fcn16model = fcn_32s_to_16s(fcn32model)


# INFO : dummy image array to test the model passes
imarr = np.ones((image_size,image_size, 3))
imarr = np.expand_dims(imarr, axis=0)

#testmdl = Model(fcn32model.input, fcn32model.layers[10].output) # works fine
testmdl = fcn16model # works fine
testmdl.predict(imarr).shape


if (testmdl.predict(imarr).shape != (1, image_size, image_size, 21)):
    print('WARNING: size mismatch will impact some test cases')


fcn16model.summary() # visual inspection of model architecture


# ## Load VGG weigths from .mat file
# 
# #### https://www.vlfeat.org/matconvnet/pretrained/#semantic-segmentation
# ##### Download from console with :
# wget https://www.vlfeat.org/matconvnet/models/pascal-fcn16s-dag.mat
# 

from scipy.io import loadmat


data = loadmat('pascal-fcn16s-dag.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
p = data['params']
description = data['meta'][0,0].classes[0,0].description


l.shape, p.shape, description.shape


class2index = {}
for i, clname in enumerate(description[0,:]):
    class2index[str(clname[0])] = i
    
print(sorted(class2index.keys()))


if False: # inspection of data structure
    print(dir(l[0,31].block[0,0]))
    print(dir(l[0,36].block[0,0]))


for i in range(0, p.shape[1]-1, 2):
    print(i,
          str(p[0,i].name[0]), p[0,i].value.shape,
          str(p[0,i+1].name[0]), p[0,i+1].value.shape)


for i in range(l.shape[1]):
    print(i,
          str(l[0,i].name[0]), str(l[0,i].type[0]),
          [str(n[0]) for n in l[0,i].inputs[0,:]],
          [str(n[0]) for n in l[0,i].outputs[0,:]])


# documentation for the dagnn.Crop layer :
# https://github.com/vlfeat/matconvnet/blob/master/matlab/%2Bdagnn/Crop.m


def copy_mat_to_keras(kmodel):
    
    kerasnames = [lr.name for lr in kmodel.layers]

    prmt = (0, 1, 2, 3) # WARNING : important setting as 2 of the 4 axis have same size dimension
    
    for i in range(0, p.shape[1]-1, 2):
        matname = '_'.join(p[0,i].name[0].split('_')[0:-1])
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print('found : ', (str(matname), kindex))
            l_weights = p[0,i].value
            l_bias = p[0,i+1].value
            f_l_weights = l_weights.transpose(prmt)
            if False: # WARNING : this depends on "image_data_format":"channels_last" in keras.json file
                f_l_weights = np.flip(f_l_weights, 0)
                f_l_weights = np.flip(f_l_weights, 1)
            print(f_l_weights.shape, kmodel.layers[kindex].get_weights()[0].shape)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
        else:
            print('not found : ', str(matname))


#copy_mat_to_keras(fcn32model)
copy_mat_to_keras(fcn16model)


im = Image.open('rgb.jpg') # http://www.robots.ox.ac.uk/~szheng/crfasrnndemo/static/rgb.jpg
im = im.crop((0,0,319,319)) # WARNING : manual square cropping
im = im.resize((image_size,image_size))


plt.imshow(np.asarray(im))
print(np.asarray(im).shape)


crpim = im # WARNING : we deal with cropping in a latter section, this image is already fit
preds = prediction(fcn16model, crpim, transform=False) # WARNING : transfrom=True requires a code change (dim order)


#imperson = preds[0,class2index['person'],:,:]
print(preds.shape)
imclass = np.argmax(preds, axis=3)[0,:,:]
print(imclass.shape)
plt.figure(figsize = (15, 7))
plt.subplot(1,3,1)
plt.imshow( np.asarray(crpim) )
plt.subplot(1,3,2)
plt.imshow( imclass )
plt.subplot(1,3,3)
plt.imshow( np.asarray(crpim) )
masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#plt.imshow( imclass, alpha=0.5 )
plt.imshow( masked_imclass, alpha=0.5 )


# List of dominant classes found in the image
for c in np.unique(imclass):
    print(c, str(description[0,c][0]))


bspreds = bytescale(preds, low=0, high=255)

plt.figure(figsize = (15, 7))
plt.subplot(2,3,1)
plt.imshow(np.asarray(crpim))
plt.subplot(2,3,3+1)
plt.imshow(bspreds[0,:,:,class2index['background']], cmap='seismic')
plt.subplot(2,3,3+2)
plt.imshow(bspreds[0,:,:,class2index['person']], cmap='seismic')
plt.subplot(2,3,3+3)
plt.imshow(bspreds[0,:,:,class2index['bicycle']], cmap='seismic')





get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import math
import copy

import skimage.io as io
from scipy.misc import bytescale


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import merge


from utils import fcn32_blank, fcn_32s_to_16s, prediction


# ## Build model architecture
# 

# ### Fully Convolutional Networks for Semantic Segmentation
# ##### Jonathan Long, Evan Shelhamer, Trevor Darrell
# 
# www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
# 
# Extract from the article relating to the model architecture.
# 
# The model is derived from VGG16.
# 
# **remark** : deconvolution and conv-transpose are synonyms, they perform up-sampling
# 
# #### 4.1. From classifier to dense FCN
# 
# We decapitate each net by discarding the final classifier layer [**code comment** : *this is why fc8 is not included*], and convert all fully connected layers to convolutions.
# 
# We append a 1x1 convolution with channel dimension 21 [**code comment** : *layer named score_fr*] to predict scores for each of the PASCAL classes (including background) at each of the coarse output locations, followed by a deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs as described in Section 3.3.
# 
# 
# #### 4.2. Combining what and where
# We define a new fully convolutional net (FCN) for segmentation that combines layers of the feature hierarchy and
# refines the spatial precision of the output.
# While fully convolutionalized classifiers can be fine-tuned to segmentation as shown in 4.1, and even score highly on the standard metric, their output is dissatisfyingly coarse.
# The 32 pixel stride at the final prediction layer limits the scale of detail in the upsampled output.
# 
# We address this by adding skips that combine the final prediction layer with lower layers with finer strides.
# This turns a line topology into a DAG [**code comment** : *this is why some latter stage layers have 2 inputs*], with edges that skip ahead from lower layers to higher ones.
# As they see fewer pixels, the finer scale predictions should need fewer layers, so it makes sense to make them from shallower net outputs.
# Combining fine layers and coarse layers lets the model make local predictions that respect global structure.
# 
# We first divide the output stride in half by predicting from a 16 pixel stride layer.
# We add a 1x1 convolution layer on top of pool4 [**code comment** : *the score_pool4_filter layer*] to produce additional class predictions.
# We fuse this output with the predictions computed on top of conv7 (convolutionalized fc7) at stride 32 by adding a 2x upsampling layer and summing [**code comment** : *layer named sum*] both predictions [**code warning** : *requires first layer crop to insure the same size*].
# 
# Finally, the stride 16 predictions are upsampled back to the image [**code comment** : *layer named upsample_new*].
# 
# We call this net FCN-16s.
# 
# ### Remark :
# **The original paper mention that FCN-8s (slightly more complex architecture) does not provide much improvement so we stopped at FCN-16s**
# 

image_size = 64*8 # INFO: initially tested with 256, 448, 512


fcn32model = fcn32_blank(image_size)


#fcn32model.summary() # visual inspection of model architecture


fcn16model = fcn_32s_to_16s(fcn32model)


# INFO : dummy image array to test the model passes
imarr = np.ones((3,image_size,image_size))
imarr = np.expand_dims(imarr, axis=0)

#testmdl = Model(fcn32model.input, fcn32model.layers[10].output) # works fine
testmdl = fcn16model # works fine
testmdl.predict(imarr).shape


if (testmdl.predict(imarr).shape != (1,21,image_size,image_size)):
    print('WARNING: size mismatch will impact some test cases')


fcn16model.summary() # visual inspection of model architecture


# ## Load VGG weigths from .mat file
# 
# #### https://www.vlfeat.org/matconvnet/pretrained/#semantic-segmentation
# ##### Download from console with :
# wget https://www.vlfeat.org/matconvnet/models/pascal-fcn16s-dag.mat
# 

from scipy.io import loadmat


data = loadmat('pascal-fcn16s-dag.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
p = data['params']
description = data['meta'][0,0].classes[0,0].description


l.shape, p.shape, description.shape


class2index = {}
for i, clname in enumerate(description[0,:]):
    class2index[str(clname[0])] = i
    
print(sorted(class2index.keys()))


if False: # inspection of data structure
    print(dir(l[0,31].block[0,0]))
    print(dir(l[0,36].block[0,0]))


for i in range(0, p.shape[1]-1, 2):
    print(i,
          str(p[0,i].name[0]), p[0,i].value.shape,
          str(p[0,i+1].name[0]), p[0,i+1].value.shape)


for i in range(l.shape[1]):
    print(i,
          str(l[0,i].name[0]), str(l[0,i].type[0]),
          [str(n[0]) for n in l[0,i].inputs[0,:]],
          [str(n[0]) for n in l[0,i].outputs[0,:]])


# documentation for the dagnn.Crop layer :
# https://github.com/vlfeat/matconvnet/blob/master/matlab/%2Bdagnn/Crop.m


def copy_mat_to_keras(kmodel):
    
    kerasnames = [lr.name for lr in kmodel.layers]

    prmt = (3,2,0,1) # WARNING : important setting as 2 of the 4 axis have same size dimension
    
    for i in range(0, p.shape[1]-1, 2):
        matname = '_'.join(p[0,i].name[0].split('_')[0:-1])
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print 'found : ', (str(matname), kindex)
            l_weights = p[0,i].value
            l_bias = p[0,i+1].value
            f_l_weights = l_weights.transpose(prmt)
            f_l_weights = np.flip(f_l_weights, 2)
            f_l_weights = np.flip(f_l_weights, 3)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
        else:
            print 'not found : ', str(matname)


#copy_mat_to_keras(fcn32model)
copy_mat_to_keras(fcn16model)


im = Image.open('rgb.jpg') # http://www.robots.ox.ac.uk/~szheng/crfasrnndemo/static/rgb.jpg
im = im.crop((0,0,319,319)) # WARNING : manual square cropping
im = im.resize((image_size,image_size))


plt.imshow(np.asarray(im))


crpim = im # WARNING : we deal with cropping in a latter section, this image is already fit
preds = prediction(fcn16model, crpim, transform=True)


#imperson = preds[0,class2index['person'],:,:]
imclass = np.argmax(preds, axis=1)[0,:,:]

plt.figure(figsize = (15, 7))
plt.subplot(1,3,1)
plt.imshow( np.asarray(crpim) )
plt.subplot(1,3,2)
plt.imshow( imclass )
plt.subplot(1,3,3)
plt.imshow( np.asarray(crpim) )
masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#plt.imshow( imclass, alpha=0.5 )
plt.imshow( masked_imclass, alpha=0.5 )


# List of dominant classes found in the image
for c in np.unique(imclass):
    print c, str(description[0,c][0])


bspreds = bytescale(preds, low=0, high=255)

plt.figure(figsize = (15, 7))
plt.subplot(2,3,1)
plt.imshow(np.asarray(crpim))
plt.subplot(2,3,3+1)
plt.imshow(bspreds[0,class2index['background'],:,:], cmap='seismic')
plt.subplot(2,3,3+2)
plt.imshow(bspreds[0,class2index['person'],:,:], cmap='seismic')
plt.subplot(2,3,3+3)
plt.imshow(bspreds[0,class2index['bicycle'],:,:], cmap='seismic')





get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers


# ## Linear auto-encoder : like PCA
# ### We get a linear model by removing activation functions
# 

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

if True: # no sparsity constraint
    encoded = Dense(encoding_dim, activation=None)(input_img)
else:
    encoded = Dense(encoding_dim, activation=None,
                    activity_regularizer=regularizers.activity_l1(10e-5))(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation=None)(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


# train autoencoder to reconstruct MNIST digits
# use a per-pixel binary crossentropy loss
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()





from __future__ import print_function # for python 2.7 users


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D


from keras.applications import vgg16


trueVGG = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)


from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering


# ## Build model architecture
# 

def convblock(cdim, nb, bits=3):
    L = []
    
    for k in range(1,bits+1):
        convname = 'block'+str(nb)+'_conv'+str(k)
        L.append( Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname) ) # Keras 2
    
    L.append( AveragePooling2D((2, 2), strides=(2, 2)) ) # WARNING : MaxPooling2D in **true** VGG model
    
    return L


def vgg_headless(): # we remove the top classification layers
    
    mdl = Sequential()
        
    # First layer is a dummy-permutation = Identity to specify input shape
    mdl.add( Permute((1,2,3), input_shape=(224,224,3)) ) # WARNING : 0 is the sample dim

    for l in convblock(64, 1, bits=2):
        mdl.add(l)

    for l in convblock(128, 2, bits=2):
        mdl.add(l)
        
    for l in convblock(256, 3, bits=3):
        mdl.add(l)
            
    for l in convblock(512, 4, bits=3):
        mdl.add(l)
            
    for l in convblock(512, 5, bits=3):
        mdl.add(l)
        
    return mdl


pseudoVGG = vgg_headless()


pseudoNames = [lr.name for lr in pseudoVGG.layers]

for lr in trueVGG.layers:
    if ('_conv' in lr.name):
        idx = pseudoNames.index(lr.name)
        print(lr.name, idx)
        Ws, Bs = lr.get_weights()
        pseudoVGG.layers[idx].set_weights([Ws, Bs])


pseudoVGG.summary() # visual inspection of model architecture


featureLayerIndex = 5
featureModel = Model(inputs=pseudoVGG.layers[0].input, outputs=pseudoVGG.layers[featureLayerIndex].output)


im = Image.open('ak.png') # WARNING : this image is well centered and square
im = im.resize((224,224))


plt.imshow(np.asarray(im))


def pred(kmodel, crpimg, transform=False):
    
    # transform=True seems more robust but I think the RGB channels are not in right order
    
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:,:,0] -= 129.1863
        imarr[:,:,1] -= 104.7624
        imarr[:,:,2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        aux = copy.copy(imarr)
        #imarr[:, :, 0] = aux[:, :, 2]
        #imarr[:, :, 2] = aux[:, :, 0]

        #imarr[:,:,0] -= 129.1863
        #imarr[:,:,1] -= 104.7624
        #imarr[:,:,2] -= 93.5940

    #imarr = imarr.transpose((2,0,1)) # INFO : for 'th' setting of 'dim_ordering'
    imarr = np.expand_dims(imarr, axis=0)

    return kmodel.predict(imarr)


crpim = im # WARNING : we deal with cropping in a latter section, this image is already fit

pred(featureModel, crpim, transform=False).shape


# ### See documentation at :
# - https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# - https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
# 

# build the VGG16 network
model = vgg16.VGG16(include_top=False, weights='imagenet')

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


from keras import backend as K

layer_name = 'block5_conv3'
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])

input_img = model.layers[0].input # WARNING : not sure about this line

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])


import numpy as np

img_width = 128
img_height = 128

# we start from a gray image with some noise
input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step





from theano.sandbox import cuda


get_ipython().magic('matplotlib inline')
from __future__ import division, print_function


import math
import numpy as np
import random
import sys

from numpy.random import normal
import matplotlib.pyplot as plt


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Activation, merge, Flatten, Dropout, Lambda
from keras.layers import LSTM, SimpleRNN, TimeDistributed
from keras.models import Model, Sequential
from keras.layers.merge import Add, add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.constraints import nonneg
from keras.layers.convolutional import *
from keras import backend as K
from keras.utils.data_utils import get_file


look_back = 12
batch_size = 1


def mllklh(args): # minus log-likelihood of gaussian
    var_t, eps2_t = args
    return 0.5*math.log(2*math.pi) + 0.5*K.log(var_t)  + 0.5*(eps2_t/var_t)


#len(Xs), Xs[0].shape, modelUNR.predict([X[10:15,0] for X in Xs]).shape


# ### RNN architecture : 2 models that share same weights
# 

MatPlus = np.zeros((look_back-1,look_back))
MatPlus[:,1:] = np.eye(look_back-1)
#print(MatPlus)

MatMinus = np.zeros((look_back-1,look_back))
MatMinus[:,:-1] = np.eye(look_back-1)
#print(MatMinus)


Dplus = Dense(look_back-1,
              weights=[MatPlus.T, np.zeros((look_back-1,))],
              input_dim=look_back)
Dplus.trainable = False

Dminus = Dense(look_back-1,
              weights=[MatMinus.T, np.zeros((look_back-1,))],
              input_dim=look_back)
Dminus.trainable = False


I = Input(batch_shape=(batch_size, look_back,1), dtype='float32')
I2 = Lambda(lambda x : K.square(x), output_shape=(look_back,1))(I)
rnn = SimpleRNN(return_sequences=True, unroll=False,
                units=1, input_shape=(look_back, 1),
                bias_constraint=nonneg(), # insure positive var
                kernel_constraint=nonneg(), # insure positive var
                recurrent_constraint=nonneg(), # insure positive var
                activation=None,
                stateful=True)
O1 = rnn(I2)
O1f = Flatten()(O1)
O1m = Dminus(O1f)

V = Lambda(lambda x : K.sqrt(x), output_shape=(look_back,))(O1f) # get volatility

I2f = Flatten()(I2)
I2p = Dplus(I2f)

Errors = Lambda(mllklh, output_shape=(look_back-1,))([O1m,I2p])

Error = Lambda(lambda x : K.sum(x, axis=1), output_shape=(look_back-1,))(Errors)

modelT = Model(inputs=I, outputs=Errors) # training model

def special_loss(dummy, errorterms):
    return errorterms

modelT.compile(optimizer='adadelta', loss=special_loss)

modelV = Model(inputs=I, outputs=V) # simulation model


Dplus.get_weights()[0].shape, Dplus.get_weights()[1].shape


I._keras_shape, I2._keras_shape, O1._keras_shape, O1f._keras_shape, V._keras_shape, Errors._keras_shape, Error._keras_shape


onearr = np.ones((batch_size, look_back, 1)).astype('float32')
print(I2.eval({I:onearr}).shape)
print(O1.eval({I:onearr}).shape)
print(O1f.eval({I:onearr}).shape)
print(V.eval({I:onearr}).shape)
print(Errors.eval({I:onearr}).shape)
print(Error.eval({I:onearr}).shape)
print(modelT.predict(onearr).shape)
print(modelV.predict(onearr).shape)


rnn(I2).eval({I:onearr}) # dry run to allow weight setting
rnn.set_weights([np.array([[0.5]]),np.array([[1]]),np.array([1.5])])
print( O1f.eval({I:onearr}) )
print( O1m.eval({I:onearr}) )


#print( I2f.eval({I:onearr}) )
#print( I2p.eval({I:onearr}) )


Error.eval({I:onearr}).shape


modelT.summary()


# #### Manually specify a GARCH model
# 

kappa = 0.000003
alpha = 0.85
beta = 0.10
lvar = kappa / (1-alpha-beta)
print(math.sqrt(lvar)*math.sqrt(255))


# #### Copy the known GARCH parameters as initial weights for the RNN
# 

math.sqrt(lvar) # standard deviation of simulated data set


F = 1/math.sqrt(lvar) # will have to scale training data by F and Kappa by F^2 (alpha and beta unchanged)


rnn(I2).eval({I:onearr}) # dry run to allow weight setting
rnn.set_weights([np.array([[beta]]),np.array([[alpha]]),np.array([kappa])*F*F])


# #### Simulate the GARCH dynamic
# 

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, allow_overlap=True):
    dataX, dataY = [], []
    if allow_overlap:
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    else:
        # non overlap
        for i in range(0, int(dataset.shape[0]/batch_size)*batch_size-look_back, look_back):
            #print(i)
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

train = []
trainvars = []
var_t = lvar
for t in range(250*8):
    eps = math.sqrt(var_t) * normal()
    var_t = kappa + alpha * var_t + beta * (eps*eps)
    train.append(eps) # percent
    trainvars.append(var_t)
train = np.array(train).reshape(-1,1)
trainX, trainY = create_dataset(train, look_back, allow_overlap=False)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#print(trainX, trainY)


trainX.shape, trainY.shape, trainX.transpose((0,2,1)).shape


plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(np.cumsum(train))
plt.subplot(2,1,2)
plt.plot(np.sqrt(trainvars)*math.sqrt(255))
plt.show()


var_t = 0#lvar
check_vars = []
#for eps in train[0:(50*look_back),0]:
#    var_t = kappa + alpha * var_t + beta * (eps*eps)
for k, eps in enumerate(train[(50*look_back):((50*look_back)+3*batch_size*look_back),0]):
    var_t = kappa + alpha * var_t + beta * (eps*eps)
    check_vars.append(var_t)
    if k<3*look_back:
        print(math.sqrt(var_t*255))
        if ((k>0) and ((k+1) % look_back == 0)):
            print('-------', k, '--------')


rnn.states[0].get_value().shape, rnn.states[0].get_value()[:,0]*math.sqrt(255)/F


trainX.transpose((0,2,1)).shape


A = trainX.transpose((0,2,1))[50,:,0]
B = train[(50*look_back):((50*look_back)+1*look_back),0]

var_t = 0

print( math.sqrt((kappa + alpha * var_t + beta * (A[0]**2))*255) )
print( math.sqrt((kappa + alpha * var_t + beta * (B[0]**2))*255) )


np.sqrt(np.array(check_vars)*255)[0], np.squeeze(Vs.reshape((1,-1)), axis=0)[0]


rnn.reset_states()


#Vs = V.eval({I:trainX.transpose((0,2,1))[100:(100+batch_size),:,:].astype('float32')*F})*math.sqrt(255)/F
Vs0 = modelV.predict( trainX.transpose((0,2,1))[50:(50+batch_size),:,:].astype('float32')*F )*math.sqrt(255)/F
Vs1 = modelV.predict( trainX.transpose((0,2,1))[51:(51+batch_size),:,:].astype('float32')*F )*math.sqrt(255)/F
Vs2 = modelV.predict( trainX.transpose((0,2,1))[52:(52+batch_size),:,:].astype('float32')*F )*math.sqrt(255)/F
Vs = np.vstack([Vs0,Vs1,Vs2])
#np.squeeze(Vs.reshape((1,-1)))#[0:20,0], Vs[0,:], Vs[1,:]
#Vs[0:3,:]
plt.figure(figsize=(10,7))
plt.plot( np.sqrt(np.array(check_vars)*255), c='red', marker='+' )
plt.plot( np.squeeze(Vs.reshape((1,-1)), axis=0), c='black' )


# #### Compute the model Loss before training to insure loss goes down
# 

test_arr = trainX.transpose((0,2,1))[50:(50+batch_size),:,:].astype('float32')
ErrorStart = np.sum(Errors.eval({I:test_arr*F }))
print(ErrorStart)


print( modelT.predict(trainX.transpose((0,2,1))[50:(50+batch_size),:,:]*F)[0] )
print('-------------------------')
print( modelV.predict(trainX.transpose((0,2,1))[50:(50+batch_size),:,:]*F)[0]*math.sqrt(255)/F )


max_batches = int(trainX.transpose((0,2,1)).shape[0]/batch_size)
max_batches


# #### A few runs of training for sanity checks
# 

modelT.optimizer.lr.set_value(1e-1) # 1e-3 seems too large !
#modelT.optimizer.lr.get_value()


Ydummy = trainX.transpose((0,2,1))
#print(Ydummy.shape)
#print( trainX.transpose((0,2,1)).shape )
hist0 = modelT.fit(trainX.transpose((0,2,1))[0:(max_batches*batch_size),:,:]*F,
                   Ydummy[0:(max_batches*batch_size),0:-1,0],
                   epochs=10,
                   batch_size=batch_size,
                   shuffle=False,
                   verbose=0)


# #### Compare Trained-Loss to check it did not worsen
# 

test_arr = trainX.transpose((0,2,1))[50:50+batch_size:,:].astype('float32')
ErrorEnd = np.sum(Errors.eval({I:test_arr*F}))
print(ErrorStart, ErrorEnd)


# #### Compare long-term vol to check similarity
# 

math.sqrt(rnn.get_weights()[2][0]/(1-rnn.get_weights()[0][0][0]-rnn.get_weights()[1][0][0])*255)/F


kappa*F*F, rnn.get_weights()[2][0], rnn.get_weights()[0][0][0], rnn.get_weights()[1][0][0]





# Rather than importing everything manually, we'll make things easy
#   and load them all in utils.py, and just import them from there.
get_ipython().magic('matplotlib inline')
import utils; reload(utils)
from utils import *


get_ipython().magic('matplotlib inline')
from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_batches, plot_confusion_matrix, get_data


from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image


#path = "../data/dogsandcats_small/" # we copied a fraction of the full set for tests
path = "../data/dogsandcats/"
model_path = path + "models/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    print('Done')


from vgg16 import Vgg16


batch_size = 100


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, 
                batch_size=batch_size, class_mode='categorical'):
    return gen.flow_from_directory(path+dirname, target_size=(224,224), 
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches('valid', shuffle=False, batch_size=batch_size) # no shuffle as we store conv output
trn_batches = get_batches('train', shuffle=False, batch_size=batch_size) # no shuffle as we store conv output


val_batches.filenames[0:10]


val_labels = onehot(val_batches.classes)
trn_labels = onehot(trn_batches.classes)


# DONT USE IT FOR NOW
if False:
    realvgg = Vgg16()
    conv_layers, fc_layers = split_at(realvgg.model, Convolution2D)
    conv_model = Sequential(conv_layers)


vggbase = Vgg16()
vggbase.model.pop()
vggbase.model.pop()


# ### Will take 1 or 2 minutes to complete the 1st time
# 

# DONT USE IT FOR NOW
if False:
    try:
        val_features = load_array(model_path+'valid_convlayer_features.bc')
        if False: # force update
            raise
    except:
        print('Missing file')
        val_features = conv_model.predict_generator(val_batches, val_batches.nb_sample)
        save_array(model_path + 'valid_convlayer_features.bc', val_features)


try:
    val_vggfeatures = load_array(model_path+'valid_vggbase_features.bc')
    if False: # force update
        raise
except:
    print('Missing file')
    val_vggfeatures = vggbase.model.predict_generator(val_batches, val_batches.nb_sample)
    save_array(model_path + 'valid_vggbase_features.bc', val_vggfeatures)


# ### Will take a few minutes (maybe 10) to complete the 1st time
# 

# DONT USE IT FOR NOW
if False:
    try:
        trn_features = load_array(model_path+'train_convlayer_features.bc')
        if False: # force update
            raise
    except:
        print('Missing file')
        trn_features = conv_model.predict_generator(trn_batches, trn_batches.nb_sample)
        save_array(model_path + 'train_convlayer_features.bc', trn_features)


try:
    trn_vggfeatures = load_array(model_path+'train_vggbase_features.bc')
    if False: # force update
        raise
except:
    print('Missing file')
    trn_vggfeatures = vggbase.model.predict_generator(trn_batches, trn_batches.nb_sample)
    save_array(model_path + 'train_vggbase_features.bc', trn_vggfeatures)


# ### Ready to train the model
# 

ll_layers = [BatchNormalization(input_shape=(4096,)),
             Dropout(0.25),
             Dense(2, activation='softmax')]
ll_model = Sequential(ll_layers)
ll_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


ll_model.optimizer.lr = 10*1e-5
ll_model.fit(trn_vggfeatures, trn_labels, validation_data=(val_vggfeatures, val_labels), nb_epoch=5)


#ll_model.save_weights(model_path+'llmodel_finetune1.h5')
#ll_model.load_weights(model_path+'llmodel_finetune1.h5')


test_batches = get_batches('test', shuffle=False, batch_size=batch_size, class_mode=None)
testfiles = test_batches.filenames
testfiles[0:10]


# ### Will take a few minutes (maybe 5) to complete the 1st time
# 

try:
    test_vggfeatures = load_array(model_path+'test_vggbase_features.bc')
    if False: # force update
        raise
except:
    print('Missing file')
    test_vggfeatures = vggbase.model.predict_generator(test_batches, test_batches.nb_sample)
    save_array(model_path + 'test_vggbase_features.bc', test_vggfeatures)


test_preds = ll_model.predict_on_batch(test_vggfeatures)


assert(len(test_preds) == 12500)


test_preds[0:10]


dog_idx = 1
Z1 = [{'id':int(f.split('/')[-1].split('.')[0]), 'label':min(max(round(p[dog_idx],5),0.0001),0.9999)} 
      for f, p in zip(testfiles, test_preds)]
def comp(x,y):
    return int(x['id']) - int(y['id'])
Z1 = sorted(Z1, comp)
Z1[0:18]


import csv
        
with open('predictions.csv', 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for z in Z1:
        writer.writerow(z)








