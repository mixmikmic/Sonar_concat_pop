# # Listing 3-1 Illustate 2D convolution of Images through a Toy example
# 

## Illustate 2D convolution of Images through a Toy example

import scipy.signal
import numpy as np

# Take a 7x7 image as example

image = np.array([[1, 2, 3, 4, 5, 6, 7],
                 [8, 9, 10, 11, 12, 13, 14],
                 [15, 16, 17, 18, 19, 20, 21],
                 [22, 23, 24, 25, 26, 27, 28],
                 [29, 30, 31, 32, 33, 34, 35],
                 [36, 37, 38, 39, 40, 41, 42],
                 [43, 44, 45, 46, 47, 48, 49]])

# Defined an image processing kernel

filter_kernel = np.array([[-1, 1, -1],
                          [-2, 3, 1],
                          [2, -4, 0]])

# Convolve the image with the filter kernel through scipy 2D convolution to produce an output image of same dimension as that of the input

I = scipy.signal.convolve2d(image, filter_kernel,mode='same', boundary='fill', fillvalue=0)
print 'Scipy convolve2d output'
print(I)

# We replicate the same logic of a Scipy 2D convolution by following the below steps
# a) The boundaries need to be extended in both directions for the image and padded with zeroes.
#    For convolving the 7x7 image by 3x3 kernel the dimensions needs to be extended by (3-1)/2 i.e 1
#    on either size for each dimension. So a skeleton image of 9x9 image would be created
#    in which the boundaries of 1 pixel are pre-filled with zero.
# b) The kernel needs to be flipped i.e rotated by 180 degrees
# c) The flipped kernel needs to placed at each cordinate location for the image and then the sum of
#    cordinatewise product with the image intensities need to be computed. These sum for each co-ordinate would give
#    the intensities for the output image.

row,col=7,7

## Rotate the filter kernel twice by 90 degree to get 180 rotation

filter_kernel_flipped = np.rot90(filter_kernel,2)

## Pad the boundaries of the image with zeroes and fill the rest from the original image

image1 = np.zeros((9,9))

for i in xrange(row):
    for j in xrange(col):
        image1[i+1,j+1] = image[i,j]

#print(image1)

## Define the output image

image_out = np.zeros((row,col))

## Dynamic shifting of the flipped filter at each image cordinate and then computing the convolved sum.

for i in xrange(1,1+row):
    for j in xrange(1,1+col):
        arr_chunk = np.zeros((3,3))
        for k,k1 in zip(xrange(i-1,i+2),xrange(3)):
            for l,l1 in zip(xrange(j-1,j+2),xrange(3)):
                arr_chunk[k1,l1] = image1[k,l]

        image_out[i-1,j-1] = np.sum(np.multiply(arr_chunk,filter_kernel_flipped))
print "2D convolution implementation"
print(image_out) 


# # Listing 3-2 Convolution of an Image with Mean filter 
# 

import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
get_ipython().run_line_magic('matplotlib', 'inline')

img = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
mean = 0
var = 100
sigma = var**0.5
row,col = np.shape(gray)
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col)
gray_noisy = gray + gauss
print "Image after applying Gaussian Noise"
plt.imshow(gray_noisy,cmap='gray')


## Mean filter
Hm = np.array([[1,1,1],[1,1,1],[1,1,1]])/float(9)
Gm = convolve2d(gray_noisy,Hm,mode='same')
plt.imshow(Gm,cmap='gray') 
print "Image after convolving with Mean filter"


# # Listing 3-3. Median Filter illustration
# 

#----------------------------------------------------------------------------------------
## First create an image with Salt and Pepper Noise 
#----------------------------------------------------------------------------------------
## Generate random integers from 0 to 20
## If the value is zero we will replace the image pixel with a low value of 0 that corresponds to a black pixel
## If the value is 20 we will replace the image pixel with a high value of 255 that correspondsa to a white pixel
## Since we have taken 20 intergers and out of which we will only tag integers 1 and 20 as salt and pepper noise
## hence we would have approximately 10% of the overall pixels as salt and pepper noise. If we want to reduce it
## to 5 % we can taken integers from 0 to 40 and then treat 0 as indicator for black pixel and 40 as an indicator for white pixel.

np.random.seed(0)

gray_sp = gray*1
sp_indices = np.random.randint(0,21,[row,col])

for i in xrange(row):
    for j in xrange(col):
        if sp_indices[i,j] == 0:
            gray_sp[i,j] = 0
        if sp_indices[i,j] == 20:
            gray_sp[i,j] = 255
plt.imshow(gray_sp,cmap='gray')
print "Image after applying Salt and Pepper Noise"






#-----------------------------------------------------------------------------------------------------------
# Remove the Salt and Pepper Noise 
#-----------------------------------------------------------------------------------------------------------
## Now we want to remove the salt and pepper noise through a median filter.
## Using the opencv Median Filter for the same

gray_sp_removed = cv2.medianBlur(gray_sp,3)
plt.imshow(gray_sp_removed,cmap='gray')
print"Removing Salt and Pepper Noise with OpenCV Median Filter"


##Implementation of the 3x3 Median Filter without using opencv

gray_sp_removed_exp = gray*1

for i in xrange(row):
    for j in xrange(col):
        local_arr = []
        for k in xrange(np.max([0,i-1]),np.min([i+2,row])):
            for l in xrange(np.max([0,j-1]),np.min([j+2,col])):
                local_arr.append(gray_sp[k,l])

        gray_sp_removed_exp[i,j] = np.median(local_arr)
plt.imshow(gray_sp_removed_exp,cmap='gray')
print "Image produced by applying Median Filter Logic"        


# # Listing 3-4. Illustration of the Guassian Filter
# 

# Creating the Gaussian Filter 
Hg = np.zeros((20,20))

for i in xrange(20):
    for j in xrange(20):
        Hg[i,j] = np.exp(-((i-10)**2 + (j-10)**2)/10)

plt.imshow(Hg,cmap='gray')
print "Gaussian Blur Filter"


gray_blur = convolve2d(gray,Hg,mode='same')
plt.imshow(gray_blur,cmap='gray')
print "Image after convolving with  Gaussiab Blur Filter Created above"


gray_high = gray - gray_blur
plt.imshow(gray_high,cmap='gray')
print "Figh Frequency Component of Image"


gray_enhanced = gray + 0.025*gray_high
plt.imshow(gray_enhanced,cmap='gray')
print "Enhanced Image with some portion of High Frequency Component added"


# # Listing 3-5 Convolution using a Sobel Filter
# 

Hx = np.array([[ 1,0, -1],[2,0,-2],[1,0,-1]],dtype=np.float32)
Gx = convolve2d(gray,Hx,mode='same')
plt.imshow(Gx,cmap='gray')
print "Image after convolving with Horizontal Sobel Filter"


Hy = np.array([[ -1,-2, -1],[0,0,0],[1,2,1]],dtype=np.float32)
Gy = convolve2d(gray,Hy,mode='same')
plt.imshow(Gy,cmap='gray')
print "Image after convolving with Vertical Sobel Filter"


G = (Gx*Gx + Gy*Gy)**0.5
plt.imshow(G,cmap='gray')
print 'Image after combining outputs from both Horizontal and Vertical Sobel Filters'


# # Listing 3-6 Convolutional Neural Network for Digit Recognition on the MNIST dataset
# 

##################################################
##Import the required libraries and read the MNIST dataset
##################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.examples.tutorials.mnist import input_data
import time
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

###########################################
## Set the value of the Parameters
###########################################

learning_rate = 0.01
epochs = 20
batch_size = 256
num_batches = mnist.train.num_examples/batch_size
input_height = 28
input_width = 28
n_classes = 10
dropout = 0.75
display_step = 1
filter_height = 5
filter_width = 5
depth_in = 1
depth_out1 = 64
depth_out2 = 128



###########################################
# input output definition
###########################################
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)
###########################################
## Store the weights
## Number of weights of filters to be learnt in 'wc1' => filter_height*filter_width*depth_in*depth_out1
## Number of weights of filters to be learnt in 'wc1' => filter_height*filter_width*depth_out1*depth_out2
## No of Connections to the fully Connected layer => Each maxpooling operation reduces the image size to 1/4.
## So two maxpooling reduces the imase size to /16. There are depth_out2 number of images each of size 1/16 ## of the original image size of input_height*input_width. So there is total of
## (1/16)*input_height* input_width* depth_out2 pixel outputs which when connected to the fully connected layer ## with 1024 units would provide (1/16)*input_height* input_width* depth_out2*1024 connections.
###########################################
weights = {
'wc1' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])),
'wc2' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])),
'wd1' : tf.Variable(tf.random_normal([(input_height/4)*(input_height/4)* depth_out2,1024])),
'out' : tf.Variable(tf.random_normal([1024,n_classes]))
}
#################################################
## In the 1st Convolutional Layer there are 64 feature maps and that corresponds to 64 biases in 'bc1'
## In the 2nd Convolutional Layer there are 64 feature maps and that corresponds to 128 biases in 'bc2'
## In the Fully Connected Layer there are 1024units and that corresponds to 1024 biases in 'bd1'
## In the output layet there are 10 classes for the Softmax and that corresponds to 10 biases in 'out'
#################################################
biases = {
'bc1' : tf.Variable(tf.random_normal([64])),
'bc2' : tf.Variable(tf.random_normal([128])),
'bd1' : tf.Variable(tf.random_normal([1024])),
'out' : tf.Variable(tf.random_normal([n_classes]))
}


##################################################
## Create the different layers
##################################################

'''C O N V O L U T I O N L A Y E R'''
def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

''' P O O L I N G L A Y E R'''
def maxpool2d(x,stride=2):
    return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')
##################################################
## Create the feed forward model
##################################################
def conv_net(x,weights,biases,dropout):
##################################################
## Reshape the input in the 4 dimensional image
## 1st dimension - image index
## 2nd dimension - height
## 3rd dimension - width
## 4th dimension - depth
    x = tf.reshape(x,shape=[-1,28,28,1])
##################################################
## Convolutional layer 1
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,2)
## Convolutional layer 2
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2,2)
## Now comes the fully connected layer
    fc1 = tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
## Apply Dropout
    fc1 = tf.nn.dropout(fc1,dropout)
## Output class prediction
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out


#######################################################
# Defining the tensorflow Ops for different activities
#######################################################
pred = conv_net(x,weights,biases,keep_prob)
# Define loss function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
## initializing all variables
init = tf.global_variables_initializer()
####################################################
## Launch the execution Graph
####################################################
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):
            
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob: 1.})
            if epochs % display_step == 0:
                print("Epoch:", '%04d' % (i+1),
                "cost=", "{:.9f}".format(loss),
                "Training accuracy","{:.5f}".format(acc))
    print('Optimization Completed')

    y1 = sess.run(pred,feed_dict={x:mnist.test.images[:256],keep_prob: 1})
    test_classes = np.argmax(y1,1)
    print('Testing Accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob: 1}))
    f, a = plt.subplots(1, 10, figsize=(10, 2))

    for i in range(10):
        a[i].imshow(np.reshape(mnist.test.images[i],(28, 28)))
        print test_classes[i]

end_time = time.time()
print('Total processing time:',end_time - start_time)


# # Listing 3-7 Real World use of Convolutional Neural Network
# 

########################################################

## Load the relevant libraries
## Download the data from https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening

######################################################## 
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import time
from keras.utils import np_utils
import os 
import tensorflow as tf

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64,64), cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        #path = os.path.join('.', 'Downloads', 'Intel','train', fld, '*.jpg')
        path = os.path.join('/media', 'santanu','9eb9b6dc-b380-486e-b4fd-c424a325b976','Kaggle Competitions','Intel','train', fld, '*.jpg')
        files = glob.glob(path)
        
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
            
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('/media', 'santanu','9eb9b6dc-b380-486e-b4fd-c424a325b976','Kaggle Competitions','Intel','Additional', fld, '*.jpg')
        files = glob.glob(path)
        
        for fl in files:
            flbase = os.path.basename(fl)
            #print fl
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('/media', 'santanu','9eb9b6dc-b380-486e-b4fd-c424a325b976','Kaggle Competitions','Intel','test', fld, '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)
    path = os.path.join('/media', 'santanu','9eb9b6dc-b380-486e-b4fd-c424a325b976','Kaggle Competitions','Intel','stg2', fld, '*.jpg')
    files = sorted(glob.glob(path))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)
    
    return X_test, X_test_id



def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 2,3, 1))
    train_data = train_data.transpose((0, 1,3, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 3)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0,2,3,1))
    train_data = test_data.transpose((0, 1,3, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

##########################################################
## Read and Normalalize the train data
########################################################## 

#train_data, train_target, train_id = read_and_normalize_train_data()

##########################################################
## Shuffle the input training data to aid Stochastic gradient descent
########################################################## 
from random import shuffle
# Given list1 and list2
list1_shuf = []
list2_shuf = []
index_shuf = range(len(train_data))
shuffle(index_shuf)
for i in index_shuf:
    list1_shuf.append(train_data[i,:,:,:])
    list2_shuf.append(train_target[i,])
list1_shuf = np.array(list1_shuf,dtype=np.uint8)
list2_shuf = np.array(list2_shuf,dtype=np.uint8)

##########################################################
## TensorFlow activities for Network Definition and Training
##########################################################
## Create the different layers 



## Create  the different layers

channel_in = 3 
channel_out = 64
channel_out1 = 128

'''C O N V O L U T I O N    L A Y E R'''
def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


''' P O O L I N G  L A Y E R'''
def maxpool2d(x,stride=2):
    return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')

## Create the final model 

def conv_net(x,weights,biases,dropout):
    
    ## Convolutional 1
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,stride=2)
    ## Convolutional 2
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2,stride=2)
    ## Convolutional 3
    conv3 = conv2d(conv2,weights['wc3'],biases['bc3'])
    conv3 = maxpool2d(conv3,stride=2)
    ## Fully connected layer
        
    fc1 = tf.reshape(conv3,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    ## Apply Dropout 
    fc1 = tf.nn.dropout(fc1,dropout)
    fc2 = tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    ## Apply Dropout 
    fc2 = tf.nn.dropout(fc2,dropout)
    
    ## Output class prediction 
    
    out = tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    return out 

######################################################
## Define several Parameters for the Network and learning
####################################################### 

start_time = time.time()
learning_rate = 0.01
epochs = 200
batch_size = 128
num_batches = list1_shuf.shape[0]/128
input_height = 64 
input_width = 64
n_classes = 3
dropout = 0.5
display_step = 1
filter_height = 3
filter_width = 3
depth_in = 3
depth_out1 = 64
depth_out2 = 128
depth_out3 = 256


# input output definitition 
x = tf.placeholder(tf.float32,[None,input_height,input_width,depth_in])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

## Store the weights

weights = { 
    'wc1' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])),
    'wc2' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])),
    'wc3' : tf.Variable(tf.random_normal([filter_height,filter_width,depth_out2,depth_out3])),
    'wd1' : tf.Variable(tf.random_normal([(input_height/8)*(input_height/8)*256,512])),
    'wd2' : tf.Variable(tf.random_normal([512,512])),
    'out' : tf.Variable(tf.random_normal([512,n_classes]))

}

biases = { 
    'bc1' : tf.Variable(tf.random_normal([64])),
    'bc2' : tf.Variable(tf.random_normal([128])),
    'bc3' : tf.Variable(tf.random_normal([256])),
    'bd1' : tf.Variable(tf.random_normal([512])),
    'bd2' : tf.Variable(tf.random_normal([512])),
    'out' : tf.Variable(tf.random_normal([n_classes]))

}

# the model 

pred = conv_net(x,weights,biases,keep_prob)

# Define loss function and optimizer 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model 

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))



## initializing all variables

init = tf.global_variables_initializer()

## Launch the execution Graph
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(epochs):
        
        for j in range(num_batches):
            
            batch_x,batch_y = list1_shuf[i*(batch_size):(i+1)*(batch_size)],list2_shuf[i*(batch_size):(i+1)*(batch_size)]
            sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob: 1.})
            
        
        if epochs % display_step == 0:
            print("Epoch:", '%04d' % (i+1),
                  "cost=", "{:.9f}".format(loss),
                  "Training accuracy","{:.5f}".format(acc))
            
            
    print('Optimization Completed')
   
          
end_time = time.time()
print('Total processing time:',end_time - start_time)




# # Listing 3-8 Transfer Learning with Inception V3
# 

# # Listing 3-9 Transfer Learning with Pre-trained VGG16 
# 

import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
sys.path.append("/home/santanu/models/slim")
import cv2
from scipy.misc import imresize
from nets import vgg
from preprocessing import vgg_preprocessing
from mlxtend.preprocessing import shuffle_arrays_unison
checkpoints_dir = '/home/santanu/checkpoints'

slim = tf.contrib.slim

learning_rate = 0.01
batch_size = 32

cat_train = '/home/santanu/CatvsDog/train/cat/'
dog_train = '/home/santanu/CatvsDog/train/dog/'

all_images = os.listdir(cat_train) + os.listdir(dog_train)
train_images, validation_images = train_test_split(all_images, train_size=0.8, test_size=0.2)

MEAN_VALUE = np.array([103.939, 116.779, 123.68])

################################################
# Logic to read the Images and also do mean correction
################################################ 
def image_preprocess(img_path,width,height):
    img = cv2.imread(img_path)
    img = imresize(img,(width,height))
    img = img - MEAN_VALUE
    return(img)

################################################
# Create generator for Image batches so that only the processed 
# batch is in memory
################################################ 
def data_gen_small(images, batch_size, width,height):
        """
        data_dir: where the actual images are kept
        mask_dir: where the actual masks are kept
        images: the filenames of the images we want to generate batches from
        batch_size: self explanatory
        dims: the dimensions in which we want to rescale our images
        """
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                data_dir = ' '
                # images
                if images[i].split('.')[0] == 'cat':
                    labels.append(1)
                    data_dir = cat_train
                else:
                    if images[i].split('.')[0] == 'dog':
                        labels.append(0)
                        data_dir = dog_train
                #print 'data_dir',data_dir
                img_path = data_dir + images[i]
                array_img = image_preprocess(img_path,width,height)
                imgs.append(array_img)
                
            imgs = np.array(imgs)
            labels = np.array(labels)
            labels = np.reshape(labels,(batch_size,1))
            yield imgs,labels
            
#######################################################
## Defining the generators for training and validation batches
####################################################### 
train_gen = data_gen_small(train_images,batch_size,224,224)
val_gen = data_gen_small(validation_images,batch_size,224,224)


with tf.Graph().as_default():
    
    
    
    x = tf.placeholder(tf.float32,[None,224,224,3])
    y = tf.placeholder(tf.float32,[None,1])
    
    W1 =tf.Variable(tf.random_normal([4096,1],mean=0.0,stddev=0.02),name='W1')
    b1 = tf.Variable(tf.random_normal([1],mean=0.0,stddev=0.02),name='b1')
    

##############################################
## Load the VGG16 model from slim extract the 
##fully connected layer before the final output layer
###############################################    
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(x,
                               num_classes=1000,
                               is_training=False)
        fc_7 = end_points['vgg_16/fc7']
        fc_7 = tf.reshape(fc_7, [-1,W1.get_shape().as_list()[0]])
    
    logitx = tf.nn.bias_add(tf.matmul(fc_7,W1),b1)
    probx = tf.nn.sigmoid(logitx)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitx,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[W1,b1])
    
##############################################
## Load the VGG16 model from slim extract the fully connected layer
## before the final output layer
###############################################    
        
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # Load weights
        init_fn(sess)
        val_x,val_y = next(val_gen)
        for i in xrange(1):
            for j in xrange(50):
                batch_x,batch_y = next(train_gen)
                sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
                cost_train = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
                cost_val = sess.run(cost,feed_dict={x:val_x,y:val_y})
                prob_out = sess.run(probx,feed_dict={x:val_x,y:val_y})
                print "Training Cost",cost_train,"Validation Cost",cost_val
        out_val = (prob_out > 0.5)*1
        print 'accuracy', np.sum(out_val == val_y)*100/float(len(val_y))

        
       


class_dict ={1:"Cat",0:"Dog"}
out_val = (prob_out > 0.5)*1
print 'accuracy', np.sum(out_val == val_y)*100/float(len(val_y))
plt.imshow(val_x[0] + MEAN_VALUE)
print "Actual class:",class_dict[val_y[0][0]],"Predicted Class:",class_dict[out_val[0][0]]


class_dict ={1:"Cat",0:"Dog"}
out_val = (prob_out > 0.5)*1
print 'accuracy', np.sum(out_val == val_y)*100/float(len(val_y))
plt.imshow(val_x[2] + MEAN_VALUE)
print "Actual class:",class_dict[val_y[2][0]],"Predicted Class:",class_dict[out_val[2][0]]


# # Listing 5-1. Computation of Pi through Monte Carlo Sampling 
# 

import numpy as np
number_sample = 100000
inner_area,outer_area = 0,0

for i in range(number_sample):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    if (x**2 + y**2) < 1 :
        inner_area += 1
    outer_area += 1

print("The computed value of Pi:",4*(inner_area/float(outer_area))) 


# # Listing 5-2. Bivariate Gaussian distribution Sampling through Metropolis Algorithm 
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Now lets generate this with one of the Markov Chain Monte Carlo method called Metropolis Hastings algorithm
# Our assumed transition probabilities would follow normal distribution X2 ~ N(X1,Covariance= [[0.2 , 0],[0,0.2]])
import time
start_time = time.time()

# Set up constants and initial variable conditions
num_samples=100000
prob_density = 0

## Plan is to sample from a Bivariate Guassian Distribution with mean (0,0) and covariance of
## 0.7 between the two variables

mean = np.array([0,0])
cov = np.array([[1,0.7],[0.7,1]])
cov1 = np.matrix(cov)
mean1 = np.matrix(mean)
x_list,y_list = [],[]
accepted_samples_count = 0

## Normalizer of the Probility distibution
## This is not actually required since we are taking ratio of probabilities for inference

normalizer = np.sqrt( ((2*np.pi)**2)*np.linalg.det(cov))
## Start with initial Point (0,0)
x_initial, y_initial = 0,0
x1,y1 = x_initial, y_initial
for i in xrange(num_samples):
    
## Set up the Conditional Probability distribution taking the existing point
## as the mean and a small variance = 0.2 so that points near the existing point
## have a high chance of getting sampled.

    mean_trans = np.array([x1,y1])
    cov_trans = np.array([[0.2,0],[0,0.2]])
    x2,y2 = np.random.multivariate_normal(mean_trans,cov_trans).T
    X = np.array([x2,y2])
    X2 = np.matrix(X)
    X1 = np.matrix(mean_trans)

    ## Compute the probability density of te existing point and the new sampled
    ## point

    mahalnobis_dist2 = (X2 - mean1)*np.linalg.inv(cov)*(X2 - mean1).T
    prob_density2 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist2)
    mahalnobis_dist1 = (X1 - mean1)*np.linalg.inv(cov)*(X1 - mean1).T
    prob_density1 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist1)

    ## This is the heart of the algorithm. Comparing the ratio of Probability density of the new
    ## point and the existing point(acceptance_ratio) and selecting the new point if it is have more probability
    ## density. If it has less probability it is randomly selected with the probability of getting
    ## selected being proportional to the ratio of the acceptance ratio
    acceptance_ratio = prob_density2[0,0] / float(prob_density1[0,0])

    if (acceptance_ratio >= 1) | ((acceptance_ratio < 1) and (acceptance_ratio >= np.random.uniform(0,1)) ) :
        x_list.append(x2)
        y_list.append(y2)
        x1 = x2
        y1 = y2
        accepted_samples_count += 1

end_time = time.time()

print ('Time taken to sample ' + str(accepted_samples_count) + ' points ==> ' + str(end_time - start_time) + ' seconds' )
print 'Acceptance ratio ===> ' , accepted_samples_count/float(100000)
print "Mean of the Sampled Points"
print np.mean(x_list),np.mean(y_list)
print "Covariance matrix of the Sampled Points"
print np.cov(x_list,y_list) 
## Time to display the samples generated
plt.xlabel('X')
plt.ylabel('Y')
print "Scatter plot for the Sampled Points"
plt.scatter(x_list,y_list,color='black')


# # Listing 5-3a. Restricted Boltzmann Machine Implementation with MNIST dataset 
# 

##Import the Required libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

## Read the MNIST files
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

## Set up the parameters for training 


n_visible      = 784
n_hidden    = 500
display_step = 1
num_epochs = 200 
batch_size = 256 
lr         = tf.constant(0.001, tf.float32)

## Define the tensorflow variables for weights and biases as well as placeholder for input
x  = tf.placeholder(tf.float32, [None, n_visible], name="x") 
W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") 
b_h = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="b_h")) 
b_v = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="b_v")) 

## Converts the probability into discrete binary states i.e. 0 and 1 
def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
          
  
## Gibbs sampling step
def gibbs_step(x_k):
        h_k = sample(tf.sigmoid(tf.matmul(x_k, W) + b_h)) 
        x_k = sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_v))
        return x_k
## Run multiple gives Sampling step starting from an initital point     
def gibbs_sample(k,x_k):
    for i in range(k):
        x_out = gibbs_step(x_k) 
# Returns the gibbs sample after k iterations
    return x_out

# Constrastive Divergence algorithm
# 1. Through Gibbs sampling locate a new visible state x_sample based on the current visible state x    
# 2. Based on the new x sample a new h as h_sample    
x_s = gibbs_sample(2,x) 
h_s = sample(tf.sigmoid(tf.matmul(x_s, W) + b_h)) 

# Sample hidden states based given visible states
h = sample(tf.sigmoid(tf.matmul(x, W) + b_h)) 
# Sample visible states based given hidden states
x_ = sample(tf.sigmoid(tf.matmul(h, tf.transpose(W)) + b_v))

# The weight updated based on gradient descent 
size_batch = tf.cast(tf.shape(x)[0], tf.float32)
W_add  = tf.multiply(lr/size_batch, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_s), h_s)))
bv_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(x, x_s), 0, True))
bh_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(h, h_s), 0, True))
updt = [W.assign_add(W_add), b_v.assign_add(bv_add), b_h.assign_add(bh_add)]

# TensorFlow graph execution

with tf.Session() as sess:
    # Initialize the variables of the Model
    init = tf.global_variables_initializer()
    sess.run(init)
    
    total_batch = int(mnist.train.num_examples/batch_size)
    # Start the training 
    for epoch in range(num_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run the weight update 
            batch_xs = (batch_xs > 0)*1
            _ = sess.run([updt], feed_dict={x:batch_xs})
            
        # Display the running step 
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1))
                  
    print("RBM training Completed !")
    
    
    out = sess.run(h,feed_dict={x:(mnist.test.images[:20]> 0)*1})
    label = mnist.test.labels[:20]
    
    plt.figure(1)
    for k in range(20):
        plt.subplot(4, 5, k+1)
        image = (mnist.test.images[k]> 0)*1
        image = np.reshape(image,(28,28))
        plt.imshow(image,cmap='gray')
       
    plt.figure(2)
    
    for k in range(20):
        plt.subplot(4, 5, k+1)
        image = sess.run(x_,feed_dict={h:np.reshape(out[k],(-1,n_hidden))})
        image = np.reshape(image,(28,28))
        plt.imshow(image,cmap='gray')
        print(np.argmax(label[k]))
        
    W_out = sess.run(W)
    
    
    sess.close()


# # Listing 5-3b. Basic Implementation of Deep Belief Network
# 

##Import the Required libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

## Read the MNIST files
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

## Set up the parameters for training 


n_visible      = 784
n_hidden    = 500
display_step = 1
num_epochs = 200 
batch_size = 256 
lr         = tf.constant(0.001, tf.float32)
learning_rate_train = tf.constant(0.01, tf.float32)
n_classes = 10
training_iters = 200
## Define the tensorflow variables for weights and biases as well as placeholder for input
x  = tf.placeholder(tf.float32, [None, n_visible], name="x") 
y  = tf.placeholder(tf.float32, [None,10], name="y") 

W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") 
b_h = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="b_h")) 
b_v = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="b_v")) 
W_f = tf.Variable(tf.random_normal([n_hidden,n_classes], 0.01), name="W_f") 
b_f = tf.Variable(tf.zeros([1, n_classes],  tf.float32, name="b_f")) 
## Converts the probability into discrete binary states i.e. 0 and 1 
def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
          
  
## Gibbs sampling step
def gibbs_step(x_k):
        h_k = sample(tf.sigmoid(tf.matmul(x_k, W) + b_h)) 
        x_k = sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_v))
        return x_k
## Run multiple gives Sampling step starting from an initital point     
def gibbs_sample(k,x_k):
    for i in range(k):
        x_out = gibbs_step(x_k) 
# Returns the gibbs sample after k iterations
    return x_out

# Constrastive Divergence algorithm
# 1. Through Gibbs sampling locate a new visible state x_sample based on the current visible state x    
# 2. Based on the new x sample a new h as h_sample    
x_s = gibbs_sample(2,x) 
h_s = sample(tf.sigmoid(tf.matmul(x_s, W) + b_h)) 

# Sample hidden states based given visible states
h = sample(tf.sigmoid(tf.matmul(x, W) + b_h)) 
# Sample visible states based given hidden states
x_ = sample(tf.sigmoid(tf.matmul(h, tf.transpose(W)) + b_v))

# The weight updated based on gradient descent 
size_batch = tf.cast(tf.shape(x)[0], tf.float32)
W_add  = tf.multiply(lr/size_batch, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_s), h_s)))
bv_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(x, x_s), 0, True))
bh_add = tf.multiply(lr/size_batch, tf.reduce_sum(tf.subtract(h, h_s), 0, True))
updt = [W.assign_add(W_add), b_v.assign_add(bv_add), b_h.assign_add(bh_add)]
#--------------------------------------------------------------

## Ops for the Classification Network

#-------------------------------------------------------------- 
h_out = tf.sigmoid(tf.matmul(x, W) + b_h)
logits = tf.matmul(h_out,W_f) + b_f
prob = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_train).minimize(cost)
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#--------------------------------------------------------------                   
# TensorFlow graph execution of Unsupervised Pre-training with RBM
#--------------------------------------------------------------
with tf.Session() as sess:
    # Initialize the variables of the Model
    init = tf.global_variables_initializer()
    sess.run(init)
    
    total_batch = int(mnist.train.num_examples/batch_size)
    # Start the training 
    for epoch in range(num_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run the weight update 
            batch_xs = (batch_xs > 0)*1
            _ = sess.run([updt], feed_dict={x:batch_xs})
            
        # Display the running step 
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1))
                  
    print("RBM training Completed !")
    
    
    out = sess.run(h,feed_dict={x:(mnist.test.images[:20]> 0)*1})
    label = mnist.test.labels[:20]
    
    plt.figure(1)
    for k in range(20):
        plt.subplot(4, 5, k+1)
        image = (mnist.test.images[k]> 0)*1
        image = np.reshape(image,(28,28))
        plt.imshow(image,cmap='gray')
       
    plt.figure(2)
    
    for k in range(20):
        plt.subplot(4, 5, k+1)
        image = sess.run(x_,feed_dict={h:np.reshape(out[k],(-1,n_hidden))})
        image = np.reshape(image,(28,28))
        plt.imshow(image,cmap='gray')
        print(np.argmax(label[k]))
#-----------------------------------------
## Invoke the classification now
#-----------------------------------------
    
    for i in xrange(training_iters):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if i % 10 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print "Iter " + str(i) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc)
        
    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    print "Testing Accuracy:",         sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256]})

    sess.close()


# # Listing 5-4 Sparse Auto Encoder Implementation in TensorFlow 
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 1000
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 32*32 # 1st layer num features
#n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1,n_input])),
}    
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h1'])),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    return layer_1

def logfunc(x, x2):
    return tf.multiply( x, tf.log(tf.div(x,x2)))

def KL_Div(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = logfunc(rho,rho_hat) + logfunc(invrho, invrhohat)
    return logrho


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
rho_hat = tf.reduce_mean(encoder_op,1)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost_m = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost_sparse = 0.01*tf.reduce_sum(KL_Div(0.2,rho_hat))
#cost_reg = 0.0001* (tf.nn.l2_loss(weights['decoder_h1']) + tf.nn.l2_loss(weights['encoder_h1']))
cost_reg = 0.0001*tf.nn.l2_loss(weights['encoder_h1'])
cost = tf.add(cost_reg,tf.add(cost_m,cost_sparse))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:10]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)),cmap='gray')
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)),cmap='gray')
    #f.show()
    #plt.draw()
    #plt.waitforbuttonpress()
    #dec = sess.run(weights['decoder_h1'])
    enc = sess.run(weights['encoder_h1'])
end_time = time.time()
print('elapsed time:',end_time - start_time)




# # Plot of the Sparse feature weights learnt
# 

img_coll = []
for i in xrange(1024):
    img = np.array(enc.T[i,:])
    img = np.reshape(img,(28,28))
    img_coll.append(img)

img_coll = np.array(img_coll)
f, a = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
        a[0][i].imshow(np.reshape(img_coll[i], (28, 28)),cmap='gray')
        a[1][i].imshow(np.reshape(img_coll[10 + i], (28, 28)),cmap='gray')
        a[2][i].imshow(np.reshape(img_coll[20 + i], (28, 28)),cmap='gray')
        a[3][i].imshow(np.reshape(img_coll[30 + i], (28, 28)),cmap='gray')
        a[4][i].imshow(np.reshape(img_coll[40 + i], (28, 28)),cmap='gray')
        a[5][i].imshow(np.reshape(img_coll[50 + i], (28, 28)),cmap='gray')
        a[6][i].imshow(np.reshape(img_coll[60 + i], (28, 28)),cmap='gray')
        a[7][i].imshow(np.reshape(img_coll[70 + i], (28, 28)),cmap='gray')
        a[8][i].imshow(np.reshape(img_coll[80 + i], (28, 28)),cmap='gray')
        a[9][i].imshow(np.reshape(img_coll[90 + i], (28, 28)),cmap='gray')
        


# # Listing 5-5 Denoising Auto Encoder Implementation in TensorFlow 
# 

# # Remove the Guassian Noise 
# 

# Import the required library

import tensorflow.contrib.layers as lays
import numpy as np
from skimage import transform
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 

def autoencoder(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

# Introduce Gaussian Noise
def noisy(image):
    row,col= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy
    

# Introduce Salt and Pepper Noise
def s_p(image):
    row,col = image.shape
    s_vs_p = 0.5
    amount = 0.05
    out = np.copy(image)
      # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

      # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
ae_inputs_noise = tf.placeholder(tf.float32, (None, 32, 32, 1)) 
ae_outputs = autoencoder(ae_inputs_noise)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate

# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            image_arr = []
            for i in xrange(len(batch_img)):
                img = batch_img[i,:,:,0]
                img = noisy(img)
                image_arr.append(img)
            image_arr = np.array(image_arr)
            image_arr = image_arr.reshape(-1,32,32,1)
            batch_img = image_arr
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img,ae_inputs_noise:image_arr})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))

    # test the trained network
    batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = resize_batch(batch_img)
    image_arr = []
    
    for i in xrange(50):
        img = batch_img[i,:,:,0]
        img = noisy(img)
        image_arr.append(img)
    image_arr = np.array(image_arr)
    image_arr = image_arr.reshape(-1,32,32,1)
    batch_img = image_arr
            
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs_noise: batch_img})[0]

    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images with Gaussian Noise')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    plt.show()


# # Remove the Salt and Pepper Noise
# 

batch_size = 1000  # Number of samples in each batch
epoch_num = 10     # Number of epochs to train the network
lr = 0.001        # Learning rate

# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            image_arr = []
            for i in xrange(len(batch_img)):
                img = batch_img[i,:,:,0]
                img = s_p(img)
                image_arr.append(img)
            image_arr = np.array(image_arr)
            image_arr = image_arr.reshape(-1,32,32,1)
            #batch_img = image_arr
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs_noise:image_arr,ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))

    # test the trained network
    batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = resize_batch(batch_img)
    image_arr = []
    
    for i in xrange(50):
        img = batch_img[i,:,:,0]
        img = s_p(img)
        image_arr.append(img)
    image_arr = np.array(image_arr)
    image_arr = image_arr.reshape(-1,32,32,1)
    #batch_img = image_arr
            
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs_noise: image_arr})[0]

    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Noisy Images with Salt and Pepper Noise')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(image_arr[i, ..., 0], cmap='gray')
    plt.show()


