# # STYLE TRANSFER
# 

from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras import applications
from keras import backend as K


img_width = 300
img_height = 300
content_weight = 0.025  # the weight given to content loss
style_weight = 1.0    # the weight given to style loss
tv_weight = 1.0     # the weight given to total variance loss


# load image and convert to a format acceptable by vgg16
# vgg expect BGR instead of RGB and means to be deducted from each channel

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width,img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = applications.vgg16.preprocess_input(img)
    return img


# vgg expects the images in a certain format, so after converting
# to that format, you have to convert them back to display and save

def deprocess_image(x):
    x = x.reshape((img_width,img_height, 3))
    
    # remove zero center by mean pixel
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    
    #BGR -> RGB
    x = x[:,:,::-1]
    x = np.clip(x,0,255).astype('uint8')
    return x


# this is just to test and see the shape this function returns

# place a content.jpg and style.jpg file in your folder

preprocess_image('content.jpg').shape


# get tensor representations of our images
base_image = K.variable(preprocess_image('content.jpg'))
style_reference_image = K.variable(preprocess_image('style.jpg'))


base_image


combination_image = K.placeholder((1,img_width,img_height,3))


combination_image


# combine the 3 images into a single tensor
input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)


input_tensor.get_shape()


# build the vgg16 network with 3 images as input
# load pretrained image net weights

model = applications.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)


# get outputs of each layer from the model

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


outputs_dict # lists output of all the layers in the vgg16 network


# define util functions for neural style loss

# gram matrix of image tensor
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels= 3
    size = img_width*img_height
    return K.sum(K.square(S-C))/(4.*(channels**2)*(size**2))


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def total_variation_loss(x):
    a = K.square(x[:,:img_width-1,:img_height-1,:]- x[:,1:,:img_width-1,:])
    b = K.square(x[:,:img_width-1,:img_height-1,:]- x[:,:img_height-1,1:,:])
    return K.sum(K.pow(a+b, 1.25))
    


# combine loss into a single scalar

loss = K.variable(0.)
layer_features = outputs_dict['block4_conv2']

base_image_features = layer_features[0, :,:,:]
combination_features = layer_features[2,:,:,:]

# add content loss
loss += content_weight * content_loss(base_image_features, combination_features)


feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']


# add style loss

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight/ len(feature_layers))*sl

# add total variance loss
loss += tv_weight * total_variation_loss(combination_image)


# get gradients of generated image wrt loss

grads = K.gradients(loss, combination_image)


# create array of loss, followed by gradients

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)


f_outputs = K.function([combination_image], outputs)


# helper function to find loss and gradients

def eval_loss_and_grads(x):

    x = x.reshape((1,img_width, img_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# helper class to display information at 
# different stages while the model is training

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()


# randomly initialize the combination image to a random
# set of pixels

x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# display initial random image
# this image will become the final combined image after optimization
display_x = np.reshape(x, (img_width, img_height, 3))
plt.imshow(display_x)


for i in range(10):
    print("Start of iteration", i)
    start_time = time.time()
    
    # use l-bfgs algorithm which is more efficient than regular gradient descent
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                    fprime=evaluator.grads, maxfun=20)
    print('Current loss value', min_val)
    #save generated image
    img = deprocess_image(x.copy())
    fname = 'result_at_iteration%d.png'%i
    imsave(fname, img)
    end_time = time.time()
    print("Image saved as", fname)
    print("End of iteration %d in %d secs" % (i,end_time - start_time))





import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


vggmodel = applications.VGG16(include_top=False, weights='imagenet')


datagen = ImageDataGenerator(rescale=1./255)


# you are passing your images through the pretrained vgg network
# all the way up until the fully connected layers

generator = datagen.flow_from_directory('data/train',
                                       target_size=(150, 150),
                                       batch_size=16,
                                       class_mode=None,
                                       shuffle=False)

bottleneck_features_train = vggmodel.predict_generator(generator, 2000/16)

np.save(open('bottleneck_features_train.npy','wb'), bottleneck_features_train)


# reload the file that has been pre created and predicted on

# preload the training data and create the true output labels

train_data = np.load(open('bottleneck_features_train.npy','rb'))

train_labels = np.array([0]*1000 + [1]*1000)


train_data.shape


# test data to check your accuracy on

validation_data = np.load(open('bottleneck_features_validation.npy','rb'))

validation_labels = np.array([0]*400 + [1]*400)


# create your own fully connected network to make predictions on dog vs cat

model = Sequential()

model.add(Flatten(input_shape=(4,4,512)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# define the loss function and the gradient descent function

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# train on training data

model.fit(train_data, train_labels,
         epochs=30,
         batch_size=16,
         validation_data=(validation_data, validation_labels))








# # predict on your own image
# 

import matplotlib.pyplot as plt
from scipy.misc import imresize

get_ipython().magic('matplotlib inline')


# read in your image
img = plt.imread('dog.jpg')


# show your image
plt.imshow(img)


# resize my image
img = imresize(img, (150,150,3))


plt.imshow(img)


img.shape


# tells you there is one image in this set
img = np.expand_dims(img, axis=0)


img.shape


convolved_img = vggmodel.predict(img,1)


convolved_img.shape


new_dn_model = Sequential()

new_dn_model.add(Flatten(input_shape=(4,4,512)))
new_dn_model.add(Dense(256, activation='relu'))
new_dn_model.add(Dropout(0.5))
new_dn_model.add(Dense(1, activation='sigmoid'))

new_dn_model.load_weights('bottleneck_fc_model.h5')


new_dn_model.predict_classes(convolved_img)





# # Linear Regression Workshop
# 

import numpy as np


import tensorflow as tf


# Step 1- import data

points = np.genfromtxt('data.csv', delimiter=',')


# separate out data
x = points[:,0]
y = points[:,1]


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


plt.scatter(x,y)
plt.show()


# randomly initialize weights and biases

slope = tf.Variable([1], dtype=tf.float32)
intercept = tf.Variable([0], dtype=tf.float32)


# define hyper parameter
rate = 0.000001
num_iterations = 1000


# define data placeholders

x_val = tf.placeholder(tf.float32)
y_val = tf.placeholder(tf.float32)


# define the model

predicted_y = slope*x_val + intercept


# find your error

error = tf.reduce_sum(tf.square(predicted_y - y_val))


# find grdient and update weights

optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)

# minimize the error
train = optimizer.minimize(error)


# initialize tf session

sess = tf.Session()

# initialize only the variables

init = tf.global_variables_initializer()

sess.run(init)


# train your model for certain number of iterations

for p in range(num_iterations):
    
    sess.run(train, feed_dict={x_val:x, y_val:y})


sess.run(slope)


sess.run(intercept)


def predict(m,b,x_vals):
    
    ys = []
    
    for i in range(len(x_vals)):
        
        predicted_y = m*x_vals[i]+b
        ys.append(predicted_y)
        
    return ys


final_predicted_slope = sess.run(slope)
final_predicted_intercept = sess.run(intercept)


final_predicted_intercept


plt.scatter(x,y)
plt.plot(x, predict(final_predicted_slope, final_predicted_intercept,x))
plt.show()


predict(final_predicted_slope, final_predicted_intercept, [65])





