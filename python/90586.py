get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import joblib
from sklearn.decomposition import PCA


# # Train PCA
# 
# Fit a PCA model on a random sample of 100k (should capture the distribution of the variation pretty well)
# 
# #### Notes:
# 
# * Scale pixel to [0,1] (not strictly necessary)
# * Don't use PCA whitening
# * Use 1000 components
# 

def train_pca(filepath, n_components, N=100000):
    f = h5py.File(filepath, 'r')
    X_dataset, y_dataset = list(f.items())
    X_dataset, y_dataset = np.moveaxis(X_dataset[1][:], 1, 3), y_dataset[1][:]
    
    np.random.seed(0)
    index = np.random.choice(len(X_dataset), N, replace=False)
    X_dataset, y_dataset = X_dataset[index], y_dataset[index]
    
    print('Sample Size: %d' % N)
    print('Data Types X=%s, y=%s' % (X_dataset.dtype, y_dataset.dtype))
    print('Shape X=%s, y=%s' % (X_dataset.shape, y_dataset.shape))
    
    pca = PCA(n_components=n_components, whiten=False)
    
    start = time.time()
   
    # Reshape to 1-D array
    shape = X_dataset.shape
    X_reshape = X_dataset.reshape((shape[0], -1)) / 255.
    X_reshape.shape
   
    pca.fit(X_reshape)
    X_pca = pca.transform(X_reshape)
    
    end = time.time()
    elapsed = end - start
    print('Fit time elapsed: {}'.format(elapsed))
    
    return pca
   
n_components = 500
filepath = '../../data/svhn/svhn_format_2.hdf5'
pca = train_pca(filepath, n_components=n_components)


model_path = 'saved_models/pca_%d.pkl' % n_components
print('Dumping pca (%d) model to: %s' % (n_components, model_path)) 
joblib.dump(pca, model_path)


# # Test PCA
# 
# Display original and reconstructed images to ensure that PCA worked correctly.
# 

# Test PCA model
pca = joblib.load(model_path)
pca


def display_grid(dataset, digit_size=32, grid_size=5, seed=None):
    # Display some digits to figure out what's going on
    figure = np.zeros((digit_size * grid_size, digit_size * grid_size, 3))
   
    if seed is not None:
        np.random.seed(seed)
    for i in range(grid_size):
        for j in range(grid_size):
            digit = dataset[np.random.randint(len(dataset))]
            d_x, d_y = i * digit_size, j * digit_size
            figure[d_x:d_x + digit_size, d_y:d_y + digit_size, :] = digit.astype(int)
            
    plt.figure(figsize=(5, 5))
    plt.imshow(figure)
    plt.show()


f = h5py.File(filepath, 'r')
sample_size = 1000
X_dataset, y_dataset = list(f.items())
X_dataset, y_dataset = np.moveaxis(X_dataset[1][:], 1, 3), y_dataset[1][:]

print("Originals")
index = np.random.choice(len(X_dataset), sample_size, replace=False)
X_dataset, y_dataset = X_dataset[index], y_dataset[index]
display_grid(X_dataset, seed=0)

print("Reconstructed PCA")
X_reshape = X_dataset.reshape((X_dataset.shape[0], -1)) / 255.
X_pca = pca.transform(X_reshape)
X_recon = np.clip(pca.inverse_transform(X_pca), 0.0, 0.999)
X_recon_reshape = X_recon.reshape(X_dataset.shape) * 255.
display_grid(X_recon_reshape, seed=0)


# Reconstructed PCA images looks pretty similar.
# 




get_ipython().magic('matplotlib inline')
import numpy as np
import time
import keras
import pandas as pd
import math
import joblib
import matplotlib.pyplot as plt

from random import randint

from IPython.display import display

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Activation, Dropout, Conv2D, Conv2DTranspose
from keras.engine import Layer
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model, Sequential
from keras import metrics
from keras import backend as K
from keras_tqdm import TQDMNotebookCallback
from keras.datasets import mnist
from tqdm import tnrange, tqdm_notebook


img_rows, img_cols, img_chns = 28, 28, 1
original_img_size = (img_rows, img_cols, img_chns)
batch_size = 500
epochs = 1000
hidden_units = 8000
hidden_layers = 2
learning_rate = 0.0005
dropout = 0.1


# # Binarized MNIST
# 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.round(X_train.reshape(X_train.shape[0], img_rows * img_cols) / 255.)
X_test = np.round(X_test.reshape(X_test.shape[0], img_rows * img_cols) / 255.)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


def display_digits(X, digit_size=28, n=10):
    figure = np.zeros((digit_size * n, digit_size * n))
    
    for i in range(n):
        for j in range(n):
            index = np.random.randint(0, X.shape[0])
            digit = X[index].reshape(digit_size, digit_size)
            
            x = i * digit_size
            y = j * digit_size
            figure[x:x + digit_size, y:y + digit_size] = digit
    
    plt.figure(figsize=(n, n))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    
display_digits(X_train)


# # Custom Layer for MADE masking
# 

# from keras.layers import activations
from keras.layers import initializers
from keras.layers import activations
from keras.layers import regularizers
from keras.layers import constraints
from keras.engine import InputSpec

class MaskingDense(Layer):
    """ Just copied code from keras Dense layer and added masking """

    def __init__(self, units, out_units,
                 hidden_layers=1,
                 dropout_rate=0.0,
                 random_input_order=False,
                 activation='relu',
                 out_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MaskingDense, self).__init__(**kwargs)
        
        self.input_sel = None
        self.random_input_order = random_input_order
        self.rate = min(1., max(0., dropout_rate))
        self.kernel_sels = []
        self.units = units
        self.out_units = out_units
        self.hidden_layers = hidden_layers
        self.activation = activations.get(activation)
        self.out_activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
    def dropout_wrapper(self, inputs, training):
        if 0. < self.rate < 1.:
            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape=None, seed=None)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        
        return inputs
        
    def build_layer_weights(self, input_dim, units, use_bias=True):
        kernel = self.add_weight(shape=(input_dim, units),
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
     
        if use_bias:
            bias = self.add_weight(shape=(units,),
                                   initializer=self.bias_initializer,
                                   name='bias',
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        else:
            bias = None
        
        return kernel, bias
    
    def build_mask(self, shape, prev_sel, is_output):
        if is_output:
            input_sel = self.input_sel
        else:
            # Disallow D-1 because it would violate auto-regressive property
            # Disallow 0 because it would just createa a constant node
            # Disallow unconnected units by sampling min from previous layer
            input_sel = [randint(np.min(prev_sel), shape[-1] - 2) for i in range(shape[-1])]
            
        def vals():
            for x in range(shape[-2]):
                for y in range(shape[-1]):
                    if is_output:
                        yield 1 if prev_sel[x] < input_sel[y] else 0
                    else:
                        yield 1 if prev_sel[x] <= input_sel[y] else 0
        
        return K.constant(list(vals()), dtype='float32', shape=shape), input_sel
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
           
        self.kernels, self.biases = [], []
        self.kernel_masks, self.kernel_sels = [], []
        shape = (input_shape[-1], self.units)
       
        self.input_sel = np.arange(input_shape[-1])
        if self.random_input_order:
            np.random.shuffle(self.input_sel)
        prev_sel = self.input_sel
        for x in range(self.hidden_layers):
            # Hidden layer
            kernel, bias = self.build_layer_weights(*shape)
            self.kernels.append(kernel)
            self.biases.append(bias)
            
            # Hidden layer mask
            kernel_mask, kernel_sel = self.build_mask(shape, prev_sel, is_output=False)
            self.kernel_masks.append(kernel_mask)
            self.kernel_sels.append(kernel_sel)
        
            prev_sel = kernel_sel
            shape = (self.units, self.units)
            
        # Direct connection between input/output
        direct_shape = (input_shape[-1], self.out_units)
        self.direct_kernel, _ = self.build_layer_weights(*direct_shape, use_bias=False)
        self.direct_kernel_mask, self.direct_sel = self.build_mask(direct_shape, self.input_sel, is_output=True)
        
        # Output layer
        out_shape = (self.units, self.out_units)
        self.out_kernel, self.out_bias = self.build_layer_weights(*out_shape)
        self.out_kernel_mask, self.out_sel = self.build_mask(out_shape, prev_sel, is_output=True)
        
        self.built = True

    def call(self, inputs, training=None):
        # Hidden layer + mask
        output = inputs
        for i in range(self.hidden_layers):
            weight = self.kernels[i] * self.kernel_masks[i]
            output = K.dot(output, weight)
            output = K.bias_add(output, self.biases[i])
            output = self.activation(output)
            output = self.dropout_wrapper(output, training)
       
        # Direct connection
        direct = K.dot(inputs, self.direct_kernel * self.direct_kernel_mask)
        direct = self.dropout_wrapper(direct, training)
        
        # out_act(bias + (V dot M_v)h(x) + (A dot M_v)x) 
        output = K.dot(output, self.out_kernel * self.out_kernel_mask)
        output = output + direct
        output = K.bias_add(output, self.out_bias)
        output = self.out_activation(output)
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_units)


def logx_loss(x, x_decoded_mean):
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean)
    return xent_loss


K.set_learning_phase(1)
        
main_input = Input(shape=(img_rows * img_cols,), name='main_input')
mask_1 = MaskingDense(hidden_units, img_rows * img_cols, 
                      hidden_layers=hidden_layers,
                      dropout_rate=dropout,
                      random_input_order=False)(main_input)

model = Model(inputs=main_input, outputs=mask_1)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss=logx_loss)
model.summary()


start = time.time()

early_stopping = keras.callbacks.EarlyStopping('val_loss', min_delta=0.1, patience=50)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=0.001 * learning_rate)

history = model.fit(
    X_train, X_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[TQDMNotebookCallback(), early_stopping, reduce_lr],
    validation_data=(X_test, X_test),
    verbose=0
)

K.set_learning_phase(0)

done = time.time()
elapsed = done - start
print("Elapsed: ", elapsed)


df = pd.DataFrame(history.history)
display(df.describe(percentiles=[0.25 * i for i in range(4)] + [0.95, 0.99]))
df.plot(figsize=(8, 6))


def gen_image(model, num_samples=10):
    x_sample = np.random.rand(num_samples, img_rows * img_cols)
    
    # Iteratively generate each conditional pixel P(x_i | x_{1,..,i-1})
    for i in range(0, img_rows * img_cols):
        x_out = model.predict(x_sample)
            
        p = np.random.rand(num_samples)
        index = model.layers[-1].input_sel[i]
        x_sample[:, index] = (x_out[:, index] > p).astype(float)
        
    return x_sample


start = time.time()

K.set_learning_phase(0)
x_sample = gen_image(model, num_samples=100)

display_digits(x_sample, n=7)

done = time.time()
elapsed = done - start
print("Elapsed: ", elapsed)





