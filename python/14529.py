# This code is for:
# 
# J. P. Cohen, H. Z. Lo, and Y. Bengio, “Count-ception: Counting by Fully Convolutional Redundant Counting,” 2017.
# https://arxiv.org/abs/1703.08710
# 
# 
# Here is a video of the learning in progress:
# [![](http://img.youtube.com/vi/ej5bj0mlQq8/0.jpg)](https://www.youtube.com/watch?v=ej5bj0mlQq8)
# 
# The cell dataset used in this work is available from [VGG](http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip) and [Academic Torrents](http://academictorrents.com/details/b32305598175bb8e03c5f350e962d772a910641c)
# 




import sys,os,time,random
import numpy as np
import matplotlib
matplotlib.use('Agg');
import matplotlib.pyplot as plt
plt.set_cmap('jet');

import theano
import theano.tensor as T 
import lasagne

import skimage
from skimage.io import imread, imsave
import pickle
import scipy

from os import walk
print "theano",theano.version.full_version
print "lasagne",lasagne.__version__


get_ipython().magic('matplotlib inline')


if len(sys.argv) == 3 or sys.argv[0] == "jupyter": #on jupyter
    sys.argv = ['jupyter', 'jupyter','0', '32', '1', '0.005', 'sq','1']

print sys.argv;

seed = int(sys.argv[2])
print "seed",seed           #### Random seed for shuffling data and network weights
nsamples = int(sys.argv[3])
print "nsamples",nsamples   #### Number of samples (N) in train and valid
stride = int(sys.argv[4])
print "stride",stride       #### The stride at the initial layer

lr_param = float(sys.argv[5])
print "lr_param",lr_param   #### This will set the learning rate 

kern = sys.argv[6]
print "kern",kern           #### This can be gaus or sq
cov = int(sys.argv[7])
print "cov",cov             #### This is the covariance when kern=gaus


scale = 1
patch_size = 32
framesize = 256
noutputs = 1


paramfilename = str(scale) + "-" + str(patch_size) + "-cell-" + kern + str(cov) + "_cell_data.p"
datasetfilename = str(scale) + "-" + str(patch_size) + "-" + str(framesize) + "-" + kern + str(stride) + "-cell-" + str(cov) + "-dataset.p"
print paramfilename
print datasetfilename


random.seed(seed)
np.random.seed(seed)
lasagne.random.set_rng(np.random.RandomState(seed))


from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, ConcatLayer, Conv2DLayer

input_var = T.tensor4('inputs')
input_var_ex = T.ivector('input_var_ex')

def ConvFactory(data, num_filter, filter_size, stride=1, pad=0, nonlinearity=lasagne.nonlinearities.leaky_rectify):
    data = lasagne.layers.batch_norm(Conv2DLayer(
        data, num_filters=num_filter,
        filter_size=filter_size,
        stride=stride, pad=pad,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(gain='relu')))
    return data

def SimpleFactory(data, ch_1x1, ch_3x3):
    conv1x1 = ConvFactory(data=data, filter_size=1, pad=0, num_filter=ch_1x1)
    conv3x3 = ConvFactory(data=data, filter_size=3, pad=1, num_filter=ch_3x3) 
    concat = ConcatLayer([conv1x1, conv3x3])
    return concat


input_shape = (None, 1, framesize, framesize)
img = InputLayer(shape=input_shape, input_var=input_var[input_var_ex])
net = img


net = ConvFactory(net, filter_size=3, num_filter=64, pad=patch_size)
print net.output_shape
net = SimpleFactory(net, 16, 16)
print net.output_shape
net = SimpleFactory(net, 16, 32)
print net.output_shape
net = ConvFactory(net, filter_size=14, num_filter=16) 
print net.output_shape
net = SimpleFactory(net, 112, 48)
print net.output_shape
net = SimpleFactory(net, 64, 32)
print net.output_shape
net = SimpleFactory(net, 40, 40)
print net.output_shape
net = SimpleFactory(net, 32, 96)
print net.output_shape
net = ConvFactory(net, filter_size=18, num_filter=32) 
print net.output_shape
net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
print net.output_shape
net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
print net.output_shape
net = ConvFactory(net, filter_size=1, num_filter=1, stride=stride)
print net.output_shape


output_shape = lasagne.layers.get_output_shape(net)
real_input_shape = (None, input_shape[1], input_shape[2]+2*patch_size, input_shape[3]+2*patch_size)
print "real_input_shape:",real_input_shape,"-> output_shape:",output_shape





print "network output size should be",(input_shape[2]+2*patch_size)-(patch_size)





if (kern == "sq"):
    ef = ((patch_size/stride)**2.0)
elif (kern == "gaus"):
    ef = 1.0
print "ef", ef

prediction = lasagne.layers.get_output(net, deterministic=True)
prediction_count = (prediction/ef).sum(axis=(2,3))

classify = theano.function([input_var, input_var_ex], prediction)


train_start_time = time.time()
print classify(np.zeros((1,1,framesize,framesize), dtype=theano.config.floatX), [0]).shape
print time.time() - train_start_time, "sec"

train_start_time = time.time()
print classify(np.zeros((1,1,framesize,framesize), dtype=theano.config.floatX), [0]).shape
print time.time() - train_start_time, "sec"














def genGausImage(framesize, mx, my, cov=1):
    x, y = np.mgrid[0:framesize, 0:framesize]
    pos = np.dstack((x, y))
    mean = [mx, my]
    cov = [[cov, 0], [0, cov]]
    rv = scipy.stats.multivariate_normal(mean, cov).pdf(pos)
    return rv/rv.sum()

def getDensity(width, markers):
    gaus_img = np.zeros((width,width))
    for k in range(width):
        for l in range(width):
            if (markers[k,l] > 0.5):
                gaus_img += genGausImage(len(markers),k-patch_size/2,l-patch_size/2,cov)
    return gaus_img

def getMarkersCells(labelPath):        
    lab = imread(labelPath)[:,:,0]/255
    return np.pad(lab,patch_size, "constant")

def getCellCountCells(markers, (x,y,h,w), scale):
    types = [0] * noutputs
    types[0] = markers[y:y+w,x:x+h].sum()
    return types

def getLabelsCells(img, labelPath, base_x, base_y, stride):
    
    width = ((img.shape[0])/stride)
    print "label size: ", width
    markers = getMarkersCells(labelPath)
    labels = np.zeros((noutputs, width, width))
    
    
    if (kern == "sq"):
        for x in range(0,width):
            for y in range(0,width):

                count = getCellCountCells(markers,(base_x + x*stride,base_y + y*stride,patch_size,patch_size),scale)  
                for i in range(0,noutputs):
                    labels[i][y][x] = count[i]
    
    elif (kern == "gaus"):
        for i in range(0,noutputs):
            labels[i] = getDensity(width, markers[base_y:base_y+width,base_x:base_x+width])
    

    count_total = getCellCountCells(markers,(base_x,base_y,framesize+patch_size,framesize+patch_size),scale)
    return labels, count_total

def getTrainingExampleCells(img_raw, labelPath, base_x,  base_y, stride):
    
    img = img_raw[base_y:base_y+framesize,base_x:base_x+framesize]
    img_pad = np.pad(img,(patch_size)/2, "constant")
    labels, count  = getLabelsCells(img_pad, labelPath, base_x, base_y, stride)
    return img, labels, count





# ## code to debug data generation
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (18, 9)
# imgPath,labelPath,x,y = imgs[0][0],imgs[0][1], 0, 0

# print imgPath, labelPath
# # img_raw = imread(imgPath)[1]


# im = imread(imgPath)
# img_raw_raw = im.mean(axis=(2)) #grayscale

# img_raw = scipy.misc.imresize(img_raw_raw, (img_raw_raw.shape[0]/scale,img_raw_raw.shape[1]/scale))
# print img_raw_raw.shape," ->>>>", img_raw.shape

    
# #img_raw = scipy.misc.imresize(img_raw_raw, (img_raw_raw.shape[0]/scale,img_raw_raw.shape[1]/scale))
# print "img_raw",img_raw.shape
# img, lab, count = getTrainingExampleCells(img_raw, labelPath, x, y, stride)
# print "count", count

# markers = markers = getMarkersCells(labelPath)
# count = getCellCountCells(markers, (0,0,framesize,framesize), scale)
# print "count", count

# pcount = classify([[img]], [0])[0]

# lab_est = [(l.sum()/ef).astype(np.int) for l in lab]
# pred_est = [(l.sum()/ef).astype(np.int) for l in pcount]

# print "label est ",lab_est," --> predicted est ",pred_est

# fig = plt.Figure(figsize=(18, 9), dpi=160)
# gcf = plt.gcf()
# gcf.set_size_inches(18, 15)
# fig.set_canvas(gcf.canvas)

# ax2 = plt.subplot2grid((2,4), (0, 0), colspan=2)
# ax3 = plt.subplot2grid((2,4), (0, 2), colspan=3)
# ax4 = plt.subplot2grid((2,4), (1, 2), colspan=3)
# ax5 = plt.subplot2grid((2,4), (1, 0), rowspan=1)
# ax6 = plt.subplot2grid((2,4), (1, 1), rowspan=1)

# ax2.set_title("Input Image")
# ax2.imshow(img, interpolation='none', cmap='Greys_r')
# ax3.set_title("Regression target, {}x{} sliding window.".format(patch_size, patch_size))
# ax3.imshow(np.concatenate((lab),axis=1), interpolation='none')
# ax4.set_title("Predicted counts")
# ax4.imshow(np.concatenate((pcount),axis=1), interpolation='none')

# ax5.set_title("Real " + str(lab_est))
# ax5.set_ylim((0, np.max(lab_est)*2))
# ax5.set_xticks(np.arange(0, noutputs, 1.0))
# ax5.bar(range(noutputs),lab_est, align='center')
# ax6.set_title("Pred " + str(pred_est))
# ax6.set_ylim((0, np.max(lab_est)*2))
# ax6.set_xticks(np.arange(0, noutputs, 1.0))
# ax6.bar(range(noutputs),pred_est, align='center')

# #plt.imshow(img, interpolation='none', cmap='Greys_r')

# #plt.imshow(np.concatenate((lab),axis=1), interpolation='none')


# #fig.savefig('images-cell/image-' + str(i) + "-" + name + '.png')








import glob
imgs = []
for filename in glob.iglob('cells/*cell.png'):
    xml = filename.split("cell.png")[0] + "dots.png"
    imgs.append([filename,xml])


if len(imgs) == 0:
    print "Error loading data"


for path in imgs: 
    if (not os.path.isfile(path[0])):
        print path, "bad", path[0]
    if (not os.path.isfile(path[1])):
        print path, "bad", path[1]


if (os.path.isfile(datasetfilename)):
    print "reading", datasetfilename
    dataset = pickle.load(open(datasetfilename, "rb" ))
else:
    dataset = []
    print len(imgs)
    for path in imgs: 

        imgPath = path[0]
        print imgPath

        im = imread(imgPath)
        img_raw_raw = im.mean(axis=(2)) #grayscale

        img_raw = scipy.misc.imresize(img_raw_raw, (img_raw_raw.shape[0]/scale,img_raw_raw.shape[1]/scale))
        print img_raw_raw.shape," ->>>>", img_raw.shape

        labelPath = path[1]
        for base_x in range(0,img_raw.shape[0],framesize):
            for base_y in range(0,img_raw.shape[1],framesize):
                img, lab, count = getTrainingExampleCells(img_raw, labelPath, base_y, base_x, stride)
                
                lab_est = [(l.sum()/ef).astype(np.int) for l in lab]
                
                assert np.allclose(count,lab_est, 1)
                
                dataset.append((img,lab,count))
                print "img shape", img.shape, "label shape", lab.shape, "count ", count, "lab_est", lab_est
                sys.stdout.flush()
                    
    print "writing", datasetfilename
    out = open(datasetfilename, "wb",0)
    pickle.dump(dataset, out)
    out.close()
print "DONE"


# %matplotlib inline
# plt.rcParams['figure.figsize'] = (18, 9)
# plt.imshow(lab[0])


np_dataset = np.asarray(dataset)

#random.shuffle(np_dataset)
np.random.shuffle(np_dataset)

np_dataset = np.rollaxis(np_dataset,1,0)
np_dataset_x = np.asarray([[n] for n in np_dataset[0]],dtype=theano.config.floatX)
np_dataset_y = np.asarray([n for n in np_dataset[1]],dtype=theano.config.floatX)
np_dataset_c = np.asarray([n for n in np_dataset[2]],dtype=theano.config.floatX)

print "np_dataset_x", np_dataset_x.shape
print "np_dataset_y", np_dataset_y.shape
print "np_dataset_c", np_dataset_c.shape


del np_dataset


length = len(np_dataset_x)

n = nsamples

np_dataset_x_train = np_dataset_x[0:n]
np_dataset_y_train = np_dataset_y[0:n]
np_dataset_c_train = np_dataset_c[0:n]
print "np_dataset_x_train", len(np_dataset_x_train)

np_dataset_x_valid = np_dataset_x[n:2*n]
np_dataset_y_valid = np_dataset_y[n:2*n]
np_dataset_c_valid = np_dataset_c[n:2*n]
print "np_dataset_x_valid", len(np_dataset_x_valid)

np_dataset_x_test = np_dataset_x[100:]
np_dataset_y_test = np_dataset_y[100:]
np_dataset_c_test = np_dataset_c[100:]
print "np_dataset_x_test", len(np_dataset_x_test)





np_dataset_x_train.shape


np_dataset_x_train[:4,0].shape


plt.rcParams['figure.figsize'] = (15, 5)
plt.title("Example images")
plt.imshow(np.concatenate(np_dataset_x_train[:5,0],axis=1), interpolation='none', cmap='Greys_r')


plt.title("Example images")
plt.imshow(np.concatenate(np_dataset_y_train[:5,0],axis=1), interpolation='none')


plt.rcParams['figure.figsize'] = (15, 5)
plt.title("Counts in each image")
plt.bar(range(len(np_dataset_c_train)),np_dataset_c_train);


print "Total cells in training", np.sum(np_dataset_c_train[0:], axis=0)
print "Total cells in validation", np.sum(np_dataset_c_valid[0:], axis=0)
print "Total cells in testing", np.sum(np_dataset_c_test[0:], axis=0)





#to make video: ffmpeg -i image-0-%d-cell.png -vcodec libx264 aout.mp4
def processImages(name, i):
    fig = plt.Figure(figsize=(18, 9), dpi=160)
    gcf = plt.gcf()
    gcf.set_size_inches(18, 15)
    fig.set_canvas(gcf.canvas)
    
    (img, lab, count) = dataset[i]
    
    print str(i),count
    pcount = classify([[img]], [0])[0]
    
    lab_est = [(l.sum()/(ef)).astype(np.int) for l in lab]
    pred_est = [(l.sum()/(ef)).astype(np.int) for l in pcount]
    
    print str(i),"label est ",lab_est," --> predicted est ",pred_est

    ax2 = plt.subplot2grid((2,4), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((2,4), (0, 2), colspan=3)
    ax4 = plt.subplot2grid((2,4), (1, 2), colspan=3)
    ax5 = plt.subplot2grid((2,4), (1, 0), rowspan=1)
    ax6 = plt.subplot2grid((2,4), (1, 1), rowspan=1)

    ax2.set_title("Input Image")
    ax2.imshow(img, interpolation='none', cmap='Greys_r')
    ax3.set_title("Regression target, {}x{} sliding window.".format(patch_size, patch_size))
    ax3.imshow(np.concatenate((lab),axis=1), interpolation='none')
    ax4.set_title("Predicted counts")
    ax4.imshow(np.concatenate((pcount),axis=1), interpolation='none')

    ax5.set_title("Real " + str(lab_est))
    ax5.set_ylim((0, np.max(lab_est)*2))
    ax5.set_xticks(np.arange(0, noutputs, 1.0))
    ax5.bar(range(noutputs),lab_est, align='center')
    ax6.set_title("Pred " + str(pred_est))
    ax6.set_ylim((0, np.max(lab_est)*2))
    ax6.set_xticks(np.arange(0, noutputs, 1.0))
    ax6.bar(range(noutputs),pred_est, align='center')
    
    fig.savefig('images-cell/image-' + str(i) + "-" + name + '.png')








import pickle, os

directory = "network-temp/"
ext = "countception.p"

if not os.path.exists(directory):
    os.makedirs(directory)
    
def save_network(net,name):
    pkl_params = lasagne.layers.get_all_param_values(net, trainable=True)
    out = open(directory + str(name) + ext, "w", 0) #bufsize=0
    pickle.dump(pkl_params, out)
    out.close()

def load_network(net,name):
    all_param_values = pickle.load(open(directory + str(name) + ext, "r" ))
    lasagne.layers.set_all_param_values(net, all_param_values, trainable=True)


def re_init_network(net, re_seed):
    
    np.random.seed(re_seed)
    lasagne.random.set_rng(np.random.RandomState(re_seed))
    
    old = lasagne.layers.get_all_param_values(net, trainable=True)
    new = []
    for layer in old:
        shape = layer.shape
        if len(shape)<2:
            shape = (shape[0], 1)
        W= lasagne.init.GlorotUniform()(shape)
        if W.shape != layer.shape:
            W = np.squeeze(W, axis= 1)
        new.append(W)
    lasagne.layers.set_all_param_values(net, new, trainable=True)





#test accuracy
def test_perf(dataset_x, dataset_y, dataset_c):

    testpixelerrors = []
    testerrors = []
    bs = 1
    for i in range(0,len(dataset_x), bs):

        pcount = classify(dataset_x,range(i,i+bs))
        pixelerr = np.abs(pcount - dataset_y[i:i+bs]).mean(axis=(2,3))
        testpixelerrors.append(pixelerr)
        
        pred_est = (pcount/(ef)).sum(axis=(1,2,3))
        err = np.abs(dataset_c[i:i+bs].flatten()-pred_est)
        
        testerrors.append(err)
    
    return np.abs(testpixelerrors).mean(), np.abs(testerrors).mean()


print "Random performance"
print test_perf(np_dataset_x_train, np_dataset_y_train, np_dataset_c_train)
print test_perf(np_dataset_x_valid, np_dataset_y_valid, np_dataset_c_valid)
print test_perf(np_dataset_x_test, np_dataset_y_test, np_dataset_c_test)








re_init_network(net,seed)


target_var = T.tensor4('target')
lr = theano.shared(np.array(lr_param, dtype=theano.config.floatX))

#Mean Absolute Error is computed between each count of the count map
l1_loss = T.abs_(prediction - target_var[input_var_ex])

#Mean Absolute Error is computed for the overall image prediction
prediction_count2 =(prediction/ef).sum(axis=(2,3))
mae_loss = T.abs_(prediction_count2 - (target_var[input_var_ex]/ef).sum(axis=(2,3))) 

loss = l1_loss.mean()

params = lasagne.layers.get_all_params(net, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=lr)

train_fn = theano.function([input_var_ex], [loss,mae_loss], updates=updates,
                         givens={input_var:np_dataset_x_train, target_var:np_dataset_y_train})

print "DONE compiling theano functons"





batch_size = 2

print "batch_size", batch_size
print "lr", lr.eval()

best_valid_err = 99999999
best_test_err = 99999999
datasetlength = len(np_dataset_x_train)
print "datasetlength",datasetlength

for epoch in range(1000):
    start_time = time.time()

    epoch_err_pix = []
    epoch_err_pred = []
    todo = range(datasetlength)    
    
    for i in range(0,datasetlength, batch_size):
        ex = todo[i:i+batch_size]

        train_start_time = time.time()
        err_pix,err_pred = train_fn(ex)
        train_elapsed_time = time.time() - train_start_time

        epoch_err_pix.append(err_pix)
        epoch_err_pred.append(err_pred)

    valid_pix_err, valid_err = test_perf(np_dataset_x_valid, np_dataset_y_valid, np_dataset_c_valid)

    # a threshold is used to reduce processing when we are far from the goal
    if (valid_err < 20 and valid_err < best_valid_err):
        best_valid_err = valid_err
        best_test_err = test_perf(np_dataset_x_test, np_dataset_y_test,np_dataset_c_test)
        print "OOO best test (err_pix, err_pred)", best_test_err, ", epoch",epoch
        save_network(net,"best_valid_err")


    elapsed_time = time.time() - start_time
    err = np.mean(epoch_err_pix)
    acc = np.mean(np.concatenate(epoch_err_pred))
    
    if epoch % 5 == 0:
        print "#" + str(epoch) + "# (err_pix:" + str(np.around(err,3)) + ", err_pred:" +  str(np.around(acc,3)) + "), valid (err_pix:" + str(np.around(valid_pix_err,3)) + ", err_pred:" + str(np.around(valid_err,3)) +"), (time:" + str(np.around(elapsed_time,3)) + "sec)"

    #visualize training
    #processImages(str(epoch) + '-cell',0)

print "#####", "best_test_acc", best_test_err, "stride", stride, sys.argv


print "Done"





#load best network
load_network(net,"best_valid_err")





def compute_counts(dataset_x):

    bs = 1
    ests = []
    for i in range(0,len(dataset_x), bs):
        pcount = classify(dataset_x,range(i,i+bs))
        pred_est = (pcount/(ef)).sum(axis=(1,2,3))        
        ests.append(pred_est)
    return ests





get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (15, 5)
plt.title("Training Data")

pcounts = compute_counts(np_dataset_x_train)
plt.bar(np.arange(len(np_dataset_c_train))-0.1,np_dataset_c_train, width=0.5, label="Real Count");
plt.bar(np.arange(len(np_dataset_c_train))+0.1,pcounts, width=0.5,label="Predicted Count");
plt.tight_layout()
plt.legend()


get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (15, 5)
plt.title("Valid Data")

pcounts = compute_counts(np_dataset_x_valid)
plt.bar(np.arange(len(np_dataset_c_valid))-0.1,np_dataset_c_valid, width=0.5, label="Real Count");
plt.bar(np.arange(len(np_dataset_c_valid))+0.1,pcounts, width=0.5,label="Predicted Count");
plt.tight_layout()
plt.legend()


get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (15, 5)
plt.title("Test Data")

pcounts = compute_counts(np_dataset_x_test)
plt.bar(np.arange(len(np_dataset_c_test))-0.1,np_dataset_c_test, width=0.5, label="Real Count");
plt.bar(np.arange(len(np_dataset_c_test))+0.1,pcounts, width=0.5,label="Predicted Count");
plt.tight_layout()
plt.legend()


get_ipython().magic('matplotlib inline')
processImages('test',8)


get_ipython().magic('matplotlib inline')
processImages('test',1)


get_ipython().magic('matplotlib inline')
processImages('test',2)


get_ipython().magic('matplotlib inline')
processImages('test',3)


get_ipython().magic('matplotlib inline')
processImages('test',4)


get_ipython().magic('matplotlib inline')
processImages('test',5)


get_ipython().magic('matplotlib inline')
processImages('test',6)

















# # This does not work yet!
# 







import theano
import numpy as np
import matplotlib.pylab as plt
import csv, os, random, sys
get_ipython().magic('matplotlib inline')

import lasagne
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import Conv2DLayer, InputLayer, ConcatLayer
from lasagne.layers import DenseLayer, Pool2DLayer, FlattenLayer

print "theano",theano.version.full_version
print "lasagne",lasagne.__version__





#Set seed for random numbers:
np.random.seed(1234)
lasagne.random.set_rng(np.random.RandomState(1234))





##Uncomment these lines to fetch the dataset
#!wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#!tar -xvf cifar-10-python.tar.gz


data_dir_cifar10 = os.path.join(".", "cifar-10-batches-py")

def one_hot(x, n):
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def _load_batch_cifar10(filename, dtype='float32'):
    path = os.path.join(data_dir_cifar10, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0 # scale between [0, 1]
    labels = one_hot(batch['labels'], n=10) # convert labels to one-hot representation
    return data.astype(dtype), labels.astype(dtype)

def cifar10(dtype='float32', grayscale=True):
    x_train = []
    t_train = []
    for k in xrange(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)

    return x_train, t_train, x_test, t_test

# load data
x_train, t_train, x_test, t_test = cifar10(dtype=theano.config.floatX,grayscale=False)
labels_test = np.argmax(t_test, axis=1)

print "x_train.shape:",x_train.shape

# reshape data
x_train = x_train.reshape((x_train.shape[0], 3, 32, 32))
x_test = x_test.reshape((x_test.shape[0], 3, 32, 32))

cifar10_names = ['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']


# x_train = x_train[:256]
# t_train = t_train[:256]





get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(np.transpose(x_train[0], (1,2,0)),interpolation='none', cmap='gray');





import theano
import theano.tensor as T
import lasagne
import lasagne.layers
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer, MaxPool2DLayer, Upscale2DLayer
from lasagne.layers import ConcatLayer, DropoutLayer, ReshapeLayer, TransposedConv2DLayer


input_data = T.tensor4('cifar10')
input_var_ex = T.ivector('input_var_ex')

inshape = (None, 3, 32,32)
numhidden = 2
num_classes = 10

def ConvFactory(data, num_filter, filter_size, stride=1, pad=(0, 0), nonlinearity=lasagne.nonlinearities.rectify):
    data = lasagne.layers.batch_norm(Conv2DLayer(
        data, num_filters=num_filter,
        filter_size=filter_size,
        stride=stride, pad=pad,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(gain='relu')))
    return data

def DownsampleFactory(data, ch_3x3):
    conv = ConvFactory(data=data, filter_size=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1))
    pool = Pool2DLayer(data, pool_size=3, stride=(2, 2), pad=(1, 1), mode='max')
    concat = ConcatLayer([conv, pool])
    return concat

def SimpleFactory(data, ch_1x1, ch_3x3):
    conv1x1 = DeConvFactory(data=data, filter_size=(1, 1), pad=(0, 0), num_filter=ch_1x1)
    conv3x3 = DeConvFactory(data=data, filter_size=(3, 3), pad=(1, 1), num_filter=ch_3x3) 
    concat = ConcatLayer([conv1x1, conv3x3])
    return concat

def DeConvFactory(data, num_filter, filter_size, output_size=None, stride=1, pad=(0, 0), nonlinearity=lasagne.nonlinearities.rectify):
    data = lasagne.layers.batch_norm(TransposedConv2DLayer(
        data, num_filters=num_filter,
        filter_size=filter_size, stride=stride, crop=pad,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(gain='relu'),
        output_size=output_size))
    return data

def UpsampleFactory(data, ch_3x3):
    conv = DeConvFactory(data=data, filter_size=2, stride=2, num_filter=ch_3x3, pad='valid')
    pool = Upscale2DLayer(data, scale_factor=2)
    concat = ConcatLayer([conv, pool])
    return concat



net = InputLayer(shape=inshape, input_var=input_data[input_var_ex])
print net.output_shape
l_in = net


net = ConvFactory(data=net, filter_size=(3,3), pad=(1,1), num_filter=96)
print net.output_shape
net = SimpleFactory(net, 32, 32)
print net.output_shape
net = SimpleFactory(net, 32, 48)
print net.output_shape
net = DownsampleFactory(net, 80)
print net.output_shape
net = SimpleFactory(net, 112, 48)
print net.output_shape
net = SimpleFactory(net, 96, 64)
print net.output_shape
net = SimpleFactory(net, 80, 80)
print net.output_shape
net = SimpleFactory(net, 48, 96)
print net.output_shape
net = DownsampleFactory(net, 96)
print net.output_shape
net = SimpleFactory(net, 176, 160)
print net.output_shape
net = SimpleFactory(net, 176, 160)
print net.output_shape

net_before_hidden = net

##########

net = FlattenLayer(net)  
net = lasagne.layers.DenseLayer(net, num_units=numhidden,
                                     W=lasagne.init.GlorotUniform(),
                                     nonlinearity=None)
l_hidden = net

print net.output_shape


numfilters = 336
size = 8
net = lasagne.layers.DenseLayer(net, num_units=numfilters*size*size,
                                     W=lasagne.init.GlorotUniform(),
                                     nonlinearity=None)

net = ReshapeLayer(net, ([0],numfilters,size,size))
print net.output_shape

##########

net = SimpleFactory(net, 176, 160)
print net.output_shape
net = SimpleFactory(net, 176, 160)
print net.output_shape
net = UpsampleFactory(net, 96)
print net.output_shape
net = SimpleFactory(net, 48, 96)
print net.output_shape
net = SimpleFactory(net, 80, 80)
print net.output_shape
net = SimpleFactory(net, 96, 64)
print net.output_shape
net = SimpleFactory(net, 112, 48)
print net.output_shape
net = UpsampleFactory(net, 80)
print net.output_shape
net = SimpleFactory(net, 32, 48)
print net.output_shape
net = SimpleFactory(net, 32, 32)
print net.output_shape

net = DeConvFactory(net, filter_size=(3,3), pad=1, num_filter=3)
print net.output_shape

l_out = net





target_var = T.matrix('targets')

prediction = lasagne.layers.get_output(l_out)
#prediction = prediction.clip(0,1)
hidden = lasagne.layers.get_output(l_hidden)

#define how to make prediction
ae_reconstruct = theano.function(
    inputs=[input_var_ex],
    outputs=prediction,
    givens={input_data: x_train}
)

#define how to output embedding
ae_embed = theano.function(
    inputs=[input_var_ex],
    outputs=lasagne.layers.get_output(l_hidden),
    givens={input_data: x_train}
)

ae_reconstruct_test = theano.function(
    inputs=[input_var_ex],
    outputs=prediction,
    givens={input_data: x_test}
)





batch_size = 128
ae_embedding = np.array([])
for i in range(0,len(x_train),batch_size):
    ae_embedding = np.append(ae_embedding, ae_embed(range(i,min(i+batch_size,len(x_train)))))

ae_embedding = ae_embedding.reshape((len(x_train), numhidden))


plt.rcParams['figure.figsize'] = (25, 10)
plt.scatter(ae_embedding[:, 0], ae_embedding[:, 1], lw=0,c=np.argmax(t_train, axis=1));





import sklearn.manifold
coor_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(ae_embedding[:1000])
plt.rcParams['figure.figsize'] = (25, 10)
plt.scatter(coor_tsne[:, 0], coor_tsne[:, 1], lw=0, c=np.argmax(t_train, axis=1)[:1000]);





# create train functions 
lr = theano.shared(np.array(0., dtype=theano.config.floatX))

params_ae = lasagne.layers.get_all_params(l_out, trainable=True)

target_var = T.ivector('target')
rloss = lasagne.objectives.squared_error(prediction,input_data[input_var_ex]).mean()

updates_ae = lasagne.updates.adam(rloss, params_ae, learning_rate=lr)
f_train_ae = theano.function([input_var_ex], 
                          [rloss],
                          updates=updates_ae,
                          givens={input_data: x_train},
                          allow_input_downcast=True)





f_train_ae([0,1])








lr.set_value(0.0001)
batch_size = 64
print "batch_size",batch_size

best_valid_error = 9999999.

for j in range(100):
    
    batch_err = []  
    
    # shuffle batches
    todo = range(len(x_train))
    random.shuffle(todo)
    
    for i in range(0,len(x_train),batch_size):
        examples = todo[i:i+batch_size]
        err = f_train_ae(examples)
        batch_err.append(err)
        
    err_result = np.asarray(batch_err).mean(axis=0)
    
    
    ### Test error
    ae_reconstruction = np.array([])
    for i in range(0,len(x_test),batch_size):
        ae_reconstruction = np.append(ae_reconstruction, ae_reconstruct_test(range(i,min(i+batch_size,len(x_test)))))

    ae_reconstruction = ae_reconstruction.reshape((len(x_test), 3, 32, 32))
    valid_error = ((ae_reconstruction - x_test)**2).mean()
    
    best_valid_error = min(best_valid_error,valid_error)
    
    if j % 1 == 0:    
        print j, err_result, valid_error, best_valid_error





batch_size = 128
ae_embedding = np.array([])
for i in range(0,len(x_train),batch_size):
    ae_embedding = np.append(ae_embedding, ae_embed(range(i,min(i+batch_size,len(x_train)))))

ae_embedding = ae_embedding.reshape((len(x_train), numhidden))


# view first two dimensions
plt.rcParams['figure.figsize'] = (25, 10)
plt.scatter(ae_embedding[:, 0], ae_embedding[:, 1], lw=0,c=np.argmax(t_train, axis=1));


import sklearn.manifold
coor_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(ae_embedding[:10000])
plt.rcParams['figure.figsize'] = (25, 10)
plt.scatter(coor_tsne[:, 0], coor_tsne[:, 1], lw=0, c=np.argmax(t_train, axis=1)[:10000]);

















ae_reconstruction = ae_reconstruct(range(0,batch_size))


plt.rcParams['figure.figsize'] = (25, 5)
plt.imshow(np.transpose(x_train[0], (1,2,0)),interpolation='none', cmap='gray');


plt.rcParams['figure.figsize'] = (25, 5)
plt.imshow(np.transpose(ae_reconstruction[0], (1,2,0)),interpolation='none', cmap='gray');











ae_reconstruction = ae_reconstruct_test(range(0,batch_size))


plt.rcParams['figure.figsize'] = (25, 5)
plt.imshow(np.transpose(x_test[1], (1,2,0)),interpolation='none', cmap='gray');


plt.rcParams['figure.figsize'] = (25, 5)
plt.imshow(np.transpose(ae_reconstruction[1], (1,2,0)),interpolation='none', cmap='gray');





























# GAN Example - Joseph Paul Cohen 2017
# 
# This is trained on MNIST based on DC-GAN and code from Francis Dutil
# 

import theano
import numpy as np
import matplotlib.pylab as plt
import csv, os, random, sys

import lasagne
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import Conv2DLayer, InputLayer, ConcatLayer
from lasagne.layers import DenseLayer, Pool2DLayer, FlattenLayer

print "theano",theano.version.full_version
print "lasagne",lasagne.__version__





#Set seed for random numbers:
np.random.seed(1234)
lasagne.random.set_rng(np.random.RandomState(1234))





if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

import gzip
def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
    data = np.asarray([np.rot90(np.fliplr(x[0])) for x in data])
    data = data.reshape(-1, 1, 28, 28)
    return data / np.float32(255)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

x_train = load_mnist_images('train-images-idx3-ubyte.gz')
t_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
t_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
x_train, x_val = x_train[:-10000], x_train[-10000:]
t_train, t_val = t_train[:-10000], t_train[-10000:]





get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(x_train[0][0],interpolation='none', cmap='gray');





num_units = 1024
encoder_size = 100
noise_size = 100





import theano
import theano.tensor as T
import lasagne
import lasagne.layers
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer, MaxPool2DLayer, Upscale2DLayer
from lasagne.layers import ConcatLayer, DropoutLayer, ReshapeLayer, TransposedConv2DLayer


gen_input_var = T.matrix("gen_input_var")

gen = lasagne.layers.InputLayer(shape=(None, noise_size),input_var=gen_input_var)
print gen.output_shape

gen = lasagne.layers.ReshapeLayer(gen, (-1, noise_size, 1, 1))
print gen.output_shape
gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units, filter_size=4, stride=1))
print gen.output_shape

gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units/2, filter_size=8, stride=1))
print gen.output_shape

gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units/4, filter_size=16, stride=1))
print gen.output_shape

gen = TransposedConv2DLayer(gen, num_filters=num_units/8, filter_size=6)
print gen.output_shape

gen = Conv2DLayer(gen, num_filters=1, filter_size=4, nonlinearity=lasagne.nonlinearities.sigmoid)
print gen.output_shape


lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

disc = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=None)
print disc.output_shape

disc = BatchNormLayer(Conv2DLayer(disc, num_filters=num_units/4, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = BatchNormLayer(Conv2DLayer(disc, num_filters=num_units/2, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = (Conv2DLayer(disc, num_filters=num_units, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = FlattenLayer(disc)
disc = DenseLayer(disc, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
print disc.output_shape








# create train functions 
lr = theano.shared(np.array(0., dtype=theano.config.floatX))

gen_output = lasagne.layers.get_output(gen)

one = T.constant(1., dtype=theano.config.floatX)
input_real = T.tensor4('target')


disc_output_fake = lasagne.layers.get_output(disc, inputs=gen_output)
disc_output_real = lasagne.layers.get_output(disc, inputs=input_real)
disc_loss = -(T.log(disc_output_real) + T.log(one-disc_output_fake)).mean()
disc_params = lasagne.layers.get_all_params(disc, trainable=True)
disc_updates = lasagne.updates.adam(disc_loss, disc_params, learning_rate=lr, beta1=0.5)


gen_loss = -T.log(disc_output_fake).mean()
gen_params = lasagne.layers.get_all_params(gen, trainable=True)
gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=lr, beta1=0.5)


print "Computing functions"

gen_fn = theano.function([gen_input_var], gen_output, allow_input_downcast=True)

train_gen_fn = theano.function([gen_input_var], 
                               [gen_loss],
                               updates=gen_updates, 
                               allow_input_downcast=True)

disc_fn = theano.function([input_real], disc_output_real, allow_input_downcast=True)

train_disc_fn = theano.function([gen_input_var, input_real], 
                                [disc_loss],
                                updates=disc_updates,
                                allow_input_downcast=True)

print "Done"





noise = np.random.uniform(size=(10, noise_size))
img = gen_fn(noise)
print img.shape

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (20, 5)
plt.imshow(np.concatenate(img[:,0], axis=1),interpolation='none', cmap='gray');





disc_fn(img)


disc_fn(x_train[:10])








noise = np.random.normal(size=(1, noise_size))
train_gen_fn(noise)





train_disc_fn(noise, x_train[:1])








from scipy.ndimage.filters import gaussian_filter
def permute(img):
        
    rotation = random.choice([True, False])
    flip = random.choice([True, False])
    blur = random.randint(0,1)
    pad = random.randint(0,5)
    
    if (pad != 0):
        img = img[:,pad:-pad,pad:-pad]
        img = [scipy.misc.imresize(img[0], image_size), 
               scipy.misc.imresize(img[1], image_size), 
               scipy.misc.imresize(img[2], image_size)]
    
    if (blur != 0):
        img = gaussian_filter(img, sigma=blur)
    
    #rotate 
    if rotation:
        img = [np.rot90(img[0], 2),np.rot90(img[1], 2) ,np.rot90(img[2], 2)]
    
    #flip 
    if (flip):
        img = np.fliplr(img)  
    
    return img





num_samples = 1000





# ### D
# lr.set_value(0.0001)
# for i in range(10):
#     errs = []
#     for i in range(10):
#         noise = np.random.normal(size=(num_samples, noise_size))
#         samples = np.random.randint(0,len(x_train),num_samples)
#         err = train_disc_fn(noise, x_train[samples])
#         errs.append(err)
#     print "d",np.mean(errs)





# ### G
# lr.set_value(0.001)
# for i in range(1000):
#     errs = []
#     for i in range(5):
#         noise = np.random.normal(size=(num_samples, noise_size))
#         err = train_gen_fn(noise)
#         errs.append(err)
#     print "g",np.mean(errs)





lr.set_value(0.0001)
for j in range(100):
    err = 1
    while err > 0.5:
        noise = np.random.normal(size=(num_samples, noise_size))
        samples = np.random.randint(0,len(x_train),num_samples)
        err = train_disc_fn(noise, x_train[samples])[0]
    print "d",err

    err = 1
    while err > 0.5:
        noise = np.random.normal(size=(num_samples, noise_size))
        err = train_gen_fn(noise)[0]
    print "g",err





noise = np.random.uniform(size=(10, noise_size))
img = gen_fn(noise)
print img.shape

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (20, 5)
plt.imshow(np.concatenate(img[:,0], axis=1),interpolation='none', cmap='gray');

















# Interactive Manual Spatial Transformer Network Layer Demo with Lasagne!
# This is a tool made to help understand what the Spatial Transformer Network Layer is doing. An image is put into the network and then you can manually control the 6 parameters of the spatial transformer layer's theta parameter. The layer will transform the image and output a downsampled version that is zoomed into a region of the image which is show here in real time. 
# 
# This is good to help debug the bounds and constraits that you should impose on the theta vector. These layers seem unforgiving once you have moved outside of reasonable parameters. 
# 
# Here is a demo: http://www.youtube.com/watch?v=zPvhz6KDRyg
# [![](http://img.youtube.com/vi/zPvhz6KDRyg/0.jpg)](http://www.youtube.com/watch?v=zPvhz6KDRyg)
# 

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano, theano.tensor as T 

import lasagne, lasagne.layers

print ("theano",theano.version.full_version)
print ("lasagne",lasagne.__version__)





import skimage, scipy
from skimage.io import imread, imsave


get_ipython().system('wget -c https://i.imgur.com/3skvA.jpg')





get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (18, 9)

img_raw = imread("3skvA.jpg")
print ("img_raw", img_raw.shape)
img_raw = scipy.misc.imresize(img_raw, (img_raw.shape[0]/2,img_raw.shape[1]/2,img_raw.shape[2]))

plt.imshow(img_raw);





input_var = T.tensor4('inputs')
input_shape = (None, 3,1016,2048)
img = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

loc_var = T.matrix('loc')
loc_shape = (None,6)
loc = lasagne.layers.InputLayer(shape=loc_shape, input_var=loc_var)

img_trans = lasagne.layers.TransformerLayer(img, loc, downsample_factor=5.0)
print ("Transformer network output shape: ", img_trans.output_shape)


output_shape = lasagne.layers.get_output_shape(img_trans)
print ("input_shape:",input_shape,"-> output_shape:",output_shape)


img_trans_output = lasagne.layers.get_output(img_trans)
loc_output = lasagne.layers.get_output(loc)

f_transform = theano.function([input_var, loc_var], 
                              img_trans_output, 
                              allow_input_downcast=True)

print ("DONE building output functions")





from __future__ import print_function
from ipywidgets import interact, interactive, fixed, FloatSlider
import ipywidgets as widgets


def explore(t11=0.5,t12=0,t13=0,t21=0,t22=1,t23=0):
    o = f_transform([np.transpose(img_raw, (2, 0, 1))], [[t11,t12,t13,t21,t22,t23]])
    oo = np.transpose(o[0],(1,2,0))
    plt.imshow(256-oo.astype(int), interpolation='none');


interact(explore, 
         t11=FloatSlider(value=1,min=0, max=2, step=0.05,continuous_update=True),
         t12=FloatSlider(value=0,min=-1, max=1, step=0.05,continuous_update=True),
         t13=FloatSlider(value=0,min=-1, max=1, step=0.05,continuous_update=True),
         t21=FloatSlider(value=0,min=-1, max=1, step=0.05,continuous_update=True),
         t22=FloatSlider(value=1,min=0, max=2, step=0.05,continuous_update=True),
         t23=FloatSlider(value=0,min=-1, max=1, step=0.05,continuous_update=True),
        );














# GAN Example
# 
# The samples do not look good yet but the basics are here
# 

import theano
import numpy as np
import matplotlib.pylab as plt
import csv, os, random, sys

import lasagne
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import Conv2DLayer, InputLayer, ConcatLayer
from lasagne.layers import DenseLayer, Pool2DLayer, FlattenLayer

print "theano",theano.version.full_version
print "lasagne",lasagne.__version__





#Set seed for random numbers:
np.random.seed(1234)
lasagne.random.set_rng(np.random.RandomState(1234))





##Uncomment these lines to fetch the dataset
#!wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#!tar -xvf cifar-10-python.tar.gz


data_dir_cifar10 = os.path.join(".", "cifar-10-batches-py")

def one_hot(x, n):
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def _load_batch_cifar10(filename, dtype='float32'):
    path = os.path.join(data_dir_cifar10, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0 # scale between [0, 1]
    labels = one_hot(batch['labels'], n=10) # convert labels to one-hot representation
    return data.astype(dtype), labels.astype(dtype)

def cifar10(dtype='float32', grayscale=True):
    x_train = []
    t_train = []
    for k in xrange(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)

    return x_train, t_train, x_test, t_test

# load data
x_train, t_train, x_test, t_test = cifar10(dtype=theano.config.floatX,grayscale=False)
labels_test = np.argmax(t_test, axis=1)

print "x_train.shape:",x_train.shape

# reshape data
x_train = x_train.reshape((x_train.shape[0], 3, 32, 32))
x_test = x_test.reshape((x_test.shape[0], 3, 32, 32))

cifar10_names = ['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']


# x_train = x_train[:10000]
# t_train = t_train[:10000]





get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(np.transpose(x_train[0], (1,2,0)),interpolation='none', cmap='gray');





num_units = 100
encoder_size = 100
noise_size = 10





import theano
import theano.tensor as T
import lasagne
import lasagne.layers
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer, MaxPool2DLayer, Upscale2DLayer
from lasagne.layers import ConcatLayer, DropoutLayer, ReshapeLayer, TransposedConv2DLayer


gen_input_var = T.matrix("gen_input_var")

gen = lasagne.layers.InputLayer(shape=(None, noise_size),input_var=gen_input_var)
print gen.output_shape

gen = lasagne.layers.ReshapeLayer(gen, (-1, noise_size, 1, 1))
print gen.output_shape
gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units, filter_size=4, stride=1))
print gen.output_shape

gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units/2, filter_size=6, stride=1))
print gen.output_shape

gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units/2, filter_size=14, stride=1))
print gen.output_shape

gen = TransposedConv2DLayer(gen, num_filters=num_units/4, filter_size=8)
print gen.output_shape

gen = TransposedConv2DLayer(gen, num_filters=3, filter_size=4, nonlinearity=lasagne.nonlinearities.sigmoid)
print gen.output_shape


lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

disc = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=None)
print disc.output_shape

disc = BatchNormLayer(Conv2DLayer(disc, num_filters=num_units/4, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = BatchNormLayer(Conv2DLayer(disc, num_filters=num_units/2, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = BatchNormLayer(Conv2DLayer(disc, num_filters=num_units, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = FlattenLayer(disc)
disc = DenseLayer(disc, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
print disc.output_shape




















# create train functions 
lr = theano.shared(np.array(0., dtype=theano.config.floatX))

gen_output = lasagne.layers.get_output(gen)

one = T.constant(1., dtype=theano.config.floatX)
input_real = T.tensor4('target')


disc_output_fake = lasagne.layers.get_output(disc, inputs=gen_output)
disc_output_real = lasagne.layers.get_output(disc, inputs=input_real)
disc_loss = -(T.log(disc_output_real) + T.log(one-disc_output_fake)).mean()
disc_params = lasagne.layers.get_all_params(disc, trainable=True)
disc_updates = lasagne.updates.adam(disc_loss, disc_params, learning_rate=lr, beta1=0.5)


gen_loss = -T.log(disc_output_fake).mean()
gen_params = lasagne.layers.get_all_params(gen, trainable=True)
gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=lr, beta1=0.5)


print "Computing functions"

gen_fn = theano.function([gen_input_var], gen_output, allow_input_downcast=True)

train_gen_fn = theano.function([gen_input_var], 
                               [gen_loss],
                               updates=gen_updates, 
                               allow_input_downcast=True)

disc_fn = theano.function([input_real], disc_output_real, allow_input_downcast=True)

train_disc_fn = theano.function([gen_input_var, input_real], 
                                [disc_loss],
                                updates=disc_updates,
                                allow_input_downcast=True)

print "Done"








noise = np.random.uniform(size=(1, noise_size))
img = gen_fn(noise)
print img.shape

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(np.transpose(img[0], (1,2,0)),interpolation='none', cmap='gray');


get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(np.transpose(x_train[0], (1,2,0)),interpolation='none', cmap='gray');


disc_fn(img)


disc_fn(x_train[:1])








num_samples = 1000


# lr.set_value(0.001)
# for i in range(100):
#     errs = []
#     for i in range(10):
#         noise = np.random.normal(size=(num_samples, noise_size))
#         samples = np.random.randint(0,len(x_train),num_samples)
#         err = train_disc_fn(noise, x_train[samples])
#         errs.append(err)
#     print "d",np.mean(errs)





# lr.set_value(0.001)
# for i in range(1000):
#     errs = []
#     for i in range(5):
#         noise = np.random.normal(size=(num_samples, noise_size))
#         err = train_gen_fn(noise)
#         errs.append(err)
#     print "g",np.mean(errs)





lr.set_value(0.001)
for j in range(100):
    errs = []
    for i in range(10):
        noise = np.random.normal(size=(num_samples, noise_size))
        samples = np.random.randint(0,len(x_train),num_samples)
        err = train_disc_fn(noise, x_train[samples])
        errs.append(err)
    print "d",np.mean(errs)
    errs = []
    for i in range(1):
        noise = np.random.normal(size=(num_samples, noise_size))
        err = train_gen_fn(noise)
        errs.append(err)
    print "g",np.mean(errs)





noise = np.random.uniform(size=(1, noise_size))
img = gen_fn(noise)
print img.shape

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(np.transpose(img[0], (1,2,0)),interpolation='none', cmap='gray');





# At the bottom of this notebook there is an interactive MXNet example. It Lets you vary the inputs with sliders and will compute outputs and gradients. You can even edit the network to make it more complex!
# 
# A video of this working is here: https://www.youtube.com/watch?v=-KmImwP5eGk
# Joseph Paul Cohen 2016 (Code free for non-commercial use)
# 
# # This version is old!
# The next version makes it easy to increase and decrease the number of parameters. Check out the next version here: [mxnet-vary-inputs-slideexamples.ipynb](https://github.com/ieee8023/NeuralNetwork-Examples/blob/master/mxnet/mxnet-vary-inputs-slideexamples.ipynb) 
# 

import mxnet as mx
import cmath
import numpy as np

from __future__ import print_function
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets


def compute(s=None, x0=1, x1=1, x2=1, w0=1, w1=1, w2=1):

    # Specify inputs we declared 
    args={'x0': mx.nd.array([x0]),
          'x1': mx.nd.array([x1]),
          'x2': mx.nd.array([x2]),
          'w0': mx.nd.array([w0]),
          'w1': mx.nd.array([w1]),
          'w2': mx.nd.array([w2])
         }


    sym = s.get_internals()
    blob_names = sym.list_outputs()
    sym_group = []
    for i in range(len(blob_names)):
        if blob_names[i] not in args:
            x = sym[i]
            if blob_names[i] not in sym.list_outputs():
                x = mx.symbol.BlockGrad(x, name=blob_names[i])
            sym_group.append(x)
    sym = mx.symbol.Group(sym_group)


    # Bind to symbol and create executor
    c_exec = sym.simple_bind(
                    ctx=mx.cpu(),
                    x0 = args['x0'].shape,
                    x1 = args['x1'].shape,
                    w0 = args['w0'].shape, 
                    w1 = args['w1'].shape, 
                    x2 = args['x2'].shape,
                    w2 = args['w2'].shape)

    # Copy input values into executor memory
    c_exec.copy_params_from(arg_params = args)

    # Perform computation forward to populate outputs
    c_exec.forward()

    values = []
    values = values + [(k,v.asnumpy()[0]) for k,v in zip(sym.list_arguments(),c_exec.arg_arrays)]
    values = values + [(k,v.asnumpy()[0]) for k,v in zip(sym.list_outputs(),c_exec.outputs)]

    # Bind to symbol and create executor
    c_exec = s.simple_bind(
                    ctx=mx.cpu(),
                    x0 = args['x0'].shape,
                    x1 = args['x1'].shape,
                    w0 = args['w0'].shape, 
                    w1 = args['w1'].shape, 
                    x2 = args['x2'].shape, 
                    w2 = args['w2'].shape)

    # Copy input values into executor memory
    c_exec.copy_params_from(arg_params = args)

    # Perform computation forward to populate outputs
    c_exec.forward()

    # Backpropagate to calculate gradients 
    c_exec.backward(out_grads=mx.nd.array([1]))

    grads = []
    grads = grads + [(k,v.asnumpy()[0]) for k,v in zip(s.list_arguments(),c_exec.grad_arrays)]

    # Use these for debugging
    #for k,v in values: print("%20s=%.03f"% (k,v))
    #for k,v in grads: print("%20s=%.03f"% ("dout/d%s"%k,v))
    
    values_dict = dict(values)
    grads_dict = dict(grads)

    # Print computation graph of s.get_internals() because is shows more nodes
    a = plot_network2(sym, shape={
                                  "w0":(1,), 
                                  "x0":(1,),
                                  "w1":(1,),
                                  "x1":(1,),
                                  "x2":(1,),
                                  "w2":(1,)},
                            node_attrs={"shape":'rect',"fixedsize":'false'},
                     values_dict=values_dict, grads_dict=grads_dict)
    #Rotate the graphviz object that is returned
    a.body.extend(['rankdir=RL', 'size="10,5"'])

    del c_exec
    del sym
    
    #Show it. Use a.render() to write it to disk
    return a


## Here we define a new print network function

from __future__ import absolute_import
from mxnet.symbol import Symbol
import json
import re
import copy

def plot_network2(symbol, title="plot", shape=None, node_attrs={}, values_dict=None, grads_dict=None):
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    draw_shape = False
    if shape != None:
        draw_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes == None:
            raise ValueError("Input shape is incompete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    #print(conf)
    heads = set([x[0] for x in conf["heads"]])  # TODO(xxx): check careful
    #print(heads)
    # default attributes of node
    node_attr = {"shape": "box", "fixedsize": "true",
                 "width": "1.3", "height": "0.8034", "style": "filled"}
    # merge the dcit provided by user and the default one
    node_attr.update(node_attrs)
    dot = Digraph(name=title)
    dot.body.extend(['rankdir=RL', 'size="10,5"'])
    # color map
    cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5")

    # make nodes
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        name = node["name"]
        # input data
        attr = copy.deepcopy(node_attr)
        label = op

        if op == "null":
            label = node["name"]
            if grads_dict != None and label in grads_dict:
                label = label + ("\n d%s: %.2f" % (label, grads_dict[label]))
            
        attr["fillcolor"] = cm[1]
        
        if op == "Convolution":
            label = "Convolution\n%sx%s/%s, %s" % (_str2tuple(node["param"]["kernel"])[0],
                                                   _str2tuple(node["param"]["kernel"])[1],
                                                   _str2tuple(node["param"]["stride"])[0],
                                                   node["param"]["num_filter"])
            attr["fillcolor"] = cm[1]
        elif op == "FullyConnected":
            label = "FullyConnected\n%s" % node["param"]["num_hidden"]
            attr["fillcolor"] = cm[1]
        elif op == "BatchNorm":
            attr["fillcolor"] = cm[3]
        elif op == "Activation" or op == "LeakyReLU":
            label = "%s\n%s" % (op, node["param"]["act_type"])
            attr["fillcolor"] = cm[2]
        elif op == "Pooling":
            label = "Pooling\n%s, %sx%s/%s" % (node["param"]["pool_type"],
                                               _str2tuple(node["param"]["kernel"])[0],
                                               _str2tuple(node["param"]["kernel"])[1],
                                               _str2tuple(node["param"]["stride"])[0])
            attr["fillcolor"] = cm[4]
        elif op == "Concat" or op == "Flatten" or op == "Reshape":
            attr["fillcolor"] = cm[5]
        elif op == "Softmax":
            attr["fillcolor"] = cm[6]
        else:
            attr["fillcolor"] = cm[0]

        dot.node(name=name, label=label, **attr)
    
    # add edges
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        name = node["name"]
        inputs = node["inputs"]
        for item in inputs:
            input_node = nodes[item[0]]
            input_name = input_node["name"]
            attr = {"dir": "back", 'arrowtail':'open'}
                
            label = ""
            if values_dict != None and input_name in values_dict:
                label = "%.2f" % values_dict[input_name]               
                
            if values_dict != None and input_name + "_output" in values_dict:
                label = "%.2f" % values_dict[input_name + "_output"]
                
            #if grads_dict != None and input_name in grads_dict:
            #    label = label + ("/%.2f" %grads_dict[input_name])
    
            attr["label"] = label.replace("_","")
            dot.edge(tail_name=name, head_name=input_name, **attr)
    return dot


## Outputs from a node are shown on the edges and the gradients are shown in the box
## Modify the s object in compute different functions

# Declare input values in mxnet type
w0 = mx.symbol.Variable('w0')
x0 = mx.symbol.Variable('x0')
w1 = mx.symbol.Variable('w1')
x1 = mx.symbol.Variable('x1')
w2 = mx.symbol.Variable('w2')
x2 = mx.symbol.Variable('x2')

# Form expression using overloaded +,-,*, and / operators
# Use special mx methods to achieve other operations
n = ((w0*x0 + w1*x1) * x2 + w2)
s = mx.symbol.Activation(data=n, name='relu1', act_type="relu")+0

interact(compute, s=fixed(s), x0=1.0, w0=1.0, x1=1.0, w1=1.0, x2=1.0, w2=1.0)





## Outputs from a node are shown on the edges and the gradients are shown in the box
## Modify the s object in compute different functions

# Declare input values in mxnet type
w0 = mx.symbol.Variable('w0')
x0 = mx.symbol.Variable('x0')
w1 = mx.symbol.Variable('w1')
x1 = mx.symbol.Variable('x1')
w2 = mx.symbol.Variable('w2')
x2 = mx.symbol.Variable('x2')

# Form expression using overloaded +,-,*, and / operators
# Use special mx methods to achieve other operations
n = ((w0*x0*x2 + w1*x1*x2*x0) * x2*w2)
s = mx.symbol.Activation(data=n, name='relu1', act_type="relu")+0

interact(compute, s=fixed(s), x0=1.0, w0=1.0, x1=1.0, w1=1.0, x2=1.0, w2=1.0)











# ## RandomOut implementation for MXNet
# This notebook is a demo of the RandomOut algorithm. It is implemented as a Monitor that can be passed to the fit method of FeedForward model object. Every epoch the monitor will be invoked and test that every convolutional filter has a CGN value greater than the tau value passed in. If a filter fails the check then it is reinitialized using the initializer from the model.
# 
# The code is set up to train the 28x28 inception arch on the CIFAR-10 dataset. It can be run on multiple GPUs by setting the num_devs variable.
# 
# Using the default script parameters (on 8 GPUs) after 20 epochs we achieve the following testing accuracy:
# + wo/RandomOut = 0.7075
# + w/RandomOut = 0.7929
# 
# Paper: https://arxiv.org/abs/1602.05931
# 
# ShortScience.org: http://www.shortscience.org/paper?bibtexKey=journals/corr/CohenL016
# 
# This nodebook can be run from the command line using: 
# 
#     jupyter nbconvert randomout-cifar-inception.ipynb --to script
#     python randomout-cifar-inception.py
# 

import mxnet as mx


import numpy as np
import cmath
import graphviz
import argparse
import os, sys


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--tau', type=float, default=1e-30)
parser.add_argument('--randomout', type=str, default="True")
parser.add_argument('--network', type=str, default="inception-28-small")
parser.add_argument('-f', type=str, default='')
args = parser.parse_args()
args.f = ''

# setup logging
import logging
logging.getLogger().handlers = []
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#logging.root = logging.getLogger(str(args))
logging.root = logging.getLogger()
logging.debug("test")


import importlib
softmax = importlib.import_module('symbol_' + args.network).get_symbol(10)


# If you'd like to see the network structure, run the plot_network function
a = mx.viz.plot_network(symbol=softmax.get_internals(),node_attrs={'shape':'rect','fixedsize':'false'},
                       shape={"data":(1,3, 28, 28)}) 

a.body.extend(['rankdir=RL', 'size="40,5"'])
#a


mx.random.seed(args.seed)
num_epoch = args.epochs
batch_size = args.batch_size
num_devs = 1
model = mx.model.FeedForward(ctx=[mx.gpu(i) for i in range(num_devs)], symbol=softmax, num_epoch = num_epoch,
                             learning_rate=0.1, momentum=0.9, wd=0.00001
                             ,optimizer=mx.optimizer.Adam()
                            )


import get_data
get_data.GetCifar10()

train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=4)
# test iterator make batch of 128 image, and center crop each image into 3x28x28 from original 3x32x32
# Note: We don't need round batch in test because we only test once at one time
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        round_batch=False,
        preprocess_threads=4)





from mxnet.ndarray import NDArray
from mxnet.base import NDArrayHandle
from mxnet import ndarray

class RandomOutMonitor(mx.monitor.Monitor):
    
    def __init__(self, initializer, network, tau=0.000001, *args,**kwargs):
        mx.monitor.Monitor.__init__(self, 1, *args, **kwargs) 
        self.tau = tau
        self.initializer = initializer
        
        # here the layers we want to subject to the threshold are specified
        targetlayers = [x for x in network.list_arguments() if x.startswith("conv") and x.endswith("weight")]
        self.targetlayers = targetlayers
        
        logging.info("RandomOut active on layers: %s" % self.targetlayers)
        
    def toc(self):
        for exe in self.exes:
            for array in exe.arg_arrays:
                array.wait_to_read()
        for exe in self.exes:
            for name, array in zip(exe._symbol.list_arguments(), exe.arg_arrays):
                self.queue.append((self.step, name, self.stat_func(array)))
                
        for exe in self.exes:
            weights = dict(zip(softmax.list_arguments(), exe.arg_arrays))
            grads = dict(zip(softmax.list_arguments(), exe.grad_arrays))
            numFilters = 0
            for name in self.targetlayers:
            
                filtersg = grads[name].asnumpy()
                filtersw = weights[name].asnumpy()

                #get random array to copy over
                filtersw_rand = mx.nd.array(filtersw.copy())
                self.initializer(name, filtersw_rand)
                filtersw_rand = filtersw_rand.asnumpy()
                
                agrads = [0.0] * len(filtersg)
                for i in range(len(filtersg)):
                    agrads[i] = np.absolute(filtersg[i]).sum()
                    if agrads[i] < self.tau:
                        numFilters = numFilters+1
                        #logging.info("RandomOut: filter %i of %s has been randomized because CGN=%f" % (i,name,agrads[i]))
                        filtersw[i] = filtersw_rand[i]

                #logging.info("%s, %s, %s" % (name, min(agrads),np.mean(agrads)))
            
                weights[name] = mx.nd.array(filtersw)
                #print filtersw
            if numFilters >0:
                #logging.info("numFilters replaced: %i"%numFilters)   
                exe.copy_params_from(arg_params=weights)
            
        self.activated = False
        return []
    





train_dataiter.reset()
if args.randomout == "True":
    model.fit(X=train_dataiter,
        eval_data=test_dataiter,
        eval_metric="accuracy",
        batch_end_callback=mx.callback.Speedometer(batch_size)
        ,monitor=RandomOutMonitor(initializer = model.initializer, network=softmax, tau=args.tau)
        )
else:
    model.fit(X=train_dataiter,
        eval_data=test_dataiter,
        eval_metric="accuracy",
        batch_end_callback=mx.callback.Speedometer(batch_size)
        )








