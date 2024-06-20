# ## Visualizing GoogLeNet filters
# 
# This is an ipython notebook to generate visualizations of GoogLeNet filters, for some more info refer to [this blogpost](https://auduno.github.io/2016/06/18/peeking-inside-convnets/).
# 
# To run this code, you'll need an installation of Caffe with built pycaffe libraries, as well as the python libraries numpy, scipy and PIL. For instructions on how to install Caffe and pycaffe, refer to the installation guide [here](http://caffe.berkeleyvision.org/installation.html). Before running the ipython notebooks, you'll also need to download the [GoogLeNet model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet), and modify the variables ```pycaffe_root``` to refer to the path of your pycaffe installation (if it's not already in your python path) and ```model_path``` to refer to the path of the downloaded GoogLeNet caffe model. Also uncomment the line that enables GPU mode if you have built Caffe with GPU-support and a suitable GPU available.
# 

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys
from IPython.display import clear_output, Image, display
from scipy.misc import imresize

pycaffe_root = "/your/path/here/caffe/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

model_name="GoogLeNet"
model_path = '/your/path/here/caffe_models/bvlc_googlenet/' # substitute your path here
# modified deploy.prototxt, switched relus to leaky relus
net_fn   = './googlenet_deploy_mod.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
means = np.float32([104.0, 117.0, 123.0])

#caffe.set_mode_gpu() # uncomment this if gpu processing is available

net = caffe.Classifier(net_fn, param_fn,
                       mean = means, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def showarray(a, f, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def make_step(net, step_size=1.5, end='inception_4c/output', clip=True, focus=None, sigma=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob

    dst = net.blobs[end]
    net.forward(end=end)
    
    one_hot = np.zeros_like(dst.data)
    filter_shape = dst.data.shape
    if len(filter_shape) > 2:
        # backprop only activation in middle of filter
        one_hot[0,focus,(filter_shape[2]-1)/2,(filter_shape[3]-1)/2] = 1.
    else:
        one_hot.flat[focus] = 1.
    dst.diff[:] = one_hot
    
    net.backward(start=end)
    g = src.diff[0]
    
    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 
        
    src.data[0] = blur(src.data[0], sigma)
    
    dst.diff.fill(0.)

def deepdraw(net, base_img, octaves, random_crop=True, visualize=True, focus=None,
    clip=True, **step_params):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "starting drawing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
        _,imw,imh = image.shape
        
        # select layer
        layer = o['layer']
        
        for i in xrange(o['iter_n']):
            if imw > w:
                if random_crop:
                    # randomly select a crop 
                    #ox = random.randint(0,imw-224)
                    #oy = random.randint(0,imh-224)
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = np.random.normal(mid_x, width_x*0.3, 1)
                    ox = int(np.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = np.random.normal(mid_y, width_y*0.3, 1)
                    oy = int(np.clip(oy,0,imh-h))
                    # insert the crop into src.data[0]
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
                else:
                    ox = (imw-w)/2.
                    oy = (imh-h)/2.
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
            else:
                ox = 0
                oy = 0
                src.data[0] = image.copy()

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            
            make_step(net, end=layer, clip=clip, focus=focus, 
                      sigma=sigma, step_size=step_size)
            
            if visualize:
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                if i % 1 == 0:
                    showarray(vis,"./filename"+str(i)+".jpg")
            
            if i % 10 == 0:
                print 'finished step %d in octave %d' % (i,e)
            
            # insert modified image back into original image (if necessary)
            image[:,ox:ox+w,oy:oy+h] = src.data[0]
        
        print "octave %d image:" % e
        showarray(deprocess(net, image),"./octave_"+str(e)+".jpg")
            
    # returning the resulting image
    return deprocess(net, image)


octaves = [
    {
        'layer':'inception_4c/output',
        'iter_n':200,
        'start_sigma':2.5,
        'end_sigma':1.1,
        'start_step_size':12.,
        'end_step_size':10.,
    },
    {
        'layer':'inception_4c/output',
        'iter_n':100,
        'start_sigma':1.1,
        'end_sigma':0.78*1.1,
        'start_step_size':10.,
        'end_step_size':8.
    },
    {
        'layer':'inception_4c/output',
        'scale':1.05,
        'iter_n':100,
        'start_sigma':0.78*1.1,
        'end_sigma':0.78,
        'start_step_size':8.,
        'end_step_size':6.
    },
    {
        'layer':'inception_4c/output',
        'scale':1.05,
        'iter_n':50,
        'start_sigma':0.78*1.1,
        'end_sigma':0.40,
        'start_step_size':6.,
        'end_step_size':1.5
    },
    {
        'layer':'inception_4c/output',
        'scale':1.05,
        'iter_n':25,
        'start_sigma':0.4,
        'end_sigma':0.3,
        'start_step_size':1.5,
        'end_step_size':0.5
    }
]

# get original input size of network
original_w = net.blobs['data'].width
original_h = net.blobs['data'].height
# the background color of the initial image
background_color = np.float32([250.0, 250.0, 250.0])
# generate initial random image
gen_image = np.random.normal(background_color, 8, (original_w, original_h, 3))

# which filter in layer to visualize (conv5 has 512 filters)
imagenet_class = 411

# generate class visualization via octavewise gradient ascent
gen_image = deepdraw(net, gen_image, octaves, focus=imagenet_class, 
                 random_crop=True, visualize=False)

# save image
#img_fn = '_'.join([model_name, "deepdraw", str(imagenet_class)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)





# Code to generate paths with generated images from GoogLeNet, i.e. "painting" with GoogLeNet. See [this blogpost](http://auduno.com/post/125837418083/drawing-with-googlenet) for details.
# 
# Before running the code, insert your pycaffe path in ```pycaffe_root``` and insert the path to the googlenet model in ```model_path```. Download the bvlc googlenet model from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) if you haven't done so already.
# 

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys,math,time
from IPython.display import clear_output, Image, display
from scipy.misc import imresize

pycaffe_root = "/your/path/here/caffe/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

model_name = "GoogLeNet"
model_path = '/your/path/here/caffe_models/bvlc_googlenet/' # substitute your path here
net_fn   = '../deploy_googlenet_updated.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
mean = np.float32([104.0, 117.0, 123.0])

#caffe.set_mode_gpu() # uncomment this if gpu processing is available
net = caffe.Classifier(net_fn, param_fn,
                       mean = mean, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def showarray(a, f, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def make_step(net, step_size=1.5, end='inception_4c/output', clip=True, clip_mask=None, focus=None, sigma=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob

    dst = net.blobs[end]
    net.forward(end=end)

    one_hot = np.zeros_like(dst.data)
    one_hot.flat[focus] = 1.
    dst.diff[:] = one_hot

    net.backward(start=end)
    g = src.diff[0]
    
    if not clip_mask is None:
        g *= clip_mask
    
    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 
        
    src.data[0] = blur(src.data[0], sigma)
    
    dst.diff.fill(0.)

def deepdraw(net, base_img, octaves, focus, visualize=True, clip=True, clip_gradient=None, pathfun=None):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "starting drawing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    current_scale = 1.0
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
            current_scale *= o['scale']
        _,imw,imh = image.shape
        
        # select layer
        layer = o['layer']

        for i in xrange(o['iter_n']):
            if pathfun:
                coords = pathfun.getPathPoint()

                # scale points
                mid = [0,0]
                mid[0] = coords[0]*current_scale - w/2
                mid[1] = coords[1]*current_scale - h/2

                # randomize crop to get smoother edges
                width = max((w*current_scale)-w,10.)
                ox = np.random.normal(mid[0], width*0.3, 1)
                ox = int(np.clip(ox,0,imw-w))
                height = max((h*current_scale)-h,10.)
                oy = np.random.normal(mid[1], height*0.3, 1)
                oy = int(np.clip(oy,0,imh-h))        
                #print "orig coords : %f,%f, new coords : %f,%f" % (coords[0],coords[1],ox,oy)
            else:
                # draw in middle of image
                ox = imw/2. - w/2
                oy = imh/2. - h/2

            #ox = coords[0]*current_scale - 224/2
            #oy = coords[1]*current_scale - 224/2

            #mid_x = coords[0]*current_scale - 224/2
            #mid_y = coords[1]*current_scale - 224/2
            #width = max((224*current_scale)-224,10.)
            #ox = np.random.normal(mid_x, width*0.3, 1)
            #ox = int(np.clip(ox,0,imw-224))
            #oy = np.random.normal(mid_y, width*0.3, 1)
            #oy = int(np.clip(oy,0,imw-224))

            # insert the crop into src.data[0]
            src.data[0] = image[:,ox:ox+224,oy:oy+224]

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

            make_step(net, end=layer, clip=clip, clip_mask=clip_gradient, 
                      focus=focus, sigma=sigma, step_size=step_size)

            if visualize:
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                showarray(vis,"./filename"+str(i)+".jpg")

            if i > 0 and i % 10 == 0:
                print 'finished step %d in octave %d' % (i,e)

            # insert modified image into image
            image[:,ox:ox+224,oy:oy+224] = src.data[0]
        
        print "octave %d image:" % e
        showarray(deprocess(net, image),"./octave_"+str(e)+".jpg")
            
    # returning the resulting image
    return deprocess(net, image)


octaves = [
    {
        'layer':'loss3/classifier',
        'iter_n':190*10,
        'start_sigma':2.5,
        'end_sigma':0.78,
        'start_step_size':12.,
        'end_step_size':9.
    },
    {
        'layer':'loss3/classifier',
        'scale':1.2,
        'iter_n':150*10,
        'start_sigma':0.78*1.2,
        'end_sigma':0.78,
        'start_step_size':9.,
        'end_step_size':6.
    },
    {
        'layer':'loss2/classifier',
        'scale':1.2,
        'iter_n':150*10,
        'start_sigma':0.78*1.2,
        'end_sigma':0.44,
        'start_step_size':6.,
        'end_step_size':3.
    },
    {
        'layer':'loss1/classifier',
        'scale':1.0,
        'iter_n':10*10,
        'start_sigma':0.44,
        'end_sigma':0.304,
        'start_step_size':3.,
        'end_step_size':2.
    }
]

start_dim = [900,900] # dimension of the picture we generate

focus = 970 # which class to "draw" with

original_w = net.blobs['data'].width
original_h = net.blobs['data'].height
background_color = np.float32([250.0, 250.0, 250.0])
gen_image = np.random.normal(background_color, 8, (start_dim[0], start_dim[1], 3))

# load circular clipping mask, used to avoid artifacts from square gradients
clip_img = PIL.Image.open("./clipping_masks/clipping_mask_circle.png")
clip_arr = np.asarray(clip_img.convert("L"))
clip_arr = 1. - clip_arr/255.


# We create a class to generate random points from a path. This class is then input to the deepdraw method and used to decide where to do a gradient descent step when generating the image. Overall this will have the effect of generating images along the shape of the path.
# 
# Note that drawing a path takes over 2 hours if processed on CPU, so it's recommended to have a caffe installation with GPU support for drawing paths.
# 

# simple generator for random sine path points
class SinePath:
    def __init__(self, x_start, x_end, y_origin, amplitude, cycles=1.):
        self.x_start = x_start
        self.x_end = x_end
        self.y_origin = y_origin
        self.amplitude = amplitude
        self.cycles = cycles
    def getPathPoint(self):
        rand_point = random.random()
        x_point = rand_point*(self.x_end-self.x_start) + self.x_start
        y_point = self.amplitude * math.sin(2*math.pi*rand_point*self.cycles) + self.y_origin
        return (y_point,x_point)

start_x = original_w/2. + 25
end_x = start_dim[0] - start_x
origin_y = start_dim[1]/2.
#amplitude = start_dim[1]-start_x
path = SinePath(start_x, end_x, origin_y, 100)

gen_image = np.random.normal(background_color, 8, (start_dim[0], start_dim[1], 3))

starttime = time.time()
gen_image = deepdraw(net, gen_image, octaves, focus, visualize=False, clip_gradient=clip_arr, pathfun=path)
print "took seconds : %f" % (time.time()-starttime)

# save image
#img_fn = '_'.join([model_name, "deepdraw_path", str(focus)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)


# simple generator for random circle path points
class CirclePath:
    def __init__(self, midpoint, radius):
        self.midpoint = midpoint
        self.radius = radius
    def getPathPoint(self):
        angle = random.random()*math.pi*2
        x = math.cos(angle)*self.radius + self.midpoint[0]
        y = math.sin(angle)*self.radius + self.midpoint[1]
        return (y,x)

mid = [start_dim[0]/2., start_dim[1]/2.]
path = CirclePath(mid, start_dim[0]*0.3)

gen_image = np.random.normal(background_color, 8, (start_dim[0], start_dim[1], 3))

starttime = time.time()
gen_image = deepdraw(net, gen_image, octaves, focus, visualize=False, clip_gradient=clip_arr, pathfun=path)
print "took seconds : %f" % (time.time()-starttime)

# save image
#img_fn = '_'.join([model_name, "deepdraw_path", str(focus)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)


# simple generator for random straight path points
class StraightPath:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def getPathPoint(self):
        randpoint = random.random()
        x = start[0] + randpoint*(end[0]-start[0])
        y = start[1] + randpoint*(end[1]-start[1])
        return (y,x)

start = [original_w/2. + 25,start_dim[1]/2. - 100]
end = [start_dim[0]-original_w/2.-25,start_dim[1]/2. + 100]
path = StraightPath(start,end)

gen_image = np.random.normal(background_color, 8, (start_dim[0], start_dim[1], 3))

starttime = time.time()
gen_image = deepdraw(net, gen_image, octaves, focus, visualize=False, clip_gradient=clip_arr, pathfun=path)
print "took seconds : %f" % (time.time()-starttime)

# save image
#img_fn = '_'.join([model_name, "deepdraw_path", str(focus)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)





# ## Visualizing VGG-S filters
# 
# This is an ipython notebook to generate visualizations of VGG-S filters, for some more info refer to [this blogpost](https://auduno.github.io/2016/06/18/peeking-inside-convnets/).
# 
# To run this code, you'll need an installation of Caffe with built pycaffe libraries, as well as the python libraries numpy, scipy and PIL. For instructions on how to install Caffe and pycaffe, refer to the installation guide [here](http://caffe.berkeleyvision.org/installation.html). Before running the ipython notebooks, you'll also need to download the [VGG-S model](https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9), and modify the variables ```pycaffe_root``` to refer to the path of your pycaffe installation (if it's not already in your python path) and ```model_path``` to refer to the path of the downloaded VGG-S caffe model. Also uncomment the line that enables GPU mode if you have built Caffe with GPU-support and a suitable GPU available.
# 

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys
from IPython.display import clear_output, Image, display
from scipy.misc import imresize

pycaffe_root = "/your/path/here/caffe/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

model_name="VGGS"
model_path = '/your/path/here/vggs_model/' # substitute your path here
net_fn   = './VGG_CNN_S_deploy_mod.prototxt' # added force_backward : true to prototxt
param_fn = model_path + 'VGG_CNN_S.caffemodel'
means = np.float32([104.0, 117.0, 123.0])

#caffe.set_mode_gpu() # uncomment this if gpu processing is available

net = caffe.Classifier(net_fn, param_fn,
                       mean = means, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def showarray(a, f, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def make_step(net, step_size=1.5, end='inception_4c/output', clip=True, focus=None, sigma=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob

    dst = net.blobs[end]
    net.forward(end=end)
    
    one_hot = np.zeros_like(dst.data)
    filter_shape = dst.data.shape
    if len(filter_shape) > 2:
        # backprop only activation in middle of filter
        one_hot[0,focus,(filter_shape[2]-1)/2,(filter_shape[3]-1)/2] = 1.
    else:
        one_hot.flat[focus] = 1.
    dst.diff[:] = one_hot
    
    net.backward(start=end)
    g = src.diff[0]
    
    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 
        
    src.data[0] = blur(src.data[0], sigma)
    
    dst.diff.fill(0.)

def deepdraw(net, base_img, octaves, random_crop=True, visualize=True, focus=None,
    clip=True, **step_params):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "starting drawing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
        _,imw,imh = image.shape
        
        # select layer
        layer = o['layer']
        
        for i in xrange(o['iter_n']):
            if imw > w:
                if random_crop:
                    # randomly select a crop 
                    #ox = random.randint(0,imw-224)
                    #oy = random.randint(0,imh-224)
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = np.random.normal(mid_x, width_x*0.3, 1)
                    ox = int(np.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = np.random.normal(mid_y, width_y*0.3, 1)
                    oy = int(np.clip(oy,0,imh-h))
                    # insert the crop into src.data[0]
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
                else:
                    ox = (imw-w)/2.
                    oy = (imh-h)/2.
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
            else:
                ox = 0
                oy = 0
                src.data[0] = image.copy()

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            
            make_step(net, end=layer, clip=clip, focus=focus, 
                      sigma=sigma, step_size=step_size)
            
            if visualize:
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                if i % 1 == 0:
                    showarray(vis,"./filename"+str(i)+".jpg")
            
            if i % 10 == 0:
                print 'finished step %d in octave %d' % (i,e)
            
            # insert modified image back into original image (if necessary)
            image[:,ox:ox+w,oy:oy+h] = src.data[0]
        
        print "octave %d image:" % e
        showarray(deprocess(net, image),"./octave_"+str(e)+".jpg")
            
    # returning the resulting image
    return deprocess(net, image)


octaves = [
    {
        'layer':'conv5',
        'iter_n':200,
        'start_sigma':2.5,
        'end_sigma':1.1,
        'start_step_size':12.*0.25,
        'end_step_size':10.*0.25,
    },
    {
        'layer':'conv5',
        'iter_n':100,
        'start_sigma':1.1,
        'end_sigma':0.78*1.1,
        'start_step_size':10.*0.25,
        'end_step_size':8.*0.25
    },
    {
        'layer':'conv5',
        'scale':1.05,
        'iter_n':100,
        'start_sigma':0.78*1.1,
        'end_sigma':0.78,
        'start_step_size':8.*0.25,
        'end_step_size':6.*0.25
    },
    {
        'layer':'conv5',
        'scale':1.05,
        'iter_n':50,
        'start_sigma':0.78*1.1,
        'end_sigma':0.40,
        'start_step_size':6.*0.25,
        'end_step_size':1.5*0.25
    },
    {
        'layer':'conv5',
        'scale':1.05,
        'iter_n':25,
        'start_sigma':0.4,
        'end_sigma':0.1,
        'start_step_size':1.5*0.25,
        'end_step_size':0.5*0.25
    }
]

# get original input size of network
original_w = net.blobs['data'].width
original_h = net.blobs['data'].height
# the background color of the initial image
background_color = np.float32([250.0, 250.0, 250.0])
# generate initial random image
gen_image = np.random.normal(background_color, 8, (original_w, original_h, 3))

# which filter in layer to visualize (conv5 has 512 filters)
imagenet_class = 10

# generate class visualization via octavewise gradient ascent
gen_image = deepdraw(net, gen_image, octaves, focus=imagenet_class, 
                 random_crop=True, visualize=False)

# save image
#img_fn = '_'.join([model_name, "deepdraw", str(imagenet_class)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)


# Code for "mixing" of class visualizations. See [this blogpost](http://auduno.com/post/125837418083/drawing-with-googlenet) for details.
# 
# Before running the code, insert your pycaffe path in ```pycaffe_root``` and insert the path to the googlenet model in ```model_path```. Download the bvlc googlenet model from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) if you haven't done so already.
# 

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys,time,string
from IPython.display import clear_output, Image, display
from scipy.misc import imresize

pycaffe_root = "/your/path/here/caffe/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

model_name = "GoogLeNet"
model_path = '/your/path/here/caffe_models/bvlc_googlenet/' # substitute your path here
net_fn   = '../deploy_googlenet_updated.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
mean = np.float32([104.0, 117.0, 123.0])

#caffe.set_mode_gpu() # uncomment this if gpu processing is available
net = caffe.Classifier(net_fn, param_fn,
                       mean = mean, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def showarray(a, f, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def make_step(net, step_size=1.5, end='inception_4c/output', clip=True, focus=None, sigma=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob

    dst = net.blobs[end]
    net.forward(end=end)

    one_hot = np.zeros_like(dst.data)
    one_hot.flat[focus] = 1.
    dst.diff[:] = one_hot

    net.backward(start=end)
    g = src.diff[0]
    
    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 
        
    src.data[0] = blur(src.data[0], sigma)
    
    dst.diff.fill(0.)
    
def deepdraw(net, base_img, octaves, focus, visualize=True, clip=True):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "starting drawing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    current_scale = 1.0
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
            current_scale *= o['scale']
        _,imw,imh = image.shape
        
        # select layer
        layer = o['layer']

        for i in xrange(o['iter_n']):
            for f in focus.keys():
                # randomly select a 224x224 crop centered on midpoint
                mid_x = focus[f][1]*current_scale - 224/2
                mid_y = focus[f][0]*current_scale - 224/2
                width = max((224*current_scale)-224,10.)
                ox = np.random.normal(mid_x, width*0.3, 1)
                ox = int(np.clip(ox,0,imw-224))
                oy = np.random.normal(mid_y, width*0.3, 1)
                oy = int(np.clip(oy,0,imw-224))
                
                # insert the crop into src.data[0]
                src.data[0] = image[:,ox:ox+224,oy:oy+224]

                sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
                step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

                make_step(net, end=layer, clip=clip, focus=f, sigma=sigma, step_size=step_size)

                if visualize:
                    vis = deprocess(net, src.data[0])
                    if not clip: # adjust image contrast if clipping is disabled
                        vis = vis*(255.0/np.percentile(vis, 99.98))
                    showarray(vis,"./filename"+str(i)+".jpg")
                
                if i > 0 and i % 10 == 0:
                    print 'finished step %d in octave %d for focus %d' % (i,e,f)

                # insert modified image into image
                image[:,ox:ox+224,oy:oy+224] = src.data[0]
        
        print "octave %d image:" % e
        showarray(deprocess(net, image),"./octave_"+str(e)+".jpg")
            
    # returning the resulting image
    return deprocess(net, image)


# Define the dimension of the final image via ```start_dim```, then define which classes and where to draw them via ```focus```.
# 

octaves = [
    {
        'layer':'loss3/classifier',
        'iter_n':190,
        'start_sigma':2.5,
        'end_sigma':0.78,
        'start_step_size':6.,
        'end_step_size':6.
    },
    {
        'layer':'loss3/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.78,
        'start_step_size':6.,
        'end_step_size':6.
    },
    {
        'layer':'loss2/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.44,
        'start_step_size':3.,
        'end_step_size':3.
    },
    {
        'layer':'loss1/classifier',
        'scale':1.0,
        'iter_n':10,
        'start_sigma':0.44,
        'end_sigma':0.304,
        'start_step_size':3.,
        'end_step_size':3.
    }
]

start_dim = [350,350] # dimension of the picture we generate

# mix of gibbon and poncho
focus = {
    735 : [175,200], # which class (key) and where to draw it (coordinates)
    368 : [175,150],
}

background_color = np.float32([250.0, 250.0, 250.0])
gen_image = np.random.normal(background_color, 8, (start_dim[0], start_dim[1], 3))

starttime = time.time()
gen_image = deepdraw(net, gen_image, octaves, focus, visualize=False)
print "it took %f seconds" % (time.time()-starttime)

# save image
#focs = [str(k) for k in focus.keys()]
#foc_fname = string.join(focs,"_")
#img_fn = '_'.join([model_name, "deepdraw_mixing_classes", foc_fname+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)


# mix of gorilla and french horn
focus = {
    566 : [175,175],
    366 : [175,150]
}

gen_image = np.random.normal(background_color, 8, (start_dim[0], start_dim[1], 3))
starttime = time.time()
gen_image = deepdraw(net, gen_image, octaves, focus, visualize=False)
print "it took %f seconds" % (time.time()-starttime)

# save image
#focs = [str(k) for k in focus.keys()]
#foc_fname = string.join(focs,"_")
#img_fn = '_'.join([model_name, "deepdraw_mixing_classes", foc_fname+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)





# Code to generate GoogLeNet class visualizations in the shape of a clipping mask. See [this blogpost](http://auduno.com/post/125837418083/drawing-with-googlenet) for details.
# 
# Before running the code, insert your pycaffe path in ```pycaffe_root``` and insert the path to the googlenet model in ```model_path```. Download the bvlc googlenet model from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) if you haven't done so already.
# 

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys,string,math,time
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
from scipy.misc import imresize

pycaffe_root = "/your/path/here/caffe/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

model_name = "GoogLeNet"
model_path = '/your/path/here/caffe_models/bvlc_googlenet/' # substitute your path here
net_fn   = '../deploy_googlenet_updated.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
mean = np.float32([104.0, 117.0, 123.0])

#caffe.set_mode_gpu() # uncomment this if gpu processing is available
net = caffe.Classifier(net_fn, param_fn,
                       mean = mean, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def showarray(a, f, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def make_step(net, step_size=1.5, end='inception_4c/output', clip=True, focus=None, sigma=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob

    dst = net.blobs[end]
    net.forward(end=end)

    one_hot = np.zeros_like(dst.data)
    one_hot.flat[focus] = 1.
    dst.diff[:] = one_hot

    net.backward(start=end)
    g = src.diff[0]
    
    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 
        
    src.data[0] = blur(src.data[0], sigma)
    
    dst.diff.fill(0.)

def deepdraw(net, base_img, octaves, focus, visualize=True, clip=True, clip_arr=None):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "starting drawing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    current_scale = 1.0
    for e,o in enumerate(octaves):
        if 'scale' in o:
            image = nd.zoom(image, (1,o['scale'],o['scale']))
            current_scale *= o['scale']
            # scale clip_mask as well
            clip_arr = np.abs(np.round(nd.zoom(clip_arr, (o['scale'],o['scale']))))
        _,imw,imh = image.shape
        
        # select layer
        layer = o['layer']

        for i in xrange(o['iter_n']):
            # randomly select a 224x224 crop
            ox = random.randint(0,imw-224)
            oy = random.randint(0,imh-224)

            # insert the crop into src.data[0]
            src.data[0] = image[:,ox:ox+224,oy:oy+224]

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

            make_step(net, end=layer, clip=clip, focus=focus, sigma=sigma, step_size=step_size)

            if visualize:
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                showarray(vis,"./filename"+str(i)+"_"+str(f)+".jpg")

            if i > 0 and i % 10 == 0:
                print 'finished step %d in octave %d' % (i,e)

            # insert modified image into image
            image[:,ox:ox+224,oy:oy+224] = src.data[0]

            # apply clipping mask
            if not clip_arr is None:
                image *= clip_arr
                image[np.where(np.tile(clip_arr,(3,1,1)) == 0.)] = 255.
        
        print "octave %d image:" % e
        showarray(deprocess(net, image),"./octave_"+str(e)+".jpg")
            
    # returning the resulting image
    return deprocess(net, image)


# We load the clipping mask as black and white image and convert it to a numeric clipping mask
# 

# get clipping mask
clip_img = PIL.Image.open("./clipping_masks/clipping_mask_a.png")
clip_arr = np.asarray(clip_img.convert("L"))
clip_arr = 1. - clip_arr/255.
clip_arr = np.round(clip_arr)

octaves = [
    {
        'layer':'loss3/classifier',
        'iter_n':190,
        'start_sigma':2.5,
        'end_sigma':0.78,
        'start_step_size':12.,
        'end_step_size':12.
    },
    {
        'layer':'loss3/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.78,
        'start_step_size':6.,
        'end_step_size':6.
    },
    {
        'layer':'loss2/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.44,
        'start_step_size':6.,
        'end_step_size':3.
    },
    {
        'layer':'loss1/classifier',
        'scale':1.0,
        'iter_n':10,
        'start_sigma':0.44,
        'end_sigma':0.304,
        'start_step_size':3.,
        'end_step_size':3.
    }
]

focus = 368 # capuchins

start_dim = [clip_arr.shape[0],clip_arr.shape[1]]
background_color = np.float32([250.0, 250.0, 250.0])
gen_image = np.random.normal(background_color, 8, (start_dim[0], start_dim[1], 3))

starttime = time.time()
gen_image = deepdraw(net, gen_image, octaves, focus, visualize=False, clip_arr=clip_arr)
print "took seconds : %f" % (time.time()-starttime)

# save image
#img_fn = '_'.join([model_name, "deepdraw_clipping_mask", str(focus)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)





# ###Visualizing classes with GoogLeNet
# 
# This is an ipython notebook to generate visualizations of classes with GoogLeNet, for some more info refer to [this blogpost](http://auduno.com/post/125362849838/visualizing-googlenet-classes), and for some examples of generated images see [this](https://goo.gl/photos/8qcvjnYBQVSGG2eN6) album of highlights or [this](https://goo.gl/photos/FfsZZektqpZkdDnKA) album of all 1000 imagenet classes.
# 
# To run this code, you'll need an installation of Caffe with built pycaffe libraries, as well as the python libraries numpy, scipy and PIL. For instructions on how to install Caffe and pycaffe, refer to the installation guide [here](http://caffe.berkeleyvision.org/installation.html). Before running the ipython notebooks, you'll also need to download the [bvlc_googlenet model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet), and modify the variables ```pycaffe_root``` to refer to the path of your pycaffe installation (if it's not already in your python path) and ```model_path``` to refer to the path of the googlenet caffe model. Also uncomment the line that enables GPU mode if you have built Caffe with GPU-support and a suitable GPU available.
# 

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys
from IPython.display import clear_output, Image, display
from scipy.misc import imresize

pycaffe_root = "/your/path/here/caffe/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

model_name = "GoogLeNet"
model_path = '/your/path/here/caffe_models/bvlc_googlenet/' # substitute your path here
net_fn   = './deploy_googlenet_updated.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
mean = np.float32([104.0, 117.0, 123.0])

#caffe.set_mode_gpu() # uncomment this if gpu processing is available
net = caffe.Classifier(net_fn, param_fn,
                       mean = mean, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def showarray(a, f, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


# Definition of the main gradient ascent functions. Note that these are based on the [deepdream code](https://github.com/google/deepdream/blob/master/dream.ipynb) published by Google as well as [this code](https://github.com/kylemcdonald/deepdream/blob/master/dream.ipynb) by Kyle McDonald.
# 

def make_step(net, step_size=1.5, end='inception_4c/output', clip=True, focus=None, sigma=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    
    dst = net.blobs[end]
    net.forward(end=end)

    one_hot = np.zeros_like(dst.data)
    one_hot.flat[focus] = 1.
    dst.diff[:] = one_hot

    net.backward(start=end)
    g = src.diff[0]
    
    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 
        
    src.data[0] = blur(src.data[0], sigma)
    
    # reset objective for next step
    dst.diff.fill(0.)

def deepdraw(net, base_img, octaves, random_crop=True, visualize=True, focus=None,
    clip=True, **step_params):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "starting drawing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
        _,imw,imh = image.shape
        
        # select layer
        layer = o['layer']

        for i in xrange(o['iter_n']):
            if imw > w:
                if random_crop:
                    # randomly select a crop 
                    #ox = random.randint(0,imw-224)
                    #oy = random.randint(0,imh-224)
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = np.random.normal(mid_x, width_x*0.3, 1)
                    ox = int(np.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = np.random.normal(mid_y, width_y*0.3, 1)
                    oy = int(np.clip(oy,0,imh-h))
                    # insert the crop into src.data[0]
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
                else:
                    ox = (imw-w)/2.
                    oy = (imh-h)/2.
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
            else:
                ox = 0
                oy = 0
                src.data[0] = image.copy()

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            
            make_step(net, end=layer, clip=clip, focus=focus, 
                      sigma=sigma, step_size=step_size)

            if visualize:
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                if i % 1 == 0:
                    showarray(vis,"./filename"+str(i)+".jpg")
            
            if i % 10 == 0:
                print 'finished step %d in octave %d' % (i,e)
            
            # insert modified image back into original image (if necessary)
            image[:,ox:ox+w,oy:oy+h] = src.data[0]
        
        print "octave %d image:" % e
        showarray(deprocess(net, image),"./octave_"+str(e)+".jpg")
            
    # returning the resulting image
    return deprocess(net, image)


# #### Generating the class visualizations
# 
# The ```octaves``` list determines in which order we optimize layers, as well as how many iterations and scaling on each octave. For each octave, parameters are:
# * ```layer``` : which layer to optimize
# * ```iter_n``` : how many iterations
# * ```scale``` : by what factor (if any) to scale up the base image before proceeding
# * ```start_sigma``` : the initial radius of the gaussian blur
# * ```end_sigma``` : the final radius of the gaussian blur
# * ```start_step_size``` : the initial step size of the gradient ascent
# * ```end_step_size``` : the final step size of the gradient ascent
# 
# The choice of octave parameters below will give decent images, and is the one used for visualizations in the blogpost. However, the choice of parameters was a bit arbitrary, so feel free to experiment. Note that generating an image will take around 1 minute with GPU-enabled Caffe, or 10-15 minutes if you're running purely on CPU, depending on your computer performance.
# 

# these octaves determine gradient ascent steps
octaves = [
    {
        'layer':'loss3/classifier',
        'iter_n':190,
        'start_sigma':2.5,
        'end_sigma':0.78,
        'start_step_size':11.,
        'end_step_size':11.
    },
    {
        'layer':'loss3/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.78,
        'start_step_size':6.,
        'end_step_size':6.
    },
    {
        'layer':'loss2/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.44,
        'start_step_size':6.,
        'end_step_size':3.
    },
    {
        'layer':'loss1/classifier',
        'iter_n':10,
        'start_sigma':0.44,
        'end_sigma':0.304,
        'start_step_size':3.,
        'end_step_size':3.
    }
]

# get original input size of network
original_w = net.blobs['data'].width
original_h = net.blobs['data'].height
# the background color of the initial image
background_color = np.float32([200.0, 200.0, 200.0])
# generate initial random image
gen_image = np.random.normal(background_color, 8, (original_w, original_h, 3))

# which imagenet class to visualize
imagenet_class = 13

# generate class visualization via octavewise gradient ascent
gen_image = deepdraw(net, gen_image, octaves, focus=imagenet_class, 
                 random_crop=True, visualize=False)

# save image
#img_fn = '_'.join([model_name, "deepdraw", str(imagenet_class)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)


# This choice of octave parameters tends to give more coherent images, but has a little bit less detail.
# 

octaves = [
    {
        'layer':'loss3/classifier',
        'iter_n':190,
        'start_sigma':2.5,
        'end_sigma':0.78,
        'start_step_size':11.,
        'end_step_size':11.
    },
    {
        'layer':'loss3/classifier',
        'scale':1.2,
        'iter_n':450,
        'start_sigma':0.78*1.2,
        'end_sigma':0.40,
        'start_step_size':6.,
        'end_step_size':3.
    }
]
imagenet_class = 244
gen_image = np.random.normal(background_color, 8, (original_w, original_h, 3))
gen_image = deepdraw(net, gen_image, octaves, focus=imagenet_class, 
                 random_crop=True, visualize=False)

#img_fn = '_'.join([model_name, "deepdraw", str(imagenet_class)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)





