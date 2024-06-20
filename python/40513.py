# # Introduction to Cloud Machine Learning with Flask API and CNTK
# 
# One of the best ways to operationalize a machine learning system is through an API. In this notebook we show how to deploy scalable CNTK models for image classification through an API. The frameworks used in the solution are:
# 
# * [CNTK](https://github.com/Microsoft/CNTK/): Microsoft's Cognitive Toolkit is the deep learning library we used to compute the [Convolutional Neural Network](https://miguelgfierro.com/blog/2016/a-gentle-introduction-to-convolutional-neural-networks/) (CNN) model that identifies images.
# * [Flask](http://flask.pocoo.org/) is one of the most popular frameworks to develop APIs in python. 
# * [CherryPy](http://cherrypy.org/) is a lightweight web framework for python. We use it as a web server to host the machine learning application.  
# 
# Here we present an overview of the application. The main procedure is executed by the CNTK CNN. The network is a [pretrained ResNet with 152 layers](https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet.model). The CNN was trained on [ImageNet dataset](http://www.image-net.org/), which contains 1.2 million images divided into 1000 different classes. The CNN is accessible through the flask API, which provides an end point `/api/v1/classify_image` that can be called to classify an image. CherryPy is is the server framework where the application is hosted. It also balances the load, in such a way that several concurrent queries can be executed. Externally, there is the client that can be any desktop or mobile that sends an image to the application to be analyzed and receives the response.   
# 
# <p style="center;">
# <img src="https://miguelgfierro.com/img/upload/2017/04/17/api_overview_350.png" />
# </p>
# 
# 
# 
# 

#load libraries
import os,sys
import pkg_resources
from flask import Flask, render_template, request, send_file
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import wget
import numpy as np
from PIL import Image, ImageOps
from urllib.request import urlretrieve
import requests
from cntk import load_model, combine
from io import BytesIO, StringIO
import base64
from IPython.core.display import display, HTML
import aiohttp
import asyncio
import json
import random

print("System version: {}".format(sys.version))
print("Flask version: {}".format(pkg_resources.get_distribution("flask").version))
print("CNTK version: {}".format(pkg_resources.get_distribution("cntk").version))


# ## Image classification with a pretrained CNTK model
# The first step is to download a pretrained model. CNTK has a wide range of [different pretrained models](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Classification) that can be used for image classification.
# 
# 

def maybe_download_model(filename='ResNet_18.model'):
    if(os.path.isfile(filename)):
        print("Model %s already downloaded" % filename)
    else:
        model_name_to_url = {
        'AlexNet.model':   'https://www.cntk.ai/Models/AlexNet/AlexNet.model',
        'AlexNetBS.model': 'https://www.cntk.ai/Models/AlexNet/AlexNetBS.model',
        'VGG_16.model': 'https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet.model',
        'VGG_19.model': 'https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet.model',
        'InceptionBN.model': 'https://www.cntk.ai/Models/Caffe_Converted/BNInception_ImageNet.model',
        'ResNet_18.model': 'https://www.cntk.ai/Models/ResNet/ResNet_18.model',
        'ResNet_50.model': 'https://www.cntk.ai/Models/Caffe_Converted/ResNet50_ImageNet.model',
        'ResNet_101.model': 'https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet.model',
        'ResNet_152.model': 'https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet.model'
        }
        url = model_name_to_url[filename] 
        wget.download(url, out=filename)


# For this example we are going to use ResNet with 152 layers, which has a top-5 error of 6.71% in ImageNet.
# 

get_ipython().run_cell_magic('time', '', "model_name = 'ResNet_152.model'\nIMAGE_MEAN = 0 # in case the CNN rests the mean for the image\nmaybe_download_model(model_name)")


# Together with the model, we need the classification information. The [synsets file](synsets.txt) maps the output of the network, which is an number between 0 and 999, with the class name.
# 

def read_synsets(filename='synsets.txt'):
    with open(filename, 'r') as f:
        synsets = [l.rstrip() for l in f]
        labels = [" ".join(l.split(" ")[1:]) for l in synsets]
    return labels

labels = read_synsets()
print("Label length: ", len(labels))
print(labels[:5])


# Next we are going to prepare some helper functions to read images with PIL and plot them.
# 

def read_image_from_file(filename):
    img = Image.open(filename)
    return img
def read_image_from_ioreader(image_request):
    img = Image.open(BytesIO(image_request.read())).convert('RGB')
    return img
def read_image_from_request_base64(image_base64):
    img = Image.open(BytesIO(base64.b64decode(image_base64)))
    return img
def read_image_from_url(url):
    img = Image.open(requests.get(url, stream=True).raw)
    return img


def plot_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Let's test the different input images and plot them.
# 

imagepath = 'neko.jpg'
img_cat = read_image_from_file(imagepath)
plot_image(img_cat)


imagefile = open(imagepath, 'rb')
print(type(imagefile))
img = read_image_from_ioreader(imagefile)
plot_image(img)


imagefile = open(imagepath, 'rb')
image_base64 = base64.b64encode(imagefile.read())
print("String of %d characters" % len(image_base64))
img = read_image_from_request_base64(image_base64)
plot_image(img)


imageurl = 'https://pbs.twimg.com/profile_images/269279233/llama270977_smiling_llama_400x400.jpg'
img_llama = read_image_from_url(imageurl)
plot_image(img_llama)


# Once we have the image, the model file and the sysntets, the next step is to load the model and perform a prediction. We need to process the image to swap the RGB channels and resize to the input size of ImageNet, which is `224x224`.
# 

get_ipython().run_cell_magic('time', '', 'z = load_model(model_name)')


def softmax(vect):
    return np.exp(vect) / np.sum(np.exp(vect), axis=0)


def get_preprocessed_image(my_image, mean_image):
    #Crop and center the image
    my_image = ImageOps.fit(my_image, (224, 224), Image.ANTIALIAS)
    #Transform the image for CNTK format
    my_image = np.array(my_image, dtype=np.float32)
    # RGB -> BGR
    bgr_image = my_image[:, :, ::-1] 
    image_data = np.ascontiguousarray(np.transpose(bgr_image, (2, 0, 1)))
    image_data -= mean_image
    return image_data


def predict(model, image, labels, number_results=5):
    img = get_preprocessed_image(image, IMAGE_MEAN)
    # Use last layer to make prediction
    arguments = {model.arguments[0]: [img]}
    result = model.eval(arguments)
    result = np.squeeze(result)
    prob = softmax(result)
    # Sort probabilities 
    prob_idx = np.argsort(result)[::-1][:number_results]
    pred = [labels[i] for i in prob_idx]
    return pred
 


# Now let's predict the class of some of the images
# 

resp = predict(z, img_llama, labels, 2)
print(resp)
resp = predict(z, img_cat, labels, 3)
print(resp)
resp = predict(z, read_image_from_url('http://www.awf.org/sites/default/files/media/gallery/wildlife/Hippo/Hipp_joe.jpg'), labels, 5)
print(resp)


# ## Set up Flask API
# 
# Let´s start the flask server. The code can be found in the file [cntk_api.py](cntk_api.py). To start it in localhost, first set `DEVELOPMENT=True` in the file [config.py](config.py) and execute it inside a cntk environment:
# 
# ```bash
# source activate my-cntk-env
# python cntk_api.py
# ```
# You will get something like this:
# ```bash
# * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
# * Restarting with stat
# * Debugger is active!
# ```
# First we will test that the API works locally, for that we created a sample web page [hello.html](templates/hello.html) that is going to be rendered when you call the API root.
# 
# The code for this operation is as simple as this:
# ```python
# @app.route('/')
# def hello():
#     return render_template('hello.html')
# ```
# 
# In order to execute the api from this notebook you can use the magic background function executing the file using the binary in the cntk environment:
# 

get_ipython().run_cell_magic('bash', '--bg ', '/home/my-user/anaconda3/envs/my-cntk-env/bin/python /home/my-user/sciblog_support/Intro_to_Machine_Learning_API/cntk_api.py')


res = requests.get('http://127.0.0.1:5000/')
display(HTML(res.text))


# Flask allows for simple routing of services. Let's create the main end point `/api/v1/classify_image`. The service accepts an image in bytes or in a url. Both requests are converted to a PIL image. In case the request is incorrect the API returns a bad request. The image is analyzed using `predict` method, that returns the top-5 classification results given the CNN model and the labels. Finally everything is formatted as a json.
# 
# ```python
# @app.route('/api/v1/classify_image', methods=['POST'])
# def classify_image():
#     if 'image' in request.files:
#         cherrypy.log("Image request")
#         image_request = request.files['image']
#         img = read_image_from_ioreader(image_request)
#     elif 'url' in request.json: 
#         cherrypy.log("JSON request: {}".format(request.json['url']))
#         image_url = request.json['url']
#         img = read_image_from_url(image_url)
#     else:
#         cherrypy.log("Bad request")
#         abort(BAD_REQUEST)
#     resp = predict(model, img, labels, 5)
#     return make_response(jsonify({'message': resp}), STATUS_OK)
# ```
# 
# 
# Let's first force a bad request:
# 

headers = {'Content-type':'application/json'}
data = {'param':'1'}
res = requests.post('http://127.0.0.1:5000/api/v1/classify_image', data=json.dumps(data), headers=headers)
print(res.text)


# Now we are going to use the end point with a image from an URL.
# 

get_ipython().run_cell_magic('time', '', "imageurl = 'https://pbs.twimg.com/profile_images/269279233/llama270977_smiling_llama_400x400.jpg'\ndata = {'url':imageurl}\nres = requests.post('http://127.0.0.1:5000/api/v1/classify_image', data=json.dumps(data), headers=headers)\nprint(res.text)")


# Finally, we are going to test the API with an image loaded from disk.
# 

get_ipython().run_cell_magic('time', '', "imagepath = 'neko.jpg'\nimage_request = open(imagepath, 'rb')\nfiles_local = {'image': image_request}\nres = requests.post('http://127.0.0.1:5000/api/v1/classify_image', files=files_local)\nprint(res.text)")


# ## CNTK API with CherryPy
# 
# There are multiple solutions to setup a production API with Flask. One option is using [Nginx](https://www.nginx.com/) server and [Gunicorn](http://gunicorn.org/) as a load balancer. Here you can find [a great example](https://github.com/ilkarman/CV_end_to_end/blob/master/01-VM/FlaskVM.md) combining these two technologies to serve machine learning models. Another way is to use [Apache](https://en.wikipedia.org/wiki/Apache_HTTP_Server) like in [this example](https://www.vioan.eu/blog/2016/10/10/deploy-your-flask-python-app-on-ubuntu-with-apache-gunicorn-and-systemd/). Here a simple flask application is set up using Apache server and Gunicorn. 
# 
# In this notebook we are going to use [CherryPy](http://cherrypy.org/). It has the following features that the authors announce in their web page:
# 
# * A reliable, HTTP/1.1-compliant, WSGI thread-pooled webserver. WSGI stands for Web Server Gateway Interface. It is a specification that describes how a web server communicates with web applications, and how web applications can be chained together to process one request.
# * CherryPy is now more than ten years old and it is has proven to be very fast and stable. In [this benchmark](https://blog.appdynamics.com/engineering/a-performance-analysis-of-python-wsgi-servers-part-2/) CherryPy is tested against several other solutions.
# * Built-in profiling, coverage, and testing support.
# * Powerful and easy-to-use [configuration system](http://docs.cherrypy.org/en/latest/basics.html#config). 
# 
# All these features make CherryPy a good solution for quickly develop production APIs.
# 
# The code to set up the server is fairly simple:
# 
# ```python
# def run_server():
#     # Enable WSGI access logging via Paste
#     app_logged = TransLogger(app)
# 
#     # Mount the WSGI callable object (app) on the root directory
#     cherrypy.tree.graft(app_logged, '/')
# 
#     # Set the configuration of the web server
#     cherrypy.config.update({
#         'engine.autoreload_on': True,
#         'log.screen': True,
#         'log.error_file': "cherrypy.log",
#         'server.socket_port': PORT,
#         'server.socket_host': '0.0.0.0',
#         'server.thread_pool': 50, # 10 is default
#     })
# 
#     # Start the CherryPy WSGI web server
#     cherrypy.engine.start()
#     cherrypy.engine.block()
# ```
# 
# To start the server, we need to set `DEVELOPMENT=False` in the file [config.py](config.py) and execute it inside a cntk environment:
# 
# ```bash
# source activate my-cntk-env
# python cntk_api.py
# ```
# You will get something like this:
# 
# ```bash
# [16/Apr/2017:17:52:43] ENGINE Bus STARTING
# [16/Apr/2017:17:52:43] ENGINE Started monitor thread 'Autoreloader'.
# [16/Apr/2017:17:52:43] ENGINE Started monitor thread '_TimeoutMonitor'.
# [16/Apr/2017:17:52:43] ENGINE Serving on http://0.0.0.0:5000
# [16/Apr/2017:17:52:43] ENGINE Bus STARTED
# ```
# The first step is to test if the root end point is working:
# 

server_name = 'http://the-name-of-your-server'
port = 5000


root_url = '{}:{}'.format(server_name, port)


res = requests.get(root_url)
display(HTML(res.text))


# Now, as we did before, let's test the classification API with an image from a URL and an image from bytes.
# 

end_point = root_url + '/api/v1/classify_image' 
#print(end_point)


get_ipython().run_cell_magic('time', '', "imageurl = 'https://pbs.twimg.com/profile_images/269279233/llama270977_smiling_llama_400x400.jpg'\ndata = {'url':imageurl}\nheaders = {'Content-type':'application/json'}\nres = requests.post(end_point, data=json.dumps(data), headers=headers)\nprint(res.text)")


get_ipython().run_cell_magic('time', '', "imagepath = 'neko.jpg'\nimage_request = open(imagepath, 'rb')\nfiles = {'image': image_request}\nres = requests.post(end_point, files=files)\nprint(res.text)")


# ## Bombardment
# Finally let's do a funny part. Now that we have the API set up with CherryPy, let's test how the system performs under a big number of concurrent requests. The first step is to select the request to execute. We are going to use this handsome hippo.
# 

# Get hippo
hippo_url = "http://www.awf.org/sites/default/files/media/gallery/wildlife/Hippo/Hipp_joe.jpg"

fname = urlretrieve(hippo_url, "bhippo.jpg")[0]
img_bomb = read_image_from_file(fname)
plot_image(img_bomb)


# Next, we define the number of request and how many of them are concurrent.
# 

NUM = 100
concurrent = 10


# We prepare the images for the load test. For that we prepare `NUM` different images, which are modified by one pixel.
# 

def gen_variations_of_one_image(num, filename):
    out_images = []
    imagefile = open(filename, 'rb')
    img = Image.open(BytesIO(imagefile.read())).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    # Flip the colours for one-pixel
    # "Different Image"
    for i in range(num):
        diff_img = img.copy()
        rndm_pixel_x_y = (random.randint(0, diff_img.size[0]-1), 
                          random.randint(0, diff_img.size[1]-1))
        current_color = diff_img.getpixel(rndm_pixel_x_y)
        diff_img.putpixel(rndm_pixel_x_y, current_color[::-1])
        # Turn image into IO
        ret_imgio = BytesIO()
        diff_img.save(ret_imgio, 'PNG')
        out_images.append(ret_imgio.getvalue())
    return out_images


get_ipython().run_cell_magic('time', '', "# Save same file multiple times in memory as IO\nimages = gen_variations_of_one_image(NUM, fname)\nurl_list = [[end_point, {'image':pic}] for pic in images]")


# In the following series of functions we define an asynchronous request service. It is in charge of bombarding the end point. 
# 

def handle_req(data):
    return json.loads(data.decode('utf-8'))
 
def chunked_http_client(num_chunks, s):
    # Use semaphore to limit number of requests
    semaphore = asyncio.Semaphore(num_chunks)
    @asyncio.coroutine
    # Return co-routine that will work asynchronously and respect
    # locking of semaphore
    def http_get(dta):
        nonlocal semaphore
        with (yield from semaphore):
            url, img = dta
            response = yield from s.request('post', url, data=img)
            body = yield from response.content.read()
            yield from response.wait_for_close()
        return body
    return http_get

    
def run_experiment(urls, _session):
    http_client = chunked_http_client(num_chunks=concurrent, s=_session)
    
    # http_client returns futures, save all the futures to a list
    tasks = [http_client(url) for url in urls]
    dfs_route = []
    
    # wait for futures to be ready then iterate over them
    for future in asyncio.as_completed(tasks):
        data = yield from future
        try:
            out = handle_req(data)
            dfs_route.append(out)
        except Exception as err:
            print("Error {0}".format(err))
    return dfs_route


# Finally, let's run the scalability test. If you take a look at the terminal, you will see the logs with all the requests being executed.
# 

get_ipython().run_cell_magic('time', '', "# Expect to see some 'errors' meaning requests are expiring on 'queue'\n# i.e. we can't increase concurrency any more\nwith aiohttp.ClientSession() as session:  # We create a persistent connection\n    loop = asyncio.get_event_loop()\n    complete_responses = loop.run_until_complete(run_experiment(url_list, session)) ")


print("Number of sucessful queries: {} of {}".format(len(complete_responses), NUM))
print(complete_responses[:5])


# ## Conclusion
# 
# In this notebook we have shown how to deploy an API for image classification using deep learning. We have used Microsoft's Cognitive Toolkit (CNTK), Flask and CherryPy. The solution is easy to implement and ready for production environments. Happy coding!
# 

# # Introduction to Convolutional Neural Networks
# 
# Convolutional Neural Networks (CNN) are one of the key components in the success of Deep Learning and the new Artificial Intelligence revolution. They are specially advantageous in tasks such as object detection, scene understanding and, recently, natural language processing. In this jupyter notebook I will explain what is a convolution and how to train a CNN with the character recognition dataset MNIST.
# 
# This jupyter notebook is a support for the article in my blog: [A Gentle Introduction to Convolutional Neural Networks](https://miguelgfierro.com/blog/2016/a-gentle-introduction-to-convolutional-neural-networks/).
# 

# Load all needed libraries
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import time
import cv2
import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# ## Convolution
# 
# CNN are the key resource in deep learning. They are based on a mathematical operation called convolution. A convolution is just a multiplication of an input image (which is a matrix) times a kernel (which is another matrix).
# 
# In opencv there is a function called filter2D that allows to generate convolutions. 
# 

def plot_image(image, image2=None):
    # Show one image
    plt.subplot(121)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        plt.imshow(image, cmap = plt.get_cmap('gray'))
    plt.axis("off")
    plt.xticks([]), plt.yticks([])
    if image2 is not None:
        # Show two images
        plt.subplot(122)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            plt.imshow(image2)
        else:
            plt.imshow(image2, cmap = plt.get_cmap('gray'))
        plt.axis("off") 
        plt.xticks([]), plt.yticks([])
    plt.show()


im = cv2.imread("Lenna.png")
plot_image(im)


## Sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")
# Laplacian kernel used to detect edges
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")
# Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
# Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")


dst = cv2.filter2D(im,-1,sharpen)
plot_image(im, dst)


dst = cv2.filter2D(im,-1,laplacian)
plot_image(im, dst)


dst = cv2.filter2D(im,-1,sobelX)
plot_image(im, dst)


dst = cv2.filter2D(im,-1,sobelY)
plot_image(im, dst)


# ## Character recognition with MNIST dataset
# 
# We are going to use MNIST dataset and Lenet CNN architecture to showcase a deep learning task consisting in recognizing handwritting characters. 
# 
# [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains a training set of 60,000 examples, and a test set of 10,000 examples of hand writting characters.
# 

mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p]
Y = mnist.target[p]

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X[i].reshape((28,28)), cmap='Greys_r')
    plt.axis('off')
plt.show()

X = X.astype(np.float32)/255
X_train = X[:60000].reshape((-1, 1, 28, 28))
X_test = X[60000:].reshape((-1, 1, 28, 28))
Y_train = Y[:60000]
Y_test = Y[60000:]

batch_size = 100
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)


# ## CNN model: Lenet
# 
# Lenet architecture was published by [Yann LeCun et al.](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) in 1998. The LeNet architecture uses convolutions and pooling to increase the performance. For many years, LeNet was the most accurate algorithm for character recognition and supposed a great advance in deep neural networks, long before the appearance of GPUs and CUDA.
# 
# As it can be seen in the following code Lenet has 4 groups of hidden layers, two convolutions and two fully connected layers. Each convolution is followed by an activation and a pooling. 
# 

# Network symbolic representation
data = mx.symbol.Variable('data')
input_y = mx.sym.Variable('softmax_label')  # placeholder for output

conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2)) 

flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500) 
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10) 
lenet = mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")

# Lenet visualization
mx.viz.plot_network(lenet)


# ## Training
# Once we have the model and the data, let's put it together and train the CNN.
# 

model = mx.model.FeedForward(
    ctx = mx.cpu(),      # Run on CPU (can also use GPU: ctx = mx.gpu(0))
    symbol = lenet,       # Use the network we just defined
    num_epoch = 10,       # Train for 10 epochs
    learning_rate = 0.1,  # Learning rate
    optimizer = 'sgd',    # The optimization method is Stochastic Gradient Descent
    momentum = 0.9,       # Momentum for SGD with momentum
    wd = 0.00001)         # Weight decay for regularization

tic = time.time()
model.fit(
    X = train_iter,  # Training data set
    eval_data = test_iter,  # Testing data set. MXNet computes scores on test set every epoch
    eval_metric = ['accuracy'],  # Metric for evaluation: accuracy. Other metrics can be defined
    batch_end_callback = mx.callback.Speedometer(batch_size, 200))  # Logging module to print out progress
print("Finished training in %.0f seconds" % (time.time() - tic))


# # Visualization of Football Matches using the Lean Startup Method
# 
# [Lean Startup](https://en.wikipedia.org/wiki/Lean_startup) has been a breakthrough during the last years in the entrepreneurial landscape. But this movement is not only a methodology that can be used to create successful businesses, it can also be applied to other domains like Data Science. In this [post](https://miguelgfierro.com/blog/2016/how-to-develop-a-data-science-project-using-the-lean-startup-method/), I discuss how to apply the Lean Startup method to a Data Science project. As an example, I created a project that visualizes all football matches that took place in the UEFA Champions League since 1955.
# 
# There are several methodologies to develop data science projects. At Microsoft, we developed and intensively use the [Team Data Science Process](https://azure.microsoft.com/en-us/documentation/learning-paths/data-science-process/) (TDSP). This methodology enables us to efectively implement projects while collaborating between teams inside and outside the company.
# 
# In some situations, when starting a Data Science project, we have a clear view of the business case, access to all customer data and a clear roadmap of what the customer wants. In that situation, the Lean Startup process has little value. However, when there is uncertainty, when the customer doesn't know what he wants or when we don't know if a product is going to be sucessful, then the Lean Startup method can prove its benefit.
# 
# Implementing the Lean Startup method in the TDSP is easy, we can use all the tools TDSP proposes. In the Lean Startup method the priority is to reduce (or eliminate) the uncertainty to understand what the customer really wants.
# 
# So following the Lean Startup method, first we have to set the hyphotesis. The next step is to build a [Minimun Viable Product](https://en.wikipedia.org/wiki/Minimum_viable_product) (MVP). The MVP has two important features, it helps us to validate the hypothesis or ideas we proposed and it is a complete, end to end product with the minimum number of features. The final step is to show the MVP to the customer and measure its impact.
# 

#Load all libraries
import os,sys  
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
from datashader import reductions
from datashader.colors import colormap_select, Hot, inferno
from datashader.bokeh_ext import InteractiveImage
from bokeh.palettes import Greens3, Blues3, Blues4, Blues9, Greys9
from bokeh.plotting import figure, output_notebook
from bokeh.tile_providers import WMTSTileSource, STAMEN_TONER, STAMEN_TERRAIN
from functools import partial
import wget
import zipfile
import math
from difflib import SequenceMatcher

output_notebook()
#print(sys.path)
print(sys.version)


# ## Visualization of Champions League matches
# Visualization is one of the key parts in a Data Science project. It allows us to get a global sense of our data and to understand better our results. 
# 
# There are many free and non-free tools in the market to make data visualization. One of my favourites is [datashader](https://github.com/bokeh/datashader), an open source python library that allows to visualize big amounts of data with a clean and nice API. 
# 
# ### MVP1: Initial match visualization
# 
# We can easily create a visualization of the Champion League matches from 1955 to 2016 using datashader. For that we need a dataset of the matches, such as [this one](https://github.com/jalapic/engsoccerdata/blob/master/data-raw/champs.csv) and the coordinates of the stadiums of the teams, that you can find [here](http://opisthokonta.net/?cat=34). This last dataset has the stadiums coordinates only of teams form England, Spain, France, Germany and Scotland. We will still need the data from other countries such as Italy, Portugal, Netherlands and many others, but for our first MVP we don't care about it, our objective is to reach a minimum product as fast as possible to reduce uncertainty, and 5 countries is enough for now. 
# 
# The first step is to treat the data.
# 

df_stadium = pd.read_csv("stadiums.csv", usecols=['Team','Stadium','Latitude','Longitude','Country'])
print("Number of rows: %d" % df_stadium.shape[0])
dd1 = df_stadium.take([0,99, 64, 121])
dd1


# The next step is to match the club names in the dataset of coordinates with the those in the dataset of matches. They are similar but not always exactly the same, for example, in the dataset of coordinates we have `Real Madrid FC` and in the dataset of matches we have `Real Madrid`. Furthermore, in the first one there are several entries for some teams, like `Atletico Madrid`, `Atletico Madrid B` or `Atletico Madrid C` meaning they are the teams from the first division and from other divisions. 
# 

df_match = pd.read_csv('champions.csv', usecols=['Date','home','visitor','hcountry','vcountry'])
df_match = df_match.rename(columns = {'hcountry':'home_country', 'vcountry':'visitor_country'})
df_teams_champions = pd.concat([df_match['home'], df_match['visitor']])
teams_champions = set(df_teams_champions)
print("Number of teams that have participated in the Champions League: %d" % len(teams_champions))
print("Number of matches in the dataset: %d" % df_match.shape[0])
df_match.head()


# To find the string similarity you can use different methods. Here we will use a simple method to calculate it with `difflib`.
# 

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_info_similar_team(team, df_stadium, threshold=0.6, verbose=False):
    max_rank = 0
    max_idx = -1
    stadium = "Unknown"
    latitude = np.NaN
    longitude = np.NaN
    for idx, val in enumerate(df_stadium['Team']):
        rank = similar(team, val)
        if rank > threshold:
            if(verbose): print("%s and %s(Idx=%d) are %f similar." % (team, val, idx, rank))
            if rank > max_rank:
                if(verbose): print("New maximum rank: %f" %rank)
                max_rank = rank
                max_idx = idx
                stadium = df_stadium['Stadium'].iloc[max_idx]
                latitude = df_stadium['Latitude'].iloc[max_idx]
                longitude = df_stadium['Longitude'].iloc[max_idx]
    return stadium, latitude, longitude
print(get_info_similar_team("Real Madrid FC", df_stadium, verbose=True))
print(get_info_similar_team("Atletico de Madrid FC", df_stadium, verbose=True))
print(get_info_similar_team("Inter Milan", df_stadium, verbose=True))
 


# The next step is to create a dataframe relating each match with the stadium coordinates of each team
# 

get_ipython().run_cell_magic('time', '', "df_match_stadium = df_match\nhome_stadium_index = df_match_stadium['home'].map(lambda x: get_info_similar_team(x, df_stadium))\nvisitor_stadium_index = df_match_stadium['visitor'].map(lambda x: get_info_similar_team(x, df_stadium))\ndf_home = pd.DataFrame(home_stadium_index.tolist(), columns=['home_stadium', 'home_latitude', 'home_longitude'])\ndf_visitor = pd.DataFrame(visitor_stadium_index.tolist(), columns=['visitor_stadium', 'visitor_latitude', 'visitor_longitude'])\ndf_match_stadium = pd.concat([df_match_stadium, df_home, df_visitor], axis=1, ignore_index=False)")


print("Number of missing values for home teams: %d out of %d" % (df_match_stadium['home_stadium'].value_counts()['Unknown'], df_match_stadium.shape[0]))
df1 = df_match_stadium['home_stadium'] == 'Unknown'
df2 = df_match_stadium['visitor_stadium'] == 'Unknown'
n_complete_matches = df_match_stadium.shape[0] - df_match_stadium[df1 | df2].shape[0]
print("Number of matches with complete data: %d out of %d" % (n_complete_matches, df_match_stadium.shape[0]))
df_match_stadium.head()


# Now, even though there are many entries in the dataset that don't have any value, we are going to create a dataframe with the teams that do have values and advance in the project. This dataframe finds the combination of teams (home and visitor) that have values and concatenate each other to create the map.
# 

def aggregate_dataframe_coordinates(dataframe):
    df = pd.DataFrame(index=np.arange(0, n_complete_matches*3), columns=['Latitude','Longitude'])
    count = 0
    for ii in range(dataframe.shape[0]):
        if dataframe['home_stadium'].loc[ii]!= 'Unknown' and dataframe['visitor_stadium'].loc[ii]!= 'Unknown':
            df.loc[count] = [dataframe['home_latitude'].loc[ii], dataframe['home_longitude'].loc[ii]]
            df.loc[count+1] = [dataframe['visitor_latitude'].loc[ii], dataframe['visitor_longitude'].loc[ii]]
            df.loc[count+2] = [np.NaN, np.NaN]
            count += 3
    return df
df_agg = aggregate_dataframe_coordinates(df_match_stadium)
df_agg.head()


# We have to transform the latitude and longitude coordinates to [web mercator](https://en.wikipedia.org/wiki/Web_Mercator) format in order to be able to represent it in a map using bokeh. Mercator coordinates are a cilindrical projection of the World coordinates. It was invented in 1569 by [Gerardus Mercator](https://en.wikipedia.org/wiki/Mercator_projection) and became the standard format for nautical purposes. The web mercator format is an adaptation of the original mercator format and it is currently used by most modern map systems such as Google Maps, Bing Maps or OpenStreetMaps.
# 

def to_web_mercator(yLat, xLon):
    # Check if coordinate out of range for Latitude/Longitude
    if (abs(xLon) > 180) and (abs(yLat) > 90):  
        return
 
    semimajorAxis = 6378137.0  # WGS84 spheriod semimajor axis
    east = xLon * 0.017453292519943295
    north = yLat * 0.017453292519943295
 
    northing = 3189068.5 * math.log((1.0 + math.sin(north)) / (1.0 - math.sin(north)))
    easting = semimajorAxis * east
 
    return [easting, northing]
df_agg_mercator = df_agg.apply(lambda row: to_web_mercator(row['Latitude'], row['Longitude']), axis=1)
df_agg_mercator.head()


# The next step is to plot the trayectories in the map using datashader
# 

plot_width  = 850
plot_height = 600
x_range = (-1.9e6, 5.9e6)
y_range = (3.7e6, 9.0e6)
def create_image(x_range=x_range, y_range=y_range, w=plot_width, h=plot_height, cmap=None):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.line(df_agg_mercator, 'Latitude', 'Longitude',  ds.count())
    #img = tf.shade(agg, cmap=reversed(Blues3), how='eq_hist')
    #img = tf.shade(agg, cmap=reversed(Greens3), how='eq_hist')    
    img = tf.shade(agg, cmap=cmap, how='eq_hist')
    return img

def base_plot(tools='pan,wheel_zoom,reset',plot_width=plot_width, plot_height=plot_height,**plot_args):
    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range, outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0, **plot_args)
    
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    
    return p


ArcGIS=WMTSTileSource(url='http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{Z}/{Y}/{X}.png')
p = base_plot()
p.add_tile(ArcGIS)
#InteractiveImage(p, create_image, cmap=inferno)


# The function `InteractiveImage` is not rendered in github. I made a snapshot and show it here:
# <p align="center">
# <img src="map.JPG" alt="Matches between teams in the Champions League" width="60%"/>
# </p>
# 
# This is the first MVP which provides a final product. It is imperfect, it doesn't have all the teams in Europe and some of the matches are wronly represented. However, we got some data, transform it and plot it in a map.
# 

# ### MVP2: Improving the map to show all teams in Europe
# 
# Now that we have the map, we can start to improve it. If you are into football, you will notice that there are several points in the north of Spain, that corresponds to Sporting de Gijon. Sadly for Sporting supporters, they have never reached to the Champions. Instead, Sporting Clube de Portugal has participated several times in the championship, but since the current dataset doesn't have teams from Portugal, the system mistakenly thinks that `Sporting CP` from `champions.csv` is the Sporting de Gijon from `stadiums.csv`. So lets fix this issue by getting the stadiums coordinates from the rest of the countries in Europe.  
# We can get that info from [wikidata](https://query.wikidata.org/). Using `SPARQL` language we can get the information we need:
# ```SQL
# SELECT ?clubLabel ?venueLabel ?coordinates ?countryLabel  WHERE {
#   ?club wdt:P31 wd:Q476028.
#   ?club wdt:P115 ?venue.
#   ?venue wdt:P625 ?coordinates.
#   ?club wdt:P17 ?country.
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
# }
# ORDER BY ?clubLabel
# ```
# This generates 4435 Results in 10111 ms that can be saved to a csv file.

df_stadium_read = pd.read_csv('stadiums_wikidata.csv', usecols=['clubLabel','venueLabel','coordinates','countryLabel'])
df_stadium_read.tail()


# The first step is to clean the column coordinates. For that we will use a regex pattern. The pattern `[-+]?[0-9]*\.?[0-9]+` finds any signed float in a string. Then we create two patterns separated by a space and name the columns using this format: `(?P<Longitude>)`. Finally we have to concatente the club information with the coordinates.
# 

df_temp = df_stadium_read['coordinates'].str.extract('(?P<Longitude>[-+]?[0-9]*\.?[0-9]+) (?P<Latitude>[-+]?[0-9]*\.?[0-9]+)', expand=True)
df_stadium_new = pd.concat([df_stadium_read['clubLabel'],df_stadium_read['venueLabel'], df_temp, df_stadium_read['countryLabel']], axis=1) 
df_stadium_new = df_stadium_new.rename(columns = {'clubLabel':'Team', 'venueLabel':'Stadium','countryLabel':'Country'})
print("Number of rows: %d" % df_stadium_new.shape[0])
unique_teams_stadium = list(set(df_stadium_new['Team']))
print("Unique team's name number: %d" % len(unique_teams_stadium))
df_stadium_new.take(list(range(3388,3393)))


# As it can be seen in the previous dataframe, we came into another problem. The new dataset contains all team instances, all over the world. There are some teams that have the same name in different countries, there is a Real Madrid team from USA and South Africa, and a similar name within a country. In our case, we are only interested in the instance `Real Madrid, Estadio Santiago Bernabeu, -3.68835, 40.45306, Spain`. So how can we find an automated way to filter the correct teams that have participated in the Champions League?
# 
# A practical approach is to combine an automated and manual way. With the data we have so far we can automatically filter the two first entries using the country. We can get the country info from `champions.csv` dataset. To distinguish teams from the same country we will filter them manually. 
# 
# The first step then is to get a dataframe with all the teams that have participated in Champions and their country of origin. Then we have to remove the repeated entries and rename the country code to the country name that can be found in wikidata. 

df_match_home = df_match[['home','home_country']]
df_match_home = df_match_home.rename(columns={'home':'Team','home_country':'Country'})
df_match_visitor = df_match[['visitor','visitor_country']]
df_match_visitor = df_match_visitor.rename(columns={'visitor':'Team','visitor_country':'Country'})
df_champions_teams = pd.concat([df_match_home,df_match_visitor], axis=0, ignore_index=True)
df_champions_teams = df_champions_teams.drop_duplicates()
print("Number of unique teams: %d" % df_champions_teams.shape[0])
country_dict = {'ALB':'Albania',
                'AND':'Andorra',
                'ARM':'Armenia',
                'AUT':'Austria',
                'AZE':'Azerbaijan',
                'BEL':'Belgium',
                'BIH':'Bosnia and Herzegovina',
                'BLR':'Belarus',
                'BUL':'Bulgaria',
                'CRO':'Croatia',
                'CYP':'Cyprus',
                'CZE':'Czech Republic',
                'DEN':'Denmark',
                #'ENG':'England',
                'ENG':'United Kingdom',
                'ESP':'Spain',
                'EST':'Estonia',
                'FIN':'Finland',
                'FRA':'France',
                'FRO':'Feroe Islands',
                'GEO':'Georgia',
                'GER':'Germany',
                'GIB':'Gibraltar',
                'GRE':'Greece',
                'HUN':'Hungary',
                'ITA':'Italy',
                'IRL':'Ireland',
                'ISL':'Iceland',
                'ISR':'Israel',
                'KAZ':'Kazakhstan',
                'LTU':'Lithuania',
                'LUX':'Luxembourg',
                'LVA':'Latvia',
                'MDA':'Moldova',
                'MKD':'Macedonia',
                'MLT':'Malta',
                'MNE':'Montenegro',
                'NED':'Netherlands',
                #'NIR':'Northern Ireland',
                'NIR':'United Kingdom',
                'NOR':'Norwey',
                'POL':'Poland',
                'POR':'Portugal',
                'ROU':'Romania',
                'RUS':'Russia',
                #'SCO':'Scotland',
                'SCO':'United Kingdom',
                'SMR':'San Marino',
                'SRB':'Serbia',
                'SUI':'Switzerland',
                'SVK':'Slovakia',
                'SVN':'Slovenia',
                'SWE':'Sweden',
                'TUR':'Turkey',
                'UKR':'Ukrania',
                #'WAL':'Wales',
                'WAL':'United Kingdom'}
df_champions_teams['Country'].replace(country_dict, inplace=True)
#df_champions_teams.to_csv('match_unique.csv')# To check that the mapping is correct
df_champions_teams.sort_values(by='Team',inplace=True)
df_champions_teams = df_champions_teams.reset_index(drop=True)
df_champions_teams.head()


# Once we have the list of all teams that have participated in the Champions League, we have to generate a new dataset relating each Champions League matches with the coordinates of the team stadiums. For that we will use the function `similar` to match a the name of the team in the different datasets similarly as we did before. 
# 
# Once the csv has been generated, let's manually erase the combinations that are not correct and save everything in a new file. We won't correct those entries that are not matched, a Data Science project is better out than perfect!! 
# 

get_ipython().run_cell_magic('time', '', 'def get_info_similar_team_country(team, country, df_stadium, df, threshold, verbose):\n    team2 = "Unknown"\n    stadium = "Unknown"\n    latitude = np.NaN\n    longitude = np.NaN\n    cols = list(df)\n    for idx, val in enumerate(df_stadium[\'Team\']):\n        rank = similar(team, val)\n        if rank > threshold and country == df_stadium[\'Country\'].iloc[idx]:\n            if(verbose): print("%s and %s(Idx=%d) are %f similar and from the same country %s." \n                               % (team, val, idx, rank, country))\n            team2 = df_stadium[\'Team\'].iloc[idx]\n            stadium = df_stadium[\'Stadium\'].iloc[idx]\n            latitude = df_stadium[\'Latitude\'].iloc[idx]\n            longitude = df_stadium[\'Longitude\'].iloc[idx]\n            dtemp = pd.DataFrame([[team, team2, stadium, latitude, longitude, country]], columns=cols)\n            df = df.append(dtemp, ignore_index=True)\n    #if there is no match, register it\n    if(team2 == "Unknown"):\n        df_nomatch = pd.DataFrame([[team, team2, stadium, latitude, longitude, country]], columns=cols)\n        df = df.append(df_nomatch, ignore_index=True)\n    return df\n\ndef generate_new_stadium_dataset(df_champions_teams, df_stadium_new, threshold=0.6, verbose=False):\n    df = pd.DataFrame(columns=[\'Team\', \'Team2\', \'Stadium\', \'Latitude\',\'Longitude\',\'Country\'])\n    for idx, row in df_champions_teams.iterrows():\n        df = get_info_similar_team_country(row[\'Team\'],row[\'Country\'], df_stadium_new, df, \n                                           threshold=threshold, verbose=verbose)\n    return df\n\nverbose = False # You can change this to True to see all the combinations\nthreshold = 0.5\ndf_stadiums_champions = generate_new_stadium_dataset(df_champions_teams, df_stadium_new, threshold, verbose)\ndf_stadiums_champions.to_csv(\'stadiums_champions.csv\', index=False)')


# After we filtered the entries in the csv, let's load again the data and repeat the process.
# 

df_stadiums_champions = pd.read_csv('stadiums_champions_filtered.csv', usecols=['Team','Stadium','Latitude','Longitude','Country'])
df_stadiums_champions.head()


# As previously, we create a dataframe that relates each match with the coordinates of its stadium 
# 

df_match_stadium_new= df_match
home_stadium_index = df_match_stadium_new['home'].map(lambda x: get_info_similar_team(x, df_stadiums_champions))
visitor_stadium_index = df_match_stadium_new['visitor'].map(lambda x: get_info_similar_team(x, df_stadiums_champions))
df_home = pd.DataFrame(home_stadium_index.tolist(), columns=['home_stadium', 'home_latitude', 'home_longitude'])
df_visitor = pd.DataFrame(visitor_stadium_index.tolist(), columns=['visitor_stadium', 'visitor_latitude', 'visitor_longitude'])
df_match_stadium_new = pd.concat([df_match_stadium_new, df_home, df_visitor], axis=1, ignore_index=False)
df1 = df_match_stadium_new['home_stadium'] == 'Unknown'
df2 = df_match_stadium_new['visitor_stadium'] == 'Unknown'
n_complete_matches = df_match_stadium_new.shape[0] - df_match_stadium_new[df1 | df2].shape[0]
print("Number of matches with complete data: %d out of %d" % (n_complete_matches, df_match_stadium_new.shape[0]))
df_match_stadium_new.head()


# The next step is to aggregate the coordinates and transform them to mercator format. 
# 

df_agg = aggregate_dataframe_coordinates(df_match_stadium_new)
df_agg_mercator = df_agg.apply(lambda row: to_web_mercator(row['Latitude'], row['Longitude']), axis=1)
print("Number of rows: %d" % df_agg_mercator.shape[0])
df_agg_mercator.head()


# Finally, we plot the coordinates in the map.
# 

get_ipython().run_cell_magic('time', '', '#InteractiveImage(p, create_image, cmap=inferno)')


# Finally, the image for github:
# <p align="center">
# <img src="map2.JPG" alt="Matches between teams in the Champions League" width="60%"/>
# </p>
# 
# This is the second MVP. In this case we have the correct connections between teams in Europe. 
# 

# ### MVP3: improving design
# 
# If you take a look at the map you will notice that it is fairly ugly. The final quality of a product will drop if we don't show a good visualization. The next step is to improve the map changing the background and colors.  
# 
# 

ArcGIS2 = WMTSTileSource(url='http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png')
p2 = base_plot()
p2.add_tile(ArcGIS2)
cmap = reversed(Blues3)
#cmap = reversed(Greens3)
#InteractiveImage(p2, create_image, cmap=cmap)


# Finally, the image for github:
# <p align="center">
# <img src="map3.JPG" alt="Matches between teams in the Champions League" width="40%"/>
# </p>
# 
# We developed three different MVPs iterating on the project. The Lean Startup method provides a very easy way to remove uncertainty and get a minimum product very quickly that we can show to the customer. 
# 
# Happy data sciencing!
# 

