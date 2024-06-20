# There is a nice discussion that is intuitive and easy to follow on Metropolis-Hastings Algorithm on [Wikipedia](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm). [Here](http://galton.uchicago.edu/~eichler/stat24600/Handouts/l12.pdf) are some more nice notes on MH. Simulated Annealing, a method inspired my Metropolis-Hastings Algorithm is also discussed here.
# 
# The pseudo code for Metropolis-Hastings Algorithm is discussed below --
# 
# * Initialize $x_0$.
#     * For $i = 1 \text{ to } N-1$
#         * Sample $u \sim \mathcal{U}_{[0, 1]}$
#         * Sample $x^{*} \sim q(x|x^{(i)})$
#         * If $u < \mathcal{A}(x^(i), x^*) = \min\{\frac{p(x^*)q(x^{(i)}|x^*)}{p(x^{(i)})q(x^*|x^{(i)})}\}$
#             * $x^{i+1} = x^{*}$
#         * else
#             * $x^{i+1} = x^{(i)}$
# 

# Import the required modules
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Let's sample a random distribution and plot the histogram of the samples
x = np.hstack((np.random.normal(7, 1, 10000), np.random.normal(1, 2, 10000)))
print('Expected Value of x  = {} with variance = {}'.format(np.mean(x), np.var(x)))
plt.hist(x, bins=100)
plt.title('Samples drawn from a Gaussian Distribution')
plt.show()


# This function returns the unnormalized probabilites 
def p(x):
    return (norm(7, 1).pdf(x) + norm(1, 2).pdf(x))/2

# Initialize x_0
x = 5
# Number of samples
N = 10000
# I think when we are not sure
# we should keep the `sigma` as large
# as possible, so that we sample the entire space
sigma = 100
# List of sampled points
x_sampled = []
for i in range(N-1):
    # `u` lies in [0, 1]
    u = np.random.uniform()
    # Sample `x_star` from a gaussian distribution centered around `x`
    x_star = norm(x, sigma).rvs()
    if u < min(1, (p(x_star)*norm(x_star, sigma).pdf(x))/(p(x)*norm(x, sigma).pdf(x_star))):
        x = x_star
    x_sampled.append(x)


# Plot the sampled distribution
print('Expected Value of x  = {} with variance = {}'.format(np.mean(x_sampled), np.var(x_sampled)))
plt.hist(x_sampled, bins=100)
plt.title('Samples drawn from a Gaussian Distribution')
plt.show()


# # There are some serious issues with this code. 
# # It is work in progress.
# 

get_ipython().magic('pylab inline')
# Import the OpenCV bindings
import cv2
# To Increase the image size
pylab.rcParams['figure.figsize'] = (8.0, 8.0)


# Load the Image
im = cv2.imread('../data/features/newyork.jpg')
# Convert the image to grayscale
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imshow(im)
title('Input Image')
axis('off')


gauss_ims = []
lap_ims = []
# Number of octaves
no_octaves = 3
# Number of scales per octace
no_scales = 4

# Construct the Gaussian and Laplacian Pyramid
for i in range(no_octaves):
    gauss_ims.append([])
    lap_ims.append([])
    im = cv2.resize(im_gray, (0,0), fx=1./2**i, fy=1./2**i)
    for j in range(no_scales):
        im = cv2.pyrDown(im) 
        gauss_ims[-1].append(im)
    for j in range(len(gauss_ims[-1])-1):
        print gauss_ims[-1][j].shape, gauss_ims[-1][j+1].shape
        lap_ims[-1].append(cv2.subtract(gauss_ims[-1][j], cv2.pyrUp(gauss_ims[-1][j+1])))


# Display the Gaussian Pyramid
for i in range(len(gauss_ims)):
    for j in range(len(gauss_ims[i])):
        figure()
        imshow(gauss_ims[i][j], cmap=gray())
        title('Octave = {}, Scale = {}, Shape = {}'.format(i, j, gauss_ims[i][j].shape))
        axis('off')

# Display the Laplacian Pyramid
for i in range(len(lap_ims)):
    for j in range(len(lap_ims[i])):
        figure()
        imshow(lap_ims[i][j], cmap=gray())
        title('Octave = {}, Scale = {}, Shape = {}'.format(i, j, lap_ims[i][j].shape))
        axis('off')


keypoints = []
# Finding out max-min
figure()
for i in range(len(lap_ims)):
    for j in range(len(lap_ims[i])-2):
        # Get the current and adjacent Scales
        im_up = cv2.pyrUp(lap_ims[i][j+2])
        im_center = lap_ims[i][j+1]
        im_down = cv2.pyrDown(lap_ims[i][j])
        print i, j, lap_ims[i][j+2].shape, lap_ims[i][j+1].shape, lap_ims[i][j].shape
        print i, j, im_up.shape, im_center.shape, im_down.shape
        for k in range(1, im_center.shape[1]-1):
             for l in range(im_center.shape[0]-1):
                 if np.all(im_up[k-1:k+2, l-1:l+2] > im_center[k, l]) and np.all(im_down[k-1:k+2, l-1:l+2] > im_center[k, l]) and np.all(im_center[k-1:k+2, l-1:l+2] >= im_center[k, l]):
                     keypoints.append(((k, l), i, j+1))     


# Load the Image
im = cv2.imread('../data/features/newyork.jpg')
# Draw the image with the detected corners
for keypoint in keypoints:
    cv2.circle(im, tuple(i*2**(keypoint[2]+1)*2**(keypoint[1]) for i in keypoint[0][::-1]), 4, (0, 0, 255), -1)
imshow(im)
title('Input Image with detected features')
axis('off')


# # Displaying Video in IPython Notebook
# 

# Import the required modules
get_ipython().magic('pylab inline')
import cv2
from IPython.display import clear_output


# Grab the input device, in this case the webcam
# You can also give path to the video file
vid = cv2.VideoCapture("../data/deepdream/deepdream.mp4")

# Put the code in try-except statements
# Catch the keyboard exception and 
# release the camera device and 
# continue with the rest of code.
try:
    while(True):
        # Capture frame-by-frame
        ret, frame = vid.read()
        if not ret:
            # Release the Video Device if ret is false
            vid.release()
            # Message to be displayed after releasing the device
            print "Released Video Resource"
            break
        # Convert the image from OpenCV BGR format to matplotlib RGB format
        # to display the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Turn off the axis
        axis('off')
        # Title of the window
        title("Input Stream")
        # Display the frame
        imshow(frame)
        show()
        # Display the frame until new frame is available
        clear_output(wait=True)
except KeyboardInterrupt:
    # Release the Video Device
    vid.release()
    # Message to be displayed after releasing the device
    print "Released Video Resource"
    


# # Below is a video of the above code in action. 
# 

get_ipython().run_cell_magic('html', '', '<!-- TODO -->\n<iframe width="560" height="315" src="https://www.youtube.com/embed/2TT1EKPV_hc" frameborder="0" allowfullscreen></iframe>')


# # Ranking Algorithm
# 
# In this notebook, Markov Chain is used to rank the teams that played in EPL 2016-2017.
# 
# The algorithm used discussed in --
# 
# ```
# ColumbiaX: Machine Learning
# Lecture 20
# Prof. John Paisley
# Department of Electrical Engineering
# & Data Science Institute
# Columbia University
# ```
# 
# ## Dataset 
# 
# Data is downloaded from [here](http://www.football-data.co.uk/englandm.php). The key for the dataset is [here](http://www.football-data.co.uk/notes.txt).
# 

# Import the required modules
from __future__ import division
import numpy as np
import csv


file_name = "../data/ranking-algorithm/epl_16_17.csv"
x = [];
# Read the CSV files
with open(file_name, 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    next(csvreader)
    for row in csvreader:
        x.append([row[2], row[3], int(row[4]), int(row[5])])


# The team names
teams = {x: i for i, x in enumerate({match[0] for match in x})}
teams_rev = {v: k for k, v in teams.items()} 
# Convert into nice numpy array
x = np.array([[teams[match[0]], teams[match[1]], match[2], match[3]] for match in x], dtype=np.int32)
no_teams = len(teams)


# Prepare the Transition matrix
trans = np.zeros((no_teams, no_teams), dtype=np.float32)
for match_id in xrange(x.shape[0]):
    i, j = x[match_id][0], x[match_id][1]
    i_score, j_score = x[match_id][2], x[match_id][3]
    den = i_score + j_score
    if den > 0:
        trans[i, i] += (i_score > j_score) +  i_score / den
        trans[j, j] += (i_score < j_score) +  j_score / den
        trans[i, j] += (i_score < j_score) +  j_score / den
        trans[j, i] += (i_score > j_score) +  i_score / den


# Normalize the transition matrix
norm = np.sum(trans, axis=1) 
trans_norm = trans / np.expand_dims(norm, axis=0)


# Perform the eigenvalue decomposition of the transition matrix
w, v = np.linalg.eig(trans_norm.T)
# Normalize the 1st eigenvector that corresponds to eigenvalue = 1
s_d = v[:, 0].real / np.sum(v[:, 0].real)
# Sort s_d
sorted_ranking = np.argsort(s_d)[::-1]
# Prepare a list to display 
best_teams = [(teams_rev[i], s_d[i]) for i in sorted_ranking]


print("The rankings of the teams are")
for team in best_teams:
    print team


# One of the possible reason for City below United and Southampton might be that City performed better against the good teams.
# 

# In this system I have coded a Kalman Filter from scratch. 
# The code is based on the pseudo code given in this [pdf](http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam04-ekf-4.pdf). -- Page 15
# 
# As an aside, I will recommend to go through the corresponding video lectures.
# 
# --bikz05
# 

# Load the pylab enviroment
get_ipython().magic('pylab inline')
# Just some cosmetics !!
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10


# Add gaussian noise to the signal
pure = np.linspace(0, 100, 101)
noise = np.random.normal(0, 10, 101)
signal = pure + noise
# Show the signal
plot(signal, pure)


# Write the prediction function
def performPrediction(mean_system, sigma_system, noise_system, A):
    mean_system = A*mean_system;
    sigma_system = A*sigma_system*A + noise_system
    return mean_system, sigma_system


# Write the correction function
def performCorrection(measurement, mean_system, sigma_system, noise_system, noise_measurement, C):
    kalman_gain = sigma_system*C*(C*sigma_system*C+noise_measurement)**(-1)
    mean_system += kalman_gain*(measurement - C*mean_system)
    sigma_system = (1 - kalman_gain*C)*sigma_system
    return mean_system, sigma_system


# Gaussian Function
# Not used -- just wanted to check something.
def gaussian(x, mean, sigma):
    return 1/np.power(2*np.pi*sigma, 1/2)*np.exp(-(x-mean)**2/(2*sigma))


# Initialize the system
# Randomly tuned
# Notice that `noise_system` and `noise_measurement` will make 
# a lot of difference as to how the filter works

# All the predicted states are stacked here
x_predicted = []

# Predicted Initial State
# Intentially gave large values
mean_system = 100
sigma_system = 1

# The A and C matrices
A = 1
C = 1

# Tuning these is very important *****
noise_system = 1
noise_measurement = 10

# Perform State update and Kalman update
for measurement in signal:
    # Perform prediction
    mean_system, sigma_system = performPrediction(mean_system, sigma_system, noise_system, A)
    # Perform correction
    mean_system, sigma_system = performCorrection(measurement, mean_system, sigma_system, noise_system, noise_measurement, C)
    # Append 
    x_predicted.append(mean_system)
    
# Plot the measurements and the predictions
# This is what we measure
plot(x_predicted, pure)
# This is what the Kalman Filter thinks
plot(signal, pure)
# This is what is reality!!
plot(pure, pure)

# Notice the difference below


# # Derivation 
# 
# Let Probability of inliers  = 
# $w$ = $\frac{\text{Number of inliers}}{\text{Total number of points}}$
# 
# Probability that the $k$ points in the minimal set are all inliers = $w^k$
# 
# Probability that atleast one of the $k$ point is outlier = $1 - w^k$
# 
# Probability atleast one of the $k$ point is outlier in n iterations = $(1 - w^k)*n$
# 
# If p is the probability that after $n$ iteration we finds an inlier. 
# Then there is $1 - p$ probability that after $n$ iterations,
# we don't have atleast one of the $k$ points as outlier in $n$ iterations.
# 
# So, we get -- 
# 
# \begin{align}
# 1 - p &= (1 - w^k)*n \\ 
# log(1 - p) &= n log(1 - w^k) \n &= \frac{log(1 - p)}{log(1 - w^k)} \\end{align}
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


def iters_needed(k, p=.99, w = .5):
    """
    Returns the number of ransac iterations needed.
    :param k : Minimal set of points.
    :param w : Percentage of inliers.
    :param p : Desired Accuracy
    """
    return np.log(1-p)/np.log(1 - np.power(w, k))


x = range(1, 10)
y = [iters_needed(x_i) for x_i in x]
plt.plot(x, y)
plt.title('Number of iterations VS Minimal set of inliers')
plt.xlabel('Number of iterations')
plt.ylabel('Minimal set of inliers')
plt.show()





# # Computing Integral Image
# 
# ## Formula
# 
# $I(x, y) = i(x,y) + I(x-1, y) + I(x, y-1) + I(x-1, y-1)$
# 
# where -- 
# 
# 1. $(x, y)$ are the co-ordinate locations
# 2. $I$ is the integral image
# 3. $i$ is the original image 
# 

# Load the required modules
get_ipython().magic('pylab inline')
import cv2


def imshow2(im_title, im):
    ''' This is function to display the image'''
    plt.figure()
    plt.title(im_title)
    plt.axis("off")
    if len(im.shape) == 2:
        plt.imshow(im, cmap = "gray")
    else:
        im_display = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        plt.imshow(im_display)
    plt.show()


# Load and display the image
im = cv2.imread('../data/template-matching/wayne-rooney.jpg')
im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
imshow2("Image", im_gray)


# Create the integral image using OpenCV
int_im = cv2.integral(im_gray)
# TODO -- Write own implementation
# Box Filtering using integral image
int_im = np.zeros(im_gray.shape, np.int64)
filter_size = 25
for i in range(1, int_im.shape[0]):
    for j in range(1, int_im.shape[1]):
        int_im[i][j] = im_gray[i][j] + int_im[i-1][j] + int_im[i][j-1] - int_im[i-1][j-1]


# Box Filtering using integral image
im_filtered = np.zeros(im_gray.shape, im_gray.dtype)
filter_size = 25
for i in range(filter_size, int_im.shape[0]-filter_size):
    for j in range(filter_size,int_im.shape[1]-filter_size):
        im_filtered[i][j] = (int_im[i][j] - int_im[i-filter_size][j] - int_im[i][j-filter_size] + int_im[i-filter_size][j-filter_size])/(filter_size*filter_size)
imshow2("Filtered Image", im_filtered)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
get_ipython().magic('matplotlib inline')


# # Monte Carlo Sampling from a Gaussian Distribution
# 

# Gaussian Distribution
x = np.random.normal(5, 1, 10000)
# Law of law numbers -- This mean will be equal to the expected value when the number of samples is large
# Central Limit Theorem -- this mean is Gaussian Distributed.
print('Expected Value of x  = {} with variance = {}'.format(np.mean(x), np.var(x)))
plt.hist(x)
plt.title('Samples drawn from a Gaussian Distribution')
plt.show()


# # Monte Carlo Sampling from a Uniform Distribution
# 

# Uniform Distribution
x = np.random.rand(10000)
# Law of law numbers -- This mean will be equal to the expected value when the number of samples is large
# Central Limit Theorem -- This mean is Gaussian Distributed.
print('Expected Value of x  = {} with variance = {}'.format(np.mean(x), np.var(x)))
plt.hist(x)
plt.title('Samples drawn from a Uniform Distribution')
plt.show()


# # Sampling from a Inverse CDF Function
# 
# This method can be described in 2 steps as --
# 
# 1. Sample $u$ from a Uniform Distribution $\mathcal{U}(0, 1)$.
# 2. Get a sample $x$ from the by using the inverse CDF function $f^{-1}(u)$.
# 
# CAVEAT - We can only use this method when the inverse CDF is defined. 
# 
# ![img inverse-cdf](https://upload.wikimedia.org/wikipedia/commons/2/24/InverseFunc.png)
# *Graph of the inversion technique from ${\displaystyle x}$ to ${\displaystyle F(x)}$. On the bottom right we see the regular function and in the top left its inversion. [Image and Caption from Wikipedia]*

# SIDE NOTE FOR ME -- `scipy.stats.rv_continuous` is the base class.

# Sample u from a uniform distribution
u = np.random.rand(10000)
# Sample x from the inverse CDF function
# In Scipy it's called ppf
samples = norm.ppf(u, 0, 1)
plt.hist(samples)
plt.title('Samples drawn from a Gaussian Distribution')
plt.show()


# # Accept Reject Method
# 
# When the inverse CDF function is not defined, but the PDF is still available then we can use the accept reject method. 
# 

# CODE -- TODO


# # Naive Bayes on MNIST dataset
# The first step is to download the handwritten image dataset. 
# 

get_ipython().magic('pylab inline')
# Fetch the MNIST handwritten digit dataset
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home="../data")


# # Let's display some data :)
# 
# Now let's explore the data.
# 

# Display the number of samples 
print "(Number of samples, No. of pixels) = ", mnist.data.shape

# Display 9 number randomly selectly
for c in range(1, 10):
    subplot(3, 3,c)
    i = randint(mnist.data.shape[0])
    im = mnist.data[i].reshape((28,28))
    axis("off")
    title("Label = {}".format(mnist.target[i]))
    imshow(im, cmap='gray')


# # Split the data into training and testing data
# 

# Split the data into training and test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.05, random_state=42)

# Which is same as 
# x_train = mnist.data[:split]
# y_train = mnist.target[:split]
# x_test = mnist.data[split:]
# y_test = mnist.target[split:]


# # Prepare the classifier
# 

# Create the Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


# # Perform the predictions and display the results
# 

# Perform the predictions
clf.fit(x_train,y_train)
# Perform the predictions
y_predicted = clf.predict(x_test)
# Calculate the accuracy of the prediction
from sklearn.metrics import accuracy_score
print "Accuracy = {} %".format(accuracy_score(y_test, y_predicted)*100)
# Cross validate the scores
from sklearn.metrics import classification_report
print "Classification Report \n {}".format(classification_report(y_test, y_predicted, labels=range(0,10)))


# # Image Blending using Gaussian Pyramids
# 

get_ipython().magic('pylab inline')
import cv2;


def imshow2(im_title, im):
    ''' This is function to display the image'''
    plt.figure();
    plt.title(im_title);
    plt.axis("off");
    if len(im.shape) == 2:
        plt.imshow(im, cmap = "gray");
    else:
        im_display = cv2.cvtColor(im, cv2.COLOR_RGB2BGR);
        plt.imshow(im_display);
    plt.show();


# Load the display the images
im_left = cv2.imread('../data/blending/lion.jpg');
im_right = cv2.imread('../data/blending/tiger.jpg');
imshow2("Left Image", im_left);
imshow2("Right Image", im_right);


# Compute the Gaussian Pyramids
gauss_levels = 7;
gauss_reduce_left = im_left;
gauss_pyr_left = [gauss_reduce_left];
gauss_reduce_right = im_right;
gauss_pyr_right = [gauss_reduce_right];
for i in range(gauss_levels):
    gauss_reduce_left = cv2.pyrDown(gauss_reduce_left);
    gauss_pyr_left.append(gauss_reduce_left);
    gauss_reduce_right = cv2.pyrDown(gauss_reduce_right);
    gauss_pyr_right.append(gauss_reduce_right);

# Prepare the Laplacian Pyramids
lap_pyr_left = []
lap_pyr_right = []
for i in range(1, gauss_levels):
    h, w = gauss_pyr_left[i-1].shape[:2];
    left_append = cv2.subtract(gauss_pyr_left[i-1], cv2.pyrUp(gauss_pyr_left[i], dstsize=(w, h)));
    right_append = cv2.subtract(gauss_pyr_right[i-1], cv2.pyrUp(gauss_pyr_right[i], dstsize=(w, h)));
    lap_pyr_left.append(left_append)
    lap_pyr_right.append(right_append)

'''
# Display the pyramids 
for index, image in enumerate(gauss_pyr_left):
    imshow2("Level {}".format(index), image)
for index, image in enumerate(lap_pyr_left):
    imshow2("Level {}".format(index), image)  
'''
pass


# Combine the images
com_lap = [];
for left, right in zip(lap_pyr_left, lap_pyr_right):
    cols = left.shape[1]
    com_lap.append(np.hstack((left[:, :cols/2], right[:, cols/2:])))

'''
# Display the combined images
# Uncomment to see the Laplacian Pyramid
for index, image in enumerate(com_lap):
    imshow2("Level {}".format(index), image) 
'''
pass


# Now reconstruct
cols = gauss_pyr_left[-2].shape[1]
im_blended = np.hstack((gauss_pyr_left[-2][:, :cols/2], gauss_pyr_right[-2][:, cols/2:]));
for i in range(len(com_lap)-1, -1, -1):
    h, w = com_lap[i].shape[:2];
    im_blended = cv2.pyrUp(im_blended, dstsize=(w, h));
    im_blended = cv2.add(im_blended, com_lap[i]);

# Display the Images
imshow2("Without Blending", np.hstack((im_left[:, :cols/2], im_right[:, cols/2:])))
imshow2("With Blending", im_blended);


