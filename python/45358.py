# # Behaviour
# 
# Modules from low frequency to high frequency:
# 
# 4-01
# 
# (Modules in path planning are boxed)
# 
# * Behaviour planner has to incorporate much data (timescales 10s+)
# 
# #### Sample output:
# 
# ```
# {
#     "target_lane_id" : 2,
#     "target_leading_vehicle_id": null,
#     "target_speed" : 15.0,
#     "seconds_to_reach_target" : 10.0,
# }
# ```
# 
# ### The Behaviour Problem
# 
# 4-02
# 
# Behaviour planner: 
# - Takes map, route and predictions as input
# - Output: suggest states (maneuvers) that are 
# 
# Trajectory planner: 
# - Responsible for executing maneuvers in a collision-free, smooth and safe way
# Driver executes and is responsible for safety (if navigator issues instructions that will immediately result in a crash, the driver will wait to execute)
# Navigator 
# 
# 
# ### A solution: Finite State Machines
# 
# 4-03
# 
# - Discrete states connected by transitions
# - Begin at some start state
# - Decide to go to which state to transition to using a state transition function
# 
# Strengths:
# - Self-documenting, easy to reason about: map logical state to physical state
# - Maintainable (for small state spaces)
# 
# Weaknesses:
# - Easily abused: may just continue adding new states if problem changes or you discover things you hadn't considered before, leading to sloppy code
# - Not maintainable (for large state spaces)
# 
# 
# ### States for self-driving cars
# 
# 
# - Tradeoff between wanting a small, maintainable state space and wanting to include all the states we need
# 
# Sample list of state spaces for highway driving:
# 
# (Ideation process: Brainstorm list of ideas and prune list)
# 
# Pruned to 
# - Keep lane
# - Change lane left
# - Change lane right
# - Prepare lane change left
#     - Needed because pre-change lane left,
#         - it's safer to match the speed of the left lane
#         - without this manouver, you'd have to just wait for a gap to appear in the left lane
#         - Unclear when to turn on the turn signal (ideally want to turn it on a few seconds before lane change)
# - Prepare lane change right
# 
# (Pruned many states because they are implementations of e.g. keep lane, change lane left/right).
# 
# #### State details (in Frenet coordinates)
# 
# 4-05
# 
# ### Transition Functions
# 
# #### Inputs
# * Predictions, map, speed limit, localisation data, current state
# 
# #### Implementation (Pseudocode)
# - Generate rough trajectories for each accessible 'next state' and select the state with the trajectory with the lowest cost.
# 

def transition_function(predictions, current_fsm_state, current_pose, cost_functions, weights):
    # only consider states which can be reached from current FSM state.
    possible_successor_states = successor_states(current_fsm_state)

    # keep track of the total cost of each state.
    costs = []
    for state in possible_successor_states:
        # generate a rough idea of what trajectory we would
        # follow IF we chose this state.
        trajectory_for_state = generate_trajectory(state, current_pose, predictions)

        # calculate the "cost" associated with that trajectory.
        cost_for_state = 0
        for i in range(len(cost_functions)) :
            # apply each cost function to the generated trajectory
            cost_function = cost_functions[i]
            cost_for_cost_function = cost_function(trajectory_for_state, predictions)

            # multiply the cost by the associated weight
            weight = weights[i]
            cost_for_state += weight * cost_for_cost_function
        costs.append({'state' : state, 'cost' : cost_for_state})

    # Find the minimum cost state.
    best_next_state = None
    min_cost = 9999999
    for i in range(len(possible_successor_states)):
        state = possible_successor_states[i]
        cost  = costs[i]
        if cost < min_cost:
            min_cost = cost
            best_next_state = state 

    return best_next_state


# ### Designing cost functions
# 
# #### 1. Vehicle speed
# * Want te get to destination quickly but don't want to break the law.
#     * Suppose maximum cost if you're above the speed limit
#     * Zero cost at
#     * Cost of not moving: bad but not as bad as exceeding the speed limit.
#     * Arbitrarily connect points we've plotted with a linear function
#     * May want to parameterise these
#     
# 4-06
# 
# #### 2. Lane Choice
# 
# Options:
# * Lane Change (LC)
# * Keep Lane (KL)
# 
# 4-07
# 
# Establish relevant variables:
# * $\Delta s = s_G - s$
#     * Longitudinal distance the vehicle has before it needs to get to the goal lane
#     * Cost should be inversely propertional to $\Delta s$ (lane change costs more important when we're closer to the goal because it's more likely we won't make it in time)
# * $\Delta d = d_G - d_{LC/KL}$
#     * lateral distance between goal lane and options being considered
#     * Cost should be propertional to $\Delta d$ 
# 
# * Want to normalise cost such that cost is always in interval [0,1].
#     * choose e.g. $ \text{cost} = 1 - e^{-\frac{|\Delta d|}{\Delta s}}$
#     
#     
# #### Discussion of cost function design
# 
# 4-08
# 
# * New problems like not being aggressive enough about turning left at traffic lights
# * Regression testing: define some set of test cases (situations with corresponding expected behaviour) and testing them when redesigning cost functions
# * Ideally each cost function has a specific function
#     * -> e.g. Define a few cost functinos associated with speed (one for safety, one for legality) 
#     
# 4-09
# 
# * Different priorities in different situations
#     * add 'obeys traffic rules' cost function if at an intersection and traffic light just turned yellow.
#     
# ### Scheduling Compute Time
# * Behaviour module decisions take longer time and don't change as quickly.
# 
# * Behaviour needs info from prediction and localisation to begin its second cycle
#     * But prediction in the middle of an update cycle.
#     * Use slightly-out-of-date data so we don't block the pipeline for downsteram components (which have higher update frequency)
#     
# ### Implement a Behaviour planner
# 

# ### Introduction
# 
# Outcome of this module: 
# * Implement sensor fusion algorithm to track pedestrian relative to a car by fusing lidar and radar using a Kalman filter.
# 
# Sensor fusion needs to happen quickly: need to use high-performance language (C++) instead of Python.
# 
# ### Sensors on a self-driving car
# * Two stereo cameras that act like eyes
# * Traffic signals may be on other side of intersection -> use special lens to give camera range to detect 
# 
# #### Radar
# Stands for 'radio detection and ranging'.
# * sits behind front bumpers
# * Uses Doppler effect (change in frequency of radar waves) to measure velocity directly. 
# * Gives us velocity as an independent parameter and allows sensor fusion algorithm to converge faster.
# * Generates radar maps to help with localisation. 
#     * Provides measurements to objects without direct line of sight. Approx 150 deg field of vision, 200m range. 
#     * Resolution in vertical direction limited.
#     * Radar clutter: Reflection across static objects e.g. soda cans on the street or manhole covers can cause problems. -> disregard static objects.
# 
# #### Lidar
# Stands for 'light detection and ranging'.
# * Infrared laser beam to det distance. 
# * Lasers pulsed, pulses reflected by objects. Reflections return a point cloud that represents these objects.
# * Relies on difference between two or more scans to calculate velocity (as opposed to measuring it directly as with radars)
# * 900nm wavelength range, some use longer wavelengths -> better in rain or fog.
# [](2-3.png)
# 

# # Motion Models
# 
# ## Bicycle motion model: from a car to a bicycle
# * Ignore verticle dynamics of the model.
# * Assume front wheels like one wheel, likewise with back wheels.
# * Assume car controlled like a bicycle.
# 
# ### Yaw rate equations
# 
# Yaw: Rotation of the car about the z-axis.
# 
# ![](images/12.1.png)
# ![](images/12.2_seenotes.png)
# 
# ### Frames of reference: a comparison (Localisation vs Sensor Fusion)
# * Coordinates: L uses vehicle or map coordinates, S uses only vehicle coordinates.
# * Position of car described in: L: map. S: car always assumed to be at origin of the vehicle coord system.
# * Sensor measurements in: L, S (both) vehicle coords.
# * Map: L: Map landmarks in map coordinates. S: No map.
# 
# ### Roll, Pitch and Yaw
# * Roll: Rotation of the car about the x-axis.
# * Pitch: Rotation of the car about the y-axis.
#     * Important in hilly places.
# 
# ![](images/12.3.png)
# 
# ### Odometry: Motion sensor data
# * Commonly from wheel sensors (number of wheel rotations -> distance travelled)
# * Errors on 
#     * wet (slipping wheels travel less than expected + slide when braking) roads or 
#     * bumpy roads (overestimates distance since car moves up and down vs assumed it doesn't move vertically).
#     
#     
# ![](images/12.4.png)

# Using deep neural networks to train SDCs:
# * Called behavioural cloning 
# * or end-to-end learning because the network is learning to predict the correct steering angle and speed using only the inputs from the sensors.
# 
# Deep Learning vs Robotic approach:
# * Both approaches are being used
# * Robotic approach involves detailed knowledge about sensors, controls and planning
# * DL allows us to build a feedback loop.
# 
# 

# ### Algorithms Covered:
# - Naive Bayes
# - SVMs
# - Decision Trees
# 

# # Transfer Learning
# 
# Transfer learning: repurposing a network for a different task.
# 
# Bryan Catanzaro: wrote library than became CUDNN. Lol.
# 
# GPU: Optimisod far throughput computation (simult)
#     - since update lots of pixels on screen at the same time.
#     - dl computations involve a lot of parallelisation.
# CPU: latency -> running a single thread of instructions as quickly as possible
# 
# Rule of thumb: GPU 5x faster than CPU.
# 
# 
# Stanford University Notes on Transfer Learning: http://cs231n.github.io/transfer-learning/
# 
# 
# Why transfer learning is useful
# * Can use intel stored in trained networks to accelerate own progress
# - Your dataset may be small, so networks pre-trained on much data can help you generalise.
# 
# DL more popular recently because of
# - Increased availability of labelled data
# - Increase in computational power
# 
# - 1990s LeNet
# - ImageNet image classification competition 
#     - 2012 AlexNet (Fundamental architecture
#         - Used parallelism of 40-byte GPUs to train network in about a week.
#         - Pioneered use of ReLUs in activation.
#         - Dropout to avoid overfitting.
#         - 15% error vs 26% error of winner in 2011.
# - 
# 

# ## Labs
# Transfer Learning with TensorFlow
# Transfer learning is the practice of starting with a network that has already been trained, and then applying that network to your own problem.
# 
# Because neural networks can often take days or even weeks to train, transfer learning (i.e. starting with a network that somebody else has already spent a lot of time training) can greatly shorten training time.
# 
# So how we do go about applying transfer learning? Two popular methods are feature extraction and finetuning.
# 
# Feature extraction. We take a pretrained neural network and replace the final layer (classification layer) with a new classification layer for the new dataset or perhaps even a small neural network (eventually has a classification layer). During training the weights in all the old layers are frozen, only the weights for the new layer we added are trained. In other words, the gradient doesn't flow backwards past the first new layer we add.
# Finetuning. Essentially the same as feature extraction except the weights of the old model aren't frozen. The network is trained end-to-end.
# In this lab and the one later in the section we're going to focus on feature extraction since it's less computationally intensive.
# 
# ### Feature Extraction
# 
# The problem is that AlexNet was trained on the ImageNet database, which has 1000 classes of images. You can see the classes in the caffe_classes.py file. None of those classes involves traffic signs.
# 
# In order to successfully classify our traffic sign images, you need to remove the final, 1000-neuron classification layer and replace it with a new, 43-neuron classification layer.
# 
# This is called feature extraction, because you're basically extracting the images features captured by the penultimate layer, and passing them to a new classification layer.
# 
# #### Solution
# First, I figure out the shape of the final fully connected layer, in my opinion this is the trickiest part. To do that I have to figure out the size of the output from fc7. Since it's a fully connected layer I know it's shape will be 2D so the second (or last) element of the list will be the size of the output. fc7.get_shape().as_list()[-1] does the trick. I then combine this with the number of classes for the Traffic Sign dataset to get the shape of the final fully connected layer, shape = (fc7.get_shape().as_list()[-1], nb_classes). The rest of the code is just the standard way to define a fully connected in TensorFlow. Finally, I calculate the probabilities via softmax, probs = tf.nn.softmax(logits).
# 

import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the specifically for 
# traffic signs model.
fc7 = AlexNet(resized, feature_extract=True)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))


# ### Training the Feature Extractor
# The feature extractor you just created works, in the sense that data will flow through the network and result in predictions.
# 
# But the predictions aren't accurate, because you haven't yet trained the new classification layer.
# 
# In order to do that, you'll need to read in the training dataset and train the network with cross entropy.
# 
# Training AlexNet (even just the final layer!) can take a little while, so if you don't have a GPU, running on a subset of the data is a good alternative. As a point of reference one epoch over the training set takes roughly 53-55 seconds with a GTX 970.
# 
# 

# Simplified AlexNet used as a starting point for CV today
# 
# Others:
# - Some feature
# 

# ### VGGNet (VGG)
# 
# From Visual Geometry group at Oxford.
# 
# Architecture:
# * Long series of 3x3 convs 
# * broken up by series of 2x2 pooling layers
# * finished by a trio of fully connected layers at the end.
# 
# Starting point for working on other image classifcation tasks.
# VGG Strengths: 
# * Flexibility
# 
# Claim: Bias towards action needed to be a successful deep learning practitioner. Experiment continually.
# 

# ### GoogLeNet
# 
# * Runs really fast.
# * Inception module: trains well and is efficiently deployable.
#     - At each layer of your convnet, you can make a choice - pooling or convolution (1x1 or 3x3 or 5x5?) -> use all of them instead of choosing only one.
#     - Composition of averaging pooling followed by 1x1, then 1x1 followed by 3x3 ... and then concatenate the output.
#     - Choose params in such a way such that tho total number of parameters is small (so GoogLeNet runs nearly as fast as AlexNet,, can run in real time).
#     - -> Like ensembles.
# 
# ### ResNet (Microsoft)
# * 152 layers (vs AlexNet 8 layers, VGG 19, GoogLeNet 22)
# * Same structure repeated again and again like VGG. -> Add connections to NN that skill layers so
# * 3% on ImageNet, > human accuracy.
# 

get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : pink !important;} </style>')


# Camera Calibration with OpenCV
# ===
# 
# ### Run the code in the cell below to extract object points and image points for camera calibration.  
# 

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib qt')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('calibration_wide/GO*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


# ### If the above cell ran sucessfully, you should now have `objpoints` and `imgpoints` needed for camera calibration.  Run the cell below to calibrate, calculate distortion coefficients, and test undistortion on an image!
# 

import pickle
get_ipython().magic('matplotlib inline')

# Test undistortion on an image
img = cv2.imread('calibration_wide/test_image.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)





# # Model Predictive Control
# 
# * Reframe problem of following trajectory as an optimisation problem
# * Predict result of trajectory and select trajectory with minimum cost
# * Implement first set of actuation. Take new state to calculate new optimal trajectory. Incremental calculating: 'receding horizon control'.
# 
# ### Cost functions
# #### Cost elements
# * State
#     * Cross-track error
#     * Orientation error
#     * Velocity error (velocity relative to reference velocity)
#     * Euclidean distance between current position and destination
# * Control input
#     * Large steering angle (jerking steering wheel)
#         * ```cost += pow(delta[t], 2);```
#     * Change-rate of control input to add smoothness
#         * ```for (int t = 0; t < N-1; t++) {
#   cost += pow(delta[t+1] - delta[t], 2)
#   cost += pow(a[t+1] - a[t], 2)
# }```
#     * Large change in steering angle: (larger `multiplier` -> smoother steering angle change)
#         * ```fg[0] += multiplier * CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);```
# 
# #### Notes
# * Prediction horizon T = N * dt
#     * T: Duration over which future predictions are made
#     * N: Number of timesteps in the horizon
#     * dt: time elapsed between actuations, i.e. length of each timestep
#     * Guidelines: T should be as large as possible, dt as small as possible.
#         * But if T > a few seconds, environment changes enough that prediction doesn't make sense
#         * Large N: high computational cost
#         * Large dt: infrequent actuations, hard to accurately approximate a continuous reference trajectory (discretisation error)
#     
# ### Model Predictive Control Algorithm
# Setup:
# 
# 1. Define the length of the trajectory, N, and duration of each timestep, dt.
# 2. Define vehicle dynamics and actuator limitations along with other constraints.
# 3. Define the cost function.
# 
# Loop:
# 
# 1. We pass the current state as the initial state to the model predictive controller.
# 2. We call the optimization solver. Given the initial state, the solver will return the vector of control inputs that minimizes the cost function. The solver we'll use is called Ipopt.
# 3. We apply the first control input to the vehicle.
# 4. Back to 1.
# 
# #### Problem: Latency
# * Delay between actuation command and execution (e.g. 100ms)
# * Can model into system with MPC (vs PID controller hard to do that)
# 

# # Vehicle Models
# 
# * Models that describe how the vehicle moves
# * Tradeoff between tractability and accuracy of models
# * Kinematic and dynamic models:
#     * Kinematic models
#         * Ignore tire forces, gravity and mass
#         * Work better at low and moderate speeds
#     * Dynamic models
#         * May encompass tire forces, longitudinal and lateral forces, inertia, gravity, air resistance, drag, mass, and the geometry of the vehicle
#         * May even take internal vehicle forces into account - for example, how responsive the chassis suspension is
# 
# #### Vehicle State [x,y,ψ,v]
# * X, y coordinates
# * Orientation
# * Velocity
# 
# ## Kinematic models
# 
# ### Actuators [δ,a]
# Actuator inputs allow us to control the vehicle state. 
# 
# Most cars have three actuators: 
# * the steering wheel
# * the throttle pedal and 
# * the brake pedal. 
# 
# For simplicity we'll consider the throttle and brake pedals as a singular actuator, with negative values signifying braking and positive values signifying acceleration.
# 
# Simplified:
# * δ for steering angle and 
# a for acceleration (throttle/brake combined).
# 
# x = x + v * cos(psi) * dt
# y = y + v * sin(psi) * dt
# 
# 
# v=v+a∗dt
# * a in [-1,1]
# 
# ψ=ψ+(v/L_f)*δ∗dt
# * Add multiplicative factor of the steering angle, δ, to ψ
# * L_f measures the distance between the front of the vehicle and its center of gravity. 
#     * The larger the vehicle, the slower the turn rate.
#     * Testing the validity of a model:
#         * If the radius of the circle generated from driving the test vehicle around in a circle with a constant velocity and steering angle is similar to that of your model in the simulation, then you're on the right track.
# 
# 
# ### Global Kinematic Model
# 

# Autonomous vehicle system steps:
# * Perception system estimates the state of the surrounding environment
# * Localisation block compares model to a map to figure out where the vehicle is
# * Path planning block charts a trajectory (using environmental model, map, vehicle location)
#     * Trajectory typically passed on as a (third degree) polynomial
# * Control loop applies actuators to follow this trajectory.
# 
# Capture in new state [x,y,ψ,v,cte,eψ].
# 

# ### Fitting Polynomials
# 
# Fit polynomials to waypoints (x, y) in C++ using Eigen.
# 

# ### Error
# 
# Control loop applies actuators to minimize the error between the reference trajectory and the vehicle’s actual path.
#   * i.e. minimise (1) cross-track error and (2) predicted difference, or angle, between the vehicle orientation and trajectory orientation. We’ll call this the psi error (epsi).
#   * How: minimise difference between predicted path and reference trajectory.
#   
# #### Cross Track Error
# CTE: error between the center of the road and the vehicle's position.
# 
# * cte_{t+1} = cte_t + v_t + sin(epsi_t) * dt
#     * current cte + the change in error caused by the vehicle's movement
# * cte_t = f(x_t) - y_t 
# 
# #### Orientation error
# * eψ=eψ+(v/L_f)*δ∗dt
# * eψ_t = ψ_t - ψdes_t
#     * ψdes is desired orientation, arctan(f'(x_t))
# 
# ![](images/18.1.png)
# 

# ## Dynamic Models
# * Interactions beween tires and road
#     * Longitudinal (forward or backward) and lateral (side to side) force
#     
# ![](images/18.2.png)    
# 
# #### Slip Angle
# * Angle between the velocity vector of the wheel and the orientation of the wheel itself.
# * α=arctan(vyw/vxw)
#     * where vxw and vyw are the longitudinal and lateral velocities of the wheel.
# * Required to generate lateral force
#     * Else inertia would carry the vehicle off the road.
# 
# ![](images/18.3.png)
# 
# #### Slip Ratio
# 
# * Mismatch between speed of vehicle wheel and the expected longitudinal velocity -> tire slides (in addition to rolling)
# * Required to generate longitudinal force
# 
# ![](images/18.4.png)
# ![](images/18.5.png)
# 
# ### Tire models
# * e.g. Pacejka Tire Model (the Magic Tire Formula)
# ![](images/18.6.png)
# 
# ### Actuator Constraints
# * E.g. bounds on steering angle [-30,30] degrees 
#     * Vehicle can't have steering angle of 90 degrees.
#     * Nonholonomic (can't move in arbitrary directions) model
# * Bounds on acceleration [-1, 1].
# 

