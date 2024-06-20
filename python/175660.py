# # K Means clustering algorithm
# 

# K Means is an unsupervised clustering learning algorithm.In this notebook K Means has been implemented from scratch on the iris dataset.
# Steps to implement K Means-:
# 1. Select k random points as cluster centers.
# 2. Compute the euclidean distance of all the points from the centers(centroids) and assign it to the nearest cluster(center).
# 3. Compute the mean of the points in a cluster and the mean value becomes the new cluster center.
# 4. Keep on repeating 2 and 3 until the centroids do not change.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from copy import deepcopy


data=pd.read_csv("/home/sourojit/tfmodel/Iris.csv")


data.head()


# # Handling the Data
# 

# The iris dataset contains 4 features and a label.The label denotes the species to which each sample belongs.Since, K Means is an unsupervised learning algorithm we remove the labels column.
# 

x_point=data["SepalLength"].values
y_point=data["PetalWidth"].values
points=np.array(list(zip(x_point,y_point)))
plt.scatter(x_point,y_point,c="red")
plt.show()


# We randomly select two features to work on which gives the best distribution of points.This is important as we want our clusters to be distinctly visible to the naked eye when we plot the data. 
# 

clusters=3


points.shape


centroid_x=[5,5,7.5]
centroid_y=[0,1,1.5]


# Selecting centers randomly
# 

centroid=np.array(list(zip(centroid_x,centroid_y)))


centroid


plt.scatter(x_point,y_point,c="red")
plt.scatter(centroid_x,centroid_y,marker="*",c='g')
plt.show()


# Points plotted along with the initial centroids
# 

centroid_old=np.zeros(centroid.shape)


cluster=np.zeros(len(points))


error=np.linalg.norm(centroid-centroid_old)


error


# The error term denotes the difference between the centroids in the current iteration and the previous iteration.This term is important because it will determine if our cluster centroids have converged or not.
# 

# # K Means Algorithm
# 

# In each iteration of the while loop each sample is being assigned to its nearest cluster centroid.After each sample has been asigned to a centroid, a new centroid is computed by taking the mean of all the samples assigned to a specific cluster.This is done for all clusters.
# 

while error!=0:
    for i in range(len(points)):
        distance=[np.linalg.norm(points[i]-centroids) for centroids in centroid]
        c=distance.index(min(distance))
        cluster[i]=c
    centroid_old=deepcopy(centroid)    
    for i in range(len(centroid)):
        cluster_points=[]
        for j in range(len(cluster)):
            if cluster[j]==i:
                cluster_points.append(points[j])
        centroid[i]=np.mean(cluster_points,axis=0)
        centroid[i]
    error=np.linalg.norm(centroid-centroid_old)
        


error


centroid


colors=['r','g','b']
ax = plt.subplot()
for i in range(clusters):
    cluster_points=np.array([points[j] for j in range(len(points)) if cluster[j]==i])
    ax.scatter(cluster_points[:,0],cluster_points[:,1],c=colors[i])
ax.scatter(centroid[:,0],centroid[:,1],marker="*",c="black")
plt.show()


# # K- Nearest Neighbours Classification  
# 

# getting the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import random
style.use('fivethirtyeight')


# In this notebook, we implement the algorithm from the basics. For the purpose of visualization, we will first test the model on 2-dimensional data points. Later, we employ the same model on the Iris Dataset.
# 

# knn method (3 neighbors by default)
def knn(data,predict,k=3):
    
    # data is the training set 
    # predict is a single test data instance
    # k is number of nearest neighbors used for classification (user parameter)
    
    if len(data)>=k:
        warnings.warn('K is set to value less than total voting groups!')
    
    # distances stores distance and label of the training points from the test point
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    # sort distances in increasing order and take the label of the least k among them
    votes = [i[1] for i in sorted(distances)[:k]]
    
    # find the label which occurs the most and proportion of the its occurence
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result , confidence   


# train set
# 3 points each in 2 classes ('k': black, 'r':red)
dataset = {'k':[[1,2],[2,3],[3,1]] , 'r':[[6,5],[7,7],[8,6]]}

# test instance
new_features = [5,7]

result = knn(dataset,new_features,3)
print(result)


# plotting the points
[[plt.scatter(ii[0],ii[1],s=100,c=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],s=50,c=result[0])

plt.show()


# # Applying on Iris Dataset 
# 

# importing the Iris Dataset
df = pd.read_csv('Iris.csv')
species = df['Species'].unique()
df.drop(['Id'],1,inplace=True)
df.head()


# converting the dataframe to a list
full_data = df.values.tolist()

# shuffling the records
random.shuffle(full_data)

# splitting the dataset into train(80%) and test sets(20%)
test_size = 0.2
train_set = {species[0]:[],species[1]:[],species[2]:[]}
test_set = {species[0]:[],species[1]:[],species[2]:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
for i in train_data:
    train_set[i[-1]].append(i[:-1])    


# Calculating the accuracy. Also displaying the confidence in case of incorrect prediction.
# 

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = knn(train_set,data,k=5)
        if vote==group:
            correct +=1
        else:
            print(confidence)
        total +=1

print('Accuracy:',correct/total)


# # Linear Regression from scratch
# 

# In statistics, linear regression is a linear approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X.
# 

# In this notebook, we demonstrate the case of one explanatory variable called simple linear regression since it easy to visualize. The technique demonstrated can easily be extended to apply for more than one explanatory variable (multiple linear regression)
# 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import random
style.use('fivethirtyeight')

#plotting some random points 
x = np.array([1,2,3,4,5,6],dtype=np.float64)  #single independent variable
y = np.array([6,4,5,2,1,1],dtype=np.float64)  #dependent variable

plt.scatter(x,y)
plt.show()


# # Finding the best fit line
# 

def best_fit_slope_intercept(x,y):
    m = (mean(x)*mean(y)-mean(x*y))/(mean(x)**2 - mean(x*x))
    b = mean(y) - m*mean(x)
    return m,b


#find the best fit line for the given point set
m,b = best_fit_slope_intercept(x,y) 

#finding the regression line
reg_line = [(m*i)+b for i in x]

#plotting the given points
plt.scatter(x,y)

#plotting the regression line
plt.plot(x,reg_line)
plt.show()


# # Measurement of Goodness of Fit
# 

def squared_error(y_orig,y_line):
    return sum((y_orig-y_line)**2)

def coeff_det(y_orig,y_line):
    y_mean_line = [mean(y_orig) for i in y_orig]
    sse = squared_error(y_orig,y_line)
    sst = squared_error(y_orig,y_mean_line)
    return 1-(sse/sst)

print(coeff_det(y,reg_line))


# # Creating our dataset 
# 

# hm :  size of dataset (number of points)
# var:  variance in the y-value 
# step: change in y-value for consecutive x-values
# correlation: whether y-value increases with x-value or not.

def create_dataset(hm,var,step=2,correlation=False):
    val =1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-var,var)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]        
    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)


#Creating dataset
xs,ys = create_dataset(100,20,2,'pos')

#Fitting the best fit line
m,b = best_fit_slope_intercept(xs,ys)
reg_line = [(m*i)+b for i in xs]

#Plotting the points and the regression line
plt.scatter(xs,ys)
plt.plot(xs,reg_line)
plt.show()

#Finding the coefficient of determination
print(coeff_det(ys,reg_line))


# This regression line can now be used to make predictions for new unseen x-values.
# 

# # Basic Decision Tree
# What I feel taht, best way to learn and understand any new machine learning method is to sit down and implement the algorithm.
# Here It is , a very simple and easily understandable way to implement a  Decision Tree Algorithm.
# 

from IPython.display import Image                        
url='https://raw.githubusercontent.com/Lightning-Bug/ML-Starter-Pack/master/Decision%20Tree%20Classifier/Images/TreeSample.png'
Image(url,width=400, height=400)


# ##### Shown Above Image Is just an Example of how Our Decision Tree will look but with Different Parameters and different number of branches
# Lets go to the code part now.
# 

# Let’s imagine we want to predict rain (1) and no-rain (0) for a given day. We have two predictors:
# 
#    - **x1** is weather type (0 = partly cloudy, 1 = cloudy, 2 = sunny)
#    - **x2** is atmospheric pressure (0 = low, 1 = high)
# 

import numpy as np
x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])


# The idea behind **Decision trees** is that, given our training set, the method learns a set of rules that help us classify a new example. 
# An example rule could be: if the weather is partly cloudy and pressure is low, then it’s going to rain.
# And a few more examples you can add according to your knowledge of rain and its relation with pressure :p
# 

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}


# The idea is to split the data according to one (or multiple) attributes, so that we end up with sub-sets that (ideally) have a single outcome.
# 
# For example, if we split our training data by the attribute x1, we end up with **3 sets**, and you can see how two of the splittings are **pure, as they contain only zeros.**
# 
#    - x1 = 0: y = [0]
#    - x1 = 1: y = [0, 0]
#    - x1 = 2: y = [1, 1, 0]
#      
# The splitting for **x1 = 2** is unfortunately not pure, therefore we need to split this set into even more subsets.
# 
# The code for splitting a set is fairly simple: the following routine takes an array as input and returns a dictionary that **maps each unique value to its indices.**
# 

# ### Picking which attribute to split
# 
# An aspect that we need to figure out still is how to pick which attribute to use for the splitting. Ideally, we want the attribute that give us the better (purest) splits.
# 
# A standard measure of **“Purity”** can be obtained by taking the opposite of a quantity called Shannon entropy (if you’ve ever taken thermodynamics, you’ll know that entropy, is a measure of “Disorder” in a system).
# 
# Let’s assume we have a urn with red and green balls, we want a quantity that should be at its minimum when the urn is filled completely with green or red balls (min disorder),and at its minimum when we got half green and half red balls (maximum disorder). Given that $f_g$ is the fraction of green balls and $f_r$ is the fraction of red balls, taking the opposite of H
# 
# satisfies this property:
# ** H = − $f_r$log2($f_r$)−$f_g$log2($f_g$)**
# 

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


# You can see, in fact,
# −1*log2(1)−0*log2(0)=0 
# 
# −0.5*log2(0.5)−0.5*log2(0.5)=1
# 
# Therefore Tricks for you :
#  - If both are **same** the entropy is **1.**
#  - If any one is **zero**, then entropy is **0.**
#  
# Now that we have a measure of purity, to select the most convenient attribute for splitting, we should check if the sets improves the purity than the un-splitted set.
# 
# This measure of purity improvement can be described mathematically through a quantity called mutual information (in the decision tree literature this is often referred as information gain).
# 
# Mutual information is the difference between the entropy of the unsplitted set, and the average of the entropy of each split, weighted by the number of elements in the subset. A concrete example is as follows:
# I(y,x)=H(y)−[px=0H(y|x=0)+px=1H(y|x=1))]
# 
# where:
# - y is the original set
# - x is the attribute we are using for splitting that assumes the values {0, 1}
# - H(y∥x=k) is the entropy of the subset that corresponds to the attribute value x=k, and px=k is the proportion of elements in that subset. The implementation is again straightforward.
# 


def mutual_information(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


# To summarize, the general idea is as follow:
# 
#    - Select the most convenient attribute using the mutual information criterion.
#    - Split using the selected attribute
#    - For every subset, if the subset is not pure (or empty), recursively split this subset by picking another attribute (until you ran out of attributes).
# 

from pprint import pprint

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y


    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%d = %d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res

X = np.array([x1, x2]).T
pprint(recursive_split(X, y))


# **There’s much more about decision trees, but with these building blocks you've got a kick start and its not too hard to understand how to go about it, So, just go and Explore :) **
# 




# I am implementing here a simple 3-layer neural network from scratch. Although created, you might understand that the network is inefficient.
# 

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
# Default figure size
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(10.0,8.0)


# I am now generating a dataset using the make_moons function.
# 

# Generate a dataset and plot it
np.random.seed(0)
X,y=sklearn.datasets.make_moons(200,noise=0.20)
plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.Spectral)


# From the above scatter plot, The dataset we generated has two classes, plotted as red and blue points. You can think of the blue dots as male patients and the red dots as female patients, with the x- and y- axis being medical measurements
# 

# Our goal is to train a Machine Learning classifier that predicts the correct class (male or female) given the x- and y- coordinates. Note that the data is not linearly separable, we can't draw a straight line that separates the two classes. This means that linear classifiers, such as Logistic Regression, won't be able to fit the data unless you hand-engineer non-linear features (such as polynomials) that work well for the given dataset.
# 

# Helper function to plot decision boundary
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# Let's now build a 3-layer neural network with one input layer, one hidden layer, and one output layer. The number of nodes in the input layer is determined by the dimensionality of our data, 2. Similarly, the number of nodes in the output layer is determined by the number of classes we have, also 2. (Because we only have 2 classes we could actually get away with only one output node predicting 0 or 1, but having 2 makes it easier to extend the network to more classes later on). The input to the network will be x- and y- coordinates and its output will be two probabilities, one for class 0 ("female") and one for class 1 ("male").
# 

# We start by defining parameters and variables for gradient descent
# 

# Training set size
num_examples=len(X)

# Setting dimensionalities
nn_input_dim=2
nn_output_dim =2
# Gradient descent parameters
epsilon=0.01
reg_lambda=0.01


# Helper function to evaluate total loss on dataset
def calculate_loss(model):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    z1=X.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    exp_scores=np.exp(z2)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    # Calculating loss
    correct_logprobs=-np.log(probs[range(num_examples),y])
    data_loss=np.sum(correct_logprobs)
    # Add regularization term
    data_loss+=reg_lambda/2*(np.sum(np.square(W1))+np.sum(np.square(W2)))
    return 1./num_examples*data_loss


# We also implement a helper function to calculate the output of the network. It does forward propagation as defined above and returns the class with the highest probability.
# 

# Helper function to predict output
def predict(model,x):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    # Forward Propagation
    z1=x.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    exp_scores=np.exp(z2)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    return np.argmax(probs,axis=1)


# This function learns parameters for the neural network and returns the model
def build_model(nn_hdim,num_passes=20000,print_loss=False):
    np.random.seed(0)
    W1=np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim)
    b1=np.zeros((1,nn_hdim))
    W2=np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim)
    b2=np.zeros((1,nn_output_dim))
    model={}
    for i in range(0,num_passes):
    	# Forward propagation
        z1=X.dot(W1)+b1
        a1=np.tanh(z1)
        z2=a1.dot(W2)+b2
        exp_scores=np.exp(z2)
        probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        # Backward propagation
        delta3=probs
        delta3[range(num_examples),y]-=1
        dW2=(a1.T).dot(delta3)
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(a1,2))
        dW1=np.dot(X.T,delta2)
        db1=np.sum(delta2,axis=0)
        # Adding regularization 
        dW2+=reg_lambda*W2
        dW1+=reg_lambda*W1
        # Gradient descent parameter update
        W1+= -epsilon*dW1
        b1+=-epsilon*db1
        W2+=-epsilon*dW2
        b2+=-epsilon*db2
        # Assign new paramters to model
        model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        if print_loss and i %1000 ==0:
            print("Loss after iteration %i: %f"%(i,calculate_loss(model)))
        return model


# A network with hidden layer of size 3 
# 

# Build a model with three-dimensional hidden layer           
model= build_model(3,print_loss=True)
plot_decision_boundary(lambda x:predict(model,x))
plt.title("Decision boundary layer for hidden layer size")
plt.show()


    
    



plt.figure(figsize=(16,32))
hidden_layer_dimensions=[1,2,3,4,5,20,50]
for i,nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5,2,i+1)
    plt.title('hidden layer size %d'% nn_hdim)
    model=build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model,x))
    
    





# # Modelling a Support Vector Machine
# 
# In this notebook, a basic version of an SVM is implemented. This is only for understanding the inner working of the classifier. By no means is this implementation optimized enough to be applied on a large scale.
# 

# # SVM
# An SVM model is a representation of the data as points in space, mapped so that the instances of the separate categories are divided by a clear gap that is as wide as possible. In other words, an SVM is a Large Margin Classifier.
# 

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


# **SVM CLASS** It is useful to model an SVM as an object because training an SVM is a computationally expensive task. Therefore, in order to not retrain for every new test data, we build an object. Once trained, the classifier is very quick at predicting the label of new data.
# 
# It is recommended to go through more comprehensive resources for understanding the crux of the SVM problem.
# SVM Training is a Convex Optimization Problem. As we have already stated, the model attempts to find out the hyperplane causing maximum separation of the classes. Thus, minimizing the margin, when stated mathematically, is a Convex Opt. problem.
# 

class Support_Vector_Machine:
    # initializing the svm class object
    def __init__(self,visualization=True):
        # visualization (bool): whether or not the user wants to visually see the decision boundary hyperplane
        # colors : Setting the colors of different classes. Red for positive class. Blue for negative class.        
        
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    #train
    def fit(self,data):
        # data (dict): input datapoints([x0,x1]) with attached class label (1 or -1)
        
        self.data = data
        
        # opt_dict stores the possible [w,b] values for a given magnitude for w 
        # {||w|| : [w,b] }
        opt_dict = {}
        
        # these transforms when applied to w yield same magnitude but possibly lower value of cost function
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]
        
        # getting the maximum and the minimum feature value from the given dataset
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                   
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        
        # step sizes to be taken while optimizing the cost function
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,]
        

        b_range_multiple = 5
        # we dont need to take as small steps with b as with w
        b_multiple = 5
        
        # latest_optimum stores the most recent value of the cost function while optimization
        # start with a high value
        latest_optimum = self.max_feature_value * 10
        
        
        # Convex Optimization Loop
        for step in step_sizes:
            # initialize w for the iteration with the latest optimimum value
            w = np.array([latest_optimum,latest_optimum])
            
            # we can do this because convex
            optimized = False
            
            while not optimized:
                # Looping over values of b
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                  self.max_feature_value*b_range_multiple, 
                                  step*b_multiple):
                    
                    # Consider each transformed value of w as w_t
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        
                        # checking if the dataset satisfies the found combination of w_t and b 
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                
                                # the training datapoints must lie outside the SV margins 
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                         
                        # if valid combination is found, store it
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                
                # check if we jumped over the minima
                if w[0]<0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
            
            # optimized for this step size
            # assign the most optimum values obtained for w and b
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
        
    # predict a test point   
    def predict(self,features):
        # sign(x.w+b) decides the class of the test datapoint
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        
        # if the user wishes to view the svm then plot the test points
        if classification != 0  and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
        return classification
    
    
    # to visually see the SVM in action
    def visualize(self):
        # plot the training datapoints
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        # hyperplane = x.w + b
        # v = x.w + b
        # values of v for:
        # psv = 1
        # nsv = -1
        # dec = 0
        
        # returns the y-coordinate(x1) for plotting the point with x0 abcissa
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]
        
        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        
        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')
        
        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')
        
        # (w.x+b) = 0
        # decision boundary hyperplane
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')
        
        plt.show()


# input dataset
data_dict = {
    -1: np.array([[1,7],
                  [2,8],
                  [3,8]]),
    1: np.array([[5,1],
                 [6,-1],
                 [7,3]])
}


# train the model
svm = Support_Vector_Machine()
svm.fit(data=data_dict)


# predicting new unseen data
predict_us = [[0,10],
             [1,3],
             [3,4],
             [3,5],
             [5,5],
             [5,6],
             [6,-5],
             [5,8]]

for p in predict_us:
    svm.predict(p)
    
svm.visualize()    


# ##  Support Vector Machines(SVM)
# It is a supervised machine learning algorithm which can be used for classification as well as regression problems.
# In this notebook we will see input features belonging to two seperate classes are scattered.The SVM will find out the best hyperplane to seperate the two classes.
# 

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pickle
import pandas as pd
import numpy as np
import math                            #Import Modules
import random
from sklearn import svm


# Extracting the dataset and putting x features in z1 and y labels in z2. x1,x2 are the feauture points when y=+1. y1,y2 are the feauture points when y=-1.
# 

df=pd.read_csv("ex8a.txt", sep=' ',
                  names = ["Label", "x1", "x2"])
x1=[]
x2=[]
y1=[]                  #Extract Data
y2=[]
z1=[]
z2=[]
arr=[]
for i in range(len(df["x1"])):
    arr.append(0)
    if int(df["Label"][i])==1:
        s=[]
        q=str(df["x2"][i])
        k=str(df["x1"][i])
        x1.append(float(k[2:len(k)]))
        x2.append(float(q[2:len(q)]))
        s.append(float(k[2:len(k)]))
        s.append(float(q[2:len(q)]))
        z1.append(s)
        z2.append(1)
    else:
        s=[]
        q=str(df["x2"][i])
        k=str(df["x1"][i])
        y1.append(float(k[2:len(k)]))
        y2.append(float(q[2:len(q)]))
        s.append(float(k[2:len(k)]))
        s.append(float(q[2:len(q)]))
        z1.append(s)
        z2.append(-1)


# Kernel function helps in computing high-dimensional feature vectors.The kernel used here is the rbf kernel.It maps the input features in an efficient way.A matrix is created that precomputes all the kernel values pre-hand. This matrix is know as the gram matrix(ker) which is computationally efficient. 
# 


ker=np.zeros((len(z2),len(z2)))

#Creating a Gram Matrix

def kernel(k1,k2):
    s=100*(np.linalg.norm(np.array(k1)-np.array(k2))**2)
    return np.exp(-s)
for i in range(len(z2)):
    for j in range(len(z2)):
        ker[i][j]+=kernel(z1[i],z1[j])
with open("dic.pickle","wb") as f:
    pickle.dump(ker,f)


with open("dic.pickle","rb") as f: #Loading the pickle file
    ker=pickle.load(f)


def kernel(k1,k2):               #Rbf Kernel Function
    s=100*(np.linalg.norm(np.array(k1)-np.array(k2))**2)
    return np.exp(-s)


# a is the lagrange multiplier alpha and y is the label. The value of c is the total sum of the x_test datapoint with all the x inputs and there corresponding alpha and label. The x_test and the input x mapping is done using the kernel function.
# 

def svm_algorithm(j,y,a,b): #Running the kernel
    c=0
    for i in range(len(y)):
        c+=(a[i]*y[i]*(ker[i,j]))
    return (c+b)


# This computes the same thing as above but the only difference is that the gram matrix(ker) created is not used. 
# 

def svm_algorithmplot(x_test,x,y,a,b): #Running the kernel for visualisation
    c=0
    for i in range(len(y)):
        c+=(a[i]*y[i]*(kernel(x_test,x[i])))
    return (c+b)


# maxandmin calculates the upper(H) and lower(L) bound for the lagrange multiplier alpha(a).
# 

def maxandmin(y1,y2,a1,a2,c):  #SMO Min and Max Calculator
    if y1!=y2:
        k=[max(0,a2-a1),min(c,c+a2-a1)]
    else:
        k=[max(0,a2+a1-c),min(c,a2+a1)]
    return k


# SMO algorithm is used to optimize the Support Vector Machine(SVM). It is a mathematically a bit complex algorithm as it solves a dual optimization problem.Please read the SMO.pdf for more details about algorithm.
# 

def smo_optimization(x,y,arr,bias,c,maxpass,tol=0.001):  #SMO Algorithm 
    a=arr
    b=bias
    iter=0
    while (iter<maxpass):
        numalphas=0
        z=len(y)
        for i in range(z):
            s=svm_algorithm(i,y,a,b)-y[i]
            if ((y[i]*s < -tol and a[i]<c) or (y[i]*s >tol and a[i]>0)):
                k=random.randint(0,z-1)
                t=svm_algorithm(k,y,a,b)-y[k]
                ai_old=a[i]
                ak_old=a[k]
                d=maxandmin(y[i],y[k],a[i],a[k],c)
                if (d[0]==d[1]):
                    continue
                neta=(2*ker[i,k])-ker[i,i]-ker[k,k]
                if neta>=0:
                    continue
                a[k]=a[k]-((y[k]*(s-t))/neta)
                if (a[k]>d[1]):
                    a[k]=d[1]
                elif (a[k]<d[0]):
                    a[k]=d[0]
                else:
                    a[k]=a[k]
                if abs(a[k]-ak_old)<0.00001:
                    continue
                a[i]=a[i]-(y[i]*y[k]*(a[k]-ak_old))
                b1=b-s-(y[i]*(a[i]-ai_old)*ker[i,i])-(y[k]*(a[k]-ak_old)*ker[i,k])
                b2=b-t-(y[i]*(a[i]-ai_old)*ker[i,k])-(y[k]*(a[k]-ak_old)*ker[k,k])
                if (a[i]>0 and a[i]<c):
                    b=b1
                elif (a[k]>0 and a[k]<c):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                numalphas+=1
            if numalphas==0:
                iter+=1
            else:
                iter=0
    return ([a,b])
sumo=smo_optimization(z1,z2,arr,0,1,20)
with open("alphas.pickle","wb") as f:
    pickle.dump(sumo,f)


with open("alphas.pickle","rb") as f:
    sumo=pickle.load(f)


accuracy=0
ks=[]
for i in range(len(z2)):
    ts= svm_algorithm(i,z2,sumo[0],sumo[1])    #SVM algorithm is run and the labels are predicted. 
    if ts>0:
        ks.append(1)
    else: 
        ks.append(-1)
for i in range(len(z2)):
    if (ks[i]==z2[i]):
        accuracy+=1
print ("The Accuracy of the Support Vector Machine is",(accuracy/len(z2)*100),'%')


# The steps done below are used for visualization. They are followed from the scikit-learn documentaion.
# 

z1=np.array(z1) #Preparing for Visualization
z2=np.array(z2)
h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = z1[:, 0].min() - 1, z1[:, 0].max() + 1
y_min, y_max = z1[:, 1].min() - 1, z1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


ss=[]
st=np.c_[xx.ravel(), yy.ravel()]
for i in range(len(st)):                                 #SVM algorithm is run and the labels are predicted.
    tt= svm_algorithmplot(st[i],z1,z2,sumo[0],sumo[1])    
    if tt>0:
        ss.append(1)
    else: 
        ss.append(-1)
with open("visuals.pickle","wb") as f:
    pickle.dump(ss,f)


# Finally the datapoints are visualized along with the hyperplane.
# 

with open("visuals.pickle","rb") as f:
    ss=pickle.load(f)
Z=np.array(ss)
plt.scatter( z1[:, 0], z1[:, 1],c=np.array(z2),cmap=plt.cm.Spectral,label="points")
Z=Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Wistia,label="boundary")
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title('Support Vector Machines')
plt.legend()
plt.show()





# # Collaborative Filtering On Anime Data
# 

# import relevant libraries 

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import operator
get_ipython().run_line_magic('matplotlib', 'inline')


anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')


# Before alteration the ratings dataset uses a "-1" to represent missing ratings.
# I'm replacing these placeholders with a null value because I will later be calculating 
# the average rating per user and don't want the average to be distorted
# 

rating.rating.replace({-1: np.nan}, regex=True, inplace = True)
rating.head()


# I'm only interest in finding recommendations for the TV category

anime_tv = anime[anime['type']=='TV']
anime_tv.head()


# Join the two dataframes on the anime_id columns

merged = rating.merge(anime_tv, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)


# For computing reasons I'm limiting the dataframe length to the first 10,000 users

merged=merged[['user_id', 'name', 'user_rating']]
merged_sub= merged[merged.user_id <= 10000]
merged_sub.head()


# For collaborative filtering we'll need to create a pivot table of users on one axis and tv show names along the other. The pivot table will help us in defining the similarity between users and shows to better predict who will like what.
# 

piv = merged_sub.pivot_table(index=['user_id'], columns=['name'], values='user_rating')


print(piv.shape)
piv.head()


# Normalize the values
piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)


# Drop all columns containing only zeros representing users who did not rate
piv_norm.fillna(0, inplace=True)
piv_norm = piv_norm.T
piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]


# Our data needs to be in a sparse matrix format to be read by the following functions
piv_sparse = sp.sparse.csr_matrix(piv_norm.values)


# These matrices show us the computed cosine similarity values 
# between each user/user array pair and item/item array pair.
# 

item_similarity = cosine_similarity(piv_sparse)
user_similarity = cosine_similarity(piv_sparse.T)


# Inserting the similarity matricies into dataframe objects

item_sim_df = pd.DataFrame(item_similarity, index = piv_norm.index, columns = piv_norm.index)
user_sim_df = pd.DataFrame(user_similarity, index = piv_norm.columns, columns = piv_norm.columns)


# This function will return the top 10 shows with the highest cosine similarity value

def top_animes(anime_name):
    count = 1
    print('Similar shows to {} include:\n'.format(anime_name))
    for item in item_sim_df.sort_values(by = anime_name, ascending = False).index[1:11]:
        print('No. {}: {}'.format(count, item))
        count +=1  


# This function will return the top 5 users with the highest similarity value 

def top_users(user):
    
    if user not in piv_norm.columns:
        return('No data available on user {}'.format(user))
    
    print('Most Similar Users:\n')
    sim_values = user_sim_df.sort_values(by=user, ascending=True).loc[:,user].tolist()[1:11]
    sim_users = user_sim_df.sort_values(by=user, ascending=True).index[1:11]
    zipped = zip(sim_users, sim_values,)
    for user, sim in zipped:
        print('User #{0}, Similarity value: {1:.2f}'.format(user, sim)) 


# This function constructs a list of lists containing the highest rated shows per similar user
# and returns the name of the show along with the frequency it appears in the list

def similar_user_recs(user):
    
    if user not in piv_norm.columns:
        return('No data available on user {}'.format(user))
    
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    best = []
    most_common = {}
    
    for i in sim_users:
        max_score = piv_norm.loc[:, i].max()
        best.append(piv_norm[piv_norm.loc[:, i]==max_score].index.tolist())
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1
    sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list[:5]    


# This function calculates the weighted average of similar users
# to determine a potential rating for an input user and show

def predicted_rating(anime_name, user):
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:1000]
    user_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:1000]
    rating_list = []
    weight_list = []
    for j, i in enumerate(sim_users):
        rating = piv.loc[i, anime_name]
        similarity = user_values[j]
        if np.isnan(rating):
            continue
        elif not np.isnan(rating):
            rating_list.append(rating*similarity)
            weight_list.append(similarity)
    return sum(rating_list)/sum(weight_list)    


top_animes('Cowboy Bebop')


top_users(3)


similar_user_recs(3)


predicted_rating('Cowboy Bebop', 3)


# Below we'll see how the predict_rating function performs compared to the observed rated values for user 3.
# 

# Creates a list of every show watched by user 3

watched = piv.T[piv.loc[3,:]>0].index.tolist()


# Make a list of the squared errors between actual and predicted value

errors = []
for i in watched:
    actual=piv.loc[3, i]
    predicted = predicted_rating(i, 3)
    errors.append((actual-predicted)**2)


# This is the mean squared error for user 3
np.mean(errors)


# ##  Support Vector Machines(SVM)
# It is a supervised machine learning algorithm which can be used for classification as well as regression problems.
# In this notebook we will see input features belonging to two seperate classes. The SVM will find out the best hyperplane to seperate the two classes.
# 

import matplotlib.pyplot as plt  #Importing Modules
from matplotlib import style
style.use('ggplot')
import pandas as pd
import numpy as np
import math
import random 


# Extracting the dataset and putting x features in z1 and y labels in z2. x1,x2 are the feauture points when y=+1. y1,y2 are the feauture points when y=-1.
# 

df=pd.read_csv("twofeature.txt", sep=' ',
                  names = ["Label", "x1", "x2"])
x1=[]
x2=[]
y1=[]
y2=[]
z1=[]
z2=[]
arr=[] 
for i in range(len(df["x1"])):           
    arr.append(0)                                    
    if int(df["Label"][i])==1:             
        s=[]
        q=str(df["x2"][i])
        k=str(df["x1"][i])
        x1.append(float(k[2:len(k)]))
        x2.append(float(q[2:len(q)]))
        s.append(float(k[2:len(k)]))
        s.append(float(q[2:len(q)]))
        z1.append(s)
        z2.append(1)
    else:
        s=[]
        q=str(df["x2"][i])
        k=str(df["x1"][i])
        y1.append(float(k[2:len(k)]))
        y2.append(float(q[2:len(q)]))
        s.append(float(k[2:len(k)]))
        s.append(float(q[2:len(q)]))
        z1.append(s)
        z2.append(-1)


# Kernel function helps in computing high-dimensional feature vectors.It maps the input features in an efficient way.For this notebook the simplest linear kernel function is used.
# 

def kernel(k1,k2):#Kernel Function
    return np.dot(k1,k2) 


# a is the lagrange multiplier alpha and y is the label. The value of c is the total sum of the x_test datapoint with all the x inputs and there corresponding alpha and label. The x_test and the input x mapping is done using the kernel function. 
# 

def svm_algorithm(x,x_test,y,a,b): #SVM algorithm
    c=0                           
    for i in range(len(y)):
        c+=(a[i]*y[i]*kernel(x[i],x_test))
    return (c+b)


# maxandmin calculates the upper(H) and lower(L) bound for the lagrange multiplier alpha(a).
# 

def maxandmin(y1,y2,a1,a2,c):  #To find L and H
    if y1!=y2:
        k=[max(0,a2-a1),min(c,c+a2-a1)]
    else:
        k=[max(0,a2+a1-c),min(c,a2+a1)]
    return k


# SMO algorithm is used to optimize the Support Vector Machine(SVM). It is a mathematically a bit complex algorithm as it solves a dual optimization problem.Please read the SMO.pdf for more details about algorithm.
# 

def smo_optimization(x,y,arr,bias,c,maxpass,tol=0.001): #Solving SVM using SMO algorithm
    a=arr                             #Read the SMO.pdf for details
    b=bias
    iter=0
    while (iter<maxpass):
        numalphas=0
        z=len(y)
        for i in range(z):
            s=svm_algorithm(x,x[i],y,a,b)-y[i]
            if ((y[i]*s < -tol and a[i]<c) or (y[i]*s >tol and a[i]>0)):
                k=random.randint(0,z-1)
                t=svm_algorithm(x,x[k],y,a,b)-y[k]
                ai_old=a[i]
                ak_old=a[k]
                d=maxandmin(y[i],y[k],a[i],a[k],c)
                if (d[0]==d[1]):
                    continue
                neta=(2*kernel(x[i],x[k]))-kernel(x[i],x[i])-kernel(x[k],x[k])
                if neta>=0:
                    continue
                a[k]=a[k]-((y[k]*(s-t))/neta)
                if (a[k]>d[1]):
                    a[k]=d[1]
                elif (a[k]<d[0]):
                    a[k]=d[0]
                else:
                    a[k]=a[k]
                if abs(a[k]-ak_old)<0.00001:
                    continue
                a[i]=a[i]-(y[i]*y[k]*(a[k]-ak_old))
                b1=b-s-(y[i]*(a[i]-ai_old)*kernel(x[i],x[i]))-(y[k]*(a[k]-ak_old)*kernel(x[i],x[k]))
                b2=b-t-(y[i]*(a[i]-ai_old)*kernel(x[i],x[k]))-(y[k]*(a[k]-ak_old)*kernel(x[k],x[k]))
                if (a[i]>0 and a[i]<c):
                    b=b1
                elif (a[k]>0 and a[k]<c):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                numalphas+=1
            if numalphas==0:
                iter+=1
            else:
                iter=0
    return ([a,b])


# C is the l1 regularization term.The C parameter is used to handle outliers.The algorithm is run with value of C as 1.
# 

#Running the algorithm with C=1
sumo=smo_optimization(z1,z2,arr,0,1,1000)
accuracy=0
ks=[]
for i in range(len(z2)):
    ts= svm_algorithm(z1,z1[i],z2,sumo[0],sumo[1])    #SVM algorithm is run and the labels are predicted. 
    if ts>0:
        ks.append(1)
    else:
        ks.append(-1)
for i in range(len(z2)):
    if (ks[i]==z2[i]):
        accuracy+=1
print ("The Accuracy of the Support Vector Machine is",(accuracy/len(z2)*100),'%')
w0=0
w1=0
for i in range(len(z1)):
    w0+=sumo[0][i]*z2[i]*z1[i][0]
    w1+=sumo[0][i]*z2[i]*z1[i][1]
a=-w0/w1                                          #Slope of the hyperplane is found out.
xx = np.linspace(0,5)                             #Datapoints are now visualized.
yy = (a*xx) -(sumo[1]/w1)
plt.plot(xx,yy,'k-',color='plum',label='Hyperplane')
plt.scatter(x1,x2,color='r',label='+1')
plt.scatter(y1,y2,color='b',label='-1')
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Support Vector Machines')
plt.legend()
plt.show()


# C is the l1 regularization term.The C parameter is used to handle outliers.The algorithm is run with value of C as 100.
# 

#Running the algorithm with C=100
sumo=smo_optimization(z1,z2,arr,0,100,1000)
accuracy=0
ks=[]
for i in range(len(z2)):
    ts= svm_algorithm(z1,z1[i],z2,sumo[0],sumo[1]) #SVM algorithm is run and the labels are predicted. 
    if ts>0:
        ks.append(1)
    else:
        ks.append(-1)
for i in range(len(z2)):
    if (ks[i]==z2[i]):
        accuracy+=1
print ("The Accuracy of the Support Vector Machine is",(accuracy/len(z2)*100),'%')
w0=0
w1=0
for i in range(len(z1)):
    w0+=sumo[0][i]*z2[i]*z1[i][0]
    w1+=sumo[0][i]*z2[i]*z1[i][1]
a=-w0/w1                                                      #Slope of the hyperplane is found out.
xx = np.linspace(0,5)                                          #The datapoints are visualized.
yy = (a*xx) -(sumo[1]/w1)
plt.plot(xx,yy,'k-',color='plum',label='Hyperplane')
plt.scatter(x1,x2,color='r',label='+1')
plt.scatter(y1,y2,color='b',label='-1')
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Support Vector Machines')
plt.legend()
plt.show()





# # Basic Decision Tree from scratch
# 
# Go the defination and algorithm of Decision Tree .
# 
# Below I have given a stepby step example and explained in each step.
# 

url='https://raw.githubusercontent.com/Lightning-Bug/ML-Starter-Pack/master/Decision%20Tree%20Classifier/Images/BasicPic.png'

from IPython.display import Image                        
Image(url,width=400, height=400)


# **<em> Here is the visualization of the table we will be implementing in the below code <em/>**
# 

from __future__ import print_function


# <br />
# Format: each row is an example. <br />
# The last column is the label. <br />
# The first two columns are features.
# 

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]


# These are used only to print the tree. <br/>
# 

header = ["color", "diameter", "label"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


# Demo:
unique_vals(training_data, 0)
# unique_vals(training_data, 1)


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


#Demo:
class_counts(training_data)


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


is_numeric(7)
# is_numeric("Red")


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


# Let's write a question for a numeric attribute
Question(1, 3)


# -> How about one for a categorical attribute
# 


q = Question(0, 'Green')
q


# Let's pick an example from the training set...
example = training_data[0]
#and see if it matches the question
q.match(example) # this will be true, since the first example is Green.


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# Let's partition the training data based on whether rows are Red.
# 

true_rows, false_rows = partition(training_data, Question(0, 'Red'))
# This will contain all the 'Red' rows.
true_rows


# This will contain everything else.
false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


# **Let's look at some example to understand how Gini Impurity works.**
# 

#First, we'll look at a dataset with no mixing.
no_mixing = [['Apple'],
              ['Apple']]
# this will return 0
gini(no_mixing)


# Now, we'll look at dataset with a 50:50 apples:oranges ratio
# 

some_mixing = [['Apple'],
               ['Orange']]
# this will return 0.5 - meaning, there's a 50% chance of misclassifying
# a random example we draw from the dataset.
gini(some_mixing)



# Now, we'll look at a dataset with many different labels
# 

lots_of_mixing = [['Apple'],
                  ['Orange'],
                  ['Grape'],
                  ['Grapefruit'],
                  ['Blueberry']]
# This will return 0.8
gini(lots_of_mixing)
#######


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


# Calculate the uncertainy of our training data.
current_uncertainty = gini(training_data)
current_uncertainty


# How much information do we gain by partioning on 'Green'?
# 

true_rows, false_rows = partition(training_data, Question(0, 'Green'))
info_gain(true_rows, false_rows, current_uncertainty)



true_rows, false_rows = partition(training_data, Question(0,'Red'))
info_gain(true_rows, false_rows, current_uncertainty)


true_rows, false_rows = partition(training_data, Question(0,'Red'))

# Here, the true_rows contain only 'Grapes'.
true_rows


false_rows


# On the other hand, partitioning by Green doesn't help so much.
true_rows, false_rows = partition(training_data, Question(0,'Green'))

# We've isolated one apple in the true rows.
true_rows


false_rows


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


# Find the best question to ask first for our toy dataset.
# 

best_gain, best_question = find_best_split(training_data)
best_question
# FYI: is color == Red is just as good. See the note in the code above
# where I used '>='.


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# Keep This things in mind, we'll be doing this in the next step
#    - Try partitioing the dataset on each of the unique attribute,
#    - calculate the information gain,
#    - and return the question that produces the highest gain.
# 

def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


my_tree = build_tree(training_data)


print_tree(my_tree)


# **-----------------------------------------------------------------------------------------------------------------**
# Now, 
#   - Decide whether to follow the true-branch or the false-branch.
#   - Compare the feature / value stored in the node,
#   - to the example we're considering.
# 

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# - The tree predicts the 1st row of our
# - training data is an apple with confidence 1.
# 

classify(training_data[0], my_tree)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


print_leaf(classify(training_data[0], my_tree))
##


# On the 2nd example, the confidence is lower
print_leaf(classify(training_data[1], my_tree))


# Evaluate
testing_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]


for row in testing_data:
    print ("Actual: %s. Predicted: %s" %
           (row[-1], print_leaf(classify(row, my_tree))))


# **Here You have seen a very basic Example of prediction by certain rules(or questions which we made) and the data was also ours.
# Now you are ready to play with big sets and create your own decision tree, just follow the same procedure and design your own decision tree and have a kickass time. <br/>
# ALL THE BEST ;) **.
# 




# # Visualizing the Data
# 

# import Useful libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

colors = sns.color_palette()


# load datasets and store into variables
anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')


# Lets see how the mean rating fluxuates across each category of anime in the data
# 

anime = anime.dropna()
plt.figure(figsize=(10,8))

for i, col in enumerate(anime.type.unique()):
    ax = plt.subplot(3, 2, i + 1)
    plt.yticks([.5, .4, .3, .2, .1, 0])
    plt.ylim(ymax=.6)
    sns.kdeplot(anime[anime['type']==col].rating, shade=True, label=col, color = colors[i])   


# There is definitely a rating difference across media category. Music and ONA (Original Net Animation) tend to be viewed less favorably than mediums such as TV and Movies which have a larger mean rating.
# 

rating.rating.replace({-1: np.nan}, regex=True, inplace = True)


anime_tv = anime[anime['type']=='TV']
anime_tv.head()


# join the two dataframes on the anime_id columns
merged = rating.merge(anime_tv, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)


# reogranize the dataframe to the desired columns
merged = merged[['user_id', 'anime_id', 'name', 'genre', 'type', 'episodes', 'user_rating', 'rating']]
merged.head()


# Lets see how the critc review varies from the users' for a given show

merged[merged['name']=='One Punch Man'].user_rating.plot(kind='hist', color = colors[0], alpha=.7)
plt.axvline(merged[merged['name']=='One Punch Man'].rating.mean(), color=colors[0], linestyle='dashed', linewidth=2, label='Critic Rating')

plt.xlabel('Audience Rating')
plt.title('One Punch Man User Rating Histogram')
plt.legend(loc = 'upper left')


# A quick check on how one of my favorite show's user ratings compare with it's critic score
# 

# Shows with the highest count of 10 star ratings
highest_10_count = merged[merged.user_rating == 10].groupby('name').rating.count().sort_values(ascending = False)


highest_10_count[:10 -1].plot(kind='bar', width = 0.8,color = colors[0], alpha=0.7)
plt.title('Count of 10* Ratings')


# Series of average rating per user
user_rating_mean = merged.groupby('user_id').user_rating.mean().dropna()


sns.kdeplot(user_rating_mean, shade=True, color = colors[3], label='User Rating') 
plt.xticks([2,4,6,8,10])
plt.xlim(0,11)
plt.title("Density of Users' Average Rating")
plt.xlabel('Rating')


# This graph gives us an idea of how users rated. It reflects that users' in this dataset have a tendancy to rate things pretty favorably. With a spike at 10 representing the users who only rated one (or very few) items perfectly.
# 

# series of user standard deviations
user_std = merged.groupby('user_id').user_rating.std().dropna()


sns.kdeplot(user_std, shade=True, color = colors[2], label='User Std') 
plt.xlim(-1,5)
plt.title('Density of User Standard Deviations')
plt.xlabel('Standard Deviation')


# This chart shows us that the majority of users ratings don't vary too greatly (only a couple points). Again a spike on the margin suggesting users who only voted for a single show
# 

# Series of user rating counts
user_rating_count = rating.dropna().groupby('user_id').rating.count()


user_rating_count.plot(kind='kde', color=colors[1])
plt.axvline(rating.dropna().groupby('user_id').rating.count().mean(), color=colors[1], linestyle='dashed', linewidth=2)
plt.xlim(-100, 400)
plt.xlabel('Rating Count')
plt.title('Density of Ratings Count Per User')


# This graph depicts an intense skew to the right, although the majority of people rate only a few titles, some users have rated hundreds or more.
# 
# ### Conclusion :
# Here you have learnt how to visualize your data in different plots and shading them for more convinience.<br/>
# That's it for this one.<br/>
# **ALL THE BEST :)**
# 




# # Naive Bayes
# 
# It is a generative learing algorithm and a popular classification algorithm .The algorithm makes the assumption that x's are conditionally independent of the y. This assumption is called Naive Bayes assumption.
# In this notebook we will use Naive Bayes to build an email spam filter that will classify messages according to whether they are unsolicited commercial (spam) email, or non-spam email. Classifying emails is one example of a broader set of problems called text classification.
# 

import pandas as pd
import numpy as np #Importing Important Modules
import math
import pickle


# Created a dictionary with 2500 keys(words) and setting their value to 1. The reason of putting the value of 1
# instead of zero is because of the laplace smoothing of the numerator.If 0 was used then there would be a case that a word never occured in training but is present in the test case so at that point the probability will go to zero. This problem is fixed by laplace.After reading the data was pickled. Pickling helps faster reading of dataset.
# 

dic1={}          #dic1 contains words appeared in non spam emails.
dic2={}          #dic2 contains words appeared in spam emails.
for i in range(1,2501):
    dic1.update({i:1})
    dic2.update({i:1})
k=[dic1,dic2]
with open("dic.pickle","wb") as f:
    pickle.dump(k,f)


with open("dic.pickle","rb") as f:
    k=pickle.load(f)                                        #k[0] contains words appeared in non spam emails.
                                                            #k[1] contains words appeared in spam emails.
v=2500
df=pd.read_csv("train-features.txt", sep=' ',
                  names = ["DocID", "DicID", "Occ"])
s=df["DocID"]

#reading the file and giving them respective headers
#DocId- Document number,DicID-Dictionary token number (1-2500),Occ-No. of times occured in the respective document.



# c variable is the DocId counter. The first value of DocId is 1.r is counting the total number of words occurances(occ) in a doc file. The loop is run to go through all the files. When the doc id changes the else part executes so then c value is incremented,r's value is appended in the list a. Now r is reset to 0 and the new r value of the next doc is added. 
# 

##Training the classifier
c=1
r=0                       #Counting the length of each words in the document
a=[]                       #a is a list of all the lengths of document like a[0] is the no. of words in first document
for i in range(len(s)):
    if (s[i])==c:
        r+=df["Occ"][i]
    else:
        a.append(r)
        c+=1                                     
        r=r-r
        r+=df["Occ"][i] 
a.append(r)
b=a[0:350]             #Dividing the lenghts into two lists. As 0-350 documents are not spam(0) and 350-700 are spam(1) 
a=a[350:700]
nsp=sum(b)+v   #v is length of the dictionary ie 2500, it is added due to laplace smoothing
sp=sum(a)+v
sums=[nsp,sp]
with open("dicsum.pickle","wb") as f:
   pickle.dump(sums,f)                    #Pickling is used for faster loading of Data.

sums=[]
with open("dicsum.pickle","rb") as f:
   sums=pickle.load(f)


# k[0] is the not-spam dictionary and k[1] is the spam dictionary in which the occurances of each word is added according to the 2500 keys(words).
# 

for i in range(len(s)):              #Updating the non spam and spam dictionary by adding the occurance of the word.
    if int(s[i])<=350:
        k[0][(df["DicID"][i])]+=df["Occ"][i]
    else:
        k[1][(df["DicID"][i])]+=df["Occ"][i]
            
with open("classydicl.pickle","wb") as f:
   pickle.dump(k,f)


with open("classydicl.pickle","rb") as f:
    q=pickle.load(f)                    #Our numerator and denominator are both ready.Now we Divide.

for keys in (q[0]):
    q[0][keys]=np.divide(q[0][keys],sums[0])
    q[1][keys]=np.divide(q[1][keys],sums[1])
    

with open("newclassydic.pickle","wb") as f:
   pickle.dump(q,f)


with open("newclassydic.pickle","rb") as f:
    k=pickle.load(f)
#newclassydic is our trained classifier
#k loads the new classifier k[0] contains non spam and k[1] contains spam.


# The classifier has been trained on the training dataset.There are 50% spam and 50% non-spam labels in our training dataset.Log is used to prevent underflow errors.It's time to check accuracy on the test data set. The variable e calculates probability of not-spam and f calculates the probability of spam. 
# 

##Testing The Naive Bayes Classifier

df=pd.read_csv("test-features.txt", sep=' ',
                  names = ["DocID", "DicID", "Occ"])  #reading the file and giving them respective headers
s=df["DocID"]
t=df["DicID"]
u=df["Occ"]
x=np.log(0.50)                  #0.50 is the probability of spam and non spam dataset in our training data.
y=np.log(0.50)                  #x is the prob of non spam and y is the prob of of spam
                                                   #Applying the naive bayes algorithm.We are adding the log instead of multipying due to underflow.
z=1
arr=[]
for i in range(len(s)):
    if (s[i]==z):
        e=(k[0][t[i]])*(u[i])
        f=(k[1][t[i]])*(u[i])
        x+=np.log(e)
        y+=np.log(f)
    else:
        z+=1
        if x>y:
            arr.append(0)
        else:
            arr.append(1)
        x=np.log(0.50)
        y=np.log(0.50)
        e=(k[0][t[i]])*(u[i])
        f=(k[1][t[i]])*(u[i])
        x+=np.log(e)
        y+=np.log(f)
if x>y:
    arr.append(0)
else:
    arr.append(1)
df=pd.read_csv("test-labels.txt",names = ["LabelId"])  #reading the file and giving them respective header.
accuracy=0
l=df["LabelId"]
for i in range(len(arr)):      #Comparing test label and prediction(arr)
    if (l[i]==arr[i]):
        accuracy+=1
accuracy=accuracy/len(arr)
print ("Accuracy of the Naive Bayes Algorithm is",accuracy*100.0)
submission = pd.DataFrame(arr)
submission.to_csv('prediction.txt',index = False)#Creates prediction into a new file. 





# # Spam Classifier using Multinomial Naive Bayes from scratch
# 

# In this notebook I am going to develop a simple email or sms spam classifier using Multinomial Naive Bayes Algorithm written from scratch. For that purpose we first need to know basic algorithm behind naive bayes. 
#  Here I am using SMS dataset form UCI ,that contains almost 5500 examples.
# 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df = pd.read_csv("spam.csv",encoding='latin-1')


df.head()


# I will map the Y_label which is 'v1' column to {0,1}.
# 

dict = {'ham':0,'spam':1}
df['v1'] = df['v1'].map(dict)
df.head()


# Deleting unecessary columns from the dataframe.
# 

del df['Unnamed: 2']
del df['Unnamed: 3']
del df['Unnamed: 4']


# Now first we need to find a way to represent the text data to a numerical form.To do this I will be using CountVectorizer that creates a dictionary of all the words present in that corpus(i.e. the whole text document).Then we can transform the text data into a matrix form whose (i,j)th elemment is nothong but the number of times the jth word has appeared in the ith document or example.
# 

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
c_vec = CountVectorizer(lowercase=1,min_df=.00001,stop_words='english')
c_vec.fit(df['v2'].values)


# Spliting the dataframe into train and test dataframe .
# 

train_df = df[0:5000]
test_df = df[5000:]
test_df.index=(range(test_df.shape[0]))
Y_train = train_df['v1'].values


# 
# First method we have to write for naive bayes is for calculating probability of spam and ham class.We just have to find how many spam sms and ham sms is there in data and then divide it by total number of examples.
# 

def prob_y(Y_train,num_class=2):
    p_y = np.zeros([num_class,])
    n_y = np.zeros([num_class,])
    d_y = Y_train.shape[0]
    for i in range(Y_train.shape[0]):
        n_y[Y_train[i]] = n_y[Y_train[i]]+1
    p_y = n_y/d_y
    return p_y


p_y = prob_y(Y_train)
p_y


# 
# The next method is prob_xy which is P(X|Y) . It is the probabilty of getting the word X in the class Y.This function produces a 
# num_class * num_of_words matrix. First column of the matrix contains P(X|Y=0) and second column P(X|Y=1).
# 

def prob_xy(c_vec,train_df,Y_train,num_class=2):
    d_y = np.zeros([num_class,])+len(c_vec.vocabulary_)
    p_xy = np.zeros([num_class,len(c_vec.vocabulary_)])
    for i in np.unique(Y_train):
        temp_df = train_df[train_df['v1']==i]
        temp_x = c_vec.transform(temp_df['v2'].values)
        n_xy = np.sum(temp_x,axis=0)+1
        d_y[i] = d_y[i]+np.sum(temp_x)
        p_xy[i] = n_xy/d_y[i] 
    return p_xy


p_xy = prob_xy(c_vec,train_df,Y_train,2)
p_xy


# Now we come to final stage of this algorithm where we have to find P(Y|X) i.e. the probability of a document X to belong to class Y . From Bayes theorem in probability theory , P(Y|X) = P(X|Y) * P(Y)/P(X) . 
# And then finally the class label Y for a document X will be accroding to  max(P(Y=0|X),P(Y=1|X)).
# 

def classify(c_vec,test_df,p_xy,p_y,num_class=2):
    pred = []
    pre_yx = []
    for doc in test_df['v2'].values:
        temp_doc = (c_vec.transform([doc])).todense()
        temp_prob = np.zeros([num_class,])
        for i in range(num_class):
            temp_prob[i] = np.prod(np.power(p_xy[i],temp_doc))*p_y[i]
        pred.append(np.argmax(temp_prob))
    return pred


pred = classify(c_vec,test_df,p_xy,p_y,num_class=2)


# Now that our classification is done , we will find the accuracy for both the training and test data.
# 

def accuracy(pred,Y):
    return np.sum(pred==Y)/Y.shape[0]


Y_test = test_df['v1'].values
test_accuracy = accuracy(pred,Y_test)
print('Test Data Accuaracy = '+str(test_accuracy)) 


pred_train = classify(c_vec,train_df,p_xy,p_y,num_class=2)
print('Train Data Accuracy = '+str(accuracy(pred_train,Y_train)))





