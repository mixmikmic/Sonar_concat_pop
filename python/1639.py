# <h1 align = 'center'> Neural Networks Demystified </h1>
# <h2 align = 'center'> Part 5: Numerical Gradient Checking </h2>
# 
# 
# <h4 align = 'center' > @stephencwelch </h4>
# 

from IPython.display import YouTubeVideo
YouTubeVideo('pHMzNW8Agq4')


# Last time, we did a bunch of calculus to find the rate of change of our cost, J, with respect to our parameters, W. Although each calculus step was pretty straight forward, it’s still easy to make mistakes. What’s worse, is that our network doesn’t have a good way to tell us that it’s broken – code with incorrectly implemented gradients may appear to be functioning just fine.
# 

# This is the most nefarious kind of error when building complex systems. Big, in-your-face errors suck initially, but it’s clear that you must fix this error for your work to succeed. More subtle errors can be more troublesome because they hide in your code and steal hours of your time, slowly degrading performance, while you wonder what the problem is. 
# 

# A good solution here is to test the gradient computation part of our code, just as developer would unit test new portions of their code. We’ll combine a simple understanding of the derivative with some mild cleverness to perform numerical gradient checking. If our code passes this test, we can be quite confident that we have computed and coded up our gradients correctly. 
# 

# To get started, let’s quickly review derivatives. Derivates tell us the slope, or how steep a function is. Once you’re familiar with calculus, it’s easy to take for granted the inner workings of the derivative - we just accept that the derivative of x^2 is 2x by the power rule. However, depending on how mean your calculus teacher was, you may have spent months not being taught the power rule, and instead required to compute derivatives using the definition. Taking derivatives this way is a bit tedious, but still important - it provides us a deeper understanding of what a derivative is, and it’s going to help us solve our current problem. 
# 

# The definition of the derivative is really a glorified slope formula. The numerator gives us the change in y values, while the denominator is convenient way to express the change in x values. By including the limit, we are applying the slope formula across an infinitely small region – it’s like zooming in on our function, until it becomes linear. 
# 
# 

# The definition tells us to zoom in until our x distance is infinitely small, but computers can’t really handle infinitely small numbers, especially when they’re in the bottom parts of fractions - if we try to plug in something too small, we will quickly lose precision. The good news here is that if we plug in something reasonable small, we can still get surprisingly good numerical estimates of the derivative. 
# 

# We’ll modify our approach slightly by picking a point in the middle of the interval we would like to test, and call the distance we move in each direction epsilon. 
# 

# Let’s test our method with a simple function, x squared. We’ll choose a reasonable small value for epsilon, and compute the slope of x^2 at a given point by finding the function value just above and just below our test point. We can then compare our result to our symbolic derivative 2x, at the test point. If the numbers match, we’re in business!
# 

get_ipython().magic('pylab inline')
#Import Code from previous videos:
from partFour import *


def f(x):
    return x**2


epsilon = 1e-4
x = 1.5


numericalGradient = (f(x+epsilon)- f(x-epsilon))/(2*epsilon)


numericalGradient, 2*x


# Add helper functions to our neural network class: 
# 

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
        


# We can use the same approach to numerically evaluate the gradient of our neural network. It’s a little more complicated this time, since we have 9 gradient values, and we’re interested in the gradient of our cost function. We’ll make things simpler by testing one gradient at a time. We’ll “perturb” each weight - adding epsilon to the current value and  computing the cost function, subtracting epsilon from the current value and computing the cost function, and then computing the slope between these two values. 
# 

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 


# We’ll repeat this process across all our weights, and when we’re done we’ll have a numerical gradient vector, with the same number of values as we have weights. It’s this vector we would like to compare to our official gradient calculation. We see that our vectors appear very similar, which is a good sign, but we need to quantify just how similar they are. 
# 

NN = Neural_Network()


numgrad = computeNumericalGradient(NN, X, y)
numgrad


grad = NN.computeGradients(X,y)
grad


# A nice way to do this is to divide the norm of the difference by the norm of the sum of the vectors we would like to compare. Typical results should be on the order of 10^-8 or less if you’ve computed your gradient correctly. 
# 

norm(grad-numgrad)/norm(grad+numgrad)


# And that’s it, we can now check our computations and eliminate gradient errors before they become a problem. Next time we’ll train our Neural Network. 
# 

# <h1 align = 'center'> Neural Networks Demystified </h1>
# <h2 align = 'center'> Part 6: Training </h2>
# 
# 
# <h4 align = 'center' > @stephencwelch </h4>
# 

from IPython.display import YouTubeVideo
YouTubeVideo('9KM9Td6RVgQ')


# So far we’ve built a neural network in python, computed a cost function to let us know how well our network is performing, computed the gradient of our cost function so we can train our network, and last time we numerically validated our gradient computations. After all that work, it’s finally time to train our neural network. 
# 

# Back in part 3, we decided to train our network using gradient descent. While gradient descent is conceptually pretty straight forward, its implementation can actually be quite complex- especially as we increase the size and number of layers in our neural network. If we just march downhill with consistent step sizes, we may get stuck in a local minimum or flat spot, we may move too slowly and never reach our minimum, or we may move to quickly and bounce out of our minimum. And remember, all this must happen in high-dimensional space, making things significantly more complex. Gradient descent is a wonderfully clever method, but provides no guarantees that we will converge to a good solution, that we will converge to a solution in a certain amount of time, or that we will converge to a solution at all.  
# 

# The good and bad news here is that this problem is not unique to Neural Networks - there’s an entire field dedicated to finding the best combination of inputs to minimize the output of an objective function: the field of Mathematical Optimization. The bad news is that optimization can be a bit overwhelming; there are many different techniques we could apply to our problem. 
# 

# Part of what makes the optimization challenging is the broad range of approaches covered - from very rigorous, theoretical methods to hands-on, more heuristics-driven methods. Yann Lecun’s 1998 publication Efficient BackProp presents an excellent review of various optimization techniques as applied to neural networks. 
# 

# Here, we’re going to use a more sophisticated variant on gradient descent, the popular Broyden-Fletcher-Goldfarb-Shanno numerical optimization algorithm. The BFGS algorithm overcomes some of the limitations of plain gradient descent by estimating the second derivative, or curvature, of the cost function surface, and using this information to make more informed movements downhill. BFGS will allow us to find solutions more often and more quickly. 
# 

# We’ll use the BFGS implementation built into the scipy optimize package, specifically within the minimize function. To use BFGS, the minimize function requires us to pass in an objective function that accepts a vector of parameters, input data, and output data, and returns both the cost and gradients. Our neural network implementation doesn’t quite follow these semantics, so we’ll use a wrapper function to give it this behavior. We’ll also pass in initial parameters, set the jacobian parameter to true since we’re computing the gradient within our neural network class, set the method to BFGS, pass in our input and output data, and some options. Finally, we’ll implement a callback function that allows us to track the cost function value as we train the network. Once the network is trained, we’ll replace the original, random parameters, with the trained parameters. 
# 

get_ipython().magic('pylab inline')
#Import code from previous videos:
from partFive import *


from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',                                  args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
    


# If we plot the cost against the number of iterations through training, we should see a nice, monotonically decreasing function. Further, we see that the number of function evaluations required to find the solution is less than 100, and far less than the 10^27 function evaluation that would have been required to find a solution by brute force, as shown in part 3. Finally, we can evaluate our gradient at our solution and see very small values – which make sense, as our minimum should be quite flat. 
# 

NN = Neural_Network()


T = trainer(NN)


T.train(X,y)


plot(T.J)
grid(1)
xlabel('Iterations')
ylabel('Cost')


NN.costFunctionPrime(X,y)


# The more exciting thing here is that we finally have a trained network that can predict your score on a test based on how many hours you sleep and how many hours you study the night before. If we run our training data through our forward method now, we see that our predictions are excellent. We can go one step further and explore the input space for various combinations of hours sleeping and hours studying, and maybe we can find an optimal combination of the two for your next test. 
# 

NN.forward(X)


y


#Test network for various combinations of sleep/study:
hoursSleep = linspace(0, 10, 100)
hoursStudy = linspace(0, 5, 100)

#Normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

#Create 2-d versions of input for plotting
a, b  = meshgrid(hoursSleepNorm, hoursStudyNorm)

#Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()


allOutputs = NN.forward(allInputs)


#Contour Plot:
yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

CS = contour(xx,yy,100*allOutputs.reshape(100, 100))
clabel(CS, inline=1, fontsize=10)
xlabel('Hours Sleep')
ylabel('Hours Study')


#3D plot:

##Uncomment to plot out-of-notebook (you'll be able to rotate)
#%matplotlib qt

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100),                        cmap=cm.jet)

ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')


# Our results look pretty reasonable, and we see that for our model, sleep actually has a bigger impact on your grade than studying – something I wish I had realized when I was in school. So we’re done, right? 
# 	Nope. 
# 

# We’ve made possibly the most dangerous and tempting error in machine learning – overfitting. Although our network is performing incredibly well (maybe too well) on our training data – that doesn’t mean that our model is a good fit for the real world, and that’s what we’ll work on next time. 
# 

# <h1 align = 'center'> Neural Networks Demystified </h1>
# <h2 align = 'center'> Part 1: Data and Architecture </h2>
# 
# 
# <h4 align = 'center' > @stephencwelch </h4>
# 

from IPython.display import YouTubeVideo
YouTubeVideo('bxe2T-V8XRs')


# <h3 align = 'center'> Variables </h3>
# 
# |Code Symbol | Math Symbol | Definition | Dimensions
# | :-: | :-: | :-: | :-: |
# |X|$$X$$|Input Data, each row is an example| (numExamples, inputLayerSize)|
# |y |$$y$$|target data|(numExamples, outputLayerSize)|
# 

# Let’s say you want to predict some output value y, given some input value X. For example, maybe you want to predict your score on a test based on how many hours you sleep and how many hours you study the night before. To use a machine learning approach, we first need some data. Let’s say for the last three tests, you recorded your number of hours of studying, your number of hours sleeping, and your score on the test. We'll use the programming language python to store our data in 2-dimensional numpy arrays.
# 

get_ipython().magic('pylab inline')


# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)


X


y


# Now that we have some data, we’re going to use it to train a model to predict how you will do on your next test, based on how many hours you sleep and how many hours you study. This is called a supervised regression problem. It’s supervised because our examples have inputs and outputs. It’s a regression problem because we’re predicting your test score, which is a continuous output. If we we’re predicting your letter grade, this would be called a classification problem, and not a regression problem.
# 

# There are an overwhelming number of models within machine learning, here we’re going to use a particularly interesting one called an artificial neural network. These guys are loosely based on how the neurons in your brain work, and have been particularly successful recently at solving really big, really hard problems.
# 

# Before we throw our data into the model, we need to account for the differences in the units of our data. Both of our inputs are in hours, but our output is a test score, scaled between 0 and 100. Neural networks are smart, but not smart enough to guess the units of our data. It’s kind of like asking our model to compare apples to oranges, where most learning models really only want to compare apples to apples. The solution is to scale our data, this way our model only sees standardized units. Here, we're going to take advantage of the fact that all of our data is positive, and simply divide by the maximum value for each variable, effectively scaling the result between 0 and 1. 
# 

X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100


X


y


# Now we can build our Neural Net. We know our network must have 2 inputs and 1 output, because these are the dimensions of our data. We’ll call our output y hat, because it’s an estimate of y, but not the same as y. Any layer between our input and output layer is called a hidden layer. Recently, researchers have built networks with many many hidden layers. These are known as a deep belief networks, giving rise to the term deep learning. Here, we’re going to use 1 hidden layer with 3 hidden units, but if we wanted to build a deep neural network, we would just stack a bunch of layers together.
# 

from IPython.display import Image
i = Image(filename='images/simpleNetwork.png')
i


# In neural net visuals, circles represent neurons and lines represent synapses. Synapses have a really simple job, they take a value from their input, multiply it by a specific weight, and output the result. Neurons are a little more complicated. Their job is to add together the outputs of all their synapses, and apply an activation function. Certain activation functions allow neural nets to model complex non-linear patterns, that simpler models may miss. For our neural net, we’ll use sigmoid activation functions. Next, we'll build out our neural net in python.
# 

