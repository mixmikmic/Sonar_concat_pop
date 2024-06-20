# # GPyOpt: The tool for Bayesian Optimization 
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# ## Reference Manual index
# 
# *Last updated Friday, 11 March 2016.*
# 
# =====================================================================================================
# 
# 1. **What is GPyOpt?**
# 
# 2. **Installation and setup**
# 
# 3. **First steps with GPyOpt and Bayesian Optimization**
# 
# 4. **Alternative GPyOpt interfaces:  Modular Bayesian optimization and Spearmint**
# 
# 5. **What can I do with GPyOpt?**
#     1. Bayesian optmization with arbitrary restrictions.
#     2. Parallel Bayesian optimization.
#     3. Mixing different types of variables.
#     4. Armed bandits problems.
#     5. Tuning Scikit-learn models.
#     5. Integrating the model hyperparameters.
#     6. Currently supported models and acquisitions.
#     7. Using various cost evaluations functions.
# 
# =====================================================================================================
# 

# ## 1. What is GPyOpt?
# 
# [GPyOpt](http://sheffieldml.github.io/GPy/) is a tool for optimization (minimization) of black-box functions using Gaussian processes. It has been implemented in [Python](https://www.python.org/download/releases/2.7/) by the [group of Machine Learning](http://ml.dcs.shef.ac.uk/sitran/) (at SITraN) of the University of Sheffield. 
# 
# GPyOpt is based on [GPy](https://github.com/SheffieldML/GPy), a library for Gaussian process modeling in Python. [Here](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb) you can also find some notebooks about GPy functionalities. GPyOpt is a tool for Bayesian Optimization but we also use it for academic dissemination in [Gaussian Processes Summer Schools](gpss.cc), where you can find some extra labs and a variety of talks on Gaussian processes and Bayesian optimization.
# 
# The purpose of this manual is to provide a guide to use GPyOpt. The framework is [BSD-3 licensed](https://opensource.org/licenses/BSD-3-Clause) and we welcome collaborators to develop new functionalities. If you have any question or suggestions about the notebooks, please contact (Javier Gonzalez) j.h.gonzalez@sheffield.ac.uk. For general questions about the package please write an issue in the [GitHub repository](https://github.com/SheffieldML/GPyOpt).
# 
# <img src="./files/figures/gpyopt.jpeg" alt="Drawing" style="width: 500px;"/>
# 

# ## 2. Installation and setup
# 

# The simplest way to install GPyOpt is using pip. Ubuntu users can do:
# 
# ```
# sudo apt-get install python-pip
# pip install gpyopt
# ```
# 
# If you'd like to install from source, or want to contribute to the project (e.g. by sending pull requests via github), read on. Clone the repository in GitHub and add it to your $PYTHONPATH.
# 
# 
# ```
# git clone git@github.com:SheffieldML/GPyOpt.git ~/SheffieldML
# echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.bashrc
# ```
# 
# There are a number of dependencies that you may need to install. Three of them are needed to ensure the good behaviour of the package. These are, GPy, numpy and scipy. Other dependencies, such as DIRECT, cma and pyDOE are optional and only are required for in some options of the module. All of them are pip installable.
# 
# 

# ## 3. First steps with GPyOpt and Bayesian Optimization
# 
# The tutorial [Introduction to Bayesian Optimization with GPyOpt](./GPyOpt_reference_manual.ipynb) reviews some basic concepts on Bayesian optimization and shows some basic GPyOpt functionalities. It is a manual for beginners who want to start using the package.
# 
# <img src="./files/figures/iteration001.png" alt="Drawing" style="width: 500px;"/>
# 

# ## 4. Alternative GPyOpt interfaces:  Modular Bayesian optimization and Spearmint
# 
# GPyOpt has different interfaces oriented to different types of users. Apart from the general interface (detailed in the introductory manual) you can use GPyOpt in a modular way: you can implement and use your some elements of the optimization process, such us a new model or acquisition function, but still use the main backbone of the package. You can check the [GPyOpt: Modular Bayesian Optimization](./GPyOpt_modular_bayesian_optimization.ipynb) notebook if you are interested on using GPyOpt this way. 
# 
# Also, we have developed and GPyOpt interface with Spearmint. This means that if you use Spearmint you can re-run your experiments with GPyOpt by changing only one line of code (TODO).
# 

# ## 5. What can I do with GPyOpt?
# 
# There are several options implemented in GPyOpt that allows to cover a wide range of specific optimization problems. We have implemented a collection of notebooks to explain these functionalities separately but they can be easily combined. 
# 

# ### 5.1. Bayesian optmization with arbitrary restrictions
# 

# With GPyOpt you can solve optimization problems with arbitrary non trivial restrictions. Have a look to the notebook [GPyOpt: Bayesian Optimization with fixed constrains](./GPyOpt_constrained_optimization.ipynb) if you want to know more about how to use GPyOpt in these type of problems.
# 

# ### 5.2 Parallel Bayesian optimization
# The main bottleneck when using Bayesian optimization is the cost of evaluating the objective function. In the notebook [GPyOpt: parallel Bayesian Optimization](GPyOpt_parallel_optimization.ipynb) you can learn more about the different parallel methods currently implemented in GPyOPt.  
# 

# ### 5.3 Mixing different types of variables
# In GPyOpt you can easily combine different types of variables in the optimization. Currently you can use discrete an continuous variables. The way GPyOpt handles discrete variables is by marginally optimizing the acquisition functions over combinations of feasible values. This may slow down the optimization if many discrete variables are used but it avoids rounding errors. See the notebook entitled [GPyOpt: mixing different types of variables](./GPyOpt_mixed_domain.ipynb) for further details. 
# 

# ### 5.4 Armed bandits problems
# 
# Armed bandits optimization problems are a particular case of Bayesian Optimization that appear when the domain of the function to optimize is entirely discrete. This has several advantages with respect to optimize in continuous domains. The most remarkable is that the optimization of the acquisition function can be done by taking the $arg min$ of all candidate points while the rest of the BO theory applies. In the notebook [GPyOpt: armed bandits optimization](./GPyOpt_bandits_optimization.ipynb) you can check how to use GPyOpt in these types of problems.       
# 

# ### 5.5 Tuning scikit-learn models
# 
# [Scikit-learn](http://scikit-learn.org/stable/) is a very popular library with a large variety of useful methods in Machine Learning. Have a look to the notebook [GPyOpt: configuring Scikit-learn methods](GPyOpt_scikitlearn.ipynb) to learn how learn the parameters of Scikit-learn methods using GPyOpt. You will learn how to automatically tune the parameters of a Support Vector Regression.
# 

# ### 5.6 Integrating the model hyper parameters
# Maximum Likelihood estimation can be a very instable choice to tune the surrogate model hyper parameters, especially in the fist steps of the optimization. When using a GP model as a surrogate of your function you can integrate the most common acquisition functions with respect to the parameters of the model. Check the notebook [GPyOpt: integrating model hyperparameters](./GPyOpt_integrating_model_hyperparameters.ipynb) to check how to use this option.
# 

# ### 5.7 Currently supported surrogate models and acquistions
# 
# GPyOpt is constantly under development. In the notebook [GPyOpt: available surrogate models and acquisitions](./GPyOpt_models.ipynb) you can check the currently supported models and acquisitions.
# 

# ### 5.8 Using various cost evaluation functions
# The cost of evaluating the objective can be a crucial factor in the optimization process. Check the notebook [GPyOpt: dealing with cost functions](./GPyOpt_cost_functions.ipynb) to learn how to use arbitrary cost functions, including the objective evaluation time.
# 




# # GPyOpt: interface with Spearmint
# 
# ### Written by Javier Gonzalez and Zhenwen Dai, University of Sheffield.
# 
# 
# *Last updated Monday, 14 March 2016.*
# 

# To be announced.
# 

# # GPyOpt: configuring a GPy model :)
# 
# ### Written by Javier Gonzalez, Zhenwen Dai and Max Zwiessele, University of Sheffield.
# 
# *Last updated Tuesday, 3 Jun 2016.*
# 

# In this notebook we are going to create a [black hole](https://en.wikipedia.org/wiki/Black_hole) of Gaussian process that will collapse the entire machine learinig universe: We will configure a GPy model using GPyOpt by using a GPy model as surrogate model for the likelihood of original GPy model.
# 

get_ipython().magic('pylab inline')
import GPy
import GPyOpt
import numpy as np
from sklearn import svm
from GPy.models import GPRegression
from numpy.random import seed
np.random.seed(12345)


# As we did for the scikit-learn SVR example we use the Olympic marathon dataset available in GPy and we split the original dataset into the training data (first 20 data points) and testing data (last 7 data points). 
# 

# Let's load the dataset
GPy.util.datasets.authorize_download = lambda x: True
data = GPy.util.datasets.olympic_marathon_men()
X = data['X']
Y = data['Y']
X_train = X[:20]
Y_train = Y[:20]
X_test = X[20:]
Y_test = Y[20:]


# Let's first create a GPy model. We add some arbitraty numbers to the parameters of the kernel.
# 

k =  GPy.kern.Matern32(1, variance=2, lengthscale=1)   + GPy.kern.Linear(1, variances=1)   + GPy.kern.Bias(1, variance=5)

m = GPRegression(X_train, Y_train, kernel=k,
                 normalizer=True)
print m


# Now we plot hor the model looks for the training and testing data.
# 

Y_train_pred, Y_train_pred_var = m.predict(X_train)
Y_test_pred, Y_test_pred_var = m.predict(X_test)

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')

plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')


# Not very good, as we expected. Now let's optimise the model parameters. We will do that using the default optimiser and by using Bayesian optmization. We start with this latest option. We first have a look to the parameters we need to tune and we create the domain where the optimisation is going to be carried out.
# 

# Model parameters
m.parameter_names()


# List containing the description of the domain where we will perform the optmisation
domain = [
{'name': 'Mat32.variance',          'type': 'continuous', 'domain': (1,4.)},
{'name': 'Mat32.lengthscale',       'type': 'continuous', 'domain': (50.,150.)},
{'name': 'Linear.variances',        'type': 'continuous', 'domain': (1e-5,6)},
{'name': 'Bias.variance',           'type': 'continuous', 'domain': (1e-5,6)},
{'name': 'Gaussian_noise.variance', 'type': 'continuous', 'domain': (1e-5,4.)}
]


# We will minimize the minus marinal log-likelihood. We wrapp it to create our objetive function.
# 

def f_lik(x):
    m[:] = x
    return m.objective_function()


# Now create the GPyOpt object and run the optimization procedure. We will use the expected improvement integrated over the GP hyperparameters (so the black hole created is not so dangerous). 
# 

opt = GPyOpt.methods.BayesianOptimization(f = f_lik,                  
                                          domain = domain,
                                          normalize_Y= True,
                                          exact_feval = True,
                                          model_type= 'GP',
                                          acquisition_type ='EI',       
                                          acquisition_jitter = 0.25)   


# And we run the optimization for 50 iterations.
# 

# it may take a few seconds
opt.run_optimization(max_iter=100)
opt.plot_convergence()


# Let's show the best parameters found. They differ significantly from the default parameters.
# 

x_best = opt.X[np.argmin(opt.Y)].copy()
m[:] = x_best
print("The best model optimized with GPyOpt:")
print m


# And we print the model for the training and testing datasets.
# 

Y_train_pred, Y_train_pred_var = m.predict(X_train)
Y_test_pred, Y_test_pred_var = m.predict(X_test)

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')

plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')


# Now we optimize the model using the default GPy optimizer. To do this we first restrict the search of the parameters the same way we did for the Bayesian optimisation case.
# 

m[:] = opt.X[0].copy()

m.kern.Mat32.variance.constrain_bounded(1,4.)
m.kern.Mat32.lengthscale.constrain_bounded(50,150.)
m.kern.linear.variances.constrain_bounded(1e-5,6)
m.kern.bias.variance.constrain_bounded(1e-5,6)
m.Gaussian_noise.variance.constrain_bounded(1e-5,4.)


m.kern


m.optimize()


Y_train_pred, Y_train_pred_var = m.predict(X_train)
Y_test_pred, Y_test_pred_var = m.predict(X_test)

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')

plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')


print("The best model optimized with the default optimizer:")
print m


# # Introduction to Bayesian Optimization with GPyOpt 
# 
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# *Last updated Friday, 11 March 2016.*
# 
# =====================================================================================================
# 1. **How to use GPyOpt?**
# 
# 2. **The Basics of Bayesian Optimization**
#     1. Gaussian Processes
#     2. Acquisition functions
#     3. Applications of Bayesian Optimization 
# 
# 3. **1D optimization example**
# 
# 4. **2D optimization example**
# 
# =====================================================================================================
# 

# ## 1. How to use GPyOpt?
# 

# We start by loading GPyOpt and GPy.
# 

get_ipython().magic('pylab inline')
import GPy
import GPyOpt
from numpy.random import seed


# GPyOpt is easy to use as a black-box functions optimizer. To start you only need: 
# 
# * Your favorite function $f$ to minimize. We use $f(x)=2x^2$ in this toy example, whose global minimum is at x=0.
# 

def myf(x):
    return (2*x)**2


# * A set of box constrains, the interval [-1,1] in our case. You can define a list of dictionaries where each element defines the name, type and domain of the variables. 
# 

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)}]


# * A budget, or number of allowed evaluations of $f$.
# 

max_iter = 15


# With this three pieces of information GPyOpt has enough to find the minimum of $f$ in the selected region. GPyOpt solves the problem in two steps. First, you need to create a GPyOpt object that stores the problem (f and and box-constrains). You can do it as follows.
# 

myProblem = GPyOpt.methods.BayesianOptimization(myf,bounds)


# Next you need to run the optimization for the given budget of iterations. This bit it is a bit slow because many default options are used. In the next notebooks of this manual you can learn how to change other parameters to optimize the optimization performance.
# 

myProblem.run_optimization(max_iter)


# Now you can check the best found location $x^*$ by
# 

myProblem.x_opt


# and the predicted value value of $f$ at $x^*$ optimum by
# 

myProblem.fx_opt


# And that's it! Keep reading to learn how GPyOpt uses Bayesian Optimization to solve this an other optimization problem. You will also learn all the features and options that you can use to solve your problems efficiently. 
# 
# =====================================================================================================
# 

# ## 2. The Basics of Bayesian Optimization
# 
# Bayesian optimization (BO) is an strategy for global optimization of black-box functions [(Snoek et al., 2012)](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf). Let $f: {\mathcal X} \to R$ be a L-Lipschitz  continuous function defined on a compact subset ${\mathcal X} \subseteq R^d$. We are interested in solving the global optimization problem of finding
# $$ x_{M} = \arg \min_{x \in {\mathcal X}} f(x). $$
# 
# We assume that $f$ is a *black-box* from which only perturbed evaluations of the type $y_i = f(x_i) + \epsilon_i$, with $\epsilon_i \sim\mathcal{N}(0,\psi^2)$, are  available. The goal is to make a series of $x_1,\dots,x_N$ evaluations of $f$ such that the *cumulative regret* 
# $$r_N= Nf(x_{M})- \sum_{n=1}^N f(x_n),$$ 
# is minimized. Essentially, $r_N$ is minimized if we start evaluating $f$ at $x_{M}$ as soon as possible. 
# 
# There are two crucial bits in any Bayesian Optimization (BO) procedure approach.
# 
# 1. Define a **prior probability measure** on $f$: this function will capture the our prior beliefs on $f$. The prior will be updated to a 'posterior' using the available data.
# 
# 2. Define an **acquisition function** $acqu(x)$: this is a criteria to decide where to sample next in order to gain the maximum information about the location of the global maximum of $f$.
# 
# Every time a new data point is collected. The model is re-estimated and the acquisition function optimized again until convergence. Given a prior over the function $f$ and an acquisition function, a BO procedure will converge to the optimum of $f$ under some conditions [(Bull, 2011)](http://arxiv.org/pdf/1101.3501.pdf).
# 

# ### 2.1 Prior probability meassure on $f$: Gaussian processes
# 

# A Gaussian process (GP) is a probability distribution across classes functions, typically smooth, such that each linear finite-dimensional restriction is multivariate Gaussian [(Rasmussen and Williams, 2006)](http://www.gaussianprocess.org/gpml). GPs are fully parametrized by a mean $\mu(x)$ and a covariance function $k(x,x')$.  Without loss of generality $\mu(x)$ is assumed to be zero. The covariance function $k(x,x')$ characterizes the smoothness and other properties of $f$. It is known as the
# kernel of the process and has to be continuous, symmetric and positive definite. A widely used kernel is the square exponential, given by
# 
# $$ k(x,x') = l \cdot \exp{ \left(-\frac{\|x-x'\|^2}{2\sigma^2}\right)} $$
# where $\sigma^2$ and and $l$ are positive parameters. 
# 
# To denote that $f$ is a sample from a GP with mean $\mu$ and covariance $k$ we write 
# 
# $$f(x) \sim \mathcal{GP}(\mu(x),k(x,x')).$$ 
# 
# For regression tasks, the most important feature of GPs is that process priors are conjugate to the likelihood from finitely many observations $y= (y_1,\dots,y_n)^T$ and $X =\{x_1,...,x_n\}$, $x_i\in \mathcal{X}$ of the form $y_i = f(x_i) + \epsilon_i $
# where $\epsilon_i \sim \mathcal{N} (0,\sigma^2)$. We obtain the Gaussian posterior posterior $f(x^*)|X, y, \theta \sim \mathcal{N}(\mu(x^*),\sigma^2(x^*))$, where $\mu(x^*)$ and $\sigma^2(x^*)$ have close form. See [(Rasmussen and Williams, 2006)](http://www.gaussianprocess.org/gpml) for details.
# 

# ### 2.2 Acquisition Function
# 
# Acquisition functions are designed represents our beliefs over the maximum of $f(x)$. Denote by $\theta$ the parameters of the GP model and by $\{x_i,y_i\}$ the available sample. Three of the most common acquisition functions, all available in GPyOpt are:
# 
# * **Maximum probability of improvement (MPI)**:
# 
# $$acqu_{MPI}(x;\{x_n,y_n\},\theta) = \Phi(\gamma(x)), \mbox{where}\   \gamma(x)=\frac{\mu(x;\{x_n,y_n\},\theta)-f(x_{best})-\psi}{\sigma(x;\{x_n,y_n\},\theta)}.$$
# 
# 
# * **Expected improvement (EI)**:
# 
# $$acqu_{EI}(x;\{x_n,y_n\},\theta) = \sigma(x;\{x_n,y_n\},\theta) (\gamma(x) \Phi(\gamma(x))) + N(\gamma(x);0,1).$$
# 
# * **Upper confidence bound (UCB)**:
# 
# $$acqu_{UCB}(x;\{x_n,y_n\},\theta) = -\mu(x;\{x_n,y_n\},\theta)+\psi\sigma(x;\{x_n,y_n\},\theta).$$
# 
# $\psi$ is a tunable parameters that help to make the acquisition functions more flexible. Also, in the case of the UBC, the parameter $\eta$ is useful to define the balance between the importance we give to the mean and the variance of the model. This is know as the **exploration/exploitation trade off**.
# 

# ### 2.3 Applications of Bayesian Optimization
# 
# Bayesian Optimization has been applied to solve a wide range of problems. Among many other, some nice applications of Bayesian Optimization include: 
# 
# 
# * Sensor networks (http://www.robots.ox.ac.uk/~parg/pubs/ipsn673-garnett.pdf),
# 
# * Automatic algorithm configuration (http://www.cs.ubc.ca/labs/beta/Projects/SMAC/papers/11-LION5-SMAC.pdf), 
# 
# * Deep learning (http://www.mlss2014.com/files/defreitas_slides1.pdf), 
# 
# * Gene design (http://bayesopt.github.io/papers/paper5.pdf),
# 
# * and a long etc!
# 
# In this Youtube video you can see Bayesian Optimization working in a real time in a robotics example. [(Calandra1 et al. 2008)](http://www.ias.tu-darmstadt.de/uploads/Site/EditPublication/Calandra_LION8.pdf) 
# 

from IPython.display import YouTubeVideo
YouTubeVideo('ualnbKfkc3Q')


# ## 3. One dimensional example
# 
# In this example we show how GPyOpt works in a one-dimensional example a bit more difficult that the one we analyzed in Section 3. Let's consider here the Forrester function 
# 
# $$f(x) =(6x-2)^2 \sin(12x-4)$$ defined on the interval $[0, 1]$. 
# 
# The minimum of this function is located at $x_{min}=0.78$. The Forrester function is part of the benchmark of functions of GPyOpt. To create the true function, the perturbed version and boundaries of the problem you need to run the following cell. 
# 

get_ipython().magic('pylab inline')
import GPy
import GPyOpt

# Create the true and perturbed Forrester function and the boundaries of the problem
f_true= GPyOpt.objective_examples.experiments1d.forrester()          # noisy version
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]  # problem constrains 


# We plot the true Forrester function.
# 

f_true.plot()


# As we did in Section 3, we need to create the GPyOpt object that will run the optimization. We specify the function, the boundaries and we add the type of acquisition function to use. 
# 

# Creates GPyOpt object with the model and anquisition fucntion
seed(123)
myBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f,            # function to optimize       
                                             domain=bounds,        # box-constrains of the problem
                                             acquisition_type='EI',
                                             exact_feval = True) # Selects the Expected improvement


# Now we want to run the optimization. Apart from the number of iterations you can select 
# how do you want to optimize the acquisition function. You can run a number of local optimizers (acqu_optimize_restart) at random or in grid (acqu_optimize_method).     
# 

# Run the optimization
max_iter = 15     # evaluation budget
max_time = 60     # time budget 
eps      = 10e-6  # Minimum allows distance between the las two observations

myBopt.run_optimization(max_iter, max_time, eps)                     


# When the optimization is done you should receive a message describing if the method converged or if the maximum number of iterations was reached. In one dimensional examples, you can see the result of the optimization as follows.
# 

myBopt.plot_acquisition()


myBopt.plot_convergence()


# In problems of any dimension two evaluations plots are available.
# 
# * The distance between the last two observations.
# 
# * The value of $f$ at the best location previous to each iteration.
# 
# To see these plots just run the following cell.
# 

myBopt.plot_convergence()


# Now let's make a video to track what the algorithm is doing in each iteration. Let's use the LCB in this case with parameter equal to 2.
# 

# starts the optimization, 
import numpy as np
X_initial = np.array([[0.2],[0.4],[0.6]])

iterBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f,                 
                                             domain=bounds,        
                                             acquisition_type='EI',
                                             X = X_initial,
                                             exact_feval = True,
                                             normalize_Y = False,
                                             acquisition_jitter = 0.01)

iterBopt.model.model.kern.variance.constrain_fixed(2.5)

iterBopt.plot_acquisition('./figures/iteration%.03i.png' % (0))

from IPython.display import clear_output
N_iter = 15

for i in range(N_iter):
    clear_output()
    iterBopt.run_optimization(max_iter=1) 
    iterBopt.plot_acquisition('./figures/iteration%.03i.png' % (i + 1))


# ## 4. Two dimensional example
# 
# Next, we try a 2-dimensional example. In this case we minimize the use the Six-hump camel function 
# 
# $$f(x_1,x_2) = \left(4-2.1x_1^2 = \frac{x_1^4}{3} \right)x_1^2 + x_1x_2 + (-4 +4x_2^2)x_2^2,$$
# 
# in $[-3,3]\times [-2,2]$. This functions has two global minimum, at $(0.0898,-0.7126)$ and $(-0.0898,0.7126)$. As in the previous case we create the function, which is already in GPyOpt. In this case we generate observations of the function perturbed with white noise of $sd=0.1$.
# 

# create the object function
f_true = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
f_sim = GPyOpt.objective_examples.experiments2d.sixhumpcamel(sd = 0.1)
bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
         {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]
f_true.plot()


# We create the GPyOpt object. In this case we use the Lower Confidence bound acquisition function to solve the problem.
# 

# Creates three identical objects that we will later use to compare the optimization strategies 
myBopt2D = GPyOpt.methods.BayesianOptimization(f_sim.f,
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type='LCB',  
                                              normalize_Y = True,
                                              acquisition_weight = 2)    


# We run the optimization for 40 iterations and show the evaluation plot and the acquisition function.
# 

# runs the optimization for the three methods
max_iter = 40  # maximum time 40 iterations
max_time = 60  # maximum time 60 seconds

myBopt2D.run_optimization(max_iter,max_time,verbosity=False)            


# Finally, we plot the acquisition function and the convergence plot.
# 

myBopt2D.plot_acquisition() 


myBopt2D.plot_convergence()


# # GPyOpt: available surrogate models and acquisitions
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# ## Reference Manual index
# 
# *Last updated Friday, 11 March 2016.*
# 

# ### 1. Supported models
# 
# The surrogate models supported in GPyOpt are:
# 
# * **Standard Gaussian Processes** with standard MLE over the model hyperparameters: select ``model_type = GP`` in the GPyOpt wrapper.
# 
# 
# * **Standard Gaussian Processes with MCMC** sampling over the model hyperparameters: select ``model_type = GP_MCMC`` in the GPyOpt wrapper.
# 
# 
# * **Sparse Gaussian processes**: select ``model_type = sparseGP`` in the GPyOpt wrapper. 
# 
# 
# * **Random Forrest**: select ``model_type = RF``. To illustrate GPyOpt modularity, we have also wrapped the random forrest method implemetented in Scikit-learn.
# 

# ### 2. Supported acquisiitions
# 
# The supported acquisition functions in GPyOpt are:
# 
# * **Expected Improvement**: select ``acquisition_type = EI`` in the GPyOpt wrapper.
# 
# 
# * **Expected Improvement integrated over the model hyperparameters**: select ``acquisition_type = EI_MCMC`` in the GPyOpt wrapper. Only works if ``model_type`` is set to ``GP_MCMC``.
# 
# 
# * **Maximum Probability of Improvement**: select ``acquisition_type = MPI`` in the GPyOpt wrapper.
# 
# 
# * **Maximum Probability of Improvement integrated over the model hyperparameters**: select ``acquisition_type = MPI_MCMC`` in the GPyOpt wrapper. Only works if ``model_type`` is set to ``GP_MCMC``.
# 
# 
# * **GP-Lower confidence bound**: select ``acquisition_type = LCB`` in the GPyOpt wrapper.
# 
# 
# * **GP-Lower confidence bound integrated over the model hyperparameters**: select ``acquisition_type = LCB_MCMC`` in the GPyOpt wrapper. Only works if ``model_type`` is set to ``GP_MCMC``.
# 
# 
# 

# # Creating new surrogate models for GPyOpt
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# ## Reference Manual index
# 
# *Last updated Friday, 11 Jun 2016.*
# 

# You can create and use your own surrogate models functions in GPyOpt. To do it just complete the following template.
# 

# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.models.base import BOModel
import numpy as np

class NewModel(BOModel):
   
    """
    General template to create a new GPyOPt surrogate model

    :param normalize Y: wheter the outputs are normalized (default, false)

    """

    # SET THIS LINE TO True of False DEPENDING IN THE ANALYTICAL GRADIENTS OF THE PREDICTIONS ARE AVAILABLE OR NOT
    analytical_gradient_prediction = False

    def __init__(self, normalize_Y=True, **kwargs ):

        # ---
        # ADD TO self... THE REST OF THE PARAMETERS OF YOUR MODEL
        # ---
        
        self.normalize_Y = normalize_Y
        self.model = None

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        self.X = X
        self.Y = Y
        
        # ---
        # ADD TO self.model THE MODEL CREATED USING X AND Y.
        # ---
        
        
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        self.X = X_all
        self.Y = Y_all
        
        if self.normalize_Y:
            Y_all = (Y_all - Y_all.mean())/(Y_all.std())
        
        if self.model is None: 
            self._create_model(X_all, Y_all)
        else: 
            # ---
            # AUGMENT THE MODEL HERE AND REUPDATE THE HIPER-PARAMETERS
            # ---
            pass
                 
    def predict(self, X):
        """
        Preditions with the model. Returns posterior means m and standard deviations s at X. 
        """

        # ---
        # IMPLEMENT THE MODEL PREDICTIONS HERE (outputs are numpy arrays with a point per row)
        # ---
        
        return m, s
    
    def get_fmin(self):
        return self.model.predict(self.X).min()


# # GPyOpt: configuring Scikit-learn methods
# 
# ### Written by Javier Gonzalez and Zhenwen Dai, University of Sheffield.
# 
# *Last updated Tuesday, 15 March 2016.*
# 

# The goal of this notebook is to use GPyOpt to tune the parameters of Machine Learning algorithms. In particular, we will shows how to tune the hyper-parameters for the [Support Vector Regression (SVR)](http://www.svms.org/regression/SmSc98.pdf) implemented in [Scikit-learn](http://scikit-learn.org/stable/). Given the standard interface of Scikit-learn, other models can be tuned in a similar fashion. 
# 

# We start loading the requires modules.
# 

get_ipython().magic('pylab inline')
import GPy
import GPyOpt
import numpy as np
from sklearn import svm
from numpy.random import seed
seed(12345)


# For this example we will use the Olympic marathon dataset available in GPy. 
# We split the original dataset into the training data (first 20 data points) and testing data (last 7 data points). The performance of SVR is evaluated in terms of Rooted Mean Squared Error (RMSE) on the testing data.
# 

# Let's load the dataset
GPy.util.datasets.authorize_download = lambda x: True # prevents requesting authorization for download.
data = GPy.util.datasets.olympic_marathon_men()
X = data['X']
Y = data['Y']
X_train = X[:20]
Y_train = Y[:20,0]
X_test = X[20:]
Y_test = Y[20:,0]


# Let's first see the results with the default kernel parameters.
# 

from sklearn import svm
svr = svm.SVR()
svr.fit(X_train,Y_train)
Y_train_pred = svr.predict(X_train)
Y_test_pred = svr.predict(X_test)
print("The default parameters obtained: C="+str(svr.C)+", epilson="+str(svr.epsilon)+", gamma="+str(svr.gamma))


# We compute the RMSE on the testing data and plot the prediction. With the default parameters, SVR does not give an OK fit to the training data but completely miss out the testing data well.
# 

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')
plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')
print("RMSE = "+str(np.sqrt(np.square(Y_test_pred-Y_test).mean())))


# Now let's try Bayesian Optimization. We first write a wrap function for fitting with SVR. The objective is the RMSE from cross-validation. We optimize the parameters in *log* space.
# 

nfold = 3
def fit_svr_val(x):
    x = np.atleast_2d(np.exp(x))
    fs = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        fs[i] = 0
        for n in range(nfold):
            idx = np.array(range(X_train.shape[0]))
            idx_valid = np.logical_and(idx>=X_train.shape[0]/nfold*n, idx<X_train.shape[0]/nfold*(n+1))
            idx_train = np.logical_not(idx_valid)
            svr = svm.SVR(C=x[i,0], epsilon=x[i,1],gamma=x[i,2])
            svr.fit(X_train[idx_train],Y_train[idx_train])
            fs[i] += np.sqrt(np.square(svr.predict(X_train[idx_valid])-Y_train[idx_valid]).mean())
        fs[i] *= 1./nfold
    return fs

## -- Note that similar wrapper functions can be used to tune other Scikit-learn methods


# We set the search interval of $C$ to be roughly $[0,1000]$ and the search interval of $\epsilon$ and $\gamma$ to be roughtly $[1\times 10^{-5},0.1]$.
# 

domain       =[{'name': 'C',      'type': 'continuous', 'domain': (0.,7.)},
               {'name': 'epsilon','type': 'continuous', 'domain': (-12.,-2.)},
               {'name': 'gamma',  'type': 'continuous', 'domain': (-12.,-2.)}]


# We, then, create the GPyOpt object and run the optimization procedure. It might take a while.
# 

opt = GPyOpt.methods.BayesianOptimization(f = fit_svr_val,            # function to optimize       
                                          domain = domain,         # box-constrains of the problem
                                          acquisition_type ='LCB',       # LCB acquisition
                                          acquisition_weight = 0.1)   # Exploration exploitation


# it may take a few seconds
opt.run_optimization(max_iter=50)
opt.plot_convergence()


# Let's show the best parameters found. They differ significantly from the default parameters.
# 

x_best = np.exp(opt.X[np.argmin(opt.Y)])
print("The best parameters obtained: C="+str(x_best[0])+", epilson="+str(x_best[1])+", gamma="+str(x_best[2]))
svr = svm.SVR(C=x_best[0], epsilon=x_best[1],gamma=x_best[2])
svr.fit(X_train,Y_train)
Y_train_pred = svr.predict(X_train)
Y_test_pred = svr.predict(X_test)


# We can see SVR does a reasonable fit to the data. The result could be further improved by increasing the *max_iter*. 
# 

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')
plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')
print("RMSE = "+str(np.sqrt(np.square(Y_test_pred-Y_test).mean())))


# # GPyOpt: dealing with cost fuctions
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# ## Reference Manual index
# 
# *Last updated Friday, 11 March 2016.*
# 

# GPyOpt allows to consider function evaluation costs in the optimization.
# 

get_ipython().magic('pylab inline')
import GPyOpt


# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.branin()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds     
objective_true.plot()


domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]


def mycost(x):
    cost_f  = np.atleast_2d(.1*x[:,0]**2 +.1*x[:,1]**2).T
    cost_df = np.array([0.2*x[:,0],0.2*x[:,1]]).T
    return cost_f, cost_df


# plot the cost fucntion
grid = 400
bounds = objective_true.bounds
X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
x1, x2 = np.meshgrid(X1, X2)
X = np.hstack((x1.reshape(grid*grid,1),x2.reshape(grid*grid,1)))

cost_X, _ = mycost(X)


# Feasible region
plt.contourf(X1, X2, cost_X.reshape(grid,grid),100, alpha=1,origin ='lower')
plt.title('Cost function')
plt.colorbar()


get_ipython().magic('pinfo GPyOpt.methods.BayesianOptimization')


from numpy.random import seed
seed(123)
BO = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            domain = domain, 
                                            initial_design_numdata = 5,
                                            acquisition_type = 'EI',              
                                            normalize_Y = True,
                                            exact_feval = False,
                                            acquisition_jitter = 0.05)  


seed(123)
BO_cost = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            cost_withGradients = mycost,
                                            initial_design_numdata =5,
                                            domain = domain,                  
                                            acquisition_type = 'EI',              
                                            normalize_Y = True,
                                            exact_feval = False,
                                            acquisition_jitter = 0.05)    


BO.plot_acquisition()


BO_cost.plot_acquisition()


BO_cost.run_optimization(15)
BO_cost.plot_acquisition()


BO.run_optimization(15)
BO.plot_acquisition()


# # GPyOpt: parallel Bayesian optimization
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# *Last updated Tuesday, 15 March 2016.*
# 

# In this noteboook we are going to learn how to use GPyOpt to run parallel BO methods. The goal of these approaches is to make use of all the computational power or our machine to perform the optimization. For instance, if we hace a computer with 4 cores, we may want to make 4 evaluations of $f$ in parallel everytime we test the performance of the algorithm. 
# 
# In this notebook we will use the **Local Penalization** method describe in the paper *Batch Bayesian Optimization via Local Penalization*.
# 

from IPython.display import HTML 
HTML('<iframe src=http://arxiv.org/pdf/1505.08052v4.pdf width=700 height=550></iframe>')


get_ipython().magic('pylab inline')
import GPyOpt


# As in previous examples we use a synthetic objective function but you can think about doing the same with any function you like. In this case, we use the Branin function. For the optimization we will perturb the evaluations with Gaussian noise with sd = 0.1.
# 

# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.branin()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds        


domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]


objective_true.plot()


# As in previous cases, we create a GPyOpt object with the desing space and fucntion to optimize. In this case we need to select the evaluator type, which in this case is the *local penalization method* the batch size and the number of cores that we want to use. The evaluation of the function will be splitted accross the available cores.
# 

batch_size = 4
num_cores = 4


from numpy.random import seed
seed(123)
BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            domain = domain,                  
                                            acquisition_type = 'EI',              
                                            normalize_Y = True,
                                            initial_design_numdata = 10,
                                            evaluator_type = 'local_penalization',
                                            batch_size = batch_size,
                                            num_cores = num_cores,
                                            acquisition_jitter = 0)    


# We will optimize this function by running 10 parallel evaluations in 3 cores of our machine. 
# 

# --- Run the optimization for 10 iterations
max_iter = 10                                                     
BO_demo_parallel.run_optimization(max_iter)


# We plot the resutls. Observe that the final number of evaluations that we will make is $10*4=40$. 
# 

BO_demo_parallel.plot_acquisition()


BO_demo_parallel.suggested_sample


# See how the method explores the space using the four parallel evaluations of $f$ and it is able to identify the location of the three minima. 
# 

objective_noisy.min


# # GPyOpt: Modular Bayesian Optimization 
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# 
# *Last updated Friday, 11 March 2016.*
# 

# In the [Introduction Bayesian Optimization GPyOpt](./GPyOpt_reference_manual.ipynb) we showed how GPyOpt can be used to solve optimization problems with some basic functionalities. The object 
# 
# ```
# GPyOpt.methods.BayesianOptimization
# ```
# is used to initialize the desired functionalities, such us the acquisition function, the initial design or the model. In some cases we want to have control over those objects and we may want to replace some element in the loop without having to integrate the new elements in the base code framework. This is now possible through the modular implementation of the package using the
# 
# ```
# GPyOpt.methods.ModularBayesianOptimization
# ```
# 
# class. In this notebook we are going to show how to use the backbone of GPyOpt to run a Bayesian optimization algorithm in which we will use our own acquisition function. In particular we are going to use the Expected Improvement integrated over the jitter parameter. That is
# 
# $$acqu_{IEI}(x;\{x_n,y_n\},\theta) = \int acqu_{EI}(x;\{x_n,y_n\},\theta,\psi) p(\psi;a,b)d\psi $$
# where $p(\psi;a,b)$ is, in this example, the distribution [$Beta(a,b)$](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.beta.html).
# 
# This acquisition is not available in GPyOpt, but we will implement and use in this notebook. The same can be done for other models, acquisition optimizers etc.
# 
# As usual, we start loading GPy and GPyOpt.
# 

get_ipython().magic('pylab inline')
import GPyOpt
import GPy


# In this example we will use the Branin function as a test case.
# 

# --- Function to optimize
func  = GPyOpt.objective_examples.experiments2d.branin()
func.plot()


# Because we are won't use the pre implemented wrapper, we need to create the classes for each element of the optimization. In total we need to create:
# 
# * Class for the **objective function**,
# 

objective = GPyOpt.core.task.SingleObjective(func.f)


# * Class for the **design space**,
# 

space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},
                                    {'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])


# * Class for the **model type**,
# 

model = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)


# * Class for the **acquisition optimizer**,
# 

aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)


# * Class for the **initial design**,
# 

initial_design = GPyOpt.util.stats.initial_design('random', space, 5)


# * Class for the **acquisition function**. Because we want to use our own acquisition, we need to implement a class to handle it. We will use the currently available Expected Improvement to create an integrated version over the jitter parameter. Samples will be generated using a beta distribution with parameters a and b as it is done using the default [numpy beta function](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.beta.html).
# 

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
from numpy.random import beta

class jitter_integrated_EI(AcquisitionBase):
    def __init__(self, model, space, optimizer=None, cost_withGradients=None, par_a=1, par_b=1, num_samples= 100):
        super(jitter_integrated_EI, self).__init__(model, space, optimizer)
        
        self.par_a = par_a
        self.par_b = par_b
        self.num_samples = num_samples
        self.samples = beta(self.par_a,self.par_b,self.num_samples)
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)
    
    def acquisition_function(self,x):
        acqu_x = np.zeros((x.shape[0],1))       
        for k in range(self.num_samples):
            self.EI.jitter = self.samples[k]
            acqu_x +=self.EI.acquisition_function(x)           
        return acqu_x/self.num_samples
    
    def acquisition_function_withGradients(self,x):
        acqu_x      = np.zeros((x.shape[0],1))       
        acqu_x_grad = np.zeros(x.shape)
        
        for k in range(self.num_samples):
            self.EI.jitter = self.samples[k]       
            acqu_x_sample, acqu_x_grad_sample =self.EI.acquisition_function_withGradients(x) 
            acqu_x += acqu_x_sample
            acqu_x_grad += acqu_x_grad_sample           
        return acqu_x/self.num_samples, acqu_x_grad/self.num_samples


# Now we initialize the class for this acquisition and we plot the histogram of the used samples to integrate the acquisition.
# 

acquisition = jitter_integrated_EI(model, space, optimizer=aquisition_optimizer, par_a=1, par_b=10, num_samples=2000)
xx = plt.hist(acquisition.samples,bins=50)


# * Finally we create the class for the **type of evaluator**,
# 

# --- CHOOSE a collection method
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)


# With all the classes on place,including the one we have created for this example, we can now create the **Bayesian optimization object**.
# 

bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design)


# And we run the optimization.
# 

max_iter  = 10                                            
bo.run_optimization(max_iter = max_iter) 


# We plot the acquisition and the diagnostic plots.
# 

bo.plot_acquisition()
bo.plot_convergence()


# # Creating new acquisitions for GPyOpt
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# ## Reference Manual index
# 
# *Last updated Friday, 11 Jun 2016.*
# 

# You can create and use your own aquisition functions in GPyOpt. To do it just complete the following template.
# 

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
    
class AcquisitionNew(AcquisitionBase):
    
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    
    def __init__(self, model, space, optimizer, cost_withGradients=None, **kwargs):
        self.optimizer = optimizer
        super(AcquisitionNew, self).__init__(model, space, optimizer)
        
        # --- UNCOMMENT ONE OF THE TWO NEXT BITS
             
        # 1) THIS ONE IF THE EVALUATION COSTS MAKES SENSE
        #
        # if cost_withGradients == None:
        #     self.cost_withGradients = constant_cost_withGradients
        # else:
        #     self.cost_withGradients = cost_withGradients 

        # 2) THIS ONE IF THE EVALUATION COSTS DOES NOT MAKE SENSE
        #
        # if cost_withGradients == None:
        #     self.cost_withGradients = constant_cost_withGradients
        # else:
        #     print('LBC acquisition does now make sense with cost. Cost set to constant.')  
        #     self.cost_withGradients = constant_cost_withGradients


    def _compute_acq(self,x):
        
        # --- DEFINE YOUR AQUISITION HERE (TO BE MAXIMIZED)
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array  
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the 
        # values of the acquisition at x.
        #
        
        return f_acqu_x
    
    def _compute_acq_withGradients(self, x):
        
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array  
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the 
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        
        return f_acqu_x, df_acqu_x
            


# # GPyOpt: mixing different types of variables
# 
# ### Written by Javier Gonzalez, University of Sheffield.
# 
# 
# *Last updated Monday, 14 March 2016.*
# 

# In this notebook we are going to see how to used GPyOpt to solve optimizaiton problems in which the domain of the fucntion is defined in terms of a variety of continous and discrete variables. To this end we start by loading GPyOpt. 
# 

get_ipython().magic('pylab inline')
import GPyOpt
from numpy.random import seed
seed(123)


# We will use the **Alpine1** function, that it is available in the benchmark of functions of the package. This function is defined for arbitrary dimension. In this example we will work in dimension 5. The functional form of the Alpine1 function is:
# 
# $$f(x_1,x_2,x_3,x_4,x_5)=\sum_{i=1}^{5} \lvert {x_i \sin \left( x_i \right) + 0.1 x_i} \rvert$$
# 

# We load the function from GPyOpt, assuming that noisy free values will be sampled.
# 

func  = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=5) 


# We will consider that variables $x_1$ and $x_2$ are continuous and defined in the interval $[-5,5]$, variable $x_3$ is takes continuous values in the interval $[-1,5]$ and the variables $x_4$ and $x_5$ are discrete and take values $\{-3,0,3\}$ and $\{-5,-1,1,5\}$ respectively. Next we define this domain to use it in GPyOpt.
# 

mixed_domain =[{'name': 'var1_2', 'type': 'continuous', 'domain': (-10,10),'dimensionality': 2},
               {'name': 'var3', 'type': 'continuous', 'domain': (-8,3)},
               {'name': 'var4', 'type': 'discrete', 'domain': (-2,0,2)},
               {'name': 'var5', 'type': 'discrete', 'domain': (-1,5)}]


# And that's it! We can proceed now with the optimization as we have seen in previous examples. We just need to select, the model, acquisition, optimizer, etc. Given the previous domain, GPyOpt will handle for use the fact that we have discrete variables. This is taken into account in the optimization but also when generating initial values, etc. Let's see how it works when we use the Expected improvement.
# 

myBopt = GPyOpt.methods.BayesianOptimization(f=func.f,                   # function to optimize       
                                             domain=mixed_domain,        # box-constrains of the problem
                                             initial_design_numdata = 20,# number data initial design
                                             acquisition_type='EI',      # Expected Improvement
                                             exact_feval = True)         # True evaluations


# Now, we run the optimization for 20 iterations or a maximum of 60 seconds and we show the convergence plots.
# 

max_iter = 10
max_time = 60

myBopt.run_optimization(max_iter, max_time)


myBopt.X


myBopt.plot_convergence()


# The current best found value is:
# 

myBopt.x_opt


