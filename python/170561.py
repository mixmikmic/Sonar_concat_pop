import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('qtconsole', '--colors=linux')
plt.style.use('ggplot')


# # Chapter 4 - Inferences with Gaussians
# ## 4.1 Inferring a mean and standard deviation
# 
# 
# Inferring the mean and variance of a Gaussian distribution. 
# $$ \mu \sim \text{Gaussian}(0, .001)  $$
# $$ \sigma \sim \text{Uniform} (0, 10)  $$
# $$ x_{i} \sim \text{Gaussian} (\mu, \frac{1}{\sigma^2})  $$
# 

# Data
x = np.array([1.1, 1.9, 2.3, 1.8])
n = len(x)
    
with pm.Model() as model1:
    # prior
    mu = pm.Normal('mu', mu=0, tau=.001)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    # observed
    xi = pm.Normal('xi',mu=mu, tau=1/(sigma**2), observed=x)
    # inference
    trace = pm.sample(1e3, njobs=2)

pm.traceplot(trace[50:]);


from matplotlib.ticker import NullFormatter
nullfmt = NullFormatter()         # no labels
y = trace['mu'][50:]
x = trace['sigma'][50:]

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y, c=[1, 1, 1], alpha=.5)

# now determine nice limits by hand:
binwidth1 = 0.25
axScatter.set_xlim((-.01, 10.5))
axScatter.set_ylim((-0, 5))

bins1 = np.linspace(-.01, 10.5, 20)
axHistx.hist(x, bins=bins1)
bins2 = np.linspace(-0, 5, 20)
axHisty.hist(y, bins=bins2, orientation='horizontal')

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim());

print('The mu estimation is: ', y.mean())
print('The sigma estimation is: ', x.mean())


# ### Note from Junpeng Lao
# There are might be divergence warning (Uniform prior on sigma is not a good idea in general), which you can further visualize below
# 

# display the total number and percentage of divergent
divergent = trace['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace)
print('Percentage of Divergent %.5f' % divperc)

# scatter plot for the identifcation of the problematic neighborhoods in parameter space
plt.figure(figsize=(6, 6))
y = trace['mu']
x = trace['sigma']
plt.scatter(x[divergent == 0], y[divergent == 0], color='r', alpha=.05)
plt.scatter(x[divergent == 1], y[divergent == 1], color='g', alpha=.5);


# ## 4.2 The seven scientists
# 
# 
# The model:
# $$ \mu \sim \text{Gaussian}(0, .001)  $$
# $$ \lambda_{i} \sim \text{Gamma} (.001, .001)  $$
# $$ \sigma = 1/{\sqrt\lambda_{i}} $$  
# $$ x_{i} \sim \text{Gaussian} (\mu, \lambda_{i})  $$
# 
# The mean is the same for all seven scientists, while the standard deviations are different 
# 

# data
x = np.array([-27.020,3.570,8.191,9.898,9.603,9.945,10.056])
n = len(x)

with pm.Model() as model2: 
    # prior
    mu = pm.Normal('mu', mu=0, tau=.001)
    lambda1 = pm.Gamma('lambda1', alpha=.01, beta=.01, shape=(n))
    # sigma = pm.Deterministic('sigma',1 / sqrt(lambda1))
    # observed
    xi = pm.Normal('xi',mu = mu, tau = lambda1, observed = x )

    # inference
    trace2 = pm.sample(5000, njobs=2)

burnin = 1000
pm.traceplot(trace2[burnin:]);

mu = trace2['mu'][burnin:]
lambda1 = trace2['lambda1'][burnin:]

print('The mu estimation is: ', mu.mean())
print('The sigma estimation is: ')
for i in np.mean(np.squeeze(lambda1),axis=0):
    print(1 / np.sqrt(i))


# ## 4.3 Repeated measurement of IQ
# 
# 
# The model:
# $$ \mu_{i} \sim \text{Uniform}(0, 300)  $$
# $$ \sigma \sim \text{Uniform} (0, 100)  $$
# $$ x_{ij} \sim \text{Gaussian} (\mu_{i}, \frac{1}{\sigma^2})  $$
# 
# Data Come From Gaussians With Different Means But Common Precision
# 

# Data
y = np.array([[90,95,100],[105,110,115],[150,155,160]])
ntest = 3
nsbj = 3

import sys
eps = sys.float_info.epsilon

with pm.Model() as model3:
    # mu_i ~ Uniform(0, 300)
    mui = pm.Uniform('mui', 0, 300, shape=(nsbj,1))

    # sg ~ Uniform(0, 100)
    # sg = pm.Uniform('sg', .0, 100)
    
    # It is more stable to use a Gamma prior
    lambda1 = pm.Gamma('lambda1', alpha=.01, beta=.01)
    sg = pm.Deterministic('sg',1 / np.sqrt(lambda1))
    
    # y ~ Normal(mu_i, sg)
    yd = pm.Normal('y', mu=mui, sd=sg, observed=y)
    
    trace3 = pm.sample(5e3, njobs=2)


burnin = 500
pm.traceplot(trace3[burnin:]);

mu = trace3['mui'][burnin:]
sigma = trace3['sg'][burnin:]

print('The mu estimation is: ', np.mean(mu, axis=0))
print('The sigma estimation is: ',sigma.mean())


import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('qtconsole', '--colors=linux')
plt.style.use('ggplot')

from matplotlib import gridspec
from theano import tensor as tt
from scipy import stats


# # Chapter 13 - Extrasensory perception
# 
# ## 13.1 Evidence for optional stopping
# The negative relation between the number of subjects and effect size suggests that the results are contaminated by optional stopping. Here we inferred on the correlation coefficient (same as in chapter 5):  
# 
# $$ \mu_{1},\mu_{2} \sim \text{Gaussian}(0, .001)  $$
# $$ \sigma_{1},\sigma_{2} \sim \text{InvSqrtGamma} (.001, .001)  $$
# $$ r \sim \text{Uniform} (-1, 1) $$  
# $$ x_{i} \sim \text{MvGaussian} ((\mu_{1},\mu_{2}), \begin{bmatrix}\sigma_{1}^2 & r\sigma_{1}\sigma_{2}\\r\sigma_{1}\sigma_{2} & \sigma_{2}^2\end{bmatrix}^{-1})  $$
# 

# Sample size N and effect size E in the Bem experiments
N = np.array([100, 150, 97, 99, 100, 150, 200, 100, 50])
E = np.array([.25, .20, .25, .20, .22, .15, .09, .19, .42])

y = np.vstack([N, E]).T
n,n2 = np.shape(y) # number of experiments

with pm.Model() as model1:
    # r∼Uniform(−1,1)
    r = pm.Uniform('r', lower=-1, upper=1)
    
    # μ1,μ2∼Gaussian(0,.001)
    mu = pm.Normal('mu', mu=0, tau=.001, shape=n2)
    
    # σ1,σ2∼InvSqrtGamma(.001,.001)
    lambda1 = pm.Gamma('lambda1', alpha=.001, beta=.001)
    lambda2 = pm.Gamma('lambda2', alpha=.001, beta=.001)
    sigma1 = pm.Deterministic('sigma1', 1/np.sqrt(lambda1))
    sigma2 = pm.Deterministic('sigma2', 1/np.sqrt(lambda2))
    
    cov = pm.Deterministic('cov', tt.stacklists([[lambda1**-1, r*sigma1*sigma2],
                                                [r*sigma1*sigma2, lambda2**-1]]))
    
    tau1 = pm.Deterministic('tau1', tt.nlinalg.matrix_inverse(cov))
    
    yd = pm.MvNormal('yd', mu=mu, tau=tau1, observed=y)

    trace1=pm.sample(3e3, njobs=2)

pm.traceplot(trace1, varnames=['r']);


from scipy.stats.kde import gaussian_kde
burnin=50
r = trace1['r'][burnin:]

fig = plt.figure(figsize=(12, 6)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
ax0 = plt.subplot(gs[0])
ax0.scatter(y[:, 0], y[:, 1], s=50)
plt.axis([0, 215, 0, .5])
plt.xlabel('Number of Subjects')
plt.ylabel('Effect Size')

my_pdf = gaussian_kde(r)
x=np.linspace(-1, 1, 200)
ax1 = plt.subplot(gs[1])
ax1.plot(x, my_pdf(x), 'r', lw=2.5, alpha=0.6,) # distribution function
ax1.plot(x, np.ones(x.size)*.5, 'k--', lw=2.5, alpha=0.6, label='Prior')
posterior = my_pdf(0)             # this gives the pdf at point delta = 0
prior     = .5       # height of order-restricted prior at delta = 0
BF01      = posterior/prior
print ('the Bayes Factor is %.5f'%(1/BF01))
ax1.plot([0, 0], [posterior, prior], 'k-', 
         [0, 0], [posterior, prior], 'ko',
         lw=1.5, alpha=1)
# ax1.hist(r, bins=100, normed=1,alpha=.3)
plt.xlabel('Correlation r')
plt.ylabel('Density')

# Compare to approximation Jeffreys (1961), pp. 289-292:
import scipy as scipy 
freqr=scipy.corrcoef(y[:, 0], y[:, 1])
BF_Jeffrey=1/(((2*(n-1)-1)/np.pi)**.5 * (1-freqr[0,1]**2)**(.5*((n-1)-3)))
print ('the approximation Jeffreys Bayes Factor is %.5f'%(BF_Jeffrey))

# Compare to exact solution Jeffreys (numerical integration):
BF_Jeffex = scipy.integrate.quad(lambda rho: ((1-rho**2)**((n-1)/2)) / ((1-rho*freqr[0,1])**((n-1)-.5)), -1, 1)
print('the exact Jeffreys Bayes Factor is %.5f'%(BF_Jeffex[0]/2))

plt.show()


# ## 13.2 Evidence for differences in ability
#   
# $$ \mu_{1},\mu_{2} \sim \text{Gaussian}(0, .001)  $$
# $$ \sigma_{1},\sigma_{2} \sim \text{InvSqrtGamma} (.001, .001)  $$
# $$ r \sim \text{Uniform} (0, 1) $$  
# $$ \hat\theta_{i} \sim \text{MvGaussian} ((\mu_{1},\mu_{2}), \begin{bmatrix}\sigma_{1}^2 & r\sigma_{1}\sigma_{2}\\r\sigma_{1}\sigma_{2} & \sigma_{2}^2\end{bmatrix}^{-1})  $$  
# $$ \theta_{ij} =  \Phi(\hat\theta_{ij})$$
# $$ k_{ij} \sim \text{Binomial}(\theta_{ij},n)$$
# 

# Data
# Proportion correct on erotic pictures, block 1 and block 2:
prc1er=np.array([0.6000000, 0.5333333, 0.6000000, 0.6000000, 0.4666667, 
                 0.6666667, 0.6666667, 0.4000000, 0.6000000, 0.6000000,
                 0.4666667, 0.6666667, 0.4666667, 0.6000000, 0.3333333,
                 0.4000000, 0.4000000, 0.2666667, 0.3333333, 0.5333333,
                 0.6666667, 0.5333333, 0.6000000, 0.4000000, 0.4666667, 
                 0.7333333, 0.6666667, 0.6000000, 0.6666667, 0.5333333,
                 0.5333333, 0.6666667, 0.4666667, 0.3333333, 0.4000000,
                 0.5333333, 0.4000000, 0.4000000, 0.3333333, 0.4666667,
                 0.4000000, 0.4666667, 0.4666667, 0.5333333, 0.3333333,
                 0.7333333, 0.2666667, 0.6000000, 0.5333333, 0.4666667,
                 0.4000000, 0.5333333, 0.6666667, 0.4666667, 0.5333333,
                 0.5333333, 0.4666667, 0.4000000, 0.4666667, 0.6666667,
                 0.4666667, 0.3333333, 0.3333333, 0.3333333, 0.4000000,
                 0.4000000, 0.6000000, 0.4666667, 0.3333333, 0.3333333,
                 0.6666667, 0.5333333, 0.3333333, 0.6000000, 0.4666667,
                 0.4666667, 0.4000000, 0.3333333, 0.4666667, 0.5333333,
                 0.8000000, 0.4000000, 0.5333333, 0.5333333, 0.6666667,
                 0.6666667, 0.6666667, 0.6000000, 0.6000000, 0.5333333,
                 0.3333333, 0.4666667, 0.6666667, 0.5333333, 0.3333333,
                 0.3333333, 0.2666667, 0.2666667, 0.4666667, 0.6666667])

prc2er=np.array([0.3333333, 0.6000000, 0.5333333, 0.2666667, 0.6666667,
                 0.5333333, 0.6666667, 0.4666667, 0.4666667, 0.6666667,
                 0.4000000, 0.6666667, 0.2666667, 0.4000000, 0.4666667,
                 0.3333333, 0.5333333, 0.6000000, 0.3333333, 0.4000000,
                 0.4666667, 0.4666667, 0.6000000, 0.5333333, 0.5333333,
                 0.6000000, 0.5333333, 0.6666667, 0.6000000, 0.2666667,
                 0.4666667, 0.4000000, 0.6000000, 0.5333333, 0.4000000,
                 0.4666667, 0.5333333, 0.3333333, 0.4000000, 0.4666667,
                 0.8000000, 0.6000000, 0.2000000, 0.6000000, 0.4000000,
                 0.4000000, 0.2666667, 0.2666667, 0.6000000, 0.4000000,
                 0.4000000, 0.4000000, 0.4000000, 0.4000000, 0.6666667,
                 0.7333333, 0.5333333, 0.5333333, 0.3333333, 0.6000000,
                 0.5333333, 0.5333333, 0.4666667, 0.5333333, 0.4666667,
                 0.5333333, 0.4000000, 0.4000000, 0.4666667, 0.6000000,
                 0.6000000, 0.6000000, 0.4666667, 0.6000000, 0.6666667,
                 0.5333333, 0.4666667, 0.6000000, 0.2000000, 0.5333333,
                 0.4666667, 0.4000000, 0.5333333, 0.5333333, 0.5333333,
                 0.5333333, 0.6000000, 0.6666667, 0.4000000, 0.4000000,
                 0.5333333, 0.8000000, 0.6000000, 0.4000000, 0.2000000,
                 0.6000000, 0.6666667, 0.4666667, 0.4666667, 0.4666667])             
Nt = 60
xobs = np.vstack([prc1er, prc2er]).T*Nt
n,n2 = np.shape(xobs) # number of participants

plt.figure(figsize=[5, 5])
plt.hist2d(xobs[:, 0], xobs[:, 1], cmap = 'binary')
plt.xlabel('Performance Session 1', fontsize=15)
plt.ylabel('Performance Session 2', fontsize=15)
plt.axis([0, 60, 0, 60])
plt.show()


def phi(x):
    # probit transform
    return 0.5 + 0.5 * pm.math.erf(x/pm.math.sqrt(2))

with pm.Model() as model2:
    # r∼Uniform(−1,1)
    r =  pm.Uniform('r', lower=0, upper=1)
    
    # μ1,μ2∼Gaussian(0,.001)
    mu = pm.Normal('mu', mu=0, tau=.001, shape=n2)
    
    # σ1,σ2∼InvSqrtGamma(.001,.001)
    lambda1 = pm.Gamma('lambda1', alpha=.001, beta=.001, testval=100)
    lambda2 = pm.Gamma('lambda2', alpha=.001, beta=.001, testval=100)
    sigma1 = pm.Deterministic('sigma1', 1/np.sqrt(lambda1))
    sigma2 = pm.Deterministic('sigma2', 1/np.sqrt(lambda2))
    
    cov = pm.Deterministic('cov', tt.stacklists([[lambda1**-1, r*sigma1*sigma2],
                                                [r*sigma1*sigma2, lambda2**-1]]))
    
    tau1 = pm.Deterministic('tau1', tt.nlinalg.matrix_inverse(cov))

    thetai = pm.MvNormal('thetai', mu=mu, tau=tau1, shape=(n, n2))
    theta = phi(thetai)
    kij = pm.Binomial('kij', p=theta, n=Nt, observed=xobs)
    
    trace2=pm.sample(3e3, njobs=2)

pm.traceplot(trace2, varnames=['r']);


from scipy.stats.kde import gaussian_kde
r = trace2['r'][1000:]

fig = plt.figure(figsize=(6, 6)) 

my_pdf = gaussian_kde(r)
x1=np.linspace(0, 1, 200)

plt.plot(x1, my_pdf(x1), 'r', lw=2.5, alpha=0.6,) # distribution function
plt.plot(x1, np.ones(x1.size), 'k--', lw=2.5, alpha=0.6, label='Prior')
posterior = my_pdf(0)        # this gives the pdf at point delta = 0
prior     = 1       # height of order-restricted prior at delta = 0
BF01      = posterior/prior
print ('the Bayes Factor is %.5f'%(1/BF01))
plt.plot([0, 0], [posterior, prior], 'k-', 
         [0, 0], [posterior, prior], 'ko',
         lw=1.5, alpha=1)
# ax1.hist(r, bins=100, normed=1,alpha=.3)
plt.xlim([0, 1])
plt.xlabel('Correlation r')
plt.ylabel('Density')

# Compare to exact solution Jeffreys (numerical integration):
freqr=scipy.corrcoef(xobs[:, 0]/Nt, xobs[:, 1]/Nt)
BF_Jeffex = scipy.integrate.quad(lambda rho: ((1-rho**2)**((n-1)/2)) / ((1-rho*freqr[0, 1])**((n-1)-.5)), 0, 1)
print('the exact Jeffreys Bayes Factor is %.5f'%(BF_Jeffex[0]))

plt.show()


# ## 13.3 Evidence for the impact of extraversion
#   
# $$ \mu_{1},\mu_{2} \sim \text{Gaussian}(0, .001)  $$
# $$ \sigma_{1},\sigma_{2} \sim \text{InvSqrtGamma} (.001, .001)  $$
# $$ r \sim \text{Uniform} (-1, 1) $$  
# $$ \hat\theta_{i} \sim \text{MvGaussian} ((\mu_{1},\mu_{2}), \begin{bmatrix}\sigma_{1}^2 & r\sigma_{1}\sigma_{2}\\r\sigma_{1}\sigma_{2} & \sigma_{2}^2\end{bmatrix}^{-1})  $$  
# $$ \theta_{i1} =  \Phi(\hat\theta_{i1})$$
# $$ \theta_{i2} =  100\Phi(\hat\theta_{i2})$$
# $$ k_{i} \sim \text{Binomial}(\theta_{i1},n)$$
# $$ x_{i} \sim \text{Gaussian}(\theta_{i2},\lambda^x)$$
# 

k2 = np.array([36, 32, 36, 36, 28, 40, 40, 24, 36, 36, 28, 40, 28, 
       36, 20, 24, 24, 16, 20, 32, 40, 32, 36, 24, 28, 44,
       40, 36, 40, 32, 32, 40, 28, 20, 24, 32, 24, 24, 20, 
       28, 24, 28, 28, 32, 20, 44, 16, 36, 32, 28, 24, 32,
       40, 28, 32, 32, 28, 24, 28, 40, 28, 20, 20, 20, 24,
       24, 36, 28, 20, 20, 40, 32, 20, 36, 28, 28, 24, 20,
       28, 32, 48, 24, 32, 32, 40, 40, 40, 36, 36, 32, 20,
       28, 40, 32, 20, 20, 16, 16, 28, 40])
       
x2 = np.array([50, 80, 79, 56, 50, 80, 53, 84, 74, 67, 50, 45, 62, 
       65, 71, 71, 68, 63, 67, 58, 72, 73, 63, 54, 63, 70, 
       81, 71, 66, 74, 70, 84, 66, 73, 78, 64, 54, 74, 62, 
       71, 70, 79, 66, 64, 62, 63, 60, 56, 72, 72, 79, 67, 
       46, 67, 77, 55, 63, 44, 84, 65, 41, 62, 64, 51, 46,
       53, 26, 67, 73, 39, 62, 59, 75, 65, 60, 69, 63, 69, 
       55, 63, 86, 70, 67, 54, 80, 71, 71, 55, 57, 41, 56, 
       78, 58, 76, 54, 50, 61, 60, 32, 67])
       
nsubjs = len(k2)
ntrials = 60
sigmax = 3

plt.figure(figsize=[8, 5])
plt.scatter(x2, k2)
plt.xlabel('Extraversion', fontsize=15)
plt.ylabel('Performance Session 2', fontsize=15)
plt.axis([0,100,0,60])
plt.show()


with pm.Model() as model3:
    # r∼Uniform(−1,1)
    r =  pm.Uniform('r', lower=0, upper=1)
    
    # μ1,μ2∼Gaussian(0,.001)
    mu = pm.Normal('mu', mu=0, tau=.001, shape=n2)
    
    # σ1,σ2∼InvSqrtGamma(.001,.001)
    lambda1 = pm.Gamma('lambda1', alpha=.001, beta=.001, testval=100)
    lambda2 = pm.Gamma('lambda2', alpha=.001, beta=.001, testval=100)
    sigma1 = pm.Deterministic('sigma1', 1/np.sqrt(lambda1))
    sigma2 = pm.Deterministic('sigma2', 1/np.sqrt(lambda2))
    
    cov = pm.Deterministic('cov', tt.stacklists([[lambda1**-1, r*sigma1*sigma2],
                                                [r*sigma1*sigma2, lambda2**-1]]))
    
    tau1 = pm.Deterministic('tau1', tt.nlinalg.matrix_inverse(cov))
    
    thetai = pm.MvNormal('thetai', mu=mu, tau=tau1, shape=(n, n2))
    theta1 = phi(thetai[:,0])
    theta2 = 100 * phi(thetai[:,1])
    
    ki = pm.Binomial('ki', p=theta1, n=Nt, observed=k2)
    xi = pm.Normal('xi', mu=theta2, sd=sigmax, observed=x2)
    
    trace3=pm.sample(3e3, njobs=2)

pm.traceplot(trace3, varnames=['r']);


r = trace3['r'][1000:]

fig = plt.figure(figsize=(6, 6)) 

my_pdf = gaussian_kde(r)
x1=np.linspace(-1, 1, 200)

plt.plot(x1, my_pdf(x1), 'r', lw=2.5, alpha=0.6,) # distribution function
plt.plot(x1, np.ones(x1.size)*.5, 'k--', lw=2.5, alpha=0.6, label='Prior')
posterior = my_pdf(0)        # this gives the pdf at point delta = 0
prior     = .5       # height of order-restricted prior at delta = 0
BF01      = posterior/prior
print ('the Bayes Factor is %.5f'%(1/BF01))
plt.plot([0, 0], [posterior, prior], 'k-', 
         [0, 0], [posterior, prior], 'ko',
         lw=1.5, alpha=1)
# ax1.hist(r, bins=100, normed=1,alpha=.3)
plt.xlim([-1, 1])
plt.xlabel('Correlation r')
plt.ylabel('Density')

plt.show()


import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('qtconsole', '--colors=linux')
plt.style.use('ggplot')

from matplotlib import gridspec
from theano import tensor as tt
from scipy import stats


# # Chapter 18 - Heuristic decision-making
# 
# ## 18.1 Take-the-best
# The take-the-best (TTB) model of decision-making (Gigerenzer & Goldstein, 1996) is a simple but influential account of how people choose between two stimuli on some criterion, and a good example of the general class of heuristic decision-making models (e.g., Gigerenzer & Todd, 1999; Gigerenzer & Gaissmaier, 2011; Payne, Bettman, & Johnson, 1990).
# 
# 
# $$ t_q = \text{TTB}_{s}(\mathbf a_q,\mathbf b_q)$$
# $$ \gamma \sim \text{Uniform}(0.5,1)$$  
# $$ y_{iq} \sim
# \begin{cases}
# \text{Bernoulli}(\gamma) & \text{if $t_q = a$} \\text{Bernoulli}(1- \gamma) & \text{if $t_q = b$} \\text{Bernoulli}(0.5) & \text{otherwise}
# \end{cases}  $$
# 

import scipy.io as sio
matdata = sio.loadmat('data/StopSearchData.mat')

y = np.squeeze(matdata['y'])
m = np.squeeze(np.float32(matdata['m']))
p = np.squeeze(matdata['p'])
v = np.squeeze(np.float32(matdata['v']))
x = np.squeeze(np.float32(matdata['x']))

# Constants
n, nc = np.shape(m)  # number of stimuli and cues
nq, _ = np.shape(p)  # number of questions
ns, _ = np.shape(y)  # number of subjects


s = np.argsort(v)  # s[1:nc] <- rank(v[1:nc])
t = []
# TTB Model For Each Question
for q in range(nq):
    # Add Cue Contributions To Mimic TTB Decision
    tmp1 = np.zeros(nc)
    for j in range(nc):
        tmp1[j] = (m[p[q, 0]-1, j]-m[p[q, 1]-1, j])*np.power(2, s[j])
    # Find if Cue Favors First, Second, or Neither Stimulus
    tmp2 = np.sum(tmp1)
    tmp3 = -1*np.float32(-tmp2 > 0)+np.float32(tmp2 > 0)
    t.append(tmp3+1)
t = np.asarray(t, dtype=int)
tmat = np.tile(t[np.newaxis, :], (ns, 1))


with pm.Model() as model1:
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])
    yiq = pm.Bernoulli('yiq', p=gammat[tmat], observed=y)
    trace1 = pm.sample(3e3, njobs=2, tune=1000)
    
pm.traceplot(trace1);


ppc = pm.sample_ppc(trace1, samples=100, model=model1)
yiqpred = np.asarray(ppc['yiq'])
fig = plt.figure(figsize=(16, 8))
x1 = np.repeat(np.arange(ns)+1, nq).reshape(ns, -1).flatten()
y1 = np.repeat(np.arange(nq)+1, ns).reshape(nq, -1).T.flatten()

plt.scatter(y1, x1, s=np.mean(yiqpred, axis=0)*200, c='w')
plt.scatter(y1[y.flatten() == 1], x1[y.flatten() == 1], marker='x', c='r')
plt.plot(np.ones(100)*24.5, np.linspace(0, 21, 100), '--', lw=1.5, c='k')
plt.axis([0, 31, 0, 21])
plt.show()


# ## 18.2 Stopping
# A common comparison (e.g., Bergert & Nosofsky, 2007; Lee & Cummins, 2004) is between TTB and a model often called the Weighted ADDitive (WADD) model, which sums the evidence for both decision alternatives over all available cues, and chooses the one with the greatest evidence.
# 
# $$ \phi \sim \text{Uniform}(0,1)$$
# $$ z_i \sim \text{Bernoulli}(\phi)$$
# $$ \gamma \sim \text{Uniform}(0.5,1)$$  
# $$ t_{iq} = 
# \begin{cases}
# \text{TTB}\,(\mathbf a_q,\mathbf b_q) & \text{if $z_i = 1$} \\text{WADD}\,(\mathbf a_q,\mathbf b_q) & \text{if $z_i = 0$} \\end{cases}  $$  
# 
# $$ y_{iq} \sim
# \begin{cases}
# \text{Bernoulli}(\gamma) & \text{if $t_{iq} = a$} \\text{Bernoulli}(1- \gamma) & \text{if $t_{iq} = b$} \\text{Bernoulli}(0.5) & \text{otherwise}
# \end{cases}  $$
# 

# Question cue contributions template
qcc = np.zeros((nq, nc))
for q in range(nq):
    # Add Cue Contributions To Mimic TTB Decision
    for j in range(nc):
        qcc[q, j] = (m[p[q, 0]-1, j]-m[p[q, 1]-1, j])

qccmat = np.tile(qcc[np.newaxis, :, :], (ns, 1, 1))
# TTB Model For Each Question
s = np.argsort(v)  # s[1:nc] <- rank(v[1:nc])
smat = np.tile(s[np.newaxis, :], (ns, nq, 1))
ttmp = np.sum(qccmat*np.power(2, smat), axis=2)
tmat = -1*(-ttmp > 0)+(ttmp > 0)+1
t = tmat[0]
# tmat = np.tile(t[np.newaxis, :], (ns, 1))        

# WADD Model For Each Question
xmat = np.tile(x[np.newaxis, :], (ns, nq, 1))
wtmp = np.sum(qccmat*xmat, axis=2)
wmat = -1*(-wtmp > 0)+(wtmp > 0)+1
w = wmat[0]

print(t)
print(w)


with pm.Model() as model2:
    phi = pm.Beta('phi', alpha=1, beta=1, testval=.01)
    
    zi = pm.Bernoulli('zi', p=phi, shape=ns,
                      testval=np.asarray([1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    zi_ = tt.reshape(tt.repeat(zi, nq), (ns, nq))
    
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])

    t2 = tt.switch(tt.eq(zi_, 1), tmat, wmat)
    yiq = pm.Bernoulli('yiq', p=gammat[t2], observed=y)

    trace2 = pm.sample(3e3, njobs=2, tune=1000)
    
pm.traceplot(trace2);


fig = plt.figure(figsize=(16, 4))
zitrc = trace2['zi'][1000:]
plt.bar(np.arange(ns)+1, 1-np.mean(zitrc, axis=0))
plt.yticks([0, 1], ('TTB', 'WADD'))
plt.xlabel('Subject')
plt.ylabel('Group')
plt.axis([0, 21, 0, 1])
plt.show()


# ## 18.3 Searching
# 
# 
# $$ s_i \sim \text{Uniform}((1,...,9),...,(9,...,1))$$
# $$ t_{iq} = \text{TTB}_{si}(\mathbf a_q,\mathbf b_q)$$
# $$ \gamma \sim \text{Uniform}(0.5,1)$$  
# $$ y_{iq} \sim
# \begin{cases}
# \text{Bernoulli}(\gamma) & \text{if $t_{iq} = a$} \\text{Bernoulli}(1- \gamma) & \text{if $t_{iq} = b$} \\text{Bernoulli}(0.5) & \text{otherwise}
# \end{cases}  $$
# 

with pm.Model() as model3:
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])
    
    v1 = pm.HalfNormal('v1', sd=1, shape=ns*nc)    
    s1 = pm.Deterministic('s1', tt.argsort(v1.reshape((ns, 1, nc)), axis=2))
    smat2 = tt.tile(s1, (1, nq, 1))  # s[1:nc] <- rank(v[1:nc])
    
    # TTB Model For Each Question
    ttmp = tt.sum(qccmat*tt.power(2, smat2), axis=2)
    tmat = -1*(-ttmp > 0)+(ttmp > 0)+1
    
    yiq = pm.Bernoulli('yiq', p=gammat[tmat], observed=y)


# It is important to notice here that, the sorting operation `s[1:nc] <- rank(v[1:nc])` is likely breaks the smooth property in geometry. Method such as NUTS and ADVI is likely return wrong estimation as the nasty geometry will lead the sampler to stuck in some local miminal.  
# For this reason, we use Metropolis to sample from this model.
# 

with model3:
    # trace3 = pm.sample(3e3, njobs=2, tune=1000)
    trace3 = pm.sample(1e5, step=pm.Metropolis(), njobs=2)

pm.traceplot(trace3, varnames=['gamma', 'v1']);


burnin = 50000
# v1trace = np.squeeze(trace3['v1'][burnin:])
# s1trace = np.argsort(v1trace, axis=2)
s1trace = np.squeeze(trace3[burnin:]['s1'])
for subj_id in [12, 13]:
    subj_s = np.squeeze(s1trace[:,subj_id-1,:])
    unique_ord = np.vstack({tuple(row) for row in subj_s})
    num_display = 10
    print('Subject %s' %(subj_id))
    print('There are %s search orders sampled in the posterior.'%(unique_ord.shape[0]))
    mass_ = []
    for s_ in unique_ord:
        mass_.append(np.mean(np.sum(subj_s == s_, axis=1) == len(s_)))
    mass_ = np.asarray(mass_)
    sortmass = np.argsort(mass_)[::-1]
    for i in sortmass[:num_display]:
        s_ = unique_ord[i]
        print('Order=(' + str(s_+1) + '), Estimated Mass=' + str(mass_[i]))


# The return order is not at all similar to the result in JAGS (as shown in the book on p.233). However, the cue 2 is searched before cue 6 in Subject 12 and vice versa in Subject 13, which is the same as in the book.
# 

# ## 18.4 Searching and stopping
# 
# 
# $$ \phi_{i} \sim \text{Uniform}(0,1)$$
# $$ z_{iq} \sim \text{Bernoulli}(\phi_{i})$$
# $$ s_i \sim \text{Uniform}((1,...,9),...,(9,...,1))$$
# $$ \gamma \sim \text{Uniform}(0.5,1)$$  
# $$ t_{iq} = 
# \begin{cases}
# \text{TTB}_{si}\,(\mathbf a_q,\mathbf b_q) & \text{if $z_{iq} = 1$} \\text{WADD}\,(\mathbf a_q,\mathbf b_q) & \text{if $z_{iq} = 0$} \\end{cases}  $$  
# $$ y_{iq} \sim
# \begin{cases}
# \text{Bernoulli}(\gamma) & \text{if $t_{iq} = a$} \\text{Bernoulli}(1- \gamma) & \text{if $t_{iq} = b$} \\text{Bernoulli}(0.5) & \text{otherwise}
# \end{cases}  $$
# 

with pm.Model() as model4:
    phi = pm.Beta('phi', alpha=1, beta=1, testval=.01)
    
    zi = pm.Bernoulli('zi', p=phi, shape=ns,
                      testval=np.asarray([1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    zi_ = tt.reshape(tt.repeat(zi, nq), (ns, nq))
    
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])

    v1 = pm.HalfNormal('v1', sd=1, shape=ns*nc)    
    s1 = pm.Deterministic('s1', tt.argsort(v1.reshape((ns, 1, nc)), axis=2))
    smat2 = tt.tile(s1, (1, nq, 1))  # s[1:nc] <- rank(v[1:nc])
    
    # TTB Model For Each Question
    ttmp = tt.sum(qccmat*tt.power(2, smat2), axis=2)
    tmat = -1*(-ttmp > 0) + (ttmp > 0) + 1
            
    t2 = tt.switch(tt.eq(zi_, 1), tmat, wmat)
    yiq = pm.Bernoulli('yiq', p=gammat[t2], observed=y)
    
    trace4 = pm.sample(1e5, step=pm.Metropolis())
    
burnin=50000
pm.traceplot(trace4[burnin:], varnames=['phi', 'gamma']);


ppc = pm.sample_ppc(trace4[burnin:], samples=100, model=model4)
yiqpred = np.asarray(ppc['yiq'])
fig = plt.figure(figsize=(16, 8))
x1 = np.repeat(np.arange(ns)+1, nq).reshape(ns, -1).flatten()
y1 = np.repeat(np.arange(nq)+1, ns).reshape(nq, -1).T.flatten()

plt.scatter(y1, x1, s=np.mean(yiqpred, axis=0)*200, c='w')
plt.scatter(y1[y.flatten() == 1], x1[y.flatten() == 1], marker='x', c='r')
plt.plot(np.ones(100)*24.5, np.linspace(0, 21, 100), '--', lw=1.5, c='k')
plt.axis([0, 31, 0, 21])
plt.show()


import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('qtconsole', '--colors=linux')
plt.style.use('ggplot')

from matplotlib import gridspec
from theano import tensor as tt
from scipy import stats


# # Chapter 17 - The GCM model of categorization
# 
# ## 17.1 The GCM model
# The Generalized Context Model (GCM: Nosofsky, 1984, 1986) is an influential and empirically successful model of categorization. It is intended to explain how people make categorization decisions in a task where stimuli are presented, one at a time, over a sequence of trials, and must be classified into one of a small number of categories (usually two) based on corrective feedback.
# The GCM assumes that stimuli are stored as exemplars, using their values along underlying stimulus dimensions, which correspond to points in a multidimensional psychological space. The GCM then assumes people make similarity comparisons between the current stimulus and the exemplars, and base their decision on the overall similarities to each category.
# 
# $$ c \sim \text{Uniform}(0,5) $$
# $$ w \sim \text{Uniform}(0,1) $$  
# $$ b = \frac{1}{2}$$  
# $$ d_{ij}^m = \lvert p_{im} - p_{jm}\rvert $$  
# $$ s_{ij} = \text{exp}[-c\,(w d^1_{ij}+(1-w)d^2_{ij})] $$  
# $$ r_i = \frac{b \sum_{j}a_{j}s_{ij}}{b \sum_{j}a_{j}s_{ij}\,+\,(1-b) \sum_{j}(1-a_{j})s_{ij}}$$  
# $$ y_i \sim \text{Binomial}(r_i,t)$$  
# 

import scipy.io as sio
matdata = sio.loadmat('data/KruschkeData.mat')

nstim = 8
nsubj = 40
t = nstim*nsubj
a = matdata['a'][0]
y = matdata['y'][:,0]

d1 = matdata['d1']
d2 = matdata['d2']
x = matdata['x']


a1 = np.repeat(2-a, nstim).reshape(nstim, nstim).T

with pm.Model() as model1:
    c = pm.Uniform('c', lower=0, upper=5)
    w = pm.Uniform('w', lower=0, upper=1)
    b = .5
    sij = tt.exp(-c*(w*d1+(1-w)*d2))
    
    sum_ajsij = tt.sum(a1*sij, axis=1)
    sum_majsij = tt.sum((1-a1)*sij, axis=1)
    
    ri = pm.Deterministic('ri', (b*sum_ajsij)/(b*sum_ajsij+(1-b)*sum_majsij))
    yi = pm.Binomial('yi', p=ri, n=t, observed=y)
       
    trace1=pm.sample(3e3, njobs=2)

pm.traceplot(trace1, varnames=['c', 'w']);


# Fig. 17.3
fig = plt.figure(figsize=(10, 5))
ctr = trace1['c']
wtr = trace1['w']
plt.scatter(ctr, wtr, alpha=.05)
plt.xlabel('Generalization')
plt.ylabel('Attention Weight')
plt.axis((0, 5, 0, 1))
plt.show()


# Fig. 17.4
ppc = pm.sample_ppc(trace1, samples=500, model=model1)

fig = plt.figure(figsize=(10, 5))
yipred = ppc['yi']
ax = sns.violinplot(data=yipred)
plt.plot(np.float32(x).T*40, color='gray', alpha=.1)
plt.plot(np.mean(np.float32(x).T*40, axis=1), color='r', alpha=1)
plt.xticks(np.arange(8), ''.join(map(str, np.arange(1, 9))))
plt.yticks([0, t], ('B', 'A'))
plt.xlabel('Stimulus')
plt.ylabel('Category Decision')
plt.show()


# ## 17.2 Individual differences in the GCM
# 
# 
# $$ c_k \sim \text{Uniform}(0,5) $$
# $$ w_k \sim \text{Uniform}(0,1) $$  
# $$ b_k = \frac{1}{2}$$  
# $$ d_{ij}^m = \lvert p_{im} - p_{jm}\rvert $$  
# $$ s_{ijk} = \text{exp}[-c_k\,(w_k d^1_{ij}+(1-w_k)d^2_{ij})] $$  
# $$ r_{ik} = \frac{b_k \sum_{j}a_{j}s_{ijk}}{b_k \sum_{j}a_{j}s_{ijk}\,+\,(1-b_k) \sum_{j}(1-a_{j})s_{ijk}}$$  
# $$ y_{ik} \sim \text{Binomial}(r_{ik},n)$$  
# 

# Fig. 17.5
sns.set(style='ticks')

# Create a dataset with many short random walks
x2 = np.float32(x)
subjvect=[]
stimvect=[]
respvect=[]
for i in range(nstim):
    for j in range(nsubj):
        subjvect.append(j+1)
        stimvect.append(i+1)
        respvect.append(x2[j, i])
df = pd.DataFrame(np.c_[respvect, stimvect, subjvect],
                  columns=['Resp', 'Stim', 'Sbj'])

grid = sns.FacetGrid(df, col='Sbj', hue='Sbj', col_wrap=8, size=1.5)
grid.map(plt.plot, 'Stim', 'Resp', marker='o', ms=4)
# Adjust the tick positions and labels
grid.set(xticks=np.arange(1, 9), yticks=[0, 8],
         xlim=(0, 9), ylim=(-1, 9))
# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)


a1 = np.tile(2-a,[nstim,1])[:,:,np.newaxis]
y2 = x2.transpose()
d1_t = np.tile(d1[:, :, np.newaxis], [1, 1, nsubj])
d2_t = np.tile(d2[:, :, np.newaxis], [1, 1, nsubj])

with pm.Model() as model2:
    c = pm.Uniform('c', lower=0, upper=5, shape=(1, 1, nsubj))
    w = pm.Uniform('w', lower=0, upper=1, shape=(1, 1, nsubj))
    b = .5

    sij = tt.exp(-c*(w*d1_t+(1-w)*d2_t))
    
    sum_ajsij = tt.sum(a1*sij, axis=1)
    sum_majsij = tt.sum((1-a1)*sij, axis=1)
    
    ri = pm.Deterministic('ri', (b*sum_ajsij)/(b*sum_ajsij+(1-b)*sum_majsij))
    
    yi = pm.Binomial('yi', p=ri, n=nstim, observed=y2)
    
    trace2 = pm.sample(3e3, njobs=2)
    
plt.style.use('ggplot')
pm.traceplot(trace2, varnames=['c', 'w']);


# Fig. 17.7
burnin=2500
fig = plt.figure(figsize=(10, 5))
ctr=np.squeeze(trace2['c'][burnin:])
wtr=np.squeeze(trace2['w'][burnin:])
from matplotlib import cm as cm
colors = cm.rainbow(np.linspace(0, 1, nsubj))
for i in range(nsubj):
    plt.scatter(ctr[:, i], wtr[:, i], color=colors[i], alpha=.025)
    
plt.scatter(np.mean(ctr, axis=0), np.mean(wtr, axis=0), s=50, color='black', alpha=1)
plt.xlabel('Generalization')
plt.ylabel('Attention Weight')
plt.axis((0, 5, 0, 1))
plt.show()


# ## 17.3 Latent groups in the GCM
# 
# 
# $$ \mu_{1}^w,\delta \sim \text{Uniform}(0,1)$$
# $$ \mu_{2}^w = \text{min}(1,\mu_{1}^w+\delta) $$
# $$ \sigma^w \sim \text{Uniform}(0.01,1)$$
# $$ \mu^c \sim \text{Uniform}(0,5)$$
# $$ \sigma^c \sim \text{Uniform}(0.01,3)$$
# $$ \phi^c,\phi^g \sim \text{Uniform}(0,1)$$
# $$ z_{k}^c \sim \text{Bernoulli}(\phi^c)$$
# $$ z_{k}^g \sim \text{Bernoulli}(\phi^g)$$
# 
# $$ c_k \sim \text{Gaussian}(\mu^c,\frac{1}{(\sigma^c)^2})_{\mathcal I(0,5)} $$
# $$ w_k \sim
# \begin{cases}
# \text{Gaussian}(\mu_{1}^w,\frac{1}{(\sigma^w)^2})_{\mathcal I(0,1)}  & \text{if $z_{k}^c = 0,z_{k}^g = 0$} \\text{Gaussian}(\mu_{2}^w,\frac{1}{(\sigma^w)^2})_{\mathcal I(0,1)}  & \text{if $z_{k}^c = 0,z_{k}^g = 1$}
# \end{cases}  $$
# 
# $$ b_k = \frac{1}{2}$$  
# $$ d_{ij}^m = \lvert p_{im} - p_{jm}\rvert $$  
# $$ s_{ijk} = \text{exp}[-c_k\,(w_k d^1_{ij}+(1-w_k)d^2_{ij})] $$  
# 
# $$ r_{ik} =
# \begin{cases}
# \frac{b_k \sum_{j}a_{j}s_{ijk}}{b_k \sum_{j}a_{j}s_{ijk}\,+\,(1-b_k) \sum_{j}(1-a_{j})s_{ijk}}  & \text{if $z_{k}^c = 0$} \\frac{1}{2}  & \text{if $z_{k}^c = 1$}
# \end{cases}  $$
# 
# 
# $$ y_{ik} \sim \text{Binomial}(r_{ik},n)$$  
# 

# BoundNormal1 = pm.Bound(pm.Normal, lower=0., upper=5.)
# BoundNormal2 = pm.Bound(pm.Normal, lower=0., upper=1.)

with pm.Model() as model3:
    mu1w = pm.Uniform('mu1w', lower=0, upper=1, testval=.05)
    delta = pm.Uniform('delta', lower=0, upper=1, testval=.75)
    mu2w = pm.Deterministic('mu2w', tt.clip(mu1w+delta, 0, 1))
    
    sigmaw = pm.Uniform('sigmaw', lower=.01, upper=1, testval=.05)
    muc = pm.Uniform('muc', lower=0, upper=5, testval=1.4)
    sigmac = pm.Uniform('sigmac', lower=.01, upper=3, testval=.45)
    
    phic = pm.Uniform('phic', lower=0, upper=1, testval=.1)
    phig = pm.Uniform('phig', lower=0, upper=1, testval=.8)

    zck = pm.Bernoulli('zck', p=phic, shape=nsubj)
    zcg = pm.Bernoulli('zcg', p=phig, shape=nsubj)
#     zck_ = pm.theanof.tt_rng().uniform(size=(1, nsubj))
#     zck = pm.Deterministic('zck', tt.repeat(tt.lt(zck_, phic), nstim, axis=0))
#     zcg_ = pm.theanof.tt_rng().uniform(size=(nsubj,))
#     zcg = pm.Deterministic('zcg', tt.lt(zcg_, phig))
    b = .5
    
    # c = BoundNormal1('c', mu=muc, sd=sigmac, shape=(1, 1, nsubj))
    c = tt.clip(pm.Normal('c', mu=muc, sd=sigmac, shape=(1, 1, nsubj)), 0, 5)
    muw = pm.Deterministic('muw', tt.switch(tt.eq(zcg, 0), mu1w, mu2w))
    # w = BoundNormal2('w', mu=muw.all(), sd=sigmaw, shape=(1, 1, nsubj))
    w = tt.clip(pm.Normal('w', mu=muw, sd=sigmaw, shape=(1, 1, nsubj)), 0, 1)
    
    sij = tt.exp(-c*(w*d1_t+(1-w)*d2_t))
    
    sum_ajsij = tt.sum(a1*sij, axis=1)
    sum_majsij = tt.sum((1-a1)*sij, axis=1)
    
    ri1 = pm.Deterministic('ri1', (b*sum_ajsij)/(b*sum_ajsij+(1-b)*sum_majsij))
    ri2 = tt.constant(np.ones((nstim, nsubj))*.5)
    ri = pm.Deterministic('ri', tt.squeeze(tt.switch(tt.eq(zck, 0), ri1, ri2)))
    
    yi = pm.Binomial('yi', p=ri, n=nstim, observed=y2)


import theano
with model3:
    trace3 = pm.sample(3e3, njobs=2)

#     # Using ADVI
#     s = theano.shared(pm.floatX(1))
#     inference = pm.ADVI(cost_part_grad_scale=s)
#     # ADVI has nearly converged
#     inference.fit(n=20000)
#     # It is time to set `s` to zero
#     s.set_value(0)
#     approx = inference.fit(n=10000)
#     trace3 = approx.sample_vp(3000) 
#     elbos1 = -inference.hist

pm.traceplot(trace3, varnames=['mu1w', 'delta', 'sigmaw', 
                               'muc', 'sigmac', 
                               'phic', 'phig']);


# Fig. 17.9
burnin = 1000
zc = np.squeeze(trace3['zck'][burnin:])
zg = np.squeeze(trace3['zcg'][burnin:])
z = (zc == 0) * (zg+1) + 3 * (zc == 1)
z1 = np.zeros((nsubj, 3))
for i in range(nsubj):
    sbjid = z[:, i]
    z1[i] = [np.sum(sbjid == 1)/len(sbjid), np.sum(sbjid == 2)/len(sbjid), np.sum(sbjid == 3)/len(sbjid)]
ord1 = []
for i in range(3):
    ordtmp = np.argsort(z1[:, i])
    ordtmp = ordtmp[z1[ordtmp, i] > .5]
    ord1.extend(ordtmp)
ord1 = np.asarray(ord1)
fig = plt.figure(figsize=(16, 8))

xpl = np.arange(40)[::-1]
plt.plot(xpl, z1[ord1, 0], marker='s', markersize=12)
plt.plot(xpl, z1[ord1, 1], marker='o', markersize=12)
plt.plot(xpl, z1[ord1, 2], marker='^', markersize=12)
plt.legend(['Attend Height', 'Attend Position', 'Contaminant'])
plt.ylim([-.05, 1.05])
plt.xlim([-1, 40])
plt.xlabel('Subject')
plt.ylabel('Membership Probability')
plt.show()


# Fig. 17.10
nppc_sample = 500
ppc = pm.sample_ppc(trace3[burnin:], samples=nppc_sample, model=model3)

fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 3)
for ip in range(3):
    ax = plt.subplot(gs[ip])
    idx = np.nonzero(z1[:, ip] > .5)[0]
    
    yipred = np.zeros((nppc_sample*len(idx), 8))
    for i, i2 in enumerate(idx):
        yipred1 = ppc['yi'][:, :, i2]
        yipred[0+i*nppc_sample:nppc_sample+i*nppc_sample, :] = yipred1
    
    ax = sns.violinplot(data=yipred)
    plt.plot(y2[:, idx], color='gray', alpha=.4)
    plt.plot(np.mean(y2[:, idx], axis=1), color='red', alpha=1)
    plt.xticks(np.arange(8), ''.join(map(str, np.arange(1, 9))))
    plt.yticks([0, nstim], ('B', 'A'))
    plt.xlabel('Stimulus')
    plt.ylabel('Category Decision')
plt.show()


import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('qtconsole', '--colors=linux')
plt.style.use('ggplot')

from matplotlib import gridspec
from theano import tensor as tt
from scipy import stats


# # Chapter 16 - The BART model of risk taking
# 
# ## 16.1 The BART model
# Balloon Analogue Risk Task (BART: Lejuez et al., 2002): Every trial in this task starts by showing a balloon representing a small monetary value. The subject can then either transfer the money to a virtual bank account, or choose to pump, which adds a small amount of air to the balloon, and increases its value. There is some probability, however, that pumping the balloon will cause it to burst, causing all the money to be lost. A trial finishes when either the subject has transferred the money, or the balloon has burst.
# 
# $$ \gamma^{+} \sim \text{Uniform}(0,10) $$
# $$ \beta \sim \text{Uniform}(0,10) $$
# $$ \omega = -\gamma^{+} \,/\,\text{log}(1-p) $$
# $$ \theta_{jk} = \frac{1} {1+e^{\beta(k-\omega)}} $$
# $$ d_{jk} \sim \text{Bernoulli}(\theta_{jk}) $$
# 

p = .15  # (Belief of) bursting probability
ntrials = 90   # Number of trials for the BART

Data = pd.read_csv('data/GeorgeSober.txt', sep='\t')
# Data.head()
cash = np.asarray(Data['cash']!=0, dtype=int)
npumps = np.asarray(Data['pumps'], dtype=int)

options = cash + npumps

d = np.full([ntrials,30], np.nan)
k = np.full([ntrials,30], np.nan)
# response vector
for j, ipumps in enumerate(npumps):
    inds = np.arange(options[j],dtype=int)
    k[j,inds] = inds+1
    if ipumps > 0:
        d[j,0:ipumps] = 0
    if cash[j] == 1:
        d[j,ipumps] = 1
        
indexmask = np.isfinite(d)
d = d[indexmask]
k = k[indexmask]


with pm.Model():
    gammap = pm.Uniform('gammap', lower=0, upper=10, testval=1.2)
    beta = pm.Uniform('beta', lower=0, upper=10, testval=.5)
    omega = pm.Deterministic('omega', -gammap/np.log(1-p))
    
    thetajk = 1 - pm.math.invlogit(- beta * (k - omega))
    
    djk = pm.Bernoulli('djk', p=thetajk, observed=d)
    
    trace = pm.sample(3e3, njobs=2)
    
pm.traceplot(trace, varnames=['gammap', 'beta']);


from scipy.stats.kde import gaussian_kde
burnin=2000
gammaplus = trace['gammap'][burnin:]
beta = trace['beta'][burnin:]

fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3)

ax0 = plt.subplot(gs[0])
ax0.hist(npumps, bins=range(1, 9), rwidth=.8, align='left')
plt.xlabel('Number of Pumps', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

ax1 = plt.subplot(gs[1])
my_pdf1 = gaussian_kde(gammaplus)
x1=np.linspace(.5, 1, 200)
ax1.plot(x1, my_pdf1(x1), 'k', lw=2.5, alpha=0.6) # distribution function
plt.xlim((.5, 1))
plt.xlabel(r'$\gamma^+$', fontsize=15)
plt.ylabel('Posterior Density', fontsize=12)

ax2 = plt.subplot(gs[2])
my_pdf2 = gaussian_kde(beta)
x2=np.linspace(0.3, 1.3, 200)
ax2.plot(x2, my_pdf2(x2), 'k', lw=2.5, alpha=0.6,) # distribution function
plt.xlim((0.3, 1.3))
plt.xlabel(r'$\beta$', fontsize=15)
plt.ylabel('Posterior Density', fontsize=12);


# ## 16.2 A hierarchical extension of the BART model
#   
#   
# $$ \mu_{\gamma^{+}} \sim \text{Uniform}(0,10) $$
# $$ \sigma_{\gamma^{+}} \sim \text{Uniform}(0,10) $$
# $$ \mu_{\beta} \sim \text{Uniform}(0,10) $$
# $$ \sigma_{\beta} \sim \text{Uniform}(0,10) $$
# $$ \gamma^{+}_i \sim \text{Gaussian}(\mu_{\gamma^{+}}, 1/\sigma_{\gamma^{+}}^2) $$
# $$ \beta_i \sim \text{Gaussian}(\mu_{\beta}, 1/\sigma_{\beta}^2) $$
# $$ \omega_i = -\gamma^{+}_i \,/\,\text{log}(1-p) $$
# $$ \theta_{ijk} = \frac{1} {1+e^{\beta_i(k-\omega_i)}} $$
# $$ d_{ijk} \sim \text{Bernoulli}(\theta_{ijk}) $$
# 

p = .15  # (Belief of) bursting probability
ntrials = 90   # Number of trials for the BART
Ncond = 3

dall = np.full([Ncond,ntrials,30], np.nan)
options = np.zeros((Ncond,ntrials))
kall = np.full([Ncond,ntrials,30], np.nan)
npumps_ = np.zeros((Ncond,ntrials))

for icondi in range(Ncond):
    if icondi == 0:
        Data = pd.read_csv('data/GeorgeSober.txt',sep='\t')
    elif icondi == 1:
        Data = pd.read_csv('data/GeorgeTipsy.txt',sep='\t')
    elif icondi == 2:
        Data = pd.read_csv('data/GeorgeDrunk.txt',sep='\t')
    # Data.head()
    cash = np.asarray(Data['cash']!=0, dtype=int)
    npumps = np.asarray(Data['pumps'], dtype=int)
    npumps_[icondi,:] = npumps
    options[icondi,:] = cash + npumps
    # response vector
    for j, ipumps in enumerate(npumps):
        inds = np.arange(options[icondi,j],dtype=int)
        kall[icondi,j,inds] = inds+1
        if ipumps > 0:
            dall[icondi,j,0:ipumps] = 0
        if cash[j] == 1:
            dall[icondi,j,ipumps] = 1
            
indexmask = np.isfinite(dall)
dij = dall[indexmask]
kij = kall[indexmask]
condall = np.tile(np.arange(Ncond,dtype=int),(30,ntrials,1))
condall = np.swapaxes(condall,0,2)
cij = condall[indexmask]


with pm.Model() as model2:
    mu_g = pm.Uniform('mu_g', lower=0, upper=10)
    sigma_g = pm.Uniform('sigma_g', lower=0, upper=10)
    mu_b = pm.Uniform('mu_b', lower=0, upper=10)
    sigma_b = pm.Uniform('sigma_b', lower=0, upper=10)
    
    gammap = pm.Normal('gammap', mu=mu_g, sd=sigma_g, shape=Ncond)
    beta = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=Ncond)
    
    omega = -gammap[cij]/np.log(1-p)
    thetajk = 1 - pm.math.invlogit(- beta[cij] * (kij - omega))
    
    djk = pm.Bernoulli("djk", p=thetajk, observed=dij)
    
    approx = pm.fit(n=100000, method='advi',
                    obj_optimizer=pm.adagrad_window
                    )  # type: pm.MeanField
    start = approx.sample(draws=2, include_transformed=True)
    trace2 = pm.sample(3e3, njobs=2, init='adapt_diag', start=list(start))
    
pm.traceplot(trace2, varnames=['gammap', 'beta']);


burnin=1000
gammaplus = trace2['gammap'][burnin:]
beta = trace2['beta'][burnin:]
ylabels = ['Sober', 'Tipsy', 'Drunk']

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(3, 3)
for ic in range(Ncond):
    ax0 = plt.subplot(gs[0+ic*3])
    ax0.hist(npumps_[ic], bins=range(1, 10), rwidth=.8, align='left')
    plt.xlabel('Number of Pumps', fontsize=12)
    plt.ylabel(ylabels[ic], fontsize=12)

    ax1 = plt.subplot(gs[1+ic*3])
    my_pdf1 = gaussian_kde(gammaplus[:, ic])
    x1=np.linspace(.5, 1.8, 200)
    ax1.plot(x1, my_pdf1(x1), 'k', lw=2.5, alpha=0.6) # distribution function
    plt.xlim((.5, 1.8))
    plt.xlabel(r'$\gamma^+$', fontsize=15)
    plt.ylabel('Posterior Density', fontsize=12)

    ax2 = plt.subplot(gs[2+ic*3])
    my_pdf2 = gaussian_kde(beta[:, ic])
    x2=np.linspace(0.1, 1.5, 200)
    ax2.plot(x2, my_pdf2(x2), 'k', lw=2.5, alpha=0.6) # distribution function
    plt.xlim((0.1, 1.5))
    plt.xlabel(r'$\beta$', fontsize=15)
    plt.ylabel('Posterior Density', fontsize=12);


import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('qtconsole', '--colors=linux')
plt.style.use('ggplot')

from matplotlib import gridspec
from theano import tensor as tt
from scipy import stats


# # Chapter 15 - The SIMPLE model of memory
# 
# ## 15.1 The SIMPLE model
# Brown, Neath, and Chater (2007) proposed the SIMPLE (Scale-Invariant Memory, Perception, and LEarning) model, which, among various applications, has been applied to the basic memory phenomenon of free recall.
# 
# $$ c_x \sim \text{Uniform}(0,100)$$
# $$ s_x \sim \text{Uniform}(0,100)$$
# $$ t_x \sim \text{Uniform}(0,1) $$  
# $$ \eta_{ijx} = \text{exp}(-\,c_x \,\lvert \text{log} T_{ix}\,-\,\text{log} T_{jx}\rvert)$$  
# $$ d_{ijx} = \frac{\eta_{ijx}} {\sum_{k}\eta_{ikx}}$$  
# $$ r_{ijx} = \frac{1} {1\,+\,\text{exp}(-\,s_{x}(d_{ijx}\,-\,t_{x}))}$$  
# $$ \theta_{ix} = \text{min}(1,\sum_{k}r_{ikx})$$
# $$ y_{ix} \sim \text{Binomial}(\theta_{ix},\eta_{x})$$
# 
# Model correction on SIMPLE model could be done by replacing $\theta_{ix} = \text{min}(1,\sum_{k}r_{ikx})$ with $\theta_{ix} = 1\,-\,\prod_{k}(1-r_{ikx})$ (see Box 15.2 on page 200)
# 

y = pd.read_csv('data/k_M.txt', ',', header=None)
n = np.array([1440, 1280, 1520, 1520, 1200, 1280])
listlength = np.array([10, 15, 20, 20, 30, 40])
lagall = np.array([2, 2, 2, 1, 1, 1])
offset = np.array([15, 20, 25, 10, 15, 20])
dsets = 6
m = np.zeros(np.shape(y))
for i in range(dsets):
    m[i, 0:listlength[i]]=offset[i]+np.arange((listlength[i])*lagall[i], 0, -lagall[i])
pc = pd.read_csv('data/pc_M.txt', ',', header=None)
pcmat = np.asarray(pc).T


ymat = np.asarray(y)
nitem = m.shape[1]
m2 = m
m2[m2==0] = 1e-5 # to avoid NaN in ADVI
nmat = np.repeat(n[:,np.newaxis], nitem, axis=1)
mmat1 = np.repeat(m2[:,:,np.newaxis],nitem,axis=2)
mmat2 = np.transpose(mmat1, (0, 2, 1))
mask = np.where(ymat>0)

with pm.Model() as simple1:
    cx = pm.Uniform('cx', lower=0, upper=100, shape=dsets, testval=np.ones(dsets)*20)
    sx = pm.Uniform('sx', lower=0, upper=100, shape=dsets)
    tx = pm.Uniform('tx', lower=0, upper=1, shape=dsets)
    
    # Similarities
    eta = tt.exp(-cx[:,np.newaxis,np.newaxis]*abs(tt.log(mmat1)-tt.log(mmat2)))
    etasum = tt.reshape(tt.repeat(tt.sum(eta, axis=2), nitem), (dsets, nitem, nitem))
    
    # Discriminabilities
    disc = eta/etasum
    
    # Response Probabilities
    resp = 1/(1+tt.exp(-sx[:, np.newaxis, np.newaxis]*(disc-tx[:, np.newaxis, np.newaxis])))
    
    # Free Recall Overall Response Probability
    # theta = tt.clip(tt.sum(resp, axis=2), 0., .999)
    theta=1-tt.prod(1-resp, axis=2)
    
    yobs = pm.Binomial('yobs', p=theta[mask], n=nmat[mask], observed=ymat[mask])
    trace = pm.sample(3e3, njobs=2, init='advi+adapt_diag')
    
pm.traceplot(trace, varnames=['cx', 'sx', 'tx']);


ymat = np.asarray(y).T
mmat = m.T

with pm.Model() as simple1:
    cx = pm.Uniform('cx', lower=0, upper=100, shape=dsets, testval=np.ones(dsets)*20)
    sx = pm.Uniform('sx', lower=0, upper=100, shape=dsets)
    tx = pm.Uniform('tx', lower=0, upper=1, shape=dsets)
    
    yobs=[]
    for x in range(dsets):
        sz = listlength[x]
        # Similarities
        m1 = np.array([mmat[0:sz, x], ]*sz).T
        m2 = np.array([mmat[0:sz, x], ]*sz)

        eta = tt.exp(-cx[x]*abs(tt.log(m1)-tt.log(m2)))
        etasum = tt.reshape(tt.repeat(tt.sum(eta, axis=1), sz), (sz, sz))
        # Discriminabilities
        disc = eta/etasum
        # Response Probabilities
        resp = 1/(1+tt.exp(-sx[x]*(disc-tx[x])))
        # Free Recall Overall Response Probability
        theta = tt.clip(tt.sum(resp, axis=1), 0, .999)
        # theta=1-tt.prod(1-resp,axis=1)
        
        yobs.append([pm.Binomial("yobs_%x"%x, p=theta, n=n[x], observed=ymat[0:sz, x])])
        
    trace = pm.sample(3e3, njobs=2, init='advi+adapt_diag')
    
pm.traceplot(trace, varnames=['cx', 'sx', 'tx']);


# The above two model does the same thing, but surprising using list compression is actually faster.
# 

ymat = np.asarray(y).T
mmat = m.T

fig = plt.figure(figsize=(16, 8))
fig.text(0.5, -0.02, 'Serial Position', ha='center', fontsize=20)
fig.text(-0.02, 0.5, 'Probability Correct', va='center', rotation='vertical', fontsize=20)

burnin=1000
totalsamp=3e3
ppcsamples=200

gs = gridspec.GridSpec(2, 3)
for ip in range(dsets):
    ax = plt.subplot(gs[ip])
    ay=ymat[:, ip]/n[ip] # pcmat[:,ip]

    cxt=trace['cx'][:, ip]
    sxt=trace['sx'][:, ip]
    txt=trace['tx'][:, ip]
    
    sz = listlength[ip]
    # Similarities
    m1 = np.array([mmat[0:sz, ip], ]*sz).T
    m2 = np.array([mmat[0:sz, ip], ]*sz)
    for ips in np.random.randint(burnin, totalsamp, ppcsamples):
        
        eta=np.exp(-cxt[ips]*abs(np.log(m1)-np.log(m2)))
        etasum=np.reshape(np.repeat(np.sum(eta, axis=1), sz), (sz, sz))
        # Discriminabilities
        disc = eta/etasum
        # Response Probabilities
        resp = 1/(1+np.exp(-sxt[ips]*(disc-txt[ips])))
        # Free Recall Overall Response Probability
        theta = np.minimum(np.sum(resp, axis=1), .999)
        ax.plot(theta, alpha=.05)
    
    ax.plot(ay[ay!=0], marker='o', alpha=.5)
    plt.axis((0, 41, 0, 1))
    plt.title(str(listlength[ip])+'-'+str(lagall[ip]))

plt.tight_layout()
plt.show()


# ## 15.2 A hierarchical extension of SIMPLE
# 
# 
# $$ c \sim \text{Uniform}(0,100)$$
# $$ s \sim \text{Uniform}(0,100)$$
# $$ a_{1} \sim \text{Uniform}(-1,0) $$
# $$ a_{2} \sim \text{Uniform}(0,1) $$
# $$ t_x = a_{1}W_x + a_{2} $$  
# $$ \eta_{ijx} = \text{exp}(-\,c_x \,\lvert \text{log} T_{ix}\,-\,\text{log} T_{jx}\rvert)$$  
# $$ d_{ijx} = \frac{\eta_{ijx}} {\sum_{k}\eta_{ikx}}$$  
# $$ r_{ijx} = \frac{1} {1\,+\,\text{exp}(-\,s_{x}(d_{ijx}\,-\,t_{x}))}$$  
# $$ \theta_{ix} = \text{min}(1,\sum_{k}r_{ikx})$$
# $$ y_{ix} \sim \text{Binomial}(\theta_{ix},\eta_{x})$$
# 
# Model correction on SIMPLE model could be done by replacing $\theta_{ix} = \text{min}(1,\sum_{k}r_{ikx})$ with $\theta_{ix} = 1\,-\,\prod_{k}(1-r_{ikx})$ (see Box 15.2 on page 200)
# 

ymat = np.asarray(y)
nitem = m.shape[1]
m2 = m
m2[m2==0] = 1e-5 # to avoid NaN in ADVI
nmat = np.repeat(n[:,np.newaxis], nitem, axis=1)
mmat1 = np.repeat(m2[:,:,np.newaxis],nitem,axis=2)
mmat2 = np.transpose(mmat1, (0, 2, 1))
mask = np.where(ymat>0)
W = listlength

with pm.Model() as simple2:
    cx = pm.Uniform('cx', lower=0, upper=100, testval=21)
    sx = pm.Uniform('sx', lower=0, upper=100, testval=10)
    a1 = pm.Uniform('a1', lower=-1, upper=0, testval=-.002)
    a2 = pm.Uniform('a2', lower=0, upper=1, testval=.64)
    tx = pm.Deterministic('tx', a1*W+a2)
    
    # Similarities
    eta = tt.exp(-cx*abs(tt.log(mmat1)-tt.log(mmat2)))
    etasum = tt.reshape(tt.repeat(tt.sum(eta, axis=2), nitem), (dsets, nitem, nitem))
    
    # Discriminabilities
    disc = eta/etasum
    
    # Response Probabilities
    resp = 1/(1+tt.exp(-sx*(disc-tx[:,np.newaxis,np.newaxis])))
    
    # Free Recall Overall Response Probability
    # theta = tt.clip(tt.sum(resp, axis=2), 0., .999)
    theta=1-tt.prod(1-resp, axis=2)
    
    yobs = pm.Binomial('yobs', p=theta[mask], n=nmat[mask], observed=ymat[mask])
    trace2_ = pm.sample(3e3, njobs=2, init='advi+adapt_diag')
    
pm.traceplot(trace2_, varnames=['cx', 'sx', 'tx']);


ymat = np.asarray(y).T
mmat = m.T
W = listlength

with pm.Model() as simple2:
    cx = pm.Uniform('cx', lower=0, upper=100, testval=21)
    sx = pm.Uniform('sx', lower=0, upper=100, testval=10)
    a1 = pm.Uniform('a1', lower=-1, upper=0, testval=-.002)
    a2 = pm.Uniform('a2', lower=0, upper=1, testval=.64)
    tx = pm.Deterministic('tx', a1*W+a2)
    
    yobs=[]
    for x in range(dsets):
        sz = listlength[x]
        # Similarities
        m1 = np.array([mmat[0:sz, x], ]*sz).T
        m2 = np.array([mmat[0:sz, x], ]*sz)

        eta=tt.exp(-cx*abs(tt.log(m1)-tt.log(m2)))
        etasum=tt.reshape(tt.repeat(tt.sum(eta, axis=1), sz), (sz, sz))
        # Discriminabilities
        disc = eta/etasum
        # Response Probabilities
        resp = 1/(1+tt.exp(-sx*(disc-tx[x])))
        # Free Recall Overall Response Probability
        theta = tt.clip(tt.sum(resp, axis=1), 0, .999)
        # theta=1-tt.prod(1-resp,axis=1)
        
        yobs.append([pm.Binomial('yobs_%x'%x, p=theta, n=n[x], observed=ymat[0:sz, x])])
        
    trace2 = pm.sample(3e3, njobs=2, init='advi+adapt_diag')
    
pm.traceplot(trace2, varnames=['cx', 'sx', 'tx']);


trace = trace2

ymat = np.asarray(y).T
mmat = m.T

fig = plt.figure(figsize=(16, 8))
fig.text(0.5, -0.02, 'Serial Position', ha='center', fontsize=20)
fig.text(-0.02, 0.5, 'Probability Correct', va='center', rotation='vertical', fontsize=20)

burnin=1000
totalsamp=3e3
ppcsamples=200

gs = gridspec.GridSpec(2, 3)
for ip in range(dsets):
    ax = plt.subplot(gs[ip])
    ay=ymat[:, ip]/n[ip] # pcmat[:,ip]

    cxt=trace['cx']
    sxt=trace['sx']
    txt=trace['tx'][:, ip]
    
    sz = listlength[ip]
    # Similarities
    m1 = np.array([mmat[0:sz, ip], ]*sz).T
    m2 = np.array([mmat[0:sz, ip], ]*sz)
    for ips in np.random.randint(burnin, totalsamp, ppcsamples):
        
        eta=np.exp(-cxt[ips]*abs(np.log(m1)-np.log(m2)))
        etasum=np.reshape(np.repeat(np.sum(eta, axis=1), sz), (sz, sz))
        # Discriminabilities
        disc = eta/etasum
        # Response Probabilities
        resp = 1/(1+np.exp(-sxt[ips]*(disc-txt[ips])))
        # Free Recall Overall Response Probability
        theta = np.minimum(np.sum(resp, axis=1), .999)
        ax.plot(theta, alpha=.05)
    
    ax.plot(ay[ay!=0], marker='o', alpha=.5)
    plt.axis((0, 41, 0, 1))
    plt.title(str(listlength[ip])+'-'+str(lagall[ip]))

plt.tight_layout();


fig = plt.figure(figsize=(16, 4))
ax1=plt.subplot(1,3,1)
pm.plot_posterior(trace=trace,varnames=['sx'],ax=ax1);
ax1.set_xlabel('Threshold Noise (s)')
ax1.set_xlim([9,11])
ax2=plt.subplot(1,3,2)
pm.plot_posterior(trace=trace,varnames=['cx'],ax=ax2);
ax2.set_xlabel('Threshold Noise (s)')
ax2.set_xlim([18,24]);


