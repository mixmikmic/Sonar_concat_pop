# Bernoulli trials are one of the simplest experimential setups: there are a number of iterations of some activity, where each iteration (or trial) may turn out to be a "success" or a "failure". From the data on T trials, we want to estimate the probability of "success".
# 
# Since it is such a simple case, it is a nice setup to use to describe some of Python's capabilities for estimating statistical models. Here I show estimation from the Bayesian perspective, via Metropolis-Hastings MCMC methods.
# 
# In [another post](./bernoulli_trials_classical.html) I show estimation of the problem in Python using the classical / frequentist approach.
# 
# <!-- TEASER_END -->
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sympy as sp
import pymc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.special import gamma

from sympy.interactive import printing
printing.init_printing()


# ### Setup
# 
# Let $y$ be a Bernoulli trial:
# 
# $y \sim \text{Bernoulli}(\theta) = \text{Binomial}(1, \theta)$
# 
# The probability density function, or marginal likelihood function, is:
# 
# $$p(y|\theta) = \theta^{y} (1-\theta)^{1-y} = \begin{cases}
# \theta & y = 1 \1 - \theta & y = 0
# \end{cases}$$
# 

# Simulate data
np.random.seed(123)

nobs = 100
theta = 0.3
Y = np.random.binomial(1, theta, nobs)


# Plot the data
fig = plt.figure(figsize=(7,3))
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1]) 
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.plot(range(nobs), Y, 'x')
ax2.hist(-Y, bins=2)

ax1.yaxis.set(ticks=(0,1), ticklabels=('Failure', 'Success'))
ax2.xaxis.set(ticks=(-1,0), ticklabels=('Success', 'Failure'));

ax1.set(title=r'Bernoulli Trial Outcomes $(\theta=0.3)$', xlabel='Trial', ylim=(-0.2, 1.2))
ax2.set(ylabel='Frequency')

fig.tight_layout()


# ### Bayesian Estimation
# 
# Using Bayes' rule:
# 
# $$p(\theta|Y) = \frac{p(Y|\theta) p(\theta)}{p(Y)}$$
# 
# To perform Bayesian estimation, we need to construct the __posterior__ $p(\theta|Y)$ given:
# 
# - the (joint) __likelihood__ $P(Y|\theta)$
# - the __prior__ $p(\theta)$
# - the __marginal probability density function__ $P(Y)$
# 
# to perform the estimation, we need to specify the functional forms of the likelihood and the prior. The marginal pdf of $Y$ is a constant with respect to $\theta$, so it does not need to specified for our purposes.
# 

# ### Likelihood function
# 
# Consider a sample of $T$ draws from the random variable $y$. The joint likelihood of observing any specific sample $Y = (y_1, ..., y_T)'$ is given by:
# 
# $$
# \begin{align}
# p(Y|\theta) & = \prod_{i=1}^T \theta^{y_i} (1-\theta)^{1-y_i} \& = \theta^{s} (1 - \theta)^{T-s}
# \end{align}
# $$
# 
# where $s = \sum_i y_i$ is the number of observed "successes", and $T-s$ is the number of observed "failures".
# 

t, T, s = sp.symbols('theta, T, s')

# Create the function symbolically
likelihood = (t**s)*(1-t)**(T-s)

# Convert it to a Numpy-callable function
_likelihood = sp.lambdify((t,T,s), likelihood, modules='numpy')


# ### Prior
# 
# Since $\theta$ is a probability value, our prior must respect $\theta \in (0,1)$. We will use the (conjugate) Beta prior:
# 
# $\theta \sim \text{Beta}(\alpha_1, \alpha_2)$
# 
# so that $(\alpha_1, \alpha_2)$ are the model's hyperparameters. Then the prior is specified as:
# 
# $$
# p(\theta;\alpha_1,\alpha_2) = \frac{1}{B(\alpha_1, \alpha_2)} \theta^{\alpha_1-1} (1 - \theta)^{\alpha_2 - 1}
# $$
# 
# where $B(\alpha_1, \alpha_2)$ is the Beta function. Note that to have a fully specified prior, we need to also specify the hyperparameters.
# 

# For alpha_1 = alpha_2 = 1, the Beta distribution
# degenerates to a uniform distribution
a1 = 1
a2 = 1

# Prior Mean
prior_mean = a1 / (a1 + a2)
print 'Prior mean:', prior_mean

# Plot the prior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g');

# Cleanup
ax.set(title='Prior Distribution', ylim=(0,12))
ax.legend(['Prior']);


# ### Posterior
# 
# #### Analytically
# 
# Given the prior and the likelihood function, we can try to find the kernel of the posterior analytically. In this case, it will be possible:
# 
# $$
# \begin{align}
# p(\theta|Y;\alpha_1,\alpha_2) & = \frac{P(Y|\theta) P(\theta)}{P(Y)} \& \propto P(Y|\theta) P(\theta) \& = \theta^s (1-\theta)^{T-s} \frac{1}{B(\alpha_1, \alpha_2)} \theta^{\alpha_1-1} (1 - \theta)^{\alpha_2 - 1} \& \propto \theta^{s+\alpha_1-1} (1 - \theta)^{T-s+\alpha_2 - 1} \\end{align}
# $$
# 
# The last line is identifiable as the kernel of a beta distribution with parameters $(\hat \alpha_1, \hat \alpha_2) = (s+\alpha_1, T-s+\alpha_2)$
# 
# Thus the posterior is given by
# 
# $$
# P(\theta|Y;\alpha_1,\alpha_2) = \frac{1}{B(\hat \alpha_1, \hat \alpha_2)} \theta^{\hat \alpha_1 - 1} (1-\theta)^{\hat \alpha_2 -1}
# $$
# 

# Find the hyperparameters of the posterior
a1_hat = a1 + Y.sum()
a2_hat = a2 + nobs - Y.sum()

# Posterior Mean
post_mean = a1_hat / (a1_hat + a2_hat)
print 'Posterior Mean (Analytic):', post_mean

# Plot the analytic posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');

# Plot the prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g');

# Cleanup
ax.set(title='Posterior Distribution (Analytic)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior']);


# #### Metropolis-Hastings: Pure Python
# 
# Although since in this case the posterior can be found analytically for the conjugate Beta prior, we can also arrive at it as the stationary distribution of a Markov chain with Metropolis-Hastings transition kernel.
# 
# To do this, we need a proposal distribution $q(\theta|\theta^{[g]})$, and here we will use a random walk proposal: $\theta^* = \theta^{[g]} + \eta_t$ where $\eta_t \sim \text{Normal}(0,\sigma^2)$ where $\sigma^2$ will be set to get a desired acceptance ratio.
# 

#%%timeit
print 'Timing: 1 loops, best of 3: 356 ms per loop'

# Metropolis-Hastings parameters
G1 = 1000 # burn-in period
G = 10000 # draws from the (converged) posterior

# Model parameters
sigma = 0.1
thetas = [0.5]             # initial value for theta
etas = np.random.normal(0, sigma, G1+G) # random walk errors
unif = np.random.uniform(size=G1+G)     # comparators for accept_probs

# Callable functions for likelihood and prior
prior_const = gamma(a1) * gamma(a2) / gamma(a1 + a2)
mh_ll = lambda theta: _likelihood(theta, nobs, Y.sum())
def mh_prior(theta):
    prior = 0
    if theta >= 0 and theta <= 1:
        prior = prior_const*(theta**(a1-1))*((1-theta)**(a2-1))
    return prior
mh_accept = lambda theta: mh_ll(theta) * mh_prior(theta)

theta_prob = mh_accept(thetas[-1])

# Metropolis-Hastings iterations
for i in range(G1+G):
    # Draw theta
    
    # Generate the proposal
    theta = thetas[-1]
    theta_star = theta + etas[i]
    theta_star_prob = mh_accept(theta_star)
    # Calculate the acceptance probability
    accept_prob = theta_star_prob / theta_prob
    
    # Append the new draw
    if accept_prob > unif[i]:
        theta = theta_star
        theta_prob = theta_star_prob
    thetas.append(theta)


# We can describe the posterior using the draws after the chain has converged (i.e. following the burn-in period):
# 

# Posterior Mean
print 'Posterior Mean (MH):', np.mean(thetas[G1:])

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas[G1:], bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via pure Python (10,000 Draws; 1,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);


# #### Metropolis-Hastings: Cython
# 
# The runtime of 356ms is not bad, by we may be able to improve matters by writing it in Cython, a pseudo-language which is then compiled into a C extension that we can call from our Python code. In the right circumstances, this can speed up code dramatically.
# 
# Although I am not an expert in MATLAB, a pretty much direct port of this code to MATLAB (almost identical to the Cython code below) runs in about 400ms, so pure Python and MATLAB appear to be reasonably similar.
# 

get_ipython().magic('load_ext cythonmagic')


get_ipython().run_cell_magic('cython', '', '\nimport numpy as np\nfrom scipy.special import gamma\ncimport numpy as np\ncimport cython\n\nfrom libc.math cimport pow\n\ncdef double likelihood(double theta, int T, int s):\n    return pow(theta, s)*pow(1-theta, T-s)\n\ncdef double prior(double theta, double a1, double a2, double prior_const):\n    if theta < 0 or theta > 1:\n        return 0\n    return prior_const*pow(theta, a1-1)*pow(1-theta, a2-1)\n\ncdef np.ndarray[np.float64_t, ndim=1] draw_posterior(np.ndarray[np.float64_t, ndim=1] theta, double eta, double unif, int T, int s, double a1, double a2, double prior_const):\n    cdef double theta_star, theta_star_prob, accept_prob\n    \n    theta_star = theta[0] + eta\n    theta_star_prob = likelihood(theta_star, T, s) * prior(theta_star, a1, a2, prior_const)\n    \n    accept_prob = theta_star_prob / theta[1]\n    \n    if accept_prob > unif:\n        theta[0] = theta_star\n        theta[1] = theta_star_prob\n        \n    return theta\n\ndef mh(double theta_init, int T, int s, double sigma, double a1, double a2, int G1, int G):\n    \n    cdef np.ndarray[np.float64_t, ndim = 1] theta, thetas, etas, unif\n    cdef double prior_const, theta_prob\n    cdef int t\n    \n    prior_const = gamma(a1) * gamma(a2) / gamma(a1 + a2)\n    theta_prob = likelihood(theta_init, T, s) * prior(theta_init, a1, a2, prior_const)\n    \n    theta = np.array([theta_init, theta_prob])\n    \n    thetas = np.zeros((G1+G,))\n    etas = np.random.normal(0, sigma, G1+G)\n    unif = np.random.uniform(size=G1+G)\n    \n    for t in range(G1+G):\n        theta = draw_posterior(theta, etas[t], unif[t], T, s, a1, a2, prior_const)\n        thetas[t] = theta[0]\n        \n    return thetas')


#%%timeit
print 'Timing: 10 loops, best of 3: 20.7 ms per loop'
thetas = mh(0.5, nobs, Y.sum(), sigma, a1, a2, G1, G)


# Notice that using Cython, we've sped up the code by a factor of about 17-20 from pure Python or MATLAB.
# 

# Posterior Mean
print 'Posterior Mean (MH):', np.mean(thetas[G1:])

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas[G1:], bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via Cython (10,000 Draws; 1,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);


# Now that we've improved the performance of our Metropolis-Hastings draws, we can increase the burn in period (although that is not necessary to ensure convergence in this case) and increase the number of draws from the converged posterior. Here we'll increase the burn in period and the post-convergence draws by a factor of 100 each. The total increase in runtime will almost exclusively be a result of the 100x increase in the post-convergence draws, so the runtime will likely increase by a factor of about 100).
# 
# Notice that this would be inconvenient using the pure Python or MATLAB code, since it would take about $100 \times 0.4 \text{s} \approx 40\text{s}$. Fortunately, our Cython implementation can run it in about $100 \times 0.02 \text{s} \approx 2\text{s}$.
# 

G1 = 100000
G = 1000000


#%%timeit
print 'Timing: 1 loops, best of 3: 2.09 s per loop'
thetas = mh(0.5, nobs, Y.sum(), sigma, a1, a2, G1, G)


# Posterior Mean
print 'Posterior Mean (MH):', np.mean(thetas[G1:])

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas[G1:], bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via Cython (1,000,000 Draws; 100,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);


# And with this many post-convergence draws, we can match the analytic posterior mean to 4 decimal places.
# 

# #### Metropolis-Hastings: PyMC
# 
# We can also make use of the [PyMC](https://github.com/pymc-devs/pymc) package to do Metropolis-Hastings runs for us. It is about twice as slow as the custom pure Python approach we employed above (and so ~40 times slower than the Cython implementation), but it is certainly much less work to set up!
# 
# (Note: I am not well-versed in PyMC, so it is certainly possible - likely, even - that there is a more performant way to do this).
# 

G1 = 1000
G = 10000


#%%timeit
print 'Timing: 1 loops, best of 3: 590 ms per loop'

pymc_theta = pymc.Beta('pymc_theta', a1, a2, value=0.5)
pymc_Y = pymc.Bernoulli('pymc_Y', p=pymc_theta, value=Y, observed=True)

model = pymc.MCMC([pymc_theta, pymc_Y])
model.sample(iter=G+G1, burn=G1, progress_bar=False)

model.summary()
thetas = model.trace('pymc_theta')[:]


# Posterior Mean
# (use all of `thetas` b/c PyMC already removed the burn-in runs here)
print 'Posterior Mean (MH):', np.mean(thetas)

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas, bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via PyMC (10,000 Draws; 1,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);


# This gives an example of the use of the Markov Switching Model that I wrote for the [Statsmodels](https://github.com/statsmodels/statsmodels) Python package, to replicate the treatment of Filardo (1994) as given in Kim and Nelson (1999). This model demonstrates estimation with time-varying transition probabilities.
# 
# This is tested against Kim and Nelson's (1999) code (HMT_TVP.OPT), which can be found at [http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm](http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm). It also corresponds to the examples of Markov-switching models from E-views 8, which can be found at [http://www.eviews.com/EViews8/ev8ecswitch_n.html#TimeVary](http://www.eviews.com/EViews8/ev8ecswitch_n.html#TimeVary).
# 
# <!-- TEASER_END -->
# 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.mar_model import MAR


# Filardo's 1994 Industrial Production dataset: Monthly, 1947.1 - 1995.3
import re
f = open('data/filardo.prn')
data = pd.DataFrame(
    [map(float, re.split(' +', line.strip())) for line in f.readlines()[:-1]],
    index=pd.date_range('1948-01-01', '1991-04-01', freq='MS'),
    columns=['month', 'ip', 'idx']
)
data['dlip'] = np.log(data['ip']).diff()*100
# Deflated pre-1960 observations by ratio of std. devs.
# See hmt_tvp.opt or Filardo (1994) p. 302
std_ratio = data['dlip']['1960-01-01':].std() / data['dlip'][:'1959-12-01'].std()
data['dlip'][:'1959-12-01'] = data['dlip'][:'1959-12-01'] * std_ratio

data['dlidx'] = np.log(data['idx']).diff()*100
data['dmdlidx'] = data['dlidx'] - data['dlidx'].mean()

# NBER recessions
from pandas.io.data import DataReader
from datetime import datetime
usrec = DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))


# Model Setup
order = 4
nstates = 2

switch_ar = False
switch_var = False
switch_mean = True


mod = MAR(data.dlip[2:], order, nstates,
                    switch_ar=switch_ar, switch_var=switch_var, switch_mean=switch_mean,
                    tvtp_exog=data.dmdlidx[1:])
params = np.array(np.r_[
    [1.64982, -0.99472, -4.35966, -1.77043], # TVTP parameters
    [0.18947, 0.07933, 0.11094,  0.12226],   # AR parameters
    [-np.log(0.69596)],                      # Std. Dev
     [-0.86585, 0.51733]                      # Mean
])


# Filter the data
(
    marginal_densities, filtered_joint_probabilities,
    filtered_joint_probabilities_t1
) = mod.filter(params);

transitions = mod.separate_params(params)[0]

# Smooth the data
filtered_marginal_probabilities = mod.marginalize_probabilities(filtered_joint_probabilities[1:])
smoothed_marginal_probabilities = mod.smooth(filtered_joint_probabilities, filtered_joint_probabilities_t1, transitions)

# Save the data
data['filtered'] = np.r_[
    [np.NaN]*(order+2),
    filtered_marginal_probabilities[:,0]
]


import matplotlib.pyplot as plt
from matplotlib import dates
fig = plt.figure(figsize=(9,4))

ax = fig.add_subplot(111)
ax.fill_between(usrec.index, 0, usrec.USREC, color='gray', alpha=0.3)
ax.plot(data.index, data.filtered, 'k')
ax.set(
    xlim=('1948-01-01', '1991-04-01'),
    ylim=(0,1),
    title='Filtered probability of a low-production state'
);


# ## State space diagnostics
# 
# It is important to run post-estimation diagnostics on all types of models. In state space models, if the model is correctly specified, the standardized one-step ahead forecast errors should be independent and identically Normally distributed. Thus, one way to assess whether or not the model adequately describes the data is to compute the standardized residuals and apply diagnostic tests to check that they meet these distributional assumptions.
# 
# Although there are many available tests, Durbin and Koopman (2012) and Harvey (1990) suggest three basic tests as a starting point:
# 
# - **Normality**: the [Jarque–Bera test](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test)
# - **Heterskedasticity**: a test similar to the [Goldfeld-Quandt test](https://en.wikipedia.org/wiki/Goldfeld%E2%80%93Quandt_test)
# - **Serial correlation**: the [Ljung-Box test](https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test)
# 
# These have been added to Statsmodels in [this pull request (2431)](https://github.com/statsmodels/statsmodels/pull/2431), and their results are added as an additional table at the bottom of the `summary` output (see the table below for an example).
# 
# Furthermore, graphical tools can be useful in assessing these assumptions. Durbin and Koopman (2012) suggest the following four plots as a starting point:
# 
# 1. A time-series plot of the standardized residuals themselves
# 2. A histogram and kernel-density of the standardized residuals, with a reference plot of the Normal(0,1) density
# 3. A [Q-Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot) against Normal quantiles
# 4. A [correlogram](https://en.wikipedia.org/wiki/Correlogram)
# 
# To that end, I have also added a `plot_diagnostics` method which creates those following four plots.
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import statsmodels.api as sm
import seaborn as sn


from pandas_datareader.data import DataReader
lgdp = np.log(DataReader('GDPC1', 'fred', start='1984-01', end='2005-01'))


mod = sm.tsa.SARIMAX(lgdp, order=(2,1,0), seasonal_order=(3,1,0,3))
res = mod.fit()
print res.summary()

fig = res.plot_diagnostics(figsize=(11,6))
fig.tight_layout()


# # Dynamic factors and coincident indices
# 
# Factor models generally try to find a small number of unobserved "factors" that influence a subtantial portion of the variation in a larger number of observed variables, and they are related to dimension-reduction techniques such as principal components analysis. Dynamic factor models explicitly model the transition dynamics of the unobserved factors, and so are often applied to time-series data.
# 
# Macroeconomic coincident indices are designed to capture the common component of the "business cycle"; such a component is assumed to simultaneously affect many macroeconomic variables. Although the estimation and use of coincident indices (for example the [Index of Coincident Economic Indicators](http://www.newyorkfed.org/research/regional_economy/coincident_summary.html)) pre-dates dynamic factor models, in several influential papers Stock and Watson (1989, 1991) used a dynamic factor model to provide a theoretical foundation for them.
# 
# Below, we follow the treatment found in Kim and Nelson (1999), of the Stock and Watson (1991) model, to formulate a dynamic factor model, estimate its parameters via maximum likelihood, and create a coincident index.
# 

# ## Macroeconomic data
# 
# The coincident index is created by considering the comovements in four macroeconomic variables (versions of thse variables are available on [FRED](https://research.stlouisfed.org/fred2/); the ID of the series used below is given in parentheses):
# 
# - Industrial production (IPMAN)
# - Real aggregate income (excluding transfer payments) (W875RX1)
# - Manufacturing and trade sales (CMRMTSPL)
# - Employees on non-farm payrolls (PAYEMS)
# 
# In all cases, the data is at the monthly frequency and has been seasonally adjusted; the time-frame considered is 1972 - 2005.
# 

get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn

np.set_printoptions(precision=4, suppress=True, linewidth=120)


from pandas_datareader.data import DataReader

# Get the datasets from FRED
start = '1979-01-01'
# end = '2014-12-01'
end = '2016-06-01'
indprod = DataReader('IPMAN', 'fred', start=start, end=end)
income = DataReader('W875RX1', 'fred', start=start, end=end)
# sales = DataReader('CMRMTSPL', 'fred', start=start, end=end)
emp = DataReader('PAYEMS', 'fred', start=start, end=end)
# dta = pd.concat((indprod, income, sales, emp), axis=1)
# dta.columns = ['indprod', 'income', 'sales', 'emp']


# **Note**: in the most recent update on FRED (8/12/15) the time series CMRMTSPL was truncated to begin in 1997; this is probably a mistake due to the fact that CMRMTSPL is a spliced series, so the earlier period is from the series HMRMT and the latter period is defined by CMRMT.
# 
# Until this is corrected, the pre-8/12/15 dataset can be downloaded from Alfred (https://alfred.stlouisfed.org/series/downloaddata?seid=CMRMTSPL) or constructed by hand from HMRMT and CMRMT, as I do below (process taken from the notes in the Alfred xls file).
# 

HMRMT = DataReader('HMRMT', 'fred', start='1967-01-01', end=end)
CMRMT = DataReader('CMRMT', 'fred', start='1997-01-01', end=end)


HMRMT_growth = HMRMT.diff() / HMRMT.shift()
sales = pd.Series(np.zeros(emp.shape[0]), index=emp.index)

# Fill in the recent entries (1997 onwards)
sales[CMRMT.index] = CMRMT

# Backfill the previous entries (pre 1997)
idx = sales.ix[:'1997-01-01'].index
for t in range(len(idx)-1, 0, -1):
    month = idx[t]
    prev_month = idx[t-1]
    sales.ix[prev_month] = sales.ix[month] / (1 + HMRMT_growth.ix[prev_month].values)


dta = pd.concat((indprod, income, sales, emp), axis=1)
dta.columns = ['indprod', 'income', 'sales', 'emp']


dta.ix[:, 'indprod':'emp'].plot(subplots=True, layout=(2, 2), figsize=(15, 6));


# Stock and Watson (1991) report that for their datasets, they could not reject the null hypothesis of a unit root in each series (so the series are integrated), but they did not find strong evidence that the series were co-integrated.
# 
# As a result, they suggest estimating the model using the first differences (of the logs) of the variables, demeaned and standardized.
# 

# Create log-differenced series
dta['dln_indprod'] = (np.log(dta.indprod)).diff() * 100
dta['dln_income'] = (np.log(dta.income)).diff() * 100
dta['dln_sales'] = (np.log(dta.sales)).diff() * 100
dta['dln_emp'] = (np.log(dta.emp)).diff() * 100

# De-mean and standardize
dta['std_indprod'] = (dta['dln_indprod'] - dta['dln_indprod'].mean()) / dta['dln_indprod'].std()
dta['std_income'] = (dta['dln_income'] - dta['dln_income'].mean()) / dta['dln_income'].std()
dta['std_sales'] = (dta['dln_sales'] - dta['dln_sales'].mean()) / dta['dln_sales'].std()
dta['std_emp'] = (dta['dln_emp'] - dta['dln_emp'].mean()) / dta['dln_emp'].std()


# ## Dynamic factors
# 
# A general dynamic factor model is written as:
# 
# $$
# \begin{align}
# y_t & = \Lambda f_t + B x_t + u_t \f_t & = A_1 f_{t-1} + \dots + A_p f_{t-p} + \eta_t \qquad \eta_t \sim N(0, I)\u_t & = C_1 u_{t-1} + \dots + C_1 f_{t-q} + \varepsilon_t \qquad \varepsilon_t \sim N(0, \Sigma)
# \end{align}
# $$
# 
# where $y_t$ are observed data, $f_t$ are the unobserved factors (evolving as a vector autoregression), $x_t$ are (optional) exogenous variables, and $u_t$ is the error, or "idiosyncratic", process ($u_t$ is also optionally allowed to be autocorrelated). The $\Lambda$ matrix is often referred to as the matrix of "factor loadings". The variance of the factor error term is set to the identity matrix to ensure identification of the unobserved factors.
# 
# This model can be cast into state space form, and the unobserved factor estimated via the Kalman filter. The likelihood can be evaluated as a byproduct of the filtering recursions, and maximum likelihood estimation used to estimate the parameters.
# 

# ## Model specification
# 
# The specific dynamic factor model in this application has 1 unobserved factor which is assumed to follow an AR(2) proces. The innovations $\varepsilon_t$ are assumed to be independent (so that $\Sigma$ is a diagonal matrix) and the error term associated with each equation, $u_{i,t}$ is assumed to follow an independent AR(2) process.
# 
# Thus the specification considered here is:
# 
# $$
# \begin{align}
# y_{i,t} & = \lambda_i f_t + u_{i,t} \u_{i,t} & = c_{i,1} u_{1,t-1} + c_{i,2} u_{i,t-2} + \varepsilon_{i,t} \qquad & \varepsilon_{i,t} \sim N(0, \sigma_i^2) \f_t & = a_1 f_{t-1} + a_2 f_{t-2} + \eta_t \qquad & \eta_t \sim N(0, I)\\end{align}
# $$
# 
# where $i$ is one of: `[indprod, income, sales, emp ]`.
# 
# This model can be formulated using the `DynamicFactor` model built-in to Statsmodels. In particular, we have the following specification:
# 
# - `k_factors = 1` - (there is 1 unobserved factor)
# - `factor_order = 2` - (it follows an AR(2) process)
# - `error_var = False` - (the errors evolve as independent AR processes rather than jointly as a VAR - note that this is the default option, so it is not specified below)
# - `error_order = 2` - (the errors are autocorrelated of order 2: i.e. AR(2) processes)
# - `error_cov_type = 'diagonal'` - (the innovations are uncorrelated; this is again the default)
# 
# Once the model is created, the parameters can be estimated via maximum likelihood; this is done using the `fit()` method.
# 
# **Note**: recall that we have de-meaned and standardized the data; this will be important in interpreting the results that follow.
# 
# **Aside**: in their empirical example, Kim and Nelson (1999) actually consider a slightly different model in which the employment variable is allowed to also depend on lagged values of the factor - this model does not fit into the built-in `DynamicFactor` class, but can be accomodated by using a subclass to implement the required new parameters and restrictions - see Appendix A, below.
# 

# ## Parameter estimation
# 
# Multivariate models can have a relatively large number of parameters, and it may be difficult to escape from local minima to find the maximized likelihood. In an attempt to mitigate this problem, I perform an initial maximization step (from the model-defined starting paramters) using the modified Powell method available in Scipy (see the minimize documentation for more information). The resulting parameters are then used as starting parameters in the standard LBFGS optimization method.
# 

# Get the endogenous data
endog = dta.ix['1979-02-01':, 'std_indprod':'std_emp']

# Create the model
mod = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=2, error_order=2)
initial_res = mod.fit(method='powell', disp=False)
res = mod.fit(initial_res.params)


# ## Estimates
# 
# Once the model has been estimated, there are two components that we can use for analysis or inference:
# 
# - The estimated parameters
# - The estimated factor
# 

# ### Parameters
# 
# The estimated parameters can be helpful in understanding the implications of the model, although in models with a larger number of observed variables and / or unobserved factors they can be difficult to interpret.
# 
# One reason for this difficulty is due to identification issues between the factor loadings and the unobserved factors. One easy-to-see identification issue is the sign of the loadings and the factors: an equivalent model to the one displayed below would result from reversing the signs of all factor loadings and the unobserved factor.
# 
# Here, one of the easy-to-interpret implications in this model is the persistence of the unobserved factor: we find that exhibits substantial persistence.
# 

print res.summary(separate_params=False)


# ### Estimated factors
# 
# While it can be useful to plot the unobserved factors, it is less useful here than one might think for two reasons:
# 
# 1. The sign-related identification issue described above.
# 2. Since the data was differenced, the estimated factor explains the variation in the differenced data, not the original data.
# 
# It is for these reasons that the coincident index is created (see below).
# 
# With these reservations, the unobserved factor is plotted below, along with the NBER indicators for US recessions. It appears that the factor is successful at picking up some degree of business cycle activity.
# 

fig, ax = plt.subplots(figsize=(13,3))

# Plot the factor
dates = endog.index._mpl_repr()
ax.plot(dates, res.factors.filtered[0], label='Factor')
ax.legend()

# Retrieve and also plot the NBER recession indicators
rec = DataReader('USREC', 'fred', start=start, end=end)
ylim = ax.get_ylim()
ax.fill_between(dates[:len(rec)], ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1);


# ## Post-estimation
# 
# Although here we will be able to interpret the results of the model by constructing the coincident index, there is a useful and generic approach for getting a sense for what is being captured by the estimated factor. By taking the estimated factors as given, regressing them (and a constant) each (one at a time) on each of the observed variables, and recording the coefficients of determination ($R^2$ values), we can get a sense of the variables for which each factor explains a substantial portion of the variance and the variables for which it does not.
# 
# In models with more variables and more factors, this can sometimes lend interpretation to the factors (for example sometimes one factor will load primarily on real variables and another on nominal variables).
# 
# In this model, with only four endogenous variables and one factor, it is easy to digest a simple table of the $R^2$ values, but in larger models it is not. For this reason, a bar plot is often employed; from the plot we can easily see that the factor explains most of the variation in industrial production index and a large portion of the variation in sales and employment, it is less helpful in explaining income.
# 

res.plot_coefficients_of_determination(figsize=(8,2));


# ## Coincident Index
# 
# As described above, the goal of this model was to create an interpretable series which could be used to understand the current status of the macroeconomy. This is what the coincident index is designed to do. It is constructed below. For readers interested in an explanation of the construction, see Kim and Nelson (1999) or Stock and Watson (1991).
# 
# In essense, what is done is to reconstruct the mean of the (differenced) factor. We will compare it to the coincident index on published by the Federal Reserve Bank of Philadelphia (USPHCI on FRED).
# 

usphci = DataReader('USPHCI', 'fred', start='1979-01-01', end='2016-06-01')['USPHCI']
usphci.plot(figsize=(13,3));


dusphci = usphci.diff()[1:].values
def compute_coincident_index(mod, res):
    # Estimate W(1)
    spec = res.specification
    design = mod.ssm['design']
    transition = mod.ssm['transition']
    ss_kalman_gain = res.filter_results.kalman_gain[:,:,-1]
    k_states = ss_kalman_gain.shape[0]

    W1 = np.linalg.inv(np.eye(k_states) - np.dot(
        np.eye(k_states) - np.dot(ss_kalman_gain, design),
        transition
    )).dot(ss_kalman_gain)[0]

    # Compute the factor mean vector
    factor_mean = np.dot(W1, dta.ix['1972-02-01':, 'dln_indprod':'dln_emp'].mean())
    
    # Normalize the factors
    factor = res.factors.filtered[0]
    factor *= np.std(usphci.diff()[1:]) / np.std(factor)

    # Compute the coincident index
    coincident_index = np.zeros(mod.nobs+1)
    # The initial value is arbitrary; here it is set to
    # facilitate comparison
    coincident_index[0] = usphci.iloc[0] * factor_mean / dusphci.mean()
    for t in range(0, mod.nobs):
        coincident_index[t+1] = coincident_index[t] + factor[t] + factor_mean
    
    # Attach dates
    coincident_index = pd.Series(coincident_index, index=dta.index).iloc[1:]
    
    # Normalize to use the same base year as USPHCI
    coincident_index *= (usphci.ix['1992-07-01'] / coincident_index.ix['1992-07-01'])
    
    return coincident_index


# Below we plot the calculated coincident index along with the US recessions and the comparison coincident index USPHCI.
# 

fig, ax = plt.subplots(figsize=(13,3))

# Compute the index
coincident_index = compute_coincident_index(mod, res)

# Plot the factor
dates = endog.index._mpl_repr()
ax.plot(dates, coincident_index, label='Coincident index')
ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
ax.legend(loc='lower right')

# Retrieve and also plot the NBER recession indicators
ylim = ax.get_ylim()
ax.fill_between(dates[:len(rec)], ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1);


# ## Appendix 1: Extending the dynamic factor model
# 
# Recall that the previous specification was described by:
# 
# $$
# \begin{align}
# y_{i,t} & = \lambda_i f_t + u_{i,t} \u_{i,t} & = c_{i,1} u_{1,t-1} + c_{i,2} u_{i,t-2} + \varepsilon_{i,t} \qquad & \varepsilon_{i,t} \sim N(0, \sigma_i^2) \f_t & = a_1 f_{t-1} + a_2 f_{t-2} + \eta_t \qquad & \eta_t \sim N(0, I)\\end{align}
# $$
# 
# Written in state space form, the previous specification of the model had the following observation equation:
# 
# $$
# \begin{bmatrix}
# y_{\text{indprod}, t} \y_{\text{income}, t} \y_{\text{sales}, t} \y_{\text{emp}, t} \\end{bmatrix} = \begin{bmatrix}
# \lambda_\text{indprod} & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\lambda_\text{income}  & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\lambda_\text{sales}   & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\lambda_\text{emp}     & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\end{bmatrix}
# \begin{bmatrix}
# f_t \f_{t-1} \u_{\text{indprod}, t} \u_{\text{income}, t} \u_{\text{sales}, t} \u_{\text{emp}, t} \u_{\text{indprod}, t-1} \u_{\text{income}, t-1} \u_{\text{sales}, t-1} \u_{\text{emp}, t-1} \\end{bmatrix}
# $$
# 
# and transition equation:
# 
# $$
# \begin{bmatrix}
# f_t \f_{t-1} \u_{\text{indprod}, t} \u_{\text{income}, t} \u_{\text{sales}, t} \u_{\text{emp}, t} \u_{\text{indprod}, t-1} \u_{\text{income}, t-1} \u_{\text{sales}, t-1} \u_{\text{emp}, t-1} \\end{bmatrix} = \begin{bmatrix}
# a_1 & a_2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \1   & 0   & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 0   & c_{\text{indprod}, 1} & 0 & 0 & 0 & c_{\text{indprod}, 2} & 0 & 0 & 0 \0   & 0   & 0 & c_{\text{income}, 1} & 0 & 0 & 0 & c_{\text{income}, 2} & 0 & 0 \0   & 0   & 0 & 0 & c_{\text{sales}, 1} & 0 & 0 & 0 & c_{\text{sales}, 2} & 0 \0   & 0   & 0 & 0 & 0 & c_{\text{emp}, 1} & 0 & 0 & 0 & c_{\text{emp}, 2} \0   & 0   & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\end{bmatrix} 
# \begin{bmatrix}
# f_{t-1} \f_{t-2} \u_{\text{indprod}, t-1} \u_{\text{income}, t-1} \u_{\text{sales}, t-1} \u_{\text{emp}, t-1} \u_{\text{indprod}, t-2} \u_{\text{income}, t-2} \u_{\text{sales}, t-2} \u_{\text{emp}, t-2} \\end{bmatrix}
# + R \begin{bmatrix}
# \eta_t \\varepsilon_{t}
# \end{bmatrix}
# $$
# 
# the `DynamicFactor` model handles setting up the state space representation and, in the `DynamicFactor.update` method, it fills in the fitted parameter values into the appropriate locations.
# 

# The extended specification is the same as in the previous example, except that we also want to allow employment to depend on lagged values of the factor. This creates a change to the $y_{\text{emp},t}$ equation. Now we have:
# 
# $$
# \begin{align}
# y_{i,t} & = \lambda_i f_t + u_{i,t} \qquad & i \in \{\text{indprod}, \text{income}, \text{sales} \}\y_{i,t} & = \lambda_{i,0} f_t + \lambda_{i,1} f_{t-1} + \lambda_{i,2} f_{t-2} + \lambda_{i,2} f_{t-3} + u_{i,t} \qquad & i = \text{emp} \u_{i,t} & = c_{i,1} u_{i,t-1} + c_{i,2} u_{i,t-2} + \varepsilon_{i,t} \qquad & \varepsilon_{i,t} \sim N(0, \sigma_i^2) \f_t & = a_1 f_{t-1} + a_2 f_{t-2} + \eta_t \qquad & \eta_t \sim N(0, I)\\end{align}
# $$
# 
# Now, the corresponding observation equation should look like the following:
# 
# $$
# \begin{bmatrix}
# y_{\text{indprod}, t} \y_{\text{income}, t} \y_{\text{sales}, t} \y_{\text{emp}, t} \\end{bmatrix} = \begin{bmatrix}
# \lambda_\text{indprod} & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\lambda_\text{income}  & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\lambda_\text{sales}   & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\lambda_\text{emp,1}   & \lambda_\text{emp,2} & \lambda_\text{emp,3} & \lambda_\text{emp,4} & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\end{bmatrix}
# \begin{bmatrix}
# f_t \f_{t-1} \f_{t-2} \f_{t-3} \u_{\text{indprod}, t} \u_{\text{income}, t} \u_{\text{sales}, t} \u_{\text{emp}, t} \u_{\text{indprod}, t-1} \u_{\text{income}, t-1} \u_{\text{sales}, t-1} \u_{\text{emp}, t-1} \\end{bmatrix}
# $$
# 
# Notice that we have introduced two new state variables, $f_{t-2}$ and $f_{t-3}$, which means we need to update the  transition equation:
# 
# $$
# \begin{bmatrix}
# f_t \f_{t-1} \f_{t-2} \f_{t-3} \u_{\text{indprod}, t} \u_{\text{income}, t} \u_{\text{sales}, t} \u_{\text{emp}, t} \u_{\text{indprod}, t-1} \u_{\text{income}, t-1} \u_{\text{sales}, t-1} \u_{\text{emp}, t-1} \\end{bmatrix} = \begin{bmatrix}
# a_1 & a_2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \1   & 0   & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 1   & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 0 & 0 & c_{\text{indprod}, 1} & 0 & 0 & 0 & c_{\text{indprod}, 2} & 0 & 0 & 0 \0   & 0   & 0 & 0 & 0 & c_{\text{income}, 1} & 0 & 0 & 0 & c_{\text{income}, 2} & 0 & 0 \0   & 0   & 0 & 0 & 0 & 0 & c_{\text{sales}, 1} & 0 & 0 & 0 & c_{\text{sales}, 2} & 0 \0   & 0   & 0 & 0 & 0 & 0 & 0 & c_{\text{emp}, 1} & 0 & 0 & 0 & c_{\text{emp}, 2} \0   & 0   & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \0   & 0   & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\end{bmatrix} 
# \begin{bmatrix}
# f_{t-1} \f_{t-2} \f_{t-3} \f_{t-4} \u_{\text{indprod}, t-1} \u_{\text{income}, t-1} \u_{\text{sales}, t-1} \u_{\text{emp}, t-1} \u_{\text{indprod}, t-2} \u_{\text{income}, t-2} \u_{\text{sales}, t-2} \u_{\text{emp}, t-2} \\end{bmatrix}
# + R \begin{bmatrix}
# \eta_t \\varepsilon_{t}
# \end{bmatrix}
# $$
# 
# This model cannot be handled out-of-the-box by the `DynamicFactor` class, but it can be handled by creating a subclass when alters the state space representation in the appropriate way.
# 

# First, notice that if we had set `factor_order = 4`, we would almost have what we wanted. In that case, the last line of the observation equation would be:
# 
# $$
# \begin{bmatrix}
# \vdots \y_{\text{emp}, t} \\end{bmatrix} = \begin{bmatrix}
# \vdots &  &  &  &  &  &  &  &  &  &  & \vdots \\lambda_\text{emp,1}   & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\end{bmatrix}
# \begin{bmatrix}
# f_t \f_{t-1} \f_{t-2} \f_{t-3} \\vdots
# \end{bmatrix}
# $$
# 
# 
# and the first line of the transition equation would be:
# 
# $$
# \begin{bmatrix}
# f_t \\vdots
# \end{bmatrix} = \begin{bmatrix}
# a_1 & a_2 & a_3 & a_4 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\vdots &  &  &  &  &  &  &  &  &  &  & \vdots \\end{bmatrix} 
# \begin{bmatrix}
# f_{t-1} \f_{t-2} \f_{t-3} \f_{t-4} \\vdots
# \end{bmatrix}
# + R \begin{bmatrix}
# \eta_t \\varepsilon_{t}
# \end{bmatrix}
# $$
# 
# Relative to what we want, we have the following differences:
# 
# 1. In the above situation, the $\lambda_{\text{emp}, j}$ are forced to be zero for $j > 0$, and we want them to be estimated as parameters.
# 2. We only want the factor to transition according to an AR(2), but under the above situation it is an AR(4).
# 
# Our strategy will be to subclass `DynamicFactor`, and let it do most of the work (setting up the state space representation, etc.) where it assumes that `factor_order = 4`. The only things we will actually do in the subclass will be to fix those two issues.
# 
# First, here is the full code of the subclass; it is discussed below. It is important to note at the outset that none of the methods defined below could have been omitted. In fact, the methods `__init__`, `start_params`, `param_names`, `transform_params`, `untransform_params`, and `update` form the core of all state space models in Statsmodels, not just the `DynamicFactor` class.
# 

from statsmodels.tsa.statespace import tools
class ExtendedDFM(sm.tsa.DynamicFactor):
    def __init__(self, endog, **kwargs):
            # Setup the model as if we had a factor order of 4
            super(ExtendedDFM, self).__init__(
                endog, k_factors=1, factor_order=4, error_order=2,
                **kwargs)

            # Note: `self.parameters` is an ordered dict with the
            # keys corresponding to parameter types, and the values
            # the number of parameters of that type.
            # Add the new parameters
            self.parameters['new_loadings'] = 3

            # Cache a slice for the location of the 4 factor AR
            # parameters (a_1, ..., a_4) in the full parameter vector
            offset = (self.parameters['factor_loadings'] +
                      self.parameters['exog'] +
                      self.parameters['error_cov'])
            self._params_factor_ar = np.s_[offset:offset+2]
            self._params_factor_zero = np.s_[offset+2:offset+4]

    @property
    def start_params(self):
        # Add three new loading parameters to the end of the parameter
        # vector, initialized to zeros (for simplicity; they could
        # be initialized any way you like)
        return np.r_[super(ExtendedDFM, self).start_params, 0, 0, 0]
    
    @property
    def param_names(self):
        # Add the corresponding names for the new loading parameters
        #  (the name can be anything you like)
        return super(ExtendedDFM, self).param_names + [
            'loading.L%d.f1.%s' % (i, self.endog_names[3]) for i in range(1,4)]

    def transform_params(self, unconstrained):
            # Perform the typical DFM transformation (w/o the new parameters)
            constrained = super(ExtendedDFM, self).transform_params(
            unconstrained[:-3])

            # Redo the factor AR constraint, since we only want an AR(2),
            # and the previous constraint was for an AR(4)
            ar_params = unconstrained[self._params_factor_ar]
            constrained[self._params_factor_ar] = (
                tools.constrain_stationary_univariate(ar_params))

            # Return all the parameters
            return np.r_[constrained, unconstrained[-3:]]

    def untransform_params(self, constrained):
            # Perform the typical DFM untransformation (w/o the new parameters)
            unconstrained = super(ExtendedDFM, self).untransform_params(
                constrained[:-3])

            # Redo the factor AR unconstraint, since we only want an AR(2),
            # and the previous unconstraint was for an AR(4)
            ar_params = constrained[self._params_factor_ar]
            unconstrained[self._params_factor_ar] = (
                tools.unconstrain_stationary_univariate(ar_params))

            # Return all the parameters
            return np.r_[unconstrained, constrained[-3:]]

    def update(self, params, transformed=True, **kwargs):
        # Peform the transformation, if required
        if not transformed:
            params = self.transform_params(params)
        params[self._params_factor_zero] = 0
        
        # Now perform the usual DFM update, but exclude our new parameters
        super(ExtendedDFM, self).update(params[:-3], transformed=True, **kwargs)

        # Finally, set our new parameters in the design matrix
        self['design', 3, 1:4] = params[-3:]
        


# So what did we just do?
# 
# #### `__init__`
# 
# The important step here was specifying the base dynamic factor model which we were operating with. In particular, as described above, we initialize with `factor_order=4`, even though we will only end up with an AR(2) model for the factor. We also performed some general setup-related tasks.
# 
# #### `start_params`
# 
# `start_params` are used as initial values in the optimizer. Since we are adding three new parameters, we need to pass those in. If we hadn't done this, the optimizer would use the default starting values, which would be three elements short.
# 
# #### `param_names`
# 
# `param_names` are used in a variety of places, but especially in the results class. Below we get a full result summary, which is only possible when all the parameters have associated names.
# 
# #### `transform_params` and `untransform_params`
# 
# The optimizer selects possibly parameter values in an unconstrained way. That's not usually desired (since variances can't be negative, for example), and `transform_params` is used to transform the unconstrained values used by the optimizer to constrained values appropriate to the model. Variances terms are typically squared (to force them to be positive), and AR lag coefficients are often constrained to lead to a stationary model. `untransform_params` is used for the reverse operation (and is important because starting parameters are usually specified in terms of values appropriate to the model, and we need to convert them to parameters appropriate to the optimizer before we can begin the optimization routine).
# 
# Even though we don't need to transform or untransform our new parameters (the loadings can in theory take on any values), we still need to modify this function for two reasons:
# 
# 1. The version in the `DynamicFactor` class is expecting 3 fewer parameters than we have now. At a minimum, we need to handle the three new parameters.
# 2. The version in the `DynamicFactor` class constrains the factor lag coefficients to be stationary as though it was an AR(4) model. Since we actually have an AR(2) model, we need to re-do the constraint. We also set the last two autoregressive coefficients to be zero here.
# 
# #### `update`
# 
# The most important reason we need to specify a new `update` method is because we have three new parameters that we need to place into the state space formulation. In particular we let the parent `DynamicFactor.update` class handle placing all the parameters except the three new ones in to the state space representation, and then we put the last three in manually.

# Create the model
extended_mod = ExtendedDFM(endog)
initial_extended_res = extended_mod.fit(method='powell', disp=False)
extended_res = extended_mod.fit(initial_extended_res.params, maxiter=1000)
print extended_res.summary(separate_params=False)


# Although this model increases the likelihood, it is not preferred by the AIC and BIC mesaures which penalize the additional three parameters.
# 
# Furthermore, the qualitative results are unchanged, as we can see from the updated $R^2$ chart and the new coincident index, both of which are practically identical to the previous results.
# 

extended_res.plot_coefficients_of_determination(figsize=(8,2));


fig, ax = plt.subplots(figsize=(13,3))

# Compute the index
extended_coincident_index = compute_coincident_index(extended_mod, extended_res)

# Plot the factor
dates = endog.index._mpl_repr()
ax.plot(dates, coincident_index, '-', linewidth=1, label='Basic model')
ax.plot(dates, extended_coincident_index, '--', linewidth=3, label='Extended model')
ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
ax.legend(loc='lower right')
ax.set(title='Coincident indices, comparison')

# Retrieve and also plot the NBER recession indicators
ylim = ax.get_ylim()
ax.fill_between(dates[:len(rec)], ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1);


# ### Getting Started with Python Packaging
# 
# The first two weeks of the state-space project have been dedicated to introducing the Kalman filter - which was written in Cython with calls to the BLAS and LAPACK libraries linked to in Scipy - into the Statsmodels build process. A future post may describe why it was not just written in pure Python (in brief, it is because the Kalman filter is a recursive algorithm with a loop over the number of entries in a dataset, where each loop involves many matrix operations on relatively small matrices). For now, though, the source `kalman_filter.pyx` needs to be "Cythonized" into `kalman_filter.c` and then compiled into (e.g.) `kalman_filter.so`, either when the package is installed using pip, or from source (e.g. `python setup.py install`).
# 
# The first thing to figure out was the state of Python's packaging. I've had a vague sense of some of the various tools of Python packaging for a while (especially since it used to be recommended to specify `--distribute` which making a new [virtualenv](http://www.google.com)), but I built all my Cython packages either via a `python setup.py build_ext --inplace` (from the [Cython quickstart](http://docs.cython.org/src/quickstart/build.html)) or via [IPython magic](http://ipython.org/ipython-doc/2/config/extensions/cythonmagic.html).
# 
# The recommended `setup.py` file from Cython quickstart is:
# 
# ```python
# from distutils.core import setup
# from Cython.Build import cythonize
# 
# setup(
#   name = 'Hello world app',
#   ext_modules = cythonize("hello.pyx"),
# )
# ```
# 
# as you can see, this uses the [`distutils`](https://docs.python.org/2/library/distutils.html) package. However, while `distutils` is part of base Python and is standard for packaging, it, from what I could tell, `distribute` was the up-and-coming way to proceed. Would that it were that simple; it turns out that Python packaging is not for the faint of heart. A wonderful [stackoverflow answer](http://stackoverflow.com/a/14753678/603962) describes the state-of-the-art (hopefully) as of October 2013. It comes to the conclusion that [`setuptools`](https://pythonhosted.org/setuptools/) is probably the way to go, unless you only need basic packaging, in which case you should use `distutils`.
# 
# ### Setuptools
# 
# So it appeared that the way to go was to use `setuptools` (and more than personal preference, Statsmodels [uses `setuptools`](https://github.com/statsmodels/statsmodels/blob/master/setup.py#L30)). Unfortunately, I have always previously used the above snippet which is `distutils` based, and as it turns out, the magic that makes that bit of code possible *is not available in setuptools*. You can read [this mailing list conversation](https://mail.python.org/pipermail/distutils-sig/2007-September/008207.html) from September 2013 for a fascinating back-and-forth about what should be supported where, leading to the surprising conclusion that to make Setuptools automatically call Cython to build `*.pyx` files, one should *trick* it into believing there was a fake Pyrex installation.
# 
# This approach can be seen at the [repository](http://github.com/ChadFulton/pykalman_filter) for the existing Kalman filter code, or at https://github.com/njsmith/scikits-sparse (in both cases, look for the "fake_pyrex" directory in the project root).
# 
# It's often a good idea, though, to look at [NumPy](http://github.com/numpy/numpy) and [SciPy](https://github.com/scipy/scipy) for *how it should be done*, and it turns out that neither of them use a fake Pyrex directory, and neither do rely on `setuptools` (or `distutils`) to Cythonize the `*.pyx` files. Instead, they use a direct `subprocess` call to `cythonize` directly. Why do this, though?
# 
# ### NumPy and SciPy
# 
# Although at first it seemed like an awfully Byzantine and arbitrary mish-mash of roll-your-owns, where no two parties do things the same way, it turns out that the NumPy / SciPy approach agrees, in spirit, with the latest `Cython` [documentation on compilation](http://docs.cython.org/src/reference/compilation.html). The idea is that `Cython` should not be a required dependency for installation, and thus the *already Cythonized* `*.c` files should be included in the distributed package. These will be cythonized during the `python setup.py sdist` process.
# 
# So the end result is that setuptools should not be required to cythonize `*.pyx` files, it only needs to compile and link `*.c` files (which it has no problem with - no fake pyrex directory needed). Then the question is, how to cythonize the files? It turns out that the common way, as mentioned above, is to use a subprocess call to the `cythonize` binary directly (see [Statsmodels](https://github.com/statsmodels/statsmodels/blob/master/setup.py#L86), [NumPy](https://github.com/numpy/numpy/blob/master/setup.py#L187), [SciPy](https://github.com/scipy/scipy/blob/master/setup.py#L158)).

# ## Markov switching dynamic regression models
# 

# This notebook provides an example of the use of Markov switching models in Statsmodels to estimate dynamic regression models with changes in regime. It follows the examples in the Stata Markov switching documentation, which can be found at http://www.stata.com/manuals14/tsmswitch.pdf.
# 

get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn

# NBER recessions
from pandas_datareader.data import DataReader
from datetime import datetime
usrec = DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))


# ### Federal funds rate with switching intercept
# 
# The first example models the federal funds rate as noise around a constant intercept, but where the intercept changes during different regimes. The model is simply:
# 
# $$r_t = \mu_{S_t} + \varepsilon_t \qquad \varepsilon_t \sim N(0, \sigma^2)$$
# 
# where $S_t \in \{0, 1\}$, and the regime transitions according to
# 
# $$ P(S_t = s_t | S_{t-1} = s_{t-1}) =
# \begin{bmatrix}
# p_{00} & p_{10} \1 - p_{00} & 1 - p_{10}
# \end{bmatrix}
# $$
# 
# We will estimate the parameters of this model by maximum likelihood: $p_{00}, p_{10}, \mu_0, \mu_1, \sigma^2$.
# 
# The data used in this example can be found at http://www.stata-press.com/data/r14/usmacro.
# 

# Get the federal funds rate data
from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds
dta_fedfunds = pd.Series(fedfunds, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))

# Plot the data
dta_fedfunds.plot(title='Federal funds rate', figsize=(12,3))

# Fit the model
# (a switching mean is the default of the MarkovRegession model)
mod_fedfunds = sm.tsa.MarkovRegression(dta_fedfunds, k_regimes=2)
res_fedfunds = mod_fedfunds.fit()


print(res_fedfunds.summary())


# From the summary output, the mean federal funds rate in the first regime (the "low regime") is estimated to be $3.7$ whereas in the "high regime" it is $9.6$. Below we plot the smoothed probabilities of being in the high regime. The model suggests that the 1980's was a time-period in which a high federal funds rate existed.
# 

res_fedfunds.smoothed_marginal_probabilities[1].plot(
    title='Probability of being in the high regime', figsize=(12,3));


# From the estimated transition matrix we can calculate the expected duration of a low regime versus a high regime.
# 

print(res_fedfunds.expected_durations)


# A low regime is expected to persist for about fourteen years, whereas the high regime is expected to persist for only about five years.
# 

# ### Federal funds rate with switching intercept and lagged dependent variable
# 
# The second example augments the previous model to include the lagged value of the federal funds rate.
# 
# $$r_t = \mu_{S_t} + r_{t-1} \beta_{S_t} + \varepsilon_t \qquad \varepsilon_t \sim N(0, \sigma^2)$$
# 
# where $S_t \in \{0, 1\}$, and the regime transitions according to
# 
# $$ P(S_t = s_t | S_{t-1} = s_{t-1}) =
# \begin{bmatrix}
# p_{00} & p_{10} \1 - p_{00} & 1 - p_{10}
# \end{bmatrix}
# $$
# 
# We will estimate the parameters of this model by maximum likelihood: $p_{00}, p_{10}, \mu_0, \mu_1, \beta_0, \beta_1, \sigma^2$.
# 

# Fit the model
mod_fedfunds2 = sm.tsa.MarkovRegression(
    dta_fedfunds.iloc[1:], k_regimes=2, exog=dta_fedfunds.iloc[:-1])
res_fedfunds2 = mod_fedfunds2.fit()


print(res_fedfunds2.summary())


# There are several things to notice from the summary output:
# 
# 1. The information criteria have decreased substantially, indicating that this model has a better fit than the previous model.
# 2. The interpretation of the regimes, in terms of the intercept, have switched. Now the first regime has the higher intercept and the second regime has a lower intercept.
# 
# Examining the smoothed probabilities of the high regime state, we now see quite a bit more variability.
# 

res_fedfunds2.smoothed_marginal_probabilities[0].plot(
    title='Probability of being in the high regime', figsize=(12,3));


# Finally, the expected durations of each regime have decreased quite a bit.
# 

print(res_fedfunds2.expected_durations)


# ### Taylor rule with 2 or 3 regimes
# 
# We now include two additional exogenous variables - a measure of the output gap and a measure of inflation - to estimate a switching Taylor-type rule with both 2 and 3 regimes to see which fits the data better.
# 
# Because the models can be often difficult to estimate, for the 3-regime model we employ a search over starting parameters to improve results, specifying 20 random search repetitions.
# 

# Get the additional data
from statsmodels.tsa.regime_switching.tests.test_markov_regression import ogap, inf
dta_ogap = pd.Series(ogap, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))
dta_inf = pd.Series(inf, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))

exog = pd.concat((dta_fedfunds.shift(), dta_ogap, dta_inf), axis=1).iloc[4:]

# Fit the 2-regime model
mod_fedfunds3 = sm.tsa.MarkovRegression(
    dta_fedfunds.iloc[4:], k_regimes=2, exog=exog)
res_fedfunds3 = mod_fedfunds3.fit()

# Fit the 3-regime model
np.random.seed(12345)
mod_fedfunds4 = sm.tsa.MarkovRegression(
    dta_fedfunds.iloc[4:], k_regimes=3, exog=exog)
res_fedfunds4 = mod_fedfunds4.fit(search_reps=20)


print(res_fedfunds3.summary())


print(res_fedfunds4.summary())


# Due to lower information criteria, we might prefer the 3-state model, with an interpretation of low-, medium-, and high-interest rate regimes. The smoothed probabilities of each regime are plotted below.
# 

fig, axes = plt.subplots(3, figsize=(10,7))

ax = axes[0]
ax.plot(res_fedfunds4.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-interest rate regime')

ax = axes[1]
ax.plot(res_fedfunds4.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a medium-interest rate regime')

ax = axes[2]
ax.plot(res_fedfunds4.smoothed_marginal_probabilities[2])
ax.set(title='Smoothed probability of a high-interest rate regime')

fig.tight_layout()


# ### Switching variances
# 
# We can also accomodate switching variances. In particular, we consider the model
# 
# $$
# y_t = \mu_{S_t} + y_{t-1} \beta_{S_t} + \varepsilon_t \quad \varepsilon_t \sim N(0, \sigma_{S_t}^2)
# $$
# 
# We use maximum likelihood to estimate the parameters of this model: $p_{00}, p_{10}, \mu_0, \mu_1, \beta_0, \beta_1, \sigma_0^2, \sigma_1^2$.
# 
# The application is to absolute returns on stocks, where the data can be found at http://www.stata-press.com/data/r14/snp500.
# 

# Get the federal funds rate data
from statsmodels.tsa.regime_switching.tests.test_markov_regression import areturns
dta_areturns = pd.Series(areturns, index=pd.date_range('2004-05-04', '2014-5-03', freq='W'))

# Plot the data
dta_areturns.plot(title='Absolute returns, S&P500', figsize=(12,3))

# Fit the model
mod_areturns = sm.tsa.MarkovRegression(
    dta_areturns.iloc[1:], k_regimes=2, exog=dta_areturns.iloc[:-1], switching_variance=True)
res_areturns = mod_areturns.fit()


print(res_areturns.summary())


# The first regime is a low-variance regime and the second regime is a high-variance regime. Below we plot the probabilities of being in the low-variance regime. Between 2008 and 2012 there does not appear to be a clear indication of one regime guiding the economy.
# 

res_areturns.smoothed_marginal_probabilities[0].plot(
    title='Probability of being in a low-variance regime', figsize=(12,3));


# This gives an example of the use of the Markov Switching Model that I wrote for the [Statsmodels](https://github.com/statsmodels/statsmodels) Python package, to replicate Hamilton's (1989) seminal paper introducing Markov-switching models via the Hamilton Filter. It uses the Kim (1994) smoother, and matches the treatment in Kim and Nelson (1999).
# 
# This is tested against Kim and Nelson's (1999) code (HMT4_KIM.OPT), which can be found at [http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm](http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm). It also corresponds to the examples of Markov-switching models from E-views 8, which can be found at [http://www.eviews.com/EViews8/ev8ecswitch_n.html#MarkovAR](http://www.eviews.com/EViews8/ev8ecswitch_n.html#MarkovAR).
# 
# <!-- TEASER_END -->
# 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.mar_model import MAR


# Model Setup
order = 4
nstates = 2

switch_ar = False
switch_sd = False
switch_mean = True


# Hamilton's 1989 GNP dataset: Quarterly, 1947.1 - 1995.3
f = open('data/gdp4795.prn')
data = pd.DataFrame(
    [float(line) for line in f.readlines()[:-3]],
    index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
    columns=['gnp']
)
data['dlgnp'] = np.log(data['gnp']).diff()*100
data = data['1952-01-01':'1984-10-01']

# NBER recessions
from pandas.io.data import DataReader
from datetime import datetime
usrec = DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))


mod = MAR(data.dlgnp, 4, nstates)
params = np.array([
    1.15590, -2.20657,
    0.08983, -0.01861, -0.17434, -0.08392,
    -np.log(0.79619),
    -0.21320, 1.12828
])


# Filter the data
(
    marginal_densities, filtered_joint_probabilities,
    filtered_joint_probabilities_t1
) = mod.filter(params);

transitions = mod.separate_params(params)[0]

# Smooth the data
filtered_marginal_probabilities = mod.marginalize_probabilities(filtered_joint_probabilities[1:])
smoothed_marginal_probabilities = mod.smooth(filtered_joint_probabilities, filtered_joint_probabilities_t1, transitions)

# Save the data
data['filtered'] = np.r_[
    [np.NaN]*order,
    filtered_marginal_probabilities[:,0]
]
data['smoothed'] = np.r_[
    [np.NaN]*order,
    smoothed_marginal_probabilities[:,0]
]


import matplotlib.pyplot as plt
from matplotlib import dates
fig = plt.figure(figsize=(9,9))

ax = fig.add_subplot(211)
ax.fill_between(usrec.index, 0, usrec.USREC, color='gray', alpha=0.3)
ax.plot(data.index, data.filtered, 'k')
ax.set(
    xlim=('1952-04-01', '1984-12-01'),
    ylim=(0,1),
    title='Filtered probability of a recession (GDP: 1952:II - 1984:IV)'
);

ax = fig.add_subplot(212)
ax.fill_between(usrec.index, 0, usrec.USREC, color='gray', alpha=0.3)
ax.plot(data.index, data.smoothed, 'k')
ax.set(
    xlim=('1952-04-01', '1984-12-01'),
    ylim=(0,1),
    title='Smoothed probability of a recession (GDP: 1952:II - 1984:IV)'
);


# This post demonstrates the use of the Self-Exciting Threshold Autoregression module I wrote for the [Statsmodels](https://github.com/statsmodels/statsmodels) Python package, to analyze the often-examined [Sunspots dataset](http://www.ngdc.noaa.gov/stp/solar/ssndata.html). In particular, I pick up where the Sunspots section of the Statsmodels [ARMA Notebook example](http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma.html) leaves off, and look at estimation and forecasting of SETAR models.
# 
# <!-- TEASER_END -->
# 

# <!-- TEASER_END -->
# 

# 
# 
# Note: here we consider the raw Sunspot series to match the ARMA example, although many sources in the literature apply a transformation to the series before modeling.
# 

# <!-- TEASER_END -->
# 

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.setar_model as setar_model
import statsmodels.tsa.bds as bds


print sm.datasets.sunspots.NOTE


dta = sm.datasets.sunspots.load_pandas().data


dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]


dta.plot(figsize=(12,8));


# First we'll fit an AR(3) process to the data as in the ARMA Notebook Example.
# 

arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit()
print arma_mod30.params
print arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic


# To test for non-linearity, we can use the BDS test on the residuals of the linear AR(3) model.
# 

bds.bds(arma_mod30.resid, 3)


# The null hypothesis of the BDS test is that the given series is an iid process (independent and identically distributed). Here the p-values are small enough that we can confidently reject the null (of iid).
# 
# This suggests there may be an underlying non-linear structure. To try and capture this, we'll fit a SETAR(2) model to the data to allow for two regimes, and we let each regime be an AR(3) process. Here we're not specifying the delay or threshold values, so they'll be optimally selected from the model.
# 
# Note: In the summary, the \gamma parameter(s) are the threshold value(s).
# 

setar_mod23 = setar_model.SETAR(dta, 2, 3).fit()
print setar_mod23.summary()


# Note that the The AIC and BIC criteria prefer the SETAR model to the AR model.
# 

# We can also directly test for the appropriate model, noting that an AR(3) is the same as a SETAR(1;1,3), so the specifications are nested.
# 
# Note: this is a bootstrapped test, so it is rather slow until improvements can be made.
# 

setar_mod23 = setar_model.SETAR(dta, 2, 3).fit()
f_stat, pvalue, bs_f_stats = setar_mod23.order_test() # by default tests against SETAR(1)
print pvalue


# The null hypothesis is a SETAR(1), so it looks like we can safely reject it in favor of the SETAR(2) alternative.
# 
# One thing to note, though, is that the default assumptions of `order_test()` is that there is homoskedasticity, which may be unreasonable here. So we can force the test to allow for heteroskedasticity of general form (in this case it doesn't look like it matters, however).
# 

f_stat_h, pvalue_h, bs_f_stats_h = setar_mod23.order_test(heteroskedasticity='g')
print pvalue


# Note that the BDS test still rejects the null when considering the residuals of the series, although with less strength than it did the AR(3) model. We can take a look at the residual plot to see that it appears the errors may have a mean of zero, but may not exhibit homoskedasticity (see Hansen (1999) for more details).
# 

print bds.bds(setar_mod23.resid, 3)
setar_mod23.resid.plot(figsize=(10,5));


# We have two new types of parameters estimated here compared to an ARMA model. The delay and the threshold(s). The delay parameter selects which lag of the process to use as the threshold variable, and the thresholds indicate which values of the threshold variable separate the datapoints into the (here two) regimes.
# 
# The confidence interval for the threshold parameter is generated (as in Hansen (1997)) by inverting the likelihood ratio statistic created from  considering the selected threshold value against ecah alternative threshold value, and comparing against critical values for various confidence interval levels. We can see that graphically by plotting the likelihood ratio sequence against each alternate threshold.
# 
# Alternate thresholds that correspond to likelihood ratio statistics less than the critical value are included in a confidence set, and the lower and upper bounds of the confidence interval are the smallest and largest threshold, respectively, in the confidence set.
# 

setar_mod23.plot_threshold_ci(0, figwidth=10, figheight=5);


# As in the ARMA Notebook Example, we can take a look at in-sample dynamic prediction and out-of-sample forecasting.
# 

predict_arma_mod30 = arma_mod30.predict('1990', '2012', dynamic=True)
predict_setar_mod23 = setar_mod23.predict('1990', '2012', dynamic=True)


ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_arma_mod30.plot(ax=ax, style='r--', linewidth=2, label='AR(3) Dynamic Prediction');
ax = predict_setar_mod23.plot(ax=ax, style='k--', linewidth=2, label='SETAR(2;3,3) Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0));


# It appears the dynamic prediction from the SETAR model is able to track the observed datapoints a little better than the AR(3) model. We can compare with the root mean square forecast error, and see that the SETAR does slightly better.
# 

def rmsfe(y, yhat):
    return (y.sub(yhat)**2).mean()
print 'AR(3):        ', rmsfe(dta.SUNACTIVITY, predict_arma_mod30)
print 'SETAR(2;3,3): ', rmsfe(dta.SUNACTIVITY, predict_setar_mod23)


# If we extend the forecast window, however, it is clear that the SETAR model is the only one that even begins to fit the shape of the data, because the data is cyclic.
# 

predict_arma_mod30_long = arma_mod30.predict('1960', '2012', dynamic=True)
predict_setar_mod23_long = setar_mod23.predict('1960', '2012', dynamic=True)
ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_arma_mod30_long.plot(ax=ax, style='r--', linewidth=2, label='AR(3) Dynamic Prediction');
ax = predict_setar_mod23_long.plot(ax=ax, style='k--', linewidth=2, label='SETAR(2;3,3) Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0));


print 'AR(3):        ', rmsfe(dta.SUNACTIVITY, predict_arma_mod30_long)
print 'SETAR(2;3,3): ', rmsfe(dta.SUNACTIVITY, predict_setar_mod23_long)


# This gives an example of the use of the Markov Switching Model that I wrote for the [Statsmodels](https://github.com/statsmodels/statsmodels) Python package, to replicate the treatment of Kim, Nelson, and Startz (1998) as given in Kim and Nelson (1999). This model demonstrates estimation with regime heteroskedasticity (switching of variances) and fixed means (all at zero).
# 
# This is tested against Kim and Nelson's (1999) code (STCK_V3.OPT), which can be found at [http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm](http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm). It also corresponds to the examples of Markov-switching models from E-views 8, which can be found at [http://www.eviews.com/EViews8/ev8ecswitch_n.html#RegHet](http://www.eviews.com/EViews8/ev8ecswitch_n.html#RegHet).
# 
# <!-- TEASER_END -->
# 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.mar_model import MAR


# Model Setup
order = 0
nstates = 3

switch_ar = False
switch_var = True
switch_mean = [0,0,0]


# Equal-Weighted Excess Returns
f = open('data/ew_excs.prn')
data = pd.DataFrame(
    [float(line.strip()) for line in f.readlines()[:-3]],
    # Not positive these are the right dates...
    index=pd.date_range('1926-01-01', '1995-10-01', freq='MS'),
    columns=['ewer']
)
data = data[0:732]
data['dmewer'] = data['ewer'] - data['ewer'].mean()


mod = MAR(data.dmewer, 0, nstates,
          switch_ar=switch_ar, switch_var=switch_var, switch_mean=switch_mean)
params = np.array([
    16.399767, 12.791361, 0.522758, 4.417225, -5.845336, -3.028234,
    6.704260/2,  5.520378/2,  3.473059/2
])


# Filter the data
(
    marginal_densities, filtered_joint_probabilities,
    filtered_joint_probabilities_t1
) = mod.filter(params);

transitions = mod.separate_params(params)[0]

# Smooth the data
filtered_marginal_probabilities = mod.marginalize_probabilities(filtered_joint_probabilities[1:])
smoothed_marginal_probabilities = mod.smooth(filtered_joint_probabilities, filtered_joint_probabilities_t1, transitions)

# Save the data
data['smoothed_low'] = np.r_[
    [np.NaN]*order,
    smoothed_marginal_probabilities[:,0]
]
data['smoothed_medium'] = np.r_[
    [np.NaN]*order,
    smoothed_marginal_probabilities[:,1]
]
data['smoothed_high'] = np.r_[
    [np.NaN]*order,
    smoothed_marginal_probabilities[:,2]
]


import matplotlib.pyplot as plt
from matplotlib import dates
fig = plt.figure(figsize=(9,13))

ax = fig.add_subplot(311)
ax.plot(data.index, data.smoothed_low, 'k')
ax.set(
    xlim=('1926-01-01', '1986-12-01'),
    ylim=(0,1),
    title='Smoothed probability of a low-variance state for stock returns'
);
ax = fig.add_subplot(312)
ax.plot(data.index, data.smoothed_medium, 'k')
ax.set(
    xlim=('1926-01-01', '1986-12-01'),
    ylim=(0,1),
    title='Smoothed probability of a medium-variance state for stock returns'
);
ax = fig.add_subplot(313)
ax.plot(data.index, data.smoothed_high, 'k')
ax.set(
    xlim=('1926-01-01', '1986-12-01'),
    ylim=(0,1),
    title='Smoothed probability of a high-variance state for stock returns'
);


# ## Markov switching autoregression models
# 

# This notebook provides an example of the use of Markov switching models in Statsmodels to replicate a number of results presented in Kim and Nelson (1999). It applies the Hamilton (1989) filter the Kim (1994) smoother.
# 
# This is tested against the Markov-switching models from E-views 8, which can be found at http://www.eviews.com/EViews8/ev8ecswitch_n.html#MarkovAR or the Markov-switching models of Stata 14 which can be found at http://www.stata.com/manuals14/tsmswitch.pdf.
# 

get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn

# NBER recessions
from pandas_datareader.data import DataReader
from datetime import datetime
usrec = DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))


# ### Hamilton (1989) switching model of GNP
# 
# This replicates Hamilton's (1989) seminal paper introducing Markov-switching models. The model is an autoregressive model of order 4 in which the mean of the process switches between two regimes. It can be written:
# 
# $$
# y_t = \mu_{S_t} + \phi_1 (y_{t-1} - \mu_{S_{t-1}}) + \phi_2 (y_{t-2} - \mu_{S_{t-2}}) + \phi_3 (y_{t-3} - \mu_{S_{t-3}}) + \phi_4 (y_{t-4} - \mu_{S_{t-4}}) + \varepsilon_t
# $$
# 
# Each period, the regime transitions according to the following matrix of transition probabilities:
# 
# $$ P(S_t = s_t | S_{t-1} = s_{t-1}) =
# \begin{bmatrix}
# p_{00} & p_{10} \p_{01} & p_{11}
# \end{bmatrix}
# $$
# 
# where $p_{ij}$ is the probability of transitioning *from* regime $i$, *to* regime $j$.
# 
# The model class is `MarkovAutoregression` in the time-series part of `Statsmodels`. In order to create the model, we must specify the number of regimes with `k_regimes=2`, and the order of the autoregression with `order=4`. The default model also includes switching autoregressive coefficients, so here we also need to specify `switching_ar=False` to avoid that.
# 
# After creation, the model is `fit` via maximum likelihood estimation. Under the hood, good starting parameters are found using a number of steps of the expectation maximization (EM) algorithm, and a quasi-Newton (BFGS) algorithm is applied to quickly find the maximum.
# 

# Get the RGNP data to replicate Hamilton
from statsmodels.tsa.regime_switching.tests.test_markov_autoregression import rgnp
dta_hamilton = pd.Series(rgnp, index=pd.date_range('1951-04-01', '1984-10-01', freq='QS'))

# Plot the data
dta_hamilton.plot(title='Growth rate of Real GNP', figsize=(12,3))

# Fit the model
mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
res_hamilton = mod_hamilton.fit()


print(res_hamilton.summary())


# We plot the filtered and smoothed probabilities of a recession. Filtered refers to an estimate of the probability at time $t$ based on data up to and including time $t$ (but excluding time $t+1, ..., T$). Smoothed refers to an estimate of the probability at time $t$ using all the data in the sample.
# 
# For reference, the shaded periods represent the NBER recessions.
# 

fig, axes = plt.subplots(2, figsize=(7,7))
ax = axes[0]
ax.plot(res_hamilton.filtered_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.3)
ax.set(xlim=(dta_hamilton.index[4], dta_hamilton.index[-1]), ylim=(0, 1),
       title='Filtered probability of recession')

ax = axes[1]
ax.plot(res_hamilton.smoothed_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.3)
ax.set(xlim=(dta_hamilton.index[4], dta_hamilton.index[-1]), ylim=(0, 1),
       title='Smoothed probability of recession')

fig.tight_layout()


# From the estimated transition matrix we can calculate the expected duration of a recession versus an expansion.
# 

print(res_hamilton.expected_durations)


# In this case, it is expected that a recession will last about one year (4 quarters) and an expansion about two and a half years.
# 

# ### Kim, Nelson, and Startz (1998) Three-state Variance Switching
# 
# This model demonstrates estimation with regime heteroskedasticity (switching of variances) and no mean effect. The dataset can be reached at http://econ.korea.ac.kr/~cjkim/MARKOV/data/ew_excs.prn.
# 
# The model in question is:
# 
# $$
# \begin{align}
# y_t & = \varepsilon_t \\varepsilon_t & \sim N(0, \sigma_{S_t}^2)
# \end{align}
# $$
# 
# Since there is no autoregressive component, this model can be fit using the `MarkovRegression` class. Since there is no mean effect, we specify `trend='nc'`. There are hypotheized to be three regimes for the switching variances, so we specify `k_regimes=3` and `switching_variance=True` (by default, the variance is assumed to be the same across regimes).
# 

# Get the dataset
raw = pd.read_table('ew_excs.prn', header=None, skipfooter=1, engine='python')
raw.index = pd.date_range('1926-01-01', '1995-12-01', freq='MS')

dta_kns = raw.ix[:'1986'] - raw.ix[:'1986'].mean()

# Plot the dataset
dta_kns[0].plot(title='Excess returns', figsize=(12, 3))

# Fit the model
mod_kns = sm.tsa.MarkovRegression(dta_kns, k_regimes=3, trend='nc', switching_variance=True)
res_kns = mod_kns.fit()


print(res_kns.summary())


# Below we plot the probabilities of being in each of the regimes; only in a few periods is a high-variance regime probable.
# 

fig, axes = plt.subplots(3, figsize=(10,7))

ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-variance regime for stock returns')

ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a medium-variance regime for stock returns')

ax = axes[2]
ax.plot(res_kns.smoothed_marginal_probabilities[2])
ax.set(title='Smoothed probability of a high-variance regime for stock returns')

fig.tight_layout()


# ### Filardo (1994) Time-Varying Transition Probabilities
# 
# This model demonstrates estimation with time-varying transition probabilities. The dataset can be reached at http://econ.korea.ac.kr/~cjkim/MARKOV/data/filardo.prn.
# 
# In the above models we have assumed that the transition probabilities are constant across time. Here we allow the probabilities to change with the state of the economy. Otherwise, the model is the same Markov autoregression of Hamilton (1989).
# 
# Each period, the regime now transitions according to the following matrix of time-varying transition probabilities:
# 
# $$ P(S_t = s_t | S_{t-1} = s_{t-1}) =
# \begin{bmatrix}
# p_{00,t} & p_{10,t} \p_{01,t} & p_{11,t}
# \end{bmatrix}
# $$
# 
# where $p_{ij,t}$ is the probability of transitioning *from* regime $i$, *to* regime $j$ in period $t$, and is defined to be:
# 
# $$
# p_{ij,t} = \frac{\exp\{ x_{t-1}' \beta_{ij} \}}{1 + \exp\{ x_{t-1}' \beta_{ij} \}}
# $$
# 
# Instead of estimating the transition probabilities as part of maximum likelihood, the regression coefficients $\beta_{ij}$ are estimated. These coefficients relate the transition probabilities to a vector of pre-determined or exogenous regressors $x_{t-1}$.
# 

# Get the dataset
dta_filardo = pd.read_table('filardo.prn', sep=' +', header=None, skipfooter=1, engine='python')
dta_filardo.columns = ['month', 'ip', 'leading']
dta_filardo.index = pd.date_range('1948-01-01', '1991-04-01', freq='MS')

dta_filardo['dlip'] = np.log(dta_filardo['ip']).diff()*100
# Deflated pre-1960 observations by ratio of std. devs.
# See hmt_tvp.opt or Filardo (1994) p. 302
std_ratio = dta_filardo['dlip']['1960-01-01':].std() / dta_filardo['dlip'][:'1959-12-01'].std()
dta_filardo['dlip'][:'1959-12-01'] = dta_filardo['dlip'][:'1959-12-01'] * std_ratio

dta_filardo['dlleading'] = np.log(dta_filardo['leading']).diff()*100
dta_filardo['dmdlleading'] = dta_filardo['dlleading'] - dta_filardo['dlleading'].mean()

# Plot the data
dta_filardo['dlip'].plot(title='Standardized growth rate of industrial production', figsize=(13,3))
plt.figure()
dta_filardo['dmdlleading'].plot(title='Leading indicator', figsize=(13,3));


# The time-varying transition probabilities are specified by the `exog_tvtp` parameter.
# 
# Here we demonstrate another feature of model fitting - the use of a random search for MLE starting parameters. Because Markov switching models are often characterized by many local maxima of the likelihood function, performing an initial optimization step can be helpful to find the best parameters.
# 
# Below, we specify that 20 random perturbations from the starting parameter vector are examined and the best one used as the actual starting parameters. Because of the random nature of the search, we seed the random number generator beforehand to allow replication of the result.
# 

mod_filardo = sm.tsa.MarkovAutoregression(
    dta_filardo.ix[2:, 'dlip'], k_regimes=2, order=4, switching_ar=False,
    exog_tvtp=sm.add_constant(dta_filardo.ix[1:-1, 'dmdlleading']))

np.random.seed(12345)
res_filardo = mod_filardo.fit(search_reps=20)


print(res_filardo.summary())


# Below we plot the smoothed probability of the economy operating in a low-production state, and again include the NBER recessions for comparison.
# 

fig, ax = plt.subplots(figsize=(12,3))

ax.plot(res_filardo.smoothed_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.3)
ax.set(xlim=(dta_filardo.index[6], dta_filardo.index[-1]), ylim=(0, 1),
       title='Smoothed probability of a low-production state');


# Using the time-varying transition probabilities, we can see how the expected duration of a low-production state changes over time:
# 

res_filardo.expected_durations[0].plot(
    title='Expected duration of a low-production state', figsize=(12,3));


# During recessions, the expected duration of a low-production state is much higher than in an expansion.
# 

# An astonishing variety of time series econometrics problems can
# be handled in one way or another by putting a model into state
# space form and applying the Kalman filter, providing optimal
# estimates of latent state variables conditioning on observed
# data and the loglikelihood of parameters. Better still, writing
# code to run through the Kalman filter recursions is very
# straightforward in many of the popular software packages (e.g.
# Python, MATLAB) and can be accomplished in fewer than 50 lines of code.
# 
# Considering a time-invariant state-space model such
# as<sup>3</sup>:
# 
# $$
# \begin{align}
# y_t & = Z \alpha_t + \varepsilon_t \qquad & \varepsilon_t \sim N(0, H) \\\alpha_{t+1} & = T \alpha_t + \eta_t \qquad & \eta_t \sim N(0, Q) \\\alpha_0 & \sim N(a_0, P_0) ~ \text{known}
# \end{align}
# $$
# 
# the Kalman filter can be written as
# 

import numpy as np

def kalman_filter(y, Z, H, T, Q, a_0, P_0):
    # Dimensions
    k_endog, nobs = y.shape
    k_states = T.shape[0]

    # Allocate memory for variables
    filtered_state = np.zeros((k_states, nobs))
    filtered_state_cov = np.zeros((k_states, k_states, nobs))
    predicted_state = np.zeros((k_states, nobs+1))
    predicted_state_cov = np.zeros((k_states, k_states, nobs+1))
    forecast = np.zeros((k_endog, nobs))
    forecast_error = np.zeros((k_endog, nobs))
    forecast_error_cov = np.zeros((k_endog, k_endog, nobs))
    loglikelihood = np.zeros((nobs+1,))

    # Copy initial values to predicted
    predicted_state[:, 0] = a_0
    predicted_state_cov[:, :, 0] = P_0

    # Kalman filter iterations
    for t in range(nobs):

        # Forecast for time t
        forecast[:, t] = np.dot(Z, predicted_state[:, t])

        # Forecast error for time t
        forecast_error[:, t] = y[:, t] - forecast[:, t]

        # Forecast error covariance matrix and inverse for time t
        tmp1 = np.dot(predicted_state_cov[:, :, t], Z.T)
        forecast_error_cov[:, :, t] = (
            np.dot(Z, tmp1) + H
        )
        forecast_error_cov_inv = np.linalg.inv(forecast_error_cov[:, :, t])
        determinant = np.linalg.det(forecast_error_cov[:, :, t])

        # Filtered state for time t
        tmp2 = np.dot(forecast_error_cov_inv, forecast_error[:,t])
        filtered_state[:, t] = (
            predicted_state[:, t] +
            np.dot(tmp1, tmp2)
        )

        # Filtered state covariance for time t
        tmp3 = np.dot(forecast_error_cov_inv, Z)
        filtered_state_cov[:, :, t] = (
            predicted_state_cov[:, :, t] -
            np.dot(
                np.dot(tmp1, tmp3),
                predicted_state_cov[:, :, t]
            )
        )

        # Loglikelihood
        loglikelihood[t] = -0.5 * (
            np.log((2*np.pi)**k_endog * determinant) +
            np.dot(forecast_error[:, t], tmp2)
        )

        # Predicted state for time t+1
        predicted_state[:, t+1] = np.dot(T, filtered_state[:, t])

        # Predicted state covariance matrix for time t+1
        tmp4 = np.dot(T, filtered_state_cov[:, :, t])
        predicted_state_cov[:, :, t+1] = np.dot(tmp4, T.T) + Q
        
        predicted_state_cov[:, :, t+1] = (
            predicted_state_cov[:, :, t+1] + predicted_state_cov[:, :, t+1].T
        ) / 2

    return (
        filtered_state, filtered_state_cov,
        predicted_state, predicted_state_cov,
        forecast, forecast_error, forecast_error_cov,
        loglikelihood
    )


# So why then did I write nearly 15,000 lines of code to
# contribute Kalman filtering and state-space models to the
# Statsmodels project?
# 
# 1. **Performance**: It should run fast
# 2. **Wrapping**: It should be easy to use
# 3. **Testing**: It should run correctly
# 
# ### Performance
# 
# The Kalman filter basically consists of iterations (loops) and
# matrix operations. It is well known that loops perform poorly
# in interpreted languages like Python<sup>1</sup>, and also that
# matrix operations are ultimately performed by the highly
# optimized [BLAS](http://www.netlib.org/blas/) and
# [LAPACK](http://www.netlib.org/lapack/) libraries, regardless
# of the high-level programming language used.<sup>2</sup> This
# suggests two things:
# 
# - Fast code should be compiled (not interpreted)
# - Fast code should call the BLAS / LAPACK libraries as soon
#    as possible (not through intermediate functions)
# 
# These two things are possible using
# [Cython](http://cython.org/), a simple extension of Python
# syntax that allows compilation to C and direct interaction with
# BLAS and LAPACK. All of the heavy lifting of the Kalman
# filtering I contributed to Statsmodels is performed in Cython,
# which allows for very fast execution.
# 
# It might seem like this approach eliminates the whole benefit
# of using a high-level language like Python - in fact, why not
# just use C or Fortran if we're going to ultimately compile the
# code? First, Cython is quite similar to Python, so future
# maintenance is easier, but more importantly end-user Python
# code can interact with it directly. In this way, we get the
# best of both worlds: the speed of compiled code where
# performance is needed and the ease of interpreted code where it
# isn't.
# 
# An $AR(1)$ model can be written in state space form as
# 
# $$
# \begin{align}
#     y_t & = \alpha_t \\\
#     \alpha_{t+1} & = \phi_1 \alpha_t + \eta_t \qquad \eta_t \sim N(0, \sigma_\eta^2)
# \end{align}
# $$
# 
# and it can specified in Python and the Kalman filter applied
# using the following code:
# 

from scipy.signal import lfilter

# Parameters
nobs = 100
phi = 0.5
sigma2 = 1.0

# Example dataset
np.random.seed(1234)
eps = np.random.normal(scale=sigma2**0.5, size=nobs)
y = lfilter([1], [1, -phi], eps)[np.newaxis, :]

# State space
Z = np.array([1.])
H = np.array([0.])
T = np.array([phi])
Q = np.array([sigma2])

# Initial state distribution
a_0 = np.array([0.])
P_0 = np.array([sigma2 / (1 - phi**2)])

# Run the Kalman filter
res = kalman_filter(y, Z, H, T, Q, a_0, P_0)


# Comparing the above Kalman filter with the implementation in
# Statsmodels for the $AR(1)$ model yields the following runtimes
# in milliseconds for a single filter application, where $nobs$
# is the length of the time series (reasonable measures were
# taken to ensure these timings are meaningful, but not
# extraordinary measures):
# 
# 
# | `nobs`        | Python (ms) | MATLAB (ms) | Cython (ms)   |
# |---------------|-------------|-------------|---------------|
# | $10$   &nbsp; | $0.742$     | $0.326$     | $0.106$       |
# | $10^2$        | $6.39$      | $3.040$     | $0.161$       |
# | $10^3$        | $67.1$      | $32.5$      | $0.668$       |
# | $10^4$        | $662.0$     | $311.3$     | $6.1$         |
# 
# Across hundreds or thousands of iterations (as in maximum
# likelihood estimation or MCMC methods), these differences can
# be substantial. Also, other Kalman filtering methods, such as
# the univariate approach of Koopman and Durbin (2000) used with
# large dimensional observations $y_t$, can add additional inner
# loops, increasing the importance of compiled code.
# 
# ### Wrapping
# 
# One of the main reaons that using Python or MATLAB is
# preferrable to C or Fortran is that code in higher-level 
# lanaguages is more expressive and more readable. Even though
# the performance sensitive code has been written in Cython, we
# want to take advantage of the high-level language features in
# Python proper to make specifying, filtering, and estimating
# parameters of state space models as natural as possible. For
# example, an $ARMA(1,1)$ model can be written in state-space
# form as
# 
# $$
# \begin{align}
#     y_t & = \begin{bmatrix} 1 & \theta_1 \end{bmatrix} \begin{bmatrix} \alpha_{1t} \\ \alpha_{2t} \end{bmatrix} \\    \begin{bmatrix} \alpha_{1t+1} \\ \alpha_{2t+1} \end{bmatrix} & = \begin{bmatrix}
#     \phi_1 & 0 \    1 & 0
#     \end{bmatrix} \begin{bmatrix} \alpha_{1t} \\ \alpha_{2t} \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \eta_t \qquad \eta_t \sim N(0, \sigma_\eta^2)
# \end{align}
# $$
# 
# where $\theta\_1$, $\phi\_1$, and $\sigma\_\eta^2$ are unknown
# parameters. Estimating them via MLE has been made very
# easy in the Statsmodels state space library; the model can be
# specified and estimated with the following code
# 
# **Note**: this code has been updated on July 31, 2015 to
# reflect an update to the Statsmodels code base.
# 
# **Note**: this code has been updated on June 17, 2016 to
# reflect a further update to the Statsmodels code base, and
# also to estimate an ARMA(1,1) model as shown above.
# 

import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm

# True model parameters for an AR(1)
nobs = int(1e3)
true_phi = 0.5
true_sigma = 1**0.5

# Simulate a time series
np.random.seed(1234)
disturbances = np.random.normal(0, true_sigma, size=(nobs,))
endog = lfilter([1], np.r_[1, -true_phi], disturbances)

# Construct the model for an ARMA(1,1)
class ARMA11(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(ARMA11, self).__init__(endog, k_states=2, k_posdef=1,
                                     initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1., 0]
        self['transition'] = [[0, 0],
                              [1., 0]]
        self['selection', 0, 0] = 1.

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(ARMA11, self).update(params, transformed, **kwargs)

        self['design', 0, 1] = params[0]
        self['transition', 0, 0] = params[1]
        self['state_cov', 0, 0] = params[2]

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0.,0.,1]  # these are very simple

# Create and fit the model
mod = ARMA11(endog)
res = mod.fit()
print(res.summary())


# Whereas the above example showed an ad-hoc creation and
# estimation of a specific model, the power of object-oriented
# programming in Python can be leveraged to create generic and
# reusable estimation classes. For example, for the common class
# of (Seasonal) Autoregressive Integrated Moving Average models
# (optionally with exogenous regressors), an `SARIMAX` class has
# been written to automate the creation and estimation of those
# types of models. For example, an
# $SARIMA(1,1,1) \times (0,1,1,4)$ model of GDP can be specified
# and estimated as (an added bonus is that we can download the
# GDP data on-the-fly from FRED using Pandas):
# 
# **Note**: this code has been updated on June 17, 2016 to
# reflect an update to the Statsmodels code base and to use the
# `pandas_datareader` package.
# 

import statsmodels.api as sm
from pandas_datareader.data import DataReader

gdp = DataReader('GDPC1', 'fred', start='1959', end='12-31-2014')

# Create the model, here an SARIMA(1,1,1) x (0,1,1,4) model
mod = sm.tsa.SARIMAX(gdp, order=(1,1,1), seasonal_order=(0,1,1,4))

# Fit the model via maximum likelihood
res = mod.fit()
print(res.summary())


# This type of built-in model should be familiar to those who
# work with programs like Stata (which also has a built-in
# SARIMAX model). The benefit of Python and Statsmodels is that
# you can build *your own* classes of models which behave just
# as smoothly and seamlessly as those that are "built-in". By
# building on top of the state space functionality in
# Statsmodels, you get a lot for free while still retaining the
# flexibility to write any kind of model you want.
# 
# For example, a local linear trend model can be created for
# re-use in the following way:
# 
# **Note**: this code has been updated on June 17, 2016 to
# reflect an update to the Statsmodels code base.
# 

"""
Univariate Local Linear Trend Model
"""
import pandas as pd

class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, trend=True):
        # Model properties
        self.trend = trend

        # Model order
        k_states = 2
        k_posdef = 1 + self.trend

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
        )

        # Initialize the matrices
        self['design'] = np.array([1, 0])
        self['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self['selection'] = np.eye(k_states)[:,:k_posdef]

        # Initialize the state space model as approximately diffuse
        self.initialize_approximate_diffuse()
        # Because of the diffuse initialization, burn first two loglikelihoods
        self.loglikelihood_burn = 2

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

        # The parameters depend on whether or not we have a trend
        param_names = ['sigma2.measurement', 'sigma2.level']
        if self.trend:
            param_names += ['sigma2.trend']
        self._param_names = param_names

    @property
    def start_params(self):
        return [0.1] * (2 + self.trend)

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

        # Observation covariance
        self['obs_cov',0,0] = params[0]

        # State covariance
        self[self._state_cov_idx] = params[1:]


# Now, we have a generic class that can fit local linear trend
# models (if `trend=True`) and also local level models (if
# `trend=False`). For example, we can model the annual flow
# volume of the Nile river using a local linear trend model:
# 

y = sm.datasets.nile.load_pandas().data
y.index = pd.date_range('1871', '1970', freq='AS')

mod1 = LocalLinearTrend(y['volume'], trend=True)
res1 = mod1.fit()
print res1.summary()


# It looks as though the presense of a stochastic trend
# is not adding anything to the model (and the parameter is not
# estimated well in any case) - refitting without the trend is
# easy:
# 

mod2 = LocalLinearTrend(y['volume'], trend=False)
res2 = mod2.fit()
print res2.summary()


# Instead of constructing our own custom class, this particular example could be estimated using the `UnobservedComponents` model in the Statsmodels state space library.
# 
# ### Testing
# 
# It is no good to have fast code that is easy to use if it gives
# the wrong answer. For that reason, a large part of creating
# production ready code is constructing unit tests comparing
# the module's output to known values to make sure everything
# works. The state space model code in Statsmodels has 455 unit
# tests covering everything from the filter output
# (`filtered_state`, `logliklelihood`, etc.) to state space
# creation (e.g. the `SARIMAX` class) and maximum likelihood
# estimation (estimated parameters, maximized likelihood values,
# standard errors, etc.).
# 
# ### Bibliography
# 
# Durbin, James, and Siem Jan Koopman. 2012.
# Time Series Analysis by State Space Methods: Second Edition.
# Oxford University Press.
# 
# Koopman, S. J., and J. Durbin. 2000.
# “Fast Filtering and Smoothing for Multivariate State Space Models.”
# Journal of Time Series Analysis 21 (3): 281–96.
# 
# ### Footnotes
# 
# [1] This can be improved with a JIT compiler like
# [Numba](http://numba.pydata.org/).
# 
# [2] Python, MATLAB, Mathematica, Stata, Gauss, Ox, etc. all
# ultimately rely on BLAS and LAPACK libraries for performing
# operations on matrices.
# 
# [3] See Durbin and Koopman (2012) for notation.
# 
# [4] A [proposal](http://legacy.python.org/dev/peps/pep-0465/)
# is in place to create an infix matrix multiplication
# operator in Python
# 

# ### The initialization problem
# 
# The Kalman filter is a recursion for optimally making inferences about an unknown state variable given a related observed variable. In particular, if the state variable at time $t$ is represented by $\alpha_t$, then the (linear, Gaussian) Kalman filter takes as input the mean and variance of that state conditional on observations up to time $t-1$ and provides as output the filtered mean and variance of the state at time $t$ and the predicted mean and variance of the state at time $t$.
# 
# More concretely, we denote (see Durbin and Koopman (2012) for all notation)
# 
# $$
# \begin{align}
# \alpha_t \mid Y_{t-1} & \sim N(a_t, P_t) \\alpha_t \mid Y_{t} & \sim N(a_{t|t}, P_{t|t}) \\alpha_{t+1} \mid Y_{t} & \sim N(a_{t+1}, P_{t+1}) \\end{align}
# $$
# 
# Then the inputs to the Kalman filter recursion are $a_t$ and $P_t$ and the outputs are $a_{t \mid t}, P_{t \mid t}$ (called *filtered* values) and $a_{t+1}, P_{t+1}$ (called *predicted* values).
# 
# This process is done for $t = 1, \dots, n$. While the predicted values as outputs of the recursion are available as inputs to subsequent iterations, an important question is *initialization*: what values should be used as inputs to start the very first recursion.
# 
# Specifically, when running the recursion for $t = 1$, we need as inputs $a_1, P_1$. These values define, respectively, the expectation and variance / covariance matrix for the initial state $\alpha_1 \mid Y_0$. Here, though, $Y_0$ denotes the observation of *no data*, so in fact we are looking for the *unconditional* expectation and variance / covariance matrix of $\alpha_1$. The question is how to find these.
# 
# In general this is a rather difficult problem (for example for non-stationary proceses) but for stationary processes, an analytic solution can be found.
# 

# ### Stationary processes
# 
# A (covariance) stationary process is, very roughly speaking, one for which the mean and covariances are not time-dependent. What this means is that we can solve for the unconditional expectation and variance explicity (this section results from Hamilton (1994), Chapter 13)
# 
# The state equation for a state-space process (to which the Kalman filter is applied) is
# 
# $$
# \alpha_{t+1} = T \alpha_t + \eta_t
# $$
# 
# Below I set up the elements of a typical state equation like that which would be found in the ARMA case, where the transition matrix $T$ is a sort-of companion matrix. I'm setting it up in such a way that I'll be able to adjust the dimension of the state, so we can see how some of the below methods scale.
# 

import numpy as np
from scipy import linalg

def state(m=10):
    T = np.zeros((m, m), dtype=complex)
    T[0,0] = 0.6 + 1j
    idx = np.diag_indices(m-1)
    T[(idx[0]+1, idx[1])] = 1
    
    Q = np.eye(m)
    
    return T, Q


# #### Unconditional mean
# 
# Taking unconditional expectations of both sides yields:
# 
# $$
# E[\alpha_{t+1}] = T E[ \alpha_t] + E[\eta_t]
# $$
# 
# or $(I - T) E[\alpha_t] = 0$ and given stationarity this means that the unique solution is $E[\alpha_t] = 0$ for all $t$. Thus in initializing the Kalman filter, we set $a_t = E[\alpha_t] = 0$.
# 
# #### Unconditional variance / covariance matrix
# 
# Slightly more tricky is the variance / covariance matrix. To find it (as in Hamilton) post-multiply by the transpose of the state and take expectations:
# 
# $$
# E[\alpha_{t+1} \alpha_{t+1}'] = E[(T \alpha_t + \eta_t)(\alpha_t' T' + \eta_t')]
# $$
# 
# This yields an equation of the form (denoting by $\Sigma$ and $Q$ the variance / covariance matrices of the state and disturbance):
# 
# $$
# \Sigma = T \Sigma T' + Q
# $$
# 
# Hamilton then shows that this equation can be solved as:
# 
# $$
# vec(\Sigma) = [I - (T \otimes T)]^{-1} vec(Q)
# $$
# 
# where $\otimes$ refers to the Kronecker product. There are two things that jump out about this equation:
# 
# 1. It can be easily solved. In Python, it would look something like:
#    ```python
#    m = T.shape[0]
#    Sigma = np.linalg.inv(np.eye(m**2) - np.kron(T, T)).dot(Q.reshape(Q.size, 1)).reshape(n,n)
#    ```
# 2. It will scale very poorly (in terms of computational time) with the dimension of the state-space ($m$). In particular, you have to take the inverse of an $m^2 \times m^2$ matrix.
# 
# Below I take a look at the timing for solving it this way using the code above (`direct_inverse`) and using built-in scipy direct method (which uses a linear solver rather than taking the inverse, so it is a bit faster)s
# 

def direct_inverse(A, Q):
    n = A.shape[0]
    return np.linalg.inv(np.eye(n**2) - np.kron(A,A.conj())).dot(Q.reshape(Q.size, 1)).reshape(n,n)

def direct_solver(A, Q):
    return linalg.solve_discrete_lyapunov(A, Q)

# Example
from numpy.testing import assert_allclose
np.set_printoptions(precision=10)
T, Q = state(3)
sol1 = direct_inverse(T, Q)
sol2 = direct_solver(T, Q)

assert_allclose(sol1,sol2)


# Timings for m=1
T, Q = state(1)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')


# Timings for m=5
T, Q = state(5)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')


# Timings for m=10
T, Q = state(10)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')


# Timings for m=50
T, Q = state(50)
get_ipython().magic('timeit direct_inverse(T, Q)')
get_ipython().magic('timeit direct_solver(T, Q)')


# ### Lyapunov equations
# 
# As you can notice by looking at the name of the scipy function, the equation describing the unconditional variance / covariance matrix, $\Sigma = T \Sigma T' + Q$ is an example of a discrete Lyapunov equation.
# 
# One place to turn to improve performance on matrix-related operations is to the underlying Fortran linear algebra libraries: BLAS and LAPACK; if there exists a special-case solver for discrete time Lyapunov equations, we can call that function and be done.
# 
# Unfortunately, no such function exists, but what does exist is a special-case solver for *Sylvester* equations (\*trsyl), which are equations of the form $AX + XB = C$. Furthermore, the *continuous* Lyapunov equation, $AX + AX^H + Q = 0$ is a special case of a Sylvester equation. Thus if we can transform the discrete to a continuous Lyapunov equation, we can then solve it quickly as a Sylvester equation.
# 
# The current implementation of the scipy discrete Lyapunov solver does not do that, although their continuous solver `solve_lyapunov` does call `solve_sylvester` which calls \*trsyl. So, we need to find a transformation from discrete to continuous and directly call `solve_lyapunov` which will do the heavy lifting for us.
# 
# It turns out that there are several transformations that will do it. See Gajic, Z., and M.T.J. Qureshi. 2008. for details. Below I present two bilinear transformations, and show their timings.
# 

def bilinear1(A, Q):
    A = A.conj().T
    n = A.shape[0]
    eye = np.eye(n)
    B = np.linalg.inv(A - eye).dot(A + eye)
    res = linalg.solve_lyapunov(B.conj().T, -Q)
    return 0.5*(B - eye).conj().T.dot(res).dot(B - eye)

def bilinear2(A, Q):
    A = A.conj().T
    n = A.shape[0]
    eye = np.eye(n)
    AI_inv = np.linalg.inv(A + eye)
    B = (A - eye).dot(AI_inv)
    C = 2*np.linalg.inv(A.conj().T + eye).dot(Q).dot(AI_inv)
    return linalg.solve_lyapunov(B.conj().T, -C)

# Example:
T, Q = state(3)
sol3 = bilinear1(T, Q)
sol4 = bilinear2(T, Q)

assert_allclose(sol1,sol3)
assert_allclose(sol3,sol4)


# Timings for m=1
T, Q = state(1)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')


# Timings for m=5
T, Q = state(5)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')


# Timings for m=10
T, Q = state(10)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')


# Timings for m=50
T, Q = state(50)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')


# Notice that this method does so well we can even try $m=500$.
# 

# Timings for m=500
T, Q = state(500)
get_ipython().magic('timeit bilinear1(T, Q)')
get_ipython().magic('timeit bilinear2(T, Q)')


# ### Final thoughts
# 
# The first thing to notice is *how much better* the bilinear transformations do as $m$ grows large. They are able to take advantage of the special formulation of the problem so as to avoid many calculations that a generic inverse (or linear solver) would have to do. Second, though, for small $m$, the original analytic solutions are actually better.
# 
# I have submitted a [pull request to Scipy](https://github.com/scipy/scipy/pull/3748) to augment the `solve_discrete_lyapunov` for large $m$ ($m >= 10$) using the second bilinear transformation to solve it as a Sylvester equation.
# 

# # ARMA(1, 1) - CPI Inflation
# 
# This notebook contains the example code from "State Space Estimation of Time Series Models in Python: Statsmodels" for the ARMA(1, 1) model of CPI inflation.
# 

# These are the basic import statements to get the required Python functionality
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ## Data
# 
# For this example, we consider modeling quarterly CPI inflation. Below, we retrieve the data directly from the [Federal Reserve Economic Database](https://fred.stlouisfed.org/) using the `pandas_datareader` package.
# 

# Get the data from FRED
from pandas_datareader.data import DataReader
cpi = DataReader('CPIAUCNS', 'fred', start='1971-01', end='2016-12')
cpi.index = pd.DatetimeIndex(cpi.index, freq='MS')
inf = np.log(cpi).resample('QS').mean().diff()[1:] * 400

# Plot the series to see what it looks like
fig, ax = plt.subplots(figsize=(13, 3), dpi=300)
ax.plot(inf.index, inf, label=r'$\Delta \log CPI$')
ax.legend(loc='lower left')
ax.yaxis.grid();


# ## State space model
# 
# The ARMA(1, 1) model is:
# 
# $$
# y_t = \phi y_{t-1} + \varepsilon_t + \theta_1 \varepsilon_{t-1}, \qquad \varepsilon_t \sim N(0, \sigma^2)
# $$
# 
# and it can be written in state-space form as:
# 
# $$
# \begin{align}
# y_t & = \underbrace{\begin{bmatrix} 1 & \theta_1 \end{bmatrix}}_{Z} \underbrace{\begin{bmatrix} \alpha_{1,t} \\ \alpha_{2,t} \end{bmatrix}}_{\alpha_t} \    \begin{bmatrix} \alpha_{1,t+1} \\ \alpha_{2,t+1} \end{bmatrix} & = \underbrace{\begin{bmatrix}
#         \phi & 0 \        1      & 0     \    \end{bmatrix}}_{T} \begin{bmatrix} \alpha_{1,t} \\ \alpha_{2,t} \end{bmatrix} +
#     \underbrace{\begin{bmatrix} 1 \\ 0 \end{bmatrix}}_{R} \underbrace{\varepsilon_{t+1}}_{\eta_t} \\end{align}
# $$
# 
# Below we construct a custom class, `ARMA11`, to estimate the ARMA(1, 1) model.
# 

from statsmodels.tsa.statespace.tools import (constrain_stationary_univariate,
                                              unconstrain_stationary_univariate)

class ARMA11(sm.tsa.statespace.MLEModel):
    start_params = [0, 0, 1]
    param_names = ['phi', 'theta', 'sigma2']

    def __init__(self, endog):
        super(ARMA11, self).__init__(
            endog, k_states=2, k_posdef=1, initialization='stationary')

        self['design', 0, 0] = 1.
        self['transition', 1, 0] = 1.
        self['selection', 0, 0] = 1.

    def transform_params(self, params):
        phi = constrain_stationary_univariate(params[0:1])
        theta = constrain_stationary_univariate(params[1:2])
        sigma2 = params[2]**2
        return np.r_[phi, theta, sigma2]

    def untransform_params(self, params):
        phi = unconstrain_stationary_univariate(params[0:1])
        theta = unconstrain_stationary_univariate(params[1:2])
        sigma2 = params[2]**0.5
        return np.r_[phi, theta, sigma2]

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(ARMA11, self).update(params, **kwargs)

        self['design', 0, 1] = params[1]
        self['transition', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[2]


# ## Maximum likelihood estimation
# 
# With this class, we can instantiate a new object with the inflation data and fit the model by maximum likelihood methods.
# 

inf_model = ARMA11(inf)
inf_results = inf_model.fit()

print(inf_results.summary())


# Notice that the diagnostic tests reported in the lower table suggest that our residuals do not appear to be white noise - in particular, we can reject at the 5% level the null hypotheses of serial independence (Ljung-Box test), homoskedasticity (Heteroskedasticity test), and normality (Jarque-Bera test).
# 
# To further investicate the residuals, we can produce diagnostic plots.
# 

inf_results.plot_diagnostics(figsize=(13, 5));


# We can also produce in-sample one-step-ahead predictions and out-of-sample forecasts:
# 

# Construct the predictions / forecasts
inf_forecast = inf_results.get_prediction(start='2005-01-01', end='2020-01-01')

# Plot them
fig, ax = plt.subplots(figsize=(13, 3), dpi=300)

forecast = inf_forecast.predicted_mean
ci = inf_forecast.conf_int(alpha=0.5)

ax.fill_between(forecast.ix['2017-01-02':].index, -3, 7, color='grey',
                alpha=0.15)
lines, = ax.plot(forecast.index, forecast)
ax.fill_between(forecast.index, ci['lower CPIAUCNS'], ci['upper CPIAUCNS'],
                alpha=0.2)

p1 = plt.Rectangle((0, 0), 1, 1, fc="white")
p2 = plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3)
ax.legend([lines, p1, p2], ["Predicted inflation",
                            "In-sample one-step-ahead predictions",
                            "Out-of-sample forecasts"], loc='upper left')
ax.yaxis.grid()


# And we can produce impulse responses.
# 

# Construct the impulse responses
inf_irfs = inf_results.impulse_responses(steps=10)

print(inf_irfs)


# ## ARMA(1, 1) in Statsmodels via SARIMAX
# 
# The large class of seasonal autoregressive integrated moving average models - SARIMAX(p, d, q)x(P, D, Q, S) - is implemented in Statsmodels in the `sm.tsa.SARIMAX` class.
# 
# First, we'll check that fitting an ARMA(1, 1) model by maximum likelihood using `sm.tsa.SARIMAX` gives the same results as our `ARMA11` class, above.
# 

inf_model2 = sm.tsa.SARIMAX(inf, order=(1, 0, 1))
inf_results2 = inf_model2.fit()

print(inf_results2.summary())


# ## Metropolis-Hastings - ARMA(1, 1)
# 
# Here we show how to estimate the ARMA(1, 1) model via Metropolis-Hastings using PyMC. Recall that the ARMA(1, 1) model has three parameters: $(\phi, \theta, \sigma^2)$.
# 
# For $\phi$ and $\theta$ we specify uniform priors of $(-1, 1)$, and for $1 / \sigma^2$ we specify a $\Gamma(2, 4)$ prior.
# 

import pymc as mc

# Priors
prior_phi = mc.Uniform('phi', -1, 1)
prior_theta = mc.Uniform('theta', -1, 1)
prior_precision = mc.Gamma('precision', 2, 4)

# Create the model for likelihood evaluation
model = sm.tsa.SARIMAX(inf, order=(1, 0, 1))

# Create the "data" component (stochastic and observed)
@mc.stochastic(dtype=sm.tsa.statespace.MLEModel, observed=True)
def loglikelihood(value=model, phi=prior_phi, theta=prior_theta, precision=prior_precision):
    return value.loglike([phi, theta, 1 / precision])

# Create the PyMC model
pymc_model = mc.Model((prior_phi, prior_theta, prior_precision, loglikelihood))

# Create a PyMC sample and perform sampling
sampler = mc.MCMC(pymc_model)
sampler.sample(iter=10000, burn=1000, thin=10)


# Plot traces
mc.Matplot.plot(sampler)


# ## Gibbs Sampling - ARMA(1, 1)
# 
# Here we show how to estimate the ARMA(1, 1) model via Metropolis-within-Gibbs Sampling.
# 

from scipy.stats import multivariate_normal, invgamma

def draw_posterior_phi(model, states, sigma2):
    Z = states[0:1, 1:]
    X = states[0:1, :-1]

    tmp = np.linalg.inv(sigma2 * np.eye(1) + np.dot(X, X.T))
    post_mean = np.dot(tmp, np.dot(X, Z.T))
    post_var = tmp * sigma2

    return multivariate_normal(post_mean, post_var).rvs()

def draw_posterior_sigma2(model, states, phi):
    resid = states[0, 1:] - phi * states[0, :-1]
    post_shape = 3 + model.nobs
    post_scale = 3 + np.sum(resid**2)

    return invgamma(post_shape, scale=post_scale).rvs()

np.random.seed(17429)

from scipy.stats import norm, uniform
from statsmodels.tsa.statespace.tools import is_invertible

# Create the model for likelihood evaluation and the simulation smoother
model = ARMA11(inf)
sim_smoother = model.simulation_smoother()

# Create the random walk and comparison random variables
rw_proposal = norm(scale=0.3)

# Create storage arrays for the traces
n_iterations = 10000
trace = np.zeros((n_iterations + 1, 3))
trace_accepts = np.zeros(n_iterations)
trace[0] = [0, 0, 1.]  # Initial values

# Iterations
for s in range(1, n_iterations + 1):
    # 1. Gibbs step: draw the states using the simulation smoother
    model.update(trace[s-1], transformed=True)
    sim_smoother.simulate()
    states = sim_smoother.simulated_state[:, :-1]

    # 2. Gibbs step: draw the autoregressive parameters, and apply
    # rejection sampling to ensure an invertible lag polynomial
    phi = draw_posterior_phi(model, states, trace[s-1, 2])
    while not is_invertible([1, -phi]):
        phi = draw_posterior_phi(model, states, trace[s-1, 2])
    trace[s, 0] = phi

    # 3. Gibbs step: draw the variance parameter
    sigma2 = draw_posterior_sigma2(model, states, phi)
    trace[s, 2] = sigma2

    # 4. Metropolis-step for the moving-average parameter
    theta = trace[s-1, 1]
    proposal = theta + rw_proposal.rvs()
    if proposal > -1 and proposal < 1:
        acceptance_probability = np.exp(
            model.loglike([phi, proposal, sigma2]) -
            model.loglike([phi, theta, sigma2]))

        if acceptance_probability > uniform.rvs():
            theta = proposal
            trace_accepts[s-1] = 1
    trace[s, 1] = theta
    
# For analysis, burn the first 1000 observations, and only
# take every tenth remaining observation
burn = 1000
thin = 10
final_trace = trace[burn:][::thin]


from scipy.stats import gaussian_kde

fig, axes = plt.subplots(2, 2, figsize=(13, 5), dpi=300)

phi_kde = gaussian_kde(final_trace[:, 0])
theta_kde = gaussian_kde(final_trace[:, 1])
sigma2_kde = gaussian_kde(final_trace[:, 2])

axes[0, 0].hist(final_trace[:, 0], bins=20, normed=True, alpha=1)
X = np.linspace(0.75, 1.0, 5000)
line, = axes[0, 0].plot(X, phi_kde(X))
ylim = axes[0, 0].get_ylim()
vline = axes[0, 0].vlines(final_trace[:, 0].mean(), ylim[0], ylim[1],
                          linewidth=2)
axes[0, 0].set(title=r'$\phi$')

axes[0, 1].hist(final_trace[:, 1], bins=20, normed=True, alpha=1)
X = np.linspace(-0.9, 0.0, 5000)
axes[0, 1].plot(X, theta_kde(X))
ylim = axes[0, 1].get_ylim()
vline = axes[0, 1].vlines(final_trace[:, 1].mean(), ylim[0], ylim[1],
                          linewidth=2)
axes[0, 1].set(title=r'$\theta$')

axes[1, 0].hist(final_trace[:, 2], bins=20, normed=True, alpha=1)
X = np.linspace(4, 8.5, 5000)
axes[1, 0].plot(X, sigma2_kde(X))
ylim = axes[1, 0].get_ylim()
vline = axes[1, 0].vlines(final_trace[:, 2].mean(), ylim[0], ylim[1],
                          linewidth=2)
axes[1, 0].set(title=r'$\sigma^2$')

p1 = plt.Rectangle((0, 0), 1, 1, alpha=0.7)
axes[0, 0].legend([p1, line, vline],
                  ["Histogram", "Gaussian KDE", "Sample mean"],
                  loc='upper left')

axes[1, 1].plot(final_trace[:, 0], label=r'$\phi$')
axes[1, 1].plot(final_trace[:, 1], label=r'$\theta$')
axes[1, 1].plot(final_trace[:, 2], label=r'$\sigma^2$')
axes[1, 1].legend(loc='upper left')
axes[1, 1].set(title=r'Trace plots')
fig.tight_layout()


# ## Expanded model: SARIMAX
# 
# We'll try a more complicated model now: SARIMA(3, 0, 0)x(0, 1, 1, 4). The ability to include a seasonal effect is important, since the data series we selected is not seasonally adjusted.
# 
# We also add an explanatory "impulse" variable to account for the clear outlier in the fourth quarter of 2008. We'll estimate this model by maximum likelihood.
# 

outlier_exog = pd.Series(np.zeros(len(inf)), index=inf.index, name='outlier')
outlier_exog['2008-10-01'] = 1
inf_model3 = sm.tsa.SARIMAX(inf, order=(3, 0, 0), seasonal_order=(0, 1, 1, 4), exog=outlier_exog)
inf_results3 = inf_model3.fit()

print(inf_results3.summary())


inf_results3.plot_diagnostics(figsize=(13, 5));


# In terms of the reported model diagnostics, this model appears to do a better job of explaining the inflation series - although it is far from perfect.
# 

# Construct the predictions / forecasts
# Notice that now to forecast we need to provide an expanded
# outlier_exog series for the new observations
outlier_exog_fcast = np.zeros((13,1))
inf_forecast3 = inf_results3.get_prediction(
    start='2005-01-01', end='2020-01-01', exog=outlier_exog_fcast)

# Plot them
fig, ax = plt.subplots(figsize=(13, 3), dpi=300)

forecast = inf_forecast3.predicted_mean
ci = inf_forecast3.conf_int(alpha=0.5)

ax.fill_between(forecast.ix['2017-01-02':].index, -10, 7, color='grey',
                alpha=0.15)
lines, = ax.plot(forecast.index, forecast)
ax.fill_between(forecast.index, ci['lower CPIAUCNS'], ci['upper CPIAUCNS'],
                alpha=0.2)

p1 = plt.Rectangle((0, 0), 1, 1, fc="white")
p2 = plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3)
ax.legend([lines, p1, p2], ["Predicted inflation",
                            "In-sample one-step-ahead predictions",
                            "Out-of-sample forecasts"], loc='lower left')
ax.yaxis.grid()


