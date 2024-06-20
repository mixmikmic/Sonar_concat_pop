# Copyright &copy; 2015 Ondrej Martinsky, All rights reserved
# 
# [www.quantandfinancial.com](http://www.quantandfinancial.com)
# # LU Decomposition
# 

get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, multiply, dot
from scipy.linalg import lu


# ### Gaussian Elimination
# Code for calculating **X** from **M*X=Y** where *M* is either lower or upper diagonal matrix
# 

def solve_l(m, y):  # solves x from m*x = y
    assert (m==tril(m)).all()        # assert matrix is lower diagonal
    assert (m.shape[0]==m.shape[1])  # Assert matrix is square matrix
    N=m.shape[0]
    x=zeros(N)                      # Vector of roots
    for r in range(N):
        s = 0
        for c in range(r):
            s += m[r,c]*x[c]            
        x[r] = (y[r]-s) / m[r,r]
    assert allclose(dot(m,x), y)    # Check solution
    return x

def solve_u(m, y):
    m2 = fliplr(flipud(m))     # flip matrix LR and UD, so upper diagonal matrix becomes lower diagonal
    y2 = y[::-1]               # flip array
    x2 = solve(m2, y2)
    x = x2[::-1]
    assert allclose(dot(m,x), y) # Check solution
    return x

def solve(m, y):
    if (m==tril(m)).all():
        return solve_l(m,y)
    else:
        return solve_u(m,y)


# ### Solving using L U decomposition
# 

# Unknowns
x_org = array([2, 4, 1])
print(x_org)


# Coefficients
m = array([[2,-1,1],[3,3,9],[3,3,5]])
print(m)


# Results
y = dot(m,x_org)
print(y)


# Note: matrix dot-product is not commutative, but is associative
p, l, u = lu(m, permute_l=False)
pl, u = lu(m, permute_l=True)
assert (dot(p,l)==pl).all()
assert (dot(pl,u)==m).all()
assert (pinv(p)==p).all()


print(l) # Lower diagonal matrix, zero element above the principal diagonal


print(u) # Upper diagnonal matrix, zero elements below the principal diagonal


print(p) # Permutation matrix for "l"


assert (l*u==multiply(l,u)).all()          # memberwise multiplication
assert (m==dot(dot(p,l),u)).all()          # matrix multiplication, M=LU


assert (pinv(p)==p).all()
#   P*L*U*X = Y
#   L*U*X = pinv(P)*Y
#   set Z=U*X
#   L*Z = P*Y (solve Z)
z = solve(l, dot(p,y))
#   solve X from U*X=Z
x = solve(u, z)


assert allclose(x_org,x)
print(x)





# Copyright &copy; 2012 Ondrej Martinsky, All rights reserved
# 
# [www.quantandfinancial.com](http://www.quantandfinancial.com)
# # Time Value of Money
# 

# ### Implementation
# scroll below for examples
# 

from optimization import newton
class TVM:
    bgn, end = 0, 1

    def __str__(self):
        return "n=%f, r=%f, pv=%f, pmt=%f, fv=%f" % (
            self.n, self.r, self.pv, self.pmt, self.fv)

    def __init__(self, n=0.0, r=0.0, pv=0.0, pmt=0.0, fv=0.0, mode=end):
        self.n = float(n)
        self.r = float(r)
        self.pv = float(pv)
        self.pmt = float(pmt)
        self.fv = float(fv)
        self.mode = mode

    def calc_pv(self):
        z = pow(1 + self.r, -self.n)
        pva = self.pmt / self.r
        if (self.mode == TVM.bgn): pva += self.pmt
        return -(self.fv * z + (1 - z) * pva)

    def calc_fv(self):
        z = pow(1 + self.r, -self.n)
        pva = self.pmt / self.r
        if (self.mode == TVM.bgn): pva += self.pmt
        return -(self.pv + (1 - z) * pva) / z

    def calc_pmt(self):
        z = pow(1 + self.r, -self.n)
        if self.mode == TVM.bgn:
            return (self.pv + self.fv * z) * self.r / (z - 1) / (1 + self.r)
        else:
            return (self.pv + self.fv * z) * self.r / (z - 1)

    def calc_n(self):
        pva = self.pmt / self.r
        if (self.mode == TVM.bgn): pva += self.pmt
        z = (-pva - self.pv) / (self.fv - pva)
        return -log(z) / log(1 + self.r)

    def calc_r(self):
        def function_fv(r, self):
            z = pow(1 + r, -self.n)
            pva = self.pmt / r
            if (self.mode == TVM.bgn): pva += self.pmt
            return -(self.pv + (1 - z) * pva) / z
        return newton(f=function_fv, fArg=self, x0=.05,
                      y=self.fv, maxIter=1000, minError=0.0001)


# ### Calculation of monthly payments
# 

TVM(n=25*12, r=.04/12, pv=500000, fv=0).calc_pmt()


# ### Internal Rate of Return
# 

TVM(n=10*2, pmt=6/2, pv=-80, fv=100).calc_r()


# ### How much I can borrow ?
# 

TVM(n=25*12, r=.04/12, pmt=-2000, fv=0).calc_pv()


# Copyright &copy; 2013 Ondrej Martinsky, All rights reserved
# 
# [www.quantandfinancial.com](http://www.quantandfinancial.com)
# # Portfolio Optimization and Black-Litterman
# 

# #### Mathematical symbols used in this notebook
# 
# | Python symbol | Math Symbol | Comment
# | -- | -- | --
# | rf | $r$ | risk free rate
# | lmb | $\lambda$ | risk aversion coefficient
# | C | $C$ | Assets covariance matrix
# | V | $V$ | Assets variances (diagonal in covariance matrix)
# | W | $W$ | Assets weights
# | R | $R$ | Assets returns
# | mean | $\mu$ | Portfolio historical return
# | var | $\sigma$ | Portfolio historical variance
# | Pi | $\Pi$ | Portfolio equilibrium excess returns
# | tau | $\tau$ | Scaling factor for Black-litterman
# 

get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = (10, 6)
import scipy.optimize
from pandas import *


# ## Few helper functions
# 

# Calculates portfolio mean return
def port_mean(W, R):
    return sum(R * W)

# Calculates portfolio variance of returns
def port_var(W, C):
    return dot(dot(W, C), W)

# Combination of the two functions above - mean and variance of returns calculation
def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)


# Given risk-free rate, assets returns and covariances, this function calculates
# mean-variance frontier and returns its [x,y] points in two arrays
def solve_frontier(R, C, rf):
    def fitness(W, R, C, r):
        # For given level of return r, find weights which minimizes portfolio variance.
        mean, var = port_mean_var(W, R, C)
        penalty = 100 * abs(
            mean - r)  # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
        return var + penalty

    frontier_mean, frontier_var = [], []
    n = len(R)  # Number of assets in the portfolio
    for r in linspace(min(R), max(R), num=20):  # Iterate through the range of returns on Y axis
        W = ones([n]) / n  # start optimization with equal weights
        b_ = [(0, 1) for i in range(n)]
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
        optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success:
            raise BaseException(optimized.message)
        # add point to the efficient frontier [x,y] = [optimized.x, r]
        frontier_mean.append(r)
        frontier_var.append(port_var(optimized.x, C))
    return array(frontier_mean), array(frontier_var)


# Given risk-free rate, assets returns and covariances, this function calculates
# weights of tangency portfolio with respect to sharpe ratio maximization
def solve_weights(R, C, rf):
    def fitness(W, R, C, rf):
        mean, var = port_mean_var(W, R, C)  # calculate mean/variance of the portfolio
        util = (mean - rf) / sqrt(var)  # utility = Sharpe ratio
        return 1 / util  # maximize the utility, minimize its inverse value
    n = len(R)
    W = ones([n]) / n  # start optimization with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights for boundaries between 0%..100%. No leverage, no shorting
    c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # Sum of weights must be 100%
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success: raise BaseException(optimized.message)
    return optimized.x


class Result:
    def __init__(self, W, tan_mean, tan_var, front_mean, front_var):
        self.W=W
        self.tan_mean=tan_mean
        self.tan_var=tan_var
        self.front_mean=front_mean
        self.front_var=front_var
        
def optimize_frontier(R, C, rf):
    W = solve_weights(R, C, rf)
    tan_mean, tan_var = port_mean_var(W, R, C)  # calculate tangency portfolio
    front_mean, front_var = solve_frontier(R, C, rf)  # calculate efficient frontier
    # Weights, Tangency portfolio asset means and variances, Efficient frontier means and variances
    return Result(W, tan_mean, tan_var, front_mean, front_var)

def display_assets(names, R, C, color='black'):
    scatter([C[i, i] ** .5 for i in range(n)], R, marker='x', color=color), grid(True)  # draw assets
    for i in range(n): 
        text(C[i, i] ** .5, R[i], '  %s' % names[i], verticalalignment='center', color=color) # draw labels

def display_frontier(result, label=None, color='black'):
    text(result.tan_var ** .5, result.tan_mean, '   tangent', verticalalignment='center', color=color)
    scatter(result.tan_var ** .5, result.tan_mean, marker='o', color=color), grid(True)
    plot(result.front_var ** .5, result.front_mean, label=label, color=color), grid(True)  # draw efficient frontier


# ## Load historical prices
# 

# Function loads historical stock prices of nine major S&P companies and returns them together
# with their market capitalizations, as of 2013-07-01
def load_data():
    symbols = ['XOM', 'AAPL', 'MSFT', 'JNJ', 'GE', 'GOOG', 'CVX', 'PG', 'WFC']
    cap = {'XOM': 403.02e9, 'AAPL': 392.90e9, 'MSFT': 283.60e9, 'JNJ': 243.17e9, 'GE': 236.79e9,
           'GOOG': 292.72e9, 'CVX': 231.03e9, 'PG': 214.99e9, 'WFC': 218.79e9}
    n = len(symbols)
    prices_out, caps_out = [], []
    for s in symbols:
        dataframe = pandas.read_csv('data/%s.csv' % s, index_col=None, parse_dates=['date'])
        prices = list(dataframe['close'])[-500:] # trailing window 500 days
        prices_out.append(prices)
        caps_out.append(cap[s])
    return symbols, prices_out, caps_out

names, prices, caps = load_data()
n = len(names)


# ## Estimate assets historical return and covariances
# 

# Function takes historical stock prices together with market capitalizations and
# calculates weights, historical returns and historical covariances
def assets_historical_returns_and_covariances(prices):
    prices = matrix(prices)  # create numpy matrix from prices
    # create matrix of historical returns
    rows, cols = prices.shape
    returns = empty([rows, cols - 1])
    for r in range(rows):
        for c in range(cols - 1):
            p0, p1 = prices[r, c], prices[r, c + 1]
            returns[r, c] = (p1 / p0) - 1
    # calculate returns
    expreturns = array([])
    for r in range(rows):
        expreturns = append(expreturns, numpy.mean(returns[r]))
    # calculate covariances
    covars = cov(returns)
    expreturns = (1 + expreturns) ** 250 - 1  # Annualize returns
    covars = covars * 250  # Annualize covariances
    return expreturns, covars

W = array(caps) / sum(caps) # calculate market weights from capitalizations
R, C = assets_historical_returns_and_covariances(prices)
rf = .015  # Risk-free rate


# ##### Asset returns and weights
# 

display(pandas.DataFrame({'Return': R, 'Weight (based on market cap)': W}, index=names).T)


# ##### Asset covariances
# 

display(pandas.DataFrame(C, columns=names, index=names))


# ## Mean-Variance Optimization (based on historical returns)
# 

res1 = optimize_frontier(R, C, rf)

display_assets(names, R, C, color='blue')
display_frontier(res1, color='blue')
xlabel('variance $\sigma$'), ylabel('mean $\mu$'), show()
display(pandas.DataFrame({'Weight': res1.W}, index=names).T)


# ## Black-litterman reverse optimization
# 

# Calculate portfolio historical return and variance
mean, var = port_mean_var(W, R, C)

lmb = (mean - rf) / var  # Calculate risk aversion
Pi = dot(dot(lmb, C), W)  # Calculate equilibrium excess returns


# ##### Mean-variance Optimization (based on equilibrium returns)
# 

res2 = optimize_frontier(Pi+rf, C, rf)

display_assets(names, R, C, color='red')
display_frontier(res1, label='Historical returns', color='red')
display_assets(names, Pi+rf, C, color='green')
display_frontier(res2, label='Implied returns', color='green')
xlabel('variance $\sigma$'), ylabel('mean $\mu$'), legend(), show()
display(pandas.DataFrame({'Weight': res2.W}, index=names).T)


# ##### Determine views to the equilibrium returns and prepare views (Q) and link (P) matrices
# 

def create_views_and_link_matrix(names, views):
    r, c = len(views), len(names)
    Q = [views[i][3] for i in range(r)]  # view matrix
    P = zeros([r, c])
    nameToIndex = dict()
    for i, n in enumerate(names):
        nameToIndex[n] = i
    for i, v in enumerate(views):
        name1, name2 = views[i][0], views[i][2]
        P[i, nameToIndex[name1]] = +1 if views[i][1] == '>' else -1
        P[i, nameToIndex[name2]] = -1 if views[i][1] == '>' else +1
    return array(Q), P

views = [('MSFT', '>', 'GE', 0.02),
         ('AAPL', '<', 'JNJ', 0.02)]

Q, P = create_views_and_link_matrix(names, views)
print('Views Matrix')
display(DataFrame({'Views':Q}))
print('Link Matrix')
display(DataFrame(P))


# ##### Optimization based on Equilibrium returns with adjusted views
# 

tau = .025  # scaling factor

# Calculate omega - uncertainty matrix about views
omega = dot(dot(dot(tau, P), C), transpose(P))  # 0.025 * P * C * transpose(P)
# Calculate equilibrium excess returns with views incorporated
sub_a = inv(dot(tau, C))
sub_b = dot(dot(transpose(P), inv(omega)), P)
sub_c = dot(inv(dot(tau, C)), Pi)
sub_d = dot(dot(transpose(P), inv(omega)), Q)
Pi_adj = dot(inv(sub_a + sub_b), (sub_c + sub_d))

res3 = optimize_frontier(Pi + rf, C, rf)

display_assets(names, Pi+rf, C, color='green')
display_frontier(res2, label='Implied returns', color='green')
display_assets(names, Pi_adj+rf, C, color='blue')
display_frontier(res3, label='Implied returns (adjusted views)', color='blue')
xlabel('variance $\sigma$'), ylabel('mean $\mu$'), legend(), show()
display(pandas.DataFrame({'Weight': res2.W}, index=names).T)





# Copyright &copy; 2013 Ondrej Martinsky, All rights reserved
# 
# [www.quantandfinancial.com](http://www.quantandfinancial.com)
# # Diversification benefit in mean-variance portfolio optimization
# 

get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = (10, 6)
import scipy.optimize
from pandas import *


r_a = 0.04
r_b = 0.05
stdev_a = 0.07
stdev_b = 0.08


for correl in linspace(-1,1,5):
    X, Y = [], []
    for w_a in linspace(0,1,100):
        w_b = 1 - w_a
        r = r_a * w_a + r_b * w_b
        var = w_a**2 * stdev_a**2 + w_b**2 * stdev_b**2 + 2*w_a*w_b*stdev_a*stdev_b*correl
        stdev = sqrt(var)
        X.append(stdev)
        Y.append(r)
    plot(X,Y,label='Correlation %0.0f%%' % (100*correl))
xlabel('Standard Deviation $\sigma$'), ylabel('Expected return $r$')
title('Diversification benefit for different levels of correlation between two assets')
legend();





