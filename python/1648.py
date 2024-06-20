# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="45%" align="right" border="4">
# 

# # Fourier-based Option Pricing
# 

# For several reasons, it is beneficial to have available alternative valuation and pricing approaches to the Monte Carlo simulation approach. One application area is to **benchmark Monte Carlo-based valuation results** against other (potentially more accurate) results. Another area is **model calibration to liquidly traded vanilla instruments** where generally faster numerial methods can be applied.
# 
# This part introduces **Fouried-based valuation functions** and benchmarks valuation results from the "standard", simulation-based DX Analytics modeling approach to output of those functions. 
# 

import dx
import datetime as dt


# ## Risk Factors
# 

# The examples and benchmarks to follow rely on four different models:
# 
# * geometric Brownian motion (Black-Scholes-Merton 1973)
# * jump diffusion (Merton 1976)
# * stochastic volatility (Heston 1993)
# * stochastic volatility jump diffusion (Bates 1996)
# 
# For details on these models and the Fourier-based option pricing approach refer to Hilpisch (2015) (cf. http://eu.wiley.com/WileyCDA/WileyTitle/productCd-1119037999.html).
# 
# We first define the single **market and valuation environments**.
# 

# constant short rate
r = dx.constant_short_rate('r', 0.01)


# geometric Brownian motion
me = dx.market_environment('me', dt.datetime(2015, 1, 1))
me.add_constant('initial_value', 100.)
me.add_constant('volatility', 0.2)
me.add_constant('final_date', dt.datetime(2015, 12, 31))
me.add_constant('currency', 'EUR')


# jump component
me.add_constant('lambda', 0.4)
me.add_constant('mu', -0.6)
me.add_constant('delta', 0.2)


# stochastic volatiltiy component
me.add_constant('rho', -.5)
me.add_constant('kappa', 5.0)
me.add_constant('theta', 0.02)
me.add_constant('vol_vol', 0.3)


# valuation environment
val_env = dx.market_environment('val_env', dt.datetime(2015, 1, 1))
val_env.add_constant('paths', 55000)
    # 25,000 paths
val_env.add_constant('frequency', 'D')
    # weekly frequency
val_env.add_curve('discount_curve', r)
val_env.add_constant('starting_date', dt.datetime(2015, 1, 1))
val_env.add_constant('final_date', dt.datetime(2015, 12, 31))


# add valuation environment to market environment
me.add_environment(val_env)


# Equipped with the single market environments and the valuation environment, we can instantiate the **simulation model objects**.
# 

gbm = dx.geometric_brownian_motion('gbm', me)


jd = dx.jump_diffusion('jd', me)


sv = dx.stochastic_volatility('sv', me)


svjd = dx.stoch_vol_jump_diffusion('svjd', me)


# ## Plain Vanilla Put and Call Options
# 

# Based on the just defined risk factors, we define 8 diffent options---a **European put and call option per risk factor**, respectively.
# 

# market environment for the options
me_option = dx.market_environment('option', dt.datetime(2015, 1, 1))
me_option.add_constant('maturity', dt.datetime(2015, 12, 31))
me_option.add_constant('strike', 100.)
me_option.add_constant('currency', 'EUR')
me_option.add_environment(me)
me_option.add_environment(val_env)


euro_put_gbm = dx.valuation_mcs_european_single('euro_put', gbm, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_gbm = dx.valuation_mcs_european_single('euro_call', gbm, me_option,
                                  'np.maximum(maturity_value - strike, 0)')


euro_put_jd = dx.valuation_mcs_european_single('euro_put', jd, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_jd = dx.valuation_mcs_european_single('euro_call', jd, me_option,
                                  'np.maximum(maturity_value - strike, 0)')


euro_put_sv = dx.valuation_mcs_european_single('euro_put', sv, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_sv = dx.valuation_mcs_european_single('euro_call', sv, me_option,
                                  'np.maximum(maturity_value - strike, 0)')


euro_put_svjd = dx.valuation_mcs_european_single('euro_put', svjd, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_svjd = dx.valuation_mcs_european_single('euro_call', svjd, me_option,
                                  'np.maximum(maturity_value - strike, 0)')


# ## Valuation Benchmarking
# 

# In this sub-section, we benchmark the **Monte Carlo value estimates** against the **Fourier-based pricing results**.
# 

import numpy as np
import pandas as pd


# We first define some parameters used throughout.
# 

freq = '2m'  # used for maturity definitions
periods = 3  # number of intervals for maturity grid
strikes = 5  # number of strikes per maturity
initial_value = 100  # initial value for all risk factors
start = 0.8  # lowest strike in percent of spot
end = 1.2  # highest strike in percent of spot
start_date = '2015/3/1'  # start date for simulation/pricing


# ### Geometric Brownian Motion
# 

# We need to initialize the valuation object first.
# 

euro_put_gbm.present_value()
  # method call needed for initialization


# There is a **valuation class for European put and call options in the Black-Scholes-Merton model** available called `BSM_european_option`. It is based on the analytical pricing formula for that model and is instantiated as follows:
# 

bsm_option = dx.BSM_european_option('bsm_opt', me_option)


# The following routine benchmarks the Monte Carlo value estimates for the **European put option** against the output from the valuation object based on the analytical pricing formula. The results are quite good since this model is quite easy to discretize exactly and therefore generally shows good convergence of the Monte Carlo estimates.
# 

get_ipython().run_cell_magic('time', '', "# European put\nprint('%4s  | %7s | %7s | %7s | %7s | %7s' % ('T', 'strike', 'mcs', 'fou', 'dif', 'rel'))\nfor maturity in pd.date_range(start=start_date, freq=freq, periods=periods):\n    bsm_option.maturity = maturity\n    euro_put_gbm.update(maturity=maturity)\n    for strike in np.linspace(start, end, strikes) * initial_value:\n        T = (maturity - me_option.pricing_date).days / 365.\n        euro_put_gbm.update(strike=strike)\n        mcs = euro_put_gbm.present_value()\n        bsm_option.strike = strike\n        ana = bsm_option.put_value()\n        print('%4.3f | %7.3f | %7.4f | %7.4f | %7.4f | %7.2f '\n                % (T, strike, mcs, ana, mcs - ana, (mcs - ana) / ana * 100))")


# The same now for the **European call option**.
# 

euro_call_gbm.present_value()
  # method call needed for initialization


get_ipython().run_cell_magic('time', '', "# European calls\nprint('%4s  | %7s | %7s | %7s | %7s | %7s' % ('T', 'strike', 'mcs', 'fou', 'dif', 'rel'))\nfor maturity in pd.date_range(start=start_date, freq=freq, periods=periods):\n    euro_call_gbm.update(maturity=maturity)\n    for strike in np.linspace(start, end, strikes) * initial_value:\n        T = (maturity - me_option.pricing_date).days / 365.\n        euro_call_gbm.update(strike=strike)\n        mcs = euro_call_gbm.present_value()\n        bsm_option.strike = strike\n        bsm_option.maturity = maturity\n        ana = bsm_option.call_value()\n        print('%4.3f | %7.3f | %7.4f | %7.4f | %7.4f | %7.2f ' \\\n                % (T, strike, mcs, ana, mcs - ana, (mcs - ana) / ana * 100))")


# ### Benchmarking Function
# 

# All other valuation benchmarks are generated with **Fourier-based pricing functions** for which the handling is identical. We therefore use the following function for the benchmarks from now on:
# 

def valuation_benchmarking(valuation_object, fourier_function):
    print('%4s  | %7s | %7s | %7s | %7s | %7s' % ('T', 'strike', 'mcs', 'fou', 'dif', 'rel'))
    for maturity in pd.date_range(start=start_date, freq=freq, periods=periods):
        valuation_object.update(maturity=maturity)
        me_option.add_constant('maturity', maturity)
        for strike in np.linspace(start, end, strikes) * initial_value:
            T = (maturity - me_option.pricing_date).days / 365.
            valuation_object.update(strike=strike)
            mcs = valuation_object.present_value()
            me_option.add_constant('strike', strike)
            fou = fourier_function(me_option)
            print('%4.3f | %7.3f | %7.4f | %7.4f | %7.4f | %7.3f '
                % (T, strike, mcs, fou, mcs - fou, (mcs - fou) / fou * 100))


# ### Jump Diffusion
# 

# The next model is the jump diffusion as proposed by **Merton (1976)**.
# 

euro_put_jd.present_value()
  # method call needed for initialization


# There is a Fourier-based pricing function available which is called `M76_put_value` and which is used for the benchmarking for the **European put options** that follows.
# 

get_ipython().magic('time valuation_benchmarking(euro_put_jd, dx.M76_put_value)')


# Accordingly, the benchmarking for the **European call options** based on the Fourier-based `M76_call_value` function.
# 

euro_call_jd.present_value()
  # method call needed for initialization


get_ipython().magic('time valuation_benchmarking(euro_call_jd, dx.M76_call_value)')


# ### Stochastic Volatility
# 

# Stochastic volatility models like the one of **Heston (1993)** are popular to reproduce implied volatility smiles observed in markets. First, the benchmarking for the **European put options** using the Fourier-based `H93_put_value` function.
# 

euro_put_sv.present_value()
  # method call needed for initialization


get_ipython().magic('time valuation_benchmarking(euro_put_sv, dx.H93_put_value)')


# Second, the benchmarking for the **European call options** based on the Fourier-based `H93_call_value` function.
# 

euro_call_sv.present_value()
  # method call needed for initialization


get_ipython().magic('time valuation_benchmarking(euro_call_sv, dx.H93_call_value)')


# ### Stochastic Volatility Jump-Diffusion
# 

# Finally, we consider the combination of the stochastic volatility and jump diffusion models from before as proposed by **Bates (1996)**. The Fourier-based pricing function for **European put options** is called `B96_put_value`.
# 

euro_put_svjd.present_value()
  # method call needed for initialization


get_ipython().magic('time valuation_benchmarking(euro_put_svjd, dx.B96_put_value)')


# The Fourier-based counterpart function for **European call options** is called `B96_call_value`.
# 

euro_call_svjd.present_value()
  # method call needed for initialization


get_ipython().magic('time valuation_benchmarking(euro_call_svjd, dx.B96_call_value)')


# ## Sources of Errors
# 

# Numerical methods like Monte Carlo simulation might suffer from different **sources of errors**, like for example:
# 
# * **discretization error**: every **discretization of a continuous time interval**---or a continuous state space to this end---leads to a so-called discretization error
# * **approximation errors**: DX Analytics uses in several places approximative, **Euler-based discretization schemes** (e.g. for performance reasons and to allow for consistent correlation modeling) which are known to be biased
# * **numerical errors**: the approximation of a continuous probability distribution by a **finite, discrete set of (pseudo-) random numbers** introduces also errors
# 

# **Copyright, License & Disclaimer**
# 
# &copy; Dr. Yves J. Hilpisch | The Python Quants GmbH
# 
# DX Analytics (the "dx library") is licensed under the GNU Affero General Public License
# version 3 or later (see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)).
# 
# DX Analytics comes with no representations
# or warranties, to the extent permitted by applicable law.
# 
# 
# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>
# 
# [http://tpq.io](http://tpq.io) | [team@tpq.io](mailto:team@tpq.io) | [http://twitter.com/dyjh](http://twitter.com/dyjh)
# 
# **Quant Platform** |
# [http://quant-platform.com](http://quant-platform.com)
# 
# **Derivatives Analytics with Python (Wiley Finance)** |
# [http://derivatives-analytics-with-python.com](http://derivatives-analytics-with-python.com)
# 
# **Python for Finance (O'Reilly)** |
# [http://python-for-finance.com](http://python-for-finance.com)
# 

