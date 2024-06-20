# # Tutorial for assessing the riskiness of single assets
# 
# ## Steps 
# 
# 1. Import the needed libraries for the tutorial
# 2. Create the starting and ending periods with the Datetime library
# 3. Pull Apple's stock data remotely from Yahoo Finance for those specified periods
# 4. Check the daily returns of Apple's stock for those periods
# 5. Check the daily volatility of Apple's daily returns and convert it to percentage
# 6. Plot a histogram of Apple's daily returns to visualize the volatility
# 7. Extend the tutorial by pulling Adjusted Closing prices for Apple, Facebook and Tesla from Yahoo Finance
# 8. Check the daily returns of the three companies
# 9. Check the volatility of those daily returns
# 10. Visualise the volatility by means of a histogram with the returns of each asset stacked against each other
#  
# 

# Import needed libraries

import pandas as pd
import numpy as np
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# Specify starting and end periods with Datetime

start = datetime(2016,1,1)
end = datetime(2017,1,1)


# Get Apple's stock info

apple = web.DataReader('AAPL', data_source='yahoo', start=start, end=end)


# Check the data

apple.head()


# Slice the Adjusted Closing prices we need 

aapl_close = apple['Adj Close']
aapl_close.head()


# Calculate daily returns 

daily_returns = aapl_close.pct_change()
daily_returns.head()


# Check the volatility of Apple's daily returns

daily_volatility = daily_returns.std()
daily_volatility


# just making the float a bit human readable ;) 

print(str(round(daily_volatility, 5) * 100) + '%')


daily_returns.hist(bins=50, alpha=0.8, color='blue', figsize=(8,6));


# Let's have fun by comparing the volatility of three stocks. Pull Ajdusted closing prices for Apple, Fb and Tesla

assets = ['AAPL', 'FB', 'TSLA']

df = pd.DataFrame()

for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo', start=start, end=end)['Adj Close']
    
df.head()


# Check the daily returns of the three companies

asset_returns_daily = df.pct_change()
asset_returns_daily.head()


# Check the volatility of the daily returns of the three companines

asset_volatility_daily = asset_returns_daily.std()
asset_volatility_daily


# Visualise the daily returns of the three companies stacked against each other. Notice the most/least volatile?

asset_returns_daily.plot.hist(bins=50, figsize=(10,6));


# As seen in the histogram, Tesla's daily returns are the most volatile with the biggest 'spreads'

asset_volatility_daily.max()


# No surprise Apple's daily returns is the least volatile with such a small spread

asset_volatility_daily.min()


# # Go ahead and check the daily volatilities of stocks you find interesting
# 




# # **Let's check out the calculation of rates of returns on single assets**
# 
# The steps are as follows;
# 
# 1. Import Python's number crunching libraries
# 2. Use panda's data reader to get real world stock information of Apple
# 3. Explore the data
# 4. Calculate rate of return using the simple returns fomula 
# 5. Calculate rate of return using log returns
# 
# **More info on Pandas Datareader:** https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#
# 
# ** Documentation for .shift() method:** https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html
# 
# **More info on the differences between simple returns and log returns:** https://quant.stackexchange.com/questions/4160/discrete-returns-versus-log-returns-of-assets
# 

# Step 1 (import python's number crunchers)

import pandas as pd
import numpy as np
from pandas_datareader import data as web


# Step 2 & 3 (Get Apple stock information using Pandas Datareader)

data = pd.DataFrame()

tickers = ['AAPL']

for item in tickers:
    data[item] = web.DataReader(item, data_source='yahoo', start='01-01-2000')['Adj Close']

data.head()


# Step 4 (Simple Returns with the formula)
# .shift() method to use previous value 

simple_returns1 = (data / data.shift(1)) - 1
simple_returns1.head()


# Still Step 4 (Simple Returns formula expressed as a method)
# Same result as above
# Alternative solution

simple_returns2 = data.pct_change()
simple_returns2.head()


# Step 5 (Getting log returns)

log_returns = np.log(data / data.shift(1))
log_returns.head()





# # Tutorial for calculating the rate of returns for a portfolio of various securities
# 
# ## Steps
# 
# 1. **Import Python's number crunchers and the randomising library**
# 2. **Use the output of the scrape S&P 500 list (just copied the output list)**
# 3. **Make 5 random selections with random.random as our imaginary portfolio**
# 4. **Remotely pull Adjusted Closing prices for the 5 randomly selected stocks from Yahoo Finance**
# 5. **Calculate the daily returns of assets in the porftolio**
# 6. **Annualise daily returns in the portfolio**
# 7. **Create random weights matches the number of assets in the portfolio**
# 8. **Make sure that the summation of the weights = 1**
# 9. **Calculate the returns of our imaginary portfolio**
# 10. **Convert the raw float into a percentage**
# 
# More info:
# 
# **Random.random**: https://docs.python.org/3/library/random.html#random.random
# 
# **Random.sample:** https://docs.python.org/3/library/random.html#random.sample
# 
# **.pct_change()**: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pct_change.html
# 

#import python's number crunchers and a randomizer

import numpy as np
import pandas as pd
from pandas_datareader import data as web
import random


# just copied output list from previous scraped list

scraped_tickers = ['MMM', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'AAP', 'AES', 'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'ADM', 'ARNC', 'AJG', 'AIZ', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 'BAC', 'BK', 'BCR', 'BAX', 'BBT', 'BDX', 'BRK.B', 'BBY', 'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BHF', 'BMY', 'AVGO', 'BF.B', 'CHRW', 'CA', 'COG', 'CPB', 'COF', 'CAH', 'CBOE', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNC', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHTR', 'CHK', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 'GLW', 'COST', 'COTY', 'CCI', 'CSRA', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DOV', 'DWDP', 'DPS', 'DTE', 'DRE', 'DUK', 'DXC', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EVHC', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT', 'GWW', 'HAL', 'HBI', 'HOG', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HPQ', 'HUM', 'HBAN', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC', 'ICE', 'IBM', 'INCY', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'KORS', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'PCAR', 'PKG', 'PH', 'PDCO', 'PAYX', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'Q', 'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL', 'CRM', 'SBAC', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIG', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SPGI', 'SWK', 'SPLS', 'SBUX', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYF', 'SNPS', 'SYY', 'TROW', 'TGT', 'TEL', 'FTI', 'TXN', 'TXT', 'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS', 'TSCO', 'TDG', 'TRV', 'TRIP', 'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB', 'UA', 'UAA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'HCN', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS']


# checked the number of items in our list

len(scraped_tickers)


# make a portfolio of randomly 5 listed stocks from the big list

random_selection = random.sample(scraped_tickers, 5)
random_selection


# pull Adjusted closing prices with Pandas datareader and check the head of this data

data = pd.DataFrame()

for item in random_selection:
    data[item] = web.DataReader(item, data_source='yahoo', start='15-09-2016')['Adj Close']

data.head()


# simple daily returns with .pct_change() method

daily_simple_returns = data.pct_change()
daily_simple_returns.head()


# annualise daily returns. 250 trading days in a year

annual_returns = daily_simple_returns.mean() * 250
annual_returns


# number of assets in the randomly selected portfolio

num_assets = len(random_selection)
num_assets


# sum of weights must equal 1. 
# (a / a+b) + (b / a+b) = 1 
# applying this logic above

weights = np.random.random(num_assets)
weights = weights / sum(weights)
weights


# check if the sum of weights is indeed = 1

sum(weights)


# calculate expected returns of the portfolio 

port_returns_expected = np.sum(weights * annual_returns)
port_returns_expected


# convert the float into a percentage cos why not ;)

print(str(round(port_returns_expected * 100, 2)) + '%')


# # Go ahead and play with this tutorial. Try various imaginary portfolios. You only get better with practise 
# 




