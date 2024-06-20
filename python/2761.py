# # Episode 3 -- Summary Statistics
# 

# Problem: Given a bunch of datapoints:
# 
# * characterize the distribution in one or two values
# * Characterization should be robust to outliers
# 
# Equivalent of an elevator pitch for a data sets.
# 
# Problem: This is inherently impossible
# 

# ## Mean Value
# 
# The _mean value_ of $x_1, \dots, x_n$ is defined as
# 
# $$ \mu = mean(x_1, \dots, x_n) = \frac{1}{n} \sum_{i=1}^n x_i. $$
# 
# - Represnets center of mass
# - If the values are close together this is a good representative
# 

import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


def mean(X):
    return float(sum(X)) / len(X)

X = np.loadtxt("DataSets/RequestRates.csv", delimiter=",")[:,1]
print "mean=", mean(X)

# Plot
def mark(m,height=1,style='r'):
    plt.plot([m,m],[0,height],style)

def plot_mean(X):
    sns.rugplot(X, color='grey', height=-1)
    mark(mean(X))
    plt.show()

plt.figure(figsize=(14,2))
plot_mean(X)


# Mean values can be atypical
plt.figure(figsize=(14,2))
plot_mean([1,2,0.4,1.2,100,110,112])


# # Application: Plotting
# 
# * A monitoring graph rarely shows you the full data: Not enough pixels!
# * Need to choose an summary statistic to pre-aggregate the data.
# * Common choice: mean
# 

# ## Peak Erosion is caused by mean-aggregation (default)
# 

# <figure>
# <img src="img/peak_erosion_1.png" width="80%">
# <img src="img/peak_erosion_2.png" width="80%">
# <figcaption>Peak erosion: Same peak is shown with height 0.08 and 0.03</figcaption>
# </figure>
# 

# To avoid peak erosion use either:
# 
# * Histogram Aggregation
# 
# <figure>
# <img src="img/example_peak_histogram.png" width="80%">
# <figcaption>Peak erosion: Show all collected data using histogram aggregation</figcaption>
# </figure>
# 
# * Demo Histogram: https://parlette.circonus.com/trending/graphs/edit/3447a986-d388-4191-82c9-cdacc1af9c79
# * Share: https://share.circonus.com/shared/graphs/3447a986-d388-4191-82c9-cdacc1af9c79/i6PIm8
# 
# * Another example
# <img src="img/example_histogram_aggregation2.png">
# 

# * Max Aggregation
# 
# <figure>
# <img src="img/peak_erosion_3.png" width="80%">
# <figcaption>Peak erosion: Same peak is shown with height 0.08 and 0.03</figcaption>
# </figure>
# 
# * Demo: https://parlette.circonus.com/trending/graphs/view/65d896dd-2be3-4be9-a76d-6fec209358b1
# * Share: https://share.circonus.com/shared/graphs/65d896dd-2be3-4be9-a76d-6fec209358b1/bZ7pNb
# 

# # Deviation Measures
# 
# 1. The _maximal deviation_ is defined as
# 
# $$ maxdev(x_1,\dots,x_n) = max \{ |x_i - \mu| \,|\, i=1,\dots,n\}.$$
# 
# 2. The _mean absolute deviation_ is defined as
# 
# $$ mad(x_1,\dots,x_n) = \frac{1}{n} \sum_{i=1}^n |x_i - \mu|.$$
# 
# 3. The _standard deviation_ is defined as
# 
# $$ \sigma = stddev(x_1,\dots,x_n) =  \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2}.$$
# 
# 
# * Measure the 'typical' displacement from the mean value.
# * Standard deviation is popular because it has extremely nice mathematical properties.
# 

def max_dev(X):
    m = mean(X)
    return max(abs(x - m) for x in X)

def mad(X):
    m = mean(X)
    return sum(abs(x - m) for x in X) / float(len(X))

def stddev(X):
    m = mean(X)
    return math.pow(sum((x - m)**2 for x in X) / len(X), 0.5)

# Plotting helper function
def plot_mean_dev(X, m, s, new_canvas=True):
    print "mean = ", m
    print "dev  = ", s
    if new_canvas: plt.figure(figsize=(14,1))
    sns.rugplot(X, color='grey')
    plt.plot([m,m],[0,-0.09],'r-' )
    plt.plot([m-s,m-s],[0,-0.08],'b-')
    plt.plot([m+s,m+s],[0,-0.08],'b-')
    plt.plot([m-s,m+s],[-0.04,-0.04],'b--')
    if new_canvas:  plt.show()


X = np.loadtxt("DataSets/RequestRates.csv", delimiter=",")[:,1]
    
print "Maximal deviation"
plot_mean_dev(X,mean(X),max_dev(X))

print "Standard Deviation"
plot_mean_dev(X,mean(X),stddev(X))

print "Mean Absolute Deviation"
plot_mean_dev(X,mean(X),mad(X))


# Standard deviation is a good deviation for normal distributed data
X = [ np.random.normal() for x in range(3000) ]
plt.hist(X, bins=30, alpha=0.7, normed=True)
plot_mean_dev(X,mean(X),stddev(X), False)


# Large effect on Outliers
X = X + [200]

print "Maximal deviation"
plot_mean_dev(X,mean(X),max_dev(X))

print "Standard Deviation"
plot_mean_dev(X,mean(X),stddev(X))

print "Mean Absolute Deviation"
plot_mean_dev(X,mean(X),mad(X))


# ## Caution with Standard Deviation
# 
# - Everybody Learns about standard deviation in school
# - Beautiful mathematical properties!
# - Everybody knows 
#   - "68% of data falls within 1 std-dev of the mean"
#   - "95% falls within 2 std-dev of the mean"
#   - "99.7" falls within 3 std-dev of the mean"
# * "Problem is: this is utter nonsense". Only true for normally distributed data.
# 
# * Not good for measuring outliers!
# 
# 
# _Source:_ Janert - Data Analysis with Open Source Tools
# 

# ## War Story:
# 
# - Looking at SLA for DB response times
# - Outlier defined as value larger than $\mu + 3\sigma$
# - Look at code: Takes '0.3' percentile!
# - So always have outliers.
# - And 0.3-percentile was way too large (hours of latency).
# - Programmer changed code for 1%, 5%, 10% quantiles.
# - Finally handcoded a threshold
# - The SLA was never changed
# 
# Source: Janert - Data Analysis with Open Source Tools
# 




# # Part III: Hands On Session
# 

# ## 1. Data Wranginling
# 
# * Count number of `POST` requests logged in `DBLog.out`
# 
# * Convert `DBLog.out` to csv with fileds "Method,Path,RepsoneCode,Latency"
# 
# * Make a bash script `log2csv` that can be used as follows:  
#   `cat DBLog.out | ./log2csv > DBLog.csv`
# 

# ## 2. IPython toolchain
# 
# * Start IPython notebook
# * Import dataset "LogDB.csv"
# * Make a bar chart with "HTTP Method | Count" like so:
# 
# ```
# GET  | ################
# POST | ######
# PUT  | ##
# ```
# 
# * Make a histogram of response times
# * Make a boxplot of the response times
# * Compute the 0.95-quantile
# 

# ## 3. Quantile Checking
# 
# * Load the file `WebLatency.csv` containing request latencies for 6 servers in 5 minute intervals into a np.array
# 
# * Plot the repsonse times of server 1 against time in a scatter plot
# 
# * Calculate the averages over 15m intervals as follows:
#   
#   1. Generate a sequence of timestamps 15 minutes distance
#   2. Generate a sequence of sample windows of length 15 minutes
#   3. Count the number of samples in each window
#   4. Compute the average of the samples in each window
# 
#   Alternative approach:
#   
#   1. Make a function `getWindowStart(t)` that returns the start of the last 15m window for a given time stamp t
#   2. Add a column to the np.array cointaining the window start times for each row
#   3. Use boolean indexing to select all rows in a 15m window
#   4. Compute the count and average of samples in each window
# 
# 
# * Calculate the `0.95`, `0.99` and `1`-quantile over the 15m intervals for each day.
# 

# ## 4. Peek time regression
# 
# * Load the file `ReqRateTrend.csv` containg request rates over time.
# 
# * Calculate the maximum value for each day
# 
# * Make a linear regression of the maximum values for each day over time
# 
# Variant:
# 
# * Make a regression over daily .95 quantiles instead
# 

