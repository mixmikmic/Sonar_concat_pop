# # Fantasy Football
# 
# So - this is my first year participating in a fantasy football league. I enjoy football, but I typically only keep up with a few teams, so drafting an actual team was a bit daunting. So, like most things, I relied on data to help me out. I spent some time researching strategy, looking at projections, and even simulating some drafts. Draft day came and I felt pretty good about my team...but now I am currently 0-3 for the season. Ha.
# 
# I started to get a bit curious about why I was doing so poorly. Typically my "projected" points were pretty good each week, but my team just never seemed to deliver.
# 
# Thus, here is my first post looking at the biggest out performers and the biggest misses relative to ESPN projections. All data are from ESPN. So - lets get to it!
# 

import psycopg2
import pandas.io.sql as psql
import pandas as pd
from matplotlib import pyplot as plt
from __future__ import division #now division always returns a floating point number
import numpy as np
import seaborn as sns
get_ipython().magic('matplotlib inline')


db = psycopg2.connect("dbname='fantasyfootball' host='localhost'")

def get_combined_df():
    actual_points = psql.read_sql("""
    select name, team, position, sum(total_points) as points_actual 
    from scoring_leaders_weekly
    group by name, team, position;""", con=db)
    predicted_points = psql.read_sql("""
        select name, team, position, sum(total_points) as points_predicted
        from next_week_projections
        group by name, team, position;""", con=db)
    combined_df = actual_points.merge(predicted_points, 
                                      on=['name', 'team', 'position'], how='left')
    combined_df = combined_df.dropna()
    combined_df = combined_df[combined_df['points_predicted'] > 0]
    combined_df['points_diff'] = combined_df.points_actual - combined_df.points_predicted
    combined_df['points_diff_pct'] = (combined_df.points_actual - combined_df.points_predicted) / combined_df.points_predicted
    return combined_df

def get_top_bottom(df):
    group = df.groupby('position')
    top_list = []
    bottom_list = []
    for name, data in group:
        top = data.sort('points_diff', ascending=False)
        top_list.append(top.head())
        tail = top.tail()
        tail = tail.sort('points_diff')
        bottom_list.append(tail)
    top_df = pd.concat(top_list)
    bottom_df = pd.concat(bottom_list)
    return top_df, bottom_df

def run_analysis():
    combined_df = get_combined_df()
    top, bottom = get_top_bottom(combined_df)
    return combined_df, top, bottom


# # RESULTS
# 
# For the results, I decided to show the top 5 out performers and the top 5 under performers for the cumulative season based on the absolute point difference (not the percentage). First, here are the **out performers**. This is pretty interesting. Travis Benjamin, for example, has in the first 3 weeks produced an extra 33.6 fantasy points relative to expectations. Not too bad.
# 

combined_df, top_1, bottom_1 = run_analysis()


top_1


# Now here are the **under performers**. These are the people you don't want to be playing...For example, C.J. Anderson comes in at a whopping 42 points under expectation. Who is my running back you ask... Drew Brees takes the cake, though, under performing by 46.2 points on the season. He was injured, though.
# 

bottom_1


# Next, I wanted to take a look at the distribution of point differences by position. The below chart shows that the median player in all positions is under performing, except for TE which is pretty close to zero. There are a few break out WRs and quite a bunch of under performing running backs. The spread is also pretty wide for most of the positions.
# 

ax = sns.boxplot(combined_df.points_diff, groupby=combined_df.position)
plt.title("Distribution of Point Differences by Position")
sns.despine()


# I also looked at the distribution of actual points by position. One thing you hear in fantasy is to select RBs early because they are high variance players. Meaning that you suffer more by getting a lower ranked RB than a lower ranked QB. This is also due to the fact that a lot more RBs are getting drafted than QBs. Below is the distribution for all players and provides a general sense.
# 

ax = sns.boxplot(combined_df.points_actual, groupby=combined_df.position)
plt.title("Distribution of Actual Points by Position")
sns.despine()


# To see if the high variance difference is playing out, we can look at the top 12 quarterbacks and top 36 running backs so far in the season (assume 12 team league with 1 starting QB and 3 starting RBs). You can see below that indeed the RBs standard deviation is quite a bit higher (about 7 points) than the QBs.
# 

combined_df[combined_df['position'] == "QB"].sort('points_actual', ascending=False).head(n=12).describe()


combined_df[combined_df['position'] == "QB"].sort('points_actual', ascending=False).head(n=36).describe()


# # In conclusion
# 
# These were just some quick analyses I did to try and get a sense of which players are doing well/poorly and how various positions are performing. 
# 
# If people find this interesting, I can try and update the data as the season goes on.
# 
# I am hoping to find time to investigate the ESPN projections to see how sensical they really are. Based on the chart above, they seem to aim high, leading to many under performers. I would like to try and build my own projection model to see how well I can compare. Now that I think about it, here are the overall summary statistics below. It looks like on average ESPN is over projecting by about 4 points with a standard deviation of 10.5 points.
# 

combined_df.describe()


# #Bayesian Logistic Regression
# 
# Install PYMC3: https://pymc-devs.github.io/pymc3/#installation
# 

import pymc3 as pm
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style='ticks', palette='Set2')
import pandas as pd
import numpy as np
import math
from theano import tensor
from __future__ import division

data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]

setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
sns.despine()


basic_model = pm.Model()
X1 = X[:, 0]
X2 = X[:, 1]

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2
    logit = 1 / (1 + tensor.exp(-mu))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Bernoulli('Y_obs', p=logit, observed=y)
    
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start) # Instantiate MCMC sampling algorithm
    trace = pm.sample(2000, step, progressbar=False) # draw 2000 posterior samples using NUTS sampling


pm.traceplot(trace)


np.mean(trace.get_values('beta'), axis=0)


def predict(trace, x1, x2, threshold):
    alpha = trace.get_values('alpha').mean()
    betas = np.mean(trace.get_values('beta'), axis=0)
    linear = alpha + (x1 * betas[0]) + (x2 * betas[1])
    probability = 1 / (1 + np.exp(-linear))
    return [np.where(probability >= threshold, 1, 0), linear, probability]
def accuracy(predictions, actual):
    return np.sum(predictions == actual) / len(predictions)


predictions, logit_x, logit_y = predict(trace, X1, X2, 0.5)
accuracy(predictions, y)


plt.scatter(logit_x, logit_y)


# #Offensive Tackles in Football
# 
# So I have a brother who is currently 16 and plays high school football as an offensive tackle. He is 6 feet 2 inches and weights 250 pounds. Now that is a big kid. His dream is to play professional football one day.
# 
# The first step to playing professional football is to play college football for hopefully a good team. Not knowing much about what it takes to get recruited by a top college football team, I thought I would look for some data. Fortunately, ESPN has height and weight data on the [top 100 offensive tackles](http://espn.go.com/college-sports/football/recruiting/playerrankings/_/position/offensive-tackle/class/2015/view/position) being recruited out of high school. This little project will look at the height and weight of top recruited offensive tackles and how these values are associated with that player's rank.
# 
# #Get and Clean the Data
# 

from bs4 import BeautifulSoup
import urllib2
import pandas as pd
from pandas import DataFrame, Series
get_ipython().magic('matplotlib inline')
from __future__ import division
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
import statsmodels.api as sm


# Lets get the data from ESPN.
# 

html = urllib2.urlopen('http://espn.go.com/college-sports/football/recruiting/playerrankings/_/view/position/order/true/position/offensive-guard')
text = html.read()
soup = BeautifulSoup(text.replace('ISO-8859-1', 'utf-8'))


ht_wgt = []
for tr in soup.findAll('tr')[1:]:
    tds = tr.findAll('td')
    height = tds[4].text
    weight = tds[5].text
    grade = tds[7].text
    ht_wgt.append([height, weight, grade])


# A quick sanity check to make sure we got 100 players
# 

#should have 100
len(ht_wgt)


# Now lets drop our data into a Pandas data frame and take a look.
# 

data = DataFrame(ht_wgt, columns=['height', 'weight', 'grade'])
data.head()


# Lets clean up the data to get the values as integers and convert the height to inches. I also created a mean zero grade just to bring the grades closer to zero.
# 

data['weight'] = data.weight.astype(int)
data['grade'] = data.grade.astype(int)
hgt_str = data.height.values
hgt_str = [x.split("'") for x in hgt_str]
hgt_in = [(int(x[0]) * 12) + int(x[1]) for x in hgt_str]
data['height_inches'] = hgt_in
data['grade_meanzero'] = data.grade - data.grade.mean()
data.head()


# #Make Some Plots
# 
# Lets take a look at the distribution of height and weight and how that fits in with the ranking. Also, I will plot my brother on the plots to see how he stacks up.
# 

fig, ax = plt.subplots(3,1)
fig.set_size_inches(8.5, 11)

fig.suptitle('2015 ESPN Top 100 High School Offensive tackles',
             fontsize=16, fontweight='bold')

ax[0].hist(data.height_inches, bins = range(73,83), align='left')
ax[0].set_xlabel('Height')
ax[0].set_ylabel('Number of Players')
ax[0].annotate('Average Height: {}'.format(data.height_inches.mean()), 
             xy=(.5, .5), xytext=(.70, .7),  
             xycoords='axes fraction', textcoords='axes fraction')
ax[0].plot([75, 75], [0,40])
ax[0].set_xlim([72,82])
ax[0].set_xticks(range(73,83))
ax[0].annotate('My Brother', xy=(75, 20), xytext=(73, 25))

ax[1].hist(data.weight)
ax[1].set_xlabel('Weight in Pounds')
ax[1].set_ylabel('Number of Players')
ax[1].annotate('Average Weight: {}'.format(data.weight.mean()), 
             xy=(.5, .5), xytext=(.70, .7),  
             xycoords='axes fraction', textcoords='axes fraction')
ax[1].plot([280, 280], [0,30])
ax[1].annotate('My Brother', xy=(250, 15), xytext=(236, 20))

ax[2].scatter(data.height_inches, data.weight, s=data.grade_meanzero*15, alpha=.6)
ax[2].set_title('Bigger Circle Means Better Rank')
ax[2].set_xlabel('Height in Inches')
ax[2].set_ylabel('Weight in Pounds')
ax[2].set_xlim([72,82])
ax[2].set_xticks(range(73,83))
ax[2].scatter([75],[280], alpha=1, s=50, c=sns.color_palette("Set2", 2)[1])
ax[2].annotate('My Brother', xy=(75, 280), xytext=(73.5, 255))

fig.tight_layout()
plt.subplots_adjust(top=0.92)
sns.despine()
plt.savefig('Top100_OT.png')


# #Analysis
# 
# It looks like the sweet spot for height is between 76 and 78 inches. And it looks like taller players are getting a better rank; at least up to 78 inches. This makes some sense because you probably don't expect your players to grow much taller; you can more easily affect their weight gain if needed.
# 
# This is a very simple and somewhat silly example of data analysis, but I like it. Using data I was able to gain a better understanding of an area in which I previously had no experience. Also, I was able to give my brother some sound advice that is grounded in data - grow taller!
# 

# #What is Bayes Theorem?
# 
# Bayes theorem is what allows us to go from a **sampling (or likelihood) distribution** and a **prior distribution** to a **posterior distribution**.
# 
# ##What is a Sampling Distribution?
# 
# A sampling distribution is the probability of seeing **our data (X)** given our **parameters ($\theta$).** This is written as $p(X|\theta)$.
# 
# For example, we might have data on 1,000 coin flips. Where 1 indicates a head. This can be represented in python as
# 

import numpy as np
data_coin_flips = np.random.randint(2, size=1000)
np.mean(data_coin_flips)


# A sampling distribution allows us to specify how we think these data were generated. For our coin flips, we can think of our data as being generated from a [Bernoulli Distribution](http://en.wikipedia.org/wiki/Bernoulli_distribution). This distribution takes one **parameter** p which is the probability of getting a 1 (or a head for a coin flip). It then returns a value of 1 with probablility p and a value of 0 with probablility (1-p).
# 
# You can see how this is perfect for a coin flip. With a fair coin we know our p = .5 because we are equally likely to get a 1 (head) or 0 (tail). We can create samples from this distribution like this:
# 

bernoulli_flips = np.random.binomial(n=1, p=.5, size=1000)
np.mean(bernoulli_flips)


# Now that we have defined how we believe our data were generated, we can calculate the probability of seeing our data given our parameters $p(X|\theta)$. Since we have selected a Bernoulli distribution, we only have one parameter: p. 
# 
# We can use the **probability mass function (PMF)** of the Bernoulli distribution to get our desired probability for a single coin flip. The PMF takes a single observed data point and then given the parameters (p in our case) returns the probablility of seeing that data point given those parameters. For a Bernoulli distribution it is simple: if the data point is a 1 the PMF returns p, if the data point is a 0 it returns (1-p). We could write a quick function to do this:
# 

def bern_pmf(x, p):
    if (x == 1):
        return p
    elif (x == 0):
        return 1 - p
    else:
        return "Value Not in Support of Distribution"


# We can now use this function to get the probability of a data point give our parameters. You probably see that with p = .5 this function always returns .5
# 

print(bern_pmf(1, .5))
print(bern_pmf(0, .5))


# This is a pretty simple PMF, but other distributions can get much more complicated. So it is good to know that Scipy has most of these built in. We can draw from the PMF as follows:
# 

import scipy.stats as st
print(st.bernoulli.pmf(1, .5))
print(st.bernoulli.pmf(0, .5))


# This is nice, but what we really want to know is the probability of see all 1,000 of our data points. How do we do that? The trick here is to assume that our data are [independent and identically distributed](http://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables). This assumption allows us to say the probability of seeing all of our data is just the product of each individual probability: $p(x_{1}, ..., x_{n}|\beta) = p(x_{1}|\beta) * ... * p(x_{n}|\beta)$. This is easy to do:
# 

np.product(st.bernoulli.pmf(data_coin_flips, .5))


# How does that number help us? Well by itself, it doesn't really help too much. What we need to do now is get more of a distribution for our sampling model. Currently, we have only tested our model with p = .5, but what if p = .8? or .2? What would the probablility of our data look like then? This can be done by defining a grid of values for our p. Below I will make a grid of 100 values between 0 and 1 (because p has to be between 0 and 1) and then I will calculate the probability of seeing our data given each of these values:
# 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style='ticks', palette='Set2')

params = np.linspace(0, 1, 100)
p_x = [np.product(st.bernoulli.pmf(data_coin_flips, p)) for p in params]
plt.plot(params, p_x)
sns.despine()


# Now we are getting somewhere. We can see that the probablility of seeing our data peaks at p=.5 and almost certainly is between p=.4 and p=.6. Nice. So now we have a good idea of what p value generated our data assuming it was drawn from a Bernoulli distribution. We're done, right? Not quite...
# 
# ##Prior Distribution
# 
# Bayes theorem says that we need to think about both our sampling distribution and our prior distribution. What do I mean by prior distribution? It is the $p(\theta)$ or the probability of seeing a specific value for our parameter. In our sampling distribution we defined 100 values from 0 to 1 for our parameter p. Now we must define the prior probability of seeing each of those values. That is the probability we would have assumed before seeing any data. Most likely, we would have assumed a fair coin, which looks like the distribution above. Lets see how we can do this:
# 

fair_flips = bernoulli_flips = np.random.binomial(n=1, p=.5, size=1000)
p_fair = np.array([np.product(st.bernoulli.pmf(fair_flips, p)) for p in params])
p_fair = p_fair / np.sum(p_fair)
plt.plot(params, p_fair)
sns.despine()


# Basically we created 1,000 fair coin flips and then generated the sampling distribution like we did before (except we divided by the sum of the sampling distribution to make the values sum to 1). Now we have a "fair coin" prior on our parameters. This basically means that before we saw any data we thought coin flips were fair. And we can see that assumption in our prior distribution by the fact that our prior distribution peaks at .5 and is almost all between .4 and .6.
# 
# I know what you are thinking - this is pretty boring. The sampling and prior distributions look exactly the same. So lets mix things up. Lets keep our fair prior but change our data to be an unfair coin:
# 

unfair_flips = bernoulli_flips = np.random.binomial(n=1, p=.8, size=1000)
p_unfair = np.array([np.product(st.bernoulli.pmf(unfair_flips, p)) for p in params])
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(params, p_unfair)
axes[0].set_title("Sampling Distribution")
axes[1].plot(params, p_fair)
axes[1].set_title("Prior Distribution")
sns.despine()
plt.tight_layout()


# Ah - now this is interesting. We have strong data evidence of an unfair coin (since we generated the data we know it is unfair with p=.8), but our prior beliefs are telling us that coins are fair. How do we deal with this?
# 
# #Bayes Theorem (Posterior Distribution)
# 
# Bayes theorem is what allows us to go from our sampling and prior distributions to our posterior distribution. The **posterior distribution is the $P(\theta|X)$**. Or in English, the probability of our parameters given our data. And if you think about it that is what we really want. We are typically given our data - from maybe a survey or web traffic - and we want to figure out what parameters are most likely given our data. So how do we get to this posterior distribution? Here comes some math (don't worry it is not too bad):
# 
# By definition, we know that (If you don't believe me, check out this [page](https://people.richland.edu/james/lecture/m170/ch05-cnd.html) for a refresher): 
# * $P(A|B) = \dfrac{P(A,B)}{P(B)}$. Or in English, the probability of seeing A given B is the probability of seeing them both divided by the probability of B.
# * $P(B|A) = \dfrac{P(A,B)}{P(A)}$. Or in English, the probability of seeing B given A is the probability of seeing them both divided by the probability of A.
# 
# You will notice that both of these values share the same numerator, so:
# * $P(A,B) = P(A|B)*P(B)$
# * $P(A,B) = P(B|A)*P(A)$
# 
# Thus:
# 
# $P(A|B)*P(B) = P(B|A)*P(A)$
# 
# Which implies:
# 
# $P(A|B) = \dfrac{P(B|A)*P(A)}{P(B)}$
# 
# And plug in $\theta$ for $A$ and $X$ for $B$:
# 
# $P(\theta|X) = \dfrac{P(X|\theta)*P(\theta)}{P(X)}$
# 
# Nice! Now we can plug in some terminology we know:
# 
# $Posterior = \dfrac{likelihood * prior}{P(X)}$
# 
# But what is the $P(X)$? Or in English, the probability of our data? That sounds wierd... Let's go back to some math and use $B$ and $A$ again:
# 
# We know that $P(B) = \sum_{A} P(A,B)$ (check out this [page](http://en.wikipedia.org/wiki/Marginal_distribution) for a refresher)
# 
# And from our definitions above, we know that:
# 
# $P(A,B) = P(B|A)*P(A)$
# 
# Thus:
# 
# $P(B) = \sum_{A} P(B|A)*P(A)$
# 
# Plug in our $\theta$ and $X$:
# 
# $P(X) = \sum_{\theta} P(X|\theta)*P(\theta)$
# 
# Plug in our terminology:
# 
# $P(X) = \sum_{\theta} likelihood * prior$
# 
# Wow! Isn't that awesome! But what do we mean by $\sum_{\theta}$. This means to sum over all the values of our parameters. In our coin flip example, we defined 100 values for our parameter p, so we would have to calculated the likelihood * prior for each of these values and sum all those anwers. That is our denominator for Bayes Theorem. Thus our final answer for Bayes is:
# 
# $Posterior = \dfrac{likelihood * prior}{\sum_{\theta} likelihood * prior}$
# 
# That was a lot of text. Let's do some more coding and put everything together.

def bern_post(n_params=100, n_sample=100, true_p=.8, prior_p=.5, n_prior=100):
    params = np.linspace(0, 1, n_params)
    sample = np.random.binomial(n=1, p=true_p, size=n_sample)
    likelihood = np.array([np.product(st.bernoulli.pmf(sample, p)) for p in params])
    #likelihood = likelihood / np.sum(likelihood)
    prior_sample = np.random.binomial(n=1, p=prior_p, size=n_prior)
    prior = np.array([np.product(st.bernoulli.pmf(prior_sample, p)) for p in params])
    prior = prior / np.sum(prior)
    posterior = [prior[i] * likelihood[i] for i in range(prior.shape[0])]
    posterior = posterior / np.sum(posterior)
    
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    axes[0].plot(params, likelihood)
    axes[0].set_title("Sampling Distribution")
    axes[1].plot(params, prior)
    axes[1].set_title("Prior Distribution")
    axes[2].plot(params, posterior)
    axes[2].set_title("Posterior Distribution")
    sns.despine()
    plt.tight_layout()
    
    return posterior


example_post = bern_post()


# You will notice that I set 100 as the number of observations for the prior and likelihood. This increases the variance of our distributions. More data typically decreases the spread of a distribution. Also, as you get more data to estimate your likelihood, the prior distribution matters less. 
# 

moredata_post = bern_post(n_sample=1000)


# You can see that effect in the graphs above. Because we have more data to help us estimate our likelihood our posterior distribution is closer to our likelihood. Pretty cool.
# 
# ##Conclusion
# 
# There you have it. An introduction to Bayes Theorem. Now if you ever are doubting the fairness of a coin you know how to investigate that problem! Or maybe the probability of a population voting yes for a law? Or any other yes/no outcome. I know what you are thinking - not that impressive. You want to predict things or run a fancy algorithm... in time my friends. I actually would like to write a series of posts leading up to Bayesian Linear Regression. Hopefully this is the first post in that series :)
# 
# ###Some Side Notes
# 
# 1. You will notice that the denominator for Bayes Theorem is just a constant. So if you only want to get the maximum posterior value, you don't even need to calculate that constant. For this reason you will often see the posterior shown as proportional (or $\propto$ in math) to the likelihood * prior.
# 2. Frequentist statistics is focused on the likelihood. Or you could say that frequentists are bayesians with a non-informative prior (like a uniform distribution). But don't hate on frequentists too much; most of bayesian inference in applied settings relies of frequentists statistics.
# 3. Now that you know frequentist statistics focuses on the likelihood it is much clearer why people often misinterpret frequentist confidence intervals. The likelihood is $P(X|\theta)$ - or the probability of our data given our parameters. That is a bit wierd because we are given our data, not our parameters. What most frequentists models do is take the maximum of the likelihood distribution (or Maximum Likelihood Estimation (MLE)). Basically find what parameters maximize the probability of seeing our data. The important point here is that they are treating the data as random, so what the frequentist confidence interval is saying is that if you were to keep getting new data (maybe some more surveys) and calculate confidence intervals on each of these new samples, 95% of these samples would have confidence intervals that contain the true parameter you are trying to estimate.
# 




# #Continuous Prior
# 
# In my [introduction](http://nbviewer.ipython.org/github/tfolkman/learningwithdata/blob/master/Bayes_Primer.ipynb) to Bayes post, I went over a simple application of Bayes theorem to Bernoulli distributed data. In this post, I want to extend our example to use a continous prior.
# 
# In my last post, I ended with this code:
# 

import numpy as np
import matplotlib.pyplot as plt
from __future__ import division
get_ipython().magic('matplotlib inline')
import scipy.stats as st
import seaborn as sns
sns.set(style='ticks', palette='Set2')

def bern_post(n_params=100, n_sample=100, true_p=.8, prior_p=.5, n_prior=100):
    params = np.linspace(0, 1, n_params)
    sample = np.random.binomial(n=1, p=true_p, size=n_sample)
    likelihood = np.array([np.product(st.bernoulli.pmf(sample, p)) for p in params])
    #likelihood = likelihood / np.sum(likelihood)
    prior_sample = np.random.binomial(n=1, p=prior_p, size=n_prior)
    prior = np.array([np.product(st.bernoulli.pmf(prior_sample, p)) for p in params])
    prior = prior / np.sum(prior)
    posterior = [prior[i] * likelihood[i] for i in range(prior.shape[0])]
    posterior = posterior / np.sum(posterior)
    
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    axes[0].plot(params, likelihood)
    axes[0].set_title("Sampling Distribution")
    axes[1].plot(params, prior)
    axes[1].set_title("Prior Distribution")
    axes[2].plot(params, posterior)
    axes[2].set_title("Posterior Distribution")
    sns.despine()
    plt.tight_layout()
    
    return posterior


p = bern_post()


# You will notice the "n_params" parameter in our code. Remember, in this case we have a prior on the probability of a coin-flip being heads. Thus our prior value has to be between 0 and 1. So n_params = 100 means we split the range 0 to 1 into 100 values. This is typically refered to as a grid approach because we break up our prior value (which can be any real number from 0 to 1) into a finite number of discrete values.
# 
# But what if we represent our prior beliefs as a continuous distribution?

# #Beta Distribution
# 
# So - we need a continuous distribution that gives us a value between 0 and 1 - like our probability of heads. An obvious choice that fits this criteria is a [beta distribution](http://en.wikipedia.org/wiki/Beta_distribution).
# 
# What is this beta distribution? Lets take a look:
# 

x = np.linspace(0, 1, 100)
y = [st.beta.pdf(x1, 2, 2) for x1 in x]
plt.plot(x, y)
plt.axvline(.5)


# If you look closely at the code, you will notice the beta PDF (a PDF is just a function that will tell you the relative likelihood of a random variable to take a given value; so in the above PDF you can see that 0.5 has a relatively high likelihood) takes 2 parameters - we will call them $\alpha$ and $\beta$. With these two parameters we know the PDF of a beta function as: $$\propto x^{\alpha - 1}(1 - x)^{\beta - 1}$$ This formula comes directly from the Wikipedia link. You will notice, though, that I excluded the denominator in the formula. Thus, I said proportional to. In almost all PDF formulas the denominator is the normalizing constant. The normalizing constant is basically what makes sure the PDF integrates to 1 as a probability distribution must. And because it is a constant we can ignore it if we only want relative likelihoods and not actual probability values.
# 
# The wikipedia link has a good picture that shows how the PDF of the beta distribution changes as we change $\alpha$ and $\beta$. So how do we pick values for $\alpha$ and $\beta$?
# 
# We know (see Wikipedia) that the mean value of a beta PDF is $$\frac{\alpha}{\alpha + \beta}$$ Thus, in our picture above with $\alpha=2$ and $\beta=2$, we get a mean value of .5.
# 
# And the variance is: $$\frac{\alpha \beta}{(\alpha + \beta)^{2}(\alpha + \beta + 1)}$$ So for our parameters, we get a value of $$\frac{4}{80}=.05$$ And with a beta distribution, when $\alpha = \beta > 1$ you will always get a symmetric distribution centered around 1/2. This is good for our coin flipping example if we think our prior is a fair coin. As we increase $\alpha$ and $\beta$ our variance will get smaller, which means we have more confidence in our prior belief that our coin is fair.
# 
# Note: There are many ways to pick $\alpha$ and $\beta$. Another way I have seen is to:
# 
# let $m$ = prior belief of probability of heads
# 
# let $n$ = the number of data points used to calculate your $m$ value. If you didn't use data, then basically set $n$ higher if you are more confident in your prior. Then:
# 
# $$\alpha = mn$$ $$\beta = (1-m)n$$
# 
# In the code below you can see this in action
# 

m = .8
n = 100
alpha = m*n
beta = (1-m)*n
y = [st.beta.pdf(x1, alpha, beta) for x1 in x]
plt.plot(x, y)


# So what have we learned? 
# * Instead of using a grid of values to represent our parameters, we can actually use a continous distribution, like a beta distribution.
# * We can specify our prior beta distribution using two parameters: $\alpha$ and $\beta$
# * We can adjust $\alpha$ and $\beta$ to more reflect our prior beliefs
# 
# ##Connecting it all
# 
# So how do we use this beta distribution to get to our posterior estimate? Remember: $$p(\theta|y) \propto p(y|\theta)p(\theta)$$ This is basically Bayes' theroem and in English says the posterior distribution is proportional to the product of the likelihood and prior distributions.
# 
# Going back to our coin flip example, we defined our likelihood as: $$ \theta^{y}(1-\theta)^{n-y}$$ Where $y$ is the number of flips that are heads, $n$ is the number of total flips, and $\theta$ is the probability of getting a heads.
# 
# We will now use a Beta distribution to represent the $p(\theta)$. Thus our prior is: $$\theta^{\alpha-1}(1-\theta)^{\beta-1}$$ 
# 
# When we take the product we get: $$ \theta^{y}(1-\theta)^{n-y}\theta^{\alpha-1}(1-\theta)^{\beta-1} $$
# 
# Oh! This is nice...we can easily combine terms and get: $$\theta^{y+\alpha-1}(1-\theta)^{n-y+\beta-1} $$
# 
# If you add parenthesis in the right places you will see:  $$\theta^{(y+\alpha)-1}(1-\theta)^{(n-y+\beta)-1} $$
# 
# Which is just a Beta distribution with $$ \alpha = y + \alpha_{prior} $$ $$ \beta = n - y + \beta_{prior} $$
# 
# Wow! This is amazing because now we know exactly how our posterior is distributed. This is so awesome, there is even a name for it: **conjugate prior**. Whenever we have a prior distribution on a likelihood that simplifies to a posterior with the same distribution as the prior, we call that prior distribution a conjugate prior relative to that likelihood distribution. So a beta distribution is a conjugate prior for the bernoulli distribution. (Note: Wikipedia has a great list of [conjugate priors](http://en.wikipedia.org/wiki/Conjugate_prior)) In code:
# 

def bern_beta(n_flips, n_heads, prior_alpha, prior_beta):
    n_tails = n_flips - n_heads
    x_values = np.linspace(0, 1, 100)
    likelihood = [(x**n_heads) * ((1-x)**(n_tails)) for x in x_values]
    prior = [st.beta.pdf(x, prior_alpha, prior_beta) for x in x_values]
    posterior = [st.beta.pdf(x, prior_alpha + n_heads, prior_beta + n_tails) for x in x_values]
    
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    axes[0].plot(x_values, likelihood)
    axes[0].set_title("Sampling Distribution")
    axes[1].plot(x_values, prior)
    axes[1].set_title("Prior Distribution")
    axes[2].plot(x_values, posterior)
    axes[2].set_title("Posterior Distribution")
    sns.despine()
    plt.tight_layout()


bern_beta(n_flips=100, n_heads=10, prior_alpha=50, prior_beta=50)


# ##Knowing the Posterior Distribution is Great
# 
# Now imagine we only care about the average posterior value. If we make the above assumptions of a bernoulli likelihood and beta prior we know our posterior is distributed as a beta distribution:
# 
# $$Beta(y + \alpha_{prior}, n - y + \beta_{prior})$$
# 
# And we know the mean value of our beta distribution to be:
# 
# $$\frac{y + \alpha_{prior}}{y + \alpha_{prior} + n - y + \beta_{prior}}$$
# 

n = 100
y = 10
prior_alpha = 50
prior_beta = 50
posterior_mean = (y + prior_alpha) / (y + prior_alpha + n - y + prior_beta)
print("Posterior Mean: {}".format(posterior_mean))


# Nice - no need for anything other than some algebra :)
# 

# # Predicting Fantasy Football Points
# 
# If you read my last [post](http://nbviewer.ipython.org/github/tfolkman/learningwithdata/blob/master/Biggest_Misses.ipynb) you will know that I recently started fantasy football and my team isn't doing so great. Currently 0 and 4. Ha!
# 
# What seemed strange to me, though, is that my team kept underperforming relative to the ESPN projections. The consistent underperformance lead me to try and develop my own prediction model to see if I couldn't maybe do a better job.
# 
# Skip the next few blocks of code to see more discussion.
# 

import sqlalchemy
from sqlalchemy.orm import create_session
from sklearn import preprocessing
from collections import namedtuple
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
get_ipython().magic('matplotlib inline')


engine = sqlalchemy.create_engine('postgresql://localhost/fantasyfootball')    
session = create_session(bind=engine)


class Prediction:

    
    def __init__(self, test_proj_week, position='ALL'):
        self.train_proj_week = test_proj_week - 1
        self.test_proj_week = test_proj_week
        self.position = position


    def make_prediction(self):
        encoders = self.create_position_factors()
        
        self.train_data = self.week_df(self.train_proj_week, encoders.position,
                             encoders.team, self.position)
        test_data = self.week_df(self.test_proj_week, encoders.position, encoders.team, self.position)
        
        clf = RandomForestRegressor(n_estimators=5000, max_depth=5)
        clf.fit(self.train_data.X, self.train_data.y)
        model_predicted_points = clf.predict(test_data.X)
        
        results = self.rmean_sq(test_data.y.values, model_predicted_points)
        espn = self.rmean_sq(test_data.y.values, test_data.espn_proj.values)
        
        # Put some variables in self for easier access
        self.my_prediction = model_predicted_points
        self.model = clf
        self.results = results
        self.espn_results = espn
        self.actual = test_data.y.values
        self.espn_prediction = test_data.espn_proj.values
        
        self.get_combined_df(test_data)
      
    
    def get_combined_df(self, data):
        df=pd.concat([data.X, data.index, data.espn_proj, data.y], axis=1)
        df['name'] = df['index'].str.split("_").str.get(0)
        df['team'] = df['index'].str.split("_").str.get(1)
        df['position'] = df['index'].str.split("_").str.get(2)
        df['my_prediction'] = self.my_prediction
        self.combined_test_df = df
        
    
    def report(self):
        print("Prediction for Week {0} for {1} position(s)".format(self.test_proj_week, self.position))
        print("My RMSE: {}".format(self.results.rmse))
        print("ESPN RMSE: {}".format(self.espn_results.rmse))
        self.plot_feature_importance()
        plt.title("Feature Importance", fontsize=20)
        plt.show()
        self.plot_dist_comp()
        plt.title("Distribution of RMSE", fontsize=20)
        plt.show()
        self.scatter_plot(self.actual, self.my_prediction)
        plt.title("My Predictions", fontsize=20)
        plt.show()
        self.scatter_plot(self.actual, self.espn_prediction)
        plt.title("ESPN Predictions", fontsize=20)
        plt.show()
        
    
    def plot_feature_importance(self):
        plt.figure(figsize=(8,5))
        df = pd.DataFrame()
        df['fi'] = self.model.feature_importances_
        df['name'] = self.train_data.X.columns
        df = df.sort('fi')
        sns.barplot(x=df.name, y=df.fi)
        plt.xticks(rotation='vertical')
        sns.despine()
        
    
    def plot_dist_comp(self):
        plt.figure(figsize=(8,5))
        sns.distplot(self.results.array, label="Me")
        sns.distplot(self.espn_results.array, label="ESPN")
        plt.legend()
        sns.despine()
        
    
    def scatter_plot(self, x, y):
        plt.figure(figsize=(8,5))
        max_v = 40
        x_45 = np.arange(0, max_v)
        plt.scatter(x, y)
        plt.plot(x_45, x_45)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.ylim([0, max_v])
        plt.xlim([0, max_v])
    
    
    def week_df(self, proj_week, position_encoder, team_encoder, position = 'ALL'):
        
        # Get actual data for all previous weeks
        actual_data = pd.read_sql_query("""select name, team, position, opponent, week,
                                    at_home, total_points, won_game, opponent_score, team_score
                                    from scoring_leaders_weekly""", engine)
        
        # Calculate how team's perform on average against positions for fantasy points
        team_data = pd.DataFrame(actual_data.groupby(['opponent', 'position']).total_points.mean())
        team_data.reset_index(level=0, inplace=True)
        team_data.reset_index(level=0, inplace=True)
        team_data.rename(columns={'total_points': 'opponent_points'}, inplace=True)
        actual_data = actual_data.merge(team_data, on=['opponent', 'position'], how='left')
        team_data.rename(columns={'opponent_points': 'next_opponent_points', 'opponent': 'next_opponent'}, inplace=True)
        
        actual_data['index'] = actual_data.name + "_" + actual_data.team + "_" + actual_data.position
        actual_data.week = actual_data.week.astype(int)
        actual_data = actual_data[actual_data.week < proj_week]

        # Calculate the average values for previous week metrics
        wgt_df = actual_data[['opponent_points', 'index', 'at_home', 'total_points',
                              'won_game', 'opponent_score', 'team_score']]
        group_wgt_df = wgt_df.groupby('index')
        player_df = group_wgt_df.mean()
        player_df.reset_index(level=0, inplace=True)
        
        # Get the opponent data for the next week as well as espn projection
        predicted_data = pd.read_sql_query("""select name, team, position, opponent as next_opponent,
                                            at_home as next_at_home, total_points as predicted_points
                                            from next_week_projections
                                            where week = '{0}'""".format(proj_week), engine)
        predicted_data['index'] = predicted_data.name + "_" + predicted_data.team + "_" + predicted_data.position
        predicted_data.drop(['name', 'team'], axis=1, inplace=True)

        # Start combining everything - messy - sorry...
        X = player_df.merge(predicted_data, on='index', how='left')
        X = X.dropna()

        # Get the actual result as our target
        actual_result = pd.read_sql_query("""select name, team, position, total_points as actual_points
                                            from scoring_leaders_weekly
                                            where week = '{0}'""".format(proj_week), engine)
        actual_result['index'] = actual_result.name + "_" + actual_result.team + "_" + actual_result.position
        actual_result.drop(['name', 'team', 'position'], axis=1, inplace=True)

        X = X.merge(actual_result, on='index', how='left')
        X = X.merge(team_data, on=['position', 'next_opponent'], how='left')
        X = X.dropna()
        if position != 'ALL':
            X = X[X.position == position]
        y = X.actual_points

        X['team'] = X['index'].str.split("_").str.get(1)

        # Sklearn won't create factors for you, so encode the categories to integers
        X['team_factor'] = team_encoder.transform(X.team)
        if position == 'ALL':
            X['position_factor'] = position_encoder.transform(X.position)
        X['next_opponent_factor'] = team_encoder.transform(X.next_opponent)

        espn = X['predicted_points']
        index = X['index']
        X.drop(['predicted_points', 'actual_points', 'team', 'position', 'next_opponent', 'index'], axis=1, inplace=True)
        
        # Return named tuple of all the data I need
        week_tuple = namedtuple('Week', ['X', 'y', 'espn_proj', 'index'])
        return week_tuple(X, y, espn, index)
    
    
    def create_position_factors(self):
        # Convert positions into integer categories
        position_encoder = preprocessing.LabelEncoder()
        positions = np.ravel(pd.read_sql_query("""select distinct position from scoring_leaders_weekly;""", engine).values)
        position_encoder.fit(positions)

        # Convert team names into integer categories
        team_encoder = preprocessing.LabelEncoder()
        teams = np.ravel(pd.read_sql_query("""select distinct team from scoring_leaders_weekly;""", engine).values)
        team_encoder.fit(teams)
        encoders = namedtuple('encoders', ['team', 'position'])
        return encoders(team_encoder, position_encoder)
    
    
    def rmean_sq(self, y_true, y_pred):
        rmse = namedtuple('rmse', ['rmse', 'array'])
        sq_error = []
        assert len(y_true) == len(y_pred)
        for i in range(len(y_true)):
            sq_error.append((y_true[i] - y_pred[i])**2)
        return rmse(np.sqrt(np.mean(sq_error)), np.sqrt(sq_error))


# # Put my model to the test
# 
# If you look through the class above, you will see that I am using a random forest regressor as my model of choice with 5,000 trees and a max depth of 5. I did some cross validation testing with a few other models and hyper parameter selection and this seemed to work the best.
# 
# I use a handful of variables including things like a player's average fantasy points, the average number of game's a player has won, and a player's position (more on variables in a bit). I tried modeling each position individually, but a global model provided a better root mean squared error in the end.
# 
# So - lets try it out. What I will be doing is predicting fantasy points for players in week 3 of the season. To do this, I will use week 2 as my training set. Thus, I use week 1 data to train my model to predict week 2 outcomes. I then take this trained model and use it with week 1 and 2 data to predict week 3 fantasy points for players. Here we go...
# 

week_3_proj_all = Prediction(3)
week_3_proj_all.make_prediction()
week_3_proj_all.report()


# # Analysis
# 
# ## RMSE 
# 
# First, you will notice that I achieved a better RMSE than ESPN! Yay! The next thing you should notice is that I barely did better. But RMSE is just a single number, so lets dive a bit deeper.
# 
# ## RMSE Distribution
# 
# Looking at the second graph you will see both distributions of RMSE. ESPN is doing a better job of getting more of their prediction errors close to zero. While my model has less area in the tail, so doesn't have as many big prediction errors as ESPN does. I also have more prediction errors in the 5-10 value range. These results can also be seen in the scatter plots. ESPN's predictions don't perform very well with very high actual fantasy point values. My model also suffers from this problem. This seems to make sense; these are most likely big games for these players with above average results that would be hard to predict with any confidence. My big win seems to be for players performing in the 5-15 point range. My model handles these better than ESPN with less low predictions in this range. Both models also have a lot of errors when players have low actual values. My model tends to over predict these, while ESPN has over and under predictions. These are probably players with bad games, so my model based on past, better performance is over shoting. ESPN probably has some better domain knowledge that can sometimes lead to over predicting a bad performance. 
# 
# ## Feature Importance
# 
# I found these results to be very fascinating. You will immediately notice that **next_opponent_points** is very important in my model. This variable is the average fantasy points that the opponent team the player will be against in week 3 has allowed for the player's position in the previous weeks for the season. I think this is sometimes called points against and it is very important. This is something I really overlooked in my drafting. I didn't really consider a player's schedule at all. These results suggest that analyzing a player's schedule can be extremely important. A great running back could have some bad fantasy games if up against some good run defenses.
# 
# The next variable, **total_points** is the player's average fantasy points thus far in the season. **Opponent_points** is like **next_opponent_points** but it is the average value for that player's team for all the teams he has played thus far in the season. **Next_opponent_factor** is just the actual team the player will be playing in week 3. **Team_factor** is the player's team. **Team_score** is the average score for the player's team for past games. **Position_factor** is the player's position. **Opponent_score** is the same as **team_score** but for the player's opponents. **At_home** is average number of games at home. **Next_at_home** is whether next game is at home. **Won_game** is average number of games won in the past.
# 

data = week_3_proj_all.combined_test_df
data['my_error'] = np.sqrt((data['my_prediction'] - data['actual_points'])**2)
data['espn_error'] = np.sqrt((data['predicted_points'] - data['actual_points'])**2)
data = data[['position', 'my_error', 'espn_error']]
data = pd.melt(data, id_vars=['position'], value_vars=['my_error', 'espn_error'])
data.columns = ['position', 'type', 'error']


sns.barplot(x='position', y='error', hue='type', data=data)
sns.despine()


# # Results by Position
# 

# The above bar chart shows how the models do by position. There are not too many surprises here. Less volatile positions like K and TE have lower average errors while positions like QB and RB have higher average errors. ESPN beats me across the board on average except for RB (but all values are close). One sort of interesting point is that my model has a noteable smaller standard deviation for QBs. But again, all differences are slight.
# 

# # Conclusion
# 
# Predicting fantasy points is hard. On average the models are only off by about 6.5 points, but can miss pretty big when players do very well or poorly, which is really what you usually care about. This is inheritely a challenging problem - how do you use past data to try and determine a future performance that differs from that past (like a break out game or a slump)? On some level, I imagine this is impossible, but think we could maybe do better. For instance, my model doesn't account for injury at all. Knowing that a player is coming off an injury could be very valuable. Or knowing that a key offensive lineman is out could really affect a RBs performance. I think the NFL has a lot of cool work that could be done with the right data and some deep thought about how these data could be used to understand performance.
# 
# Second, I was very happy to gain some more insight into what variables seem to matter when trying to predict a player's fantasy points. In the future, I will be paying a lot more attention to a player's opponents.
# 
# Lastly, I think it is really cool that with only a handful of variables a model could be developed that performs very similar to ESPN's prediction model. This is what I **love** about data science. I was able to use data to gain a much deeper undertanding of an area in which I am far from an expert. I have no idea how ESPN creates their predictions, but I imagine they put some effort into it, and I think it is awesome that with just my laptop and some data I could create a competing model. If anyone from ESPN is reading this...I'd love to connect and learn some more about your model :). 
# 
# P.S. - I will try and post some updates on the performance of the models as the season progresses if anyone is interested.
# 

