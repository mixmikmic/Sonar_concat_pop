# Stitch Fix, Jupyter, GitHub, and the Long Tail
# ==
# 
# At Stitch Fix we are avid users of Jupyter for research at both the personal and team scales. At the personal level, Jupyter is a great interface to research the question at hand. It captures the workflow of the research where we can take detailed notes on the code and explain models with written content and mathematical equations. 
# 
# At the team level, Jupyter is a great tool for communication. Notebooks allow one to fluidly mix text, code, equations, data, and plots in whatever way makes the most sense to explain something. You can organize your explanation around your thought process rather than around the artificial lines determined by the tools you’re using.  You have a single Jupyter notebook instead of a bunch of disconnected files, one containing SQL code, another Python, a third LaTeX to typeset equations.
# 
# When the analysis is finished, the Jupyter notebook remains a “living” interactive research notebook.  Re-doing the analysis using new data or different assumptions is simple.
# 
# Stitch Fix provides clothing to its clients based on their style preferences.  We send a client five pieces of clothing we predict they’ll like, and the client chooses what to keep. Inevitably, some pieces of clothing will be more popular than others. In some cases, a few select items may be unpopular. 
# 
# The largest benefit from adding a single style of clothing to our line of inventory comes from the most popular one. Each of the less popular styles, by itself, contributes less.  However, there are *many* of the less popular ones, reflecting the fact that our clients are unique in their fashion preferences. Together, the value in the "long tail" can match or exceed the value of the few products in the "head."  Catering to the long tail allows us to save our clients the time they would otherwise spend searching through many retail stores to find clothing that’s unique to their tastes.
# 
# But, where do we draw the line on how far into the long tail we should support? Below we investigate this question using the Jupyter Notebook. The portability and flexibility of the Notebooks allows us to easily share the analysis with others.  GitHub integration allows a great new possibility: other researches can fork the notebook to extend or alter the analysis according to their own particular interests!
# 

# Is the value in the head or the tail?
# --
# 
# We will approximate the number of each style of clothing sold as a power law of the rank $r$ by sales volume.  The most popular style has $r=1$, the second most popular $r=2$, and the least popular has $r=N$.  Consumer preferences dictate the shape of the curve. 
# 
# Even though we may want to carry an infinite number of styles of clothing, it's important to keep $N$ finite so that the integrals converge!  For the moment we will consider a scaled-down version of Stitch Fix that only carrys 100 styles of clothing and sells a volume of $V=5,000$ units per year.
# 
# The volume of each style sold is
# \begin{equation}
# v(r) = \frac{A}{r^n}
# \end{equation}
# where $A$ is a normalization constant and $n$ is the index of the power law.  The value of $n$ will determine how much value is in the head versus the tail.
# 
# Approximating the product distribution as continuous so it can be written as an integral, the normalization constant is set by the constraint
# \begin{equation}
# \int_1^N \frac{A\,dr}{r^n} = V
# \end{equation}
# so
# \begin{equation}
# A = \frac{(n-1) V}{1-N^{1-n}} 
# \end{equation}
# 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')
sns.set_context('talk')
sns.set_style('darkgrid')


inventory = 100.0
volume = 5000.0

rr = np.linspace(1,inventory,100)
ns = [0.25, 0.75, 1.25, 1.75]

fig, ax = plt.subplots(figsize=(10, 6))

for nn in ns:
    norm = (nn-1)*volume/(1-inventory**(1-nn))
    ax.plot(rr, norm/rr**nn, label='$n=%g$' % nn)

ax.legend()
ax.set_xlabel('Rank by Sales Volume $r$')
ax.set_ylabel('Units Sold')
ax.set_title('Sales volume of each product by rank')
ax.set_ylim(0,100)


# All of these distributions have the same area under the curve, so they represent the same total number of units sold.  Smaller values of $n$ give flatter distributions (less head, more tail) and larger values of $n$ give more head-heavy distributions.
# 
# What is the _total_ value in the head versus the tail?  Define the head to be the 10% of styles with the largest sales volume, the tail to be the 50% of styles with the lowest sales volumes, and the middle to be those in between.
# 
# That is, the head, tail, and middle look like this:
# 

# Same plot as above
fig, ax = plt.subplots(figsize=(10, 6))

for nn in ns:
    norm = (nn-1)*volume/(1-inventory**(1-nn))
    ax.plot(rr, norm/rr**nn, label='$n=%g$' % nn)

ax.set_xlabel('Rank by Sales Volume $r$')
ax.set_ylabel('Units Sold')
ax.set_title('Sales volume of each product by rank')
ax.set_ylim(0,100)

# Ask seaborn for some pleasing colors
c1, c2, c3 = sns.color_palette(n_colors=3)

# Add transparent rectangles
head_patch = plt.matplotlib.patches.Rectangle((1,0), 9, 100, alpha=0.25, color=c1)
middle_patch = plt.matplotlib.patches.Rectangle((11,0), 39, 100, alpha=0.25, color=c2)
tail_patch = plt.matplotlib.patches.Rectangle((51,0), 48, 100, alpha=0.25, color=c3)
ax.add_patch(head_patch)
ax.add_patch(middle_patch)
ax.add_patch(tail_patch)

# Add text annotations
ax.text(5,50,"Head", color=c1, fontsize=16, rotation=90)
ax.text(25,80,"Middle", color=c2, fontsize=16)
ax.text(75,80,"Tail", color=c3, fontsize=16)


# How many units from the head, tail, and middle are sold?  Integrate over the sales rank distribution to get the sales volume in the head:
# 
# \begin{equation}
# V_H = \int_1^{f_H N} \frac{A\, dr}{r^n} = \frac{V(N^{n-1} - f_H^{1-n})}{N^{n-1} - 1}
# \end{equation}
# where $f_H=0.1$
# 
# The volume in the tail is
# 
# \begin{equation}
# V_T = \int_{f_T N}^N \frac{A\, dr}{r^n} = \frac{V(f_T^{1-n}-1)}{N^{n-1}-1}
# \end{equation}
# where $f_T=0.5$
# 
# And the middle:
# 
# \begin{equation}
# V_M = \int_{f_H N}^{f_T N} \frac{A\, dr}{r^n} = \frac{V(f_H^{1-n} - f_T^{1-n})}{N^{n-1}-1}
# \end{equation}
# 

f_head = 0.1
f_tail = 0.5

ns = np.linspace(0,2,100)
nm1 = ns-1.0

head = volume*(inventory**nm1 - f_head**-nm1)/(inventory**nm1-1)
middle = volume*(f_head**-nm1 - f_tail**-nm1)/(inventory**nm1-1)
tail = volume*(f_tail**-nm1 - 1)/(inventory**nm1-1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ns, head, label='Head')
ax.plot(ns, middle, label='Middle')
ax.plot(ns, tail, label='Tail')
ax.legend(loc='upper left')
ax.set_ylabel('Units Sold')
ax.set_xlabel('Power law index $n$')


# For $n>1$, the head has most of value.  As $n$ falls, the middle and tail become important.
# 

# How many styles of clothing should we carry?
# --
# 
# We can choose expand our inventory from $N$ to $N+1$ styles of clothing.  How many additional units will we sell?
# 
# This is just the sales volume distribution $n(r)$ evaluated at $r=N+1$
# 
# \begin{equation}
# \frac{d V}{d N} = \frac{(n-1) V}{(1-N^{1-n})(N+1)^n} 
# \end{equation}
# 

marginal_benefit = ((ns-1)*volume)/((1-inventory**(1-ns))*(inventory+1)**ns)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ns, marginal_benefit)
ax.set_ylabel('Additional Units Sold')
ax.set_xlabel('Power law index $n$')
ax.set_title('Marginal Benefit of Expanding Inventory')


# Our scaled down version of Stitch Fix can expect to sell an additional $\simeq 40$ pieces of clothing per year if the sales distribution has $n=0.25$, or an additional $\simeq 3$ units per year if $n=1.5$  Whether expanding the inventory is worth it to get those additional sales depends entirely on how much it costs to bring one additional product into the inventory.
# 

# # NBA Free throw analysis
# 
# Now let's see some of these methods in action on real world data.
# I'm not a basketball guru by any means, but I thought it would be fun to see whether we can find players that perform differently in free throws when playing at home versus away.
# [Basketballvalue.com](http://basketballvalue.com/downloads.php) has
# some nice play by play data on season and playoff data between 2007 and 2012, which we will use for this analysis.
# It's not perfect, for example it only records player's last names, but it will do for the purpose of demonstration.
# 
# 
# ## Getting data:
# 
# - Download and extract play by play data from 2007 - 2012 data from http://basketballvalue.com/downloads.php
# - Concatenate all text files into file called `raw.data`
# - Run following to extract free throw data into `free_throws.csv`
# ```
# cat raw.data | ack Free Throw | sed -E 's/[0-9]+([A-Z]{3})([A-Z]{3})[[:space:]][0-9]*[[:space:]].?[0-9]{2}:[0-9]{2}:[0-9]{2}[[:space:]]*\[([A-z]{3}).*\][[:space:]](.*)[[:space:]]Free Throw.*(d|\))/\1,\2,\3,\4,\5/ ; s/(.*)d$/\10/ ; s/(.*)\)$/\11/' > free_throws.csv
# ```
# 

from __future__ import division

import pandas as pd
import numpy as np
import scipy as sp

import scipy.stats

import toyplot as tp


# ## Data munging
# 
# Because only last name is included, we analyze "player-team" combinations to avoid duplicates.
# This could mean that the same player has multiple rows if he changed teams.
# 

df = pd.read_csv('free_throws.csv', names=["away", "home", "team", "player", "score"])

df["at_home"] = df["home"] == df["team"]

df.head()


# ## Overall free throw%
# 
# We note that at home the ft% is slightly higher, but there is not much difference
# 

df.groupby("at_home").mean()


# ## Aggregating to player level
# 
# We use a pivot table to get statistics on every player.
# 

sdf = pd.pivot_table(df, index=["player", "team"], columns="at_home", values=["score"], 
               aggfunc=[len, sum], fill_value=0).reset_index()
sdf.columns = ['player', 'team', 'atm_away', 'atm_home', 'score_away', 'score_home']

sdf['atm_total'] = sdf['atm_away'] + sdf['atm_home']
sdf['score_total'] = sdf['score_away'] + sdf['score_home']

sdf.sample(10)


# ## Individual tests
# 
# For each player, we assume each free throw is an independent draw from a Bernoulli distribution with probability $p_{ij}$ of succeeding where $i$ denotes the player and $j=\{a, h\}$ denoting away or home, respectively.
# 
# Our null hypotheses are that there is no difference between playing at home and away, versus the alternative that there is a difference.
# While you could argue a one-sided test for home advantage is also appropriate, I am sticking with a two-sided test.
# 
# $$\begin{aligned}
# H_{0, i}&: p_{i, a} = p_{i, h},\H_{1, i}&: p_{i, a} \neq p_{i, h}.
# \end{aligned}$$
# 
# To get test statistics, we conduct a simple two-sample proportions test, where our test statistic is:
# 
# $$Z = \frac{\hat p_h - \hat p_a}{\sqrt{\hat p (1-\hat p) (\frac{1}{n_h} + \frac{1}{n_a})}}$$
# 
# where
# - $n_h$ and $n_a$ are the number of attempts at home and away, respectively
# - $X_h$ and $X_a$ are the number of free throws made at home and away
# - $\hat p_h = X_h / n_h$ is the MLE for the free throw percentage at home
# - likewise, $\hat p_a = X_a / n_a$ for away ft%
# - $\hat p = \frac{X_h + X_a}{n_h + n_a}$ is the MLE for overall ft%, used for the pooled variance estimator 
# 
# Then we know from Stats 101 that $Z \sim N(0, 1)$ under the null hypothesis that there is no difference in free throw percentages.
# 
# For a normal approximation to hold, we need $np > 5$ and $n(1-p) > 5$, since $p \approx 0.75$, let's be a little conservative and say we need at least 50 samples for a player to get a good normal approximation.
# 
# This leads to data on 936 players, and for each one we compute Z, and the corresponding p-value.
# 

data = sdf.query('atm_total > 50').copy()
len(data)


data['p_home'] = data['score_home'] / data['atm_home']
data['p_away'] = data['score_away'] / data['atm_away']
data['p_ovr'] = (data['score_total']) / (data['atm_total'])

# two-sided
data['zval'] = (data['p_home'] - data['p_away']) / np.sqrt(data['p_ovr'] * (1-data['p_ovr']) * (1/data['atm_away'] + 1/data['atm_home']))
data['pval'] = 2*(1-sp.stats.norm.cdf(np.abs(data['zval'])))

# one-sided testing home advantage
# data['zval'] = (data['p_home'] - data['p_away']) / np.sqrt(data['p_ovr'] * (1-data['p_ovr']) * (1/data['atm_away'] + 1/data['atm_home']))
# data['pval'] = (1-sp.stats.norm.cdf(data['zval']))


data.sample(10)


canvas = tp.Canvas(800, 300)
ax1 = canvas.axes(grid=(1, 2, 0), label="Histogram p-values")
hist_p = ax1.bars(np.histogram(data["pval"], bins=50, normed=True), color="steelblue")
hisp_p_density = ax1.plot([0, 1], [1, 1], color="red")
ax2 = canvas.axes(grid=(1, 2, 1), label="Histogram z-values")
hist_z = ax2.bars(np.histogram(data["zval"], bins=50, normed=True), color="orange")
x = np.linspace(-3, 3, 200)
hisp_z_density = ax2.plot(x, sp.stats.norm.pdf(x), color="red")


# # Global tests
# 
# We can test the global null hypothesis, that is, there is no difference in free throw % between playing at home and away for any player using both Fisher's Combination Test and the Bonferroni method.
# Which one is preferred in this case? I would expect to see many small difference in effects rather than a few players showing huge effects, so Fisher's Combination Test probably has much better power.
# 
# ## Fisher's combination test
# 
# We expect this test to have good power: if there is a difference between playing at home and away we would expect to see a lot of little effects.
# 

T = -2 * np.sum(np.log(data["pval"]))
print 'p-value for Fisher Combination Test: {:.3e}'.format(1 - sp.stats.chi2.cdf(T, 2*len(data)))


# ## Bonferroni's method
# 
# The theory would suggest this test has a lot less power, it's unlikely to have a few players where the difference is relatively huge.
# 

print '"p-value" Bonferroni: {:.3e}'.format(min(1, data["pval"].min() * len(data)))


# ## Conclusion
# 
# Indeed, we find a small p-value for Fisher's Combination Test, while Bonferroni's method does not reject the null hypothesis.
# In fact, if we multiply the smallest p-value by the number of hypotheses, we get a number larger than 1, so we aren't even remotely close to any significance.
# 

# # Multiple tests
# 
# So there definitely seems some evidence that there is a difference in performance.
# If you tell a sport's analyst that there is evidence that at least some players that perform differently away versus at home, their first question will be: "So who is?"
# Let's see if we can properly answer that question.
# 
# ## Naive method
# 
# Let's first test each null hypothesis ignoring the fact that we are dealing with many hypotheses. Please don't do this at home!
# 

alpha = 0.05
data["reject_naive"] = 1*(data["pval"] < alpha)

print 'Number of rejections: {}'.format(data["reject_naive"].sum())


# If we don't correct for multiple comparisons, there are actually 65 "significant" results (at $\alpha = 0.05$), which corresponds to about 7% of the players.
# We expect around 46 rejections by chance, so it's a bit more than expected, but this is a bogus method so no matter what, we should discard the results.
# 
# 
# 
# ## Bonferroni correction
# 
# Let's do it the proper way though, first using Bonferroni correction.
# Since this method is basically the same as the Bonferroni global test, we expect no rejections:
# 

from statsmodels.sandbox.stats.multicomp import multipletests


data["reject_bc"] = 1*(data["pval"] < alpha / len(data))
print 'Number of rejections: {}'.format(data["reject_bc"].sum())


# Indeed, no rejections.
# 
# ## Benjamini-Hochberg
# 
# Let's also try the BHq procedure, which has a bit more power than Bonferonni.
# 

is_reject, corrected_pvals, _, _ = multipletests(data["pval"], alpha=0.1, method='fdr_bh')


data["reject_fdr"] = 1*is_reject
data["pval_fdr"] = corrected_pvals
print 'Number of rejections: {}'.format(data["reject_fdr"].sum())


# Even though the BHq procedure has more power, we can't reject any of the individual hypothesis, hence we don't find sufficient evidence for any of the players that free throw performance is affected by location.
# 
# 
# # Taking a step back
# 
# If we take a step back and take another look at our data, we quickly find that we shouldn't be surprised with our results.
# In particular, our tests are clearly underpowered. 
# That is, the probability of rejecting the null hypothesis when there is a true effect is very small given the effect sizes that are reasonable.
# 
# While there are definitely sophisticated approaches to power analysis, we can use a [simple tool](http://statpages.org/proppowr.html) to get a rough estimate.
# The free throw% is around 75% percent, and at that level it takes almost 2500 total attempts to detect a difference in ft% of 5% ($\alpha = 0.05$, power = $0.8$), and 5% is a pretty remarkable difference when only looking at home and away difference.
# For most players, the observed difference is not even close to 5%, and we have only 11 players in our dataset with more than 2500 free throws.
# 
# 
# To have any hope to detect effects for those few players that have plenty of data, the worst thing one can do is throw in a bunch of powerless tests.
# It would have been much better to restrict our analysis to players where we have a lot of data.
# Don't worry, I've already done that and again we cannot reject a single hypothesis.
# 
# So unfortunately it seems we won't be impressing our friends with cool results, more likely we will be the annoying person pointing out the fancy stats during a game don't really mean anything.
# 
# There is one cool take-away though: Fisher's combination test did reject the global null hypothesis even though each single test had almost no power, combined they did yield a significant result.
# If we aggregate the data across all players first and then conduct a single test of proportions, it turns out we cannot reject that hypothesis.
# 

len(data.query("atm_total > 2500"))


reduced_data = data.query("atm_total > 2500").copy()

is_reject2, corrected_pvals2, _, _ = multipletests(reduced_data["pval"], alpha=0.1, method='fdr_bh')
reduced_data["reject_fdr2"] = 1*is_reject2
reduced_data["pval_fdr2"] = corrected_pvals2


print 'Number of rejections: {}'.format(reduced_data["reject_fdr2"].sum())


