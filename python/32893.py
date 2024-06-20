# # Project 5: Capstone Project
# ### Learning to Trade Using Q-Learning
# <sub>Uirá Caiado. Aug 10, 2016<sub>
# 
# 
# #### Abstract
# 
# *In this project, I will present an adaptive learning model to trade a single stock under the reinforcement learning framework. This area of machine learning consists in training an agent by reward and punishment without needing to specify the expected action. The agent learns from its experience and develops a strategy that maximizes its profits. The simulation results show initial success in bringing learning techniques to build algorithmic trading strategies.*
# 

# ## 1. Introduction
# 
# In this section, I will provide a high-level overview of the project, define the problem addressed and the metric used to measure the performance of the model created.
# 
# ### 1.1. Project Overview
# ```
# Udacity:
# 
# In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
# - Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?
# - Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?
# ```
# Nowadays, algo trading represents almost half of all cash equity trading in western Europe. In advanced markets, it already [accounts](http://en.resenhadabolsa.com.br/portfolio-category/the-distributionintermediation-industry-in-brazil-challenges-and-trends/) for over 40%-50% of total volume. In Brazil its market share is not as large – currently about 10% – but is expected to rise in the years ahead as markets and players go digital.
# 
# As automated strategies are becoming increasingly popular, building an intelligent system that can trade many times a day and adapts itself to the market conditions and still consistently makes money is a subject of keen interest of any market participant.
# 
# Given that it is hard to produce such strategy, in this project I will try to build an algorithm that just does better than a random agent, but learns by itself how to trade. To do so, I will feed my agent with four days of information about every trade and change in the [top of the order book](https://goo.gl/k1dDYZ) in the [PETR4](https://pt.wikipedia.org/wiki/Petrobras) - one of the most liquidity assets in Brazilian Stock Market - in a Reinforcement Learning Framework. Later on, I will test what it has learned in a newest dataset.
# 
# The dataset used in this project is also known as [level I order book data](https://www.thebalance.com/order-book-level-2-market-data-and-depth-of-market-1031118) and includes all trades and changes in the prices and total quantities at best Bid (those who wants to buy the stock) and Offer side (those who intends to sell the stock).

# 
# ### 1.2. Problem Statement
# ```
# Udacity:
# 
# In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
# - Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?
# - Have you thoroughly discussed how you will attempt to solve the problem?
# - Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?
# ```
# [Algo trading](http://goo.gl/b9jAqE) strategies usually are programs that follow a predefined set of instructions to place its orders. 
# 
# The primary challenge to this approach is building these rules in a way that it can consistently generate profit without being too sensitive to market conditions. Thus, the goal of this project is to develop an adaptive learning model that can learn by itself those rules and trade a particular asset using reinforcement learning framework under an environment that replays historical high-frequency data.
# 
# As \cite{chan2001electronic} described, reinforcement learning can be considered as a model-free approximation of dynamic programming. The knowledge of the underlying processes is not assumed but learned from experience. The agent can access some information about the environment state as the order flow imbalance, the sizes of the best bid and offer and so on. At each time step $t$, It should generate some valid action, as buy stocks or insert a limit order at the Ask side. The agent also should receive a reward or a penalty at each time step if it is already carrying a position from previous rounds or if it has made a trade (the cost of the operations are computed as a penalty). Based on the rewards and penalties it gets, the agent should learn an optimal policy for trade this particular stock, maximizing the profit it receives from its actions and resulting positions.
# 
# ```
# Udacity Reviewer:
# 
# This is really quite close! I'm marking as not meeting specifications because you should fully outline your solution here. You've outlined your strategy regarding reinforcement learning, but you should also address things like data preprocessing, choosing your state space etc. Basically, this section should serve as an outline for your entire solution. Just add a paragraph or two to fully outline your proposed methodology and you're good to go.
# ```
# 
# This project starts with an overview of the dataset and shows how the environment states will be represented in Section 2. The same section also dives in the reinforcement learning framework and defines the benchmark used at the end of the project. Section 3 discretizes the environment states by transforming its variables and clustering them into six groups. Also describes the implementation of the model and the environments, as well as and the process of improvement made upon the algorithm used. Section 4 presents the final model and compares statistically its performance to the benchmark chosen. Section 5 concludes the project with some closing remarks and possible improvements.
# 

# 
# ### 1.3. Metrics
# ```
# Udacity:
# 
# In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
# - Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?
# - Have you provided reasonable justification for the metrics chosen based on the problem and solution?
# ```
# 
# ```
# Udacity Reviewer:
# 
# The section on metrics should address any statistics or metrics that you'll be using in your report. What you've written in your benchmark section is roughly what we're looking for for the metrics section and vice versa. I'd recommend changing the subtitles to clarify this. If it's more logical to introduce the benchmark before explaining your metrics, you could combine the 'Benchmark' and 'Metrics' subsections into a single 'Benchmark and Metrics' section. 
# ```
# 
# Different metrics are used to support the decisions made throughout the project. We use the mean [Silhouette Coefficient](http://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient) of all samples to justify the clustering method chosen to reduce the state space representation of the environment. As exposed in the scikit-learn documentation, this coefficient is composed by the mean intra-cluster distance ($a$) and the mean nearest-cluster distance ($b$) for each sample. The score for a single cluster is given by $s = \frac{b-a}{\max{a, \, b}}$.This scores are so average down to all samples and varying between $1$ (the best one) and $-1$ (the worst value).
# 
# Then, we use [sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio) to help us understanding the performance impact of different values to the model parameters. The Sharpe is measure upon the first difference ($\Delta r$) of the accumulated PnL curve of the model. So, the first difference is defined as $\Delta r = PnL_t - PnL_{t-1}$.
# 
# Finally, as we shall justify latter, the performance of my agent will be compared to the performance of a random agent. These performances will be measured primarily of Reais made (the Brazilian currency) by the agents. To compared the final PnL of both agents in the simulations, we will perform a one-sided [Welch's unequal variances t-test](https://goo.gl/Je2ZLP) for the null hypothesis that the learning agent has the expected PnL greater than the random agent. As the implementation of the [t-test in the scipy](https://goo.gl/gs222c) assumes a two-sided t-test, to perform the one-sided test, we will divide the p-value by $2$ to compare to a critical value of $0.05$ and requires that the t-value is greater than zero. In the next section, I will detail the behavior of learning agent.

# ## 2. Analysis
# 
# In this section, I will explore the data set that will be used in the simulation, define and justify the inputs employed in the state representation of the algorithm, explain the reinforcement learning techniques used and provide a benchmark.
# 
# ### 2.1. Data Exploration
# ```
# Udacity:
# 
# In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
# - If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?
# - If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?
# - If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?
# - Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)
# ```
# The dataset used is composed by level I order book data from PETR4, a stock traded at BMFBovespa Stock Exchange. Includes 45 trading sessions from 07/25/2016 to 09/26/2016. I will use one day to create the scalers of the features used, that I shall explain. Then, I will use four days to train and test the model, and after each training session, I will validate the policy found in an unseen dataset from the subsequent day. The data was collected from Bloomberg. 
# 
# In the figure below can be observed how the market behaved on the days that the out-of-sample will be performed. In this charts are plotted the number of cents that an investment of the same amount of money in PETR4 and BOVA11 would have varied on these market sessions. BOVA11 is an [ETF](https://en.wikipedia.org/wiki/Exchange-traded_fund) that can be used as a proxy to the [Bovespa Index](https://goo.gl/TsuUVH), the Brazilian Stock Exchange Index. As can seem, PETR4 was relatively more volatile than the rest of the market.

import zipfile
s_fname = "data/data_0725_0926.zip"
s_fname2 = "data/bova11_2.zip"
archive = zipfile.ZipFile(s_fname, 'r')
archive2 = zipfile.ZipFile(s_fname2, 'r')
l_fnames = archive.infolist()


import qtrader.eda as eda; reload(eda);
df_last_pnl = eda.plot_cents_changed(archive, archive2)


# *Let's start by looking at the size of the files that can be used in the simulation:*
# 

def foo():
    f_total = 0.
    f_tot_rows = 0.
    for i, x in enumerate(archive.infolist()):
        f_total += x.file_size/ 1024.**2
        for num_rows, row in enumerate(archive.open(x)):
            f_tot_rows += 1
        print "{}:\t{:,.0f} rows\t{:0.2f} MB".format(x.filename, num_rows + 1, x.file_size/ 1024.**2)
    print '=' * 42
    print "TOTAL\t\t{} files\t{:0.2f} MB".format(i+1,f_total)
    print "\t\t{:0,.0f} rows".format(f_tot_rows)

get_ipython().magic('time foo()')


# There are 45 files, each one has 110,000 rows on average, resulting in 5,631,273 rows at total and almost  230 MB of information. Now, let's look at the structure of one of them:
# 

import pandas as pd
df = pd.read_csv(archive.open(l_fnames[0]), index_col=0, parse_dates=['Date'])
df.head()


# Each file is composed of four different fields. The column $Date$ is the timestamp of the row and has a precision of seconds. $Type$ is the kind of information that the row encompasses. The type "TRADE" relates to an actual trade that has happened. "BID" is related to changes in the best Bid level and "ASK," to the best Offer level. $Price$ is the current best bid or ask and $Size$ is the cumulated quantity on that price and side.
# 
# All this data will be used to create the environment where my agent will operate. This environment is an order book, where the agent will be able to insert limit orders and execute trades at the best prices. The order book is represented by two binary trees, one for the Bid and other for the Ask side. As can be seen in the table below, the nodes of these trees are sorted by price (price level) in ascending order on the Bid side and descending order on the ask side. At each price level, there are other binary trees sorted by order of arrival. The first order to arrive is the first order filled when coming in a trade.
# 

import qtrader.simulator as simulator
import qtrader.environment as environment
e = environment.Environment()
sim = simulator.Simulator(e)
get_ipython().magic('time sim.run(n_trials=1)')


sim.env.get_order_book()


# The environment will answer with the agent's current position and Profit and Loss (PnL) every time the agent executes a trade or has an order filled. The cost of the trade will be accounted as a penalty.
# 
# The agent also will be able to sense the state of the environment and include it in its own state representation. So, this intern state will be represented by a set of variables about the current situation os the market and the state of the agent, given by:
# 
# - $qOFI$ : integer. The net order flow at the bid and ask in the last 10 seconds
# - $book\_ratio$ : float. The Bid size over the Ask size
# - $position$: integer. The current position of my agent. The maximum is $100$
# - $OrderBid$: boolean. If the agent has order at the bid side
# - $OrderAsk$: boolean. If the agent has order at the ask side
# 

# 
# ```
# Udacity:
# 
# Exploratory Visualization:
# 
# In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
# - Have you visualized a relevant characteristic or feature about the dataset or input data?
# - Is the visualization thoroughly analyzed and discussed?
# - If a plot is provided, are the axes, title, and datum clearly defined?
# ```

# Regarding the measure of the Order Flow Imbalance (OFI), there are many ways to measure it. \cite{cont2014price} argued the *order flow imbalance* is a measure of supply/demand imbalance and defines it as a sum of individual event contribution $e_n$ over time intervals $\left[ t_{k-1}, \; t_k \right]$, such that:
# 
# $$OFI_k = \sum^{N(t_k)}_{n=N(t_{k-1})+1} e_n$$
# 
# Where $N(t_k)$ and $N(t_{k-1}) + 1$ are index of the first and last event in the interval. The $e_n$ was defined by the authors as a measure of the contribution of the $n$-th event to the size of the bid and ask queues:
# 
# $$e_n = \mathbb{1}_{P_{n}^{B} \geq P_{n-1}^{B}} q^{B}_{n} - \mathbb{1}_{P_{n}^{B} \leq P_{n-1}^{B}}  q^{B}_{n-1} - \mathbb{1}_{P_{n}^{A} \leq P_{n-1}^{A}} q^{A}_{n} + \mathbb{1}_{P_{n}^{A} \geq P_{n-1}^{A}}  q^{A}_{n-1}$$
# 
# Where $q^{B}_{n}$ and $q^{A}_{n}$ are linked to the cumulated quantities at the best bid and ask in the time $n$. The subscript $n-1$ is related to the last observation. $\mathbb{1}$ is an [indicator](https://en.wikipedia.org/wiki/Indicator_function) function. In the figure below is ploted the 10-second log-return of PETR4 against the contemporaneous OFI. [Log-return](https://quantivity.wordpress.com/2011/02/21/why-log-returns/) is defined as $\ln{r_t} = \ln{\frac{P_t}{P_{t-1}}}$, where $P_t$ is the current price of PETR4 and $P_{t-1}$ is the previous one.
# 

import qtrader.eda as eda; reload(eda);
s_fname = "data/petr4_0725_0818_2.zip"
get_ipython().magic('time eda.test_ofi_indicator(s_fname, f_min_time=20.)')


import pandas as pd
df = pd.read_csv('data/ofi_petr.txt', sep='\t')
df.drop('TIME', axis=1, inplace=True)
df.dropna(inplace=True)
ax = sns.lmplot(x="OFI", y="LOG_RET", data=df, markers=["x"], palette="Set2", size=4, aspect=2.)
ax.ax.set_title(u'Relation between the Log-return and the $OFI$\n', fontsize=15);
ax.ax.set_ylim([-0.004, 0.005])
ax.ax.set_xlim([-400000, 400000])


# As described by \cite{cont2014price} in a similar test, the figure suggests that order flow imbalance is a stronger driver of high-frequency price changes and this variable will be used to describe the current state of the order book.
# 

# ### 2.2. Algorithms and Techniques
# ```
# Udacity:
# 
# In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
# - Are the algorithms you will use, including any default variables/parameters in the project clearly defined?
# - Are the techniques to be used thoroughly discussed and justified?
# - Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?
# ```
# Based on \cite{cont2014price}, the algo trading might be conveniently modeled in the framework of reinforcement learning. As suggested by \cite{du1algorithm}, this framework adjusts the parameters of an agent to maximize the expected payoff or reward generated due to its actions. Therefore, the agent learns a policy that tells him the actions it must perform to achieve its best performance. This optimal policy is exactly what we hope to find when we are building an automated trading strategy.
# 
# According to \cite{chan2001electronic}, Markov decision processes (MDPs) are the most common model when implementing reinforcement learning.  The MDP model of the environment consists, among other things, of a discrete set of states $S$ and a discrete set of actions taken from $A$. In this project, depending on the position of the learner(long or short), at each time step $t$ it will be allowed to choose an action $a_t$ from different subsets from the action space $A$ , that consists of six possibles actions:
# 
# $$a_{t} \in \left (None,\, buy,\, sell,\, best\_bid,\, best\_ask,\, best\_both \right)$$
# 
# Where $None$ indicates that the agent shouldn't have any order in the market. $Buy$ and $Sell$ means that the agent should execute a market order to buy or sell $100$ stocks (the size of an order will always be a hundred shares). This kind of action will be allowed based on a [trailing stop](https://goo.gl/SVmVzJ) of 4 cents. $best\_bid$ and $best\_ask$ indicate that the agent should keep order at best price just in the mentioned side and $best\_both$, it should have ordered at best price in both sides.

# So, at each discrete time step $t$, the agent senses the current state $s_t$ and choose to take an action $a_t$. The environment responds by providing the agent a reward $r_t=r(s_t, a_t)$ and by producing the succeeding state $s_{t+1}=\delta(s_t, a_t)$. The functions $r$ and $\delta$ only depend on the current state and action (it is [memoryless](https://en.wikipedia.org/wiki/Markov_process)), are part of the environment and are not necessarily known to the agent.
# 
# The task of the agent is to learn a policy $\pi$ that maps each state to an action ($\pi: S \rightarrow A$), selecting its next action $a_t$ based solely on the current observed state $s_t$, that is $\pi(s_t)=a_t$. The optimal policy, or control strategy, is the one that produces the greatest possible cumulative reward over time. So, stating that:
# 
# $$V^{\pi}(s_t)= r_t + \gamma r_{t+1} + \gamma^2 r_{t+1} + ... = \sum_{i=0}^{\infty} \gamma^{i} r_{t+i}$$
# 
# Where $V^{\pi}(s_t)$ is also called the discounted cumulative reward and it represents the cumulative value achieved by following an policy $\pi$ from an initial state $s_t$ and $\gamma \in [0, 1]$ is a constant that determines the relative value of delayed versus immediate rewards. It is one of the
# 
# If we set $\gamma=0$, only immediate rewards is considered. As $\gamma \rightarrow 1$, future rewards are given greater emphasis relative to immediate reward. The optimal policy $\pi^{*}$ that will maximizes $V^{\pi}(s_t)$ for all states $s$ can be written as:
# 
# $$\pi^{*} = \underset{\pi}{\arg \max} \, V^{\pi} (s)\,\,\,\,\,, \,\, \forall s$$
# 
# However, learning $\pi^{*}: S \rightarrow A$ directly is difficult because the available training data does not provide training examples of the form $(s, a)$. Instead, as \cite{Mitchell} explained, the only available information is the sequence of immediate rewards $r(s_i, a_i)$ for $i=1,\, 2,\, 3,\,...$
# 
# So, as we are trying to maximize the cumulative rewards $V^{*}(s_t)$ for all states $s$, the agent should prefer $s_1$ over $s_2$ wherever $V^{*}(s_1) > V^{*}(s_2)$. Given that the agent must choose among actions and not states, and it isn't able to perfectly predict the immediate reward and immediate successor for every possible state-action transition, we also must learn $V^{*}$ indirectly.
# 
# To solve that, we define a function $Q(s, \, a)$ such that its value is the maximum discounted cumulative reward that can be achieved starting from state $s$ and applying action $a$ as the first action. So, we can write:
# 
# $$Q(s, \, a) = r(s, a) + \gamma V^{*}(\delta(s, a))$$
# 
# As $\delta(s, a)$ is the state resulting from applying action $a$ to state $s$ (the successor) chosen by following the optimal policy, $V^{*}$ is the cumulative value of the immediate successor state discounted by a factor $\gamma$. Thus,  what we are trying to achieve is
# 
# $$\pi^{*}(s) = \underset{a}{\arg \max} Q(s, \, a)$$
# 
# It implies that the optimal policy can be obtained even if the agent just uses the current action $a$ and state $s$ and chooses the action that maximizes $Q(s,\, a)$. Also, it is important to notice that the function above implies that the agent can select optimal actions even when it has no knowledge of the functions $r$ and $\delta$.
# 
# Lastly, according to \cite{Mitchell}, there are some conditions to ensure that the reinforcement learning converges toward an optimal policy. On a deterministic MDP, the agent must select actions in a way that it visits every possible state-action pair infinitely often. This requirement can be a problem in the environment that the agent will operate.
# 
# As the most inputs suggested in the last subsection was defined in an infinite space, in section 3 I will discretize those numbers before use them to train my agent, keeping the state space representation manageable, hopefully. We also will see how \cite{Mitchell} defined a reliable way to estimate training values for $Q$, given only a sequence of immediate rewards $r$.
# 

# ### 2.3. Benchmark
# ```
# Udacity:
# 
# In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
# - Has some result or value been provided that acts as a benchmark for measuring performance?
# - Is it clear how this result or value was obtained (whether by data or by hypothesis)?
# ```
# 
# In 1988, the Wall Street Journal created a [Dartboard Contest](http://www.automaticfinances.com/monkey-stock-picking/), where Journal staffers threw darts at a stock table to select their assets, while investment experts picked their own stocks. After six months, they compared the results of the two methods. After adjusting the results to risk level, they found out that the pros barely have beaten the random pickers.
# 
# Given that, the benchmark used to measure the performance of the learner will be the amount of money made, in Reais, by a random agent. So, my goal will be to outperform this agent, that should just produce some random action from a set of allowed actions taken from $A$ at each time step $t$.
# 
# Just like my learner, the set of action can change over time depending on the open position, that is limited to $100$ stocks at most, on any side. When it reaches its limit, it will be allowed just to perform actions that decrease its position. So, for instance, if it already [long](https://goo.gl/GgXJgR) in $100$ shares, the possible moves would be $\left (None,\, sell,\, best\_ask \right)$. If it is [short](https://goo.gl/XFR7q3), it just can perform $\left (None,\, buy,\, best\_bid\right)$.
# 
# The performance will be measured primarily in the money made by the agents (that will be optimized by the learner). First, I will analyze if the learning agent was able to improve its performance on the same dataset after different trials. Later on, I will use the policy learned to simulate the learning agent behavior in a different dataset and then I will compare the final Profit and Loss of both agents. All data analyzed will be obtained by simulation.
# 
# As the last reference, in the final section, we will compare the total return of the learner to a strategy of buy-and-hold in BOVA11 and in the stock traded to check if we are consistently beating the market and not just being profitable, as the Udacity reviewer noticed.

# ## 3. Methodology
# 
# In this section, I will discretize the input space and implement an agent to learn the Q function.
# 
# ### 3.1 Data Preprocessing
# ```
# Udacity:
# 
# In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
# - If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?
# - Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?
# - If no preprocessing is needed, has it been made clear why?
# ```

# As mentioned before, I will implement a Markov decision processes (MDP) that requires, among other things, of a discrete set of states $S$. Apart from the input variables $position$, $OrderBid$, $OrderAsk$, the other variables are defined in an infinite domain. I am going to discretize those inputs, so my learning agent can use them in the representation of their intern state. In the Figure bellow, we can see the distribution of those variables. The data was produced using the first day of the dataset.
# 

import pandas as pd
df = pd.read_csv('data/ofi_petr.txt', sep='\t')
df.drop(['TIME', 'DELTA_MID'], axis=1, inplace=True)
df.dropna(inplace=True)


# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(df.ix[:, ['OFI', 'BOOK_RATIO']],
                  alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ```
# Udacity Reviewer:
# 
# Please be sure to specify how you are doing this (I'd recommend giving the formula).
# ```
# The scale of the variables is very different, and, in the case of the $BOOK\_RATIO$, it presents a logarithmic distribution. I will apply a logarithm transformation on this variable and rescale both to lie between a given minimum and maximum value of each feature using the function [MinMaxScaler](http://scikit-learn.org/stable/modules/preprocessing.html) from scikit-learn. So, both variable will be scaled to lie between $0$ and $1$ by applying the formula $z_{i} =\frac{x_i - \min{X}}{\max{X} - \min{X}}$. Where $z$ is the transformed variable, $x_i$ is the variable to be transformed and $X$ is a vector with all $x$ that will be transformed. The result of the transformation can be seen in the figure below.
# 

import sklearn.preprocessing as preprocessing
import numpy as np


scaler_ofi = preprocessing.MinMaxScaler().fit(pd.DataFrame(df.OFI))
scaler_bookratio = preprocessing.MinMaxScaler().fit(pd.DataFrame(np.log(df.BOOK_RATIO)))
d_transformed = {}
d_transformed['OFI'] = scaler_ofi.transform(pd.DataFrame(df.OFI)).T[0]
d_transformed['BOOK_RATIO'] = scaler_bookratio.transform(pd.DataFrame(np.log(df.BOOK_RATIO))).T[0]


df_transformed = pd.DataFrame(d_transformed)
pd.scatter_matrix(df_transformed.ix[:, ['OFI', 'BOOK_RATIO']],
                    alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# As mentioned before, in an MDP environment the agent must visit every possible state-action pair infinitely often. If I just bucketize the variables and combine them, I will end up with a huge number of states to explore. So, to reduce the state space, I am going to group those variables using K-Means and Gaussian Mixture Model (GMM) clustering algorithm. Then I will quantify the "goodness" of the clustering results by calculating each data point's [silhouette coefficient](https://goo.gl/FUVD50). The silhouette coefficient for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). In the figure below, I am going to calculate the mean silhouette coefficient to K-Means and GMM using a different number of clusters. Also, I will test different covariance structures to GMM.
# 

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import time


reduced_data = df_transformed.ix[:, ['OFI', 'BOOK_RATIO']]
reduced_data.columns = ['Dimension 1', 'Dimension 2']
range_n_clusters = [2, 3, 4, 5, 6, 8, 10]

f_st = time.time()
d_score = {}
d_model = {}
s_key = "Kmeans"
d_score[s_key] = {}
d_model[s_key] = {}
for n_clusters in range_n_clusters:
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    preds = clusterer.fit_predict(reduced_data)
    d_model[s_key][n_clusters] = clusterer
    d_score[s_key][n_clusters] = metrics.silhouette_score(reduced_data, preds)
print "K-Means took {:0.2f} seconds to run over all complexity space".format(time.time() - f_st)

f_avg = 0

for covar_type in ['spherical', 'diag', 'tied', 'full']:
    f_st = time.time()
    s_key = "GMM_{}".format(covar_type)
    d_score[s_key] = {}
    d_model[s_key] = {}
    for n_clusters in range_n_clusters:
        
        # TODO: Apply your clustering algorithm of choice to the reduced data 
        clusterer = GMM(n_components=n_clusters,
                        covariance_type=covar_type,
                        random_state=10)
        clusterer.fit(reduced_data)
        preds = clusterer.predict(reduced_data)
        d_model[s_key][n_clusters] = clusterer
        d_score[s_key][n_clusters] = metrics.silhouette_score(reduced_data, preds)
        f_avg += time.time() - f_st
        
print "GMM took {:0.2f} seconds on average to run over all complexity space".format(f_avg / 4.)


import pandas as pd
ax = pd.DataFrame(d_score).plot()
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score\n")
ax.set_title("Performance vs Complexity\n", fontsize = 16);


# The maximum score has happened using 2 clusters.However, I believe that the market can't be simplified that much. So, I will use the K-means with six centroids to group the variables. In the figure below we can see how the algorithm classified the data. Also, in the following table, the centroid was put in their original scales.
# 

# get centers
sample_preds = []
centers = d_model["Kmeans"][6].cluster_centers_
preds = d_model["Kmeans"][6].fit_predict(reduced_data)


# Display the results of the clustering from implementation
import qtrader.eda as eda; reload(eda);
eda.cluster_results(reduced_data, preds, centers)


# recovering data
log_centers = centers.copy()
df_aux = pd.DataFrame([np.exp(scaler_bookratio.inverse_transform(log_centers.T[0].reshape(1, -1))[0]),
                      scaler_ofi.inverse_transform(log_centers.T[1].reshape(1, -1))[0]]).T
df_aux.columns = df_transformed.columns
df_aux.index.name = 'CLUSTER'
df_aux.columns = ['BOOK RATIO', 'OFI']
df_aux.round(2)


# Curiously, the algorithm gave more emphasis on the $BOOK\_RATIO$  when its value was very large (the bid size almost eight times greater than the ask size) or tiny (when the bid size was one tenth of the ask size). The other cluster seems mostly dominated by the $OFI$. In the next subsection, I will discuss how I have implemented the Q-learning, how I intend to perform the simulations and make some tests. *Lastly, let's serialize the objects used in clusterization to be used later. *
# 

import pickle
pickle.dump(d_model["Kmeans"][6] ,open('data/kmeans_2.dat', 'w'))
pickle.dump(scaler_ofi, open('data/scale_ofi_2.dat', 'w'))
pickle.dump(scaler_bookratio, open('data/scale_bookratio_2.dat', 'w'))
print 'Done !'


# ### 3.2. Implementation
# 
# ```
# Udacity:
# 
# In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
# - Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?
# - Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?
# - Was there any part of the coding process (e.g., writing complicated functions) that should be documented?
# ```
# 
# As we have seen, learning the Q function corresponds to learning the optimal policy. According to \cite{Mohri_2012}, the optimal state-action value function $Q^{*}$ is defined for all $(s, \, a) \in S \times A$ as the expected return for taking the action $a \in A$ at the state $s \in S$, following the optimal policy. So, it can be written as \cite{Mitchell} suggested:
# 
# $$V^{*}(s) = \underset{a'}{\arg \max} \, Q(s, \, a')$$
# 
# Using this relationship, we can write a recursive definition of Q function, such that:
# 
# $$Q(s, \, a) = r(s, a) + \gamma \, \underset{a'}{\max} \, Q(\delta(s,\, a), \, a')$$
# 
# The recursive nature of the function above implies that our agent doesn't know the actual $Q$ function. It just can estimate $Q$, that we will refer as $\hat{Q}$. It will represents is hypothesis $\hat{Q}$ as a large table that attributes each pair $(s\, , \, a)$ to a value for $\hat{Q}(s,\, a)$ - the current hypothesis about the actual but unknown value $Q(s, \, a)$. I will initialize this table with zeros, but it could be filled with random numbers, according to \cite{Mitchell}. Still according to him, the agent repeatedly should observe its current state $s$ and do the following:
# 
# 

# ###### Algorithm 1: Update Q-table
# - Observe the current state $s$ and the allowed actions $A^{*}$:
#     - Choose some action $a \, \in \, A^{*}$ and execute it
#     - Receive the immediate reward $r = r(s, a)$
#     - if there is no entry $(s, \, a)$
#         - initialize the table entry $\hat{Q}(s, \, a)$ to zero
#     - Observe the new state $s' = \delta(s, \,a)$. 
#     - Updates the table entry for $\hat{Q}(s, \, a)$ following:
#         - $\hat{Q}(s, \, a) \leftarrow r + \gamma \underset{a'}{\max} \hat{Q}(s', \, a')$
# - $s \leftarrow s'$
# 
# The main issue in the strategy presented in Algorithm 1 is that the agent could overcommit to actions that presented positive $\hat{Q}$ values early in the simulation, failing to explore other actions that could present even higher values. \cite{Mitchell} proposed to use a probabilistic approach to select actions, assigning higher probabilities to action with high $\hat{Q}$ values, but given to every action at least a nonzero probability. So, I will implement the following relation:
# 
# $$P(a_i\, | \,s ) = \frac{k ^{\hat{Q}(s, a_i)}}{\sum_j k^{\hat{Q}(s, a_j)}}$$
# 
# Where $P(a_i\, | \,s )$ is the probability of selecting the action $a_i$ given the state $s$. The constant $k$ is positive and determines how strongly the selection favors action with high $\hat{Q}$ values.
# 
# Ideally, to optimize the policy found, the agent should iterate over the same dataset repeatedly until it is not able to improve its PnL. Later on, the policy learned will be tested against the same dataset to check its consistency. Lastly, this policy will be tested on the subsequent day of the training session. So, before perform the out-of-sample test, we will use the following procedure:
#  
# ###### Algorithm 2: Train-Test Q-Learning Trader
# - **for** each trial in total iterations desired **do**:
#     - **for** each observation in the session **do**:
#         - Update the table $\hat{Q}$
#     - Save the table $\hat{Q}$ indexed by the trial ID
# - **for** each trial in total iterations made **do**:
#     - Load the table $\hat{Q}$ related to the current trial
#     - **for** each observation in the session **do**:
#         - Observe the current state $s$ and the allowed actions $A^{*}$:
#             - **if** $s \, \notin  \, \hat{Q}$: Close out open positions or do nothing
#             - **else**: Choose the optimal action $a \, \in \, A^{*}$ and execute it
# 

# Each training session will include data from the largest part of a trading session, starting at 10:30 and closing at 16:30. Also, the agent will be allowed to hold a position of just 100 shares at maximum (long or short). When the training session is over, all positions from the learner will be closed out so the agent always will start a new session without carrying positions.
# 
# The agent will be allowed to take action every $2$ seconds and, due to this delay, every time it decides to insert limit orders, it will place it 1 cent worst than the best price. So, if the best bid is $12.00$ and the best ask is $12.02$, if the agent chooses the action $BEST\_BOTH$, it should include a buy order at $11.99$ and a sell order at $12.03$. It will be allowed to cancel these orders after 2 seconds. However, if these orders are filled in the mean time, the environment will inform the agent so it can update its current position. Even though, it just will take new actions after passed those 2 seconds.
# 
# ```
# Udacity Reviewer:
# 
# Please be sure to note any complications that occurred during the coding process. Otherwise, this section is simply excellent
# ```
# One of the biggest complication of the approach proposed in this project was to find out a reasonable representation of the environment state that wasn't too big to visit each state-action pair sufficiently often but was still useful in the learning process. In the next subsection, I will try different configurations of $k$ and $\gamma$ to try to improve the performance of the learning agent over the same trial.
# 

# 
# ### 3.3. Refinement
# ```
# Udacity:
# 
# In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
# - Has an initial solution been found and clearly reported?
# - Is the process of improvement clearly documented, such as what techniques were used?
# - Are intermediate and final solutions clearly reported as the process is improved?
# ```

# As mentioned before, we should iterate over the same dataset and check the policy learned on the same observations until convergence. Given the time required to perform each train-test iteration, "until convergence" will be 10 repetitions. We are going to train the model on the dataset from 08/15/2016. After each iteration, we will check how the agent would perform using the policy it has just learned. The agent in the first training session will use $\gamma=0.7$ and $k=0.3$. In the figure below are the results of the first round of iterations:
# 

# analyze the logs from the in-sample tests
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Fri_Oct__7_002946_2016.log'  # 15 old
# s_fname = 'log/train_test/sim_Wed_Oct__5_110344_2016.log'  # 15
# s_fname = 'log/train_test/sim_Thu_Oct__6_165539_2016.log'  # 25
# s_fname = 'log/train_test/sim_Thu_Oct__6_175507_2016.log'  # 35
# s_fname = 'log/train_test/sim_Thu_Oct__6_183555_2016.log'  # 5
get_ipython().magic("time d_rtn_train_1 = eda.simple_counts(s_fname, 'LearningAgent_k')")


import qtrader.eda as eda; reload(eda);
eda.plot_train_test_sim(d_rtn_train_1)


# The curve Train in the charts is the PnL obtained during the training session when the agent was allowed to explore new actions randomly. The test is the PnL obtained using strictly the policy learned. 
# 
# Although the agent was able to profit at the end of every single round,  "Convergence" is something that I can not claim. For instance, the PnL was worst in the first round than in the first one. I believe this stability of the results is difficult to obtain in day-trading. For example, even if the agent think that it should buy before the market goes up, it doesn't depending on its will if its order is filled.
# 
# We will target on improving the final PnL of the agent. However, less variability of the results is desired, especially at the beginning of the day, when the strategy didn't make any money yet. So, we also will look at the [sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio) of the first difference of the cumulated PnL produced by each configuration.
# 
# First, we are going to iterate through some values for $k$ and look at its performance in the training phase at the first hours of the training session. We also will use just 5 iterations here to speed up the tests.
# 

# improving K
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Thu_Oct__6_133518_2016.log'
get_ipython().magic("time d_rtn_k = eda.count_by_k_gamma(s_fname, 'LearningAgent_k', 'k')")


import pandas as pd
import matplotlib.pyplot as plt

f, na_ax = plt.subplots(1, 4, sharex=True, sharey=True)
for ax1, s_key in zip(na_ax.ravel(), ['0.3', '0.8', '1.3', '2.0']):
    df_aux = pd.Series(d_rtn_k[s_key][5])
    df_filter = pd.Series([x.hour for x in df_aux.index])
    df_aux = df_aux[((df_filter < 15)).values]
    df_aux.reset_index(drop=True, inplace=True)
    df_aux.plot(legend=False, ax=ax1)
    df_first_diff = df_aux - df_aux.shift()
    df_first_diff = df_first_diff[df_first_diff != 0]
    f_sharpe = df_first_diff.mean()/df_first_diff.std()
    ax1.set_title('$k = {}$ | $sharpe = {:0.2f}$'.format(s_key, f_sharpe), fontsize=10)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL Changing K\n'
f.suptitle(s_title, fontsize=16, y=1.03);


# When the agent was set to use $k=0.8$ and $k=2.0$, it achieved very similar results and Sharpe ratios. As the variable $k$ control the likelihood of the agent try new actions based on the Q value already observed, I will prefer the smallest value because it improves the chance of the agent to explore. Now, let's perform the same analysis varying only the $\gamma$:
# 

# improving Gamma
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Thu_Oct__6_154516_2016.log'
get_ipython().magic("time d_rtn_gammas = eda.count_by_k_gamma(s_fname, 'LearningAgent_k', 'gamma')")


import pandas as pd
import matplotlib.pyplot as plt

f, na_ax = plt.subplots(1, 4, sharex=True, sharey=True)
for ax1, s_key in zip(na_ax.ravel(), ['0.3', '0.5', '0.7', '0.9']):
    df_aux = pd.Series(d_rtn_gammas[s_key][5])
    df_filter = pd.Series([x.hour for x in df_aux.index])
    df_aux = df_aux[((df_filter < 15)).values]
    df_aux.reset_index(drop=True, inplace=True)
    df_aux.plot(legend=False, ax=ax1)
    df_first_diff = df_aux - df_aux.shift()
    f_sharpe = df_first_diff.mean()/df_first_diff.std()
    ax1.set_title('$\gamma = {}$ | $sharpe = {:0.2f}$'.format(s_key, f_sharpe), fontsize=10)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time Step', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL Changing Gamma\n'
f.suptitle(s_title, fontsize=16, y=1.03);


# As explained before, as $\gamma$ approaches one, future rewards are given greater emphasis about the immediate reward. When it is zero, only immediate rewards is considered. Despite the fact that the best parameter was $\gamma = 0.9$, I am not comfortable in giving so little attention to immediate rewards. It sounds dangerous when we talk about stock markets. So, I will choose to use $\gamma = 0.5$ arbitrarily in the next tests. *In the figure below, the agent is trained using $\gamma=0.5$ and $k=0.8$. [the next chart is not used in the final version]*
# 

# analyze the logs from the in-sample tests
import qtrader.eda as eda;reload(eda);
# s_fname = 'log/train_test/sim_Fri_Oct__7_002946_2016.log'  # 15 old
s_fname = 'log/train_test/sim_Wed_Oct__5_110344_2016.log'  # 15
# s_fname = 'log/train_test/sim_Thu_Oct__6_165539_2016.log'  # 25
# s_fname = 'log/train_test/sim_Thu_Oct__6_175507_2016.log'  # 35
# s_fname = 'log/train_test/sim_Thu_Oct__6_183555_2016.log'  # 5
get_ipython().magic("time d_rtn_train_2 = eda.simple_counts(s_fname, 'LearningAgent_k')")


import qtrader.eda as eda; reload(eda);
eda.plot_train_test_sim(d_rtn)


# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Fri_Oct__7_003943_2016.log'  # idx = 15 old
get_ipython().magic("time d_rtn_test_1 = eda.simple_counts(s_fname, 'LearningAgent_k')")


# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Wed_Oct__5_111812_2016.log'  # idx = 15
get_ipython().magic("time d_rtn_test_2 = eda.simple_counts(s_fname, 'LearningAgent_k')")


# compare the old with the data using the new configuration
import pandas as pd
df_plot = pd.DataFrame(d_rtn_test_1['pnl']['test']).mean(axis=1).fillna(method='ffill')
ax1 = df_plot.plot(legend=True, label='old')
df_plot = pd.DataFrame(d_rtn_test_2['pnl']['test']).mean(axis=1).fillna(method='ffill')
df_plot.plot(legend=True, label='new', ax=ax1)
ax1.set_title('Cumulative PnL Produced by New\nand Old Configurations')
ax1.set_xlabel('Time')
ax1.set_ylabel('PnL');


# In the figure above, an agent was trained using $\gamma=0.5$ and $k=0.8$ and its performance in out-of-sample test is compared to the previous implementation. In this case, the dataset from 07/16/2016 was used. the current configuration improved the performance of the model. We will discuss the final results in the next section.
# 

# ## 4. Results
# 
# In this section, I will evaluate the final model, test its robustness and compare its performance to the benchmark established earlier.
# 
# ### 4.1. Model Evaluation and Validation
# ```
# Udacity:
# 
# In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
# - Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?
# - Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?
# - Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?
# - Can results found from the model be trusted?
# ```

# One of the last questions that remain is if the model can make money in different scenarios. To test the robustness of the final model, I am going to use the same framework in very spaced days.
# 
# As each round of training and testing sessions takes 20-30 minutes to complete, I will check its performance just on three different days. I have already used the file of index 15 in the last tests. Now, I am going to use the files with index 5, 25 and 35 to train new models, and use the files with index 6, 26 and 36 to perform out-of-sample tests. In the Figure below we can see how the model performed in different unseen datasets.
# 

# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
l_fname = ['log/train_test/sim_Thu_Oct__6_171842_2016.log',  # idx = 25
           'log/train_test/sim_Thu_Oct__6_181611_2016.log',  # idx = 35
           'log/train_test/sim_Thu_Oct__6_184852_2016.log']  # idx = 5
def foo(l_fname):
    d_learning_k = {}
    for idx, s_fname in zip([25, 35, 5], l_fname):
        d_learning_k[idx] = eda.simple_counts(s_fname, 'LearningAgent_k')
    return d_learning_k

get_ipython().magic('time d_learning_k = foo(l_fname)')


import pandas as pd
import matplotlib.pyplot as plt

f, na_ax = plt.subplots(1, 3, sharey=True)
for ax1, idx in zip(na_ax.ravel(), [5, 25, 35]):
    df_plot = pd.DataFrame(d_learning_k[idx]['pnl']['test']).mean(axis=1)
    df_plot.fillna(method='ffill').plot(legend=False, ax=ax1)
    ax1.set_title('idx: {}'.format(idx + 1), fontsize=10)
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL in Diferent Days\n'
f.suptitle(s_title, fontsize=16, y=1.03);


# The model was able to make money in two different days after being trained in the previous session to each day. The performance of the third day was pretty bad. However, even wasting a lot of money at the beginning of the day, the agent was able to recover the most of its loss at the end of the session.
# 
# Looking at just to this data, the performance of the model looks very unstable and a little disapointing. In the next subsection, we will see why it is not that bad.
# 

# ### 4.2. Justification
# ```
# Udacity:
# 
# In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
# - Are the final results found stronger than the benchmark result reported earlier?
# - Have you thoroughly analyzed and discussed the final solution?
# - Is the final solution significant enough to have solved the problem?
# ```
# 

# Lastly, I am going to compare the final model with the performance of a random agent. We are going to compare the performance of those agents in an out-of-sample test.
# 
# As the learning agent follows strictly the policy learned, I will simulate the operations of this agent on the datasets tested just once. Even though I had run more trials, the return would be the same. However, I will simulate the operations of the random agent $20$ times at each dataset. As this agent can take any action at each run, the performance can be very good or very bad. So, I will compare the performance of the learning agent to the average performance of the random agent.
# 
# In the figure below we can see how much money each one has made in the first dataset used in this project, from 08/16/2016. The learning agent was trained using data from 08/15/2016, the previous day.
# 

# analyze the logs from the out-of-sample random agent
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Wed_Oct__5_111907_2016.log'  # idx = 15
get_ipython().magic("time d_rtn_test_1r = eda.simple_counts(s_fname, 'BasicAgent')")


import pandas as pd
import scipy
ax1 = pd.DataFrame(d_rtn_test_2['pnl']['test']).mean(axis=1).fillna(method='ffill').plot(legend=True, label='LearningAgent_k')
pd.DataFrame(d_rtn_test_1r['pnl']['test']).mean(axis=1).fillna(method='ffill').plot(legend=True, label='RandomAgent', ax=ax1)
ax1.set_title('Cumulative PnL Comparision\n')
ax1.set_xlabel('Time')
ax1.set_ylabel('PnL');
#performs t-test
a = [float(pd.DataFrame(d_rtn_test_2['pnl']['test']).iloc[-1].values)] * 2
b = list(pd.DataFrame(d_rtn_test_1r['pnl']['test']).fillna(method='ffill').iloc[-1].values)
tval, p_value = scipy.stats.ttest_ind(a, b, equal_var=False)


# A Welch's unequal variances t-test was conducted to compare if the PnL of the learner was greater than the PnL of a random agent. There was a significant difference between the performances (t-value $\approx 7.93$;  p-value $< 0.000$). These results suggest that learning agent really outperformed the random agent, the chosen benchmark. Finally, I am going to perform the same test using the datasets used in the previous subsection.
# 

print "t-value = {:0.6f}, p-value = {:0.8f}".format(tval, p_value)


# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
l_fname = ['log/train_test/sim_Thu_Oct__6_172024_2016.log',  # idx = 25
           'log/train_test/sim_Thu_Oct__6_181735_2016.log',  # idx = 35
           'log/train_test/sim_Thu_Oct__6_184957_2016.log']  # idx = 5
def foo(l_fname):
    d_basic = {}
    for idx, s_fname in zip([25, 35, 5], l_fname):
        d_basic[idx] = eda.simple_counts(s_fname, 'BasicAgent')
    return d_basic

get_ipython().magic('time d_basic = foo(l_fname)')


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

f, na_ax = plt.subplots(1, 3, sharey=True)
l_stattest = []
for ax1, idx in zip(na_ax.ravel(), [5, 25, 35]):
    # plot results
    df_learning_agent = pd.DataFrame(d_learning_k[idx]['pnl']['test']).mean(axis=1)
    df_learning_agent.fillna(method='ffill').plot(legend=True, label='LearningAgent_k', ax=ax1)
    df_random_agent = pd.DataFrame(d_basic[idx]['pnl']['test']).mean(axis=1)
    df_random_agent.fillna(method='ffill').plot(legend=True, label='RandomAgent', ax=ax1)
    #performs t-test
    a = [float(pd.DataFrame(d_learning_k[idx]['pnl']['test']).iloc[-1].values)] * 2
    b = list(pd.DataFrame(d_basic[idx]['pnl']['test']).iloc[-1].values)
    tval, p_value = scipy.stats.ttest_ind(a, b, equal_var=False)
    l_stattest.append({'key': idx+1,'tval': tval, 'p_value': p_value/2})
    # set axis
    ax1.set_title('idx: ${}$ | p-value : ${:.3f}$'.format(idx+1, p_value/2.), fontsize=10)
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL Comparision in Diferent Days\n'
f.suptitle(s_title, fontsize=16, y=1.03);


pd.DataFrame(l_stattest)


# In the dataset with index 26 and 36, the random agent outperformed the learning agent most the time, but at the end of these days, the learning agent was able to catch up the random agent performance.
# On this days, the t-test also rejected that the PnL of the learner was greater than the PnL from the random agent. In the dataset with index 6, the learning agent outperformed the random agent by a large margin, also confirmed by the t-test (t-value $\approx 5.97$;  p-value $< 0.000$). Curiously, in the worst day of the test, the random agent also performed poorly, suggesting that it wasn't a problem of my agent, but something that has happened on the market.
# 
# I believe these results are encoraging because they suggested that using the same learning framework on different days we can successfully find practical solutions that adapt well to new circumstances.
# 

# ## 5. Conclusion
# 
# In this section, I will discuss the final result of the model, summarize the entire problem solution and suggest some improvements that could be made.
# 
# ### 5.1. Final Remarks
# 
# ```
# Udacity:
# 
# Free-Form Visualization:
# 
# In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
# - Have you visualized a relevant or important quality about the problem, dataset, input data, or results?
# - Is the visualization thoroughly analyzed and discussed?
# - If a plot is provided, are the axes, title, and datum clearly defined?
# ```
# 
# In this project, We have proposed the use of the reinforcement learning framework to build an agent that learns how to trade according to the market states and its own conditions. After that the agent's policy was optimized to the previous sessions to the days it would have traded (in the out-of-sample tests), the agent would have been able to generate the result exposed in the figure below.

# group all data generated previously
df_aux = pd.concat([pd.DataFrame(d_learning_k[5]['pnl']['test']),
                    pd.DataFrame(d_rtn_test_2['pnl']['test']),
                    pd.DataFrame(d_learning_k[25]['pnl']['test']),
                    pd.DataFrame(d_learning_k[35]['pnl']['test'])])
d_data = df_aux.to_dict()
df_plot = eda.make_df(d_data).reset_index(drop=True)[1]

df_aux = pd.concat([pd.DataFrame(d_basic[5]['pnl']['test']).mean(axis=1),
                    pd.DataFrame(d_rtn_test_1r['pnl']['test']).mean(axis=1),
                    pd.DataFrame(d_basic[25]['pnl']['test']).mean(axis=1),
                    pd.DataFrame(d_basic[35]['pnl']['test']).mean(axis=1)])
d_data = pd.DataFrame(df_aux).to_dict()
df_plot2 = eda.make_df(d_data).reset_index(drop=True)[0]
ax1 = df_plot.plot(legend=True, label='LearningAgent_k')
df_plot2.plot(legend=True, label='RandomAgent')
ax1.set_title('Cumulated PnL from Simulations\n', fontsize=16)
ax1.set_ylabel('PnL')
ax1.set_xlabel('Time Step');


# The chart above shows the accumulated return in four different days generated by the learning agent  and by the random agent. Although the learning agent has not made money all the time, it still beat the performance of the random agent on the period of the tests. It also would have beaten a buy-and-hold strategy in BOVA11 and PETR4. Both would have lost money in the period, R\$ $-14,00$ and R\$ $-8,00$, respectively.
# 

((df_last_pnl)*100).sum()


# ```
# Udacity:
# 
# Reflection:
# 
# In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
# - Have you thoroughly summarized the entire process you used for this project?
# - Were there any interesting aspects of the project?
# - Were there any difficult aspects of the project?
# - Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?
# ```

# To find the optimal policy we have used Q-Learning, a model-free approach to reinforcement learning. We trained the agent by simulating several runs on the same dataset, allowing the agent to explore the results of different actions on the same environment. So, we have back-tested the policy learned on the same dataset and found out that the policy learned not always converge to a better one. We noticed that this non-convergence could be related to the nature of our problem.
# 
# So, we refined the model testing different configurations of the model parameters and compare the PnL of the new policy to the old one backtesting them against a different dataset. Finally, after we selected the best parameters, we trained the model in different days and tested against the subsequent sessions. 
# 
# We compared these results to the returns of a random agent and concluded that our model was significantly better during the period of the tests.
# 
# One of the most interesting parts of this project was to define the state representation of the environment. I find out that when we increase the state space too much, it becomes very hard the agent learns an acceptable policy in the number of the trials we have used. The number of trials used was mostly determined by the time it took to run (several minutes)
# 
# It was interesting to see that, even clustering the variables using k-means, the agent was still capable of using the resulting clusters to learn something useful from the environment. 
# 
# Building the environment was the most difficult and challenging part of the entire project. Not just find an adequate structure for build the order book wasn't trivial, but make the environment operates it correctly was difficult. It has to manage different orders from various agents and ensure that each agent can place, cancel or fill orders (or have orders been filled) in the right sequence.
# 
# Overall, I believe that the simulation results have shown initial success in bringing reinforcement learning techniques to build algorithmic trading strategies. Develop a strategy that doesn't perform any [arbitrage](https://en.wikipedia.org/wiki/Arbitrage) and still never lose money is something very unlikely to happen. This agent was able to mimic the performance of an average random agent sometimes and outperforms it other times. In the long run, It would be good enough.
# 

# ### 5.2. Improvement
# ```
# Udacity:
# 
# In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
# - Are there further improvements that could be made on the algorithms or techniques you used in this project?
# - Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?
# - If you used your final solution as the new benchmark, do you think an even better solution exists?
# ```
# 

# Many areas could be explored to improve the current model and refine the test results. I wasn't able to achieve a stable solution using Q-Learning, and I believe that it is most due to the non-deterministic nature of the problem. So, we could test [Recurrent Reinforcement Learning](https://goo.gl/4U4ntD), for instance, which \cite{du1algorithm} argued that it could outperform Q-learning in the sense of stability and computational convenience.
# 
# Also, I believe that different state representations should be tested much deeper. The state observed by the agent is one of the most relevant aspects of reinforcement learning problems and probably there are better representations that the one used in this project to the given task.
# 
# Another future extension to that project also could include a more realistic environment, where other agents respond to the actions of the learning agent, and lastly, we could test other reward functions to the problem posed. Would be interesting to include some future information in the response of the environment to the actions of the agent, for example, to see how it would affect the policies learned.
# 

# *Style notebook and change matplotlib defaults*
# 

#loading style sheet
from IPython.core.display import HTML
HTML( open('ipython_style.css').read())


#changing matplotlib defaults
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2", 10))





# ## Building The Environment
# <sub>Uirá Caiado. Aug 19, 2016<sub>
# 
# ### My Goal, For Now
# 

# Here I will build the environment where my agent will be trained. As it is a pretty complicated task, I created this notebook just to explore some options. Test an automated strategy is something pretty difficult, to tell the truth. There are some issues that I could talk about: historical prices are just one path from a stochastic process, the presence of the strategy on the market could change it etc.
# 
# However, would be interesting if the solution explored in my capstone could capture some aspect of the real structure of the markets. So, I will build an environment that will use historical data, but I will build it in a "agent-based model" fashion. Probably I will not do that to this project, but this structure can be useful for some future projects that I intend to do after completing this one.
# 
# I will use TAQ data from Bloomberg; It consists of 'level I' data from the PETR4 Brazilian Stock. I will explain about all these terms and data in my "official" project; here I just intend to share my experiments in a less formal way. Level I data is the information of the first queue of order book (bid and ask), and trades that have been made (TAQ is an acronym for Trade and Quote). It alone is not enough to create an order book, but I will try to do that. If I complete this task, I could exclude the historical data and include some random agents to build a Monte Carlo of the order book, for example. I will try to build it upon the codes for the SmartCab project. Instead of the gridlike world, my world will be discrete, but an order book.
# 

# ### Exploring The Data
# 
# So, let's start looking at the data that I have available. It is data from the last 19 days from PETR4 stock from Brazilian BMFBovespa. I choose this share because it is on of the most active stocks in our market, but still the data produced is manageable (60 to 90 thousands of rows by day).
# 

import zipfile
s_fname = "data/petr4_0725_0818.zip"
archive = zipfile.ZipFile(s_fname, 'r')
def foo():
    f_total = 0.
    f_tot_rows = 0.
    for i, x in enumerate(archive.infolist()):
        f_total += x.file_size/ 1024.**2
        for num_rows, row in enumerate(archive.open(x)):
            f_tot_rows += 1
        print "{}:\t{:,.0f} rows\t{:0.2f} MB".format(x.filename, num_rows + 1, x.file_size/ 1024.**2)
    print '=' * 42
    print "TOTAL\t\t{} files\t{:0.2f} MB".format(i+1,f_total)
    print "\t\t{:0,.0f} rows".format(f_tot_rows)

get_ipython().magic('time foo()')


# Ok, all files together have 91 MB. It is not too much actually. Maybe I could try to get more data later. Well, let's read one of them.
# 

import pandas as pd
df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
df.head()


print "{:%m/%d/%Y}".format(df.Date[0])
print df.groupby('Type').count()['Date']


# As it is a "Trade And quotes" file, I was already expecting that there was the same number os Bid and Ask rows in the file. Well... it wasn't a problem to read this file. Well, I will read this file row by row now and include the prices in a binomial tree (the structure I intend to use to keep the order book). Let's see how long it takes.
# 

from bintrees import FastRBTree


def foo():
    for idx, row in df.iterrows():
        pass

print "time to iterate the rows:"
get_ipython().magic('time foo()')


def foo():
    bid_tree = FastRBTree()
    ask_tree = FastRBTree()
    for idx, row in df.iterrows():
        if row.Type == 'BID':
            bid_tree.insert(row['Price'], row['Size'])
        elif row.Type == 'ASK':
            ask_tree.insert(row['Price'], row['Size'])

print "time to insert everything in binary trees:"
get_ipython().magic('time foo()')


# It is not excelent, but it is ok for now. I will see how bad it will be when I include all the logic needed. Now, let's visualize the prices on that day
# 

df_aux = df[df.Type == 'TRADE'].Price
df_aux.index = df[df.Type == 'TRADE'].Date
ax = df_aux.plot()
ax.set_title("Price fluctuation of PETR4\n");


# Know what, let's visualize the prices of all days. I will plot the cumulated returns of this series
# 

import pandas as pd

df_all = None

for i, x in enumerate(archive.infolist()):
    df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
    ts_date = df.Date[0].date()
    df.Date = ["{:%H:%M:%S}".format(x) for x in df.Date]
    df = df[df.Type == "TRADE"]
    df = pd.DataFrame(df.groupby('Date').last()['Price'])
    if i == 0:
        df_all = df.copy()
        df_all.columns = [ts_date] 
    else:
        df_aux = df.copy()
        df_aux.columns = [ts_date]
        df_all = df_all.join(df_aux)
df_all.index = pd.to_datetime(df_all.index)
df_all = df_all.fillna(method='ffill')
df_all = df_all.dropna()


import numpy as np
df_logrtn = np.log(df_all/df_all.shift())
df_logrtn = df_logrtn[[(x.hour*60 + x.minute) < (16*60 + 55) for x in df_logrtn.index]]
ax = df_logrtn.cumsum().plot(legend=False)
ax.set_title('Cumulative Log-Returns of PETR4 in 19 different days\n', fontsize=16)
ax.set_ylabel('Return');


# Interesting, isn't it?! It looks like the output of a Monte Carlo simulation. Well, the simulator that I will build should produce exactly this output.
# 

# ### Pre-processing The Data
# 
# There is a problem using this data to do a simulation: the order of events. As there is just the first line of the order book, when happened a trade that filled more than an price level at once, I won't have the order in the second price level in my structure to be filled. So, I need to create it beforehand. To do so, I need to preprocess the dataset, so I can include this event. What I will do is sum up the trades that happend in sequence and include a new rows between the sequences to preper the dataset.
# 

import qtrader.preprocess as preprocess
s_fname = "data/petr4_0725_0818.zip"
preprocess.make_zip_file(s_fname)


# Now, let's check if I keep the same traded quantity in all files
# 

import zipfile
import pandas as pd
s_fname = "data/petr4_0725_0818.zip"
archive = zipfile.ZipFile(s_fname, 'r')


f_total = 0.
i_same = 0
i_different = 0
for i, x in enumerate(archive.infolist()):
    s_fname = 'data/petr4_0725_0818_2/' + x.filename
    df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
    df2 = pd.read_csv(s_fname, index_col=0, parse_dates=['Date'])
    f_all = (df.ix[df.Type=='TRADE', 'Price'] * df.ix[df.Type=='TRADE', 'Size']).sum()
    f_all2 = (df2.ix[df2.Type=='TRADE', 'Price'] * df2.ix[df2.Type=='TRADE', 'Size']).sum()
    if f_all == f_all2:
        i_same += 1
    else:
        i_different += 1

print "{} files has the same number of trades".format(i_same)
print "{} files has DIFFERENT number of trades".format(i_different)
    


# Nice. I am going to use this files in the next steps.
# 

# ### Building an Order book
# 
# I imagine a structure where my environment is an order book that operates in discrete time steps. At each time step, it allows randomly the agents (at each step) to take new actions. Then, the Environment updates the order book according to these messages.
# 
# At the beginning of each time step, all agent see the same state of the order book. Some of them will try to execute orders at the same price, for example, and the environment will accept just this kind of message while there are orders to be filled at that particular price. After that, It should distribute all the new states to the agents.
# 
# It also should keep the prices organized and, inside each price, should follow the orders arranged by "arrival time". Always that an agent grow his quantity of an existing order, it should be moved to the end of the queue on that price. The message that the agent should send to the environment should be something like that
# 
# ```
# {'instrumento_symbol': 'PETR4',
#  'agent_id': 10,
#  'order_entry_step': 15,
#  'order_status': 'New'
#  'last_order_id': 11,
#  'order_id': 11,
#  'order_side': 'BID',
#  'agressor_indicator': 'Neutral',
#  'order_price': 12.11,
#  'total_qty_order': 100,
#  'traded_qty_order': 0}
#  ```
#  
# The limit order book will be ordered first by price and then by arrival. So, my first class should be the order itself. The second class should be the price level (that is a group of orders). The third one, the side of the book (a collection of Price Levels). Finally, the order book (the bid and ask side). So, now I need to create all the structure that handle these interations. If should receive an order and answer to the environment if it was acepted , the new ID, and, if there was a trade, the informations about the trade so the environment can update the state of the agents. I need a list of prices and another of orders. Oh... and a list of agents that have a list of their current orders on the market, but it is something handle by the environment. So, let's do that. First, let's implement the order structure.
# 

# example of message
d_msg = {'instrumento_symbol': 'PETR4',
         'agent_id': 10,
         'order_entry_step': 15,
         'order_status': 'New',
         'last_order_id': 0,
         'order_id': 0,
         'order_side': 'BID',
         'order_price': 12.11,
         'total_qty_order': 100,
         'traded_qty_order': 0}


class Order(object):
    '''
    A representation of a single Order
    '''
    def __init__(self, d_msg):
        '''
        Instantiate a Order object. Save all parameter as attributes
        :param d_msg: dictionary.
        '''
        # keep data extract from file
        self.d_msg = d_msg.copy()
        self.d_msg['org_total_qty_order'] = self.d_msg['total_qty_order']
        f_q1 = self.d_msg['total_qty_order']
        f_q2 = self.d_msg['traded_qty_order']
        self.d_msg['total_qty_order'] = f_q1 - f_q2
        self.order_id = d_msg['order_id']
        self.last_order_id = d_msg['last_order_id']
        self.name = "{:07d}".format(d_msg['order_id'])
        self.main_id = self.order_id

    def __str__(self):
        '''
        Return the name of the Order
        '''
        return self.name

    def __repr__(self):
        '''
        Return the name of the Order
        '''
        return self.name

    def __eq__(self, other):
        '''
        Return if a Order has equal order_id from the other
        :param other: Order object. Order to be compared
        '''
        return self.order_id == other.order_id

    def __ne__(self, other):
        '''
        Return if a Order has different order_id from the other
        :param other: Order object. Order to be compared
        '''
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Allow the Order object be used as a key in a hash table. It is used by
        dictionaries
        '''
        return self.order_id.__hash__()

    def __getitem__(self, s_key):
        '''
        Allow direct access to the inner dictionary of the object
        :param i_index: integer. index of the l_legs attribute list
        '''
        return self.d_msg[s_key]


my_order = Order(d_msg)
print "My id is {} and the price is {:0.2f}".format(my_order['order_id'], my_order['order_price'])
print "The string representation of the order is {}".format(my_order)


# If the ID is zero, I will consider as the new order. Well... now I need to organize orders at the same price by arrival time and update its IDs. Also, if an existing order increases the quantity, should have been marked with a new ID. If the quantity decreases, it should keep the same place on the queue. So, the price level need to know what is the last general ID used by the Order Book:
# 

from bintrees import FastRBTree

class PriceLevel(object):
    '''
    A representation of a Price level in the book
    '''
    def __init__(self, f_price):
        '''
        A representation of a PriceLevel object
        '''
        self.f_price = f_price
        self.i_qty = 0
        self.order_tree = FastRBTree()

    def add(self, order_aux):
        '''
        Insert the information in the tree using the info in order_aux. Return
        is should delete the Price level or not
        :param order_aux: Order Object. The Order message to be updated
        '''
        # check if the order_aux price is the same of the self
        if order_aux['order_price'] != self.f_price:
            raise DifferentPriceException
        elif order_aux['order_status'] == 'limit':
            self.order_tree.insert(order_aux.main_id, order_aux)
            self.i_qty += int(order_aux['total_qty_order'])
        # check if there is no object in the updated tree (should be deleted)
        return self.order_tree.count == 0

    def delete(self, i_last_id, i_old_qty):
        '''
        Delete the information in the tree using the info in order_aux. Return
        is should delete the Price level or not
        :param i_last_id: Integer. The previous secondary order id
        :param i_old_qty: Integer. The previous order qty
        '''
        # check if the order_aux price is the same of the self
        try:
            self.order_tree.remove(i_last_id)
            self.i_qty -= i_old_qty
        except KeyError:
            raise DifferentPriceException
        # check if there is no object in the updated tree (should be deleted)
        return self.order_tree.count == 0

    def __str__(self):
        '''
        Return the name of the PriceLevel
        '''
        return '{:,.0f}'.format(self.i_qty)

    def __repr__(self):
        '''
        Return the name of the PriceLevel
        '''
        return '{:,.0f}'.format(self.i_qty)

    def __eq__(self, other):
        '''
        Return if a PriceLevel has equal price from the other
        :param other: PriceLevel object. PriceLevel to be compared
        '''
        # just to make sure that there is no floating point discrepance
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return abs(self.f_price - f_aux) < 1e-4

    def __gt__(self, other):
        '''
        Return if a PriceLevel has a gerater price from the other.
        Bintrees uses that to compare nodes
        :param other: PriceLevel object. PriceLevel to be compared
        '''
        # just to make sure that there is no floating point discrepance
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return (f_aux - self.f_price) > 1e-4

    def __lt__(self, other):
        '''
        Return if a Order has smaller order_id from the other. Bintrees uses
        that to compare nodes
        :param other: Order object. Order to be compared
        '''
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return (f_aux - self.f_price) < -1e-4

    def __ne__(self, other):
        '''
        Return if a Order has different order_id from the other
        :param other: Order object. Order to be compared
        '''
        return not self.__eq__(other)


my_order = Order(d_msg)


# create different orders at the same price
d_msg1 = d_msg.copy()
d_msg1['order_id'] = 1
order1 = Order(d_msg1)
d_msg2 = d_msg.copy()
d_msg2['order_id'] = 2
order2 = Order(d_msg2)
d_msg3 = d_msg.copy()
d_msg3['order_id'] = 3
order3 = Order(d_msg3)


my_price = PriceLevel(d_msg['order_price'])
my_price.add(order1)
my_price.add(order2)
my_price.add(order3)


print "There is {} shares at {:.2f}".format(my_price, my_price.f_price)
print 'the orders in the book are: {}'.format(dict(my_price.order_tree))


# Ok. Now, let's delete two of them
# 

my_price.delete(1, 100)
my_price.delete(2, 100)
print "There is {} shares at {:.2f}".format(my_price, my_price.f_price)
print 'the orders in the book are: {}'.format(dict(my_price.order_tree))


# I will let to the Environment handle the IDs, so the order book just need to keep the data ordered. Now, I am going to implement the Book side (a collection of Price levels) and the Limit Order Book (a collection of BookSide). You can see the final implementation on the file `book.py`.
# 

import qtrader.book as book; reload(book);


my_book = book.LimitOrderBook('PETR4')


d_msg0 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 0,
          'order_price': 12.12,
          'order_side': 'ASK',
          'order_status': 'New',
          'total_qty_order': 400,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}


d_msg1 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 1,
          'order_price': 12.11,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg2 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 2,
          'order_price': 12.11,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg3 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 3,
          'order_price': 12.10,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 200,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg4 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 4,
          'order_price': 12.10,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg5 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 3,
          'order_price': 12.10,
          'order_side': 'BID',
          'order_status': 'Replaced',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg6 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 1,
          'order_price': 12.11,
          'order_side': 'BID',
          'order_status': 'Filled',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Passive'}


# include several orders
my_book.update(d_msg0)
my_book.update(d_msg1)
my_book.update(d_msg2)
my_book.update(d_msg3)
my_book.update(d_msg4)
my_book.get_n_top_prices(5)


# test cancelation
my_book.update(d_msg5)
my_book.get_n_top_prices(5)


# checking if the order of Ids are OK
x = my_book.book_bid.price_tree.get(12.10)
x.order_tree


# test a trade
my_book.update(d_msg6)
my_book.get_n_top_prices(5)


my_book.get_basic_stats()


# Ok, everything looks right. Let's play with some real data now.
# 

# ### Create the Planner
# 
# I want to build a structure that I could simulate with any kind of data: real data, "semi-real" data and artificial data. Using artificial data, I could allow the agent change the environment. The other two option, I would just replay the market. The first one, I would need "level II" data, that is not easily acquired, and it is harder to handle. The second one, I would use just "level I" data, which some vendors provide, like Bloomberg. However, there are just the grouped data of the best- bid and offer.
# 
# For now, what I need is a framework to interact with the book and a planner (maybe). This planner won't do anything if we are using the artificial market, just return the actions of each agent. If it is historical data, It should take the data and translate to the Environment as actions of each agent. So, my environment could update the order book using just the messages dicionaries and update each agent if that message was acepted or not. I guess that the role of the Environment should be update each agent with informations that them could use to build their states and rewards. The planner should handle the "macro-behaviour" of each agent (like, know something about the true price, etc).
# 
# So, let's come back to the data that was explored at the begining of this notebook. the first thing that I should do is to do a basic reshape on the data to make it work with my book.
# 

import zipfile
s_fname = "data/petr4_0725_0818.zip"
archive = zipfile.ZipFile(s_fname, 'r')

f_total = 0.
for i, x in enumerate(archive.infolist()):
    f_total += x.file_size/ 1024.**2
    for num_rows, row in enumerate(archive.open(x)):
        pass


import pandas as pd
df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
df.head(5)


# Something that I need to take into account is that the book didn't start before the first BID or ASK, that both are related to the aggregated best bid and ask, and TRADE can be an aggression on either side. As I want to make a book with multiple queues, I will keep the orders on the book and just modify it when the price returns to that level. So, before decrease the BID or increase the ASK, I should check if I have an order for that price and modify it before cancel the better price. Also, I know that the TRADE flag can be a cross order, something that I don't want to account for. I also want to keep track of the changes on the best queue of the order book and the trades separated.
# 

def translate_row(idx, row, i_order_id):
    '''
    '''
    if row.Type != 'TRADE' and row['Size'] > 100:
        d_rtn = {'agent_id': 10,
                 'instrumento_symbol': 'PETR4',
                 'new_order_id': i_order_id + 1,
                 'order_entry_step': idx,
                 'order_id': i_order_id + 1,
                 'order_price': row['Price'],
                 'order_side': row.Type,
                 'order_status': 'New',
                 'total_qty_order': row['Size'],
                 'traded_qty_order': 0,
                 'agressor_indicator': 'Neutral'}
        return i_order_id + 1,  d_rtn


# test the structure
import qtrader.book as book; reload(book);
my_book = book.LimitOrderBook('PETR4')
for idx, row in df.iterrows():
    i_id = my_book.i_last_order_id
    t_rtn = translate_row(idx, row, i_id)
    if t_rtn:
        my_book.i_last_order_id = t_rtn[0]
        my_book.update(t_rtn[1])
    if idx == 1000:
        break


my_book.get_n_top_prices(5)


my_book.get_basic_stats()


# Well... I would say that I need to handle better this orders. I need to cancel them to the bid and don't cross each other. So... Also, note the quantity on each price. It is not feasable. I need that the function knows the book. Let's see.
# 

def translate_row(idx, row, my_book):
    '''
    '''
    l_msg = []
    if row.Type != 'TRADE' and row['Size'] % 100 == 0:
        # recover the best price
        f_best_price = my_book.get_best_price(row.Type)
        i_order_id = my_book.i_last_order_id + 1
        # check if there is orders in the row price
        obj_ordtree = my_book.get_orders_by_price(row.Type, row['Price'])
        if obj_ordtree:
            # cant present more than 2 orders (mine and market)
            assert len(obj_ordtree) <= 2, 'More than two offers'
            # get the first order
            obj_order = obj_ordtree.nsmallest(1)[0][1]
            # check if should cancel the best price
            b_cancel = False
            if row.Type == 'BID' and row['Price'] < f_best_price:
                # check if the price in the row in smaller
                obj_ordtree2 = my_book.get_orders_by_price(row.Type)
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())
            elif row.Type == 'ASK' and row['Price'] > f_best_price:
                obj_ordtree2 = my_book.get_orders_by_price(row.Type)
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())

            
            
            # replace the current order
            i_old_id = obj_order.main_id
            i_new_id = obj_order.main_id
            if row['Size'] > obj_order['total_qty_order']:
                i_new_id = my_book.i_last_order_id + 1
                d_rtn = obj_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())


            # Replace the order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': i_old_id,
                     'order_entry_step': idx,
                     'new_order_id': i_new_id,
                     'order_price': row['Price'],
                     'order_side': row.Type,
                     'order_status': 'Replaced',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn.copy())
        else:               
            # if the price is not still in the book, include a new order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': my_book.i_last_order_id + 1,
                     'order_entry_step': idx,
                     'new_order_id': my_book.i_last_order_id + 1,
                     'order_price': row['Price'],
                     'order_side': row.Type,
                     'order_status': 'New',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn)
        return l_msg


# test the structure
import qtrader.book as book; reload(book);
import pprint
import time
f_start = time.time()
my_book = book.LimitOrderBook('PETR4')
self = my_book
for idx, row in df.iterrows():
    l_msg = translate_row(idx, row, my_book)
    if l_msg:
        for msg in l_msg:
            my_book.update(msg)
#     if idx == 10000:
#         break
"It took {:0.2f} seconds to process {:0,.0f} rows".format(time.time() - f_start, idx + 1)


my_book.get_basic_stats()


my_book.get_n_top_prices(100)


# It is looking right, but I still need to treat Trades. Let's do it. As I know that some trades are cross orders (it is not passing through the book), I will pre-process the data to exclude them, so my simulator will just handle data that relevant.
# 

df.head()


def foo():
    best_bid = None
    best_ask = None
    i_num_cross = 0
    l_cross = []
    for idx, row in df.iterrows():
        if row.Size % 100 == 0:
            if row.Type == 'BID':
                best_bid = row.copy()
            elif row.Type == 'ASK':
                best_ask = row.copy()
            else:
                if not isinstance(best_bid, type(None)):
                    if row.Price == best_bid.Price:
                        if row.Size > best_bid.Size:
    #                         print 'cross-bid', idx
                            i_num_cross += 1
                            l_cross.append(idx)

                if not isinstance(best_ask, type(None)):
                    if row.Price == best_ask.Price:
                        if row.Size > best_ask.Size:
    #                         print 'cross-ask', idx
                            i_num_cross += 1
                            l_cross.append(idx)
    print "number of cross-orders: {:.0f}".format(i_num_cross)
    return l_cross

get_ipython().magic('time l_cross = foo()')


# I guess that I will not need to treat that. Just need to ignores trades that there are not enough in the book to filled them. Let´s first start by building an object to read the data files so I don't need to load all the data to a dataframe before translating the rows to the order book.
# 

import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.book as book; reload(book);
import time
s_fname = "data/petr4_0725_0818.zip"
my_test = matching_engine.BloombergMatching(None, "PETR4", 200, s_fname)
f_start = time.time()
for i in xrange(my_test.max_nfiles):
    for d_data in my_test:
        pass
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)


# Hm... I was reading all the files in 6 seconds before.... It looks bad, isn't it? Let's translate the messages and update the books using the last function implemented and process the trades.
# 

import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.book as book; reload(book);
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_test = matching_engine.BloombergMatching(None, "PETR4", 200, s_fname)
f_start = time.time()

for i in xrange(my_test.max_nfiles):
    quit = False
    while True:
        try:
            l_msg = my_test.next()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break
#     break
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)


# ... Yeh... There is some changes to do before conclude it. I need to make the translator handle multiple orders. It is handling multiple messages just in Trades, not when there are limit orders. I am not sure if I will use multiple agents by price, but I need to handle more than one agent anyway because the agent that I intend to create. But know what, before implement it, I guess that it is better to introduce the last peace of this study: the environment
# 

my_test.my_book.get_n_top_prices(5)


# ### Finally, the Environment
# 
# My environment should know what is the best bid and best price at each iteration, and keep a list of the trades that have happened. Each agent could have multiple orders at each time. However, I will allow the most of them to handle a single order at each time to simplify my problem. Each agent should receive some variables that it could use to compose its inner-state at each time step. Also, should be given the reward if it is the case. As I probably should test what kind of reward it receives, maybe the environment should pass just a dictionary of variables, and the agent decides what it will use as state and what it will use as the reward... and what it will not use at all.
# 
# As I stated before, my order matching still has a important limitation. It assumes that all price levels have just a single order, and it will not be the case, as my agent will interact with the environment. I could even include more orders at each price level... but I guess that I will keep it "simple" (it is not simple, already).
# 
# Something that I will have to take care is that, as I dealing with historical data (and I wnat to keep it in that way), I will need to make my translator to modify the order book to acomodate the order of my agent. My agent will always use the minimum quantity possible, so I need to discount it when my agent include an order in any price (and decrease it when the agent cancel that order). I also will have an agent that will have all order of the market (and that will perform the trades). So, let's start by just instantiating an environment and set the initial agents to a simulation
# 

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint


e = environment.Environment()
a = e.create_agent(environment.Agent)
e.set_primary_agent(a)
pprint.pprint(dict(e.agent_states))


# Ok, now let's try to simulate the order book, as before
# 

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_env = environment.Environment()
f_start = time.time()

for i in xrange(my_env.order_matching.max_nfiles):
    quit = False
    my_env.reset()
    while True:
        try:
            l_msg = my_env.step()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break

print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)


# More than 5 minutes to read all files. Well, if you think, there are more than 2 million rows in total. It implies that there are more than 2 million times that my agent will receive something from the environment. So, maybe it is not so bad. Maybe I should even restrict the number of steps inside a particular file, so I could use the remain data to perform a back test. By the way, it is not a problem that I intend to address here. For now, my goal is just building the world where I can include agents to interact. It is already pretty complicated so far. Let's keep moving. Now, I will make the agent to sense the world and receive a reward by their actions. One of the measued that I want to include is the order flow imbalance that I will explain better in the main notebook. For now, the form of the measure is:
# 
# $$e_n = \mathbb{1}_{P_{n}^{B} \geq P_{n-1}^{B}} q^{B}_{n} - \mathbb{1}_{P_{n}^{B} \leq P_{n-1}^{B}}  q^{B}_{n-1} + \mathbb{1}_{P_{n}^{A} \leq P_{n-1}^{A}} q^{A}_{n} + \mathbb{1}_{P_{n}^{A} \geq P_{n-1}^{A}}  q^{A}_{n-1}$$
# 
# Where $A$ in related to the ask side, $B$ to the Bid side, $n$ to the current observation and $n-1$ to the last one. $\mathbb{1}$ is an [Indicator](https://en.wikipedia.org/wiki/Indicator_function) function and $P$ is the price and $q$, the quantity. I will also hold the value of this variables in chunck of 10 seconds. So, let's see
# 

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_env = environment.Environment()
f_start = time.time()

for i in xrange(my_env.order_matching.max_nfiles):
    quit = False
    my_env.reset()
    while True:
        try:
#             if my_env.order_matching.idx == 6:
#                 if my_env.order_matching.i_nrow == 107410:
#                     raise NotImplementedError
            if my_env.order_matching.i_nrow > 9:
                break
            l_msg = my_env.step()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break
    break
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)


# Now, I need to calculate the reward. Both the inputs from environment and the reward is something that I will need to explore in the main notebook. For now, the reward will be just the PnL from the last step to this step. Porblably I will penalize the agent to keep orders in the order book later. Just to make it "pays the risk". For now, let's see:
# 

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_env = environment.Environment()
f_start = time.time()

for i in xrange(my_env.order_matching.max_nfiles):
    quit = False
    my_env.reset()
    while True:
        try:
#             if my_env.order_matching.idx == 6:
#                 if my_env.order_matching.i_nrow == 107410:
#                     raise NotImplementedError
            if my_env.order_matching.i_nrow > 9:
                break
            l_msg = my_env.step()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break
    break
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)


# It is pretty bad. It between 8 to 12 minutes to run, depending on the machine used. Well, I will try to optimize it later. Now, it is time to make the part that I was avoiding at all costs =). I need to allow a random agent interact with this world and receives whatever need to receive. I will need to improve my translator for doing that (at this point, my agent can not change too much what have happened). Also, I will not print out the updates in Zombie agent, but just in my learning agent, that will happen to be updated just once per second (from the simulation, not from real time). First, let's allow the environment to deal with multiple orders by price and trades when the prices cross each other.
# 

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import qtrader.agent as agent; reload(agent);
import qtrader.simulator as simulator; reload(simulator);


e = environment.Environment()
a = e.create_agent(agent.BasicAgent)
e.set_primary_agent(a)

sim = simulator.Simulator(e)
get_ipython().magic('time sim.run(n_trials=20)')


# Ok, now, let's implement my basic agent that will take actions randomly. I will let it insert limit order with 10 cents of spread between the bid price the the agent order I alow it to update its state just once a hour.
# 

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import qtrader.agent as agent; reload(agent);
import qtrader.simulator as simulator; reload(simulator);
import qtrader.translators as translators; reload(translators);


e = environment.Environment()
a = e.create_agent(agent.BasicAgent, f_min_time=3600.)
e.set_primary_agent(a)

sim = simulator.Simulator(e)
get_ipython().magic('time sim.run(n_trials=30)')


# WOOOOWWWWWWWWWWW ... it worked. Now.... after almost a month doing that (today is 16th september), I can finally ...... start the Udacity projet =/
# 




# *Style notebook and change matplotlib defaults*
# 

#loading style sheet
from IPython.core.display import HTML
HTML( open('ipython_style.css').read())


#changing matplotlib defaults
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2", 10))





