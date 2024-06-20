# Company XYZ started a subscription model in January, 2015. You get hired as a ﬁrst data scientist at the end of August and, as a ﬁrst task, you are asked to help executives understand how the subscription model is doing.
# 
# Therefore, you decide to pull data from all the users who subscribed in January and see, for each month, how many of them unsubscribed. In particular, your boss is interested in:
# 
# * A model that predicts monthly retention rate for the diﬀerent subscription price points
# * Based on your model, for each price point, what percentage of users is still subscribed after at least 12 months?
# * How do user country and source aﬀect subscription retention rate? How would you use these ﬁndings to improve the company revenue?
# 
# # Index
# * [Load the data](#Load-the-data)
# * [Answer questions 1](#Answer-question-1)
#     * [calculate total number by the end of each billing cycle for each 'monthly-cost'](#calculate-total-number-by-the-end-of-each-billing-cycle-for-each-'monthly-cost')
#     * [fit Linear Regression model](#fit-Linear-Regression-model)
#     * [predict on billing cycles from 9~12](#predict-on-billing-cycles-from-9~12)
# * [Answer question 2](#Answer-question-2)
# * [Answer question 3](#Answer-question-3)
#     * [how country affects retention rate?](#how-country-affects-retention-rate?)
#     * [how source affects retention rate?](#how-source-affects-retention-rate?)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression
get_ipython().magic('matplotlib inline')


# ## Load the data
# 

subscriptions = pd.read_csv("subscription.csv",index_col='user_id')

# 'subscription_signup_date' is always Jan, 2015 in this table. useless, so delete it
del subscriptions['subscription_signup_date']

# rename some long column name to short ones, which is easier to read
subscriptions.rename(columns={'subscription_monthly_cost':'monthly_cost',
                              'billing_cycles':'bill_cycles'},inplace=True)


# check the data, have a feeling about it
subscriptions.sample(10)


# ## Answer question 1
# A model that predicts monthly retention rate for the diﬀerent subscription price points
# 
# ### calculate total number by the end of each billing cycle for each 'monthly cost'
# 

count_by_cost = subscriptions.groupby('monthly_cost').apply(lambda df: df.bill_cycles.value_counts()).unstack()
# for index 'n', the value is the #people who paid 'n' billing cycles
count_by_cost


# for each row in 'count_by_cost', we perform a reverse cumsum to get the #people by the end of each billing cycles
total_by_cost = count_by_cost.apply(lambda s: s.iloc[::-1].cumsum().iloc[::-1],axis=1).transpose()
total_by_cost


total_by_cost.plot()


# ### fit Linear Regression model
# from above plot, we can draw following conclusions:
# 1. for each 'monthly_cost', there are only 8 samples.
#     * ** complex model will overfit on samll dataset, so I decide to use a simple regression model - Linear Regression **
# 2. the remaining #subscribers by the end of each billing cycles has a nonlinear relationship with #billing_cycles
#     * ** so I need to include some nonlinear transformation of 'billing_cycles' **
# 3. by no means, the target, #subscribers should be non-negative
#     * <span style='color:orange;font-weight:bold;font-size:1.5em'>so I cannot fit on original target, i.e., '#subscribers at the end of each billing cycle', but on log(#subscribers).</span>
#     * <span style='color:orange;font-weight:bold;font-size:1.5em'>then after obtaining the fitted value, we transform back to '#subscribers' by exp(), which can guarantee the result is always positive</span>
# 

def make_time_features(t):
    """
    three features:
    1. t: #cycles
    2. t-square: square of #cycles
    3. logt: log(#cycles)
    """
    return pd.DataFrame({'t': t,'logt': np.log(t),'tsquare':t*t },index = t)

def fit_linear_regression(s):
    """
    target:
    log(s): s is #subscribers left by the end of each billing cycle
    do this transformation, to guarantee that, after tranforming back, the fitted result is always positive
    """
    X = make_time_features(s.index)
    return LinearRegression().fit(X,np.log(s))


lr_by_cost = total_by_cost.apply(fit_linear_regression,axis=0)


# ### predict on billing cycles from 9~12
# 
# because
# 1. <span style='color:orange;'>there are so few examples (only 8) for each model. we cannot afford to split out a separate test set, which will further reduce the data used for training</span>
# 2. <span style='color:orange;'>the question is predicting near future, cycles from 9 to 12, which I assume there is no significant difference from our training data.</span>
# 
# due to above two considerations, I don't use normal cross validation to check how my model fits, but just plot the true values and my predictions to ** visualize ** how my model fits.
# 

allt = np.arange(1,13)
Xwhole = make_time_features(allt)
Xwhole


# call each cost's model to fit on above features
predicts = lr_by_cost.apply(lambda lr: pd.Series(lr.predict(Xwhole),index=allt)).transpose()
predicts = predicts.applymap(np.exp)


predicts


fig,axes = plt.subplots(3,1,sharex=True)
monthly_costs = [29,49,99]
for index,cost in enumerate(monthly_costs):
    ax = axes[index]
    total_by_cost.loc[:,cost].plot(ax = ax,label='true values')
    predicts.loc[:,cost].plot(ax=ax,label='predictions')
    ax.legend(loc='best')
    ax.set_title('monthly cost = {}'.format(cost))
plt.rc('figure',figsize=(5,10))


# combine the real values and predictions together to check how the fits going
# 

pd.merge(total_by_cost,predicts,how='right',left_index=True,right_index=True,suffixes = ('_true','_pred'))


# ## Answer question 2
# Based on your model, for each price point, what percentage of users is still subscribed after at least 12 months?

predicts.loc[12,:]/predicts.loc[1,:]


# ## Answer question 3
# How do user country and source aﬀect subscription retention rate? How would you use these ﬁndings to improve the company revenue?

def calc_retention_rate(s):
    """
    input: 
        s: n-th value is #subscribers who paid 'n' cycles
    return:
        retention rate by the end of each cycle
    """
    r = s.iloc[::-1].cumsum().iloc[::-1]
    return r/r.iloc[0]

def retention_rate_by(colname):
    """
    step 1. group subscribers based on certain column, e.g., country or source
    step 2. for each group, count #subscribers who paid 'n' cycles
    step 3. for each group, calculate retention rate for each cycle
    """
    counts = subscriptions.groupby(colname).apply(lambda df: df.bill_cycles.value_counts()).unstack()
    return counts.apply(calc_retention_rate, axis=1).transpose()


# ### how country affects retention rate?
# 

retention_rate_by_country = retention_rate_by('country')
retention_rate_by_country


retention_rate_by_country.plot(marker='o')


# rank countries by August's retention rate
retention_rate_by_country.iloc[-1,:].sort_values(ascending=False)


# based on above result, we can divide coutries into 3 classes:
# 1. customers from China and Indian are most loyal ones. 
#     * not only retention rate is high, but also the 'dropping rate' is the slowest
#     * this may be because the good economic situations in these two countries, and also because rich people in these two countries love buying foreign products to show their 'social class'.
#     * to increase the revenue, ** we should keep tight touch to customers in China and Indian. for examples, sending coupons to them from time to time.**
# 2. UK, US, Germany has medium 'retention rate'
# 3. France, Italy, Spain has the lowest 'retention rate'
#     * maybe because the poor economic conditions in these countries
#     * ** we may consider to lower the 'monthly cost' in these countries, to keep more customers in subscription **
# 

# ### how source affects retention rate?
# 

retention_rate_by_source = retention_rate_by('source')
retention_rate_by_source


retention_rate_by_source.plot(marker='x')


# from above result, we can see that subscribers from 'friend_referral' are the loyalest, much more loyal then subscribers from advertisement and search engine. 
# 
# To improve the revenue, ** we can launch some program to improve 'user referral'. for example, current subscribers can invite new users to subscribe. if your friends subscribe, you get rewarded with a certain amount of money or credit.**
# 




# You are looking at data from an e-commerce website. The site is very simple and has just 4 pages:
# 1. The ﬁrst page is the home page. When you come to the site for the ﬁrst time, you can only land on the home page as a ﬁrst page.
# 2. From the home page, the user can perform a search and land on the search page.
# 3. From the search page, if the user clicks on a product, she will get to the payment page, where she is asked to provide payment information in order to buy that product.
# 4. If she does decide to buy, she ends up on the conﬁrmation page
# 
# The company CEO isn't very happy with the volume of sales and, especially, of sales coming from new users. Therefore, she asked you to investigate whether there is something wrong in the conversion funnel or, in general, if you could suggest how conversion rate can be improved.
# 

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().magic('matplotlib inline')


# # Index
# 
# # Load the data
# 

allusers = pd.read_csv("home_page_table.csv",index_col="user_id")
users_to_search = pd.read_csv("search_page_table.csv",index_col="user_id")
users_to_pay = pd.read_csv("payment_page_table.csv",index_col="user_id")
users_to_confirm = pd.read_csv("payment_confirmation_table.csv",index_col="user_id")


allusers.loc[users_to_search.index,"page"] = users_to_search.page
allusers.loc[users_to_pay.index,"page"] = users_to_pay.page
allusers.loc[users_to_confirm.index,"page"] = users_to_confirm.page


# give it a better, more clear name
allusers.rename(columns={'page':'final_page'},inplace=True)


# change string to ordered-categorical feature
pages = ["home_page","search_page","payment_page","payment_confirmation_page"]
allusers["final_page"] = allusers.final_page.astype("category",categories = pages,ordered=True)


user_infos = pd.read_csv("user_table.csv",index_col="user_id")
user_infos.loc[:,"date"] = pd.to_datetime(user_infos.date)


allusers = allusers.join(user_infos)
allusers.head()


allusers.to_csv("all_users.csv",index_label="user_id")


# # Answer question 1
# <span style='color:blue;font-size:1.2em'>A full picture of funnel conversion rate for both desktop and mobile</span>
# 

def conversion_rates(df):
    stage_counts = df.final_page.value_counts()
    # #users converts from current page
    convert_from = stage_counts.copy()

    total = df.shape[0]
    for page in stage_counts.index:
        n_left = stage_counts.loc[page]# how many users just stop at current page
        n_convert = total - n_left
        convert_from[page] = n_convert
        total = n_convert

    cr = pd.concat([stage_counts,convert_from],axis=1,keys=["n_drop","n_convert"])
    cr["convert_rates"] = cr.n_convert.astype(np.float)/(cr.n_drop + cr.n_convert)
    cr['drop_rates'] = 1 - cr.convert_rates

    return cr


allusers.groupby('device').apply(conversion_rates)


allusers.groupby('device')['final_page'].apply(lambda s: s.value_counts(normalize=True)).unstack()


# # Answer question 2
# <span style='color:blue;font-size:1.2em'>Some insights on what the product team should focus on in order to improve conversion rate as well as anything you might discover that could help improve conversion rate.</span>
# 

allusers.head()


X = allusers.copy()


X.device.value_counts()


X['from_mobile'] = (X.device == 'Mobile').astype(int)
del X['device']


X['is_male'] = (X.sex == 'Male').astype(int)
del X['sex']


X['converted'] = (X.final_page == 'payment_confirmation_page').astype(int)
del X['final_page']


X.converted.mean()# a highly imbalanced classification problem


# ## Impact of date
# 

X.date.describe()


X['weekday'] = X.date.dt.weekday_name
del X['date']


X.head()


X.groupby('weekday')['converted'].agg(['count','mean']).sort_values(by='mean',ascending=False)


# ## Impact of sex
# 

X.groupby('is_male')['converted'].agg(['count','mean']).sort_values(by='mean',ascending=False)


# ## Statistical Test
# 

X = pd.get_dummies(X,prefix='',prefix_sep='')
X.head()


y = X.converted
X = X.loc[:,X.columns != 'converted']


scores, pvalues = chi2(X,y)


pd.DataFrame({'chi2_score':scores,'chi2_pvalue':pvalues},index=X.columns).sort_values(by='chi2_score',ascending=False)











del X['Tuesday']# remove one redundant feature





dt = DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,min_samples_split=20)
dt.fit(X,y)
export_graphviz(dt,feature_names=X.columns,class_names=['NotConvert','Converted'],
                proportion=True,leaves_parallel=True,filled=True)





# Company XYZ has started a new referral program on Oct, 31. Each user who refers a new user will get 10$ in credit when the new user buys something.
# 
# The program has been running for almost a month and the Growth Product Manager wants to know if it's been successful. She is very excited cause, since the referral program started, the company saw a spike in number of users and wants you to be able to give her some data she can show to her boss.
# 
# * Can you estimate the impact the program had on the site?
# * Based on the data, what would you suggest to do as a next step?
# * The referral program wasn't really tested in a rigorous way. It simply started on a given day for all users and you are drawing conclusions by looking at the data before and after the test started. What kinds of risks this approach presents? Can you think of a better way to test the referral program and measure its impact?

import datetime
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# # Index
# * [Load the data](#Load-the-data)
# * [Hypothesis test on all data](#Hypothesis-test-on-all-data)
# * [Hypothesis test grouped by country](#Hypothesis-test-grouped-by-country)
#     * [daily spent change in each country](#daily-spent-change-in-each-country)
#     * [daily customers change in each country](#daily-customers-change-in-each-country)
#     * [daily transactions change in each country](#daily-transactions-change-in-each-country)
#     * [Country-based conclusion](#Country-based-conclusion)
# * [Answer question 1](#Answer-question-1)
# * [Answer question 2](#Answer-question-2)
# * [Answer question 3](#Answer-question-3)
# 
# 
# ## Load the data
# 

referral = pd.read_csv("referral.csv")
del referral['device_id']
referral['date'] = pd.to_datetime( referral.date )


referral.head()# glance the data


dt_referral_starts = datetime.datetime(2015,10,31)


referral.date.describe()


(pd.Series(referral.date.unique()) >= dt_referral_starts).value_counts()


# There are 28 days before the program, and 28 days after the program. User Referral program starts right in the middle, 
# 
# ## Hypothesis test on all data
# 

def count_spent(df):
    d = {}
    d['n_purchase'] = df.shape[0]# number of purchase in that day
    d['total_spent'] = df.money_spent.sum() # total money spent in that day
    d['n_customer'] = df.user_id.unique().shape[0] # how many customers access the store that day
    return pd.Series(d)


def daily_statistics(df):
    """
    given a dataframe
    1.  group by day, and return '#purchase','total spent money','#customers' on each day
    2.  split daily data into two groups, before the program and after the program
    3.  for each 'sale index' ('#purchase','total spent money','#customers'), 
        calculate the mean before/after the program, their difference, and pvalue 
    """
    grpby_day = df.groupby('date').apply(count_spent)

    grpby_day_before = grpby_day.loc[grpby_day.index < dt_referral_starts, :]
    grpby_day_after = grpby_day.loc[grpby_day.index >= dt_referral_starts, :]

    d = []
    colnames = ['total_spent','n_purchase','n_customer']
    for col in colnames:
        pre_data = grpby_day_before.loc[:,col]
        pre_mean = pre_data.mean()

        post_data = grpby_day_after.loc[:,col]
        post_mean = post_data.mean()

        result = ss.ttest_ind(pre_data, post_data, equal_var=False)
        # either greater or smaller, just one-tail test
        pvalue = result.pvalue / 2 

        d.append({'mean_pre':pre_mean,'mean_post':post_mean,'mean_diff':post_mean - pre_mean,
                  'pvalue':pvalue})

    # re-order the columns
    return pd.DataFrame(d,index = colnames).loc[:,['mean_pre','mean_post','mean_diff','pvalue']]


daily_statistics(referral)


# <a id='whole_result'></a>although after launching the 'user referral' program, in all three 'sale index', i.e., 'daily purchase activity', 'daily money spent', 'daily customers', are all increased, however, <span style='color:orange;font-size:1.5em;font-weight:bold'>none of those increment are significant</span>. (by using a ** 0.05 ** significant level)
# 

# ## Hypothesis test grouped by country
# 

referral.country.value_counts()


daily_stat_bycountry = referral.groupby('country').apply(daily_statistics)


daily_stat_bycountry


# from above result, we know <span style='color:blue;font-weight:bold'>'User Referral' program has different effect in different countries</span>. The program boosts the sales in some country, but in some other countries, <span style='color:red;font-weight:bold'>it even decrease the sales.</span>
# 

# ### daily spent change in each country
# 

daily_stat_bycountry.xs('total_spent',level=1).sort_values(by='pvalue')


# from above result, if we loose the significant level=0.1, then
# * <span style='color:orange;font-weight:bold'>daily spent in 'CH' and 'DE' are significantly decreased.</span>
# * <span style='color:orange;font-weight:bold'>'MX','IT','FR','ES','UK', their daily spent are significant increased.</span>
# * <span style='color:orange;font-weight:bold'>'US' and 'CA' has some improvement in daily spent, but NOT significant.</span>
# 

# ### daily customers change in each country
# 

daily_stat_bycountry.xs('n_customer',level=1).sort_values(by='pvalue')


# from above result, 
# * <span style='color:orange;font-weight:bold'>daily customers in 'CH' and 'DE' are significantly decreased.</span>
# * <span style='color:orange;font-weight:bold'>'MX','IT','FR','ES', their daily customers are significant increased.
# 

# ### daily transactions change in each country
# 

daily_stat_bycountry.xs('n_purchase',level=1).sort_values(by='pvalue')


# ## Country-based conclusion
# 

# * <span style='color:orange;font-weight:bold;font-size:1.5em'>the program fails in CH and DE, it significantly decrease the sales in these two countries.</span>
# * <span style='color:orange;font-weight:bold;font-size:1.5em'>the program succeeds in 'MX','IT','FR','ES', it significantly increase the sales.</span>
# * <span style='color:orange;font-weight:bold;font-size:1.5em'>the program doesn't seem have any significant effect on UK,CA,US, especially on CA and US.</span>
# 

# ## Answer question 1
# Can you estimate the impact the program had on the site?
# 
# according to the analysis above, the program [doesn't seem have significant impacts to the whole company as a whole](#whole_result).
# 
# however, based on each country, I find the program has [different impact on different country](#Country-based-conclusion):
# * ** the program fails in CH and DE, it significantly decrease the sales in these two countries.**
# * ** the program succeeds in 'MX','IT','FR','ES', it significantly increase the sales. **
# * ** the program doesn't seem have any significant effect on UK,CA,US, especially on CA and US.**

# ## Answer question 2
# Based on the data, what would you suggest to do as a next step?

# 1. first I suggest perform more accurate A/B test ([see question 3's answer](#Answer-question-3)) and collect more data, to study the impact of the program
# 2. since the program has different impact in different country, I suggest studying the reason of such difference. ** for example, does the program has any cultural conflicts in CH and DE? **
# 

# ## Answer question 3
# The referral program wasn't really tested in a rigorous way. It simply started on a given day for all users and you are drawing conclusions by looking at the data before and after the test started. What kinds of risks this approach presents? Can you think of a better way to test the referral program and measure its impact?
# 
# this approach isn't an accurate A/B test. "User Referral" program isn't the only difference between control group and test group. for example, there may be some special holiday after Oct 31 in some country. or just because the weather get colder after Oct 31, people's requirement on some goods are increased.
# 
# To get more accurate impact of the program, we need to perform a more careful A/B test. for example:
# * during the same peroid of time
# * randomly split the customers into two groups, and let only one group know the User Referral program.
# * run the experiment some time, then perform the t-test to see whether some 'sale performance index' (e.g., daily spent, daily customers, daily transactions) have significant changes or not.

# Company XYZ is an online grocery store. In the current version of the website, they have manually grouped the items into a few categories based on their experience. 
# 
# However, they now have a lot of data about user purchase history. Therefore, they would like to put the data into use! This is what they asked you to do: 
# * The company founder wants to meet with some of the best customers to go through a focus group with them. You are asked to send the ID of the following customers to the founder: 
#     * the customer who bought the most items overall in her lifetime 
#     * for each item, the customer who bought that product the most 
# * Cluster items based on user co-purchase history. That is, create clusters of products that have the highest probability of being bought together. The goal of this is to replace the old/manually created categories with these new ones. Each item can belong to just one cluster.
# 

import re
from collections import Counter
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')


# # Index
# * [Load the data](#Load-the-data)
# * [Build user-item-count matrix](#Build-user-item-count-matrix)
# * [Answer question 1](#Answer-question-1)
# * [Answer question 2](#Answer-question-2)
# * [Build item-item-similarity matrix](#Build-item-item-similarity-matrix)
# * [Answer question 3: Clustering](#Clustering)
# 

# ## Load the data
# 

items = pd.read_csv("item_to_id.csv", index_col='Item_id')
items.sort_index(inplace=True)
items.head()


purchase_history = pd.read_csv("purchase_history.csv")
purchase_history.head()


# ## Build user-item count matrix
# 

def item_counts_by_user(same_user_df):
    # 'sum' here is adding two lists into one big list
    all_item_ids = same_user_df['id'].str.split(',').sum()
    # transform from string to int, make it easier to be sorted later
    return pd.Series(Counter(int(id) for id in all_item_ids))


user_item_counts = purchase_history.groupby("user_id").apply(item_counts_by_user).unstack(fill_value=0)


user_item_counts.shape


# each row in user_item_counts represents one user
# each column in user_item_counts represents one item
# [u,i] holds the number which user 'u' boughts item 'i'
user_item_counts.sample(5)


# ## Answer question 1
# 
# the customer who bought the most items overall in her lifetime
# 

# we assume each "item id" in the purchase history stands for 'item_count=1'
user_item_total = user_item_counts.sum(axis=1)
print "custom who bought most in lifetime is: {}, and he/she bought {} items".format(user_item_total.argmax(),user_item_total.max())


# ## Answer question 2
# for each item, the customer who bought that product the most
# 

max_user_byitem = user_item_counts.apply(lambda s: pd.Series([s.argmax(), s.max()], index=["max_user", "max_count"]))
max_user_byitem = max_user_byitem.transpose()
max_user_byitem.index.name = "Item_id"


# join with item name
max_user_byitem = max_user_byitem.join(items).loc[:, ["Item_name", "max_user", "max_count"]]
max_user_byitem


# ## Build item-item similarity matrix
# 

# A is |U|*|I|, and each item is normalized
A = normalize(user_item_counts.values, axis=0)
item_item_similarity = A.T.dot(A)
item_item_similarity = pd.DataFrame(item_item_similarity,
                                    index=user_item_counts.columns,
                                    columns=user_item_counts.columns)


item_item_similarity.head() # get a feeling about the data


# ## Clustering
# 

pca = PCA()
# rotate by PCA, making it easier to be visualized later
items_rotated = pca.fit_transform(item_item_similarity)
items_rotated = pd.DataFrame(items_rotated,
                             index=user_item_counts.columns,
                             columns=["pc{}".format(index+1) for index in xrange(items.shape[0])])


# show the total variance which can be explained by first K principle components
explained_variance_by_k = pca.explained_variance_ratio_.cumsum()
plt.plot(range(1,len(explained_variance_by_k)+1),explained_variance_by_k,marker="*")


def show_clusters(items_rotated,labels):
    """
    plot and print clustering result
    """
    fig = plt.figure(figsize=(15, 15))
    colors =  itertools.cycle (["b","g","r","c","m","y","k"])

    grps = items_rotated.groupby(labels)
    for label,grp in grps:
        plt.scatter(grp.pc1,grp.pc2,c=next(colors),label = label)

        print "*********** Label [{}] ***********".format(label)
        names = items.loc[ grp.index,"Item_name"]
        for index, name in enumerate(names):
            print "\t<{}> {}".format(index+1,name)

    # annotate
    for itemid in items_rotated.index:
        x = items_rotated.loc[itemid,"pc1"]
        y = items_rotated.loc[itemid,"pc2"]
        name = items.loc[itemid,"Item_name"]
        name = re.sub('\W', ' ', name)
        plt.text(x,y,name)

    # plt.legend(loc="best")


def cluster(n_clusters,n_components=48):
    """
    n_components=K, means use first K principle components in the clustering
    n_clusters: the number of clusters we want to cluster
    """
    print "first {} PC explain {:.1f}% variances".format(n_components,
                                                         100 * sum(pca.explained_variance_ratio_[:n_components]))

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(items_rotated.values[:, :n_components])

    # display results
    show_clusters(items_rotated, kmeans.labels_)


# choose best K (i.e., number of clusters)
inertias = []
silhouettes = []

ks = range(2,30)
for k in ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(items_rotated)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(items_rotated, kmeans.predict(items_rotated)))


fig = plt.figure(figsize=(10,4))
fig.add_subplot(1,2,1)
plt.plot(ks,inertias,marker='x')# want to use elbow method to find best k

fig.add_subplot(1,2,2)
plt.plot(ks,silhouettes,marker='o')# the higher the better


# based on above plots, ** either elbow method nor silhouette_score can give better hint about the number of clusters **. I decide to try multiple K, and choose the best one according to common sense.
# 

# use all the components
cluster(n_clusters=15)





import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans


get_ipython().magic('matplotlib inline')
sns.set_context('notebook')


# # Index
# * [Load the data](#Load-the-data)
# * [Answer question 1](#Answer-question-1)
# * [Answer question 2](#Answer-question-2)
# * [Answer question 3](#Answer-question-3)
#     * [Prepare features](#Prepare-features)
#     * [Reduce dimensions and visualize](#Reduce-dimensions-and-visualize)
#     * [Kmeans to cluster](#Kmeans-to-cluster)
#     * [Check the result](#Check-the-result)
# 

# ## Load the data
# 

holders = pd.read_csv('cc_info.csv',index_col='credit_card')
holders.rename(columns={'credit_card_limit':'credit_limit'},inplace=True)


transactions = pd.read_csv('transactions.csv')
transactions['date'] = pd.to_datetime(transactions.date)
transactions.rename(columns={'transaction_dollar_amount':'amount'},inplace=True)


# ## Answer question 1
# Your boss wants to identify those users that in your dataset never went above the monthly credit card limit (calendar month). The goal of this is to automatically increase their limit. Can you send him the list of Ids?

transactions.date.dt.year.value_counts()


# First, we need to calculate each user's monthly spent.
# 

def monthly_spent_byuser(df):
    # I have checked the data already, all transactions happen in year 2015
    # so I can just group by month
    return df.groupby(df.date.dt.month)['amount'].agg('sum')


# first group by 'credit_card' (i.e., by user)
# then sum up all spent by month
card_month_spents = transactions.groupby("credit_card").apply(monthly_spent_byuser).unstack(fill_value=0)


# join with 'credit_limit' to simplify the comparison
card_month_spents = card_month_spents.join(holders.credit_limit)
card_month_spents.head()


# Then, we check whether each user has exceed his credit limit before.
# 

n_months = card_month_spents.shape[1]-1
def is_never_above_limit(s):
    limit = s.loc['credit_limit']
    return (s.iloc[0:n_months] <= limit).all()

is_user_never_exceed_limit = card_month_spents.apply(is_never_above_limit,axis=1)

users_never_exceed_limit = card_month_spents.loc[is_user_never_exceed_limit ,:].index


users_never_exceed_limit


with open("users_never_exceed_limit.txt","wt") as outf:
    for cardno in users_never_exceed_limit:
        outf.write('{}\n'.format(cardno))


# ## Answer question 2
# On the other hand, she wants you to implement an algorithm that as soon as a user goes above her monthly limit, it triggers an alert so that the user can be notiﬁed about that.We assume here that at the beginning of the new month, user total money spent gets reset to zero (i.e. she pays the card fully at the end of each month). Build a function that for each day, returns a list of users who went above their credit card monthly limit on that day.
# 

class MonthSpentMonitor(object):

    def __init__(self,credit_limits):
        """
        card_limits is a dictionary
        key=card number, value=credit limit
        """
        self.total_spent = defaultdict(float)
        self.credit_limits = credit_limits

    def reset(self):
        self.total_spent.clear()

    def count(self,daily_transaction):
        """
        daily_transaction: a dict
        key=card number, value=amount
        """
        for cardno,amount in daily_transaction:
            self.total_spent[cardno] += amount

        # assume 'credit_limits' always can find the cardno
        # otherwise, raise KeyError, which is a good indicator showing something is wrong
        return [ cardno for cardno,total in self.total_spent.viewitems() if total > self.credit_limits[cardno]]


# <span style='color:red;'>Due to time limitation, and since the question doesn't provide enough information about the requirement (e.g. input format), I just provide above codes. if given enough time and more clear API specification, I will write some test code to test/demonstrate above codes.</span>
# 

# ### Answer question 3
# Finally, your boss is very concerned about frauds cause they are a huge cost for credit card companies. She wants you to implement an unsupervised algorithm that returns all transactions that seem unusual and are worth being investigated further.
# 

# ### Prepare features
# I think there are two factors which impact a transaction is fraud or not:
# 1. if the transaction violates that user's consumption habit. For example, if a user spend less then 200 each transaction most of the time, then a transaction more than 1000 will be highly suspicious.
# 2. if user spend the money far from his home, although it is possible due to traveling, but it's still very suspicious.
# 
# Although the data provide each transaction's geometric information and card holder's home, <span style='color:red'>unfortunately, due to time limits, I cannot relate each transation's 'Long' and 'Lat' with that card holder's home address</span>. so I have to drop the second factor listed above, and <span style='color:orange;font-size:1.5em'>only make features from user's previous comsumption history.</span>
# 
# <span style='color:red;font-weight:bold'>if given more time, I would use some Map Web API to map Long/Lat to address, and compare with card-holder's address, which will be a very useful feature to detect credit fraud.</span>
# 

def statistics_by_card(s):
    ps = [25, 50, 75]
    d = np.percentile(s,ps)
    return pd.Series(d,index=['{}%'.format(p) for p in ps])

tran_statistics = transactions.groupby('credit_card')['amount'].apply(statistics_by_card).unstack()


tran_statistics.head()


# then merge the 'transaction', 'previous consumption history' and 'credit limit' together, put all useful information about the transaction in one DataFrame
# 

# merge 'transaction' with 'previous consumption statistics'
temp = pd.merge(transactions,tran_statistics,how='left',left_on='credit_card',right_index=True)

# merge with credit limit
transactions = pd.merge(temp,holders.loc[:,['credit_limit']],how='left',left_on='credit_card',right_index=True)


transactions.tail()


# save it for later use
transactions.to_csv('extend_transactions.csv',index=False)


# ### Reduce dimensions and visualize
# 

# we only care about current amount and previous consumption history, so we can keep those useful features
# 

X = transactions.loc[:,['amount','25%','50%','75%','credit_limit']]


X.describe()


# then I want to reduce X to 2D, and visualize it to get some hint. However, <span style='color:orange'>since credit_limit has much higher variance, I need to scale each feature to unit variance before applying PCA</span>, otherwise, the principle components will be highly aligned with 'credit_limit' which doesn't provide useful information.
# 

X = scale(X)


# Use PCA to reduce feature matrix to 2D
# 

pca = PCA(n_components=2)
X2d = pca.fit_transform(X)
X2d = pd.DataFrame(X2d,columns=['pc1','pc2'])


plt.scatter(X2d.pc1,X2d.pc2,alpha=0.3)


# ### Kmeans to cluster
# 

# above plot shows a good sign, that is, the data is well seperated. also above plot give me some hint, may be the data can be grouped into 6 clusters.
# 
# then I use Kmeans algorithm to perform the clustering.
# 

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters,n_jobs=-1)
kmeans.fit(X)


X2d['label'] = kmeans.labels_
print X2d.label.value_counts()


# above result also show a good sign, cluster 3 has apparently fewer transcations compared with others. This makes senses, because fraud activity, isn't that normal.
# 
# then I plot the clusters in 2D, to check their distribution.
# 

colors = itertools.cycle( ['r','g','b','c','m','y','k'] )

plt.rc('figure',figsize=(10,6))
for label in  xrange(n_clusters) :
    temp = X2d.loc[X2d.label == label,:]
    plt.scatter(temp.pc1,temp.pc2,c=next(colors),label=label,alpha=0.3)

plt.legend(loc='best')


X2d.head()


g = sns.FacetGrid(X2d, hue="label")
g.map(plt.scatter, "pc1", "pc2", alpha=0.3)
g.add_legend();


# ### Check the result
# the cyan points represents the suspicious transactions. Let's pick those transactions out, and check whether they make sense or not.
# 

suspicious_label = X2d.label.value_counts().argmin()
suspicious_label


suspect = transactions.loc[X2d.label==suspicious_label,['credit_card','amount','25%','50%','75%','credit_limit','date']]
suspect.to_csv('suspect.csv',index=False)


suspect.sample(10)


# open 'suspect.csv' or see the randomly sampled suspicious transactions list above, we can find <span style='color:orange;font-weight:bold;font-size:1.5em'>their amount is much higher than that user's 75th percentile in his previous consumption history.</span>
# 
# for example, in 12697-th transaction, that user's 75th percentile is 89.04, then suddenly it comes a transaction with 977.38, which is very suspicious and need further investigation. 
# 

labels = ["amount",'75%']
plt.hist(suspect.loc[:,labels].values,bins=50,label=labels)
plt.legend(loc='best')


# histogram of the "consumption amount" and that user's 75th percentile of consumption records, also shows, <span style='color:orange;font-weight:bold;font-size:1.5em'>selected transaction has 'amount' much higher than '75th percentile', which is very suspicious and worth further investigation.</span>
# 




