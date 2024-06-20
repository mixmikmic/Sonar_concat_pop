# In this chapter we cover
# - Efficient markets hypothesis: strong form v. weak form
# - Random walk
# - persistence and regression to the mean
# - Fundamental vs. Technical Analysis
# - What the literature says: twitter+mood, momentum, january effect
# - herd behavior, information cascades, private information, game theory
# - red queen games
# - trade people not prices - harder to uncover and more robust
# - biases and errors: survivorship, data mining fallacy , stateful strategies
# 

import pandas as pd
import numpy as np
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
pd.set_option('display.max_colwidth', 200)


#!pip install pandas_datareader


import pandas_datareader as pdr

start_date = pd.to_datetime('2010-01-01')
stop_date = pd.to_datetime('2016-03-01')

spy = pdr.data.get_data_yahoo('SPY', start_date, stop_date)

spy


spy_c = spy['Close']


fig, ax = plt.subplots(figsize=(15,10))
spy_c.plot(color='k')
plt.title("SPY", fontsize=20)


first_open = spy['Open'].iloc[0]
first_open


last_close = spy['Close'].iloc[-1]
last_close


last_close - first_open


spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])


spy['Daily Change'].sum()


np.std(spy['Daily Change'])


spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))


spy['Overnight Change'].sum()


np.std(spy['Overnight Change'])


# daily returns
daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
daily_rtn


daily_rtn.hist(bins=50, color='lightblue', figsize=(12,8))


# intra day returns
id_rtn = ((spy['Close'] - spy['Open'])/spy['Open'])*100
id_rtn


id_rtn.hist(bins=50, color='lightblue', figsize=(12,8))


# overnight returns
on_rtn = ((spy['Open'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
on_rtn


on_rtn.hist(bins=50, color='lightblue', figsize=(12,8))


def get_stats(s, n=252):
    s = s.dropna()
    wins = len(s[s>0])
    losses = len(s[s<0])
    evens = len(s[s==0])
    mean_w = round(s[s>0].mean(), 3)
    mean_l = round(s[s<0].mean(), 3)
    win_r = round(wins/losses, 3)
    mean_trd = round(s.mean(), 3)
    sd = round(np.std(s), 3)
    max_l = round(s.min(), 3)
    max_w = round(s.max(), 3)
    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
    cnt = len(s)
    print('Trades:', cnt,          '\nWins:', wins,          '\nLosses:', losses,          '\nBreakeven:', evens,          '\nWin/Loss Ratio', win_r,          '\nMean Win:', mean_w,          '\nMean Loss:', mean_l,          '\nMean', mean_trd,          '\nStd Dev:', sd,          '\nMax Loss:', max_l,          '\nMax Win:', max_w,          '\nSharpe Ratio:', sharpe_r)


get_stats(daily_rtn)


get_stats(id_rtn)


get_stats(on_rtn)


def get_signal(x):
    val = np.random.rand()
    if val > .5:
        return 1
    else:
        return 0


for i in range(1000):
    spy['Signal_' + str(i)] = spy.apply(get_signal, axis=1)


spy


#spy.to_csv('/Users/alexcombs/Downloads/spy.csv', index=False)
spy = pd.read_csv('/Users/alexcombs/Downloads/spy.csv')
#spy.drop([x for x in spy.columns is 'Signal' in x])


sumd={}
for i in range(1000):
    sumd.update({i: np.where(spy['Signal_' + str(i)].iloc[1:]==1, spy['Overnight Change'].iloc[1:],0).sum()})


returns = pd.Series(sumd).to_frame('return').sort_values('return', ascending=0)


returns


mystery_rtn = pd.Series(np.where(spy['Signal_270'].iloc[1:]==1,spy['Overnight Change'].iloc[1:],0))


get_stats(mystery_rtn)


start_date = pd.to_datetime('2000-01-01')
stop_date = pd.to_datetime('2016-03-01')

sp = pdr.data.get_data_yahoo('SPY', start_date, stop_date)


sp


fig, ax = plt.subplots(figsize=(15,10))
sp['Close'].plot(color='k')
plt.title("SPY", fontsize=20)


long_day_rtn = ((sp['Close'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100


(sp['Close'] - sp['Close'].shift(1)).sum()


get_stats(long_day_rtn)


long_id_rtn = ((sp['Close'] - sp['Open'])/sp['Open'])*100


(sp['Close'] - sp['Open']).sum()


get_stats(long_id_rtn)


long_on_rtn = ((sp['Open'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100


(sp['Open'] - sp['Close'].shift(1)).sum()


get_stats(long_on_rtn)


for i in range(1, 21, 1):
    sp.loc[:,'Close Minus ' + str(i)] = sp['Close'].shift(i)


sp


sp20 = sp[[x for x in sp.columns if 'Close Minus' in x or x == 'Close']].iloc[20:,]


sp20


sp20 = sp20.iloc[:,::-1]


sp20


from sklearn.svm import SVR


clf = SVR(kernel='linear')


len(sp20)


X_train = sp20[:-2000]
y_train = sp20['Close'].shift(-1)[:-2000]


X_test = sp20[-2000:-1000]
y_test = sp20['Close'].shift(-1)[-2000:-1000]


model = clf.fit(X_train, y_train)


preds = model.predict(X_test)


preds


len(preds)


tf = pd.DataFrame(list(zip(y_test, preds)), columns=['Next Day Close', 'Predicted Next Close'], index=y_test.index)


tf


cdc = sp[['Close']].iloc[-2000:-1000]
ndo = sp[['Open']].iloc[-2000:-1000].shift(-1)


tf1 = pd.merge(tf, cdc, left_index=True, right_index=True)
tf2 = pd.merge(tf1, ndo, left_index=True, right_index=True)
tf2.columns = ['Next Day Close', 'Predicted Next Close', 'Current Day Close', 'Next Day Open']


tf2


def get_signal(r):
    if r['Predicted Next Close'] > r['Next Day Open'] + 1:
        return 0
    else:
        return 1


def get_ret(r):
    if r['Signal'] == 1:
        return ((r['Next Day Close'] - r['Next Day Open'])/r['Next Day Open']) * 100
    else:
        return 0


tf2 = tf2.assign(Signal = tf2.apply(get_signal, axis=1))
tf2 = tf2.assign(PnL = tf2.apply(get_ret, axis=1))


tf2


(tf2[tf2['Signal']==1]['Next Day Close'] - tf2[tf2['Signal']==1]['Next Day Open']).sum()


(sp['Close'].iloc[-2000:-1000] - sp['Open'].iloc[-2000:-1000]).sum()


get_stats((sp['Close'].iloc[-2000:-1000] - sp['Open'].iloc[-2000:-1000])/sp['Open'].iloc[-2000:-1000] * 100)


get_stats(tf2['PnL'])


#!pip install fastdtw


from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance


tseries = []
tlen = 5
for i in range(tlen, len(sp), tlen):
    pctc = sp['Close'].iloc[i-tlen:i].pct_change()[1:].values * 100
    res = sp['Close'].iloc[i-tlen:i+1].pct_change()[-1] * 100
    tseries.append((pctc, res))


len(tseries)


tseries[0]


dist_pairs = []
for i in range(len(tseries)):
    for j in range(len(tseries)):
        dist = dtw_dist(tseries[i][0], tseries[j][0])
        dist_pairs.append((i,j,dist,tseries[i][1], tseries[j][1]))


dist_frame = pd.DataFrame(dist_pairs, columns=['A','B','Dist', 'A Ret', 'B Ret'])


sf = dist_frame[dist_frame['Dist']>0].sort_values(['A','B']).reset_index(drop=1)


sfe = sf[sf['A']<sf['B']]


winf = sfe[(sfe['Dist']<=1)&(sfe['A Ret']>0)]


winf


plt.plot(np.arange(4), tseries[6][0])


plt.plot(np.arange(4), tseries[598][0])


excluded = {}
return_list = []
def get_returns(r):
    if excluded.get(r['A']) is None:
        return_list.append(r['B Ret'])
        if r['B Ret'] < 0:
            excluded.update({r['A']:1})


winf.apply(get_returns, axis=1);


get_stats(pd.Series(return_list))








import pandas as pd
import numpy as np
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity


cosine_similarity(np.array([4,0,5,3,5,0,0]).reshape(1,-1),                  np.array([0,4,0,4,0,5,0]).reshape(1,-1))


cosine_similarity(np.array([4,0,5,3,5,0,0]).reshape(1,-1),                  np.array([2,0,2,0,1,0,0]).reshape(1,-1))


cosine_similarity(np.array([-.25,0,.75,-1.25,.75,0,0])                  .reshape(1,-1),                  np.array([0,-.33,0,-.33,0,.66,0])                  .reshape(1,-1))


cosine_similarity(np.array([-.25,0,.75,-1.25,.75,0,0])                  .reshape(1,-1),                  np.array([.33,0,.33,0,-.66,0,0])                  .reshape(1,-1))


user_x = [0,.33,0,-.66,0,33,0]
user_y = [0,0,0,-1,0,.5,.5]

cosine_similarity(np.array(user_x).reshape(1,-1),                  np.array(user_y).reshape(1,-1))


user_x = [0,.33,0,-.66,0,33,0]
user_z = [0,-.125,0,-.625,0,.375,.375]

cosine_similarity(np.array(user_x).reshape(1,-1),                  np.array(user_z).reshape(1,-1))


s1 = [-1.0,0.0,0.0,0.0,1.0]
s2 = [-1.66,0.0,.33,0.0,1.33]

cosine_similarity(np.array(s1).reshape(1,-1),                  np.array(s2).reshape(1,-1))


# ## Build a Recommendation Engine
# 

myun = 'YOUR_USERNAME'
mypw = 'YOUR_USER_TOKEN'


# ### Get the repos I have starred
# 

my_starred_repos = []
def get_starred_by_me():
    resp_list = []
    last_resp = ''
    first_url_to_get = 'https://api.github.com/user/starred'
    first_url_resp = requests.get(first_url_to_get, auth=(myun,mypw))
    last_resp = first_url_resp
    resp_list.append(json.loads(first_url_resp.text))
    
    while last_resp.links.get('next'):
        next_url_to_get = last_resp.links['next']['url']
        next_url_resp = requests.get(next_url_to_get, auth=(myun,mypw))
        last_resp = next_url_resp
        resp_list.append(json.loads(next_url_resp.text))
        
    for i in resp_list:
        for j in i:
            msr = j['html_url']
            my_starred_repos.append(msr)


get_starred_by_me()


my_starred_repos


len(my_starred_repos)


my_starred_users = []
for ln in my_starred_repos:
    right_split = ln.split('.com/')[1]
    starred_usr = right_split.split('/')[0]
    my_starred_users.append(starred_usr)


my_starred_users


len(my_starred_users)


# ### Looks like some duplication because I starred multiple repos for some users
# 

len(set(my_starred_users))


# ### Now let's get the repos they starred
# 

starred_repos = {k:[] for k in set(my_starred_users)}
def get_starred_by_user(user_name):
    starred_resp_list = []
    last_resp = ''
    first_url_to_get = 'https://api.github.com/users/'+ user_name +'/starred'
    first_url_resp = requests.get(first_url_to_get, auth=(myun,mypw))
    last_resp = first_url_resp
    starred_resp_list.append(json.loads(first_url_resp.text))
    
    while last_resp.links.get('next'):
        next_url_to_get = last_resp.links['next']['url']
        next_url_resp = requests.get(next_url_to_get, auth=(myun,mypw))
        last_resp = next_url_resp
        starred_resp_list.append(json.loads(next_url_resp.text))
        
    for i in starred_resp_list:
        for j in i:
            sr = j['html_url']
            starred_repos.get(user_name).append(sr)


for usr in list(set(my_starred_users)):
    print(usr)
    try:
        get_starred_by_user(usr)
    except:
        print('failed for user', usr)


len(starred_repos)


# ### Now we'll build a vocabulary that includes all the repos starred by the users I starred
# 

repo_vocab = [item for sl in list(starred_repos.values()) for item in sl]


repo_set = list(set(repo_vocab))


len(repo_set)


all_usr_vector = []
for k,v in starred_repos.items():
    usr_vector = []
    for url in repo_set:
        if url in v:
            usr_vector.extend([1])
        else:
            usr_vector.extend([0])
    all_usr_vector.append(usr_vector)


len(all_usr_vector)


df = pd.DataFrame(all_usr_vector, columns=repo_set, index=starred_repos.keys())


df


len(df.columns)


# ### Now I need to add myself to this dataframe to find the similarity between myself and the other users
# 

my_repo_comp = []
for i in df.columns:
    if i in my_starred_repos:
        my_repo_comp.append(1)
    else:
        my_repo_comp.append(0)


mrc = pd.Series(my_repo_comp).to_frame(myun).T


mrc


mrc.columns = df.columns


fdf = pd.concat([df, mrc])


fdf


l2 = my_starred_repos


l1 = fdf.iloc[-1,:][fdf.iloc[-1,:]==1].index.values


a = set(l1)
b = set(l2)


b.difference(a)


from sklearn.metrics import jaccard_similarity_score
from scipy.stats import pearsonr


sim_score = {}
for i in range(len(fdf)):
    ss = pearsonr(fdf.iloc[-1,:], fdf.iloc[i,:])
    sim_score.update({i: ss[0]})


sf = pd.Series(sim_score).to_frame('similarity')


sf


sf.sort_values('similarity', ascending=False)


fdf.index[5]


fdf.iloc[71,:][fdf.iloc[71,:]==1]


all_recs = fdf.iloc[[31,5,71,79],:][fdf.iloc[[31,5,71,79],:]==1].fillna(0).T


all_recs[(all_recs==1).all(axis=1)]


str_recs_tmp = all_recs[all_recs[myun]==0].copy()
str_recs = str_recs_tmp.iloc[:,:-1].copy()
str_recs


str_recs[(str_recs==1).all(axis=1)]


str_recs[str_recs.sum(axis=1)>1]





# What we'll cover in this chapter
# - What topics have been the most shared in the past year?
# - What does the research on virality say?
# - A look at headlines
# - topic modelling and visualization
# - cosine similarity

import requests
import pandas as pd
import numpy as np
import json
import time
from selenium import webdriver

pd.set_option('display.max_colwidth', 200)


browser = webdriver.PhantomJS()
browser.set_window_size(1080,800)
browser.get("http://www.ruzzit.com/en-US/Timeline?media=Articles&timeline=Year1&networks=All")
time.sleep(3)

pg_scroll_count = 50

while pg_scroll_count:
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(15)
    pg_scroll_count -= 1

titles = browser.find_elements_by_class_name("article_title")
link_class = browser.find_elements_by_class_name("link_read_more_article")
stats = browser.find_elements_by_class_name("ruzzit_statistics_area")


all_data = []
for title, link, stat in zip(titles, link_class, stats):
    all_data.append((title.text,                     link.get_attribute("href"),                     stat.find_element_by_class_name("col-md-12").text.split(' shares')[0],
                     stat.find_element_by_class_name("col-md-12").text.split('tweets\n')[1].split('likes\n0')[0],
                     stat.find_element_by_class_name("col-md-12").text.split('1\'s\n')[1].split(' pins')[0],
                     stat.find_element_by_class_name("col-md-12").text.split('pins\n')[1]))


all_data


df = pd.DataFrame(all_data, columns=['title', 'link', 'fb', 'lnkdn', 'pins', 'date'])
df


#browser.save_screenshot('/Users/alexcombs/Desktop/testimg.png')


df = df.assign(redirect = df['link'].map(lambda x: requests.get(x).url))


df


def check_home(x):
    if '.com' in x:
        if len(x.split('.com')[1]) < 2:
            return 1
        else:
            return 0
    else:
        return 0


def check_img(x):
    if '.gif' in x or '.jpg' in x:
        return 1
    else:
        return 0


df = df.assign(pg_missing = df['pg_missing'].map(check_home))


df = df.assign(img_link = df['redirect'].map(check_img))


df


df[df['pg_missing']==1]


len(df[df['pg_missing']==1])


len(df[df['img_link']==1])


df[df['pg_missing']==1]


dfc = df[(df['img_link']!=1)&(df['pg_missing']!=1)]


dfc


def get_data(x):
    try:
        data = requests.get('https://api.embedly.com/1/extract?key=SECRET_KEY&url=' + x)
        json_data = json.loads(data.text)
        return json_data
    except:
        print('Failed')
        return None


dfc = dfc.assign(json_data = dfc['redirect'].map(get_data))


dfc_bak = dfc


dfc


def get_title(x):
    try:
        return x.get('title')
    except:
        return None


dfc = dfc.assign(title = dfc['json_data'].map(get_title))


def get_site(x):
    try:
        return x.get('provider_name')
    except:
        return None


dfc = dfc.assign(site = dfc['json_data'].map(get_site))


def get_images(x):
    try:
        return len(x.get('images'))
    except:
        return None


dfc = dfc.assign(img_count = dfc['json_data'].map(get_images))


def get_entities(x):
    try:
        return [y.get('name') for y in x.get('entities')]
    except:
        return None


dfc = dfc.assign(entities = dfc['json_data'].map(get_entities))


def get_html(x):
    try:
        return x.get('content')
    except:
        return None


dfc = dfc.assign(html = dfc['json_data'].map(get_html))


dfc[::-1]


from bs4 import BeautifulSoup


def text_from_html(x):
    try:
        soup = BeautifulSoup(x, 'lxml')
        return soup.get_text()
    except:
        return None


dfc = dfc.assign(text = dfc['html'].map(text_from_html))


dfc[::-1]


# ## Clean up data for counts and date
# 

def clean_counts(x):
    if 'M' in str(x):
        d = x.split('M')[0]
        dm = float(d) * 1000000
        return dm
    elif 'k' in str(x):
        d = x.split('k')[0]
        dk = float(d.replace(',','')) * 1000
        return dk
    elif ',' in str(x):
        d = x.replace(',','')
        return int(d)
    else:
        return x


dfc = dfc.assign(fb = dfc['fb'].map(clean_counts))


dfc = dfc.assign(lnkdn = dfc['lnkdn'].map(clean_counts))


dfc = dfc.assign(pins = dfc['pins'].map(clean_counts))


dfc = dfc.assign(date = pd.to_datetime(dfc['date'], dayfirst=True))


dfc


def get_word_count(x):
    if not x is None:
        return len(x.split(' '))
    else:
        return None


dfc = dfc.assign(word_count = dfc['text'].map(get_word_count))


dfc[['text','word_count']][::-1]


# ## Get Main Image Colors
# 

import matplotlib.colors as mpc


def get_hex(x):
    try:
        if x.get('images'):
            main_color = x.get('images')[0].get('colors')[0].get('color')
            return mpc.rgb2hex([(x/255) for x in main_color])
    except:
        return None


def get_rgb(x):
    try:
        if x.get('images'):
            main_color = x.get('images')[0].get('colors')[0].get('color')
            return main_color
    except:
        return None


dfc = dfc.assign(main_hex = dfc['json_data'].map(get_hex))
dfc = dfc.assign(main_rgb = dfc['json_data'].map(get_rgb))


dfc


dfc['img_count'].value_counts().to_frame('count')


fig, ax = plt.subplots(figsize=(8,6))
y = dfc['img_count'].value_counts().sort_index()
x = y.sort_index().index
plt.bar(x, y, color='k', align='center')
plt.title('Image Count Frequency', fontsize=16, y=1.01)
ax.set_xlim(-.5,5.5)
ax.set_ylabel('Count')
ax.set_xlabel('Number of Images')


#dfc.to_json('/Users/alexcombs/Desktop/viral_data.json')
dfc = pd.read_json('/Users/alexcombs/Desktop/viral_data.json')


mci = dfc['main_hex'].value_counts().to_frame('count')
mci


mci['color'] = ' '


def color_cells(x):
    return 'background-color: ' + x.index


mci.style.apply(color_cells, subset=['color'], axis=0)


def get_csplit(x):
    try:
        return x[0], x[1], x[2]
    except:
        return None, None, None


dfc['reds'], dfc['greens'], dfc['blues'] = zip(*dfc['main_rgb'].map(get_csplit))


dfc


from sklearn.cluster import KMeans


np.sqrt(256)


clf = KMeans(n_clusters=16)


clf.fit(dfc[['reds', 'greens', 'blues']].dropna())


clusters = pd.DataFrame(clf.cluster_centers_, columns=['r', 'g', 'b'])


clusters


def hexify(x):
    rgb = [round(x['r']), round(x['g']), round(x['b'])]
    hxc = mpc.rgb2hex([(x/255) for x in rgb])
    return hxc


clusters.index = clusters.apply(hexify, axis=1)


clusters


clusters['color'] = ' '


clusters


clusters.style.apply(color_cells, subset=['color'], axis=0)


dfc[dfc['title'].isnull()]


# ## Headline Analysis
# 

from nltk.util import ngrams
from nltk.corpus import stopwords
import re

def get_word_stats(txt_series, n, rem_stops=False):
    txt_words = []
    txt_len = []
    for w in txt_series:
        if w is not None:
            if rem_stops == False:
                word_list = [x for x in ngrams(re.findall('[a-z0-9\']+', w.lower()), n)]
            else:
                word_list = [y for y in ngrams([x for x in re.findall('[a-z0-9\']+', w.lower())                                                if x not in stopwords.words('english')], n)]
            word_list_len = len(list(word_list))
            txt_words.extend(word_list)
            txt_len.append(word_list_len)
    return pd.Series(txt_words).value_counts().to_frame('count'), pd.DataFrame(txt_len, columns=['count'])


hw,hl = get_word_stats(dfc['title'], 3, 1)


hw


hl.describe()


tt = dfc[~dfc['title'].isnull()]


tt[tt['title'].str.contains('Dies')]


dfc['site'].value_counts().to_frame()


# ## Examine the Body Content
# 

hw,hl = get_word_stats(dfc['text'], 3, 1)


hw


# ## Build Predictive Model
# 

from sklearn.ensemble import RandomForestRegressor


all_data = dfc.dropna(subset=['img_count', 'word_count'])


all_data.reset_index(inplace=True, drop=True)


all_data


train_index = []
test_index = []
for i in all_data.index:
    result = np.random.choice(2, p=[.65,.35])
    if result == 1:
        test_index.append(i)
    else:
        train_index.append(i)


print('test length:', len(test_index), '\ntrain length:', len(train_index))


sites = pd.get_dummies(all_data['site'])


sites


y_train = all_data.iloc[train_index]['fb'].astype(int)
X_train_nosite = all_data.iloc[train_index][['img_count', 'word_count']]


X_train = pd.merge(X_train_nosite, sites.iloc[train_index], left_index=True, right_index=True)


y_test = all_data.iloc[test_index]['fb'].astype(int)
X_test_nosite = all_data.iloc[test_index][['img_count', 'word_count']]


X_test = pd.merge(X_test_nosite, sites.iloc[test_index], left_index=True, right_index=True)


clf = RandomForestRegressor(n_estimators=100)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


y_actual = y_test


deltas = pd.DataFrame(list(zip(y_pred, y_actual, (y_pred - y_actual)/(y_actual))), columns=['predicted', 'actual', 'delta'])


deltas


deltas['delta'].describe()


a = pd.Series([10,10,10,10])
b = pd.Series([12,8,8,12])


np.sqrt(np.mean((b-a)**2))/np.mean(a)


(b-a).mean()


np.sqrt(np.mean((y_pred-y_actual)**2))/np.mean(y_actual)





# ## Use title n-grams
# 

from sklearn.feature_extraction.text import CountVectorizer


vect = CountVectorizer(ngram_range=(1,3))


X_titles_all = vect.fit_transform(all_data['title'])


X_titles_train = X_titles_all[train_index]


X_titles_test = X_titles_all[test_index]


len(X_titles_train.toarray())


len(X_titles_test.toarray())


len(X_train)


len(X_test)


X_test = pd.merge(X_test, pd.DataFrame(X_titles_test.toarray(), index=X_test.index), left_index=True, right_index=True)


X_train = pd.merge(X_train, pd.DataFrame(X_titles_train.toarray(), index=X_train.index), left_index=True, right_index=True)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


deltas = pd.DataFrame(list(zip(y_pred, y_actual, (y_pred - y_actual)/(y_actual))), columns=['predicted', 'actual', 'delta'])


deltas


np.sqrt(np.mean((y_pred-y_actual)**2))/np.mean(y_actual)


# ## Add Title Word Count
# 

all_data = all_data.assign(title_wc = all_data['title'].map(lambda x: len(x.split(' '))))


X_train = pd.merge(X_train, all_data[['title_wc']], left_index=True, right_index=True)


X_test = pd.merge(X_test, all_data[['title_wc']], left_index=True, right_index=True)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


y_actual = y_test


np.sqrt(np.mean((y_pred-y_actual)**2))/np.mean(y_actual)








