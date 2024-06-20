import pandas as pd


# - [Click-through ad data from Kaggle competition](https://www.kaggle.com/c/avazu-ctr-prediction/data)
# - train_subset is first 10K rows of 6+GB set
# 

df = pd.read_csv('data/train_subset.csv')


df.head(3)


# how many features should we have after?
len(df['device_id'].unique())


# Features are $\theta$ = [$N^+$, $N^-$, $log(N^+)-log(N^-)$, isRest]
# 
# $N^+$ = $p(+)$ = $n^+/(n^+ + n^-)$
# 
# $N^-$ = $p(-)$ = $n^-/(n^+ + n^-)$
# 
# $log(N^+)-log(N^-)$ = $\frac{p(+)}{p(-)}$
# 
# isRest = back-off bin (not shown here)
# 

def click_counting(x, bin_column):
    clicks = pd.Series(x[x['click'] > 0][bin_column].value_counts(), name='clicks')
    no_clicks = pd.Series(x[x['click'] < 1][bin_column].value_counts(), name='no_clicks')
    
    counts = pd.DataFrame([clicks,no_clicks]).T.fillna('0')
    counts['total'] = counts['clicks'].astype('int64') + counts['no_clicks'].astype('int64')
    
    return counts

def bin_counting(counts):
    counts['N+'] = counts['clicks'].astype('int64').divide(counts['total'].astype('int64'))
    counts['N-'] = counts['no_clicks'].astype('int64').divide(counts['total'].astype('int64'))
    counts['log_N+'] = counts['N+'].divide(counts['N-'])

#    If we wanted to only return bin-counting properties, we would filter here
    bin_counts = counts.filter(items= ['N+', 'N-', 'log_N+'])
    return counts, bin_counts


# bin counts example: device_id
bin_column = 'device_id'
device_clicks = click_counting(df.filter(items= [bin_column, 'click']), bin_column)
device_all, device_bin_counts = bin_counting(device_clicks)


# check to make sure we have all the devices
len(device_bin_counts)


device_all.sort_values(by = 'total', ascending=False).head(4)


# We can see how this can change model evaluation time by comparing raw vs. bin-counting size
from sys import getsizeof

print('Our pandas Series, in bytes: ', getsizeof(df.filter(items= ['device_id', 'click'])))
print('Our bin-counting feature, in bytes: ', getsizeof(device_bin_counts))





import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as modsel
import sklearn.preprocessing as preproc


# ## Load and prep Yelp reviews data
# 

## Load Yelp Business data
biz_f = open('data/yelp/v6/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json')
biz_df = pd.DataFrame([json.loads(x) for x in biz_f.readlines()])
biz_f.close()

## Load Yelp Reviews data
review_file = open('data/yelp/v6/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')
review_df = pd.DataFrame([json.loads(x) for x in review_file.readlines()])
review_file.close()


biz_df.shape


review_df.shape


# Pull out only Nightlife and Restaurants businesses
two_biz = biz_df[biz_df.apply(lambda x: 'Nightlife' in x['categories'] 
                                        or 'Restaurants' in x['categories'], 
                              axis=1)]


two_biz.shape


biz_df.shape


## Join with the reviews to get all reviews on the two types of business
twobiz_reviews = two_biz.merge(review_df, on='business_id', how='inner')


twobiz_reviews.shape


twobiz_reviews.to_pickle('data/yelp/v6/yelp_dataset_challenge_academic_dataset/twobiz_reviews.pkl')


twobiz_reviews = pd.read_pickle('data/yelp/v6/yelp_dataset_challenge_academic_dataset/twobiz_reviews.pkl')


# Trim away the features we won't use
twobiz_reviews = twobiz_reviews[['business_id', 
                                 'name', 
                                 'stars_y', 
                                 'text', 
                                 'categories']]


# Create the target column--True for Nightlife businesses, and False otherwise
twobiz_reviews['target'] = twobiz_reviews.apply(lambda x: 'Nightlife' in x['categories'],
                                                axis=1)


## Now pull out each class of reviews separately, 
## so we can create class-balanced samples for training
nightlife = twobiz_reviews[twobiz_reviews.apply(lambda x: 'Nightlife' in x['categories'], axis=1)]
restaurants = twobiz_reviews[twobiz_reviews.apply(lambda x: 'Restaurants' in x['categories'], axis=1)]


nightlife.shape


restaurants.shape


nightlife_subset = nightlife.sample(frac=0.1, random_state=123)
restaurant_subset = restaurants.sample(frac=0.021, random_state=123)


nightlife_subset.shape


restaurant_subset.shape


nightlife_subset.to_pickle('data/yelp/v6/yelp_dataset_challenge_academic_dataset/nightlife_subset.pkl')
restaurant_subset.to_pickle('data/yelp/v6/yelp_dataset_challenge_academic_dataset/restaurant_subset.pkl')


nightlife_subset = pd.read_pickle('data/yelp/v6/yelp_dataset_challenge_academic_dataset/nightlife_subset.pkl')
restaurant_subset = pd.read_pickle('data/yelp/v6/yelp_dataset_challenge_academic_dataset/restaurant_subset.pkl')


combined = pd.concat([nightlife_subset, restaurant_subset])


combined['target'] = combined.apply(lambda x: 'Nightlife' in x['categories'],
                                    axis=1)


combined


# Split into training and test data sets
training_data, test_data = modsel.train_test_split(combined, 
                                                   train_size=0.7, 
                                                   random_state=123)


training_data.shape


test_data.shape


# Represent the review text as a bag-of-words 
bow_transform = text.CountVectorizer()
X_tr_bow = bow_transform.fit_transform(training_data['text'])


len(bow_transform.vocabulary_)


X_tr_bow.shape


X_te_bow = bow_transform.transform(test_data['text'])


y_tr = training_data['target']
y_te = test_data['target']


# Create the tf-idf representation using the bag-of-words matrix
tfidf_trfm = text.TfidfTransformer(norm=None)
X_tr_tfidf = tfidf_trfm.fit_transform(X_tr_bow)


X_te_tfidf = tfidf_trfm.transform(X_te_bow)


X_tr_l2 = preproc.normalize(X_tr_bow, axis=0)
X_te_l2 = preproc.normalize(X_te_bow, axis=0)


# ## Classify with logistic regression
# 

def simple_logistic_classify(X_tr, y_tr, X_test, y_test, description, _C=1.0):
    ## Helper function to train a logistic classifier and score on test data
    m = LogisticRegression(C=_C).fit(X_tr, y_tr)
    s = m.score(X_test, y_test)
    print ('Test score with', description, 'features:', s)
    return m


m1 = simple_logistic_classify(X_tr_bow, y_tr, X_te_bow, y_te, 'bow')
m2 = simple_logistic_classify(X_tr_l2, y_tr, X_te_l2, y_te, 'l2-normalized')
m3 = simple_logistic_classify(X_tr_tfidf, y_tr, X_te_tfidf, y_te, 'tf-idf')


# ## Tune regularization parameters using grid search
# 

param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
bow_search = modsel.GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
l2_search = modsel.GridSearchCV(LogisticRegression(), cv=5,
                               param_grid=param_grid_)
tfidf_search = modsel.GridSearchCV(LogisticRegression(), cv=5,
                                   param_grid=param_grid_)


bow_search.fit(X_tr_bow, y_tr)


bow_search.best_score_


l2_search.fit(X_tr_l2, y_tr)


l2_search.best_score_


tfidf_search.fit(X_tr_tfidf, y_tr)


tfidf_search.best_score_


bow_search.best_params_


l2_search.best_params_


tfidf_search.best_params_


bow_search.cv_results_


import pickle


results_file = open('tfidf_gridcv_results.pkl', 'wb')
pickle.dump(bow_search, results_file, -1)
pickle.dump(tfidf_search, results_file, -1)
pickle.dump(l2_search, results_file, -1)
results_file.close()


pkl_file = open('tfidf_gridcv_results.pkl', 'rb')
bow_search = pickle.load(pkl_file)
tfidf_search = pickle.load(pkl_file)
l2_search = pickle.load(pkl_file)
pkl_file.close()


search_results = pd.DataFrame.from_dict({'bow': bow_search.cv_results_['mean_test_score'],
                               'tfidf': tfidf_search.cv_results_['mean_test_score'],
                               'l2': l2_search.cv_results_['mean_test_score']})
search_results


# ## Plot cross validation results
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


ax = sns.boxplot(data=search_results, width=0.4)
ax.set_ylabel('Accuracy', size=14)
ax.tick_params(labelsize=14)
plt.savefig('tfidf_gridcv_results.png')


m1 = simple_logistic_classify(X_tr_bow, y_tr, X_te_bow, y_te, 'bow', 
                              _C=bow_search.best_params_['C'])
m2 = simple_logistic_classify(X_tr_l2, y_tr, X_te_l2, y_te, 'l2-normalized', 
                              _C=l2_search.best_params_['C'])
m3 = simple_logistic_classify(X_tr_tfidf, y_tr, X_te_tfidf, y_te, 'tf-idf', 
                              _C=tfidf_search.best_params_['C'])


bow_search.cv_results_['mean_test_score']





import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib notebook')
sns.set_style('whitegrid')

from sklearn import linear_model
from sklearn.model_selection import cross_val_score


# # Log Transform on Yelp Reviews Dataset
# 

## Load the data
biz_f = open('data/yelp/v6/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json')
biz_df = pd.DataFrame([json.loads(x) for x in biz_f.readlines()])
biz_f.close()


## Compute the log transform of the review count
biz_df['log_review_count'] = np.log10(biz_df['review_count'] + 1)


## Visualize the distribution of review counts before and after log transform
plt.figure()
ax = plt.subplot(2,1,1)
biz_df['review_count'].hist(ax=ax, bins=100)
ax.tick_params(labelsize=14)
ax.set_xlabel('review_count', fontsize=14)
ax.set_ylabel('Occurrence', fontsize=14)

ax = plt.subplot(2,1,2)
biz_df['log_review_count'].hist(ax=ax, bins=100)
ax.tick_params(labelsize=14)
ax.set_xlabel('log10(review_count))', fontsize=14)
ax.set_ylabel('Occurrence', fontsize=14)


## Train linear regression models to predict the average stars rating of a business,
## using the review_count feature with and without log transformation
## Compare the 10-fold cross validation score of the two models
m_orig = linear_model.LinearRegression()
scores_orig = cross_val_score(m_orig, biz_df[['review_count']], biz_df['stars'], cv=10)
m_log = linear_model.LinearRegression()
scores_log = cross_val_score(m_log, biz_df[['log_review_count']], biz_df['stars'], cv=10)
print("R-squared score without log transform: %0.5f (+/- %0.5f)" % (scores_orig.mean(), scores_orig.std() * 2))
print("R-squared score with log transform: %0.5f (+/- %0.5f)" % (scores_log.mean(), scores_log.std() * 2))


# # Log Transform on Online News Popularity Dataset
# 

df = pd.read_csv('data/UCI_Online_News_Popularity/OnlineNewsPopularity/OnlineNewsPopularity.csv', delimiter=', ')


df['log_n_tokens_content'] = np.log10(df['n_tokens_content'] + 1)


df


news_orig_model = linear_model.LinearRegression()
scores_orig = cross_val_score(news_orig_model, df[['n_tokens_content']], df['shares'], cv=10)

news_log_model = linear_model.LinearRegression()
scores_log = cross_val_score(news_log_model, df[['log_n_tokens_content']], df['shares'], cv=10)

print("R-squared score without log transform: %0.5f (+/- %0.5f)" % (scores_orig.mean(), scores_orig.std() * 2))

print("R-squared score with log transform: %0.5f (+/- %0.5f)" % (scores_log.mean(), scores_log.std() * 2))


# ## Plot the distribution of number of tokens with and without log transform
# 

plt.figure()
ax = plt.subplot(2,1,1)
df['n_tokens_content'].hist(ax=ax, bins=100)
ax.tick_params(labelsize=14)
ax.set_xlabel('Number of Words in Article', fontsize=14)
ax.set_ylabel('Number of Articles', fontsize=14)

ax = plt.subplot(2,1,2)
df['log_n_tokens_content'].hist(ax=ax, bins=100)
ax.tick_params(labelsize=14)
ax.set_xlabel('Log of Number of Words', fontsize=14)
ax.set_ylabel('Number of Articles', fontsize=14)


# ## Visualize the correlation between the input and the output
# 

plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.scatter(df['n_tokens_content'], df['shares'])
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Number of Words in Article', fontsize=14)
ax1.set_ylabel('Number of Shares', fontsize=14)

ax2 = plt.subplot(2,1,2)
ax2.scatter(df['log_n_tokens_content'], df['shares'])
ax2.tick_params(labelsize=14)
ax2.set_xlabel('Log of the Number of Words in Article', fontsize=14)
ax2.set_ylabel('Number of Shares', fontsize=14)


plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.scatter(biz_df['review_count'], biz_df['stars'])
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Review Count', fontsize=14)
ax1.set_ylabel('Average Star Rating', fontsize=14)

ax2 = plt.subplot(2,1,2)
ax2.scatter(biz_df['log_review_count'], biz_df['stars'])
ax2.tick_params(labelsize=14)
ax2.set_xlabel('Log of Review Count', fontsize=14)
ax2.set_ylabel('Average Star Rating', fontsize=14)





import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

from skimage.feature import hog
from skimage import data, color, exposure


image = color.rgb2gray(data.chelsea())


fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax2.set_adjustable('box-forced')


import numpy as np


gx = np.empty(image.shape, dtype=np.double)
gx[:, 0] = 0
gx[:, -1] = 0
gx[:, 1:-1] = image[:, :-2] - image[:, 2:]
gy = np.empty(image.shape, dtype=np.double)
gy[0, :] = 0
gy[-1, :] = 0
gy[1:-1, :] = image[:-2, :] - image[2:, :]


# ## fig, (ax1, ax2, ax3) = plt.subplots(1, 3, 
#                                     figsize=(12,8), 
#                                     sharex=True, 
#                                     sharey=True)
# 
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Original image')
# ax1.set_adjustable('box-forced')
# 
# ax2.axis('off')
# ax2.imshow(gx, cmap=plt.cm.gray)
# ax2.set_title('Horizontal gradients')
# ax2.set_adjustable('box-forced')
# 
# ax3.axis('off')
# ax3.imshow(gy, cmap=plt.cm.gray)
# ax3.set_title('Vertical gradients')
# ax3.set_adjustable('box-forced')
# 




import pandas as pd
import json


# Load the first 10 reviews
f = open('data/yelp/v6/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')
js = []
for i in range(10):
    js.append(json.loads(f.readline()))
f.close()
review_df = pd.DataFrame(js)
review_df.shape


# ### Using spacy: [Installation instructions for spacy](https://spacy.io/docs/usage/)
# 

import spacy


# model meta data
spacy.info('en')


# preload the language model
nlp = spacy.load('en')


# Keeping it in a pandas dataframe
doc_df = review_df['text'].apply(nlp)

type(doc_df)


type(doc_df[0])


doc_df[4]


# spacy gives you both fine grained (.pos_) + coarse grained (.tag_) parts of speech    
for doc in doc_df[4]:
    print(doc.text, doc.pos_, doc.tag_)


# spaCy also does noun chunking for us

print([chunk for chunk in doc_df[4].noun_chunks])


# ### Using [Textblob](https://textblob.readthedocs.io/en/dev/)
# 

from textblob import TextBlob


# The default tagger in TextBlob uses the PatternTagger, the same as [pattern](https://www.clips.uantwerpen.be/pattern), which is fine for our example. To use the NLTK tagger, we can specify the pos_tagger when we call TextBlob. More [here](http://textblob.readthedocs.io/en/dev/advanced_usage.html#advanced).
# 

blob_df = review_df['text'].apply(TextBlob)

type(blob_df)


type(blob_df[4])


blob_df[4].tags


# blobs in TextBlob also have noun phrase extraction

print([np for np in blob_df[4].noun_phrases])


