get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
np.random.seed(45)
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


indicator = pd.read_csv('indicator1.csv')
indicator.head()


# Drop the code column and year column
# 

indicator.drop(indicator.columns[[0, 2]], axis=1, inplace=True)
indicator.head()


indicator.info()


# Use describe() method to show the summary statistics of numeric attributes.
# 

indicator.describe()


# The count, mean, min and max rows are self-explanatory. The std shows standard deviation. The 25%, 50% and 75% rows show the corresponding percentiles.
# 
# To get a feel of what type of the data we are dealing with, we plot a histogram for each numeric attribute.
# 

indicator.hist(bins=50, figsize=(20, 15))
plt.savefig('numeric_attributes.png')
plt.show()


# Observations: 
# 
# These attributes have very different scales, we will need to apply feature scaling.
# 
# Many histogram are right skewed. This may make it harder for some machine learning algorithms to detect patterns. We will need to transform them to more normal distributions.
# 

# check for correlation between attributes.
# 

from pandas.plotting import scatter_matrix

attributes = ["GDP_per_capita", "Hours_do_tax", "Days_reg_bus", "Cost_start_Bus",
              "Bus_tax_rate", "Ease_Bus"]
scatter_matrix(indicator[attributes], figsize=(12, 8))
plt.savefig("scatter_matrix_plot.png")
plt.show()


# It seems GDP per Capita has a negative correlation with Ease of doing business. The other attributes all have a positive correlation with Ease of doing business. Let's find the most promising attribute to predict the Ease of doing business.
# 

from sklearn.linear_model import LinearRegression
X = indicator.drop(['country', 'Ease_Bus'], axis=1)
regressor = LinearRegression()
regressor.fit(X, indicator.Ease_Bus)


print('Estimated intercept coefficient:', regressor.intercept_)


print('Number of coefficients:', len(regressor.coef_))


pd.DataFrame(list(zip(X.columns, regressor.coef_)), columns = ['features', 'est_coef'])


# The most promising attribute to predict the "ease of doing business" is the "days spent to register a business", so let’s zoom in on their correlation scatterplot.
# 

indicator.plot(kind="scatter", x="Days_reg_bus", y="Ease_Bus",
             alpha=0.8)
plt.savefig('scatter_plot.png')


# The correlation is indeed very strong; you can clearly see the upward trend and the points are not too dispersed.
# 

# Split the data into training and test
# 

from sklearn.cross_validation import train_test_split
y = indicator.Ease_Bus

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Build a Linear Regression model
# 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


regressor.score(X_test, y_test)


# So in our model, 60.7% of the variability in Y can be explained using X.
# 

from sklearn.metrics import mean_squared_error
regressor_mse = mean_squared_error(y_pred, y_test)

import math
math.sqrt(regressor_mse)


# So we are an average of 33.59 away from the ground true score on "ease of doing business" when making predictions on our test set.
# 
# The median score of "Ease of doing business" is 95, so a typical prediction error of 33.59 is not very satisfying. This is an example of a model underfitting the training data. When this happens it can mean that the features do not provide enough information to make good predictions, or that the model is not powerful enough. The main ways to fix underfitting are to select more features from Wordbank indicators(e.g., "getting credit", 'registering property" and so on). 
# 

regressor.predict([[41096.157300, 5.0, 3, 58.7, 161.0]])


indicator.loc[indicator['country'] == 'Belgium']


regressor.predict([[42157.927990, 0.4, 2, 21.0, 131.0]])


indicator.loc[indicator['country'] == 'Canada']


plt.scatter(regressor.predict(X_train), regressor.predict(X_train)-y_train, c='indianred', s=40)
plt.scatter(regressor.predict(X_test), regressor.predict(X_test)-y_test, c='b', s=40)
plt.hlines(y=0, xmin=0, xmax=200)
plt.title('Residual plot using training(red) and test(blue) data')
plt.ylabel('Residual')
plt.savefig('residual_plot.png')


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD

book = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
book.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
user = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
user.columns = ['userID', 'Location', 'Age']
rating = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
rating.columns = ['userID', 'ISBN', 'bookRating']


rating.head()


user.head()


book.head()


combine_book_rating = pd.merge(rating, book, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
combine_book_rating.head()


# ### Filter to only popular books
# 
# Remove rows where book title is missing
# 

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])


book_ratingCount = (combine_book_rating.
     groupby(by = ['bookTitle'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalRatingCount'})
     [['bookTitle', 'totalRatingCount']]
    )
book_ratingCount.head()


# #### Now we can merge the total rating count data into the rating data, giving us exactly what we need to filter out the lesser known books.
# 

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
rating_with_totalRatingCount.head()


pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_ratingCount['totalRatingCount'].describe())


# #### The median book has only been rated one time. Let’s take a look at the top of the distribution.
# 

print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))


# #### So about 1% of books have 50 ratings, 2% have 29 ratings. Since we have so many books in our data, we’ll limit it to the top 1%, this will give us 2713 different books. 
# 

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_book.head()


# #### Filtering to US users only
# 

combined = rating_popular_book.merge(user, left_on = 'userID', right_on = 'userID', how = 'left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
us_canada_user_rating.head()


if not us_canada_user_rating[us_canada_user_rating.duplicated(['userID', 'bookTitle'])].empty:
    initial_rows = us_canada_user_rating.shape[0]

    print('Initial dataframe shape {0}'.format(us_canada_user_rating.shape))
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    current_rows = us_canada_user_rating.shape[0]
    print('New dataframe shape {0}'.format(us_canada_user_rating.shape))
    print('Removed {0} rows'.format(initial_rows - current_rows))


us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)


from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)


query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


# Perfect! "Green Mile Series" books are definitely should be recommended one after another.   
# 

us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)


us_canada_user_rating_pivot2.head()


us_canada_user_rating_pivot2.shape


X = us_canada_user_rating_pivot2.values.T
X.shape


import sklearn
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
matrix.shape


import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)
corr.shape


us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)
coffey_hands = us_canada_book_list.index("The Green Mile: Coffey's Hands (Green Mile Series)")
print(coffey_hands)


corr_coffey_hands  = corr[coffey_hands]


list(us_canada_book_title[(corr_coffey_hands<1.0) & (corr_coffey_hands>0.9)])


# The results look great!
# 




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## The Data
# 
# The data consists of three tables: ratings, books info, and users info.
# 

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


# ### Ratings Data
# 
# The ratings data set provides a list of ratings that users have given to books. It includes 1,149,780 records and 3 fields: userID, ISBN, and rating.
# 

print(ratings.shape)
print(list(ratings.columns))


ratings.head()


# ### Ratings Distribution
# 

plt.rc("font", size=15)
ratings.bookRating.value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()


# ### Books dataset
# 
# This dataset provides books details. It includes 271360 records and 8 fields: ISBN, book title, book author, publisher and so on.
# 

print(books.shape)
print(list(books.columns))


books.head()


# ### Users dataset 
# 
# This dataset provides the user demographic information. It includes 278858 records and 3 fields: user id, location and age.
# 

print(users.shape)
print(list(users.columns))


users.head()


# ### Age distribution
# 
# The most active users are among 20-30s.
# 

users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('system2.png', bbox_inches='tight')
plt.show()


# ## Recommendations based on rating counts
# 

rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False).head()


# The book with ISBN "0971880107" received the most ratings. Let's find out which books are in the top 5.
# 

most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209', '0060928336', '0312195516'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
most_rated_books_summary


# The book that received the most ratings in this data set is Rich Shapero's Wild Animus. Something in common among these five most rated books - they are fictions or novels. The recommender suggests that novels and fictions are popular and likely receive more ratings. And if someone likes "Wild Animus", probably we should recommend him(her) "The Lovely Bones: A Novel".
# 

# ## Recommendations based on correlations
# 
# Find out the average rating and the number of ratings each book received.
# 

average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
average_rating.sort_values('ratingCount', ascending=False).head()


# #### Observation: 
# 
# In this dataet, the book that received the most ratings is not highly rated at all. So if we were set to use recommendations based on rating counts, we would definitely make mistaks here.
# 

# #### To ensure statistical significance, users with less than 200 ratings, and books with less than 100 ratings are excluded.
# 

counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]


# ### Rating matrix
# 
# Convert the table to a 2D matrix. The matrix will be sparse because not every user rate every book.
# 

ratings_pivot = ratings.pivot(index='userID', columns='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
print(ratings_pivot.shape)
ratings_pivot.head()


# Let's find out which books are correlated with the 2nd most rated book "The Lovely Bones: A Novel". To blatantly quote from the Wikipedia: It is the story of a teenage girl who, after being raped and murdered, watches from her personal Heaven as her family and friends struggle to move on with their lives while she comes to terms with her own death.
# 

bones_ratings = ratings_pivot['0316666343']
similar_to_bones = ratings_pivot.corrwith(bones_ratings)
corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR', ascending=False).head(10)


# We obtained the books' ISBNs, but we need to find out the names of the books to see whether they make sense.
# 

books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', '0345342968', '0060930535', '0375707972', '0684872153'], 
                                  index=np.arange(9), columns=['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on='ISBN')
corr_books


# Let's select three books to examine from the above highly correlated list "The Nanny Diaries: A Novel", "The Pilot's Wife: A Novel" and "Where the heart is". 
# 
# "The Nanny Diaries" satirizes upper class Manhattan society as seen through the eyes of their children's caregivers. 
# 
# Written by the same author of "The Lovely Bones", "The Pilot's Wife" is the third novel in Shreve's informal trilogy to be set in a large beach house on the New Hampshire coast that used to be a conventis.
# 
# "Where the Heart Is" dramatizes in detail the tribulations of lower-income and foster children in the United States.
# 
# These three books sound right to me to be highly correlated with "The Lovely Bones". Seems our correlation recommender system is working.
# 




import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("Online_Retail.xlsx")
df.head()


import seaborn as sns
sns.set_palette("husl")
sns.set(rc={'image.cmap': 'coolwarm'})
get_ipython().magic('matplotlib inline')


df.dtypes


import datetime as dt
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date


df = df[pd.notnull(df['CustomerID'])]


df = df[(df['Quantity']>0)]


df['Sales'] = df['Quantity'] * df['UnitPrice']
cols_of_interest = ['CustomerID', 'InvoiceDate', 'Sales']
df = df[cols_of_interest]


print(df.head())
print('Number of Entries: %s' % len(df))


# * frequency represents the number of repeat purchases the customer has made. This means that it’s one less than the total number of purchases.
# * T represents the age of the customer in whatever time units chosen (daily, in our dataset). This is equal to the duration between a customer’s first purchase and the end of the period under study.
# * recency represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer’s first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)
# 

from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes.estimation import *

data = summary_data_from_transaction_data(df, 'CustomerID', 'InvoiceDate', monetary_value_col='Sales', observation_period_end='2011-12-9')
data.head()


# ### Basic Frequency/Recency analysis using the BG/NBD model
# We’ll use the BG/NBD model first, because this is the simplest to start with.
# 

from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])
print(bgf)


# After fitting, we have lots of nice methods and properties attached to the fitter object.
# 
# For small samples sizes, the parameters can get implausibly large, so by adding an l2 penalty the likelihood, we can control how large these parameters can be. This is implemented as setting as positive penalizer_coef in the initialization of the model. In typical applications, penalizers on the order of 0.001 to 0.1 are effective.
# 

# ### Visualizing our Frequency/Recency Matrix
# Consider: a customer bought from us every day for three weeks straight, and we haven’t heard from them in months. What are the chances they are still “alive”? Pretty small. On the other hand, a customer who historically buys from us once a quarter, and bought last quarter, is likely still alive. We can visualize this relationship using the Frequency/Recency matrix, which computes the expected number of transactions a artificial customer is to make in the next time period, given his or her recency (age at last purchase) and frequency (the number of repeat transactions he or she has made).
# 

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)


# We can see that if a customer has bought 120 times from us, and their latest purchase was when they were 350 days old (given the individual is 350 days old), then they are our best customer (bottom-right). Customers who have purchased a lot and purchased recently will likely be the best customers in the future.
# 
# Customers who have purchased a lot but not recently (top-right corner), have probably dropped out.
# 

# There’s also that beautiful “tail” around (20,250). That represents the customer who buys infrequently, and we’ve not seen him or her very recently, so they might buy again - we’re not sure if they dropped out or just between purchases.
# 
# We can predict which customers are still alive:
# 

from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)


# Customers who have purchased recently are almost surely "alive". 
# 
# Customers who have purchased a lot but not recently, are likely to have dropped out. And the more they bought in the past, the more likely they have dropped out. They are represented in the upper-right.
# 
# The matrix lets us estimate a behavioral propensity that we can never observe if someone is alive or not. 
# 

# ### Ranking customers from best to worst
# 

# Let’s return to our customers and rank them from “highest expected purchases in the next period” to lowest. Models expose a method that will predict a customer’s expected purchases in the next period using their history.
# 

t = 1
data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, data['frequency'], data['recency'], data['T'])
data.sort_values(by='predicted_purchases').tail(5)


# We can see that the customer who has made 131 purchases, and bought very recently from us, has a probability of 29.8% to buy again in the next period (tomorrow).
# 

# ### Assessing model fit
# we can predict and we can visualize our customers’ behaviour, but is our model correct? There are a few ways to assess the model’s correctness. The first is to compare our data versus artificial data simulated with your fitted model’s parameters.
# 

from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)


# This plot tells us for all customers in our data, how many purchases they make in our observation period, and what does the model predict? Looks nice?
# 
# More importantly, it tells us that our model doesn’t suck. So, we can continue on with our analysis.

# ### More model fitting
# We can partition the dataset into a calibration period dataset and a holdout dataset. This is important as we want to test how our model performs on data not yet seen (think cross-validation in standard machine learning literature). Lifetimes has a function to partition our dataset like this:
# 

from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(df, 'CustomerID', 'InvoiceDate',
                                        calibration_period_end='2011-06-08',
                                        observation_period_end='2011-12-9' )   
print(summary_cal_holdout.head())


from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)


# ### Customer Predictions
# Based on customer history, we can predict what an individuals future purchases might look like:
# 

t = 10 #predict purchases in 10 periods
individual = data.iloc[20]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])


data.iloc[20]


# Customer ID 12370's predicted purchase is 0.008919 in 10 periods.
# 

# ### Customer Probability Histories
# Given a customer transaction history, we can calculate their historical probability of being alive, according to our trained model. For example:
# 

from lifetimes.plotting import plot_history_alive

id = 12347
days_since_birth = 365
sp_trans = df.loc[df['CustomerID'] == id]
plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate')


# ### Estimating customer lifetime value using the Gamma-Gamma model
# For this whole time we didn’t take into account the economic value of each transaction and we focused mainly on transactions’ occurrences. To estimate this we can use the Gamma-Gamma submodel. But first we need to create summary data from transactional data also containing economic values for each transaction (i.e. profits or revenues).
# 

returning_customers_summary = data[data['frequency']>0]

print(returning_customers_summary.head())


# ### The Gamma-Gamma model and the independence assumption
# The model we are going to use to estimate the CLV for our userbase is called the Gamma-Gamma submodel, which relies upon an important assumption. The Gamma-Gamma submodel, in fact, assumes that there is no relationship between the monetary value and the purchase frequency. In practice we need to check whether the Pearson correlation between the two vectors is close to 0 in order to use this model.
# 

returning_customers_summary[['monetary_value', 'frequency']].corr()


# At this point we can train our Gamma-Gamma submodel and predict the conditional, expected average lifetime value of our customers.
# 

from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])
print(ggf)


# We can now estimate the average transaction value:
# 

print(ggf.conditional_expected_average_profit(
        data['frequency'],
        data['monetary_value']
    ).head(10))


print("Expected conditional average profit: %s, Average profit: %s" % (
    ggf.conditional_expected_average_profit(
        data['frequency'],
        data['monetary_value']
    ).mean(),
    data[data['frequency']>0]['monetary_value'].mean()
))


# While for computing the total CLV using the [DCF method](https://en.wikipedia.org/wiki/Discounted_cash_flow) adjusting for cost of capital:
# 

# refit the BG model to the summary_with_money_value dataset
bgf.fit(data['frequency'], data['recency'], data['T'])

print(ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    data['frequency'],
    data['recency'],
    data['T'],
    data['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
).head(10))





