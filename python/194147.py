import tweepy
import codecs
from time import sleep


## fill in your Twitter credentials 
access_token = ''
access_token_secret = ''
consumer_key = ''
consumer_secret = ''

## Set up an instance of the REST API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


#stuff = api.user_timeline(screen_name = 'Inspire_Us', count = 100)

#for status in stuff:
#    print(status.text)


# # 1. Retweet the tweet containing specific words
# 

for tweet in tweepy.Cursor(api.search, q='#motivation').items():
    try:
        print('\nTweet by: @' + tweet.user.screen_name)

        tweet.retweet()
        print('Retweeted the tweet')

        sleep(3600)

    except tweepy.TweepError as e:
        print(e.reason)

    except StopIteration:
        break


# # 2. Tweet on your timeline
# 

api.update_status(status="This is a sample tweet using Tweepy with python")


# # 3. Add to favourites and follow the user
# 

for tweet in tweepy.Cursor(api.search, q='#motivation').items():
    try:
        print('\nTweet by: @' + tweet.user.screen_name)

        # Favorite the tweet
        tweet.favorite()
        print('Favorited the tweet')

        # Follow the user who tweeted
        tweet.user.follow()
        print('Followed the user')

        #sleep(5)
        sleep(3600)

    except tweepy.TweepError as e:
        print(e.reason)

    except StopIteration:
        break       
        


# # Basic Chatbot using ChatterBot
# 

import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer


#Let's give a name to your chat bot
chatbot = ChatBot('Veronica')

#Train the bot
chatbot.set_trainer(ListTrainer)
chatbot.train(['What is your name?', 'My name is Veronica'])


#let's test the bot
chatbot.get_response('What is your name?')


#Let's train the bot with more data
conversations = [
     'Are you an artist?', 'No, are you mad? I am a bot',
     'Do you like big bang theory?', 'Bazinga!',
     'What is my name?', 'Natasha',
     'What color is the sky?', 'Blue, stop asking me stupid questions'
]

chatbot.train(conversations)


chatbot.get_response('Do you like big bang theory?')


#Let's use chatterbot corpus to train the bot
from chatterbot.trainers import ChatterBotCorpusTrainer
chatbot.set_trainer(ChatterBotCorpusTrainer)
chatbot.train("chatterbot.corpus.english")


chatbot.get_response('Who is the President of America?')


chatbot.get_response('What language do you speak?')


chatbot.get_response('Hi')


# The bot needs some intelligence. 
# 

import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#Run the below piece of code for the first time
#nltk.download('stopwords')


message_data = pd.read_csv("spam.csv",encoding = "latin")
message_data.head()


message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)


message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})


message_data.groupby('Spam/Not_Spam').describe()


message_data_copy = message_data['message'].copy()


def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)


message_data_copy = message_data_copy.apply(text_preprocess)


message_data_copy


vectorizer = TfidfVectorizer("english")


message_mat = vectorizer.fit_transform(message_data_copy)
message_mat


message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat, 
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)


# Let's try using stemming and normalizing length of the messages
# 

def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words


message_data_copy = message_data_copy.apply(stemmer)
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)


message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat, 
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)


# Accuracy score improved. Let's try normalizing length.
# 

message_data['length'] = message_data['message'].apply(len)
message_data.head()


length = message_data['length'].as_matrix()
new_mat = np.hstack((message_mat.todense(),length[:, None]))


message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(new_mat, 
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)


import pandas as pd
import numpy as np


ted_data = pd.read_csv("ted_main.csv")
ted_data.head(3)


# Let's have a look how many values are missing.
ted_data.isnull().sum()


#Lets have a look at the data and see identify Object/Categorial values and Continuous values
ted_data.dtypes


# Looking at the snapshot of the data, the name column contains the 
# title of ted talks too. There are already seperate columns for name(main speaker) and title. 
# We can drop this extra column.
# 

#Drop the name column
ted_data = ted_data.drop(['name'], axis = 1)
ted_data.columns


# Looking at the data, the dates are in Unix timestamp format. Let get that formatted too in datetime format.
# 

from datetime import datetime
def convert(x):
    return pd.to_datetime(x,unit='s')


ted_data['film_date'] = ted_data['film_date'].apply(convert)
ted_data['published_date'] = ted_data['published_date'].apply(convert)
ted_data.head()


# Some columns contain data in dictionary and list format. We will not look at those column yet.
# 

#Lets see who talked a lot - top 20
import seaborn as sns
ax = sns.barplot(x="duration", y="main_speaker", data=ted_data.sort_values('duration', ascending=False)[:20])


#Let's see which video got the most views
ax = sns.barplot(x="views", y="main_speaker", data=ted_data.sort_values('views', ascending=False)[:20])


# I can see the some imilar names in both the plots. Looking at both the plots it seems,
# the long talks got more views? Let's try and verify it.
# 

#let's see the distribution of views
sns.distplot(ted_data[ted_data['views'] < 0.4e7]['views'])


#let's see the distribution of duration
sns.distplot(ted_data[ted_data['duration'] < 0.4e7]['duration'])


ax = sns.jointplot(x='views', y='duration', data=ted_data)


# Seems like we were wrong. There is no relationship with the length and duration.
# 

#Lets see the ditribution of comments.
sns.distplot(ted_data[ted_data['comments'] < 500]['comments'])


# Do you think most viewed videos/popular videos will have more comments?

sns.jointplot(x='views', y='comments', data=ted_data)


# Seems like we have found a relationship here.
# 

# Let's see if our top speakers have got more discussion/ comments?

ted_data[['title', 'main_speaker','views', 'comments', 'duration']].sort_values('views', ascending=False).head(20)


# Looking at the data above, Richard Dawkins has not got many views but has
# got a lot comments looks like his topic was controversial.
# 

# Now lets try and find out when ted talk were filmed the most
# 

ted_data.head(1)


talk_month = pd.DataFrame(ted_data['film_date'].map(lambda x: x.month).value_counts()).reset_index()
talk_month.columns = ['month', 'talks']
talk_month.head()


sns.barplot(x='month', y='talks', data=talk_month)


#Import necessary libraries
import plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np


#create dataframe from csv 
breast_cancer_dataframe = pd.read_csv('data.csv')

#get features information
#breast_cancer_dataframe.info()


#extract a few sample data to have a look
breast_cancer_dataframe.head()


#data cleaning step - remove the columns or rows with missing values and the ID as it doesn't have any relevance in anaysis
breast_cancer_df = breast_cancer_dataframe.drop(['id','Unnamed: 32'], axis = 1)


#dropping the column called diagnosis and having a columns of 0 and 1 instead --> 1 for M(Malignant) and 0 for B(Benign)
breast_cancer_df= pd.get_dummies(breast_cancer_df,'diagnosis',drop_first=True) 

#check if new column is added and contains 0 and 1
breast_cancer_df.head()


#First Plotly chart - Bar chart to see the count of Malignant and Benign in our data

#create data to feed into the plot - x-axis will hold the name of diagnosis
#and y axis will have the counts according the number of matches found in diagnosis column in our dataframe

color = ['red','green']
data = [go.Bar(x=['Malignant','Benign'],
y=[breast_cancer_dataframe.loc[breast_cancer_dataframe['diagnosis']=='M'].shape[0],
   breast_cancer_dataframe.loc[breast_cancer_dataframe['diagnosis']=='B'].shape[0]],
   marker=dict(color=color) 

)]

#create the layout of the chart by defining titles for chart, x-axis and y-axis
layout = go.Layout(title='Breast Cancer - Diagnosis',
xaxis=dict(title='Diagnosis'),
yaxis=dict(title='Number of people')
)

#Imbed data and layout into charts figure using Figure function
fig = go.Figure(data=data, layout=layout)

#Use plot function of plotly to visualize the data
py.offline.plot(fig)


#breast_cancer_df.std()


'''
data = [go.Bar(x=breast_cancer_dataframe['radius_mean'],
y= breast_cancer_dataframe['texture_mean'])]

layout = go.Layout(title='Radius Mean v/s Texture Mean',
xaxis=dict(title='Radius Mean '),
yaxis=dict(title='Texture Mean')
)

fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig)
'''


#Heatmap - to visualize the correlation between features/factors given in the dataset

#calculate the pairwise correlation of columns - Pearson correlation coefficient. 
z = breast_cancer_df.corr()

#use Heatmap function available in plotly and create trace(collection of data) for plot
trace = go.Heatmap(
            x=z.index,       #set x as the feature id/name
            y=z.index,       #set y as the feature id/name
            z=z.values,      #set z as the correlation matrix values, 
                             #these values will be used to show the coloring on heatmap,
                             #which will eventually define which coefficient has more impact or are closly related
            colorscale='Viridis', #colorscale to define different colors for different range of values in correlation matrix
    )

#set the title of the plot
title = "plotting the correlation matrix of the breast cancer dataset"

##create the layout of the chart by defining title for chart, height and width of it
layout = go.Layout(
    title=title,          # set plot title
    autosize=False,       # turn off autosize 
    height=800,           # plot's height in pixels 
    width=800             # plot's height in pixels 
)

#covert the trace into list object before passing thru Figure function
data = go.Data([trace])

#Imbed data and layout into plot using Figure function
fig = go.Figure(data=data, layout=layout)

#Use plot function of plotly to visualize the data
py.offline.plot(fig)


# Plotting Bar Chart and Heatmap using Seaborn library. Seaborn code is a bit less than what we need to code for Plotly but the plot can be more interactive using plotly.
# 

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

sns.countplot(x='diagnosis',data = breast_cancer_dataframe,palette='BrBG')


plt.figure(figsize= (10,10), dpi=100)
sns.heatmap(breast_cancer_df.corr())





