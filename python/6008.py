# # Custom Sources - URL
# 

import nltk
from urllib import urlopen


# We can bring in text directly from a url if the source itself is text.
# 

url = "http://www.gutenberg.org/cache/epub/26275/pg26275.txt"


odyssey_str = urlopen(url).read()


type(odyssey_str)


odyssey_str[3:49]


odyssey_tokens = nltk.word_tokenize(odyssey_str.decode('utf-8'))


len(odyssey_tokens)


odyssey_text = nltk.Text(odyssey_tokens)


print odyssey_text[:8]


# # Projects - Sentiment Analysis
# 

import nltk
import csv
import numpy as np


negative = []
with open("words_negative.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        negative.append(row)


positive = []
with open("words_positive.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        positive.append(row)


positive[:10]


negative[:10]


def sentiment(text):
    temp = [] #
    text_sent = nltk.sent_tokenize(text)
    for sentence in text_sent:
        n_count = 0
        p_count = 0
        sent_words = nltk.word_tokenize(sentence)
        for word in sent_words:
            for item in positive:
                if(word == item[0]):
                    p_count +=1
            for item in negative:
                if(word == item[0]):
                    n_count +=1

        if(p_count > 0 and n_count == 0): #any number of only positives (+) [case 1]
            #print "+ : " + sentence
            temp.append(1)
        elif(n_count%2 > 0): #odd number of negatives (-) [case2]
            #print "- : " + sentence
            temp.append(-1)
        elif(n_count%2 ==0 and n_count > 0): #even number of negatives (+) [case3]
            #print "+ : " + sentence
            temp.append(1)
        else:
            #print "? : " + sentence
            temp.append(0)
    return temp


sentiment("It was terribly bad.")


sentiment("Actualluty, it was not bad at all.")


sentiment("This is a sentance about nothing.")


mylist = sentiment("I saw this movie the other night. I can say I was not disappointed! The actiing and story line was amazing and kept me on the edge of my seat the entire time. While I did not care for the music, it did not take away from the overall experience. I would highly recommend this movie to anyone who enjoys thirllers.")


comments = []
with open("reviews.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        comments.append(row)


comments[0]


for review in comments:
    print "\n"
    print np.average(sentiment(str(review)))
    print review


# Is some of our projects we will be using fuctions to repetedly call parts of our code.
# 
# We can define a function by giving it a name, and telling the function any values we plan on passing along.
# 

def my_function(something):
    return something


my_function("hello")


my_function(2)


def information(word):
    return "Word: " + str(word) + ", Length: " + str(len(word))


information("hello")


information("language")





# # Custom Sources - CSV (Spreadsheet)
# 

import nltk
import csv


# Importing text stored in cells on a spreadsheet or in a csv file (comma separated values).
# 

comments = []
with open("reviews.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        comments.append(row)


comments[0]


# This command will give us a list, where each element is another list of all the tokens for each comment.
# 

tokens = [nltk.word_tokenize(str(entry)) for entry in comments]


tokens[0]


# # Tokenization, Tagging, Chunking - Part of Speech Tagging
# 

import nltk


# A part of speech tagger will identify the part of speech for a sequence of words.
# 

text = "I walked to the cafe to buy coffee before work."


tokens = nltk.word_tokenize(text)


nltk.pos_tag(tokens)


# Part of speech key.
# 

nltk.help.upenn_tagset()


nltk.pos_tag(nltk.word_tokenize("I will have desert."))


nltk.pos_tag(nltk.word_tokenize("They will desert us."))


# Create a list of all nouns.
# 

md = nltk.corpus.gutenberg.words("melville-moby_dick.txt")


md_norm = [word.lower() for word in md if word.isalpha()]


md_tags = nltk.pos_tag(md_norm,tagset="universal")


md_tags[:5]


md_nouns = [word for word in md_tags if word[1] == "NOUN"]


nouns_fd = nltk.FreqDist(md_nouns)


nouns_fd.most_common()[:10]  


# # NLTK and the Basics - Conditional Frequency Distribution
# 

import nltk


# A conditional frequency distribution counts multiple cases or conditions. 
# 

names = [("Group A", "Paul"),("Group A", "Mike"),("Group A", "Katy"),("Group B", "Amy"),("Group B", "Joe"),("Group B", "Amy")]


names


# Running a regular frequency distribution on the list...
# 

nltk.FreqDist(names)


# Running a conditional frequency distribution on the list...
# 

nltk.ConditionalFreqDist(names)


# # NLTK and the Basics - Regular Expressions
# 

import nltk
import re


alice = nltk.corpus.gutenberg.words("carroll-alice.txt")


# Finding every word that start with "new".
# 

set([word for word in alice if re.search("^new",word)])


# Finding every word that ends with "ful".
# 

set([word for word in alice if re.search("ful$",word)])


# Finding words that are six characters long and have two n's in the middle.
# 

set([word for word in alice if re.search("^..nn..$",word)])


# Finding words that start with "c", "h", and "t", and end in "at".
# 

set([word for word in alice if re.search("^[chr]at$",word)])


# Finding words of any length that have two n's.
# 

set([word for word in alice if re.search("^.*nn.*$",word)])


# # NLTK and the Basics - Bigrams
# 

import nltk


# Bigrams, sometimes called 2grams, or ngrams (when dealing with a different number), is a way of looking at word sequences.
# 

text = "I think it might rain today."


tokens = nltk.word_tokenize(text)


tokens


bigrams =  nltk.bigrams(tokens)


for item in bigrams:
    print item


trigrams = nltk.trigrams(tokens)


for item in trigrams:
    print item


from nltk.util import ngrams


text = "If it is nice out, I will go to the beach."


tokens = nltk.word_tokenize(text)


bigrams = ngrams(tokens,2)


for item in bigrams:
    print item


fourgrams = ngrams(tokens,4)


for item in fourgrams:
    print item


# We can build a function to find any ngram.
# 

def n_grams(text,n):
    tokens = nltk.word_tokenize(text)
    grams = ngrams(tokens,n)
    return grams


text = "I think it might rain today, but if it is nice out, I will go to the beach."


grams = n_grams(text, 5)


for item in grams:
    print item


# # NLTK and the Basics - Counting Text
# 

import nltk


# NLTK comes with pre-packed text data. 
# 
# Project Gutenberg is a group that digitizes books and literature that are mostly in the pubic domain.  These works make great examples for practicing NLP.  If you interested in Project Gutenberg, I recommend checking out their site. http://www.gutenberg.org/wiki/Main_Page
# 

nltk.corpus.gutenberg.fileids()


# You will notice that every file name has the letter "u" before it. The 'u' is part of the external representation of the file name, meaning it's a Unicode string as opposed to a byte string. It is not part of the string.
# 

md = nltk.corpus.gutenberg.words("melville-moby_dick.txt")


md[:8]


# We can count how many times a word appears in the book.
# 

md.count("whale")


md.count("boat")


md.count("Ahab")


md.count("laptop")


# We can get an idea of how long the book is by seeing how many items are in our list.
# 

len(md)


# We can see how many unique words are used in the book.
# 

md_set = set(md)


len(md_set)


# We can calculate the average number of times any given word is used in the book.
# 

from __future__ import division #we import this since we are using Python 2.7


len(md)/len(md_set)


# We can look at the book as a lists of sentences.
# 

md_sents = nltk.corpus.gutenberg.sents("melville-moby_dick.txt")


# We can calculate the average number of words per sentence in the book.
# 

len(md)/len(md_sents)


# # NLTK and the Basics - Frequency Distribution
# 

import nltk


alice = nltk.corpus.gutenberg.words("carroll-alice.txt")


# Frequency distribution takes a list of words and counts how many times each word is seen.
# 

alice_fd = nltk.FreqDist(alice)


alice_fd


alice_fd["Rabbit"]


# We can find the 15 most common words seen.
# 

alice_fd.most_common(15)


# A word used only once in a collection of text is called a hapax legomenon.
# 

alice_fd.hapaxes()[:15]


# # Python Refresher - Dictionaries
# 

# A Python dictionary stores key-value pairs.
# 

d = {
    'Python': 'programming', 
    'English': "natural", 
    'French': 'natrual', 
    'Ruby' : 'programming', 
    'Javascript' : 'programming'
}


d


type(d)


d["Python"]


d["English"]


# We can add new entries to a dictionary.
# 

d['Scala'] = 'programming'


d


# Values can also be numbers.
# 

d["languages known"] = 3


d


d.keys()


d.values()


len(d)


# # Custom Sources - Text File
# 

import nltk


di = open("dec_independence.txt")


di_text = di.read()


di_text


type(di_text)


nltk.word_tokenize(di_text)


di_token = nltk.word_tokenize(di_text)


nltk.FreqDist(di_token)


# # Projects - Gender Prediction
# 

import nltk
import random


from nltk.corpus import names


names.fileids()


import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')


cfd = nltk.ConditionalFreqDist((fileid,name[-2:]) for fileid in names.fileids() for name in names.words(fileid))


plt.figure(figsize=(50,10))
cfd.plot()


# The plot shows us that a decent numbner of ending letter pairs have tend to lean towords female or male names.  Let's use this as our feature to build a feature set.
# 

def name_feature(name):
    return {'pair': name[-2:]}


name_feature("Katy")


name_list = ([(name, 'male') for name in names.words('male.txt')] + [(name, "female") for name in names.words('female.txt')])


name_list[:10]


name_list[-10:]


random.shuffle(name_list)


name_list[:10]


features = [(name_feature(name), gender) for (name,gender) in name_list]


features[:10]


len(features)/2


training_set = features[:3972]
testing_set = features[3972:]


# We can use the Naive Bayes Classifier to train our model. https://en.wikipedia.org/wiki/Naive_Bayes_classifier
# 

classifier = nltk.NaiveBayesClassifier.train(training_set)


male_names = names.words('male.txt')
"Carmello" in male_names


classifier.classify(name_feature("Carmello"))


nltk.classify.accuracy(classifier, testing_set)


# # Custom Sources - Exporting
# 

import nltk


alice = nltk.corpus.gutenberg.words("carroll-alice.txt")


alice = alice[:1000]


# We have to untokenize the text.
# 

alice_str = ' '.join(alice)


alice_str


# We create a new file.
# 

new_file = open('export.txt', 'w')


new_file.write(alice_str)


new_file.close()


# # Python Refresher - Lists
# 

# A Python list stores comma separated values.  In our cases these values will be strings, and numbers.
# 

mylist = ["a","b","c"]


mylist


mylist2 = [1,2,3,4,5]
mylist2


# Each item in the list has a position or index.  By using a list index you can get back individual list item.
# 
# Remember that in programming, counting starts at 0, so to get the first item, we would call index 0.
# 

mylist[0]


mylist2[0]


# We can also use a range of indexes to call  back a range from out list.
# 

mylist2[0:2]


# The first number tells us where to start while the second tells us where to end and is exclusive.  If we don't enter the fist number, we will get back the first x items, where x is the second index number we provide.
# 

mylist[:2]


mylist2[:3]


# We can also call the ends of a list by doing the following.
# 

mylist[-2:]


# # Python Refresher - Loops and Conditionals
# 

mylist = ["Python","Ruby","Javascript","HTML"]


mylist


# With a for loop, we can iterate through this list.
# 

for item in mylist:
    print item


# We can also write a for loop this way...
# 

[item for item in mylist]


# We can use for loops to carry out actions.
# 

for item in mylist:
    print item + " is a fun programming language."


newlist = [item + " is a fun programming language." for item in mylist]


newlist


# We can use if statements to look for special conditions.
# 

x = 10


if x > 5:
    print "It looks like x is greater than 5"


if x > 5 and x < 20:
    print "Hello"


number_list = [1,2,3,4,5,6,7,8,9,10]


for number in number_list:
    if number%2 == 0:
        print str(number) + " is even"
    elif number == 7:
        print str(number) + " is the best number!"
    else:
        print str(number) + " is odd"


# # Custom Sources - HTML
# 

import nltk
from urllib import urlopen


# Websites are written in HTML, so when you pull information directly from a site, you will get all the code back along with the text.
# 

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"


html = urlopen(url).read()


html


# We will use a Python library called BeautifulSoup in order to strip away the HTML code.
# 

from bs4 import BeautifulSoup


web_str = BeautifulSoup(html).get_text()


web_tokens = nltk.word_tokenize(web_str)


web_tokens[0:25]


# With a little bit of manual work we can find the main body of text.
# 

start = web_str.find("Python is a widely used general-purpose, high-level programming language.")


# The end of the first section of the Wikipedia entry ends with "CPython is managed by the non-profit Python Software Foundation." 
# 

end = web_str.find("CPython is managed by the non-profit Python Software Foundation.")


last_sent = len("CPython is managed by the non-profit Python Software Foundation.")


intro = web_str[start:end+last_sent]


intro_tokens = nltk.word_tokenize(intro)


print intro_tokens


