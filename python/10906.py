# # Topic Model Parts of Speech
# 
# This is a notebook for trying to use topic models for classifying sets of text that are more syntactically similar than topically similar. This notebook attempts to distinguish between discussion and conclusion section of scientific papers.
# 
# Below we are loading the dataset for use.
# 

from __future__ import print_function
from time import time
import os
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np

import pickle

validDocsDict = dict()
fileList = os.listdir("PubMedPOS")
for f in fileList:
    validDocsDict.update(pickle.load(open("PubMedPOS/" + f, "rb")))


# Here we are setting some vaiables to be used below and defining a function for printing the top words in a topic for the topic modeling.
# 

n_samples = len(validDocsDict.keys())
n_features = 1000
n_topics = 2
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# # Pre-process data
# 
# Here we are preprocessing data for use later. This code only grabs the discussion and conclusion sections of the data. We are also creating appropriate labels for the data and spliting the documents up to train and test sets.
# 

print("Loading dataset...")
t0 = time()
documents = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0

for k in validDocsDict.keys():
    if k.startswith("conclusion"):
        labels.append("conclusion")
        documents.append(validDocsDict[k])
        concCount += 1
        concLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("discussion"):
        labels.append("discussion")
        documents.append(validDocsDict[k])
        discCount += 1
        discLengthTotal += len(validDocsDict[k].split(' '))

print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)

train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.6)


# Here we are splitting the data up some more to train different models. Discussion and conclusion sections are being put into their own training sets. A TFIDF vectorizer is trained with the whole dataset of conclusion AND discussion sections. The multiple different training sets are then transformed using this vectorizer to get vector encodings of the text normalized to sum to 1 which accounts for differing lengths of conclusion and discussion sections.
# 

trainSetOne = []
trainSetTwo = []

for x in range(len(train)):
    if labelsTrain[x] == "conclusion":
        trainSetOne.append(train[x])
    else:
        trainSetTwo.append(train[x])

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, ngram_range = (1,4))
t0 = time()
tf = tf_vectorizer.fit_transform(train)

tfSetOne = tf_vectorizer.transform(trainSetOne)
tfSetTwo = tf_vectorizer.transform(trainSetTwo)
tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf
trainSetOne = tfSetOne
trainSetTwo = tfSetTwo

print("done in %0.3fs." % (time() - t0))


# # LDA With Two Topics
# 
# Define an LDA topic model on the whole data set with two topics. This is trying to see if the topic model can define the difference between the two groups automatically and prints the top words per topic.
# 

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# Transform the unknown data through the topic model and calculate which topic it is more associated with according to the ratios. Calculate how many of each type (conclusion and discussion) go into each topic (1 or 2).
# 

results = lda.transform(test)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    print(str(labelsTest[x]) + " " + str(val1/total) + " " + str(val2/total))
    if val1 > val2:
        if labelsTest[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labelsTest[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1


# Print out the results from the topic transforms.
# 

print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))


# Get the parameters for the LDA.
# 

lda.get_params()


# # Basic Classifiers
# 
# Train three basic classifiers to solve the problem. Try Gaussian, Bernoulli and K Nearest Neighbors classifiers and calculate how accurate they are.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())
numRight = 0
numWrongDisc = 0
numWrongConc = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1
    else:
        if classResults[item] == "discussion":
            numWrongDisc += 1
        else:
            numWrongConc += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))
print("Incorrectly classified as discussion: " + str(numWrongDisc))
print("Incorrectly classified as conclusion: " + str(numWrongConc))
print(len(classResults))


# # Two Topic Models
# 
# Define two topic models with 20 topics each, one on discussion sections and one on conclusion sections. Then transform both the train and test sets using both topic models to get 40 features for each sample based on the probability distribution for each topic in each LDA.
# 

ldaSet1 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
ldaSet2 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)


ldaSet1.fit(trainSetOne)
print_top_words(ldaSet1, tf_feature_names, n_top_words)


ldaSet2.fit(trainSetTwo)
print_top_words(ldaSet2, tf_feature_names, n_top_words)


results1 = ldaSet1.transform(train)
results2 = ldaSet2.transform(train)

resultsTest1 = ldaSet1.transform(test)
resultsTest2 = ldaSet2.transform(test)


results = np.hstack((results1, results2))
resultsTest = np.hstack((resultsTest1, resultsTest2))


# Define two classifiers using the transformed train and test sets from the topic models. Print out the accuracy of each one.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# Normalize the results of each sample of 40 features so they sum to 1. Then train two more classifiers using the data and print out the accuracy of each.
# 

for x in range(len(results)):
    total = 0
    for y in range(len(results[x])):
        total += results[x][y]
    for y in range(len(results[x])):
        results[x][y] = results[x][y]/total
        
for x in range(len(resultsTest)):
    total = 0
    for y in range(len(resultsTest[x])):
        total += resultsTest[x][y]
    for y in range(len(resultsTest[x])):
        resultsTest[x][y] = resultsTest[x][y]/total


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# # Topic Model Two Datasets Memory Efficient Fuzzy
# 
# This is a notebook for trying to use topic models for classifying sets of text that are more syntactically similar than topically similar. This notebook attempts to distinguish between discussion and conclusion section of scientific papers. This modifies the sections with random words from the introduction sections. It also reads the second dataset in a more memory efficient way.
# 
# Below we are loading the two datasets for use.
# 

from __future__ import print_function
from time import time
from random import randint

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np
import os
import pickle

validDocsDict = dict()
fileList = os.listdir("BioMedProcessed")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedProcessed/" + f, "rb")))


# Here we are setting some vaiables to be used below and defining a function for printing the top words in a topic for the topic modeling.
# 

n_samples = len(validDocsDict.keys())
n_features = 10000
n_topics = 2
n_top_words = 30
lengthOfIntroToAdd = 700

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# # Preprocess Data
# 
# Here we are preprocessing data for use later. This code only grabs the discussion and conclusion sections of the data. We are also creating appropriate labels for the data and spliting the documents up to train and test sets. We do this for both sets of data and then for a combined set of data.
# 

print("Loading dataset...")
t0 = time()
documents = []
introductionSections = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0
introCount = 0

for k in validDocsDict.keys():
    if k.startswith("conclusion"):
        labels.append("conclusion")
        documents.append(validDocsDict[k])
        concCount += 1
        concLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("discussion"):
        labels.append("discussion")
        documents.append(validDocsDict[k])
        discCount += 1
        discLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("introduction") and len(validDocsDict[k]) > 10000:
        introCount += 1
        introductionSections.append(validDocsDict[k])

print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)
print(introCount)


# Here we are reading in the files of the second dataset and only keeping the important sections. We are reading the files in one file at a time to be more memory efficient. Also, note that because the PubMed dataset is much larger, we are only reading in a third of the files.
# 

validDocs2 = []
labels2 = []
fileList = os.listdir("PubMedProcessed")
for f in fileList[0:len(fileList)/3]:
    tempDict = pickle.load(open("PubMedProcessed/" + f, "rb"))
    for item in tempDict.keys():
        if item.startswith("conclusion"):
            labels2.append("conclusion")
            validDocs2.append(tempDict[item])
        elif item.startswith("discussion"):
            labels2.append("discussion")
            validDocs2.append(tempDict[item])
        elif item.startswith("introduction") and len(tempDict[item]) > 10000:
            introCount += 1
            introductionSections.append(tempDict[item])

print(len(validDocs2))
print(introCount)


# Here we are adding random introduction words to the conclusion and discussion sections to replicate noise. Because the sections are tfidf vectorized, it is not important where in the section they are inserted.
# 

for item in range(len(documents)):
    intro = introductionSections[randint(0, len(introductionSections) - 1)].split(" ")
    randNum = randint(0, len(intro) - lengthOfIntroToAdd)
    introWords = intro[randNum:randNum + lengthOfIntroToAdd]
    documents[item] = documents[item] + " ".join(introWords)

for item in range(len(validDocs2)):
    intro = introductionSections[randint(0, len(introductionSections) - 1)].split(" ")
    randNum = randint(0, len(intro) - lengthOfIntroToAdd)
    introWords = intro[randNum:randNum + lengthOfIntroToAdd]
    validDocs2[item] = validDocs2[item] + " ".join(introWords)
    
train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.1)


# Here we are splitting the data up some more to train different models. Discussion and conclusion sections are being put into their own training sets. A TFIDF vectorizer is trained with the whole dataset of conclusion AND discussion sections from both data sets. The multiple different training sets are then transformed using this vectorizer to get vector encodings of the text normalized to sum to 1 which accounts for differing lengths of conclusion and discussion sections and between data sets.
# 

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, stop_words='english')
t0 = time()
tf_vectorizer.fit(train)
tf = tf_vectorizer.transform(train)

tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf

pubTest = tf_vectorizer.transform(validDocs2)

print("done in %0.3fs." % (time() - t0))


# Create a simple two topic LDA on the data and see how that splits the Conclusion and Discussion sections. Then, transform the second dataset using that LDA and see how that splits the Conclusion and Discussion sections.
# 

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

results = lda.transform(test)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    if val1 > val2:
        if labelsTest[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labelsTest[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1
print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))


results = lda.transform(pubTest)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    if val1 > val2:
        if labels2[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labels2[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1
print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))


# # Basic Classifiers Between Two Datasets
# 
# Train and test two Bernoulli classifiers (one where dataset 1 is trained and one where dataset 2 is trained) and print out the results of accuracy.
# 

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(pubTest.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labels2[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(pubTest.toarray(), labels2)

classResults = classifier.predict(train.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


probas = classifier.predict_log_proba(train.toarray())


TotalRight = 0
TotalWrong = 0
numRight = 0
numWrong = 0
RightNumbers = []
WrongNumbers = []
for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        TotalRight += probas[item][0] + probas[item][1]
        numRight += 1
        RightNumbers.append(probas[item][0] + probas[item][1])
    else:
        TotalWrong += probas[item][0] + probas[item][1]
        numWrong += 1
        WrongNumbers.append(probas[item][0] + probas[item][1])

print(str(TotalRight * 1.0 / numRight))
print(str(TotalWrong * 1.0 / numWrong))


# # Decision Trees Between Two Datasets
# Same concept as above, trying with decision trees instead of Bernoulli
# 

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(pubTest.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labels2[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(pubTest.toarray(), labels2)

classResults = classifier.predict(train.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


probas = classifier.predict_log_proba(train.toarray())


TotalRight = 0
TotalWrong = 0
numRight = 0
numWrong = 0
RightNumbers = []
WrongNumbers = []
for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        TotalRight += probas[item][0] + probas[item][1]
        numRight += 1
        RightNumbers.append(probas[item][0] + probas[item][1])
    else:
        TotalWrong += probas[item][0] + probas[item][1]
        numWrong += 1
        WrongNumbers.append(probas[item][0] + probas[item][1])

print(str(TotalRight * 1.0 / numRight))
print(str(TotalWrong * 1.0 / numWrong))





# # Data Processing for Topic Model Test including Parts of Speech
# 
# Getting the data from the repository...don't run unless you don't have the data!
# 
# !apt-get -y install curl
# 
# !curl -o BioMedSent/BioMedSentences.tar.zip http://i.stanford.edu/hazy/opendata/bmc/bmc_full_dddb_20150927_9651bf4a468cefcea30911050c2ca6db.tar.bzip2
# 
# http://i.stanford.edu/hazy/opendata/pmc/pmc_dddb_full_20150927_3b20db570e2cb90ab81c5c6f63babc91.tar.bzip2
# 

# # Import Data
# 
# This section defines the Sentence object used when importing and saving the data. Grab the files in a directory and process a subset of them.
# 

#Import Statements
import string
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

#Sentence object definition for data import and processing 
class Sentence:
    def __init__(self, document, sentenceNumber, wordList, lemmaList, posList):
        self.document = document
        self.sentenceNumber = sentenceNumber
        self.wordList = wordList
        self.lemmaList = lemmaList
        self.posList = posList
        self.sentence = " ".join([word for word in wordList if word not in string.punctuation])
        self.lemmaSent = " ".join([word for word in lemmaList if word not in string.punctuation])

#Get the files we want to process and put them into a list of lists called sentList
fileList = os.listdir("../PubMed/pmc_dddb_full")
sentList = []
fileList.sort()
for n in range(35, 36):
    f = open("../PubMed/pmc_dddb_full/" + fileList[n], 'r')
    for line in f:
        sentList.append(line.split('\t'))

len(sentList)


# 
# Now that we have all of the sentences in a list of lists grab the first element of each sentence list (the document id) and add that to a docList. Make this docList a set so we have the number of unique documents.
# 

docList = []
for thing in sentList:
    docList.append(thing[0])

len(set(docList))


# # Process Data
# 
# Define the processSent function for use by the multiprocessing part of the code. This function takes off some of the structure of parts of the data (removing the {,}, and ") and defines the Sentence object with all the appropriate parts.
# 
# We then use 14 cores (if available) for the Pool object and apply the processSent function to every sentence.
# 

sentObjList = []
def processSent(item):
    wordList = item[3].replace('"',"").lstrip("{").rstrip("}").split(",")
    wordList = filter(None, wordList)
    posList = item[4].split(",")
    lemmaList = item[6].replace('"',"").lstrip("{").rstrip("}").split(",")
    lemmaList = filter(None, lemmaList)
    return Sentence(item[0], item[1], wordList, lemmaList, posList)

po = Pool(16)
results = [po.apply_async(processSent, args = (sent,)) for sent in sentList]
po.close()
po.join()
output = [p.get() for p in results]
sentObjList = output
sentObjList[7].lemmaSent


# 
# Now that the sentences are processed, we need to find which sections these sentences should be atributed. For most of these papers, section headers are one word sentences. We are looking for common section headers and saving the sentence numbers for that section in that document.
# 

headingsDict = defaultdict(dict)

for sent in sentObjList:
    if len(sent.wordList) == 1:
        #print(sent.wordList)
        word = string.upper(sent.wordList[0]).strip()
        if word == 'INTRODUCTION' or word == 'BACKGROUND':
            headingsDict[sent.document]["introduction"] = sent.sentenceNumber
        elif word == 'METHODS':
            headingsDict[sent.document]["methods"] = sent.sentenceNumber
        elif word == 'RESULTS':
            headingsDict[sent.document]["results"] = sent.sentenceNumber
        elif word == 'DISCUSSION':
            headingsDict[sent.document]["discussion"] = sent.sentenceNumber
        elif word == 'CONCLUSION':
            headingsDict[sent.document]["conclusion"] = sent.sentenceNumber
        elif word == 'REFERENCES':
            headingsDict[sent.document]["references"] = sent.sentenceNumber
        

headingsDict.keys()


# 
# Now the sentences need to be tagged to their appropriate section and concatenated into one string per section per document.
# 
# The sentences are assigned a section by whichever section they are closest to (that is less than their sentence number). For example, if introduction had sentence number 5 and methods had sentence number 25, sentence number 20 would be assigned to introduction.
# 
# This is done for each sentence in each document and joined by spaces into a one string per section per document. Finally, only the documents that contain an introduction, discussion, and conclusion are kept and put into the validDocsDict dictionary
# 

documentDict = defaultdict(list)
docPartsDict = defaultdict(lambda : defaultdict(list))
docPartsCombinedDict = defaultdict(dict)

for item in sentObjList:
    documentDict[item.document].append(item)
    
for document in documentDict.keys():
    docSentList = documentDict[document]
    introNum = int(headingsDict[document].get("introduction", -1))
    methoNum = int(headingsDict[document].get("methods", -1))
    resultNum = int(headingsDict[document].get("results", -1))
    discussNum = int(headingsDict[document].get("discussion", -1))
    conclusionNum = int(headingsDict[document].get("conclusion", -1))
    refNum = int(headingsDict[document].get("references", -1))

    for sent in docSentList:
        label = "noSection"
        dist = int(sent.sentenceNumber)
        sentNumber = int(sent.sentenceNumber)
        
        if dist > sentNumber - introNum and sentNumber - introNum > 0:
            label = "introduction"
            dist = sentNumber - introNum
        if dist > sentNumber - methoNum and sentNumber - methoNum > 0:
            label = "methods"
            dist = sentNumber - methoNum
        if dist > sentNumber - resultNum and sentNumber - resultNum > 0:
            label = "results"
            dist = sentNumber - resultNum
        if dist > sentNumber - discussNum and sentNumber - discussNum > 0:
            label = "discussion"
            dist = sentNumber - discussNum
        if dist > sentNumber - conclusionNum and sentNumber - conclusionNum > 0:
            label = "conclusion"
            dist = sentNumber - conclusionNum
        if dist > sentNumber - refNum and sentNumber - refNum > 0:
            label = "references"
            dist = sentNumber - refNum
        if sent.sentence.strip().lower() not in ["introduction", "methods", "results", "discussion", "conclusion", "references"]:
            docPartsDict[document][label].append(sent)
    
    for x in docPartsDict[document].keys():
        docPartsCombinedDict[document][x] = " ".join(" ".join(y.posList) for y in sorted(docPartsDict[document][x], key=lambda z: z.sentenceNumber))

validDocsDict = defaultdict(dict)

for doc in docPartsCombinedDict.keys():
    tempKeys = docPartsCombinedDict[doc].keys()
    if 'introduction' in tempKeys and 'discussion' in tempKeys and 'conclusion' in tempKeys:
        validDocsDict[doc] = docPartsCombinedDict[doc]

print(str(len(docPartsCombinedDict.keys())))
print(str(len(validDocsDict.keys())))


# 
# Take the valid documents in the validDocsDict and output to a pickle file with the key part_docid with the part being introduction, methods, etc. and the docid allowing for document tracking.
# 

outputDict = dict()
for doc in validDocsDict.keys():
    for part in validDocsDict[doc].keys():
        outputDict[part + "_" + doc] = validDocsDict[doc][part]

pickle.dump(outputDict, open("PubmedPos35.p", "wb"))


# # Neural Network Test
# 
# This script attempts to create a neural network to solve the problem of classifying a document as part of discussion or conclusion section in scientific papers. This attempts to replicate work done in the paper "Character-level Convolutional Networks for Text Classification" by Zhang. 
# 
# Here we input our libraries and set some basic parameters.
# 

import keras
import theano
from __future__ import print_function
from time import time

import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split

import numpy as np
import os
import pickle

BATCH_SIZE = 16
FIELD_SIZE = 5 * 300
STRIDE = 1
N_FILTERS = 200


# Define a function that will turn a string into a list of Ascii numbers based on each character in the string.
# 

def vectorizeData(text):
    textList = list(text)
    returnList = []
    for item in textList[:1014]:
        returnList.append(ord(item))
    return returnList


# # Preprocess Data
# 
# Import the dataset.
# 

validDocsDict = dict()
fileList = os.listdir("BioMedProcessed")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedProcessed/" + f, "rb")))

#validDocsDict2 = dict()
#fileList = os.listdir("PubMedProcessed")
#for f in fileList:
#    validDocsDict2.update(pickle.load(open("PubMedProcessed/" + f, "rb")))


# Define some parameters for use later. This was developed to handle multiple datasets. Take the conclusion and discussion sections that are at least charLength number of characters. Vectorize that data and put them in the documents list. Then split the data up into multiple different train/test sets. 
# 

print("Loading dataset...")
t0 = time()
documents = []
testPubDocuments = []
allDocuments = []
labels = []
testPubLabels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0
charLength = 1014
charList = []

#combinedDicts = validDocsDict.copy()
#combinedDicts.update(validDocsDict2.copy())

for k in validDocsDict.keys():
    if k.startswith("conclusion") and len(validDocsDict[k]) >= charLength:
        labels.append(0)
        documents.append(vectorizeData(validDocsDict[k]))
        charList.extend(vectorizeData(validDocsDict[k]))
        concCount += 1
        concLengthTotal += len(validDocsDict[k])
    elif k.startswith("discussion") and len(validDocsDict[k]) >= charLength:
        labels.append(1)
        documents.append(vectorizeData(validDocsDict[k]))
        charList.extend(vectorizeData(validDocsDict[k]))
        discCount += 1
        discLengthTotal += len(validDocsDict[k])

charList = set(charList)
        
#for k in validDocsDict2.keys():
#    if k.startswith("conclusion"):
#        testPubLabels.append("conclusion")
#        testPubDocuments.append(vectorizeData(validDocsDict2[k]))
#        concCount += 1
#        concLengthTotal += len(validDocsDict2[k])
#    elif k.startswith("discussion"):
#        testPubLabels.append("discussion")
#        testPubDocuments.append(vectorizeData(validDocsDict2[k]))
#        discCount += 1
#        discLengthTotal += len(validDocsDict2[k])
        
#for k in combinedDicts.keys():
#    if k.startswith("conclusion"):
#        allDocuments.append(vectorizeData(combinedDicts[k]))
#    elif k.startswith("discussion"):
#        allDocuments.append(vectorizeData(combinedDicts[k]))
        
print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)


train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.95)
test1, test2, labelsTest1, labelsTest2 = train_test_split(test, labelsTest, test_size = 0.9)
print(len(train))
print(len(labelsTrain))


# Get an identity matrix from the length of the charList set (to know how many features we have in the set). This identity matrix is used in the one-hot encodding of the characters. For each character in the charList set, we assign a different row of the identity matrix. Then we create X_train and X_test sets using this mapping to convert character ascii numbers to one-hot encoddings. We also create Y_train which maps each section (discussion or conclusion) to a length two one-hot encodded vector.
# 

npVecs = np.eye(len(charList))
numToVec = dict()
labelsToVec = dict()
labelsToVec[0] = np.array([1,0])
labelsToVec[1] = np.array([0,1])
counter = 0
for item in charList:
    numToVec[item] = npVecs[counter]
    counter += 1
X_train = np.array([np.array([numToVec[x[y]] for y in x]) for x in train])
Y_train = np.array([np.array(labelsToVec[x]) for x in labelsTrain])
X_test = np.array([np.array([numToVec[x[y]] for y in x]) for x in test1])


X_train.shape
#X_train = np.expand_dims(X_train, axis = 1)


Y_train


# # Creating and Running the Neural Network
# 
# Define the model for use in the neural network. This model was taken from the Zhang paper and is attempting to replicate their work. 
# 

# VGG-like convolution stack
model = Sequential()
model.add(Convolution1D(256, 7, border_mode = 'valid', input_shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(Activation('relu'))
model.add(MaxPooling1D(3))

model.add(Convolution1D(256, 7, border_mode = 'valid')) 
model.add(Activation('sigmoid'))
model.add(MaxPooling1D(3))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('relu'))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('sigmoid'))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('relu'))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('sigmoid'))
model.add(MaxPooling1D(3))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(2048))
model.add(Dropout(0.5))
model.add(Dense(2))


# Compile the model and start running the model on the X_train and Y_train
# 

model.compile(loss='categorical_crossentropy', optimizer='adadelta')


model.fit(X_train, Y_train, nb_epoch=5000, batch_size=BATCH_SIZE, verbose=1, 
          show_accuracy=True, validation_split=0.1)


# Get the predicted classes for independent test set data, compare them with known labels and output the accuracy.
# 

Y_guess = model.predict_classes(X_test)


numCorrect = 0
for item in range(len(labelsTest1)):
    if Y_guess[item] == labelsTest1[item]:
        numCorrect += 1
print(numCorrect)
print(numCorrect * 1.0 / len(labelsTest1))


# # Topic Model Fuzzy
# 
# This is a notebook for trying to use topic models for classifying sets of text that are more syntactically similar than topically similar. This notebook attempts to distinguish between discussion and conclusion section of scientific papers. This notebook also augments the data with random sentences from the introduction section.
# 
# Below we are loading the dataset for use.
# 

from __future__ import print_function
from time import time
import os
from random import randint

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np

import pickle

validDocsDict = dict()
fileList = os.listdir("BioMedProcessed")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedProcessed/" + f, "rb")))


# Here we are setting some vaiables to be used below and defining a function for printing the top words in a topic for the topic modeling.
# 

n_samples = len(validDocsDict.keys())
n_features = 1000
n_topics = 2
n_top_words = 30
lengthOfIntroToAdd = 500

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# # Pre-process data
# 
# Here we are preprocessing data for use later. This code grabs the introduction, discussion and conclusion sections of the data. We are also creating appropriate labels for the data and spliting the documents up to train and test sets.
# 

print("Loading dataset...")
t0 = time()
documents = []
introductionSections = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0
introCount = 0

for k in validDocsDict.keys():
    if k.startswith("conclusion"):
        labels.append("conclusion")
        documents.append(validDocsDict[k])
        concCount += 1
        concLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("discussion"):
        labels.append("discussion")
        documents.append(validDocsDict[k])
        discCount += 1
        discLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("introduction") and len(validDocsDict[k]) > 5000:
        introCount += 1
        introductionSections.append(validDocsDict[k])

print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)
print(introCount)


# Now we add some random introduction sections to the discussion and conclusion sections to add some noise.
# 

for item in range(len(documents)):
    if labels[item] == "conclusion":
        intro = introductionSections[randint(0, len(introductionSections) - 1)].split(" ")
        randNum = randint(0, len(intro) - lengthOfIntroToAdd)
        introWords = intro[randNum:randNum + lengthOfIntroToAdd]
        documents[item] = documents[item] + " ".join(introWords)

train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.1)


# Here we are splitting the data up some more to train different models. Discussion and conclusion sections are being put into their own training sets. A TFIDF vectorizer is trained with the whole dataset of conclusion AND discussion sections. The multiple different training sets are then transformed using this vectorizer to get vector encodings of the text normalized to sum to 1 which accounts for differing lengths of conclusion and discussion sections.
# 

trainSetOne = []
trainSetTwo = []

for x in range(len(train)):
    if labelsTrain[x] == "conclusion":
        trainSetOne.append(train[x])
    else:
        trainSetTwo.append(train[x])

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(train)

tfSetOne = tf_vectorizer.transform(trainSetOne)
tfSetTwo = tf_vectorizer.transform(trainSetTwo)
tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf
trainSetOne = tfSetOne
trainSetTwo = tfSetTwo

print("done in %0.3fs." % (time() - t0))


# # LDA With Two Topics
# 
# Define an LDA topic model on the whole data set with two topics. This is trying to see if the topic model can define the difference between the two groups automatically and prints the top words per topic.
# 

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# Transform the unknown data through the topic model and calculate which topic it is more associated with according to the ratios. Calculate how many of each type (conclusion and discussion) go into each topic (1 or 2).
# 

results = lda.transform(test)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    print(str(labelsTest[x]) + " " + str(val1/total) + " " + str(val2/total))
    if val1 > val2:
        if labelsTest[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labelsTest[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1


# Print out the results from the topic transforms.
# 

print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))


# Get the parameters for the LDA.
# 

lda.get_params()


# # Basic Classifiers
# 
# Train three basic classifiers to solve the problem. Try Gaussian, Bernoulli and K Nearest Neighbors classifiers and calculate how accurate they are.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# # Two Topic Models
# 
# Define two topic models with 20 topics each, one on discussion sections and one on conclusion sections. Then transform both the train and test sets using both topic models to get 40 features for each sample based on the probability distribution for each topic in each LDA.
# 

ldaSet1 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
ldaSet2 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)


ldaSet1.fit(trainSetOne)
print_top_words(ldaSet1, tf_feature_names, n_top_words)


ldaSet2.fit(trainSetTwo)
print_top_words(ldaSet2, tf_feature_names, n_top_words)


results1 = ldaSet1.transform(train)
results2 = ldaSet2.transform(train)

resultsTest1 = ldaSet1.transform(test)
resultsTest2 = ldaSet2.transform(test)


results = np.hstack((results1, results2))
resultsTest = np.hstack((resultsTest1, resultsTest2))


# Define two classifiers using the transformed train and test sets from the topic models. Print out the accuracy of each one.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# Normalize the results of each sample of 40 features so they sum to 1. Then train two more classifiers using the data and print out the accuracy of each.
# 

for x in range(len(results)):
    total = 0
    for y in range(len(results[x])):
        total += results[x][y]
    for y in range(len(results[x])):
        results[x][y] = results[x][y]/total
        
for x in range(len(resultsTest)):
    total = 0
    for y in range(len(resultsTest[x])):
        total += resultsTest[x][y]
    for y in range(len(resultsTest[x])):
        resultsTest[x][y] = resultsTest[x][y]/total


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# # Topic Model Parts of Speech
# 
# This is a notebook for trying to use topic models for classifying sets of text that are more syntactically similar than topically similar. This notebook attempts to distinguish between discussion and conclusion section of scientific papers.
# 
# Below we are loading the dataset for use.
# 

from __future__ import print_function
from time import time
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np

import pickle

my_randoms1 = random.sample(xrange(31), 16)

validDocsDict = dict()
fileList1 = os.listdir("BioMedPOS")
for index, files1 in enumerate(fileList1):
    if index in my_randoms1:
        validDocsDict.update(pickle.load(open("BioMedPOS/" + files1, "rb")))
    
my_randoms2 = random.sample(xrange(10), 5)
    
fileList2 = os.listdir("PubMedPOS")
for index, files2 in enumerate(fileList2):
    if index in my_randoms2:
        validDocsDict.update(pickle.load(open("PubMedPOS/" + files2, "rb"))) 


# Here we are setting some vaiables to be used below and defining a function for printing the top words in a topic for the topic modeling.
# 

n_samples = len(validDocsDict.keys())
n_features = 200
n_topics = 2
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# # Pre-process data
# 
# Here we are preprocessing data for use later. This code only grabs the discussion and conclusion sections of the data. We are also creating appropriate labels for the data and spliting the documents up to train and test sets.
# 

print("Loading dataset...")
t0 = time()
documents = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0

for k in validDocsDict.keys():
    if k.startswith("conclusion"):
        labels.append("conclusion")
        documents.append(validDocsDict[k])
        concCount += 1
        concLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("discussion"):
        labels.append("discussion")
        documents.append(validDocsDict[k])
        discCount += 1
        discLengthTotal += len(validDocsDict[k].split(' '))

print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)

train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.1)


# Here we are splitting the data up some more to train different models. Discussion and conclusion sections are being put into their own training sets. A TFIDF vectorizer is trained with the whole dataset of conclusion AND discussion sections. The multiple different training sets are then transformed using this vectorizer to get vector encodings of the text normalized to sum to 1 which accounts for differing lengths of conclusion and discussion sections.
# 

trainSetOne = []
trainSetTwo = []

for x in range(len(train)):
    if labelsTrain[x] == "conclusion":
        trainSetOne.append(train[x])
    else:
        trainSetTwo.append(train[x])

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
#tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features)
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, ngram_range = (1,4))
t0 = time()
tf = tf_vectorizer.fit_transform(train)

tfSetOne = tf_vectorizer.transform(trainSetOne)
tfSetTwo = tf_vectorizer.transform(trainSetTwo)
tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf
trainSetOne = tfSetOne
trainSetTwo = tfSetTwo

print("done in %0.3fs." % (time() - t0))


# # LDA With Two Topics
# 
# Define an LDA topic model on the whole data set with two topics. This is trying to see if the topic model can define the difference between the two groups automatically and prints the top words per topic.
# 

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# Transform the unknown data through the topic model and calculate which topic it is more associated with according to the ratios. Calculate how many of each type (conclusion and discussion) go into each topic (1 or 2).
# 

results = lda.transform(test)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    print(str(labelsTest[x]) + " " + str(val1/total) + " " + str(val2/total))
    if val1 > val2:
        if labelsTest[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labelsTest[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1


# Print out the results from the topic transforms.
# 

print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))


# Get the parameters for the LDA.
# 

lda.get_params()


# # Basic Classifiers
# 
# Train three basic classifiers to solve the problem. Try Gaussian, Bernoulli and K Nearest Neighbors classifiers and calculate how accurate they are.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# # Two Topic Models
# 
# Define two topic models with 20 topics each, one on discussion sections and one on conclusion sections. Then transform both the train and test sets using both topic models to get 40 features for each sample based on the probability distribution for each topic in each LDA.
# 

ldaSet1 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
ldaSet2 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)


ldaSet1.fit(trainSetOne)
print_top_words(ldaSet1, tf_feature_names, n_top_words)


ldaSet2.fit(trainSetTwo)
print_top_words(ldaSet2, tf_feature_names, n_top_words)


results1 = ldaSet1.transform(train)
results2 = ldaSet2.transform(train)

resultsTest1 = ldaSet1.transform(test)
resultsTest2 = ldaSet2.transform(test)


results = np.hstack((results1, results2))
resultsTest = np.hstack((resultsTest1, resultsTest2))


# Define three classifiers using the transformed train and test sets from the topic models. Print out the accuracy of each one.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# Normalize the results of each sample of 40 features so they sum to 1. Then train two more classifiers using the data and print out the accuracy of each.
# 

for x in range(len(results)):
    total = 0
    for y in range(len(results[x])):
        total += results[x][y]
    for y in range(len(results[x])):
        results[x][y] = results[x][y]/total
        
for x in range(len(resultsTest)):
    total = 0
    for y in range(len(resultsTest[x])):
        total += resultsTest[x][y]
    for y in range(len(resultsTest[x])):
        resultsTest[x][y] = resultsTest[x][y]/total


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))





# # Topic Model Two Datasets
# 
# This is a notebook for trying to use topic models for classifying sets of text that are more syntactically similar than topically similar. This notebook attempts to distinguish between discussion and conclusion section of scientific papers.
# 
# Below we are loading the two datasets for use.
# 

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np
import os
import pickle

validDocsDict = dict()
fileList = os.listdir("BioMedProcessed")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedProcessed/" + f, "rb")))

validDocsDict2 = dict()
fileList = os.listdir("PubMedProcessed")
for f in fileList:
    validDocsDict2.update(pickle.load(open("PubMedProcessed/" + f, "rb")))


# Here we are setting some vaiables to be used below and defining a function for printing the top words in a topic for the topic modeling.
# 

n_samples = len(validDocsDict.keys())
n_features = 10000
n_topics = 2
n_top_words = 30


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# # Preprocess Data
# 
# Here we are preprocessing data for use later. This code only grabs the discussion and conclusion sections of the data. We are also creating appropriate labels for the data and spliting the documents up to train and test sets. We do this for both sets of data and then for a combined set of data.
# 

print("Loading dataset...")
t0 = time()
documents = []
testPubDocuments = []
allDocuments = []
labels = []
testPubLabels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0

combinedDicts = validDocsDict.copy()
combinedDicts.update(validDocsDict2.copy())

for k in validDocsDict.keys():
    if k.startswith("conclusion"):
        labels.append("conclusion")
        documents.append(validDocsDict[k])
        concCount += 1
        concLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("discussion"):
        labels.append("discussion")
        documents.append(validDocsDict[k])
        discCount += 1
        discLengthTotal += len(validDocsDict[k].split(' '))
        
for k in validDocsDict2.keys():
    if k.startswith("conclusion"):
        testPubLabels.append("conclusion")
        testPubDocuments.append(validDocsDict2[k])
        concCount += 1
        concLengthTotal += len(validDocsDict2[k].split(' '))
    elif k.startswith("discussion"):
        testPubLabels.append("discussion")
        testPubDocuments.append(validDocsDict2[k])
        discCount += 1
        discLengthTotal += len(validDocsDict2[k].split(' '))
        
for k in combinedDicts.keys():
    if k.startswith("conclusion"):
        allDocuments.append(combinedDicts[k])
    elif k.startswith("discussion"):
        allDocuments.append(combinedDicts[k])
        
print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)

train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.1)


# Here we are splitting the data up some more to train different models. Discussion and conclusion sections are being put into their own training sets. A TFIDF vectorizer is trained with the whole dataset of conclusion AND discussion sections from both data sets. The multiple different training sets are then transformed using this vectorizer to get vector encodings of the text normalized to sum to 1 which accounts for differing lengths of conclusion and discussion sections and between data sets.
# 

trainSetOne = []
trainSetTwo = []

for x in range(len(train)):
    if labelsTrain[x] == "conclusion":
        trainSetOne.append(train[x])
    else:
        trainSetTwo.append(train[x])

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, stop_words='english')
t0 = time()
tf_vectorizer.fit(allDocuments)
tf = tf_vectorizer.transform(train)

tfSetOne = tf_vectorizer.transform(trainSetOne)
tfSetTwo = tf_vectorizer.transform(trainSetTwo)
tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf
trainSetOne = tfSetOne
trainSetTwo = tfSetTwo

pubTest = tf_vectorizer.transform(testPubDocuments)

print("done in %0.3fs." % (time() - t0))


# # LDA With Two Topics
# 
# Define an LDA topic model on the whole data set with two topics. This is trying to see if the topic model can define the difference between the two groups automatically and prints the top words per topic. This is only performed on the first data set.
# 

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# Transform the unknown data through the topic model and calculate which topic it is more associated with according to the ratios. Calculate how many of each type (conclusion and discussion) go into each topic (1 or 2). This is done just for the first dataset.
# 

results = lda.transform(test)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    print(str(labelsTest[x]) + " " + str(val1/total) + " " + str(val2/total))
    if val1 > val2:
        if labelsTest[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labelsTest[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1


# Print out the results from the topic transforms.
# 

print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))


# Get the parameters for the LDA.
# 

lda.get_params()


# # Basic Classifiers
# 
# Train three basic classifiers to solve the problem. Try Gaussian, Bernoulli and K Nearest Neighbors classifiers and calculate how accurate they are.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# # Two Topic Models
# 
# Define two topic models with 20 topics each, one on discussion sections and one on conclusion sections. Then transform both the train and test sets using both topic models to get 40 features for each sample based on the probability distribution for each topic in each LDA.
# 

ldaSet1 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
ldaSet2 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)


ldaSet1.fit(trainSetOne)
print_top_words(ldaSet1, tf_feature_names, n_top_words)


ldaSet2.fit(trainSetTwo)
print_top_words(ldaSet2, tf_feature_names, n_top_words)


results1 = ldaSet1.transform(train)
results2 = ldaSet2.transform(train)

resultsTest1 = ldaSet1.transform(test)
resultsTest2 = ldaSet2.transform(test)


results = np.hstack((results1, results2))
resultsTest = np.hstack((resultsTest1, resultsTest2))


# Define two classifiers using the transformed train and test sets from the topic models. Print out the accuracy of each one.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# Normalize the results of each sample of 40 features so they sum to 1. Then train two more classifiers using the data and print out the accuracy of each.
# 

for x in range(len(results)):
    total = 0
    for y in range(len(results[x])):
        total += results[x][y]
    for y in range(len(results[x])):
        results[x][y] = results[x][y]/total
        
for x in range(len(resultsTest)):
    total = 0
    for y in range(len(resultsTest[x])):
        total += resultsTest[x][y]
    for y in range(len(resultsTest[x])):
        resultsTest[x][y] = resultsTest[x][y]/total
        


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# # Basic Classifiers Between Two Datasets
# 
# Train and test two Bernoulli classifiers (one where dataset 1 is trained and one where dataset 2 is trained) and print out the results of accuracy.
# 

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(pubTest.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == testPubLabels[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(pubTest.toarray(), testPubLabels)

classResults = classifier.predict(train.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# This notebook follows the Training example given by Kaggle https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words to display each document as a Bag-of-Words and then uses Scikit-learn's logistic regression to train a model.
# 
# The data in this initial test is the labeled dataset of 50,000 IMDB movie reviews for sentiment analysis.  This was obtained from Kaggle.  https://www.kaggle.com/c/word2vec-nlp-tutorial/data  The labeled training data was split into 90% train, 10% test.
# 
# While Pythia will not be doing sentiment analysis, using this data was a quick and easy way to see how a bag-of-words + logistic regression could work for novelty detection.  All that is really needed for this type of analysis is a labels column (in this dataset that is sentiment) and text (which is review in this dataset).
# 
# In the future we will be looking at better to see how spacy could substitute in for BeautifulSoup and nltk, 
# 




import pandas as pd
from bs4 import BeautifulSoup
import nltk
import numpy as np
from sklearn import linear_model


data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


#do a simple split by using "ID" with trailing 3 before the _

train = data[data["id"].str.contains("3_")==False]
test = data[data["id"].str.contains("3_")==True]


print train.shape, test.shape


2500/25000.0 *100 #so split is perfect 10%


train.head()


test.head()


sum(train["sentiment"])/22500.0 #training set is nicely half positive and half negative as well


#The tutorial goes through the steps in the function to show what each is doing they are...


# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison
print train["review"][0]
print example1.get_text()


import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print letters_only


lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words


#Some of the most common stopwords, you would normally get this through a package
stopwords  = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


#The Main function is review_to_words which does all the text process of cleaning and splitting


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = stopwords                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


clean_review = review_to_words( train["review"][0] )
print clean_review


train_labels = train["sentiment"]
test_labels = test["sentiment"]


print "Cleaning and parsing the training set movie reviews...\n"
# Get the number of reviews based on the dataframe column size
num_reviews = data["review"].size

# Initialize an empty list to hold the clean reviews
#When the data was split the train, test sets kept the index which we will use to our advantage here
clean_train_reviews = []
clean_test_reviews = []
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )
    try:
        clean_train_reviews.append(review_to_words(train["review"][i] ))
    except:
        clean_test_reviews.append(review_to_words(test["review"][i] ))


print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


#Also make the test data into the correct format
test_data_features = vectorizer.fit_transform(clean_test_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
test_data_features = test_data_features.toarray()


# Let us look at what the data now looks like
# 

print train_data_features.shape


print sum(train_data_features[0]), max(train_data_features[0]), train_data_features[0]


print sum(test_data_features[0]), max(test_data_features[0]), test_data_features[0]


# And Finally use Logistic Regression to train a model and perform a test
# 

#try using the BagOfWords with the logistic regression
logreg = linear_model.LogisticRegression(C=1e5)


# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(train_data_features, train_labels)


#Now that we have something trained we can check if it is accurate with the test set


preds = logreg.predict(test_data_features)


#because the label is zero or one the root difference is simply the absolute difference between predicted and actual
rmse = sum(abs(preds-test_labels))/float(len(test_labels)) 
print rmse


#Not a very good model as it is just every so slightly better than random





#Some additional data analysis of the vocabulary and model...


# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab[:10]


import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag


# # NLTK
# 
# Trying out functionality of NLTK word and character counts
# 

import re
import nltk.data
from nltk import wordpunct_tokenize

text = '''There are two ways of constructing a software design:
One way is to make it so simple that there are obviously no deficiencies and
the other way is to make it so complicated that there are no obvious deficiencies.'''
# — C.A.R. Hoare, The 1980 ACM Turing Award Lecture

# split into words by punctuations
# remove punctuations and all '-' words
RE = re.compile('[0-9a-z-]', re.I)
words = filter(lambda w: RE.search(w) and w.replace('-', ''), wordpunct_tokenize(text))

wordc = len(words)
charc = sum(len(w) for w in words)

sent = nltk.data.load('tokenizers/punkt/english.pickle')

sents = sent.tokenize(text)
sentc = len(sents)

print words
print charc, wordc, sentc
print 4.71 * charc / wordc + 0.5 * wordc / sentc - 21.43


# # Readability Scores
# 
# This calculates different readability scores for text - a test to see how it works
# 
# The Dale-Chall score is the best to use, according to academic research: http://www.ncpublicschools.org/docs/superintendents/memos/2014/01/readability.pdf
# 

# reference: https://pypi.python.org/pypi/textstat/

from textstat.textstat import textstat
if __name__ == '__main__':
    test_data = """Playing games has always been thought to be important to the development of well-balanced and creative children; however, what part, if any, they should play in the lives of adults has never been researched that deeply. I believe that playing games is every bit as important for adults as for children. Not only is taking time out to play games with our children and other adults valuable to building interpersonal relationships but is also a wonderful way to release built up tension."""

print textstat.flesch_reading_ease(test_data) #doesn't work when punctuation removed because it counts sentences
print textstat.smog_index(test_data)
print textstat.flesch_kincaid_grade(test_data)
print textstat.coleman_liau_index(test_data)
print textstat.automated_readability_index(test_data)
print textstat.dale_chall_readability_score(test_data) #this is the best one to use, according to academic research
print textstat.difficult_words(test_data)
print textstat.linsear_write_formula(test_data)
print textstat.gunning_fog(test_data)
print textstat.text_standard(test_data)


# # Loading Dataset
# 
# This takes the same functionality from the original code, loads one test file (since I was having performance issues) and appending the Dale-Chall readability scores so we can try and classify basd on those. This includes multi-threading for 6 processors becuase it is such a calculation intensive process, even when loading just one file.
# 

from __future__ import print_function
from time import time
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split
from textstat.textstat import textstat

import numpy as np

import pickle

validDocsDict = dict()

file_name = "TestDocsPub_kimtest.p"

validDocsDict = dict()
fileList1 = os.listdir("BioMedProcessed")

validDocsDict.update(pickle.load(open("BioMedProcessed/" + file_name, "rb")))


n_samples = len(validDocsDict.keys())
n_features = 1000
n_topics = 2
n_top_words = 30


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


from multiprocessing import Pool

print("Loading dataset...")
t0 = time()
#documents = []
readability = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
#concCount = 0
#discCount = 0

def f(k):
    for k in validDocsDict.keys():
        if k.startswith("conclusion"):
            labels.append(0)
            #documents.append(validDocsDict[k])
            readability.append(textstat.dale_chall_readability_score(validDocsDict[k]))
            #concCount += 1
            #concLengthTotal += len(validDocsDict[k].split(' '))
        elif k.startswith("discussion"):
            labels.append(1)
            #documents.append(validDocsDict[k])
            readability.append(textstat.dale_chall_readability_score(validDocsDict[k]))
            #discCount += 1
            #discLengthTotal += len(validDocsDict[k].split(' '))
            
po = Pool(6)
results = [po.apply_async(f, args = (k,)) for k in validDocsDict.keys()]
po.close()
po.join()
output = [p.get() for p in results]

#print(len(documents))
#print(concLengthTotal * 1.0/ concCount)
#print(discLengthTotal * 1.0/ discCount)
print(len(readability))
#print(concCount + discCount)

train, test, labelsTrain, labelsTest = train_test_split(readability, labels, test_size = 0.1)


# # Test Classifiers
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.ensemble import BaggingClassifier

classifier = BaggingClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# # Data Processing for Topic Model Test
# 
# Getting the data from the repository...don't run unless you don't have the data!
# 
# !apt-get -y install curl
# 
# !curl -o BioMedSent/BioMedSentences.tar.zip http://i.stanford.edu/hazy/opendata/bmc/bmc_full_dddb_20150927_9651bf4a468cefcea30911050c2ca6db.tar.bzip2
# 
# http://i.stanford.edu/hazy/opendata/pmc/pmc_dddb_full_20150927_3b20db570e2cb90ab81c5c6f63babc91.tar.bzip2
# 

# # Import Data
# 
# This section defines the Sentence object used when importing and saving the data. Grab the files in a directory and process a subset of them.
# 

#Import Statements
import string
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import h5py

#Sentence object definition for data import and processing 
class Sentence:
    def __init__(self, document, sentenceNumber, wordList, lemmaList, posList):
        self.document = document
        self.sentenceNumber = sentenceNumber
        self.wordList = wordList
        self.lemmaList = lemmaList
        self.posList = posList
        self.sentence = " ".join([word for word in wordList if word not in string.punctuation])
        self.lemmaSent = " ".join([word for word in lemmaList if word not in string.punctuation])

#Get the files we want to process and put them into a list of lists called sentList
fileList = os.listdir("../PubMed/pmc_dddb_full")
sentList = []
fileList.sort()
for n in range(3, 5):
    f = open("../PubMed/pmc_dddb_full/" + fileList[n], 'r')
    for line in f:
        sentList.append(line.split('\t'))

len(sentList)


# 
# Now that we have all of the sentences in a list of lists grab the first element of each sentence list (the document id) and add that to a docList. Make this docList a set so we have the number of unique documents.
# 

docList = []
for thing in sentList:
    docList.append(thing[0])

len(set(docList))


# # Process Data
# 
# Define the processSent function for use by the multiprocessing part of the code. This function takes off some of the structure of parts of the data (removing the {,}, and ") and defines the Sentence object with all the appropriate parts.
# 
# We then use 14 cores (if available) for the Pool object and apply the processSent function to every sentence.
# 

sentObjList = []
def processSent(item):
    wordList = item[3].replace('"',"").lstrip("{").rstrip("}").split(",")
    wordList = filter(None, wordList)
    posList = item[4].split(",")
    lemmaList = item[6].replace('"',"").lstrip("{").rstrip("}").split(",")
    lemmaList = filter(None, lemmaList)
    return Sentence(item[0], item[1], wordList, lemmaList, posList)

po = Pool(20)
results = [po.apply_async(processSent, args = (sent,)) for sent in sentList]
po.close()
po.join()
output = [p.get() for p in results]
sentObjList = output
sentObjList[7].lemmaSent


# 
# Now that the sentences are processed, we need to find which sections these sentences should be atributed. For most of these papers, section headers are one word sentences. We are looking for common section headers and saving the sentence numbers for that section in that document.
# 

headingsDict = defaultdict(dict)

for sent in sentObjList:
    if len(sent.wordList) == 1:
        #print(sent.wordList)
        word = string.upper(sent.wordList[0]).strip()
        if word == 'INTRODUCTION' or word == 'BACKGROUND':
            headingsDict[sent.document]["introduction"] = sent.sentenceNumber
        elif word == 'METHODS':
            headingsDict[sent.document]["methods"] = sent.sentenceNumber
        elif word == 'RESULTS':
            headingsDict[sent.document]["results"] = sent.sentenceNumber
        elif word == 'DISCUSSION':
            headingsDict[sent.document]["discussion"] = sent.sentenceNumber
        elif word == 'CONCLUSION':
            headingsDict[sent.document]["conclusion"] = sent.sentenceNumber
        elif word == 'REFERENCES':
            headingsDict[sent.document]["references"] = sent.sentenceNumber
        

headingsDict.keys()


# 
# Now the sentences need to be tagged to their appropriate section and concatenated into one string per section per document.
# 
# The sentences are assigned a section by whichever section they are closest to (that is less than their sentence number). For example, if introduction had sentence number 5 and methods had sentence number 25, sentence number 20 would be assigned to introduction.
# 
# This is done for each sentence in each document and joined by spaces into a one string per section per document. Finally, only the documents that contain an introduction, discussion, and conclusion are kept and put into the validDocsDict dictionary
# 

documentDict = defaultdict(list)
docPartsDict = defaultdict(lambda : defaultdict(list))
docPartsCombinedDict = defaultdict(dict)

for item in sentObjList:
    documentDict[item.document].append(item)
    
for document in documentDict.keys():
    docSentList = documentDict[document]
    introNum = int(headingsDict[document].get("introduction", -1))
    methoNum = int(headingsDict[document].get("methods", -1))
    resultNum = int(headingsDict[document].get("results", -1))
    discussNum = int(headingsDict[document].get("discussion", -1))
    conclusionNum = int(headingsDict[document].get("conclusion", -1))
    refNum = int(headingsDict[document].get("references", -1))

    for sent in docSentList:
        label = "noSection"
        dist = int(sent.sentenceNumber)
        sentNumber = int(sent.sentenceNumber)
        
        if dist > sentNumber - introNum and sentNumber - introNum > 0:
            label = "introduction"
            dist = sentNumber - introNum
        if dist > sentNumber - methoNum and sentNumber - methoNum > 0:
            label = "methods"
            dist = sentNumber - methoNum
        if dist > sentNumber - resultNum and sentNumber - resultNum > 0:
            label = "results"
            dist = sentNumber - resultNum
        if dist > sentNumber - discussNum and sentNumber - discussNum > 0:
            label = "discussion"
            dist = sentNumber - discussNum
        if dist > sentNumber - conclusionNum and sentNumber - conclusionNum > 0:
            label = "conclusion"
            dist = sentNumber - conclusionNum
        if dist > sentNumber - refNum and sentNumber - refNum > 0:
            label = "references"
            dist = sentNumber - refNum
        if sent.sentence.strip().lower() not in ["introduction", "methods", "results", "discussion", "conclusion", "references"]:
            docPartsDict[document][label].append(sent)
    
    for x in docPartsDict[document].keys():
        docPartsCombinedDict[document][x] = " ".join(y.sentence for y in sorted(docPartsDict[document][x], key=lambda z: z.sentenceNumber))

validDocsDict = defaultdict(dict)

for doc in docPartsCombinedDict.keys():
    tempKeys = docPartsCombinedDict[doc].keys()
    if 'introduction' in tempKeys and 'discussion' in tempKeys and 'conclusion' in tempKeys:
        validDocsDict[doc] = docPartsCombinedDict[doc]

print(str(len(docPartsCombinedDict.keys())))
print(str(len(validDocsDict.keys())))


# 
# Take the valid documents in the validDocsDict and output to a pickle file with the key part_docid with the part being introduction, methods, etc. and the docid allowing for document tracking.
# 

f = h5py.File('PubMed2.hdf5','w')

partsGroups = {}
for doc in validDocsDict.keys():
    for part in validDocsDict[doc].keys():
        if part not in partsGroups.keys():
            partsGroups[part] = f.create_group(part)
        partsGroups[part].create_dataset(doc, data=np.string_(validDocsDict[doc][part]))
f.close()


# # Topic Model Fuzzy Parts of Speech
# 
# This is a notebook for trying to use topic models for classifying sets of text that are more syntactically similar than topically similar. This notebook attempts to distinguish between discussion and conclusion section of scientific papers. This notebook also augments the data with random sentences from the introduction section.
# 
# Below we are loading the dataset for use.
# 

from __future__ import print_function
from time import time
import os
from random import randint

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np

import pickle

validDocsDict = dict()
fileList = os.listdir("BioMedPOS")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedPOS/" + f, "rb")))


# Here we are setting some vaiables to be used below and defining a function for printing the top words in a topic for the topic modeling.
# 

n_samples = len(validDocsDict.keys())
n_features = 5000
n_topics = 2
n_top_words = 30
lengthOfIntroToAdd = 500

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# # Pre-process data
# 
# Here we are preprocessing data for use later. This code grabs the introduction, discussion and conclusion sections of the data. We are also creating appropriate labels for the data and spliting the documents up to train and test sets.
# 

print("Loading dataset...")
t0 = time()
documents = []
introductionSections = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0
introCount = 0

for k in validDocsDict.keys():
    if k.startswith("conclusion"):
        labels.append("conclusion")
        documents.append(validDocsDict[k])
        concCount += 1
        concLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("discussion"):
        labels.append("discussion")
        documents.append(validDocsDict[k])
        discCount += 1
        discLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("introduction") and len(validDocsDict[k]) > 5000:
        introCount += 1
        introductionSections.append(validDocsDict[k])

print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)
print(introCount)


# Now we add some random introduction sections to the discussion and conclusion sections to add some noise.
# 

for item in range(len(documents)):
    if labels[item] == "conclusion":
        documents[item] = documents[item] + documents[item]
    #intro = introductionSections[randint(0, len(introductionSections) - 1)].split(" ")
    #randNum = randint(0, len(intro) - lengthOfIntroToAdd)
    #introWords = intro[randNum:randNum + lengthOfIntroToAdd]
    #documents[item] = documents[item] + " ".join(introWords)

train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.5)


# Here we are splitting the data up some more to train different models. Discussion and conclusion sections are being put into their own training sets. A TFIDF vectorizer is trained with the whole dataset of conclusion AND discussion sections. The multiple different training sets are then transformed using this vectorizer to get vector encodings of the text normalized to sum to 1 which accounts for differing lengths of conclusion and discussion sections.
# 

trainSetOne = []
trainSetTwo = []

for x in range(len(train)):
    if labelsTrain[x] == "conclusion":
        trainSetOne.append(train[x])
    else:
        trainSetTwo.append(train[x])

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, ngram_range = (1,6))
t0 = time()
tf = tf_vectorizer.fit_transform(train)

tfSetOne = tf_vectorizer.transform(trainSetOne)
tfSetTwo = tf_vectorizer.transform(trainSetTwo)
tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf
trainSetOne = tfSetOne
trainSetTwo = tfSetTwo

print("done in %0.3fs." % (time() - t0))


# # LDA With Two Topics
# 
# Define an LDA topic model on the whole data set with two topics. This is trying to see if the topic model can define the difference between the two groups automatically and prints the top words per topic.
# 

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# Transform the unknown data through the topic model and calculate which topic it is more associated with according to the ratios. Calculate how many of each type (conclusion and discussion) go into each topic (1 or 2).
# 

results = lda.transform(test)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    print(str(labelsTest[x]) + " " + str(val1/total) + " " + str(val2/total))
    if val1 > val2:
        if labelsTest[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labelsTest[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1


# Print out the results from the topic transforms.
# 

print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))


# Get the parameters for the LDA.
# 

lda.get_params()


# # Basic Classifiers
# 
# Train three basic classifiers to solve the problem. Try Gaussian, Bernoulli and K Nearest Neighbors classifiers and calculate how accurate they are.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())
numRight = 0
numWrongDisc = 0
numWrongConc = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1
    else:
        if classResults[item] == "discussion":
            numWrongDisc += 1
        else:
            numWrongConc += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))
print("Incorrectly classified as discussion: " + str(numWrongDisc))
print("Incorrectly classified as conclusion: " + str(numWrongConc))
print(len(classResults))


# # Two Topic Models
# 
# Define two topic models with 20 topics each, one on discussion sections and one on conclusion sections. Then transform both the train and test sets using both topic models to get 40 features for each sample based on the probability distribution for each topic in each LDA.
# 

ldaSet1 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
ldaSet2 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)


ldaSet1.fit(trainSetOne)
print_top_words(ldaSet1, tf_feature_names, n_top_words)


ldaSet2.fit(trainSetTwo)
print_top_words(ldaSet2, tf_feature_names, n_top_words)


results1 = ldaSet1.transform(train)
results2 = ldaSet2.transform(train)

resultsTest1 = ldaSet1.transform(test)
resultsTest2 = ldaSet2.transform(test)


results = np.hstack((results1, results2))
resultsTest = np.hstack((resultsTest1, resultsTest2))


# Define two classifiers using the transformed train and test sets from the topic models. Print out the accuracy of each one.
# 

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


# Normalize the results of each sample of 40 features so they sum to 1. Then train two more classifiers using the data and print out the accuracy of each.
# 

for x in range(len(results)):
    total = 0
    for y in range(len(results[x])):
        total += results[x][y]
    for y in range(len(results[x])):
        results[x][y] = results[x][y]/total
        
for x in range(len(resultsTest)):
    total = 0
    for y in range(len(resultsTest[x])):
        total += resultsTest[x][y]
    for y in range(len(resultsTest[x])):
        resultsTest[x][y] = resultsTest[x][y]/total


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))





# # Data Processing for Topic Model Test
# 
# Getting the data from the repository...don't run unless you don't have the data!
# 
# !apt-get -y install curl
# 
# !curl -o BioMedSent/BioMedSentences.tar.zip http://i.stanford.edu/hazy/opendata/bmc/bmc_full_dddb_20150927_9651bf4a468cefcea30911050c2ca6db.tar.bzip2
# 
# http://i.stanford.edu/hazy/opendata/pmc/pmc_dddb_full_20150927_3b20db570e2cb90ab81c5c6f63babc91.tar.bzip2
# 

# # Import Data
# 
# This section defines the Sentence object used when importing and saving the data. Grab the files in a directory and process a subset of them.
# 

#Import Statements
import string
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

#Sentence object definition for data import and processing 
class Sentence:
    def __init__(self, document, sentenceNumber, wordList, lemmaList, posList):
        self.document = document
        self.sentenceNumber = sentenceNumber
        self.wordList = wordList
        self.lemmaList = lemmaList
        self.posList = posList
        self.sentence = " ".join([word for word in wordList if word not in string.punctuation])
        self.lemmaSent = " ".join([word for word in lemmaList if word not in string.punctuation])

#Get the files we want to process and put them into a list of lists called sentList
fileList = os.listdir("../PubMed/pmc_dddb_full")
sentList = []
fileList.sort()
for n in range(27, 28):
    f = open("../PubMed/pmc_dddb_full/" + fileList[n], 'r')
    for line in f:
        sentList.append(line.split('\t'))

len(sentList)


# 
# Now that we have all of the sentences in a list of lists grab the first element of each sentence list (the document id) and add that to a docList. Make this docList a set so we have the number of unique documents.
# 

docList = []
for thing in sentList:
    docList.append(thing[0])

len(set(docList))


# # Process Data
# 
# Define the processSent function for use by the multiprocessing part of the code. This function takes off some of the structure of parts of the data (removing the {,}, and ") and defines the Sentence object with all the appropriate parts.
# 
# We then use 14 cores (if available) for the Pool object and apply the processSent function to every sentence.
# 

sentObjList = []
def processSent(item):
    wordList = item[3].replace('"',"").lstrip("{").rstrip("}").split(",")
    wordList = filter(None, wordList)
    posList = item[4].split(",")
    lemmaList = item[6].replace('"',"").lstrip("{").rstrip("}").split(",")
    lemmaList = filter(None, lemmaList)
    return Sentence(item[0], item[1], wordList, lemmaList, posList)

po = Pool(16)
results = [po.apply_async(processSent, args = (sent,)) for sent in sentList]
po.close()
po.join()
output = [p.get() for p in results]
sentObjList = output
sentObjList[7].lemmaSent


# 
# Now that the sentences are processed, we need to find which sections these sentences should be atributed. For most of these papers, section headers are one word sentences. We are looking for common section headers and saving the sentence numbers for that section in that document.
# 

headingsDict = defaultdict(dict)

for sent in sentObjList:
    if len(sent.wordList) == 1:
        #print(sent.wordList)
        word = string.upper(sent.wordList[0]).strip()
        if word == 'INTRODUCTION' or word == 'BACKGROUND':
            headingsDict[sent.document]["introduction"] = sent.sentenceNumber
        elif word == 'METHODS':
            headingsDict[sent.document]["methods"] = sent.sentenceNumber
        elif word == 'RESULTS':
            headingsDict[sent.document]["results"] = sent.sentenceNumber
        elif word == 'DISCUSSION':
            headingsDict[sent.document]["discussion"] = sent.sentenceNumber
        elif word == 'CONCLUSION':
            headingsDict[sent.document]["conclusion"] = sent.sentenceNumber
        elif word == 'REFERENCES':
            headingsDict[sent.document]["references"] = sent.sentenceNumber
        

headingsDict.keys()


# 
# Now the sentences need to be tagged to their appropriate section and concatenated into one string per section per document.
# 
# The sentences are assigned a section by whichever section they are closest to (that is less than their sentence number). For example, if introduction had sentence number 5 and methods had sentence number 25, sentence number 20 would be assigned to introduction.
# 
# This is done for each sentence in each document and joined by spaces into a one string per section per document. Finally, only the documents that contain an introduction, discussion, and conclusion are kept and put into the validDocsDict dictionary
# 

documentDict = defaultdict(list)
docPartsDict = defaultdict(lambda : defaultdict(list))
docPartsCombinedDict = defaultdict(dict)

for item in sentObjList:
    documentDict[item.document].append(item)
    
for document in documentDict.keys():
    docSentList = documentDict[document]
    introNum = int(headingsDict[document].get("introduction", -1))
    methoNum = int(headingsDict[document].get("methods", -1))
    resultNum = int(headingsDict[document].get("results", -1))
    discussNum = int(headingsDict[document].get("discussion", -1))
    conclusionNum = int(headingsDict[document].get("conclusion", -1))
    refNum = int(headingsDict[document].get("references", -1))

    for sent in docSentList:
        label = "noSection"
        dist = int(sent.sentenceNumber)
        sentNumber = int(sent.sentenceNumber)
        
        if dist > sentNumber - introNum and sentNumber - introNum > 0:
            label = "introduction"
            dist = sentNumber - introNum
        if dist > sentNumber - methoNum and sentNumber - methoNum > 0:
            label = "methods"
            dist = sentNumber - methoNum
        if dist > sentNumber - resultNum and sentNumber - resultNum > 0:
            label = "results"
            dist = sentNumber - resultNum
        if dist > sentNumber - discussNum and sentNumber - discussNum > 0:
            label = "discussion"
            dist = sentNumber - discussNum
        if dist > sentNumber - conclusionNum and sentNumber - conclusionNum > 0:
            label = "conclusion"
            dist = sentNumber - conclusionNum
        if dist > sentNumber - refNum and sentNumber - refNum > 0:
            label = "references"
            dist = sentNumber - refNum
        if sent.sentence.strip().lower() not in ["introduction", "methods", "results", "discussion", "conclusion", "references"]:
            docPartsDict[document][label].append(sent)
    
    for x in docPartsDict[document].keys():
        docPartsCombinedDict[document][x] = " ".join(y.sentence for y in sorted(docPartsDict[document][x], key=lambda z: z.sentenceNumber))

validDocsDict = defaultdict(dict)

for doc in docPartsCombinedDict.keys():
    tempKeys = docPartsCombinedDict[doc].keys()
    if 'introduction' in tempKeys and 'discussion' in tempKeys and 'conclusion' in tempKeys:
        validDocsDict[doc] = docPartsCombinedDict[doc]

print(str(len(docPartsCombinedDict.keys())))
print(str(len(validDocsDict.keys())))


# 
# Take the valid documents in the validDocsDict and output to a pickle file with the key part_docid with the part being introduction, methods, etc. and the docid allowing for document tracking.
# 

outputDict = dict()
for doc in validDocsDict.keys():
    for part in validDocsDict[doc].keys():
        outputDict[part + "_" + doc] = validDocsDict[doc][part]

pickle.dump(outputDict, open("TestDocsPub27.p", "wb"))





# # Data Processing for Topic Model Test
# 
# Getting the data from the repository...don't run unless you don't have the data!
# 
# !apt-get -y install curl
# 
# !curl -o BioMedSent/BioMedSentences.tar.zip http://i.stanford.edu/hazy/opendata/bmc/bmc_full_dddb_20150927_9651bf4a468cefcea30911050c2ca6db.tar.bzip2
# 
# http://i.stanford.edu/hazy/opendata/pmc/pmc_dddb_full_20150927_3b20db570e2cb90ab81c5c6f63babc91.tar.bzip2
# 

# # Import Data
# 
# This section defines the Sentence object used when importing and saving the data. Grab the files in a directory and process a subset of them.
# 
# The only difference between this and the ETLScriptPapers is that punctuation is not stripped out, so we can differentiate between sentences for the readabilty scores.
# 

#Import Statements
import string
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

#Sentence object definition for data import and processing 
class Sentence:
    def __init__(self, document, sentenceNumber, wordList, lemmaList, posList):
        self.document = document
        self.sentenceNumber = sentenceNumber
        self.wordList = wordList
        self.lemmaList = lemmaList
        self.posList = posList
        self.sentence = " ".join([word for word in wordList]) #if word not in string.punctuation
        self.lemmaSent = " ".join([word for word in lemmaList]) #if word not in string.punctuation

#Get the files we want to process and put them into a list of lists called sentList
fileList = os.listdir("../BioMedSent/")
sentList = []
fileList.sort()
for n in range(1,2):
    f = open("../BioMedSent/" + fileList[n], 'r')
    for line in f:
        sentList.append(line.split('\t'))

len(sentList)


# 
# Now that we have all of the sentences in a list of lists grab the first element of each sentence list (the document id) and add that to a docList. Make this docList a set so we have the number of unique documents.
# 

docList = []
for thing in sentList:
    docList.append(thing[0])

len(set(docList))


# # Process Data
# 
# Define the processSent function for use by the multiprocessing part of the code. This function takes off some of the structure of parts of the data (removing the {,}, and ") and defines the Sentence object with all the appropriate parts.
# 
# We then use 14 cores (if available) for the Pool object and apply the processSent function to every sentence.
# 

sentObjList = []
def processSent(item):
    wordList = item[3].replace('"',"").lstrip("{").rstrip("}").split(",")
    wordList = filter(None, wordList)
    posList = item[4].split(",")
    lemmaList = item[6].replace('"',"").lstrip("{").rstrip("}").split(",")
    lemmaList = filter(None, lemmaList)
    return Sentence(item[0], item[1], wordList, lemmaList, posList)

po = Pool(6)
results = [po.apply_async(processSent, args = (sent,)) for sent in sentList]
po.close()
po.join()
output = [p.get() for p in results]
sentObjList = output
sentObjList[7].lemmaSent


# 
# Now that the sentences are processed, we need to find which sections these sentences should be atributed. For most of these papers, section headers are one word sentences. We are looking for common section headers and saving the sentence numbers for that section in that document.
# 

headingsDict = defaultdict(dict)

for sent in sentObjList:
    if len(sent.wordList) == 1:
        #print(sent.wordList)
        word = string.upper(sent.wordList[0]).strip()
        if word == 'INTRODUCTION' or word == 'BACKGROUND':
            headingsDict[sent.document]["introduction"] = sent.sentenceNumber
        elif word == 'METHODS':
            headingsDict[sent.document]["methods"] = sent.sentenceNumber
        elif word == 'RESULTS':
            headingsDict[sent.document]["results"] = sent.sentenceNumber
        elif word == 'DISCUSSION':
            headingsDict[sent.document]["discussion"] = sent.sentenceNumber
        elif word == 'CONCLUSION':
            headingsDict[sent.document]["conclusion"] = sent.sentenceNumber
        elif word == 'REFERENCES':
            headingsDict[sent.document]["references"] = sent.sentenceNumber
        

headingsDict.keys()


# 
# Now the sentences need to be tagged to their appropriate section and concatenated into one string per section per document.
# 
# The sentences are assigned a section by whichever section they are closest to (that is less than their sentence number). For example, if introduction had sentence number 5 and methods had sentence number 25, sentence number 20 would be assigned to introduction.
# 
# This is done for each sentence in each document and joined by spaces into a one string per section per document. Finally, only the documents that contain an introduction, discussion, and conclusion are kept and put into the validDocsDict dictionary
# 

documentDict = defaultdict(list)
docPartsDict = defaultdict(lambda : defaultdict(list))
docPartsCombinedDict = defaultdict(dict)

for item in sentObjList:
    documentDict[item.document].append(item)
    
for document in documentDict.keys():
    docSentList = documentDict[document]
    introNum = int(headingsDict[document].get("introduction", -1))
    methoNum = int(headingsDict[document].get("methods", -1))
    resultNum = int(headingsDict[document].get("results", -1))
    discussNum = int(headingsDict[document].get("discussion", -1))
    conclusionNum = int(headingsDict[document].get("conclusion", -1))
    refNum = int(headingsDict[document].get("references", -1))

    for sent in docSentList:
        label = "noSection"
        dist = int(sent.sentenceNumber)
        sentNumber = int(sent.sentenceNumber)
        
        if dist > sentNumber - introNum and sentNumber - introNum > 0:
            label = "introduction"
            dist = sentNumber - introNum
        if dist > sentNumber - methoNum and sentNumber - methoNum > 0:
            label = "methods"
            dist = sentNumber - methoNum
        if dist > sentNumber - resultNum and sentNumber - resultNum > 0:
            label = "results"
            dist = sentNumber - resultNum
        if dist > sentNumber - discussNum and sentNumber - discussNum > 0:
            label = "discussion"
            dist = sentNumber - discussNum
        if dist > sentNumber - conclusionNum and sentNumber - conclusionNum > 0:
            label = "conclusion"
            dist = sentNumber - conclusionNum
        if dist > sentNumber - refNum and sentNumber - refNum > 0:
            label = "references"
            dist = sentNumber - refNum
        if sent.sentence.strip().lower() not in ["introduction", "methods", "results", "discussion", "conclusion", "references"]:
            docPartsDict[document][label].append(sent)
    
    for x in docPartsDict[document].keys():
        docPartsCombinedDict[document][x] = " ".join(y.sentence for y in sorted(docPartsDict[document][x], key=lambda z: z.sentenceNumber))

validDocsDict = defaultdict(dict)

for doc in docPartsCombinedDict.keys():
    tempKeys = docPartsCombinedDict[doc].keys()
    if 'introduction' in tempKeys and 'discussion' in tempKeys and 'conclusion' in tempKeys:
        validDocsDict[doc] = docPartsCombinedDict[doc]

print(str(len(docPartsCombinedDict.keys())))
print(str(len(validDocsDict.keys())))


# 
# Take the valid documents in the validDocsDict and output to a pickle file with the key part_docid with the part being introduction, methods, etc. and the docid allowing for document tracking.
# 

outputDict = dict()
for doc in validDocsDict.keys():
    for part in validDocsDict[doc].keys():
        outputDict[part + "_" + doc] = validDocsDict[doc][part]

pickle.dump(outputDict, open("TestDocsPub_kimtest.p", "wb"))





