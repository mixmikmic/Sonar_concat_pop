# ## Working with HTML
# 

import urllib.request
aliceUrl = "http://www.gutenberg.org/files/11/11-h/11-h.htm"
aliceString = urllib.request.urlopen(aliceUrl).read()


print(aliceString[:200]," ... ",aliceString[-200:]) # have a peek


from bs4 import BeautifulSoup
aliceSoup = BeautifulSoup(aliceString)
aliceSoupText = aliceSoup.text # this includes the head section
print(aliceSoupText[:200]," ... ",aliceSoupText[-200:]) # have a peek


aliceSoupBodyText = aliceSoup.body.text
print(aliceSoupBodyText[:200]," ... ",aliceSoupBodyText[-200:]) # have a peek


# ## Working With XML
# 

import nltk
rjFile = nltk.data.find("corpora/shakespeare/r_and_j.xml") # search for this text
rjFile


from xml.etree.ElementTree import ElementTree
rjXml = ElementTree().parse(rjFile) # parse the file
rjXml


speeches = rjXml.findall(".//SPEECH")
len(speeches)


speakers = rjXml.findall(".//SPEAKER")
len(speeches) # same thing, each speech has a speaker tag


speakerNames = [speaker.text for speaker in speakers]


print(set(speakerNames)) # unique speakers


get_ipython().magic('matplotlib inline')
nltk.FreqDist(speakerNames).plot(20) # how many speeches for each speaker


uniqueSpeakerNames = list(set(speakerNames))
print(uniqueSpeakerNames)
# let's look at names that aren't the non-string None and aren't all uppercase
titleCaseSpeakerNames = [name for name in uniqueSpeakerNames if name != None and name != name.upper()]
nltk.Text(speakerNames).dispersion_plot(titleCaseSpeakerNames)


# let's create a dictionary with each speaker pointing to text from that speaker
speakersDict = nltk.defaultdict(str)
speeches = rjXml.findall(".//SPEECH")
for speech in speeches:
    speaker = speech.find("SPEAKER").text
    for line in speech.findall("LINE"):
        if line.text:
            speakersDict[speaker]+=line.text+"\n"

# now let's look at speech length for each speaker (different from number of speeches)
speakersLengthsDict = {}
for speaker, text in speakersDict.items():
    speakersLengthsDict[speaker]=len(text)

nltk.FreqDist(speakersLengthsDict).plot(20)


# let's look at how often Romeo and Juliet say "love" and which words are nearby
romeoTokens = nltk.word_tokenize(speakersDict["ROMEO"])
print(romeoTokens.count("love")/len(romeoTokens))
nltk.Text(romeoTokens).similar("love")
julietTokens = nltk.word_tokenize(speakersDict["JULIET"])
print(julietTokens.count("love")/len(julietTokens))
nltk.Text(julietTokens).similar("love")





# # Getting Collocates for Words
# 
# This shows how you can get collocates for a word. 
# 
# **Note:** It assumes you have the text in the same directory.
# 

# ### Checking files and getting a text
# 
# First we check what texts we have in the directory.
# 

get_ipython().magic('ls *.txt')


# Now we open the text. 
# 
# Copy in the title of the text you want to process and run the next cell. Our example uses the plain text version of Hume's [A Treatise of Human Nature by David Hume](http://www.gutenberg.org/ebooks/4705) that we downloaded.
# 

targetText = "Hume Treatise.txt"

with open(targetText, "r") as f:
    theText = f.read()

print("This string has", "{:,}".format(len(theText)), "characters")


# ### Tokenizing
# 
# Note that we are tokenizing the full Gutenberg text file which includes metadata and license information. If you want only the tokens of the book you should delete the Gutenberg information from the text file.
# 

import re
theTokens = re.findall(r'\b\w[\w-]*\b', theText.lower())
print(theTokens[:10])


# ### Finding the collocates
# 
# This will ask you what word you want collocates for create a list off collocates. Note that you can set the number of words of context.
# 

wrd2find = input("What word do you want collocates for?") # Ask for the word to search for
context = 5 # This sets the context of words on either side to grab

end = len(theTokens)
counter = 0
theCollocates = []
for word in theTokens:
    if word == wrd2find: # This checks to see if the word is what we want
        for i in range(context):
            if (counter - (i + 1)) >= 0: # This checks that we aren't at the beginning
                theCollocates.append(theTokens[(counter - (i + 1))]) # This adds words before
            if (counter + (i + 1)) < end: # This checks that we aren't at the end
                theCollocates.append(theTokens[(counter + (i + 1))]) # This adds words afte
    counter = counter + 1
    
print(theCollocates[:10])


# ### Doing things with the collocates
# 
# Now we can do various things with the list of collocates.
# 

# #### Count collocates
# 

print(len(theCollocates))


# #### Count unique words among collocates
# 

print(set(theCollocates))


# #### Tabulate top collocates
# 

import nltk
tokenDist = nltk.FreqDist(theCollocates)
tokenDist.tabulate(10)


# #### Plot top collocates
# 

import matplotlib
get_ipython().magic('matplotlib inline')
tokenDist.plot(25, title="Top Frequency Collocates for " + wrd2find.capitalize())


# #### Explort all collocates and counts as a CSV
# 
# This will create CSV file with the name of the target word with the counts.
# 

import csv
nameOfResults = wrd2find.capitalize() + ".Collocates.csv"
table = tokenDist.most_common()

with open(nameOfResults, "w") as f:
    writer = csv.writer(f)
    writer.writerows(table)
    
print("Done")


# ---
# [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) From [The Art of Literary Text Analysis](ArtOfLiteraryTextAnalysis.ipynb) by [Stéfan Sinclair](http://stefansinclair.name) &amp; [Geoffrey Rockwell](http://geoffreyrockwell.com)<br >Created August 8, 2014  (Jupyter 4.2.1)
# 




# # Getting Graphical
# 

# This notebook introduces graphical output for text processing. It's part of the [The Art of Literary Text Analysis](ArtOfLiteraryTextAnalysis.ipynb) and assumes that you've already worked through previous notebooks ([Getting Setup](GettingSetup.ipynb), [Getting Started](GettingStarted.ipynb),  [Getting Texts](GettingTexts.ipynb) and [Getting NLTK](GettingNltk.ipynb)). In this notebook we'll look in particular at
# 
# * plotting high frequency terms
# * plotting a characteristic curve of word lengths
# * plotting a distribution graph of terms
# 

# ## Graphing in Jupyter
# 

# The Anaconda bundle already comes with the Matlablib library, so nothing further to install.
# 
# However, there's a very important step needed when graphing with iPython, you need to instruct the kernel to produce graphs inline the first time you generate a graph in a notebook. That's accomplished with this code:
# 
# > %matplotlib inline
# 
# If ever you forget to do that, your notebook might become unresponsive and you'll need to shutdown the kernel and start again. Even that's not a big deal, but best to avoid it.
# 
# We can test simple graphing in a new notebook (let's call it `GettingGraphical`) to make sure that everything is working. The syntax below also shows how we can create a shorthand name for a library so that instead of always writing ```matplotlib.pyplot``` we can simply write ```plt```.
# 

import matplotlib.pyplot as plt

# make sure that graphs are embedded into our notebook output
get_ipython().magic('matplotlib inline')

plt.plot() # create an empty graph


# Wow, who knew such insights were possible with ipython, eh? :)
# 

# ## Plotting Word Frequency
# 

# The previous notebook on [Getting NLTK](GettingNltk.ipynb) explained the basics of tokenization, filtering, listing frequencies and concordances. If you need to recapitulate the essentials of the previous notebook, try running this:
# 
# ```python
# import urllib.request
# # retrieve Poe plain text value
# poeUrl = "http://www.gutenberg.org/cache/epub/2147/pg2147.txt"
# poeString = urllib.request.urlopen(poeUrl).read().decode()```
# 
# And then this, in a separate cell so that we don't read repeatedly from Gutenberg:
# 
# ```python
# import os
# # isolate The Gold Bug
# start = poeString.find("THE GOLD-BUG")
# end = poeString.find("FOUR BEASTS IN ONE")
# goldBugString = poeString[start:end]
# # save the file locally
# directory = "data"
# if not os.path.exists(directory):
#     os.makedirs(directory)
# with open("data/goldBug.txt", "w") as f:
#     f.write(goldBugString)```
# 
# Let's pick up where we left off by (re)reading our _Gold Bug_ text, tokenizing, filtering to keep only words, calculating frequencies and showing a table of the top frequency words. We'd previously created a filtered list that removed stopwords (very common syntactic words that don't carry much meaning), but for now we'll keep all words.
# 

import nltk

# read Gold Bug plain text into string
with open("data/goldBug.txt", "r") as f:
    goldBugString = f.read()

# simple lowercase tokenize
goldBugTokensLowercase = nltk.word_tokenize(goldBugString.lower())

# filter out tokens that aren't words
goldBugWordTokensLowercase = [word for word in goldBugTokensLowercase if word[0].isalpha()]

# determine frequencies
goldBugWordTokensLowercaseFreqs = nltk.FreqDist(goldBugWordTokensLowercase)

# preview the top 20 frequencies
goldBugWordTokensLowercaseFreqs.tabulate(20)


# This table is useful for ranking the top frequency terms (from left to right), though it's difficult to get a sense from the numbers of how the frequencies compare. Do the numbers drop gradually, precipitously or irregularly? This is a perfect scenario for experimenting with visualization by producing a simple graph.
# 
# In addition to the ```tabulate()``` function, the frequencies (FreqDist) object that we created as a ```plot()``` function that conveniently plots a graph of the top frequency terms. Again, in order to embed a graph in the output of an iPython Notebook we need to give the following special instruction: ```%matplotlib inline```. It's ok to repeat this several time in a notebook, but like an ```import``` statement, we really just need to do this once for the first cell in the notebook where it's relevant.
# 

# make sure that graphs are embedded into our notebook output
get_ipython().magic('matplotlib inline')

# plot the top frequency words in a graph
goldBugWordTokensLowercaseFreqs.plot(25, title="Top Frequency Word Tokens in Gold Bug")


# This graph shows not only the rank of the words (along the bottom x axis), but is also much more effective than the table at showing the steep decline in frequncy as we move away from the first words. This is actually a well-known phenomenon with natural language and is described by Zipf's law, which the [Wikipedia article](http://en.wikipedia.org/wiki/Zipf's_law) nicely summarizes:
# 
# > Zipf's law states that given some corpus of natural language utterances, the frequency of any word is inversely proportional to its rank in the frequency table. Thus the most frequent word will occur approximately twice as often as the second most frequent word, three times as often as the third most frequent word, etc. 
# 
# As we continue to explore frequency of words, it's useful to keep in mind the distinction between frequency rank and the actual number of words (tokens) that each word form (type) is contributing.
# 

# ## The Characteristic Curve of Word Lengths
# 

# One of the first examples we have of quantitative stylistics (text analysis) is an 1887 study by T.C. Mendenhall who manually counted the length of words and used that to suggest that authors had a distinctive stylistic signature, based on the average word length of their writings. In some ways this is similar to the type/token ratio we saw in the previous notebook, as it tries to measure stylistic features of texts without considering (yet) what the words may mean. It also uses all words, even the function words that authors are maybe using less deliberately. Unlike with the type/token ratios, Mendenhall's Characteristic Curve is less sensitive to changes in the total text length. If an author uses relatively longer words, chances are that style will persist throughout a text (which is different from comparing type/token ratios for a text of 1,000 words or 100,000 words).
# 
# To calculate the frequencies of terms, we can start by replacing each word in our tokens list with the length of that word. So, instead of this:
# 
# ```python
# [word for word in tokens]```
# 
# we have this:
# 
# ```python
# [len(word) for word in tokens]```
# 

goldBugLowerCaseWordTokenLengths = [len(w) for w in goldBugWordTokensLowercase]
print("first five words: ", goldBugWordTokensLowercase[:5])
print("first five word lengths: ", goldBugLowerCaseWordTokenLengths[:5])


# That looks right, "the" is 3 letters, "gold-bug" is 8, etc.
# 
# Now, just as we counted the frequencies of repeating words, we can count the frequencies of repeating word lengths.
# 

nltk.FreqDist(goldBugLowerCaseWordTokenLengths).plot()


# That was easy, but not really what we want, since the word lengths on the bottom axis are ordered by frequency (3 is the most common word length, followed by 2, and then 4). The default behaviour of ordering by frequency was useful for words, but not as useful here if we want to order by word length.
# 
# To accomplish what we want, we'll extract items from the frequency list, which provides a sorting by key (by word length), and then create a list from that.
# 
# 

goldBugLowerCaseWordTokenLengthFreqs = list(nltk.FreqDist(goldBugLowerCaseWordTokenLengths).items())
goldBugLowerCaseWordTokenLengthFreqs # sorted by word length (not frequency)


# Formally, this is a list of tuples where each line represents an item in the list and within each line item there's a fixed-order tuple of two numbers, the first for the word length and the second for the frequency. Since lists don't have a built-in ```plot()``` function – unlike FreqDist that we used previously to plot high frequency words – we need to call the graphing library directly and plot the x (word lengths) and y (frequencies).
# 

import matplotlib.pyplot as plt

goldBugLowerCaseWordTokenWordLengths = [f[0] for f in goldBugLowerCaseWordTokenLengthFreqs]
goldBugLowerCaseWordTokenWordLengthValues = [f[1] for f in goldBugLowerCaseWordTokenLengthFreqs]
plt.plot(goldBugLowerCaseWordTokenWordLengths, goldBugLowerCaseWordTokenWordLengthValues)
plt.xlabel('Word Length')
plt.ylabel('Word Count')


# That's pretty darn close to what some of Mendenhall's graphs looked like, such as this one for the first thousand words of _Oliver Twist_:
# 
# ![Characteristic Curve](images/characteristic-curve-mendenhall.png)
# 
# Thank goodness we didn't need to count tens of thousands of tokens by hand (an error-prone process) like Mendenhall did!
# 
# On its own one characteristic curve isn't terribly useful since the point is to compare an author's curve with another, but for now at least we know we can fairly easily generate the output for one text. For now, let's shift back to working with words.

# ## Graphing Distribution
# 

# As we saw in the previous notebooks, sometimes it's useful to work with all word tokens (like when measuring Zipf's Law or aggregate word length, but typically we need to strip out function words to start studying the meaning of texts. Let's recapitulate the filtering steps.
# 

stopwords = nltk.corpus.stopwords.words("English")
goldBugContentWordTokensLowercase = [word for word in goldBugWordTokensLowercase if word not in stopwords]
goldBugContentWordTokensLowercaseFreqs = nltk.FreqDist(goldBugContentWordTokensLowercase)
goldBugContentWordTokensLowercaseFreqs.most_common(20)


# Or, now that we've done some plotting, we could graph the top frequency content terms, though it may be harder to read the words.
# 

goldBugContentWordTokensLowercaseFreqs.plot(20, title="Top Frequency Content Terms in Gold Bug")


# As we'd noticed in the last notebook, the words "jupiter" and "legrand" are suspiciously high frequency (for relatively uncommon words), which may suggest that they're being used as character names in the story. We can regenerate our NLTK text object and ask for concordances of each to confirm this hypothesis. To help differentiate between upper and lowercase words, we'll re-tokenize the text and not perform any case alternation or filtering.
# 

goldBugText = nltk.Text(nltk.word_tokenize(goldBugString))
goldBugText.concordance("jupiter", lines=5)
goldBugText.concordance("legrand", lines=5)


# Are these two character names present throughout the story? One way to get a quick sense is to create a dispersion plot, which is essentially a distribution graph of occurrences. Note that [dispersion_plot()](http://www.nltk.org/api/nltk.html?highlight=dispersion_plot#nltk.text.Text.dispersion_plot) takes a list of words as an argument, but that the words are case-sensitive (unlike the ```concordance()``` function). Since case matters, for other purposes it might have been preferable to use the lowercase tokens instead.
# 

goldBugText.dispersion_plot(["Jupiter", "Legrand"])


# This graph suggests that there are many more occurrences of the character names in the first half of the text. This doesn't necessarily mean that the characters are not as present in the second half, but their names appear less often (perhaps because of a shift in dialogue structure or narrative focus).
# 

# ## Next Steps
# 

# Here are some tasks to try:
# 
# * generate a simple list of the top 20 frequency lowercase content terms (without counts, just the terms)
# * create a dispersion plot of these terms, do any other stand out as irregularly distributed?
# * try the command [goldBugText.collocations()](http://www.nltk.org/api/nltk.html?highlight=text#nltk.text.Text.collocations) – what does this do? how might it be useful?
# 
# In the next notebook we're going to look at more powerful ways of matching terms and [Searching Meaning](SearchingMeaning.ipynb).

# ---
# [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) From [The Art of Literary Text Analysis](ArtOfLiteraryTextAnalysis.ipynb) by [Stéfan Sinclair](http://stefansinclair.name) &amp; [Geoffrey Rockwell](http://geoffreyrockwell.com)<br >Created January 27, 2015 and last modified December 9, 2015 (Jupyter 4)
# 

