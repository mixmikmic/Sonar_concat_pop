# ## Create A Long Number
# 

annual_revenue = 9282904.9282872782


# ## Format Number
# 

# Format rounded to two decimal places
format(annual_revenue, '0.2f')


# Format with commas and  rounded to one decimal place
format(annual_revenue, '0,.1f')


# Format as scientific notation
format(annual_revenue, 'e')


# Format as scientific notation rounded to two deciminals
format(annual_revenue, '0.2E')


# Import modules
import pandas as pd


# Example dataframe
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
df


# ## Create one column as a function of two columns
# 

# Create a function that takes two inputs, pre and post
def pre_post_difference(pre, post):
    # returns the difference between post and pre
    return post - pre


# Create a variable that is the output of the function
df['score_change'] = pre_post_difference(df['preTestScore'], df['postTestScore'])

# View the dataframe
df


# ## Create two columns as a function of one column
# 

# Create a function that takes one input, x
def score_multipler_2x_and_3x(x):
    # returns two things, x multiplied by 2 and x multiplied by 3
    return x*2, x*3


# Create two new variables that take the two outputs of the function
df['post_score_x2'], df['post_score_x3'] = zip(*df['postTestScore'].map(score_multipler_2x_and_3x))
df


# ## Slicing Arrays
# 

# ### Explanation Of Broadcasting
# 

# Unlike many other data types, slicing an array into a new variable means that any chances to that new variable are broadcasted to the original variable. Put other way, a slice is a hotlink to the original array variable, not a seperate and independent copy of it.
# 

# Import Modules
import numpy as np


# Create an array of battle casualties from the first to the last battle
battleDeaths = np.array([1245, 2732, 3853, 4824, 5292, 6184, 7282, 81393, 932, 10834])


# Divide the array of battle deaths into start, middle, and end of the war
warStart = battleDeaths[0:3]; print('Death from battles at the start of war:', warStart)
warMiddle = battleDeaths[3:7]; print('Death from battles at the middle of war:', warMiddle)
warEnd = battleDeaths[7:10]; print('Death from battles at the end of war:', warEnd)


# Change the battle death numbers from the first battle
warStart[0] = 11101


# View that change reflected in the warStart slice of the battleDeaths array
warStart


# View that change reflected in (i.e. "broadcasted to) the original battleDeaths array
battleDeaths


# ## Indexing Arrays
# 

# Note: This multidimensional array behaves like a dataframe or matrix (i.e. columns and rows)
# 

# Create an array of regiment information
regimentNames = ['Nighthawks', 'Sky Warriors', 'Rough Riders', 'New Birds']
regimentNumber = [1, 2, 3, 4]
regimentSize = [1092, 2039, 3011, 4099]
regimentCommander = ['Mitchell', 'Blackthorn', 'Baker', 'Miller']

regiments = np.array([regimentNames, regimentNumber, regimentSize, regimentCommander])
regiments


# View the first column of the matrix
regiments[:,0]


# View the second row of the matrix
regiments[1,]


# View the top-right quarter of the matrix
regiments[:2,2:]


# ### Create a dictionary
# 

dict = {'county': ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'fireReports': [4, 24, 31, 2, 3]}


# ### Create a list from the dictionary keys
# 

# Create a list of keys
list(dict.keys())


# ### Create a list from the  dictionary values
# 

# Create a list of values
list(dict.values())


# **Python 3 has three string types**
# 
# - str() is for unicode
# - bytes() is for binary data
# - bytesarray() mutable variable of bytes
# 

# ### Create some simulated text.
# 

string = 'The quick brown fox jumped over the lazy brown bear.'


# ### Capitalize the first letter.
# 

string_capitalized = string.capitalize()
string_capitalized


# ### Center the string with periods on either side, for a total of 79 characters
# 

string_centered = string.center(79, '.')
string_centered


# ### Count the number of e's between the fifth and last character
# 

string_counted = string.count('e', 4, len(string))
string_counted


# ### Locate any e's between the fifth and last character
# 

string_find = string.find('e', 4, len(string))
string_find


# ### Are all characters are alphabet?
# 

string_isalpha = string.isalpha()
string_isalpha


# ### Are all characters digits?
# 

string_isdigit = string.isdigit()
string_isdigit


# ### Are all characters lower case?
# 

string_islower = string.islower()
string_islower


# ### Are all chracters alphanumeric?
# 

string_isalnum = string.isalnum()
string_isalnum


# ### Are all characters whitespaces?
# 

string_isalnum = string.isspace()
string_isalnum


# ### Is the string properly titlespaced?
# 

string_istitle = string.istitle()
string_istitle


# ### Are all the characters uppercase?
# 

string_isupper = string.isupper()
string_isupper


# ### Return the lengths of string
# 

len(string)


# ### Convert string to lower case
# 

string_lower = string.lower()
string_lower


# ### Convert string to lower case
# 

string_upper = string.upper()
string_upper


# ### Convert string to title case
# 

string_title = string.title()
string_title


# ### Convert string the inverted case
# 

string_swapcase = string.swapcase()
string_swapcase


# ### Remove all leading whitespaces (i.e. to the left)
# 

string_lstrip = string.lstrip()
string_lstrip


# ### Remove all leading and trailing whitespaces (i.e. to the left and right)
# 

string_strip = string.strip()
string_strip


# ### Remove all trailing whitespaces (i.e. to the right)
# 

string_rstrip = string.rstrip()
string_rstrip


# ### Replace lower case e's with upper case E's, to a maximum of 4
# 

string_replace = string.replace('e', 'E', 4)
string_replace


# In this snippet we take a list and break it up into n-size chunks. This is a very common practice when dealing with APIs that have a maximum request size.
# 
# Credit for this nifty function goes to Ned Batchelder who [posted it on StackOverflow](http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python).
# 

# Create a list of first names
first_names = ['Steve', 'Jane', 'Sara', 'Mary','Jack','Bob', 'Bily', 'Boni', 'Chris','Sori', 'Will', 'Won','Li']


# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


# Create a list that from the results of the function chunks:
list(chunks(first_names, 5))


# Create an list of items denoting the number of soldiers in each regiment, view the list
regimentSize = (5345, 6436, 3453, 2352, 5212, 6232, 2124, 3425, 1200, 1000, 1211); regimentSize


# ## One-line Method
# 
# This line of code does the same thing as the multiline method below, it is just more compact (but also more complicated to understand.
# 

# Create a list called smallRegiments that filters regimentSize to 
# find all items that fulfill the lambda function (which looks for all items under 2500).
smallRegiments = list(filter((lambda x: x < 2500), regimentSize)); smallRegiments


# ## Multi-line Method
# 
# The ease with interpreting what is happening, I've broken down the one-line filter method into multiple steps, one per line of code. This appears below.
# 

# Create a lambda function that looks for things under 2500
lessThan2500Filter = lambda x: x < 2500


# Filter regimentSize by the lambda function filter
filteredRegiments = filter(lessThan2500Filter, regimentSize)


# Convert the filter results into a list
smallRegiments = list(filteredRegiments)


# ## For Loop Equivalent
# 
# This for loop does the same as both methods above, except it uses a for loop.
# 

# ### Create a for loop that go through each item of a list and finds items under 2500
# 

# Create a variable for the results of the loop to be placed
smallRegiments_2 = []

# for each item in regimentSize,
for x in regimentSize:
    # look if the item's value is less than 2500
    if x < 2500:
        # if true, add that item to smallRegiments_2
        smallRegiments_2.append(x)


# View the smallRegiment_2 variable
smallRegiments_2


# ## Create Argument Objects
# 

# Create a dictionary of arguments
argument_dict = {'a':'Alpha', 'b':'Bravo'}

# Create a list of arguments
argument_list = ['Alpha', 'Bravo']


# ## Create A Simple Function
# 

# Create a function that takes two inputs
def simple_function(a, b):
    # and prints them combined
    return a + b


# ## Run the Function With Unpacked Arguments
# 

# Run the function with the unpacked argument dictionary
simple_function(**argument_dict)


# Run the function with the unpacked argument list
simple_function(*argument_list)


# ## Create A New File And Write To It
# 

# Create a file if it doesn't already exist
with open('file.txt', 'xt') as f:
    # Write to the file
    f.write('This file now exsits!')
    # Close the connection to the file
    f.close()


# ## Open The File And Read It
# 

# Open the file
with open('file.txt', 'rt') as f:
    # Read the data in the file
    data = f.read()
    # Close the connection to the file
    f.close()


# ## View The Contents Of The File
# 

# View the data in the file
data


# ## Delete The File
# 

# Import the os package
import os

# Delete the file
os.remove('file.txt')


# ### Import the random module
# 

import random


# ### Create a variable of the true number of deaths of an event
# 

deaths = 6


# ## Create a variable that is denotes if the while loop should keep running
# 

running = True


# ### while running is True
# 

while running:
    # Create a variable that randomly create a integer between 0 and 10.
    guess = random.randint(0,10)

    # if guess equals deaths,
    if guess == deaths:
        # then print this
        print('Correct!')
        # and then also change running to False to stop the script
        running = False
    # else if guess is lower than deaths
    elif guess < deaths:
        # then print this
        print('No, it is higher.')
    # if guess is none of the above
    else:
        # print this
        print('No, it is lower')


# By the output, you can see that the while script keeping generating guesses and checking them until guess matches deaths, in which case the script stops.
# 

# ### Create a string
# 

string = 'Strings are defined as ordered collections of characters.'


# ### Print the entire string
# 

string[:]


# ### Print the first three characters
# 

string[0:3]


# ### Print the first three characters
# 

string[:3]


# ### Print the last three characters
# 

string[-3:]


# ### Print the third to fifth character
# 

string[2:5]


# ### Print the first to the tenth character, skipping every other character
# 

string[0:10:2]


# ### Print the string in reverse
# 

string[::-1]


# ## Create Two Lists
# 

names = ['James', 'Bob', 'Sarah', 'Marco', 'Nancy', 'Sally']
ages = [42, 13, 14, 25, 63, 23]


# ## Iterate Over Both Lists At Once
# 

for name, age in zip(names, ages):
     print(name, age)


# ### Import the sys module
# 

import sys


# ### Print a string with 1 digit and one string.
# 

'This is %d %s bird!' % (1, 'dead')


# ### Print a dictionary based string
# 

'%(number)d more %(food)s' % {'number' : 1, 'food' : 'burger'}


# ### Print a string about my laptop.
# 

'My {1[kind]} runs {0.platform}'.format(sys, {'kind': 'laptop'})


# ## String Formatting Codes
# - %s string
# - %r repr string
# - %c character (integer or string)
# - %d decimal
# - %i integer
# - %x hex integer
# - %X same as X but with uppercase
# - %e floating point lowercase
# - %E floating point uppercase
# - %f floating point decimal lowercase
# - %F floating point decimal uppercase
# - %g floating point e or f
# - %G floating point E or F
# - %% literal %
# 

# ## Create A Function
# 

# Create a function that
def function(names):
    # For each name in a list of names
    for name in names:
        # Returns the name
        return name


# Create a variable of that function
students = function(['Abe', 'Bob', 'Christina', 'Derek', 'Eleanor'])


# Run the function
students


# Now we have a problem, we were only returned the name of the first student. Why? Because the function only ran the `for name in names` iteration once!
# 

# ## Create A Generator
# 
# A generator is a function, but instead of returning the `return`, instead returns an iterator. The generator below is exactly the same as the function above except I have replaced `return` with `yield` (which defines whether a function with a regular function or a generator function).
# 

# Create a generator that
def generator(names):
    # For each name in a list of names
    for name in names:
        # Yields a generator object
        yield name


# Same as above, create a variable for the generator
students = generator(['Abe', 'Bob', 'Christina', 'Derek', 'Eleanor'])


# Everything has been the same so far, but now things get interesting. Above when we ran `students` when it was a function, it returned one name. However, now that `students` refers to a generator, it yields a generator object of names!
# 

# Run the generator
students


# What can we do this a generator object? A lot! As a generator `students` will can each student in the list of students:
# 

# Return the next student
next(students)


# Return the next student
next(students)


# Return the next student
next(students)


# It is interesting to note that if we use list(students) we can see all the students still remaining in the generator object's iteration:
# 

# List all remaining students in the generator
list(students)


# ## Make Two Dictionaries
# 

importers = {'El Salvador' : 1234,
             'Nicaragua' : 152,
             'Spain' : 252
            }

exporters = {'Spain' : 252,
             'Germany' : 251,
             'Italy' : 1563
             }


# ## Find Duplicate Keys
# 

# Find the intersection of importers and exporters
importers.keys() & exporters.keys()


# ## Find Difference In Keys
# 

# Find the difference between importers and exporters
importers.keys() - exporters.keys()


# ## Find Key, Values Pairs In Common
# 

# Find countries where the amount of exports matches the amount of imports
importers.items() & exporters.items()


# ## Create Some Text 
# 

text = 'Chapter 1'


# ## Add Padding Around Text 
# 

# Add Spaces Of Padding To The Left
format(text, '>20')


# Add Spaces Of Padding To The Right
format(text, '<20')


# Add Spaces Of Padding On Each Side
format(text, '^20')


# Add * Of Padding On Each Side
format(text, '*^20')


# ## Create Sorted List
# 

sorted_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
print(sorted_list)


# ## Create Binary Search Algorithm
# 

def binary_search(sorted_list, target):

    '''This function inputs a sorted list and a target value to find and returns ....'''

    # Create variables for the index of the first and last elements
    start = 0
    end = len(sorted_list) - 1

    while end >= start:

        # Create a variable for the index of the middle term
        middle = start + (end - start) // 2  # // is integer division in Python 3.X

        # If the target value is less than the middle value of the search area
        if target < sorted_list[middle]:
            # Cut the list in half by making the new end value the old middle value minus 1
            # The minus one is because we already ruled the middle value out, so we can ignore it
            end = middle - 1

        #  Else, if the target value is greater than the middle value of the search area
        elif target > sorted_list[middle]:
            # Cut the list in half by making the new start value the old middle value plus 1
            # The plus one is because we already ruled the middle value out, so we can ignore it
            start = middle + 1

        # If it's not too high or too low, it must be just right, return the location
        else:
            return ("Found it at index: {}".format(middle))

    # If we've fallen out of the while loop the target value is not in the list
    return print("Not in list!")


# ## Conduct Binary Search
# 

# Run binary search
binary_search(sorted_list, 2)


# Thanks for [Julius](https://github.com/jss367) for the improved code.
# 

# Create a list of length 3:
armies = ['Red Army', 'Blue Army', 'Green Army']

# Create a list of length 4:
units = ['Red Infantry', 'Blue Armor','Green Artillery','Orange Aircraft']


# For each element in the first list,
for army, unit in zip(armies, units):
    # Display the corresponding index element of the second list:
    print(army, 'has the following options:', unit)


# Notice that the fourth item of the second list, `orange aircraft`, did not display.
# 

# While there are a number of good libraries out there, OpenCV is the most popular and documented library for handling images. One of the biggest hurdles to using OpenCV is installing it. However, fortunately we can use Anaconda's package manager tool conda to install OpenCV in a single line of code in our terminal: 
# 
# `conda install --channel https://conda.anaconda.org/menpo opencv3`
# 
# Afterwards, we can check the installation by opening a notebook, importing OpenCV, and checking the version number (3.1.0):
# 

# Load library
import cv2

# View version number
cv2.__version__


# ## Create List Of Tuples
# 

# Create a list of tuples where the first and second element of each 
# super is the first last names, respectively
soldiers = [('Steve', 'Miller'), ('Stacy', 'Markov'), ('Sonya', 'Matthews'), ('Sally', 'Mako')]


# ## Unpack Tuples
# 

# For the second element for each tuple in soldiers,
for _, last_name in soldiers:
    # print the second element
    print(last_name)


# Bessel's correction is the reason we use $n-1$ instead of $n$ in the calculations of sample variance and sample standard deviation.
# 
# Sample variance:
# 
# $$ s^2 = \frac {1}{n-1} \sum\_{i=1}^n  \left(x\_i - \overline{x} \right)^ 2 $$
# 
# When we calculate sample variance, we are attempting to estimate the population variance, an unknown value. To make this estimate, we estimate this unknown population variance from the mean of the squared deviations of samples from the overall sample mean. A negative sideffect of this estimation technique is that, because we are taking a sample, we are a more likely to observe observations with a smaller deviation because they are more common (e.g. they are the center of the distribution). The means that by definiton we will underestimate the population variance.
# 
# Friedrich Bessel figured out that by multiplying a biased (uncorrected) sample variance $s\_n^2 = \frac {1}{n} \sum\_{i=1}^n  \left(x\_i - \overline{x} \right)^ 2$ by $\frac{n}{n-1}$ we will be able to reduce that bias and thus be able to make an accurate estimate of the population variance and standard deviation. The end result of that multiplication is the unbiased sample variance.
# 

# ## Create some raw text
# 

# Create a list of three strings.
incoming_reports = ["We are attacking on their left flank but are losing many men.", 
               "We cannot see the enemy army. Nothing else to report.", 
               "We are ready to attack but are waiting for your orders."]


# ## Seperate by word
# 

# import word tokenizer
from nltk.tokenize import word_tokenize

# Apply word_tokenize to each element of the list called incoming_reports
tokenized_reports = [word_tokenize(report) for report in incoming_reports]

# View tokenized_reports
tokenized_reports


# Import regex
import re

# Import string
import string


regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

tokenized_reports_no_punctuation = []

for review in tokenized_reports:
    
    new_review = []
    for token in review: 
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    
    tokenized_reports_no_punctuation.append(new_review)
    
tokenized_reports_no_punctuation


# ## Remove filler words
# 

from nltk.corpus import stopwords

tokenized_reports_no_stopwords = []
for report in tokenized_reports_no_punctuation:
    new_term_vector = []
    for word in report:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    tokenized_reports_no_stopwords.append(new_term_vector)
            
tokenized_reports_no_stopwords


# Often I need or want to change the case of all items in a column of strings (e.g. BRAZIL to Brazil, etc.). There are many ways to accomplish this but I have settled on this one as the easiest and quickest.
# 

# Import pandas
import pandas as pd

# Create a list of first names
first_names = pd.Series(['Steve Murrey', 'Jane Fonda', 'Sara McGully', 'Mary Jane'])


# print the column
first_names


# print the column with lower case
first_names.str.lower()


# print the column with upper case
first_names.str.upper()


# print the column with title case
first_names.str.title()


# print the column split across spaces
first_names.str.split(" ")


# print the column with capitalized case
first_names.str.capitalize()


# You get the idea. Many more string methods are [avaliable here](https://docs.python.org/3.5/library/stdtypes.html#string-methods)
# 

# ## Create a list of names
# 

commander_names = ["Alan Brooke", "George Marshall", "Frank Jack Fletcher", "Conrad Helfrich", "Albert Kesselring"] 


# ## Sort Alphabetically By Length
# 

# To complete the sort, we will combine two operations:
# 
# - `lambda x: len(x)`, which returns the length of each string.
# - `sorted()`, which sorts a list.
# 

# Sort a variable called 'commander_names' by the length of each string
sorted(commander_names, key=lambda x: len(x))


# ## Create A List
# 

# Create a list:
armies = ['Red Army', 'Blue Army', 'Green Army']


# ## Breaking Out Of A For Loop
# 

for army in armies:
    print(army)
    if army == 'Blue Army':
        print('Blue Army Found! Stopping.')
        break


# Notice that the loop stopped after the conditional if statement was satisfied.
# 

# ## Exiting If Loop Completed
# 

# A loop will exit when completed, but using an `else` statement we can add an action at the conclusion of the loop if it hasn't been exited earlier.
# 

for army in armies:
    print(army)
    if army == 'Orange Army':
        break
else:
    print('Looped Through The Whole List, No Orange Army Found')


# ### Create a variable with the status of the conflict.
# 
# - 1 if the conflict is active
# - 0 if the conflict is not active
# - unknown if the status of the conflict is unknwon
# 

conflict_active = 1


# ### If the conflict is active print a statement
# 

if conflict_active == 1:
    print('The conflict is active.')


# ### If the conflict is active print a statement, if not, print a different statement
# 

if conflict_active == 1:
    print('The conflict is active.')
else:
    print('The conflict is not active.')


# ### If the conflict is active print a statement, if not, print a different statement, if unknown, state a third statement.
# 

if conflict_active == 1:
    print('The conflict is active.')
elif conflict_active == 'unknown':
    print('The status of the conflict is unknown')
else:
    print('The conflict is not active.')


# ## Preliminary
# 

# Import combinations with replacements from itertools
from itertools import combinations_with_replacement


# ## Create a list of objects
# 

# Create a list of objects to combine
list_of_objects = ['warplanes', 'armor', 'infantry']


# ## Find all combinations (with replacement) for the list
# 

# Create an empty list object to hold the results of the loop
combinations = []

# Create a loop for every item in the length of list_of_objects, that,
for i in list(range(len(list_of_objects))):
    # Finds every combination (with replacement) for each object in the list
    combinations.append(list(combinations_with_replacement(list_of_objects, i+1)))
    
# View the results
combinations


# Flatten the list of lists into just a list
combinations = [i for row in combinations for i in row]

# View the results
combinations


# ### Import the random module
# 

import random


# ### Create a while loop
# 

# set running to true
running = True


# while running is true
while running:
    # Create a random integer between 0 and 5
    s = random.randint(0,5)
    # If the integer is less than 3
    if s < 3:
        # Print this
        print('It is too small, starting over.')
        # Reset the next interation of the loop
        # (i.e skip everything below and restart from the top)
        continue
    # If the integer is 4
    if s == 4:
        running = False
        # Print this
        print('It is 4! Changing running to false')
    # If the integer is 5,
    if s == 5:
        # Print this
        print('It is 5! Breaking Loop!')
        # then stop the loop
        break


# ## Create A Dictionary
# 

ages = {'John': 21,
        'Mike': 52,
        'Sarah': 12,
        'Bob': 43
       }


# ## Find The Maximum Value Of The Values
# 

max(zip(ages.values(), ages.keys()))


# Import modules
import pandas as pd


raw_data = {'first_name': ['Jason', 'Jason', 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', 'Miller', 'Ali', 'Milner', 'Cooze'], 
        'age': [42, 42, 36, 24, 73], 
        'preTestScore': [4, 4, 31, 2, 3],
        'postTestScore': [25, 25, 57, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
df


# ## Find where a value exists in a column
# 

# View preTestscore where postTestscore is greater than 50
df['preTestScore'].where(df['postTestScore'] > 50)


# ## Create a list of names
# 

commander_names = ["Alan Brooke", "George Marshall", "Frank Jack Fletcher", "Conrad Helfrich", "Albert Kesselring"] 


# ## Sort Alphabetically By Last Name
# 

# To complete the sort, we will combine three operations:
# 
# - `lambda x: x.split(" ")`, which is a function that takes a string `x` and breaks it up along each blank space. This outputs a list.
# - `[-1]`, which takes the last element of a list.
# - `sorted()`, which sorts a list.
# 

# Sort a variable called 'commander_names' by the last elements of each name.
sorted(commander_names, key=lambda x: x.split(" ")[-1])


# ### Setup the originally variables and their values
# 

one = 1
two = 2


# ### View the original variables
# 

'one =', one, 'two =', two


# ### Swap the values
# 

one, two = two, one


# ### View the swapped values, notice how the values for each variable have changed
# 

'one =', one, 'two =', two


# Geocoding (converting a physical address or location into latitude/longitude) and reverse geocoding (converting a lat/long to a physical address or location) are common tasks when working with geo-data.
# 
# Python offers a number of packages to make the task incredibly easy. In the tutorial below, I use pygeocoder, a wrapper for Google's geo-API, to both geocode and reverse geocode.
# 

# ## Preliminaries
# 
# First we want to load the packages we will want to use in the script. Specifically, I am loading pygeocoder for its geo-functionality, pandas for its dataframe structures, and numpy for its missing value (np.nan) functionality.
# 

# Load packages
from pygeocoder import Geocoder
import pandas as pd
import numpy as np


# ## Create some simulated geo data
# 
# Geo-data comes in a wide variety of forms, in this case we have a Python dictionary of five latitude and longitude strings, with each coordinate in a coordinate pair separated by a comma.
# 

# Create a dictionary of raw data
data = {'Site 1': '31.336968, -109.560959',
        'Site 2': '31.347745, -108.229963',
        'Site 3': '32.277621, -107.734724',
        'Site 4': '31.655494, -106.420484',
        'Site 5': '30.295053, -104.014528'}


# While technically unnecessary, because I originally come from R, I am a big fan of dataframes, so let us turn the dictionary of simulated data into a dataframe.
# 

# Convert the dictionary into a pandas dataframe
df = pd.DataFrame.from_dict(data, orient='index')


# View the dataframe
df


# You can see now that we have a a dataframe with five rows, with each now containing a string of latitude and longitude. Before we can work with the data, we'll need to 1) seperate the strings into latitude and longitude and 2) convert them into floats. The function below does just that.
# 

# Create two lists for the loop results to be placed
lat = []
lon = []

# For each row in a varible,
for row in df[0]:
    # Try to,
    try:
        # Split the row by comma, convert to float, and append
        # everything before the comma to lat
        lat.append(float(row.split(',')[0]))
        # Split the row by comma, convert to float, and append
        # everything after the comma to lon
        lon.append(float(row.split(',')[1]))
    # But if you get an error
    except:
        # append a missing value to lat
        lat.append(np.NaN)
        # append a missing value to lon
        lon.append(np.NaN)

# Create two new columns from lat and lon
df['latitude'] = lat
df['longitude'] = lon


# Let's take a took a what we have now.
# 

# View the dataframe
df


# Awesome. This is exactly what we want to see, one column of floats for latitude and one column of floats for longitude.
# 

# ## Reverse Geocoding
# 
# To reverse geocode, we feed a specific latitude and longitude pair, in this case the first row (indexed as '0') into pygeocoder's reverse_geocoder function. 
# 

# Convert longitude and latitude to a location
results = Geocoder.reverse_geocode(df['latitude'][0], df['longitude'][0])


# Now we can take can start pulling out the data that we want.
# 

# Print the lat/long
results.coordinates


# Print the city
results.city


# Print the country
results.country


# Print the street address (if applicable)
results.street_address


# Print the admin1 level
results.administrative_area_level_1


# ## Geocoding
# 
# For geocoding, we need to submit a string containing an address or location (such as a city) into the geocode function. However, not all strings are formatted in a way that Google's geo-API can make sense of them. We can text if an input is valid by using the .geocode().valid_address function.
# 

# Verify that an address is valid (i.e. in Google's system)
Geocoder.geocode("4207 N Washington Ave, Douglas, AZ 85607").valid_address


# Because the output was True, we now know that this is a valid address and thus can print the latitude and longitude coordinates.
# 

# Print the lat/long
results.coordinates


# But even more interesting, once the address is processed by the Google geo API, we can parse it and easily separate street numbers, street names, etc. 
# 

# Find the lat/long of a certain address
result = Geocoder.geocode("7250 South Tucson Boulevard, Tucson, AZ 85756")


# Print the street number
result.street_number


# Print the street name
result.route


# And there you have it. Python makes this entire process easy and inserting it into an analysis only takes a few minutes. Good luck!
# 

# ## Create Two Lists
# 

# Create a list of theofficer's name
officer_names = ['Sodoni Dogla', 'Chris Jefferson', 'Jessica Billars', 'Michael Mulligan', 'Steven Johnson']

# Create a list of the officer's army
officer_armies = ['Purple Army', 'Orange Army', 'Green Army', 'Red Army', 'Blue Army']


# ## Construct A Dictionary From The Two Lists
# 

# Create a dictionary that is the zip of the two lists
dict(zip(officer_names, officer_armies))


# In Python, it is possible to string lambda functions together.
# 

# ### Create a series, called pipeline, that contains three mini functions
# 

pipeline = [lambda x: x **2 - 1 + 5,
            lambda x: x **20 - 2 + 3,
            lambda x: x **200 - 1 + 4]


# ### For each item in pipeline, run the lambda function with x = 3
# 

for f in pipeline:
    print(f(3))


# ## Create A Dictionary
# 

Officers = {'Michael Mulligan': 'Red Army',
            'Steven Johnson': 'Blue Army',
            'Jessica Billars': 'Green Army',
            'Sodoni Dogla': 'Purple Army',
            'Chris Jefferson': 'Orange Army'}


Officers


# ## Use Dictionary Comprehension
# 

# Display all dictionary entries where the key doesn't start with 'Chris'
{keys : Officers[keys] for keys in Officers if not keys.startswith('Chris')}


# Notice that the entry for 'Chris Jefferson' is not returned.
# 

