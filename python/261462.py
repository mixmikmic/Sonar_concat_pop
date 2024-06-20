# ## 1. The NIST Special Publication 800-63B
# <p>If you – 50 years ago – needed to come up with a secret password you were probably part of a secret espionage organization or (more likely) you were pretending to be a spy when playing as a kid. Today, many of us are forced to come up with new passwords <em>all the time</em> when signing into sites and apps. As a password <em>inventeur</em> it is your responsibility to come up with good, hard-to-crack passwords. But it is also in the interest of sites and apps to make sure that you use good passwords. The problem is that it's really hard to define what makes a good password. However, <em>the National Institute of Standards and Technology</em> (NIST) knows what the second best thing is: To make sure you're at least not using a <em>bad</em> password. </p>
# <p>In this notebook, we will go through the rules in <a href="https://pages.nist.gov/800-63-3/sp800-63b.html">NIST Special Publication 800-63B</a> which details what checks a <em>verifier</em> (what the NIST calls a second party responsible for storing and verifying passwords) should perform to make sure users don't pick bad passwords. We will go through the passwords of users from a fictional company and use python to flag the users with bad passwords. But us being able to do this already means the fictional company is breaking one of the rules of 800-63B:</p>
# <blockquote>
#   <p>Verifiers SHALL store memorized secrets in a form that is resistant to offline attacks. Memorized secrets SHALL be salted and hashed using a suitable one-way key derivation function.</p>
# </blockquote>
# <p>That is, never save users' passwords in plaintext, always encrypt the passwords! Keeping this in mind for the next time we're building a password management system, let's load in the data.</p>
# <p><em>Warning: The list of passwords and the fictional user database both contain <strong>real</strong> passwords leaked from <strong>real</strong> websites. These passwords have not been filtered in any way and include words that are explicit, derogatory and offensive.</em></p>
# 

# Importing the pandas module
import pandas as pd

# Loading in datasets/users.csv 
users = pd.read_csv('datasets/users.csv')

# Printing out how many users we've got
print(users)

# Taking a look at the 12 first users
users.head(12)


# ## 2. Passwords should not be too short
# <p>If we take a look at the first 12 users above we already see some bad passwords. But let's not get ahead of ourselves and start flagging passwords <em>manually</em>. What is the first thing we should check according to the NIST Special Publication 800-63B?</p>
# <blockquote>
#   <p>Verifiers SHALL require subscriber-chosen memorized secrets to be at least 8 characters in length.</p>
# </blockquote>
# <p>Ok, so the passwords of our users shouldn't be too short. Let's start by checking that!</p>
# 

# Calculating the lengths of users' passwords
import pandas as pd
users = pd.read_csv('datasets/users.csv')
users['length'] = users.password.str.len()
users['too_short'] = users['length'] < 8
print(users['too_short'].sum())

# Taking a look at the 12 first rows
users.head(12)


# ## 3.  Common passwords people use
# <p>Already this simple rule flagged a couple of offenders among the first 12 users. Next up in Special Publication 800-63B is the rule that</p>
# <blockquote>
#   <p>verifiers SHALL compare the prospective secrets against a list that contains values known to be commonly-used, expected, or compromised.</p>
#   <ul>
#   <li>Passwords obtained from previous breach corpuses.</li>
#   <li>Dictionary words.</li>
#   <li>Repetitive or sequential characters (e.g. ‘aaaaaa’, ‘1234abcd’).</li>
#   <li>Context-specific words, such as the name of the service, the username, and derivatives thereof.</li>
#   </ul>
# </blockquote>
# <p>We're going to check these in order and start with <em>Passwords obtained from previous breach corpuses</em>, that is, websites where hackers have leaked all the users' passwords. As many websites don't follow the NIST guidelines and encrypt passwords there now exist large lists of the most popular passwords. Let's start by loading in the 10,000 most common passwords which I've taken from <a href="https://github.com/danielmiessler/SecLists/tree/master/Passwords">here</a>.</p>
# 

# Reading in the top 10000 passwords
common_passwords = pd.read_csv("datasets/10_million_password_list_top_10000.txt",
                header=None,
                squeeze=True)

# Taking a look at the top 20
common_passwords.head(20)


# ## 4.  Passwords should not be common passwords
# <p>The list of passwords was ordered, with the most common passwords first, and so we shouldn't be surprised to see passwords like <code>123456</code> and <code>qwerty</code> above. As hackers also have access to this list of common passwords, it's important that none of our users use these passwords!</p>
# <p>Let's flag all the passwords in our user database that are among the top 10,000 used passwords.</p>
# 

# Flagging the users with passwords that are common passwords
users['common_password'] = users['password'].isin(common_passwords)

# Counting and printing the number of users using common passwords
print(users['common_password'].sum())

# Taking a look at the 12 first rows
users.head(12)


# ## 5. Passwords should not be common words
# <p>Ay ay ay! It turns out many of our users use common passwords, and of the first 12 users there are already two. However, as most common passwords also tend to be short, they were already flagged as being too short. What is the next thing we should check?</p>
# <blockquote>
#   <p>Verifiers SHALL compare the prospective secrets against a list that contains [...] dictionary words.</p>
# </blockquote>
# <p>This follows the same logic as before: It is easy for hackers to check users' passwords against common English words and therefore common English words make bad passwords. Let's check our users' passwords against the top 10,000 English words from <a href="https://github.com/first20hours/google-10000-english">Google's Trillion Word Corpus</a>.</p>
# 

# Reading in a list of the 10000 most common words
words = pd.read_csv("datasets/google-10000-english.txt", header=None,
                squeeze=True)

# Flagging the users with passwords that are common words
users['common_word'] = users['password'].str.lower().isin(words)

# Counting and printing the number of users using common words as passwords
print(users['common_word'].sum())

# Taking a look at the 12 first rows
users.head(12)


# ## 6. Passwords should not be your name
# <p>It turns out many of our passwords were common English words too! Next up on the NIST list:</p>
# <blockquote>
#   <p>Verifiers SHALL compare the prospective secrets against a list that contains [...] context-specific words, such as the name of the service, the username, and derivatives thereof.</p>
# </blockquote>
# <p>Ok, so there are many things we could check here. One thing to notice is that our users' usernames consist of their first names and last names separated by a dot. For now, let's just flag passwords that are the same as either a user's first or last name.</p>
# 

# Extracting first and last names into their own columns
users['first_name'] = users['user_name'].str.extract(r'(^\w+)', expand=False)
users['last_name'] = users['user_name'].str.extract(r'(\w+$)', expand=False)

# Flagging the users with passwords that matches their names
users['uses_name'] = (users['password'] == users['first_name']) | (users['password'] == users['last_name'])
# Counting and printing the number of users using names as passwords
print(users['uses_name'].count())

# Taking a look at the 12 first rows
users.head(12)


# ## 7. Passwords should not be repetitive
# <p>Milford Hubbard (user number 12 above), what where you thinking!? Ok, so the last thing we are going to check is a bit tricky:</p>
# <blockquote>
#   <p>verifiers SHALL compare the prospective secrets [so that they don't contain] repetitive or sequential characters (e.g. ‘aaaaaa’, ‘1234abcd’).</p>
# </blockquote>
# <p>This is tricky to check because what is <em>repetitive</em> is hard to define. Is <code>11111</code> repetitive? Yes! Is <code>12345</code> repetitive? Well, kind of. Is <code>13579</code> repetitive? Maybe not..? To check for <em>repetitiveness</em> can be arbitrarily complex, but here we're only going to do something simple. We're going to flag all passwords that contain 4 or more repeated characters.</p>
# 

### Flagging the users with passwords with >= 4 repeats
users['too_many_repeats'] = users['password'].str.contains(r'(.)\1\1\1')

# Taking a look at the users with too many repeats
users.head(12)


# ## 8. All together now!
# <p>Now we have implemented all the basic tests for bad passwords suggested by NIST Special Publication 800-63B! What's left is just to flag all bad passwords and maybe to send these users an e-mail that strongly suggests they change their password.</p>
# 

# Flagging all passwords that are bad
#users['bad_password'] = users['password'].isin(users['too_short'] | users['common_password'] | users['common_word'] | users['uses_name'] | users['too_many_repeats'])
#val = np.union1d(users['too_short'],users['common_password'],users['common_word'],users['uses_name'],users['too_many_repeats'])
#users['bad_password'] = users['password'].isin(val)
#users['bad_password'] = users['password'] == users['too_many_repeats']
users['bad_password'] = ( 
    users['too_short'] | 
    users['common_password'] |
    users['common_word'] |
    users['uses_name'] |
    users['too_many_repeats'])

# Counting and printing the number of bad passwords
print(users['bad_password'].sum())

# Looking at the first 25 bad passwords
print(users['bad_password'].head(25))


# ## 9. Otherwise, the password should be up to the user
# <p>In this notebook, we've implemented the password checks recommended by the NIST Special Publication 800-63B. It's certainly possible to better implement these checks, for example, by using a longer list of common passwords. Also note that the NIST checks in no way guarantee that a chosen password is good, just that it's not obviously bad.</p>
# <p>Apart from the checks we've implemented above the NIST is also clear with what password rules should <em>not</em> be imposed:</p>
# <blockquote>
#   <p>Verifiers SHOULD NOT impose other composition rules (e.g., requiring mixtures of different character types or prohibiting consecutively repeated characters) for memorized secrets. Verifiers SHOULD NOT require memorized secrets to be changed arbitrarily (e.g., periodically).</p>
# </blockquote>
# <p>So the next time a website or app tells you to "include both a number, symbol and an upper and lower case character in your password" you should send them a copy of <a href="https://pages.nist.gov/800-63-3/sp800-63b.html">NIST Special Publication 800-63B</a>.</p>
# 

# Enter a password that passes the NIST requirements
# PLEASE DO NOT USE AN EXISTING PASSWORD HERE
new_password = "Hsancor1995"
print(new_password)


# ## 1. Sound it out!
# <p>Grey and Gray. Colour and Color. Words like these have been the cause of many heated arguments between Brits and Americans. Accents (and jokes) aside, there are many words that are pronounced the same way but have different spellings. While it is easy for us to realize their equivalence, basic programming commands will fail to equate such two strings. </p>
# <p>More extreme than word spellings are names because people have more flexibility in choosing to spell a name in a certain way. To some extent, tradition sometimes governs the way a name is spelled, which limits the number of variations of any given English name. But if we consider global names and their associated English spellings, you can only imagine how many ways they can be spelled out. </p>
# <p>One way to tackle this challenge is to write a program that checks if two strings sound the same, instead of checking for equivalence in spellings. We'll do that here using fuzzy name matching.</p>
# 

# Importing the fuzzy package
import fuzzy

# Exploring the output of fuzzy.nysiis

print(fuzzy.nysiis('yesterday'))
# Testing equivalence of similar sounding words
fuzzy.nysiis('tomorrow') == fuzzy.nysiis('tommorow')


# ## 2. Authoring the authors
# <p>The New York Times puts out a weekly list of best-selling books from different genres, and which has been published since the 1930’s.  We’ll focus on Children’s Picture Books, and analyze the gender distribution of authors to see if there have been changes over time. We'll begin by reading in the data on the best selling authors from 2008 to 2017.</p>
# 

# Importing the pandas module
import pandas as pd

# Reading in datasets/nytkids_yearly.csv, which is semicolon delimited.
author_df = pd.read_csv('datasets/nytkids_yearly.csv', delimiter=';')

# Looping through author_df['Author'] to extract the authors first names
first_name = []
for name in author_df['Author']:
    first_name.append(name.split()[0])

# Adding first_name as a column to author_df
author_df['first_name'] = first_name

# Checking out the first few rows of author_df
author_df.head()


# ## 3. It's time to bring on the phonics... _again_!
# <p>When we were young children, we were taught to read using phonics; sounding out the letters that compose words. So let's relive history and do that again, but using python this time. We will now create a new column or list that contains the phonetic equivalent of every first name that we just extracted. </p>
# <p>To make sure we're on the right track, let's compare the number of unique values in the <code>first_name</code> column and the number of unique values in the nysiis coded column. As a rule of thumb, the number of unique nysiis first names should be less than or equal to the number of actual first names.</p>
# 

# Importing numpy
import numpy as np

# Looping through author's first names to create the nysiis (fuzzy) equivalent
nysiis_name = []
for first_name in author_df['first_name']:
    tmp = fuzzy.nysiis(first_name)
    nysiis_name.append(tmp.split()[0])

# Adding first_name as a column to author_df
author_df['first_name'] = first_name
# Adding nysiis_name as a column to author_df
author_df['nysiis_name'] = nysiis_name

num_bananas_one = np.unique(author_df['first_name'])
lst1 = list(num_bananas_one)
num_bananas_one = np.asarray(lst1)

num_bananas_two = np.unique(author_df['nysiis_name'])
lst2 = list(num_bananas_two)
num_bananas_two = np.asarray(lst2)

# Printing out the difference between unique firstnames and unique nysiis_names:
print(str("Difference is" + str(num_bananas_one) + "," + str(num_bananas_two) + "."))


# ## 4. The inbetweeners
# <p>We'll use <code>babynames_nysiis.csv</code>, a dataset that is derived from <a href="https://www.ssa.gov/oact/babynames/limits.html">the Social Security Administration’s baby name data</a>, to identify author genders. The dataset contains unique NYSIIS versions of baby names, and also includes the percentage of times the name appeared as a female name (<code>perc_female</code>) and the percentage of times it appeared as a male name (<code>perc_male</code>). </p>
# <p>We'll use this data to create a list of <code>gender</code>. Let's make the following simplifying assumption: For each name, if <code>perc_female</code> is greater than <code>perc_male</code> then assume the name is female, if <code>perc_female</code> is less than <code>perc_male</code> then assume it is a male name, and if the percentages are equal then it's a "neutral" name.</p>
# 

import pandas as pd
# Reading in datasets/babynames_nysiis.csv, which is semicolon delimited.
babies_df = pd.read_csv('datasets/babynames_nysiis.csv', delimiter = ';')

# Looping through babies_df to and filling up gender
gender = []
for idx, row in babies_df.iterrows():
    if row[1] > row[2]:
        gender.append('F')
    elif row[1] < row[2]:
        gender.append('M')
    elif row[1] == row[2]:
        gender.append('N')
    else:
        gender
# Adding a gender column to babies_df
babies_df['gender'] = pd.Series(gender)

# Printing out the first few rows of babies_df
print(babies_df.head(10))


# ## 5. Playing matchmaker
# <p>Now that we have identified the likely genders of different names, let's find author genders by searching for each author's name in the <code>babies_df</code> DataFrame, and extracting the associated gender. </p>
# 

# This function returns the location of an element in a_list.
# Where an item does not exist, it returns -1.
def locate_in_list(a_list, element):
   loc_of_name = a_list.index(element) if element in a_list else -1
   return(loc_of_name)

# Looping through author_df['nysiis_name'] and appending the gender of each
# author to author_gender.
author_gender = []
# ...YOUR CODE FOR TASK 5...
#print(author_df['nysiis_name'])
for idx in author_df['nysiis_name']:
   index = locate_in_list(list(babies_df['babynysiis']),idx)
   #print(index)
   if(index==-1): 
       author_gender.append('Unknown')
   else: 
       author_gender.append(list(babies_df['gender'])[index])

# Adding author_gender to the author_df
# ...YOUR CODE FOR TASK 5...
author_df['author_gender'] = author_gender 

# Counting the author's genders
# ...YOUR CODE FOR TASK 5...
author_df['author_gender'].value_counts()


# ## 6. Tally up
# <p>From the results above see that there are more female authors on the New York Times best seller's list than male authors. Our dataset spans 2008 to 2017. Let's find out if there have been changes over time.</p>
# 

# Creating a list of unique years, sorted in ascending order.
years = np.unique(author_df['Year'])
# Initializing lists
males_by_yr = []
females_by_yr = []
unknown_by_yr = []

# Looping through years to find the number of male, female and unknown authors per year
# ...YOUR CODE FOR TASK 6...
for yy in years:   
   males_by_yr.append(len( author_df[ (author_df['Year']==yy) & (author_df['author_gender']=='M')  ] ))
   females_by_yr.append(len( author_df[ (author_df['Year']==yy) & (author_df['author_gender']=='F')  ] ))
   unknown_by_yr.append(len( author_df[ (author_df['Year']==yy) & (author_df['author_gender']=='Unknown')  ] ))

# Printing out yearly values to examine changes over time
# ...YOUR CODE FOR TASK 6...
print(males_by_yr)
print(females_by_yr)
print(unknown_by_yr)


# ## 7. Foreign-born authors?
# <p>Our gender data comes from social security applications of individuals born in the US. Hence, one possible explanation for why there are "unknown" genders associated with some author names is because these authors were foreign-born. While making this assumption, we should note that these are only a subset of foreign-born authors as others will have names that have a match in <code>baby_df</code> (and in the social security dataset). </p>
# <p>Using a bar chart, let's explore the trend of foreign-born authors with no name matches in the social security dataset.</p>
# 

# Importing matplotlib
import matplotlib.pyplot as plt

# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Plotting the bar chart
plt.bar(unknown_by_yr, 'year')

# [OPTIONAL] - Setting a title, and axes labels
plt.title('Awesome!')
plt.xlabel('X')
plt.ylabel('Y')


# ## 8. Raising the bar
# <p>What’s more exciting than a bar chart is a grouped bar chart. This type of chart is good for displaying <em>changes</em> over time while also <em>comparing</em> two or more groups. Let’s use a grouped bar chart to look at the distribution of male and female authors over time.</p>
# 

# Creating a new list, where 0.25 is added to each year
years_shifted = [year + 0.25 for year in years]

# Plotting males_by_yr by year
plt.bar(males_by_yr, 'year', width = 0.25, color = 'lightblue')

# Plotting females_by_yr by years_shifted
plt.bar(females_by_yr, 'year_shifted', width = 0.25, color = 'pink')

# [OPTIONAL] - Adding relevant Axes labels and Chart Title
plt.title('Awesome!')
plt.xlabel('X')
plt.ylabel('Y')


# ## 1. Meet Professor William Sharpe
# <p>An investment may make sense if we expect it to return more money than it costs. But returns are only part of the story because they are risky - there may be a range of possible outcomes. How does one compare different investments that may deliver similar results on average, but exhibit different levels of risks?</p>
# <p><img style="float: left ; margin: 5px 20px 5px 1px;" width="200" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_66/img/sharpe.jpeg"></p>
# <p>Enter William Sharpe. He introduced the <a href="https://web.stanford.edu/~wfsharpe/art/sr/sr.htm"><em>reward-to-variability ratio</em></a> in 1966 that soon came to be called the Sharpe Ratio. It compares the expected returns for two investment opportunities and calculates the additional return per unit of risk an investor could obtain by choosing one over the other. In particular, it looks at the difference in returns for two investments and compares the average difference to the standard deviation (as a measure of risk) of this difference. A higher Sharpe ratio means that the reward will be higher for a given amount of risk. It is common to compare a specific opportunity against a benchmark that represents an entire category of investments.</p>
# <p>The Sharpe ratio has been one of the most popular risk/return measures in finance, not least because it's so simple to use. It also helped that Professor Sharpe won a Nobel Memorial Prize in Economics in 1990 for his work on the capital asset pricing model (CAPM).</p>
# <p>Let's learn about the Sharpe ratio by calculating it for the stocks of the two tech giants Facebook and Amazon. As a benchmark, we'll use the S&amp;P 500 that measures the performance of the 500 largest stocks in the US.</p>
# 

# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings to produce nice plots in a Jupyter notebook
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading in the data
stock_data = pd.read_csv('datasets/stock_data.csv', parse_dates=['Date'], index_col='Date').dropna()
benchmark_data = pd.read_csv('datasets/benchmark_data.csv', parse_dates=['Date'], index_col='Date').dropna()


# ## 2. A first glance at the data
# <p>Let's take a look the data to find out how many observations and variables we have at our disposal.</p>
# 

# Display summary for stock_data
print('Stocks\n')
stock_data.info()
print(stock_data.head())
# Display summary for benchmark_data
print('\nBenchmarks\n')
benchmark_data.info()
print(benchmark_data.head())


# ## 3. Plot & summarize daily prices for Amazon and Facebook
# <p>Before we compare an investment in either Facebook or Amazon with the index of the 500 largest companies in the US, let's visualize the data, so we better understand what we're dealing with.</p>
# 

# visualize the stock_data
stock_data.plot(subplots=True, title='Stock Data')


# summarize the stock_data
stock_data.describe()


# ## 4. Visualize & summarize daily values for the S&P 500
# <p>Let's also take a closer look at the value of the S&amp;P 500, our benchmark.</p>
# 

# plot the benchmark_data
benchmark_data.plot(subplots=True, title='S&P 500')


# summarize the benchmark_data
benchmark_data.describe()


# ## 5. The inputs for the Sharpe Ratio: Starting with Daily Stock Returns
# <p>The Sharpe Ratio uses the difference in returns between the two investment opportunities under consideration.</p>
# <p>However, our data show the historical value of each investment, not the return. To calculate the return, we need to calculate the percentage change in value from one day to the next. We'll also take a look at the summary statistics because these will become our inputs as we calculate the Sharpe Ratio. Can you already guess the result?</p>
# 

# calculate daily stock_data returns
stock_returns = stock_data.pct_change()

# plot the daily returns
stock_returns.plot()


# summarize the daily returns
stock_returns.describe()


# ## 6. Daily S&P 500 returns
# <p>For the S&amp;P 500, calculating daily returns works just the same way, we just need to make sure we select it as a <code>Series</code> using single brackets <code>[]</code> and not as a <code>DataFrame</code> to facilitate the calculations in the next step.</p>
# 

# calculate daily benchmark_data returns
# ... YOUR CODE FOR TASK 6 HERE ...
sp_returns = benchmark_data['S&P 500'].pct_change()

# plot the daily returns
sp_returns.plot()


# summarize the daily returns
sp_returns.describe()


# ## 7. Calculating Excess Returns for Amazon and Facebook vs. S&P 500
# <p>Next, we need to calculate the relative performance of stocks vs. the S&amp;P 500 benchmark. This is calculated as the difference in returns between <code>stock_returns</code> and <code>sp_returns</code> for each day.</p>
# 

# calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plot the excess_returns
excess_returns.plot()


# summarize the excess_returns
excess_returns.describe()


# ## 8. The Sharpe Ratio, Step 1: The Average Difference in Daily Returns Stocks vs S&P 500
# <p>Now we can finally start computing the Sharpe Ratio. First we need to calculate the average of the <code>excess_returns</code>. This tells us how much more or less the investment yields per day compared to the benchmark.</p>
# 

# calculate the mean of excess_returns 

avg_excess_return = excess_returns.mean()

# plot avg_excess_returns
avg_excess_return.plot.bar(title='Mean of the Return Difference')


# ## 9. The Sharpe Ratio, Step 2: Standard Deviation of the Return Difference
# <p>It looks like there was quite a bit of a difference between average daily returns for Amazon and Facebook.</p>
# <p>Next, we calculate the standard deviation of the <code>excess_returns</code>. This shows us the amount of risk an investment in the stocks implies as compared to an investment in the S&amp;P 500.</p>
# 

# calculate the standard deviations
sd_excess_return = excess_returns.std()

# plot the standard deviations
sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')


# ## 10. Putting it all together
# <p>Now we just need to compute the ratio of <code>avg_excess_returns</code> and <code>sd_excess_returns</code>. The result is now finally the <em>Sharpe ratio</em> and indicates how much more (or less) return the investment opportunity under consideration yields per unit of risk.</p>
# <p>The Sharpe Ratio is often <em>annualized</em> by multiplying it by the square root of the number of periods. We have used daily data as input, so we'll use the square root of the number of trading days (5 days, 52 weeks, minus a few holidays): √252</p>
# 

# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plot the annualized sharpe ratio
annual_sharpe_ratio.plot(title='Annualized Sharpe Ratio: Stocks vs S&P 500')


# ## 11. Conclusion
# <p>Given the two Sharpe ratios, which investment should we go for? In 2016, Amazon had a Sharpe ratio twice as high as Facebook. This means that an investment in Amazon returned twice as much compared to the S&amp;P 500 for each unit of risk an investor would have assumed. In other words, in risk-adjusted terms, the investment in Amazon would have been more attractive.</p>
# <p>This difference was mostly driven by differences in return rather than risk between Amazon and Facebook. The risk of choosing Amazon over FB (as measured by the standard deviation) was only slightly higher so that the higher Sharpe ratio for Amazon ends up higher mainly due to the higher average daily returns for Amazon. </p>
# <p>When faced with investment alternatives that offer both different returns and risks, the Sharpe Ratio helps to make a decision by adjusting the returns by the differences in risk and allows an investor to compare investment opportunities on equal terms, that is, on an 'apples-to-apples' basis.</p>
# 

# Uncomment your choice.
buy_amazon = True
# buy_facebook = True


# ## 1. Introduction to Baby Names Data
# <blockquote>
#   <p>What’s in a name? That which we call a rose, By any other name would smell as sweet.</p>
# </blockquote>
# <p>In this project, we will explore a rich dataset of first names of babies born in the US, that spans a period of more than 100 years! This suprisingly simple dataset can help us uncover so many interesting stories, and that is exactly what we are going to be doing. </p>
# <p>Let us start by reading the data.</p>
# 

# Import modules
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
bnames.head()


# ## 2. Exploring Trends in Names
# <p>One of the first things we want to do is to understand naming trends. Let us start by figuring out the top five most popular male and female names for this decade (born 2011 and later). Do you want to make any guesses? Go on, be a sport!!</p>
# 

# bnames_top5: A dataframe with top 5 popular male and female names for the decade
import numpy as np
bnames_2010 = bnames.loc[bnames['year'] > 2010]
bnames_2010_agg = bnames_2010.groupby(['sex', 'name'], as_index=False)['births'].sum()
bnames_top5 = bnames_2010_agg.sort_values(['sex', 'births'], ascending=[True, False]).groupby('sex').head().reset_index(drop=True)
print(bnames_top5.head())


# ## 3. Proportion of Births
# <p>While the number of births is a useful metric, making comparisons across years becomes difficult, as one would have to control for population effects. One way around this is to normalize the number of births by the total number of births in that year.</p>
# 

# Import modules
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
bnames2 = bnames.copy()
# Compute the proportion of births by year and add it as a new column
total_births_by_year = bnames.groupby('year')['births', 'year'].transform(sum)
bnames2['prop_births'] = bnames2.births/ total_births_by_year.births
print(bnames2)


# ## 4. Popularity of Names
# <p>Now that we have the proportion of births, let us plot the popularity of a name through the years. How about plotting the popularity of the female names <code>Elizabeth</code>, and <code>Deneen</code>, and inspecting the underlying trends for any interesting patterns!</p>
# 

# Set up matplotlib for plotting in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# Import modules
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
def plot_trends(name, sex):
  data = bnames[(bnames.name == name) & (bnames.sex == sex)]
  ax = data.plot(x = "year", y = "births")
  ax.set_xlim(1880, 2016)
  return ax


# Plot trends for Elizabeth and Deneen 
for name in ('Elizabeth', 'Deneen'):
    plt.axis = plot_trends(name, 'F')
plt.xlabel('Year')
plt.ylabel('Births')
plt.show()
# How many times did these female names peak?
num_peaks_elizabeth = 3
num_peaks_deneen    = 1


# ## 5. Trendy vs. Stable Names
# <p>Based on the plots we created earlier, we can see that <strong>Elizabeth</strong> is a fairly stable name, while <strong>Deneen</strong> is not. An interesting question to ask would be what are the top 5 stable and top 5 trendiest names. A stable name is one whose proportion across years does not vary drastically, while a trendy name is one whose popularity peaks for a short period and then dies down. </p>
# <p>There are many ways to measure trendiness. A simple measure would be to look at the maximum proportion of births for a name, normalized by the sume of proportion of births across years. For example, if the name <code>Joe</code> had the proportions <code>0.1, 0.2, 0.1, 0.1</code>, then the trendiness measure would be <code>0.2/(0.1 + 0.2 + 0.1 + 0.1)</code> which equals <code>0.5</code>.</p>
# <p>Let us use this idea to figure out the top 10 trendy names in this data set, with at least a 1000 births.</p>
# 

# top10_trendy_names | A Data Frame of the top 10 most trendy names
names = pd.DataFrame()
name_and_sex_grouped = bnames.groupby(['name', 'sex'])
names['total'] = name_and_sex_grouped['births'].sum()
names['max'] = name_and_sex_grouped['births'].max()
names['trendiness'] = names['max']/names['total']

top10_trendy_names = names.loc[names['total'] > 999].sort_values(['trendiness'], ascending=False).head(10).reset_index()

print(top10_trendy_names)


# ## 6. Bring in Mortality Data
# <p>So, what more is in a name? Well, with some further work, it is possible to predict the age of a person based on the name (Whoa! Really????). For this, we will need actuarial data that can tell us the chances that someone is still alive, based on when they were born. Fortunately, the <a href="https://www.ssa.gov/">SSA</a> provides detailed <a href="https://www.ssa.gov/oact/STATS/table4c6.html">actuarial life tables</a> by birth cohorts.</p>
# <table>
# <thead>
# <tr>
# <th style="text-align:right;">year</th>
# <th style="text-align:right;">age</th>
# <th style="text-align:right;">qx</th>
# <th style="text-align:right;">lx</th>
# <th style="text-align:right;">dx</th>
# <th style="text-align:right;">Lx</th>
# <th style="text-align:right;">Tx</th>
# <th style="text-align:right;">ex</th>
# <th style="text-align:left;">sex</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">39</td>
# <td style="text-align:right;">0.00283</td>
# <td style="text-align:right;">78275</td>
# <td style="text-align:right;">222</td>
# <td style="text-align:right;">78164</td>
# <td style="text-align:right;">3129636</td>
# <td style="text-align:right;">39.98</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">40</td>
# <td style="text-align:right;">0.00297</td>
# <td style="text-align:right;">78053</td>
# <td style="text-align:right;">232</td>
# <td style="text-align:right;">77937</td>
# <td style="text-align:right;">3051472</td>
# <td style="text-align:right;">39.09</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">41</td>
# <td style="text-align:right;">0.00318</td>
# <td style="text-align:right;">77821</td>
# <td style="text-align:right;">248</td>
# <td style="text-align:right;">77697</td>
# <td style="text-align:right;">2973535</td>
# <td style="text-align:right;">38.21</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">42</td>
# <td style="text-align:right;">0.00332</td>
# <td style="text-align:right;">77573</td>
# <td style="text-align:right;">257</td>
# <td style="text-align:right;">77444</td>
# <td style="text-align:right;">2895838</td>
# <td style="text-align:right;">37.33</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">43</td>
# <td style="text-align:right;">0.00346</td>
# <td style="text-align:right;">77316</td>
# <td style="text-align:right;">268</td>
# <td style="text-align:right;">77182</td>
# <td style="text-align:right;">2818394</td>
# <td style="text-align:right;">36.45</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">44</td>
# <td style="text-align:right;">0.00351</td>
# <td style="text-align:right;">77048</td>
# <td style="text-align:right;">270</td>
# <td style="text-align:right;">76913</td>
# <td style="text-align:right;">2741212</td>
# <td style="text-align:right;">35.58</td>
# <td style="text-align:left;">F</td>
# </tr>
# </tbody>
# </table>
# <p>You can read the <a href="https://www.ssa.gov/oact/NOTES/as120/LifeTables_Body.html">documentation for the lifetables</a> to understand what the different columns mean. The key column of interest to us is <code>lx</code>, which provides the number of people born in a <code>year</code> who live upto a given <code>age</code>. The probability of being alive can be derived as <code>lx</code> by 100,000. </p>
# <p>Given that 2016 is the latest year in the baby names dataset, we are interested only in a subset of this data, that will help us answer the question, "What percentage of people born in Year X are still alive in 2016?" </p>
# <p>Let us use this data and plot it to get a sense of the mortality distribution!</p>
# 

# Read lifetables from datasets/lifetables.csv
lifetables = pd.read_csv('datasets/lifetables.csv')

# Extract subset relevant to those alive in 2016
lifetables_2016 = lifetables[lifetables['year'] + lifetables['age'] == 2016]

# Plot the mortality distribution: year vs. lx
lifetables_2016.plot(x= 'year', y= 'lx')
plt.show()
lifetables_2016.head()


# ## 7. Smoothen the Curve!
# <p>We are almost there. There is just one small glitch. The cohort life tables are provided only for every decade. In order to figure out the distribution of people alive, we need the probabilities for every year. One way to fill up the gaps in the data is to use some kind of interpolation. Let us keep things simple and use linear interpolation to fill out the gaps in values of <code>lx</code>, between the years <code>1900</code> and <code>2016</code>.</p>
# 

# Create smoothened lifetable_2016_s by interpolating values of lx
year = np.arange(1900, 2016)
mf = {"M": pd.DataFrame(), "F": pd.DataFrame()}
for sex in ["M", "F"]:
  d = lifetables_2016[lifetables_2016['sex'] == sex][["year", "lx"]]
  mf[sex] = d.set_index('year').reindex(year).interpolate().reset_index()
  mf[sex]['sex'] = sex

lifetable_2016_s = pd.concat(mf, ignore_index = True)
lifetable_2016_s.head()
print(lifetable_2016_s)


# ## 8. Distribution of People Alive by Name
# <p>Now that we have all the required data, we need a few helper functions to help us with our analysis. </p>
# <p>The first function we will write is <code>get_data</code>,which takes <code>name</code> and <code>sex</code> as inputs and returns a data frame with the distribution of number of births and number of people alive by year.</p>
# <p>The second function is <code>plot_name</code> which accepts the same arguments as <code>get_data</code>, but returns a line plot of the distribution of number of births, overlaid by an area plot of the number alive by year.</p>
# <p>Using these functions, we will plot the distribution of births for boys named <strong>Joseph</strong> and girls named <strong>Brittany</strong>.</p>
# 

def get_data(name, sex):
    name_sex = ((bnames['name'] == name) & 
                (bnames['sex'] == sex))
    data = bnames[name_sex].merge(lifetable_2016_s)
    data['n_alive'] = data['lx']/(10**5)*data['births']
    return data
    

def plot_data(name, sex):
    fig, ax = plt.subplots()
    dat = get_data(name, sex)
    dat.plot(x = 'year', y = 'births', ax = ax, 
               color = 'black')
    dat.plot(x = 'year', y = 'n_alive', 
              kind = 'area', ax = ax, 
              color = 'steelblue', alpha = 0.8)
    ax.set_xlim(1900, 2016)
    return
# Plot the distribution of births and number alive for Joseph and Brittany
    
    plot_data('Britanny', 'F')
    plot_data('Joseph', 'M')


# ## 9. Estimate Age
# <p>In this section, we want to figure out the probability that a person with a certain name is alive, as well as the quantiles of their age distribution. In particular, we will estimate the age of a female named <strong>Gertrude</strong>. Any guesses on how old a person with this name is? How about a male named <strong>William</strong>?</p>
# 

# Import modules
from wquantiles import quantile

def estimate_age(name, sex):
    data = get_data(name, sex)
    qs = [0.75, 0.5, 0.25]
    quantiles = [2016 - int(quantile(data.year, data.n_alive, q)) for q in qs]
    result = dict(zip(['q25', 'q50', 'q75'], quantiles))
    result['p_alive'] = round(data.n_alive.sum()/data.births.sum()*100, 2)
    result['sex'] = sex
    result['name'] = name
    return pd.Series(result)
# Estimate the age of Gertrude
estimate_age('Gertrude', 'F')


# ## 10. Median Age of Top 10 Female Names
# <p>In the previous section, we estimated the age of a female named Gertrude. Let's go one step further this time, and compute the 25th, 50th and 75th percentiles of age, and the probability of being alive for the top 10 most common female names of all time. This should give us some interesting insights on how these names stack up in terms of median ages!</p>
# 

# Import modules
from wquantiles import quantile
import pandas as pd
import numpy as np
bnames = pd.read_csv('datasets/names.csv.gz')
# Function to estimate age quantiles
# Create smoothened lifetable_2016_s by interpolating values of lx
# Read lifetables from datasets/lifetables.csv
lifetables = pd.read_csv('datasets/lifetables.csv')

# Extract subset relevant to those alive in 2016
lifetables_2016 = lifetables[lifetables['year'] + lifetables['age'] == 2016]
year = np.arange(1900, 2016)
mf = {"M": pd.DataFrame(), "F": pd.DataFrame()}
for sex in ["M", "F"]:
  d = lifetables_2016[lifetables_2016['sex'] == sex][["year", "lx"]]
  mf[sex] = d.set_index('year').reindex(year).interpolate().reset_index()
  mf[sex]['sex'] = sex
lifetable_2016_s = pd.concat(mf, ignore_index = True)

# Function to estimate age quantiles

def get_data(name, sex):
    name_sex = ((bnames['name'] == name) & 
                (bnames['sex'] == sex))
    data = bnames[name_sex].merge(lifetable_2016_s)
    data['n_alive'] = data['lx']/(10**5)*data['births']
    return data
def estimate_age(name, sex):
    data = get_data(name, sex)
    qs = [0.75, 0.5, 0.25]
    quantiles = [2016 - int(quantile(data.year, data.n_alive, q)) for q in qs]
    result = dict(zip(['q25', 'q50', 'q75'], quantiles))
    result['p_alive'] = round(data.n_alive.sum()/data.births.sum()*100, 2)
    result['sex'] = sex
    result['name'] = name
    return pd.Series(result)
# Estimate the age of Gertrude
estimate_age('Gertrude', 'F')
top_10_female_names = bnames.groupby(['name', 'sex'], as_index = False).agg({'births': np.sum}).sort_values('births', ascending = False).query('sex == "F"').head(10).reset_index(drop = True)
estimates = pd.concat([estimate_age(name, 'F') for name in top_10_female_names.name], axis = 1)
median_ages = estimates.T.sort_values('q50').reset_index(drop = True)


# ## 1. Meet Dr. Ignaz Semmelweis
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_20/img/ignaz_semmelweis_1860.jpeg"></p>
# <!--
# <img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_20/datasets/ignaz_semmelweis_1860.jpeg">
# -->
# <p>This is Dr. Ignaz Semmelweis, a Hungarian physician born in 1818 and active at the Vienna General Hospital. If Dr. Semmelweis looks troubled it's probably because he's thinking about <em>childbed fever</em>: A deadly disease affecting women that just have given birth. He is thinking about it because in the early 1840s at the Vienna General Hospital as many as 10% of the women giving birth die from it. He is thinking about it because he knows the cause of childbed fever: It's the contaminated hands of the doctors delivering the babies. And they won't listen to him and <em>wash their hands</em>!</p>
# <p>In this notebook, we're going to reanalyze the data that made Semmelweis discover the importance of <em>handwashing</em>. Let's start by looking at the data that made Semmelweis realize that something was wrong with the procedures at Vienna General Hospital.</p>
# 

# importing modules
import pandas as pd

# Read datasets/yearly_deaths_by_clinic.csv into yearly
yearly = pd.read_csv('datasets/yearly_deaths_by_clinic.csv')

# Print out yearly
print(yearly)


# ## 2. The alarming number of deaths
# <p>The table above shows the number of women giving birth at the two clinics at the Vienna General Hospital for the years 1841 to 1846. You'll notice that giving birth was very dangerous; an <em>alarming</em> number of women died as the result of childbirth, most of them from childbed fever.</p>
# <p>We see this more clearly if we look at the <em>proportion of deaths</em> out of the number of women giving birth. Let's zoom in on the proportion of deaths at Clinic 1.</p>
# 

# Calculate proportion of deaths per no. births
yearly["proportion_deaths"] = yearly.deaths/yearly.births

# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2
yearly1 = yearly[yearly['clinic'] == 'clinic 1']
yearly2 = yearly[yearly['clinic'] == 'clinic 2']

# Print out yearly1
print(yearly1)


# ## 3. Death at the clinics
# <p>If we now plot the proportion of deaths at both clinic 1 and clinic 2  we'll see a curious pattern...</p>
# 

# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot yearly proportion of deaths at the two clinics
ax = yearly1.plot(x='year', y='proportion_deaths', label='clinic1')
yearly2.plot(x='year', y='proportion_deaths', label='clinic2', ax=ax)
ax.set_ylabel('Proportion deaths')


# ## 4. The handwashing begins
# <p>Why is the proportion of deaths constantly so much higher in Clinic 1? Semmelweis saw the same pattern and was puzzled and distressed. The only difference between the clinics was that many medical students served at Clinic 1, while mostly midwife students served at Clinic 2. While the midwives only tended to the women giving birth, the medical students also spent time in the autopsy rooms examining corpses. </p>
# <p>Semmelweis started to suspect that something on the corpses, spread from the hands of the medical students, caused childbed fever. So in a desperate attempt to stop the high mortality rates, he decreed: <em>Wash your hands!</em> This was an unorthodox and controversial request, nobody in Vienna knew about bacteria at this point in time. </p>
# <p>Let's load in monthly data from Clinic 1 to see if the handwashing had any effect.</p>
# 

# Read datasets/monthly_deaths.csv into monthly
monthly = pd.read_csv('datasets/monthly_deaths.csv', parse_dates=['date'])

# Calculate proportion of deaths per no. births
monthly['proportion_deaths'] = monthly.deaths/monthly.births

# Print out the first rows in monthly
monthly.head()


# ## 5. The effect of handwashing
# <p>With the data loaded we can now look at the proportion of deaths over time. In the plot below we haven't marked where obligatory handwashing started, but it reduced the proportion of deaths to such a degree that you should be able to spot it!</p>
# 

# Plot monthly proportion of deaths
# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot yearly proportion of deaths at the two clinics
ax = monthly['proportion_deaths'].plot(x='date', y='proportion_deaths', label='clinic1') 
monthly['proportion_deaths'].plot(x='date', y='proportion_deaths', label='clinic2', ax=ax)
ax.set_ylabel("Proportion deaths")


# ## 6. The effect of handwashing highlighted
# <p>Starting from the summer of 1847 the proportion of deaths is drastically reduced and, yes, this was when Semmelweis made handwashing obligatory. </p>
# <p>The effect of handwashing is made even more clear if we highlight this in the graph.</p>
# 

# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# Date when handwashing was made mandatory
import pandas as pd
handwashing_start = pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing = monthly[monthly['date'] < handwashing_start]
after_washing = monthly[monthly['date'] >= handwashing_start]

# Plot monthly proportion of deaths before and after handwashing
ax = before_washing.plot(x='date', y='proportion_deaths', label='clinic1') 
after_washing.plot(x='date', y='proportion_deaths', label='clinic2', ax=ax)
ax.set_ylabel("Proportion deaths")


# ## 7. More handwashing, fewer deaths?
# <p>Again, the graph shows that handwashing had a huge effect. How much did it reduce the monthly proportion of deaths on average?</p>
# 

# Difference in mean monthly proportion of deaths due to handwashing
before_proportion = before_washing['proportion_deaths']
after_proportion = after_washing['proportion_deaths']
mean_diff = after_proportion.mean() - before_proportion.mean()
mean_diff


# ## 8. A Bootstrap analysis of Semmelweis handwashing data
# <p>It reduced the proportion of deaths by around 8 percentage points! From 10% on average to just 2% (which is still a high number by modern standards). </p>
# <p>To get a feeling for the uncertainty around how much handwashing reduces mortalities we could look at a confidence interval (here calculated using the bootstrap method).</p>
# 

# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append(boot_after.mean() - boot_before.mean())

# Calculating a 95% confidence interval from boot_mean_diff 
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
confidence_interval


# ## 9. The fate of Dr. Semmelweis
# <p>So handwashing reduced the proportion of deaths by between 6.7 and 10 percentage points, according to a 95% confidence interval. All in all, it would seem that Semmelweis had solid evidence that handwashing was a simple but highly effective procedure that could save many lives.</p>
# <p>The tragedy is that, despite the evidence, Semmelweis' theory — that childbed fever was caused by some "substance" (what we today know as <em>bacteria</em>) from autopsy room corpses — was ridiculed by contemporary scientists. The medical community largely rejected his discovery and in 1849 he was forced to leave the Vienna General Hospital for good.</p>
# <p>One reason for this was that statistics and statistical arguments were uncommon in medical science in the 1800s. Semmelweis only published his data as long tables of raw data, but he didn't show any graphs nor confidence intervals. If he would have had access to the analysis we've just put together he might have been more successful in getting the Viennese doctors to wash their hands.</p>
# 

# The data Semmelweis collected points to that:
doctors_should_wash_their_hands = True


# ## 1. Bitcoin. Cryptocurrencies. So hot right now.
# <p>Since the <a href="https://newfronttest.bitcoin.com/bitcoin.pdf">launch of Bitcoin in 2008</a>, hundreds of similar projects based on the blockchain technology have emerged. We call these cryptocurrencies (also coins or cryptos in the Internet slang). Some are extremely valuable nowadays, and others may have the potential to become extremely valuable in the future<sup>1</sup>. In fact, the 6th of December of 2017 Bitcoin has a <a href="https://en.wikipedia.org/wiki/Market_capitalization">market capitalization</a> above $200 billion. </p>
# <p><center>
# <img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_82/img/bitcoint_market_cap_2017.png" style="width:500px"> <br> 
# <em>The astonishing increase of Bitcoin market capitalization in 2017.</em></center></p>
# <p>*<sup>1</sup>- <strong>WARNING</strong>: The cryptocurrency market is exceptionally volatile and any money you put in might disappear into thin air.  Cryptocurrencies mentioned here <strong>might be scams</strong> similar to <a href="https://en.wikipedia.org/wiki/Ponzi_scheme">Ponzi Schemes</a> or have many other issues (overvaluation, technical, etc.). <strong>Please do not mistake this for investment advice</strong>. *</p>
# <p>That said, let's get to business. As a first task, we will load the current data from the <a href="https://api.coinmarketcap.com">coinmarketcap API</a> and display it in the output.</p>
# 

# Importing pandas
import pandas as pd

# Importing matplotlib and setting aesthetics for plotting later.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.style.use('fivethirtyeight')

# Reading in current data from coinmarketcap.com
current = pd.read_json("https://api.coinmarketcap.com/v1/ticker/")

# Printing out the first few lines
current.head()


# ## 2. Full dataset, filtering, and reproducibility
# <p>The previous API call returns only the first 100 coins, and we want to explore as many coins as possible. Moreover, we can't produce reproducible analysis with live online data. To solve these problems, we will load a CSV we conveniently saved on the 6th of December of 2017 using the API call <code>https://api.coinmarketcap.com/v1/ticker/?limit=0</code> named <code>datasets/coinmarketcap_06122017.csv</code>. </p>
# 

# Reading datasets/coinmarketcap_06122017.csv into pandas
dec6 = pd.read_csv('datasets/coinmarketcap_06122017.csv')

# Selecting the 'id' and the 'market_cap_usd' columns
market_cap_raw = dec6[['id','market_cap_usd']]

# Counting the number of values
print(market_cap_raw.count())


# ## 3. Discard the cryptocurrencies without a market capitalization
# <p>Why do the <code>count()</code> for <code>id</code> and <code>market_cap_usd</code> differ above? It is because some cryptocurrencies listed in coinmarketcap.com have no known market capitalization, this is represented by <code>NaN</code> in the data, and <code>NaN</code>s are not counted by <code>count()</code>. These cryptocurrencies are of little interest to us in this analysis, so they are safe to remove.</p>
# 

# Filtering out rows without a market capitalization
cap = market_cap_raw.query('market_cap_usd > 0')

# Counting the number of values again
print(cap.count())


# ## 4. How big is Bitcoin compared with the rest of the cryptocurrencies?
# <p>At the time of writing, Bitcoin is under serious competition from other projects, but it is still dominant in market capitalization. Let's plot the market capitalization for the top 10 coins as a barplot to better visualize this.</p>
# 

#Declaring these now for later use in the plots
TOP_CAP_TITLE = 'Top 10 market capitalization'
TOP_CAP_YLABEL = '% of total cap'

# Selecting the first 10 rows and setting the index
cap10 = cap.head(10).set_index('id')

# Calculating market_cap_perc
cap10 = cap10.assign(market_cap_perc = lambda x: (x.market_cap_usd/cap.market_cap_usd.sum())*100)

# Plotting the barplot with the title defined above 
ax = cap10.market_cap_perc.head(10).plot.bar(title=TOP_CAP_TITLE)

# Annotating the y axis with the label defined above
ax.set_ylabel(TOP_CAP_YLABEL)


# ## 5. Making the plot easier to read and more informative
# <p>While the plot above is informative enough, it can be improved. Bitcoin is too big, and the other coins are hard to distinguish because of this. Instead of the percentage, let's use a log<sup>10</sup> scale of the "raw" capitalization. Plus, let's use color to group similar coins and make the plot more informative<sup>1</sup>. </p>
# <p>For the colors rationale: bitcoin-cash and bitcoin-gold are forks of the bitcoin <a href="https://en.wikipedia.org/wiki/Blockchain">blockchain</a><sup>2</sup>. Ethereum and Cardano both offer Turing Complete <a href="https://en.wikipedia.org/wiki/Smart_contract">smart contracts</a>. Iota and Ripple are not minable. Dash, Litecoin, and Monero get their own color.</p>
# <p><sup>1</sup> <em>This coloring is a simplification. There are more differences and similarities that are not being represented here.</em></p>
# <p><sup>2</sup> <em>The bitcoin forks are actually <strong>very</strong> different, but it is out of scope to talk about them here. Please see the warning above and do your own research.</em></p>
# 

# Colors for the bar plot
COLORS = ['orange', 'green', 'orange', 'cyan', 'cyan', 'blue', 'silver', 'orange', 'red', 'green']

# Plotting market_cap_usd as before but adding the colors and scaling the y-axis  
ax = cap10.market_cap_perc.head(10).plot.bar(title=TOP_CAP_TITLE, logy=True)
# Annotating the y axis with 'USD'
ax.set_ylabel('USD')

# Final touch! Removing the xlabel as it is not very informative
ax.set_xlabel('')


# ## 6. What is going on?! Volatility in cryptocurrencies
# <p>The cryptocurrencies market has been spectacularly volatile since the first exchange opened. This notebook didn't start with a big, bold warning for nothing. Let's explore this volatility a bit more! We will begin by selecting and plotting the 24 hours and 7 days percentage change, which we already have available.</p>
# 

# Selecting the id, percent_change_24h and percent_change_7d columns
volatility = dec6[['id', 'percent_change_24h', 'percent_change_7d']]

# Setting the index to 'id' and dropping all NaN rows
volatility = volatility.set_index('id').dropna()

# Sorting the DataFrame by percent_change_24h in ascending order
volatility = volatility.sort_values(['percent_change_24h'], ascending=True)

# Checking the first few rows
volatility.head()


# ## 7. Well, we can already see that things are *a bit* crazy
# <p>It seems you can lose a lot of money quickly on cryptocurrencies. Let's plot the top 10 biggest gainers and top 10 losers in market capitalization.</p>
# 

#Defining a function with 2 parameters, the series to plot and the title
def top10_subplot(volatility_series, title):
    # Making the subplot and the figure for two side by side plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    # Plotting with pandas the barchart for the top 10 losers
    ax = (volatility_series[:10].plot.bar(color='darkred', ax=axes[0]))
    
    # Setting the figure's main title to the text passed as parameter
    fig.suptitle(title)
    
    # Setting the ylabel to '% change'
    ax.set_ylabel("% change")
    
    # Same as above, but for the top 10 winners
    ax = (volatility_series[-10:].plot.bar(color='darkblue', ax=axes[1]))
    
    # Returning this for good practice, might use later
    return fig, ax

DTITLE = "24 hours top losers and winners"

# Calling the function above with the 24 hours period series and title DTITLE  
fig, ax = top10_subplot(volatility.percent_change_24h, DTITLE)


# ## 8. Ok, those are... interesting. Let's check the weekly Series too.
# <p>800% daily increase?! Why are we doing this tutorial and not buying random coins?<sup>1</sup></p>
# <p>After calming down, let's reuse the function defined above to see what is going weekly instead of daily.</p>
# <p><em><sup>1</sup> Please take a moment to understand the implications of the red plots on how much value some cryptocurrencies lose in such short periods of time</em></p>
# 

# Sorting in ascending order
volatility7d = volatility.sort_values(by='percent_change_7d', ascending=True)

WTITLE = "Weekly top losers and winners"

# Calling the top10_subplot function
fig, ax = top10_subplot(volatility7d.percent_change_7d, WTITLE)


# ## 9. How small is small?
# <p>The names of the cryptocurrencies above are quite unknown, and there is a considerable fluctuation between the 1 and 7 days percentage changes. As with stocks, and many other financial products, the smaller the capitalization, the bigger the risk and reward. Smaller cryptocurrencies are less stable projects in general, and therefore even riskier investments than the bigger ones<sup>1</sup>. Let's classify our dataset based on Investopedia's capitalization <a href="https://www.investopedia.com/video/play/large-cap/">definitions</a> for company stocks. </p>
# <p><sup>1</sup> <em>Cryptocurrencies are a new asset class, so they are not directly comparable to stocks. Furthermore, there are no limits set in stone for what a "small" or "large" stock is. Finally, some investors argue that bitcoin is similar to gold, this would make them more comparable to a <a href="https://www.investopedia.com/terms/c/commodity.asp">commodity</a> instead.</em></p>
# 

# Selecting everything bigger than 10 billion 
largecaps = cap.query('market_cap_usd > 10000000000')

# Printing out largecaps
print(largecaps)


# ## 10. Most coins are tiny
# <p>Note that many coins are not comparable to large companies in market cap, so let's divert from the original Investopedia definition by merging categories.</p>
# <p><em>This is all for now. Thanks for completing this project!</em></p>
# 

import numpy as np
# Making a nice function for counting different marketcaps from the
# "cap" DataFrame. Returns an int.
# INSTRUCTORS NOTE: Since you made it to the end, consider it a gift :D
def capcount(query_string):
    return cap.query(query_string).count().id

# Labels for the plot
lABELS = ["biggish", "micro", "nano"]

# Using capcount count the biggish cryptos
biggish = capcount('market_cap_usd > 300000000')

# Same as above for micro ...
micro = capcount('market_cap_usd > 50000000 and market_cap_usd < 300000000')

# ... and for nano
nano =  capcount('market_cap_usd < 50000000')

# Making a list with the 3 counts
values = [biggish, micro, nano]

# Plotting them with matplotlib 
ind = np.arange(len(values))
plt.bar(ind, values)
#plt.xticks(ind, LABELS)


# ## 1. Dr. John Snow
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_final1.png"></p>
# <p>Dr. John Snow (1813-1858) was a famous British physician and is widely recognized as a legendary figure in the history of public health and a leading pioneer in the development of anesthesia. Some even say one of the greatest physicians of all time.</p>
# <p>As a leading advocate of both anesthesia and hygienic practices in medicine, he not only experimented with ether and chloroform but also designed a mask and method how to administer it. He personally administered chloroform to Queen Victoria during the births of her eighth and ninth children, in 1853 and 1857, which assured a growing public acceptance of the use of anesthetics during childbirth.</p>
# <p>But, as we will show later, not all his life was just a success. John Snow is now also recognized as one of the founders of modern epidemiology <em>(some also consider him as the founder of data visualization, spatial analysis, data science in general, and many other related fields)</em> for his scientific and pretty modern data approach in identifying the source of a cholera outbreak in Soho, London in 1854, but it wasn't always like this. In fact, for a long time, he was simply ignored by the scientific community and currently is very often mythified. </p>
# <p>In this notebook, we're not only going to rediscover his "data story", but reanalyze the data that he collected in 1854 and recreate his famous map (also called The Ghost Map).</p>
# 

# Loading in the pandas module
import pandas as pd

# Reading in the data
deaths = pd.read_csv('datasets/deaths.csv')

# Print out the shape of the dataset
print(deaths.shape)

# Printing out the first 5 rows
deaths.head()


# ## 2. Cholera attacks!
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_cholera1.jpg"></p>
# <p>Prior to John Snow's discovery cholera was a regular visitor to London’s overcrowded and unsanitary streets. During the time of the third cholera outbreak, it was one of the most studied subjects (between years 1839-1856 over 700 studies and essays were published in London alone) and nearly all of the authors believed the outbreaks were due to miasma or "bad air". </p>
# <p>It was John Snow's pioneering work with anesthesia and gases that made him doubt the miasma model of the disease. Originally he formulated and published his theory that cholera is spread by water or food  in an essay On the Mode of Communication of Cholera (before the outbreak in 1849). The essay received negative reviews in the Lancet and the London Medical Gazette. </p>
# <p>We know now that he was right, but Dr. Snow's dilemma was how to prove it? His first step to getting there was checking the data. Our dataset has 489 rows of data in 3 columns but to work with dataset more easily we will first make few changes. </p>
# 

# Summarizing the content of deaths
deaths.info()

# Define the new names of your columns
newcols = {
    'Death': 'death_count',
    'X coordinate': 'x_latitude', 
    'Y coordinate': 'y_longitude' 
    }

# Rename your columns
deaths.rename(columns = newcols, inplace=True)

# Describe the dataset 
deaths.describe()


# ## 3. You know nothing, John Snow!
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_cholera_king2.png"></p>
# <p>It was somehow unthinkable that one man could debunk the miasma theory and prove that all the others got it wrong, so his work was mostly ignored. His medical colleagues simply said: "You know nothing, John Snow!"</p>
# <p>As already mentioned John Snow's first attempt to debunk the "miasma" theory ended with negative reviews. However, a reviewer made a helpful suggestion in terms of what evidence would be compelling: the crucial natural experiment would be to find people living side by side with lifestyles similar in all respects except for the water source. The cholera outbreak in Soho, London in 1854 gave Snow the opportunity not only to save lives this time but also to further test and improve his theory. But what about the final proof that he was right?  </p>
# <p>We now know how John Snow did it, so let's get the data right first.</p>
# 

# Create `locations` by subsetting only Latitude and Longitude from the dataset 
locations = deaths.loc[:, ['x_latitude','y_longitude']]

# Create `deaths_list` by transforming the DataFrame to list of lists 
deaths_list = locations.values.tolist()

# Check the length of the list
print(len(deaths_list))


# ## 4. The Ghost Map
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_original.jpg"> </p>
# <p>His original map, unfortunately, is not available (it might never even existed). We can see the famous one that he drew about a year later in 1855, though, and it is displayed in this cell. Because the map depicts and visualizes the deaths sometimes it is called also <strong>The Ghost Map</strong>. </p>
# <p>We now know how John Snow did it and have the data too, so let's recreate his map using modern techniques. </p>
# 

# Plot the data on map (map location is provided) using folium and for loop for plotting all the points
import folium

map = folium.Map(location=[51.5132119,-0.13666], tiles='Stamen Toner', zoom_start=17)
for point in range(0, len(deaths_list)):
    folium.CircleMarker(deaths_list[point], radius=8, color='red', fill=True, fill_color='red', opacity = 0.4).add_to(map)
map


# ## 5. It's the pump!
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_caricature1.jpg"></p>
# <p>After marking the deaths on the map, what John Snow saw was not a random pattern (we saw this on our recreation of The Ghost Map too). The majority of the deaths were concentrated at the corner of Broad Street (now Broadwick Street) and Cambridge Street (now Lexington Street). A cluster of deaths around the junction of these streets was the epicenter of the outbreak, but what was there? Yes, a water pump.</p>
# <p>John Snow at the time already had a developed theory that cholera spreads through water, so to test this he marked on the map also the locations of the water pumps nearby. And here it was, the whole picture.</p>
# <p>By combining the location of deaths related to cholera with locations of the water pumps, Snow was able to show that the majority were clustered around one particular public water pump in Broad Street, Soho. Finally, he had the proof that he needed.</p>
# <p>We will now do the same and add the locations of the pumps to our recreation of The Ghost Map.</p>
# 

# Import the data
pumps = pd.read_csv('datasets/pumps.csv')

# Subset the DataFrame and select just ['X coordinate', 'Y coordinate'] columns
locations_pumps = pumps.loc[:, ['X coordinate','Y coordinate']]

# Transform the DataFrame to list of lists in form of ['X coordinate', 'Y coordinate'] pairs
pumps_list = locations_pumps.values.tolist()

# Create a for loop and plot the data using folium (use previous map + add another layer)
map1 = map
for point in range(0, len(pumps_list)):
    folium.Marker(pumps_list[point], popup=pumps['Pump Name'][point]).add_to(map1)
map1


# ## 6.  You know nothing, John Snow! (again)
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_map1.jpg"></p>
# <p>So, John Snow finally had his proof that there was a connection between deaths as a consequence of the cholera outbreak and the public water pump that was probably contaminated. But he didn't just stop there and investigated further.</p>
# <p>He was looking for anomalies now (we would now say "outliers in data") and found two in fact where there were no deaths. First was brewery right on the Broad Street, so he went there and learned that they drank mostly beer (in other words not the water from the local pump, which confirms his theory that the pump is the source of the outbreak). The second building without any deaths was workhouse near Poland street where he learned that their source of water was not the pump on the Broad Street (confirmation again). The locations of both buildings are visualized also on the map on the left.</p>
# <p>He was now sure, and although officials did not trust him nor his theory they removed the handle to the pump next day, 8th of September 1854. John Snow later collected and published in his famous book also all the data about deaths in chronological order, before and after the peak of the outbreak and we will now analyze and compare the effect when the handle was removed.</p>
# 

# Import the data the right way
dates = pd.read_csv('datasets/dates.csv', parse_dates=['date'])

# Set the Date when handle was removed (8th of September 1854)
handle_removed = pd.to_datetime('1854/9/8')

# Create new column `day_name` in `dates` DataFrame with names of the day 
dates['day_name'] = dates['date'].dt.weekday_name

# Create new column `handle` in `dates` DataFrame based on a Date the handle was removed 
dates['handle'] = dates['date'] > handle_removed

# Check the dataset and datatypes
dates.info()

# Create a comparison of how many cholera deaths and attacks there were before and after the handle was removed
dates.groupby(['handle']).sum()


# ## 7. The picture worth a thousand words
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_pump1.jpg"> </p>
# <p>Removing the handle from the pump prevented any more of the infected water from being collected. The spring below the pump was later found to have been contaminated with sewage. This act was later recognized as an early example of epidemiology, public health medicine and the application of science (the germ theory of disease) in a real-life crisis. </p>
# <p>A replica of the pump, together with an explanatory and memorial plaque and without a handle was erected in 1992  near the location of the original close to the back wall of what today is the John Snow pub. The site is subtly marked with a pink granite kerbstone in front of a small wall plaque.</p>
# <p>We can learn a lot from John Snow's data. We can take a look at absolute counts, but this observation could lead us to a wrong conclusion so let's take a different look on the data using Bokeh. </p>
# <p>Thanks to John Snow we have the data in chronological order (i.e. as time series data), so the best way to see the whole picture is to visualize it and look at it the way he saw it while writing <em>On the Mode of Communication of Cholera (1855)</em>.</p>
# 

import bokeh
from bokeh.plotting import output_notebook, figure, show
output_notebook(bokeh.resources.INLINE)

# Set up figure
p = figure(plot_width=900, plot_height=450, x_axis_type='datetime', tools='lasso_select, box_zoom, save, reset, wheel_zoom',
          toolbar_location='above', x_axis_label='Date', y_axis_label='Number of Deaths/Attacks', 
          title='Number of Cholera Deaths/Attacks before and after 8th of September 1854 (removing the pump handle)')

# Plot on figure
p.line(dates['date'], dates['deaths'], color='red', alpha=1, line_width=3, legend='Cholera Deaths')
p.circle(dates['date'], dates['deaths'], color='black', nonselection_fill_alpha=0.2, nonselection_fill_color='grey')
p.line(dates['date'], dates['attacks'], color='black', alpha=1, line_width=2, legend='Cholera Attacks')

show(p)


# ## 8. John Snow's myth & Did we learn something?
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_132/img/johnsnow_water1.jpg"> </p>
# <p>From the previous interactive visualization, we can clearly see that the peak of the cholera outbreak happened before removing the handle and it was already in decline (downside trajectory) before the 8th of September 1854.</p>
# <p>This different view on the data is very important because in case that we compare just absolute numbers this could lead us to wrong conclusion that removing the handle on Broad Street pump for sure stopped the outbreak, which is simply not true (it surely did help but did not stop the outbreak) and John Snow was aware of this (he just did what needed to be done and never aspired to become a hero).</p>
# <p>But people love stories about heroes and other myths (definitely more than science or data science). According to John Snow's myth, he was the superhero who in two days defied their equals by hypothesizing that cholera was a waterborne disease. Despite no one listening to him, he bravely continued drawing his map, convinced local authorities to remove the handle of the infected water pump with his findings, and stopped the outbreak. John Snow saved the lives of many Londoners.</p>
# <p>If we take a better look behind this story, we can find also the true John Snow, who was fighting the disease with limited tools and wanted to get proof that he was right and "knew something" about cholera. He just did what he could with limited time and always boiled his water before drinking.</p>
# 

# Based on John Snow's map and the data John Snow collected, what would you say?
john_snow_knows_nothing = False


# ## 1. Introduction
# <p>Everyone loves Lego (unless you ever stepped on one). Did you know by the way that "Lego" was derived from the Danish phrase leg godt, which means "play well"? Unless you speak Danish, probably not. </p>
# <p>In this project, we will analyze a fascinating dataset on every single lego block that has ever been built!</p>
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_10/datasets/lego-bricks.jpeg" alt="lego"></p>
# 

# Nothing to do here


# ## 2. Reading Data
# <p>A comprehensive database of lego blocks is provided by <a href="https://rebrickable.com/downloads/">Rebrickable</a>. The data is available as csv files and the schema is shown below.</p>
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_10/datasets/downloads_schema.png" alt="schema"></p>
# <p>Let us start by reading in the colors data to get a sense of the diversity of lego sets!</p>
# 

# Import modules
import pandas as pd

# Read colors data
colors = pd.read_csv('datasets/colors.csv')

# Print the first few rows
colors.head()


# ## 3. Exploring Colors
# <p>Now that we have read the <code>colors</code> data, we can start exploring it! Let us start by understanding the number of colors available.</p>
# 

# How many distinct colors are available?
num_colors = colors.shape[0]
print(num_colors)


# ## 4. Transparent Colors in Lego Sets
# <p>The <code>colors</code> data has a column named <code>is_trans</code> that indicates whether a color is transparent or not. It would be interesting to explore the distribution of transparent vs. non-transparent colors.</p>
# 

# colors_summary: Distribution of colors based on transparency
colors_summary = colors.groupby(colors['is_trans']).count()
colors_summary


# ## 5. Explore Lego Sets
# <p>Another interesting dataset available in this database is the <code>sets</code> data. It contains a comprehensive list of sets over the years and the number of parts that each of these sets contained. </p>
# <p><img src="https://imgur.com/1k4PoXs.png" alt="sets_data"></p>
# <p>Let us use this data to explore how the average number of parts in Lego sets has varied over the years.</p>
# 

get_ipython().run_line_magic('matplotlib', 'inline')
# Read sets data as `sets`
sets = pd.read_csv('datasets/sets.csv')
# Create a summary of average number of parts by year: `parts_by_year`
parts_by_year = sets[['year', 'num_parts']].groupby('year', as_index=False).count()
# Plot trends in average number of parts by year
parts_by_year.plot(x = 'year', y = 'num_parts')
parts_by_year.head()


# ## 6. Lego Themes Over Years
# <p>Lego blocks ship under multiple <a href="https://shop.lego.com/en-US/Themes">themes</a>. Let us try to get a sense of how the number of themes shipped has varied over the years.</p>
# 

# themes_by_year: Number of themes shipped by year
themes_by_year = sets[['year', 'theme_id']].groupby('year', as_index=False).count()
themes_by_year.head()


# ## 7. Wrapping It All Up!
# <p>Lego blocks offer an unlimited amount of fun across ages. We explored some interesting trends around colors, parts, and themes. </p>
# 

# Nothing to do here


# ## 1. Introduction
# <p>Version control repositories like CVS, Subversion or Git can be a real gold mine for software developers. They contain every change to the source code including the date (the "when"), the responsible developer (the "who"), as well as little message that describes the intention (the "what") of a change.</p>
# <p><a href="https://commons.wikimedia.org/wiki/File:Tux.svg">
# <img style="float: right;margin:5px 20px 5px 1px" width="150px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_111/img/tux.png" alt="Tux - the Linux mascot">
# </a></p>
# <p>In this notebook, we will analyze the evolution of a very famous open-source project &ndash; the Linux kernel. The Linux kernel is the heart of some Linux distributions like Debian, Ubuntu or CentOS. </p>
# <p>We get some first insights into the work of the development efforts by </p>
# <ul>
# <li>identifying the TOP 10 contributors and</li>
# <li>visualizing the commits over the years.</li>
# </ul>
# <p>Linus Torvalds, the (spoiler alert!) main contributor to the Linux kernel (and also the creator of Git), created a <a href="https://github.com/torvalds/linux/">mirror of the Linux repository on GitHub</a>. It contains the complete history of kernel development for the last 13 years.</p>
# <p>For our analysis, we will use a Git log file with the following content:</p>
# 

# Printing the content of git_log_excerpt.csv
print(open('datasets/git_log_excerpt.csv'))


# ## 2. Reading in the dataset
# <p>The dataset was created by using the command <code>git log --encoding=latin-1 --pretty="%at#%aN"</code>. The <code>latin-1</code> encoded text output was saved in a header-less csv file. In this file, each row is a commit entry with the following information:</p>
# <ul>
# <li><code>timestamp</code>: the time of the commit as a UNIX timestamp in seconds since 1970-01-01 00:00:00 (Git log placeholder "<code>%at</code>")</li>
# <li><code>author</code>: the name of the author that performed the commit (Git log placeholder "<code>%aN</code>")</li>
# </ul>
# <p>The columns are separated by the number sign <code>#</code>. The complete dataset is in the <code>datasets/</code> directory. It is a <code>gz</code>-compressed csv file named <code>git_log.gz</code>.</p>
# 

# Loading in the pandas module
import pandas as pd

# Reading in the log file
git_log = pd.read_csv(
    'datasets/git_log.gz',
    sep='#',
    encoding='latin-1',
    header=None,
    names=['timestamp', 'author']
)

# Printing out the first 5 rows
git_log.head(5)


# ## 3. Getting an overview
# <p>The dataset contains the information about every single code contribution (a "commit") to the Linux kernel over the last 13 years. We'll first take a look at the number of authors and their commits to the repository.</p>
# 

# calculating number of commits
number_of_commits = git_log['timestamp'].count()

# calculating number of authors
number_of_authors = git_log['author'].value_counts(dropna=True).count()

# printing out the results
print("%s authors committed %s code changes." % (number_of_authors, number_of_commits))


# ## 4. Finding the TOP 10 contributors
# <p>There are some very important people that changed the Linux kernel very often. To see if there are any bottlenecks, we take a look at the TOP 10 authors with the most commits.</p>
# 

# Identifying the top 10 authors
top_10_authors = git_log['author'].value_counts().head(10)

# Listing contents of 'top_10_authors'
top_10_authors


# ## 5. Wrangling the data
# <p>For our analysis, we want to visualize the contributions over time. For this, we use the information in the <code>timestamp</code> column to create a time series-based column.</p>
# 

# converting the timestamp column
git_log['timestamp'] = pd.to_datetime(git_log['timestamp'], unit='s')

# summarizing the converted timestamp column
git_log['timestamp'].describe()


# ## 6. Treating wrong timestamps
# <p>As we can see from the results above, some contributors had their operating system's time incorrectly set when they committed to the repository. We'll clean up the <code>timestamp</code> column by dropping the rows with the incorrect timestamps.</p>
# 

# determining the first real commit timestamp
first_commit_timestamp = git_log['timestamp'].iloc[-1]

# determining the last sensible commit timestamp
last_commit_timestamp = pd.to_datetime('today')

# filtering out wrong timestamps
corrected_log = git_log[(git_log['timestamp']>=first_commit_timestamp)&(git_log['timestamp']<=last_commit_timestamp)]

# summarizing the corrected timestamp column
corrected_log['timestamp'].describe()


# ## 7. Grouping commits per year
# <p>To find out how the development activity has increased over time, we'll group the commits by year and count them up.</p>
# 

# Counting the no. commits per year
commits_per_year = corrected_log.groupby(
    pd.Grouper(
        key='timestamp', 
        freq='AS'
        )
    ).count()

# Listing the first rows
commits_per_year.head()


# ## 8. Visualizing the history of Linux
# <p>Finally, we'll make a plot out of these counts to better see how the development effort on Linux has increased over the the last few years. </p>
# 

# Setting up plotting in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the data
commits_per_year.plot(kind='line', title='Development effort on Linux', legend=False)


# ## 9.  Conclusion
# <p>Thanks to the solid foundation and caretaking of Linux Torvalds, many other developers are now able to contribute to the Linux kernel as well. There is no decrease of development activity at sight!</p>
# 

# calculating or setting the year with the most commits to Linux
year_with_most_commits = 2016


