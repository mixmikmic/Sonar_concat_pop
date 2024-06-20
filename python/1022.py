# First, we'll import some libraries of Python functions that we'll need to run this code
import pandas as pd
import sqlite3
import xlrd as xl

# Select an excel file saved in the same folder as this SQL_Practice file. 
path = ('book_db.xlsx')

# if this .sqlite db doesn't already exists, this will create it
# if the .sqlite db *does* already exist, this establishes the desired connection
con = sqlite3.connect("book_db.sqlite")

# this pulls out the names of the sheets in the workbook. We'll use these to name the different tables in the SQLite database that we'll create
table_names = xl.open_workbook(path).sheet_names()

# this loop makes it possible to use any other .xls sheet, since the sheet names aren't called out specifically
for table in table_names:
    df = pd.read_excel(path, sheetname='{}'.format(table))
    con.execute("DROP TABLE IF EXISTS {}".format(table))
    pd.io.sql.to_sql(df, "{}".format(table), con, index=False)

# now the spreadsheets are in tables in a mini database!

# Finally, a little function to make it easy to run queries on this mini-database
def run(query):
    results = pd.read_sql("{}".format(query), con)
    return results


# Some notes about common web metrics:
# - **Rank**: Position that your ad appears in relation to other advertisements (refers to paid search ads).
# - **CTR (Click-Through-Rate)**: Rate at which users click on an ad after they see it. 
#     - If 10 people see your advertisement and 1 person clicks on it, your CTR is 10%.
# - **CPC (Cost Per Click):** How much you spent each time a user clicked on your advertisement. With online advertising, you frequently only pay the site displaying your ad (or the ad network, like Google) when a user clicks on your advertisement. 
# - **RPC (Revenue per Click):** The average revenue you make each time a user clicks on your advertisement. 
#     - If you get 10 clicks and make one sale worth 10 dollars, then your RPC is 1 dollar. 
# - **Conversion**: The completion of a desired action. This might be a sale, a signup or registration, participation in a survey, a video view, etc. 
# - **Conversion Rate**: The rate at which users go from clicking to actually completing the desired action.
#     - If you get 10 clicks and 1 person completes the purchase, your conversion rate is 10%.
# - **COS (Cost of Sale)**: How much you spent to get 1 customer to convert. Don't confuse with CPA (cost per acquisition), which usually only refers to when you only pay your ad publisher when a customer makes a purchase)
#     - If you get 10 clicks, spend 50 cents per click, and 1 customer makes a purchase, your COS is 5 dollars
# - **Customer Lifetime Value**: The total revenue a customer has generated for you. Often this takes into account returns and user-support costs.
# - **RPM (Revenue Per Thousand Impressions**: Total revenue for every 1000 times the ad was displayed
# 

run('''
PRAGMA TABLE_INFO(transactions)
''')


run('''
SELECT  
    author,
    COUNT(DISTINCT(title)) as unique_titles,
    SUM(CASE WHEN gender = 'F' THEN price*purchases end) AS female_revenue,
    SUM(CASE WHEN gender = 'M' THEN price*purchases end) AS male_revenue
FROM
    books B
    JOIN transactions T ON B.id = T.bookid
    JOIN users U on U.id = T.userID
    JOIN authors A on A.id = B.AuthorID
GROUP BY author
ORDER BY female_revenue + male_revenue DESC
LIMIT 10
    ''')


run('''
SELECT 
    title, 
    author,
    SUM(clicks)/COUNT(*) as CTR,
    SUM(spend)/SUM(clicks) as CPC,
    SUM(price*purchases)/SUM(clicks) as RPC,
    SUM(purchases) as Conversions, 
    SUM(purchases)/SUM(clicks) as Conversion_Rate,
    SUM(spend)/SUM(purchases) as COS,
    SUM(price*purchases)/COUNT(*)*1000 RPM
FROM
    books B
    JOIN transactions T ON B.id = T.bookid
    JOIN users U on U.id = T.userID
    JOIN authors A on A.id = B.AuthorID
GROUP BY title
ORDER BY RPM DESC
LIMIT 10
    ''')





# What might the future developer workforce look like?
# ====================================================
# **Morgan White**
# Data Bootcamp, Spring 2016

# Introduction
# ------------
# Diversity, as a topic, has haunted technology companies and development teams for some time now. Their teams are notoriously white and male, in large part because this reflects the demographic of people who have traditionally been in these academic programs. 
# 
# However, this lack of diversity has been in the news a lot lately, [with major scandals at Google](http://time.com/3904408/google-workplace-diversity/ "Title"). There are many programs encouraging minority and female participation in STEM programs, and it feels like perhaps the next generation of developers will start to look more like the population of the world. 
# 
# ..Or will it? 
# 

import sys                             # system module
import pandas as pd                    # data package
import matplotlib.pyplot as plt        # graphics module  
import datetime as dt                  # date and time module
import numpy as np                     # foundation for Pandas
import seaborn.apionly as sns          # fancy matplotlib graphics (no styling)
from pandas.io import data, wb         # worldbank data

# plotly imports
from plotly.offline import iplot, iplot_mpl  # plotting functions
import plotly.graph_objs as go               # ditto
import plotly                                # just to print version and init notebook
import cufflinks as cf                       # gives us df.iplot that feels like df.plot
cf.set_config_file(offline=True, offline_show_link=False)

# these lines make our graphics show up in the notebook
get_ipython().magic('matplotlib inline')
plotly.offline.init_notebook_mode()

# check versions (overkill, but why not?)
print('Python version:', sys.version)
print('Pandas version: ', pd.__version__)
print('Plotly version: ', plotly.__version__)
print('Today: ', dt.date.today())


# Data
# ----
# **2013 National AP Exams**
# I thought a natural starting point was to ask the question: how are current students buying into the existing education programs, and what do those demographics look like over the years? 
# 
# It turns out the AP exam for Computer Science only became available a few years ago, but [the engagement numbers and related demographic information](http://home.cc.gatech.edu/ice-gt/556 "Title) is freely available. 
# 
# 

url = 'http://home.cc.gatech.edu/ice-gt/uploads/556/DetailedStateInfoAP-CS-A-2006-2013-with-PercentBlackAndHIspanicByState-fixed.xlsx'
ap0 = pd.read_excel(url, sheetname=1, header=0)
ap0.shape


ap = ap0.drop("Unnamed: 2", 1)


ap.head(5)


# **U.S. Census Data**
# 
# Placeholder to import 2012 U.S. Census Data. [It looks like they have demographic/state data for people in STEM careers]("http://www.census.gov/people/io/""Title"), but the link to the files is leading to a 404 page on their site. I have emailed them to let them know, and hope to have this data next week. 
# 

# 
# **Additional Data**
# 
# I'm working on finding more raw data on current demographics of developers as a benchmark. I've also discovered [this survey](http://stackoverflow.com/research/developer-survey-2016 "Title"), but am still figuring out how to get to the raw data in order to import it. 
# 




# #The Effect of Monetary Policy in the European Union since 1995
# 
# August 2015
# 
# Written by Susan Chen at NYU Stern with help from Professor David Backus
# 
# Contact: <jiachen2017@u.northwestern.edu>
# 
# 
# This project was inspired by an IMF staff paper titled ["The Real Effects of Monetary Policy in the European Union: What are the Differences?"](https://www.imf.org/external/Pubs/FT/staffp/1998/06-98/pdf/ramaswam.pdf) by Ramana Ramaswamy and Torsten Sløk.
# 
# This paper examined the effects of monetary policy in the European Union using data from the period 1972-1995. Based on results from a vector autogression model, it was discovered that the European Union countries broadly fall into two groups in terms of how their output responds to a contractionary monetary shock. In one group (Austria, Belgium, Finland, Germany, the Netherlands, and the United Kingdom), the effects take 11 to 12 quarters to occur, whereas in the second (Denmark, France, Italy, Portugal, Spain, and Sweden), it is only 5 to 6 quarters. However, the effects in the former group are twice as severe as in the latter, with decline in output at 0.7-0.9 percent from the baseline in the former versus 0.4-0.6 percent in the latter. 
# 
# I wanted to pick up where the paper left off and examine what the effects of monetary policy have been like in the European Union since 1995, just a few years before the euro was introduced in 1999. Essentially, my project reverse engineers the results of this paper with the distinction of using data from 1995 to 2014. In addition, although several other countries have since been introduced into the EU, I chose to look at the same set of twelve countries in order to do a direct comparison and for sake of simplicity. 
# 
# ##Abstract
# 
# In comparison to the estimations made in Ramaswamy and Sløk's paper, the results in my project indicate that since the introduction of the euro, the response of output in these EU countries to an interest rate shock has changed significantly. These countries no longer fall in the same two groups in terms of percent deviation from the baseline and amount of periods it takes for the effect to bottom out. 
# 
# In regards to percent deviation of output from the baseline, Austria, Belgium, Denmark, France, Germany, Italy, the Netherlands, Portugal, and the United Kingdom fall in the range 0.03 - 0.06 percent. Finland, Spain, and Sweden experience a higher deviation that falls in the 0.08 - 0.10 percent range. 
# 
# In regards to the amount of periods it takes for the monetary policy effect to bottom out, Austria, Belgium, Finland, Italy, Sweden, and the United Kingdom require 8 - 9 periods, while Denmark, France, Germany, the Netherlands, Portugal, and Spain require slightly more time at 10 - 12 periods. The differences between the number of periods here are almost negligible. 
# 

import pandas as pd
import numpy as np
import statsmodels.tsa.api as tsa
import pandas.io.data as web
import datetime


# #####Packages Imported
# 
# I use **pandas**, a Python package that allows for fast data manipulation and analysis. In pandas, a dataframe allows for storing related columns of data. I use **pandas.io.data** to extract data from FRED which is directly formatted into a dataframe. I also use **numpy**, a Python package for scientific computing, for the mathematical calculations that were needed to fit the data to a vector autoregression model. Lastly, I use **statmodels**, a Python module used for a variety of statistical computations, for creating a vector autoregression model. 
# 

# ##Creating the Data Set 
# 
# Using the FRED api and data from the OECD website, I created a list of dataframes. Each dataframe contains a country's real GDP, CPI, and short-term interest rate. The code can be easily edited to include the exchange rate in the dataframes, which was done as an addition in the IMF paper. For now, I have chosen to exclude that portion from this project. Ramaswamy and Sløk's paper carry out the estimations both with and without the exchange rate in the VAR and found that its inclusion in the VAR largely did not change the results with the exception of Sweden where the response of output to an interest rate shock was dampened. 
# 

#FRED Remote Data Access API for Real GDP data 

start = datetime.datetime(1995, 1, 1)
end = datetime.datetime(2014, 12, 30)

gdp=web.DataReader(['NAEXKP01FRQ189S','NAEXKP01ITQ189S','NAEXKP01DKQ189S','NAEXKP01SEQ652S','NAEXKP01ESQ652S','NAEXKP01PTQ652S','NAEXKP01DEQ189S','NAEXKP01NLQ189S','NAEXKP01BEQ189S','NAEXKP01ATQ189S','NAEXKP01FIQ189S','NAEXKP01GBQ652S'], 
"fred", start, end)

gdp.columns = ['France', 'Italy', 'Denmark', 'Sweden', 'Spain', 'Portugal', 'Germany', 'Netherlands', 'Belgium', 'Austria', 'Finland', 'United Kingdom']


#FRED Remote Data Access API for Exchange rate data 

exchange=web.DataReader(['CCUSMA02FRQ618N','CCUSMA02ITQ618N','CCUSMA02DKQ618N','CCUSMA02SEQ618N','CCUSMA02ESQ618N','CCUSMA02PTQ618N','CCUSMA02DEQ618N','CCUSMA02NLQ618N' ,'CCUSMA02BEQ618N', 'CCUSMA02ATQ618N','CCUSMA02FIQ618N','CCUSMA02GBQ618N'],
"fred", start, end)

exchange.columns = ['France', 'Italy', 'Denmark', 'Sweden', 'Spain', 'Portugal', 'Germany', 'Netherlands', 'Belgium', 'Austria', 'Finland', 'United Kingdom']


#Data downloaded from OECD and read into python using pandas

file1 = '/users/susan/desktop/cpiquarterlyoecd.csv' # file location #dates row replaced with a datetime format
cpi_df = pd.read_csv(file1) 
cpi_df = cpi_df.transpose() #OECD has years as columns and countries as rows 
cpi = cpi_df.drop(cpi_df.index[0]) #drop blank 'location' row
cpi.index = pd.to_datetime(cpi.index) #convert dates to datetime format
cpi.columns = ['Austria','Belgium','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia','Luxembourg','Netherlands','Poland','Portugal','Slovak Republic','Slovenia','Spain','Sweden','United Kingdom']

file2 = '/users/susan/desktop/interestquarterlyoecd.csv' # file location
interest_df = pd.read_csv(file2)
interest_df = interest_df.transpose() #OECD has years as columns and countries as rows 
interest = interest_df.drop(interest_df.index[0]) #drop blank 'location' row
interest.index = pd.to_datetime(interest.index) #convert dates to datetime format
interest.columns = ['Austria','Belgium','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia','Luxembourg','Netherlands','Poland','Portugal','Slovak Republic','Slovenia','Spain','Sweden','United Kingdom']


#Creating a list of dataframes organized by country 

by_country = {}

for country in gdp.columns:
    country_df = pd.concat([gdp[country], cpi[country], interest[country]], axis = 1) #add exchange[country] if including exchange rates
    country_df.columns = ['RealGDP', 'CPI', 'Interest'] #add 'Exchange' if necessary 
    country_df = country_df.convert_objects(convert_numeric = True)
    country_df = country_df.dropna()
    by_country[country] = country_df


# ##Dickey-Fuller test
# 
# To determine whether the data is non-stationary or not, I ran the Augmented Dickey-Fuller unit root test with a maximum of two lags and with constant and trend. In this test, the existence of a unit root, hence non-stationarity, is the null hypothesis. Critical values for the t-statistic are taken from MacKinnon (2010) {1%: -4.08, 5%: -3.47, 10%: -3.16}
# 
# We are unable to reject the null hypothesis based on the t-statistics for real GDP and CPI in all EU countries with the exception of Germany's real GDP given a 5% significance. Therefore, for the most part, the levels of both prices and output are nonstationary. 
# 
# Normally, taking the first difference of the data will yield stationarity. However, Ramaswamy and Sløk argue that there is a trade-off between statistical efficiency and potential loss of long-relationships among the variables when time series data is differenced. In addition, the impulse response of the first difference of output to an interest rates shock implies that monetary shocks have a permanent impact on the level of output. However, the impulse response of the level of output to an interest rate shock, past values of the variables determine whether a monetary shock is long lasting or not. For these reasons, I will be using the levels of the data as well. 
# 

def fuller(country):
    for country in gdp.columns:
        print (country)    
        print ('Real GDP:' , (tsa.stattools.adfuller(by_country[country].RealGDP, maxlag = 2, regression = 'ct'))[0]) #prints the t-statistic
        print ('CPI:' , tsa.stattools.adfuller(by_country[country].CPI, maxlag = 2, regression = 'ct')[0])
        print ('Interest:' , tsa.stattools.adfuller(by_country[country].Interest, maxlag = 2, regression = 'ct')[0])
        print ('---')

fuller(country)


# ##Impulse Response Functions
# 
# Impulse response functions display the response of current and future values of each variable to a one unit increase in the current value of one of the VAR errors. This is done with the assumption that this error returns to zero in following periods and that all other errors are equal to zero. 
# 
# The impulse response functions are estimated with a three variable VAR, the variables being level of output, level of prices, and short term interest rates. I select the Bayesian information criterion (BIC) for VAR order selection because it returns the smallest number of lags in comparison with other information criterion. In addition, it yielded stable regression results. The number of lags for all countries is 2, with the exception of Denmark, which is 1. The impulse responses are estimated with the first 24 periods. We are focusing on the 'Interest -> Real GDP' graph which is interest rate shock on real GDP. 
# 

def varmodel(country):
    mdata = by_country[country]
    data = np.log(mdata)
    model = tsa.VAR(data)
    res = model.fit(model.select_order()['bic'])
    irf = res.irf(24)
    irf.plot()
    #print (res.summary())  
    #print (res.is_stable())


varmodel('France')


varmodel('Italy')


varmodel('Denmark')


varmodel('Sweden')


varmodel('Spain')


varmodel('Portugal')


varmodel('Germany')


varmodel('Netherlands')


varmodel('Belgium')


varmodel('Austria')


varmodel('Finland')


varmodel('United Kingdom')


# ##Conclusion
# 
# The specific ways in which the countries may be grouped may be less significant than the fact that the implementation of a single financial market and common monetary policy has significantly reduced the differences in the transmission of monetary policy among the countries. It is also notable that in addition to the narrowing of differences between countries, the figures themselves are much smaller. At the same time, most of the countries seem to experience a longer lasting effect from the transmission of monetary policy. 
# 
# Given the turbulence that the financial crisis has caused over the past few years, which may have been reflected in the results of this project, it can be assumed that the intended effect of a shared financial market and monetary policy has not been achieved quite yet. However, this provides the possibility for further research, which is made more interesting by the potential changes that may occur within the European Union in the near future. 
# 

# ##Data Sources
# 
# Real GDP and exchange rate data were obtained through the FRED api for Python. Real GDP is in total GDP by expenditure in terms of national currency and is seasonally adjusted. Figures for Sweden, Spain, Portugal, and United Kingdom are in 2000 chained national currency. Exchange rates are averages of daily rates in terms of national currency to US dollar and are not seasonally adjusted.
# 
# [CPI](https://data.oecd.org/price/inflation-cpi.htm) and [short term interest rate](https://data.oecd.org/interest/short-term-interest-rates.htm) data were downloaded from the OECD website. CPI is for total items and Index 2010=1. Interest rates are total percentages per annum. 
# 
# All data is quarterly covering the period 1995:01 - 2014:04 with the exceptions of Austria, Italy, and the Netherlands for which there was incomplete Real GDP data so the data instead covers the period 1996:01 - 2014:04. 
# 
# 
# ####Citations: 
# 
# OECD (2015), Inflation (CPI) (indicator). doi: 10.1787/eee82e6e-en (Accessed on 06 August 2015)
# 
# OECD (2015), Short-term interest rates (indicator). doi: 10.1787/2cc37d77-en (Accessed on 06 August 2015)
# 

# #The Rise and Fall of the US Employment-Population Ratio
# 
# A research project at NYU's Stern School of Business.  
# Written by David Cai (txc202@nyu.edu) under the direction of David Backus, July 2015.  
# 
# ## Abstract
# 
# After the Great Recession, while the unemployment rate has almost returned to pre-2007 levels, the employment-population ratio has not made a similar recovery. I explore the roles of aging and other effects on employment by examining the labor force participation rate, a similar indicator that is less sensitive to cyclical variation. I also decompose the employment-population ratio into specific demographic groups to explore their contributions to the overall change.
# 
# ## The Employment-Population Ratio and the Unemployment Rate
# 
# Historically, for more over two decades from 1989 to 2010, the employment-population ratio has generally moved in line with the unemployment rate, albeit in an inverse direction (Figure 1). However, from 2011 onwards, these two indicators have begun to diverge. Despite the unemployment rate improving to almost pre-recession levels, the employment-population ratio has failed to increase by the same amount. This finding indicates that past 2011, some component of the employment-population ratio other than the unemployment rate must have changed. 
# 
# Mathematically, the employment-population ratio can be decomposed into the product of the labor force participation rate and the employment rate of the labor force. Alternatively, the employment-population ratio can be represented as the labor force participation rate multiplied by one minus the unemployment rate. Since the unemployment rate before the crisis has been roughly equal to its level today, the change in the labor force participation rate represents the largest contribution to the decline in the employment-population ratio.
# 

"""
Creates a figure using FRED data
Uses pandas Remote Data Access API
Documentation can be found at http://pandas.pydata.org/pandas-docs/stable/remote_data.html
"""

get_ipython().magic('matplotlib inline')
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

start, end = dt.datetime(1989, 1, 1), dt.datetime(2015, 6, 1) # Set the date range of the data
data = web.DataReader(['EMRATIO', 'UNRATE', 'USREC'],'fred', start, end) # Choose data series you wish to download
data.columns = ['Empl Pop Ratio', 'Unemployment Rate', 'Recession'] 
plt.figure(figsize=plt.figaspect(0.5))

data['Empl Pop Ratio'].plot()
plt.xlabel('')
plt.text(dt.datetime(1990, 1, 1), 64.25, 'Employment-', fontsize=11, weight='bold')
plt.text(dt.datetime(1990, 1, 1), 63.75, 'Population Ratio', fontsize=11, weight='bold')

data['Unemployment Rate'].plot(secondary_y=True, color = 'r')
plt.text(dt.datetime(1990, 1, 1), 4, 'Unemployment Rate', fontsize=11, weight='bold')

def get_recession_months():
    rec_dates = data['Recession']
    one_vals = np.where(rec_dates == 1) 
    rec_startind = rec_dates.index[one_vals]
    return rec_startind

def shade_recession(dates):
    for date in dates:
        plt.axvspan(date, date+relativedelta(months=+1), color='gray', alpha=0.1, lw=0)
    
shade_recession(get_recession_months())

plt.suptitle('Figure 1. Employment-Population Ratio and Unemployment, 1989-2015', fontsize=12, weight='bold')
plt.show()


# Source: Figure created using data from the Bureau of Labor Statistics (BLS) accessed through the Federal Reserve Economic Data (FRED). This graph is updated from Moffitt (2012)’s Figure 2. Recession data is from NBER accessed through FRED.
# 

# ## Labor Force Participation
# 
# Since 1976, the labor force participation rate has trended upwards until hitting a peak around 2000 (Figure 2). Aaronson et al. (2006) note that this trend can be extended back to the early 1960s, with labor force participation rising from less than 60 percent its peak of 67.3 percent in 2000. After 2000, a reversal of the previous trend emerged, with a new trend of labor force decline until today. Aaronson et al. point out that a prolonged decline in labor force participation is unprecedented in the postwar era, thus leading observers to wonder if long-term structural changes in the labor market have occurred.
# 
# After the publication of the 2006 paper, the labor force participation rate has continued to fall. Revisiting this issue, Aaronson et al. (2014) examine the decline in labor force participation from 2007 onwards. They attempt to break down the factors contributing to this decline into structural and cyclical components. The authors find that 1.3 percent, or nearly one half, of the 2.8 percent decline in the aggregate participation rate can be attributable population aging. Moreover, they note the contributions of declines in specific age/sex categories, such as among youth and adult men. Finally, they discover the existence of a cyclical component; however, its magnitude is much more uncertain. Of these three components, population aging represents the largest contributor to the labor force participation decline.
# 

start, end = dt.datetime(1976, 1, 1), dt.datetime(2015, 3, 1)
data = web.DataReader(['CIVPART', 'USREC'], 'fred', start, end)
data.columns = ['LFPR', 'Recession']
plt.figure(figsize=plt.figaspect(0.5))
data['LFPR'].plot(color = 'k')
plt.xlabel('')
shade_recession(get_recession_months())
plt.suptitle('Figure 2. Labor Force Participation Rate, 1976-2015', fontsize=12, fontweight='bold')
plt.show()


# Source: Figure created using data from the Bureau of Labor Statistics (BLS) accessed through the Federal Reserve Economic Data (FRED). This graph is adapted from Aaronson et al. (2014)’s Figure 9. Recession data is from NBER accessed through FRED.
# 

# ###Changes in the Age Distribution
# 
# As population aging is the largest contributor to the labor force participation decline, further analysis is necessary to understand its nature. Aaronson et al. (2014) observe that the proportion of the working age population reported as retired in the Current Population Survey (CPS) has increased by more than one percent in 2014 compared to 2007, accounting for the majority of the 1.3 percent effect of aging. The authors argue that this change is the result of a shift of the age distribution of the population, as the leading edge of the baby boom generation reaches age 62. However, on the contrary, within-age participation rates have increased since 2007, making a positive contribution to total labor force participation (Figure 3). Aaronson et al. (2014) make a similar finding, observing that within-age retirement rates have decreased, likely due to changes in social security and pensions, increased education levels, and longer life spans. These same factors can also explain the increase in the within-age participation rates among older cohorts. That said, the most important implication of Figure 3 is that labor force participation rates decrease with age. As the population age distribution shifts towards older ages, overall labor force participation can be expected to decrease.
# 

#file = '/Users/davidcai/lfpr.csv'
file = 'https://raw.githubusercontent.com/DaveBackus/Data_Bootcamp/master/Code/Projects/lfpr.csv'
df = pd.read_csv(file, index_col=0)

start, end = dt.datetime(1980, 1, 1), dt.datetime(2010, 1, 1)
data = web.DataReader('USREC', 'fred', start, end)
data.columns=['Recession']

# Take a simple averages of ratios for men and women
df["Age 62"] = df[["M62-64", "W62-64"]].mean(axis=1)
df["Age 65"] = df[["M65-69", "W65-69"]].mean(axis=1)
df["Age 70"] = df[["M70-74", "W70-74"]].mean(axis=1)
df["Age 75"] = df[["M75-79", "W75-79"]].mean(axis=1)

# Convert years into datetime series
df.index = df.index.astype(str) + "-1-1"
df.index = pd.to_datetime(df.index)
plt.figure(figsize=(plt.figaspect(0.5)))

df["Age 62"].plot()
df["Age 65"].plot()
df["Age 70"].plot()
df["Age 75"].plot()

plt.text(dt.datetime(2007, 1, 1), 42, 'Age 62', fontsize=11, weight='bold')
plt.text(dt.datetime(2007, 1, 1), 25, 'Age 65', fontsize=11, weight='bold')
plt.text(dt.datetime(2007, 1, 1), 15, 'Age 70', fontsize=11, weight='bold')
plt.text(dt.datetime(2007, 1, 1), 6, 'Age 75', fontsize=11, weight='bold')

shade_recession(get_recession_months())

plt.suptitle('Figure 3. Labor Force Participation Rates, By Age, 1980-2010', fontsize=12, fontweight='bold')
plt.show()


# Source: Figure created using author's calculations, working on calculations from Leonesio et al. (2012), available at http://www.ssa.gov/policy/docs/ssb/v72n1/v72n1p59-text.html#chart1. Data is originally from Current Population Survey (CPS) monthly files. Recession data is from NBER accessed through FRED.
# 
# Notes: I employ a oversimplification by taking a simple average of male and female participation rates to determine overall participation rates.
# 

# ##Demographic Specific Employment Trends
# 
# In addition to examining the contribution of the labor force participation rate in order to explain the decline in the employment-population ratio, an alternative approach is possible. Moffitt (2012) decomposes the aggregate employment-population ratio into contributions from specific demographic groups. After breaking down the overall employment population ratio into ratios for men and women, Moffitt observes different employment trends between the sexes (Figure 4). For men, he notes, on average, the ratio declined from 1970 to 1983, remained constant from 1983 to 2000, and continued to fall from 2000 onwards. For women, the ratio increased from 1970 to 2000 but began to decrease from 2000 onwards. Moffitt observes that men's wages declined from 1999-2007 while women's wages increased over the same period, which may account for differences in their employment trends. Moffitt concludes that while that "about half of the decline [in participation rate] among men can be explained by declines in wage rates and by changes in nonlabor income and family structure,” the factors contributing to the employment decline among women are less clear. Moreover, after considering other proposed factors as taxes and government transfers, Moffitt finds their contributions insignificant and unlikely to explain the employment decline.
# 

start, end = dt.datetime(1970, 1, 1), dt.datetime(2015, 3, 1)
data = web.DataReader(['LNS12300001', 'EMRATIO','LNS12300002', 'USREC'], 'fred', start, end)
data.columns=['Men', 'Overall', 'Women', 'Recession']
plt.figure(figsize=plt.figaspect(0.5))

data["Men"].plot()
data["Overall"].plot()
data["Women"].plot()
plt.xlabel('')

plt.text(dt.datetime(1971, 1, 1), 71, 'Men', fontsize=11, weight='bold')
plt.text(dt.datetime(1971, 1, 1), 52, 'Overall', fontsize=11, weight='bold')
plt.text(dt.datetime(1971, 1, 1), 37, 'Women', fontsize=11, weight='bold')

shade_recession(get_recession_months())

plt.suptitle('Figure 4. Employment Population Ratios, Overall and by Sex, 1970-2015', fontsize=12, fontweight='bold')
plt.show()


# Source: Figure created using data from the Bureau of Labor Statistics (BLS) accessed through the Federal Reserve Economic Data (FRED). This graph is updated from Moffitt (2012)’s Figure 1. Recession data is from NBER accessed through FRED.
# 

# ## Conclusion
# 
# Despite ongoing research, the decline in the employment-population ratio is still not well understood. Since the employment-population ratio can be broken down into the product of the labor participation rate and one minus the unemployment rate, a change in the employment-population ratio can be attributed to contributions from either or both of the two components. Six years after the end of the Great Recession, the unemployment rate has almost recovered fully while the employment-population ratio has failed to keep pace. This finding indicates that the majority of the decline in the employment-population ratio since 2007 can be attributed to contributions from changes in the labor force participation rate, which has led researchers to concentrate on this area.
# 
# Studying recent trends in the labor force participation rate, Aaronson et al. (2014) argue that nearly one half of the decline can be attributed to population aging. Of the rest, economists have separated the remaining component into cyclical and residual factors. Mofitt (2012) took an alternate approach in explaining the change in the employment-population ratio by studying employment trends in different age-sex groups. He found that a disproportionate amount of the decline could be attributed to less educated and younger groups for both sexes, but that employment trends and reasons for the decline were different between the sexes. While Moffitt chose not to focus on the labor force participation rate, he provides important contributions to the area of residual effects, where much more research is needed.
# 

# ##References
# 
# Aaronson, S., Cajner, T., Fallick, B., Galbis-Reig, F., Smith, C., & Wascher, W. (2014). Labor Force Participation: Recent Developments and Future Prospects. *Brookings Papers on Economic Activity*, 2014(2), 197-275.
# 
# Aaronson, S., Fallick, B., Figura, A., Pingle, J. F., & Wascher, W. L. (2006). The recent decline in the labor force participation rate and its implications for potential labor supply. *Brookings Papers on Economic Activity*, 2006(1), 69-154.
# 
# Donovan, S. A. (2015). An Overview of the Employment-Population Ratio.
# 
# Leonesio, M. V., Bridges, B., Gesumaria, R., & Del Bene, L. (2012). The increasing labor force participation of older workers and its effect on the income of the aged. *Social security bulletin*, 72(1), 59-77.
# Chicago	
# 
# Moffitt, R. A., DAVIS, S. J., & MAS, A. (2012). The Reversal of the Employment-Population Ratio in the 2000s: Facts and Explanations [with Comments and Disscussion]. *Brookings Papers on Economic Activity*, 201-264.
# 




# # Survey Data
# 
# The raw SCF survey data can be accessed from either the "Full Public Data Set" or the "Summary Extract Public Data" datasets. The only difference between the two is that the Extract dataset contains fewer variables, and the variable names were re-defined to make them a little bit more user friendly (e.g. total income = 'X5729' in the Full dataset and = 'income' in the Extract dataset). There are also a number of variables in the Extract dataset that were constructed from various variables in the Full dataset. For example the Extract dataset variable 'networth' doesn't appear in the Full dataset, but all the individual components required to construct 'networth' does.
# 
# There IS a big difference between the codebooks for the two datasets, though. The [Full dataset codebook](http://www.federalreserve.gov/econresdata/scf/files/codebk2013.txt) is 45,000 lines of text, which can make it absurdly difficult to find what you're looking for-- especially if you're only looking for variable names or definitions. The [Extract dataset codebook](http://sda.berkeley.edu/data/scfcomb2013/Doc/hcbk.htm) is hosted by UCal Berkeley and is much, much easier to work with. 
# 
# Here are some tips/hints that should save you a lot of time. All of this info is in the codebook, the problem is finding it buried in the 45,000 lines of text.
# 
# * ### ***J-codes vs. X-codes:***
# 
#    Every question from the SCF survey (post 1989) is tabulated and given a code that begins with 'X'. For example, total income is coded as X5729 in the Full dataset and 'highest level of education completed by the head of household' is X5901.
#     
#    You'll also notice that each X-code has a corresponding J-code in the Full dataset (e.g. X5729 vs. J5729). The J-codes are essentially flags, and give additional information as to whether a particular observation/variable combo was altered/imputed/missing/etc.
#     
#    *Search for 'Definitions of the "J" Variables' in the main codebook to find the J-code definitions*
#     
#    *Search for 'LIST OF VARIABLES' in the  main codebook to find definitions of the various X-codes*
#    
#    ***Note: no J-codes for survey years before 1992. Variable names for these datasets start with a 'b' rather than 'X'. ***
#     
# 
# * ### ***Weights:***
# 
#    In order to make the ~5,000 households covered by the survey more representative of the ~125 million total households in America, each observation receives a weight. The weight is supposed to estimate how many households each observation in the dataset is supposed to represent. e.g. a weight of 1,050 implies that that particular household is estimated to be representative of 1,050 households in the US during that particular year.  
#     
#    In the Full dataset, the weight variable is X42001. In the Extract dataset, the weight variable is 'wgt'.
#    
#    These weights complicate things like determining standard errors or running regressions (apparently). There's a lot of info on this in the Full Dataset codebook. There's a dataset called "Replicate Weight Files" on the SCF website that is designed to help determine standard errors/confidence intervals/etc. 
#     
#    ***Note: none of the variables in either dataset are pre-weighted.***  
#     
#     
# * ### ***The dataset you're seeing is 5x larger than actual dataset:***
# 
#   The [Berkeley SCF site](http://sda.berkeley.edu/data/scfcomb2013/Doc/hcbk.htm) does a good job explaining this --
#   
#   "Users should be aware that the dataset appears to have five times the number of cases as it should have. This dataset uses multiple imputation to supply estimates for any missing data point. Since there are five imputed values for each missing value, there are in effect five datasets, which have been combined into the current dataset. Therefore users should not attempt to calculate standard errors or confidence intervals for the point estimates produced by the SDA analysis programs. Although the (weighted) point estimates are accurate, the calculation of the complex standard errors and confidence intervals would require access to additional variables that are not included in the present dataset and would also require calculation algorithms that are not currently available in SDA.
# 
#   A weight variable (WGT) is provided for the analysis of the data. The sum of the weights for each year is the number of households in the U.S. for that year. However, if you run an analysis that includes all eight years in the dataset, the sum of the weights is eight times the average number of households in those years. Nevertheless, statistics like means and percentages will not be affected by those artificially inflated weighted N's."
# 

# # Downloading/Importing SCF Data
# 
# The SCF datasets (both Full and Summary Extract versions) are available in (1) SAS, (2) Stata, and (3) Fixed-width formats. The easiest way to read in the data is via Pandas pd.read_stata function. 
# 
# The files are pretty large, so they're compressed in a .zip file. We'll have to unzip the file first before reading it into Python.
# 
# *** We'll work with the Summary Extract data below ***
# 

import pandas as pd   #The data package
import sys            #The code below wont work for any versions before Python 3. This just ensures that (allegedly).
assert sys.version_info >= (3,5)


import requests
import io
import zipfile     #Three packages we'll need to unzip the data

"""
The next two lines of code converts the URL into a format that works
with the "zipfile" package.
"""
url2013 = 'http://www.federalreserve.gov/econresdata/scf/files/scfp2013s.zip'
url2013_requested = requests.get(url2013)  

"""
Next, zipfile downloads, unzips, and saves the file to your computer. 'url2013_unzipped' 
contains the file path for the file.
"""
zipfile2013 = zipfile.ZipFile(io.BytesIO(url2013_requested.content))        
url2013_unzipped = zipfile2013.extract(zipfile2013.namelist()[0]) 


df2013 = pd.read_stata(url2013_unzipped)



df2013.head(10)       #Returns the first 10 rows of the dataframe


# 
# 
# We'll also be looking at prior-year surveys, so I'll condense the unzipping processes above into a function out of laziness
# 
# **Note: The Summary Extract datasets are not available for survey years prior to 1989.**
# 

def unzip_survey_file(year = '2013'):
    import requests, io, zipfile
    import pandas as pd
    
    if int(year) <1989:
        url = 'http://www.federalreserve.gov/econresdata/scf/files/'+year+'_scf'+year[2:]+'bs.zip'
    
    else: 
        url = 'http://www.federalreserve.gov/econresdata/scf/files/scfp'+year+'s.zip'    
        
    url = requests.get(url)
    url_unzipped = zipfile.ZipFile(io.BytesIO(url.content))
    return url_unzipped.extract(url_unzipped.namelist()[0])

df1983 = pd.read_stata(unzip_survey_file(year = '1983'))
df1992 = pd.read_stata(unzip_survey_file(year = '1992'))
df2001 = pd.read_stata(unzip_survey_file(year = '2001'))

"""
There is no Summary Extract dataset for 1983, so we'll rename the variable names in the 1983 Full 
dataset so that they correspond to the variable names in the other survey years.

Also, 161 out of the 4262 total households covered in the 1983 survey actually reported having 
negative income. This isn't the case for the other survey years we are considering, and it 
complicates our analysis a bit below. Because of this, we drop any obs. that report negative 
incomes before proceeding. This has a near-zero impact on any of our results, since all but 2 
of these observations recieve a weight of zero. The two non-zero weight observations reporting
negative incomes account for only <0.05% of the total population, so not much is lost be 
excluding them.

Going forward: it might be worthwhile to figure out why there are instances of negative incomes
in the 1983 survey yet none for the other years. 
"""
df1983 = df1983.rename(columns = {'b3201':'income', 'b3324':'networth', 'b3015' : 'wgt'})

df1983 = df1983[df1983['income']>=0]


# # Distribution of Income in the United States
# 
# How equal, or unequal, is the distribution of income across households in the US? On way to visualize this is by plotting what's known as a [Lorenz curve](https://en.wikipedia.org/wiki/Lorenz_curve).
# 
# The Lorenz curve plots the proportion of overall income earned (y-axis) against the bottom x% of earners (x-axis). If total income in the US was uniformly distributed across all households, than we would expect the bottom 50% of earners to account for 50% of the total income. Likewise, the bottom 80% of earners should account for 80% of total income, etc. etc. The Lorenz curve in this "perfect equality" case would just be a straight line going from (x,y) = (0%,0%) to (x,y) = (100%,100%).
# 
# In reality, income in most countries is far from being uniformly distributed. As we will see, the bottom 50% of earners in the U.S. accounted for XX% of total income during 2013 while the bottom 80% accounted for just XX%. This means the Lorenz curve for the U.S. won't be linear, but instead "bowed" towards the bottom right-hand corner. The heralded Gini coefficient is actually estimated by measuring the area between a country's *actual* Lorenz curve and the linear Lorenz curve we'd see in the "perfect equality" case.
# 
# ***Note:*** It would make our lives a lot easier if the income percentile groupings of households in our *sample* accurately reflected the *actual* income percentile groupings in the US. This isn't the case, unfortunately, and we'll have to weight the observations in order to get a more accurate picture of the income distribution.
# 
# The function below produces accurate, weighted percentiles for income, net worth, or whatever other variable in the SCF dataset we're interest in. About 99% of code was taken from [here](http://stackoverflow.com/a/29677616), but I modified it a bit to suit our needs better.
# 



def weighted_percentiles(data, variable, weights, percentiles = [], 
                         dollar_amt = False, subgroup = None, limits = []):
    """
    data               specifies what dataframe we're working with
    
    variable           specifies the variable name (e.g. income, networth, etc.) in the dataframe
    
    percentiles = []   indicates what percentile(s) to return (e.g. 90th percentile = .90)
    
    weights            corresponds to the weighting variable in the dataframe
    
    dollar_amt = False returns the percentage of total income earned by that percentile 
                       group (i.e. bottom 80% of earners earned XX% of total)
                         
    dollar_amt = True  returns the $ amount earned by that percentile (i.e. 90th percentile
                       earned $X)
                         
    subgroup = ''      isolates the analysis to a particular subgroup in the dataset. For example
                       subgroup = 'age' would return the income distribution of the age group 
                       determined by the limits argument
                       
    limits = []        Corresponds to the subgroup argument. For example, if you were interesting in 
                       looking at the distribution of income across heads of household aged 18-24,
                       then you would input "subgroup = 'age', limits = [18,24]"
                         
    """
    import numpy 
    a  = list()
    data[variable+weights] = data[variable]*data[weights]
    if subgroup is None:
        tt = data
    else:
        tt = data[data[subgroup].astype(int).isin(range(limits[0],limits[1]+1))] 
    values, sample_weight = tt[variable], tt[weights]
    
    for index in percentiles: 
        values = numpy.array(values)
        index = numpy.array(index)
        sample_weight = numpy.array(sample_weight)

        sorter = numpy.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_percentiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_percentiles /= numpy.sum(sample_weight)
        a.append(numpy.interp(index, weighted_percentiles, values))
    
    if dollar_amt is False:    
        return[tt.loc[tt[variable]<=a[x],
                      variable+weights].sum()/tt[variable+weights].sum() for x in range(len(percentiles))]
    else:
        return a



get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt                            

def figureprefs(data, variable = 'income', labels = False, legendlabels = []):
    
    percentiles = [i * 0.05 for i in range(20)]+[0.99, 1.00]

    fig, ax = plt.subplots(figsize=(10,8));

    ax.set_xticks([i*0.1 for i in range(11)]);       #Sets the tick marks
    ax.set_yticks([i*0.1 for i in range(11)]);

    vals = ax.get_yticks()                           #Labels the tick marks
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);
    ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals]);

    ax.set_title('Lorenz Curve: United States, 1983 vs. 2013',  #Axes titles
                  fontsize=18, loc='center');
    ax.set_ylabel('Cumulative Percent of Total Income', fontsize = 12);
    ax.set_xlabel('Percent of Familes Ordered by Incomes', fontsize = 12);
    
    if type(data) == list:
        values = [weighted_percentiles(data[x], variable,
                    'wgt', dollar_amt = False, percentiles = percentiles) for x in range(len(data))]
        for index in range(len(data)):
            plt.plot(percentiles,values[index],
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels[index]);
            for num in [10, 19, 20]:
                ax.annotate('{:3.1f}%'.format(values[index][num]*100), 
                    xy=(percentiles[num], values[index][num]),
                    ha = 'right', va = 'center', fontsize = 12);

    else:
        values = weighted_percentiles(data, variable,
                    'wgt', dollar_amt = False, percentiles = percentiles)
        plt.plot(percentiles,values,
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels);

    plt.plot(percentiles,percentiles, linestyle =  '--', color='k',
            label='Perfect Equality');
   
    legend(loc = 2)

    

years_graph = [df2013, df1983]
labels = ['2013', '1983']

figureprefs(years_graph, variable = 'income', legendlabels = labels);


# # Is the distribution of income in the U.S. becoming more or less equal?
# 
# It's clear from the graph above that the distribution of income in the U.S. has become more unequal since 1983. This is reflected by the fact that the Lorenz curve has clearly shifted to the right. The top 1% of earners increased their share from 12.6% (100%-80.3%) of total income in 1983 to 19.7% in 2013. The top 5% increased their share from 26% in 1983 to 36.2% in 2013.
# 
# We try to illustrate which income percentile groups become worse-off or better-off below. The graph we produce below plots average annual (real) income growth between 1983 and 2013 against income percentile. 
# 
# To see what's going on here, consider the following: a family in the 50th percentile made \$46,075 per year in 1983 (in 2013 dollars) while in 2013 a family in the 50th percentile earned \$46,668 per year (again in 2013 dollars). The real growth in income for a family in the 50th percentile was therefore just 1.29% (0.046% per year on average). A family in the 90th percentile, however, saw their income increase a whopping 70.23% (2.40% per year on average) in real terms. This definitely makes it seem like higher income families have fared relatively better over the past 31 years than lower, or average income families.
# 


"""
Note: All Summary Extract data for survey years 1989 and later have been adjusted for inflation
(2013=100). This isn't the case for survey data prior to 1989, so we'll have to adjust the 1983 
data manually.
"""

from pandas.io import wb                                            # World Bank api

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)  #Ignore these two lines


cpi = wb.download(indicator='FP.CPI.TOTL' , country= 'USA', start=1983, end=2013)  #CPI

"""
The World Bank CPI series is indexed so that 2010 = 100. We'll have to re-index it so that 2013 = 100
to be consistent with the other data.
"""
cpi1983 = (100/cpi['FP.CPI.TOTL'][2013-2013])*cpi['FP.CPI.TOTL'][2013-1983]/100
df1983['realincome'] = df1983['income']/cpi1983



percentiles = [i * 0.01 for i in range(1,100)]+[0.99]+[0.999]

incomes = pd.DataFrame({'2001': weighted_percentiles(df2001, 'income', 'wgt', dollar_amt = True, percentiles =percentiles),
'2013': weighted_percentiles(df2013, 'income', 'wgt', dollar_amt = True, percentiles = percentiles),
'1992': weighted_percentiles(df1992, 'income', 'wgt', dollar_amt = True, percentiles = percentiles),
'1983': weighted_percentiles(df1983, 'realincome', "wgt", dollar_amt = True, percentiles = percentiles)})

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(percentiles,(incomes['2013']-incomes['1983'])/incomes['1983']/(2013-1983+1),
         linewidth = 2.0, label = '1983-2013');
yvals = ax.get_yticks()
ax.set_xticks([i * 0.1 for i in range(11)])
xvals = ax.get_xticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in yvals]);
ax.set_xticklabels(['{:3.0f}'.format(x*100) for x in xvals]);
ax.set_title('Annual real income growth by income percentile',  #Axes titles
                  fontsize=18, loc='center');
ax.axhline(y=0,xmin = 0, xmax = 1, linestyle = '--', color = 'k');
ax.set_ylabel('Average annual growth rate of real income');
ax.set_xlabel('Income percentile');
legend(loc=2);


# The figure above is pretty revealing, but the picture might be even more stark for the period 1992-2001, which corresponds roughly to President Clinton's two terms in office (or as close as we can get with the SCF, which is conducted every three years). 
# 
# This isn't to say that the other decades aren't interesting, but this timeframe seems to be one with the most rapid growth in top incomes.
# 
# Real income growth exploded for top earners in the 1990s. The top 1% of earners saw their income grow by 8.35% per year on average in real terms between 1992 and 2001. The top 0.1% faired even better, with average real income growth of 18.70%.
# 
# That being said, just about every income percentile saw better-than-average growth between 1992 and 2001. Median income (50th percentile) grew 2.15 per year on average in real terms, which is much better than the 0.046% per year average real income growth the same percentile group experienced over the period 1983-2013.
# 

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(percentiles,(incomes['2001']-incomes['1992'])/incomes['1992']/(2001-1992+1),
         linewidth = 2.0, label = '1992-2001');
plt.plot(percentiles,(incomes['2013']-incomes['1983'])/incomes['1983']/(2013-1983+1),
         linewidth = 2.0, label = '1983-2013');
yvals = ax.get_yticks()
ax.set_xticks([i * 0.1 for i in range(11)])
xvals = ax.get_xticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in yvals]);
ax.set_xticklabels(['{:3.0f}'.format(x*100) for x in xvals]);
ax.set_title('Annual real income growth by income percentile',  #Axes titles
                  fontsize=18, loc='center');
ax.axhline(y=0,xmin = 0, xmax = 1, linestyle = '--', color = 'k');
ax.set_ylabel('Average annual growth rate of real income');
ax.set_xlabel('Income percentile');
legend(loc=2);


# # Is there a relationship between GDP per capita and PISA scores?
# 
# July 2015 
# 
# Written by Susan Chen at NYU Stern with help from Professor David Backus
# 
# Contact: <jiachen2017@u.northwestern.edu>
# 
# ##About PISA
# 
# Since 2000, the Programme for International Student Assessment (PISA) has been administered every three years to evaluate education systems around the world. It also gathers family and education background information through surveys. The test, which assesses 15-year-old students in reading, math, and science, is administered to a total of around 510,000 students in 65 countries. The duration of the test is two hours, and it contains a mix of open-ended and multiple-choice questions. Learn more about the test [here](http://www.oecd.org/pisa/test/).  
# 
# I am interested in seeing if there is a correlation between a nation's wealth and their PISA scores. Do wealthier countries generally attain higher scores, and if so, to what extent? I am using GDP per capita as the economic measure of wealth because this is information that could be sensitive to population numbers so GDP per capita in theory should allow us to compare larger countries (in terms of geography or population) with small countries. 
# 
#  
# ##Abstract 
# 
# In terms of the correlation between GDP per capita and each component of the PISA, the r-squared values for an OLS regression model, which usually reflect how well the model fits the data, are 0.57, 0.63, and 0.57 for reading, math, and science, respectively. Qatar and Vietnam, outliers, are excluded from the model.
# 

# ####Packages Imported
# I use **matplotlib.pyplot** to plot scatter plots. I use **pandas**, a Python package that allows for fast data manipulation and analysis, to organize my dataset. I access World Bank data through the remote data access API for pandas, **pandas.io**.  I also use **numpy**, a Python package for scientific computing, for the mathematical calculations that were needed to fit the data more appropriately. Lastly, I use **statmodels.formula.api**, a Python module used for a variety of statistical computations, for running an OLS linear regression. 
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pandas.io import wb


# ##Creating the Dataset
# 
# PISA 2012 scores are downloaded as an excel file from the [statlink](http://dx.doi.org/10.1787/888932937035) on page 21 of the published [PISA key findings](http://www.oecd.org/pisa/keyfindings/pisa-2012-results-volume-I.pdf). I deleted the explanatory text surrounding the table. I kept only the "Mean Score in PISA 2012" column for each subject and then saved the file as a csv. Then, I read the file into pandas and renamed the columns. 
# 

file1 = '/users/susan/desktop/PISA/PISA2012clean.csv' # file location
df1 = pd.read_csv(file1)

#pandas remote data access API for World Bank GDP per capita data
df2 = wb.download(indicator='NY.GDP.PCAP.PP.KD', country='all', start=2012, end=2012)


df1


#drop multilevel index 
df2.index = df2.index.droplevel('year') 


df1.columns = ['Country','Math','Reading','Science']
df2.columns = ['GDPpc']


#combine PISA and GDP datasets based on country column  
df3 = pd.merge(df1, df2, how='left', left_on = 'Country', right_index = True) 


df3.columns = ['Country','Math','Reading','Science','GDPpc']


#drop rows with missing GDP per capita values
df3 = df3[pd.notnull(df3['GDPpc'])] 


print (df3)


# ##Excluding Outliers 
# 
# I initially plotted the data and ran the regression without excluding any outliers. The resulting r-squared values for reading, math, and science were 0.29, 0.32, and 0.27, respectively. Looking at the scatter plot, there seem to be two obvious outliers, Qatar and Vietnam. I decided to exclude the data for these two countries because the remaining countries do seem to form a trend. I found upon excluding them that the correlation between GDP per capita and scores was much higher. 
# 
# Qatar is an outlier as it placed relatively low, 63rd out of the 65 countries, with a relatively high GDP per capita at about $131000. Qatar has a high GDP per capita for a country with just 1.8 million people, and only 13% of which are Qatari nationals. Qatar is a high income economy as it contains one of the world's largest natural gas and oil reserves.
# 
# [Vietnam](http://www.economist.com/blogs/banyan/2013/12/education-vietnam) is an outlier because it placed relatively high, 17th out of the 65 countries, with a relatively low GDP per capita at about $4900. Reasons for Vietnam's high score may be due to the investment of the government in education and the uniformity of classroom professionalism and discipline found across countries. At the same time, rote learning is much more emphasized than creative thinking, and it is important to note that many disadvantaged students are forced to drop out, reasons which may account for the high score. 
# 

df3.index = df3.Country #set country column as the index 
df3 = df3.drop(['Qatar', 'Vietnam']) # drop outlier


# ##Plotting the Data
# 
# I use the log of the GDP per capita to plot against each component of the PISA on a scatter plot. 
# 

Reading = df3.Reading
Science = df3.Science
Math = df3.Math
GDP = np.log(df3.GDPpc)

#PISA reading vs GDP per capita
plt.scatter(x = GDP, y = Reading, color = 'r') 
plt.title('PISA 2012 Reading scores vs. GDP per capita')
plt.xlabel('GDP per capita (log)')
plt.ylabel('PISA Reading Score')
plt.show()

#PISA math vs GDP per capita
plt.scatter(x = GDP, y = Math, color = 'b')
plt.title('PISA 2012 Math scores vs. GDP per capita')
plt.xlabel('GDP per capita (log)')
plt.ylabel('PISA Math Score')
plt.show()

#PISA science vs GDP per capita
plt.scatter(x = GDP, y = Science, color = 'g')
plt.title('PISA 2012 Science scores vs. GDP per capita')
plt.xlabel('GDP per capita (log)')
plt.ylabel('PISA Science Score')
plt.show()


# ##Regression Analysis
# 
# The OLS regression results indicate that the there is a 0.57 correlation betweeen reading scores and GDP per capita, 0.63 between math scores and GDP per capita, and 0.57 between science scores and GDP per capita. 
# 

lm = smf.ols(formula='Reading ~ GDP', data=df3).fit()
lm2.params
lm.summary()


lm2 = smf.ols(formula='Math ~ GDP', data=df3).fit()
lm2.params
lm2.summary()


lm3 = smf.ols(formula='Science ~ GDP', data=df3).fit()
lm3.params
lm3.summary()


# ##Conclusion
# 
# The results show that countries with a higher GDP per capita seem to have a relatively higher advantage even though correlation does not imply causation. GDP per capita only reflects the potential of the country to divert financial resources towards education, and not how much is actually allocated to education. While the correlation is not weak, it is not strong enough to indicate the fact that a country's greater wealth will lead to a better education system. Deviations from the trend line would show that countries with similar performance on the PISA can vary greatly in terms of GDP per capita. The two outliers, Vietnam and Qatar, are two examples of that. At the same time, great scores are not necessarily indicative of a great educational system. There are many factors that need to be taken into consideration when evaluating a country's educational system, such as secondary school enrollment, and this provides a a great opportunity for further research. 
# 

# ##Data Sources
# 
# PISA 2012 scores are downloaded from the [statlink](http://dx.doi.org/10.1787/888932937035) on page 21 of the published [PISA key findings](http://www.oecd.org/pisa/keyfindings/pisa-2012-results-volume-I.pdf).
# 
# GDP per capita data is accessed through the World Bank API for Pandas. Documentation is found [here](http://pandas.pydata.org/pandas-docs/stable/remote_data.html#remote-data-wb). GDP per capita is based on PPP and is in constant 2011 international dollars (indicator: NY.GDP.PCAP.PP.KD). 
# 

