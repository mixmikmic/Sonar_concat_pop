# ##  Introduction to Data Visualization with Altair
# 

# Notebook to accompany article on PB Python.
# http://pbpython.com/altair-intro.html
# 

import pandas as pd
from altair import Chart, X, Y, Axis, SortField


get_ipython().magic('matplotlib inline')


# Read in the sample CSV file containing the MN 2014 capital budget
# 

budget = pd.read_csv("https://github.com/chris1610/pbpython/raw/master/data/mn-budget-detail-2014.csv")


budget.head()


# Make a basic plot of the top 10 expenditures using the default pandas plotting functions
# 

budget_top_10 = budget.sort_values(by='amount',ascending=False)[:10]


budget_top_10.plot(kind="bar",x=budget_top_10["detail"],
                   title="MN Capital Budget - 2014",
                   legend=False)


# Build a similar plot in Altair
# 

c = Chart(budget_top_10).mark_bar().encode(
    x='detail',
    y='amount')
c


# Convert to a horizontal bar graph
# 

c = Chart(budget_top_10).mark_bar().encode(
    y='detail',
    x='amount')
c


# Look at the dictionary containing the JSON spec
# 

c.to_dict(data=False)


# Use X() and Y() - does nothing in this example but is useful for future steps
# 

Chart(budget_top_10).mark_bar().encode(
    x=X('detail'),
    y=Y('amount')
)


# Add in color codes based on the category. This automatically includes a legend.
# 

Chart(budget_top_10).mark_bar().encode(
    x=X('detail'),
    y=Y('amount'),
    color="category"
)


# Show what happens if we run on all data, not just the top 10
# 

Chart(budget).mark_bar().encode(
    x='detail',
    y='amount',
    color='category')


# Filter data to only amounts >= $10M
# 

Chart(budget).mark_bar().encode(
    x='detail:N',
    y='amount:Q',
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )


# Swap X and Y in order to see it as a horizontal bar chart
# 

Chart(budget).mark_bar().encode(
    y='detail:N',
    x='amount:Q',
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )


# Equivalent approach using X and Y classes
# 

Chart(budget).mark_bar().encode(
    x=X('detail:O',
        axis=Axis(title='Project')),
    y=Y('amount:Q',
        axis=Axis(title='2014 Budget')),
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )


# Order the projects based on their total spend
# 

Chart(budget).mark_bar().encode(
    x=X('detail:O', sort=SortField(field='amount', order='descending', op='sum'),
        axis=Axis(title='Project')),
    y=Y('amount:Q',
        axis=Axis(title='2014 Budget')),
    color='category').transform_data(
      filter='datum.amount >= 10000000',
        )


# Summarize the data at the category level
# 

Chart(budget).mark_bar().encode(
    x=X('category', sort=SortField(field='amount', order='descending', op='sum'),
        axis=Axis(title='Category')),
    y=Y('sum(amount)',
        axis=Axis(title='2014 Budget')))


c = Chart(budget).mark_bar().encode(
    y=Y('category', sort=SortField(field='amount', order='descending', op='sum'),
        axis=Axis(title='Category')),
    x=X('sum(amount)',
        axis=Axis(title='2014 Budget')))
c


c.to_dict(data=False)





# This (article] [http://pbpython.com/simple-graphing-pandas.html] will walk through how to start doing some simple graphing in pandas.
# I am using a new data file that is the same format as my previous article but includes data for only 20 customers.
# First we are going to import pandas, numpy and matplot lib. 
# I am also showing the versions I'm testing so you can make sure yours is compatible.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.__version__


# Next, enable iPython to display matplotlib graphs. As an alternative you can run ipython notebook.
# 

get_ipython().magic('matplotlib inline')


# We will read in the file like we did in the previous article but I'm going to tell it to treat the date column as a date field so I can do some re-sampling later.
# 

sales=pd.read_csv("sample-salesv2.csv",parse_dates=['date'])
sales.head()


# Now that we have read in the data, we can do some quick analysis
# 

sales.describe()


# We can actually learn some pretty helpful info from this simple command:
# For example, we can tell that customers on average purchases 10.3 items per transaction and that the average cost of the transaction was $579.84. It is also easy to see the min and max so you understand the range of the data.
# 

sales['unit price'].describe()


# It is easy to call describe on a single column too. I can see that my average price is \$56.18 but it ranges from \$10.06 to \$99.97.
# 

# I am showing the output of dtypes so that you can see that the date column is a datetime field. I also scan this to make sure that any columns that have numbers are floats or ints so that I can do additional analysis in the future.
# 

sales.dtypes


# Now we remove some columns to make additional analysis easier.
# 

customers = sales[['name','ext price','date']]
customers.head()


# This representation has multiple lines for each customer. In order to understand purchasing patterns, let's group all the customers by name.
# 

customer_group = customers.groupby('name')
customer_group.size()


# Now that our data is in a simple format to manipulate, let's determine how much each customer purchased during our time frame.
# 
# The sum function allows us to quickly sum up all the values by customer. We can also sort the data using the sort command.
# 

sales_totals = customer_group.sum()
sales_totals.sort_values(by=['ext price']).head()


# Now that we know what the data look like, tt is very simple to create a quick bar chart plot.
# 

my_plot = sales_totals.plot(kind='bar')


# Unfortunately this chart is a little ugly. With a few tweaks we can make it a little more impactful.
# Let's try:
# - sorting the data in descending order.
# - Removing the legend
# - Adding a title
# - Labeling the axes
# 

my_plot = sales_totals.sort_values(by=['ext price'],ascending=False).plot(kind='bar',legend=None,title="Total Sales by Customer")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales ($)")


# This actually tells us a little about our biggest customers and how much difference there is between their sales and our smallest customers.
# 
# Now, let's try to see how the sales break down by category.
# 

customers = sales[['name','category','ext price','date']]
customers.head()


# We can use groupby to organize the data by category and name.
# 

category_group=customers.groupby(['name','category']).sum()
category_group.head()


# The category representation looks good but we need to break it apart to graph it as a stacked bar graph. Unstack can do this for us.
# 

category_group.unstack().head()


# Now plot it.
# 

my_plot = category_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Customer")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")


# Now clean some of this up a little bit.
# We can specify the figure size and customize the legend.
# 

my_plot = category_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Customer",figsize=(9, 7))
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")
my_plot.legend(["Total","Belts","Shirts","Shoes"], loc=9,ncol=4)


# Now that we know who the biggest customers are and how they purchase products, we might want to look at purchase patterns in more detail.
# 
# Let's take another look at the data and try to see how large the individual purchases are. A histogram allows us to group purchases together so we can see how big the customer transactions are.
# 

purchase_patterns = sales[['ext price','date']]
purchase_patterns.head()


purchase_plot = purchase_patterns['ext price'].hist(bins=20)
purchase_plot.set_title("Purchase Patterns")
purchase_plot.set_xlabel("Order Amount($)")
purchase_plot.set_ylabel("Number of orders")


# After looking at this group
# 
# We can look at purchase patterns over time. We can see that most of our transactions are less than $500 and only a very few are about $1500.
# 

# Another interesting way to look at the data would be by sales over time. Do we have certain months where we are busier than others?
# 
# Let's get the data down to order size and date.

purchase_patterns = sales[['ext price','date']]
purchase_patterns.head()


# If we want to analyze the data by date, we need to set the date column as the index.
# 

purchase_patterns = purchase_patterns.set_index('date')
purchase_patterns.head()


# One of the really cool things that pandas allows us to do is resample the data. If we want to look at the data by month, we can easily resample and sum it all up.
# 

# purchase_patterns.resample('M',how=sum)
# 

# Plotting the data is now very easy
# 

purchase_plot = purchase_patterns.resample('M').sum().plot(title="Total Sales by Month",legend=None)


# December is our peak month and April is the slowest.
# 
# Let's say we really like this plot and want to save it somewhere for a presentation.
# 

fig = purchase_plot.get_figure()
fig.savefig("total-sales.png")





# Follow up notebook that describes how accurately prophet predicted several weeks of website traffic.
# Full article is [here](http://pbpython.com/prophet-accuracy.html)
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')


proj = pd.read_excel('https://github.com/chris1610/pbpython/blob/master/data/March-2017-forecast-article.xlsx?raw=True')


# Look at the prediction (aka yhat) values
# 

proj[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# Convert the values from log format and filter for the right daterange
# 

proj["Projected_Sessions"] = np.exp(proj.yhat).round()
proj["Projected_Sessions_lower"] = np.exp(proj.yhat_lower).round()
proj["Projected_Sessions_upper"] = np.exp(proj.yhat_upper).round()

final_proj = proj[(proj.ds > "3-5-2017") & 
                  (proj.ds < "5-20-2017")][["ds", "Projected_Sessions_lower", 
                                            "Projected_Sessions", "Projected_Sessions_upper"]]


# Read in the Google Analytics file
# 

actual = pd.read_excel('Traffic_20170306-20170519.xlsx')
actual.columns = ["ds", "Actual_Sessions"]


actual.head()


# Combine the predictions and the merge
# 

df = pd.merge(actual, final_proj)
df.head()


# See how big the delta is between the prediction and actual
# 

df["Session_Delta"] = df.Actual_Sessions - df.Projected_Sessions
df.Session_Delta.describe()


# Need to convert to just a date in order to keep plot from throwing errors
df['ds'] = df['ds'].dt.date


# Quick Plot of the delta
# 

fig, ax = plt.subplots(figsize=(9, 6))
df.plot("ds", "Session_Delta", ax=ax)
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right');


# More comprehensive plot showing the upper and lower bound
# 

fig, ax = plt.subplots(figsize=(9, 6))
df.plot(kind='line', x='ds', y=['Actual_Sessions', 'Projected_Sessions'], ax=ax, style=['-','--'])
ax.fill_between(df['ds'].values, df['Projected_Sessions_lower'], df['Projected_Sessions_upper'], alpha=0.2)
ax.set(title='Pbpython Traffic Prediction Accuracy', xlabel='', ylabel='Sessions')
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')


# ## Learning More About Pandas by Building and Using a Weighted Average Function
# 

# This notebook is based on the article on [Pbpython.com](http://pbpython.com/weighted-average.html). Please reference it for the background and additional details
# 

import pandas as pd
import numpy as np


# Read in our sample sales data that includes projected price for our new product launch
# 

sales = pd.read_excel("https://github.com/chris1610/pbpython/blob/master/data/sales-estimate.xlsx?raw=True", sheetname="projections")
sales


# Show the mean for our current and new product price
# 

print(sales["Current_Price"].mean())
print(sales["New_Product_Price"].mean())


# Calculate the weighted average using the long form
# 

print((sales["Current_Price"] * sales["Quantity"]).sum() / sales["Quantity"].sum())
print((sales["New_Product_Price"] * sales["Quantity"]).sum() / sales["Quantity"].sum())


# Use np.average to simplify the formula
# 

print(np.average(sales["Current_Price"], weights=sales["Quantity"]))
print(np.average(sales["New_Product_Price"], weights=sales["Quantity"]))


# For maximum flexibility, build our own weighted average function
# 

def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


# Call the weighted average on all of the data
# 

print(wavg(sales, "Current_Price", "Quantity"))
print(wavg(sales, "New_Product_Price", "Quantity"))


# Group the data by manager
# 

sales.groupby("Manager").apply(wavg, "Current_Price", "Quantity")


sales.groupby("Manager").apply(wavg, "New_Product_Price", "Quantity")


# You can also group by state
# 

sales.groupby("State").apply(wavg, "New_Product_Price", "Quantity")


# You can also group by multiple criteria and the function will work correctly.
# 

sales.groupby(["Manager", "State"]).apply(wavg, "New_Product_Price", "Quantity")


# Example of applying multiple aggregation functions
# 

f = {'New_Product_Price': ['mean'],'Current_Price': ['median'], 'Quantity': ['sum', 'mean']}
sales.groupby("Manager").agg(f)


# Similar method to group multiple custom functions together into a single DataFrame
# 

data_1 = sales.groupby("Manager").apply(wavg, "New_Product_Price", "Quantity")
data_2 = sales.groupby("Manager").apply(wavg, "Current_Price", "Quantity")


summary = pd.DataFrame(data=dict(s1=data_1, s2=data_2))
summary.columns = ["New Product Price","Current Product Price"]
summary.head()


# Finally, numpy has an average function that can be used:
# 

np.average(sales["Current_Price"], weights=sales["Quantity"])


# Use a lambda function for it to work with grouped data
# 

sales.groupby("Manager").apply(lambda x: np.average(x['New_Product_Price'], weights=x['Quantity']))





# ### Predicting The Future with Python
# #### PYMNTOs, June 8, 2017
# 

import pandas as pd
import numpy as np
from fbprophet import Prophet


get_ipython().magic('matplotlib inline')


# Read in the source data - downloaded from google analytics
df = pd.read_excel('https://github.com/chris1610/pbpython/blob/master/data/All-Web-Site-Data-Audience-Overview.xlsx?raw=True')


df.head()


# Convert to log format
df['Sessions'] = np.log(df['Sessions'])


# Need to name the columns like this in order for prophet to work
df.columns = ["ds", "y"]


# Create the model
m1 = Prophet()
m1.fit(df)


# Predict out a year
future1 = m1.make_future_dataframe(periods=365)
forecast1 = m1.predict(future1)


m1.plot(forecast1);


m1.plot_components(forecast1);


articles = pd.DataFrame({
  'holiday': 'publish',
  'ds': pd.to_datetime(['2014-09-27', '2014-10-05', '2014-10-14', '2014-10-26', '2014-11-9',
                        '2014-11-18', '2014-11-30', '2014-12-17', '2014-12-29', '2015-01-06',
                        '2015-01-20', '2015-02-02', '2015-02-16', '2015-03-23', '2015-04-08',
                        '2015-05-04', '2015-05-17', '2015-06-09', '2015-07-02', '2015-07-13',
                        '2015-08-17', '2015-09-14', '2015-10-26', '2015-12-07', '2015-12-30',
                        '2016-01-26', '2016-04-06', '2016-05-16', '2016-06-15', '2016-08-23',
                        '2016-08-29', '2016-09-06', '2016-11-21', '2016-12-19', '2017-01-17',
                        '2017-02-06', '2017-02-21', '2017-03-06']),
  'lower_window': 0,
  'upper_window': 5,
})
articles.head()


m2 = Prophet(holidays=articles).fit(df)
future2 = m2.make_future_dataframe(periods=365)
forecast2 = m2.predict(future2)
m2.plot(forecast2);


m2.plot_components(forecast2);


forecast2.head()


forecast2["Sessions"] = np.exp(forecast2.yhat).round()


forecast2[["ds", "Sessions"]].tail()





# # Amortization Example in Pandas
# 

# Code to support article at [Practical Business Python](http://pbpython.com/amortization-model.html)
# 

# Setup the imports and matplotlib plotting
# 

import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib


get_ipython().magic('matplotlib inline')


matplotlib.style.use('ggplot')


# Define a function to build an amortization table/schedule and return the value as a pandas DataFrame
# 

def amortization_table(interest_rate, years, payments_year, principal, addl_principal=0, start_date=date.today()):
    """ Calculate the amortization schedule given the loan details
    
     Args:
        interest_rate: The annual interest rate for this loan
        years: Number of years for the loan
        payments_year: Number of payments in a year
        principal: Amount borrowed
        addl_principal (optional): Additional payments to be made each period. Assume 0 if nothing provided.
                                   must be a value less then 0, the function will convert a positive value to
                                   negative
        start_date (optional): Start date. Will start on first of next month if none provided

    Returns:
        schedule: Amortization schedule as a pandas dataframe
        summary: Pandas dataframe that summarizes the payoff information
    """
    # Ensure the additional payments are negative
    if addl_principal > 0:
        addl_principal = -addl_principal
    
    # Create an index of the payment dates
    rng = pd.date_range(start_date, periods=years * payments_year, freq='MS')
    rng.name = "Payment_Date"
    
    # Build up the Amortization schedule as a DataFrame
    df = pd.DataFrame(index=rng,columns=['Payment', 'Principal', 'Interest', 
                                         'Addl_Principal', 'Curr_Balance'], dtype='float')
    
    # Add index by period (start at 1 not 0)
    df.reset_index(inplace=True)
    df.index += 1
    df.index.name = "Period"
    
    # Calculate the payment, principal and interests amounts using built in Numpy functions
    per_payment = np.pmt(interest_rate/payments_year, years*payments_year, principal)
    df["Payment"] = per_payment
    df["Principal"] = np.ppmt(interest_rate/payments_year, df.index, years*payments_year, principal)
    df["Interest"] = np.ipmt(interest_rate/payments_year, df.index, years*payments_year, principal)
        
    # Round the values
    df = df.round(2) 
    
    # Add in the additional principal payments
    df["Addl_Principal"] = addl_principal
    
    # Store the Cumulative Principal Payments and ensure it never gets larger than the original principal
    df["Cumulative_Principal"] = (df["Principal"] + df["Addl_Principal"]).cumsum()
    df["Cumulative_Principal"] = df["Cumulative_Principal"].clip(lower=-principal)
    
    # Calculate the current balance for each period
    df["Curr_Balance"] = principal + df["Cumulative_Principal"]
    
    # Determine the last payment date
    try:
        last_payment = df.query("Curr_Balance <= 0")["Curr_Balance"].idxmax(axis=1, skipna=True)
    except ValueError:
        last_payment = df.last_valid_index()
    
    last_payment_date = "{:%m-%d-%Y}".format(df.loc[last_payment, "Payment_Date"])
        
    # Truncate the data frame if we have additional principal payments:
    if addl_principal != 0:
                
        # Remove the extra payment periods
        df = df.ix[0:last_payment].copy()
        
        # Calculate the principal for the last row
        df.ix[last_payment, "Principal"] = -(df.ix[last_payment-1, "Curr_Balance"])
        
        # Calculate the total payment for the last row
        df.ix[last_payment, "Payment"] = df.ix[last_payment, ["Principal", "Interest"]].sum()
        
        # Zero out the additional principal
        df.ix[last_payment, "Addl_Principal"] = 0
        
    # Get the payment info into a DataFrame in column order
    payment_info = (df[["Payment", "Principal", "Addl_Principal", "Interest"]]
                    .sum().to_frame().T)
       
    # Format the Date DataFrame
    payment_details = pd.DataFrame.from_items([('payoff_date', [last_payment_date]),
                                               ('Interest Rate', [interest_rate]),
                                               ('Number of years', [years])
                                              ])
    # Add a column showing how much we pay each period.
    # Combine addl principal with principal for total payment
    payment_details["Period_Payment"] = round(per_payment, 2) + addl_principal
    
    payment_summary = pd.concat([payment_details, payment_info], axis=1)
    return df, payment_summary
    


# ## Examples of running the function
# 

schedule1, stats1 = amortization_table(0.05, 30, 12, 100000, addl_principal=0)


# Take a look at the start and end of the table as well as the summary stats
# 

schedule1.head()


schedule1.tail()


stats1


# Try running some other scenarios and combining them into a single DataFrame
# 

schedule2, stats2 = amortization_table(0.05, 30, 12, 100000, addl_principal=-200)
schedule3, stats3 = amortization_table(0.04, 15, 12, 100000, addl_principal=0)


# Combine all the scenarios into 1 view
pd.concat([stats1, stats2, stats3], ignore_index=True)


schedule3.head()


# ## Examples of plotting the data
# 

schedule1.plot(x='Payment_Date', y='Curr_Balance', title="Pay Off Timeline");


fig, ax = plt.subplots(1, 1)
schedule1.plot(x='Payment_Date', y='Curr_Balance', label="Scenario 1", ax=ax)
schedule2.plot(x='Payment_Date', y='Curr_Balance', label="Scenario 2", ax=ax)
schedule3.plot(x='Payment_Date', y='Curr_Balance', label="Scenario 3", ax=ax)
plt.title("Pay Off Timelines");


schedule1["Cum_Interest"] = schedule1["Interest"].abs().cumsum()
schedule2["Cum_Interest"] = schedule2["Interest"].abs().cumsum()
schedule3["Cum_Interest"] = schedule3["Interest"].abs().cumsum()

fig, ax = plt.subplots(1, 1)


schedule1.plot(x='Payment_Date', y='Cum_Interest', label="Scenario 1", ax=ax)
schedule2.plot(x='Payment_Date', y='Cum_Interest', label="Scenario 2", ax=ax, style='+')
schedule3.plot(x='Payment_Date', y='Cum_Interest', label="Scenario 3", ax=ax)

ax.legend(loc="best");


fig, ax = plt.subplots(1, 1)

y1_schedule = schedule1.set_index('Payment_Date').resample("A")["Interest"].sum().abs().reset_index()
y1_schedule["Year"] = y1_schedule["Payment_Date"].dt.year
y1_schedule.plot(kind="bar", x="Year", y="Interest", ax=ax, label="30 Years @ 5%")

plt.title("Interest Payments");


