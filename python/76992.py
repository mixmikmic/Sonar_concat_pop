# # Purpose
# 
# When working on a problem in Python, I work on writing code that can first do a job once. Then, I make that code into a function so I can repeat the solution as many times as needed without copying and pasting. This notebook shows my development process for the individual pieces of the stocker class. It is messy on purpose! When developing, I usually concentrate on writing functional and then go back and write proper documentation and clean things up. I preserved this notebook to show how I think things through in code.
# 

# # Stock Price Predictions
# 
# We can also make testable predictions by restricting our model to a subset of the data, and then using the forecasted values to see how correct our model is. We will restrict data up until the end of 2016, and then try to make predictions for 2017.
# 

# quandl for financial data, pandas and numpy for data manipulation
# prophet for additive models
import quandl 
import pandas as pd
import numpy as np
import fbprophet

# Plotting in the Jupyter Notebook
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Plotting parameters
import matplotlib
inline_rc = dict(matplotlib.rcParams)
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 22
matplotlib.rcParams['text.color'] = 'k'



# ## Retrieve Data from Quandl
# 

quandl.ApiConfig.api_key = 'rFsSehe51RLzREtYhLfo'

# Using Tesla data
tesla_stocks = quandl.get('WIKI/TSLA')


# Set the index (dates) to a column 
tesla_stocks.reset_index(level=0, inplace=True)

# Rename the columns for prophet and create a year column
tesla_stocks.rename(columns={'Date': 'ds', 'Adj. Close': 'y'}, inplace=True)
tesla_stocks['Year'] = [date.year for date in tesla_stocks['ds']]


# Training and testing data
tesla_train = tesla_stocks[tesla_stocks['Year'] < 2017]
tesla_test = tesla_stocks[tesla_stocks['Year'] > 2016]


# ## Prophet Model for Predicting Entire Year at Once
# 

# Create a new model 
tesla_prophet = fbprophet.Prophet(changepoint_prior_scale=0.2)

# Train the model
tesla_prophet.fit(tesla_train)

# Number of days to make predictions 
days = (max(tesla_test['ds']) - min(tesla_test['ds'])).days

# Future dataframe
tesla_forecast = tesla_prophet.make_future_dataframe(periods = days, freq = 'D')

# Make forecasts
tesla_forecast = tesla_prophet.predict(tesla_forecast)


# ## Accuracy and Profits (or losses) for Entire Year at Once
# 

tesla_results = tesla_forecast.merge(tesla_test, how = 'inner', on = 'ds')
tesla_results = tesla_results[['ds', 'y', 'yhat']]

# Predicted difference between stock prices
tesla_results['pred_diff'] = (tesla_results['yhat']).diff()

# Actual difference between stock prices
tesla_results['real_diff'] = (tesla_results['y']).diff()


# Correct direction column
tesla_results['correct'] = (np.sign(tesla_results['pred_diff']) == np.sign(tesla_results['real_diff'])) * 1
print('Correct direction predicted: {:0.2f}% of days.'.format(100 * np.mean(tesla_results['correct'])))


# ## Value of Predictions
# 

# Need to include the adjusted open to calculate profits or losses
tesla_results = tesla_results.merge(tesla_stocks[['ds', 'Adj. Open']], how = 'left', on = 'ds')
tesla_results['daily_change'] = abs(tesla_results['y'] - tesla_results['Adj. Open'])


# Only invest on days with predicted increase in stock price
tesla_pred_increase = tesla_results[tesla_results['pred_diff'] > 0]
tesla_pred_increase.reset_index(inplace=True)

profits = []

# If buy and stock goes up, add change to profits (1000 shares)
# If buy and stock goes down, subtract change from profit
for i, correct in enumerate(tesla_pred_increase['correct']):
    if correct == 1:
        profits.append(1000 * tesla_pred_increase.ix[i, 'daily_change'])
    else:
        profits.append(-1 * 1000 * tesla_pred_increase.ix[i, 'daily_change'])
        
tesla_pred_increase['profit'] = profits


print('Predicted profits for entire year at once: {:.0f} $.'.format(np.sum(tesla_pred_increase['profit'])))


smart_profit = 1000 * (tesla_results.ix[len(tesla_results) - 1, 'y'] - tesla_results.ix[0, 'Adj. Open'])
print('Buy and Hold for entire year profits: {:.0f} $.'.format(smart_profit))


# # Visualizations of Entire Year at Once
# 

# Dataframe for plotting
tesla_plot = pd.merge(tesla_stocks, tesla_forecast, on = 'ds', how = 'inner')


# ## Predictions and Actual Values
# 

# Set up the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot the actual values
ax.plot(tesla_plot['ds'], tesla_plot['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
# Plot the predicted values
ax.plot(tesla_plot['ds'], tesla_plot['yhat'], 'darkslateblue',linewidth = 3.2, label = 'Predicted');

# Plot the uncertainty interval as ribbon
ax.fill_between(tesla_plot['ds'].dt.to_pydatetime(), tesla_plot['yhat_upper'], tesla_plot['yhat_lower'], alpha = 0.6, 
               facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Uncertainty')

# Put a vertical line at the start of predictions
plt.vlines(x='2017-01-01', ymin=0, ymax=max(tesla_plot['y']), colors = 'r',
           linestyles='dashed')

# Plot formatting
plt.legend(loc = 2, prop={'size': 14}); plt.xlabel('Date'); plt.ylabel('Price $');
plt.grid(linewidth=0.6, alpha = 0.6)
plt.title('Tesla Stock Price Observed and Predicted');


# ## Predicted and Smart Profits over Year 
# 

# Merge results with the predicted increase containing profits
# tesla results contains actual data and predicted values
tesla_results = pd.merge(tesla_results, tesla_pred_increase[['ds', 'profit']], on = 'ds', how='left')


# Total predicted profit at each day
tesla_results['total_profit'] = tesla_results['profit'].cumsum()


# Forward fill the total predicted profits
tesla_results.ix[0, 'total_profit'] = 0
tesla_results['total_profit'] = tesla_results['total_profit'].ffill()


# Calculate the profits from buying and holding
smart_profits = []
first_open = tesla_results.ix[0, 'Adj. Open']

for i, close in enumerate(tesla_results['y']):
    smart_profits.append(1000 * (close - first_open))
   
# No need for cumulative sum because smart profits are unrealized
tesla_results['total_smart'] = smart_profits


# Final profit and final smart used for locating text
final_profit = tesla_results.ix[len(tesla_results) - 1, 'total_profit']
final_smart = tesla_results.ix[len(tesla_results) - 1, 'total_smart']

# text location
last_date = tesla_results.ix[len(tesla_results) - 1, 'ds']
text_location = (last_date - pd.DateOffset(months = 1)).date()

plt.style.use('dark_background')
plt.figure(figsize=(12, 8))

# Plot smart profits
plt.plot(tesla_results['ds'], tesla_results['total_smart'], 'b',
         linewidth = 3, label = 'Smart Profits') 

# Plot prediction profits
plt.plot(tesla_results['ds'], tesla_results['total_profit'], 
         color = 'g' if final_profit > 0 else 'r',
         linewidth = 3, label = 'Prediction Profits')

# Display final values on graph
plt.text(x = text_location, 
         y =  final_profit + (final_profit / 40),
         s = '%d$' % final_profit,
        color = 'w' if final_profit > 0 else 'r',
        size = 18)
plt.text(x = text_location, 
         y =  final_smart + (final_smart / 40),
         s = '%d$' % final_smart,
        color = 'w' if final_smart > 0 else 'r',
        size = 18);

# Plot formatting
plt.ylabel('Profit  (US $)'); plt.xlabel('Date'); 
plt.title('Tesla 2017 Prediction Profits and Smart Profits');
plt.legend(loc = 2, prop={'size': 16});
plt.grid(alpha=0.2); 


# ### Plotting Lines Segments with Different Colors for Losses and Profits (Not Used)
# 
# This is really complicated code that I copied and pasted from the matplotlib documentation. It does work, but because I don't entirely understand it, I decided to go with the simpler graph.
# 

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


x = np.array(list(range(len(tesla_results))))
xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
           'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

y = np.array(tesla_results['total_profit'])

# Create a colormap for red, green and blue and a norm to color
# f' < -0.5 red, f' > 0.5 blue, and the rest green
cmap = ListedColormap(['r', 'g'])
norm = BoundaryNorm([0], cmap.N)

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create the line collection object, setting the colormapping parameters.
# Have to set the actual values used for colormapping separately.
lc = LineCollection(segments, label = 'Prediction Profits', cmap=cmap, norm=norm)
lc.set_array(y)
lc.set_linewidth(3)

xticks = list(range(0, x.max(), int(x.max() / 12)))

fig1 = plt.figure(figsize=(10,8))
fig1.autofmt_xdate()
plt.gca().add_collection(lc)
plt.plot(x, tesla_results['total_smart'], label = 'Smart Profit')
plt.xticks(xticks, xlabels);
plt.legend();


# # Forecast the next month at Once
# 

# Make new model and fit
forecast_prophet = fbprophet.Prophet(daily_seasonality=False,
                                     changepoint_prior_scale=0.3)
forecast_prophet.fit(tesla_stocks)

# Make future dataframe and predict
future_tesla = forecast_prophet.make_future_dataframe(periods=30, freq='D')
future_tesla = forecast_prophet.predict(future_tesla)


# Only want weekdays
weekdays = [i for i, date in enumerate(future_tesla['ds']) if date.dayofweek < 5]
future_tesla = future_tesla.ix[weekdays, :]

# Only predictions
future_tesla = future_tesla[future_tesla['ds'] > max(tesla_stocks['ds'])]


# Difference and buy/sell recommedation
future_tesla['diff'] = future_tesla['yhat'].diff()
future_tesla = future_tesla.dropna()
future_tesla['buy'] = [1 if diff > 0 else 0 for diff in future_tesla['diff']]


# Create two separate dataframes for plotting
future_buy = future_tesla[future_tesla['buy'] == 1]
future_sell = future_tesla[future_tesla['buy'] == 0]

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# Plot of future predictions
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 18
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the estimates
ax.plot(future_buy['ds'], future_buy['yhat'], 'g^', ms = 16, label = 'Buy')
ax.plot(future_sell['ds'], future_sell['yhat'], 'rv', ms = 16, label = 'Sell')

# Plot errorbars
ax.errorbar(future_tesla['ds'].dt.to_pydatetime(), future_tesla['yhat'], 
            yerr = future_tesla['yhat_upper']- future_tesla['yhat_lower'], 
            capthick=2, color = 'k',linewidth = 3,
           ecolor='darkblue', capsize = 6, elinewidth = 2, label = 'Pred with Range')

# Plot formatting
plt.legend(loc = 2, prop={'size': 12});
plt.xticks(rotation = '45')
plt.ylabel('Predicted Stock Price (US $)');
plt.xlabel('Date'); plt.title('Buy and Sell Predictions for Tesla');


print('Recommeded Buy Dates:\n')
for date, change in zip(future_buy['ds'], future_buy['diff']):
    print('Date: {} \t Predicted Change: {:.2f}$.'.format(date.date(), change))


print('Recommended Sell Dates:\n')
for date, change in zip(future_sell['ds'], future_sell['diff']):
    print('Date: {} \t Predicted Change: {:.2f}$.'.format(date.date(), change))


# # Evaluate on One Year One Day at a Time
# 

# stock = quandl.get('WIKI/TSLA')
# stock.reset_index(level=0, inplace=True)

# stock = stock.rename(columns={'Date': 'ds', 'Adj. Close': 'y'})

# # Year column
# stock['Year'] = [date.year for date in stock['ds']]

# # Only going to use three years of past data 
# stock = stock[stock['Year'] > 2013]

# stock.head()


# We now need to iterate through every day from start of 2017 up until the end of the data, making a prediction one day at a time. We want to train our model on all the data up until the day we are predicting, and then predict only one day. This may take a little time, but we can speed things up if we do not make predictions for weekends. This will be a more realistic prediction than one year at a time, becuase if we were actively trading we would like to train a new model for every day.
# 

# # Make a dataframe to hold all the results
# results = stock.copy()
# results['yhat'] = 0
# results['yhat_upper'] = 0
# results['yhat_lower'] = 0

# # Set the index for locating
# results = results.set_index('ds')


# day_range = pd.date_range(start=pd.datetime(2017, 1, 1), end=max(stock['ds']), freq='D')
# eval_length = (day_range[-1] - day_range[0]).days


# for i, day in enumerate(day_range):
#     print('{:.2f}% Complete.\r'.format(100 * i / eval_length), end='')
#     # Do not make predictions on Sat or Sun
#     if day.weekday() == 5 or day.weekday() == 6:
#         pass
        
#     else:
#         # Select all the training data up until prediction day
#         train = stock[stock['ds'] < day.date()]
#         # Create a prophet model and fit on the training data
#         model = fbprophet.Prophet(daily_seasonality=False, changepoint_prior_scale=0.2)
#         model.fit(train)
        
#         # Make a future dataframe with one day
#         future = model.make_future_dataframe(periods=1, freq='D')
#         future = model.predict(future)
        
#         # Extract the relevant prediction information
#         yhat = future.ix[len(future) - 1, 'yhat']
#         yhat_upper = future.ix[len(future) - 1, 'yhat_upper']
#         yhat_lower = future.ix[len(future) - 1, 'yhat_lower']
        
#         # Assign the dates to the results dataframe
#         results.ix[day, 'yhat'] = yhat
#         results.ix[day, 'yhat_upper'] = yhat_upper
#         results.ix[day, 'yhat_lower'] = yhat_lower
    


# # Visualization Functions
# 

def plot_predictions(stock_df, forecast_df, ticker='TSLA'):
    # merge the two dataframes
    stock_plot = pd.merge(stock_df, forecast_df, on = 'ds', how = 'inner')
    
    # Plot parameters
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    matplotlib.rcParams['axes.labelsize'] = 16
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['axes.titlesize'] = 22
    matplotlib.rcParams['text.color'] = 'k'

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot the actual values
    ax.plot(stock_plot['ds'], stock_plot['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observed')
    ax.plot(stock_plot['ds'], stock_plot['yhat'], 'darkslateblue',linewidth = 3.2, label = 'Predicted');

    # Plot the uncertainty interval
    ax.fill_between(stock_plot['ds'].dt.to_pydatetime(), stock_plot['yhat_upper'], stock_plot['yhat_lower'], alpha = 0.6, 
                   facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Uncertainty')

    # Put a vertical line at the start of predictions
    plt.vlines(x='2017-01-01', ymin=0, ymax=max(stock_plot['y']), colors = 'r',
               linestyles='dashed')
    plt.legend(loc = 2, prop={'size': 14}); plt.xlabel('Date'); plt.ylabel('Price $');
    plt.grid(linewidth=0.6, alpha = 0.6)
    plt.title('%s Price Observed and Predicted' % ticker);


def plot_profits(results, ticker):
    # Total predicted profit at each day
    results['total_profit'] = results['profit'].cumsum()
    
    # Forward fill the total predicted profits
    results.ix[0, 'total_profit'] = 0
    results['total_profit'] = results['total_profit'].ffill()
    
    # Calculate the profits from buying and holding
    smart_profits = []
    first_open = results.ix[0, 'Adj. Open']

    for i, close in enumerate(results['y']):
        smart_profits.append(1000 * (close - first_open))
    
    # Column with daily profits
    # No need for cumulative total because smart profits are unrealized
    results['total_smart'] = smart_profits
    
    # Final total profit and smart profit for plotting
    final_profit = results.ix[len(results) - 1, 'total_profit']
    final_smart = results.ix[len(results) - 1, 'total_smart']
    
    # Last date for location of text annotation
    last_date = results.ix[len(results) - 1, 'ds']
    text_location = (last_date - pd.DateOffset(months = 1)).date()

    # Plot parameters
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['axes.labelsize'] = 16
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['axes.titlesize'] = 22
    matplotlib.rcParams['text.color'] = 'k'
    
    # Set the style
    plt.style.use('dark_background')
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Smart Profits
    plt.plot(results['ds'], results['total_smart'], 'b',
         linewidth = 3.0, label = 'Smart Profits') 
    
    # Prediction Profits
    plt.plot(results['ds'], results['total_profit'], 
         color = 'g' if final_profit > 0 else 'r',
         linewidth = 3.0, label = 'Prediction Profits')
    
    # Labels and Title
    plt.ylabel('Profit  (US $)'); plt.xlabel('Date'); 
    plt.title('%s 2017 Prediction Profits and Smart Profits' % ticker);
    
    # Legend and grid
    plt.legend(loc = 2, prop={'size': 16});
    plt.grid(alpha=0.6); 
    
    # Add label for final total profit
    plt.text(x = text_location, 
         y =  final_profit +  (final_profit / 40),
         s = '%d $' % final_profit,
        color = 'w' if final_profit > 0 else 'r',
        size = 16)
    # Add label for final smart profits
    plt.text(x = text_location, 
         y =  final_smart + (final_smart / 40),
         s = '%d $' % final_smart,
        color = 'w' if final_smart > 0 else 'r',
        size = 16);


# ## Forecast and Graph Function
# 

def forecast_and_plot(stock, ticker, changepoint_prior_scale=0.2):
    # Make new model and fit
    forecast_prophet = fbprophet.Prophet(daily_seasonality=False,
                                         changepoint_prior_scale=changepoint_prior_scale)
    forecast_prophet.fit(stock)

    # Make future dataframe and predict
    future = forecast_prophet.make_future_dataframe(periods=30, freq='D')
    future = forecast_prophet.predict(future)

    # Only want weekdays
    weekdays = [i for i, date in enumerate(future['ds']) if date.dayofweek < 5]
    future = future.ix[weekdays, :]

    # Only predictions
    future = future[future['ds'] >= max(stock['ds'])]
    
    # Difference between predictions
    future['diff'] = future['yhat'].diff()
    
    # Drop rows with na values
    future = future.dropna()
    
    # Buy/sell recommendation
    future['buy'] = [1 if diff > 0 else 0 for diff in future['diff']]

    # Create two separate dataframes for plotting
    future_buy = future[future['buy'] == 1]
    future_sell = future[future['buy'] == 0]

    # Plotting Parameters
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['axes.titlesize'] = 18
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(future_buy['ds'], future_buy['yhat'], 'g^', ms = 16, label = 'Buy')
    ax.plot(future_sell['ds'], future_sell['yhat'], 'rv', ms = 16, label = 'Sell')
    ax.errorbar(future['ds'].dt.to_pydatetime(), future['yhat'], 
            yerr = future['yhat_upper']- future['yhat_lower'], 
            capthick=2, color = 'k',linewidth = 3,
           ecolor='darkblue', capsize = 6, elinewidth = 2, label = 'Pred with Range')

    plt.legend(loc = 2, prop={'size': 12});
    plt.xticks(rotation = '45')
    plt.ylabel('Predicted Stock Price (US $)');
    plt.xlabel('Date'); plt.title('Buy and Sell Predictions for %s' % ticker);

    # Predicted buy dates
    print('Recommeded Buy Dates:\n')
    for date, change in zip(future_buy['ds'], future_buy['diff']):
        print('Date: {} \t Predicted Change: {:.2f}$.'.format(date.date(), change))
    
    # Predicted sell dates
    print('\nRecommended Sell Dates:\n')
    for date, change in zip(future_sell['ds'], future_sell['diff']):
        print('Date: {} \t Predicted Change: {:.2f}$.'.format(date.date(), change))
    
    print('\n')


# # Stock Predictor Function 
# 
# This will be turned into a class because it makes more sense.
# 

def stock_predictor(ticker='TSLA'):
    try:
        # Using years from 2011 onwards
        stock = quandl.get('WIKI/%s' % ticker.upper(), start_date = '2011-01-01')
    except Exception as e:
        print('Invalid Stock Ticker')
        print(e)
        return
    
    # Change the index to a Date column
    stock_clean = stock.reset_index()[['Date', 'Adj. Close', 'Adj. Open']]
    
    # Create a year column
    stock_clean['Year'] = [date.year for date in stock_clean['Date']]
    
    # Rename for prophet training
    stock_clean = stock_clean.rename(columns={'Date': 'ds', 'Adj. Close': 'y'})
    
    # Training and Testing Sets
    stock_train = stock_clean[stock_clean['Year'] < 2017]
    stock_test = stock_clean[stock_clean['Year'] > 2016]
    
    # Create the prophet model and fit on training set
    stock_prophet = fbprophet.Prophet(daily_seasonality=False,
                                      changepoint_prior_scale=0.2)
    stock_prophet.fit(stock_train)
    
    # Number of days to predict
    days = (max(stock_test['ds']) - min(stock_test['ds'])).days
    
    # Make forecasts for entire length of test set + one week
    stock_forecast = stock_prophet.make_future_dataframe(periods=days + 7, freq = 'D')
    stock_forecast = stock_prophet.predict(stock_forecast)
    
    # Plot the entire series
    plot_predictions(stock_clean, stock_forecast, ticker)
    
    # Dataframe for predictions and test values
    results = stock_forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    
    # Merge to get acutal values (y)
    results = pd.merge(results, stock_test[['ds', 'y']], on = 'ds', how='right')
    
    # Merge to get daily opening values (Adj. Open)
    results = results.merge(stock_clean[['ds', 'Adj. Open']], on = 'ds', how ='inner')
    
    # Columns of daily changes
    results['pred_diff'] = results['yhat'].diff()
    results['real_diff'] = results['y'].diff()
    
    # Whether the prediction was right or wrong
    # Multiply by 1 to convert to an integer
    results['correct'] = (np.sign(results['pred_diff']) == np.sign(results['real_diff'])) * 1
    
    # Calculate daily change in price
    results['daily_change'] = abs(results['y'] - results['Adj. Open'])
    
    # Only buy if predicted to increase
    results_pred_increase = results[results['pred_diff'] > 0]
        
    # Calculate profits or losses
    profits = []
    for i, correct in enumerate(results_pred_increase['correct']):
        if correct == 1:
            profits.append(1000 * results.ix[i, 'daily_change'])
        else:
            profits.append(-1 * 1000 * results.ix[i, 'daily_change'])
    
    results_pred_increase['profit'] = profits
    
    # Dataframe for plotting profits
    results = pd.merge(results, results_pred_increase[['ds', 'profit']], on = 'ds', how = 'left')
    
    # Plot the profits (or losses)
    plot_profits(results, ticker)
    
    # Calculate total profit if buying 1000 shares every day
    total_profit = int(np.sum(profits))
    
    # Calculate total profit if buying and holding 1000 shares for entire time
    first_price = int(results[results['ds'] == min(results['ds'])]['y'])
    last_price = int(results[results['ds'] == max(results['ds'])]['y'])

    # Smart profit
    smart_profit = (last_price - first_price) * 1000
    
    # Total accuracy is percentage of correct predictions
    accuracy = np.mean(results['correct']) *  100
    
    performance = {'pred_profit': total_profit, 'smart_profit': smart_profit, 'accuracy': accuracy}
    
    print('You were correct {:.2f}% of the time.'.format(accuracy))
    print('Your profit from playing the stock market in {}: {:.0f} $.'.format(ticker, total_profit))
    print('The buy and hold profit (smart strategy) for {}: {:.0f} $.'.format(ticker, smart_profit))
    print('Thanks for playing the stock market!')
    
    return(stock_clean)


stock = stock_predictor('TSLA')


forecast_and_plot(stock, ticker = 'TSLA')


stock = stock_predictor('CAT')


stock = stock_predictor('MSFT')


forecast_and_plot(stock, ticker = 'MSFT')





# # Purpose
# 
# After writing a ton of separate functions to act on stock data, I realized a much better approach was to make a class that could hold data and also had different functions for prediction, plotting, analysis and anything else that might be needed. Having already developed all of these functions separately, I thought it made a lot of sense to combine them into a single class. This notebook will build up the class, covering many data manipulation, plotting, and even modeling concepts in a single example! 
# 
# After completing the class in this notebook, I then moved it to a separate Python script, stocker.py, so I could import it from an interactive Python session or a Jupyter Notebook session. My development process was therefore: 
# 
# 1. Write code to do each desired task once (in Jupyter Notebook).
# 2. Encapsulate code in functions (in Jupyter).
# 3. Write a class to group all functions together (in Jupyter). 
# 4. Export the finished class to a Python script (I use Sublime Text 3 for editing entire scripts).
# 5. Import the class into a different Jupyter notebook and test out the functions.
# 6. Fix the functions that do not work and update the Python script as required in Sublime Text. 
# 
# The work in this notebook falls under steps 2 and 3. Another notebook (Stocker Development) covered step 1. The result of step 4 can be found in stocker.py. Step 5 is documented in Stocker Analysis Usage and Stocker Prediction Usage. Step 6 was done in Sublime Text and the script shows the final result. 
# 
# Stocker is still a work in progress, and I encourage anyone to try it out and let me know what doesn't work! 
# 

# Quandl for financial analysis, pandas and numpy for data manipulation
# fbprophet for additive models, #pytrends for Google trend data
import quandl
import pandas as pd
import numpy as np
import fbprophet
import pytrends
from pytrends.request import TrendReq

# matplotlib pyplot for plotting
import matplotlib.pyplot as plt

import matplotlib


# Class for analyzing and (attempting) to predict future prices
# Contains a number of visualizations and analysis methods
class Stocker():
    
    # Initialization requires a ticker symbol
    def __init__(self, ticker):
        
        # Enforce capitalization
        ticker = ticker.upper()
        
        # Symbol is used for labeling plots
        self.symbol = ticker
        
        # api personal key
        quandl.ApiConfig.api_key = 'rFsSehe51RLzREtYhLfo'

        # Retrieval the financial data
        try:
            stock = quandl.get('WIKI/%s' % ticker)
        
        except Exception as e:
            print('Error Retrieving Data.')
            print(e)
            return
        
        # Set the index to a column called Date
        stock = stock.reset_index(level=0)
        
        # Columns required for prophet
        stock['ds'] = stock['Date']
        stock['y'] = stock['Adj. Close']
        stock['Daily Change'] = stock['Adj. Close'] - stock['Adj. Open']
        
        # Data assigned as class attribute
        self.stock = stock.copy()
        
        # Minimum and maximum date in range
        self.min_date = min(stock['Date'])
        self.max_date = max(stock['Date'])
        
        # Find max and min prices and dates on which they occurred
        self.max_price = np.max(self.stock['y'])
        self.min_price = np.min(self.stock['y'])
        
        self.min_price_date = self.stock[self.stock['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.stock[self.stock['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]
        
        # The starting price (starting with the opening price)
        self.starting_price = float(self.stock.ix[0, 'Adj. Open'])
        
        # The most recent price
        self.most_recent_price = float(self.stock.ix[len(self.stock) - 1, 'y'])
        
        # Prophet parameters
        self.changepoint_prior_scale = 0.2
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None
        
        print('{} Stocker Initialized. Data covers {} to {}.'.format(self.symbol,
                                                                     self.min_date.date(),
                                                                     self.max_date.date()))
        
    # Basic Historical Plots and Basic Statistics
    def plot_stock(self, start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic'):
        
        self.reset_plot()
        
        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
            
        # Convert to pandas date time
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if end_date.date() < start_date.date():
            print('End Date must be later than start date.')
            return
        
        # Check to make sure dates are in the data
        if (start_date not in list(self.stock['Date'])):
            print('Start Date not in data (either out of range or not a trading day.)')
            return
        elif (end_date not in list(self.stock['Date'])):
            print('End Date not in data (either out of range or not a trading day.)')
            return 
            
        # Useful statistics
        
        
        stock_plot = self.stock[(self.stock['Date'] >= start_date.date()) & (self.stock['Date'] <= end_date.date())]
        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        
        for i, stat in enumerate(stats):
            
            stat_min = min(stock_plot[stat])
            stat_max = max(stock_plot[stat])

            stat_avg = np.mean(stock_plot[stat])
            
            date_stat_min = stock_plot[stock_plot[stat] == stat_min]['Date']
            date_stat_min = date_stat_min[date_stat_min.index[0]].date()
            date_stat_max = stock_plot[stock_plot[stat] == stat_max]['Date']
            date_stat_max = date_stat_max[date_stat_max.index[0]].date()
            
            current_stat = float(stock_plot[stock_plot['Date'] == end_date][stat])
            
            print('Maximum {} = {:.2f} on {}.'.format(stat, stat_max, date_stat_max))
            print('Minimum {} = {:.2f} on {}.'.format(stat, stat_min, date_stat_min))
            print('Current {} = {:.2f}.\n'.format(stat, current_stat))
            
            # Percentage y-axis
            if plot_type == 'pct':
                # Simple Plot 
                plt.style.use('fivethirtyeight');
                if stat == 'Daily Change':
                    plt.plot(stock_plot['Date'], 100 * stock_plot[stat],
                         color = colors[i], linewidth = 2.4, alpha = 0.9,
                         label = stat)
                else:
                    plt.plot(stock_plot['Date'], 100 * (stock_plot[stat] -  stat_avg) / stat_avg,
                         color = colors[i], linewidth = 2.4, alpha = 0.9,
                         label = stat)

                plt.xlabel('Date'); plt.ylabel('Change Relative to Average (%)'); plt.title('%s Stock History' % self.symbol); 
                plt.legend(prop={'size':10})
                plt.grid(color = 'k', alpha = 0.4); 

            # Stat y-axis
            elif plot_type == 'basic':
                plt.style.use('fivethirtyeight');
                plt.plot(stock_plot['Date'], stock_plot[stat], color = colors[i], linewidth = 3, label = stat, alpha = 0.8)
                plt.xlabel('Date'); plt.ylabel('US $'); plt.title('%s Stock History' % self.symbol); 
                plt.legend(prop={'size':10})
                plt.grid(color = 'k', alpha = 0.4); 
      
        plt.show();
        
    # Reset the plotting parameters to clear style formatting
    # Not sure if this should be a static method
    @staticmethod
    def reset_plot():
        
        # Restore default parameters
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        
        # Adjust a few parameters to liking
        matplotlib.rcParams['figure.figsize'] = (8, 5)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'
    
    # Method to linearly interpolate prices on the weekends
    def resample(self, dataframe):
        # Change the index and resample at daily level
        dataframe = dataframe.set_index('ds')
        dataframe = dataframe.resample('D')
        
        # Reset the index and interpolate nan values
        dataframe = dataframe.reset_index(level=0)
        dataframe = dataframe.interpolate()
        return dataframe
    
    # Remove weekends from a dataframe
    def remove_weekends(self, dataframe):
        
        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)
        
        weekends = []
        
        # Find all of the weekends
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)
            
        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)
        
        return dataframe
    
    
    # Calculate and plot profit from buying and holding shares for specified date range
    def buy_and_hold(self, start_date=None, end_date=None, nshares=1):
        self.reset_plot()
        
        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
            
        # Convert to pandas datetime for indexing dataframe
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if end_date.date() < start_date.date():
            print('End Date must be later than start date.')
            return
        
        # Check to make sure dates are in the data
        if (start_date not in list(self.stock['Date'])):
            print('Start Date not in data (either out of range or not a trading day.)')
            return
        elif (end_date not in list(self.stock['Date'])):
            print('End Date not in data (either out of range or not a trading day.)')
            return 
        
        # Find starting and ending price of stock
        start_price = float(self.stock[self.stock['Date'] == start_date]['Adj. Open'])
        end_price = float(self.stock[self.stock['Date'] == end_date]['Adj. Close'])
        
        # Make a profit dataframe and calculate profit column
        profits = self.stock[(self.stock['Date'] >= start_date) & (self.stock['Date'] <= end_date)]
        profits['hold_profit'] = nshares * (profits['Adj. Close'] - start_price)
        
        # Total profit
        total_hold_profit = nshares * (end_price - start_price)
        
        print('{} Total buy and hold profit from {} to {} for {} shares = ${:.2f}'.format
              (self.symbol, start_date.date(), end_date.date(), nshares, total_hold_profit))
        
        # Plot the total profits 
        plt.style.use('dark_background')
        
        # Location for number of profit
        text_location = (end_date - pd.DateOffset(months = 1)).date()
        
        # Plot the profits over time
        plt.plot(profits['Date'], profits['hold_profit'], 'b', linewidth = 3)
        plt.ylabel('Profit ($)'); plt.xlabel('Date'); plt.title('Buy and Hold Profits for {} {} to {}'.format(
                                                                self.symbol, start_date.date(), end_date.date()))
        
        # Display final value on graph
        plt.text(x = text_location, 
             y =  total_hold_profit + (total_hold_profit / 40),
             s = '$%d' % total_hold_profit,
            color = 'g' if total_hold_profit > 0 else 'r',
            size = 14)
        
        plt.grid(alpha=0.2)
        plt.show();
        
    # Create a prophet model without training
    def create_model(self):

        # Make the model
        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,  
                                  weekly_seasonality=self.weekly_seasonality, 
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)
        
        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        
        return model
    
    # Graph the effects of altering the changepoint prior scale (cps)
    def changepoint_prior_analysis(self, changepoint_priors=[0.001, 0.05, 0.1, 0.2], colors=['b', 'r', 'grey', 'gold']):
    
        # Training and plotting with 4 years of data
        train = self.stock[(self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=3)).date())]
        
        # Iterate through all the changepoints and make models
        for i, prior in enumerate(changepoint_priors):
            # Select the changepoint
            self.changepoint_prior_scale = prior
            
            # Create and train a model with the specified cps
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=180, freq='D')
            
            # Make a dataframe to hold predictions
            if i == 0:
                predictions = future.copy()
                
            future = model.predict(future)
            
            # Fill in prediction dataframe
            predictions['%.3f_yhat_upper' % prior] = future['yhat_upper']
            predictions['%.3f_yhat_lower' % prior] = future['yhat_lower']
            predictions['%.3f_yhat' % prior] = future['yhat']
         
        # Remove the weekends
        predictions = self.remove_weekends(predictions)
        
        # Plot set-up
        self.reset_plot()
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(1, 1)
        
        # Actual observations
        ax.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Observations')
        color_dict = {prior: color for prior, color in zip(changepoint_priors, colors)}

        # Plot each of the changepoint predictions
        for prior in changepoint_priors:
            # Plot the predictions themselves
            ax.plot(predictions['ds'], predictions['%.3f_yhat' % prior], linewidth = 1.2,
                     color = color_dict[prior], label = '%.3f prior scale' % prior)
            
            # Plot the uncertainty interval
            ax.fill_between(predictions['ds'].dt.to_pydatetime(), predictions['%.3f_yhat_upper' % prior],
                            predictions['%.3f_yhat_lower' % prior], facecolor = color_dict[prior],
                            alpha = 0.3, edgecolor = 'k', linewidth = 0.6)
                            
        # Plot labels
        plt.legend(prop={'size': 10})
        plt.xlabel('Date'); plt.ylabel('Stock Price ($)'); plt.title('Effect of Changepoint Prior Scale');
        plt.show()
            
    # Basic prophet model for specified number of days  
    def create_prophet_model(self, days=0, resample=False):
        
        self.reset_plot()
        
        model = self.create_model()
        
        # Fit on the stock history for past 3 years
        stock_history = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = 3)).date()]
        
        if resample:
            stock_history = self.resample(stock_history)
        
        model.fit(stock_history)
        
        # Make and predict for next year with future dataframe
        future = model.make_future_dataframe(periods = days, freq='D')
        future = model.predict(future)
        
        if days > 0:
            # Print the predicted price
            print('Predicted Price on {} = ${:.2f}'.format(
                future.ix[len(future) - 1, 'ds'].date(), future.ix[len(future) - 1, 'yhat']))

            title = '%s Historical and Predicted Stock Price'  % self.symbol
        else:
            title = '%s Historical and Modeled Stock Price' % self.symbol
        
        # Set up the plot
        fig, ax = plt.subplots(1, 1)

        # Plot the actual values
        ax.plot(stock_history['ds'], stock_history['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        
        # Plot the predicted values
        ax.plot(future['ds'], future['yhat'], 'forestgreen',linewidth = 2.4, label = 'Modeled');

        # Plot the uncertainty interval as ribbon
        ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.3, 
                       facecolor = 'g', edgecolor = 'k', linewidth = 1.4, label = 'Uncertainty')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10}); plt.xlabel('Date'); plt.ylabel('Price $');
        plt.grid(linewidth=0.6, alpha = 0.6)
        plt.title(title);
        plt.show()
        
        return model, future
      
    # Evaluate prediction model for one year
    def evaluate_prediction(self, start_date=None, end_date=None, nshares = 1000):
        
        # Default start date is one year before end of data
        # Default end date is end date of data
        if start_date is None:
            start_date = self.max_date - pd.DateOffset(years=1)
        if end_date is None:
            end_date = self.max_date
            
        # Convert to pandas datetime for indexing dataframe
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if end_date.date() < start_date.date():
            print('End Date must be later than start date.')
            return
        
        # Check to make sure dates are in the data
        if (start_date not in list(self.stock['Date'])):
            print('Start Date not in data (either out of range or not a trading day.)')
            return
        elif (end_date not in list(self.stock['Date'])):
            print('End Date not in data (either out of range or not a trading day.)')
            return 
        
        # Training data starts 3 years before start date and goes up to start date
        train = self.stock[(self.stock['Date'] < start_date.date()) & 
                           (self.stock['Date'] > (start_date - pd.DateOffset(years=3)).date())]
        
        # Testing data is specified in the range
        test = self.stock[(self.stock['Date'] >= start_date.date()) & (self.stock['Date'] <= end_date.date())]
        
        # Create and train the model
        model = self.create_model()
        model.fit(train)
        
        # Make a future dataframe and predictions
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)
        
        # Merge predictions with the known values
        test = pd.merge(test, future, on = 'ds', how = 'inner')
        
        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()
        
        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1
        
        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])
        
        # Only playing the stocks when we predict the stock will increase
        test_pred_increase = test[test['pred_diff'] > 0]
        
        test_pred_increase.reset_index(inplace=True)
        prediction_profit = []
        
        # Iterate through all the predictions and calculate profit from playing
        for i, correct in enumerate(test_pred_increase['correct']):
            
            # If we predicted up and the price goes up, we gain the difference
            if correct == 1:
                prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
            # If we predicted up and the price goes down, we lose the difference
            else:
                prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
        
        test_pred_increase['pred_profit'] = prediction_profit
        
        # Put the profit into the test dataframe
        test = pd.merge(test, test_pred_increase[['ds', 'pred_profit']], on = 'ds', how = 'left')
        test.ix[0, 'pred_profit'] = 0
    
        # Profit for either method at all dates
        test['pred_profit'] = test['pred_profit'].cumsum().ffill()
        test['hold_profit'] = nshares * (test['y'] - float(test.ix[0, 'y']))
        
        # Display information
        print('You played the stock market in {} from {} to {} with {} shares.\n'.format(
            self.symbol, start_date.date(), end_date.date(), nshares))
        
        print('Predicted price on {} = ${:.2f}.'.format(max(future['ds']).date(), future.ix[len(future) - 1, 'yhat']))
        print('Actual price on    {} = ${:.2f}.\n'.format(max(test['ds']).date(), test.ix[len(test) - 1, 'y']))
        
        # Display some friendly information about the perils of playing the stock market
        print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
        print('When the model predicted a decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))
        print('The total profit using the Prophet model = ${:.2f}.'.format(np.sum(prediction_profit)))
        print('The Buy and Hold strategy profit =         ${:.2f}.'.format(float(test.ix[len(test) - 1, 'hold_profit'])))
        print('\nThanks for playing the stock market!\n')
        
        # Reset the plot
        self.reset_plot()
        
        # Set up the plot
        fig, ax = plt.subplots(1, 1)

        # Plot the actual values
        ax.plot(train['ds'], train['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        ax.plot(test['ds'], test['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        
        # Plot the predicted values
        ax.plot(future['ds'], future['yhat'], 'navy', linewidth = 2.4, label = 'Predicted');

        # Plot the uncertainty interval as ribbon
        ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.6, 
                       facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Uncertainty')

        # Put a vertical line at the start of predictions
        plt.vlines(x=min(test['ds']).date(), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                   linestyles='dashed', label = 'Prediction Start')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 8}); plt.xlabel('Date'); plt.ylabel('Price $');
        plt.grid(linewidth=0.6, alpha = 0.6)
                   
        plt.title('%s Prophet Evaluation for Past Year' % self.symbol);
        plt.show();
        
        # Plot the predicted and actual profits over time
        self.reset_plot()
        
        # Final profit and final smart used for locating text
        final_profit = test.ix[len(test) - 1, 'pred_profit']
        final_smart = test.ix[len(test) - 1, 'hold_profit']

        # text location
        last_date = test.ix[len(test) - 1, 'ds']
        text_location = (last_date - pd.DateOffset(months = 1)).date()

        plt.style.use('dark_background')

        # Plot smart profits
        plt.plot(test['ds'], test['hold_profit'], 'b',
                 linewidth = 1.8, label = 'Buy and Hold') 

        # Plot prediction profits
        plt.plot(test['ds'], test['pred_profit'], 
                 color = 'g' if final_profit > 0 else 'r',
                 linewidth = 1.8, label = 'Prediction')

        # Display final values on graph
        plt.text(x = text_location, 
                 y =  final_profit + (final_profit / 40),
                 s = '$%d' % final_profit,
                color = 'g' if final_profit > 0 else 'r',
                size = 18)
        
        plt.text(x = text_location, 
                 y =  final_smart + (final_smart / 40),
                 s = '$%d' % final_smart,
                color = 'g' if final_smart > 0 else 'r',
                size = 18);

        # Plot formatting
        plt.ylabel('Profit  (US $)'); plt.xlabel('Date'); 
        plt.title('Predicted versus Buy and Hold Profits');
        plt.legend(loc = 2, prop={'size': 10});
        plt.grid(alpha=0.2); 
        plt.show()
        
    def retrieve_google_trends(self, search, date_range):
        
        # Set up the trend fetching object
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [search]

        try:
        
            # Create the search object
            pytrends.build_payload(kw_list, cat=0, timeframe=date_range[0], geo='', gprop='news')
            
            # Retrieve the interest over time
            trends = pytrends.interest_over_time()

            related_queries = pytrends.related_queries()

        except Exception as e:
            print('\nGoogle Search Trend retrieval failed.')
            print(e)
            return
        
        return trends, related_queries
        
    def changepoint_date_analysis(self, search=None):
        self.reset_plot()

        model = self.create_model()
        
        # Use past three years of data
        train = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = 3)).date()]
        model.fit(train)
        
        # Predictions of the training data (no future periods)
        future = model.make_future_dataframe(periods=0, freq='D')
        future = model.predict(future)
    
        train = pd.merge(train, future[['ds', 'yhat']], on = 'ds', how = 'inner')
        
        changepoints = model.changepoints
        train = train.reset_index(drop=True)
        
        # Create dataframe of only changepoints
        change_indices = []
        for changepoint in (changepoints):
            change_indices.append(train[train['ds'] == changepoint.date()].index[0])
        
        c_data = train.ix[change_indices, :]
        deltas = model.params['delta'][0]
        
        c_data['delta'] = deltas
        c_data['abs_delta'] = abs(c_data['delta'])
        
        # Sort the values by maximum change
        c_data = c_data.sort_values(by='abs_delta', ascending=False)

        # Limit to 10 largest changepoints
        c_data = c_data[:10]

        # Separate into negative and positive changepoints
        cpos_data = c_data[c_data['delta'] > 0]
        cneg_data = c_data[c_data['delta'] < 0]

        # Changepoints and data
        if not search:
        
            print('\nChangepoints sorted by slope rate of change (2nd derivative):\n')
            print(c_data.ix[:, ['Date', 'Adj. Close', 'delta']][:5])

            # Line plot showing actual values, estimated values, and changepoints
            self.reset_plot()
            
            # Set up line plot 
            plt.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Stock Price')
            plt.plot(future['ds'], future['yhat'], color = 'navy', linewidth = 2.0, label = 'Modeled')
            
            # Changepoints as vertical lines
            plt.vlines(cpos_data['ds'].dt.to_pydatetime(), ymin = min(train['y']), ymax = max(train['y']), 
                       linestyles='dashed', color = 'r', 
                       linewidth= 1.2, label='Negative Changepoints')

            plt.vlines(cneg_data['ds'].dt.to_pydatetime(), ymin = min(train['y']), ymax = max(train['y']), 
                       linestyles='dashed', color = 'darkgreen', 
                       linewidth= 1.2, label='Positive Changepoints')

            plt.legend(prop={'size':10});
            plt.xlabel('Date'); plt.ylabel('Price ($)'); plt.title('Stock Price with Changepoints')
            plt.show()
        
        # Search for search term in google news
        # Show related queries, rising related queries
        # Graph changepoints, search frequency, stock price
        if search:
            date_range = ['%s %s' % (str(min(train['Date']).date()), str(max(train['Date']).date()))]

            # Get the Google Trends for specified terms and join to training dataframe
            trends, related_queries = self.retrieve_google_trends(search, date_range)

            if (trends is None)  or (related_queries is None):
                print('No search trends found for %s' % search)
                return

            print('\n Top Related Queries: \n')
            print(related_queries[search]['top'].head())

            print('\n Rising Related Queries: \n')
            print(related_queries[search]['rising'].head())

            # Upsample the data for joining with training data
            trends = trends.resample('D')

            trends = trends.reset_index(level=0)
            trends = trends.rename(columns={'date': 'ds', search: 'freq'})

            # Interpolate the frequency
            trends['freq'] = trends['freq'].interpolate()

            # Merge with the training data
            train = pd.merge(train, trends, on = 'ds', how = 'inner')

            # Normalize values
            train['y_norm'] = train['y'] / max(train['y'])
            train['freq_norm'] = train['freq'] / max(train['freq'])
            
            self.reset_plot()

            # Plot the normalized stock price and normalize search frequency
            plt.plot(train['ds'], train['y_norm'], 'k-', label = 'Stock Price')
            plt.plot(train['ds'], train['freq_norm'], color='goldenrod', label = 'Search Frequency')

            # Changepoints as vertical lines
            plt.vlines(cpos_data['ds'].dt.to_pydatetime(), ymin = 0, ymax = 1, 
                       linestyles='dashed', color = 'r', 
                       linewidth= 1.2, label='Negative Changepoints')

            plt.vlines(cneg_data['ds'].dt.to_pydatetime(), ymin = 0, ymax = 1, 
                       linestyles='dashed', color = 'darkgreen', 
                       linewidth= 1.2, label='Positive Changepoints')

            # Plot formatting
            plt.legend(prop={'size': 10})
            plt.xlabel('Date'); plt.ylabel('Normalized Values'); 
            plt.title('%s Stock Price and Search Frequency for %s' % (self.symbol, search))
            plt.show()
        
    # Predict the future price for a given range of days
    def predict_future(self, days=30):
        
        # Use past three years for training
        train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=3)).date()]
        
        model = self.create_model()
        
        model.fit(train)
        
        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)
        
        # Only concerned with future dates
        future = future[future['ds'] >= max(self.stock['Date']).date()]
        
        # Remove the weekends
        future = self.remove_weekends(future)
        
        # Calculate whether increase or not
        future['diff'] = future['yhat'].diff()
    
        future = future.dropna()

        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff'] > 0) * 1
        
        # Rename the columns for presentation
        future = future.rename(columns={'ds': 'Date', 'yhat': 'estimate', 'diff': 'change', 
                                        'yhat_upper': 'upper', 'yhat_lower': 'lower'})
        
        future_increase = future[future['direction'] == 1]
        future_decrease = future[future['direction'] == 0]
        
        # Print out the dates
        print('\nPredicted Increase: \n')
        print(future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        print('\nPredicted Decrease: \n')
        print(future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        self.reset_plot()
        
        # Set up plot
        plt.style.use('fivethirtyeight')
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 12
        
        # Plot the predictions and indicate if increase or decrease
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot the estimates
        ax.plot(future_increase['Date'], future_increase['estimate'], 'g^', ms = 12, label = 'Pred. Increase')
        ax.plot(future_decrease['Date'], future_decrease['estimate'], 'rv', ms = 12, label = 'Pred. Decrease')

        # Plot errorbars
        ax.errorbar(future['Date'].dt.to_pydatetime(), future['estimate'], 
                    yerr = future['upper'] - future['lower'], 
                    capthick=1.4, color = 'k',linewidth = 2,
                   ecolor='darkblue', capsize = 4, elinewidth = 1, label = 'Pred with Range')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10});
        plt.xticks(rotation = '45')
        plt.ylabel('Predicted Stock Price (US $)');
        plt.xlabel('Date'); plt.title('Predictions for %s' % self.symbol);
        plt.show()
        
    def changepoint_prior_validation(self, changepoint_priors = [0.001, 0.05, 0.1, 0.2]):
                               
        # Select three years of training data starting 4 years ago and going until 3 years ago
        train = self.stock[(self.stock['Date'] < (max(self.stock['Date']) - pd.DateOffset(years=1)).date()) & 
                           (self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=4)).date())]
        
        # Testing data used for answers
        test = self.stock[(self.stock['Date'] >= (max(self.stock['Date']) - pd.DateOffset(years=1)).date())]
        eval_days = (max(test['Date']).date() - min(test['Date']).date()).days
        
        results = pd.DataFrame(0, index = list(range(len(changepoint_priors))), 
                               columns = ['cps', 'train_err', 'train_range', 'test_err', 'test_range'])
        
        # Iterate through all the changepoints and make models
        for i, prior in enumerate(changepoint_priors):
            results.ix[i, 'cps'] = prior
            
            # Select the changepoint
            self.changepoint_prior_scale = prior
            
            # Create and train a model with the specified cps
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=eval_days, freq='D')
                
            future = model.predict(future)
            
            # Training results and metrics
            train_results = pd.merge(train, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_train_error = np.mean(abs(train_results['y'] - train_results['yhat']))
            avg_train_uncertainty = np.mean(abs(train_results['yhat_upper'] - train_results['yhat_lower']))
            
            results.ix[i, 'train_err'] = avg_train_error
            results.ix[i, 'train_range'] = avg_train_uncertainty
            
            # Testing results and metrics
            test_results = pd.merge(test, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_test_error = np.mean(abs(test_results['y'] - test_results['yhat']))
            avg_test_uncertainty = np.mean(abs(test_results['yhat_upper'] - test_results['yhat_lower']))
            
            results.ix[i, 'test_err'] = avg_test_error
            results.ix[i, 'test_range'] = avg_test_uncertainty
            
        print(results)
        
        # Plot of training and testing average errors
        self.reset_plot()
        
        plt.plot(results['cps'], results['train_err'], 'bo-', ms = 8, label = 'Train Error')
        plt.plot(results['cps'], results['test_err'], 'r*-', ms = 8, label = 'Test Error')
        plt.xlabel('Changepoint Prior Scale'); plt.ylabel('Avg. Absolute Error ($)');
        plt.title('Training and Testing Curves as Function of CPS')
        plt.grid(color='k', alpha=0.3)
        plt.xticks(results['cps'], results['cps'])
        plt.legend(prop={'size':10})
        plt.show();
        
        # Plot of training and testing average uncertainty
        self.reset_plot()

        plt.plot(results['cps'], results['train_range'], 'bo-', ms = 8, label = 'Train Range')
        plt.plot(results['cps'], results['test_range'], 'r*-', ms = 8, label = 'Test Range')
        plt.xlabel('Changepoint Prior Scale'); plt.ylabel('Avg. Uncertainty ($)');
        plt.title('Uncertainty in Estimate as Function of CPS')
        plt.grid(color='k', alpha=0.3)
        plt.xticks(results['cps'], results['cps'])
        plt.legend(prop={'size':10})
        plt.show();


tesla = Stocker('TSLA')


tesla.changepoint_date_analysis(search='Model S')





# # Purpose
# 
# In an earlier notebook, I attemped to make a stock prediction model. This failed to say the least. However, I think there is still much to be salvaged from the attempt. Python code written for a foolish task (predicting the stock market) is not written in vain, but an ideal opportunity to learn some new skills. I had a ton of fun making the stock model, and I wanted to still show the process which ultimately is more important than the end result. After writing a ton of separate functions to act on stock data, I realized a much better approach was to make a class that could hold data and also had different functions for prediction, plotting, analysis and anything else that might be needed. Having already developed all of these functions separately, I thought it made a lot of sense to combine them into a single class. This notebook will build up the class, covering many data manipulation, plotting, and even modeling concepts in a single example! 
# 

# Quandl for financial analysis, pandas and numpy for data manipulation
# fbprophet for additive models, #pytrends for Google trend data
import quandl
import pandas as pd
import numpy as np
import fbprophet
import pytrends
from pytrends.request import TrendReq

# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import matplotlib


# Class for analyzing and (attempting) to predict future prices
# Contains a number of visualizations and analysis methods
class Stocker():
    
    # Initialization requires a ticker symbol
    def __init__(self, ticker):
        
        # Enforce capitalization
        ticker = ticker.upper()
        
        # Symbol is used for labeling plots
        self.symbol = ticker
        
        # api personal key
        quandl.ApiConfig.api_key = 'rFsSehe51RLzREtYhLfo'

        # Retrieval the financial data
        try:
            stock = quandl.get('WIKI/%s' % ticker)
        
        except Exception as e:
            print('Error Retrieving Data.')
            print(e)
            return
        
        # Set the index to a column called Date
        stock = stock.reset_index(level=0)
        
        # Columns required for prophet
        stock['ds'] = stock['Date']
        stock['y'] = stock['Adj. Close']
        
        # Data assigned as class attribute
        self.stock = stock.copy()
        
        # Minimum and maximum date in range
        self.min_date = min(stock['Date'])
        self.max_date = max(stock['Date'])
        
        # Find max and min prices and dates on which they occurred
        self.max_price = np.max(self.stock['y'])
        self.min_price = np.min(self.stock['y'])
        
        self.min_price_date = self.stock[self.stock['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.stock[self.stock['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]
        
        # The starting price (starting with the opening price)
        self.starting_price = float(self.stock.ix[0, 'Adj. Open'])
        
        # The most recent price
        self.most_recent_price = float(self.stock.ix[len(self.stock) - 1, 'y'])
        
        # This can be changed by user
        self.changepoint_prior_scale = 0.2
        
        print('{} Stocker Initialized. Data covers {} to {}.'.format(self.symbol,
                                                                     self.min_date.date(),
                                                                     self.max_date.date()))
        
    # Basic Historical Plot and Basic Statistics
    def plot_stock(self, start_date=None, end_date=None):
        
        self.reset_plot()
        
        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
            
        # Convert to pandas date time
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if end_date.date() < start_date.date():
            print('End Date must be later than start date.')
            return
        
        # Check to make sure dates are in the data
        if (start_date not in list(self.stock['Date'])):
            print('Start Date not in data (either out of range or not a trading day.)')
            return
        elif (end_date not in list(self.stock['Date'])):
            print('End Date not in data (either out of range or not a trading day.)')
            return 
            
        # Useful statistics
        print('Maximum price = ${:.2f} on {}.'.format(self.max_price, self.max_price_date.date()))
        print('Minimum price = ${:.2f} on {}.'.format(self.min_price, self.min_price_date.date()))
        print('Current price = ${:.2f}.'.format(self.most_recent_price))
        
        stock_plot = self.stock[(self.stock['Date'] >= start_date.date()) & (self.stock['Date'] <= end_date.date())]
        
        # Simple Plot 
        plt.style.use('fivethirtyeight');
        plt.plot(stock_plot['Date'], stock_plot['Adj. Close'], 'r-', linewidth = 3)
        plt.xlabel('Date'); plt.ylabel('Closing Price ($)'); plt.title('%s Stock Price History' % self.symbol); 
        plt.grid(color = 'k', alpha = 0.4); plt.show()
      
    # Reset the plotting parameters to clear style formatting
    # Not sure if this should be a static method
    @staticmethod
    def reset_plot():
        
        # Restore default parameters
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        
        # Adjust a few parameters to liking
        matplotlib.rcParams['figure.figsize'] = (8, 5)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'
    
    # Method to linearly interpolate prices on the weekends
    def resample(self, dataframe):
        # Change the index and resample at daily level
        dataframe = dataframe.set_index('ds')
        dataframe = dataframe.resample('D')
        
        # Reset the index and interpolate nan values
        dataframe = dataframe.reset_index(level=0)
        dataframe = dataframe.interpolate()
        return dataframe
    
    # Remove weekends from a dataframe
    def remove_weekends(self, dataframe):
        
        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)
        
        weekends = []
        
        # Find all of the weekends
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday) == 5 | (date.weekday == 6):
                weekends.append(i)
            
        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)
        
        return dataframe
    
    
    # Calculate and plot profit from buying and holding shares for specified date range
    def buy_and_hold(self, start_date=None, end_date=None, nshares=1):
        self.reset_plot()
        
        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
            
        # Convert to pandas datetime for indexing dataframe
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if end_date.date() < start_date.date():
            print('End Date must be later than start date.')
            return
        
        # Check to make sure dates are in the data
        if (start_date not in list(self.stock['Date'])):
            print('Start Date not in data (either out of range or not a trading day.)')
            return
        elif (end_date not in list(self.stock['Date'])):
            print('End Date not in data (either out of range or not a trading day.)')
            return 
        
        # Find starting and ending price of stock
        start_price = float(self.stock[self.stock['Date'] == start_date]['Adj. Open'])
        end_price = float(self.stock[self.stock['Date'] == end_date]['Adj. Close'])
        
        # Make a profit dataframe and calculate profit column
        profits = self.stock[(self.stock['Date'] >= start_date) & (self.stock['Date'] <= end_date)]
        profits['hold_profit'] = nshares * (profits['Adj. Close'] - start_price)
        
        # Total profit
        total_hold_profit = nshares * (end_price - start_price)
        
        print('{} Total buy and hold profit from {} to {} for {} shares = ${:.2f}'.format
              (self.symbol, start_date.date(), end_date.date(), nshares, total_hold_profit))
        
        # Plot the total profits 
        plt.style.use('dark_background')
        
        # Location for number of profit
        text_location = (end_date - pd.DateOffset(months = 1)).date()
        
        # Plot the profits over time
        plt.plot(profits['Date'], profits['hold_profit'], 'b', linewidth = 3)
        plt.ylabel('Profit ($)'); plt.xlabel('Date'); plt.title('Buy and Hold Profits for {} {} to {}'.format(
                                                                self.symbol, start_date.date(), end_date.date()))
        
        # Display final value on graph
        plt.text(x = text_location, 
             y =  total_hold_profit + (total_hold_profit / 40),
             s = '$%d' % total_hold_profit,
            color = 'g' if total_hold_profit > 0 else 'r',
            size = 14)
        
        plt.grid(alpha=0.2)
        plt.show();
        
    # Create a prophet model without training
    def create_model(self, **kwargs):

        # Make the model
        model = fbprophet.Prophet(daily_seasonality=False,  weekly_seasonality=False,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                 **kwargs)
        
        # Add monthly seasonality
        model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        
        return model
    
    # Graph the effects of altering the changepoint prior scale (cps)
    def changepoint_prior_analysis(self, changepoint_priors=[0.001, 0.05, 0.1, 0.2], colors=['b', 'r', 'grey', 'gold']):
    
        # Training and plotting with 4 years of data
        train = self.stock[(self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=3)).date())]
        
        # Iterate through all the changepoints and make models
        for i, prior in enumerate(changepoint_priors):
            # Select the changepoint
            self.changepoint_prior_scale = prior
            
            # Create and train a model with the specified cps
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=180, freq='D')
            
            # Make a dataframe to hold predictions
            if i == 0:
                predictions = future.copy()
                
            future = model.predict(future)
            
            # Fill in prediction dataframe
            predictions['%.3f_yhat_upper' % prior] = future['yhat_upper']
            predictions['%.3f_yhat_lower' % prior] = future['yhat_lower']
            predictions['%.3f_yhat' % prior] = future['yhat']
         
        # Remove the weekends
        predictions = self.remove_weekends(predictions)
        
        # Plot set-up
        self.reset_plot()
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(1, 1)
        
        # Actual observations
        ax.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Observations')
        color_dict = {prior: color for prior, color in zip(changepoint_priors, colors)}

        # Plot each of the changepoint predictions
        for prior in changepoint_priors:
            # Plot the predictions themselves
            ax.plot(predictions['ds'], predictions['%.3f_yhat' % prior], linewidth = 1.2,
                     color = color_dict[prior], label = '%.3f prior scale' % prior)
            
            # Plot the uncertainty interval
            ax.fill_between(predictions['ds'].dt.to_pydatetime(), predictions['%.3f_yhat_upper' % prior],
                            predictions['%.3f_yhat_lower' % prior], facecolor = color_dict[prior],
                            alpha = 0.3, edgecolor = 'k', linewidth = 0.6)
                            
        # Plot labels
        plt.legend(prop={'size': 10})
        plt.xlabel('Date'); plt.ylabel('Stock Price ($)'); plt.title('Effect of Changepoint Prior Scale');
        plt.show()
            
    # Basic prophet model for the next 365 days   
    def create_prophet_model(self, resample=False):
        
        self.reset_plot()
        
        model = self.create_model()
        
        # Fit on the stock history for past 3 years
        stock_history = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = 3)).date()]
        
        if resample:
            stock_history = self.resample(stock_history)
        
        model.fit(stock_history)
        
        # Make and predict for next year with future dataframe
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)
        
        # Print the predicted price
        print('Predicted Price on {} = ${:.2f}'.format(
            future.ix[len(future) - 1, 'ds'].date(), future.ix[len(future) - 1, 'yhat']))
        
        # Set up the plot
        fig, ax = plt.subplots(1, 1)

        # Plot the actual values
        ax.plot(stock_history['ds'], stock_history['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        
        # Plot the predicted values
        ax.plot(future['ds'], future['yhat'], 'forestgreen',linewidth = 2.4, label = 'Predicted');

        # Plot the uncertainty interval as ribbon
        ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.3, 
                       facecolor = 'g', edgecolor = 'k', linewidth = 1.4, label = 'Uncertainty')

        # Put a vertical line at the start of predictions
        plt.vlines(x=max(self.stock['Date']).date(), ymin=min(future['yhat_lower']), 
                   ymax=max(future['yhat_upper']), colors = 'r',
                   linestyles='dashed')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10}); plt.xlabel('Date'); plt.ylabel('Price $');
        plt.grid(linewidth=0.6, alpha = 0.6)
        plt.title('%s Historical and Predicted Stock Price'  % self.symbol);
        plt.show()
        
        return model, future
      
    # Evaluate prediction model for one year
    def evaluate_prediction(self, start_date=None, end_date=None, nshares = 1000):
        
        # Default start date is one year before end of data
        # Default end date is end date of data
        if start_date is None:
            start_date = self.max_date - pd.DateOffset(years=1)
        if end_date is None:
            end_date = self.max_date
            
        # Convert to pandas datetime for indexing dataframe
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if end_date.date() < start_date.date():
            print('End Date must be later than start date.')
            return
        
        # Check to make sure dates are in the data
        if (start_date not in list(self.stock['Date'])):
            print('Start Date not in data (either out of range or not a trading day.)')
            return
        elif (end_date not in list(self.stock['Date'])):
            print('End Date not in data (either out of range or not a trading day.)')
            return 
        
        # Training data starts 3 years before start date and goes up to start date
        train = self.stock[(self.stock['Date'] < start_date.date()) & 
                           (self.stock['Date'] > (start_date - pd.DateOffset(years=3)).date())]
        
        # Testing data is specified in the range
        test = self.stock[(self.stock['Date'] >= start_date.date()) & (self.stock['Date'] <= end_date.date())]
        
        # Create and train the model
        model = self.create_model()
        model.fit(train)
        
        # Make a future dataframe and predictions
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)
        
        # Merge predictions with the known values
        test = pd.merge(test, future, on = 'ds', how = 'inner')
        
        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()
        
        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1
        
        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])
        
        # Only playing the stocks when we predict the stock will increase
        test_pred_increase = test[test['pred_diff'] > 0]
        
        test_pred_increase.reset_index(inplace=True)
        prediction_profit = []
        
        # Iterate through all the predictions and calculate profit from playing
        for i, correct in enumerate(test_pred_increase['correct']):
            
            # If we predicted up and the price goes up, we gain the difference
            if correct == 1:
                prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
            # If we predicted up and the price goes down, we lose the difference
            else:
                prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
        
        test_pred_increase['pred_profit'] = prediction_profit
        
        # Put the profit into the test dataframe
        test = pd.merge(test, test_pred_increase[['ds', 'pred_profit']], on = 'ds', how = 'left')
        test.ix[0, 'pred_profit'] = 0
    
        # Profit for either method at all dates
        test['pred_profit'] = test['pred_profit'].cumsum().ffill()
        test['hold_profit'] = nshares * (test['y'] - float(test.ix[0, 'y']))
        
        # Display information
        print('You played the stock market in {} from {} to {} with {} shares.\n'.format(
            self.symbol, start_date.date(), end_date.date(), nshares))
        
        print('Predicted price on {} = ${:.2f}.'.format(max(future['ds']).date(), future.ix[len(future) - 1, 'yhat']))
        print('Actual price on    {} = ${:.2f}.\n'.format(max(test['ds']).date(), test.ix[len(test) - 1, 'y']))
        
        # Display some friendly information about the perils of playing the stock market
        print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
        print('When the model predicted a decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))
        print('The total profit using the Prophet model = ${:.2f}.'.format(np.sum(prediction_profit)))
        print('The Buy and Hold strategy profit =         ${:.2f}.'.format(float(test.ix[len(test) - 1, 'hold_profit'])))
        print('\nThanks for playing the stock market!\n')
        
        # Reset the plot
        self.reset_plot()
        
        # Set up the plot
        fig, ax = plt.subplots(1, 1)

        # Plot the actual values
        ax.plot(train['ds'], train['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        ax.plot(test['ds'], test['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        
        # Plot the predicted values
        ax.plot(future['ds'], future['yhat'], 'navy', linewidth = 2.4, label = 'Predicted');

        # Plot the uncertainty interval as ribbon
        ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.6, 
                       facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Uncertainty')

        # Put a vertical line at the start of predictions
        plt.vlines(x=min(test['ds']).date(), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                   linestyles='dashed', label = 'Prediction Start')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 8}); plt.xlabel('Date'); plt.ylabel('Price $');
        plt.grid(linewidth=0.6, alpha = 0.6)
                   
        plt.title('%s Prophet Evaluation for Past Year' % self.symbol);
        plt.show();
        
        # Plot the predicted and actual profits over time
        self.reset_plot()
        
        # Final profit and final smart used for locating text
        final_profit = test.ix[len(test) - 1, 'pred_profit']
        final_smart = test.ix[len(test) - 1, 'hold_profit']

        # text location
        last_date = test.ix[len(test) - 1, 'ds']
        text_location = (last_date - pd.DateOffset(months = 1)).date()

        plt.style.use('dark_background')

        # Plot smart profits
        plt.plot(test['ds'], test['hold_profit'], 'b',
                 linewidth = 1.8, label = 'Buy and Hold') 

        # Plot prediction profits
        plt.plot(test['ds'], test['pred_profit'], 
                 color = 'g' if final_profit > 0 else 'r',
                 linewidth = 1.8, label = 'Prediction')

        # Display final values on graph
        plt.text(x = text_location, 
                 y =  final_profit + (final_profit / 40),
                 s = '$%d' % final_profit,
                color = 'g' if final_profit > 0 else 'r',
                size = 18)
        
        plt.text(x = text_location, 
                 y =  final_smart + (final_smart / 40),
                 s = '$%d' % final_smart,
                color = 'g' if final_smart > 0 else 'r',
                size = 18);

        # Plot formatting
        plt.ylabel('Profit  (US $)'); plt.xlabel('Date'); 
        plt.title('Predicted versus Buy and Hold Profits');
        plt.legend(loc = 2, prop={'size': 10});
        plt.grid(alpha=0.2); 
        plt.show()
        
    def retrieve_google_trends(self, term, date_range):
        
        # Set up the trend fetching object
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [term]

        try:
        
            # Create the search object
            pytrends.build_payload(kw_list, cat=0, timeframe=date_range[0], geo='', gprop='')
            
            # Retrieve the interest over time
            trends = pytrends.interest_over_time()

        except Exception as e:
            print('\nGoogle Search Trend retrieval failed.')
            print(e)
            return
        
        return trends
        
    def changepoint_date_analysis(self, term=None):
        self.reset_plot()
        
        if term is None:
            term = '%s stock' % self.symbol

        model = self.create_model()
        
        # Use past three years of data
        train = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = 3)).date()]
        model.fit(train)
        
        # Predictions of the training data (no future periods)
        future = model.make_future_dataframe(periods=0, freq='D')
        future = model.predict(future)
    
        train = pd.merge(train, future[['ds', 'yhat']], on = 'ds', how = 'inner')
        
        changepoints = model.changepoints
        train = train.reset_index(drop=True)
        
        # Create dataframe of only changepoints
        change_indices = []
        for changepoint in (changepoints):
            change_indices.append(train[train['ds'] == changepoint.date()].index[0])
        
        c_data = train.ix[change_indices, :]
        deltas = model.params['delta'][0]
        
        c_data['delta'] = deltas
        c_data['abs_delta'] = abs(c_data['delta'])
        
        # Sort the values by maximum change
        c_data = c_data.sort_values(by='abs_delta', ascending=False)
        
        print('\nChangepoints sorted by slope rate of change (2nd derivative):\n')
        print(c_data.ix[:, ['Date', 'Adj. Close', 'delta']][:5])
    
    
        # Limit to 10 largest changepoints
        c_data = c_data[:10]
        
        # Line plot showing actual values, estimated values, and changepoints
        self.reset_plot()
        
        # Set up line plot 
        plt.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Stock Price')
        plt.plot(future['ds'], future['yhat'], color = 'navy', linewidth = 2.0, label = 'Estimated')
        
        # Changepoints as vertical lines
        plt.vlines(c_data['ds'].dt.to_pydatetime(), ymin = min(train['y']), ymax = max(train['y']), 
                   linestyles='dashed', color = 'r', 
                   linewidth= 1.2, label='Changepoints')
        plt.legend(prop={'size':10});
        plt.xlabel('Date'); plt.ylabel('Price ($)'); plt.title('Stock Price with Changepoints')
        plt.show()
        
        date_range = ['%s %s' % (str(min(train['Date']).date()), str(max(train['Date']).date()))]
        
        # Get the Google Trends for specified terms and join to training dataframe
        trends = self.retrieve_google_trends(term, date_range)
        
        # Upsample the data for joining with training data
        trends = trends.resample('D')
        
        trends = trends.reset_index(level=0)
        trends = trends.rename(columns={'date': 'ds', term: 'freq'})
        
        # Interpolate the frequency
        trends['freq'] = trends['freq'].interpolate()
        
        # Merge with the training data
        train = pd.merge(train, trends, on = 'ds', how = 'inner')
        
        # Normalize values
        train['y_norm'] = train['y'] / max(train['y'])
        train['freq_norm'] = train['freq'] / max(train['freq'])
        self.reset_plot()
        
        # Plot the normalized stock price and normalize search frequency
        plt.plot(train['ds'], train['y_norm'], 'k-', label = 'Stock Price')
        plt.plot(train['ds'], train['freq_norm'], color='forestgreen', label = 'Search Frequency')
        
        # Plot the changepoints as dashed vertical lines
        plt.vlines(c_data['ds'].dt.to_pydatetime(), ymin=0, ymax=1,
                   linewidth = 1.2, label='Changepoints', linestyles='dashed', color = 'r')
        
        # Plot formatting
        plt.legend(prop={'size': 10})
        plt.xlabel('Date'); plt.ylabel('Normalized Values'); plt.title('Stock Price and Search Frequency for %s' % term)
        plt.show()
        
    # Predict the future price for a given range of days
    def predict_future(self, days=30):
        
        # Use past three years for training
        train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=3)).date()]
        
        model = self.create_model()
        
        model.fit(train)
        
        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)
        
        # Only concerned with future dates
        future = future[future['ds'] >= max(self.stock['Date']).date()]
        
        # Remove the weekends
        future = self.remove_weekends(future)
        
        # Calculate whether increase or not
        future['diff'] = future['yhat'].diff()
    
        future = future.dropna()
        
        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff'] > 0) * 1
        
        # Rename the columns for presentation
        future = future.rename(columns={'ds': 'Date', 'yhat': 'estimate', 'diff': 'change', 
                                        'yhat_upper': 'upper', 'yhat_lower': 'lower'})
        
        future_increase = future[future['direction'] == 1]
        future_decrease = future[future['direction'] == 0]
        
        # Print out the dates
        print('\nPredicted Increase: \n')
        print(future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        print('\nPredicted Decrease: \n')
        print(future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        self.reset_plot()
        
        # Set up plot
        plt.style.use('fivethirtyeight')
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 12
        
        # Plot the predictions and indicate if increase or decrease
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot the estimates
        ax.plot(future_increase['Date'], future_increase['estimate'], 'g^', ms = 12, label = 'Pred. Increase')
        ax.plot(future_decrease['Date'], future_decrease['estimate'], 'rv', ms = 12, label = 'Pred. Decrease')

        # Plot errorbars
        ax.errorbar(future['Date'].dt.to_pydatetime(), future['estimate'], 
                    yerr = future['upper'] - future['lower'], 
                    capthick=1.4, color = 'k',linewidth = 2,
                   ecolor='darkblue', capsize = 4, elinewidth = 1, label = 'Pred with Range')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10});
        plt.xticks(rotation = '45')
        plt.ylabel('Predicted Stock Price (US $)');
        plt.xlabel('Date'); plt.title('Predictions for %s' % self.symbol);
        plt.show()
        
    def changepoint_prior_validation(self, changepoint_priors = [0.001, 0.05, 0.1, 0.2]):
                               
        # Select three years of training data starting 4 years ago and going until 3 years ago
        train = self.stock[(self.stock['Date'] < (max(self.stock['Date']) - pd.DateOffset(years=1)).date()) & 
                           (self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=4)).date())]
        
        # Testing data used for answers
        test = self.stock[(self.stock['Date'] >= (max(self.stock['Date']) - pd.DateOffset(years=1)).date())]
        eval_days = (max(test['Date']).date() - min(test['Date']).date()).days
        
        results = pd.DataFrame(0, index = list(range(len(changepoint_priors))), 
                               columns = ['cps', 'train_err', 'train_range', 'test_err', 'test_range'])
        
        # Iterate through all the changepoints and make models
        for i, prior in enumerate(changepoint_priors):
            results.ix[i, 'cps'] = prior
            
            # Select the changepoint
            self.changepoint_prior_scale = prior
            
            # Create and train a model with the specified cps
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=eval_days, freq='D')
                
            future = model.predict(future)
            
            # Training results and metrics
            train_results = pd.merge(train, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_train_error = np.mean(abs(train_results['y'] - train_results['yhat']))
            avg_train_uncertainty = np.mean(abs(train_results['yhat_upper'] - train_results['yhat_lower']))
            
            results.ix[i, 'train_err'] = avg_train_error
            results.ix[i, 'train_range'] = avg_train_uncertainty
            
            # Testing results and metrics
            test_results = pd.merge(test, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_test_error = np.mean(abs(test_results['y'] - test_results['yhat']))
            avg_test_uncertainty = np.mean(abs(test_results['yhat_upper'] - test_results['yhat_lower']))
            
            results.ix[i, 'test_err'] = avg_test_error
            results.ix[i, 'test_range'] = avg_test_uncertainty
            
        print(results)
        
        # Plot of training and testing average errors
        self.reset_plot()
        
        plt.plot(results['cps'], results['train_err'], 'bo', ms = 8, label = 'Train Error')
        plt.plot(results['cps'], results['test_err'], 'r*', ms = 8, label = 'Test Error')
        plt.xlabel('Changepoint Prior Scale'); plt.ylabel('Avg. Absolute Error ($)');
        plt.title('Training and Testing Curves as Function of CPS')
        plt.grid(color='k', alpha=0.3)
        plt.xticks(results['cps'], results['cps'])
        plt.legend(prop={'size':10})
        plt.show();
        
        # Plot of training and testing average uncertainty
        self.reset_plot()

        plt.plot(results['cps'], results['train_range'], 'bo', ms = 8, label = 'Train Range')
        plt.plot(results['cps'], results['test_range'], 'r*', ms = 8, label = 'Test Range')
        plt.xlabel('Changepoint Prior Scale'); plt.ylabel('Avg. Uncertainty ($)');
        plt.title('Uncertainty in Estimate as Function of CPS')
        plt.grid(color='k', alpha=0.3)
        plt.xticks(results['cps'], results['cps'])
        plt.legend(prop={'size':10})
        plt.show();


tesla = Stocker(ticker = 'tsla')


tesla.changepoint_prior_scale=0.5


tesla.evaluate_prediction(start_date='2016-01-04', end_date='2017-01-04', nshares=100)





tesla.plot_stock()


tesla = Stocker(ticker = 'TSLA')


tesla.changepoint_prior_validation()


tesla.evaluate_prophet()


tesla = Stocker(ticker = 'TSLA')
model, future = tesla.prophet_model()


tesla.changepoint_prior_effect()


model.plot_components(future)


micro = Stocker(ticker='MSFT')


model, future = micro.prophet_model()


model.plot_components(future)


micro.plot_stock(start_date='2018-01-03', end_date='2018-01-10')


stock_msft = micro.stock


stock_msft.tail(10)


micro.hold_profit(nshares=1000)


tesla = Stocker(ticker='TSLA')


tesla.plot_stock()


tesla.changepoint_prior_effect()


tesla.changepoint_prior_scale = 0.1


tesla.prophet_model()


tesla.hold_profit()


tesla.evaluate_prophet()


tesla.changepoint_analysis()


tesla.predict_range(days=60)


cat = Stocker(ticker='WIKI/CAT')


cat.plot_stock()


cat.changepoint_prior_effect()





