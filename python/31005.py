# # Bay Area Bike Share Analysis
# 

# ### Reading data
# 
# Before making the hypothesis, it is always a good practice to look at the snapshot of data in order to understand the various attributes
# 

from delorean import parse
import numpy as np
import pandas as pd


df = pd.read_csv('data/trip_data.csv', parse_dates=True)


df.shape


df.head(20)


df.dtypes


df["Subscription Type"].value_counts()


# ### Hypothesis
# 
# Based on the data, what are the questions we can answer?
# 
# 1. Given a time slot and terminal code, predict if the terminal has high traffic
# 2. Given a time slot and terminal code, perdict number of transactions

# ## Hypothesis 1
# 
# ### Given a time slot and terminal code, predict if the terminal has high traffic
# 

# ### Feature engineering
# 
# * Find & replace null values
# * Hour window
# * Target generation
# * Removal of non categorical fields
# 

pd.isnull(df["Start Date"]).all()


def get_hour(value):
    
    parsed_date = parse(value).datetime.hour
    return parsed_date
    


df["start_hour"] = df["Start Date"].apply(get_hour)


df.start_hour.value_counts()


a = parse('7/25/2016 14:13')


a.datetime.isoweekday()


def get_day(value):
    
    parsed_date = parse(value).datetime.isoweekday()
    return parsed_date


df["start_day"] = df["Start Date"].apply(get_day)


df.head()


df["end_hour"] = df["End Date"].apply(get_hour)


df["end_day"] = df["End Date"].apply(get_day)


df.head()


start_df = df.groupby(by=["start_hour", "start_day", "Start Terminal"]).count().copy()
start_df = start_df.reset_index()


start_df = start_df.ix[:, ["start_hour", "start_day", "Start Terminal", "Trip ID"]]
start_df.columns = ["hour", "day", "terminal_code", "trip_id"]
start_df.head()


end_df = df.groupby(by=["end_hour", "end_day", "End Terminal"]).count().copy()
end_df = end_df.reset_index()
end_df = end_df.ix[:, ["end_hour", "end_day", "End Terminal", "Trip ID"]]
end_df.columns = ["hour", "day", "terminal_code", "trip_id"]
end_df.head()


merged_df = start_df.merge(end_df, how="inner", on=["hour", "day", "terminal_code"])


merged_df.head()


merged_df["trip_count"] = merged_df["trip_id_x"] + merged_df["trip_id_y"]


merged_df = merged_df.ix[:, ["hour", "day", "terminal_code", "trip_count"]]


merged_df.head()


merged_df.trip_count.mean()


merged_df["target"] = 0
merged_df.ix[(merged_df.trip_count > merged_df.trip_count.mean()), "target"] = 1


merged_df.target.value_counts()


merged_df.head()





