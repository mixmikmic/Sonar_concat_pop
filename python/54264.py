import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
import os
import csv
from tqdm import tqdm
import random

DATA_DIR = "../data/"
SELECTED_DATA_DIR = "../selected-data/"
CSV_DIR = "csv/"
MOVIES_FILE = "imdb_database.json"
MOVIES_FILE_OUT = "best_movie_ratings_features.csv"
USERS_FILE_OUT = "users_ratings.csv"


# # Subset of movies
# 

def flatten_aka(d):
    try:
        return [x['name'] for x in d]
    except:
        return []

movies = pd.read_json(DATA_DIR + MOVIES_FILE)
movies = movies[movies.kind == "movie"]
movies["title"] = movies.id
movies["votes"] = movies.rating.apply(lambda x: x['votes'])
movies["rating"] = movies.rating.apply(lambda x: x['rank'])
movies["aka"] = movies.aka.apply(flatten_aka)
movies = movies.drop_duplicates(subset="title")
movies = movies[["aka", "genres", "year", "votes", "rating", "title"]].set_index("title").dropna()
movies.shape


movies = movies.sort_values(by="votes", ascending=False)
movies = movies.loc[movies.index[:1000]]
movies.sample()


movies.to_csv(SELECTED_DATA_DIR + MOVIES_FILE_OUT)


# # Users with at least N recommandations for these movies
# 

N = 50
users = pd.DataFrame()
user_files = os.listdir(DATA_DIR + CSV_DIR)
random.shuffle(user_files)
user_files = user_files[:10000]
for (i, filename) in tqdm(enumerate(user_files)):
    user = pd.read_csv(DATA_DIR + CSV_DIR + filename, header=-1)
    user = user.set_index(1)
    count = user.join(movies, how='inner', lsuffix="user_", rsuffix="movie_").size
    if count > 50:
        # get only recommandations that are in movies
        users = users.append(user[user.index.isin(movies.index)])

users.columns = ["user", "rating", "link"]
users.index.rename("movie", inplace=True)
users.sample()


#for id, recomm in tqdm(users.iterrows()):
#    assert(id in movies.index)


users.to_csv(SELECTED_DATA_DIR + USERS_FILE_OUT)





