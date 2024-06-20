# I started by doing some exploratory analsis on the IMDB dataset
# 

get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("movies_genres.csv", delimiter='\t')
df.info()


# We have a total of 117 352 movies and each of them is associated with 28 possible genres. The genres columns simply contain a 1 or 0 depending of wether the movie is classified into that particular genre or not, so the one-hot-enconding schema is alreay provided in this file.
# 

# Next we are going to calculate the absolute number of movies per genre. Note: each movie can be associated with more than one genre, we just want to know which genres have more movies.
# 

df_genres = df.drop(['plot', 'title'], axis=1)
counts = []
categories = list(df_genres.columns.values)
for i in categories:
    counts.append((i, df_genres[i].sum()))
df_stats = pd.DataFrame(counts, columns=['genre', '#movies'])
df_stats


df_stats.plot(x='genre', y='#movies', kind='bar', legend=False, grid=True, figsize=(15, 8))


# Since the `Lifestyle` has 0 instances we can just remove it from the data set
# 

df.drop('Lifestyle', axis=1, inplace=True)


# One thing that notice when working with this dataset is that there are plots written in different languages. Let's use [langedetect](https://pypi.python.org/pypi/langdetect?) tool to identify the language in which the plots are written
# 

from langdetect import detect
df['plot_lang'] = df.apply(lambda row: detect(row['plot'].decode("utf8")), axis=1)
df['plot_lang'].value_counts()


# There other languages besides English, let's just keep English plots, and save this to a new file.
# 

df = df[df.plot_lang.isin(['en'])]
df.to_csv("movies_genres_en.csv", sep='\t', encoding='utf-8')


