get_ipython().magic('matplotlib inline')
import pandas as pd
import datetime
import ast
import tldextract


# If you want access to this data ping @bstarling on slack. I have it stored in s3
# 150mb / 240k rows
df = pd.read_csv('breitbart_clean.csv', sep='\t', parse_dates=['date'])


df.set_index('date', inplace=True)
df.count()


# ### Articles by year (2 months of 2012 missing)
# 

by_year=df.groupby([pd.TimeGrouper('A')]).count()['title']
by_year


by_year.plot()


# ## Category publications by year
# 

df.groupby([pd.TimeGrouper('A'),'category']).count()['title']


# ### Top 25 authors
# 

df.groupby(['author']).count()['title'].sort_values(ascending=0).head(25)


# ### Hacky attempt to explore most common top level domains linked in articles
# 

from collections import Counter
tld_counter = Counter()


def get_tld(hrefs):
    
    # Quick and dirty, not thorough yet
    for link in ast.literal_eval(hrefs):
        top_level = tldextract.extract(link)
        top_level = top_level.domain
        tld_counter[top_level] += 1


_ = df[['hrefs']].applymap(get_tld)


tld_counter.most_common(25)





