# # Introduction to the data
# 

import sqlite3
conn = sqlite3.connect("nominations.db")
schema = conn.execute("pragma table_info(nominations);").fetchall()
first_ten = conn.execute("select * from nominations limit 10;").fetchall()

for r in schema:
    print(r)
    
for r in first_ten:
    print(r)


# # Creating the ceremonies table
# 

years_hosts = [(2010, "Steve Martin"),
               (2009, "Hugh Jackman"),
               (2008, "Jon Stewart"),
               (2007, "Ellen DeGeneres"),
               (2006, "Jon Stewart"),
               (2005, "Chris Rock"),
               (2004, "Billy Crystal"),
               (2003, "Steve Martin"),
               (2002, "Whoopi Goldberg"),
               (2001, "Steve Martin"),
               (2000, "Billy Crystal"),
            ]
create_ceremonies = "create table ceremonies (id integer primary key, year integer, host text);"
conn.execute(create_ceremonies)
insert_query = "insert into ceremonies (Year, Host) values (?,?);"
conn.executemany(insert_query, years_hosts)

print(conn.execute("select * from ceremonies limit 10;").fetchall())
print(conn.execute("pragma table_info(ceremonies);").fetchall())


# # Foreign key constraints
# 

conn.execute("PRAGMA foreign_keys = ON;")


# # Setting up one-to-many
# 

create_nominations_two = '''create table nominations_two 
(id integer primary key, 
category text, 
nominee text, 
movie text, 
character text, 
won text,
ceremony_id integer,
foreign key(ceremony_id) references ceremonies(id));
'''

nom_query = '''
select ceremonies.id as ceremony_id, nominations.category as category, 
nominations.nominee as nominee, nominations.movie as movie, 
nominations.character as character, nominations.won as won
from nominations
inner join ceremonies 
on nominations.year == ceremonies.year
;
'''
joined_nominations = conn.execute(nom_query).fetchall()

conn.execute(create_nominations_two)

insert_nominations_two = '''insert into nominations_two (ceremony_id, category, nominee, movie, character, won) 
values (?,?,?,?,?,?);
'''

conn.executemany(insert_nominations_two, joined_nominations)
print(conn.execute("select * from nominations_two limit 5;").fetchall())


# # Deleting and renaming tables
# 

drop_nominations = "drop table nominations;"
conn.execute(drop_nominations)

rename_nominations_two = "alter table nominations_two rename to nominations;"
conn.execute(rename_nominations_two)


# # Creating a join table
# 

create_movies = "create table movies (id integer primary key,movie text);"
create_actors = "create table actors (id integer primary key,actor text);"
create_movies_actors = '''create table movies_actors (id INTEGER PRIMARY KEY,
movie_id INTEGER references movies(id), actor_id INTEGER references actors(id));
'''
conn.execute(create_movies)
conn.execute(create_actors)
conn.execute(create_movies_actors)


# # Populating the movies and actors tables
# 

insert_movies = "insert into movies (movie) select distinct movie from nominations;"
insert_actors = "insert into actors (actor) select distinct nominee from nominations;"
conn.execute(insert_movies)
conn.execute(insert_actors)

print(conn.execute("select * from movies limit 5;").fetchall())
print(conn.execute("select * from actors limit 5;").fetchall())


# # Populating a join table
# 

pairs_query = "select movie,nominee from nominations;"
movie_actor_pairs = conn.execute(pairs_query).fetchall()

join_table_insert = "insert into movies_actors (movie_id, actor_id) values ((select id from movies where movie == ?),(select id from actors where actor == ?));"
conn.executemany(join_table_insert,movie_actor_pairs)

print(conn.execute("select * from movies_actors limit 5;").fetchall())





import pandas as pd
police_killings = pd.read_csv("police_killings.csv", encoding="ISO-8859-1")
police_killings.head(5)


police_killings.columns


counts = police_killings["raceethnicity"].value_counts()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.bar(range(6), counts)
plt.xticks(range(6), counts.index, rotation="vertical")


counts / sum(counts)


# ## Racial breakdown
# 
# It looks like people identified as `Black` are far overrepresented in the shootings versus in the population of the US (`28%` vs `16%`).  You can see the breakdown of population by race [here](https://en.wikipedia.org/wiki/Race_and_ethnicity_in_the_United_States#Racial_makeup_of_the_U.S._population).  
# 
# People identified as `Hispanic` appear to be killed about as often as random chance would account for (`14%` of the people killed as Hispanic, versus `17%` of the overall population).
# 
# Whites are underrepresented among shooting victims vs their population percentage, as are Asians.
# 

police_killings["p_income"][police_killings["p_income"] != "-"].astype(float).hist(bins=20)


police_killings["p_income"][police_killings["p_income"] != "-"].astype(float).median()


# # Income breakdown
# 
# According to the [Census](https://en.wikipedia.org/wiki/Personal_income_in_the_United_States), median personal income in the US is `28,567`, and our median is `22,348`, which means that shootings tend to happen in less affluent areas.  Our sample size is relatively small, though, so it's hard to make sweeping conclusions.
# 

state_pop = pd.read_csv("state_population.csv")


counts = police_killings["state_fp"].value_counts()

states = pd.DataFrame({"STATE": counts.index, "shootings": counts})


states = states.merge(state_pop, on="STATE")


states["pop_millions"] = states["POPESTIMATE2015"] / 1000000
states["rate"] = states["shootings"] / states["pop_millions"]

states.sort("rate")


# ## Killings by state
# 
# States in the midwest and south seem to have the highest police killing rates, whereas those in the northeast seem to have the lowest.
# 

police_killings["state"].value_counts()


pk = police_killings[
    (police_killings["share_white"] != "-") & 
    (police_killings["share_black"] != "-") & 
    (police_killings["share_hispanic"] != "-")
]

pk["share_white"] = pk["share_white"].astype(float)
pk["share_black"] = pk["share_black"].astype(float)
pk["share_hispanic"] = pk["share_hispanic"].astype(float)


lowest_states = ["CT", "PA", "IA", "NY", "MA", "NH", "ME", "IL", "OH", "WI"]
highest_states = ["OK", "AZ", "NE", "HI", "AK", "ID", "NM", "LA", "CO", "DE"]

ls = pk[pk["state"].isin(lowest_states)]
hs = pk[pk["state"].isin(highest_states)]


columns = ["pop", "county_income", "share_white", "share_black", "share_hispanic"]

ls[columns].mean()


hs[columns].mean()


# ## State by state rates
# 
# It looks like the states with low rates of shootings tend to have a higher proportion of blacks in the population, and a lower proportion of hispanics in the census regions where the shootings occur.  It looks like the income of the counties where the shootings occur is higher.
# 
# States with high rates of shootings tend to have high hispanic population shares in the counties where shootings occur.
# 

import pandas as pd
star_wars = pd.read_csv("star_wars.csv", encoding="ISO-8859-1")


star_wars = star_wars[pd.notnull(star_wars["RespondentID"])]


star_wars.head()


star_wars.columns


yes_no = {"Yes": True, "No": False}

for col in [
    "Have you seen any of the 6 films in the Star Wars franchise?",
    "Do you consider yourself to be a fan of the Star Wars film franchise?"
    ]:
    star_wars[col] = star_wars[col].map(yes_no)

star_wars.head()


import numpy as np

movie_mapping = {
    "Star Wars: Episode I  The Phantom Menace": True,
    np.nan: False,
    "Star Wars: Episode II  Attack of the Clones": True,
    "Star Wars: Episode III  Revenge of the Sith": True,
    "Star Wars: Episode IV  A New Hope": True,
    "Star Wars: Episode V The Empire Strikes Back": True,
    "Star Wars: Episode VI Return of the Jedi": True
}

for col in star_wars.columns[3:9]:
    star_wars[col] = star_wars[col].map(movie_mapping)


star_wars = star_wars.rename(columns={
        "Which of the following Star Wars films have you seen? Please select all that apply.": "seen_1",
        "Unnamed: 4": "seen_2",
        "Unnamed: 5": "seen_3",
        "Unnamed: 6": "seen_4",
        "Unnamed: 7": "seen_5",
        "Unnamed: 8": "seen_6"
        })

star_wars.head()


star_wars = star_wars.rename(columns={
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "ranking_1",
        "Unnamed: 10": "ranking_2",
        "Unnamed: 11": "ranking_3",
        "Unnamed: 12": "ranking_4",
        "Unnamed: 13": "ranking_5",
        "Unnamed: 14": "ranking_6"
        })

star_wars.head()


star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)


star_wars[star_wars.columns[9:15]].mean()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.bar(range(6), star_wars[star_wars.columns[9:15]].mean())


# ## Rankings
# 
# So far, we've cleaned up the data, renamed several columns, and computed the average ranking of each movie.  As I suspected, it looks like the "original" movies are rated much more highly than the newer ones.
# 

star_wars[star_wars.columns[3:9]].sum()


plt.bar(range(6), star_wars[star_wars.columns[3:9]].sum())


# # View counts
# 
# It appears that the original movies were seen by more respondents than the newer movies.  This reinforces what we saw in the rankings, where the earlier movies seem to be more popular.
# 

males = star_wars[star_wars["Gender"] == "Male"]
females = star_wars[star_wars["Gender"] == "Female"]


plt.bar(range(6), males[males.columns[9:15]].mean())
plt.show()

plt.bar(range(6), females[females.columns[9:15]].mean())
plt.show()


plt.bar(range(6), males[males.columns[3:9]].sum())
plt.show()

plt.bar(range(6), females[females.columns[3:9]].sum())
plt.show()


# ## Male/Female differences in favorite Star Wars movie and most seen movie
# 
# Interestingly, more males watches episodes 1-3, but males liked them far less than females did.
# 

# ## Birth Dates In The United States
# 
# The raw data behind the story **Some People Are Too Superstitious To Have A Baby On Friday The 13th**, which you can read [here](http://fivethirtyeight.com/features/some-people-are-too-superstitious-to-have-a-baby-on-friday-the-13th/).
# 
# We'll be working with the data set from the Centers for Disease Control and Prevention's National National Center for Health Statistics.  The data set has the following structure:
# 
# - `year` - Year
# - `month` - Month
# - `date_of_month` - Day number of the month
# - `day_of_week` - Day of week, where 1 is Monday and 7 is Sunday
# - `births` - Number of births
# 

f = open("births.csv", 'r')
text = f.read()
print(text)


lines_list = text.split("\n")
lines_list


data_no_header = lines_list[1:len(lines_list)]
days_counts = dict()

for line in data_no_header:
    split_line = line.split(",")
    day_of_week = split_line[3]
    num_births = int(split_line[4])

    if day_of_week in days_counts:
        days_counts[day_of_week] = days_counts[day_of_week] + num_births
    else:
        days_counts[day_of_week] = num_births

days_counts


# # Introduction To The Dataset
# 

csv_list = open("US_births_1994-2003_CDC_NCHS.csv").read().split("\n")


csv_list[0:10]


# # Converting Data Into A List Of Lists
# 

def read_csv(filename):
    string_data = open(filename).read()
    string_list = string_data.split("\n")[1:]
    final_list = []
    
    for row in string_list:
        string_fields = row.split(",")
        int_fields = []
        for value in string_fields:
            int_fields.append(int(value))
        final_list.append(int_fields)
    return final_list
        
cdc_list = read_csv("US_births_1994-2003_CDC_NCHS.csv")


cdc_list[0:10]


# # Calculating Number Of Births Each Month
# 

def read_csv(filename):
    string_data = open(filename).read()
    string_list = string_data.split("\n")[1:]
    final_list = []
    
    for row in string_list:
        string_fields = row.split(",")
        int_fields = []
        for value in string_fields:
            int_fields.append(int(value))
        final_list.append(int_fields)
    return final_list
        
cdc_list = read_csv("US_births_1994-2003_CDC_NCHS.csv")


def month_births(data):
    births_per_month = {}
    
    for row in data:
        month = row[1]
        births = row[4]
        if month in births_per_month:
            births_per_month[month] = births_per_month[month] + births
        else:
            births_per_month[month] = births
    return births_per_month
    
cdc_month_births = month_births(cdc_list)


cdc_month_births


# # Calculating Number Of Births Each Day Of Week
# 

def dow_births(data):
    births_per_dow = {}
    
    for row in data:
        dow = row[3]
        births = row[4]
        if dow in births_per_dow:
            births_per_dow[dow] = births_per_dow[dow] + births
        else:
            births_per_dow[dow] = births
    return births_per_dow
    
cdc_dow_births = dow_births(cdc_list)


cdc_dow_births


# # Creating A More General Function
# 

def calc_counts(data, column):
    sums_dict = {}
    
    for row in data:
        col_value = row[column]
        births = row[4]
        if col_value in sums_dict:
            sums_dict[col_value] = sums_dict[col_value] + births
        else:
            sums_dict[col_value] = births
    return sums_dict

cdc_year_births = calc_counts(cdc_list, 0)
cdc_month_births = calc_counts(cdc_list, 1)
cdc_dom_births = calc_counts(cdc_list, 2)
cdc_dow_births = calc_counts(cdc_list, 3)


cdc_year_births


cdc_month_births


cdc_dom_births


cdc_dow_births


import pandas

bike_rentals = pandas.read_csv("bike_rental_hour.csv")
bike_rentals.head()


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

plt.hist(bike_rentals["cnt"])


bike_rentals.corr()["cnt"]


def assign_label(hour):
    if hour >=0 and hour < 6:
        return 4
    elif hour >=6 and hour < 12:
        return 1
    elif hour >= 12 and hour < 18:
        return 2
    elif hour >= 18 and hour <=24:
        return 3

bike_rentals["time_label"] = bike_rentals["hr"].apply(assign_label)


# ## Error metric
# 
# The mean squared error metric makes the most sense to evaluate our error.  MSE works on continuous numeric data, which fits our data quite well.
# 

train = bike_rentals.sample(frac=.8)


test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]


from sklearn.linear_model import LinearRegression

predictors = list(train.columns)
predictors.remove("cnt")
predictors.remove("casual")
predictors.remove("registered")
predictors.remove("dteday")

reg = LinearRegression()

reg.fit(train[predictors], train["cnt"])


import numpy
predictions = reg.predict(test[predictors])

numpy.mean((predictions - test["cnt"]) ** 2)


actual


test["cnt"]


# ## Error
# 
# The error is very high, which may be due to the fact that the data has a few extremely high rental counts, but otherwise mostly low counts.  Larger errors are penalized more with MSE, which leads to a higher total error.
# 

from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(min_samples_leaf=5)

reg.fit(train[predictors], train["cnt"])


predictions = reg.predict(test[predictors])

numpy.mean((predictions - test["cnt"]) ** 2)


reg = DecisionTreeRegressor(min_samples_leaf=2)

reg.fit(train[predictors], train["cnt"])

predictions = reg.predict(test[predictors])

numpy.mean((predictions - test["cnt"]) ** 2)


# ## Decision tree error
# 
# By taking the nonlinear predictors into account, the decision tree regressor appears to have much higher accuracy than linear regression.
# 

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(min_samples_leaf=5)
reg.fit(train[predictors], train["cnt"])


predictions = reg.predict(test[predictors])

numpy.mean((predictions - test["cnt"]) ** 2)


# ## Random forest error
# 
# By removing some of the sources of overfitting, the random forest accuracy is improved over the decision tree accuracy.
# 

import pandas

board_games = pandas.read_csv("board_games.csv")
board_games = board_games.dropna(axis=0)
board_games = board_games[board_games["users_rated"] > 0]

board_games.head()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.hist(board_games["average_rating"])


print(board_games["average_rating"].std())
print(board_games["average_rating"].mean())


# ## Error metric
# 
# In this data set, using mean squared error as an error metric makes sense.  This is because the data is continuous, and follows a somewhat normal distribution.  We'll be able to compare our error to the standard deviation to see how good the model is at predictions.
# 

from sklearn.cluster import KMeans

clus = KMeans(n_clusters=5)
cols = list(board_games.columns)
cols.remove("name")
cols.remove("id")
cols.remove("type")
numeric = board_games[cols]

clus.fit(numeric)


import numpy
game_mean = numeric.apply(numpy.mean, axis=1)
game_std = numeric.apply(numpy.std, axis=1)


labels = clus.labels_

plt.scatter(x=game_mean, y=game_std, c=labels)


# ## Game clusters
# 
# It looks like most of the games are similar, but as the game attributes tend to increase in value (such as number of users who rated), there are fewer high quality games.  So most games don't get played much, but a few get a lot of players.
# 

correlations = numeric.corr()

correlations["average_rating"]


# ## Correlations
# 
# The `yearpublished` column is surprisingly highly correlated with `average_rating`, showing that more recent games tend to be rated more highly.  Games for older players (`minage` is high) tend to be more highly rated.  The more "weighty" a game is (`average_weight` is high), the more highly it tends to be rated.
# 

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
cols.remove("average_rating")
cols.remove("bayes_average_rating")
reg.fit(board_games[cols], board_games["average_rating"])
predictions = reg.predict(board_games[cols])

numpy.mean((predictions - board_games["average_rating"]) ** 2)


# ## Game clusters
# 
# The error rate is close to the standard deviation of all board game ratings.  This indicates that our model may not have high predictive power.  We'll need to dig more into which games were scored well, and which ones weren't.
# 




import pandas
import csv

jeopardy = pandas.read_csv("jeopardy.csv")

jeopardy


jeopardy.columns


jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question', 'Answer']


import re

def normalize_text(text):
    text = text.lower()
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    return text

def normalize_values(text):
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    try:
        text = int(text)
    except Exception:
        text = 0
    return text


jeopardy["clean_question"] = jeopardy["Question"].apply(normalize_text)
jeopardy["clean_answer"] = jeopardy["Answer"].apply(normalize_text)
jeopardy["clean_value"] = jeopardy["Value"].apply(normalize_values)


jeopardy


jeopardy["Air Date"] = pandas.to_datetime(jeopardy["Air Date"])


jeopardy.dtypes


def count_matches(row):
    split_answer = row["clean_answer"].split(" ")
    split_question = row["clean_question"].split(" ")
    if "the" in split_answer:
        split_answer.remove("the")
    if len(split_answer) == 0:
        return 0
    match_count = 0
    for item in split_answer:
        if item in split_question:
            match_count += 1
    return match_count / len(split_answer)

jeopardy["answer_in_question"] = jeopardy.apply(count_matches, axis=1)


jeopardy["answer_in_question"].mean()


# ## Answer terms in the question
# 
# The answer only appears in the question about `6%` of the time.  This isn't a huge number, and means that we probably can't just hope that hearing a question will enable us to figure out the answer.  We'll probably have to study.
# 

question_overlap = []
terms_used = set()
for i, row in jeopardy.iterrows():
        split_question = row["clean_question"].split(" ")
        split_question = [q for q in split_question if len(q) > 5]
        match_count = 0
        for word in split_question:
            if word in terms_used:
                match_count += 1
        for word in split_question:
            terms_used.add(word)
        if len(split_question) > 0:
            match_count /= len(split_question)
        question_overlap.append(match_count)
jeopardy["question_overlap"] = question_overlap

jeopardy["question_overlap"].mean()


# ## Question overlap
# 
# There is about `70%` overlap between terms in new questions and terms in old questions.  This only looks at a small set of questions, and it doesn't look at phrases, it looks at single terms.  This makes it relatively insignificant, but it does mean that it's worth looking more into the recycling of questions.
# 

def determine_value(row):
    value = 0
    if row["clean_value"] > 800:
        value = 1
    return value

jeopardy["high_value"] = jeopardy.apply(determine_value, axis=1)


def count_usage(term):
    low_count = 0
    high_count = 0
    for i, row in jeopardy.iterrows():
        if term in row["clean_question"].split(" "):
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1
    return high_count, low_count

comparison_terms = list(terms_used)[:5]
observed_expected = []
for term in comparison_terms:
    observed_expected.append(count_usage(term))

observed_expected


from scipy.stats import chisquare
import numpy as np

high_value_count = jeopardy[jeopardy["high_value"] == 1].shape[0]
low_value_count = jeopardy[jeopardy["high_value"] == 0].shape[0]

chi_squared = []
for obs in observed_expected:
    total = sum(obs)
    total_prop = total / jeopardy.shape[0]
    high_value_exp = total_prop * high_value_count
    low_value_exp = total_prop * low_value_count
    
    observed = np.array([obs[0], obs[1]])
    expected = np.array([high_value_exp, low_value_exp])
    chi_squared.append(chisquare(observed, expected))

chi_squared


# ## Chi-squared results
# 
# None of the terms had a significant difference in usage between high value and low value rows.  Additionally, the frequencies were all lower than `5`, so the chi-squared test isn't as valid.  It would be better to run this test with only terms that have higher frequencies.
# 

# # Introduction
# 

import pandas as pd
pd.options.display.max_columns = 99
first_five = pd.read_csv('loans_2007.csv', nrows=5)
first_five


thousand_chunk = pd.read_csv('loans_2007.csv', nrows=1000)
thousand_chunk.memory_usage(deep=True).sum()/(1024*1024)


# ### Let's try tripling to 3000 rows and calculate the memory footprint for each chunk.
# 

chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
for chunk in chunk_iter:
    print(chunk.memory_usage(deep=True).sum()/(1024*1024))


# ## How many rows in the data set?
# 

chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
total_rows = 0
for chunk in chunk_iter:
    total_rows += len(chunk)
print(total_rows)


# # Exploring the Data in Chunks
# 
# ## How many columns have a numeric type? How many have a string type?
# 

# Numeric columns
chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
for chunk in chunk_iter:
    print(chunk.dtypes.value_counts())


# Are string columns consistent across chunks?
obj_cols = []
chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)

for chunk in chunk_iter:
    chunk_obj_cols = chunk.select_dtypes(include=['object']).columns.tolist()
    if len(obj_cols) > 0:
        is_same = obj_cols == chunk_obj_cols
        if not is_same:
            print("overall obj cols:", obj_cols, "\n")
            print("chunk obj cols:", chunk_obj_cols, "\n")    
    else:
        obj_cols = chunk_obj_cols


# ### Observation 1: By default -- 31 numeric columns and 21 string columns.
# 
# ### Observation 2: It seems like one column in particular (the `id` column) is being cast to int64 in the last 2 chunks but not in the earlier chunks. Since the `id` column won't be useful for analysis, visualization, or predictive modelling let's ignore this column.
# 
# ## How many unique values are there in each string column? How many of the string columns contain values that are less than 50% unique?
# 

## Create dictionary (key: column, value: list of Series objects representing each chunk's value counts)
chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000)
str_cols_vc = {}
for chunk in chunk_iter:
    str_cols = chunk.select_dtypes(include=['object'])
    for col in str_cols.columns:
        current_col_vc = str_cols[col].value_counts()
        if col in str_cols_vc:
            str_cols_vc[col].append(current_col_vc)
        else:
            str_cols_vc[col] = [current_col_vc]


## Combine the value counts.
combined_vcs = {}

for col in str_cols_vc:
    combined_vc = pd.concat(str_cols_vc[col])
    final_vc = combined_vc.groupby(combined_vc.index).sum()
    combined_vcs[col] = final_vc


combined_vcs.keys()


# ## Optimizing String Columns
# 
# ### Determine which string columns you can convert to a numeric type if you clean them. Let's focus on columns that would actually be useful for analysis and modelling.
# 

obj_cols


useful_obj_cols = ['term', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'issue_d', 'purpose', 'earliest_cr_line', 'revol_util', 'last_pymnt_d', 'last_credit_pull_d']


for col in useful_obj_cols:
    print(col)
    print(combined_vcs[col])
    print("-----------")


# ### Convert to category
# 

convert_col_dtypes = {
    "sub_grade": "category", "home_ownership": "category", 
    "verification_status": "category", "purpose": "category"
}


# ### Convert `term` and `revol_util` to numerical by data cleaning.
# ### Convert `issue_d`, `earliest_cr_line`, `last_pymnt_d`, and `last_credit_pull_d` to datetime.
# 

chunk[useful_obj_cols]


chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000, dtype=convert_col_dtypes, parse_dates=["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"])

for chunk in chunk_iter:
    term_cleaned = chunk['term'].str.lstrip(" ").str.rstrip(" months")
    revol_cleaned = chunk['revol_util'].str.rstrip("%")
    chunk['term'] = pd.to_numeric(term_cleaned)
    chunk['revol_util'] = pd.to_numeric(revol_cleaned)
    
chunk.dtypes


chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000, dtype=convert_col_dtypes, parse_dates=["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"])
mv_counts = {}
for chunk in chunk_iter:
    term_cleaned = chunk['term'].str.lstrip(" ").str.rstrip(" months")
    revol_cleaned = chunk['revol_util'].str.rstrip("%")
    chunk['term'] = pd.to_numeric(term_cleaned)
    chunk['revol_util'] = pd.to_numeric(revol_cleaned)
    float_cols = chunk.select_dtypes(include=['float'])
    for col in float_cols.columns:
        missing_values = len(chunk) - chunk[col].count()
        if col in mv_counts:
            mv_counts[col] = mv_counts[col] + missing_values
        else:
            mv_counts[col] = missing_values
mv_counts


chunk_iter = pd.read_csv('loans_2007.csv', chunksize=3000, dtype=convert_col_dtypes, parse_dates=["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"])
mv_counts = {}
for chunk in chunk_iter:
    term_cleaned = chunk['term'].str.lstrip(" ").str.rstrip(" months")
    revol_cleaned = chunk['revol_util'].str.rstrip("%")
    chunk['term'] = pd.to_numeric(term_cleaned)
    chunk['revol_util'] = pd.to_numeric(revol_cleaned)
    chunk = chunk.dropna(how='all')
    float_cols = chunk.select_dtypes(include=['float'])
    for col in float_cols.columns:
        missing_values = len(chunk) - chunk[col].count()
        if col in mv_counts:
            mv_counts[col] = mv_counts[col] + missing_values
        else:
            mv_counts[col] = missing_values
mv_counts


import pandas

movies = pandas.read_csv("fandango_score_comparison.csv")


movies


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(movies["Fandango_Stars"])


plt.hist(movies["Metacritic_norm_round"])


# ## Fandango vs Metacritic Scores
# 
# There are no scores below a `3.0` in the Fandango reviews.  The Fandango reviews also tend to center around `4.5` and `4.0`, whereas the Metacritic reviews seem to center around `3.0` and `3.5`.
# 

import numpy

f_mean = movies["Fandango_Stars"].mean()
m_mean = movies["Metacritic_norm_round"].mean()
f_std = movies["Fandango_Stars"].std()
m_std = movies["Metacritic_norm_round"].std()
f_median = movies["Fandango_Stars"].median()
m_median = movies["Metacritic_norm_round"].median()

print(f_mean)
print(m_mean)
print(f_std)
print(m_std)
print(f_median)
print(m_median)


# ## Fandango vs Metacritic Methodology
# 
# Fandango appears to inflate ratings and isn't transparent about how it calculates and aggregates ratings.  Metacritic publishes each individual critic rating, and is transparent about how they aggregate them to get a final rating.
# 

# ## Fandango vs Metacritic number differences
# 
# The median metacritic score appears higher than the mean metacritic score because a few very low reviews "drag down" the median.  The median fandango score is lower than the mean fandango score because a few very high ratings "drag up" the mean.
# 
# Fandango ratings appear clustered between `3` and `5`, and have a much narrower random than Metacritic reviews, which go from `0` to `5`.
# 
# Fandango ratings in general appear to be higher than metacritic ratings.
# 
# These may be due to movie studio influence on Fandango ratings, and the fact that Fandango calculates its ratings in a hidden way.
# 

plt.scatter(movies["Metacritic_norm_round"], movies["Fandango_Stars"])


movies["fm_diff"] = numpy.abs(movies["Metacritic_norm_round"] - movies["Fandango_Stars"])


movies.sort_values(by="fm_diff", ascending=False).head(5)


from scipy.stats import pearsonr

r_value, p_value = pearsonr(movies["Fandango_Stars"], movies["Metacritic_norm_round"])

r_value


# ## Fandango and Metacritic correlation
# 
# The low correlation between Fandango and Metacritic scores indicates that Fandango scores aren't just inflated, they are fundamentally different.  For whatever reason, it appears like Fandango both inflates scores overall, and inflates scores differently depending on the movie.
# 

from scipy.stats import linregress

slope, intercept, r_value, p_value, stderr_slope = linregress(movies["Metacritic_norm_round"], movies["Fandango_Stars"])


pred = 3 * slope + intercept

pred


# ## Finding Residuals
# 

pred_1 = 1 * slope + intercept
pred_5 = 5 * slope + intercept
plt.scatter(movies["Metacritic_norm_round"], movies["Fandango_Stars"])
plt.plot([1,5],[pred_1,pred_5])
plt.xlim(1,5)
plt.show()


# # Introducing Thanksgiving Dinner Data
# 

import pandas as pd

data = pd.read_csv("thanksgiving.csv", encoding="Latin-1")
data.head()


data.columns


# # Filtering Out Rows From A DataFrame
# 

data["Do you celebrate Thanksgiving?"].value_counts()


data = data[data["Do you celebrate Thanksgiving?"] == "Yes"]


# # Using value_counts To Explore Main Dishes
# 

data["What is typically the main dish at your Thanksgiving dinner?"].value_counts()


data[data["What is typically the main dish at your Thanksgiving dinner?"] == "Tofurkey"]["Do you typically have gravy?"]


# # Figuring Out What Pies People Eat
# 

data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple"].value_counts()


ate_pies = (pd.isnull(data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple"])
&
pd.isnull(data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pecan"])
 &
 pd.isnull(data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pumpkin"])
)

ate_pies.value_counts()


# # Converting Age To Numeric
# 

data["Age"].value_counts()


def extract_age(age_str):
    if pd.isnull(age_str):
        return None
    age_str = age_str.split(" ")[0]
    age_str = age_str.replace("+", "")
    return int(age_str)

data["int_age"] = data["Age"].apply(extract_age)
data["int_age"].describe()


# # Findings
# 
# Although we only have a rough approximation of age, and it skews downward because we took the first value in each string (the lower bound), we can see that that age groups of respondents are fairly evenly distributed.
# 

# # Converting Income To Numeric
# 

data["How much total combined money did all members of your HOUSEHOLD earn last year?"].value_counts()


def extract_income(income_str):
    if pd.isnull(income_str):
        return None
    income_str = income_str.split(" ")[0]
    if income_str == "Prefer":
        return None
    income_str = income_str.replace(",", "")
    income_str = income_str.replace("$", "")
    return int(income_str)

data["int_income"] = data["How much total combined money did all members of your HOUSEHOLD earn last year?"].apply(extract_income)
data["int_income"].describe()


# # Findings
# 
# Although we only have a rough approximation of income, and it skews downward because we took the first value in each string (the lower bound), the average income seems to be fairly high, although there is also a large standard deviation.
# 

# # Correlating Travel Distance And Income
# 

data[data["int_income"] < 50000]["How far will you travel for Thanksgiving?"].value_counts()


data[data["int_income"] > 150000]["How far will you travel for Thanksgiving?"].value_counts()


# # Findings
# 
# It appears that more people with high income have Thanksgiving at home than people with low income.  This may be because younger students, who don't have a high income, tend to go home, whereas parents, who have higher incomes, don't.
# 

# # Linking Friendship And Age
# 

data.pivot_table(
    index="Have you ever tried to meet up with hometown friends on Thanksgiving night?", 
    columns='Have you ever attended a "Friendsgiving?"',
    values="int_age"
)


data.pivot_table(
    index="Have you ever tried to meet up with hometown friends on Thanksgiving night?", 
    columns='Have you ever attended a "Friendsgiving?"',
    values="int_income"
)


# # Findings
# 
# It appears that people who are younger are more likely to attend a Friendsgiving, and try to meet up with friends on Thanksgiving.
# 

