import os
import pandas as pd
import datetime

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from make_df import make_df


spring_start = "2004-03-01" 
spring_end = "2004-05-31"
summer_start = "2004-06-01" 
summer_end = "2004-08-31"
autumn_start = "2004-09-01" 
autumn_end = "2004-11-30"
winter_start = "2004-12-01" 
winter_end = "2004-02-28"





files= []
folder = "/Users/gianluca/Downloads/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT.part1/USA_CA_Montague-Siskiyou.County.AP.725955_TMY3/"
for file in os.listdir(folder):
    files.append(file)


path = "/Users/gianluca/Downloads/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT.part1/USA_CA_Montague-Siskiyou.County.AP.725955_TMY3/RefBldgFullServiceRestaurantNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv"
df = make_df(path)


start_date = "2004-01-01"
end_date = "2004-12-31"

workdays = pd.bdate_range(start_date,end_date)
all_days = pd.date_range(start_date,end_date)
weekends = all_days - workdays


def plot_range(file_input, output_folder, start_date, end_date):
    df = make_df(folder + file_input)
    
    workdays = pd.bdate_range(start_date,end_date)
    all_days = pd.date_range(start_date,end_date)
    weekends = all_days - workdays
    
    for day in workdays:
        plt.subplot(131)
        plt.plot(df[str(day.date())].index.hour,
                 df[str(day.date())]['Electricity:Facility [kW](Hourly)'], color=(0,0,0,0.1))
    for day in all_days:
        plt.subplot(132)
        plt.plot(df[str(day.date())].index.hour,
                 df[str(day.date())]['Electricity:Facility [kW](Hourly)'], color=(0,0,0,0.1))
    plt.title(file_input, fontsize = 50)
    for day in weekends:
        plt.subplot(133)
        plt.plot(df[str(day.date())].index.hour,
                 df[str(day.date())]['Electricity:Facility [kW](Hourly)'], color=(0,0,0,0.1))
    plt.rcParams["figure.figsize"] = (50,8)

    if not os.path.exists(folder + output_folder):
        os.makedirs(folder + output_folder)
    
    plt.savefig(folder + output_folder + "/" + file_input.split(".")[0] + "_loads.png")
    return plt.show()


file


plot_range("RefBldgWarehouseNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv", "test_plots", "2004-01-01","2004-12-31")


def plot_day(df, days):
    for day in days:
        plt.plot(df[day].index.hour,
                         df[day]['Electricity:Facility [kW](Hourly)'])
    plt.show()


df = make_df(folder + "RefBldgPrimarySchoolNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv")


# ## NB:
# Daylight Saving Time starts the 12 of March and ends the 5 of november
# 

# ```df1 = pd.concat([df[:"2004-03-11"],
#                  df["2004-03-12":"2004-11-5"].shift(1),
#                  df["2004-11-6":]])
# df1.fillna(method="bfill",inplace=True)```
# 

df = make_df(folder+"RefBldgOutPatientNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv")
plot_range("RefBldgOutPatientNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv", spring_start, spring_end)
plot_range("RefBldgOutPatientNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv", summer_start, summer_end)
plot_range("RefBldgOutPatientNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv", autumn_start, autumn_end)


for file in files:
    plot_range(file,"2004-01-01","2004-12-31")


folder


# To obtain the number of peaks we can normalize it to the mean to get something like a sinusoid, and then count the times it passes above or under it.
# In this way we avoid counting all the small peaks that have no role
# 

# # Features:
# * building kind!!! (school, office, house, etc)
# * city code
# 




import os, sys
sys.path.append("tools/")
import pandas as pd
import datetime

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


from make_buildings_dataset import make_buildings_dataset
from describe_clusters import describe_clusters


from plot_funcs import *


path = "COMMERCIAL/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT.part1/USA_CA_Montague-Siskiyou.County.AP.725955_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv"


from make_df import make_df
df = make_df(path)


days = [day.strftime("%Y-%m-%d") for day in df.index.date]


plt.rcParams["figure.figsize"] = (30,10)


plot_range(path, "2004-01-04", "2004-12-31")


#residential = make_buildings_dataset("RESIDENTIAL/")


df = pd.read_csv("all_commercial.csv").drop("Unnamed: 0", axis = 1)


from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=6)

data = df[list(str(i) for i in range(0,24))]
cluster.fit(data)


def plot_building(df_row, color = "blue"):
    if color == "hairball":
        color = (0,0,0,0.01)
    plt.plot(df_row[list(str(i) for i in range(0,24))].values[0].tolist(), color = color)


plt.rcParams["figure.figsize"] = (20,12)


cmap = plt.cm.Set1(np.linspace(0, 1, 6)).tolist()
colors = ["firebrick", "darkorange", "forestgreen", "royalblue", "mediumvioletred", "gold"]
n_plot = 231

for n_cluster in range(0, len(cluster.cluster_centers_)):
    cluster_elements = (cluster.labels_ == n_cluster)

#    fig = plt.figure()
    plt.suptitle('Commercial load clustering', fontsize=24, fontweight='bold')
    plt.subplot(n_plot)
    plt.style.use('seaborn-bright')
    plt.title("Cluster n" + str(n_cluster + 1))
    
    for row in df.ix[cluster_elements].index:
        plot_building(df.ix[[row]], color = "hairball")
    
    print("CLUSTER", str(n_cluster))
    print("total elements:", str((cluster.labels_ == n_cluster).sum()))
    for unique in df.ix[cluster.labels_ == n_cluster, "building_kind"].unique():
        print("n ", unique, "=", str(len(df.ix[cluster.labels_ == n_cluster].loc[df["building_kind"] == unique])),
             "out of", len(df.loc[df["building_kind"] == unique]))
    plt.plot(cluster.cluster_centers_[n_cluster], color = colors[n_cluster], linewidth = 3)
    
    plt.xlim([0, 23])
    plt.ylim([0, 1.1])
    plt.xlabel("Hour of the day")
    plt.ylabel("Load/Peak")
    n_plot += 1
plt.savefig("clusters.png")
plt.show()


df_row = df.ix[[1]]
a = plot_building(df.ix[[1]], color = "hairball")


for center in cluster.cluster_centers_:
    plt.plot(center)
    plt.show()


import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) #enables plotly offline plotting
py.tools.set_credentials_file(username='gianlucahmd', api_key=open('../plotly_key.txt', "r").read())

data = []
n = 1
colors = ["green", "red", "blue", "orange", "purple"]
for center in cluster.cluster_centers_:
    data.append(go.Scatter(
            x = list(range(0,24)),
            y = center,
            name = "cluster " + str(n),
            line = dict(color = colors[n - 1])
        ))
    n += 1

layout = go.Layout(
    title = "Loads Clustering for " + str(len(cluster.cluster_centers_)) + " clusters",
    xaxis = dict(
        title = "H of day",
        range = [0,24]
    ),
    yaxis = dict(
        title = "Scaled consumption",
        range = [0,1.2]
    ))

fig = go.Figure(data = data, layout = layout)
iplot(fig)


# # Notes:
# * It's impossible to separate all working days perfectly, since for instance the school in  RefBldgSecondarySchoolNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv on Presidents' Day was open for some kind of activity...
# * test different seasons for different states with different climates
# 

for i in range(0, cluster.n_clusters):
    print("N elements in cluster", str(i), "=", (cluster.labels_ == i).sum())


# ## Division in 4 clusters:
# * N elements in cluster 0 = 8420
# * N elements in cluster 1 = 2808
# * N elements in cluster 2 = 936
# * N elements in cluster 3 = 2812
# 
# ## Division in 5 clusters:
# * N elements in cluster 0 = 5382
# * N elements in cluster 1 = 2808
# * N elements in cluster 2 = 2808
# * N elements in cluster 3 = 936
# * N elements in cluster 4 = 3042
# 

# # Features:
# * building kind!!! (school, office, house, etc)
# * city code
# 

