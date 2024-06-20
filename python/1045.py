# # k-means clustering to reduce spatial data set size
# 
# This notebook reduces the size of a spatial data set by clustering with k-means. More info: http://geoffboeing.com/2014/08/visualizing-summer-travels/
# 

# magic command to display matplotlib plots inline within the ipython notebook webpage
get_ipython().magic('matplotlib inline')

# import necessary modules
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, kmeans2, whiten


# load the data set
df = pd.read_csv('data/summer-travel-gps-full.csv')
df.head()


# convert the lat-long coordinates into a two-dimensional numpy array and plot it
coordinates = df.as_matrix(columns=['lon', 'lat'])

most_index = df['city'].value_counts().head(6).index
most = pd.DataFrame(df[df['city'].isin(most_index)])
most.drop_duplicates(subset=['city'], keep='first', inplace=True)

plt.figure(figsize=(10, 6), dpi=100)
co_scatter = plt.scatter(coordinates[:,0], coordinates[:,1], c='b', edgecolor='', s=15, alpha=0.3)

plt.title('Scatter plot of the full set of GPS points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

for i, row in most.iterrows():
    plt.annotate(row['city'], 
                 xy=(row['lon'], row['lat']),
                 xytext=(row['lon'] + 1.5, row['lat'] + 0.6),
                 bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.6),
                 xycoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='k', alpha=0.8))

plt.show()


# N is the number of observations to group into k clusters
N = len(coordinates)

# normalize the coordinate data with the whiten function
# each feature is divided by its standard deviation across all observations to give it unit variance.
w = whiten(coordinates)

# k is the number of clusters to form
k = 100

# i is the number of iterations to perform
i = 50


# performs k-means on a set of observation vectors forming k clusters
# returns a k-length array of cluster centroid coordinates, and the final distortion
cluster_centroids1, distortion = kmeans(w, k, iter=i)

# plot the cluster centroids
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(cluster_centroids1[:,0], cluster_centroids1[:,1], c='y', s=100)
plt.show()


# the kmeans2 function classifies the set of observations into k clusters using the k-means algorithm
# returns a k by N array of centroids found at the last iteration of k-means,
# and an index of the centroid the i'th observation is closest to
# use optional argument minit='points' because the data is not evenly distributed
# minit='points' will choose k observations (rows) at random from data for the initial centroids
cluster_centroids2, closest_centroids = kmeans2(w, k, iter=i, minit='points')

# plot the cluster centroids
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(cluster_centroids2[:,0], cluster_centroids2[:,1], c='r', s=100)
plt.scatter(w[:,0], w[:,1], c='k', alpha=.3, s=10)
plt.show()


# plot the original full data set colored by cluster - not very useful with this many clusters
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(coordinates[:,0], coordinates[:,1], c=closest_centroids, s=100)
plt.show()


print('k =', k)
print('N =', N)

# the size of cluster_centroids1 and cluster_centroids2 should be the same as k
print(len(cluster_centroids1)) # appears some clusters collapsed, giving us a value less than k
print(len(cluster_centroids2))

# the size of closest_centroids should be the same as N
print(len(closest_centroids))

# the number of unique elements in closest_centroids should be the same as k
print(len(np.unique(closest_centroids)))


# for each set of coordinates in our full data set, add the closest_centroid from the kmeans2 clustering
rs = pd.DataFrame(df)
rs['closest_centroid'] = closest_centroids

# reduce the data set so there is only one row for each closest_centroid
rs.drop_duplicates(subset=['closest_centroid'], keep='first', inplace=True)
rs.head()


# plot the final reduced set of coordinate points
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(rs['lon'], rs['lat'], c='m', s=100)
plt.show()


# plot the cluster centroids vs the whitened coordinate points
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(cluster_centroids2[:,0], cluster_centroids2[:,1], c='r', alpha=.7, s=150)
plt.scatter(w[:,0], w[:,1], c='k', alpha=.3, s=10)
plt.show()


# plot the final reduced set of coordinate points vs the original full set
plt.figure(figsize=(10, 6), dpi=100)
rs_scatter = plt.scatter(rs['lon'], rs['lat'], c='r', alpha=.7, s=150)
df_scatter = plt.scatter(df['lon'], df['lat'], c='k', alpha=.3, s=5)

plt.title('Full data set vs k-means reduced set')
plt.legend((df_scatter, rs_scatter), ('Full set', 'Reduced set'), loc='upper left')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()


#rs.to_csv('data/summer-travel-gps-kmeans.csv', index=False)





# # Visualizing travel data with matplotlib
# 
# This notebook visualizes timestamped location data with matplotlib. More info: http://geoffboeing.com/2014/08/visualizing-summer-travels/
# 

# magic command to display matplotlib plots inline within the ipython notebook webpage
get_ipython().magic('matplotlib inline')

# import necessary modules
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.cm as cm, matplotlib.font_manager as fm
from datetime import datetime as dt
from time import time
from shapely.geometry import Polygon
from geopy.distance import great_circle
from geopandas import GeoDataFrame


# load the gps coordinate data, using the date as the full set's index
# the data files are encoded as utf-8: specify so to prevent matplotlib from choking on diacritics
df = pd.read_csv('data/summer-travel-gps-full.csv', encoding='utf-8', index_col='date', parse_dates=True)
rs = pd.read_csv('data/summer-travel-gps-dbscan.csv', encoding='utf-8')


title_font = fm.FontProperties(family='Arial', style='normal', size=20, weight='normal', stretch='normal')
label_font = fm.FontProperties(family='Arial', style='normal', size=16, weight='normal', stretch='normal')
ticks_font = fm.FontProperties(family='Arial', style='normal', size=12, weight='normal', stretch='normal')
annotation_font = fm.FontProperties(family='Arial', style='normal', size=11, weight='normal', stretch='normal')


# plot a histogram of the countries I visited
countdata = df['country'].value_counts()
ax = countdata.plot(kind='bar',                 
                    figsize=[9, 6], 
                    width=0.9, 
                    alpha=0.6, 
                    color='g',
                    edgecolor='w',
                    grid=False,
                    ylim=[0, 700])

ax.set_xticks(range(len(countdata)))
ax.set_xticklabels(countdata.index, rotation=45, rotation_mode='anchor', ha='right', fontproperties=ticks_font)
ax.yaxis.grid(True)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)

ax.set_title('Most Visited Countries', fontproperties=title_font)
ax.set_xlabel('', fontproperties=label_font)
ax.set_ylabel('Number of GPS records', fontproperties=label_font)

plt.show()


# let's re-scale that to make it look better
countdata = np.log(df['country'].value_counts())
ax = countdata.plot(kind='bar',                 
                    figsize=[9, 6], 
                    width=0.9, 
                    alpha=0.6, 
                    color='g',
                    edgecolor='w',
                    grid=False,
                    ylim=[0, 7])

ax.set_xticks(range(len(countdata)))
ax.set_xticklabels(countdata.index, rotation=45, rotation_mode='anchor', ha='right', fontproperties=ticks_font)
ax.yaxis.grid(True)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)

ax.set_title('Most Visited Countries', fontproperties=title_font)
ax.set_xlabel('', fontproperties=label_font)
ax.set_ylabel('Log of number of GPS records', fontproperties=label_font)

plt.show()


# plot a histogram of the cities I visited most
countdata = df['city'].value_counts().head(13)
xlabels = pd.Series(countdata.index)

ax = countdata.plot(kind='bar',                 
                    figsize=[9, 6], 
                    width=0.9, 
                    alpha=0.6, 
                    color='#003399',
                    edgecolor='w',
                    grid=False,
                    ylim=[0, 700])

ax.set_xticks(range(len(countdata)))
ax.set_xticklabels(countdata.index, rotation=45, rotation_mode='anchor', ha='right', fontproperties=ticks_font)
ax.yaxis.grid(True)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
    
ax.set_title('Most Visited Cities', fontproperties=title_font)
ax.set_xlabel('', fontproperties=label_font)
ax.set_ylabel('Number of GPS records', fontproperties=label_font)

plt.show()


# let's re-scale that to make it look better
countdata = np.log(df['city'].value_counts().head(13))
ax = countdata.plot(kind='bar',                 
                    figsize=[9, 6], 
                    width=0.9, 
                    alpha=0.6, 
                    color='#003399',
                    edgecolor='w',
                    grid=False,
                    ylim=[0, 7])

ax.set_xticks(range(len(countdata)))
ax.set_xticklabels(countdata.index, rotation=45, rotation_mode='anchor', ha='right', fontproperties=ticks_font)
ax.yaxis.grid(True)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
    
ax.set_title('Most Visited Cities', fontproperties=title_font)
ax.set_xlabel('', fontproperties=label_font)
ax.set_ylabel('Number of GPS records', fontproperties=label_font)

plt.show()


# get a representative point from the reduced data set for each of the most visited cities in the full set
most_index = df['city'].value_counts().head(8).index
most = pd.DataFrame(df[df['city'].isin(most_index)])
most.drop_duplicates(subset=['city'], keep='first', inplace=True)

# plot the final reduced set of coordinate points vs the original full set
fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rs['lon'], rs['lat'], c='m', edgecolor='k', alpha=.4, s=150)

# set axis labels, tick labels, and title
for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
ax.set_title('Most Visited Cities', fontproperties=title_font)
ax.set_xlabel('Longitude', fontproperties=label_font)
ax.set_ylabel('Latitude', fontproperties=label_font)

# annotate the most visited cities
for _, row in most.iterrows():
    ax.annotate(row['city'], 
                xy=(row['lon'], row['lat']),
                xytext=(row['lon'] + 1, row['lat'] + 1),
                fontproperties=annotation_font,
                bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.8),
                xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='k', alpha=0.8))
plt.show()


# get a representative point from the reduced data set for each of the most visited countries in the full set
most_index = df['country'].value_counts().head(8).index
most = pd.DataFrame(df[df['country'].isin(most_index)])
most.drop_duplicates(subset=['country'], keep='first', inplace=True)

# plot the final reduced set of coordinate points vs the original full set
fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rs['lon'], rs['lat'], c='m', edgecolor='k', alpha=.4, s=150)

# set axis labels, tick labels, and title
for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
ax.set_title('Most Visited Countries', fontproperties=title_font)
ax.set_xlabel('Longitude', fontproperties=label_font)
ax.set_ylabel('Latitude', fontproperties=label_font)

# annotate the most visited countries
for _, row in most.iterrows():
    ax.annotate(row['country'], 
                xy=(row['lon'], row['lat']),
                xytext=(row['lon'] - 1, row['lat'] - 1),
                fontproperties=annotation_font,
                bbox=dict(boxstyle="round", fc="1"),
                xycoords='data')
plt.show()


# next we'll identify the most isolated points (or clusters of points, based on some threshold distance)
start_time = time()

# what is the distance to the nearest point that is at least *threshold* miles away?
# ie, ignore all other points within this distance when identifying the next nearest point
# this treats everything within this threshold distance as a single cluster
threshold = 20

# create two new columns in the dataframe of simplified coordinates
# nearest_point will contain the index of the row of the nearest point from the original full data set
# nearest_dist will contain the value of the distance between these two points
rs['nearest_point'] = None
rs['nearest_dist'] = np.inf

# for each row (aka, coordinate pair) in the data set
for label, row in rs.iterrows():  
    
    point1 = (row['lat'], row['lon'])
    for label2, row2 in rs.iterrows():
        
        # don't compare the row to itself
        if(label != label2):
            
            # calculate the great circle distance between points            
            point2 = (row2['lat'], row2['lon'])
            dist = great_circle(point1, point2).miles

            # if this row's nearest is currently null, save this point as its nearest
            # or if this distance is smaller than the previous smallest, update the row
            if pd.isnull(rs.loc[label, 'nearest_dist']) | ((dist > threshold) & (dist < rs.loc[label, 'nearest_dist'])):
                rs.loc[label, 'nearest_dist'] = dist
                rs.loc[label, 'nearest_point'] = label2
            
print('process took {:.2f} seconds'.format(time()-start_time))


# sort the points by distance to nearest, then drop duplicates of nearest_point
most_isolated = rs.sort_values(by='nearest_dist', ascending=False).drop_duplicates(subset='nearest_point', keep='first')
most_isolated = most_isolated.head(5)
most_isolated


# plot the most isolated clusters in the data set
fig, ax = plt.subplots(figsize=[10, 6])

rs_scatter = ax.scatter(rs['lon'], rs['lat'], c='b', alpha=0.4, s=150)
df_scatter = ax.scatter(most_isolated['lon'], most_isolated['lat'], c='r', alpha=0.9, s=150)

# set axis labels, tick labels, and title
for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
ax.set_title('Most Isolated Clusters, and Distance to Next Nearest', fontproperties=title_font)
ax.set_xlabel('Longitude', fontproperties=label_font)
ax.set_ylabel('Latitude', fontproperties=label_font)

# annotate each of the most isolated clusters with city name, and distance to next nearest point + its name
for _, row in most_isolated.iterrows():
    ax.annotate(row['city'] + ', ' + str(int(row['nearest_dist'])) + ' mi. to ' + rs['city'][row['nearest_point']], 
                xy=(row['lon'], row['lat']),
                xytext=(row['lon'] + 0.75, row['lat'] + 0.25),
                fontproperties=annotation_font,
                bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.7),
                xycoords='data')
plt.show()


# # Now we'll use geopandas, shapely, and geopy to manipulate and plot summer travel data
# 

rs.head()


#load the shapefile of all countries in the world
all_countries = GeoDataFrame.from_file('shapefiles/countries_shp/world_country_admin_boundary_shapefile_with_fips_codes.shp')
all_countries.head()


# define the coordinates at the extent of our point data for our map
margin_width = 4
lon_range = [rs['lon'].min() - margin_width, rs['lon'].max() + margin_width]
lat_range = [rs['lat'].min() - margin_width, rs['lat'].max() + margin_width]

# create a rectangle from these coordinates
spatial_extent = Polygon([(lon_range[0], lat_range[0]), 
                          (lon_range[0], lat_range[1]), 
                          (lon_range[1], lat_range[1]),
                          (lon_range[1], lat_range[0])])

# one way to get the shapes is with geopandas intersection, but that chops the shapes off at the extent
#countries = all_countries['geometry'].intersection(spatial_extent)

# another way to get the shapes is geopandas intersects, which pulls the full shape
# but let's remove russia because it's too big
countries = all_countries[all_countries['geometry'].intersects(spatial_extent)]
countries = countries[countries['CNTRY_NAME'] != 'Russia']

countries.plot()


# get a representative point for each of the most visited cities
most_index = df['city'].value_counts().head(6).index
most = pd.DataFrame(rs[rs['city'].isin(most_index)])
most.drop_duplicates(subset=['city'], keep='first', inplace=True)


# draw a map of our point data on top of a basemap of country boundaries
fig = plt.figure()

# set the figure dimensions to the extent of the coordinates in our data
ydimension = int((lat_range[1] - lat_range[0]) / 4)
xdimension = int((lon_range[1] - lon_range[0]) / 4)
fig.set_size_inches(xdimension, ydimension)

# plot the country boundaries and then our point data
countries.plot(alpha=0)
rs_scatter = plt.scatter(x=rs['lon'], y=rs['lat'], c='m', edgecolor='w', alpha=0.7, s=100)

# annotate the most visited cities in the data set
for _, row in most.iterrows():
    plt.annotate(row['city'], 
                 xy=(row['lon'], row['lat']),
                 xytext=(row['lon'] + 0.5, row['lat'] - 1),
                 fontproperties=annotation_font,
                 bbox=dict(boxstyle='round', color='gray', fc='w', alpha=0.9),
                 xycoords='data')

# limit the coordinate space shown to the extent of our point data
plt.xlim(lon_range)
plt.ylim(lat_range)   

# set axis labels and title
plt.title('Map of {} GPS Coordinates in the Reduced Data Set'.format(len(rs)), fontproperties=title_font)

plt.show()


# # Now draw some pie charts to show proportions
# 

# function to produce more beautiful pie charts with matplotlib
def gbplot_pie(fractions, #values for the wedges
              labels, #labels for the wedges
              title = '', #title of the pie chart
              cm_name = 'Pastel1', #name of the matplotlib colormap to use
              autopct = '%1.1f%%', #format the value text on each pie wedge
              labeldistance = 1.05, #where to place wedge labels in relation to pie wedges
              shadow = True, #shadow around the pie
              startangle = 90, #rotate 90 degrees to start the top of the data set on the top of the pie
              edgecolor = 'w', #color of pie wedge edges
              width = 8, #width of the figure in inches
              height = 8, #height of the figure in inches
              grouping_threshold = None, #group all wedges below this value into one 'all others' wedge
              grouping_label = None): #what the label the grouped wedge
    
    # if the user passed a threshold value, group all fractions lower than it into one 'misc' pie wedge
    if not grouping_threshold==None:
        
        # if user didn't pass a label, apply a default text
        if grouping_label == None:
            grouping_label = 'Others'

        # select the rows greater than the cutoff value
        row_mask = fractions > grouping_threshold
        meets_threshold = fractions[row_mask]

        # group all other rows below the cutoff value
        all_others = pd.Series(fractions[~row_mask].sum())
        all_others.index = [grouping_label]

        # append the grouped row to the bottom of the rows to display
        fractions = meets_threshold.append(all_others)
        labels = fractions.index
    
    # get the color map then pull 1 color from it for each pie wedge we'll draw
    color_map = cm.get_cmap(cm_name)
    num_of_colors = len(fractions)
    colors = color_map([x/float(num_of_colors) for x in range(num_of_colors)])
    
    # create the figure and an axis to plot on
    fig, ax = plt.subplots(figsize=[width, height])
    
    # plot the pie
    wedges = ax.pie(fractions, 
                    labels = labels, 
                    labeldistance = labeldistance,
                    autopct = autopct,
                    colors = colors,
                    shadow = shadow, 
                    startangle = startangle)
    
    # change the edgecolor for each wedge
    for wedge in wedges[0]:
        wedge.set_edgecolor(edgecolor)
    
    # set the title and show the plot
    ax.set_title(title, fontproperties=title_font)
    plt.show()


countdata = df['city'].value_counts()
gbplot_pie(fractions = countdata,
           labels = countdata.index,
           title = 'Cities, by share of records in data set',
           grouping_threshold = 30,
           grouping_label = 'All Other Cities')


countdata = df['country'].value_counts()

# convert the pie wedge percentage into its absolute value
def my_autopct(pct):
    total = sum(countdata)
    val = int(round(pct*total)/100.0000)
    return '{v:d}'.format(v=val)

gbplot_pie(fractions = countdata,
           labels = countdata.index,
           title = 'Countries, by number of records in data set',
           autopct=my_autopct,
           grouping_threshold = 30,
           grouping_label = 'All Other Countries')


# plot a histogram of the GPS records by hour
countdata = df.groupby(df.index.hour).size()
countdata.index = ['{:02}:00'.format(hour) for hour in countdata.index]

ax = countdata.plot(kind='bar',                 
                    figsize=[9, 6], 
                    width=0.9, 
                    alpha=0.6,
                    color='c',
                    edgecolor='w',
                    grid=False,
                    ylim=[0, 120])

ax.set_xticks(range(len(countdata)))
ax.set_xticklabels(countdata.index, rotation=45, rotation_mode='anchor', ha='right', fontproperties=ticks_font)
ax.yaxis.grid(True)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
    
ax.set_title('Records in the data set, by hour of the day', fontproperties=title_font)
ax.set_xlabel('', fontproperties=label_font)
ax.set_ylabel('Number of GPS records', fontproperties=label_font)

plt.show()


# plot a histogram of the GPS records by day of week
countdata = df.groupby(df.index.weekday).size()
countdata.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = countdata.plot(kind='bar',                 
                    figsize=[6, 6], 
                    width=0.9, 
                    alpha=0.6,
                    color='c',
                    edgecolor='w',
                    grid=False,
                    ylim=[0, 300])

ax.set_xticks(range(len(countdata)))
ax.set_xticklabels(countdata.index, rotation=35, rotation_mode='anchor', ha='right', fontproperties=ticks_font)
ax.yaxis.grid(True)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)

ax.set_title('Total observations by day of the week', fontproperties=title_font)
ax.set_xlabel('', fontproperties=label_font)
ax.set_ylabel('Number of GPS records', fontproperties=label_font)

plt.show()


# lots of rows from this day in the balkans - why?
date = dt.strptime('2014-06-28', '%Y-%m-%d').date()
day_records = df[df.index.date==date]
print(len(day_records))

day_records.head()


# Ah, I had wifi in Kotor so more signals got through
# 

# plot a chart of records by date
countdata = df.groupby(df.index.date).size()
fig, ax = plt.subplots()

# create the line plot
ax = countdata.plot(kind='line',
                    figsize=[10, 5],
                    linewidth='3', 
                    alpha=0.5,
                    marker='o',
                    color='b')

# annotate the points around the balkans, for explanation
ax.annotate('Left the EU', 
            xy=('2014-06-20', 60),
            fontproperties=annotation_font,
            bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.7),
            xycoords='data')

ax.annotate('Had WiFi', 
            xy=('2014-06-23', 40),
            fontproperties=annotation_font,
            bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.7),
            xycoords='data')

ax.annotate('Return to EU', 
            xy=('2014-07-01', 53.5),
            fontproperties=annotation_font,
            bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.7),
            xycoords='data')

# set the x-ticks/labels for every nth row of the data - here, 1 tick mark per week (ie, 7 days)
xtick_data = countdata.iloc[range(0, len(countdata), 7)]
ax.xaxis.set_ticks(xtick_data.index)
ax.grid()

# set tick labels, axis labels, and title
ax.set_xticklabels(xtick_data.index, rotation=35, rotation_mode='anchor', ha='right', fontproperties=ticks_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
ax.set_title('Number of records in the data set, by date', fontproperties=title_font)
ax.set_xlabel('', fontproperties=label_font)
ax.set_ylabel('Number of GPS records', fontproperties=label_font)

plt.show()





# # DBSCAN clustering to reduce spatial data set size
# 
# This notebook reduces the size of a spatial data set by clustering with DBSCAN. More info: http://geoffboeing.com/2014/08/visualizing-summer-travels/
# 
# You might also be interested in [this notebook](https://github.com/gboeing/data-visualization/blob/master/location-history/google-location-history-cluster.ipynb) that uses this technique to cluster 1.2 million spatial data points and [this write-up](http://geoffboeing.com/2016/06/mapping-everywhere-ever-been/) of that project. Also see [here](https://en.wikipedia.org/wiki/Earth_radius#Mean_radius) for the number of kilometers in one radian.
# 

import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
get_ipython().magic('matplotlib inline')


# define the number of kilometers in one radian
kms_per_radian = 6371.0088


# load the data set
df = pd.read_csv('data/summer-travel-gps-full.csv', encoding='utf-8')
df.head()


# The scikit-learn DBSCAN haversine distance metric requires data in the form of [latitude, longitude] and both inputs and outputs are in units of radians.
# 
# ### Compute DBSCAN
# 
#   - `eps` is the physical distance from each point that forms its neighborhood
#   - `min_samples` is the min cluster size, otherwise it's noise - set to 1 so we get no noise
#   
# Extract the lat, lon columns into a numpy matrix of coordinates, then convert to radians when you call `fit`, for use by scikit-learn's haversine metric.
# 

# represent points consistently as (lat, lon)
coords = df.as_matrix(columns=['lat', 'lon'])

# define epsilon as 1.5 kilometers, converted to radians for use by haversine
epsilon = 1.5 / kms_per_radian


start_time = time.time()
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_

# get the number of clusters
num_clusters = len(set(cluster_labels))

# all done, print the outcome
message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
print(message.format(len(df), num_clusters, 100*(1 - float(num_clusters) / len(df)), time.time()-start_time))
print('Silhouette coefficient: {:0.03f}'.format(metrics.silhouette_score(coords, cluster_labels)))


# turn the clusters in to a pandas series, where each element is a cluster of points
clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])


# ### Find the point in each cluster that is closest to its centroid
# 
# DBSCAN clusters may be non-convex. This technique just returns one representative point from each cluster. First get the lat,lon coordinates of the cluster's centroid (shapely represents the *first* coordinate in the tuple as `x` and the *second* as `y`, so lat is `x` and lon is `y` here). Then find the member of the cluster with the smallest great circle distance to the centroid.
# 

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

centermost_points = clusters.map(get_centermost_point)


# unzip the list of centermost points (lat, lon) tuples into separate lat and lon lists
lats, lons = zip(*centermost_points)

# from these lats/lons create a new df of one representative point for each cluster
rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
rep_points.tail()


# pull row from original data set where lat/lon match the lat/lon of each row of representative points
# that way we get the full details like city, country, and date from the original dataframe
rs = rep_points.apply(lambda row: df[(df['lat']==row['lat']) & (df['lon']==row['lon'])].iloc[0], axis=1)
rs.to_csv('data/summer-travel-gps-dbscan.csv', encoding='utf-8')
rs.tail()


# plot the final reduced set of coordinate points vs the original full set
fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rs['lon'], rs['lat'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
df_scatter = ax.scatter(df['lon'], df['lat'], c='k', alpha=0.9, s=3)
ax.set_title('Full data set vs DBSCAN reduced set')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
plt.show()





# # Projecting and mapping spatial data with matplotlib
# 
# This notebook visualizes projected location data with matplotlib. More info: http://geoffboeing.com/2014/08/visualizing-summer-travels/
# 
# This notebook's code if fairly old now, and not particularly efficient. For more up-to-date code for loading, projecting, mapping, and analyzing spatial data, check out: https://github.com/gboeing/urban-data-science/tree/master/19-Spatial-Analysis-and-Cartography
# 

# magic command to display matplotlib plots inline within the ipython notebook webpage
get_ipython().magic('matplotlib inline')

# import necessary modules
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm
from time import time
from shapely.geometry import Polygon, Point
from geopy.distance import great_circle
from geopandas import GeoDataFrame
from descartes import PolygonPatch


# load the gps coordinate data
# the data files are encoded as utf-8: specify so to prevent matplotlib from choking on diacritics
df = pd.read_csv('data/summer-travel-gps-full.csv', encoding='utf-8')
rs = pd.read_csv('data/summer-travel-gps-dbscan.csv', encoding='utf-8')


# specify the fonts and background color for our map
title_font = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=15, weight='normal', stretch='normal')
annotation_font = fm.FontProperties(family='Bitstream Vera Sans', style='normal', size=10, weight='normal', stretch='normal')
backgroundcolor = '#e4f4ff'


# load the shapefile
all_countries = GeoDataFrame.from_file('shapefiles/world_borders/TM_WORLD_BORDERS-0.3.shp')

# the original CRS of our shapefile and point data
original_crs = all_countries.crs

# the projected CRS to convert our shapefile and point data into
target_crs = {'datum':'WGS84', 'no_defs':True, 'proj':'aea', 'lat_1':35, 'lat_2':55, 'lat_0':45, 'lon_0':10}


# change the CRS of the shapefile to the specified projected one
all_countries.to_crs(crs=target_crs, inplace=True)


# create a geometry column in our point data set for geopandas to use
rs['geometry'] = rs.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

# create a new geopandas geodataframe from the point data 
points = GeoDataFrame(rs)

# you must specify its original CRS to convert it to a different (projected) one later
points.crs = original_crs
points.head()


# convert the point data to the same projected CRS we specified earlier for our shapefile
points.to_crs(crs=target_crs, inplace=True)

# convert the projected points into discrete x and y columns for easy matplotlib scatterplotting
points['x'] = points['geometry'].map(lambda point: point.x)
points['y'] = points['geometry'].map(lambda point: point.y)    
points.head()


# calculate some margin so our data doesn't go right up to the edges of the plotting figure
x_margin_width = (points.bounds['maxx'].max() - points.bounds['minx'].min()) / 10
y_margin_width = (points.bounds['maxy'].max() - points.bounds['miny'].min()) / 3

# define the coordinates at the extent of our projected point data
xlim = (points.bounds['minx'].min() - x_margin_width, points.bounds['maxx'].max() + x_margin_width)
ylim = (points.bounds['miny'].min() - y_margin_width, points.bounds['maxy'].max() + y_margin_width)

# create a rectangle from these coordinates
spatial_extent = Polygon([(xlim[0], ylim[0]), 
                          (xlim[0], ylim[1]), 
                          (xlim[1], ylim[1]),
                          (xlim[1], ylim[0])])


# select the country shapes within the spatial extents of our point data
countries = all_countries[all_countries['geometry'].intersects(spatial_extent)]

# set dimensions in inches for the plotting figure size
xdim = (xlim[1] - xlim[0]) / 400000
ydim = (ylim[1] - ylim[0]) / 400000


# get a representative point for each of the most visited cities
most_index = df['city'].value_counts().head(6).index
most = pd.DataFrame(points[points['city'].isin(most_index)])
most.drop_duplicates(subset=['city'], keep='first', inplace=True)
most


def get_patches(countries, visited_countries):
    
    # specify the colors for our map
    facecolor = '#f7f7f7'
    visited_facecolor = '#eeeeee'
    edgecolor = '#cccccc'
    
    # create a list to contain a descartes PolygonPatch object for each Polygon in the GeoDataFrame geometry column
    patches = []

    for _, row in countries.iterrows():
        
        fc = visited_facecolor if row['NAME'] in visited_countries else facecolor
        
        # if this row contains a Polygon object
        if type(row['geometry']) == Polygon:
            patches.append(PolygonPatch(row['geometry'], fc=fc, ec=edgecolor, zorder=0))

        # else, this row contains a MultiPolygon object - this is a shapely object that contains multiple Polygon objects
        # for example, countries that contain islands will have one Polygon shape for their mainland, and others for the island geometries
        else:
            # for each sub-Polygon object in the MultiPolygon
            for polygon in row['geometry']:
                patches.append(PolygonPatch(polygon, fc=fc, ec=edgecolor, zorder=0))
    return patches                


# get a list of visited countries so we can shade those patches a different color
visited_countries = rs['country'].unique()
countries = countries.replace('The former Yugoslav Republic of Macedonia', 'Macedonia (FYROM)')


# create a figure, axis, and set the background color
fig = plt.figure(figsize=(xdim, ydim))
ax = fig.add_subplot(111)
ax.set_axis_bgcolor(backgroundcolor)

# add each patch we extracted from the GeoDataFrame's geometry column to the axis
for patch in get_patches(countries, visited_countries):
    ax.add_patch(patch)

# add the projected point data to the axis as a scatter plot
points_scatter = ax.scatter(x=points['x'], y=points['y'], c='m', alpha=0.4, s=100)

ax.set_title('Projected shapefile and GPS coordinates', fontproperties=title_font)

# set the axes limits
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# remove the tickmarks as these are projected geometries, the ticks are confusing northings/eastings
ax.set_xticks([])
ax.set_yticks([])

# annotate the most visited cities on the map
for _, row in most.iterrows():
    plt.annotate(row['city'], 
                 xy=(row['x'], row['y']),
                 xytext=(row['x'] + 35000, row['y'] - 100000),
                 bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.8),
                 xycoords='data')

plt.show()


# next we'll identify the most isolated points (or clusters of points, based on some threshold distance) 
# in the projected data set
start_time = time()

# what is the distance to the nearest point that is at least *threshold* miles away?
# ie, ignore all other points within this distance when identifying the next nearest point
# this treats everything within this threshold distance as a single cluster
threshold = 20

# create two new columns in the dataframe of simplified coordinates
# nearest_point will contain the index of the row of the nearest point from the original full data set
# nearest_dist will contain the value of the distance between these two points
points['nearest_point'] = None
points['nearest_dist'] = np.inf

# for each row (aka, coordinate pair) in the data set
for label, row in rs.iterrows():  
    
    point1 = (row['lat'], row['lon'])
    for label2, row2 in rs.iterrows():
        
        # don't compare the row to itself
        if(label != label2):
            
            # calculate the great circle distance between points            
            point2 = (row2['lat'], row2['lon'])
            dist = great_circle(point1, point2).miles

            # if this row's nearest is currently null, save this point as its nearest
            # or if this distance is smaller than the previous smallest, update the row
            if pd.isnull(points.loc[label, 'nearest_dist']) | ((dist > threshold) & (dist < points.loc[label, 'nearest_dist'])):
                points.loc[label, 'nearest_dist'] = dist
                points.loc[label, 'nearest_point'] = label2
            
print('process took %s seconds' % round(time() - start_time, 2))


# sort the points by distance to nearest, then drop duplicates of nearest_point
most_isolated = points.sort_values(by='nearest_dist', ascending=False).drop_duplicates(subset='nearest_point', keep='first')
most_isolated = most_isolated.head(5)


# plot the most isolated clusters in the data set

# create a figure, axis, and set the background color
fig = plt.figure(figsize=(xdim, ydim))
ax = fig.add_subplot(111)
ax.set_axis_bgcolor(backgroundcolor)

# add each patch we extracted from the GeoDataFrame's geometry column to the axis
for patch in get_patches(countries, visited_countries):
    ax.add_patch(patch)

# add the projected point data to the axis as a scatter plot
points_scatter = ax.scatter(points['x'], points['y'], c='m', alpha=.4, s=150)
isolated_scatter = ax.scatter(most_isolated['x'], most_isolated['y'], c='r', alpha=.9, s=150)

ax.set_title('Most Isolated Clusters, and Distance to Next Nearest', fontproperties=title_font)

# set the axes limits
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# remove the tickmarks as these are projected geometries, the ticks are confusing northings/eastings
ax.set_xticks([])
ax.set_yticks([])

# annotate each of the most isolated clusters with city name, and distance to next nearest point + its name
for _, row in most_isolated.iterrows():
    xytext = (row['x'], row['y'] - 120000) if row['city'] != 'Prizren' else (row['x'], row['y'] + 90000)
    ax.annotate(row['city'] + ', ' + str(int(row['nearest_dist'])) + ' mi. to ' + rs['city'][row['nearest_point']], 
                xy=(row['x'], row['y']),
                xytext=xytext,
                fontproperties=annotation_font,
                bbox=dict(boxstyle='round', color='k', fc='w', alpha=0.7),
                xycoords='data')

plt.show()





# # Reverse geocode latitude-longitude to city + country, worldwide
# 
# This notebook reverse geocodes a lat-long data set to city + country. 
# 
# More info: http://geoffboeing.com/2014/08/visualizing-summer-travels/
# 
# For an advanced version that uses local caching, see [this notebook](https://github.com/gboeing/data-visualization/blob/master/location-history/google-location-history-reverse-geocode.ipynb) and [this write-up](http://geoffboeing.com/2016/06/mapping-everywhere-ever-been/) of that project.
# 

# import necessary modules
import pandas as pd, requests, logging, time

# magic command to display matplotlib plots inline within the ipython notebook
get_ipython().magic('matplotlib inline')


# configure logging for our tool
lfh = logging.FileHandler('logs/reverse_geocoder.log', mode='w', encoding='utf-8')
lfh.setFormatter(logging.Formatter('%(levelname)s %(asctime)s %(message)s'))
log = logging.getLogger('reverse_geocoder')
log.setLevel(logging.INFO)
log.addHandler(lfh)
log.info('process started')


# load the gps coordinate data
df = pd.read_csv('data/summer-travel-gps-no-city-country.csv', encoding='utf-8')

# create new columns
df['geocode_data'] = ''
df['city'] = ''
df['country'] = ''

df.head()


# function that handles the geocoding requests
def reverse_geocode(latlng):
    time.sleep(0.1)
    url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng={0}'    
    request = url.format(latlng)
    log.info(request)
    response = requests.get(request)
    data = response.json()
    if 'results' in data and len(data['results']) > 0:
        return data['results'][0]


# create concatenated lat+lng column then reverse geocode each value
df['latlng'] = df.apply(lambda row: '{},{}'.format(row['lat'], row['lon']), axis=1)
df['geocode_data'] = df['latlng'].map(reverse_geocode)
df.head()


# identify municipality and country data in the json that google sent back
def parse_city(geocode_data):
    if (not geocode_data is None) and ('address_components' in geocode_data):
        for component in geocode_data['address_components']:
            if 'locality' in component['types']:
                return component['long_name']
            elif 'postal_town' in component['types']:
                return component['long_name']
            elif 'administrative_area_level_2' in component['types']:
                return component['long_name']
            elif 'administrative_area_level_1' in component['types']:
                return component['long_name']
    return None

def parse_country(geocode_data):
    if (not geocode_data is None) and ('address_components' in geocode_data):
        for component in geocode_data['address_components']:
            if 'country' in component['types']:
                return component['long_name']
    return None


df['city'] = df['geocode_data'].map(parse_city)
df['country'] = df['geocode_data'].map(parse_country)
print(len(df))
df.head()


# google's geocoder fails on anything in kosovo, so do those manually now
df.loc[df['country']=='', 'country'] = 'Kosovo'
df.loc[df['city']=='', 'city'] = 'Prizren'


# save our reverse-geocoded data set
df.to_csv('data/summer-travel-gps-full.csv', encoding='utf-8', index=False)





# # Douglas-Peucker simplification to reduce spatial data set size
# 
# This notebook uses shapely's implementation of the douglas-peucker algorithm to reduce the size of a spatial data set. The full data set consists of 1,759 lat-long coordinate points. More info: http://geoffboeing.com/2014/08/visualizing-summer-travels/
# 

# magic command to display matplotlib plots inline within the ipython notebook webpage
get_ipython().magic('matplotlib inline')

# import necessary modules
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from shapely.geometry import LineString
from time import time


# load the point data
df = pd.read_csv('data/summer-travel-gps-full.csv')
coordinates = df.as_matrix(columns=['lat', 'lon'])


# create a shapely line from the point data
line = LineString(coordinates)

# all points in the simplified object will be within the tolerance distance of the original geometry
tolerance = 0.015

# if preserve topology is set to False the much quicker Douglas-Peucker algorithm is used
# we don't need to preserve topology bc we just need a set of points, not the relationship between them
simplified_line = line.simplify(tolerance, preserve_topology=False)

print(line.length, 'line length')
print(simplified_line.length, 'simplified line length')
print(len(line.coords), 'coordinate pairs in full data set')
print(len(simplified_line.coords), 'coordinate pairs in simplified data set')
print(round(((1 - float(len(simplified_line.coords)) / float(len(line.coords))) * 100), 1), 'percent compressed')


# save the simplified set of coordinates as a new dataframe
lon = pd.Series(pd.Series(simplified_line.coords.xy)[1])
lat = pd.Series(pd.Series(simplified_line.coords.xy)[0])
si = pd.DataFrame({'lon':lon, 'lat':lat})
si.tail()


start_time = time()

# df_label column will contain the label of the matching row from the original full data set
si['df_label'] = None

# for each coordinate pair in the simplified set
for si_label, si_row in si.iterrows():    
    si_coords = (si_row['lat'], si_row['lon'])
    
    # for each coordinate pair in the original full data set
    for df_label, df_row in df.iterrows():
        
        # compare tuples of coordinates, if the points match, save this row's label as the matching one
        if si_coords == (df_row['lat'], df_row['lon']):
            si.loc[si_label, 'df_label'] = df_label
            break
            
print('process took %s seconds' % round(time() - start_time, 2))


si.tail()


# select the rows from the original full data set whose labels appear in the df_label column of the simplified data set
rs = df.loc[si['df_label'].dropna().values]

#rs.to_csv('data/summer-travel-gps-simplified.csv', index=False)
rs.tail()


# plot the final simplified set of coordinate points vs the original full set
plt.figure(figsize=(10, 6), dpi=100)
rs_scatter = plt.scatter(rs['lon'], rs['lat'], c='m', alpha=0.3, s=150)
df_scatter = plt.scatter(df['lon'], df['lat'], c='k', alpha=0.4, s=10)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Simplified set of coordinate points vs original full set')
plt.legend((rs_scatter, df_scatter), ('Simplified', 'Original'), loc='upper left')
plt.show()





