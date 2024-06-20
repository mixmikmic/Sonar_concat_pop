# ## Geographic Visualization using plotly/geoplotib - Neerja Doshi
# 

# Dataset - The dataset used here is the World Happiness Report data for 2015. It can be found here: https://www.kaggle.com/unsdsn/world-happiness/data
# 

# import the necessary packages
import pandas as pd
import numpy as np

import geoplotlib
from geoplotlib.colors import ColorMap
from geoplotlib.colors import create_set_cmap
import pyglet
from sklearn.cluster import KMeans
from geoplotlib.layers import BaseLayer
from geoplotlib.core import BatchPainter
from geoplotlib.utils import BoundingBox

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


df = pd.read_csv('2015.csv')
print(df.head())
map_data = pd.read_csv('countries.csv')
map_data.head()


# ### Choropleth
# This map shows the happiness rank of all the countries in the world in 2015. The darker the colour, the higher the rank, i.e. the happier the people in that country. It can be seen that the countries of North America (the US, Mexico and Canada), Australia, New Zealand and the western countries of Europe have the happiest citizens. Countries that have lower happiness scores are the ones that are either war struck (eg., Iraq) or are highly underdeveloped as can be said of the countries in Africa like Congo and Chad. The top 5 countries are:
# 1. Switzerland
# 2. Iceland
# 3. Denmark
# 4. Norway
# 5. Canada
# 

scl = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(240, 210, 250)"]]


# scl = [[0.0, 'rgb(50,10,143)'],[0.2, 'rgb(117,107,177)'],[0.4, 'rgb(158,154,200)'],\
#             [0.6, 'rgb(188,189,220)'],[0.8, 'rgb(218,208,235)'],[1.0, 'rgb(250,240,255)']]
data = dict(type = 'choropleth', 
            colorscale = scl,
            autocolorscale = True,
            reversescale = True,
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Happiness Rank'], 
           text = df['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Orthographic'}))

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)


# ### Symbol Map
# To take our analysis further, we plot a symbol map where the size of the circles represents the Happiness Score while the colour represents the GDP of the countries. Larger the circle, happier the citizens while darker the circle, higher the GDP.
# From this plot, we can see that the top 5 countries we have seen above definitely have a much higher GDP. ** This seems to imply that more well off a country is economically, the happier its citizens are. ** Also, the underdeveloped countries like Chad, Congo, Burundi and Togo have a very low GDP and also a very low happiness score. While we cannot directly say that low GDP implies lower happiness, it seems like an important factor. This trend remain consistent throughout all the countries. ** There are almost no countries that have a high GDP but low happiness index or vice versa. **
# 
# Also geographic location and neighbours may be playing an important role. We can clusters/regions with countries having similar GDPs and similar Happiness Scores. ** So countries that have a good economy and good relations with their neighbours benefit from mutual growth and this is also reflected in their happiness scores. Countries that have disturbed neighbourhoods, like in middle east Asia (Iraq, Afghanistan, etc.), show much lower growth/economic prosperity as well as lower happiness scores. **
# 
# One thing that can also be noted is that in general, countries which are known for their lower population densities (https://www.worldatlas.com/articles/the-10-least-densely-populated-places-in-the-world-2015.html) like Denmark and Iceland are much happier than the more densly populated countries.
# 

df = df.merge(map_data, how='left', on = ['Country'])
df.head()


df['Happiness Score'].min(), df['Happiness Score'].max()


df['text']=df['Country'] + '<br>Happiness Score ' + (df['Happiness Score']).astype(str)
scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'country names',
        lon = df['Longitude'],
        lat = df['Latitude'],
        text = df['text'],
        mode = 'markers',
        marker = dict(
            size = df['Happiness Score']*3,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df['Economy (GDP per Capita)'],
            cmax = df['Economy (GDP per Capita)'].max(),
            colorbar=dict(
                title="GDP per Capita"
            )
        ))]

layout = dict(
        title = 'Happiness Scores by GDP',
        geo = dict(
#             scope='usa',
            projection=dict( type='Mercator' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
#             subunitcolor = "rgb(217, 217, 217)",
#             countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

symbolmap = go.Figure(data = data, layout=layout)

iplot(symbolmap)


# ### Extra plot
# A cluster plot may also be used to see if the clustered regions coincide with any of the regions above. (This plot opens in a new window)
# 

"""
Example of keyboard interaction
"""

class KMeansLayer(BaseLayer):

    def __init__(self, data):
        self.data = data
        self.k = 2


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['Longitude'], self.data['Latitude'])

        k_means = KMeans(n_clusters=self.k)
        k_means.fit(np.vstack([x,y]).T)
        labels = k_means.labels_

        self.cmap = create_set_cmap(set(labels), 'hsv')
        for l in set(labels):
            self.painter.set_color(self.cmap[l])
            self.painter.convexhull(x[labels == l], y[labels == l])
            self.painter.points(x[labels == l], y[labels == l], 2)
    
            
    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        ui_manager.info('Use left and right to increase/decrease the number of clusters. k = %d' % self.k)
        self.painter.batch_draw()


    def on_key_release(self, key, modifiers):
        if key == pyglet.window.key.LEFT:
            self.k = max(2,self.k - 1)
            return True
        elif key == pyglet.window.key.RIGHT:
            self.k = self.k + 1
            return True
        return False
  




data = geoplotlib.utils.DataAccessObject(df)
geoplotlib.add_layer(KMeansLayer(data))
geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(geoplotlib.utils.BoundingBox.DK)
geoplotlib.show()





# # Introduction to Data Visualization with Seaborn  - Neerja Doshi
# ### Data file : graduates.csv
# This file contains information about students graduating from American institutes for 11 years between 1993 and 2015. For every year and Major of graduation, there is information about the distribution in terms of race, employment status, employment sector, etc.
# 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('graduates.csv')
df.head()


new_majors = df['Major'][df['Asians'] == 0].unique()
new_majors
df = df[~df.Major.isin(new_majors)]
# df.nunique()

major_df = df.groupby(['Major']).sum().reset_index()
# major_df.head().T


# There were some Majors that did not exist before 2010, thus, their values are 0 in all the other years. Going further, those major have been excluded from analysis.
# 

# ### 1. Most to least studied majors - Boxplot
# First, here is a look at the number of student pursuing a particular Major over the years from 1993-2015. From the boxplot below, it is evident that Pyschology was the most favoured major while Chemical Engineering, Physics and Astronomy were the least studied. Also, the deviation in the number of graduates in Psychology, Biological Sciences and Computer Science and Math is quite high from year to year.
# 

plt.figure(figsize=(20,10))
g = sns.boxplot(x="Major", y="Total",data=df)

g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.title('Major v/s Number of Graduates')
plt.show() 


# ### 2. Unemployment over the years - Barplot
# From the bar plot below, we can see that for all the majors, unemployment is the lowest in the years from 2001-2008 (except 2003), but it increases after 2010. 2003 is one year in which all the Majors had a sudden spike in unemployment. While the reason for this spike is unclear, it is in keepng with the fact that in 2003, unemployment rate in the US had gone up to 5.8% with 308,000 job cuts.
# 
# Psychology has a huge number of unemployed graduates. The number of unemployed graduates there is almost double that of some of the other Majors.
# 

plt.figure(figsize=(20,10))
g = sns.factorplot(x="Major", y="Unemployed", data=df, size=12, kind="bar",hue = 'Year')
g.set_xticklabels(rotation=20)
plt.title('Number of unemployed graduates per major')
plt.show()


# ### 3. Employment Rate - Scatterplot (Multivariate)
# From the scatterplot below, we can see the emplyment rate for every major over the years. As we can see, the years 2001, 2006 and 2008 had a high proportion of graduates who got employed whereas 2010 onwards, employment rate reduced quite a bit. 
# 

df['Employment Rate'] = df['Employed']/df['Total']
df.head()


plt.figure(figsize=(20,10))
g = sns.stripplot(x="Major", y="Employment Rate",hue = 'Year', data=df, size = 15)
g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.legend(loc = 'lower right')
plt.title('Emploment Rate for every major over the years')
plt.show()


# ### 4. Distribution of Employment Rate pre and post 2003 - Violin plot
# 

# In the plot below, it can be seen that post 2003, the median employment rate is slightly lower than pre 2003. Also the deviation of employment rate over the years is higher post 2003 for almost all the Majors, especially Biological Sciences, Civil Engineering and Mechanical engineering. 
# 

f = lambda x: 'new' if x.Year > 2003 else 'old'
dfc = df.copy()
dfc['old_new'] = dfc.apply(f, axis=1)


plt.figure(figsize=(20,10))
g = sns.violinplot(x="old_new", y="Employment Rate", data=dfc, hue = 'Major');
plt.legend(loc = 'lower left')
plt.title('Distribution of Employment Rate pre and post 2003')
plt.xlabel('Pre and Post 2003')
plt.show()


# ### 5. Major v/s Male:Female Ratio - Scatterplot (Bivariate)
# From the scatterplot below, it is evident that the male to female ratio is least for Psychology Biological Sciences while it is the highest for Mechanical engineering, followed by Electrical Engineering.
# 

df['Gender Ratio'] = df['Males']/df['Females']
major_df['Gender Ratio'] = major_df['Males']/major_df['Females']


plt.figure(figsize=(20,10))
g = sns.stripplot(x="Major", y='Gender Ratio', data=df,color = 'red',  size = 10, jitter = True)
g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.title('Major v/s Gender Ratio')
plt.show()


# ### 6. Year v/s Gender - Linear Regression plot
# 

df1 = df[['Major', 'Year', 'Males', 'Females']]
df1 = pd.melt(df1, ['Major', 'Year'], var_name="Gender")


plt.figure(figsize=(20,10))
sns.lmplot(x="Year", y="value", col="Gender", hue="Major", data=df1,
           col_wrap=2, ci=None, palette="husl", size=6, 
           scatter_kws={"s": 50, "alpha": 1})
plt.show()


# From this, it is evident that there is an initially there are more number of males than females, but the rate of increase in number of females, post 1999 is much higher than the increase in number of males, especially in Computer Science and Math, Biological Sciences and Psychology.
# 

# ### 7. Distribution of race for every major - Swarmplot
# Also, as can be seen from the plots below, the number of Asians and Minorities graduating from American instititutes has been increasing considerably over the years. The number of Asians in particular has increased significantly in the field of Computer Science and Math while the number of minorities has increased significantly in Psychology.
# 

plt.figure(figsize = (15,8))
g = sns.swarmplot(x="Year", y="Asians", data=df, size = 8, hue = 'Major')
g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.title('Number of Asians over the Years')
plt.show()


plt.figure(figsize = (15,8))
g = sns.swarmplot(x="Year", y="Minorities", data=df, size = 8, hue = 'Major')
plt.title('Number of Minorities over the Years')
plt.show()


# ### 8. Distribution of race by year - Grouped bar chart
# 

df1 = df[['Major', 'Year', 'Whites', 'Asians', 'Minorities']]
df1 = pd.melt(df1, ['Major', 'Year'], var_name="Race")


plt.figure(figsize=(10,20))
g = sns.factorplot(x="Year", y="value", hue="Race", data=df1, 
                   size=10, kind="bar", palette="muted", legend_out="True") # legend_out draws the legend outside the chart

g.set_ylabels("Distribution by race")
plt.title('Distribution of Race by Year')
plt.show()


# From this plot also, it is evident that the number of whites is way higher than the other races, there is a steady increase in the number of non-whites that are graduating from American institutes.
# 

# ### 9. Doctorates in every Major over the Years - Faceted histogram
# 

# Binning the data based on years into 2 i.e. before and after 2001
df.columns
df.Major.unique()
df.Year.unique()

f = lambda x: 'new' if x.Year > 2003 else 'old'
dfc = df.copy()
dfc['old_new'] = dfc.apply(f, axis=1)


g = sns.FacetGrid(dfc, col="old_new", row = 'Major', margin_titles=True)
g.map(plt.hist, "Doctorates", color="steelblue",bins = 3)
plt.show() 


# From this plot it can be seen that the number of people pursuing their doctorate increased quite a bit post 2003 in every field except in Chemical and Civil Engineering. The most significant increase was in Biological Sciences.
# 

