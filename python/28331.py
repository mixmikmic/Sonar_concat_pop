# # Getting a Python CASTable Object from an Existing CAS Table
# 
# Many of the examples in the Python series of articles here use a CASTable object to invoke actions or apply DataFrame-like syntax to CAS tables.  In those examples, the CASTable object is generally the result of an action that loads the CAS table from a data file or other data source.  But what if you have a CAS table already loaded in a session and you want to create a new CASTable object that points to it?
# 
# The first thing you need is a connection to CAS.

import swat

conn = swat.CAS(host, port, username, password)


# We'll load a table of data here so we have something to work with.  We'll also specify a table name and caslib so they are easier to reference in the next step.
# 

conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/class.csv', 
              casout=dict(name='class', caslib='casuser'))


# Using the **tableinfo** action, we can see that the table exists, however, we didn't store the output of the **read_csv** method, so we don't have a CASTable object pointing to it.
# 

conn.tableinfo(caslib='casuser')


# The solution is fairly simple, you use the **CASTable** method of the CAS connection object.  You just pass it the name of the table and the name of the CASLib just as it is printed in `In[2]` above.
# 

cls = conn.CASTable('class', caslib='CASUSER')
cls


# We now have a CASTable object that we can use to interact with.
# 

cls.to_frame()


conn.close()





# # Fetching Sorted Data using Python
# 
# Because of the nature of distributed data on a grid of computers, data isn't always organized in a way that is most useful.  In many cases, you want the data to be displayed in an ordered form.  To do this, you'll want to use the **sortby=** parameter of the **fetch** action.  There is also a Pandas DataFrame-like way of fetching data in an ordered frorm.
# 
# We first need to start with a CAS connection.
# 

import swat

conn = swat.CAS(host, port, username, password)


# We need some data to work with, so we'll upload a small data set.
# 

cars = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
cars


# Using the **fetch** action, we can bring down a sample of the data.
# 

cars.fetch(to=5)


# To bring the data down in a sorted form, we use the **sortby=** parameter.  This parameter can take a list of column names, or a list of dictionaries with the keys 'name', 'order', and 'formatted'.  The 'name' key is required.  The 'order' parameter can be set to 'ascending' or 'descending'.  The 'formatted' parameter can be set to 'raw' or 'formatted' to sort the column based on the formatted value or the raw value; by default, it sorts by raw value.  Let's start with just using column names.
# 

cars.fetch(to=5, sortby=['Cylinders', 'EngineSize'])


# To reverse the order of the EngineSize column, we need to use a dictionary for that element.
# 

cars.fetch(to=5, sortby=['Cylinders', dict(name='EngineSize', order='descending')])


# ## DataFrame-style Sorting
# 
# As with most elements of the CASTable object, you can also apply sorting behaviors just like you do with Pandas DataFrames.  After you do this, each time the data is fetched from the server, it will automatically apply the sorting options for you.  This gives the appearance of a sorted CASTable object.
# 
# The CASTable object has a **sort_values** method (it also supports the older **sort** method name).  The **sort_values** method takes a list of column names as well as optional **ascending=** and **inplace=** options.  The **ascending=** option takes a boolean (for all columns) or a list of booleans (one for each column) to indicate whether it should be sorted in ascending or descending order.  The **inplace=** option indicates whether the sort options should be applied to the CASTable that **sort_value** is called on, or if it should return a copy of the CASTable object with the sort options applied.  Note that no copying or sorting is done when the **sort_value** option is used.  The sorting only occurs when data is being brought back to the client side either through **table.fetch** directly, or through any of the other methods that use **fetch** in the background (e.g., **head**, **tail**)
# 
# Let's apply some sort options to our cars object.
# 

sorted_cars = cars.sort_values(['Cylinders', 'EngineSize'])
sorted_cars.fetch(to=5)


# Now let's fetch some data using the **head** DataFrame method.  You'll see that the data is still coming back in sorted order.
# 

sorted_cars.head()


# Since we didn't use the **inplace=** option, the original CASTable object should still bring data back in an unsorted order.
# 

cars.head()


# Now let's try applying a sort order along with the **inplace=True** option.  This will modify the CASTable object directly rather than returning a copy.
# 

cars.sort_values(['Cylinders', 'EngineSize'], ascending=[True, False], inplace=True)
cars.head()


conn.close()


# ## Conclusion
# 
# Dealing with distributed data can sometimes take some getting used to, but hopefully with these tips on how to sort your data when it is brought back to the client, you can retrieve your data in the form that you are looking for.
# 




# ![SWAT](images/swat.png)
# 
# SWAT is the open-source Python interface to SAS’ cloud-based, fault-tolerant, in-memory analytics server.
# * Connects to CAS using binary (currently Linux only) or REST interface
# * Calls CAS analytic actions and returns results in Python objects
# * Implements a Pandas API that calls CAS actions in the background

# <h3  style="color:#1f77b4">How Does it Work?</h3>
# 

import swat


conn = swat.CAS('cas01', 49786)


# <h3  style="color:#1f77b4">Calling CAS Actions</h3>
# 

conn.serverstatus()


conn.userinfo()


conn.help();


# <h3  style="color:#1f77b4"> Loading Data </h3>
# 

tbl2 = conn.read_csv('https://raw.githubusercontent.com/'
                    'sassoftware/sas-viya-programming/master/data/cars.csv', 
                     casout=conn.CASTable('cars'))
tbl2


# <h3  style="color:#1f77b4"> CASTable </h3>
# <br/>
# CASTable objects contain a reference to a CAS table as well as filtering and grouping options, and computed columns.
# 

conn.tableinfo()


tbl = conn.CASTable('attrition')


tbl.columninfo()


get_ipython().magic('pinfo tbl2')


tbl2.fetch()


# <h3  style="color:#1f77b4"> Exploring Data </h3>
# 

tbl.summary() 


tbl.freq(inputs='Attrition')


# <h3  style="color:#1f77b4"> Building Analytical Models </h3>
# 

conn.loadactionset('regression')
conn.help(actionset='regression');


output = tbl.logistic(
    target='Attrition',
    inputs=['Gender', 'MaritalStatus', 'AccountAge'],
    nominals = ['Gender', 'MaritalStatus']
) 


output.keys()


output


from swat.render import render_html
render_html(output)


# ![CAS + Python](images/swatSection1.png)

# <h3 style="color:#1f77b4"> Pandas-style DataFrame API </h3>
# <br/>
# Many Pandas DataFrame features are available on the CASTable objects.
# 

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/'
                 'sassoftware/sas-viya-programming/master/data/cars.csv')
df.describe()


tbl2.describe()


tbl2.groupby('Origin').describe()


tbl[['Gender', 'AccountAge']].head()


# <h3 style="color:#1f77b4"> Visualization </h3>
# 
# ![Visualization](images/swatVisual.png)

from bokeh.plotting import show, figure
from bokeh.charts import Bar
from bokeh.io import output_notebook
output_notebook()


output1 = tbl.freq(inputs=['Attrition'])

p = Bar(output1['Frequency'], 'FmtVar', 
        values='Frequency',
        color="#1f77b4", 
        agg='mean', 
        title="", 
        xlabel = "Attrition",
        ylabel = 'Frequency',        
        bar_width=0.8,
        plot_width=600, 
        plot_height=400 
)
show(p)


# ![Slice 'n Dice](images/swatSection2.png)

conn.tableinfo()


tbl2.groupby(['Origin', 'Type']).describe()


tbl2[['MPG_CITY', 'MPG_Highway', 'MSRP']].describe()


tbl2[(tbl2.MSRP > 90000) & (tbl2.Cylinders < 12)].head()


# ![Python + SAS](images/swatSection3.png)

conn.runcode(code='''
    data cars_temp;
        set cars;
        sqrt_MSRP = sqrt(MSRP);
        MPG_avg = (MPG_city + MPG_highway) / 2;
    run;
''')


conn.tableinfo()


conn.loadactionset('fedsql')

conn.fedsql.execdirect(query='''
    select make, model, msrp,
    mpg_highway from cars
        where msrp > 80000 and mpg_highway > 20
''')


# # Getting CAS Action Help from Python
# 
# As with most things in programming, there are multiple ways of displaying help information about CAS action sets and actions from Python.  We'll outline each of those methods in this article.
# 
# The first thing we need is a connection to CAS.
# 

import swat

conn = swat.CAS(host, port, username, password)


# ## Using the `help` Action
# 
# The CAS server has a builtin help system that will tell you about action sets and actions.  To get help for all of the loaded action sets and a description of all of the actions in those action sets, you just call the **help** action with no parameters.  In this case, we are storing the output of the action to a variable.  That result contains the same information as the printed notes, but the information in encapsulated in DataFrame structures.  Unless you are going to use the action set information programmatically, there isn't much reason to have it printed twice.
# 

out = conn.help()


# If you only want to see the help for a single action set, you can specify the action set name as a parameter.
# 

out = conn.help(actionset='simple')


# You can also specify a single action as a parameter.  Calling the **help** action this way will also print descriptions of all of the action parameters.
# 

out = conn.help(action='summary')


# ## Using Python's `help` Function
# 
# In addition to the **help** action, you can also use Python's **help** function.  In this case, you have to specify an action set or action variable.  In the code below, we will get the help for the **simple** action set.  In addition to the actions in the action set, you will also get information about the action set's Python class. 
# 

help(conn.simple)


# Alternatively, you can specify a particular action attribute.  This will print information about the action parameters and the Python action class.
# 

help(conn.simple.summary)


# ## Using IPython's ? Operator
# 
# The IPython environment has a way of invoking help as well.  It is more useful in the notebook environment where the help content will pop up in a separate pane of the browser.  To bring up help for an action set, you simply add a **?** after the action set attribute name.
# 

get_ipython().magic('pinfo conn.simple')


# The **?** operator also works with action names.
# 

get_ipython().magic('pinfo conn.simple.summary')


# ## Conclusion
# 
# Which one of the above described methods of getting help on CAS actions that you decide to use really depends on what type of information you are looking for and what environment you are in.  If you are commonly working in IPython, the **?** operator method is likely to be your best bet.  If you simply want to see what actions are available in an action set, you may just call the **help** action directly.  And if you are looking for information about the action as well as the Python action class methods, then Python's **help** function is what you are looking for.
# 

conn.close()





# ![title](http://www.sas.com/content/sascom/en_us/software/viya/_jcr_content/par/styledcontainer_95fa/par/image_e693.img.png/1473452935247.png)
# 
# # A Simple Pipeline using Hypergroup to Perform Community Detection and Network Analysis

# The study of social networks has gained importance in recent years within social and behavioral research on HIV and AIDS. Social network research offers a means to map routes of potential viral transfer, to analyze the influence of peer norms and practices on the risk behaviors of individuals. This example analyzes the results of a study of high-risk drug use for HIV prevention in Hartford, Connecticut. This social network collected on drug users has 194 nodes and 273 edges.
# 

# ## Data Preparation
# ### Import Packages:   SAS Wrapper for Analytic Transfer (SWAT) and Open Source Libraries
# 

import swat
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

# Also import networkx used for rendering a network
import networkx as nx

get_ipython().magic('matplotlib inline')


# ### Connect to Cloud Analytic Services in SAS Viya
# 

s = swat.CAS('http://viya.mycompany.com:8777') # REST API


# ### Load the Action Set for Hypergroup
# 

s.loadactionset('hypergroup')


drug_network = pd.read_csv('drug_network.csv')


# **Hypergroup** doesn't support numeric source and target columns - so make sure to cast them as varchars.
# 

drug_network['SOURCE'] = drug_network['FROM'].astype(str)
drug_network['TARGET'] = drug_network['TO'].astype(str)
drug_network.head()


if s.tableexists('drug_network').exists:
    s.CASTable('drug_network').droptable()
    
dataset = s.upload_frame(drug_network, 
                         importoptions=dict(vars=[dict(type='double'),
                                                  dict(type='double'),
                                                  dict(type='varchar'),
                                                  dict(type='varchar')]),
                          casout=dict(name='drug_network', promote=True))


# ## Data Exploration
# 

# ### Get to Know Your Data (What are the Variables?)
# 

dataset.columninfo()


dataset.head()


dataset.summary()


# ### Graph Rendering Utility
# 

def renderNetworkGraph(filterCommunity=-1, size=18, sizeVar='_HypGrp_',
                       colorVar='', sizeMultipler=500, nodes_table='nodes',
                       edges_table='edges'):
    ''' Build an array of node positions and related colors based on community '''
    nodes = s.CASTable(nodes_table)
    if filterCommunity >= 0:
        nodes = nodes.query('_Community_ EQ %F' % filterCommunity)
    nodes = nodes.to_frame()

    nodePos = {}
    nodeColor = {}
    nodeSize = {}
    communities = []
    i = 0
    for nodeId in nodes._Value_:    
        nodePos[nodeId] = (nodes._AllXCoord_[i], nodes._AllYCoord_[i])
        if colorVar: 
            nodeColor[nodeId] = nodes[colorVar][i]
            if nodes[colorVar][i] not in communities:
                communities.append(nodes[colorVar][i])
        nodeSize[nodeId] = max(nodes[sizeVar][i],0.1)*sizeMultipler
        i += 1
    communities.sort()
  
    # Build a list of source-target tuples
    edges = s.CASTable(edges_table)
    if filterCommunity >= 0:
        edges = edges.query('_SCommunity_ EQ %F AND _TCommunity_ EQ %F' % 
                            (filterCommunity, filterCommunity))
    edges = edges.to_frame()

    edgeTuples = []
    i = 0
    for p in edges._Source_:
        edgeTuples.append( (edges._Source_[i], edges._Target_[i]) )
        i += 1
    
    # Add nodes and edges to the graph
    plt.figure(figsize=(size,size))
    graph = nx.DiGraph()
    graph.add_edges_from(edgeTuples)

    # Size mapping
    getNodeSize=[nodeSize[v] for v in graph]
    
    # Color mapping
    jet = cm = plt.get_cmap('jet')
    getNodeColor=None
    if colorVar: 
        getNodeColor=[nodeColor[v] for v in graph]
        cNorm  = colors.Normalize(vmin=min(communities), vmax=max(communities))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
        # Using a figure here to work-around the fact that networkx doesn't produce a labelled legend
        f = plt.figure(1)
        ax = f.add_subplot(1,1,1)
        for community in communities:
            ax.plot([0],[0], color=scalarMap.to_rgba(community), 
                    label='Community %s' % '{:2.0f}'.format(community),linewidth=10)
        
    # Render the graph
    nx.draw_networkx_nodes(graph, nodePos, node_size=getNodeSize, 
                           node_color=getNodeColor, cmap=jet)
    nx.draw_networkx_edges(graph, nodePos, width=1, alpha=0.5)
    nx.draw_networkx_labels(graph, nodePos, font_size=11, font_family='sans-serif')
        
    if len(communities) > 0:
        plt.legend(loc='upper left',prop={'size':11})
        
    plt.title('Hartford Drug User Social Network', fontsize=30)
    plt.axis('off')
    plt.show()


# ### Execute Community and Hypergroup Detection
# 

# Create output table objects
edges = s.CASTable('edges', replace=True)
nodes = s.CASTable('nodes', replace=True)

dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    allGraphs = True,
    edges     = edges,
    vertices  = nodes
)


renderNetworkGraph()


dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    allGraphs = True,
    community = True,
    edges     = edges,
    vertices  = nodes
)


# How many hypergroups and communities do we have?

nodes.distinct()


nodes.summary()


# ### Basic Community Analysis
# 

# What are the 5 biggest communities?

topKOut = s.CASTable('topKOut', replace=True)

nodes[['_Community_']].topk(
    aggregator = 'N',
    topK       = 4,
    casOut     = topKOut
)

topKOut = topKOut.sort_values('_Rank_').head(10)
topKOut.columns


nCommunities = len(topKOut)

ind = np.arange(nCommunities)    # the x locations for the groups

plt.figure(figsize=(8, 4))
p1 = plt.bar(ind + 0.2, topKOut._Score_, 0.5, color='orange', alpha=0.75)

plt.ylabel('Vertices', fontsize=12)
plt.xlabel('Community', fontsize=12)
plt.title('Number of Nodes for the Top %s Communities' % nCommunities)
plt.xticks(ind + 0.2, topKOut._Fmtvar_)

plt.show()


# >**Note:** This shows that the biggest communities have up to 63 vertices.
# 

# What nodes belong to community 4?
# 

nodes.query('_Community_ EQ 4').head()


# What edges do we have?

edges.head()


# ### Render the network graph
# 

renderNetworkGraph(colorVar='_Community_')


# ### Reduce Number of Communities
# 
# Limit the communities to 5.
# 

dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    community = True,
    nCommunities = 5,
    allGraphs = True,
    edges     = edges,
    vertices  = nodes
)


renderNetworkGraph(colorVar='_Community_')


# ### Analyze Node Centrality
# 

# How important is a user in the network?

dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    community = True,
    nCommunities = 5,
    centrality = True,
    mergeCommSmallest = True,
    allGraphs = True,
    graphPartition = True,
    scaleCentralities = 'central1', # returns centrality values closer to 1 in the center
    edges     = edges,
    vertices  = nodes
)


nodes.head()


# Betweenness centrality quantifies the number of times a node acts as a bridge along the shortest path(s) 
# between two other nodes.  As such it describes the importance of a node in a network.
# 

renderNetworkGraph(colorVar='_Community_', sizeVar='_Betweenness_')


# ### Filter Communities
# 
# Only filter community 2.
# 

renderNetworkGraph(2, size=10, sizeVar='_CentroidAngle_', sizeMultipler=5)


s.close()


# >Falko Schulz ▪ Principal Software Developer ▪ Business Intelligence Visualization R&D ▪ SAS® Institute ▪ [falko.schulz@sas.com](mailto:falko.schulz@sas.com) ▪ http://www.sas.com
# 

# >Data used by permission from Margaret R. Weeks at the Institute of Community Resesarch (http://www.incommunityresearch.org) https://www.researchgate.net/publication/227085871_Social_Networks_of_Drug_Users_in_High-Risk_Sites_Finding_the_Connections 
# 

# # Simple Statistics in Python
# 
# The actions in CAS cover a wide variety of statistical analyses.  While we can't cover all of them here, we'll at least get you started on some of the simpler ones.
# 
# First we need to get a CAS connection set up.
# 

import swat

conn = swat.CAS(host, port, username, password)


# ## The `simple` Action Set
# 
# The basic statistics package in CAS is called **simple** and should be already loaded.  If you are using IPython, you can see what actions are available using the **?** operator.
# 

get_ipython().magic('pinfo conn.simple')


# You can also use Python's **help** function.
# 

help(conn.simple)


# Let's start off with the **summary** action.  We'll need some data, so we'll load some CSV from a local file.  Then we'll run the action on it.
# 

cars = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
out = cars.summary()
out


# The result object here is a CASResults object which is a subclass of a Python dictionary.  In this case, we only have one key "Summary".  The value for this key is a DataFrame.  We can store the DataFrame in a variable so that it's easier to work with, then we can do any of the standard Pandas DataFrame operations on it.  Here we are setting the first column as the index for the DataFrame so that we can do data selection easier later on.
# 

df = out['Summary']
df.set_index(df.columns[0], inplace=True)
df


# Now that we have an index, we can use the **loc** property of the DataFrame to select rows based on index values as well as columns based on names.
# 

df.loc[['MSRP', 'Invoice'], ['Min', 'Mean', 'Max']]


# ## DataFrame methods on CASTable objects
# 
# In the previous example, we called the **summary** action directly.  This gave us a CASResults object that contained a DataFrame with the result of the action.  You can also use many of the Pandas DataFrame methods directly on the CASTable object so that, in many ways, they are interchangeable.  One of the most common methods used on a Pandas DataFrame is the **describe** method.  This includes statistics that would normally be gotten by running variations of the **summary**, **distinct**, **topk**, and **percentile** actions.  This is all done for you and the output created is the same as what you would get from an actual Pandas DataFrame.  The difference is that in the case of the CASTable version, you can handle much, much larger data sets.
# 

cars.describe()


# Other examples of DataFrame methods that work on CASTable objects are **min**, **max**, **std**, etc.  Each of these calls **simple.summary** in the background, so if you want to use more than one, you might be better off just calling the **describe** method once to get all of them.
# 

cars.min()


cars.max()


cars.std()


# ## Conclusion
# 
# Although we have just barely scratched the surface, you should now be able to get some basic statistical results back about your data.  Whether you want to use the action API directly, or the familiar Pandas DataFrame methods is up to you.
# 

conn.close()





# # Your First CAS Connection from Python
# 
# Let's start with a gentle introduction to the Python CAS client by doing some basic operations like creating a CAS connection and running a simple action.  You'll need to have Python installed as well as the SWAT Python package from SAS, and you'll need a running CAS server.
#  
# We will be using Python 3 for our example.  Specifically, we will be using the IPython interactive prompt (type 'ipython' rather than 'python' at your command prompt).  The first thing we need to do is import SWAT and create a CAS session.  We will use the name 'mycas1' for our CAS hostname and 12345 as our CAS port name.  In this case, we will use username/password authentication, but other authentication mechanisms are also possible depending on your configuration.
# 

# Import the SWAT package which contains the CAS interface
import swat

# Create a CAS session on mycas1 port 12345
conn = swat.CAS('mycas1', 12345, 'username', 'password') 


# As you can see above, we have a session on the server.  It has been assigned a unique session ID and more user-friendly name.  In this case, we are using the binary CAS protocol as opposed to the REST interface.  We can now run CAS actions in the session.  Let's begin with a simple one: **listnodes**.
# 

# Run the builtins.listnodes action
nodes = conn.listnodes()
nodes


# The **listnodes** action returns a ``CASResults`` object (which is just a subclass of Python's ordered dictionary).  It contains one key ('nodelist') which holds a Pandas DataFrame.  We can now grab that DataFrame to do further operations on it.
# 

# Grab the nodelist DataFrame
df = nodes['nodelist']
df


# Use DataFrame selection to subset the columns.
# 

roles = df[['name', 'role']]
roles


# Extract the worker nodes using a DataFrame mask
roles[roles.role == 'worker']


# Extract the controllers using a DataFrame mask
roles[roles.role == 'controller']


# In the code above, we are doing some standard DataFrame operations using expressions to filter the DataFrame to include only worker nodes or controller nodes.  Pandas DataFrames support lots of ways of slicing and dicing your data.  If you aren't familiar with them, you'll want to get acquainted on the [Pandas web site](http://pandas.pydoc.org/).
#  
# When you are finished with a CAS session, it's always a good idea to clean up.
# 

conn.close()


# Those are the very basics of connecting to CAS, running an action, and manipulating the results on the client side.  You should now be able to jump to other topics on the Python CAS client to do some more interesting work.
# 




# # How to Get CAS Action Documentation within Python-swat
# There is publically available documentation, but sometimes you just don't want to leave Python. <br>
# Copyright (c) 2017 SAS Institute Inc.
# 

# Load the swat package and turn off note messages
import swat
swat.options.cas.print_messages = False

# set the connection: host, port, username, password
s = swat.CAS(host, port, username, password)


# ### What are the default loaded actionsets?
# 

# list all loaded actionsets
s.builtins.actionSetInfo()


# ### What are the available actions for each actionset?
# 

# list each actionset with available actions as an ordered dict
s.help()


# ### I see an action that I need, how do I see the inputs?
# 

# session.actionset.action
help(s.dataPreprocess.impute)


# ### What I need to do is in an actionset that's not loaded, how do I see all available actionsets?
# 

# list all of the actionsets, whether they are loaded or not
s.builtins.actionSetInfo(all = True)


# ### Enable help on non previously loaded actionset and list newly available actions
# 

# load in new actionset
s.builtins.loadActionSet('decisionTree')

# get help again
s.help().decisionTree


# ### Let's see the arguments of a specific action that I now have loaded
# 

help(s.decisionTree.gbtreeTrain)


s.session.endsession() # end the session


# # Filtering Your Data using Python
# 
# Sometimes you just want to look at a small subset of your data.  Luckily, CAS makes this fairly easy to do through the use of WHERE clauses and variable lists.  If you are using CASTable objects in Python, you can also use DataFrame-like operations on them to filter your view of the data as well.  We'll look at some examples of both of these here.
# 
# The first thing we need to do is create a CAS connection.
# 

import swat

conn = swat.CAS(host, port, username, password)


# Now we need some data to work with, so we'll upload the cars data set.
# 

cars = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
cars


# Using the **len** function in Python and the **table.columinfo** action, we can get some more information about the data in the CAS table.  We see that it has 428 rows and 15 columns of data.
# 

len(cars)


cars.table.columninfo()


# Let's say that we only want to see the sports cars.  We can do this in one of two ways.  We could set the **where** parameter on the CASTable object to contain the string 'Type = "Sports"', or we could use the DataFrame data selection syntax.  Let's look at the **where** parameter first.
# 
# To apply a WHERE clause to a CASTable, we can manually set the **where** attribute to the expression we want to apply.
# 

cars.where = 'Type = "Sports"'
cars


# We can look at the length and a sample of the data to see that it has been subset.
# 

len(cars)


cars.head()


# Let's remove the **where** attribute and look at the DataFrame-like ways of subsetting data.
# 

del cars.where
cars


# The **query** method on the CASTable object mimics the **query** method of DataFrames.  However, in this case, the syntax of the expression is the same as a CAS WHERE clause.  So we use the same expression from above as the argument to the **query** method.
# 

cars.query('Type = "Sports"').head()


# Unlike setting the **where** parameter on the CASTable object, the **query** method does not embed the parameter in the CASTable object.  It creates a copy of the table.  If you want to apply the query to the CASTable object, you would add the **inplace=True** option.
# 

cars.query('Type = "Sports"', inplace=True)
cars


# A very popular way of subsetting the data in a DataFrame is using Python's getitem syntax (i.e., df[...]).  You can use that same syntax on CASTable objects.  First, we need to delete the WHERE clause that we had embedded using the last **query** method.
# 

del cars.where
cars


# The way to subset a table using DataFrame syntax is to index a CASTable object (e.g., cars[...]) using a condition on a column of that CASTable (e.g., cars.Type == 'Sports').  The condition is applied to the CASTable as a filter.
# 

cars[cars.Type == 'Sports'].head()


# The way this works is the condition creates a computed column that is treated as a mask.  If you look at the result of the expression, you'll see that it creates a computed column describing the expression.
# 

cars.Type == 'Sports'


# If you look at a sample of the data created by the computed column, you'll see that it is a series of ones and zeros.
# 

(cars.Type == 'Sports').head(40)


# When this mask is applied to a table, only the rows where the condition is true (i.e., computed expression is equal to one) show up in the output.
# 

cars[cars.Type == 'Sports'].head()


# It is also possible to combine expressions using Python's & and | operators.  Due to the order of operations, you need to surround each subexpression with parentheses.
# 

cars[(cars.Type == 'Sports') & (cars.Cylinders > 6)].head()


# Alternatively, you can chain your subsets which also results in combining those conditions into a single filter.
# 

cars[cars.Type == 'Sports'][cars.Cylinders > 6].head()


# If you want to see what's going on behind the scenes, you can always store the result in a variable.  The resulting object is simply another CASTable object with a WHERE clause and computed columns for the expressions.
# 

sports8 = cars[cars.Type == 'Sports'][cars.Cylinders > 6]
sports8


conn.close()


# ## Conclusion
# 
# We've shown two different ways of filtering data in a CAS table.  First we showed how you could manually apply a WHERE clause to a CASTable object.  The next way of filtering data was based on the Pandas DataFrame API.  You can either use the **query** method with a WHERE clause, or you can use the getitem syntax of Python to generate computed columns and WHERE clauses based on Python syntax.
# 




# ## Getting Started with Creating Charts with Python
# 
# There are many Python packages available for creating charts.  Which one you use really depends on 
# what the purpose of the final plot is.  For quick results, Pandas and Seaborn are quite popular.
# For publication-ready plots, Matplotlib is a very common choice (the previous two packages are actually
# wrappers around Matplotlib). And for interactive plots, you may want to try Plot.ly or Bokeh.
# 
# The first thing we need to do is connect to CAS and upload some data.  We are using the SAS CARS dataset in CSV form here.
# 

import swat

conn = swat.CAS(host, port, username, password)

tbl = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
tbl.head()


# Let's subset the data to just the sports cars using the **query** method of the CASTable object.  This works just like the **query** method on DataFrames.  We'll then download the data into a local DataFrame using the **head** method.  We've specified a maximum numer of rows as 1,000 here.  That will cover all of the sports cars in the result.  Finally, we'll add an index to the DataFrame that contains the make and model of the car.
# 

sports = tbl.query('Type = "Sports"')
sports


df = sports.head(1000)
df.set_index(df['Make'] + ' ' + df['Model'], inplace=True)
df.head()


# Now that we have some data to work with, let's create some charts.  To enable Matplotlib to embed images directly in the notebook, use the `%matplotlib` magic command. This works with Pandas plotting, Seaborn, and Matplotlib charts.
# 

get_ipython().magic('matplotlib inline')


# ### Pandas `plot` Method
# 
# Pandas DataFrames have a property called **plot** that makes it easy to create quick charts from the
# data in the DataFrame.  In older versions of Pandas, **plot** was a method with a **kind=** attribute that
# indicated the type of plot to create.  Newer versions of **plot** have methods for each individual 
# plot type such as bar, scatter, line, etc.
# 
# In the example below, we are subsetting the DataFrame to only include MSRP and Invoice, then we are
# calling the **plot.bar** method to create bar charts of the columns in subplots.  We will also use
# the rot= parameter to rotate the x axis labels.
# 

df[['MSRP', 'Invoice']].plot.bar(figsize=(15, 8), rot=-90, subplots=True)


# ### Creating Charts using Seaborn
# 
# The next step up from the **plot** method of DataFrames is using the Seaborn package.  This package 
# is a wrapper around Matplotlib that takes some of the work out of creating graphs and adds new
# ways of styling charts.
# 
# The code below creates a figure that contains two subplots as we did before.  Seaborn is then used to 
# create bar charts in each of the axes.  Finally, the x axis labels are overridden so that they can be 
# rotated -90 degrees as we did before.
# 

import seaborn as sns
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

bar = sns.barplot(df.index, df['MSRP'], ax=ax1, color='blue')
ax1.set_ylabel('MSRP')

bar2 = sns.barplot(df.index, df['Invoice'], ax=ax2, color='green')
ax2.set_ylabel('Invoice')

labels = bar2.set_xticklabels(df.index, rotation=-90)


# ### Using Matplotlib Directly
# 
# The final entry is the static graphics line is Matplotlib itself.  Panda's **plot** method and Seaborn are 
# just wrappers around Matplotlib, but you can still use Matplotlib directly.  For this case, it doesn't look 
# a lot different than the Seaborn case.  You'll noticed that we have to do a bit more adjustment of labels
# on the x axis and the x axis is a bit wider than it needs to be.  Seaborn just helps out with those details
# automatically.
# 

import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

ax1.bar(range(len(df.index)), df['MSRP'], color='blue')
ax1.set_ylabel('MSRP')

ax2.bar(range(len(df.index)), df['Invoice'], color='green')
ax2.set_ylabel('Invoice')

ax2.set_xticks([x + 0.25 for x in range(len(df.index))])
labels = ax2.set_xticklabels(df.index, rotation=-90)


# ### Using Plot.ly and Cufflinks
# 
# The Plot.ly package can be used a couple of different ways.  There's the Plot.ly API that uses standard
# Python structures as inputs, and there is an additional package called Cufflinks that integrates 
# Plot.ly charts into Pandas DataFrames.  Since we have our data in a DataFrame, it's easier to use 
# Cufflinks to start.
# 
# The code below uses Cufflinks' **iplot** method on the DataFrame.  The **iplot** method works much like the
# standard **plot** method on DataFrames except that it uses Plot.ly as the back-end rather than 
# Matplotlib.  After importing cufflinks, we use the **go_offline** function to indicate that we are using
# local graphics rather than the hosted Plot.ly service.
# 
# The benefit to Plot.ly graphics is that they are interactive when viewed in a web browser.
# 

import cufflinks as cf

cf.go_offline()

df[['MSRP', 'Invoice']].iplot(kind='bar', subplots=True, shape=(2, 1), shared_xaxes=True)


# To do a similar plot using the standard Plot.ly API takes a bit more work.
# 

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

data = [
    go.Bar(x=df.index, y=df.MSRP, name='MSRP'),
    go.Bar(x=df.index, y=df.Invoice, name='Invoice')
]

fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, print_grid=True)
fig.append_trace(data[0], 1, 1)
fig.append_trace(data[1], 2, 1)

fig['layout']['height'] = 700
fig['layout']['margin'] = dict(b=250)

iplot(fig)


# ## Creating Charts with Bokeh
# 
# Bokeh is a popular graphics library for Python.  The charting functionality is a more recent addition, so it isn't as mature as some of the other libraries here.  However, it is an extremeley powerful and popular Python package.  This chart could still use some work with label orientation and doing the two pieces as subplots rather than separate plots, but the functionality doesn't appear to exist in this release.
# 

from bokeh.charts import Bar, show
from bokeh.io import output_notebook

output_notebook()

try: show(Bar(df, values='MSRP', ylabel='MSRP', width=1000, height=400, color='blue'))
except: pass
try: show(Bar(df, values='Invoice', ylabel='Invoice', width=1000, height=400, color='green'))
except: pass


conn.close()


# ## Conclusion
# 
# We have shown the basics of several Python charting libraries here.  Which of these (if any) that you use for your purposes really depends on your needs.  The Matplotlib-based libraries are better at static and publication-style grahpics, whereas Plot.ly and Bokeh are more tuned to interactive charting in web browsers.  Hopefully, we have given you enough information to pique your interest in one of these packages for creating charts from your CAS results.
# 




# # Exporting Data from CAS using Python
# 
# While the **save** action can export data to many formats and data sources, there are also ways of easily converting CAS table data to formats on the client as well.  Keep in mind though that while you can export large data sets on the server, you may not want to attempt to bring tens of gigabytes of data down to the client using these methods.
# 
# While you can always use the **fetch** action to get the data from a CAS table, you might just want to export the data to a file.  To make this easier, the CASTable objects support the same **to_XXX** methods as Pandas DataFrames.  This includes **to_csv**, **to_dict**, **to_excel**, **to_html**, and others.  Behind the scenes, the **fetch** action is called and the resulting DataFrame is exported to the file corresponding to the export method used.  Let's look at some examples.
# 
# First we need a connection to the server.
# 

import swat

conn = swat.CAS(host, port, username, password)


# For purposes of this example, we will load some data into the server to work with.  You may already have tables in your server that you can use.
# 

tbl = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
tbl


tbl.head()


# Now that we have a CASTable object to work with, we can export the data from the CAS table that it references to a local file.  We'll start with CSV.  The **to_csv** method will return a string of CSV data if you don't specify a filename.  We'll do it that way in the following code.
# 

print(tbl.to_csv())


print(tbl.to_html())


print(tbl.to_latex())


# There are many other **to_XXX** methods on the CASTable object, each of which corresponds to the same **to_XXX** method on Pandas DataFrames.  The CASTable methods take the same arguments as the DataFrame counterparts, so you can [read the Pandas documentation for more information](http://pandas.pydata.org/pandas-docs/stable/api.html#id12).
# 

conn.close()





# # Running Data Step from Python
# 
# The **datastep** action set in CAS allows you to run data step code with the **datastep.runcode** action.  There are a few ways to execute data step code in the Python client.  We'll cover each of them here.
# 
# Let's get a CAS connection to work with first.
# 

import swat

conn = swat.CAS(host, port, username, password)


# Now we need to get some data into our session.
# 

cls = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/class.csv',
                    casout=dict(name='class', caslib='casuser'))
cls


# ## The `datastep.runcode` Action
# 
# The most basic was to run data step code is using the **datastep.runcode** action directly.  This action runs very much like running data step in SAS.  You simply specify CAS tables rather than SAS data sets as your input and output data.  In this example, we will comput the body mass index (BMI) of the students in the class data set.  The output of the **datastep.runcode** action will contain two keys: inputTables and outputTables.  Each of those keys points to a DataFrame of the information about the input and output tables including a CASTable object in the last column.
# 

out = conn.datastep.runcode('''
   data bmi(caslib='casuser');
      set class(caslib='casuser');
      BMI = weight / (height**2) * 703;
   run;
''')
out


# We can pull the output table DataFrame out using the following line of code.  The **ix** property is a DataFrame property that allows you to extract elements from a DataFrame at indexes or labels.  In this case, we want the element in row zero, column name **casTable**.
# 

bmi = out.OutputCasTables.ix[0, 'casTable']
bmi.to_frame()


# As you can see, we have a new CAS table that now includes the BMI column.
# 

# ## The CASTable `datastep` Method
# 
# CASTable objects have a **datastep** method that does some of the work of wrapping your data step code with the appropriate input and output data sets.  When using this method, you just give the body of the data step code.  The output table name will be automatically generated.  In this case, the output of the method is a CASTable object that references the newly generated table, so you don't have to extract the CASTable from the underlying action results.
# 

bmi2 = cls.datastep('''BMI = weight / (height**2) * 703''')
bmi2.to_frame()


# ## The `casds` IPython Magic Command
# 
# The third way of running data step from Python is reserved for IPython users.  IPython has commands that are called "magics".  These commands start with **%** (for one line commands) or **%%** (for cell commands) and allow extension developers to add functionality that isn't necessarily Python-based to your environment.  Included in SWAT is a packgae called **swat.cas.magics** that can be loaded to surface the **%%casds** magic command.  The **%%casds** magic gives you the ability to enter an entire IPython cell of data step code rather than Python code.  This is especially useful in the IPython notebook interface.
# 
# Let's give the **%%casds** magic a try.  First we have to load the **swat.cas.magics** extension.
# 

get_ipython().magic('load_ext swat.cas.magics')


# Now we can use the **%%casds** magic to enter an entire cell of data step code.  The **%casds** magic requires at least one argument which contains the CAS connection object where the action should run.  In most cases, you'll want to add the **--output** option as well which specifies the name of an output variable that will be surfaced to the Python environment which contains the output of the **datastep.runcode** action.
# 

get_ipython().run_cell_magic('casds', '--output out2 conn', "\ndata bmi3(caslib='casuser');\n   set class(caslib='casuser');\n   BMI = weight / (height**2) * 703;\nrun;")


# Just as before, we can extract the output CASTable object from the returned DataFrames.
# 

bmi3 = out2.OutputCasTables.ix[0, 'casTable']
bmi3.to_frame()


# ## Conclusion
# 
# If you are an existing SAS user, you may be relieved to find that you can still use data step in the CAS environment.  Even better, you can run it from Python.  This blend of languages and environments gives you an enormous number of possibilities for data analysis, and should make SAS programmers feel right at home in Python.
# 

conn.close()





