# # Analyses with NetworkX
# 

# Social networks have become a fixture of modern life thanks to social networking sites like Facebook and Twitter. Social networks themselves are not new, however. The study of such networks dates back to the early twentieth century, particularly in the field of sociology and anthropology. It is their prevelance in mainstream applciations that have moved these types of studies to the purview of data science. 
# 
# The basis for the analyses in this notebook comes from Graph Theory- the mathmatical study of the application and properties of graphs, originally motivated by the study of games of chance. Generally speaking, this involves the study of network encoding, and measuring properties of the graph. Graph theory can be traced back to Euler's work on the Konigsberg Bridges problem (1735). However in recent decades, the rise of the social network has influenced the discpline, particularly with Computer Science graph data structures and databases. 
# 
# A Graph, then can be defined as: `G = (V, E)` consiting of a finite set of nodes denoted by `V` or `V(G)` and a collection `E` or `E(G)` of unordered pairs `{u, v}` where `u, v ∈ V`. Less formally, this is a symbolic repreentation of a network and their relationships- a set of linked nodes.
# 
# Graphs can be either directed or undirected. Directed graphs simply have ordered relationships, undirected graphs can be seen as bidirectional directed graphs. A directed graph in a social network tends to have directional semantic relationships, e.g. "friends" - Abe might be friends with Jane, but Jane might not reciprocate. Undirected social networks have more general semantic relationships, e.g. "knows". Any directed graph can easily be converted to the more general undirected graph. In this case, the adjacency matrix becomes symmetric.
# 
# A few final terms will help us in our discussion. The cardinality of vertices is called the *order* of the Graph, where as the cardinality of the edges is called the *size*. In the above graph, the order is 7 and the size is 10. Two nodes are adjacent if they share an edge, they are also called neighbors and the neighborhood of a vertex is the set of all vertices that a vertex is connected to. The number of nodes in a vertex' neighborhood is that vertex' degree. 
# 
# ## Required Python Libraries ##
# 
# The required external libraries for the tasks in this notebook are as follows:
# 
# 1. networkx
# 2. matplotlib
# 3. python-louvain
# 
# NetworkX is a well maintained Python library for the creation, manipulation, and study of the structure of complex networks. Its tools allow for the quick creation of graphs, and the library also contains many common graph algorithms. In particular NetworkX complements Python's scientific computing suite of SciPy/NumPy, Matplotlib, and Graphviz and can handle graphs in memory of 10M's of nodes and 100M's of links. NetworkX should be part of every data scientist's toolkit. 
# 
# NetworkX and Python are the perfect combination to do social network analysis. NetworkX is designed to handle data at scale, data that is relevant to modern scale social networks. The core algorithms that are included are implemented on extremely fast legacy code. Graphs are hugely flexible (nodes can be any hashable type), and there is an extensive set of native IO formats. Finally, with Python- you'll be able to access or use a myriad of data sources from databases to the Internet.
# 

get_ipython().magic('matplotlib inline')


import os
import random
import community

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tribe.utils import *
from tribe.stats import *
from operator import itemgetter

## Some Helper constants
FIXTURES = os.path.join(os.getcwd(), "fixtures")
# GRAPHML  = os.path.join(FIXTURES, "benjamin@bengfort.com.graphml")
GRAPHML  = os.path.join(FIXTURES, "/Users/benjamin/Desktop/20150814T212153Z.graphml")


# The basics of creating a NetworkX Graph:
# 

H = nx.Graph(name="Hello World Graph")
# Also nx.DiGraph, nx.MultiGraph, etc

# Add nodes manually, label can be anything hashable
H.add_node(1, name="Ben", email="benjamin@bengfort.com")
H.add_node(2, name="Tony", email="ojedatony1616@gmail.com")

# Can also add an iterable of nodes: H.add_nodes_from
print nx.info(H)


H.add_edge(1,2, label="friends", weight=0.832)

# Can also add an iterable of edges: H.add_edges_from


print nx.info(H)
# Clearing a graph is easy
H.remove_node(1)
H.clear()


# For testing and diagnostics it's useful to generate a random Graph. NetworkX comes with several graph models including:
# 
# - Complete Graph `G=nx.complete_graph(100)`
# - Star Graph `G=nx.star_graph(100)`
# - Erdős-Rényi graph, binomial graph `G=nx.erdos_renyi_graph(100, 0.20)`
# - Watts-Strogatz small-world graph `G=nx.watts_strogatz_graph(100, 0.20)`
# - Holme and Kim power law `G=nx.powerlaw_cluster_graph(100, 0.20)`
# 
# But there are so many more, see [Graph generators](https://networkx.github.io/documentation/latest/reference/generators.html) for more information on all the types of graph generators NetworkX provides. These, however are the best ones for doing research on social networks.
# 

H = nx.erdos_renyi_graph(100, 0.20)


# Accessing Nodes and Edges:
# 

print H.nodes()[1:10]
print H.edges()[1:5]
print H.neighbors(3)


# For fast, memory safe iteration, use the `_iter` methods

edges, nodes = 0,0
for e in H.edges_iter(): edges += 1
for n in H.nodes_iter(): nodes += 1
    
print "%i edges, %i nodes" % (edges, nodes)


# Accessing the properties of a graph

print H.graph['name']
H.graph['created'] = strfnow()
print H.graph


# Accessing the properties of nodes and edges

H.node[1]['color'] = 'red'
H.node[43]['color'] = 'blue'

print H.node[43]
print H.nodes(data=True)[:3]

# The weight property is special and should be numeric
H.edge[0][34]['weight'] = 0.432
H.edge[0][36]['weight'] = 0.123

print H.edge[34][0]





# Accessing the highest degree node
center, degree = sorted(H.degree().items(), key=itemgetter(1), reverse=True)[0]

# A special type of subgraph
ego = nx.ego_graph(H, center)

pos = nx.spring_layout(H)
nx.draw(H, pos, node_color='#0080C9', edge_color='#cccccc', node_size=50)
nx.draw_networkx_nodes(H, pos, nodelist=[center], node_size=100, node_color="r")
plt.show()

# Other subgraphs can be extracted with nx.subgraph


# Finding the shortest path
H = nx.star_graph(100)
print nx.shortest_path(H, random.choice(H.nodes()), random.choice(H.nodes()))

pos = nx.spring_layout(H)
nx.draw(H, pos)
plt.show()


# Preparing for Data Science Analysis
print nx.to_numpy_matrix(H)
# print nx.to_scipy_sparse_matrix(G)


# ## Serialization of Graphs
# 
# Most Graphs won't be constructed in memory, but rather saved to disk. Serialize and deserialize Graphs as follows:
# 

G = nx.read_graphml(GRAPHML) # opposite of nx.write_graphml


print nx.info(G)


# NetworkX has a ton of Graph serialization methods, and most have methods in the following format for serialization format, `format`:
# 
# - Read Graph from disk: `read_format`
# - Write Graph to disk: `write_format`
# - Parse a Graph string: `parse_format`
# - Generate a random Graph in format: `generate_format`
#     
# The list of formats is pretty impressive:
# 
# - Adjacency List
# - Multiline Adjacency List
# - Edge List
# - GEXF
# - GML
# - Pickle
# - GraphML
# - JSON
# - LEDA
# - YAML
# - SparseGraph6
# - Pajek
# - GIS Shapefile
# 
# The JSON and GraphmL are most noteworthy (for use in D3 and Gephi/Neo4j)
# 

# ## Initial Analysis of Email Network
# 
# We can do some initial analyses on our network using built in NetworkX methods.
# 

# Generate a list of connected components
# See also nx.strongly_connected_components
for component in nx.connected_components(G):
    print len(component)


len([c for c in nx.connected_components(G)])


# Get a list of the degree frequencies
dist = FreqDist(nx.degree(G).values())
dist.plot()


# Compute Power log sequence
degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence

plt.loglog(degree_sequence,'b-',marker='.')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")


# Graph Properties
print "Order: %i" % G.number_of_nodes()
print "Size: %i" % G.number_of_edges()


print "Clustering: %0.5f" % nx.average_clustering(G)


print "Transitivity: %0.5f" % nx.transitivity(G)


hairball = nx.subgraph(G, [x for x in nx.connected_components(G)][0])
print "Average shortest path: %0.4f" % nx.average_shortest_path_length(hairball)


# Node Properties
node = 'benjamin@bengfort.com' # Change to an email in your graph
print "Degree of node: %i" % nx.degree(G, node)
print "Local clustering: %0.4f" % nx.clustering(G, node)


# ## Computing Key Players
# 

# In the previous graph, we began exploring ego networks and strong ties between individuals in our social network. We started to see that actors with strong ties to other actors created clusters that centered around themselves. This leads to the obvious question: who are the key figures in the graph, and what kind of pull do they have? We'll look at a couple measures of "centrality" to try to discover this: degree centrality, betweeness centrality, closeness centrality, and eigenvector centrality.
# 
# ### Degree Centrality ###
# The most common and perhaps simplest technique for finding the key actors of a graph is to measure the degree of each vertex. Degree is a signal that determines how connected a node is, which could be a metaphor for influence or popularity. At the very least, the most connected nodes are the ones that spread information the fastest, or have the greatest effect on their community. Measures of degree tend to suffer from dillution, and benefit from statistical techniques to normalize data sets. 
# 

def nbest_centrality(graph, metric, n=10, attribute="centrality", **kwargs):
    centrality = metric(graph, **kwargs)
    nx.set_node_attributes(graph, attribute, centrality)
    degrees = sorted(centrality.items(), key=itemgetter(1), reverse=True)
    
    for idx, item in enumerate(degrees[0:n]):
        item = (idx+1,) + item
        print "%i. %s: %0.4f" % item
    
    return degrees


degrees = nbest_centrality(G, nx.degree_centrality, n=15)


# ### Betweenness Centrality ###
# 
# A _path_ is a sequence of nodes between a star node and an end node where no node appears twice on the path, and is measured by the number of edges included (also called hops). The most interesting path to compute for two given nodes is the _shortest path_, e.g. the minimum number of edges required to reach another node, this is also called the node _distance_. Note that paths can be of length 0, the distance from a node to itself.
# 

# centrality = nx.betweenness_centrality(G)
# normalized = nx.betweenness_centrality(G, normalized=True)
# weighted   = nx.betweenness_centrality(G, weight="weight")

degrees = nbest_centrality(G, nx.betweenness_centrality, n=15)


# ### Closeness Centrality ###
# 
# Another centrality measure, _closeness_ takes a statistical look at the outgoing paths fora  particular node, v. That is, what is the average number of hops it takes to reach any other node in the network from v? This is simply computed as the reciprocal of the mean distance to all other nodes in the graph, which can be normalized to `n-1 / size(G)-1` if all nodes in the graph are connected. The reciprocal ensures that nodes that are closer (e.g. fewer hops) score "better" e.g. closer to one as in other centrality scores. 
# 

# centrality = nx.closeness_centrality(graph)
# normalied  = nx.closeness_centrality(graph, normalized=True)
# weighted   = nx.closeness_centrality(graph, distance="weight")

degrees = nbest_centrality(G, nx.closeness_centrality, n=15)


# ### Eigenvector Centrality ###
# 
# The eigenvector centrality of a node, v is proportional to the sum of the centrality scores of it's neighbors. E.g. the more important people you are connected to, the more important you are. This centrality measure is very interesting, because an actor with a small number of hugely influential contacts may outrank ones with many more mediocre contacts. For our social network, hopefully it will allow us to get underneath the celebrity structure of heroic teams and see who actually is holding the social graph together. 
# 

# centrality = nx.eigenvector_centality(graph)
# centrality = nx.eigenvector_centrality_numpy(graph)

degrees = nbest_centrality(G, nx.eigenvector_centrality_numpy, n=15)


# ## Clustering and Cohesion ##
# 
# In this next section, we're going to characterize our social network as a whole, rather than from the perspective of individual actors. This task is usually secondary to getting a feel for the most important nodes; but it is a chicken and an egg problem- determining the techniques to analyze and split the whole graph can be informed by key player analyses, and vice versa. 
# 
# The _density_ of a network is the ratio of the number of edges in the network to the total number of possible edges in the network. The possible number of edges for a graph of n vertices is n(n-1)/2 for an undirected graph (remove the division for a directed graph). Perfectly connected networks (every node shares an edge with every other node) have a density of 1, and are often called _cliques_. 
# 

print nx.density(G)


# Graphs can also be analyzed in terms of distance (the shortest path between two nodes). The longest distance in a graph is called the _diameter_ of the social graph, and represents the longest information flow along the graph. Typically less dense (sparse) social networks will have a larger diameter than more dense networks. Additionally, the average distance is an interesting metric as it can give you information about how close nodes are to each other. 
# 

for subgraph in nx.connected_component_subgraphs(G):
    print nx.diameter(subgraph)
    print nx.average_shortest_path_length(subgraph)


# Let's actually get into some clustering. The python-louvain library uses NetworkX to perform community detection with the louvain method. Here is a simple example of cluster partitioning on a small, built-in social network.
# 

partition = community.best_partition(G)
print "%i partitions" % len(set(partition.values()))
nx.set_node_attributes(G, 'partition', partition)


pos = nx.spring_layout(G)
plt.figure(figsize=(12,12))
plt.axis('off')

nx.draw_networkx_nodes(G, pos, node_size=200, cmap=plt.cm.RdYlBu, node_color=partition.values())
nx.draw_networkx_edges(G,pos, alpha=0.5)


# ## Visualizing Graphs ##
# 
# NetworkX wraps matplotlib or graphviz to draw simple graphs using the same charting library we saw in the previous chapter. This is effective for smaller size graphs, but with larger graphs memory can quickly be consumed.  To draw a graph, simply use the `networkx.draw` function, and then use `pyplot.show` to display it. 
# 

nx.draw(nx.erdos_renyi_graph(20, 0.20))
plt.show()


# There is, however, a rich drawing library underneath that lets you customize how the Graph looks and is laid out with many different layout algorithms. Let's take a look at an example using one of the built-in Social Graphs: The Davis Women's Social Club.
# 

# Generate the Graph
G=nx.davis_southern_women_graph()
# Create a Spring Layout
pos=nx.spring_layout(G)

# Find the center Node
dmin=1
ncenter=0
for n in pos:
    x,y=pos[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin:
        ncenter=n
        dmin=d

# color by path length from node near center
p=nx.single_source_shortest_path_length(G,ncenter)

# Draw the graph
plt.figure(figsize=(8,8))
nx.draw_networkx_edges(G,pos,nodelist=[ncenter],alpha=0.4)
nx.draw_networkx_nodes(G,pos,nodelist=p.keys(),
                       node_size=90,
                       node_color=p.values(),
                       cmap=plt.cm.Reds_r)





