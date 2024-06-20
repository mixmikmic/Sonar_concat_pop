# ### Created by Tirthajyoti Sarkar, Ph.D., Jan 2018
# 

# # Mean-shift Clustering Technique
# 
# Mean-shift clustering aims to discover blobs in a smooth density of samples. It is a centroid based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.
# 
# Given a candidate centroid $x_i$ for iteration $t$, the candidate is updated according to the following equation:
# 
# $$x_i^{t+1} = x_i^t + m(x_i^t)$$
# 
# Where $N(x_i)$ is the neighborhood of samples within a given distance around $x_i$ and $m$ is the mean shift vector that is computed for each centroid that points towards a region of the maximum increase in the density of points. This is computed using the following equation, effectively updating a centroid to be the mean of the samples within its neighborhood:
# 
# $$m(x_i) = \frac{\sum_{x_j \in N(x_i)}K(x_j - x_i)x_j}{\sum_{x_j \in N(x_i)}K(x_j - x_i)}$$
# 
# **The algorithm automatically sets the number of clusters, instead of relying on a parameter bandwidth, which dictates the size of the region to search through**. This parameter can be set manually, but can be estimated using the provided estimate_bandwidth function, which is called if the bandwidth is not set.
# 
# **The algorithm is not highly scalable, as it requires multiple nearest neighbor searches** during the execution of the algorithm. The algorithm is guaranteed to converge, however the algorithm will stop iterating when the change in centroids is small.
# 

# ## Make the synthetic data
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


from sklearn.cluster import MeanShift
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.4,random_state=101)


X.shape


plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],edgecolors='k',c='orange',s=75)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# ## Clustering
# 

ms_model = MeanShift().fit(X)
cluster_centers = ms_model.cluster_centers_
labels = ms_model.labels_
n_clusters = len(cluster_centers)
labels = ms_model.labels_


# #### Number of detected clusters and their centers
# 

print("Number of clusters detected by the algorithm:", n_clusters)


print("Cluster centers detected at:\n\n", cluster_centers)


plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],edgecolors='k',c=ms_model.labels_,s=75)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# #### Homogeneity
# 
# Homogeneity metric of a cluster labeling given a ground truth.
# 
# A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class. This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
# 

print ("Homogeneity score:", metrics.homogeneity_score(labels_true,labels))


# #### Completeness
# 
# Completeness metric of a cluster labeling given a ground truth.
# 
# A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster. This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
# 

print("Completeness score:",metrics.completeness_score(labels_true,labels))


# ## Time complexity and model quality as the data size grows
# 

import time
from tqdm import tqdm 


n_samples = [10,20,50,100,200,500,1000,2000,3000,5000,7500,10000]
centers = [[1, 1], [-1, -1], [1, -1]]
t_ms = []
homo_ms=[]
complete_ms=[]

for i in tqdm(n_samples):
    X,labels_true = make_blobs(n_samples=i, centers=centers, cluster_std=0.4,random_state=101)
    t1 = time.time()
    ms_model = MeanShift().fit(X)
    t2=time.time()
    t_ms.append(t2-t1)
    homo_ms.append(metrics.homogeneity_score(labels_true,ms_model.labels_))
    complete_ms.append(metrics.completeness_score(labels_true,ms_model.labels_))


plt.figure(figsize=(8,5))
plt.title("Time complexity of Mean Shift\n",fontsize=20)
plt.scatter(n_samples,t_ms,edgecolors='k',c='green',s=100)
plt.plot(n_samples,t_ms,'k--',lw=3)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Number of samples",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Time taken for model (sec)",fontsize=15)
plt.show()


plt.figure(figsize=(8,5))
plt.title("Homogeneity score with data set size\n",fontsize=20)
plt.scatter(n_samples,homo_ms,edgecolors='k',c='green',s=100)
plt.plot(n_samples,homo_ms,'k--',lw=3)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Number of samples",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Homogeneity score",fontsize=15)
plt.show()


plt.figure(figsize=(8,5))
plt.title("Completeness score with data set size\n",fontsize=20)
plt.scatter(n_samples,complete_ms,edgecolors='k',c='green',s=100)
plt.plot(n_samples,complete_ms,'k--',lw=3)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Number of samples",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Completeness score",fontsize=15)
plt.show()


# ## How well the cluster detection works in the presence of noise?
# 
# Create data sets with varying degree of noise std. dev and run the model to detect clusters.
# 

noise = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,1.75,2.0]
n_clusters = []
for i in noise:
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=200, centers=centers, cluster_std=i,random_state=101)
    ms_model=MeanShift().fit(X)
    n_clusters.append(len(ms_model.cluster_centers_))


print("Detected number of clusters:",n_clusters)
plt.figure(figsize=(8,5))
plt.title("Cluster detection with noisy data\n",fontsize=20)
plt.scatter(noise,n_clusters,edgecolors='k',c='green',s=100)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Noise std.dev",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Number of clusters detected",fontsize=15)
plt.show()


# ** We see that the cluster detection works well up to a certain level of noise std. dev, after which the mean of the blobs shifts to the overall centroid and the number of detected clusters tends to 1**
# 

# ### Created by Tirthajyoti Sarkar, Ph.D.
# 

# # Affinity Propagation Clustering Technique
# 
# Affinity Propagation creates clusters by sending messages between pairs of samples until convergence. **A dataset is then described using a small number of exemplars**, which are identified as those most representative of other samples. The messages sent between pairs represent the suitability for one sample to be the exemplar of the other, which is updated in response to the values from other pairs. This updating happens iteratively until convergence, at which point the final exemplars are chosen, and hence the final clustering is given.
# 
# ## The algorithm
# 
# Let $x_1$ through $x_n$ be a set of data points, with no assumptions made about their internal structure, and let $s$ be a function that quantifies the similarity between any two points, such that $s(x_i, x_j) > s(x_i, x_k) \text{ iff } x_i$ is more similar to $x_j$ than to $x_k$. For this example, the negative squared distance of two data points was used i.e. for points xi and xk, 
# ${\displaystyle s(i,k)=-\left\|x_{i}-x_{k}\right\|^{2}} {\displaystyle s(i,k)=-\left\|x_{i}-x_{k}\right\|^{2}}$
# 
# The diagonal of s i.e. ${s(i,i)}$ is particularly important, as it represents the input preference, meaning how likely a particular input is to become an exemplar. When it is set to the same value for all inputs, it controls how many classes the algorithm produces. A value close to the minimum possible similarity produces fewer classes, while a value close to or larger than the maximum possible similarity, produces many classes. It is typically initialized to the median similarity of all pairs of inputs.
# 
# The algorithm proceeds by alternating two message passing steps, to update two matrices:
# 
# * The "responsibility" matrix *$R$* has values $r(i, k)$ that quantify how well-suited $x_k$ is to serve as the exemplar for $x_i$, relative to other candidate exemplars for $x_i$.
# 
# * The "availability" matrix *$A$* contains values $a(i, k)$ that represent how "appropriate" it would be for $x_i$ to pick $x_k$ as its exemplar, taking into account other points' preference for $x_k$ as an exemplar.
# 
# Both matrices are initialized to all zeroes, and can be viewed as log-probability tables. The algorithm then performs the following updates iteratively:
# 
# First, responsibility updates are sent around: 
# $$ {\displaystyle r(i,k)\leftarrow s(i,k)-\max _{k'\neq k}\left\{a(i,k')+s(i,k')\right\}} $$
# Then, availability is updated per
# 
# $$ {\displaystyle a(i,k)\leftarrow \min \left(0,r(k,k)+\sum _{i'\not \in \{i,k\}}\max(0,r(i',k))\right)} $$ 
# for ${\displaystyle i\neq k}$ and
# $$ {\displaystyle a(k,k)\leftarrow \sum _{i'\neq k}\max(0,r(i',k))} $$
# 
# The iterations are performed until either the cluster boundaries remain unchanged over a number of iterations, or after some predetermined number of iterations. The exemplars are extracted from the final matrices as those whose 'responsibility + availability' for themselves is positive i.e.
# 
# $${\displaystyle (r(i,i)+a(i,i))>0} {\displaystyle (r(i,i)+a(i,i))>0})$$
# 
# ## Pros and cons
# 
# Affinity Propagation can be interesting as **it automatically chooses the number of clusters based on the data provided**. For this purpose, the two important parameters are the preference, which controls how many exemplars are used, and the damping factor which damps the responsibility and availability messages to avoid numerical oscillations when updating these messages.
# 
# **The main drawback of Affinity Propagation is its complexity**. The algorithm has a time complexity of the order $O(N^2 T)$, where $N$ is the number of samples and $T$ is the number of iterations until convergence. Further, the memory complexity is of the order $O(N^2)$ if a dense similarity matrix is used, but reducible if a sparse similarity matrix is used. This makes Affinity Propagation most appropriate for small to medium sized datasets.
# 

# ## Make some synthetic data
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,random_state=0)


X.shape


plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],edgecolors='k',c='orange',s=75)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# ## Clustering
# 

# Compute Affinity Propagation
af_model = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af_model.cluster_centers_indices_
labels = af_model.labels_
n_clusters_ = len(cluster_centers_indices)


# #### Number of detected clusters and their centers
# 

print("Number of clusters detected by the algorithm:", n_clusters_)


print("Cluster centers detected at:\n\n", X[cluster_centers_indices])


plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1],edgecolors='k',c=af_model.labels_,s=75)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# #### Homogeneity
# 
# Homogeneity metric of a cluster labeling given a ground truth.
# 
# A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class. This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
# 

print ("Homogeneity score:", metrics.homogeneity_score(labels_true,labels))


# #### Completeness
# 
# Completeness metric of a cluster labeling given a ground truth.
# 
# A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster. This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
# 

print("Completeness score:",metrics.completeness_score(labels_true,labels))


# #### Prediction
# 

x_new = [0.5,0.4]
x_pred = af_model.predict([x_new])[0]


print("New point ({},{}) will belong to cluster {}".format(x_new[0],x_new[1],x_pred))


x_new = [-0.5,0.4]
x_pred = af_model.predict([x_new])[0]


print("New point ({},{}) will belong to cluster {}".format(x_new[0],x_new[1],x_pred))


# ## Time complexity and model quality as the data size grows
# 

import time
from tqdm import tqdm 


n_samples = [10,20,50,100,200,500,1000,2000,3000,5000,7500,10000]
centers = [[1, 1], [-1, -1], [1, -1]]
t_aff = []
homo_aff=[]
complete_aff=[]

for i in tqdm(n_samples):
    X,labels_true = make_blobs(n_samples=i, centers=centers, cluster_std=0.5,random_state=0)
    t1 = time.time()
    af_model = AffinityPropagation(preference=-50,max_iter=50).fit(X)
    t2=time.time()
    t_aff.append(t2-t1)
    homo_aff.append(metrics.homogeneity_score(labels_true,af_model.labels_))
    complete_aff.append(metrics.completeness_score(labels_true,af_model.labels_))


plt.figure(figsize=(8,5))
plt.title("Time complexity of Affinity Propagation\n",fontsize=20)
plt.scatter(n_samples,t_aff,edgecolors='k',c='green',s=100)
plt.plot(n_samples,t_aff,'k--',lw=3)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Number of samples",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Time taken for model (sec)",fontsize=15)
plt.show()


plt.figure(figsize=(8,5))
plt.title("Homogeneity score with data set size\n",fontsize=20)
plt.scatter(n_samples,homo_aff,edgecolors='k',c='green',s=100)
plt.plot(n_samples,homo_aff,'k--',lw=3)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Number of samples",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Homogeneity score",fontsize=15)
plt.show()


plt.figure(figsize=(8,5))
plt.title("Completeness score with data set size\n",fontsize=20)
plt.scatter(n_samples,complete_aff,edgecolors='k',c='green',s=100)
plt.plot(n_samples,complete_aff,'k--',lw=3)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Number of samples",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Completeness score",fontsize=15)
plt.show()


# ## How well the cluster detection works in the presence of noise? Can damping help?
# 
# Create data sets with varying degree of noise std. dev and run the model to detect clusters. Also, play with damping parameter to see the effect.
# 

noise = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,1.75,2.0]
n_clusters = []
for i in noise:
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=200, centers=centers, cluster_std=i,random_state=101)
    af_model=AffinityPropagation(preference=-50,max_iter=500,convergence_iter=15,damping=0.5).fit(X)
    n_clusters.append(len(af_model.cluster_centers_indices_))  


print("Detected number of clusters:",n_clusters)
plt.figure(figsize=(8,5))
plt.title("Cluster detection with noisy data for low damping=0.5\n",fontsize=16)
plt.scatter(noise,n_clusters,edgecolors='k',c='green',s=100)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Noise std.dev",fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Number of clusters detected",fontsize=15)
plt.show()


noise = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,1.75,2.0]
n_clusters = []
for i in noise:
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=200, centers=centers, cluster_std=i,random_state=101)
    af_model=AffinityPropagation(preference=-50,max_iter=500,convergence_iter=15,damping=0.9).fit(X)
    n_clusters.append(len(af_model.cluster_centers_indices_))  


print("Detected number of clusters:",n_clusters)
plt.figure(figsize=(8,5))
plt.title("Cluster detection with noisy data for high damping=0.9\n",fontsize=16)
plt.scatter(noise,n_clusters,edgecolors='k',c='green',s=100)
plt.grid(True)
plt.xticks(fontsize=15)
plt.xlabel("Noise std.dev",fontsize=15)
plt.yticks([i for i in range(2,10)],fontsize=15)
plt.ylabel("Number of clusters detected",fontsize=15)
plt.show()


# ** We see that for low damping factor is not good for cluster detection as it creates oscillatory predictions. Higher damping stablizes the prediction. For low noise std. dev, the prediction is correct and for higher noise it deviates slowly.**
# 

# <h1 style="font-family:verdana;color:red">DBSCAN Clustering</h1>
# 
# The [**```DBSCAN```**](http://scikit-learn.org/stable/modules/clustering.html#dbscan) algorithm views clusters as areas of high density separated by areas of low density. 
# <p style="font-size:125%;color:blue">Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped.</p>
# 
# The central component to the DBSCAN is the concept of core samples, which are samples that are in areas of high density. A cluster is therefore a set of core samples, each close to each other (measured by some distance measure) and a set of non-core samples that are close to a core sample (but are not themselves core samples).
# 

# ## Clustering
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ### Make moon-shaped and doughnut-shaped data
# 

from sklearn import cluster, datasets


n_samples = 1500
noisy_circles,circle_labels = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons,moon_labels = datasets.make_moons(n_samples=n_samples, noise=.1)


noisy_moons=np.array(noisy_moons)
noisy_circles = np.array(noisy_circles)


plt.figure(figsize=(8,5))
plt.title("Half-moon shaped data", fontsize=18)
plt.grid(True)
plt.scatter(noisy_moons[:,0],noisy_moons[:,1])
plt.show()


plt.figure(figsize=(8,5))
plt.title("Concentric circles of data points", fontsize=18)
plt.grid(True)
plt.scatter(noisy_circles[:,0],noisy_circles[:,1])
plt.show()


# ### Can k-means identify the right clusters?
# 

km=cluster.KMeans(n_clusters=2)


km.fit(noisy_moons)


plt.figure(figsize=(8,5))
plt.title("Half-moon shaped data", fontsize=18)
plt.grid(True)
plt.scatter(noisy_moons[:,0],noisy_moons[:,1],c=km.labels_)
plt.show()


km.fit(noisy_circles)


plt.figure(figsize=(8,5))
plt.title("Concentric circles of data points", fontsize=18)
plt.grid(True)
plt.scatter(noisy_circles[:,0],noisy_circles[:,1],c=km.labels_)
plt.show()


# ### How does DBSCAN perform?
# 

dbs = cluster.DBSCAN(eps=0.1)


dbs.fit(noisy_moons)


plt.figure(figsize=(8,5))
plt.title("Half-moon shaped data", fontsize=18)
plt.grid(True)
plt.scatter(noisy_moons[:,0],noisy_moons[:,1],c=dbs.labels_)
plt.show()


dbs.fit(noisy_circles)


plt.figure(figsize=(8,5))
plt.title("Concentric circles of data points", fontsize=18)
plt.grid(True)
plt.scatter(noisy_circles[:,0],noisy_circles[:,1],c=dbs.labels_)
plt.show()


# ## Speed difference between reading numerical data from plain CSV vs. using _.npy_ file format
# 

import numpy as np
import time


n_samples=1000000


with open('fdata.txt', 'w') as fdata:
    for _ in range(n_samples):
        fdata.write(str(10*np.random.random())+',')


t1=time.time()
array_direct = np.fromfile('fdata.txt',dtype=float, count=-1,sep=',').reshape(1000,1000)
t2=time.time()
print(array_direct)
print('\nShape: ',array_direct.shape)
print(f"Time took to read: {t2-t1} seconds.")


t1=time.time()
with open('fdata.txt','r') as fdata:
    datastr=fdata.read()
lst = datastr.split(',')
lst.pop()
array_lst=np.array(lst,dtype=float).reshape(1000,1000)
t2=time.time()
print(array_lst)
print('\nShape: ',array_lst.shape)
print(f"Time took to read: {t2-t1} seconds.")


np.save('fnumpy.npy',array_lst)


t1=time.time()
array_reloaded = np.load('fnumpy.npy')
t2=time.time()
print(array_reloaded)
print('\nShape: ',array_reloaded.shape)
print(f"Time took to load: {t2-t1} seconds.")


t1=time.time()
array_reloaded = np.load('fnumpy.npy').reshape(10000,100)
t2=time.time()
print(array_reloaded)
print('\nShape: ',array_reloaded.shape)
print(f"Time took to load: {t2-t1} seconds.")


# ### Speed enhancement as the sample size grows...
# 

n_samples=[100000*i for i in range(1,11)] 
time_lst_read=[]
time_npy_read=[]

for sample_size in n_samples:
    with open('fdata.txt', 'w') as fdata:
        for _ in range(sample_size):
            fdata.write(str(10*np.random.random())+',')

    t1=time.time()
    with open('fdata.txt','r') as fdata:
        datastr=fdata.read()
    lst = datastr.split(',')
    lst.pop()
    array_lst=np.array(lst,dtype=float)
    t2=time.time()
    time_lst_read.append(1000*(t2-t1))
    print("Array shape:",array_lst.shape)

    np.save('fnumpy.npy',array_lst)

    t1=time.time()
    array_reloaded = np.load('fnumpy.npy')
    t2=time.time()
    time_npy_read.append(1000*(t2-t1))
    print("Array shape:",array_reloaded.shape)
    
    print(f"Processing done for {sample_size} samples\n")


import matplotlib.pyplot as plt


plt.figure(figsize=(8,5))
#plt.xscale('log')
#plt.yscale('log')
plt.scatter(n_samples,time_lst_read)
plt.scatter(n_samples,time_npy_read)
plt.legend(['Normal read from CSV','Read from .npy file'])
plt.show()


time_npy_read





