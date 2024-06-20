# # K-Nearest Neighbor Classifier
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import sklearn
import matplotlib
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd), ('Scipy', scipy), ('Sklearn', sklearn))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


def get_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target


import math
import numpy as np
import copy
import collections

class knn_classifier:
    
    def __init__(self, n_neighbors=5):
        """
        KNearestNeighbors is a distance based classifier that returns
        predictions based on the nearest points in the feature space.
        ---
        In: n_neighbors (int) - how many closest neighbors do we consider
        """
        if n_neighbors > 0:
            self.k = int(n_neighbors)
        else:
            print("n_neighbors must be >0. Set to 5!")
            self.k = 5
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        """
        Makes a copy of the training data that can live within the class.
        Thus, the model can be serialized and used away from the original
        training data. 
        ---
        In: X (features), y (labels); both np.array or pandas dataframe/series
        """
        self.X = copy.copy(self.pandas_to_numpy(X))
        self.y = copy.copy(self.pandas_to_numpy(y))
        
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)
    
    def predict(self, X):
        """
        Iterates through all points to predict, calculating the distance
        to all of the training points. It then passes that to a sorting function
        which returns the most common vote of the n_neighbors (k) closest training
        points.
        ___
        In: new data to predict (np.array, pandas series/dataframe)
        Out: predictions (np.array)
        """
        X = self.pandas_to_numpy(X)
        results = []
        for x in X:
            local_results = []
            for (x2,y) in zip(self.X,self.y):
                local_results.append([self.dist_between_points(x,x2),y])
            results.append(self.get_final_predict(local_results))
        return np.array(results).reshape(-1,1)
            
    def get_final_predict(self,results):
        """
        Takes a list of [distance, label] pairs and sorts by distance,
        returning the mode vote for the n_neighbors (k) closest votes. 
        ---
        In: [[distance, label]] list of lists
        Output: class label (int)
        """
        results = sorted(results, key=lambda x: x[0])
        dists, votes = zip(*results)
        return collections.Counter(votes[:self.k]).most_common(1)[0][0]

    def dist_between_points(self, a, b):
        """
        Calculates the distance between two vectors.
        ---
        Inputs: a,b (np.arrays)
        Outputs: distance (float)"""
        assert np.array(a).shape == np.array(b).shape
        return np.sqrt(np.sum((a-b)**2))
    
    def score(self, X, y):
        """
        Uses the predict method to measure the accuracy of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: accuracy (float)
        """
        pred = self.predict(X)
        correct = 0
        for i,j in zip(y,pred):
            if i == j:
                correct+=1
        return float(correct)/float(len(y))


X,y = get_data()


def shuffle_data(X, y):
    assert len(X) == len(y)
    permute = np.random.permutation(len(y))
    return X[permute], y[permute]

def train_test_split_manual(X, y, test_size=0.3):
    nX, ny = shuffle_data(X,y)
    split_index = int(len(X)*test_size)
    testX = nX[:split_index]
    trainX = nX[split_index:]
    testy = ny[:split_index]
    trainy = ny[split_index:]
    return trainX, testX, trainy, testy


x_train, x_test, y_train, y_test = train_test_split_manual(X,y,test_size=0.3)


knn = knn_classifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)


from sklearn.neighbors import KNeighborsClassifier

sk_knn = KNeighborsClassifier(n_neighbors=5)
sk_knn.fit(x_train,y_train)
sk_knn.score(x_test,y_test)


myscore = []
skscore = []
for k in range(1,55)[::2]:
    knn = knn_classifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    myscore.append(knn.score(x_test,y_test))
    sk_knn = KNeighborsClassifier(n_neighbors=k)
    sk_knn.fit(x_train,y_train)
    skscore.append(sk_knn.score(x_test,y_test))
    
plt.plot(range(1,55)[::2],myscore,'r-',lw=4,label="kNN Scratch")
plt.plot(range(1,55)[::2],skscore,'-k',label="kNN SkLearn")
plt.legend(loc="lower left", fontsize=16);





# # Random Forest (an extension on the Decision Tree Class)
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


import sys 
sys.path.append('../modules')
from decision_tree_classifier import decision_tree_classifier
import collections
import pandas as pd
import numpy as np

class random_forest_classifier:
    
    def __init__(self, n_trees = 10, max_depth=None, n_features='sqrt', mode='rfnode', seed=None):
        """
        Random Forest Classifier uses bootstrapping and column randomization
        to generate n_trees different datasets and then applies a decision 
        tree to each dataset. The final prediction is an ensemble of all created trees.
        ---
        Params:
        n_trees (int): number of bootstrapped trees to grow for ensembling
        max_depth (int): maximum number of splits to make in each tree)
        n_features: The number of columns to include in the models. 
                    Options: numeric value (e.g. 4 => 4 columns used)
                             "sqrt" (square root of the number of cols in input data)
                             "div3" (number of input cols divided by 3)
        mode: If mode='rfnode' the column randomization happens at each node. Otherwise
              Each tree gets one randomized set of columns for all nodes in that tree.
        seed: Random seed to allow for reproducibility.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.tree_filter_pairs = []
        self.mode = mode
        if seed:
            self._seed = seed
            np.random.seed(seed)
        
    def find_number_of_columns(self, X):
        """
        Uses the user input for n_features to decide how many columns should
        be included in each model. Uses the shape of X to decide the final number
        if 'sqrt' is called. 
        ---
        Input: X (array, dataframe, or series)
        """
        if isinstance(self.n_features, int):
            return self.n_features
        if self.n_features == 'sqrt':
            return int(np.sqrt(X.shape[1])+0.5)
        if self.n_features == 'div3':
            return int(X.shape[1]/3+0.5)
        else:
            raise ValueError("Invalid n_features selection")
    
    def get_bagged_data(self, X, y):
        """
        Chooses random rows to populate a bootstrapped dataset, with replacement.
        Maintains the correlation between X and y
        ---
        Input: X, y (arrays)
        Outputs: randomized X,y (arrays)
        """
        index = np.random.choice(np.arange(len(X)),len(X))
        return X[index], y[index]
    
    def randomize_columns(self,X):
        """
        Chooses a set of columns to keep from the input data. These are
        randomly drawn, according the number requested by the user. The data
        is filtered and only the allowed columns are returned, along with the
        filter. 
        ---
        Input: X (array)
        Output: filtered_X (array), filter (array)
        """
        num_col = self.find_number_of_columns(X)
        filt = np.random.choice(np.arange(0,X.shape[1]),num_col,replace=False)
        filtered_X = self.apply_filter(X, filt)
        return filtered_X, filt
    
    def apply_filter(self, X, filt):
        """
        Given X and a filter, only the columns matching the index values
        in filter are returned.
        ---
        Input: X (array), filter (array of column IDs)
        Output: filtered_X (array)
        """
        filtered_X = X.T[filt]
        return filtered_X.T
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)
    
    def fit(self, X, y):
        """
        Generates the bootstrapped data, decides which column to keep,
        and then uses the decision tree class to build a model on each 
        bootstrapped and column-randomized dataset. Each tree is stored 
        as part of the model for later use, along with the appropriate
        filter - which is needed to filter new data for use with the model.
        ---
        Input: X, y (arrays, dataframe, or series)
        """
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        try:
            self.base_filt = [x for x in range(X.shape[1])]
        except IndexError:
            self.base_filt = [0]
        for _ in range(self.n_trees):
            filt = self.base_filt
            bagX, bagy = self.get_bagged_data(X,y)
            if self.mode == 'rftree':
                bagX, filt = self.randomize_columns(bagX)
            new_tree = decision_tree_classifier(self.max_depth, mode=self.mode, n_features=self.n_features)
            new_tree.fit(bagX, bagy)
            self.tree_filter_pairs.append((new_tree, filt))
    
    def predict(self, X):
        """
        Uses the list of tree models built in the fit, doing a predict with each
        model. The associated filter is applied to X, so the model sees the columns
        it has learned about. The final prediction uses the mode of all the trees 
        predictions.
        ---
        Input: X (array, dataframe, or series)
        Output: Class ID (int)
        """
        X = self.pandas_to_numpy(X)
        self.predicts = []
        for tree, filt in self.tree_filter_pairs:
            filtered_X = self.apply_filter(X, filt)
            self.predicts.append(tree.predict(filtered_X))
        self.pred_by_row = np.array(self.predicts).T
        
        ensemble_predict = []
        for row in self.pred_by_row:
            ensemble_predict.append(collections.Counter(row).most_common(1)[0][0])
        return ensemble_predict
    
    def score(self, X, y):
        """
        Uses the predict method to measure the accuracy of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: accuracy (float)
        """
        pred = self.predict(X)
        correct = 0
        for i,j in zip(y,pred):
            if i == j:
                correct+=1
        return float(correct)/float(len(y))


# # Let's try it out with the Iris dataset
# 

def get_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target


X,y = get_data()


from data_splitting import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


rf = random_forest_classifier(n_trees=25, n_features='sqrt', mode='rfnode', seed=42)
rf.fit(X_train, y_train)


preds = rf.predict(X_test)
for i,j in zip(preds[10:40:2], rf.pred_by_row[10:40:2]):
    print(j,i)


# Make sure each node is getting randomized properly in rfnode mode.
# 

for tr in rf.tree_filter_pairs:
    print(tr[0].tree.filt)        


rf.score(X_test,y_test)


accs = []
for n in range(1,100,5):
    rf = random_forest_classifier(n_trees=n, mode='rfnode')
    rf.fit(X_train, y_train)
    accs.append(rf.score(X_test, y_test))


plt.plot(range(1,100,5),accs,'r')
plt.xlabel("Num. Trees")
plt.ylabel("Accuracy Score")
plt.title("Accuracy vs Num Trees (Mean Acc: %.3f)"%round(np.mean(accs),3));


# # Now let's play with some more complicated data
# 

from sklearn.datasets import load_wine
X = load_wine().data
y = load_wine().target


from data_splitting import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


rf = random_forest_classifier(n_trees=10, mode='rfnode')
rf.fit(X_train, y_train)
rf.score(X_test, y_test)


from sklearn.dummy import DummyClassifier
dc = DummyClassifier()
dc.fit(X_train, y_train)
dc.score(X_test, y_test)


accs = []
for n in range(1,100,5):
    rf = random_forest_classifier(n_trees=n, mode='rfnode', seed=42)
    rf.fit(X_train, y_train)
    accs.append(rf.score(X_test, y_test))


plt.plot(range(1,100,5),accs,'r')
plt.xlabel("Num. Trees")
plt.ylabel("Accuracy Score")
plt.title("Accuracy vs Num Trees (Mean Acc: %.3f)"%round(np.mean(accs),3));


# # Now let's look at a few trees to see whether they're all being built the same way.
# 

X,y = get_data()
from data_splitting import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


rf = random_forest_classifier(n_trees=5, n_features='sqrt', mode='rfnode')
rf.fit(X_train, y_train)


for i, tree in enumerate(rf.tree_filter_pairs):
    print("--- Tree %i ---"%i,"\n")
    tree[0].print_tree()
    print("\n\n")





# # Spectral Clustering 
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


import numpy as np
import pandas as pd
import sys
sys.path.append('../../modules')
from kmeans import kmeans

class spectral_clustering:
    
    def __init__(self, k=3, connectivity=20, svd_dims=3, affinity='neighbors', bandwidth=1.):
        self.k = k
        self.connect = connectivity
        self.dims = svd_dims
        if affinity in ['neighbors', 'rbf']:
            self.affinity_type = affinity
        else:
            print("Not a valid affinity type, default to 'neighbors'.")
            self.affinity_type = 'neighbors'
        self.bandwidth = bandwidth
    
    def rbf_kernel(self, x1, x2, sig=1.):
        """
        Returns the rbf affinity between two points (x1 and x2),
        for a given bandwidth (standard deviation).
        ---
        Inputs: 
            x1; point 1(array)
            x2; point 2(array)
            sig; standard deviation (float)
        """
        diff = np.sqrt(np.sum((x1-x2)**2))
        norm = 1/(np.sqrt(2*np.pi*sig**2))
        return norm*np.exp(-diff**2/(2*sig**2))
    
    def compute_distance_between_all_points(self, pt1, pts, connectivity=None):
        """
        Returns the distance between points. Currently only uses Euclidean distance.
        ---
        Input: data point, all data points (np arrays)
        Output: Distance (float)
        """
        if self.affinity_type == 'neighbors':
            x = np.sqrt(np.sum((pt1 - pts)**2, axis=1))
            idxs = x.argsort()[:connectivity]
            filt = np.ones(len(x), dtype=bool)
            filt[idxs] = False
            x[filt] = 0.
            x[~filt] = 1.
        elif self.affinity_type == 'rbf':
            x = []
            for p in pts:
                x.append(self.rbf_kernel(pt1, p, sig=self.bandwidth))
        return x
    
    def fit(self, X):
        X = self.pandas_to_numpy(X)
        self.original_data = np.copy(X)
        self.similarity = np.array([self.compute_distance_between_all_points(p,X, connectivity=self.connect) for p in X])
        self.similarity /= max(self.similarity.ravel())
        self.U, self.Sigma, self.VT = self.do_svd(self.similarity)
        self.kmeans = kmeans(k=self.k)
        self.kmeans.fit(self.U)
        
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
        
    def transform_to_svd_space(self,X):
        sig_inv = np.linalg.inv(self.Sigma)
        return np.dot(np.dot(X,self.U),sig_inv)
    
    def predict(self, X):
        X = self.pandas_to_numpy(X)
        sim_space = [self.compute_distance_between_all_points(p,self.original_data, connectivity=self.connect) for p in X]
        transformed_X = np.array([self.transform_to_svd_space(x) for x in sim_space])
        return self.kmeans.predict(transformed_X)
    
    def do_svd(self, similarity):
        dims = self.dims
        U, Sigma, VT = np.linalg.svd(similarity)
        VT = VT[:dims,:]
        U = U[:,:dims]
        Sigma = np.diag(Sigma[:dims])
        return U, Sigma, VT
        
    def plot_similarity_matrix(self):
        plt.figure(dpi=200)
        plt.imshow(self.similarity, cmap=plt.cm.Blues)
        plt.xlabel("Point ID", fontsize=16)
        plt.ylabel("Point ID", fontsize=16)
        plt.title("Similarity Matrix (1 for neighbors, 0 for not)", fontsize=16);
        plt.colorbar(cmap=plt.cm.Blues);
        
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)


def get_data(n_clust = 3):
    X1 = np.random.normal(-5,1,50).reshape(-1,1)
    y1 = np.random.normal(-5,1,50).reshape(-1,1)
    for _ in range(n_clust-1):
        X2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        y2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        X1 = np.vstack((X1,X2)).reshape(-1,1)
        y1 = np.vstack((y1,y2)).reshape(-1,1)
    X = np.hstack((X1,y1))
    return X

X = get_data(n_clust=5)
#np.random.shuffle(X)
plt.scatter(X[:,0],X[:,1]);


sc = spectral_clustering(k=5)
preds = sc.fit_predict(X)


plt.scatter(X[:,0],X[:,1],c=preds);


# Let's make sure this is working on new data. For my dataset (this will change if you re-run), the cluster around (-5,-5) is labeled as cluster 1. Let's see if a new point at (-5,-5) is put into cluster 1 correctly.
# 

for x, y in zip(X[:10],preds[:10]):
    print("Point at: ", x)
    print("Cluster Num: ", y)
print("---")
print("At [-5, -5], should be same cluster.")
print("Prediction is: ",sc.predict([[-5,-5]]))


sc.plot_similarity_matrix()


# ## Now let's play with some circular data that KMeans can't handle
# 

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=400, factor=0.5, random_state=0, noise=0.05)
plt.scatter(X[:,0],X[:,1]);


sc = spectral_clustering(k=2, affinity='neighbors')
sc.fit(X)


sc.plot_similarity_matrix()


preds = sc.predict(X)
plt.scatter(X[:,0],X[:,1],c=preds);


# ## Now let's play with the RBF Kernel - UNDER CONSTRUCTION
# 

sc = spectral_clustering(k=2, affinity='rbf', bandwidth=0.3, svd_dims=15)
sc.fit(X)


sc.plot_similarity_matrix()


preds = sc.predict(X)
plt.scatter(X[:,0],X[:,1],c=preds);





# # Gaussian Naive Bayes
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


import pandas as pd
import numpy as np
from collections import defaultdict

class gaussian_naive_bayes:
    
    def __init__(self):
        """
        Gaussian Naive Bayes builds it's understanding of the data by
        applying Bayes rule and calculating the conditional probability of
        being a class based on a probabilistic understanding of how the 
        class has behaved before. We will assume each feature is normally
        distributed in its own space, then use a gaussian PDF to calculate
        the probability of a class based on behavior. 
        """
        self._prob_by_class = defaultdict(float)
        self._cond_means = defaultdict(lambda: defaultdict(float))
        self._cond_std = defaultdict(lambda: defaultdict(float))
        self._log_prob_by_class = defaultdict(float)
        self._data_cols = None
        
    def gaus(self, x, mu=0, sig=1):
        """
        Returns the probability of x given the mean and standard
        deviation provided - assuming a Gaussian probability.
        ---
        Inputs: x (the value to find the probability for, float),
        mu (the mean value of the feature in the training data, float),
        sig (the standard deviation of the feature in the training data, float)
        Outputs: probability (float)
        """
        norm = 1/(np.sqrt(2*np.pi*sig**2))
        return norm*np.exp(-(x-mu)**2/(2*sig**2))
    
    def fit(self, X, y):
        """
        For each class, we find out what percentage of the data is that class.
        We then filter the data so only the rows that are that class remain,
        and then go column by column - calculating the mean and standard dev
        for the values of that column, given the class. We store all of these
        values to be used later for predictions.
        ---
        Input: X, data (array/DataFrame)
        y, targets (array/Series)
        """
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        if not self._data_cols:
            try: 
                self._data_cols = X.shape[1]
            except IndexError:
                self._data_cols = 1
        X = self.check_feature_shape(X)
        
        self._classes = np.unique(y)
        
        for cl in self._classes:
            self._prob_by_class[cl] = len(y[y == cl])/len(y)
            self._log_prob_by_class[cl] = np.log(self._prob_by_class[cl])
            filt = (y == cl)
            filtered_data = X[filt]
            for col in range(self._data_cols):
                self._cond_means[cl][col] = np.mean(filtered_data.T[col])
                self._cond_std[cl][col] = np.std(filtered_data.T[col])
                
    def predict(self, X):
        """
        Wrapper to return only the class of the prediction
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict")
    
    def predict_proba(self, X):
        """
        Wrapper to return probability of each class of the prediction
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict_proba")
    
    def predict_log_proba(self, X):
        """
        Wrapper to return log of the probability of each class of 
        the prediction.
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict_log_proba")
    
    def _predict(self, X, mode="predict"):
        """
        For each data point, we go through and calculate the probability
        of it being each class. We do so by sampling the probability of
        seeing each value per feature, then combining them together with 
        the class probability. We work in the log space to fight against
        combining too many really small or large values and under/over 
        flowing Python's memory capabilities for a float. Depending on the mode
        we return either the prediction, the probabilities for each class,
        or the log of the probabilities for each class.
        ---
        Inputs: X, data (array/DataFrame)
        mode: type of prediction to return, defaults to single prediction mode
        """
        X = self.pandas_to_numpy(X)
        X = self.check_feature_shape(X)
        results = []
        for row in X:
            beliefs = []
            for cl in self._classes:
                prob_for_class = self._log_prob_by_class[cl]
                for col in range(self._data_cols):
                    if self._cond_std[cl][col]:
                        p = self.gaus(row[col],mu=self._cond_means[cl][col],sig=self._cond_std[cl][col])
                        logp = np.log(p)
                        prob_for_class += logp
                beliefs.append([cl, prob_for_class])
            
            if mode == "predict_log_proba":
                _, log_probs = zip(*beliefs)
                results.append(log_probs)
            
            elif mode == "predict_proba":
                _, probs = zip(*beliefs)
                unlog_probs = np.exp(probs)
                normed_probs = unlog_probs/np.sum(unlog_probs)
                results.append(normed_probs)
            
            else:
                sort_beliefs = sorted(beliefs, key=lambda x: x[1], reverse=True)
                results.append(sort_beliefs[0][0])
        
        return results
    
    def score(self, X, y):
        """
        Uses the predict method to measure the accuracy of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: accuracy (float)
        """
        pred = self.predict(X)
        correct = 0
        for i,j in zip(y,pred):
            if i == j:
                correct+=1
        return float(correct)/float(len(y))
      
    def check_feature_shape(self, X):
        """
        Helper function to make sure any new data conforms to the fit data shape
        ---
        In: numpy array, (unknown shape)
        Out: numpy array, shape: (rows, self.data_cols)"""
        return X.reshape(-1,self._data_cols)
            
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return np.array(x)
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)


# ### Let's Test!
# 
# We'll start by loading the iris dataset. We're trying to guess flower species by length of flower petals and sepals.
# 

from sklearn.datasets import load_iris
X, y = load_iris().data, load_iris().target


nb = gaussian_naive_bayes()
nb.fit(X,y)


# Let's take a look at the mean value for each column, given each type of species (0, 1, 2).
# 

nb._cond_means


# Now let's look at how we're predicting probability wise.
# 

nb.predict_proba(X[0:2])


nb.predict_log_proba(X[0:2])


# We get a score of 96%, the same as SkLearn!
# 

nb.score(X,y)


from sklearn.naive_bayes import GaussianNB

nb_sk = GaussianNB()
nb_sk.fit(X,y)
nb_sk.score(X,y)


# Let's visualize all of our probability distributions for each feature, given each class. We can see that in terms of Petal Length, it's very unlikely a Setosa will ever have >2 cm, a Versicoulor us unlikely to be outside the range of 3-6 cm, and the Virginica is unlikely to be outside the 4-7cm range. So if we see a petal length of 7 cm, we know it's very likely to be a Virginica! That's the intution that Naive Bayes is built upon.
# 

gaus = nb.gaus
means = nb._cond_means
std = nb._cond_std
X = np.linspace(-2,10,100)
fig, ax = plt.subplots(3,4, figsize=(16,10))
for cl in nb._classes:
    for col in range(nb._data_cols):
        ax[cl][col].plot(X,gaus(X,mu=means[cl][col],sig=std[cl][col]), lw=3)
        ax[cl][col].grid(True)
        
cols = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
rows = ['Setosa','Versicolour','Virginica']

fig.suptitle('Probability Distributions by Class and Feature', fontsize=18, fontweight='bold', y=1.04)

for aa, col in zip(ax[0], cols):
    aa.set_title(col, fontsize=16)

for aa, row in zip(ax[:,0], rows):
    aa.set_ylabel(row, rotation=90, fontsize=16)

fig.tight_layout()


# ## Now Let's try with MNIST
# 
# MNIST is a common dataset where numbers are hand written and we're trying to "read" the digit.
# 

from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target

shuffle = np.random.permutation(range(len(y)))
X = X[shuffle]
y = y[shuffle]
X_train = X[:-100]
y_train = y[:-100]
X_test = X[-100:]
y_test = y[-100:]


# Let's split up our data and make sure we have all the different numbers in our training set.
# 

plt.hist(y_train);


# Let's take a look at a digit. We see that we're measuring the darkness of each pixel. We'll be treating those as our features, with each pixel getting its own column. We see the first two pixels are empty, and the numerics are 0's. Then we see that we have an "8" and that's a fairly grey pixel, the "12" next to it is much darker. So we'll be building our intution based on "are these pixels normally dark for a 3? Then it's likely to be a 3."
# 

plt.imshow(X[2].reshape(8,8))
print(X[2].reshape(8,8))
plt.grid(False)


nb = gaussian_naive_bayes()
nb.fit(X_train,y_train)


# We are 88% accurate on these digits!
# 

nb.score(X_test,y_test)


fig, ax = plt.subplots(2,4,figsize=(16,8))
preds = []
np.random.seed(42)
for i,x in enumerate(np.random.choice(range(X.shape[0]),size=8)):
    I = i//4
    J = i%4
    ax[I][J].imshow(X[x].reshape(8,8))
    ax[I][J].grid(False)
    preds.append(nb.predict(X[x]))
print("Predictions: ",preds)





# # Ridge - L2 Regularized Regression (An SGD Wrapper)
# 

import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


import numpy as np
import pandas as pd
import sys
sys.path.append('../../modules')
from sgd_regressor import sgd_regressor

class ridge_regressor(sgd_regressor):
    
    def __init__(self, n_iter=100, alpha=0.01, verbose=False, return_steps=False, fit_intercept=True, 
                 dynamic=True, loss='ols', epsilon=0.1, lamb=1e-6, l1_perc = 0.5):
        """
        Ridge Regressor - This is a wrapper on the SGD class where the regularization is set
        to the L2 Norm. All other functionality is the same as the SGD class.
        ---
        KWargs:
        
        n_iter: number of epochs to run in while fitting to the data. Total number of steps
        will be n_iter*X.shape[0]. 
        
        alpha: The learning rate. Moderates the step size during the gradient descent algorithm.
        
        verbose: Whether to print out coefficient information during the epochs
        
        return_steps: If True, fit returns a list of the coefficients at each update step for diagnostics
        
        fit_intercept: If True, an extra coefficient is added with no associated feature to act as the
                       base prediction if all X are 0.
                       
        dynamic: If true, an annealing scedule is used to scale the learning rate. 
        
        lamb: Stands for lambda. Sets the strength of the regularization. Large lambda causes large
              regression. If regularization is off, this does not apply to anything.
              
        l1_perc: If using elastic net, this variable sets what portion of the penalty is L1 vs L2. 
                 If regularize='EN' and l1_perc = 1, equivalent to regularize='L1'. If 
                 regularize='EN' and l1_perc = 0, equivalent to regulzarize='L2'.
        """
        self.coef_ = None
        self.trained = False
        self.n_iter = n_iter
        self.alpha_ = alpha
        self.verbosity = verbose
        self._return_steps = return_steps
        self._fit_intercept = fit_intercept
        self._next_alpha_shift = 0.1 # Only used if dynamic=True
        self._dynamic = dynamic
        self._regularize = 'L2'
        self._lamb = lamb
        self._l1_perc = l1_perc


# # Let's Gen some data to see how it behaves
# 

def gen_data(rows = 200, gen_coefs = [2,4], gen_inter = 0):
    X = np.random.rand(rows,len(gen_coefs))
    y = np.sum(np.tile(np.array(gen_coefs),(X.shape[0],1))*X,axis=1)
    y = y + np.random.normal(0,0.5, size=X.shape[0])
    y = y + gen_inter
    return X, y

actual_coefs = [10,8,9,10,11]
X, y = gen_data(gen_coefs=actual_coefs[1:], gen_inter=actual_coefs[0])


# It will work with Pandas or Numpy arrays. Let's play with Pandas for now.
# 

import pandas as pd
cols = []
for i in range(X.shape[1]):
    cols.append('X'+str(i))
data = pd.DataFrame(X, columns=cols)
data['y'] = y
data.head()


ridge = ridge_regressor(n_iter=500, alpha=1e-3, verbose=False, dynamic=False, return_steps=True, lamb=1e-6)


steps = ridge.fit(data.iloc[:,:-1],data.iloc[:,-1])


ridge.coef_


test_X, test_y = gen_data(rows=200, gen_coefs=actual_coefs[1:], gen_inter=actual_coefs[0])
pred_y = sgd.predict(test_X)
test_err = pred_y - test_y


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

sns.distplot(test_err);


from scipy.stats import normaltest
print(normaltest(test_err))


plt.scatter(test_y, pred_y, s=50)
temp = np.linspace(min(test_y),max(test_y),100)
plt.plot(temp,temp,'r-')
plt.xlabel("True y")
plt.ylabel("Predicted y");


# ## Let's look at how the optimization looks in the Coefficient Space!
# 

def plot_beta_space(steps, components = (0,1), last_300=False, zoom=False):
    plt.figure(figsize=(20,16))
    try:
        B0 = np.array(steps).T[components[0]]
        B1 = np.array(steps).T[components[1]]
    except:
        print("Couldn't find those components, defaulting to (0,1)")
        B0 = np.array(steps).T[0]
        B1 = np.array(steps).T[1]
    if last_300:
        steps_to_show=-300
        skip = 2
        plt.scatter(B0[steps_to_show::skip],B1[steps_to_show::skip],c=plt.cm.rainbow(np.linspace(0,1,len(B0[steps_to_show::skip]))));
        plt.scatter(steps[steps_to_show][0],steps[steps_to_show][1],c='r',marker='x', s=400,label='Start')
        plt.scatter(steps[-1][0],steps[-1][1],c='k',marker='x', s=400,label='End')
        plt.title("Movement in the Coefficient Space, Last "+str(-steps_to_show)+" steps!",fontsize=32);
    else: 
        plt.scatter(B0[::25],B1[::25],c=plt.cm.rainbow(np.linspace(0,1,len(B0[::25]))));
        plt.scatter(steps[0][0],steps[0][1],c='r',marker='x', s=400,label='Start')
        plt.scatter(steps[-1][0],steps[-1][1],c='k',marker='x', s=400,label='End')
        plt.title("Movement in the Coefficient Space",fontsize=32);
    plt.legend(fontsize=32, loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#333333');
    plt.xlabel("B"+str(components[0]),fontsize=26)
    plt.ylabel("B"+str(components[1]),fontsize=26);
    if zoom:
        plt.ylim(min(B1[steps_to_show::skip]), max(B1[steps_to_show::skip]))
        plt.xlim(min(B0[steps_to_show::skip]), max(B0[steps_to_show::skip]));


plot_beta_space(steps)


# ** Now let's look at the last 300 steps. NOTE THE SCALE CHANGE! **
# 

plot_beta_space(steps, last_300=True)


from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')

def plot_beta_space3D(steps, components = (0,1)):
    def cost_function(x,y):
        return (x-actual_coefs[components[0]])**2 + (y-actual_coefs[components[1]])**2
    
    plot_vals_x = []
    plot_vals_y = []
    plot_vals_z = []
    for b1 in np.linspace(0,20,100):
        for b2 in np.linspace(0,20,100):
            cost = cost_function(b1,b2)
            plot_vals_x.append(b1)
            plot_vals_y.append(b2)
            plot_vals_z.append(cost)
    
    try:
        B0 = np.array(steps).T[components[0]]
        B1 = np.array(steps).T[components[1]]
    except:
        print("Couldn't find those components, defaulting to (0,1)")
        B0 = np.array(steps).T[0]
        B1 = np.array(steps).T[1]
    
    Z = cost_function(B0, B1)+10
    fig = plt.figure(figsize=(20,16))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(plot_vals_x,plot_vals_y,plot_vals_z, cmap=plt.cm.Blues, linewidth=0.2, alpha=0.4)
    ax.scatter(B0[::5],B1[::5],Z[::5],c='k',s=150);
    ax.set_xlabel("B0", fontsize=20, labelpad=20)
    ax.set_ylabel("B1", fontsize=20, labelpad=20)
    ax.set_zlabel("Cost Function Value", fontsize=20, labelpad=20);
    return ax

ax = plot_beta_space3D(steps)
ax.view_init(25, 75)


# # Now let's test with some "real data"
# 

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


X = pd.DataFrame(StandardScaler().fit_transform(load_boston().data))
y = pd.DataFrame(load_boston().target)


X.describe()


# Simplest form - fit intercept, no dynamic learning for comparison
ridge = ridge_regressor(n_iter=1000, fit_intercept=True, lamb=1e-6)
ridge.fit(X.iloc[:100],y[:100])


ridge.coef_


pred = ridge.predict(X.iloc[100:])
plt.scatter(y.iloc[100:], pred, s=50, alpha=0.5)
temp = np.linspace(min(test_y),max(test_y),100)
plt.plot(temp,temp,'r-')
plt.xlabel("True y")
plt.ylabel("Predicted y");





# # Bagging Classifier (an extension on the Decision Tree Class)
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


import sys 
sys.path.append('../modules')
from decision_tree_classifier import decision_tree_classifier
import collections
import pandas as pd
import numpy as np

class bagging_classifier:
    
    def __init__(self, n_trees = 10, max_depth=None):
        """
        Bagging Classifier uses bootstrapping to generate n_trees different
        datasets and then applies a decision tree to each dataset. The final 
        prediction is an ensemble of all created trees.
        ---
        Params:
        n_trees (int): number of bootstrapped trees to grow for ensembling
        max_depth (int): maximum number of splits to make in each tree)
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def get_bagged_data(self, X, y):
        """
        Chooses random rows to populate a bootstrapped dataset, with replacement.
        Maintains the correlation between X and y
        ---
        Input: X, y (arrays)
        Outputs: randomized X,y (arrays)
        """
        index = np.random.choice(np.arange(len(X)),len(X))
        return X[index], y[index]
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)
    
    def fit(self, X, y):
        """
        Generates the bootstrapped data then uses the decision tree
        class to build a model on each bootstrapped dataset. Each tree
        is stored as part of the model for later use.
        ---
        Input: X, y (arrays, dataframe, or series)
        """
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        for _ in range(self.n_trees):
            bagX, bagy = self.get_bagged_data(X,y)
            new_tree = decision_tree_classifier(self.max_depth)
            new_tree.fit(bagX, bagy)
            self.trees.append(new_tree)
            
    def predict(self, X):
        """
        Uses the list of tree models built in the fit, doing a predict with each
        model. The final prediction uses the mode of all the trees predictions.
        ---
        Input: X (array, dataframe, or series)
        Output: Class ID (int)
        """
        X = self.pandas_to_numpy(X)
        self.predicts = []
        for tree in self.trees:
            self.predicts.append(tree.predict(X))
        self.pred_by_row = np.array(self.predicts).T
        
        ensemble_predict = []
        for row in self.pred_by_row:
            ensemble_predict.append(collections.Counter(row).most_common(1)[0][0])
        return ensemble_predict
    
    def score(self, X, y):
        """
        Uses the predict method to measure the accuracy of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: accuracy (float)
        """
        pred = self.predict(X)
        correct = 0
        for i,j in zip(y,pred):
            if i == j:
                correct+=1
        return float(correct)/float(len(y))


# # Let's try it out with the Iris dataset
# 

def get_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target


X,y = get_data()


from data_splitting import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


bc = bagging_classifier(n_trees=100)
bc.fit(X_train, y_train)


preds = bc.predict(X_test)
for i,j in zip(preds[10:40:2], bc.pred_by_row[10:40:2]):
    print(j,i)


bc.score(X_test,y_test)


accs = []
for n in range(1,100,5):
    bc = bagging_classifier(n_trees=n)
    bc.fit(X_train, y_train)
    accs.append(bc.score(X_test, y_test))


plt.plot(range(1,100,5),accs,'r')
plt.xlabel("Num. Trees")
plt.ylabel("Accuracy Score")
plt.title("Accuracy vs Num Trees (Mean Acc: %.3f)"%round(np.mean(accs),3));


# # Now let's play with some more complicated data
# 

from sklearn.datasets import load_wine
X = load_wine().data
y = load_wine().target


from data_splitting import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


bc = bagging_classifier(n_trees=100)
bc.fit(X_train, y_train)
bc.score(X_test, y_test)


from sklearn.dummy import DummyClassifier
dc = DummyClassifier()
dc.fit(X_train, y_train)
dc.score(X_test, y_test)


accs = []
for n in range(1,100,5):
    bc = bagging_classifier(n_trees=n)
    bc.fit(X_train, y_train)
    accs.append(bc.score(X_test, y_test))


plt.plot(range(1,100,5),accs,'r');
plt.xlabel("Num. Trees")
plt.ylabel("Accuracy Score");
plt.title("Accuracy vs Num Trees (Mean Acc: %.3f)"%round(np.mean(accs),3));





# # Let's build Linear Regression from Scratch
# 

# Imports and Version Checking
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


# ## The actual class for linear_regression.
# 

import numpy as np

class linear_regression:
    
    def __init__(self, w_intercept=True):
        self.coef_ = None
        self.intercept = w_intercept
        self.is_fit = False
        
    def add_intercept(self,X):
        """
        Adds an 'all 1's' bias term to function as the y-intercept
        """
        if type(X) == type(np.array([5])):
            rows = X.shape[0]
        else:
            X = np.array([[X]])
            rows = 1
        inter = np.ones(rows).reshape(-1,1)
        return np.hstack((X,inter))
        
    def fit(self, X, y):
        """
        Read in X (all features) and y (target) and use the Linear Algebra solution
        to extract the coefficients for Linear Regression.
        """
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        if self.intercept:
            X = self.add_intercept(X)
        temp_xtx = np.linalg.inv(np.dot(X.T,X))
        temp_xty = np.dot(X.T,y)
        self.coef_ = np.dot(temp_xtx,temp_xty)
        self.is_fit = True
    
    def predict(self,X):
        """
        Takes in a new X value (that must be the same shape as the original X for fitting)
        and returns the predicted y value, using the coefficients from fitting.
        """
        if not self.is_fit:
            raise ValueError("You have to run the 'fit' method before using predict!")
        if type(X) == type([5]):
            X = np.array(X)
        if type(X) == type(5) or type(X) == type(5.):
            X = np.array([X])
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.intercept:
            X = self.add_intercept(X)
        return np.dot(X,self.coef_)[0][0]
    
    def score(self, X, true):
        """
        Takes in X, y pairs and measures the performance of the model.
        Returns negative mean squared error.
        ---
        Inputs: X, y (features, labels; np.arrays)
        Outputs: Negative Mean Square Error (float)
        """
        pred = self.predict(X)
        mse = np.mean(np.square(true-pred))
        return -mse


def gen_data(coef=3.5, intercept=5., num_points=100):
    X = np.random.uniform(0,10,num_points)
    y = coef*X + np.random.normal(0,1.5,100) + intercept
    return X,y

X,y = gen_data()
lr = linear_regression(w_intercept=True)
lr.fit(X,y)
lr.coef_


import seaborn as sns
model_vals = []
for val in np.linspace(0,10,100):
    model_vals.append(float(lr.predict(val)))


plt.scatter(X,y,s=75)
plt.plot(np.linspace(0,10,100),model_vals,'r-',lw=4)
plt.xlabel('X')
plt.ylabel('y')
plt.title("y vs X");


X_test, y_test = gen_data()
lr.score(X_test, y_test)


import sys
sys.path.append('../modules')
from stats_regress import *
pred = lr.predict(X_test)
test_model_results(X_test, y_test, pred)


# ## Let's see it in action with X^2
# 

X = np.linspace(0,10,100)
y = 2.5*X*X + np.random.normal(0,1.5,100) + 5.

lr = linear_regression(w_intercept=True)
lr.fit(X*X,y)
lr.coef_


lr.predict(0)


import seaborn as sns
model_vals = []
for val in np.linspace(0,10,100):
    model_vals.append(float(lr.predict(val*val)))


plt.scatter(X,y,s=75)
plt.plot(np.linspace(0,10,100),model_vals,'r-',lw=4)
plt.xlabel('X')
plt.ylabel('y')
plt.title("y vs X");


# ## Check Error handling for fit-predict
# 

linear_regression = linear_regression()
linear_regression.predict([1])





# # Let's build a Markov Chain text generator
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import sklearn
import matplotlib
import sys
import scipy

libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd), ('Scipy', scipy), ('Sklearn', sklearn))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


# The main idea is that we pull word groups (ngrams) and figure out what word comes after each group of words. We store those "after" words and their relationship to our "key" words. Once we have that, we can randomly draw from our "after" words every time we get a new key. Then we chain that over and over to build whole setences and phrases. We'll use the dictionary "markov_keys" to store the keys and after words... then we will sample from that.
# 

import numpy as np

class markov_chain:
    
    def __init__(self, text, from_file=True, ngram=2, random_state=None):
        """
        Markov Chains are great for generating text based on previously seen text. 
        Here we'll either read from file or from one big string, then generate a 
        probabilistic understanding of the document by using ngrams as keys and
        storing all possible following words. We can then generate sentences
        using random dice and this object.
        ---
        Inputs
            text: either the path to a file containing the text or the text (string)
            from_file: whether the text is in a file or note (bool)
            ngram: how many words to use as a key for the text generation
            random_state: used to set the random state for reproducibility
        """
        self.ngram = int(ngram)
        self.markov_keys = dict()
        self._from_file = from_file
        if type(text) != type("string"):
            raise TypeError("'text' must be a PATH or string object")
        if from_file:
            self.path = text
        else:
            self.raw = text
        self.text_as_list = None
        if random_state:
            np.random.seed(random_state)
        self.create_probability_object()

    def preprocess(self):
        """
        Opens and cleans the text to be learned. If self.from_file, it reads
        from the path provided. The cleaning is very minor, just lowercasing
        and getting rid of quotes. Creates a list of words from the text.
        """
        if self._from_file:
            with open(self.path,'r') as f:
                self.raw = f.read()
        self.text_as_list = self.raw.lower().replace('"','').replace("'","").split()

    def markov_group_generator(self,text_as_list):
        """
        Generator that creates the ngram groupings to act as keys.
        Just grabs ngram number of words and puts them into a tuple
        and yields that upon iteration request.
        ---
        Inputs
            text_as_list: the text after preprocessing (list)
        Outputs
            keys: word groupings of length self.ngram (tuple)
        """
        if len(text_as_list) < self.ngram+1:
            raise ValueError("NOT A LONG ENOUGH TEXT!")
            return

        for i in range(self.ngram,len(text_as_list)):
            yield tuple(text_as_list[i-self.ngram:i+1])

    def create_probability_object(self):
        """
        Steps through the text, pulling keys out and keeping track
        of which words follow the keys. Duplication is allowed for 
        values for each key - but all keys are unique.
        """
        if self.markov_keys:
            print("Probability Object already built!")
            return
        if not self.text_as_list:
            self.preprocess()
        for group in self.markov_group_generator(self.text_as_list):
            word_key = tuple(group[:-1])
            if word_key in self.markov_keys:
                self.markov_keys[word_key].append(group[-1])
            else:
                self.markov_keys[word_key] = [group[-1]]
    
    def generate_sentence(self, length=25, starting_word_id=None):
        """
        Given a seed word, pulls the key associated with that word and 
        samples from the values available. Then moves to the newly generated 
        word and gets the key associated with it, and generates again. 
        Repeats until the sentence is 'length' words long.
        ---
        Inputs
            length: how many words to generate (int)
            starting_word_id: what word to use as seed, by location (int)
        Outputs
            gen_words: the generated sentence, including seed words (string)
        """
        if not self.markov_keys:
            raise ValueError("No probability object built. Check initialization!")
        
        if (not starting_word_id or type(starting_word_id) != type(int(1)) 
            or starting_word_id < 0 or starting_word_id > len(self.text_as_list)-self.ngram):
            starting_word_id = np.random.randint(0,len(self.text_as_list)-self.ngram)
            
        gen_words = self.text_as_list[starting_word_id:starting_word_id+self.ngram]
        
        while len(gen_words) < length:
            seed = tuple(gen_words[-self.ngram:])
            gen_words.append(np.random.choice(self.markov_keys[seed]))
        return ' '.join(gen_words)
    
    def print_key_value_pairs(self, num_keys=20):
        """
        Iterates through the probability object, printing key-value
        pairs. 
        ---
        Input
        num_keys: how many pairs to show (int)
        """
        i = 1
        for key,value in self.markov_keys.items():
            print(key,value)
            print()
            i+=1
            if i>int(num_keys):
                break


# ## Read from H.P. Lovecraft text found on Project Gutenberg
# 

MC = markov_chain('../data/lovecraft.txt',ngram=2)


MC.print_key_value_pairs()


# We asked for ngrams = 2, so our key will be two words. Then we'll see what words are "allowed" to come after it based on what we learned from the text. So in our text, we see the phrase "the nameless" many times. Most of the times we see it, it's followed by "city" so you can see that we have lots of "city" stored as possible words to use next.
# 

MC.print_key_value_pairs(num_keys=1)


print(MC.generate_sentence(length=100, starting_word_id=25))


# ## Test reading from a string directly
# 

test_text = '''the moon, and all conjectures about the new york police detective named thomas f. malone, 
now on maenalus, pan sighs and stretches in his monstrous labours. the following day, though the localities 
were over what seemed to have lost their memory, they said, been considerable discussion about the mindless 
demon-sultan azathoth? just before two oclock, and through the eternal fishing. that fishing paid less and 
less reluctant to discuss. at least to possess the most decadent of communities. all this is not, however, 
wholly displace the exultation. he stopped for the gowned, slippered old man in the world, i think,'''

MC = markov_chain(test_text, from_file=False, ngram=1)


MC.print_key_value_pairs(num_keys=10)


MC.generate_sentence()





# # Bernoulli Naive Bayes
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


import pandas as pd
import numpy as np
from collections import defaultdict

class bernoulli_naive_bayes:
    
    def __init__(self, smoothing = 1.):
        """
        Bernoulli Naive Bayes builds it's understanding of the data by
        applying Bayes rule and calculating the conditional probability of
        being a class based on a probabilistic understanding of how the 
        class has behaved before. We only care if a feature is zero or non-zero
        in this style of naive bayes and will calculate our conditional probabilities
        accordingly. 
        ---
        Inputs:
        smoothing: the Laplace smoothing factor overcome the problem of multiplying
        a 0 probability, that causes the total probability to be 0.
        """
        self._prob_by_class = defaultdict(float)
        self._cond_probs = defaultdict(lambda: defaultdict(float))
        self._log_prob_by_class = defaultdict(float)
        self._log_cond_probs = defaultdict(lambda: defaultdict(float))
        self._data_cols = None
        self._smoothing = smoothing
    
    def fit(self, X, y):
        """
        For each class, we find out what percentage of the data is that class.
        We then filter the data so only the rows that are that class remain,
        and then go column by column - calculating what percentage of rows are
        non-zero, given the class. We store all of these values to be used later 
        for predictions. We also store the log of these values for later prediction.
        ---
        Input: X, data (array/DataFrame)
        y, targets (array/Series)
        """
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        if not self._data_cols:
            try: 
                self._data_cols = X.shape[1]
            except IndexError:
                self._data_cols = 1
        X = self.check_feature_shape(X)
        self._classes = np.unique(y)
        
        for cl in self._classes:
            self._prob_by_class[cl] = len(y[y == cl])/len(y)
            self._log_prob_by_class[cl] = np.log(self._prob_by_class[cl])
            denom = len(y[y == cl])
            filt = (y == cl)
            filtered_data = X[filt]
            for col in range(self._data_cols):
                binarized_column = filtered_data.T[col] > 0
                num_ones = np.sum(binarized_column)
                #smoothing applied here so we never get a zero probability
                self._cond_probs[cl][col] = (num_ones+self._smoothing)/(denom+self._smoothing) 
                self._log_cond_probs[cl][col] = np.log(self._cond_probs[cl][col])
                
    def predict(self, X):
        """
        Wrapper to return only the class of the prediction
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict")
    
    def predict_proba(self, X):
        """
        Wrapper to return probability of each class of the prediction
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict_proba")
    
    def predict_log_proba(self, X):
        """
        Wrapper to return log of the probability of each class of 
        the prediction.
        ---
        Input: X, data (array/dataframe)
        """
        return self._predict(X, mode="predict_log_proba")
    
    def _predict(self, X, mode="predict"):
        """
        For each data point, we go through and calculate the probability
        of it being each class. We do so by using the probability of
        seeing each value per feature, then combining them together with 
        the class probability. We work in the log space to fight against
        combining too many really small or large values and under/over 
        flowing Python's memory capabilities for a float. Depending on the mode
        we return either the prediction, the probabilities for each class,
        or the log of the probabilities for each class.
        ---
        Inputs: X, data (array/DataFrame)
        mode: type of prediction to return, defaults to single prediction mode
        """
        X = self.pandas_to_numpy(X)
        X = self.check_feature_shape(X)
        X = (X > 0).astype(int) # convert to 1 or 0
        results = []
        for row in X:
            beliefs = []
            for cl in self._classes:
                prob_for_class = self._log_prob_by_class[cl]
                for col in range(self._data_cols):
                    p = self._log_cond_probs[cl][col]
                    # The row or (1-row) chooses either the 0 or 1 probability
                    # based on whether our row is a 0 or 1.
                    prob_for_class += p*row[col] + (1-p)*(1-row[col])
                beliefs.append([cl, prob_for_class])
            
            if mode == "predict_log_proba":
                _, log_probs = zip(*beliefs)
                results.append(log_probs)
            
            elif mode == "predict_proba":
                _, probs = zip(*beliefs)
                unlog_probs = np.exp(probs)
                normed_probs = unlog_probs/np.sum(unlog_probs)
                results.append(normed_probs)
            
            else:
                sort_beliefs = sorted(beliefs, key=lambda x: x[1], reverse=True)
                results.append(sort_beliefs[0][0])
        
        return results
    
    def score(self, X, y):
        """
        Uses the predict method to measure the accuracy of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: accuracy (float)
        """
        pred = self.predict(X)
        correct = 0
        for i,j in zip(y,pred):
            if i == j:
                correct+=1
        return float(correct)/float(len(y))
      
    def check_feature_shape(self, X):
        """
        Helper function to make sure any new data conforms to the fit data shape
        ---
        In: numpy array, (unknown shape)
        Out: numpy array, shape: (rows, self.data_cols)"""
        return X.reshape(-1,self._data_cols)
            
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return np.array(x)
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)


# ### Let's test it!
# 
# Let's generate some data to test with. We'll use the example of senators voting on 4 different issues (only 3 of which are relevant) and then trying to predict which party the senator is from.
# 

def get_data():
    votes = [0,1]
    senators = np.random.choice(votes, replace=True, size=(100,4))
    df = pd.DataFrame(senators, columns=['vote1','vote2','vote3','vote4'])
    
    def calculate_party(row):
        x = row['vote1']
        y = row['vote2']
        z = row['vote3']

        party = 0.7*x + 0.5*y - z + np.random.normal(0,0.3)
        if party > 0.1:
            return 'Dem'
        elif party > 0.01:
            return 'Ind'
        else:
            return 'Rep'
    
    df['party'] = df.apply(calculate_party,axis=1)
    print(df.party.value_counts())
    return df.iloc[:,:-1],df.iloc[:,-1]
    


X, y = get_data()


nb = bernoulli_naive_bayes()
nb.fit(X.iloc[:90],y.iloc[:90])


# Let's look at the probability of voting YES on each issue by what party the senators in our training data were.
# 

nb._cond_probs


# Now we can predict!
# 

nb.predict(X.iloc[0:2])


nb.predict_proba(X.iloc[0:2])


nb.predict_log_proba(X.iloc[0:2])


# We have an accuracy of 90%, which is the same as SkLearn's accuracy!
# 

nb.score(X.iloc[90:],y.iloc[90:])


from sklearn.naive_bayes import BernoulliNB

nb_sk = BernoulliNB()
nb_sk.fit(X.iloc[:90],y.iloc[:90])
nb_sk.score(X.iloc[90:],y.iloc[90:])


# Let's visualize the vote probability by party - by looking at this we can see which YES votes tend to indicate Democrat, Independent, or Republican. Our model is just learning that - in this data sample at least - that Democrats are much more likely to vote yes on the first issue than Republicans. The same is true for Independents, but since they are such a small part of our sample, we aren't likely to guess Independent unless we're VERY sure.
# 

probs = np.zeros((3,4))
for cl, d in nb._cond_probs.items():
    for val in d.items():
        if cl == 'Dem':
            i=0
        if cl == 'Ind':
            i=1
        if cl == 'Rep':
            i=2
        probs[i][val[0]] = val[1]


plt.style.use('seaborn')
fig_plot = plt.imshow(probs, cmap='Blues', interpolation='nearest')
plt.grid(False)
plt.xticks()
ax = plt.gca()
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['vote 1','vote 2','vote 3','vote 4'], fontsize=14)
ax.set_yticks([0,1,2])
ax.set_yticklabels(['Dem','Ind','Rep'], fontsize=14);
fig = plt.gcf()
cbar = fig.colorbar(fig_plot, ticks=[0, 0.5, 1]);
plt.title("Probability of Vote by Party (Training Data)", fontsize=16);





# ## Train-Test Split From Scratch
# 

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))


def train_test_split(X, y, test_size=0.3):
    """
    Takes in features and labels and returns X_train, X_test, y_train, and y_test
    ----
    In: X (features), y (labels), test_size (percentage of data to go into test)
    Out: X_train, X_test, y_train, and y_test
    """
    X = np.array(X)
    y = np.array(y)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    splitter = np.random.choice([0,1],size=y.shape,p=[1-test_size,test_size])
    for x,y,z in zip(X,y,splitter):
        if z == 0:
            X_train.append(x)
            y_train.append(y)
        else:
            X_test.append(x)
            y_test.append(y)
    return X_train, X_test, y_train, y_test


from sklearn.datasets import load_boston
X = load_boston().data
y = load_boston().target


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


len(X_test)/(len(X_train)+len(X_test))


def plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=0,title="Checking Train-Test Split"):
    x_plot = []
    x_plot_test = []
    for j in X_train:
        x_plot.append(j[feature_num])
    for j in X_test:
        x_plot_test.append(j[feature_num])

    plt.figure(figsize=(8,6))
    plt.scatter(x_plot, y_train, c='b')
    plt.scatter(x_plot_test, y_test, c='r')
    plt.xlabel("Feature " + str(feature_num))
    plt.ylabel("Y");
    plt.title(title);


plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=12)


# # Cross Validation from Scratch
# 

class cross_val:
    
    def __init__(self, show_plot=False, feat_num=0):
        """
        The Cross-Val object contains several objects that the user may want to 
        use later, including a final copy of the best model.
        ---
        Params:
        show_plot: should it plot the data showing the splits
        feat_num: if show_plot, which feature should be used (by column num)
        best_model: the model with the lowest MSE is kept for later usage
        """
        self.show_plot = show_plot
        self.feat_num = feat_num
        self.best_model = None
        self.best_model_score = None
        
    def plot_single_feature_vs_label(self, X_train, X_test, y_train, y_test, feature_num=0, 
                                     title="Checking Train-Test Split"):
        """
        This helper method is to make plots of the data being split 
        with one feature vs the target label, showing each fold for 
        visual inspection of the splits. 
        """
        x_plot = []
        x_plot_test = []
        for j in X_train:
            x_plot.append(j[feature_num])
        for j in X_test:
            x_plot_test.append(j[feature_num])

        plt.figure(figsize=(8,6))
        plt.scatter(x_plot, y_train, c='b')
        plt.scatter(x_plot_test, y_test, c='r')
        plt.xlabel("Feature " + str(feature_num))
        plt.ylabel("Y");
        plt.title(title);

    def plot_coefs(self):
        """
        This method shows the coefficient values for each fold in a plot.
        If there are 10 coefficient, there will be 10 plots. If there were 3
        folds, each plot will contain 3 points.
        """
        if not self.coefs:
            print("Either your model doesn't have coefficients, or you")
            print("must run cross_validation_scores first!")
            return            
        for coef in range(len(self.coefs[0])):
            plot_x = []
            plot_y = []
            i=1
            for fold in self.coefs:
                plot_x.append(i)
                plot_y.append(fold[coef])
                i+=1
            plt.figure(figsize=(10,8))
            plt.plot(plot_x,plot_y)
            plt.plot(plot_x,[np.mean(plot_y)]*len(plot_x),'r--')
            plt.ylabel("coef "+str(coef))
            plt.xlabel("Fold ID")
            plt.xticks([x for x in range(1,FOLDS+1)])
            plt.title("Variation of Coefficient Across Folds")
        
    def cross_validation_scores(self, model, X, y, k=5, random_seed=42):
        """
        Splits the dataset into k folds by randomly assigning each row a
        fold ID. Afterwards, k different models are built with each fold being
        left out once and used for testing the model performance.
        ---
        Inputs:
        model: must be a class object with fit/predict methods. 
        X: feature matrix (array)
        y: labels (array)
        k: number of folds to create and use
        random_seed: sets the random number generator seed for reproducibility
        """
        X = np.array(X)
        y = np.array(y)
        self.score_folds = []
        coefs = []
        fold_nums = [x for x in range(k)]
        np.random.seed(random_seed)
        splitter = np.random.choice(fold_nums,size=y.shape)
        best_score = None
        for fold in fold_nums:
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for x2,y2,z2 in zip(X,y,splitter):
                if z2 == fold:
                    X_test.append(x2)
                    y_test.append(y2)
                else:
                    X_train.append(x2)
                    y_train.append(y2)
            model.fit(X_train,y_train)
            current_score = model.score(X_test, y_test)
            self.score_folds.append(current_score)
            if not best_score or current_score > best_score:
                best_score = current_score
                self.best_model = model
                self.best_model_score = current_score
            if model.coef_.any():
                coefs.append(model.coef_)
            if self.show_plot:
                plot_title = "CV Fold " + str(fold)
                plot_single_feature_vs_label(X_train, X_test, y_train, y_test, feature_num=self.feat_num, 
                                             title=plot_title)
        if coefs:
            self.coefs = coefs     
        
    def print_report(self):
        """
        After the CV has been run, this method will print some summary statistics
        as well as the coefficients from the model.
        """
        print("Mean Score: ", np.mean(self.score_folds))
        print("Score by fold: ", self.score_folds)
        if self.coefs:
            print("Coefs (by fold): ")
            for i,c in enumerate(self.coefs):
                print("Fold ",i,": ",c)


from sklearn.datasets import load_boston
X = load_boston().data
y = load_boston().target


import sys 
sys.path.append('../modules')
from OLS import OLS
cv = cross_val()
FOLDS = 10
cv.cross_validation_scores(OLS(w_intercept=True), X, y, k=FOLDS)


cv.coefs


cv.plot_coefs()


cv.print_report()


cv.best_model.predict(X[0].reshape(-1,13))


print(cv.best_model)
print(cv.best_model_score)


cv = cross_val(show_plot=True, feat_num=12)
cv.cross_validation_scores(OLS(), X, y, k=3)





