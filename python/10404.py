# $$ \LaTeX \text{ command declarations here.}
# \newcommand{\R}{\mathbb{R}}
# \renewcommand{\vec}[1]{\mathbf{#1}}
# $$
# 

# # EECS 545:  Machine Learning
# ## Lecture 19:  Unsupervised Learning: PCA and ICA
# * Instructor:  **Jacob Abernethy**
# * Date:  March 28, 2016
# 
# 
# *Lecture Exposition*: Saket
# 

# ## References
# 
# This lecture draws from following resources:
# 
# 

# ## Independent Component Analysis
# 
# 
# * Also called: “blind source separation”
# 
# 
# * Suppose N independent signals are mixed, and sensed by N independent sensors.
#     * Cocktail party with speakers and microphones.
#     * EEG with brain wave sources and sensors.
# 
# 
# * Can we reconstruct the original signals, given the mixed data from the sensors?

# * The sources s must be independent.
#     * And they **must be non-Gaussian**.
#     * If Gaussian, then there is no way to find unique independent components.
# 
# 
# * Linear mixing to get the sensor signals x.
#     * $x = As$
#     * or $s = Wx$ (i.e., $W = A^{-1}$ )
#     
# 
# * A is called bases; W is called filters
# 

# ## ICA: Algorithm
# 
# * There are several formulations of ICA:
#     * Maximum likelihood
#     * Maximizing non-Gaussianity (popular)
#     
# 
# * Common steps of ICA (e.g., FastICA):
#     * Apply PCA whitening (aka sphering) to the data
#     * Find orthogonal unit vectors along which the that non-Gaussianity are maximized
#     $$ \underset{W}{\\max}\hspace{1em} f(W \tilde x) \hspace{2em} s.t.\hspace{1em} WW^T = I$$
#     where f(x) can be “kurtosis”, L1 norm, etc.
# 

# ## Step 1: PCA whitening (preprocessing for ICA)
# 
# * Apply PCA: $ \Sigma = U \Lambda U^T $ 
# 
# * Project (rotate) to the principal components
# 
# * “Scale” each axis so that the transformed data has identity as covariance.
# 
# <img src="pca.jpg" align="middle">
# 

# ## Step 2: Maximization
# 
# * Rotate to maximize non-Gaussianity
# 
# <img src="ica_1.png" align="middle">
# $\hspace{6em}x \hspace{8em} x_{PCA} = U^Tx \hspace{6em} x_{PCA} = \Lambda^{-\frac{1}{2}} U^Tx \hspace{6em} x_{ICA} = V \Lambda^{-\frac{1}{2}} U^Tx$ 
# 

# ## Mixture Example
# 
# * Input Signals and Density
# <img src="ica_mix.png" align="middle">
# 

# * Remove correlations by whitening (sphering) the data.
# 
# <img src="ica_2.png" align="middle">
# 

# To whiten the input data:
# 
# * We want a linear transformation $ y = V x $
# 
# * So the components are uncorrelated: $\mathbb{E}[yy^T] = I $
# 
# * Given the original covariance $ C = \mathbb{E}[xx^T]$
# 
# * We can use $V = C^{-\frac {1}{2}} $
# 
# * Because $ \mathbb{E}[yy^T] = \mathbb{E}[Vxx^TV^T] = C^{-\frac {1}{2}} C C^{-\frac {1}{2}} = I $  
# 

# * Step 1 of FastICA
# 
# <img src="ica_3.png" align="middle">
# 

# * Step 2 of ICA
# 
# <img src="ica_4.png" align="middle">
# 

# * Step 3 of ICA
# 
# <img src="ica_5.png" align="middle">
# 

# * Step 4 of ICA
# 
# <img src="ica_6.png" align="middle">
# 

# * Step 5 of ICA: note that $p(y_1 ,y_2 ) = p(y_1 ) p(y_2 )$
# 
# <img src="ica_7.png" align="middle">
# 

# Example from scikit: Blind source separation using FastICA

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

###############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

###############################################################################
# Plot results

def plotResults():
    plt.figure()

    models = [X, S, S_, H]
    names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals', 
         'PCA recovered signals']
    colors = ['red', 'steelblue', 'orange']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


get_ipython().magic('matplotlib inline')
plotResults()


# ## ICA: summary
# 
# * Learning can be done by PCA whitening followed kurtosis maximization.
# 
# 
# * ICA is widely used for “blind-source separation.”
# 
# 
# * The ICA components can be used for features.
# 
# 
# * Limitation: difficult to learn overcomplete bases due to the orthogonality constraint
# 

# ## Principal Component Analysis
# 
# #### High Dimensional data
# 
# * $\dots$ may have low-dimensional structure.
# 
# <img src="pca_1.png" align="middle">
# 
# * The data is 100x100-dimensional.
# * But there are only three degrees of freedom, so it lies on a 3-dimensional subspace. (on a non-linear manifold, in this case)
# 

# * Given a set $X = \{x_n\}$ of observations 
#     * in a space of dimension $D$, 
#     * find a subspace of dimension $M < D$ 
#     * that captures most of its variability.
# 
# 
# * PCA can be described as either:
#     * maximizing the variance of the projection, or
#     * minimizing the squared approximation error.
# 

# ## Two Descriptions of PCA
# 
# * Approximate with the projection:
# 
#     * Maximize variance, or
# 
#     * Minimize squared error
#     
# <img src="pca_2.png" height = "300px" width = "300px"  align="middle">
# 

# ## Equivalent Descriptions
# 
# * With mean at the origin $ c_i^2 = a_i^2 + b_i^2 $
# 
# * With constant $c_i^2$
# 
# * Minimizing $b_i^2$
# 
# * Maximizes $a_i^2$
# 
# * And vice versa
# 
# <img src="pca_3.png" height = "300px" width = "300px"  align="middle">
# 

# ## First Principal Component
# 
# * Given data points $\{x_n\}$ in $D$-dim space.
# 
#     * Mean $\bar x = \frac{1}{N} \sum_{n=1}^{N} x_n $
# 
#     * Data covariance ($D \times D$ matrix): 
#      $ S = \frac{1}{N} \sum_{n=1}^{N}(x_n - \bar x)(x_n - \bar x)^T$
# 
# 
# * Let $u_1$ be the principal component we want.
#     * Length 1: $u_1^T u_1 = 1$
#     * Projection of $x_n$: $u_1^T x_n$
# 

# * Maximize the projection variance:
# 
# $$ S = \frac{1}{N} \sum_{n=1}^{N}\{u_1^Tx_n - u_1^T \bar x \}^2 = u_1^TSu_1$$
# 
# * Use a Lagrange multiplier to enforce $u_1^T u_1 = 1$
# 
# * Maximize: $u_1^T S u_1 + \lambda(1-u_1^T u_1)$
# 
# * Derivative is zero when $ Su_1 = \lambda u_1$
#     * That is, $u_1^T S u_1 = \lambda $
# 
# * So $u_1$ is eigenvector of $S$ with largest eigenvalue.
# 

# ## PCA by Maximizing Variance
# 
# * Repeat to find the M eigenvectors of the data covariance matrix S corresponding to the M largest eigenvalues.
# 
# 
# * We can do the same thing by minimizing the squared error of the projection.
# 

# ## Learning features via PCA
# 
# #### Example: Eigenfaces
# <img src="pca_9.png"  align="middle">
# 

## scikit example: Faces recognition example using eigenfaces and SVMs

from __future__ import print_function

from time import time
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

#print("Extracting the top %d eigenfaces from %d faces"
#      % (n_components, X_train.shape[0]))
#t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
#print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

#print("Projecting the input data on the eigenfaces orthonormal basis")
#t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

#print("Fitting the classifier to the training set")
#t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
#print("done in %0.3fs" % (time() - t0))
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

#print("Predicting people's names on the test set")
#t0 = time()
y_pred = clf.predict(X_test_pca)
#print("done in %0.3fs" % (time() - t0))

#print(classification_report(y_test, y_pred, target_names=target_names))
#print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]



plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


# ## Limits to PCA
# 
# * Maximizing variance is not always the best way to make the structure visible.
# 
# 
# * PCA vs Fisher’s linear discriminant
# 
# <img src="pca_6.png" height = "300px" width = "300px"  align="middle">
# 

# ## Probabilistic PCA
# 
# * We can view PCA as solving a probabilistic latent variable problem.
# 
# 
# * Describe a distribution $p(x)$ in $D$-dimensional space, in terms of a latent variable $z$ in $M$-dimensional space.
# $ x  = Wz + \mu + \epsilon $ where
# $ p(z) =  \mathcal (z| 0, I) $
# 
# 
# * W is a $D \times M$ linear transformation from $z$ to $x$ 
# 

# * Given the generative model
# $$ x  = Wz + \mu + \epsilon $$
# 
# 
# * we can infer
# 
# $$ \mathbb{E}[x]  = \mathbb{E}[Wz + \mu + \epsilon] = \mu $$
# 
# $$ \begin {align}
# cov[x]  
# &= \mathbb{E}[(Wz + \epsilon)(Wz + \epsilon)^T] \&= \mathbb{E}[(Wzz^TW^T] + \mathbb{E}[ \epsilon \epsilon^T] \&= WW^T + \sigma^2 I
# \end{align}
# $$
# 
# 
# 
# 

# * The generative model
# $$ x  = Wz + \mu + \epsilon $$
# 
# can be illustrated
# 
# <img src="pca_7.png" height = "600px" width = "600px"  align="middle">
# 
# 

# ## Likelihood of Probabilistic PCA
# 
# * (Marginal) likelihood
# $$ \begin{align}
# ln \{p(X|\mu, W, \sigma^2)\}
# &= \sum_n p(x_n| W, \mu, \sigma^2) \&= -\frac{ND}{2} ln 2\pi - \frac{N}{2} ln |C| -\frac{1}{2} \sum_n (x_n - \mu)^TC^{-1}(x_n - \mu) \C = WW^T + \sigma^2 I 
# \end{align}
# $$
# 
# 
# * We can simply maximize this likelihood function with respect to $\mu, W, \sigma$.
# 

# ## Maximum Likelihood Parameters
# 
# * Mean: $\mu = \bar x$ 
# 
# * Noise: $ \sigma_{ML}^2 = \frac{1}{D-M} \sum_{i=M+1} ^{D} \lambda_i $
# 
# * W: $W_{ML} = U_M (L_M - \sigma^2 I)^{\frac{1}{2}} R $
# 
# where $L_M$ is diag with the $M$ largest eigenvalues
# and $U_M$ is the $M$ corresponding eigenvectors
# And $R$ is an arbitrary $M\times M$ rotation (i.e., $z$ can be defined by rotating “back”)
# 

# ## Maximum likelihood by EM
# 
# * Latent variable model
# 
# $$ p(z) = \mathcal {N}(z|0, I) $$
# $$ p(x|z) = \mathcal{N}(x|Wz + \mu, \sigma^2 I) $$
# 
# 
# * E-step: Estimate the posterior $Q(z)=P(z|x)$ 
#     * Use linear Gaussian
# 
# 
# * M-step: Maximize the data-completion likelihood given $Q(z)$:
# $$ \underset{\theta = \{\mu, W, \sigma\}}{\\max}\hspace{1em}  \sum_i \sum_{z^{(i)}} Q(z^{(i)}) log P_\theta (x^{(i)}, z^{(i)}) $$ 
# 

# ## Finding PCA params by EM
# 
# <img src="pca_8.png"  align="middle">
# 

# ## Bayesian PCA (sketch)
# 
# * Note that the maximum likelihood for probabilistic PCA is still a point estimate on $W$.
# 
# * Main idea of Bayesian PCA: Put a prior on $W$
# $$ p(W|\alpha) = \prod_i (\frac{\alpha_i}{2\pi})^\frac{D}{2} exp(-\frac{1}{2} \alpha_i w_i^T w_i) $$
# 
# * Maximize the marginal likelihood (i.e., marginalize $W$)
# $$ p(X|\alpha, \mu, \sigma^2) = \int p(X|W, \mu, \sigma^2) p(W|\alpha) d\alpha $$ 
# 

# ## Kernel PCA
# 
# * Suppose the regularity that allows dimensionality reduction is non-linear.
# 
# <img src="pca_4.png" height = "300px" width = "300px"  align="middle"> <img src="pca_5.png" height = "300px" width = "300px"  align="middle">
# 

# * As with regression and classification, we can transform the raw input data {xn} to a set of feature values
# 
# $$ \{x_n\} \rightarrow \{\phi(x_n)\} $$
# 
# 
# * Linear PCA (on the nonlinear feature space) gives us a linear subspace in the feature value space, corresponding to nonlinear structure in the data space.
# 

# * Define a kernel, to avoid having to evaluate the feature vectors explicitly.
# $$ \kappa (x, x') = \phi(x)^T \phi(x') $$
# 
# 
# * Define the Gram matrix K of pairwise similarities among the data points:
# $$ K_{nm} = \phi (x_n)^T \phi(x_m) = \kappa (x_n, x_m) $$
# 
# 
# * Express PCA in terms of the kernel,
#     * Some care is required to centralize the data.
# 

# Scikit Example: Kernel PCA


# Authors: Mathieu Blondel
#          Andreas Mueller
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot results
def plotResults():
    plt.figure()
    plt.subplot(2, 2, 1, aspect='equal')
    plt.title("Original space")
    reds = y == 0
    blues = y == 1

    plt.plot(X[reds, 0], X[reds, 1], "ro")
    plt.plot(X[blues, 0], X[blues, 1], "bo")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
    # projection on the first principal component (in the phi space)
    Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
    plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

    plt.subplot(2, 2, 2, aspect='equal')
    plt.plot(X_pca[reds, 0], X_pca[reds, 1], "ro")
    plt.plot(X_pca[blues, 0], X_pca[blues, 1], "bo")
    plt.title("Projection by PCA")
    plt.xlabel("1st principal component")
    plt.ylabel("2nd component")

    plt.subplot(2, 2, 3, aspect='equal')
    plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro")
    plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")
    plt.title("Projection by KPCA")
    plt.xlabel("1st principal component in space induced by $\phi$")
    plt.ylabel("2nd component")

    plt.subplot(2, 2, 4, aspect='equal')
    plt.plot(X_back[reds, 0], X_back[reds, 1], "ro")
    plt.plot(X_back[blues, 0], X_back[blues, 1], "bo")
    plt.title("Original space after inverse transform")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

    plt.show()


plotResults()


get_ipython().magic('matplotlib inline')
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# # EECS 545:  Machine Learning
# ## Lecture 11:  Bias-Variance Tradeoff, Cross Validation, ML Advice
# * Instructor:  **Jacob Abernethy**
# * Date:  February 17, 2015
# 
# *Lecture Exposition Credit: Valli & Ben*
# 

# ### Today's Lecture: * Machine Learning Advice * 
# - How does one go about choosing and applying an ML algorithm?
# - How does one improve the performance of an ML algorithm?

# ### A question to ponder about
# #### What is the goal of Machine Learning and ML algorithms?
# 
# Common Goal:
#    - Not to learn an exact representation of the training data itself. 
#    - Build a statistical model of the process which generates the data (Statistical Inference).
#    - This is important if the algorithm is to have good generalization performance.
# 

# ### At the beginning...
# 
# - Suppose you are given some dataset and are asked to analyze it, say as a research project.
# - What is the first thing you will do once you are given this task?

from sklearn.datasets import make_classification
X, y = make_classification(1000, n_features=5, n_informative=2, 
                           n_redundant=2, n_classes=2, random_state=0)

from pandas import DataFrame
df = DataFrame(np.hstack((X, y[:, None])), 
               columns = list(range(5)) + ["class"])


df[:5]


# ### Where to start?
# 
# - Analyze the data and preprocess using simple statistical measurements and tools. Particularly maybe look for:
#  - Number of features? Number of classes? (for classification)
#  - Mean, Median, Mode?
#  - Correlation? 
#  - Dataset size? Missing samples?
#  - Samples labeled? 

# ### Visualization
# 
# - Pro: Can often be more useful than mathematical statistical analysis to get a good grasp of what the dataset looks like.
# - Con: High-dimensional data can be hard to visualize.
# 
# Some helpful visualizations for the 5-dimensional dataset follow. 
# 

# Pairwise feature plot

_ = sns.pairplot(df[:50], vars=[0, 1, 2, 3, 4], hue="class", size=1.5)


# Correlation Plot
plt.figure(figsize=(10, 10));
_ = sns.heatmap(df.corr(), annot=False)


# ### General Approaches to a Data Problem
# 
# After doing some visualization and simple statistical analysis or preprocessing of data, how should one proceed?

# ### Approach 1: Careful Design 
# - Things to do:
#  - Engineer/Select exactly the right features.
#  - Collect the right dataset.
#  - Design the right algorithms.
# - Implement and hope it works.
# 

# ### Approach 1:  Careful Design
# 
# - Pros: 
#  - Can lead to new, elegant and scalable algorithms. 
#  - Contributions to ML theory are generally done using this approach.
# - Cons: 
#  - Can be time consuming. Slow time to market for companies. 
#  - "Premature optimization is the root of all evil." - Donald Knuth (Note: while this quote was intended to talk about programming, premature statistical optimizatio can also be quite evil.)
# 

# ### Approach 2: Build and Fix
# - Implement something quickly.
# - Run error analyses and diagnoses to see if anything can be improved. Repeat until some performance criteria is met or goal is reached.
# 

# ### Approach 2:  Build and Fix
# 
# - Pros: 
#  - Easy especially with vast computing resources (can try different methods more easily). 
#  - Fast time to market. 
# - Cons: 
#  - Not systematic. 
#  - Can miss out on the reasoning behind why a model works well. 
# 

# ### Choosing a Method
# 
# - Not easy to immediately decide what to use. Some things to consider first: 
#  - Supervised vs. Unsupervised vs. Semi-supervised vs. Active Learning vs. Reinforcement Learning ...?
#  - Generative vs. Discriminative? 
#  - Parametric vs. Non-parametric?

# ### Choosing a Method
# 
# - Still wondering how to go about choosing methods from an applied viewpoint? 
#  - There are many guides (see next few slides). 
#  - Go ahead and try different algorithms! (Similar to approach 2) We will also talk about how to measure performance and deal with poor performance later.
# 

from IPython.display import Image
Image(filename='images/sklearn_sheet.png', width=800, height=600) 


from IPython.display import Image
Image(filename='images/azure_sheet.png', width=800, height=600) 


# ### Building a Statistical Model (Statistical Inference) 
# #### Estimators
# 
# - ML Algorithms can in general be thought of as "estimators."
# > **Estimator:** A statistic (a function of data) that is used to infer the value of an unknown parameter in a statistical model.
# 
# - Suppose there is a fixed parameter $f$ that needs to be estimated. An estimator of $f$ is a function that maps the sample space to a set of sample estimates, denoted $\hat{f}$.
# 

# ### Noise
# 
# - For most problems in Machine Learning, the relationship is functional but noisy.
# 
# - Mathematically, $y = f(x) + \epsilon$ where $\epsilon$ is noise with mean $0$ variance $\sigma^2$
# 

# ### A Mathematical to view the goal of Machine Learning
# 
# - Let the training set be $D = \{\mathbf{x}_1, ..., \mathbf{x}_n\}, \mathbf{x}_i \in \mathbb{R}^d$.
# - Goal: Find $\hat{f}$ that minimizes some **Loss function**, $L(y, \hat{f})$, which measures how good predictions are for **both** 
#  - Points in $D$ (the **sample**), and, 
#  - Points ***out of sample*** (outside $D$).
# - Cannot minimize both perfectly because the relationship between $y$ and $\mathbf{x}$ is noisy.
#  - ***Irreducible error***.
# 

# adapted from http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html
def plot_loss_functions():
    xmin, xmax = -4, 4
    xx = np.linspace(xmin, xmax, 100)
    plt.plot(xx, xx ** 2, 'm-',
             label="Quadratic loss")
    plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], 'k-',
             label="Zero-one loss")
    plt.plot(xx, 1/(1 + np.exp(xx)), 'b-',
             label="Sigmoid loss")
    plt.plot(xx, np.where(xx < 1, 1 - xx, 0), 'g-',
             label="Hinge loss")
    plt.plot(xx, np.log2(1 + np.exp(-xx)), 'r-',
             label="Log loss")
    plt.plot(xx, np.exp(-xx), 'c-',
             label="Exponential loss")
    plt.ylim((0, 8))
    plt.legend(loc="best")
    plt.xlabel(r"Decision function $f(x)$")
    plt.ylabel("$L(y, f)$")


# ### Loss Functions
# 
# There are many loss functions. Useful to understand what the advantages and disadvantages of each are and when to use them.
# 

# Demonstrate some loss functions
plot_loss_functions()


# ### Brief Aside on Loss Functions (1)
# 
# - 0-1 Loss
#  - Used for classification. 
#  - Not convex! 
#   - Not practical since optimization problems become intractable!
#   - "Surrogate Loss functions" that are convex and differentiable can be used instead.
# - Sigmoid Loss 
#  - Convex and Differentiable! Can be used for classification.
#  - Used in Neural Networks as an activation function (to provide nonlinearity and differentiability).
# 

# ### Brief Aside on Loss Functions (2)
# - Quadratic Loss: 
#  - Commonly used for regression.
#  - Influenced by outliers.
# - Hinge Loss
#  - Used in SVMs. 
#  - Robust to outliers.
#  - Do not provide well calibrated probabilities.
# 

# ### Brief Aside on Loss Functions (3)
# - Log-loss
#  - Used in Logistic regression.
#  - Influenced by outliers. 
#  - Provides well calibrated probabilities (can be interpreted as confidence levels).
# - Exponential Loss
#  - Used for Boosting.
#  - Very susceptible to outliers.
# - Many more: Huber Loss, Rectifiers, Hyperbolic Tangent, Perceptron Loss, Cross Entropy Loss ...
# 

# ### Risk
# 
# For a given Loss Function, we can calculate a "Risk" function that gives the expected loss or error (this is calculated in different ways depending on whether a frequentist or bayesian approach is taken). See Nikulin, M.S. (2001), Robert Christian (2007) for more details.
# 
# To simplify the explanations for the next few slides and introduce the Bias-Variance decomposition, we will consider the quadratic loss function, $L(y, \hat{f}) = (y - \hat{f})^2$, whose associated risk function is simply given by $\mathbb{E}[(y - \hat{f})^2]$.
# 
# Thus, let the expected error of an estimator $\hat{f}$ be given by $\mathbb{E}[(y - \hat{f})^2]$.
# 
# We can now go on to expand the above expression. 
# 

# ### Decomposing the expected error (1)
# 
# $\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[y^2 - 2 \cdot y \cdot \hat{f} + {\hat{f}}^2]$
# 
# By linearity of expectations, we then have: 
# 
# $\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[y^2] - \mathbb{E}[2 \cdot y \cdot \hat{f}] + \mathbb{E}[{\hat{f}}^2]$
# 

# ### Decomposing the expected error (2)
# 
# $\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[y^2] - \mathbb{E}[2 \cdot y \cdot \hat{f}] + \mathbb{E}[{\hat{f}}^2]$
# 
# Now, $Var[X] = \mathbb{E}[{X}^2] - {\mathbb{E}[X]}^2$
# 
# So, $\mathbb{E}[X^2] = Var[X] + {\mathbb{E}[X]}^2$
# 
# Thus, we have $\mathbb{E}[(y - \hat{f})^2] = Var[y] + {\mathbb{E}[y]}^2 - \mathbb{E}[2 \cdot y \cdot \hat{f}] + Var[\hat{f}] + {\mathbb{E}[{\hat{f}}]}^2$
# 

# ### Decomposing the expected error (3)
# 
# $\mathbb{E}[(y - \hat{f})^2] = Var[y] + {\mathbb{E}[y]}^2 - \mathbb{E}[2 \cdot y \cdot \hat{f} + Var[\hat{f}] + {\mathbb{E}[{\hat{f}}]}^2$
# 
# $\begin{align} \mathbb{E}[y] &= \mathbb{E}[f + \epsilon] \               &= \mathbb{E}[f] + \mathbb{E}[\epsilon] \text{ (linearity of expectations)}\               &= \mathbb{E}[f] + 0 \               &= f \text{ (} \because f \text{ is determinstic)}\end{align}$
#                
# $\begin{align} Var[y] &= \mathbb{E}[(y - \mathbb{E}[y])^2] \                      &= \mathbb{E}[(y - f)^2] \                      &= \mathbb{E}[(f + \epsilon - f)^2] \                      &= \mathbb{E}[\epsilon^2] = \sigma^2 \end{align}$
#                 
# 
# Thus, $\mathbb{E}[(y - \hat{f})^2] = \sigma^2 + f^2 - \mathbb{E}[2 \cdot y \cdot \hat{f}] + Var[\hat{f}] + {\mathbb{E}[{\hat{f}}]}^2$
# 

# ### Decomposing the expected error (4)
# 
# $\mathbb{E}[(y - \hat{f})^2] = \sigma^2 + f^2 - \mathbb{E}[2 \cdot y \cdot \hat{f}] + Var[\hat{f}] + {\mathbb{E}[{\hat{f}}]}^2$
# 
# Note that $y$ is random ***only*** in $\epsilon$ (again, $f$ is deterministic). 
# 
# Also, $\epsilon$ is ***independent*** from $\hat{f}$.
# 
# $\begin{align}\text{Hence, }  \mathbb{E}[2 \cdot y \cdot \hat{f}] 
#                       &= \mathbb{E}[2 \cdot y \cdot \hat{f}]\                      &= \mathbb{E}[2 \cdot y] \cdot \mathbb{E}[\hat{f}] \text{ (by independence) }\                      &= 2 \cdot \mathbb{E}[y] \cdot \mathbb{E}[\hat{f}] \                      &= 2 \cdot f \cdot \mathbb{E}[\hat{f}] \end{align}$
#    
# Thus, we now have $\mathbb{E}[(y - \hat{f})^2] = \sigma^2 + f^2 - 2 \cdot f \cdot \mathbb{E}[\hat{f}] + Var[\hat{f}] + {\mathbb{E}[{\hat{f}}]}^2$
# 

# ### Decomposing the expected error (5)
# 
# $\mathbb{E}[(y - \hat{f})^2] = \sigma^2 + Var[\hat{f}] + f^2 - 2 \cdot f \cdot \mathbb{E}[\hat{f}] + {\mathbb{E}[{\hat{f}}]}^2$
# 
# Now, $f^2 - 2 \cdot f \cdot \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}]^2 = (f - \mathbb{E}[\hat{f}])^2$ 
# 
# $\implies \mathbb{E}[(y - \hat{f})^2] = \sigma^2 + Var[\hat{f}] + (f - \mathbb{E}[\hat{f}])^2$
# 
# $\begin{align} \text{Finally, } \mathbb{E}[f - \hat{f}] 
#                         &= \mathbb{E}[f] - \mathbb{E}[\hat{f}] \text{ (linearity of expectations)} \                        &= f - \mathbb{E}[\hat{f}] \end{align}$
#                      
# So, $\mathbb{E}[(y - \hat{f})^2] = \sigma^2 + Var[\hat{f}] + {\mathbb{E}[f - \hat{f}]}^2$
# 

# ### Bias-Variance Decomposition
# 
# $\mathbb{E}[(y - \hat{f})^2] = \underbrace{{\sigma^2}}_\text{irreducible error} + \overbrace{{Var[\hat{f}]}}^\text{Variance} + \underbrace{{\mathbb{E}[f - \hat{f}]}^2}_{\text{Bias}^2}$
# 

# ### Bias and Variance Formulae
# 
# Bias of an estimator, $B(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$
# 
# Variance of an estimator, $Var(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]$
# 

# ### An example to explain Bias/Variance and illustrate the tradeoff 
# 
# - Consider estimating a sinusoidal function. 
# 
# (Example that follows is inspired by Yaser Abu-Mostafa's CS 156 Lecture titled "Bias-Variance Tradeoff"
# 

import pylab as pl
def plot_fit(x, y, p, show):
    xfit = np.linspace(0, 2, 1000)
    yfit = np.polyval(p, xfit)
    if show:
        pl.scatter(x, y, c='k')
        pl.plot(xfit, yfit)
        pl.hold('on')
        pl.xlabel('x')
        pl.ylabel('y')


def calc_errors(p):
    x = np.linspace(0, 2, 1000)
    errs = []
    for i in x:
        errs.append(abs(np.polyval(p, i) - np.sin(np.pi * i)) ** 2)
    return errs


def polyfit_sin(degree, iterations, num_points=2, show=True):
    total = 0
    l = []
    coeffs = []
    errs = [0] * len(np.linspace(0, 2, 1000))
    for i in range(iterations):
        np.random.seed()
        x = 2 * np.random.random(num_points) # Pick random points from the sinusoid with co-domain [0, 2)
        y = np.sin(np.pi * x)
        p = np.polyfit(x, y, degree)  
        y_poly = [np.polyval(p, x_i) for x_i in x]  
        plot_fit(x, y, p, show)
        total += sum(abs(y_poly - y) ** 2) # calculate Squared Error (Squared Error) 
        coeffs.append(p)
        errs = np.add(calc_errors(p), errs)
    return total / iterations, errs / iterations, np.mean(coeffs, axis = 0), coeffs


# Estimate two points of sin(pi * x) with a constant once
# Ignore return values for now, we will return to these later
_, _, _, _ = polyfit_sin(0, 1)


# Estimate two points of sin(pi * x) with a constant 5 times
_, _, _, _ = polyfit_sin(0, 5)


# Estimate two points of sin(pi * x) with a constant 100 times
_, _, _, _ = polyfit_sin(0, 100)


# Estimate two points of sin(pi * x) with a constant 500 times
MSE, errs, mean_coeffs, coeffs_list = polyfit_sin(0, 500)


x = np.linspace(0, 2, 1000)

# Polynomial with mean coeffs.
p = np.poly1d(mean_coeffs)

# Calculate Bias
errs_ = []
for i in x:
    errs_.append(abs(np.sin(np.pi * i) - np.polyval(p, i)) ** 2)
print("Bias: "  + str(np.mean(errs_)))


x = np.linspace(0, 2, 1000)

diffs = []

# Calculate Variance
for coeffs in coeffs_list:
    p = np.poly1d(coeffs)
    for i in x:
        diffs.append(abs(np.polyval(np.poly1d(mean_coeffs), i) - np.polyval(p, i)) ** 2)  
print("Variance: "  + str(np.mean(diffs)))


# Error Bars plot

xfit = np.linspace(0, 2, 1000)
yfit = np.polyval(np.poly1d(mean_coeffs), xfit)
pl.scatter(xfit, yfit, c='g')
pl.hold('on')
pl.plot(xfit, np.sin(np.pi * xfit))
pl.errorbar(xfit, yfit, yerr = errs, c='y', ls="None", zorder=0)
pl.xlabel('x')
pl.ylabel('y')


# Estimate two points of sin(pi * x) with a line 1 times
MSE, _, _, _ = polyfit_sin(1, 1)
print(MSE)
# Note: Perfect fit! (floating point math cause non-zero MSE)


# Estimate two points of sin(pi * x) with a line 5 times
_, _, _, _ = polyfit_sin(1, 5)


# Estimate two points of sin(pi * x) with a line 100 times
_, _, _, _ = polyfit_sin(1, 100)


# Estimate two points of sin(pi * x) with a line 500 times
MSE, errs, mean_coeffs, coeffs = polyfit_sin(1, 500)


x = np.linspace(0, 2, 1000)

# Polynomial with mean coeffs.
p = np.poly1d(mean_coeffs)

# Calculate Bias
errs_ = []
for i in x:
    errs_.append(abs(np.sin(np.pi * i) - np.polyval(p, i)) ** 2)
print("Bias: " + str(np.mean(errs_)))


x = np.linspace(0, 2, 1000)

diffs = []

# Calculate Variance
for coeff in coeffs:
    p = np.poly1d(coeff)
    for i in x:
        diffs.append(abs(np.polyval(np.poly1d(mean_coeffs), i) - np.polyval(p, i)) ** 2)  
print("Variance: "  + str(np.mean(diffs)))


# Error bars plot

xfit = np.linspace(0, 2, 1000)
yfit = np.polyval(np.poly1d(mean_coeffs), xfit)
pl.scatter(xfit, yfit, c='g')
pl.hold('on')
pl.plot(xfit, np.sin(np.pi * xfit))
pl.errorbar(xfit, yfit, yerr = errs, c='y', ls="None", zorder=0)
pl.xlabel('x')
pl.ylabel('y')


# ### Summary
# 
# - Simpler Model (Constant)
#  - High(er) Bias: $\approx 0.5$
#  - Low(er) Variance: $\approx 0.25$
# - More Complex Model (Affine)
#  - Low(er) Bias: $\approx 0.21$
#  - High(er) Variance: $\approx 1.5$ ($ \gg 0.25!)$
# 
# Moral: (According to generalization performance) a constant is a better model than a linear model for approximating a sinusoid!
# 

# ### Bias Variance Tradeoff
# 
# #### Central problem in supervised learning. 
# 
# Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. 
# 
# - High Variance: 
#  - Model represents the training set well. 
#  - Overfit to noise or unrepresentative training data. 
#  - Poor generalization performance
# 
# 
# - High Bias: 
#  - Simplistic models.
#  - Fail to capture regularities in the data.
#  - May give better generalization performance.
# 

# ### Interpretations of Bias
#  - Captures the errors caused by the simplifying assumptions of a model.
#  - Captures the average errors of a model across different training sets.
# 

# ### Interpretations of Variance
#  - Captures how much a learning method moves around the mean. 
#  - How different can one expect the hypotheses of a given model to be?
#  - How sensitive is an estimator to different training sets?

# ### Complexity of Model
# 
# - Simple models generally have high bias and complex models generally have low bias. 
# - Simple models generally have low variance andcomplex models generally have high variance.
# 
# 
# - Underfitting / Overfitting
#  - High variance is associated with overfitting.
#  - High bias is associated with underfitting.
# 

# ### Training set size
#  
# - Decreasing the training set size
#  - Helps with a high bias algorithm: 
#   - Will in general not help in improving performance. 
#   - Can attain the same performance with smaller training samples however.
#   - Additional advantage of increases in speed.
# 
# 
# - Increase the training set size
#  - Decreases Variance by reducing overfitting.
# 

# ### Number of features
# - Increasing the number of features.
#  - Decreases bias at the expense of increasing the variance.
# 
# - Decreasing the number of features.
#  - Dimensionality reduction can decrease variance by reducing over-fitting.
# 

# ### Features 
# 
# Many techniques for engineering and selecting features (Feature Engineering and Feature Extraction)
#  - PCA, Isomap, Kernel PCA, Autoencoders, Latent sematic analysis, Nonlinear dimensionality reduction, Multidimensional Scaling
#  
# 
# The importance of features
#      - "Coming up with features is difficult, time-consuming, requires expert knowledge. 
#         Applied machine learning is basically feature engineering" - Andrew Ng
#      - "... some machine learning projects succeed and some fail. 
#         What makes the difference? Easily the most important factor is the features used." - Pedro Domingo
# 

# ### Regularization (Changing $\lambda$)
# 
# Regularization is designed to impose simplicity by adding a penalty term that depends on the charactistics of the parameters.
# 
# - Decrease Regularization. 
#  - Reduces bias (allows the model to be more complex).
#  
#  
# - Increase Regularization.
#  - Reduces variance by reducing overfitting (again, regularization imposes "simplicity.") 
# 

# ### Ideal bias and variance?
# 
# - All is not lost. Bias and Variance can both be lowered through some methods:
#  - Ex: Boosting (learning from weak classifiers).
# 
# - The sweet spot for a model is the level of complexity at which the increase in bias is equivalent to the reduction in variance. 
# 

# ### Model Selection 
# 
# - ML Algorithms generally have a lot of parameters that must be chosen. A natural question is then "How do we choose them?"
#  - Examples: Penalty for margin violation (C), Polynomial Degree in polynomial fitting
# 
# - Simple Idea: 
#  - Construct models $M_i, i = 1, ..., n$.
#  - Train each of the models to get a hypothesis $h_i, i = 1, ..., n$.
#  - Choose the best.
# - Does this work? No! Overfitting. This brings us to cross validation.
# 

# ### Hold-Out Cross Validation 
# 
# (1) Randomly split the training data $D$ into $D_{train}$ and $D_{val}$, say 70% of the data and 30% of the data respectively.
# 
# (2) Train each model $M_i$ on $D_{train}$ only, each time getting a hypothesis $h_i$.
# 
# (3) Select and output hypothesis $h_i$ that had the smallest error on the held out validation set.
# 
# Disadvantages: 
#  - Waste some sizable amount of data (30\% in the above scenario) so that less training examples are available.
#  - Using only some data for training and other data for validation.
# 

# ### K-Fold Cross Validation (Step 1)
# 
# Randomly split the training data $D$ into $K$ ***disjoint*** subsets of $N/K$ training samples each.
#  - Let these subsets be denoted $D_1, ..., D_K$.
# 

# ### K-Fold Cross Validation (Step 2)
# 
# For each model $M_i$, we evaluate the model as follows: 
#  - Train the model $M_i$ on $D \setminus D_k$ (all of the subsets except subset $D_k$) to get hypothesis $h_i(k)$.
#  - Test the hypothesis $h_i(k)$ on $D_k$ to get the error (or loss) $\epsilon_i(k)$.
#  - Estimated generalization error for model $M_i$ is then given by $e^g_i = \frac{1}{K} \sum \limits_{k = 1}^K \epsilon_i (k)$
# 

# ### K-Fold Cross Validation (Step 3)
# 
# Pick the model $M_i^*$ with the lowest estimated generalization error $e^{g*}_i$ and retrain the model on the entire training set, thus giving the final hypothesis $h^*$ that is output.
# 

# ### Three Way Data Splits
# 
# - If model selection and true error estimates are to be computed simaltaneously, the data needs to be divided into three disjoin sets.
# 
# - Training set: A set of examples used for learning
# - Validation set: A set of examples used to tune the hyperparameters of a classifier.
# - Test Set: A set of examples used *** only *** to assess the performance of a fully-trained model.
# 

# ### Procedure Outline
# 
# 1. Divide the available data into training, validation and test set
# 2. Select a model (and hyperparameters)
# 3. Train the model using the training set
# 4. Evaluate the model using the validation set
# 5. Repeat steps 2 through 4 using different models (and hyperparameters)
# 6. Select the best model (and hyperparameter) and train it using data from the training and validation set
# 7. Assess this final model using the test set
# 

# ### How to choose hyperparameters?
# 
# Cross Validation is only useful if we have some number of models. This often means constructing models each with a different combination of hyperparameters.
# 

# ### Random Search
#  - Just choose each hyperparameter randomly (possibly within some range for each.)
#  - Pro: Easy to implement. Viable for models with a small number of hyperparameters and/or low dimensional data.
#  - Con: Very inefficient for models with a large number of hyperparameters or high dimensional data (curse of dimensionality.)
# 

# ### Grid Search / Parameter Sweep
#  - Choose a subset for each of the parameters.
#   - Discretize real valued parameters with step sizes as necessary.
#  - Output the model with the best cross validation performance. 
#  - Pro: "Embarassingly Parallel" (Can be easily parallelized)
#  - Con: Again, curse of dimensionality poses problems.
# 

# ### Bayesian Optimization
#  
# - Assumes that there is a smooth but noisy relation that acts as a mapping from hyperparameters to the objective function.
# 
# - Gather observations in such a manner as to evaluate the machine learning model the least number of times while revealing as much information as possible about the mapping and, in particular, the location of the optimum.
# 
# - Exploration vs. Exploitation problem.
# 

# ### Learning Curves
# Provide a visualization for diagnostics such as:
# - Bias / variance
# - Convergence 
# 

# Image from Andrew Ng's Stanford CS229 lecture titled "Advice for applying machine learning"
from IPython.display import Image
Image(filename='images/HighVariance.png', width=800, height=600)

# Testing error still decreasing as the training set size increases. Suggests increasing the training set size.
# Large gap Between Training and Test Error.


# Image from Andrew Ng's Stanford CS229 lecture titled "Advice for applying machine learning"
from IPython.display import Image
Image(filename='images/HighBias.png', width=800, height=600)

# Training error is unacceptably high.
# Small gap between training error and testing error.


# ### Convergence
# 
# - Approach 1: 
#  - Measure gradient of the learning curve.
#  - As learning curve gradient approaches 0, the model has been trained. Choose threshold to stop training.
# 
# - Approach 2: 
#  - Measure change in the model parameters each iteration of the algorithm.
#  - One can assume that training is complete when the change in model parameters is below some threshold.
# 

# ### Diagnostics related to Convergence (1)
# - Convergence too slow? 
#  - Try using Newton's method.
#  - Larger step size. 
#   - Note that too large of a step size could also lead to slow convergence (but the learning curves in general will then suggest instability if "oscillations" are occuring.)
#  - Decrease batch size if using a batch based optimization algorithm.
# 

# ### Diagnostics related to Convergence (2)
# 
# - Are the learning curves stable? If not: 
#  - Switch to a batch style optimization algorithm if not already using one (like minibatch gradient descent / gradient descent).
#  - Increase batch sizes if already using one.
# - Some algorithms always ensure a decrease or increase in the objective function each iterations. Ensure that this is the case if the optimization algorithm being used provides such guarantees.
# 

# ### Ablative Analysis
# 
# - Similar to the idea of cross validation, except for components of a system.
# 
# - Example: Simple Logisitic Regression on spam classification gives 94% performance.
#  - 95% with spell correction
#  - 96% with top 100 most commonly used words removed
#  - 98% with extra sender and receiver information 
#  - 99% overall performance
# 

