# # Chapter 3 - Modeling and prediction
# 

get_ipython().magic('pylab inline')


# ### The Titanic dataset
# 
# We use the Pandas library to import the Titanic survival dataset.
# 

import pandas
data = pandas.read_csv("data/titanic.csv")
data[:5]


# We make a 80/20% train/test split of the data
data_train = data[:int(0.8*len(data))]
data_test = data[int(0.8*len(data)):]


# ### Preparing the data
# 

# The categorical-to-numerical function from chapter 2
# Changed to automatically add column names
def cat_to_num(data):
    categories = unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["%s=%s" % (data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)


def prepare_data(data):
    """Takes a dataframe of raw data and returns ML model features
    """
    
    # Initially, we build a model only on the available numerical values
    features = data.drop(["PassengerId", "Survived", "Fare", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
    
    # Setting missing age values to -1
    features["Age"] = data["Age"].fillna(-1)
    
    # Adding the sqrt of the fare feature
    features["sqrt_Fare"] = sqrt(data["Fare"])
    
    # Adding gender categorical value
    features = features.join( cat_to_num(data['Sex']) )
    
    # Adding Embarked categorical value
    features = features.join( cat_to_num(data['Embarked']) )
    
    return features


# ### Building a logistic regression classifier with Scikit-Learn
# 

#cat_to_num(data['Sex'])
features = prepare_data(data_train)
features[:5]


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features, data_train["Survived"])


# Make predictions
model.predict(prepare_data(data_test))


# The accuracy of the model on the test data
# (this will be introduced in more details in chapter 4)
model.score(prepare_data(data_test), data_test["Survived"])


# ### Non-linear model with Support Vector Machines
# 

from sklearn.svm import SVC
model = SVC()
model.fit(features, data_train["Survived"])


model.score(prepare_data(data_test), data_test["Survived"])


# ### Classification with multiple classes: hand-written digits
# 
# We use the popular non-linear multi-class K-nearest neighbor algorithm to predict hand-written digits from the MNIST dataset.
# 

mnist = pandas.read_csv("data/mnist_small.csv")
mnist_train = mnist[:int(0.8*len(mnist))]
mnist_test = mnist[int(0.8*len(mnist)):]


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(mnist_train.drop("label", axis=1), mnist_train['label'])


preds = knn.predict_proba(mnist_test.drop("label", axis=1))
pandas.DataFrame(preds[:5], index=["Digit %d"%(i+1) for i in range(5)])


knn.score(mnist_test.drop("label", axis=1), mnist_test['label'])


# ### Predicting numerical values with a regression model
# 
# We use the the Linear Regression algorithm to predict miles-per-gallon of various automobiles.
# 

auto = pandas.read_csv("data/auto-mpg.csv")

# Convert origin to categorical variable
auto = auto.join(cat_to_num(auto['origin']))
auto = auto.drop('origin', axis=1)

# Split in train/test set
auto_train = auto[:int(0.8*len(auto))]
auto_test = auto[int(0.8*len(auto)):]

auto[:5]


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(auto_train.drop('mpg', axis=1), auto_train["mpg"])


pred_mpg = reg.predict(auto_test.drop('mpg',axis=1))


plot(auto_test.mpg, pred_mpg, 'o')
x = linspace(10,40,5)
plot(x, x, '-');


# # Chapter 9 - Scaling ML Workflows
# 

get_ipython().magic('pylab inline')


# ### Polynomial features
# 

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import cross_val_score

iris = datasets.load_iris()

linear_classifier = LogisticRegression()
linear_scores = cross_val_score(linear_classifier, iris.data, iris.target, cv=10) #2
print "Accuracy (linear):\t%0.2f (+/- %0.2f)" % (linear_scores.mean(), linear_scores.std() * 2)


pol = PolynomialFeatures(degree=2)
nonlinear_data = pol.fit_transform(iris.data)

nonlinear_classifier = LogisticRegression()
nonlinear_scores = cross_val_score(nonlinear_classifier, nonlinear_data, iris.target, cv=10)
print "Accuracy (nonlinear):\t%0.2f (+/- %0.2f)" % (nonlinear_scores.mean(), nonlinear_scores.std() * 2)


