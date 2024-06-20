# <div class="clearfix" style="padding: 10px; padding-left: 0px">
# <img src="https://raw.githubusercontent.com/jupyter/nature-demo/master/images/jupyter-logo.png" width="150px" style="display: inline-block; margin-top: 5px;">
# <a href="http://bit.ly/tmpnbdevrax"><img src="https://cloud.githubusercontent.com/assets/836375/4916141/2732892e-64d6-11e4-980f-11afcf03ca31.png" width="150px" class="pull-right" style="display: inline-block; margin: 0px;"></a>
# </div>
# 
# ## Welcome to the Temporary Notebook (tmpnb) service!
# 
# This Notebook Server was **launched just for you**. It's a temporary way for you to try out a recent development version of the IPython/Jupyter notebook.
# 
# <div class="alert alert-warning" role="alert" style="margin: 10px">
# <p>**WARNING**</p>
# 
# <p>Don't rely on this server for anything you want to last - your server will be *deleted after 10 minutes of inactivity*.</p>
# </div>
# 
# Your server is hosted thanks to [Rackspace](http://bit.ly/tmpnbdevrax), on their on-demand bare metal servers, [OnMetal](http://bit.ly/onmetal).
# 

# ### Run some Python code!
# 
# To run the code below:
# 
# 1. Click on the cell to select it.
# 2. Press `SHIFT+ENTER` on your keyboard or press the play button (<button class='fa fa-play icon-play btn btn-xs btn-default'></button>) in the toolbar above.
# 
# A full tutorial for using the notebook interface is available [here](ipython_examples/Notebook/Index.ipynb).
# 

get_ipython().magic('matplotlib notebook')

import pandas as pd
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
import seaborn as sns

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
df.plot(); plt.legend(loc='best')


# Feel free to open new cells using the plus button (<button class='fa fa-plus icon-plus btn btn-xs btn-default'></button>), or hitting shift-enter while this cell is selected.
# 
# Behind the scenes, the software that powers this is [tmpnb](https://github.com/jupyter/tmpnb), a  Tornado application that spawns [pre-built Docker containers](https://github.com/ipython/docker-notebook) and then uses the [jupyter/configurable-http-proxy](https://github.com/jupyter/configurable-http-proxy) to put your notebook server on a unique path.
# 

# ![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png)
# ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# 
# # Welcome to Apache Spark with Python
# 
# > Apache Spark is a fast and general-purpose cluster computing system. It provides high-level APIs in Java, Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set of higher-level tools including Spark SQL for SQL and structured data processing, MLlib for machine learning, GraphX for graph processing, and Spark Streaming. 
# - http://spark.apache.org/
# 
# In this notebook, we'll train two classifiers to predict survivors in the [Titanic dataset](../edit/datasets/COUNT/titanic.csv). We'll use this classic machine learning problem as a brief introduction to using Apache Spark local mode in a notebook.

import pyspark  
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree


# First we create a [SparkContext](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext), the main object in the Spark API. This call may take a few seconds to return as it fires up a JVM under the covers.
# 

sc = pyspark.SparkContext()


# ## Sample the data
# 

# We point the context at a CSV file on disk. The result is a [RDD](http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds), not the content of the file. This is a Spark [transformation](http://spark.apache.org/docs/latest/programming-guide.html#transformations).
# 

raw_rdd = sc.textFile("datasets/COUNT/titanic.csv")


# We query RDD for the number of lines in the file. The call here causes the file to be read and the result computed. This is a Spark [action](http://spark.apache.org/docs/latest/programming-guide.html#actions).
# 

raw_rdd.count()


# We query for the first five rows of the RDD. Even though the data is small, we shouldn't get into the habit of pulling the entire dataset into the notebook. Many datasets that we might want to work with using Spark will be much too large to fit in memory of a single machine.
# 

raw_rdd.take(5)


# We see a header row followed by a set of data rows. We filter out the header to define a new RDD containing only the data rows.
# 

header = raw_rdd.first()
data_rdd = raw_rdd.filter(lambda line: line != header)


# We take a random sample of the data rows to better understand the possible values.
# 

data_rdd.takeSample(False, 5, 0)


# We see that the first value in every row is a passenger number. The next three values are the passenger attributes we might use to predict passenger survival: ticket class, age group, and gender. The final value is the survival ground truth.
# 

# ## Create labeled points (i.e., feature vectors and ground truth)
# 

# Now we define a function to turn the passenger attributions into structured `LabeledPoint` objects.
# 

def row_to_labeled_point(line):
    '''
    Builds a LabelPoint consisting of:
    
    survival (truth): 0=no, 1=yes
    ticket class: 0=1st class, 1=2nd class, 2=3rd class
    age group: 0=child, 1=adults
    gender: 0=man, 1=woman
    '''
    passenger_id, klass, age, sex, survived = [segs.strip('"') for segs in line.split(',')]
    klass = int(klass[0]) - 1
    
    if (age not in ['adults', 'child'] or 
        sex not in ['man', 'women'] or
        survived not in ['yes', 'no']):
        raise RuntimeError('unknown value')
    
    features = [
        klass,
        (1 if age == 'adults' else 0),
        (1 if sex == 'women' else 0)
    ]
    return LabeledPoint(1 if survived == 'yes' else 0, features)


# We apply the function to all rows.
# 

labeled_points_rdd = data_rdd.map(row_to_labeled_point)


# We take a random sample of the resulting points to inspect them.
# 

labeled_points_rdd.takeSample(False, 5, 0)


# ## Split for training and test
# 

# We split the transformed data into a training (70%) and test set (30%), and print the total number of items in each segment.
# 

training_rdd, test_rdd = labeled_points_rdd.randomSplit([0.7, 0.3], seed = 0)


training_count = training_rdd.count()
test_count = test_rdd.count()


training_count, test_count


# ## Train and test a decision tree classifier
# 

# Now we train a [DecisionTree](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.tree.DecisionTree) model. We specify that we're training a boolean classifier (i.e., there are two outcomes). We also specify that all of our features are categorical and the number of possible categories for each.
# 

model = DecisionTree.trainClassifier(training_rdd, 
                                     numClasses=2, 
                                     categoricalFeaturesInfo={
                                        0: 3,
                                        1: 2,
                                        2: 2
                                     })


# We now apply the trained model to the feature values in the test set to get the list of predicted outcomines.
# 

predictions_rdd = model.predict(test_rdd.map(lambda x: x.features))


# We bundle our predictions with the ground truth outcome for each passenger in the test set.
# 

truth_and_predictions_rdd = test_rdd.map(lambda lp: lp.label).zip(predictions_rdd)


# Now we compute the test error (% predicted survival outcomes == actual outcomes) and display the decision tree for good measure.
# 

accuracy = truth_and_predictions_rdd.filter(lambda v_p: v_p[0] == v_p[1]).count() / float(test_count)
print('Accuracy =', accuracy)
print(model.toDebugString())


# ## Train and test a logistic regression classifier
# 

# For a simple comparison, we also train and test a [LogisticRegressionWithSGD](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.classification.LogisticRegressionWithSGD) model.
# 

model = LogisticRegressionWithSGD.train(training_rdd)


predictions_rdd = model.predict(test_rdd.map(lambda x: x.features))


labels_and_predictions_rdd = test_rdd.map(lambda lp: lp.label).zip(predictions_rdd)


accuracy = labels_and_predictions_rdd.filter(lambda v_p: v_p[0] == v_p[1]).count() / float(test_count)
print('Accuracy =', accuracy)


# The two classifiers show similar accuracy. More information about the passengers could definitely help improve this metric.
# 

