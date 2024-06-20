# # Introduction to pandas
# 
# This is a pandas tutorial with the case of titanic.
# We will start with: 
# 
# Pandas tutorial:
# 1. Looking through the data
# 2. Exploring the variables
# 3. Cleaning the data
# 4. Transforming the data
# 5. Analysing the data
# 6. An automated way to find rules of classifications with Decision Tree (BONUS)
# 
# Looking through the data
# 1. Sample the data with head and tail
# 2. Describe the data for a quick overview
# 3. Selecting columns
# 4. Grouping the data
# 5. Plotting the data
# 
#  
# Interesting Question - Is it possible to predict the survival of the titanic passengers from the profile of the passengers? 
# 
# Exploring the variables - From here we should ask ourselves these questions
# 1. What variables do i need? - What profile characteristics about the passengers may affect survival? 
# 2. Should I transform any variables?
# 3. Are there NA Values, Outliers or Other Strange Values?
# 4. Should I Create New Variables?
# 
# Cleaning the data
# 1. Checking for data abnormalities
# 2. Handle missing values
# 3. Remove columns which are not used
# 
# Transforming the data
# 1. Transform variables into much more readable forms
# 2. Group feature for comparisons
# 3. Generate features that might help us analyse more
# 
# Analysing the data
# From the data let us try to find:
# 1. The spread of age group
# 2. The proportion of survivor with family
# 3. The proportion of male/female with certain socioeconomic status that survive, given that they are adult
# 4. The proportion of embarked port to survival

# ## Dataset Explanation
# VARIABLE DESCRIPTIONS:
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES)
#  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 
# Age is in Years; Fractional if Age less than One (1)
#  If the Age is Estimated, it is in the form xx.5
# 
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
# 
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# 
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.
# 

# # Looking through the data
# 
# 1. Sample the data with head and tail
# 2. Describe the data for a quick overview
# 3. Selecting columns
# 4. Grouping the data
# 5. Plotting the data
# 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


get_ipython().magic('matplotlib inline')


maindf = pd.read_csv('dataset/titanic/train.csv')


maindf.head()


maindf.tail()


maindf.describe()


maindf.describe(include=['object'])


# From the describes. We can find out the mean and the distribution of the variables. Describe without the object represents numerical values whereas if you put object type, it represents categorical.
# From here we can find out:
# 1. Most of the people
# 2. Most of the passengers bought the tickets for relatively lower price, but some of them. Oddly bought it pretty high. Possibly because they are from VIPs
# 

maindf.columns


# # Exploring the variables
# ## From here we can look at the following questions:
# 
# 1. How many survive in the dataset?
# 2. How many survive depending on the gender?

# ## How many survive in the dataset?
# 

conditionsurvive = (maindf.Survived==1)


maindf.loc[[0,1,2],['PassengerId','Survived','Pclass']]


# Finding survive
maindf.loc[conditionsurvive,:].shape


# Finding non survive
maindf.loc[~conditionsurvive,:].shape


# Or anotther way is to group them by survived
maindf.Survived.value_counts()


maindf.groupby('Survived').count().Name


# ##### There are 342 survived and 549 non survived
# 

# ## How many survive in the dataset?
# 

# Let's now group the data to find if most of them are male or female
maindf.groupby(['Survived','Sex']).count().Name


# Let's plot them out and see the result
maindf.groupby(['Survived','Sex']).count().Name.plot(kind='bar')


survivedsex = maindf.groupby(['Survived','Sex']).count().Name.reset_index()
maindfsurvivedis0 = survivedsex.loc[(survivedsex.Survived==0),:]
maindfsurvivedis1 = survivedsex.loc[(survivedsex.Survived==1),:]


maindfsurvivedis0


# ### There are 342 survived and 549 non survived
# ##### Out of those survived (233 are female, 109 are male ) whereas non survived ( 81 are female, 468 are male)
# 

# ## Cleaning the data
# 
# 1. Checking for data abnormalities
# 2. Handle missing values
# 3. Remove columns which are not used
# 

maindf.dtypes


maindf.info()


# ### There are missing data on the age(numerical). We need to clean that up.
# 

# Replace null in age with the average
maindf['Age'] = maindf.loc[:,['Age']].fillna(maindf['Age'].mean())


# Describing columns for analysis
# Drop useless columns such as passengerid ,name, (ticket, fare) can be described by socioeconomic status, and cabin (too much null)
cleandf = maindf.loc[:,['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']]


# # Transforming the data
# 
# 1. Transform variables into much more readable forms
# 2. Group feature for comparisons
# 3. Generate features that might help us analyse more
# 

# We can transform Pclass and Embarked
# 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
cleandf['socioeconomicstatus']=cleandf.Pclass.map({1:'upper',2:'middle',3:'lower'})


# (C = Cherbourg; Q = Queenstown; S = Southampton)
cleandf['embarkedport']=cleandf.Embarked.map({'C':'Cherbourg','Q':'Queenstown','S':'Southampton'})


# Dropping the used columns
cleandf.drop(['Pclass','Embarked'],axis=1,inplace=True)


# Group age for comparisons
cleandf.Age.hist()


# Let us try to separate this into ages
agesplit = [0,10,18,25,40,90]
agestatus = ['Adolescent','Teenager','Young Adult','Adult','Elder']

cleandf['agegroup']=pd.cut(cleandf.Age,agesplit,labels=agestatus)


# Create a feature where we count both numbers of siblings and parents
cleandf['familymembers']=cleandf.SibSp+cleandf.Parch


# Let us try to find whether the passengers are alone or not
hasfamily = (cleandf.familymembers>0)*1
cleandf['hasfamily'] = hasfamily


# Dropping the used columns
cleandf.drop(['SibSp','Parch','Age'],axis=1,inplace=True)


# Final transformed data
cleandf.head()


cleandf.to_csv('cleanedandtransformedtitanicdata.csv')


# # Analysing the data
# 
# From the data let us try to find out what profile characteristics of the passengers are related to their survival
# 
# 1. The spread of age group
# 2. The proportion of survivor with family members
# 3. The proportion of male/female with certain socioeconomic status that survive, given that they are adult
# 4. The proportion of embarked port to survival
# 

# Reading from csv
cleandf = pd.read_csv('cleanedandtransformedtitanicdata.csv')


# ## The spread of age group
# 

cleandf.agegroup.value_counts().plot(kind='bar')


# ## The proportion of survivor with family
# 

# The proportion of survivor with family
cleandf.groupby(['Survived','hasfamily']).count().agegroup.plot(kind='bar')


survived = pd.crosstab(index=cleandf.Survived, columns = cleandf.socioeconomicstatus,margins=True)
survived.columns = ['lower','middle','upper','rowtotal']
survived.index = ['died','survived','coltotal']


survived


survived/survived.ix['coltotal','rowtotal']


# Most of the lower class died, middle have same chances of survival, and upper are morelikely to survive


# ## The proportion of embarked port to survival
# 

cleandf.groupby(['Survived','embarkedport']).count().agegroup


# create a figure with two subplots
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

notsurvivors = cleandf[cleandf.Survived==0].embarkedport.value_counts()
survivors= cleandf[cleandf.Survived==1].embarkedport.value_counts()


# plot each pie chart in a separate subplot
ax1.pie(notsurvivors,labels=notsurvivors.index);
ax2.pie(survivors,labels=survivors.index);


# ##  The proportion of male/female with certain socioeconomic status that survive, given that they are adult
# 

print(cleandf.socioeconomicstatus.value_counts())
cleandf.socioeconomicstatus.value_counts().plot(kind='bar')


isadult = cleandf.agegroup=='Adult'
issurvived = cleandf.Survived==1
isnotsurvived = cleandf.Survived==0

all = cleandf[isadult].groupby(['Sex','socioeconomicstatus']).count().Survived
survived = cleandf[isadult&issurvived].groupby(['Sex','socioeconomicstatus']).count().Survived
notsurvived = cleandf[isadult&isnotsurvived].groupby(['Sex','socioeconomicstatus']).count().Survived


survivedcrosstab = pd.crosstab(index=cleandf.Survived, columns = cleandf.socioeconomicstatus,margins=True)
survivedcrosstab.columns = ['lower','middle','upper','rowtotal']
survivedcrosstab.index = ['died','survived','coltotal']


survivedcrosstab


survivedcrosstab/survivedcrosstab.ix['coltotal','rowtotal']


survivedcrosstabsex = pd.crosstab(index=cleandf.Survived, columns = [cleandf['socioeconomicstatus'],cleandf['Sex']],margins=True)


survivedcrosstabsex


# Probability of survival
(survived/all).plot(kind='bar')


# #### Upper class femailes survived more than males regardless of socioeconomic status. In general, upper class and female genders are the one benefited the most. Probably because the priorities on female and children in upper class passengers first
# 

# ## An automated way to find rules of classifications with Decision Tree (BONUS)
# 

#dropping left and sales X for the df, y for the left
X = cleandf.drop(['Survived'],axis=1)
y = cleandf['Survived']


# Clean up x by getting the dummies
X=pd.get_dummies(X)


import numpy as np
from sklearn import preprocessing,cross_validation,neighbors,svm
#splitting the train and test sets
X_train, X_test, y_train,y_test= cross_validation.train_test_split(X,y,test_size=0.2)


from sklearn import tree
clftree = tree.DecisionTreeClassifier(max_depth=3)
clftree.fit(X_train,y_train)


# Visualizing the decision tree
from sklearn import tree
from scipy import misc
import pydotplus
import graphviz

def show_tree(decisionTree, file_path):
    tree.export_graphviz(decisionTree, out_file='tree.dot',feature_names=X_train.columns)
    graph = pydotplus.graphviz.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')
    i = misc.imread(file_path)
    
    fig, ax = plt.subplots(figsize=(18, 10))    
    ax.imshow(i, aspect='auto')

# To use it
show_tree(clftree, 'tree.png')


# Finding the accuracy of decision tree

from sklearn.metrics import accuracy_score, log_loss

print('****Results****')
train_predictions = clftree.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))


# ## Conclusion:
# 1. Gender plays a very important roles, female has the most likelihood of surviving.
# 2. The higher Socioeconomic status, the more likelihood of surviving (especially for female)
# 3. Adolescent has higher chances of surviving as long as he/she does not have low socioeconomic status
# 

# # Introduction
# This is a machine learning for Titanic.
# We will start with: 
# 
# Descriptive information (exploratory)
# 1. What are the profiles of people who survive and not survive?
# 2. Is it true that kids and women are prioritized to survive?
# 3. How does the existence of siblings and parents affect the likelihood to survive?
# 
# Predictive information:
# 1. who are the ones that is more likely to survive
# 

# VARIABLE DESCRIPTIONS:
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES)
#  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 
# Age is in Years; Fractional if Age less than One (1)
#  If the Age is Estimated, it is in the form xx.5
# 
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
# 
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# 
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.
# 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


maindf = pd.read_csv('input/train.csv')
testdf = pd.read_csv('input/test.csv')
maindf.head()


#Check columns informations
maindf.info()
print("----------------------------")
testdf.info()


# Create a train set descriptors and result

#Assumes that the PassengerID,name,ticket,cabin,and embarked do not matter
#Assumes that Fare is correlated to pclass
X = maindf.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked','Fare'],axis=1)
Xtest = testdf.drop(['PassengerId','Name','Ticket','Cabin','Embarked','Fare'],axis=1)
y = maindf['Survived']


X.head()


# ## Data Preparation 
# 
# We will bin the data of:
# 1. Age
# 2. SibSp
# 3. Parch
# 4. Fare
# 

X.shape


X.describe()


#notice that the count of age is below than 714, indicate that there are empty values
#What are the empty values in each column of X
X.apply(lambda x: sum(x.isnull()),axis=0) 


#Refill empty values with Mean
X['Age'].fillna(maindf['Age'].mean(), inplace=True)
Xtest['Age'].fillna(testdf['Age'].mean(), inplace=True)


# #### Age
# **In order to better characterize the sample, the age range was classified according to growth stages: childhood (2 to 10 years), adolescence (11 to 17 years), young adult (18 to 40 years), adult (41 to 65 years) and elderly (> 65 years)**
# 
# #### Fare
# **Min to First Quarter is Low, Quarter 1 to Quarter 2 is med, Quarter 2 to Quarter 3 is high, and Quarter 3 to quarter 4 is very high**
# 
# #### Sex
# **Will be map to number male = 0 , female = 1**
# 

age_bins = [0, 2, 10, 17, 40, 65, 100]
age_group = [0,1,2,3,4,5]
X['Age']= pd.cut(X['Age'], age_bins, labels=age_group)
# age_group = ['baby', 'child', 'adolescence', 'young adult','adult','elderly']
Xtest['Age']= pd.cut(Xtest['Age'], age_bins, labels=age_group)

# fare_bins = [0,7.910400, 14.454200, 31.000000, 512.329200]
# fare_group = ['low', 'med', 'high', 'very high']
# X['Fare']= pd.cut(X['Fare'], fare_bins, labels=fare_group)


#Map Sex to 0,1
X['Sex'] = X['Sex'].map({'male':0,'female':1})
Xtest['Sex'] = Xtest['Sex'].map({'male':0,'female':1})


#SibSp would only care if the person brings spouse or sibling
#Parch would only care if the person brings parent or children

X['SibSp'][X['SibSp']>0]=1
X['Parch'][X['Parch']>0]=1
Xtest['SibSp'][Xtest['SibSp']>0]=1
Xtest['Parch'][Xtest['Parch']>0]=1

# X['WithSomebody'] = X['SibSp']+X['Parch']
X.head()


X.shape


y.shape


# ### Is it true that kids and women are prioritized to survive? 
# 

kidsorwoman = y[(X['Age']<3) | (X['Sex'] == 1)]
kidsorwoman.value_counts()
#From this result we know that kids or women are more likely to survive than die.


# ### How does the existence of siblings and parents affect the likelihood to survive?
# 

nosiblingorparent = y[X['SibSp']+ X['Parch']<1]
hassiblingorparent = y[X['SibSp']+ X['Parch']>=1]
print(nosiblingorparent.value_counts())
print('____________________')
print(hassiblingorparent.value_counts())

#From here we can see that the likelihood to survive is more if a person has anyone with him/her


# ## Training the model
# 

import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.tree import DecisionTreeClassifier


#splitting the train and test sets
X_train, X_test, y_train,y_test= cross_validation.train_test_split(X,y,test_size=0.2)


# ## DecisionTree Model Train
# 

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


pd.DataFrame(X_train,y_train).head()
accuracy = clf.score(X_test,y_test)
print(accuracy)


# ## Training Multiple Classifiers and Test Them
# 

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()


# Predict Test Set

favorite_clf = RandomForestClassifier()
favorite_clf.fit(X_train, y_train)
y_pred = pd.DataFrame(favorite_clf.predict(Xtest))


# Tidy and Export Submission
submission = pd.DataFrame({
        "PassengerId": testdf["PassengerId"]    
    })
submission['Survived'] = y_pred

submission.to_csv('submission.csv', index = False)
submission.tail()


submission.shape


