# ### Sigmoid Layer
# 

from IPython.display import Image
Image("sigmoid.png")


# ### Artifical Neural Networks
# 

Image("nn.PNG")


# ### Back Propogation
# 

Image("backprop.png")


# conda update scikit-learn
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
get_ipython().magic('matplotlib inline')


data=pd.read_csv('mnist.csv')


df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)


nn.fit(x_train,y_train)


pred=nn.predict(x_test)
#activation logistic with hidden layer sizes-> 45,90 Gave 92 % accuracy
#activation relu with hidden layer sizes-> 45,90 gave 89 % accuracy 
#Test with different combinatations of learning rate, activation and other hyper parameters and mesure the accuracy


a=y_test.values


a
count=0


for i in range(len(pred)):
    if pred[i]==a[i]:
        count=count+1


count


len(pred)


6824/8400.0





