# ## Strava analytics with ipython
# 

# First step is to get an oauth token from strava.
# 
# Create a strava application
# 
# To do that, navigate to [this link](https://www.strava.com/oauth/authorize?client_id=5966&response_type=code&redirect_uri=http://localhost&approval_prompt=auto&scope=view_private), changing the client id to match your app and authorize the strava test app. After you have clicked authorize, you will be redirected to a page that does not exist, but you can get the code from the URL :
# 
# http://localhost/token_exchange.php?state=&code=XXXX
# 
# Strava API docs :
# http://strava.github.io/api/v3/oauth/
# 

strava_oauth_code = "7ba92a5340010f4035b2f897a7c93d6a9a331b53"


import requests

payload = {
    'client_id':"5966",
    'client_secret':"b8869c83423df058bbd72319cef18bd46123b251",
    'code':strava_oauth_code
}
resp = requests.post("https://www.strava.com/oauth/token", params=payload)
assert resp.status_code == 200

access_token = resp.json()['access_token']
headers = {
    'Authorization': "Bearer " + access_token
}


access_token


resp = requests.get("https://www.strava.com/api/v3/athlete", headers=headers)
assert resp.status_code == 200
athlete = resp.json()
print athlete['firstname'], athlete['lastname']


def get_activities(page):
    params = {
        'per_page': 50,
        'page':page
    }

    resp = requests.get("https://www.strava.com/api/v3/athlete/activities",
                        params=params, headers=headers)
    assert resp.status_code == 200
    activities = resp.json()
    return activities

def get_all_activities():
    all_activities = []
    page = 1
    while True:
        activities = get_activities(page)
        page += 1
        if len(activities) == 0:
            break
        all_activities += activities
    return all_activities
    
activities = get_all_activities()
print len(activities), ' activities total'


import json
with open('activities.json', 'w') as f:
    json.dump(activities, f)


# # Learning a cosine with keras
# 

import os
os.environ['THEANO_FLAGS']='mode=FAST_COMPILE,optimizer=None,device=cpu,floatX=float32'


import numpy as np
import sklearn.cross_validation as skcv
#x = np.linspace(0, 5*np.pi, num=10000, dtype=np.float32)
x = np.linspace(0, 4*np.pi, num=10000, dtype=np.float32)
y = np.cos(x)

train, test = skcv.train_test_split(np.arange(x.shape[0]))
print train.shape
print test.shape


import pylab as pl
get_ipython().magic('matplotlib inline')
pl.plot(x, y)


X_train = x[train].reshape(-1, 1)
y_train = y[train]

print "x_train : ", X_train.min(), X_train.max()
print X_train.shape
print "y_train : ", y_train.min(), y_train.max()
print y_train.shape
assert X_train.dtype == np.float32
assert y_train.dtype == np.float32


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(1, 4, init='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dense(4, 1, init='lecun_uniform'))
model.add(Activation('tanh'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

print model.get_weights()
history = model.fit(scaler.transform(X_train), y_train, nb_epoch=10, batch_size=64, shuffle=True)


y_pred = model.predict(scaler.transform(x.reshape(-1, 1)))


model.get_weights()


pl.plot(x, y_pred, c='r', label='y_pred')
pl.plot(x, y, c='b', label='y')
pl.legend()


# ## Playing with the number of hidden units
# 

# You might want to run the example multiple times as the random initialization influences the result quite a bit.
# 

def train_plot_prediction(n_hidden):
    model = Sequential()
    model.add(Dense(1, n_hidden, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(n_hidden, 1, init='lecun_uniform'))
    model.add(Activation('tanh'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    
    history = model.fit(scaler.transform(X_train), y_train, nb_epoch=5, batch_size=64, shuffle=True,
                       verbose=False)
    
    y_pred = model.predict(scaler.transform(x.reshape(-1, 1)))
    
    pl.figure(figsize=(10, 4))
    pl.subplot(211)
    pl.title('train loss')
    pl.plot(history.epoch, history.history['loss'], label='loss')
    pl.subplot(212)
    pl.title('prediction vs ground truth')
    pl.plot(x, y_pred, c='r', label='y_pred')
    pl.plot(x, y, c='b', label='y')
    pl.legend()
    pl.tight_layout()


train_plot_prediction(1)


train_plot_prediction(2)


train_plot_prediction(3)


train_plot_prediction(4)


train_plot_prediction(5)


train_plot_prediction(10)


# ## With random forest
# 

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, max_depth=10).fit(scaler.transform(X_train), y_train)

y_pred_rf = rf.predict(scaler.transform(x.reshape(-1, 1)))


pl.figure(figsize=(10, 4))
pl.plot(x, y_pred_rf, c='r', label='y_pred')
pl.plot(x, y, c='b', label='y')
pl.legend()





