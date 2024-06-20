# # Chapter 3
# # Predicting Sports Winners with Decision Trees
# 

import pandas as pd

data_filename = "basketball.csv"
dataset = pd.read_csv(data_filename)


dataset.head()


dataset = pd.read_csv(data_filename, parse_dates=["Date"])

dataset.columns = ["Date", "Start (ET)", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Score Type", "Notes"]


dataset.head()


print(dataset.dtypes)


dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]


dataset.head()


y_true = dataset["HomeWin"].values


dataset["HomeWin"].mean()


from collections import defaultdict
won_last = defaultdict(int)


dataset["HomeLastWin"] = 0
dataset["VisitorLastWin"] = 0


for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    dataset.set_value(index, "HomeLastWin", won_last[home_team])
    dataset.set_value(index, "VisitorLastWin", won_last[visitor_team])
    
    won_last[home_team] = int(row["HomeWin"])
    won_last[visitor_team] = 1 - int(row["HomeWin"])
    


dataset.head(6)


dataset.ix[1000:1005]


X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)


from sklearn.cross_validation import cross_val_score
import numpy as np


scores = cross_val_score(clf, X_previouswins, y_true,
scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


import os
standings_filename = os.path.join("standings.csv")

standings = pd.read_csv(standings_filename, skiprows=1)


standings.head()


dataset["HomeTeamRanksHigher"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
    visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]
    dataset.set_value(index, "HomeTeamRanksHigher", int(home_rank < visitor_rank))


X_homehigher = dataset[[ "HomeTeamRanksHigher", "HomeLastWin", "VisitorLastWin",]].values


clf = DecisionTreeClassifier(random_state=14, criterion="entropy")

scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')

print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


last_match_winner = defaultdict(int)
dataset["HomeTeamWonLast"] = 0

for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))  # Sort for a consistent ordering
    # Set in the row, who won the last encounter
    home_team_won_last = 1 if last_match_winner[teams] == row["Home Team"] else 0
    dataset.set_value(index, "HomeTeamWonLast", home_team_won_last)
    # Who won this one?
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    last_match_winner[teams] = winner


dataset.ix[400:450]


X_lastwinner = dataset[[ "HomeTeamWonLast", "HomeTeamRanksHigher", "HomeLastWin", "VisitorLastWin",]].values
clf = DecisionTreeClassifier(random_state=14, criterion="entropy")

scores = cross_val_score(clf, X_lastwinner, y_true, scoring='accuracy')

print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


from sklearn.preprocessing import LabelEncoder
encoding = LabelEncoder()
encoding.fit(dataset["Home Team"].values)
home_teams = encoding.transform(dataset["Home Team"].values)
visitor_teams = encoding.transform(dataset["Visitor Team"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
X_teams = onehot.fit_transform(X_teams).todense()

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


X_all = np.hstack([X_lastwinner, X_teams])
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


X_all = np.hstack([X_lastwinner, X_teams])
clf = RandomForestClassifier(random_state=14, n_estimators=250)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


from sklearn.grid_search import GridSearchCV
parameter_space = {
    "max_features": [2, 10, 'auto'],
    "n_estimators": [100, 200],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6],
}
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))


print(grid.best_estimator_)





import numpy as np 
from PIL import Image, ImageDraw, ImageFont 
from skimage import transform as tf


def create_captcha(text, shear=0, size=(100, 30), scale=1):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"bretan/Coval-Black.otf", 22) 
    draw.text((0, 0), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    image = image / image.max()
    shape = image.shape
    # Apply scale
    shapex, shapey = (shape[0] * scale, shape[1] * scale)
    image = tf.resize(image, (shapex, shapey))
    return image


get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
image = create_captcha("GENE", shear=0.5, scale=0.6)
plt.imshow(image, cmap='Greys')


from skimage.measure import label, regionprops


from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square


def segment_image(image):
    # label will find subimages of connected non-black pixels
    labeled_image = label(image>0.2, connectivity=1, background=0)
    subimages = []
    # regionprops splits up the subimages
    for region in regionprops(labeled_image):
        # Extract the subimage
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x,start_y:end_y])
    if len(subimages) == 0:
        # No subimages found, so return the entire image
        return [image,]
    return subimages


subimages = segment_image(image)


len(subimages)


f, axes = plt.subplots(1, len(subimages), figsize=(10, 3)) 
for i in range(len(subimages)): 
    axes[i].imshow(subimages[i], cmap="gray")


from sklearn.utils import check_random_state
random_state = check_random_state(14) 
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
assert len(letters) == 26
shear_values = np.arange(0, 0.8, 0.05)
scale_values = np.arange(0.9, 1.1, 0.1)


def generate_sample(random_state=None): 
    random_state = check_random_state(random_state) 
    letter = random_state.choice(letters) 
    shear = random_state.choice(shear_values)
    scale = random_state.choice(scale_values)
    return create_captcha(letter, shear=shear, size=(30, 30), scale=scale), letters.index(letter)


image, target = generate_sample(random_state) 
plt.imshow(image, cmap="Greys") 
print("The target for this image is: {0}".format(letters[target]))


dataset, targets = zip(*(generate_sample(random_state) for i in range(1000)))
dataset = np.array([tf.resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
dataset = np.array(dataset, dtype='float') 
targets = np.array(targets)


from sklearn.preprocessing import OneHotEncoder 
onehot = OneHotEncoder() 
y = onehot.fit_transform(targets.reshape(targets.shape[0],1))


y = y.todense()


X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)


# # Classification
# 

from sklearn.neural_network import MLPClassifier


clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=14)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import f1_score


f1_score(y_pred=y_pred, y_true=y_test, average='macro')


from sklearn.metrics import classification_report


print(classification_report(y_pred=y_pred, y_true=y_test))


def predict_captcha(captcha_image, neural_network):
    subimages = segment_image(captcha_image)
    dataset = np.array([tf.resize(subimage, (20, 20)) for subimage in subimages])
    X_test = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))

    y_pred = neural_network.predict_proba(X_test)
    predictions = np.argmax(y_pred, axis=1)
    assert len(y_pred) == len(X_test)
    predicted_word = str.join("", [letters[prediction] for prediction in predictions])
    return predicted_word


clf.predict_proba


word = "GENE"
captcha = create_captcha(word, shear=0.2) 
print(predict_captcha(captcha, clf))


plt.imshow(captcha, cmap="Greys") 


subimages = segment_image(captcha)
f, axes = plt.subplots(1, len(subimages), figsize=(10, 3)) 
for i in range(len(subimages)): 
    axes[i].imshow(tf.resize(subimages[i], (20,20)), cmap="gray")


image = create_captcha(word, shear=0.2, scale=1.1, size=(len(word) * 20, 30))
plt.imshow(image)


def test_prediction(word, net, shear=0.2, scale=1):
    captcha = create_captcha(word, shear=shear, scale=scale, size=(len(word) * 20, 30))
    prediction = predict_captcha(captcha, net)
    return word == prediction, word, prediction


from nltk.corpus import words


valid_words = [word.upper() for word in words.words() if len(word) == 4]


if False:
    correct = 0
    incorrect = 0

    for word in valid_words:
        shear = random_state.choice(shear_values)
        scale = random_state.choice(scale_values)
        subimages = create_captcha(word, shear=shear, scale=scale, size=(30, len(word) * 25))


num_correct = 0 
num_incorrect = 0 
for word in valid_words: 
    shear = random_state.choice(shear_values)
    scale = random_state.choice(scale_values)
    correct, word, prediction = test_prediction(word, clf, shear=shear, scale=scale)
    #print(correct, word, prediction)
    if correct: 
        num_correct += 1 
    else: 
        num_incorrect += 1
        #if len(prediction) == len(word):
        #    print(word, prediction)
print("Number correct is {0}".format(num_correct)) 
print("Number incorrect is {0}".format(num_incorrect))


correct


p = num_correct/(num_correct+num_incorrect)
p


0.99 ** 4


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
cm = np.array([row / np.sum(row) for row in cm])
np.fill_diagonal(cm, 0)
# cm = np.log(cm + 1e-16)


plt.figure(figsize=(10, 10)) 
plt.imshow(cm, interpolation='nearest')
tick_marks = np.arange(len(letters)+1)
plt.xticks(tick_marks, letters) 
plt.yticks(tick_marks, letters) 
plt.ylabel('Actual') 
plt.xlabel('Predicted')
plt.show()


from nltk.metrics import edit_distance 
steps = edit_distance("STEP", "STOP") 
print("The number of steps needed is: {0}".format(steps))


def compute_distance(prediction, word):
    
    len_word = min(len(prediction), len(word))
    
    return len_word - sum([prediction[i] == word[i] for i in range(len_word)])


from operator import itemgetter 

def improved_prediction(word, net, dictionary, shear=0.2, scale=1.0): 
    captcha = create_captcha(word, shear=shear, scale=scale) 
    prediction = predict_captcha(captcha, net) 
    
    if prediction not in dictionary:
        distances = sorted([(word, compute_distance(prediction, word)) for word in dictionary], key=itemgetter(1))
        best_word = distances[0] 
        prediction = best_word[0]
    return word == prediction, word, prediction


num_correct = 0 
num_incorrect = 0 
for word in valid_words: 
    shear = random_state.choice(shear_values)
    scale = random_state.choice(scale_values)
    correct, word, prediction = improved_prediction(word, clf, valid_words, shear=shear, scale=scale)
    #print(correct, word, prediction)
    if correct: 
        num_correct += 1 
    else: 
        num_incorrect += 1
        #if len(prediction) == len(word):
        #    print(word, prediction)
print("Number correct is {0}".format(num_correct)) 
print("Number incorrect is {0}".format(num_incorrect))


p = num_correct/(num_correct+num_incorrect)
p





