# # Image Classification of Documents
# 

# ## 1. Setup
# To prepare your environment, you need to install some packages and enter credentials for the Watson services.
# 

# ### 1.1 Install the necessary packages
# You need the latest versions of these packages:
# python-swiftclient: is a python client for the Swift API.
# 

# ### Install IBM Cloud Object Storage Client: 
# 

get_ipython().system('pip install ibm-cos-sdk')


# ### Now restart the kernel by choosing Kernel > Restart.
# 

# ### 1.2 Import packages and libraries
# Import the packages and libraries that you'll use:
# 

import os, random
import numpy as np
import pandas as pd
import PIL
import keras
import itertools
from PIL import Image
import ibm_boto3
from botocore.client import Config


from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from skimage import feature, data, io, measure
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# ## 2. Configuration
# Add configurable items of the notebook below
# 

# ### 2.1 Add your service credentials for Object Storage
# You must create Object Storage service on IBM Cloud. To access data in a file in Object Storage, you need the Object Storage authentication credentials. Insert the Object Storage Streaming Body credentials and ensure the variable is referred as  streaming_body_1 in the following cell after removing the current contents in the cell.
# 


import sys
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_a66a3a4039de4247831dba3075f67804 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='',
    ibm_auth_endpoint="https://iam.eu-gb.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about your possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/
streaming_body_1 = client_a66a3a4039de4247831dba3075f67804.get_object(Bucket='cognitivebpm-donotdelete-pr-fqzzgidmfmfnvl', Key='Data.zip')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(streaming_body_1, "__iter__"): streaming_body_1.__iter__ = types.MethodType( __iter__, streaming_body_1 ) 



# ### 2.2 Global Variables 
# Enter the batch size for training, testing and validation dataset
# 

batch_size_train = 20
batch_size_val = 10
batch_size_test = 25
num_classes= 5
intereseted_folder='Documents'
STANDARD_SIZE=(224,224)


# # 3. Storage
# 
# ## 3.1 Extract the Dataset 
# 
# Input the zip file from object storage and extract the data onto the /home/dsxuser/work folder
# 

from io import BytesIO
import zipfile

zip_ref = zipfile.ZipFile(BytesIO(streaming_body_1.read()),'r')
paths = zip_ref.namelist()
classes_required=[]
for path in paths:
    zip_ref.extract(path)
    temp=path.split('/')
    if len(temp) > 3:
        if temp[2] not in classes_required:
            classes_required.append(temp[2])
print(classes_required)
zip_ref.close()


# # 4. Classification
# 
# ## 4.1 Create the Datset
# 

'''Converting Data Format according to the backend used by Keras
'''
datagen=keras.preprocessing.image.ImageDataGenerator(data_format=K.image_data_format())


'''Input the Training Data
'''
train_path = '/home/dsxuser/work/Data/Train_Data/'
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=classes_required, batch_size=batch_size_train)
type(train_batches)


'''Input the Validation Data
'''

val_path = '/home/dsxuser/work/Data/Val_Data/'
val_batches = ImageDataGenerator().flow_from_directory(val_path, target_size=(224,224), classes=classes_required, batch_size=batch_size_val)


'''Input the Test Data
'''
test_path = '/home/dsxuser/work/Data/Test_Data/'
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=classes_required, batch_size=batch_size_test)


test_imgs, test_labels = next(test_batches)
test_labels


y_test= [ np.where(r==1)[0][0] for r in test_labels ]
y_test


# ## 4.2 Build the Model
# 

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()


type(vgg16_model) #This is a Keras Functional API need to convert to sequential
model = Sequential() #Iterate over the functional layers and add it as a stack
for layer in vgg16_model.layers:
    model.add(layer)


model.layers.pop()
model.summary()


for layer in model.layers: #Since the model is already trained with certain weights, we dont want to change it. Let it be the same
    layer.trainable = False


model.add(Dense(5, activation='sigmoid')) # Add the last layer
model.summary()


# Complie the model
model.compile(Adam(lr=.00015), loss='categorical_crossentropy', metrics=['accuracy'])


# ## 4.3 Train the Model
# 
# The model will take about 30-45 minutes to train. 
# 

model.fit_generator(train_batches, steps_per_epoch=20, 
                    validation_data=val_batches, validation_steps=20, epochs=5, verbose=1)


# ## 4.4 Test the Model with External Test Images
# 


# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about your possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/
streaming_body_2 = client_a66a3a4039de4247831dba3075f67804.get_object(Bucket='cognitivebpm-donotdelete-pr-fqzzgidmfmfnvl', Key='test_doc-external.zip')['Body']
# add missing __iter__ method so pandas accepts body as file-like object
if not hasattr(streaming_body_2, "__iter__"): streaming_body_2.__iter__ = types.MethodType( __iter__, streaming_body_2 ) 



from io import BytesIO
import zipfile

zip_ref = zipfile.ZipFile(BytesIO(streaming_body_2.read()),'r')
paths = zip_ref.namelist()
del paths[0]
print(paths)
for path in paths:
    print(zip_ref.extract(path))
zip_ref.close()


X_test=[]
def convert_to_image(X):
    '''Function to convert all Input Images to the STANDARD_SIZE and create Training Dataset
    '''
    for f in paths:
        #fobj=get_file(f)
        #print(type(fobj))predictions= model.predict(X_test)
        if os.path.isdir(f):
            continue
        img= PIL.Image.open(f)
        img = img.resize(STANDARD_SIZE)
        img=np.array(img)
        X.append(img)
        #print(X_train)
    #print(len(X_train))
    return X
X_test=np.array(convert_to_image(X_test))
datagen.fit(X_test)


predictions= model.predict(X_test)
predictions


y_pred=[]
for i in range(len(predictions)):
    y_pred.append(np.argmax(predictions[i]))
y_pred
j = 0
for i in y_pred:
    print(paths[y_pred[j]])
    j = j + 1


print(classes_required)
index= classes_required.index('Documents')
for i in range(len(y_pred)):
    if y_pred[i] == index:
        print(paths[i])


# ## 4.5 Accuracy Testing
# 

predictions = model.predict_generator(test_batches, steps=1, verbose=0)
predictions


predictions
y_pred=[]
for i in range(len(predictions)):
    y_pred.append(np.argmax(predictions[i]))
print(y_pred)
#plots(test_imgs, titles=y_pred)

ctr=0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        ctr=ctr+1
res = ctr/len(y_pred)*100
print(res)








