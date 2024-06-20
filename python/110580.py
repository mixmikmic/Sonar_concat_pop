# # Exploring Kaggle Diabetic Retinopathy datasets
# 
# Download the dataset from the Kaggle page and store them under `inputs` folder.
# 
# References:
# - https://www.kaggle.com/c/diabetic-retinopathy-detection
# 
# ## Diabetic Retinopathy
# 
# > People with diabetes can have an eye disease called diabetic retinopathy. This is when high blood sugar levels cause damage to blood vessels in the retina. These blood vessels can swell and leak. Or they can close, stopping blood from passing through. Sometimes abnormal new blood vessels grow on the retina. All of these changes can steal your vision.
# 
# https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy
# 
# ### Severity Scale
# 
# - 0 - No DR
# - 1 - Mild
# - 2 - Moderate
# - 3 - Severe
# - 4 - Proliferative DR
# 
# References:
# - http://webeye.ophth.uiowa.edu/eyeforum/tutorials/Diabetic-Retinopathy-Med-Students/Classification.htm
# - http://arleoeye.com/services/common-eye-disorders/diabetic-retinopathy/
# - http://www.icoph.org/downloads/Diabetic-Retinopathy-Scale.pdf
# 

import pandas as pd
from glob import glob
import os
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

input_path = "../input/"


# ## Load the annotations (ground truth labels)
# 

def load_df(path):    
    def get_filename(image_id):
        return os.path.join(input_path, "train", image_id + ".jpeg")

    df_node = pd.read_csv(path)
    df_node["file"] = df_node["image"].apply(get_filename)
    df_node = df_node.dropna()
    
    return df_node

df = load_df(os.path.join(input_path, "trainLabels.csv"))
len(df)

df.head()


# ## Plot the retina images
# 

import math

def get_filelist(level=0):
    return df[df['level'] == level]['file'].values

def subplots(filelist):
    plt.figure(figsize=(16, 12))
    ncol = 3
    nrow = math.ceil(len(filelist) // ncol)
    
    for i in range(0, len(filelist)):
        plt.subplot(nrow, ncol, i + 1)
        img = cv2.imread(filelist[i])
        plt.imshow(img)


# ### Severity 0: No DR
# 
# > No abnormalities
# 

filelist = get_filelist(level=0)
subplots(filelist[:9])


# ### Severity 1: Mild
# 
# > Microaneurysms only
# 

filelist = get_filelist(level=1)
subplots(filelist[:9])


# ### Severity 2: Moderate
# 
# > More than just microaneurysms but less than Severe NPDR
# 

filelist = get_filelist(level=2)
subplots(filelist[:9])


# ### Severity 3: Severe
# 
# > Any of the following:
# - More than 20 intraretinal hemorrhages in each of 4
# quadrants
# - Definite venous beading in 2+ quadrants
# - Prominent IRMA in 1+ quadrant
# And no signs of proliferative retinopathy 
# 

filelist = get_filelist(level=3)
subplots(filelist[:9])


# ### Severity 4: Proliferative DR
# 
# > One or more of the following:
# - Neovascularization
# - Vitreous/preretinal hemorrhage 
# 

filelist = get_filelist(level=4)
subplots(filelist[:9])


# ### Class imbalance
# 

Counter(df['level'])


plt.hist(df['level'], bins=5)


# ## Exploring LUNA16 datasets
# 
# Obtain the dataset from LUNA16 page https://luna16.grand-challenge.org/data/, and store them in `inputs` folder.
# 
# Reference:
# - https://www.kaggle.com/c/data-science-bowl-2017#tutorial
# 
# ### Lung Nodule
# 
# > Lung nodules — small masses of tissue in the lung — are quite common. They appear as round, white shadows on a chest X-ray or computerized tomography (CT) scan.
# 
# > Lung nodules are usually about 0.2 inch (5 millimeters) to 1.2 inches (30 millimeters) in size. A larger lung nodule, such as one that's 30 millimeters or larger, is more likely to be cancerous than is a smaller lung nodule.
# 
# Reference:
# - http://www.mayoclinic.org/diseases-conditions/lung-cancer/expert-answers/lung-nodules/faq-20058445
# - http://emedicine.medscape.com/article/2139920-overview
# 

import SimpleITK
import numpy as np
import csv
from glob import glob
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

luna_path = "../inputs"
luna_subset_path = luna_path + "/subset*/"


# ### Dataset
# 
# The data are stored in .mhd and .raw files. .raw file contains raw images and the corresponding .mhd file contains metadata such as voxel resolution.
# 

mhd_file_list = glob(luna_subset_path + "*.mhd")

len(mhd_file_list)


# ## Plot the all slices of a scan
# 
# A scan contains about 100-300 slices. The voxel resolution (i.e. slice thickness and slice spacing) vary from sample to sample. See the metadata bundled in mhd files.
# 

import math

def plot_mhd_file(mhd_file):
    itk_img = SimpleITK.ReadImage(mhd_file) 
    img_array = SimpleITK.GetArrayFromImage(itk_img) # z,y,x ordering
    
    print("img_array.shape = ", img_array.shape)
    
    n_images = img_array.shape[0]
    ncol = 12
    nrow = math.ceil(n_images / ncol)
    
    plt.figure(figsize=(16, 16))
    for i in range(0, n_images):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(img_array[i], cmap=plt.cm.gray)

plot_mhd_file(mhd_file_list[0])


# ## Load the annoations
# 
# The annotations from doctors are stored in the csv file.
# 
# > The reference standard of our challenge consists of all nodules >= 3 mm accepted by at least 3 out of 4 radiologists.
# 
# https://luna16.grand-challenge.org/data/
# 

def load_df(path):    
    def get_filename(case):
        global mhd_file_list
        for f in mhd_file_list:
            if case in f:
                return(f)

    df_node = pd.read_csv(path)
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()
    
    return df_node

df = load_df(luna_path + "/CSVFILES/annotations.csv")

print("len(df) =", len(df))

df.head()


# ## Plot slices that contain the annotated nodules
# 

def plot_nodule(nodule_info):
    mhd_file = nodule_info[5]
    itk_img = SimpleITK.ReadImage(mhd_file) 
    img_array = SimpleITK.GetArrayFromImage(itk_img)  # z,y,x ordering
    origin_xyz = np.array(itk_img.GetOrigin())   # x,y,z  Origin in world coordinates (mm)
    spacing_xyz = np.array(itk_img.GetSpacing()) # spacing of voxels in world coor. (mm)
    center_xyz = (nodule_info[1], nodule_info[2], nodule_info[3])
    nodule_xyz = ((center_xyz - origin_xyz) // spacing_xyz).astype(np.int16)

    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    ax.imshow(img_array[nodule_xyz[2]], cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (nodule_xyz[0] - 10, nodule_xyz[1]-10),   # (x,y)
            20,          # width
            20,          # height
            linewidth=1, edgecolor='r', facecolor='none'
        )
    )

plot_nodule(df.iloc[0])
plot_nodule(df.iloc[1])
plot_nodule(df.iloc[2])
plot_nodule(df.iloc[3])


# ## Plot histogram of the nodule sizes (mm)
# 

nodule_sizes = list(df['diameter_mm'])

plt.hist(nodule_sizes, bins=30)


# ## MHD File Format
# 
# Similar to DICOM format, the file contains metadata in addition to raw images. For example,
# 
# * Spacing
#     * Voxel resoution (slice thickness and resolution) in mm
# * Origin
#     * The coordinates of the origin point in the world coordinates.
# 
# http://www.simpleitk.org/
# 

print(SimpleITK.ReadImage(mhd_file_list[0]))


# # Prediction: Sliding Window
# 

import keras
import SimpleITK
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

inputs_path = "../inputs"
luna_path = inputs_path + "/CSVFILES/"
luna_subset_path = inputs_path + "/subset*/"

df_node = pd.read_csv(luna_path + "annotations.csv")

resampled_file_list = glob("../preprocess/*.npz")

input_dim = 48


# ## Load the trained model
# 

model_path = "../notebooks/base_2017-07-05-10-04-34-val_loss_checkpoint.hdf5"
model = keras.models.load_model(model_path)


# ### Define some helper functions for getting 3D cubes
# 

import os, math

def get_seriesuid(filepath):
    basename = os.path.basename(filepath)
    
    return basename[:-8]

def switch_xz(arr):
    return np.array([arr[2], arr[1], arr[0]])

def get_nodule_index(nodule_info, new_spacing, origin_zyx):
    nodule_coordinates = np.array(nodule_info[1:4]).astype(np.float32) # x,y,z

    nodule_index = ((switch_xz(nodule_coordinates) - origin_zyx) // new_spacing).astype(np.int16)  # z,y,x
    
    return nodule_index

def get_range(x, minx, maxx, dim):
    hdim = dim // 2

    if x - hdim <= minx:
        xs = minx
        xe = xs + dim
    elif x + hdim >= maxx:
        xe = maxx
        xs = xe - dim
    else:
        xs = x - hdim
        xe = xs + dim
    
    return xs, xe

def get_3D_cube(img_array, x, y, z, dim):
    xr = get_range(x, 0, img_array.shape[2], dim)
    yr = get_range(y, 0, img_array.shape[1], dim)
    zr = get_range(z, 0, img_array.shape[0], dim)
    
    return img_array[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]


# ### Get list of cubes by sliding window
# 
# For now, we slide the window over the same slice with the annotated nodule.
# 

def get_sliding_windows(img, z, stride=8, width=input_dim):
    cube_list = []
    index_list = []

    for y in range(0, img.shape[1], stride):
        for x in range(0, img.shape[2], stride):
            index_list.append([z,y,x])

            cube = get_3D_cube(img, x, y, z, width)
            cube_list.append(np.reshape(cube, (width, width, width, 1)))
    
    return cube_list, index_list

npz_file = resampled_file_list[16]
print(npz_file)
npz_dict = np.load(npz_file)
resampled_img = npz_dict['resampled_img']
seriesuid = get_seriesuid(npz_file)
mini_df_node = df_node[df_node['seriesuid'] == seriesuid]

if len(mini_df_node) > 0:
    nodule_index_list = [get_nodule_index(nodule_info, npz_dict['new_spacing'], npz_dict['origin_zyx']) for nodule_info in mini_df_node.values] 
else:
    nodule_index_list = []

print(nodule_index_list)
mini_df_node.head()


# ## Run the loaded model
# 

z_index = nodule_index_list[0][0]
stride = 8

cube_list, index_list = get_sliding_windows(resampled_img, z_index, stride=stride)

predictions = model.predict(np.array(cube_list), batch_size=64, verbose=1)


# ### Reshape the list of predictions to 2D slice
# 

new_dim = math.ceil(resampled_img.shape[1] / stride)
prediction_slice = np.reshape(np.asarray(predictions), (new_dim, new_dim))
print(prediction_slice.shape)


# ## Plot the predicted probability map (2D slice)
# 
# See that the location of the annotated nodules has high-probability, although there're several false positives remaining.
# 

plt.imshow(prediction_slice, vmax=1, vmin=0, cmap='hot')


# ### Show the correct nodule location (the ground truth)
# 

def load_mdf(mdf_file, nodule_info):
    itk_img = SimpleITK.ReadImage(mdf_file) 
    img_array = SimpleITK.GetArrayFromImage(itk_img) # z,y,x
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coordinates (mm)
    
    nodule_center_world = np.float32(nodule_info[1:4]) # x, y, z in world coordinates (mm)
    nodule_center_idx = ((nodule_center_world - origin) // spacing).astype(np.int16)
    
    return img_array, nodule_center_idx


mhd_file_list = glob(luna_subset_path + "*.mhd")

nodule_info = mini_df_node.values[0]
seriesid = nodule_info[0]

mhd_file = next(f for f in mhd_file_list if seriesid in f)

img_array, nodule_center_idx = load_mdf(mhd_file, nodule_info)
nodule_x, nodule_y, nodule_z = nodule_center_idx

import matplotlib.patches as patches

fig, ax = plt.subplots(1)
ax.imshow(img_array[nodule_z])
ax.add_patch(
    patches.Rectangle(
        (nodule_x - 10, nodule_y - 10),
        20,          # width
        20,          # height
        linewidth=1,edgecolor='r',facecolor='none'
    )
)


# # Next Steps
# 
# Hard negative mining is one of the common techniques to reduce the false positives. Collect the false positives from the initial trained model, and restart the training using the false positive samples.
# 
# ## Ideas for improvements
# 
# By looking at the learning curve, the model is overfitting to the training set. To overcome overfitting, we can consider, for example:
# - Adding Dropout Layer
# - Data Augmentation (Ex. rotation, flip, scaling, etc.)
# - Decrease the model complexity
# - etc.
# 

