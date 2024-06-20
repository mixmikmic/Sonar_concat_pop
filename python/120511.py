from __future__ import division, print_function

import os, glob
import numpy as np
import matplotlib.pyplot as plt

from rvseg import patient

get_ipython().magic('matplotlib inline')

basedir = "/home/paperspace/Developer/datasets/rvsc/TrainingSet"


# ## Visualize Images and Masks
# 

datadir = os.path.join(basedir, "patient09")
p = patient.PatientData(datadir)

alpha = 0.2
for index in range(len(p.images)):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(p.images[index], cmap=plt.cm.gray)
    plt.plot(*p.endocardium_contours[index])
    plt.plot(*p.epicardium_contours[index])
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(p.images[index], cmap=plt.cm.gray)
    plt.imshow(p.endocardium_masks[index], cmap=plt.cm.gray, alpha=alpha)
    plt.imshow(p.epicardium_masks[index], cmap=plt.cm.gray, alpha=alpha)


# ## Construct histograms
# 
# Three-component histograms: background, cardiac muscle, right ventricle cavity.
# 
# Data is strongly biased towards background pixels, which will affect training.
# 

from collections import Counter

glob_search = os.path.join(basedir, "patient*")
datadirs = glob.glob(glob_search)

counter_background = Counter()
counter_muscle = Counter()
counter_cavity = Counter()
for datadir in datadirs:
    p = patient.PatientData(datadir)
    for image, endo_mask, epi_mask in zip(p.images, p.endocardium_masks, p.epicardium_masks):
        image = np.array(255 * (image/image.max()), dtype='uint8')
        endo_bmask = endo_mask
        epi_bmask = epi_mask
        background = image * (1 - epi_bmask)
        muscle = image * (epi_bmask - endo_bmask)
        cavity = image * endo_bmask
        counter_background += Counter(background.flatten())
        counter_muscle += Counter(muscle.flatten())
        counter_cavity += Counter(cavity.flatten())
#    plt.figure()
#    plt.imshow(image, cmap=plt.cm.gray)


sum(counter_cavity.values())/norm


def rebin(counter, factor=8):
    assert 256 % factor == 0
    counts = np.array([counter[n] for n in range(256)])
    return np.sum(counts.reshape((len(counts)//factor, factor)), axis=1)

counter_background[0] = 0
counter_muscle[0] = 0
counter_cavity[0] = 0

alpha = 0.7
factor = 8
nbins = 256//factor
x = np.arange(nbins)/(nbins-1)
width = 0.8/nbins
norm = (sum(counter_background.values()) + sum(counter_muscle.values()) +
        sum(counter_cavity.values()))
plt.bar(x, rebin(counter_background, factor)/norm,
        width=width, alpha=0.3, label='background')
plt.bar(x, rebin(counter_muscle, factor)/norm,
        width=width, alpha=alpha, label='cardiac muscle')
plt.bar(x, rebin(counter_cavity, factor)/norm,
        width=width, alpha=alpha, label='right ventricular cavity')
plt.legend()
plt.xlabel("pixel intensity")
plt.ylabel("probability")
plt.title("Pixel intensity distribution by class")


# ## Histogram equalization
# 
# Reference: http://en.wikipedia.org/wiki/Histogram_equalization
# 

from skimage import exposure

p = patient.PatientData(os.path.join(basedir, "patient11"))
n = 10
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(p.images[n], cmap=plt.cm.gray)
plt.subplot(1, 2, 2)
plt.imshow(exposure.equalize_hist(p.images[n]), cmap=plt.cm.gray)


# ## Generate Video
# 
# Create videos of beating heart of each patient. Grayscale frames are scaled by their maximum value.
# 

glob_search = os.path.join(basedir, "patient*")
datadirs = sorted(glob.glob(glob_search))

FPS=24
for datadir in datadirs:
    p = patient.PatientData(datadir)
    index = os.path.basename(datadir)[-2:]
    outfile = "out{}.mp4".format(index)
    p.write_video(outfile=outfile, FPS=FPS)


# ## Generate unittest masks for PatientData class
# 

p = patient.PatientData(os.path.join(basedir, "patient09"))


# check endo- and epi- cardium masks
print("Max pixel value (endo):", p.endocardium_masks[0].max())
print("Max pixel value (epi):",  p.epicardium_masks[0].max())

plt.imshow(p.endocardium_masks[0], cmap=plt.cm.gray, alpha=0.2)
plt.imshow(p.epicardium_masks[0], cmap=plt.cm.gray, alpha=0.2)

plt.figure()
plt.imshow(p.images[0], cmap=plt.cm.gray)
plt.imshow(p.endocardium_masks[0], cmap=plt.cm.gray, alpha=0.2)
plt.imshow(p.epicardium_masks[0], cmap=plt.cm.gray, alpha=0.2)


# Write out files for test assets
np.savetxt("endocardium-p09-0020.mask", p.endocardium_masks[0], fmt='%.3d')
np.savetxt("epicardium-p09-0020.mask", p.epicardium_masks[0], fmt='%.3d')


import matplotlib.pyplot as plt
from rvseg import dataset

get_ipython().magic('matplotlib inline')


# ## Visualize effects of data augmentation
# 

data_dir = "/home/paperspace/Developer/software/cardiac-segmentation/test-assets/"

augmentation_args = {
    'rotation_range': 180,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.05,
    'fill_mode' : 'nearest',
    'alpha': 500,
    'sigma': 20,
}

(train_generator, train_steps_in_epoch,
 val_generator, val_steps_in_epoch) = dataset.create_generators(
    data_dir=data_dir,
    batch_size=16,
    validation_split=0.0,
    mask='both',
    shuffle=True,
    seed=0,
    normalize_images=True,
    augment_training=False,
    augment_validation=False,
    augmentation_args=augmentation_args)


images, masks = next(train_generator)
for image,mask in zip(images, masks):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(image[:,:,0], cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(image[:,:,0], cmap=plt.cm.gray)
    plt.imshow(mask[:,:,1], cmap=plt.cm.gray, alpha=0.4)


# ## Verify reproducibility of shuffling
# 

data_dir = "/home/paperspace/Developer/datasets/rvsc/TrainingSet/"

seed = 1

(train_generator, train_steps_in_epoch,
 val_generator, val_steps_in_epoch) = dataset.create_generators(
    data_dir=data_dir,
    batch_size=16,
    validation_split=0.2,
    mask='inner',
    shuffle_train_val=True,
    shuffle=True,
    seed=seed,
    normalize_images=True,
    augment_training=False,
    augment_validation=False)

def gridplot(images, masks, cols):
    rows = len(images)//cols + 1
    plt.figure(figsize=(12,2.75*rows))
    for i,(image,mask) in enumerate(zip(images, masks)):
        plt.subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(image[:,:,0], cmap=plt.cm.gray)
        plt.imshow(mask[:,:,1], cmap=plt.cm.gray, alpha=0.4)

images, masks = next(train_generator)
gridplot(images, masks, cols=4)    
    
images, masks = next(val_generator)
gridplot(images, masks, cols=4)


