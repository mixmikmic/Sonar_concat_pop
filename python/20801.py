# # Read images and labels from LMDBs made by SSD
# 

# ### Load necessary libs and set up caffe/caffe_root
# 

# Make sure that caffe is on the python path:
#caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
caffe_root = '/work/caffe'
voc_labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
# LMDB directory name. Inside this directory there are data.mdb and lock.mdb
lmdb_dir = '/work/caffe/examples/VOC0712/VOC0712_test_lmdb'

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


# ### Load LabelMap and define get_labelname function
# 

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# ### Define functions to read LMDB image/labels from LMDB values
# 

'''
Breakdown of what val(ue) contains from SSD's lmdb files:
- 8 bytes of...something
- JPEG (starts with \xff\xd8, ends at \xff\xd9)
- image tags

The below functions are mostly needed to parse lmdb label data. 
I haven't figured out the format exactly, but I have annotated
one example below from VOC0712 data:

    (begin tags)
    28ffffffffffffffffff0138011000
    (begin label 1)
    1a 1e 08
    (label: \x10 = 16 = 'pottedplant')
    10
    (begin bounding boxes)
    (      0th box:   xmin        ymin        xmax        ymax)
    121a08 00 1216 0d ee7cbf3e 15 ec51b83e 1d 4e62103f 25 9134253f 3000
    (end label 1)
    (begin label 2)
    1a 1e 08
    (label: \x0b = 11 = 'diningtable')
    0b
    (      0th box:   xmin        ymin        xmax        ymax)
    121a08 00 1216 0d 2db29d3e 15 66ad0e3f 1d 91ed3c3f 25 0000803f 3000
    (end label 2)
    (begin label 3)
    1a aa01 08
    (label: \x09 = 9 = 'chair')
    09
    (      0th box:   xmin        ymin        xmax        ymax)
    121a08 00 1216 0d 5c8f023f 15 df4f0d3f 1d 5a643b3f 25 0000803f 3000
    (      1st box:   xmin        ymin        xmax        ymax)
    121a08 01 1216 0d 7593183f 15 b81e053f 1d e7fb293f 25 619e283f 3001
    (      2nd box:   xmin        ymin        xmax        ymax)
    121a08 02 1216 0d 17d90e3f 15 e8b4013f 1d 2db21d3f 25 2db21d3f 3001
    (      3rd box:   xmin        ymin        xmax        ymax)
    121a08 03 1216 0d ba498c3e 15 6f12033f 1d be9f9a3e 25 c5d9073f 3001
    (      4th box:   xmin        ymin        xmax        ymax)
    121a08 04 1216 0d ba498c3e 15 022b073f 1d 77be9f3e 25 afb9103f 3001
    (      5th box:   xmin        ymin        xmax        ymax)
    121a08 05 1216 0d df4f8d3e 15 ec0a103f 1d dbf9fe3e 25 0000803f 3000
    (end tags)
'''
def get_tags(val):
    # create label/bounding box arrays
    labels = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    
    # find beginning of tags
    stream = StringIO.StringIO(val)
    match = re.search('\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x38\x01\x10\x00', stream.read())

    # go to start of tags
    stream.seek(match.end())

    while tag_has_another_label(stream):

        label = get_next_label(stream)

        while label_has_another_box(stream):
            xmin, ymin, xmax, ymax = get_bounding_box(stream)            
            labels.append(label)
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
    
    return labels, xmins, ymins, xmaxs, ymaxs

def label_has_another_box(stream):
    bits = stream.read(2)
    if(len(bits) == 2 and struct.unpack('H', bits)[0] == 6674):
        stream.seek(-2, 1)
        return True
    else:
        stream.seek(-2, 1)
        return False

def get_next_label(stream):
    label_num = struct.unpack('b', stream.read(1))[0]
    return get_labelname(voc_labelmap, label_num)[0]

def tag_has_another_label(stream):
    bits = stream.read(1)
    if(struct.unpack('b', bits)[0] == 26): # 26 = \x1a
        # go to label
        while struct.unpack('b', stream.read(1))[0] != 8:
            continue
        return True
    else:
        return False

def get_bounding_box(stream):
    to_xmin = 7
    to_next_flt = 1
    to_end = 2    
        
    stream.seek(to_xmin, 1)
    xmin = struct.unpack('f', stream.read(4))[0]

    stream.seek(to_next_flt, 1)
    ymin = struct.unpack('f', stream.read(4))[0]

    stream.seek(to_next_flt, 1)
    xmax = struct.unpack('f', stream.read(4))[0]

    stream.seek(to_next_flt, 1)
    ymax = struct.unpack('f', stream.read(4))[0]
    
    stream.seek(to_end, 1)

    return xmin, ymin, xmax, ymax

def get_image(val):
    img_stream = StringIO.StringIO(val[8:]) 
    return Image.open(img_stream)


def display_image(image, labels, xmins, ymins, xmaxs, ymaxs):
    # plot image
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.cla()
    plt.imshow(image)
    currentAxis = plt.gca()

    # display bounding box
    for i in range(0,len(labels)):
        xmin = int(xmins[i] * image.size[0])
        ymin = int(ymins[i] * image.size[1])
        xmax = int(xmaxs[i] * image.size[0])
        ymax = int(ymaxs[i] * image.size[1])

        name = '%s'%(labels[i])
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[i % len(colors)]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.5})
        
def num_keys(curs):
    count = 0
    if curs.first():
        count += 1
        while curs.next():
            count += 1
    return count


# ### Open LMDB file and read specific image/labels given key index
# 

import caffe
import lmdb
import struct
from PIL import Image
import cStringIO as StringIO
import re

# index of image from lmdb file (0 to 4951 for VOC0712)
key_index = 0

env = lmdb.open(lmdb_dir, readonly=True)
with env.begin() as txn:
    with txn.cursor() as curs:
            
#         # Find the total number of key, value pairs
#         print "Total number of key, value pairs: ", str(num_keys(curs))
            
        # get any key/value by number
        curs.first()
        for i in range(0,key_index):
            curs.next()
        key, val = curs.item()
        
        # display key name
        print key
        
        # get tags
        labels, xmins, ymins, xmaxs, ymaxs = get_tags(val)
        
        # get image
        image = get_image(val)

        # draw image
        display_image(image, labels, xmins, ymins, xmaxs, ymaxs)





# ## Get the outputs
# 
# Lab41 trained a DenseCap model using Visual Genome's (hereafter VG) object tags. We ran the model on the ESP-Game test images (hereafter ESP) to evaluate X-corpus.
# 
# Notes: A GloVe word vector model was used for translating words in VG's vocabulary, but not in ESP's vocabulary, to ESP's vocabulary. To do the translation between predicted words not in ESP's vocabulary to a single ESP word, the highest correlated value (according to the dot product of the GloVe model vectors) of that predicted word with an ESP word was chosen.
# 

import sys
sys.path.append("attalos/")
from attalos.evaluation.evaluation import Evaluation
from oct2py import octave
octave.addpath('attalos/attalos/evaluation/')
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.core.display import display
from IPython.core.pylabtools import figsize
from scripts import load_combine_json_dir, print_list_to_columns


jsondir = "/data/fs4/datasets/espgame/ESP-ImageSet"
imgdir = "/data/fs4/datasets/espgame/ESP-ImageSet/test_images"
wordvecs_dir = "/data/fs4/teams/attalos/wordvecs"


with open(os.path.join(jsondir,"esp_test_imageset_tags_predictions.json")) as jdata:
    predicts = json.load(jdata)

with open(os.path.join(jsondir,"esp_test_imageset_tags.json")) as jdata:
    truthdicts = json.load(jdata)


# ## Visual Inspection
# 
# We need to make sure these predictions and ground truth values make sense. Below we display an image along with it's ground truth and predicted VG vocabulary detections.
# 

fname = np.random.choice(predicts.keys())
print fname
truthlbls = truthdicts[fname]
predlbls = predicts[fname]

img = Image.open(os.path.join(imgdir, fname))
display(img)

print("*" * 50 + "\nGround Truth labels:\n")
print_list_to_columns(truthlbls)

print("*" * 50 + "\nPredicted labels:\n")
print_list_to_columns(predlbls[0:10],items_per_row=6)


# ## Get GloVe vectors
# 
# The GloVe vector file is structure: each line is a vector for one word with the word as the first part of the line. Each element of the line is sperated with a space. One line examples: "dog 2.342 1.233 2.454"
# 

glove_vector_file = os.path.join(wordvecs_dir, "glove.6B.200d.txt")


f=open(glove_vector_file,'r')

glove_vec_dic = {}
for i in f.read().splitlines():
    t=i.split()
    tv=[]
    for j in t[1:]:
        tv.append(float(j))
    glove_vec_dic[t[0]]=tv

f.close()


# ## Get ESP vocab vectors
# 
# Gather all of the words in ESP vocabulary and then create a matrix with the word vectors (from the GloVe model) of the ESP vocabulary. All ESP words are in the GloVe model
# 

esp_dir = "/data/fs4/datasets/espgame"
data = np.load(os.path.join(esp_dir,"espgame-inria.npz"))
esp_vocab = data["D"] # ESP vocabulary
yTe = data ["yTe"] # ESP onehot matrix
testlist = data['testlist'] # ESP test images


esp_vocab_vec = np.zeros((len(esp_vocab),len(glove_vec_dic.values()[0])))


for i in range(len(esp_vocab)):
    esp_vocab_vec[i]=glove_vec_dic[esp_vocab[i]]


# ## Get VisualGenome vocab vectors
# 
# Gather all of the words in VG vocabulary and then create a matrix with the word vectors (from the GloVe model) of the VG vocabulary. Note: 134 out of 4688 VG words not in the GloVe model. These words are in the VG word vector matrix with 0's.
# 

with open("/data/fs4/home/justinm/andrew-attalos/andreww/VG-object-regions-dicts.json.txt") as jdata:
    vgg_json_vocab = json.load(jdata)

vgg_vocab=[]
for i in vgg_json_vocab['token_to_idx'].keys():
    vgg_vocab.append(str(i))


temp_vgg_vocab = []
for i in vgg_vocab:
    if i in glove_vec_dic.keys():
        temp_vgg_vocab.append(i)
    #if i % 500 == 0:
    #    print i, ' out of ', len(vgg_vocab)
vgg_vocab_not_in_glove = list(set(vgg_vocab)-set(temp_vgg_vocab))[:]


vgg_vocab_vec = np.zeros((len(vgg_vocab),len(glove_vec_dic.values()[0])))
print len(vgg_vocab)
print len(temp_vgg_vocab)
print len(vgg_vocab)-len(temp_vgg_vocab)
print vgg_vocab_vec.shape


for i in range(len(vgg_vocab)):
    if vgg_vocab[i] in vgg_vocab_not_in_glove:
        vgg_vocab_vec[i,:]=0
    else:
        vgg_vocab_vec[i]=glove_vec_dic[vgg_vocab[i]]


# ## Do some comparison between ESP, VisualGenome, and GloVe vocab
# 
# 14 ESP words not in VG vocabulary and was 1.7% (167 out of 9774 tags) of the total number of tags in ESP's test images
# 

esp_not_in_glove = []
esp_not_in_vgg = []
for i in esp_vocab:
    if i not in vgg_vocab:
        esp_not_in_vgg.append(i)
        print i, " not in VG vocab"
    if i not in glove_vec_dic.keys():
        esp_not_in_glove.append(i)
        print "\t",i, " not in GloVe"


# Get the word not in VG and index of the image in testlist
esp_gt_not_in_vgg={}
for i in esp_not_in_vgg:
    idxword = np.argwhere(esp_vocab==i)
    idxsimg = np.nonzero(yTe[:,idxword])[0]
    if len(idxsimg) != 0:
        esp_gt_not_in_vgg[i] = idxsimg


# print ESP word not in VG and the image index
for i in esp_gt_not_in_vgg.keys():
    print i, esp_gt_not_in_vgg[i]


# print words in VG but not in ESP
vgg_vocab_not_in_glove = list(set(vgg_vocab)-set(temp_vgg_vocab))[:]
print len(vgg_vocab_not_in_glove), " words in VG but not in GloVe"
print vgg_vocab_not_in_glove


# ## Get Correlation matrix of ESP vocab vectros with VisualGenome word vectors
# 

# convert ESP word vectors to unit vectors
esp_vocab_vec_norm = np.divide(esp_vocab_vec.T,np.linalg.norm(esp_vocab_vec,axis=1)).T


# convert VG word vectors to unit vectors. Do to the VG words not in GloVe, those vectors have length 0.

vgg_norm = np.linalg.norm(vgg_vocab_vec,axis=1)
vgg_vocab_vec_norm = vgg_vocab_vec.copy()

# Have to convert each word vector one at a time do to some word vector lengths of 0
for i in xrange(len(vgg_norm)):
    if vgg_norm[i] != 0:
        vgg_vocab_vec_norm[i] = np.divide(vgg_vocab_vec[i],vgg_norm[i])


# create the correlation matrix between the ESP word vectors and VG word Vectors
esp_vgg_corr = np.dot(esp_vocab_vec_norm,vgg_vocab_vec_norm.T)


# Testing the correlation of the same word has dot product of 1
print 'airplane' in esp_vocab
print 'airplane' in vgg_vocab
print esp_vgg_corr[np.argwhere(esp_vocab==('airplane'))[0][0],vgg_vocab.index('airplane')]


# reduce predicted tags to the top 5 without repetitive words and <UNK> tokens
# the predicted tags are in order from most confidence to least
reduced_predicts={}
top_n_words = 5
lessthan5 = 0
print "number of test images = ",len(predicts.keys())
for i in predicts.keys():
    words = predicts[i]
    newl = [] # new list of top n predicted words 
    norepeats = [] # complete list of words without repeated words and <UNK> tokens
    for j in words:
        if len(newl) >= top_n_words and j not in norepeats:
            norepeats.append(j)
        elif j != "<UNK>" and j not in newl:# and j in vocab:
            newl.append(j)
            norepeats.append(j)
    reduced_predicts[i] = newl
    if len(newl) < top_n_words:
        lessthan5+=1
        #print newl, len(newl)
        #print norepeats
print lessthan5, "/", len(predicts.keys()), " have less than 5 predicted words that are in the ESP-Games dictionay"


# list predicted tags that are not in ESP vocab
no_pred_words_in_esp_dict = {}
no_pred_words_in_glv_dict = {}
for i in reduced_predicts.keys():
    no_pred_words_in_esp = []
    no_pred_words_in_glv = []
    for j in reduced_predicts[i]:
        if str(j) not in esp_vocab:
            no_pred_words_in_esp.append(j)
        if str(j) in vgg_vocab_not_in_glove:
            no_pred_words_in_glv.append(j)
    if len(no_pred_words_in_esp)>0:
        no_pred_words_in_esp_dict[i] = no_pred_words_in_esp
        print i,no_pred_words_in_esp," predictions not in esp vocab"
    if len(no_pred_words_in_glv)>0:
        no_pred_words_in_glv_dict[i] = no_pred_words_in_esp
        print i,no_pred_words_in_glv," predictions not in GloVe dictionary"

#print len(no_pred_words_in_esp_dict.keys())," number of images that have a top 5 tag not in esp vocab"
#print len(no_pred_words_in_glv_dict.keys())," number of images that have a top 5 tag not in glove dictionary"


# Create onehot matrix of predicted tags. 
# Also translating VG words not in ESP vocab to the highest correlated word in ESP vocab
predict_arr = np.zeros(yTe.shape,dtype=np.int)
x_corpa_word_map={} # image as key, value is [VG word, correlated ESP word, dot product of words]
for i in reduced_predicts.keys():
    idximg = np.argwhere(testlist==i)[0][0]
    for j in reduced_predicts[i]:
        wordmap=[]
        idxwrd = np.argmax(esp_vgg_corr[:,vgg_vocab.index(j)])
        predict_arr[idximg,idxwrd] = 1
        if str(j) != esp_vocab[idxwrd]:
            wordmap.append([str(j),
                            esp_vocab[idxwrd],
                            esp_vgg_corr[np.argmax(esp_vgg_corr[:,vgg_vocab.index(j)]),vgg_vocab.index(j)]])
        if esp_vgg_corr[np.argmax(esp_vgg_corr[:,vgg_vocab.index(j)]),vgg_vocab.index(j)] == 0:
            print 'help, vg word has no correlation with a word in ESP \t',wordmap[-1] # fortunatelly this did not happen
    if len(wordmap) != 0:
        x_corpa_word_map[i]=wordmap


# Reduce repeated word correlations
reduced_x_corpa_word_map = []
for i in x_corpa_word_map.keys():
    for j in x_corpa_word_map[i]:
        if j not in reduced_x_corpa_word_map:
            reduced_x_corpa_word_map.append(j)
print len(reduced_x_corpa_word_map), "VG words had to be correlated to an ESP word"


# print the mapping from VG word to the highest correlated ESP word
print "[predicted word in vg, word with highest correlation to esp word, correlation]"
for i in reduced_x_corpa_word_map[:]:
    print i


# print ESP word not in VG and the image index
num_tags_not_vgg_in_esp_gt = 0
for i in esp_gt_not_in_vgg.keys():
    print i, esp_gt_not_in_vgg[i]
    num_tags_not_vgg_in_esp_gt += len(esp_gt_not_in_vgg[i])

print num_tags_not_vgg_in_esp_gt, " number of times an ESP word not in VG but was used in the ESP test images"
print np.sum(yTe), " number of ESP ground truth tags"
print num_tags_not_vgg_in_esp_gt/np.sum(yTe) * 100, "%"


# ## Running the evaluation code
# 

# Evaluate the ground truth tags (yTe) to the top n predicted tags (predict_arr)
[precision,recall,f1] = octave.evaluate(yTe.T, predict_arr.T, 5)
print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))


# These performance metrics reults are not very good and are not representative of the model's performance judging from visual inspection of a small sample. To overcome this, we'll look at a simplified version of the metrics. For precision, we'll divid the size of the intersection over the size of the union. 
# 
# $$\text{precision} = \frac{|\{\text{test}\} \cap \{\text{predicted}\}|}{|\{\text{test}\} \cup \{\text{predicted}\}|}$$
# 
# Recall evaluates the correctly predicted labels over the set of all correct labels. Since a false negative here is the absence of a predicted label and a false positive is the addition of a label we'll look at the size of the intersection of the actual and predicted over the size of the actual labels for each.
# 
# $$\text{recall} = \frac{|\{\text{test}\} \cap \{\text{predicted}\}|}{|\{\text{test}\}|}$$
# 
# And finally the F-1 score
# 
# $$\text{f1} = 2 \cdot \frac{(\text{precision} \cdot\text{recall})}{(\text{precision} + \text{recall})}$$
# 


intersection = 0
union = 0
recall_denom = 0
'''
for fname in gt_set.iterkeys():
    a = gt_set[fname]
    b = prediction_set[fname]
    intersection += len(a.intersection(b))
    union += len(a.union(b))
    recall_denom += len(a)
'''

################################################

for fname in truthdicts.keys():
    a = set(truthdicts[fname])
    b = set(reduced_predicts[fname])
    intersection += len(a.intersection(b))
    union += len(a.union(b))
    recall_denom += len(a)

intersection=float(intersection)

###############################################

precision = intersection / union
recall = intersection / recall_denom
f1 = 2 * (precision * recall) / (precision + recall)
print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))


# # Which Layers of SSD Predict Which Tags and Object Sizes?
# 

# ### Setup and load model
# 

# Set caffe root, label map, model definition, and model weights
#caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
caffe_root = '/work/caffe'
voc_labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
model_def = 'models/VGGNet/VOC0712/SSD_300x300/test.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

# Set confidence threshold (0-1) for object detection
conf_thresh = 0.6

# Set number of images to search through. Max for VOC0712 is 4952
max_images = 4952

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# ### Find layer that activates to each high confidence object in each image. Record details 
# 

# keep track of the following
labels = []    # object label
heights = []   # object height (fraction of image height)
widths = []    # object width (fraction of image width)
layers = []    # network layer that found object with high confidence

# iterate through each image
for i in range(0, max_images + 1):
    net.forward()
    
    detections = net.blobs['detection_out'].data
    
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(voc_labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    
    if top_conf.size == 0:
        continue
        
    # iterate through each confident label found by SSD
    for k in range(0,top_conf.size):
        top_conf_k = top_conf[k]
        
        # find index of this confidence in output of mbox_conf_softmax
        conf_softmax = net.blobs['mbox_conf_softmax'].data
        tconf_softmax_idx = np.where(conf_softmax == top_conf_k)

        # use index from  mbox_conf_softmax to find value in mbox_conf_reshape
        conf_reshape = net.blobs['mbox_conf_reshape'].data
        tconf_reshape = conf_reshape[tconf_softmax_idx[0], tconf_softmax_idx[1], tconf_softmax_idx[2]]

        # use value from mbox_conf_reshape to find input layer (after perm) and index of layer
        conv_layers = ["conv4_3_norm_mbox_conf_perm", "fc7_mbox_conf_perm", "conv6_2_mbox_conf_perm", "conv7_2_mbox_conf_perm", "conv8_2_mbox_conf_perm",  "pool6_mbox_conf_perm"]

        for layer in conv_layers:
            layer_data = net.blobs[layer].data
            if tconf_reshape in layer_data:
                conf_perm_name = layer
                layers.append(layer[:-15])
                
                width = top_xmax[k] - top_xmin[k]
                height = top_ymax[k] - top_ymin[k]

                widths.append(width)
                heights.append(height)
                labels.append(top_labels[k])


# ### Summarize detecting layers and labels detected
# 

layer_counts = dict((x, layers.count(x)) for x in layers)
layer_dict = dict((x, i) for i, x in enumerate(layer_counts))

print "Layer Counts", str(layer_counts)
print
label_counts = dict((str(x), labels.count(x)) for x in labels)
label_dict = dict((x, i) for i, x in enumerate(label_counts))
print "Label Counts", str(label_counts)

# create 2d array with which to make a label/layer heat map
heat_map = np.zeros((len(label_counts), len(layer_counts)))

for i in range(0, len(layers)):
    lab_idx = label_dict.get(labels[i])
    lay_idx = layer_dict.get(layers[i])
    heat_map[lab_idx, lay_idx] += 1.0 / label_counts.get(labels[i])


# ### Produce heatmap of Layers vs Labels. Labels are each normalized by total count of that label
# 

plt.pcolor(heat_map, cmap='YlOrRd')
plt.colorbar()

label_val_shift = [x + 0.5 for x in label_dict.values()]
layer_val_shift = [x + 0.5 for x in layer_dict.values()]

plt.title("SSD Layers Responsible for High Confidence (>60%) Tags of Each Label.\n(Labels Normalized by Count of Each Type)")
plt.xlabel("SSD Layer")
plt.ylabel("VOC0712 Label")
plt.yticks(label_val_shift, label_dict.keys())
plt.xticks(layer_val_shift, layer_dict.keys(), rotation='vertical')
plt.show()


# ### Produce heatmap of Layers vs Object Size
# 

# calculate "area" of each object. Image sizes unknown here.
areas = []
for i in range(0, len(widths)):
    areas.append(widths[i] * heights[i])
    
# create 2d array with which to make a label/layer heat map
k = 500 # number of bins
heat_map_area = np.zeros((k, len(layer_counts)))

for i in range(0, len(areas)):
    area_idx = int(areas[i] * k) - 1
    lay_idx = layer_dict.get(layers[i])
    heat_map_area[area_idx, lay_idx] += 1.0
    
plt.pcolor(heat_map_area, cmap='YlOrRd')
plt.colorbar()

# label_val_shift = [x + 0.5 for x in label_dict.values()]
layer_val_shift = [x + 0.5 for x in layer_dict.values()]

plt.title("SSD Layers Responsible for High Confidence (>60%) vs Label Area.\n(note that layers are not in order)")
plt.xlabel("SSD Layer")
plt.ylabel("Bin (Max Area = 1)")
# plt.yticks(label_val_shift, label_dict.keys())
plt.xticks(layer_val_shift, layer_dict.keys(), rotation='vertical')
plt.show()





# ## Get the outputs
# 
# Lab41 trained a DenseCap model using Visual Genome's (hereafter VG) object tags. Here we run the model to produce tags for all images to get evaluatation metrics.
# 
# Note: Out of the box, `run_model.lua` will output a copy of the image(s) and a result.json file. We've modified the file to only write the json output for a 2x speed improvement -- form 1 sec/image to 0.5 sec/image.
# 

# ```bash
# cd densecap/
# th run_model.lua -checkpoint densecap/models/one-hot/20160909.checkpoint.t7 -input_batch_dir visual-genome/batches/[1,2,3] -output_vis_dir densecap/predictions/one-hot/[1,2,3]```
# 

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
sys.path.append("attalos/")
import json
import h5py
import numpy as np
from PIL import Image
from oct2py import octave
from scripts import loaders
from collections import Counter
from IPython.core.display import display
#from attalos.evaluation.evaluation import Evaluation
from scripts import load_combine_json_dir, print_list_to_columns

octave.addpath('../')


# Keep all the file paths here so we don't have to hunt for them later
# 

# The json file from feedforward 
json_dir = "densecap/predictions/one-hot/"

# directory with all source images
imgdir = "visual-genome/images/"

# preprocessed training data
object_hd5 = 'VG-object-regions.h5'
object_dict = 'VG-object-regions-dicts.json'


print("Loading predictions (This will take some time)...")
results = load_combine_json_dir(json_dir, loaders.load_output_json)
print("Done!")


# ## Training data is separated into two files
# 
# `VG-object-regions.h5` contains all images, labels, bounding boxes and other important information for training.
# `VG-object-regions-dicts.json` holds all the index mappings for images and word tokens. 
# 

# simple JSON data. Holds index mappings for images and word tokens
gt_map = loaders.json_load(object_dict)

# This HDF5 file is large (127 GB). We only want labels. We'll use the 
# json data to complete the map
with h5py.File(object_hd5, 'r') as hf:    
    # labels array 
    labels = np.array(hf.get('labels'))
    
    # We will use these indices to take slices of labels. 
    # The mapping is filename_to_idx -- img_to_first_box, img_to_last_box -- labels
    img_to_first_box = np.array(hf.get('img_to_first_box'))
    img_to_last_box = np.array(hf.get('img_to_last_box'))


# ### Label-Image mapping
# The labels and images are separated into several datasets across hdf5 and json files. The funciton below consumes part of the HDF5 file that we've loaded above, namely `labels`, `img_to_first_box`, and `img_to_last_box`. Now we can do things like pass a filename to get the corresponding ground truth labels.
# 
# **Important**: Lua is 1 indexed whereas Python is 0 indexed. For example:
# 
# #### Lua: 
# ```lua
# lua_table = {'a', 'b', 'c', 'd', 'e'}
# -- No syntactic sugar for lua slicing. 
# -- Here we just want the first 2 elements
# for n=1,2 do
#     print(lua_table[n])
# end 
# -- prints a b
# ```
# 
# #### Python:
# ```python
# python_list = ['a', 'b', 'c', 'd', 'e']
# 
# for i in [1,2]:
#     print(python_list[i]
# # prints b c
# ```
# Because of this, the word and image index mappings start at 1 so we need to subtract 1 from each. 
# 

def get_corresponding_labels(filename, reference_dict, labels_arr, first_box, last_box):
    """This function will do the mapping from filename to label
    
    Parameters
    ----------
    filename : str
        name of an image file. Just the os.path.basename of the file. e.g. '55.jpg'
    reference_dict : dict
        object regions dict used for model training
    labels_arr : numpy.array
        training lables array
    first_box : numpy.array
        img_to_first_box array from training data
    last_box : numpy.array
        img_to_last_box array from training data
    
    Returns
    -------
    numpy.array with label IDs
    """
    
    idx = reference_dict['filename_to_idx'][filename] - 1
    slice_start = first_box[idx]
    slice_end = last_box[idx]
    return labels[slice_start:slice_end]


# Here we're getting summary statistics for number of ground truth images, number of tokens, and number of result images.
# 

n_gt_images = len(gt_map['idx_to_filename'].items())
n_tokens = len(gt_map['token_to_idx'].items())
n_imgs = len(results.keys())

# model predicts 1000 boxes per image
n_pred_boxes = n_imgs * 1000
print("Tokens: {0:,}\nPrediction Images: {1:,}\nPredicted Boxes: {3:,}\n\nGround Truth Images: {2:,}".format(n_tokens, n_imgs, n_gt_images, n_pred_boxes))


# ## Visual Inspection
# 
# We need to make sure these predictions and ground truth values make sense. Below we display an image along with it's ground truth and predicted labels. We're not drawing the boxes here because DenseCap resizes all inputs. 
# 

# snag a random filename 
fname = np.random.choice(gt_map['filename_to_idx'].keys())

# run through the mapping function to get the labels
label_codes = get_corresponding_labels(fname, gt_map, labels, img_to_first_box, img_to_last_box)

# open the image for display
img = Image.open(os.path.join(imgdir, fname))
display(img)

# Collect all terms and combine
terms = []
rows, cols = label_codes.shape
for row in xrange(rows):
    obj = ''
    words = []
    for col in xrange(cols):
        # The word mapping stores the token IDs as strings
        val_num = label_codes[row, col]
        val = str(val_num)
        
        if (val_num > 0) and gt_map['idx_to_token'].has_key(val):
            words.append(gt_map['idx_to_token'][val])
        terms.append(" ".join(words))
        
print("*" * 50 + "\nGround Truth labels:\n")

# function to print columns instead of one long list
print_list_to_columns(list(set(terms)))

pred_labels = list(set(map(lambda x : " ".join(x['names']), results[fname])))
print("*" * 50 + "\nPredicted labels:\n")
print_list_to_columns(pred_labels)


def get_tokens_and_score(bbox_predictions, token_to_idx):
    """This function will gather the unique tokens and 
    the best score assigned to each
    
    Parameters
    ----------
    bbox_predictions : dict
        dictionary with predictions from DenseCap. Each 
        item should have a 'score' and 'names'
    token_to_idx : dict
        mapping of words to an index/unique number
        
    Returns
    --------
    Dictionary with {term: score} mappings, dictionary 
    with missed words and counts {word, count}
    """
    out = {}
    missed = Counter()
    for item in bbox_predictions:
        name = item['names']
        
        # each name should be a list with 1 item 
        err = "Too maney tokens! Should have 1 but this one has {0}".format(len(name))
        assert len(name) == 1, err
        
        name = name[0]
        
        # attempt to squish the token into existing vocab
        # found examples of 'power line' but the existing 
        # vocab only had 'powerline'
        name_stripped = name.replace(' ', '')
        
        if token_to_idx.has_key(name):
            # 1-indexing to 0-indexing densecap/preprocessing.py 44
            name_idx = token_to_idx[name] - 1
    
        elif token_to_idx.has_key(name_stripped):
            # 1-indexing to 0-indexing densecap/preprocessing.py 44
            name_idx = token_to_idx[name_stripped] - 1 
            
        else:
            # This keeps track of the terms out of 
            # training vocabulary. 
            missed[name] += 1
            continue 
        
        score = item['score']
        if out.has_key(name_idx):
            cur_score = out[name_idx]
            if score > cur_score:
                out[name_idx] = score
        else:
            out[name_idx] = score
    return out, missed  


# array needs to be Images X Tokens
prediction_arr = np.zeros(shape=(n_gt_images, n_tokens), dtype=np.float32)

# dictionary to hold the set of unique word predictions
# we'll use this later for precision and recall numbers 
prediction_set = {}

# get the best score for each unique token
for fname, preds in results.iteritems():
    if gt_map['filename_to_idx'].has_key(fname):
        
        # 1-indexing to 0-indexing densecap/preprocessing.py 44
        idx = gt_map['filename_to_idx'][fname] - 1 
        
        scores, _ = get_tokens_and_score(preds, gt_map['token_to_idx'])
        
        # collect the tokens and scores
        tokens = []
        
        # indexing accounted for in get_tokens_and_score(...) functions
        for word_idx, score in scores.iteritems():
            prediction_arr[idx, word_idx] =  score
            tokens.append((score, word_idx))
        
        prediction_set[fname] = tokens


# ## Creating Test Evalutation Array
# 
# Here we're loading the ground truth data into an array the same shape as the prediction array.
# 

gt_array = np.zeros_like(prediction_arr)

gt_set = {}

# Loop over the ground truth map to fill in the 
# ground truth array 
for fname, idx in gt_map['filename_to_idx'].iteritems():
    gt_label_arr = get_corresponding_labels(fname, gt_map,                                             labels, img_to_first_box,                                             img_to_last_box)
    # need to go through each box and find non-zero tokens
    rows, cols = gt_label_arr.shape
    
    # 1-indexing to 0-indexing densecap/preprocessing.py 44
    idx -= 1
    
    tokens = set()
    for row in xrange(rows):
        vals = []
        for col in xrange(cols):
            val = gt_label_arr[row, col]
            if val > 0:
                # 1-indexing to 0-indexing densecap/preprocessing.py 44
                val -= 1
                vals.append(val)
        # We only trained on the last word of the truth, so only use the last word
        try:
            final_val = vals[-1]
        except IndexError:
            continue
        tokens.add(final_val)
        # To one-hot
        gt_array[idx, final_val] = 1
    gt_set[fname] = tokens


# ## Running the evaluation code
# 

# These performance metrics reults are not very good and may not be representative of the model's performance judging from visual inspection of a small sample. To overcome this, we'll look at a different version of the metrics. For precision, we'll divide the size of the intersection over the size of the union. 
# 
# $$\text{precision} = \frac{|\{\text{test}\} \cap \{\text{predicted}\}|}{|\{\text{test}\} \cup \{\text{predicted}\}|}$$
# 
# Recall evaluates the correctly predicted labels over the set of all correct labels. Since a false negative here is the absence of a predicted label and a false positive is the addition of a label we'll look at the size of the intersection of the actual and predicted over the size of the actual labels for each.
# 
# $$\text{recall} = \frac{|\{\text{test}\} \cap \{\text{predicted}\}|}{|\{\text{test}\}|}$$
# 
# And finally the F-1 score
# 
# $$\text{f1} = 2 \cdot \frac{(\text{precision} \cdot\text{recall})}{(\text{precision} + \text{recall})}$$
# 

def get_top_n(predictions, n=5):
    sorted_preds = sorted(predictions)
    ret_val = []
    for score, word in reversed(sorted_preds):
        # Only take N items
        if len(ret_val) >= n:
            break
        # Do not take duplicates
        if word in ret_val:
            continue
        else:
            ret_val.append(word)
    return ret_val


intersection = 0
union = 0
recall_denom = 0

for fname in gt_set.iterkeys():
    a = gt_set[fname]
    b = prediction_set[fname]
    b = get_top_n(b, n=5)
    intersection += len(a.intersection(b))
    union += len(a.union(b))
    recall_denom += len(a)

precision = intersection / union
recall = intersection / recall_denom
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))


# ## Get the outputs
# 
# Lab41 trained a DenseCap model using Visual Genome's (hereafter VG) object tags. Here we run the model to produce tags for all images to get evaluatation metrics.
# 
# Note: Out of the box, `run_model.lua` will output a copy of the image(s) and a result.json file. We've modified the file to only write the json output for a 2x speed improvement -- form 1 sec/image to 0.5 sec/image.
# 

# ```bash
# cd densecap/
# th run_model.lua -checkpoint densecap/models/single-object/checkpoint.t7 -input_batch_dir visual-genome/batches/ -output_vis_dir densecap/predictions/single-object```
# 

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append("attalos/")
from attalos.evaluation.evaluation import Evaluation
from oct2py import octave
octave.addpath('attalos/attalos/evaluation/')
import os
import json
import h5py
import numpy as np
from PIL import Image
from scripts import loaders
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
from IPython.core.display import display
from IPython.core.pylabtools import figsize
from scripts import load_combine_json_dir, print_list_to_columns

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')


# Keep all the file paths here so we don't need to hunt for them later
# 

# directory with result_%i.json files form DenseCap inference
json_dir = "densecap/predictions/single-object"

# The training data
object_hd5 = 'VG-object-regions.h5'
object_dict = 'VG-object-regions-dicts.json'

# base image directory
imgdir = "visual-genome/images/"


print("Loading predictions (This will take some time)...")
results = load_combine_json_dir(json_dir, loaders.load_output_json)
print("Done!")


# ## Training data is separated into two files
# 
# `VG-object-regions.h5` contains all images, labels, bounding boxes and other important information for training.
# `VG-object-regions-dicts.json` holds all the index mappings for images and word tokens. 
# 

# simple JSON data. Holds index mappings for images and word tokens
gt_map = loaders.json_load(object_dict)

# This HDF5 file is large (127 GB). We only want labels. We'll use the 
# json data to complete the map
with h5py.File(object_hd5, 'r') as hf:    
    # labels array 
    labels = np.array(hf.get('labels'))
    
    # We will use these indices to take slices of labels. 
    # The mapping is filename_to_idx -- img_to_first_box, img_to_last_box -- labels
    img_to_first_box = np.array(hf.get('img_to_first_box'))
    img_to_last_box = np.array(hf.get('img_to_last_box'))    
    


# ### Label-Image mapping
# The labels and images are separated into several datasets across hdf5 and json files. The funciton below consumes part of the HDF5 file that we've loaded above, namely `labels`, `img_to_first_box`, and `img_to_last_box`. Now we can do things like pass a filename to get the corresponding ground truth labels.
# 
# **Important**: Lua is 1 indexed and Python is 0 indexed. Because of this, the word and image index mappings start at 1 so we need to subtract 1 from each to work in Python. 
# 

def get_corresponding_labels(filename, reference_dict, labels_arr, first_box, last_box):
    """This function will do the mapping from filename to label
    
    Parameters
    ----------
    filename : str
        name of an image file. Just the os.path.basename of the file. e.g. '55.jpg'
    reference_dict : dict
        object regions dict used for model training
    labels_arr : numpy.array
        training lables array
    first_box : numpy.array
        img_to_first_box array from training data
    last_box : numpy.array
        img_to_last_box array from training data
    
    Returns
    -------
    numpy.array with label IDs
    """
    
    idx = reference_dict['filename_to_idx'][filename] - 1
    slice_start = first_box[idx]
    slice_end = last_box[idx]
    return labels[slice_start:slice_end]


n_gt_images = len(gt_map['idx_to_filename'].items())
n_tokens = len(gt_map['token_to_idx'].items())
n_imgs = len(results.keys())

# model predicts 1000 boxes per image
n_pred_boxes = n_imgs * 1000
print("Tokens: {0:,}\nPrediction Images: {1:,}\nPredicted Boxes: {3:,}\n\nGround Truth Images: {2:,}".format(n_tokens, n_imgs, n_gt_images, n_pred_boxes))


# ## Visual Inspection
# 
# We need to make sure these predictions and ground truth values make sense. Below we display an image along with it's ground truth and predicted detections. 
# 

fname = np.random.choice(gt_map['filename_to_idx'].keys())
label_codes = get_corresponding_labels(fname, gt_map, labels, img_to_first_box, img_to_last_box)

img = Image.open(os.path.join(imgdir, fname))
display(img)

terms = []
rows, cols = label_codes.shape
for row in xrange(rows):
    obj = ''
    words = []
    for col in xrange(cols):
        val_num = label_codes[row, col]
        val = str(val_num)
        
        if (val_num > 0) and gt_map['idx_to_token'].has_key(val):
            words.append(gt_map['idx_to_token'][val])
        terms.append(" ".join(words))
        
print("*" * 50 + "\nGround Truth labels:\n")
print_list_to_columns(list(set(terms)))
pred_labels = list(set(map(lambda x : " ".join(x['names']), results[fname])))
print("*" * 50 + "\nPredicted labels:\n")
print_list_to_columns(pred_labels)


# ## Creating the Predictions Array
# Now we'll load up the prediction array with the corresponding score. The output array will be $I \times T$ where $I$ is the number of images and $T$ is the number of tokens.
# 

def get_tokens_and_score(bbox_predictions, token_to_idx):
    """This function will gather the unique tokens and 
    the best score assigned to each
    
    Parameters
    ----------
    bbox_predictions : dict
        dictionary with predictions from DenseCap. Each 
        item should have a 'score' and 'names'
    token_to_idx : dict
        mapping of words to an index/unique number
        
    Returns
    --------
    Dictionary with {term: score} mappings, dictionary 
    with missed words and counts {word, count}
    """
    out = {}
    missed = Counter()
    for item in bbox_predictions:
        name = item['names']
        
        # each name should be a list with 1 item 
        err = "Too maney tokens! Should have 1 but this one has {0}".format(len(name))
        assert len(name) == 1, err
        
        name = name[0]
        
        # attempt to squish the token into existing vocab
        name_stripped = name.replace(' ', '')
        
        if token_to_idx.has_key(name):
            # 1-indexing to 0-indexing densecap/preprocessing.py 44
            name_idx = token_to_idx[name] - 1
    
        elif token_to_idx.has_key(name_stripped):
            # 1-indexing to 0-indexing densecap/preprocessing.py 44
            name_idx = token_to_idx[name_stripped] - 1 
            
        else:
            # This keeps track of the terms out of 
            # training vocabulary. 
            missed[name] += 1
            continue 
        
        score = item['score']
        if out.has_key(name_idx):
            cur_score = out[name_idx]
            if score > cur_score:
                out[name_idx] = score
        else:
            out[name_idx] = score
    return out, missed
    


# array needs to be Images X Tokens
prediction_arr = np.zeros(shape=(n_gt_images, n_tokens), dtype=np.float32)

prediction_set = {}

# Sometimes the model predicts compound phrases like 
# "red hair" and the training data may only have red OR hair
# but not combined. We believe it may be a remnent of using
# a LSTM instead of a true single class predictor
oov = Counter()

# get the best score for each unique token
for fname, preds in results.iteritems():
    if gt_map['filename_to_idx'].has_key(fname):
        
        # 1-indexing to 0-indexing densecap/preprocessing.py 44
        idx = gt_map['filename_to_idx'][fname] - 1 
        
        scores, missed = get_tokens_and_score(preds, gt_map['token_to_idx'])
        oov.update(missed)
        
        # collect unique tokens for manual precision recall
        tokens = set()
        
        for word_idx, score in scores.iteritems():
            prediction_arr[idx, word_idx] =  score
            tokens.add(word_idx)
        
        prediction_set[fname] = tokens


# There is a discrepancy between the training labes and the ground truth set. Here we plot tokens from the prediction set that were not in the ground truth. 
# 

figsize(12, 4)
top = 10
width = 0.35
ind = np.arange(top)
values = map(lambda x: x[1], oov.most_common(top))
percent = map(lambda x : x / n_pred_boxes * 100, values)
lables = map(lambda x: "\n".join(x[0].split(' ')), oov.most_common(top))
fig, ax = plt.subplots()
ax.bar(ind + width, percent, width=width)
ax.set_ylabel("Percent of Predicted Boxes")
ax.set_title("Top 10 Out-of-Vocabulary Words")
ax.set_xticks(ind + width)
ax.set_xticklabels(lables)
ax.set_xlabel("{0:0.2%} of tags are \"out-of-vocab.\"".format(sum(oov.values()) / n_pred_boxes))
plt.show()


# ## Creating Test Evalutation Array
# 
# Here we're loading the ground truth data into an array the same shape as the prediction array.
# 

gt_array = np.zeros_like(prediction_arr)

gt_set = {}

# Loop over the ground truth map to fill in the 
# ground truth array 
for fname, idx in gt_map['filename_to_idx'].iteritems():
    gt_label_arr = get_corresponding_labels(fname, gt_map,                                             labels, img_to_first_box,                                             img_to_last_box)
    # need to go through each box and find non-zero tokens
    rows, cols = gt_label_arr.shape
    
    # 1-indexing to 0-indexing densecap/preprocessing.py 44
    idx -= 1
    
    tokens = set()
    for row in xrange(rows):
        for col in xrange(cols):
            val = gt_label_arr[row, col]
            if val > 0:
                # 1-indexing to 0-indexing densecap/preprocessing.py 44
                val -= 1
                tokens.add(val)
                # To one-hot
                gt_array[idx, val] = 1
    gt_set[fname] = tokens


# ## Running the evaluation code
# 

[precision,recall,f1] = octave.evaluate(gt_array.T, np.abs(prediction_arr).T, 5)
print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))


# These performance metrics reults are not very good and are not representative of the model's performance judging from visual inspection of a small sample. To overcome this, we'll look at a simplified version of the metrics. For precision, we'll divid the size of the intersection over the size of the union. 
# 
# $$\text{precision} = \frac{|\{\text{test}\} \cap \{\text{predicted}\}|}{|\{\text{test}\} \cup \{\text{predicted}\}|}$$
# 
# Recall evaluates the correctly predicted labels over the set of all correct labels. Since a false negative here is the absence of a predicted label and a false positive is the addition of a label we'll look at the size of the intersection of the actual and predicted over the size of the actual labels for each.
# 
# $$\text{recall} = \frac{|\{\text{test}\} \cap \{\text{predicted}\}|}{|\{\text{test}\}|}$$
# 
# And finally the F-1 score
# 
# $$\text{f1} = 2 \cdot \frac{(\text{precision} \cdot\text{recall})}{(\text{precision} + \text{recall})}$$
# 

intersection = 0
union = 0
recall_denom = 0

for fname in gt_set.iterkeys():
    a = gt_set[fname]
    b = prediction_set[fname]
    intersection += len(a.intersection(b))
    union += len(a.union(b))
    recall_denom += len(a)

precision = intersection / union
recall = intersection / recall_denom
f1 = 2 * (precision * recall) / (precision + recall)
print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))


