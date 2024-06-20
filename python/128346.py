import os
import sys
import yaml
import tensorflow as tf
import sys

import hashlib


# # check paths below....
# 

# This is needed to display the images.

sys.path.append("/home/sebastian/Udacity/SDC-System-Integration/classifier/models") # point to your tensorflow dir
sys.path.append("/home/sebastian/Udacity/SDC-System-Integration/classifier/models/slim")
sys.path.append("/home/sebastian/Udacity/SDC-System-Integration/classifier/models/object_detection/")

# data directory containing rgb folder and train.yaml
PATH_TO_DATA = '/home/sebastian/Udacity/SDC-System-Integration/classifier/data' 
TF_RECORD_TRAIN_PATH =PATH_TO_DATA+'/train.record'
TF_RECORD_TEST_PATH =PATH_TO_DATA+'/test.record'



label_dict =  {
   "Green" : 1,
   "Red" : 2,
   "GreenLeft" : 3,
   "GreenRight" : 4,
   "RedLeft" : 5,
   "RedRight" : 6,
   "Yellow" : 7,
   "off" : 8,
   "RedStraight" : 9,
   "GreenStraight" : 10,
   "GreenStraightLeft" : 11,
   "GreenStraightRight" : 12,
   "RedStraightLeft" : 13,
   "RedStraightRight" : 14
   }





from object_detection.utils import dataset_util

def get_all_labels(input_yaml, riib=False):
    """ Gets all labels within label file
    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    :param input_yaml: Path to yaml file
    :param riib: If True, change path to labeled pictures
    :return: images: Labels for traffic lights
    """
    images = yaml.load(open(input_yaml, 'rb').read())

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), images[i]['path']))
        if riib:
            images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
            images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
            images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
            for box in images[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return images

def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.
    height = 720 # Image height
    width = 1280 # Image width
    filepath = example['path'] 

    filename = filepath.split('/').pop() # Filename of the image. Empty if image is not from file

    with tf.gfile.GFile(filepath, 'rb') as fid:
        encoded_image_data = fid.read()

    key = hashlib.sha256(encoded_image_data).hexdigest()
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    for box in example['boxes']:
        if box['occluded']:
            continue
        else:
            xmins.append(box['x_min']/width)
            xmaxs.append(box['x_max']/width)
            ymins.append(box['y_min']/height)
            ymaxs.append(box['y_max']/height)
            classes_text.append(box['label'])
            classes.append(label_dict.get(box['label']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example




dataset_train = get_all_labels(PATH_TO_DATA + '/train.yaml')


dataset_test = get_all_labels(PATH_TO_DATA + '/test.yaml')
# fix for test dataset
for data in dataset_test:
    data['path'] = PATH_TO_DATA+'/rgb/test/'+data['path'].split('/').pop()


#Write one big file ..around 7GB

writer_train = tf.python_io.TFRecordWriter(TF_RECORD_TRAIN_PATH)
for example in dataset_train:
    train = create_tf_example(example)
    writer_train.write(train.SerializeToString())
writer_train.close()


writer_test = tf.python_io.TFRecordWriter(TF_RECORD_TEST_PATH)
for example in dataset_test:
    test = create_tf_example(example)
    writer_test.write(test.SerializeToString())
writer_test.close()


dataset_train[0]['path'].split('/').pop()


test1 = create_tf_example(dataset_train[100])


with tf.gfile.GFile(dataset_train[100]['path'], 'rb') as fid:
    encoded_image_data = fid.read()


encoded_image_data[:100]


strings = 'png'
strings.encode('utf8')





