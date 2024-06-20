import collections
import commands
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_video_number_of_frames
from skimage.transform import resize
import cv2
import random

VIDEO_PATH = "/home/cabaf/AnetVideos/"
get_ipython().magic('matplotlib inline')


# ***Release summary***
# 

with open("activity_net.v1-2.json", "r") as fobj:
    data = json.load(fobj)

database = data["database"]
taxonomy = data["taxonomy"]
version = data["version"]

all_node_ids = [x["nodeId"] for x in taxonomy]
leaf_node_ids = []
for x in all_node_ids:
    is_parent = False
    for query_node in taxonomy:
        if query_node["parentId"]==x: is_parent = True
    if not is_parent: leaf_node_ids.append(x)
leaf_nodes = [x for x in taxonomy if x["nodeId"] in  leaf_node_ids]

vsize = commands.getoutput("du %s -lhs" % VIDEO_PATH).split("/")[0]
with open("../video_duration_info.json", "r") as fobj:
    tinfo = json.load(fobj)
total_duration = sum([tinfo[x] for x in tinfo])/3600.0

print "ActivityNet %s" % version
print "Total number of videos: %d" % len(database)
print "Total number of nodes in taxonomy: %d" % len(taxonomy)
print "Total number of leaf nodes: %d" % len(leaf_nodes)
print "Total size of downloaded videos: %s" % vsize
print "Total hours of video: %0.1f" % total_duration


# ***Category distribution***
# 

category = []
for x in database:
    cc = []
    for l in database[x]["annotations"]:
        cc.append(l["label"])
    category.extend(list(set(cc)))
category_count = collections.Counter(category)

plt.figure(num=None, figsize=(18, 8), dpi=100)
xx = np.array(category_count.keys())
yy = np.array([category_count[x] for x in category_count])
xx_idx = yy.argsort()[::-1]
plt.bar(range(len(xx)), yy[xx_idx], color=(240.0/255.0,28/255.0,1/255.0))
plt.ylabel("Number of videos per activity ")
plt.xticks(range(len(xx)), xx[xx_idx], rotation="vertical", size="small")
plt.title("ActivityNet VERSION 1.2 - Untrimmed Video Classification")
plt.show()


def get_sample_frame_from_video(videoid, duration, start_time, end_time,
                                video_path=VIDEO_PATH):
    filename = glob.glob(os.path.join(video_path, "v_%s*" % videoid))[0]
    nr_frames = get_video_number_of_frames(filename)
    fps = (nr_frames*1.0)/duration
    start_frame, end_frame = int(start_time*fps), int(end_time*fps)
    frame_idx = random.choice(range(start_frame, end_frame))
    cap = cv2.VideoCapture(filename)
    keepdoing, cnt = True, 1
    while keepdoing:
        ret, img = cap.read()
        if cnt==frame_idx:
            break
        assert ret==True, "Ended video and frame not selected."
        cnt+=1
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_random_video_from_activity(database, activity, subset="validation"):
    videos = []
    for x in database:
        if database[x]["subset"] != subset: continue
        xx = random.choice(database[x]["annotations"])
        if xx["label"]==activity:
            yy = {"videoid": x, "duration": database[x]["duration"],
                  "start_time": xx["segment"][0], "end_time": xx["segment"][1]}
            videos.append(yy)
    return random.choice(videos)


plt.figure(num=None, figsize=(18, 50), dpi=100)
idx = 1
for ll in leaf_nodes:
    activity = ll["nodeName"]
    keepdoing = True
    while keepdoing:
        try:
            video = get_random_video_from_activity(database, activity)
            img = get_sample_frame_from_video(**video)
            keepdoing = False
        except:
            keepdoing = True
    plt.subplot(20,5,idx)
    idx+=1
    plt.imshow(img), plt.axis("off"), plt.title("%s" % activity)
plt.show()





import collections
import commands
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_video_number_of_frames
from utils import get_video_resolution
from skimage.transform import resize
import filmstrip
import cv2
import random

VIDEO_PATH = "/home/cabaf/AnetVideos/"
get_ipython().magic('matplotlib inline')


# ***Release summary***
# 

with open("activity_net.v1-2.json", "r") as fobj:
    data = json.load(fobj)

database = data["database"]
taxonomy = data["taxonomy"]
version = data["version"]

all_node_ids = [x["nodeId"] for x in taxonomy]
leaf_node_ids = []
for x in all_node_ids:
    is_parent = False
    for query_node in taxonomy:
        if query_node["parentId"]==x: is_parent = True
    if not is_parent: leaf_node_ids.append(x)
leaf_nodes = [x for x in taxonomy if x["nodeId"] in  leaf_node_ids]

vsize = commands.getoutput("du %s -lhs" % VIDEO_PATH).split("/")[0]
with open("../video_duration_info.json", "r") as fobj:
    tinfo = json.load(fobj)
total_duration = sum([tinfo[x] for x in tinfo])/3600.0

category_trimmed = []
for x in database:
    cc = []
    for l in database[x]["annotations"]:
        category_trimmed.append(l["label"])
category_trimmed_count = collections.Counter(category_trimmed)

print "ActivityNet %s" % version
print "Total number of videos: %d" % len(database)
print "Total number of instances: %d" % sum([category_trimmed_count[x] for x in category_trimmed_count])
print "Total number of nodes in taxonomy: %d" % len(taxonomy)
print "Total number of leaf nodes: %d" % len(leaf_nodes)
print "Total size of downloaded videos: %s" % vsize
print "Total hours of video: %0.1f" % total_duration


# *** Category distribution ***
# 

plt.figure(num=None, figsize=(18, 8), dpi=100)
xx = np.array(category_trimmed_count.keys())
yy = np.array([category_trimmed_count[x] for x in category_trimmed_count])
xx_idx = yy.argsort()[::-1]
plt.bar(range(len(xx)), yy[xx_idx], color=(240.0/255.0,28/255.0,1/255.0))
plt.ylabel("Number of videos per activity ")
plt.xticks(range(len(xx)), xx[xx_idx], rotation="vertical", size="small")
plt.title("ActivityNet VERSION 1.2 - Activity Detection")
plt.show()


def get_random_video_from_activity(database, activity, subset="validation"):
    videos = []
    for x in database:
        if database[x]["subset"] != subset: continue
        xx = random.choice(database[x]["annotations"])
        yy = []
        if xx["label"]==activity:
            for l in database[x]["annotations"]:
                yy.append({"videoid": x, "duration": database[x]["duration"],
                           "start_time": l["segment"][0], "end_time": l["segment"][1]})
            videos.append(yy)
    return random.choice(videos)


for ll in leaf_nodes[::-1]:
    activity = ll["nodeName"]
    keepdoing = True
    while keepdoing:
        try:
            video = get_random_video_from_activity(database, activity)
            img_montage = filmstrip.get_film_strip_from_video(video)
            assert img_montage is not None, "None returned"
            keepdoing = False
        except:
            keepdoing = True
    plt.figure(num=None, figsize=(18, 4), dpi=100)
    plt.imshow(np.uint8(img_montage)), plt.title("%s" % activity)
    plt.axis("off")
    plt.show()





# # ActivityNet Challenge Proposal Task
# This notebook is intended as demo on how to format and evaluate the performance of a submission file for the proposal task. Additionally, a helper function is given to visualize the performance on the evaluation metric.
# 

import sys
sys.path.append('../Evaluation')
from eval_proposal import ANETproposal

import matplotlib.pyplot as plt
import numpy as np
import json

get_ipython().magic('matplotlib inline')


# ## Help functions to evaluate a proposal submission file and plot the metric results
# 

def run_evaluation(ground_truth_filename, proposal_filename, 
                   max_avg_nr_proposals=100, 
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=True)
    anet_proposal.evaluate()
    
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    
    return (average_nr_proposals, average_recall, recall)

def plot_metric(average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 0.95, 10)):

    fn_size = 14
    plt.figure(num=None, figsize=(6, 5))
    ax = plt.subplot(1,1,1)
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.), 
                linewidth=4, linestyle='--', marker=None)

    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)

    plt.show()


# ## Generate uniform random proposal for the validation subset
# 

get_ipython().run_cell_magic('time', '', '\n# seed the random number generator to get consistent results across multiple runs \nnp.random.seed(42)\n\nwith open("../Evaluation/data/activity_net.v1-3.min.json", \'r\') as fobj:\n    gd_data = json.load(fobj)\n\nsubset=\'validation\'\navg_nr_proposals = 100\nproposal_data = {\'results\': {}, \'version\': gd_data[\'version\'], \'external_data\': {}}\n\nfor vid_id, info in gd_data[\'database\'].iteritems():\n    if subset != info[\'subset\']:\n        continue\n    this_vid_proposals = []\n    for _ in range(avg_nr_proposals):\n        # generate random proposal center, length, and score\n        center = info[\'duration\']*np.random.rand(1)[0]\n        length = info[\'duration\']*np.random.rand(1)[0]\n        proposal = {\n                    \'score\': np.random.rand(1)[0],\n                    \'segment\': [center - length/2., center + length/2.],\n                   }\n        this_vid_proposals += [proposal]\n    \n    proposal_data[\'results\'][vid_id] = this_vid_proposals\n\nwith open("../Evaluation/data/uniform_random_proposals.json", \'w\') as fobj:\n    json.dump(proposal_data, fobj)')


# ## Evaluate the uniform random proposals and plot the metric results
# 

get_ipython().run_cell_magic('time', '', '\nuniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(\n    "../Evaluation/data/activity_net.v1-3.min.json",\n    "../Evaluation/data/uniform_random_proposals.json",\n    max_avg_nr_proposals=100,\n    tiou_thresholds=np.linspace(0.5, 0.95, 10),\n    subset=\'validation\')\n\nplot_metric(uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)')





