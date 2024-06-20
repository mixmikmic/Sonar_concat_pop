# # Fit CoFactor model to the binarized ML20M
# 

import itertools
import glob
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd
from scipy import sparse
import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')


import cofacto
import rec_eval


# ### Construct the positive pairwise mutual information (PPMI) matrix
# 

# Change this to wherever you saved the pre-processed data following [this notebook](./preprocess_ML20M.ipynb).
# 

DATA_DIR = '/hdd2/dawen/data/ml-20m/pro/'


unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())
    
unique_sid = list()
with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())


n_items = len(unique_sid)
n_users = len(unique_uid)

print n_users, n_items


def load_data(csv_file, shape=(n_users, n_items)):
    tp = pd.read_csv(csv_file)
    timestamps, rows, cols = np.array(tp['timestamp']), np.array(tp['uid']), np.array(tp['sid'])
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'), timestamps[:, None]), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq


train_data, train_raw = load_data(os.path.join(DATA_DIR, 'train.csv'))


watches_per_movie = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()


print("The mean (median) watches per movie is %d (%d)" % (watches_per_movie.mean(), np.median(watches_per_movie)))


user_activity = np.asarray(train_data.sum(axis=1)).ravel()


print("The mean (median) movies each user wathced is %d (%d)" % (user_activity.mean(), np.median(user_activity)))


vad_data, vad_raw = load_data(os.path.join(DATA_DIR, 'validation.csv'))


plt.semilogx(1 + np.arange(n_users), -np.sort(-user_activity), 'o')
plt.ylabel('Number of items that this user clicked on')
plt.xlabel('User rank by number of consumed items')
pass


plt.semilogx(1 + np.arange(n_items), -np.sort(-watches_per_movie), 'o')
plt.ylabel('Number of users who watched this movie')
plt.xlabel('Movie rank by number of watches')
pass


# ### Generate co-occurrence matrix based on the user's entire watching history
# 

def _coord_batch(lo, hi, train_data):
    rows = []
    cols = []
    for u in xrange(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
    np.save(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
    pass


from joblib import Parallel, delayed

batch_size = 5000

start_idx = range(0, n_users, batch_size)
end_idx = start_idx[1:] + [n_users]

Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data) for lo, hi in zip(start_idx, end_idx))
pass


X = sparse.csr_matrix((n_items, n_items), dtype='float32')

for lo, hi in zip(start_idx, end_idx):
    coords = np.load(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi)))
    
    rows = coords[:, 0]
    cols = coords[:, 1]
    
    tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_items, n_items), dtype='float32').tocsr()
    X = X + tmp
    
    print("User %d to %d finished" % (lo, hi))
    sys.stdout.flush()


# Note: Don't forget to delete all the temporary coo_LO_HI.npy files
# 

np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'), X.data)
np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'), X.indices)
np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'), X.indptr)


float(X.nnz) / np.prod(X.shape)


# ### Or load the pre-saved co-occurrence matrix
# 

# or co-occurrence matrix from the entire user history
dir_predix = DATA_DIR


data = np.load(os.path.join(dir_predix, 'coordinate_co_binary_data.npy'))
indices = np.load(os.path.join(dir_predix, 'coordinate_co_binary_indices.npy'))
indptr = np.load(os.path.join(dir_predix, 'coordinate_co_binary_indptr.npy'))


X = sparse.csr_matrix((data, indices, indptr), shape=(n_items, n_items))


float(X.nnz) / np.prod(X.shape)


def get_row(Y, i):
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return lo, hi, Y.data[lo:hi], Y.indices[lo:hi]


count = np.asarray(X.sum(axis=1)).ravel()


n_pairs = X.data.sum()


# ### Construct the SPPMI matrix
# 

M = X.copy()

for i in xrange(n_items):
    lo, hi, d, idx = get_row(M, i)
    M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))


M.data[M.data < 0] = 0
M.eliminate_zeros()


print float(M.nnz) / np.prod(M.shape)


# Now $M$ is the PPMI matrix. Depending on the number of negative examples $k$, we can obtain the shifted PPMI matrix as $\max(M_{wc} - \log k, 0)$
# 

# number of negative samples
k_ns = 1

M_ns = M.copy()

if k_ns > 1:
    offset = np.log(k_ns)
else:
    offset = 0.
    
M_ns.data -= offset
M_ns.data[M_ns.data < 0] = 0
M_ns.eliminate_zeros()


plt.hist(M_ns.data, bins=50)
plt.yscale('log')
pass


float(M_ns.nnz) / np.prod(M_ns.shape)


# ### Train the model
# 

scale = 0.03

n_components = 100
max_iter = 20
n_jobs = 8
lam_theta = lam_beta = 1e-5 * scale
lam_gamma = 1e-5
c0 = 1. * scale
c1 = 10. * scale

save_dir = os.path.join(DATA_DIR, 'ML20M_ns%d_scale%1.2E' % (k_ns, scale))


reload(cofacto)
coder = cofacto.CoFacto(n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=n_jobs, 
                        random_state=98765, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True, 
                        lam_theta=lam_theta, lam_beta=lam_beta, lam_gamma=lam_gamma, c0=c0, c1=c1)


coder.fit(train_data, M_ns, vad_data=vad_data, batch_users=5000, k=100)


test_data, _ = load_data(os.path.join(DATA_DIR, 'test.csv'))
test_data.data = np.ones_like(test_data.data)


n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))

params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
U, V = params['U'], params['V']


print 'Test Recall@20: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=20, vad_data=vad_data)
print 'Test Recall@50: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=50, vad_data=vad_data)
print 'Test NDCG@100: %.4f' % rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data)
print 'Test MAP@100: %.4f' % rec_eval.map_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data)


np.savez('CoFactor_K100_ML20M.npz', U=U, V=V)





# # Preprocess MovieLens-20M 
# 

import datetime
import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd
import scipy.sparse

import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')


DATA_DIR = '/hdd2/dawen/data/ml-20m/'


def timestamp_to_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)


# binarize the data (only keep ratings >= 4)
raw_data = raw_data[raw_data['rating'] > 3.5]


# sort the raw data accorindg to timestamp
raw_data = raw_data.sort_index(by=['timestamp'])


raw_data


tstamp = np.array(raw_data['timestamp'])


print("Time span of the dataset: From %s to %s" % 
      (timestamp_to_date(np.min(tstamp)), timestamp_to_date(np.max(tstamp))))


# apparently the timestamps are ordered, check to make sure

for i in xrange(tstamp.size - 1):
    if tstamp[i] > tstamp[i + 1]:
        print("not ordered")


# Confirmed the timestamps are ordered
# 

plt.hist(tstamp, bins=50)
xticks = np.linspace(tstamp[0], tstamp[-1], 10)
plt.xticks(xticks, map(lambda x: timestamp_to_date(x)[:7], xticks), rotation=90)
pass


# Now we select the data from 1995-01-01 to the last day as the dataset (i.e., all the dataset)
# 

start_t = time.mktime(datetime.datetime.strptime("1995-01-01", "%Y-%m-%d").timetuple())


raw_data = raw_data[raw_data['timestamp'] >= start_t]


# Take the first 80% of the data as train and validation set
# 

tr_vd_raw_data = raw_data[:int(0.8 * raw_data.shape[0])]


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for songs which were listened to by at least min_sc users. 
    if min_sc > 0:
        songcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(songcount.index[songcount >= min_sc])]
    
    # Only keep the triplets for users who listened to at least min_uc songs
    # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    
    # Update both usercount and songcount after filtering
    usercount, songcount = get_count(tp, 'userId'), get_count(tp, 'movieId') 
    return tp, usercount, songcount


tr_vd_raw_data, user_activity, item_popularity = filter_triplets(tr_vd_raw_data)


sparsity = 1. * tr_vd_raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
      (tr_vd_raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))


unique_uid = user_activity.index
unique_sid = item_popularity.index


song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))


with open(os.path.join(DATA_DIR, 'pro', 'unique_uid.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)


with open(os.path.join(DATA_DIR, 'pro', 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)


# Split 12.5% (10% of the total ratings) as validation set
# 

np.random.seed(13579)
n_ratings = tr_vd_raw_data.shape[0]
vad = np.random.choice(n_ratings, size=int(0.125 * n_ratings), replace=False)


vad_idx = np.zeros(n_ratings, dtype=bool)
vad_idx[vad] = True

vad_raw_data = tr_vd_raw_data[vad_idx]
train_raw_data = tr_vd_raw_data[~vad_idx]


# Make sure there is no empty users/items
# 

print "There are total of %d unique users in the training set and %d unique users in the entire dataset" % (len(pd.unique(train_raw_data['userId'])), len(unique_uid))


print "There are total of %d unique items in the training set and %d unique items in the entire dataset" % (len(pd.unique(train_raw_data['movieId'])), len(unique_sid))


train_sid = set(pd.unique(train_raw_data['movieId']))


left_sid = list()
for i, sid in enumerate(unique_sid):
    if sid not in train_sid:
        left_sid.append(sid)


move_idx = vad_raw_data['movieId'].isin(left_sid)


train_raw_data = train_raw_data.append(vad_raw_data[move_idx])
vad_raw_data = vad_raw_data[~move_idx]


print "There are total of %d unique items in the training set and %d unique items in the entire dataset" % (len(pd.unique(train_raw_data['movieId'])), len(unique_sid))


# For test data, only keep the users and items that appear in the training/validation sets
# 

test_raw_data = raw_data[int(0.8 * len(raw_data)):]


test_raw_data = test_raw_data[test_raw_data['movieId'].isin(unique_sid)]
test_raw_data = test_raw_data[test_raw_data['userId'].isin(unique_uid)]


print len(train_raw_data), len(vad_raw_data), len(test_raw_data)


# Basic data information: what's the timespan for train/test?

train_timestamp = np.asarray(tr_vd_raw_data['timestamp'])
print("train: from %s to %s" % (timestamp_to_date(train_timestamp[0]), 
                                timestamp_to_date(train_timestamp[-1])))

test_timestamp = np.asarray(test_raw_data['timestamp'])
print("test: from %s to %s" % (timestamp_to_date(test_timestamp[0]), 
                               timestamp_to_date(test_timestamp[-1])))


# ### Numerize the data into (timestamp, user_index, item_index) format
# 

def numerize(tp):
    uid = map(lambda x: user2id[x], tp['userId'])
    sid = map(lambda x: song2id[x], tp['movieId'])
    tp['uid'] = uid
    tp['sid'] = sid
    return tp[['timestamp', 'uid', 'sid']]


train_data = numerize(train_raw_data)
train_data.to_csv(os.path.join(DATA_DIR, 'pro', 'train.csv'), index=False)


vad_data = numerize(vad_raw_data)
vad_data.to_csv(os.path.join(DATA_DIR, 'pro', 'validation.csv'), index=False)


test_data = numerize(test_raw_data)
test_data.to_csv(os.path.join(DATA_DIR, 'pro', 'test.csv'), index=False)





