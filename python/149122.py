# # Test of the first vector of features
# 

import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict


mat = np.load('matrix_first_vector.npy')
redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat.shape


from sklearn.cluster import KMeans
help(KMeans)


k_means = KMeans(n_clusters=15,n_jobs=8,precompute_distances=False)


kmeans_result = k_means.fit(mat)


k_means.labels_


k_means.cluster_centers_


labels = kmeans_result.labels_.tolist()


all_malwares = ml_h.get_all_malwares
all_malwares


for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'KMeans','first_vector',l)


distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'KMeans_first_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'KMeans_first_vector')] = [(m,redis_h.client.hget(m,'label'))]


# # # First Results
# 

distrib


redis_h.client.hget('0404b8957c27de20bebb133d3cf0a28e30700f667f7c2f3fe7fde7e726b691cd','first_vector')


np.array(list(eval(_).values()))


np.linalg.norm(_)


redis_h.client.hget('003315b0aea2fcb9f77d29223dd8947d0e6792b3a0227e054be8eb2a11f443d9','first_vector')


np.array(list(eval(_).values()))


np.linalg.norm(_)


redis_h.client.hget('01259a104a0199b794b0c61fcfc657eb766b2caeae68d5c6b164a53a97874257','first_vector')


np.array(list(eval(_).values()))


np.linalg.norm(_)


redis_h.client.hget('0cfc34fa76228b1afc7ce63e284a23ce1cd2927e6159b9dea9702ad9cb2a6300','first_vector')


np.array(list(eval(_).values()))


np.linalg.norm(_)


redis_h.client.hget('0d8c2bcb575378f6a88d17b5f6ce70e794a264cdc8556c8e812f0b5f9c709198','first_vector')


np.array(list(eval(_).values()))


np.linalg.norm(_)


# # Test with the second vector of features
# 

import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict
from collections import Counter


redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat_second_mat = np.load('matrix_second_vector.npy')
mat_second_mat


from sklearn.cluster import KMeans


k_m = KMeans(n_clusters=20,n_jobs=8,precompute_distances=False)
k_m.fit(mat_second_mat)


k_m.labels_


labels = k_m.labels_.tolist()


all_malwares = ml_h.get_all_malwares
all_malwares


for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'KMeans','second_vector',l)


distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'KMeans_second_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'KMeans_second_vector')] = [(m,redis_h.client.hget(m,'label'))]


distrib


results={}
for k,v in distrib.items():
    c = Counter()
    for malware,label in v:
        c[label] +=1
    results[k]=c
results


redis_h.client.hget('003315b0aea2fcb9f77d29223dd8947d0e6792b3a0227e054be8eb2a11f443d9','second_vector')


redis_h.client.hget('0581a38d1dc61e0da50722cb6c4253d603cc7965c87e1e42db548460d4abdcae','second_vector')


redis_h.client.hget('09c04206b57bb8582faffb37e4ebb6867a02492ffc08268bcbc717708d1a8919','second_vector')





import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict


from sklearn.cluster import DBSCAN


help(DBSCAN)


mat = np.load('matrix_first_vector.npy')
redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat.shape


# # DBScan with the first vector
# 

dbscan = DBSCAN(eps=0.001,min_samples=1, metric="euclidean",n_jobs=8)


dbscan.fit(mat)


labels=dbscan.labels_.tolist()


all_malwares = ml_h.get_all_malwares
for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'DBscan','first_vector',l)


distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'DBscan_first_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'DBscan_first_vector')] = [(m,redis_h.client.hget(m,'label'))]


distrib


distrib.keys()


[(k,len(v)) for k,v in distrib.items()]


distrib[b'4']


distrib[b'0']


distrib[b'1']


# # Dbscan with the second vector
# 

import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict
from sklearn.cluster import DBSCAN


mat_second_vector = np.load('matrix_second_vector.npy')
redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat_second_vector.shape


dbscan = DBSCAN(eps=0.01,min_samples=1, metric="euclidean",n_jobs=8)


dbscan.fit(mat_second_vector)


labels=dbscan.labels_.tolist()


all_malwares = ml_h.get_all_malwares
for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'DBscan','second_vector',l)


distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'DBscan_second_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'DBscan_second_vector')] = [(m,redis_h.client.hget(m,'label'))]


distrib.keys()


sorted([(k,len(v)) for k,v in distrib.items()],key= lambda x: x[1],reverse=True )


distrib[b'1']


distrib[b'3']


distrib[b'39']


distrib[b'38']


distrib[b'105']


distrib[b'55']


distrib[b'45']


distrib[b'37']


redis_h.client.hget('e3892d2d9f87ea848477529458d025898b24a6802eb4df13e96b0314334635d0','second_vector')


redis_h.client.hget('fcfdcbdd60f105af1362cfeb3decbbbbe09d5fc82bde6ee8dfd846b2b844f972','second_vector')


distrib[b'100']


redis_h.client.hget('6c803aac51038ce308ee085f2cd82a055aaa9ba24d08a19efb2c0fcfde936c34','second_vector')


redis_h.client.hget('6217cebf11a76c888cc6ae94f54597a877462ed70da49a88589a9197173cc072','second_vector')





# # matrix with numpy
# 

import numpy as np


mat = mat = np.random.random_sample((125,125))


mat


mat[1,1]


mat[1,]


mat.tolist()


np.save('my_mat',mat)


new_mat = np.load('my_mat.npy')


new_mat == mat


test_eq = _


test_list = test_eq


all([ t for test in test_list for t in test])


# # Sparsed Matrix
# 

from scipy.sparse import csc_matrix
import scipy.sparse as sp


row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sparse_matrix = csc_matrix((data, (row, col)), shape=(3, 3))


row.shape == col.shape == data.shape


sp.issparse(sparse_matrix)


sp.issparse(mat)


new_mat_no_sparse = sparse_matrix.toarray()


sp.issparse(new_mat_no_sparse)


new_mat_no_sparse == sparse_matrix


all([ elem for row in _.tolist() for elem in row])


new_mat_no_sparse


