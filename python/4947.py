#top 5k dice keywords
NUM_CLUSTERS         = 3000 # for 25k keywords and phrases
# number of cluster synonyms to map to
NUM_CLUSTER_SYNONYMS = 5
KEY_WORDS_FILE       = "/Users/simon.hughes/Documents/Dice Data/LuceneTalk/top_5k_keywords.txt"
SYNONYMS_QRY_FILE    = "/Users/simon.hughes/Documents/Dice Data/LuceneTalk/cluster_keyword_synonym_qry.txt"
SYNONYMS_INDEX_FILE  = "/Users/simon.hughes/Documents/Dice Data/LuceneTalk/cluster_keyword_synonym_ix.txt"
PHRASES_FILE         = "/Users/simon.hughes/Documents/Dice Data/LuceneTalk/Phrases.txt"
MODEL_FILE           = "/Users/simon.hughes/Documents/Dice Data/LuceneTalk/keyword_model.w2v"
CLUSTERS_FILE        = "/Users/simon.hughes/Documents/Dice Data/LuceneTalk/%i_clusters.txt" % NUM_CLUSTERS


import numpy as np
#Shared
#just used to load phrases file
def load_stop_words(stop_words_file):
    stop_words = set()
    with open(stop_words_file) as f:
            for line in f:
                word = line.strip()
                if word[0] != "#":
                    word = word.lower()
                    stop_words.add(word)
    return stop_words

def get_vector(item, model):
    vocab = model.vocab[item]
    vector = model.syn0[vocab.index]
    return vector

def get_norm_vector(item, model):
    if item not in model.vocab:
        return None
    # for deserialized models, the norm vectors are not stored
    vec = get_vector(item, model)
    norm = np.linalg.norm(vec)
    if norm != 0:
        return vec / norm
    return vec


import time
grand_start = time.time()


import numpy as np
from collections import defaultdict

#functions
def is_valid_search_keyword(kw):
    q_kw = " " + kw + " "
    for wd in "(,), and , or , not , true , TRUE , false , FALSE ".split(","):
        if wd in q_kw:
            return False
    # remove queries with negations in them
    tokens = kw.split(" ")
    
    # remove single char keywords
    if len(tokens) == 1 and len(tokens[0]) == 1:
        return False
    
    if any(map(lambda t: t.strip().startswith("-"), tokens)):
        return False
    return True

def map_keyword(kw):
    return kw.replace(" ", "_")

def extract_clusters(ids, id2kwd):
    clusters = defaultdict(set)
    for kw_id, label in enumerate(ids):
        kw = id2kwd[kw_id]
        clusters[label].add(kw)
    return clusters

def extract_centroids(km_clusterer):
    lbl2centroid = dict()
    for i in range(len(km_clusterer.cluster_centers_)):
        centroid = km_clusterer.cluster_centers_[i]
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0.0:
            n_centroid = centroid / c_norm
        else:
            n_centroid = centroid
        lbl2centroid[i] = n_centroid
    return lbl2centroid

def compute_cluster_similarities(kwds, kwd2id, vectors, lbl2centroid):
    kwd2cluster_sims = dict()
    for kwd in kwds:
        ix = kwd2id[kwd]
        nvec = vectors[ix]
        sims = []

        for lbl, centroid in lbl2centroid.items():
            cosine_sim = np.inner(nvec, centroid)
            sims.append((lbl,cosine_sim))
        sims = sorted(sims, key = lambda (lbl,sim): -sim)
        kwd2cluster_sims[kwd] = sims
        if len(kwd2cluster_sims) % 1000 == 0:
            print("%i computed out of %i" % (len(kwd2cluster_sims), len(all_kwds)))
    return kwd2cluster_sims

# expand at query time
# use with tfidf (on cluster labels) at index time by just mapping to cluster label
def write_most_similar_clusters(topn, kwd2cluster_sims, synonym_qry_fname, synonyn_index_fname):
    kwords = sorted(kwd2cluster_sims.keys())
    cluster_label = lambda lbl: "cluster_" + str(lbl)
    
    with open(synonym_qry_fname, "w+") as qry_f:
        for kword in kwords:
            cl_sims = kwd2cluster_sims[kword]
            # unlike the other methods, we DO want to include the first cluster here
            # as it's a cluster rather than the top 10 or top 30 keyword method
            top_clusters = cl_sims[:topn]                
            if len(top_clusters) > 0:
                qry_f.write("%s=>" % kword)
                for lbl, sim in top_clusters:                    
                    qry_f.write("%s|%f " %(cluster_label(lbl),sim))
                qry_f.write("\n")
                
    with open(synonyn_index_fname, "w+") as f:
        for kword in kwords:
            # get top cluster label
            lbl, sim = kwd2cluster_sims[kword][0]
            f.write("%s=>%s\n" % (kword, cluster_label(lbl)))


import gensim, time
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load(MODEL_FILE)


phrases = load_stop_words(PHRASES_FILE)
len(phrases)


keywords = []
un_keywords = set()
with open(KEY_WORDS_FILE) as f:
    for line in f:
        kw = line.strip()
        if len(kw) > 0 and is_valid_search_keyword(kw):
            keywords.append(kw)
print("%i keywords loaded from %s" % (len(keywords), KEY_WORDS_FILE))


#get all keywords
# remove any not in the model
all_kwds = phrases.union(keywords)
#all_kwds = set(keywords)
for kwd in list(all_kwds):
    if kwd not in model.vocab:
        all_kwds.remove(kwd)
    splt = kwd.split(" ")
    # add in single word tokens from keywords
    if splt and len(splt) > 1:
        for wd in splt:
            if wd.strip() and wd in model.vocab:
                all_kwds.add(wd)

id2kwd = dict()
kwd2id = dict()
vectors = []
for term in all_kwds:
    id2kwd[len(vectors)] = term
    kwd2id[term] = len(vectors)
    vec = get_norm_vector(term, model)
    vectors.append(vec)

len(all_kwds), len(vectors)


from sklearn import cluster
from sklearn.cluster import KMeans
import time
start = time.time()

# don't parallelize (n_jobs = -1), doesn't seem to work
print("Clustering vectors into %i clusters" % NUM_CLUSTERS)
km_clusterer = KMeans(n_clusters=NUM_CLUSTERS, n_jobs=1, verbose=1, n_init=5)
ids = km_clusterer.fit_predict(vectors)

end = time.time()
print("Creating %i clusters took %i seconds" % (NUM_CLUSTERS, end - start))


lbl2cluster = extract_clusters(ids, id2kwd)
lbl2centroid = extract_centroids(km_clusterer)

len(lbl2cluster), len(lbl2centroid)


import time
start = time.time()

kwd2cluster_sims = compute_cluster_similarities(all_kwds, kwd2id, vectors, lbl2centroid)
end = time.time()
print("Sorting the clusters for each of the %i keywords took %i seconds" % (len(all_kwds),end - start))


write_most_similar_clusters(NUM_CLUSTER_SYNONYMS, kwd2cluster_sims, SYNONYMS_QRY_FILE, SYNONYMS_INDEX_FILE)


grand_end = time.time()
print("Cluster generation and processing took %i seconds" % (grand_end - grand_start))


# # Examine the Clusters
# 

lbl2cluster.values()[0:100]


# # Dump Clusters to File for Later Analysis
# 

with open(CLUSTERS_FILE, "w+") as f:
    for lbl, words in lbl2cluster.items():
        f.write(str(lbl) + "|")
        line = ",".join(sorted(words))
        f.write(line + "\n")


