# # Text Exploration
# 

import sys
print(sys.version)


# Use AzureML's data collector to log various metrics!
#from azureml.logging import current_scriptrun
#logger = current_scriptrun()
# Use AzureML's data collector to log various metrics!
#from azureml.logging import current_scriptrun
#logger = current_scriptrun()

import string, re
import pandas as pd
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import azureml
from __future__ import division
import numpy as np
import nltk
from nltk.corpus import stopwords
from azure.storage.blob import BlockBlobService

# import libraries
from __future__ import print_function
import numpy as np
from six.moves import zip
import json
import warnings
import pandas as pd
from pandas import DataFrame   
import pickle
import re
import sys 
import azureml
import string
from scipy import stats
import pip
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer     
from keras.preprocessing import sequence
import os
import tempfile  
import logging
import gensim
from gensim.models import Phrases, phrases
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec as wv
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from IPython.display import SVG
import cloudpickle
import csv
import mkl
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model
import re
import io
from os.path import dirname, join
import regex
import pandas as pd
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import azureml
from __future__ import division
import numpy as np
import nltk
from nltk.corpus import stopwords


# Imprt source text and write to a pandas dataframe
# 

import pickle
with open('biotechcleaned.pkl', 'rb') as f:
    data = pickle.load(f)
    print("Data unpickled")
    import pandas as pd
biotech_cleaned = pd.DataFrame(data)

biotech_cleaned.shape


import pickle
with open('allindustries.pkl', 'rb') as f:
    data = pickle.load(f)
    print("Data unpickled")
    import pandas as pd
allindustries = pd.DataFrame(data)

allindustries.shape


print(allindustries.Category.unique())
#print(data2['Category'].value_counts())


os.getcwd()


biotech_cleaned.dtypes


biotech_cleaned.head(3)


documents_biotech = biotech_cleaned['CleanText'].values
#documents_allindustry = allindustries['CleanText'].values
print(len(documents_biotech))


documents_biotech[0]



# tokenize, create seqs, pad
tok = Tokenizer(num_words=5000, lower=True, split=" ")
tok.fit_on_texts(biotech_cleaned['CleanText'])
biotech = tok.texts_to_sequences(biotech_cleaned['CleanText'])

import nltk 

nltk.download('punkt')
sent_lst = []

for doc in biotech_cleaned['CleanText']:
    sentences = nltk.tokenize.sent_tokenize(doc)
    for sent in sentences:
        
        word_lst = [w for w in nltk.tokenize.word_tokenize(sent) if w.isalnum()]
        sent_lst.append(word_lst)


from gensim.models import Phrases
phrases = Phrases(documents_biotech)
print(phrases)


dictionary = corpora.Dictionary(sent_lst)
print(dictionary)


# Create dictionary and save it
import os
cwd = os.getcwd()
print(os.getcwd() + "\n")


dictionary.save('stocktexts.dict')
print(dictionary)


# Create tokenized corpus and save it
corpus = [dictionary.doc2bow(sent_lst) for sent_lst in sent_lst]
corpora.MmCorpus.serialize('stocktexts.dict', corpus)


print(corpus[0])


# Create TFIDF model and save it
tfidf = models.TfidfModel(corpus)
tfidf.save('stocktexts_tfidf.model')


# Convert document texts to TFIDF
corpus_tfidf = tfidf[corpus]
print(corpus_tfidf[0])


# Extract LDA topics, using 1 pass and updating once every 1 chunk (1000 documents)
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, update_every=1, chunksize=100, passes=5)


lda.print_topics(10)


# Create index using LDA document projections
corpus_lda = lda[corpus]
index_lda = similarities.docsim.Similarity('index_lda.dat', corpus_lda, num_features=100)


print(index_lda)


# Return top 5 similar documents
index_lda.num_best = 20
index_lda[corpus_lda[1]]


print(corpus_lda[171])


print(corpus_lda[128])


# Create index using TFIDF document weights
index_tfidf = similarities.docsim.Similarity('index_tfidf.dat', corpus_tfidf, num_features=len(dictionary))


# Return top 5 similar documents
index_tfidf.num_best = 5
index_tfidf[corpus_tfidf[0]]


print(documents[172])


print(documents[173])


# # Text Exploration
# 

import sys
print(sys.version)


#  Use AzureML's data collector to log various metrics!
#from azureml.logging import current_scriptrun
#logger = current_scriptrun()
# Use AzureML's data collector to log various metrics!
#from azureml.logging import current_scriptrun
#logger = current_scriptrun()

import string, re
import pandas as pd
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import azureml
from __future__ import division
import numpy as np
import nltk
from nltk.corpus import stopwords
from azure.storage.blob import BlockBlobService

# import libraries
from __future__ import print_function
import numpy as np
from six.moves import zip
import json
import warnings
import pandas as pd
from pandas import DataFrame   
import pickle
import re
import sys 
import azureml
import string
from scipy import stats
import pip
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer     
from keras.preprocessing import sequence
import os
import tempfile  
import logging
import gensim
from gensim.models import Phrases, phrases
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec as wv
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from IPython.display import SVG
import cloudpickle
import csv
import mkl
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model
import re
import io
from os.path import dirname, join
import regex
import pandas as pd
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import azureml
from __future__ import division
import numpy as np
import nltk
from nltk.corpus import stopwords


# Imprt source text and write to a pandas dataframe
# 

import pickle
with open('biotechcleaned.pkl', 'rb') as f:
    data = pickle.load(f)
    print("Data unpickled")
    import pandas as pd
biotech_cleaned = pd.DataFrame(data)

biotech_cleaned.shape


import pickle
with open('allindustries.pkl', 'rb') as f:
    data = pickle.load(f)
    print("Data unpickled")
    import pandas as pd
allindustries = pd.DataFrame(data)

allindustries.shape


print(allindustries.Category.unique())
#print(data2['Category'].value_counts())


os.getcwd()


biotech_cleaned.dtypes


biotech_cleaned.head(3)


documents_biotech = biotech_cleaned['CleanText'].values
#documents_allindustry = allindustries['CleanText'].values
print(len(documents_biotech))


documents_biotech[0]



# tokenize, create seqs, pad
tok = Tokenizer(num_words=5000, lower=True, split=" ")
tok.fit_on_texts(biotech_cleaned['CleanText'])
biotech = tok.texts_to_sequences(biotech_cleaned['CleanText'])

import nltk 

nltk.download('punkt')
sent_lst = []

for doc in biotech_cleaned['CleanText']:
    sentences = nltk.tokenize.sent_tokenize(doc)
    for sent in sentences:
        
        word_lst = [w for w in nltk.tokenize.word_tokenize(sent) if w.isalnum()]
        sent_lst.append(word_lst)


from gensim.models import Phrases
phrases = Phrases(documents_biotech)
print(phrases)


dictionary = corpora.Dictionary(sent_lst)
print(dictionary)


# Create dictionary and save it
import os
cwd = os.getcwd()
print(os.getcwd() + "\n")


dictionary.save('stocktexts.dict')
print(dictionary)


# Create tokenized corpus and save it
corpus = [dictionary.doc2bow(sent_lst) for sent_lst in sent_lst]
corpora.MmCorpus.serialize('stocktexts.dict', corpus)


print(corpus[0])


# Create TFIDF model and save it
tfidf = models.TfidfModel(corpus)
tfidf.save('stocktexts_tfidf.model')


# Convert document texts to TFIDF
corpus_tfidf = tfidf[corpus]
print(corpus_tfidf[0])


# Extract LDA topics, using 1 pass and updating once every 1 chunk (1000 documents)
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, update_every=1, chunksize=100, passes=5)


lda.print_topics(10)


# Create index using LDA document projections
corpus_lda = lda[corpus]
index_lda = similarities.docsim.Similarity('index_lda.dat', corpus_lda, num_features=100)


print(index_lda)


# Return top 5 similar documents
index_lda.num_best = 20
index_lda[corpus_lda[1]]


print(corpus_lda[171])


print(corpus_lda[128])


# Create index using TFIDF document weights
index_tfidf = similarities.docsim.Similarity('index_tfidf.dat', corpus_tfidf, num_features=len(dictionary))


# Return top 5 similar documents
index_tfidf.num_best = 5
index_tfidf[corpus_tfidf[0]]


print(documents[172])


print(documents[173])


