# # testing scratch pad and stuff - ignore this
# 

from imp import reload
import numpy as np
import menu, preprocess, models
reload(menu)


menuitems = [('1', 'choice 1', lambda: 'you chose 1'),
             ('2', 'choice 2', lambda: 'you chose 2')
            ]

m = menu.Menu('00', 'main menu', menuitems)


x = menu.Choice(menuitems[0])


res = m()


res


'~' * 3


input('enter')


reload(preprocess)
reload(models)
ve = preprocess.BabiVectorizer()
ve.vectorize_query('Where is John?', verbose=True)


def charvectorize(word, lower=True):
    if lower:
        word = word.lower()
    idxs = [ord(c) for c in word]
    vec = np.zeros(128, int)
    for c in word:
        vec[ord(c)] = 1
    return vec
    
def dist(v1, v2):
    dv = v2 - v1
    dv = dv**2
    dv = np.sum(dv, axis=-1)
    return dv**0.5

def softdist(word1, word2, lower=True):
    v1 = charvectorize(word1, lower)
    v2 = charvectorize(word2, lower)
    return dist(v1, v2)
    
    
def matchnocase(word, vocab):
    lword = word.lower()
    listvocab = list(vocab)
    lvocab = [w.lower() for w in listvocab]
    if lword in lvocab:
        return listvocab[lvocab.index(lword)]
    return None
    

def softmatch(word, vocab, cutoff=2.):
    """Try to soft-match to catch various typos. """
    vw = charvectorize(word)
    vecs = np.array([charvectorize(w) for w in vocab])
    print(vecs.shape)
    distances = dist(vw, vecs)
    idx = np.argmin(distances)
    confidence = distances[idx]
    if confidence < cutoff:
        return vocab[idx]
    return None
    
softmatch('john?', list(ve.word_idx))
# matchnocase('MAry', ve.word_idx)


import os


os.path.normpath()


os.sep


ll


fname = 'foo/bar//spam.txt'
os.makedirs(os.path.dirname(fname), exist_ok=True)


os.path.normpath(fname)





# ### Dependencies
# - keras - obviously
# - h5py - for model checkpointing
# - keras-tqdm - because my [Jupyter notebooks freezes on the default Keras progbar](https://github.com/fchollet/keras/issues/4880). Also, it's awesome
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


'''Trains a memory network on the bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''

# compat
from __future__ import print_function

# python 
from imp import reload
from functools import reduce
import tarfile
import numpy as np
import re

# ML
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from keras_tqdm import TQDMNotebookCallback

# local libs
import preprocess
import models
reload(preprocess)


try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'two_supporting_facts_10k' #'single_supporting_fact_10k'
challenge = challenges[challenge_type]


print('Extracting stories for the challenge:', challenge_type)
train_stories = preprocess.get_stories(tar.extractfile(challenge.format('train')))
test_stories = preprocess.get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)


vocab


train_stories[0]


# Our vocab is pretty simple, and consists of the adverb 'where', people, places, prepositions, verbs, objects, definite article 'the', and two punctuation marks.
# 
# Our single adverb:  ['Where']
# 
# People: ['Daniel', 'John', 'Mary', 'Sandra']
# 
# Places: ['bathroom', 'bedroom', 'garden', 'hallway','kitchen','office']
# 
# Prepositions: ['back', 'to'] 
# 
# Verbs: ['is', 'journeyed', 'moved', 'travelled', 'went']
#  
# Articles: ['the']
#  
# Punctuanion: ['.', '?',]
# 

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
idx_to_word = {value: key for (key, value) in word_idx.items()} # reverse lookup
idx_to_word.update({0: '~'})


reload(preprocess)
ve = preprocess.BabiVectorizer()

inputs_train, queries_train, answers_train = ve.vectorize_all('train')

inputs_test, queries_test, answers_test = ve.vectorize_all('test')


print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')



class DeepMemNet:
    def __init__(self, vocab_size=22, story_maxlen=68, query_maxlen=4):
        # placeholders
        input_sequence = Input((story_maxlen,))
        question = Input((query_maxlen,))

        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=vocab_size,
                                      output_dim=64))
        input_encoder_m.add(Dropout(0.3))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=vocab_size,
                                      output_dim=query_maxlen))
        input_encoder_c.add(Dropout(0.3))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=vocab_size,
                                       output_dim=64,
                                       input_length=query_maxlen))
        question_encoder.add(Dropout(0.3))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(32)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(0.3)(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.model = model


filepath = 'dmn{:02}.hdf5'.format(0)
checkpointer = ModelCheckpoint(monitor='val_acc', filepath=filepath, verbose=1, save_best_only=False)


dmn = DeepMemNet(vocab_size=ve.vocab_size, story_maxlen=ve.story_maxlen, query_maxlen=ve.query_maxlen)
dmn.model.summary()




# train
dmn.model.fit([inputs_train, queries_train], answers_train,
          batch_size=32,
          epochs=120,
          validation_data=([inputs_test, queries_test], answers_test),
             verbose=0, callbacks=[checkpointer, TQDMNotebookCallback()])


ans = dmn.model.predict([inputs_test, queries_test])


plt.plot(ans[0])


i = 0
sentence = ve.deindex_sentence(inputs_test[i])
print(sentence)

query = ve.deindex_sentence(queries_test[i])
print(query)

print(ve.devectorize_ans(ans[i]))


