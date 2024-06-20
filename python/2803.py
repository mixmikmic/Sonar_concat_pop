# This notebook demonstrates how to change the number of topics in the model "on the fly".

import artm
print artm.version()


# Let's start by fitting a simple model with 5 topics using ``fit_offline`` algorithm. Let's  also log perplexity score after each iteration.

batch_vectorizer = artm.BatchVectorizer(data_path=r'C:\bigartm\data', data_format='bow_uci', collection_name='kos')
model = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(5)],
                  scores=[artm.PerplexityScore(name='PerplexityScore')],
                  num_document_passes = 10)
model.initialize(dictionary=batch_vectorizer.dictionary)
model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=5)
print model.score_tracker['PerplexityScore'].value


# Now let's see how to use internal method ``model.master.merge_model`` to add new topics.
# Originally, ``merge_model`` is designed to combine several ``nwt`` matrices with some weights.
# In addition, it allows you to specify which topics to include in the resulting matrix.
# If a topic doesn't exist in any of the source matrices it will be initialized with zeros.
# In the following example we "merge" just a single matrix with wegith ``1.0``.

model.master.merge_model({'nwt': 1.0}, 'test', topic_names = ['topic_{}'.format(i) for i in xrange(7)])
model.get_phi(model_name='test')[:5]


# As a side note, it is always helpful to see which matrices exist in the model.
# Normally you expect to see ``pwt`` and ``nwt`` matrix, but due to ``merge_model`` that we've execute
# there is an additional matrix named ``test``.

for model_description in model.info.model:
    print model_description


# Now, you need to modify the values by *attaching* to the model. From ``model.info`` you can easily see that the model became attached.

(test_model, test_matrix) = model.master.attach_model('test')
for model_description in model.info.model:
    print model_description


import numpy as np
test_matrix[:, [5,6]] = np.random.rand(test_matrix.shape[0], 2)
model.get_phi(model_name='test')[:5]


# Now, I'm realy not sure what will happen if you modify ``pwt`` or ``nwt``, and then use ``fit_offline``.
# That's because the ``fit_offline`` expects matrices with the same number of topics
# as described in the configuration of the model.
# However it is quite safe to use low-level methods, such as ``model.master.process_batches`` and ``model.master.normalize_model``.
# The example below shows how to use these methods to reproduce the results of ``fit_offline``. You need to figure out how to use this methods on the modified matrices (those that have different number of topics).

# Fitting model with our internal API --- process batches and normalize model
model.initialize(dictionary=batch_vectorizer.dictionary)
for i in xrange(5):
    model.master.clear_score_cache()
    model.master.process_batches(model._model_pwt, model._model_nwt,
                                 batches=[x.filename for x in batch_vectorizer.batches_list],
                                 num_document_passes = 10)
    model.master.normalize_model(model._model_pwt, model._model_nwt)
    print model.get_score('PerplexityScore').value


# As you see, perplexity values precisely reproduce the results of the ``fit_offline``.

# # How to eliminate all batches to the same dictionary
# 
# Author - Artem Popov (arti32lehtonen)
# 
# -----------------
# 
# Every batch is independent of all the batches obtained from the same collection. Hence, different token_id may correspond to the same word in different batches. This notebook shows how to change it and eliminate all batches to the same dictionary. It can be useful when you often rewrite your batches.

def eliminate_batches_to_same_dictionary(old_batches_path, new_batches_path):
    """
    Eliminate all batches from one folder to the same dictionary (in alphabetical order). 

    Parametrs:
    ----------
    old_batches_path : folder containing all batches
    
    new_batches_path : folder that will contain all batches after function implementation
    """
    
    list_of_words = get_words_from_batches(batches_path)
    main_dictionary = list_to_word_index_dictionary(list_of_words)
    
    for batch_path in sorted(glob.glob(batches_path + "/*.batch")):
        batch = artm.messages.Batch()
        
        with open(batch_path, "rb") as f:
            batch.ParseFromString(f.read())
        
        new_batch = rewrite_batch_with_dictionary(batch, main_dictionary)
        
        batch_name = batch_path[batch_path.rfind('/'):]
        
        with open(new_batches_path + batch_name, 'wb') as fout:
            fout.write(new_batch.SerializeToString())
    
    return 0 
        
         
def get_words_from_batches(batches_path):
    """
    Get set of words from the all batches and making one big dictionary for all of them
    """
    set_of_words = set()
    
    for batch_path in sorted(glob.glob(batches_path + "/*.batch")):
        batch = artm.messages.Batch()
        
        with open(batch_path, "rb") as f:
            batch.ParseFromString(f.read())
        
        set_of_words = set_of_words.union(set(batch.token))
        
    return sorted(list(set_of_words))


def list_to_word_index_dictionary(list_of_words):
    """
    Transform list of unique elements to the dictionary of format {element:element index}
    """
    return dict(zip(list_of_words, xrange(0, len(list_of_words))))


def list_to_index_word_dictionary(list_of_words):
    """
    Transform list of unique elements to the dictionary of format {element index:element}
    """

    return dict(zip( xrange(0, len(list_of_words)), list_of_words))


def rewrite_batch_with_dictionary(batch, main_dictionary):
    """
    Create new batch with the same content as the old batch, but with 
    tokens corresponds to tokens from main_dictionary
    
    Parametrs:
    ----------
    batch : old batch
    
    main_dictionary: element:element index dictionary of all collection
    """
    
    new_batch = artm.messages.Batch()
    new_batch.id = str(uuid.uuid4())
    
    for token in sorted(main_dictionary.keys()):
        new_batch.token.append(token)
        new_batch.class_id.append(u'@default_class')
    
    batch_dictionary = list_to_index_word_dictionary(batch.token)
    
    for old_item in batch.item:
        new_item = new_batch.item.add()
        new_item.id = old_item.id
        new_item.title = old_item.title

        for one_token_id, one_token_weight in zip(old_item.token_id, old_item.token_weight):
            new_item.token_id.append(main_dictionary[batch_dictionary[one_token_id]])
            new_item.token_weight.append(one_token_weight)    
    
    return new_batch


# It is easy to use it:

eliminate_batches_to_same_dictionary('batches/my_batches', 'batches/my_batches_new')


# # Topic Modeling with BigARTM
# 
# It is an inveractive book about topic modeling using [BigARTM library](http://bigartm.org/).
# 
# The book has two branches: English and [Russian](Topic_Modeling_with_BigARTM_RU.ipynb). Because of our large Russian community the branch on russian will be updated more actively.
# 
# [New Python API Example](BigARTM_example_EN.ipynb)

