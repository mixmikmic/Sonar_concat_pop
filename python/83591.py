# # Memory Representation in Dialogue Systems (Part 3)
# 
# Under construction, will update with explanations when finished.
# 
# ## Import
# 

import pandas as pd
import numpy as np
import nltk
from sklearn.metrics import accuracy_score
from neo4j.v1 import GraphDatabase, basic_auth
from collections import defaultdict


refs_utts = pd.read_pickle('resources/utts_refs.pkl')
props = pd.read_pickle('resources/restaurants_props.pkl')
len(refs_utts), len(props)


refs_utts[:5]


props[:5]


# ## Process Text
# 

stemmer = nltk.stem.snowball.EnglishStemmer()

def stem(sentence):
    return [stemmer.stem(w) for w in sentence]


test = pd.DataFrame()
test['text'] = [stem(s) for s in refs_utts.text]
test['frame'] = [tuple(stem(f.split()[1:])) for f in refs_utts.bot]
len(test)


# Remove poorly formatted frames
test = test[test.frame.map(len) == 3]
len(test)


test[:5]


knowledge = pd.DataFrame()
knowledge['restaurant'] = props.rname.copy()
knowledge['key'] = [stemmer.stem(s) for s in props.attr_key]
knowledge['value'] = [stemmer.stem(s) for s in props.attr_value]


knowledge[:5]


# A dictionary of keys to the list of values they can take
# In this instance, keys form mutually exclusive lists of values
types = knowledge[['key', 'value']]     .groupby('key')     .aggregate(lambda x: tuple(set(x)))     .reset_index()     .set_index('key')     .value     .to_dict()


types['r_cuisin'][:5]


types['r_locat']


types['r_price']


# ## Create Knowledge Graph
# 

# Create a neo4j session
driver = GraphDatabase.driver('bolt://localhost:7687', auth=basic_auth('neo4j', 'neo4j'))


# WARNING: This will clear the database when run!
def reset_db():
    session = driver.session()
    session.run('MATCH (n) DETACH DELETE n')


reset_db()


session = driver.session()

for i,row in knowledge.iterrows():
    subject, relation, obj = row.restaurant, row.key, row.value
    session.run('''
        MERGE (s:SUBJECT {name: $subject}) 
        MERGE (o:OBJECT  {name: $obj}) 
        MERGE (s)-[r:RELATION {name: $relation}]->(o)
    ''', { 
        'subject': subject,
        'relation': relation,
        'obj': obj
    })


# ## Test
# #### Baseline
# The baseline accuracy is the slot accuracy, calculated by the assumption of not knowing any frame values for any of the sentences.
# 

dont_know = tuple(types.keys())
dont_know


base_predicted = list(dont_know) * len(test)
base_actual = [w for frame in test.frame for w in frame]


accuracy_score(base_actual, base_predicted)


# #### Accuracy
# 

# Cache properties from DB
# Running this query will obtain all properties at this point in time
def get_properties():
    session = driver.session()
    return session.run('''
        MATCH ()-[r:RELATION]->(o:OBJECT) 
        RETURN collect(distinct o.name) AS properties
    ''').single()['properties']


# def get_types():
#     session = driver.session()
#     result = session.run('''
#         MATCH ()-[r:RELATION]->(o:OBJECT) 
#         RETURN collect(distinct [r.name, o.name]) AS pair
#     ''').single()[0]
    
#     g_types = defaultdict(lambda: [])
#     for k,v in result:
#         g_types[k].append(v)
#     return g_types


properties = set(get_properties())


# Hotword listener
def is_hotword(word):
    return word in properties


is_hotword('british'), is_hotword('python')


# Issue DB queries
def find_slot(prop):
    return session.run('''
        MATCH (s:SUBJECT)-[r:RELATION]->(o:OBJECT {name:$name}) 
        RETURN collect(distinct [r.name, o.name]) AS properties
    ''', {
        'name': prop
    })

def extract(result):
    return result.single()['properties'][0]


session = driver.session()
extract(find_slot('west'))


session = driver.session()
all_slots = [[find_slot(word) for word in sentence if is_hotword(word)] for sentence in test.text]
extracted_slots = [[tuple(extract(slot)) for slot in slots] for slots in all_slots]
test['slots'] = extracted_slots


def to_frame(slots):
    frame = list(dont_know)
    s = dict(slots)
    
    for i,x in enumerate(frame):
        if x in s.keys():
            frame[i] = s[x]
    
    return tuple(frame)


test['predicted'] = [to_frame(slot) for slot in test.slots]


test[:5]


predicted = [w for frame in test.predicted for w in frame]
actual = [w for frame in test.frame for w in frame]


accuracy_score(actual, predicted)


cm = nltk.ConfusionMatrix(actual, predicted)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=10))


test[test.text.map(lambda s: 'cheap' in s)]


test[test.text.map(lambda s: 'south' in s)]['text'][284]





