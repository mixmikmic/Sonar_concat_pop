import tf_glove


# # Instantiating the model
# 
# To create a new GloVe model, simply call `tf_glove.GloVeModel()`:
# 

model = tf_glove.GloVeModel(embedding_size=50, context_size=10, min_occurrences=25,
                            learning_rate=0.05, batch_size=512)


# `GloVeModel()` has several parameters:
# - **`embedding_size`**: the target dimensionality of the trained word representations. Typically between 50 and 300.
# - **`context_size`**: how many tokens on either side of a given word to include in each context window. Can be either a tuple of two ints, indicating how many token on the left and right to include, or a single int, which will be interpreted to mean symmetric context.
# - **`max_vocab_size`** *(Optional)*: the maximum size of the model's vocabulary. The model's vocabulary will be the most frequently occurring words in the corpus up to this amount. The default is 100,000.
# - **`min_occurrences`** *(Optional)*: the minimum number of times a word must have appeared in the corpus to be included in the model's vocabulary. Default is 1.
# - **`scaling_factor`** *(Optional)*: the alpha term in Eqn. 9 of Pennington et al.'s paper. Default is 3/4, which is the paper's recommendation
# - **`cooccurrence_cap`** *(Optional)*: the x_max term in Eqn. 9 of Pennington et al.'s paper. Default is 100, which is the paper's recommendation
# - **`batch_size`** *(Optional)*: the number of cooccurrences per minibatch of in training. Default is 512, which seems to work well on my machine. If training is very slow, consider playing with this.
# - **`learning_rate`** *(Optional)*: the Adagrad learning rate used in training. Default is 0.05, which is the paper's recommendation
# 

# # Reading the corpus
# 
# tf_glove needs to be fit to a corpus in order to learn word representations. To do this, we'll use `GloVeModel.fit_to_corpus(corpus)`. This method expects an iterable of iterables of strings, where each string is a token, like this:
# 
# `[["this", "is", "a", "comment", "."], ["this", "is", "another", "comment", "."]]`
# 
# That was a list of lists, but any iterable of iterables of strings should work.
# 
# ### Note on getting the dataset (if you want to follow along with these examples exactly)
# 
# For these examples, I'm going to use the dataset of Reddit comments described here: https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment
# 
# tf_glove is designed to work with any corpus, so there's no need to download this dataset. However, if you'd like to, that post has a link to a torrent for all of the comments as well as a link for just the comments from January 2015. Even just the January 2015 file is quite large (~5 GB).
# 
# I downloaded it and used
# 
# `$ head -n 1000000 RC_2015-01 > /path/to/RC_2015-01-1m_sample`
# 
# to get the 1 million comment sample file referenced below. You could also use 100k if you want to save some time. 1 million comments takes ~15 minutes to fit on my machine.
# 
# ## The code:
# 

import re
import nltk

def extract_reddit_comments(path):
    # A regex for extracting the comment body from one line of JSON (faster than parsing)
    body_snatcher = re.compile(r"\{.*?(?<!\\)\"body(?<!\\)\":(?<!\\)\"(.*?)(?<!\\)\".*}")
    with open(path) as file_:
        for line in file_:
            match = body_snatcher.match(line)
            if match:
                body = match.group(1)
                # Ignore deleted comments
                if not body == '[deleted]':
                    # Return the comment as a string (not yet tokenized)
                    yield body
                        
def tokenize_comment(comment_str):
    # Use the excellent NLTK to tokenize the comment body
    #
    # Note that we're lower-casing the comments here. tf_glove is case-sensitive,
    # so if you want 'You' and 'you' to be considered the same word, be sure to lower-case everything.
    return nltk.wordpunct_tokenize(comment_str.lower())

def reddit_comment_corpus(path):
    # A generator that returns lists of tokens representing individual words in the comment
    return (tokenize_comment(comment) for comment in extract_reddit_comments(path))

# Replace the path with the path to your corpus file
corpus = reddit_comment_corpus("/media/grady/PrimeMover/Datasets/RC_2015-01-1m_sample")


# Now, to fit the model to the corpus:
# 

model.fit_to_corpus(corpus)


# # Training the model
# 
# GloVeModel.fit_to_corpus() builds the vocabulary and cooccurrence matrix that will be used in training, but it doesn't actually train the word representations. It's time to kick off TensorFlow and train the model for real:
# 

model.train(num_epochs=50, log_dir="log/example", summary_batch_interval=1000)


# `GloVeModel.train()` has a few parameters:
# - **`num_epochs`**: How many passes through the cooccurrence matrix the training should make. The paper recommends at least 50 for `embedding_size` < 300, and 100 otherwise.
# - **`log_dir`** *(Optional)*: The path of the directory in which to log summaries for TensorBoard and t-SNE visualizations. Default is `None`, i.e. don't log anything.
# - **`summary_batch_interval`** *(Optional)*: How many minibatches between logging events for TensorBoard. Default is 1000.
# - **`tsne_epoch_interval`** *(Optional)*: How many epochs (full passes through cooccurrence matrix) between outputting a t-SNE visualization of the model's embeddings for the most frequent 1000 words in the vocabulary. Default is None, i.e. don't output t-SNE visualizations during training.
# 

# # Checking out the results
# 
# Now that we've trained the model, let's look at the results.
# 
# Use `GloVeModel.embedding_for()` to get the trained embedding for a single word:
# 

model.embedding_for("reddit")


# You can also get the model's embeddings for every word in the vocabulary like this:
# 

model.embeddings


# `GloVeModel.embeddings` will give you a NumPy matrix where each row is the model's embedding for a single word.
# 
# To make use of this, you'll want to know what row corresponds to a particular word. You can do that with `GloVeModel.id_for_word`:
# 

model.embeddings[model.id_for_word('reddit')]


# And if you want to see a 2D visualization of the learned vector space, you can use `GloVeModel.generate_tsne()`:
# 

get_ipython().magic('matplotlib inline')
model.generate_tsne()


# You might want to open that image in a new tab.
# 
# With no parameters, `GloVeModel.generate_tsne()` can be used interactively like in this notebook, but it also has parameters that will let you save the visualization to a file and adjust the size of the image and how many words appear:
# 
# - **`path`** *(Optional)*: The path at which to save the generated PNG image. Default is None, which only really makes sense for interactive environments.
# - **`size`** *(Optional)*: A tuple of (width, height) in *inches*. (Yeah, I know right? This is inherited from matplotlib.) Default is 100 x 100.  
# - **`word_count`** *(Optional)*: How many words to plot in the visualization. Default is 1000, which works fairly well for a (100 x 100) visualization.
# 

