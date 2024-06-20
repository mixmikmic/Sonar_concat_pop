# # A Convolutional Encoder Model for Neural Machine Translation

# In this tutorial we will demonstrate how to implement a state of the art convolutional encoder sequential decoder (conv2seq) architecture (Published recently at ACL'17. [Link To Paper](http://www.aclweb.org/anthology/P/P17/P17-1012.pdf)) for sequence to sequence modeling using Pytorch. While the aim of the tutorial is to make the audience comfortable with pytorch using this tutorial (with a Conv2Seq implementation as an add on), some familiarity with pytorch (or any other deep learning framework) would definitely be a plus. The agenda of this tutorial is as follows:
# 
# 1. Getting Ready with the data 
# 2. Network Definition. This includes
#     * A Convolution Encoder with residual connections
#     * An attention based RNN decoder 
# 3. Training subroutines
# 4. Model testing and Visualizations
# 
# This tutorial draws its content/design heavily from [this](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) pytorch tutorial for attention based sequence to sequence model for translation. We reuse their data selection/filtering methodology. This helps in focussing more on explaining model architecture and it's translation from formulae to code. 

# ## Data Preparation
# 
# While the paper uses the official WMT data, we stick to a relatively smaller dataset for English to French translation released as part of the Tatoeba project \[3\] which is present in the "data" directory of this project. We will later apply more filtering to restrict our focus on certain type of short sentences. 
# 
# Some examples of English-French pairs available in the data are:
# 
#     La prochaine fois, je gagnerai la partie. ==> I will win the game next time.
# 
#     Fouillez la maison ! ==>  Search the house!
# 
#     Ne vous faites pas de souci ! Je vous couvre. ==> Don't worry. I've got you covered.
# 
#     Ma famille n'est pas aussi grande que ça. ==> My family is not that large.
# 
#     Ça va être serré. ==> It's going to be close.
# 
# 
#   To get started we first import the necessary libraries
# 

from __future__ import unicode_literals, print_function, division
from io import open
from collections import namedtuple
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas as pd


# We now define some constants and variables that we will use later

use_cuda = torch.cuda.is_available() # To check if GPU is available
MAX_LENGTH = 10 # We restrict our experiments to sentences of length 10 or less
embedding_size = 256
hidden_size_gru = 256
attn_units = 256
conv_units = 256
num_iterations = 750
print_every = 100
batch_size = 1
sample_size = 1000
dropout = 0.2
encoder_layers = 3
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# Note that while the paper uses different model parameters (longer sentences, larger embedding dimensions etc.) we choose smaller values to make the architecture shallow enough for the small data that we are using.

# Next, we will define (or rather copy from [2]) some helper functions that will prove to be useful later

# Function to convert unicdoe string to plain ASCII
# Thanks to http://stackoverflow.com/a/518232/2809427

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Takes all unicode characters, converts them to ascii
# Replaces full stop with space full stop (so that Fire!
# becomes Fire !)
# Removes everything apart from alphabet characters and
# stop characters.

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Returns the cuda tensor type of a variable if CUDA is available
def check_and_convert_to_cuda(var):
    return var.cuda() if use_cuda else var


# We now read the dataset and normalize them. To be able to observe the effects of training on our small dataset quickly, we will restrict our dataset to simpler sentences, which begin with phrases like "i am", "he is", "she is" etc. These prefixes which we will be using to filter our dataset have been defined as the variable `eng_prefixes`.

data = pd.read_csv('data/eng-fra.txt', sep='\t', names=['english', 'french'])
data = data[data.english.str.lower().str.startswith(eng_prefixes)].iloc[:sample_size]

data['english'] = data.apply(lambda row: normalizeString(row.english), axis=1)
data['french'] = data.apply(lambda row: normalizeString(row.french), axis=1)


# We now have a list of sentences which are space separated words. Now, we want to convert these individual words to unique numerical ID's so that each unique word in the vocabulary is represented by a particular integer ID. To do this, we first create a function that does this mapping for us

Vocabulary = namedtuple('Vocabulary', ['word2id', 'id2word']) # A Named tuple representing the vocabulary of a particular language


def construct_vocab(sentences):
    word2id = dict()
    id2word = dict()
    word2id[SOS_TOKEN] = 0
    word2id[EOS_TOKEN] = 1
    id2word[0] = SOS_TOKEN
    id2word[1] = EOS_TOKEN
    for sentence in sentences:
        for word in sentence.strip().split(' '):
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(word2id)-1] = word
    return Vocabulary(word2id, id2word)


# Now, generating the vocabulary for source/target language is as simple as 

english_vocab = construct_vocab(data.english)
french_vocab = construct_vocab(data.french)


# The next task is to convert each sentence to a list of ID's from the corresponding vocabulary mapping. We create another helper function for it. Note that we also add a special End of Sentence `<eos>` token to mark the end of sentence. ( At decoding time, we keep generating words until the `<eos>` token has been generated )

def sent_to_word_id(sentences, vocab, eos=True):
    data = []
    for sent in sentences:
        if eos:
            end = [vocab.word2id[EOS_TOKEN]]
        else:
            end = []
        words = sent.strip().split(' ')
        
        if len(words) < MAX_LENGTH:
            data.append([vocab.word2id[w] for w in words] + end)
    return data


# And finally use this function to generate sentences with token ID's

english_data = sent_to_word_id(data.english, english_vocab)
french_data = sent_to_word_id(data.french, french_vocab)


# What we have generated now are python lists where each item in itself is a list of ID's. However, Pytorch expects a Tensor object and so we also perform that required transformation

input_dataset = [Variable(torch.LongTensor(sent)) for sent in french_data]
output_dataset = [Variable(torch.LongTensor(sent)) for sent in english_data]

if use_cuda: # And if cuda is available use the cuda tensor types
    input_dataset = [i.cuda() for i in input_dataset]
    output_dataset = [i.cuda() for i in output_dataset]


# We are now done with the required data preprocessing that is compatible with the requirements of our Encoder - Decoder architecture.

# # Encoder - Decoder Architecture
# 
# At it's core, an encoder-decoder model uses two neural networks. The first takes in the sentence token by token and produces a sentence representation (A vector of given size, say 512). Once we have this representation (presumably containing the entire semantics of the source sentence), we then use this to generate the corresponding sentence in the target language , word by word. Conventionally, recurrent neural networks have been used for both encoder and decoder [4]  as shown in the figure below ([Image Source](http://colah.github.io/posts/2015-01-Visualizing-Representations/))
# 
# 
# <img src="https://colah.github.io/posts/2015-01-Visualizing-Representations/img/Translation2-RepArrow.png" width="600" height="400" />
# 
# This however burdens the encoder by asking it to encode the entire representation of the sentence in a single vector. [5] propose a neural attention mechanism that at decoding time, apart from using this sentence representation, also tries peek at the input to get additional help in performing that decoding step.
# 
# General architecture of how attention is used to selectively focus on a particular part of input to perform decoding
# 
# <img src="http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.16.08-PM.png" width="200" height="200" />
# 
# (Image Source: WildML Blog)

# This paper replaces the recurrent encoder in the above architecture and replaces it with a Convolutional Encoder. A big benefit from this is that CNN's are highly parallelizable and thus make the training faster (and as shown in the results of the paper with equal or better performance). We now focus on the implementation of the Convolution Encoder (We recommend everyone to also try the previously mentioned RNN Encoder-Decoder tutorial on PyTorch's official tutorial page). As you will see later, the attention formulation is modified from the above architecture when using Convolutional Encoder

# # Convolutional Encoder
# 
# The main components of the convolution encoder are
# 
# * A multi-layer convolution neural network with one fixed size filter that gathers information for the given context window.
# 
# * Residual connections that combine the input of a convolution layer to its output followed by a non-linear transformation. (Note that having no pooling layer after convolution layers is essentiall for incorporating residual connections. Moreover, each input to a convolution layer has to be appropriately padded so that the output-size after applying the convolution operation can remain the same as the original input size and can thus be passed to successive convolution layers whithout reducing the input size )
# 
# * The architecture has two such Convolution networks
#   * Conv-a - The output of this encoder is used for creating the attention matrix that is used at decoding time.
#   * Conv-c - The output of this encoder is attended to (the exact formulation is discussed later) using Conv-a and is then passed to the Decoder
#   
#   
# We will now explain the architecture of in convolution encoder in general (without explicitly referring to Conv-a or Conv-c as they are structurally similar)
# 
# The input to a convolution encoder is combination (addition in this case) of individual word embeddings and their position embeddings which in the paper is given by $e_j = w_j + l_j $. Both these embeddings are learnt during training. Thus for a sentence *"La prochaine fois je gagnerai la partie"* the input to the encoder is
# 
# | Word         | Position  | Representation  |
# | -------------|:---------:| -----:|
# |    La        |  1        |  WordEmbeddingFor(*La*) + PositionEmbeddingFor(*1*)|
# |    prochaine |  2        |  WordEmbeddingFor(*prochaine*) + PositionEmbeddingFor(*2*)  |
# | fois         |  3        |  WordEmbeddingFor(*fois*) + PositionEmbeddingFor(*3*)   |
# |   je         |  4        |  WordEmbeddingFor(*je*) + PositionEmbeddingFor(*4*)|
# |   gagnerai   |  5        |  WordEmbeddingFor(*gagnerai*) + PositionEmbeddingFor(*5*)|
# |   la         |  6        |  WordEmbeddingFor(*la*) + PositionEmbeddingFor(*6*)|
# |partie        |  7        |  WordEmbeddingFor(*partie*) + PositionEmbeddingFor(*7*)|
# 
# 
# We finally begin with our encoder implementation by defining a barebone architecture

get_ipython().run_cell_magic('script', 'false', 'class ConvEncoder(nn.Module):\n    def __init__(self, vocab_size, embedding_size, dropout=0.2,\n                 num_channels_attn=512, num_channels_conv=512, max_len=MAX_LENGTH,\n                 kernel_size=3, num_layers=5):\n      pass\n    def forward(self, position_ids, sentence_as_wordids):\n      # position_ids refer to position of individual words in the sentence \n      # represented by sentence_as_wordids. \n      pass')


# Here we have the constructor with the necessary model parameters. The forward() defines the forward pass of your computational graph. Pytorch handles the backward pass of calculating gradients and updating weights on its own. We now incrementally build our encoder in the following steps. (A point worth mentioning is that while the *position_ids* can be dynamically generated on the fly as part of the computation graph, we pass them as input to make the model code minimal)

get_ipython().run_cell_magic('script', 'false', 'class ConvEncoder(nn.Module):\n    def __init__(self, vocab_size, embedding_size, dropout=0.2,\n                 num_channels_attn=512, num_channels_conv=512, max_len=MAX_LENGTH,\n                 kernel_size=3, num_layers=5):\n      super(ConvEncoder, self).__init__()\n      # Here we define the required layers that would be used in the forward pass\n      self.position_embedding = nn.Embedding(max_len, embedding_size)\n      self.word_embedding = nn.Embedding(vocab_size, embedding_size)\n      self.num_layers = num_layers\n      self.dropout = dropout\n      \n      # Convolution Layers\n      self.conv = nn.ModuleList([nn.Conv1d(num_channels_conv, num_channels_conv, kernel_size,\n                                      padding=kernel_size // 2) for _ in range(num_layers)])\n      \n    def forward(self, position_ids, sentence_as_wordids):\n      # position_ids refer to position of individual words in the sentence \n      # represented by sentence_as_wordids. \n      pass')


# The reason why we explicily use *nn.ModuleList* and not traditional Python Lists is to allow these modules to be visible to other Pytorch modules if GPU is used.
# 
# We now define the computational graph of the encoder 

class ConvEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.2,
                 num_channels_attn=512, num_channels_conv=512, max_len=MAX_LENGTH,
                 kernel_size=3, num_layers=5):
        super(ConvEncoder, self).__init__()
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv = nn.ModuleList([nn.Conv1d(num_channels_conv, num_channels_conv, kernel_size,
                                      padding=kernel_size // 2) for _ in range(num_layers)])

    def forward(self, position_ids, sentence_as_wordids):
        # Retrieving position and word embeddings 
        position_embedding = self.position_embedding(position_ids)
        word_embedding = self.word_embedding(sentence_as_wordids)
        
        # Applying dropout to the sum of position + word embeddings
        embedded = F.dropout(position_embedding + word_embedding, self.dropout, self.training)
        
        # Transform the input to be compatible for Conv1d as follows
        # Length * Channel ==> Num Batches * Channel * Length
        embedded = torch.unsqueeze(embedded.transpose(0, 1), 0)
        
        # Successive application of convolution layers followed by residual connection
        # and non-linearity
        
        cnn = embedded
        for i, layer in enumerate(self.conv):
          # layer(cnn) is the convolution operation on the input cnn after which
          # we add the original input creating a residual connection
          cnn = F.tanh(layer(cnn)+cnn)        

        return cnn


# The only difference between Conv-a and Conv-c is that of embedding size and number of convolution layers which can be easily adjusted for each using the given constructor.

# # Decoder
# We will now see how to build the decoder component of the translation model. To understand the decoder module, let's focus on the section 2 of the paper. The first paragraph indicates that we are getting a sequence of states, **z** defined as:
# $$\mathbf{z} = (z_1, z_2 \ldots z_m)$$
# 
# The next paragraph describes how a typical recurrent neural network works. This part is not necessary in understanding the implementation, since PyTorch and other Deep Learning frameworks provide methods for easy construction of these recurrent networks. What is important to understand, however, is the notations they are using for describing the inputs and outputs, since they will be useful in understanding the equations later. The paper uses LSTM as the neural network, however, for this tutorial we are using a GRU instead. Since GRU and LSTM only differ in the internal mechanism of generating hidden states and outputs, this won't affect the implementation a lot.
# 
# * $h_i$ represents the hidden state/output of the LSTM.
# * $c_i$ is the input context to the LSTM
# * $g_i$ is the embedding of the previous output of the LSTM. This gets concatenated with $c_i$ as input to the LSTM
# 
# The outputs of the encoder module are $cnn_a$ (used in generating attention matrix) and $cnn_c$ (encoded sentence).
# 
# The next word, $y_{i+1}$ is generated as:
# $$p(y_{i+1}|y_1, \ldots, y_i, \mathbf{x}) = \text{softmax}(W_oh_{i+1} + b_o)$$
# 
# For the `softmax` part, we can use PyTorch's `functional` module. For the linear transformation within the `softmax`, we can use `nn.Linear`. The input to this linear transformation is the GRU's hidden state, therefore of the same. The output will be a distribution over the entire output vocabulary, therefore equal to the output vocabulary size.
# 
# So far, our decoder should like something like this:
# 
# 

get_ipython().run_cell_magic('script', 'false', '\nclass AttnDecoder(nn.Module):\n  def __init__(self, output_vocab_size, hidden_size_gru, embedding_size,\n               n_layers_gru):\n    \n    # This will generate the embedding g_i of previous output y_i\n    self.embedding = nn.Embedding(output_size, embedding_size)\n    \n    # A GRU \n    self.gru = nn.GRU(hidden_size_gru+embedding_size, hidden_size, n_layers_gru)\n    \n    # Dense layer for output transformation\n    self.dense_o = nn.Linear(hidden_size_gru, output_vocab_size)\n    \n  def forward(self, y_i, h_i, cnn_a, cnn_c):\n    \n    # generates the embedding of previous output\n    g_i = self.embedding(y_i)\n    \n    gru_output, gru_hidden = self.gru(torch.concat(g_i, input_context), h_i)\n    # gru_output: contains the output at each time step from the last layer of gru\n    # gru_hidden: contains hidden state of every layer of gru at the end\n    \n    # We want to compute a softmax over the last output of the last layer\n    output = F.log_softmax(self.dense_o(gru_hidden[-1]))\n    \n    # We return the softmax-ed output. We also need to collect the hidden state of the GRU\n    # to be used as h_i in the next forward pass\n    \n    return output, gru_hidden')


# In the code snippet above, we haven't included the generation of `input_context` $c_i$. The paper describes the generation of $c_i$ as follows (same section):
# 
# We first transform the hidden state of GRU $h_i$ to match the size of $g_i$, the embedding of previous output and then add the embedding $g_i$.
# 
# $$d_i = W_dh_i + b_d + g_i$$
# 
# Corresponding code (rough - don't copy paste!) will be:
# 
# 
# ---
# 
# 
# ```python
# def __init__():
#   self.transform_gru_hidden = nn.Linear(gru_hidden_size, embedding_size)
# 
# def forward():
#   d_i = self.transform_gru_hidden(gru_hidden[-1]) + g_i
# ```
# 
# 
# ---
# 
# 
# Next, we generate the attention matrix $A$ as follows:
# $$a_{ij} = \frac{exp \left(d_i^Tz_j\right)}{\sum^m_{t=1}exp\left(d_i^Tz_t\right)}$$
# 
# $z_j$ (described in the next page) is the $j^{th}$ column of $cnn\_a$.
# 
# Instead of generating $a_{ij}$ individually, we can generate the entire $\mathbf{a_i}$ in one go, by modifying the equation slightly:
# 
# $$\mathbf{a_i} = \text{softmax}(d_i^T\mathbf{z})$$ where $\mathbf{z}$ now corresponds to the entire $cnn\_a$. This simplifies our implementation, since we can now quickly multiply matrices instead of iterating over individual vectors and computing the dot products.
# 
# Pythonically, we can write this roughly as:
# 
# ---
# 
# ```python
#   a_i = F.softmax(torch.bmm(d_i, cnn_a).view(1, -1))
# ```
# ---
# 
# Finally, we generate $c_i$ as
# $$c_i = \sum_{j=1}^{m}a_{ij}\left (cnn\_c(\mathbf{e})_j \right )$$
# 
# $cnn-c(\mathbf{e})_j$ corresponds to `cnn_c` that we receive as the encoder input. As before, we can transform the equation a bit so that it becomes easier to implement it:
# $$c_i = \mathbf{a}_i \left (cnn\_c \right)$$ 
# 
# ---
# 
# ```python
#   c_i = torch.bmm(a_i.view(1, 1, -1), cnn_c.transpose(1, 2))
# ```
# ---
# 
# A bit of implementation trick here - `a_i` has dimension `(sequence_length,)`. `cnn_c` has dimension `embedding_size x sequence_length`. Now, for `a_i` and `cnn_c` to be multiplied together, we need to make them compatible for multiplication. Therefore, we transpose `cnn_c` to make it `sequence_length x embedding_size` and reshape `a_i` to `1 x sequence_length`
# 
# We are almost done here with the decoder component. Few things we need to do to complete the implementation:
# * Make sure `__init__` and `forward` funcitons have all the arguments which are needed.
# * Add dropouts for embedding and decoder output $h_i$ (section 4.3, last line)
# * Add a function to initialize the hidden units of the GRU to zero after every sentence. (section 4.2, second line)
# 
# Putting everything together, our decoder module now looks like this:

class AttnDecoder(nn.Module):
  
  def __init__(self, output_vocab_size, dropout = 0.2, hidden_size_gru = 128,
               cnn_size = 128, attn_size = 128, n_layers_gru=1,
               embedding_size = 128, max_sentece_len = MAX_LENGTH):

    super(AttnDecoder, self).__init__()
    
    self.n_gru_layers = n_layers_gru
    self.hidden_size_gru = hidden_size_gru
    self.output_vocab_size = output_vocab_size
    self.dropout = dropout
    
    self.embedding = nn.Embedding(output_vocab_size, hidden_size_gru)
    self.gru = nn.GRU(hidden_size_gru + embedding_size, hidden_size_gru,
                      n_layers_gru)
    self.transform_gru_hidden = nn.Linear(hidden_size_gru, embedding_size)
    self.dense_o = nn.Linear(hidden_size_gru, output_vocab_size)

    self.n_layers_gru = n_layers_gru
    
  def forward(self, y_i, h_i, cnn_a, cnn_c):
    
    g_i = self.embedding(y_i)
    g_i = F.dropout(g_i, self.dropout, self.training)
    
    d_i = self.transform_gru_hidden(h_i) + g_i
    a_i = F.softmax(torch.bmm(d_i, cnn_a).view(1, -1))
  
    c_i = torch.bmm(a_i.view(1, 1, -1), cnn_c.transpose(1, 2))
    gru_output, gru_hidden = self.gru(torch.cat((g_i, c_i), dim=-1), h_i)
    
    gru_hidden = F.dropout(gru_hidden, self.dropout, self.training)
    softmax_output = F.log_softmax(self.dense_o(gru_hidden[-1]))
    
    return softmax_output, gru_hidden


  # function to initialize the hidden layer of GRU. 
  def initHidden(self):
    result = Variable(torch.zeros(self.n_layers_gru, 1, self.hidden_size_gru))
    if use_cuda:
        return result.cuda()
    else:
        return result


# # Training the Model
# 
# We now describe the process for training the network on the parallel dataset. In general, the steps involved in training a PyTorch model may be outlined as follows:
# 1. Initialize the network weights
# 2. Define and initialize the optimizers
# 3. Define and initialize the loss criterion
# 4. Repeat till convergence:
#    * Make a forward pass through the network
#    * Use the loss criterion to compute loss
#    * Use the optimizer to compute the gradients
#    * Backpropogate
#    
# We will describe (and implement) each of the steps described above.

# ## Initialize the network weights
# 
# This is easy. We create the objects corresponding to the `ConvEncoder` and `AttnDecoder` classes we have created above. Then we initialize the weights for different parts of the network as follows (section 4.2):
# * Convolution Layers: Samples from uniform distribution in range $(-kd^{-0.5}, kd^{0.5})$
# * Others: Samples from uniform distribution in range $(-0.05, 0.05)$

def init_weights(m):
  
    if not hasattr(m, 'weight'):
        return
    if type(m) == nn.Conv1d:
        width = m.weight.data.shape[-1]/(m.weight.data.shape[0]**0.5)
    else:
        width = 0.05
        
    m.weight.data.uniform_(-width, width)


encoder_a = ConvEncoder(len(french_vocab.word2id), embedding_size, dropout=dropout,
                        num_channels_attn=attn_units, num_channels_conv=conv_units,
                        num_layers=encoder_layers)
encoder_c = ConvEncoder(len(french_vocab.word2id), embedding_size, dropout=dropout,
                        num_channels_attn=attn_units, num_channels_conv=conv_units,
                        num_layers=encoder_layers)
decoder = AttnDecoder(len(english_vocab.word2id), dropout = dropout,
                       hidden_size_gru = hidden_size_gru, embedding_size = embedding_size,
                       attn_size = attn_units, cnn_size = conv_units)

if use_cuda:
    encoder_a = encoder_a.cuda()
    encoder_c = encoder_c.cuda()
    decoder = decoder.cuda()

encoder_a.apply(init_weights)
encoder_c.apply(init_weights)
decoder.apply(init_weights)

encoder_a.training = True
encoder_c.training = True
decoder.training = True


# ## Define and Initialize the Optimizers
# We will use Adam optimzer `torch.optim.Adam` with a learning rate of $10^{-4}$. Here's how we can do it:
# 
# ---
# 
# ```python
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
# 
# encoder_optimizer.zero_grad()
# decoder_optimizer.zero_grad()
# ```
# ---
# 

# ## Define and Initialize Loss Criterion
# We will be using Negative Log Likelihood (also known as Cross-entropy loss) as the loss criterion for our network.
# 
# ---
# ```python
# criterion = nn.NLLLoss()
# ```
# ---
# 

# ## Training Steps
# 
# We will define two functions:
# * `train`: This corresponds to one step of training. It will make a forward pass for one batch, compute loss, compute and backpropagate the gradients.
# * `trainIters`: This will sample a batch and call the train function, in a loop.
# 
# Although the paper suggests using [beam search](https://en.wikipedia.org/wiki/Beam_search) while generating the output sentence, we will use greedy decoding instead.

def trainIters(encoder_a, encoder_c, decoder, n_iters, batch_size=32, learning_rate=1e-4, print_every=100):
  
    encoder_a_optimizer = optim.Adam(encoder_a.parameters(), lr=learning_rate)
    encoder_c_optimizer = optim.Adam(encoder_c.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # Sample a training pair
    training_pairs = list(zip(*(input_dataset, output_dataset)))
    
    criterion = nn.NLLLoss()
    
    
    print_loss_total = 0
    
    # The important part of the code is the 3rd line, which performs one training
    # step on the batch. We are using a variable `print_loss_total` to monitor
    # the loss value as the training progresses
    
    for itr in range(1, n_iters + 1):
        training_pair = random.sample(training_pairs, k=batch_size)
        input_variable, target_variable = list(zip(*training_pair))
        
        loss = train(input_variable, target_variable, encoder_a, encoder_c,
                     decoder, encoder_a_optimizer, encoder_c_optimizer, decoder_optimizer,
                     criterion, batch_size=batch_size)
        
        print_loss_total += loss

        if itr % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print(print_loss_avg)
            print_loss_total=0
    print("Training Completed")


def train(input_variables, output_variables, encoder_a, encoder_c, decoder,
          encoder_a_optimizer, encoder_c_optimizer, decoder_optimizer, criterion, 
          max_length=MAX_LENGTH, batch_size=32):
    
  # Initialize the gradients to zero
  encoder_a_optimizer.zero_grad()
  encoder_c_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  for count in range(batch_size):
    # Length of input and output sentences
    input_variable = input_variables[count]
    output_variable = output_variables[count]

    input_length = input_variable.size()[0]
    output_length = output_variable.size()[0]

    loss = 0

    # Encoder outputs: We use this variable to collect the outputs
    # from encoder after each time step. This will be sent to the decoder.
    position_ids = Variable(torch.LongTensor(range(0, input_length)))
    position_ids = position_ids.cuda() if use_cuda else position_ids
    cnn_a = encoder_a(position_ids, input_variable)
    cnn_c = encoder_c(position_ids, input_variable)
    
    cnn_a = cnn_a.cuda() if use_cuda else cnn_a
    cnn_c = cnn_c.cuda() if use_cuda else cnn_c

    prev_word = Variable(torch.LongTensor([[0]])) #SOS
    prev_word = prev_word.cuda() if use_cuda else prev_word

    decoder_hidden = decoder.initHidden()

    for i in range(output_length):
      decoder_output, decoder_hidden =           decoder(prev_word, decoder_hidden, cnn_a, cnn_c)
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      prev_word = Variable(torch.LongTensor([[ni]]))
      prev_word = prev_word.cuda() if use_cuda else prev_word
      loss += criterion(decoder_output,output_variable[i])

      if ni==1: #EOS
        break

  # Backpropagation
  loss.backward()
  encoder_a_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0]/output_length


# To finally start the training, we simply call the trainIters method. (Be patient as the training will take time depending upon your machine)

trainIters(encoder_a,encoder_c, decoder, num_iterations, print_every=print_every, batch_size=batch_size)


# # Evaluation
# 
# The evaluation function will be very similar to train function minus the backpropagation part.

def evaluate(sent_pair, encoder_a, encoder_c, decoder, source_vocab, target_vocab, max_length=MAX_LENGTH):
    source_sent = sent_to_word_id(np.array([sent_pair[0]]), source_vocab)
    if(len(source_sent) == 0):
        return
    source_sent = source_sent[0]
    input_variable = Variable(torch.LongTensor(source_sent))
    
    if use_cuda:
        input_variable = input_variable.cuda()
        
    input_length = input_variable.size()[0]
    position_ids = Variable(torch.LongTensor(range(0, input_length)))
    position_ids = position_ids.cuda() if use_cuda else position_ids
    cnn_a = encoder_a(position_ids, input_variable)
    cnn_c = encoder_c(position_ids, input_variable)
    cnn_a = cnn_a.cuda() if use_cuda else cnn_a
    cnn_c = cnn_c.cuda() if use_cuda else cnn_c
    
    prev_word = Variable(torch.LongTensor([[0]])) #SOS
    prev_word = prev_word.cuda() if use_cuda else prev_word

    decoder_hidden = decoder.initHidden()
    target_sent = []
    ni = 0
    out_length = 0
    while not ni==1 and out_length < 10:
        decoder_output, decoder_hidden =             decoder(prev_word, decoder_hidden, cnn_a, cnn_c)

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        target_sent.append(target_vocab.id2word[ni])
        prev_word = Variable(torch.LongTensor([[ni]]))
        prev_word = prev_word.cuda() if use_cuda else prev_word
        out_length += 1
        
    print("Source: " + sent_pair[0])
    print("Translated: "+' '.join(target_sent))
    print("Expected: "+sent_pair[1])
    print("")


# To evaluate the learned model, simply execute the following

encoder_a.training = False
encoder_c.training = False
decoder.training = False
samples = data.sample(n=100)
for (i, row) in samples.iterrows():
    evaluate((row.french, row.english), encoder_a, encoder_c, decoder, french_vocab, english_vocab)


# ## References
# 
# 1) Jonas  Gehring,  Michael  Auli,  David  Grangier,  and Yann  Dauphin.  2017. ** A  Convolutional  Encoder Model for Neural Machine Translation.**   In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: LongPapers). Association for Computational Linguistics,Vancouver,  Canada,  pages  123–135
# [http://www.aclweb.org/anthology/P17-1012](http://www.aclweb.org/anthology/P17-1012)
# 
# 2) [** Translation with a Sequence to Sequence Network and Attention **](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) - Official PyTorch Tutorial 
# 
# 3) [**Tatoeba Project**](https://tatoeba.org/eng) (Downloaded from [http://www.manythings.org/anki/](http://www.manythings.org/anki/))
# 
# 4) Sutskever, I., Vinyals, O., and Le, Q. (2014). **Sequence to sequence learning with neural networks**. In Advances in Neural Information Processing Systems (NIPS 2014)
# 
# 5) Bahdanau, Dzmitry, Cho, Kyunghyun, and Bengio, Yoshua. **Neural machine translation by jointly learning to align and translate** arXiv:1409.0473 [cs.CL], September 2014




# # DyNet Autobatch
# 
# ## Friends don't let friends write batching code
# 
# Modern hardware processors (CPUs and GPUs) can use parallelism to a great extent.
# So batching is good for speed. But it is so annoying to write batching code for RNNs or more complex architectures. You must take care of padding, and masking, and indexing, and that's just for the easy cases...
# Not any more!
# 
# We've added a feature to [DyNet](http://github.com/clab/dynet) that will transform the way you think about and run batching code.
# The gist of it is: you aggregate a large enough computation graph to make batching possible. DyNet figures out
# the rest, and does the batching for you.
# 
# <img src="imgs/autobatch.gif" alt="An Example of Autobatching" style="width: 650px;"/>
# 
# In what follows, we show some examples of non-batched DyNet code, and then move on to show the batched version.
# 
# In order to enable auto-batching support, simply add `--dynet-autobatch 1` to the commandline flags when running a DyNet program. Check out the [paper](https://arxiv.org/abs/1705.07860) or read on for more details!
# 

# ## Dynamic Graphs, Non-batched
# 
# Let's look at some examples of non-batched code, and how simple they are to write in DyNet.
# 
# Our first example will be an **acceptor LSTM**, that reads in a sequence of vectors, passes the final vector through a linear layer followed by a softmax, and produces an output.
# 

get_ipython().system('pip install git+https://github.com/clab/dynet#egg=dynet')


import dynet as dy
import numpy as np


# acceptor LSTM
class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W       = model.add_parameters((out_dim, lstm_dim))
    
    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr() # convert the parameter into an Expession (add it to graph)
        outputs = lstm.transduce(sequence)
        result = W*outputs[-1]
        return result


# usage:
VOCAB_SIZE = 1000
EMBED_SIZE = 100

m = dy.Model()
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
acceptor = LstmAcceptor(EMBED_SIZE, 100, 3, m)


# training code
sum_of_losses = 0.0
for epoch in range(10):
    for sequence,label in [((1,4,5,1),1), ((42,1),2), ((56,2,17),1)]:
        dy.renew_cg() # new computation graph
        vecs = [embeds[i] for i in sequence]
        preds = acceptor(vecs)
        loss = dy.pickneglogsoftmax(preds, label)
        sum_of_losses += loss.npvalue()
        loss.backward()
        trainer.update()
    print sum_of_losses / 3
    sum_of_losses = 0.0
        
print "\n\nPrediction time!\n"
# prediction code:
for sequence in [(1,4,12,1), (42,2), (56,2,17)]:
    dy.renew_cg() # new computation graph
    vecs = [embeds[i] for i in sequence]
    preds = dy.softmax(acceptor(vecs))
    vals  = preds.npvalue()
    print np.argmax(vals), vals
    

    


# ---
# This was simple. Notice how each sequence has a different length, but its OK, the `LstmAcceptor` doesn't care.
# We create a new graph for each example, at exactly the desired length.
# 
# Similar to the `LstmAcceptor`, we could also write a `TreeRNN` that gets as input a tree structure and encodes it as a vector. Note that the code below is missing the support code for rerpesenting binary trees and reading trees from bracketed notation. All of these, along with the more sophisticated `TreeLSTM` version, and the training code, can be found [here](https://github.com/neulab/dynet-benchmark/blob/master/dynet-py/treenn.py).
# 

class TreeRNN(object):
    def __init__(self, model, word_vocab, hdim):
        self.W = model.add_parameters((hdim, 2*hdim))
        self.E = model.add_lookup_parameters((len(word_vocab),hdim))
        self.w2i = word_vocab

    def __call__(self, tree): return self.expr_for_tree(tree)
    
    def expr_for_tree(self, tree):
        if tree.isleaf():
            return self.E[self.w2i.get(tree.label,0)]
        if len(tree.children) == 1:
            assert(tree.children[0].isleaf())
            expr = self.expr_for_tree(tree.children[0])
            return expr
        assert(len(tree.children) == 2),tree.children[0]
        e1 = self.expr_for_tree(tree.children[0], decorate)
        e2 = self.expr_for_tree(tree.children[1], decorate)
        W = dy.parameter(self.W)
        expr = dy.tanh(W*dy.concatenate([e1,e2]))
        return expr


# ## Enter batching
# 
# Now, let's add some minibatching support. The way we go about it is very simple:
# Your only responsibility, as a programmer, is to **build a computation graph with enough material to make batching possible** (i.e., so there is something to batch). DyNet will take care of the rest.
# 
# Here is the training and prediction code from before, this time writen with batching support.
# Notice how the `LstmAcceptor` did not change, we just aggregate the loss around it.
# 

# training code: batched.
for epoch in range(10):
    dy.renew_cg()     # we create a new computation graph for the epoch, not each item.
    # we will treat all these 3 datapoints as a single batch
    losses = []
    for sequence,label in [((1,4,5,1),1), ((42,1),2), ((56,2,17),1)]:
        vecs = [embeds[i] for i in sequence]
        preds = acceptor(vecs)
        loss = dy.pickneglogsoftmax(preds, label)
        losses.append(loss)
    # we accumulated the losses from all the batch.
    # Now we sum them, and do forward-backward as usual.
    # Things will run with efficient batch operations.
    batch_loss = dy.esum(losses)/3
    print batch_loss.npvalue() # this calls forward on the batch
    batch_loss.backward()
    trainer.update()
   
print "\n\nPrediction time!\n"
# prediction code:
dy.renew_cg() # new computation graph
batch_preds = []
for sequence in [(1,4,12,1), (42,2), (56,2,17)]:
    vecs = [embeds[i] for i in sequence]
    preds = dy.softmax(acceptor(vecs))
    batch_preds.append(preds)

# now that we accumulated the prediction expressions,
# we run forward on all of them:
dy.forward(batch_preds)
# and now we can efficiently access the individual values:
for preds in batch_preds:
    vals  = preds.npvalue()
    print np.argmax(vals), vals


# ----
# Doing the same thing for the TreeRNN example is trivial: just aggregate the expressions from several trees, and then call forward. (In fact, you may receive a small boost from the auto-batching feature also within a single tree, as some computation can be batched there also.)
# 
# ## Comparison to manual batching
# We compared the speed of automatic-batching as shown above to a manualy crafted batching code, in a setting in which manual-batching excels: BiLSTM tagging where all the sentences are of the exact same length. Here, automatic batching improved the per-sentence computation time from 193ms to 16.9ms on CPU and 54.6ms to 5.03ms on GPU, resulting in an approximately 11-fold increase in sentences processed per second (5.17->59.3 on CPU and 18.3->198 on GPU).
# However, manual batching is still 1.27 times faster on CPU, and 1.76 times faster on a GPU. 
# 
# The speed in favor of manual batching seem to come mostly from the time it takes to create the computation graph itself: in manual batching we are creating a single graph with many inputs, while with automatic batching we essentially build many copies of the same graph for each batch. Should you use manual batching then? In situations in which it is very natural, like in this artificial one, sure! But in cases where manual batching is not so trivial (which is most cases, see some examples below), go ahead and use the automatic version. It works.
# 
# You can also run automatic batching on top of manually batched code. When doing this, we observe another 10% speed increase above the manual batched code, when running on the GPU. This is because the autobatching engine managed to find and exploit some additional batching opportunities. On the CPU, we did not observe any gains in this setting, but also did not observe any losses.
# 

# ## How big is the win?
# 
# So the examples above are rather simple, but how does this help on actual applications? We've run some experiments on several natural language processing tasks including POS tagging with bidirectional LSTMs, POS tagging with BiLSTMs that also have character embeddings (which is harder to batch), tree-structured neural networks, and a full-scale transition-based dependency parser. Each of these has a batch size of 64 sentences at a time, without worrying about length balancing or anything of that sort. As you can see from the results below on sentences/second, auto-batching gives you healthy gains of 3x to 9x over no auto-batching. This is with basically no effort required!
# 

# | Task              | No Autobatch (CPU) | Autobatch (CPU) | No Autobatch (GPU) | Autobatch (GPU) |
# |-------------------|--------------------|-----------------|--------------------|-----------------|
# | BiLSTM            | 16.8 | 156  | 56.2 | 367  |
# | BiLSTM w/ char    | 15.7 | 132  | 43.2 | 275  |
# | TreeNN            | 50.2 | 357  | 76.5 | 661  |
# | Transition Parser | 16.8 | 61.2 | 33.0 | 90.1 |
# 

# If you want to try these benchmarks yourself, take a look at the `...-bulk` programs in the [dynet-benchmark](http://github.com/neulab/dynet-benchmark) repository.
# 

# In the graph below you can see the number of sentences/second for training the transition-based parser with various batch sizes, on the GPU, CPU, and CPU witk MKL enabled:
# 
# <img src="imgs/bist-autobatch-speed.png" alt="Autobatching Speed in Various Batch Sizes" style="width: 600px;"/>
# 
# The following graph shows the number of sentences/second for the Tree-LSTM model for various batch sizes, and also compares to TensorFlow Fold implementation, which is another proposed solution for batching hard-to-batch architectures. Note that DyNet autobatching comfortably wins over TensorFlow fold for both GPU and CPU, with CPU being more efficient than GPU for smaller sized batches.
# 
# <img src="imgs/treelstm-autobatch-speed.png" alt="Autobatching Speed in Various Batch Sizes" style="width: 600px;"/>
# 
# 
# 
# 

# ## Miscellaneous tips
# 
# ### Should you always use batching?
# 
# It depends. In prediction time, batching is a pure win in terms of speed. In training time, the sentences/second throughput will be much better---but you will also have less parameter updates, which may make overall training slower. Experiment with different batch sizes to find a good tradeoff between the two.
# 
# ### Length-balanced batches?
# 
# It is common knowledge when writing batched code that one should arrange the batches such that all examples within the batch are of the same size.
# This is crucial for static frameworks and manual batching, as it reduces the need for padding, masking and so on.
# In our framework, this is not needed. However, you may still win some speed by having relatively-balanced batches,
# because more batching opportunities will become available.
# 
# 
# ### Tips for effective autobatching
# 
# As we said above, our only rule is "create a graph with enough material for the autobatcher to work with".
# In other words, it means delaying the call to `forward()` (or to `value()`, `npvalue()`, `scalar_value()`...) as much as possible. Beyond that, things should be transparent.
# 
# However, knowing some technicalities of DyNet and how `forward` works can help you avoid some pitfals. So here is a brief overview:
# 
# 1. The core building block of dynet are `Expression` objects. Whenever you create a new `Expression`, you extend the computation graph. 
# 
# 2. Creating an `Expression` does not entail a forward computation. We only evaluate the graph when specifically asked for it.
# 
# 3. Calls to `e.forward()`, `e.value()`, `e.npvalue()`, `e.scalar_value()`, will run forward computation **up to that expression**, and return a value.
# 
# 4. These calls will compute all the expressions that were added to the graph before `e`. These intermediary results will be cached.
# 
# 5. Asking for values for (or calling forward on) earlier expressions, will reuse the cached values.
# 
# 6. You can extend the graph further after calling forward. Later calls will compute the graph delta.
# 
# So, based on this knowledge, here is the rule:
# 
# If you created several expressions, and want to get the values for them, call forward on the **last** expression first, and then on the previous ones.
# 
# Doing it the other way around (evaluting the expressions in the order they were created) will hinder batching possibilities, because it will compute only a small incremental part of forward for each expression. On the other hand, if you run forward on the last expression first, the entire computation will happen in one chunk, batching when possible. Getting calling `npvalue()` on the earlier expressions will then return the already computed values.  
# 
# If you created a bunch of expressions and are not sure which one is the latest, you could just call the special `list` version of forward:
# ```
# dy.forward([e1,e2,...,en])
# ```
# and it will figure it out for you.
# 

# ## Loose ends
# 
# Auto-batching in DyNet works and is stable. However, some of the less common operations are not yet batched. If you have an example where you think you should be getting a nice boost from autobatching but you don't, it is most likely that you are using a non-batched operation. In any case, let us know via an issue in github, and we'll investigate this.
# 




# # Probabilistic Programming in Pyro
# LEARN HOW TO CODE A PAPER WITH STATE OF THE ART FRAMEWORKS <br>
# NIPS 2017
# 
# ## Pyro Installation Instructions
# (FOR THOSE RUNNING THE NOTEBOOK FROM HOME)
# 
# First install [PyTorch](http://pytorch.org).
# 
# Install Pyro via pip:
# 
# Python 2.7.*:
# ```python
# pip install pyro-ppl
# ```
# 
# Python 3.5:
# 
# ```python
# pip3 install pyro-ppl
# ```
# 
# Alternatively, install Pyro from source:
# 
# ```python
# git clone git@github.com:uber/pyro.git
# cd pyro
# pip install .
# ```
# Note that this particular notebook can be found on the nips-2017 branch of the [GitHub repo](https://github.com/uber/pyro) in the examples/nips2017 directory.
# 
# 
# #### Other dependencies
# Note that in order to run this notebook you will also need the following dependencies: matplotlib, observations, requests.
# 
# ## First steps
# 
# Let's start with some imports
# 

get_ipython().system('pip install pyro-ppl')


import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist


# Let's draw a sample from a unit normal distribution:
# 

mu = Variable(torch.zeros(1))   # mean zero
sigma = Variable(torch.ones(1)) # unit variance
x = dist.normal(mu, sigma)      # x is a sample from N(0,1)
print(x)


# We can compute the log pdf of the sample as follows
# 

dist.normal.log_pdf(x, mu, sigma)


# We can also declare mu as a named parameter:
# 

mu = pyro.param("mu", Variable(torch.zeros(1), requires_grad=True))
print(mu)


# ## The VAE
# 
# #### The Model
# First we define our decoder network
# 

import torch.nn as nn

z_dim=20
hidden_dim=100

nn_decoder = nn.Sequential(
    nn.Linear(z_dim, hidden_dim), 
    nn.Softplus(), 
    nn.Linear(hidden_dim, 784), 
    nn.Sigmoid()
)


# Now we can define our generative model conditioned on the observed mini-batch of images `x`:
# 

# import helper functions for Variables with requires_grad=False
from pyro.util import ng_zeros, ng_ones 

def model(x):
    batch_size=x.size(0)
    # register the decoder with Pyro (in particular all its parameters)
    pyro.module("decoder", nn_decoder)  
    # sample the latent code z
    z = pyro.sample("z", dist.normal,   
                    ng_zeros(batch_size, z_dim), 
                    ng_ones(batch_size, z_dim))
    # decode z into bernoulli probabilities
    bern_prob = nn_decoder(z)          
    # observe the mini-batch of sampled images
    return pyro.sample("x", dist.bernoulli, bern_prob, obs=x) 


# Note that instead of using the `obs` keyword in the `pyro.sample` statement we could also have used `pyro.condition`. For details on how that works see [this tutorial](http://pyro.ai/examples/intro_part_ii.html).
# 

# #### The guide
# 
# In order to do inference, we need to define a guide (a.k.a. an inference network). First we define the encoder network. Let's go ahead and define it explicitly instead of using `nn.Sequential`:
# 

class Encoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=100):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # define the forward computation on the image x
        # first compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma


# Now we can define the guide:
# 

nn_encoder = Encoder()

def guide(x):
    # register the encoder with Pyro
    pyro.module("encoder", nn_encoder)
    # encode the mini-batch of images x
    mu_z, sig_z = nn_encoder(x)
    # sample and return the latent code z
    return pyro.sample("z", dist.normal, mu_z, sig_z)


# #### Inference
# Now we're ready to do inference. First we setup our optimizer
# 

from pyro.optim import Adam
optimizer = Adam({"lr": 1.0e-3})


# Now we setup the `SVI` inference algorithm, which we will use to take gradient steps on the ELBO objective function. Note that `model` and `guide` both have the same call signature (namely they take in a mini-batch of images `x`).
# 

from pyro.infer import SVI
svi = SVI(model, guide, optimizer, loss="ELBO")


# Let's setup a basic training loop. First we setup the data loader:
# 

import torchvision.datasets as dset
import torchvision.transforms as transforms

batch_size=250
trans = transforms.ToTensor()
train_set = dset.MNIST(root='./mnist_data', train=True, 
                       transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batch_size,
                                           shuffle=True)


# Let's do 3 epochs of training and report the ELBO averaged per data point for each epoch (note that this can be somewhat slow on Azure ML)
# 

num_epochs = 3

for epoch in range(num_epochs):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, _ in train_loader:
        # wrap the mini-batch of images in a PyTorch Variable
        x = Variable(x.view(-1, 784))
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # report training diagnostics
    normalizer = len(train_loader.dataset)
    print("[epoch %03d]  average training ELBO: %.4f" % (epoch, -epoch_loss / normalizer))


# So much for the VAE. For a more fleshed out implementation and some results please see the [tutorial](http://pyro.ai/examples/vae.html).
# 
# ## Recursion with random control flow
# 
# Let's define the geometric distribution in terms of draws from a bernoulli distribution:
# 

def geom(num_trials=0, bern_prob=0.5):
    p = Variable(torch.Tensor([bern_prob]))
    x = pyro.sample('x{}'.format(num_trials), dist.bernoulli, p)
    if x.data[0] == 1:
        return num_trials  # terminate recursion
    else:
        return geom(num_trials + 1, bern_prob)  # continue recursion

# let's draw 15 samples 
for _ in range(15):
    print("%d  " % geom()),


# Note that the random variables in `geom` are generated dynamically and that different calls to `geom` can have different numbers of random variables. Also note that we take care to assign unique names to each dynamically generated random variable.
# 
# If we crank down the bernoulli probability (so that the recursion tends to terminate after a larger number of steps) we get a geometric distribution with more of a tail:
# 

for _ in range(15):
    print("%d  " % geom(bern_prob=0.1)),


# ## AIR
# 
# #### The prior
# 
# Let's build on `geom()` above to construct a recursive prior over images. First we need a prior over a single object in an image. Just like for the VAE, this prior is going to involve a decoder network. So we define that first:
# 

from torch.nn.functional import relu, sigmoid, grid_sample, affine_grid

z_dim=50

# this decodes latents z into (bernoulli pixel intensities for)
# 20x20 sized objects
class Decoder(nn.Module):
    def __init__(self, hidden_dim=200):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(z_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 20*20)

    def forward(self, z_what):
        h = relu(self.l1(z_what))
        return sigmoid(self.l2(h))

decoder = Decoder()


# Now we build the prior over a single object. Note that this prior uses (differentiable) spatial transformers to position the sampled object within the image. Most of the complexity in this code snippet is on the spatial transformer side.
# 

# define the prior probabilities for our random variables
z_where_prior_mu = Variable(torch.Tensor([3, 0, 0]))
z_where_prior_sigma = Variable(torch.Tensor([0.1, 1, 1]))
z_what_prior_mu = ng_zeros(50)
z_what_prior_sigma = ng_ones(50)

def expand_z_where(z_where):
    # Takes 3-dimensional vectors, and massages them into 
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = z_where.size(0)
    expansion_indices = Variable(torch.LongTensor([1, 0, 2, 0, 1, 3]))
    out = torch.cat((ng_zeros([1, 1]).expand(n, 1), z_where), 1)
    return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

# takes the object generated by the decoder and places it 
# within a larger image with the desired pose
def object_to_image(z_where, obj):
    n = obj.size(0)
    theta = expand_z_where(z_where)
    grid = affine_grid(theta, torch.Size((n, 1, 50, 50)))
    out = grid_sample(obj.view(n, 1, 20, 20), grid)
    return out.view(n, 50, 50)

def prior_step(t):
    # Sample object pose. This is a 3-dimensional vector representing 
    # x,y position and size.
    z_where = pyro.sample('z_where_{}'.format(t),
                          dist.normal,
                          z_where_prior_mu,
                          z_where_prior_sigma,
                          batch_size=1)

    # Sample object code. This is a 50-dimensional vector.
    z_what = pyro.sample('z_what_{}'.format(t),
                         dist.normal,
                         z_what_prior_mu,
                         z_what_prior_sigma,
                         batch_size=1)

    # Map code to pixel space using the neural network.
    y_att = decoder(z_what)

    # Position/scale object within larger image.
    y = object_to_image(z_where, y_att)

    return y


# Let's draw a few samples from the object prior to see how this looks.
# 

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

samples = [prior_step(0)[0] for _ in range(8)]

def show_images(samples):
    plt.rcParams.update({'figure.figsize': [10, 1.6] })
    f, axarr = plt.subplots(1, len(samples))

    for i, img in enumerate(samples):
        axarr[i].imshow(img.data.numpy(), cmap='gray')
        axarr[i].axis('off')

    plt.show()
    
show_images(samples)


# Now we can use `prior_step` to define a recursive prior over images:
# 

def geom_image_prior(x, step=0):
    p = Variable(torch.Tensor([0.4]))
    i = pyro.sample('i{}'.format(step), dist.bernoulli, p)
    if i.data[0] == 1:
        return x
    else:
        # add sampled object to canvas
        x = x + prior_step(step)  
        return geom_image_prior(x, step + 1)


# Let's visualize a few draws from `geom_image_prior`:
# 

x_empty = ng_zeros(1, 50, 50)
samples = [geom_image_prior(x_empty)[0] for _ in range(16)]
show_images(samples[0:8])
show_images(samples[8:16])


# We see that the images sampled from the prior have a variable number of objects, with the objects placed throughout the image. The objects are also allowed to stack (so that pixel intensities can exceed 1).
# 

# #### Reconstructing images with the trained model/guide pair
# 
# Let's take advantage of the fact that someone has trained the model and guide for us and see what we can do. One nice thing about amortized variational inference is that it allows us to use the guide (i.e. the inference network) to make quick test time predictions. Let's see how that goes in Pyro. First some imports:
# 

# this is the dataset we used for training
from observations import multi_mnist 
import numpy as np

import pyro.poutine as poutine

from air import AIR, latents_to_tensor
from viz import draw_many, tensor_to_objs, arr2img


# Next we load and preprocess the test set:
# 

def load_data():
    inpath = './multi_mnist_data'
    _, (X_np, Y) = multi_mnist(inpath, max_digits=2, canvas_size=50, seed=42)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = Variable(torch.from_numpy(X_np))
    return X

X = load_data()


# Next we instantiate an instance of the AIR `nn.module`, which contains both the model and the guide (and all the associated neural networks). Here we take care to use same hyperparameters that were used during training. We also load the learned model and guide parameters from disk.
# 

air = AIR(
    num_steps=3,
    x_size=50,
    z_what_size=50,
    window_size=28,
    encoder_net=[200],
    decoder_net=[200],
    predict_net=[200],
    bl_predict_net=[200],
    rnn_hidden_size=256,
    decoder_output_use_sigmoid=True,
    decoder_output_bias=-2,
    likelihood_sd=0.3
)   

air.load_state_dict(torch.load('air.pyro',
                    map_location={'cuda:0':'cpu'}))


# Let's pick some datapoints from the test set to reconstruct and visualize:
# 

ix = torch.LongTensor([9906, 1879, 5650,  967, 7420, 7240, 2755, 9390,   42, 5584])
n_images = len(ix)
examples_to_viz = X[ix]


# Finally we do the reconstruction and visualization.
# 

params = { 'figure.figsize': [8, 1.6] }   
plt.rcParams.update(params)
f, axarr = plt.subplots(2,n_images)

for i in range(n_images):
    img = arr2img(examples_to_viz[i].data.numpy()).convert('RGB')
    axarr[0,i].imshow(img)
    axarr[0,i].axis('off')

# run the guide and store the sampled random variables in the trace
trace = poutine.trace(air.guide).get_trace(examples_to_viz, None)
# run the prior against the samples in the trace
z, recons = poutine.replay(air.prior, trace)(examples_to_viz.size(0))
# extract the sampled random variables needed to generate the visualization
z_wheres = tensor_to_objs(latents_to_tensor(z))
# make the visualization
recon_imgs = draw_many(recons, z_wheres)
    
for i in range(n_images):
    axarr[1,i].imshow(recon_imgs[i])
    axarr[1,i].axis('off')

plt.subplots_adjust(left=0.02, bottom=0.04, right=0.98, top=0.96, wspace=0.1, hspace=0.1)
plt.savefig('air_multi_mnist_recons.png', dpi=400)
plt.show()


# The images in the top row are from the test set; the corresponding reconstructions are in the bottom row.
# Note that the small digits at the top left of each reconstructed image show the total number of objects used in the reconstruction, while the colored boxes denote the image patches used (i.e. the positions of each reconstructed object).
# 
# For a more a complete implementation with results please see the [tutorial](http://pyro.ai/examples/air.html).
# 

# ## Pyro links
# 
# [Pyro website](http://pyro.ai)<br>
# [VAE tutorial](http://pyro.ai/examples/vae.html)<br>
# [AIR tutorial](http://pyro.ai/examples/air.html)<br>
# [Documentation](http://docs.pyro.ai)<br>
# [Pyro GitHub](http://github.com/uber/pyro)<br>
# [Pyro Forum](http://forum.pyro.ai)
# 
# ## References
# 
# [1] `Auto-Encoding Variational Bayes`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
# Diederik P Kingma, Max Welling
# 
# [2] `Stochastic Backpropagation and Approximate Inference in Deep Generative Models`,
# <br/>&nbsp;&nbsp;&nbsp;&nbsp;
# Danilo Jimenez Rezende, Shakir Mohamed, Daan Wierstra
# 
# [3] `Attend, Infer, Repeat: Fast Scene Understanding with Generative Models`
# <br />&nbsp;&nbsp;&nbsp;&nbsp;
# S. M. Ali Eslami and Nicolas Heess and Theophane Weber and Yuval Tassa and Koray Kavukcuoglu and Geoffrey E. Hinton
# 
# [4] `Spatial Transformer Networks`
# <br />&nbsp;&nbsp;&nbsp;&nbsp;
# Max Jaderberg and Karen Simonyan and Andrew Zisserman
# 

# 
# # Combining Symbolic Expressions and Black-box Function Evaluations in Neural Programs
# 
# *Developed by Forough Arabshahi*
# 
# This notebook presents the code for [this](https://openreview.net/forum?id=Hksj2WWAW&noteId=Hksj2WWAW) paper.
# 
# The focus of the paper is on neural programing. Traditional methods to neural programming either rely on black-box function evaluation data or rely on program execution traces for training the neteworks. Both of these methods lack generalizability. Black-box function evaluations do not contain any infomation about the structure of the problem. Porgram execution traces, On the other hand, are expensive to collect resulting in a lack of domain coverage.
# 
# In many problems, one has access to symbolic representation of the problem that encodes the relationships between the given variables and functions in a succinct manner. For example, declarative programs greatly simplify parallel programs through the generation of symbolic computation graphs. As another example, the properties of mathematical functions are encoded through symbolic expressions. E.g. symbolic expression $x+y = y+x$ encodes the commutative property of the addition function. Therefore, symbolic expressions efficiently represent the properties of the problem, preserve structure and are readily accessible. Thus, they are a great alternative to black-box function evaluations and program execution traces. However, by themselves, they do not enable the desired task of function evaluation. 
# 
# The main contribution of this paper is combining symbolic expressions with black-box function evaluation data for training neural programmers. This results in generalizable models that are also capable of function evaluation. The paper studies modeling mathematical equations and it shows that this combination allows one to model up to 28 mathematical functions that scales up the domain by about $3.5\times$ while increasing the complexity of the mathematical equations compared to the state-of-the-art. The authors propose a dataset generation strategy that generates a balanced dataset of symbolic and function evaluation data with a good coverage of the domain under study. They propose using Tree LSTM's that mirror the parse-tree of the symbolic and function evaluation expressions for modeling mathematical equations. The paper finally evaluates the model on tasks such as equation verification and equation completion.
# 
# ***
# 
# ## Implementation Details
# 
# In this notebook, we present the code for training the tree LSTM model for the task of equation verification. There are also another notebooks attached, that covers the dataset generation.
# The code is implemented in Pythin 2.7 and uses [MxNet](https://mxnet.incubator.apache.org) as the underlying deep learning platform.
# 
# ### 1. Importing Modules
# <a id="sec:import"></a>
# 
# Let us start with importing the relevant modules.
# 
# Our designed neuralAlgonometry module is a module containing the tree class *EquationTree* and several other useful functions such as functions *readBlockJsonEquations* for reading the input data and *saveResults* for saving the results. Bellow is an example of equations that are represented using the EquationTree class.
# 
# $\sin^2(\theta) + \cos^2(\theta) = 1$ | $\sin(-2.5) = -0.6$ | Decimal expression tree for $2.5$
# - | - | -
# <img src="figs/eTree.png", width="300", height="300"/>  | <img src="figs/numTree.png", width="300", height="300"/> | <img src="figs/num_tree.png", width="300", height="300"/>
# 
# nnTreeMain is a module that contains the neural network tree classes. We use MxNet's [bucketingModule](https://mxnet.incubator.apache.org/how_to/bucketing.html) for implementing dynamic networks. Class *lstmTreeInpOut* implements treeLSTMs for the combination of symbolic and black-box function evaluation data. The implementation of the baseline models used in the paper are also present in nnTreeMain and are called *nnTreeInpOut*, *LSTMtree* and *nnTree* for treeNNs with a combination of symbolic and function evaluation data, treeLSTMs for symbolic data and treeNNs for symbolic data, respectively. Replacing lstmTreeInpOut with any of these calsses perform training and equation verification for these models.
# 
# *BucketEqIteratorInpOut* is the data iterator class used by the bucketing module and *bucketIndex* is the class that is passed to the *sym_gen* function of the bukcetingModule. precision, recall and accuracy are subclasses of mx.metric.EvalMetric.  
# 

# importing utilities
import mxnet as mx
import numpy as np
import random
import re
import copy
from neuralAlgonometry import buildNNTree, encode_equations, EquationTree, readBlockJsonEquations,                              saveResults, updateWorksheet, dumpParams, writeJson, putNodeInTree, encodeTerminals
from nnTreeMain import lstmTreeInpOut, BucketEqIteratorInpOut, bucketIndex, precision, recall, Accuracy


# ### 2. One-hot encoding of the terminals
# <a id="sec:one-hot"></a>
# 
# As stated in Sectin 2 of [the paper](https://openreview.net/forum?id=Hksj2WWAW&noteId=Hksj2WWAW), the terminals in the grammar are the leaves of the expression tree. In the neural network, these terminals are represented using the one-hot encoding. This function creates a dictionary containing the key-value pairs (terminal:index), wehre terminal, is one of the terminals, e.g. symbol $x, y$ or integers $1,2,3,\dots$ and index is the unique one-hot index. The terminals used in the paper are listed below. It is worth noting that these are the terminals for symbolic experssions only. The terminals for function evaluations are floating numbers of precision $2$ in the range $[-3.14, 3.14]$ and are represented using their decimal tree expanssions. This means that they can all be represented using the integers listed below. More explanation about how these floating point numbers are inputed to the neural network will be explained in [this section](#sec:sym_gen)
# 

# terminals:
variables = ['Symbol']
consts = ['NegativeOne', 'NaN', 'Infinity', 'Exp1', 'Pi', 'One',
          'Half', 'Integer', 'Rational', 'Float']
terminals = []
terminals.extend(variables) 
terminals.extend(consts)

intList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -2, -3]
ratList = ['2/5', '-1/2', '0']
floatList = [0.7]
varList = ['var_%d'%(d) for d in range(5)]

functionDictionary = encodeTerminals(terminals, intList, ratList, floatList, varList)
print "functionDictionary:", functionDictionary


# ### 3. Model Parameters
# <a id="sec:params"></a>
# 
# The model hyper-parameters are given in this block. You can change these hyper-parameters to tune the neural network.
# 
# As stated in the paper, the depth of an equation can be indicative of the equation's complexity. *trainDepth* and *testDepth* refers to the depth of equations included in training and testing. These are used for generating the results given in Tables 2 and 3 of the paper to assess the generalizability of the model. When trainDepth = testDepth = [1,2,3,4], train and test sets include all the equations of depths 1 through 4 and the performance of the model is assessed on unseen equations in the test set. These reults are shown in Table 2. In order to reproduce the results of Table 3 set:
# 
# trainDepth = [1,2,3], testDepth = [4]
# 
# and
# 
# trainDepth = [1,3,4], testDepth = [2]
# 

# Model parameters and training setup 
params = None
contexts = mx.cpu(0)
num_hidden = 50
vocabSize = len(functionDictionary)
emb_dimension = 50
out_dimension = 1
batch_size = 1
splitRatio = 0.8 # proportion of equations in the train set
devSplit = 1
excludeFunc = ''
trainDepth = [1,2,3,4]
testDepth = [1,2,3,4]
dropout = 0.2
lr = 0.00001 # learning rate
mom = 0.7 # momentum
wd = 0.001 # weight decay
optimizer = "adam" # name of optimizer
num_epochs = 2 # number of training epochs
load_epoch = 0 # load pre-trained model starting from epoch number load_epoch
model_prefix = "notebookModel/model0/trained_model " # path to save model checkpoints
kv_store = "device" # KV_store 

# refer to section 1. for an explanation about different neural network classes
tTypeStr = 'lstmTreeInpOut' # neural network type. Other options: 'nnTreeInpOut', 'nnTree', 'lstmTree'
tType = lstmTreeInpOut  # neural network type. Other options: nnTreeInpOut, nnTree, lstmTree

# path to data: below is a smaller dataset that runs faster.
# file data/data4000_orig_inpuOut_with_neg.json is the data used in the paper
path = "data/data1000_depth4_inpuOut.json" 
result_path = "notebookModel/model0/res" # path for saving results such as train/test accuracies and other metrics


# ### 4. Reading data
# <a id="sec:read"></a>
# 
# In this section we explain how to load the data generated using the dataset generation method. Function *readBlockJsonEquations* available in module neuralAlgonometry loads equations saved in a json format. You can input the train/test splitting ratio *splitRatio* and if desired a *devSplit* which holds out a portion of the train data for validation. This is set to 1 in this notebook meaning no validation data is used, but one can set it to, say, 0.9 to keep $10%$ of the data for validation. Validation data can be useful for assessing the models overfitting behavior during training.
# 
# The returned data includes the train/test/devEquations that are lists of objects of class *EquationTree*. train/test/devVars contains the list of variables in each equation. train/test/devLabels is a list of labels corresponding to each equation. Labels are either <font color=blue>mx.nd.array([0], dtype='float32')</font> , or <font color=blue>mx.nd.array([1], dtype='float32')</font> for incorrect and correct equations, respectively.
# 
# It should be noted that since the bucketing module needs to see all the neural network blocks when forming the first computation graph, The first equation is a synthetic equation including all the functions and terminals in the grammar.  Flag *containsNumeric* indicates weather the data contains function evaluations or if it only contains symbolic expressions. If the data contains fnuction evaluation data set this flag to True. In that case the first equation will be appended with a Number block.
# 
# In case the random seed is not sat, the created data split may not be reproducible. Therefore, one can save the original split, if further analysis needs to be done on the data using the saved data. Function *writeJson* is a function available in the neuralAlgonometry module. It saves the trees in a the json format. The format of the equations are described in the *generateData.ipynb* notebook. This is commented out in the last three lines of the cell below. But can be uncommented if it is desirable to save the splits. 
# 

random.seed(1)
mx.random.seed(1)
np.random.seed(1)
[trainEquations, trainVars, _, trainLabels,
devEquations, devVars, _, devLabels,
testEquations, testVars, _, testLabels] \
          = readBlockJsonEquations(path, trainDepth=trainDepth, testDepth=testDepth,
                                   excludeFunc=excludeFunc, splitRatio=splitRatio, devSplit=devSplit, containsNumeric=True)
    
# uncomment if need to store the original data split
# writeJson(result_path+'trainData.json', trainEquations, ranges, trainVars, trainLabels, 6)
# writeJson(result_path+'devData.json', devEquations, ranges, devVars, devLabels, 6)
# writeJson(result_path+'testData.json', testEquations, ranges, testVars, testLabels, 6)


# ### 5. Construct neural network classes
# <a id="sec:nn"></a>
# 
# In this block we construct the neural network classes for each equation in the train and test set. If you have sat devSplit to something other than 1, then you should also construct the network for your validation set. This can be done by uncommenting the last part of the code in the cell below.
# 
# Function *buildNNTree*, that is implemented in module neuralAlgonometry, traverses the input equation and constructs a treeLSTM (or another model if tType is different) that mirrors the equation's structure.  
# 
# The figures below depict the neural network constrcted using this function for the equations in [this section](#sec:import)
# 
# $\sin^2(\theta) + \cos^2(\theta) = 1$ | $\sin(-2.5) = -0.6$ 
# - | -
# <img src="figs/network_sym.png", width="400", height="400"/>  | <img src="figs/network_num.png", width="400", height="400"/>
# 

trainSamples = []
trainDataNames = []
for i, eq in enumerate(trainEquations):
    currNNTree = buildNNTree(treeType=tType , parsedEquation=eq, 
                                num_hidden=num_hidden, params=params, 
                                emb_dimension=emb_dimension, dropout=dropout)

    [currDataNames, _] = currNNTree.getDataNames(dataNames=[], nodeNumbers=[])
    trainDataNames.append(currDataNames)
    trainSamples.append(currNNTree)

testSamples = []
testDataNames = []
for i, eq in enumerate(testEquations):
    currNNTree = buildNNTree(treeType=tType , parsedEquation=eq, 
                                num_hidden=num_hidden, params=params, 
                                emb_dimension=emb_dimension, dropout=dropout)

    [currDataNames, _] = currNNTree.getDataNames(dataNames=[], nodeNumbers=[])
    testDataNames.append(currDataNames)
    testSamples.append(currNNTree)
    
# devSamples = []
# devDataNames = []
# for i, eq in enumerate(devEquations):
#     currNNTree = buildNNTree(treeType=tType , parsedEquation=eq, 
#                            num_hidden=num_hidden, params=params, 
#                            emb_dimension=emb_dimension, dropout=dropout)

#     [currDataNames, _] = currNNTree.getDataNames(dataNames=[], nodeNumbers=[])
#     devDataNames.append(currDataNames)
#     devSamples.append(currNNTree)


# ### 6. Construct data iterators
# <a id="sec:iter"></a>
# 
# Class *BucketEqIteratorInpOut* which is a subclass of <font color=blue>mx.io.DataIter</font>. It constructs the data iterators for the train and test equations. If you have sat devSplit to something other than 1, you need to have a data iterator for your validation set. This can be done by uncommenting the code below.
# 

numTrainSamples = len(trainEquations)
trainBuckets = list(xrange(numTrainSamples))

numTestSamples = len(testEquations)
testBuckets = list(xrange(numTestSamples))

train_eq, _ = encode_equations(trainEquations, vocab=functionDictionary, invalid_label=-1, 
                                     invalid_key='\n', start_label=0)
data_train  = BucketEqIteratorInpOut(enEquations=train_eq, eqTreeList=trainSamples, batch_size=batch_size, 
                             buckets=trainBuckets, labels=trainLabels, vocabSize=len(functionDictionary),
                                    invalid_label=-1)

test_eq, _ = encode_equations(testEquations, vocab=functionDictionary, invalid_label=-1, 
                             invalid_key='\n', start_label=0)
data_test  = BucketEqIteratorInpOut(enEquations=test_eq, eqTreeList=testSamples, batch_size=batch_size, 
                             buckets=testBuckets, labels=testLabels, vocabSize=len(functionDictionary),
                                    invalid_label=-1, devFlag=1)


# numDevSamples = len(devEquations)
# devBuckets = list(xrange(numDevSamples))
# dev_eq, _ = encode_equations(devEquations, vocab=functionDictionary, invalid_label=-1, 
#                              invalid_key='\n', start_label=0)
# data_dev  = BucketEqIteratorInpOut(enEquations=dev_eq, eqTreeList=devSamples, batch_size=batch_size, 
#                              buckets=devBuckets, labels=devLabels, vocabSize=len(functionDictionary),
#                                     invalid_label=-1, devFlag=1)


# ### 6. Symbol generator function for the bucketing module
# <a id="sec:sym_gen"></a>
# 
# Defining the sym_gen function for MxNet's [bucketing module](https://mxnet.incubator.apache.org/how_to/bucketing.html) and constructing the neural network model that forms the computation graph. This function returns the prediction symbol as well as the data names and label names. *data_names_corr* is a list that contains the data which contains the terminals' names. 
# 
# For the terminals that are represented with their one-hot encoding, we have a one-layer neural network block that is responsible for embedding the representation of that cell (cell $W_{sym}$ in [section 5](#sec:nn)). For floting point numbers in the range $[-3.14, 3.14]$ we have a two-layer neural network that is responsible for encoding its values (cell $W_{num}$ in [section 5](#sec:nn)). Floating point numbers are inputed as is to the neural network.
# 
# The final model is an instance of MxNet's *BucketingModule*. 
# 

cell = trainSamples[0]

def sym_gen(bucketIndexObj):

    label = mx.sym.Variable('softmax_label')

    bucketIDX = bucketIndexObj.bucketIDX
    devFlag = bucketIndexObj.devFlag

    if devFlag == 0:
        tree = trainSamples[bucketIDX]
    else:
        tree = testSamples[bucketIDX]


    [dataNames, nodeNumbers] = tree.getDataNames(dataNames=[], nodeNumbers=[])
    data_names_corr = [dataNames[i]+'_%d'%(nodeNumbers[i]) for i in range(len(dataNames))]
    nameDict = {}
    for i, dn in enumerate(dataNames):
        if dn not in nameDict:
            nameDict[dn+'_%d'%(nodeNumbers[i])] = mx.sym.Variable(name=dn+'_%d'%(nodeNumbers[i]))
        else:
            raise AssertionError("data name should not have been in the dictionary")

    if tType == lstmTreeInpOut:
        outputs, _ = tree.unroll(nameDict)
    else:
        outputs = tree.unroll(nameDict)

    pred = mx.sym.LogisticRegressionOutput(data=outputs, label=label, name='softmax')

    return pred, (data_names_corr), ('softmax_label',)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen,
    default_bucket_key  = bucketIndex(0, 0),
    context             = contexts,
    fixed_param_names  = [str(tTypeStr)+'_Equality_i2h_weight'])


# ### 7. Training
# <a id="sec:train"></a>
# 
# In this section we perform the training using model.fit($\dots$). Once ran, the training and test accuracies will be shown in the output log. Function *saveResults*, saves the precision, recall and accuracy metrics for the train and test data in the *result_path* whose value is sat in [Section 3](#sec:params)
# 

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

acc = Accuracy()
prc = precision()
rcl = recall()

if load_epoch:
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
        cell, model_prefix, load_epoch)
else:
    arg_params = None
    aux_params = None

opt_params = {
'learning_rate': lr,
'wd': wd
}

if optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
    opt_params['momentum'] = mom

model.fit(
train_data          = data_train,
eval_data           = data_test,
kvstore             = kv_store,
eval_metric         = [acc, prc, rcl],
optimizer           = optimizer,
optimizer_params    = opt_params,
initializer         = mx.init.Mixed([str(tTypeStr)+'_Equality_i2h_weight', '.*'], 
                                [mx.init.One(), mx.init.Xavier(factor_type="in", magnitude=2.34)]),
arg_params          = arg_params,
aux_params          = aux_params,
begin_epoch         = load_epoch,
num_epoch           = num_epochs,
epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, model_prefix, 1) \
                                               if model_prefix else None)

accTrain = [acc.allVals[i] for i in range(0,len(acc.allVals),2)]
accVal   = [acc.allVals[i] for i in range(1,len(acc.allVals),2)]
prcTrain = [prc.allVals[i] for i in range(0,len(prc.allVals),2)]
prcVal   = [prc.allVals[i] for i in range(1,len(prc.allVals),2)]
rclTrain = [rcl.allVals[i] for i in range(0,len(rcl.allVals),2)]
rclVal   = [rcl.allVals[i] for i in range(1,len(rcl.allVals),2)]

trainMetrics = [accTrain, prcTrain, rclTrain]
valMetrics   = [accVal,     prcVal,   rclVal]

# args
if result_path:
    saveResults(result_path+'_train.json', {}, trainMetrics, valMetrics)
    trainPredicts = model.predict(data_train).asnumpy()
    np.save(result_path+'_train_predictions.npy', trainPredicts)
    with open(result_path+'_train_labels.txt', 'wa') as labelFile:
        for lbl in trainLabels:
            labelFile.write('{0}\n'.format(lbl.asscalar()))


# ### Appendix: Pre-setup
# 
# **Note: ** MxNet's Parameter allow_extra_params should be sat to True as shown in [this commit](https://github.com/Mega-DatA-Lab/mxnet/commit/13505824699cfc39d8ea52537c56bd5aaf9639b6) for this code to work properly. This is used for handling dynamic graphs.
# 
# **Note: ** Use the _update_params(...) function in [this commit](https://github.com/Mega-DatA-Lab/mxnet/commit/960af8aa713f00e4dd6240dcc4f03867e8ac9f23) in MxNet's [model.py](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/model.py).
# 




# # Automatic Symbolic Equation Generator
# 
# *Developed by Forough Arabshahi*
# 
# This notebook presents the code for the automatic dataset generation scheme presented in [this](https://openreview.net/forum?id=Hksj2WWAW&noteId=Hksj2WWAW) paper.
# 
# This equation generation approach is based on generating new mathematical identities by performing local random changes to known identities, starting with a small number of axioms from the domain under study. These changes result in identities of similar or higher complexity (equal or larger depth), which may be correct or incorrect, that are valid expressions within a grammar. The grammar and its rules are defined in detail in [section 2.1 of the paper](https://openreview.net/forum?id=Hksj2WWAW&noteId=Hksj2WWAW). In this notebook we choose elementary algebra and trigonometry as the domain under study.
# 
# ### Generating Possible Equations
# <a id="sec:posEq"></a>
# In order to generate a new identity which may be correct or incorrect, we select an equation at random from the set of known equations, and make local changes to it. We randomly select a node in the expression tree, followed by randomly selecting one of the following actions to make the local change to the equation at the selected node:
# * ShrinkNode: Replace the node, if it’s not a leaf, with one of its children, chosen randomly. 
# * ReplaceNode: Replace the symbol at the node (i.e. the terminal or the function) with another
# compatible one, chosen randomly.
# * GrowNode: Provide the node as input to another randomly drawn function f , which then
# replaces the node. If f takes two inputs, the second input will be generated randomly from
# the set of terminals.
# * GrowSides: If the selected node is an equality, either add or multiply both sides with a
# randomly drawn number, or take both sides to the power of a randomly drawn number.
# 
# This is implemented in a function called *genNegEquation*
# 
# ### Generating Additional Correct Equations
# In order to generate only correct identities, we follow the same intuition as above, but only replace structure with others that are equal. In particular, we maintain a dictionary of valid statements (mathDictionary) that maps a mathematical statement to another. For example, the dictionary key $x + y$ has value $y + x$. We use this dictionary in our correct equation generation process where we look up patterns from the dictionary. More specifically, we look for keys that match a subtree of the equation then replace that subtree with the pattern of the value of the key. E.g. given input equation $\sin^2(\theta) + \cos^2(\theta) = 1$, this subtree matching might produce equality $\cos^2(\theta) + \sin^2(\theta) = 1$ by finding key-value pair $x + y : y + x$.
# 
# This is implemented in a function called *genPosEquation*
# 
# ***
# 
# ## Implementation Details
# 
# The dependency for this notebook if the [latex2sympy](https://github.com/augustt198/latex2sympy) package which converts a latex equation to sympy. Set the path to latex2sympy in file eqGen.py available in the root folder. Our equations are represented using the *EquationTree* object. The attributes of this object are: 
# 1. func: string that stores the function name
# 2. args: list of EquationTree objects that are the children of this node
# 3. varname: string that stores the variable name if the node is a leaf, otherwise this is ''
# 4. number: integer that indicates the node's unique number
# 4. depth: integer that indicates the depth of the subtree that is attached to the node.
# 
# below is an example of some equations
# 
# $\sin^2(\theta) + \cos^2(\theta) = 1$ | $\sin(-2.5) = -0.6$ | Decimal expression tree for $2.5$
# - | - | -
# <img src="figs/eTree.png", width="300", height="300"/>  | <img src="figs/numTree.png", width="300", height="300"/> | <img src="figs/num_tree.png", width="300", height="300"/>
# 
# ### 1. Importing Modules
# <a id="sec:import"></a>
# 
# Most of the functions are implemented in *eqGen.py*. *readJson* reads an input json in latex format and uses latex2sympy to convert it to a sympy equation. Finally, *buildEq* converts a sympy equation to an EquationTree object. *readAxioms* reads an input txt file and uses the compiler package to convert the equations to a compiler object. Function *parseEquation* then converts this to an EquationTree object. *writeJson* saves the generated EquationTree objects in a json file.
# 

get_ipython().system('pip install -i https://pypi.anaconda.org/pypi/simple antlr4-python2-runtime')


get_ipython().system('git clone https://github.com/augustt198/latex2sympy')


get_ipython().system('cd latex2sympy; antlr4 PS.g4 -o gen')


get_ipython().system('pip install mxnet')


import json
from sympy import *
import pprint
import re
import copy
import random
import compiler
import mxnet as mx
import numpy as np
import sys
sys.path.append("./latex2sympy")
from eqGen import readJson, readAxioms, parseEquation, buildEq, buildAxiom, genPosEquation,                  genNegEquation, isCorrect, writeJson
from neuralAlgonometry import catJsons


# ### 2. Parsing input axioms
# <a id="sec:axioms"></a>
# 
# In this section we a small set of input axioms from trigonometry and elementary algebra and convert them to EquationTree objects. trigonometry equations are in file *"axioms/trigonometryAxioms.json"* which are collected from the wikipedia [List of Trigonometric Identities](https://en.wikipedia.org/wiki/List_of_trigonometric_identities) page. Elementary algebra equations are in file *"axioms/algebraAxioms.txt"* and are hand generated.
# 

# path to trigonometry equations collected from wiki
inputPath = "axioms/trigonometryAxioms.json"
# path to some axioms from elementary algebra
axiomPath = "axioms/algebraAxioms.txt"
jsonAtts = ["equation", "range", "variables","labels"]

labels = []
inputRanges = []
inputEquations = [] 
eqVariables = []
parsedEquations = []
parsedRanges = []
ranges = []
inputAxioms = []
axiomVariables = []
axioms = []
axLabels = []

random.seed(1)


# Function *readJson* parses latex equations. An example of the resulting output is given below
# 

# parses latex equations from file:
readJson(inputPath, inputEquations, inputRanges, jsonAtts)
inputEquations[1]


# We use function *parseEquation* to convert these equations to a sympy equation using the latex2sympy package. An example of the output equation follows:
# 

# Converts latex equations to sympy equations using process_latex
parseEquation(inputEquations, parsedEquations, eqVariables)
parsedEquations[1]


# Function *buildEq* converts each sympy equation to an EquationTree object as shown below. The pre order traversal of the parsed equation is also shown as an example
# 

# converts equations from sympy format to EquationTree object
# equations pretty print as well as pre order traversal follows
equations = []
for i, eq in enumerate(parsedEquations):
    # building EquationTree object
    currEq = buildEq(eq, eqVariables[i])
    # assigning a unique number to each node in the tree as well as indicating subtree depth at each level
    currEq.enumerize_queue()
    equations.append(currEq)
    
    # creating training labels
    # the first equation in the input function is incorrect. It has been deliberately added
    # to include all possible functionalities in the functionDictionary. 
    # This is for compatibility with MxNet's bucketingModule.
    if i == 0:
        labels.append(mx.nd.array([0]))
    else:
        labels.append(mx.nd.array([1]))
    
print "currEq:", equations[1]
print "pre order traversal"
equations[1].preOrder()


# After parsing trigonometry axioms, we start parsing algebra axioms using python's built in compiler and function *readAxioms*. The parsed equation is shown below:
# 

# parses text equations using the compiler package and returns an equation in the compiler format
readAxioms(axiomPath, inputAxioms, axiomVariables)
inputAxioms[0]


# Once we have the compiler object, we can convert it to an EquationTree object using function *buildAxiom*. An example parsed equation as well its pre order parse is given below.
# 

# converting compiler object axioms to EquationTree objects and creating training labels
for i, ax in enumerate(inputAxioms):
    currAxiom = buildAxiom(ax, axiomVariables[i])
    currAxiom.enumerize_queue()
    axioms.append(currAxiom)
    axLabels.append(mx.nd.array([1]))
    
print "an axiom:", axioms[0]
print "pre order traversal:"
axioms[0].preOrder()


# appending algebra axioms to trigonometry axioms
equations.extend(axioms)
eqVariables.extend(axiomVariables)
labels.extend(axLabels)
print len(equations)
print len(eqVariables)
print len(labels)


# The distribution of the depth of the equations, that contains all axioms from trigonometry and algebra, is shown below. The vector shows the number of equations of depth *i* at *i*th vector position.
# 

depthMat = [0 for _ in range(26)]
for eq in equations:
    depthMat[eq.depth] += 1
print "distribution of depth of equations"
print depthMat[:10]


# ### 3. Contructing the mathDictionary
# <a id="sec:dict"></a>
# 
# Here, we construct the *mathdictionary* which is used for generating additional correct identities. This dictionary contains (key, value) pairs that are mathematically equivalent. E.g. $(x+y : y+x)$ is a (key,value) pair in the mathDictionary
# 

# constructing the mathDictionary whose (key,value) pairs are valid math equalities
# e.g. (x+y : y+x) is a (key,value) pair in the mathDictionary
# the dictionary will be updated as more correct equations are generated
mathDictionary = {}
strMathDictionary = {}
for i, eq in enumerate(equations):
    if i!=0:
        eqCopy = copy.deepcopy(eq)
        if str(eqCopy) not in strMathDictionary:
            strMathDictionary[str(eqCopy)] = 1
            mathDictionary[eqCopy.args[0]] = eqCopy.args[1]
            mathDictionary[eqCopy.args[1]] = eqCopy.args[0]
        else:
            strMathDictionary[str(eqCopy)] += 1
# for k, v in strMathDictionary.iteritems():
#     print k, v
print len(strMathDictionary)
print len(mathDictionary)


# ### 4. Generating correct equations using mathDictionary lookup
# <a id="sec:subtreeMatching"></a>
# 
# Function *genPosEquation* generates a candidate correct equation using subtree matching from a dictionary lookup. More specifically, it chooses a random node of the equation and looks for patterns in the dictionary key that match the subtree of the chosen node. The subtree is then replaced by the dictionary key's value. 
# 
# The code snippet below generates about 10 correct equations by making local changes to equations selected at random from the input axioms. If no duplicate equation is generated, this equation will be added to the list of equations. The depth of the generated equation is limited to 7 (*maxDepth*). parameter *maxDepthSoFar* along with *thrsh* control the generated number of equations of a certain depth before moving to the next depth. This is a good control for training since it ensures that a minimum number of equations of each depth are present in the final dataset. It shoul dbe noted that as depth increases, the space grows exponentially, and this does not aim to cover this exponential space. 
# 
# The distribution of the generated equations are shown below. 
# 

maxDepth = 7
numPosEq = 10
numNegEq = 10
numNegRepeats = 2
thrsh = 5

# set maxDepthSoFar to 0 to generate up to thrsh number of 
# repeated equations before moving to equations of higher depth
maxDepthSoFar = 7
totDisc = 0
for i in range(0, numPosEq):
    randInd = random.choice(range(1, len(equations)))
    while labels[randInd].asnumpy() == 0:
        randInd = random.choice(range(1, len(equations)))
    randEq = copy.deepcopy(equations[randInd])
    randEqVariable = copy.deepcopy(eqVariables[randInd])

    posEq = genPosEquation(randEq, mathDictionary, randEqVariable)
    posVars = posEq.extractVars()
    posEq.enumerize_queue()

    old = 0
    disc = 0
    tries = 0
    # this loop is to make sure there are no repeats and also that enough 
    # number of equations of a certain depth are generated
    while str(posEq) in strMathDictionary or posEq.depth > maxDepthSoFar:
        if str(posEq) in strMathDictionary:
            strMathDictionary[str(posEq)] += 1
            old += 1
            totDisc += 1
        elif posEq.depth > maxDepthSoFar:
            disc += 1
            totDisc += 1

        if old > thrsh:
            old = 0
            maxDepthSoFar += 1
            print "new max depth %d" %(maxDepthSoFar)
            if maxDepthSoFar > maxDepth:
                print "reached maximum depth"
                maxDepthSoFar = maxDepth
                break

        randInd = random.choice(range(1, len(equations)))
        randEq = equations[randInd]
        randEqVariable = copy.deepcopy(eqVariables[randInd])
        posEq = genPosEquation(randEq, mathDictionary, randEqVariable)
        posVars = posEq.extractVars()
        posEq.enumerize_queue()

    if posEq.depth <= maxDepth:
        posEqCopy = copy.deepcopy(posEq)

        if str(posEqCopy) not in strMathDictionary:
            strMathDictionary[str(posEqCopy)] = 1
            if posEqCopy.args[0] not in mathDictionary:
                mathDictionary[posEqCopy.args[0]] = posEqCopy.args[1]
            if posEqCopy.args[1] not in mathDictionary:
                mathDictionary[posEqCopy.args[1]] = posEqCopy.args[0]

            equations.append(posEq)
            eqVariables.append(posVars)
            labels.append(mx.nd.array([1]))
    else:
        totDisc += 1
        print "discarded pos equation of depth greater than %d: %s" %(maxDepth, str(posEq))

depthMat = [0 for _ in range(26)]
for eq in equations:
    depthMat[eq.depth] += 1
print "distribution of depth of equations"
print depthMat


# ### 5. Generating correct and incorrect equations using local changes
# <a id="sec:valid"></a>
# 
# Function *genNegEquation* generated a candidate correct or incorrect mathematical equation. This function takes as input an EquationTree object. It selects a node chosen at random from that equations and performs one of the operations explained in [this section](#sec:posEq) to the node depending on the type of node. We check the correctness or incorrectness of this generated equation using function *isCorrect* that uses sympy. The depth of the final equations are shown below.
# 

# generating negative equations
negLabels= [[] for _ in range(numNegRepeats)]
negEquations = [[] for _ in range(numNegRepeats)]
negEqVariables = [[] for _ in range(numNegRepeats)]
negStrMathDictionary = {}
corrNegs = 0
totDiscNeg = 0
ii = len(equations)
for i in range(1, len(equations)): 
    for rep in range(numNegRepeats):
        randInd = i
        randEq = copy.deepcopy(equations[i])
        randEqVariable = copy.deepcopy(eqVariables[randInd])

        negEq = genNegEquation(randEq, randEqVariable)
        negVars = negEq.extractVars()
        negEq.enumerize_queue()
        disc = 0
        tries = 0
        old = 0
        while str(negEq) in negStrMathDictionary or negEq.depth > maxDepth:
            if str(negEq) in negStrMathDictionary:
                negStrMathDictionary[str(negEq)] += 1
                old += 1
                totDiscNeg += 1
                # print "repeated neg equation"
            elif negEq > maxDepth:
                # print "equation larger than depth"
                disc += 1
                totDiscNeg += 1

            if old > thrsh:
                old = 0
                break

            negEq = genNegEquation(randEq, randEqVariable)
            negVars = negEq.extractVars()
            negEq.enumerize_queue()

        if negEq.depth <= maxDepth:
            
            negEqCopy = copy.deepcopy(negEq)
            try:
                isCorrect(negEq)

                if isCorrect(negEq):
                    corrNegs += 1

                    print "correct negative Eq:", negEq

                    if str(negEq) not in strMathDictionary:

                        strMathDictionary[str(negEqCopy)] = 1
                        if negEqCopy.args[0] not in mathDictionary:
                            mathDictionary[negEqCopy.args[0]] = negEqCopy.args[1]
                        if negEqCopy.args[1] not in mathDictionary:
                            mathDictionary[negEqCopy.args[1]] = negEqCopy.args[0]

                        labels.append(mx.nd.array([1]))
                        equations.append(negEq)
                        eqVariables.append(negVars)

                elif str(negEqCopy) not in negStrMathDictionary:
                        negStrMathDictionary[str(negEqCopy)] = 1

                        negLabels[rep].append(mx.nd.array([0]))
                        negEquations[rep].append(negEq)
                        negEqVariables[rep].append(negVars)
                else:
                    totDiscNeg += 1

            except:

                if str(negEqCopy) not in negStrMathDictionary:
                    negStrMathDictionary[str(negEqCopy)] = 1

                    negLabels[rep].append(mx.nd.array([0]))
                    negEquations[rep].append(negEq)
                    negEqVariables[rep].append(negVars)
                else:
                    totDiscNeg += 1

        else:
            totDiscNeg += 1
            print "discarded neg equation of depth greater than %d: %s" %(maxDepth, str(negEq))

depthMat = [0 for _ in range(26)]
for eq in negEquations[0]:
    depthMat[eq.depth] += 1
print "distribution of depth of neg equations"
print depthMat


# ### 6. Saving the generated dataset
# <a id="sec:save"></a>
# 
# Finally, we would like to save the resulting dataset which will be used to train neural network models for mathematical equalities. Function *writeJson* writes this resulting dataset to a json. In order to save an EquationTree object, we use the pre-order traversal of each equation. E.g. In order to save equation $0=0$, we construct:
# 
# "equation": {
#                * "vars": ",0,#,#,0,#,#", 
#                * "numNodes": "3", 
#                * "variables": {}, 
#                * "depth": "1,0,#,#,0,#,#", 
#                * "nodeNum": "0,1,#,#,2,#,#", 
#                * "func": "Equality,Integer,#,#,Integer,#,#"
#             },
#             
# Where # indicates a null pointer. It should be noted that the trees are binary. 
# 
# Finally function *catJsons* concatenates all the created jsons into a single file containing correct and incorrect equations that can be loaded for training.
# 

# writing equations to file
writeJson("data/data%d_pos.json"%(numPosEq), equations, ranges, eqVariables, labels, maxDepth)
for rep in range(numNegRepeats):
    writeJson("data/data%d_neg_rep%d.json"%(numNegEq,rep), negEquations[rep], ranges, negEqVariables[rep], negLabels[rep], maxDepth)


catJsons(['data/data%d_pos.json'%(numPosEq), 'data/data%d_neg_rep%d.json'%(numNegEq,0), 'data/data%d_neg_rep%d.json'%(numNegEq,1)],
          'data/data%d_final.json'%(numPosEq), maxDepth=maxDepth)





# author: Ashish Khetan, khetan2@illinois.edu, Zachary C. Lipton, Animashree Anandkumar
# 
# This notebook implements MBEM algorithm proposed in the paper "Learning From Noisy Singly-labeled Data" which is under review at ICLR 2018.
# 
# Model Bootstrapped Expectation Maximization (MBEM) is a new algorithm for training a deep learning model using noisy data collected from crowdsourcing platforms such as Amazon Mechanical Turk. MBEM outperforms classical crowdsourcing algorithm "majority vote". In this notebook, we run MBEM on CIFAR-10 dataset. We synthetically generate noisy labels given the true labels and using hammer-spammer worker distribution for worker qualities that is explained in the paper. Under the setting when the total annotation budget is fixed, that is we choose whether to collect "1" noisy label for each of the "n" training samples or collect "r" noisy labels for each of the "n/r" training examples.
# 
# we show empirically that it is better to choose the former case, that is collect "1" noisy label per example for as many training examples as possible when the total annotation budget is fixed. It takes a few hours to run this notebook and obtain the desired numerical results when using gpus. We use ResNet deep learning model for training a classifier for CIFAR-10. We use ResNet MXNET implementation given in https://github.com/tornadomeet/ResNet/. 
# 

get_ipython().system('pip install mxnet')


import mxnet as mx
import numpy as np
import logging,os
import copy
import urllib
import logging,os,sys
from scipy import stats
from random import shuffle
from __future__ import division

# Downloading data for CIFAR10
# The following function downloads .rec iterator and .lst files (MXNET iterators) for CIFAR10 
# that are used for training the deep learning model with noisy annotations
def download_cifar10():
    fname = ['train.rec', 'train.lst', 'val.rec', 'val.lst']
    testfile = urllib.URLopener()
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fname[0])
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_train.lst', fname[1])
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_val.rec',   fname[2])
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_val.lst',   fname[3])
    return fname
# download data
fname = download_cifar10()
# setting up values according to CIFAR10 dataset
# n is total number of training samples for CIFAR10
# n1 is the total number of test samples for CIFAR10 
# k is the number of classes
n, n1, k = 50000, 10000, 10
    
#setting the number of gpus that are available
gpus = None #'0,1,2,3' # if there are no gpus available set it to None.

# m is the number of workers,  gamma is the worker quality, 
# class_wise is the binary variable: takes value 1 if workers are class_wise hammer spammer 
# and 0 if workers are hammer-spammer
# k is the number of classification classes, 
# epochs is the number of epochs for ResNet model
m, gamma, class_wise, epochs, depth  = 100, 0.2, 0, 2, 20

#### main function ####    
def main(fname,n,n1,k,conf,samples,repeat,epochs,depth,gpus):    
    # defining the range of samples that are to be used for training the model
    valid = np.arange(0,samples)
    # declaring the other samples to be invalid 
    invalid = np.arange(samples,n)

    # calling function generate_labels_weight which generates noisy labels given the true labels 
    # the true lables of the examples are ascertained from the .lst files 
    # it takes as input the following:
    # name of the .lst files for the training set and the validation set
    # conf: the confusion matrices of the workers
    # repeat: number of redundant labels that need to be generated for each sample
    # for each i-th sample repeat number of workers are chosen randomly that labels the given sample
    # it returns a multi dimensional array resp_org: 
    # such that resp_org[i,j,k] is 0 vector if the a-th worker was not chosen to label the i-th example
    # else it is one-hot representation of the noisy label given by the j-th worker on the i-th example
    # workers_train_label_org: it is a dictionary. it contains "repeat" number of numpy arrays, each of size (n,k)
    # the arrays have the noisy labels given by the workers
    # workers_val_label: it is a dictionary. it contains one numpy array of size (n,k) 
    # that has true label of the examples in the validation set
    # workers_this_example: it is a numpy array of size (n,repeat).
    # it conatins identity of the worker that are used to generate "repeat" number of noisy labels for example    
    resp_org, workers_train_label_org, workers_val_label, workers_this_example = generate_labels_weight(fname,n,n1,repeat,conf)    
    #setting invalid ones 0, so that they are not used by deep learning module
    for r in range(repeat):
        workers_train_label_org['softmax'+ str(r) +'_label'][invalid] = 0       
    
    print "Algorithm: majority vote:\t\t",
    # running the baseline algorithm where the noisy labels are aggregated using the majority voting
    # calling majority voting function to aggregate the noisy labels
    pred_mv = majority_voting(resp_org[valid])    
    # call_train function takes as input the noisy labels "pred_mv", trains ResNet model for the given "depth"
    # for "epochs" run using the available "gpus". 
    # it prints the generalization error of the trained model.
    _, val_acc = call_train(n,samples,k,pred_mv,workers_val_label,fname,epochs,depth,gpus)
    print "generalization_acc:  " + str(val_acc)
    
    print "Algorithm: weighted majority vote:\t", 
    # running the another baseline algorithm where the aggregation is performed using the weighted majority vote
    # creating a numpy array to store weighted majority vote labels
    naive_agg = np.zeros((n,k))
    # generating the weighted majority vote label using the original noisy labels stored in the 
    # dictionary "workers_train_label_org"
    for r in range(repeat):
        naive_agg = naive_agg + (1/repeat)*copy.deepcopy(workers_train_label_org['softmax'+ str(r) +'_label']) 
    # calling the "call_train" function which besides printing the generalization error 
    # returns model prediction on the training examples, which is being stored in the variable "naive_pred".
    naive_pred, val_acc = call_train(n,samples,k,naive_agg[valid],workers_val_label,fname,epochs,depth,gpus)
    print "generalization_acc:  " + str(val_acc)

    print "Algorithm: MBEM:\t\t\t",    
    # running the proposed algorithm "MBEM: model bootstrapped expectation maximization" 
    # computing posterior probabilities of the true labels given the noisy labels and the worker identities.
    # post_prob_DS function takes the noisy labels given by the workers "resp_org", model prediction obtained 
    # by running "weighted majority vote" algorithm, and the worker identities.
    probs_est_labels = post_prob_DS(resp_org[valid],naive_pred[valid],workers_this_example[valid])      
    algo_agg = np.zeros((n,k))    
    algo_agg[valid] = probs_est_labels
    # calling the "call_train" function with aggregated labels being the posterior probability distribution of the 
    # examples given the model prediction obtained using the "weighted majority vote" algorithm.
    _, val_acc = call_train(n,samples,k,algo_agg[valid],workers_val_label,fname,epochs,depth,gpus)
    print "generalization_acc:  " + str(val_acc)
    
def call_train(n,samples,k,workers_train_label_use,workers_val_label,fname,epochs,depth,gpus):
    # this function takes as input aggregated labels of the training examples
    # along with name of the .rec files for training the ResNet model, depth of the model, number of epochs, and gpus information
    # it returns model prediction on the training examples.
    # we train the model twice first using the given aggregated labels and
    # second using the model prediction on the training examples on based on the first training
    # this aspect is not covered in the algorithm given in the paper. however, it works better in practice.
    # training the model twice in this fashion can be replaced by training once for sufficiently large number of epochs
    
    # first training of the model using the given aggregated labels 
    workers_train_label_use_core = np.zeros((n,k))
    workers_train_label_use_core[np.arange(samples)] = workers_train_label_use        
    pred_first_iter, val_acc = call_train_core(n,samples,k,workers_train_label_use_core,workers_val_label,fname,epochs,depth,gpus)
    # second training of the model using the model prediction on the training examples based on the first training.
    workers_train_label_use_core = np.zeros((n,k))
    workers_train_label_use_core[np.arange(samples)] = pred_first_iter[np.arange(samples)]
    pred_second_iter, val_acc = call_train_core(n,samples,k,workers_train_label_use_core,workers_val_label,fname,epochs,depth,gpus)
    return pred_second_iter, val_acc
    
def call_train_core(n,samples,k,workers_train_label_use_core,workers_val_label,fname,epochs,depth,gpus):
    # this function takes as input the same variables as the "call_train" function and it calls
    # the mxnet implementation of ResNet training module function "train" 
    workers_train_label = {} 
    workers_train_label['softmax0_label'] = workers_train_label_use_core  
    prediction, val_acc = train(gpus,fname,workers_train_label,workers_val_label,numepoch=epochs,batch_size=500,depth = depth,lr=0.5)
    model_pred = np.zeros((n,k))
    model_pred[np.arange(samples), np.argmax(prediction[0:samples],1)] = 1
    return model_pred, val_acc 

def generate_workers(m,k,gamma,class_wise):
    # Generating worker confusion matrices according to class-wise hammer-spammer distribution if class_wise ==1
    # Generating worker confusion matrices according to hammer-spammer distribution if class_wise ==0    
    # One row for each true class and columns for given answers
    
    #iniializing confusion matrices with all entries being equal to 1/k that is corresponding to a spammer worker.
    conf = (1/float(k))*np.ones((m,k,k))
    # a loop to generate confusion matrix for each worker 
    for i in range(m): 
        # if class_wise ==0 then generating worker confusion matrix according to hammer-spammer distribution
        if(class_wise==0):
            #letting the confusion matrix to be identity with probability gamma 
            if(np.random.uniform(0,1) < gamma):
                conf[i] = np.identity(k)
            # To avoid numerical issues changing the spammer matrix each element slightly    
            else:
                conf[i] = conf[i] + 0.01*np.identity(k)
                conf[i] = np.divide(conf[i],np.outer(np.sum(conf[i],axis =1),np.ones(k)))        
        else:
            # if class_wise ==1 then generating each class separately according to hammer-spammer distribution    
            for j in range(k):
                # with probability gamma letting the worker to be hammer for the j-th class
                if(np.random.uniform(0,1) < gamma):
                    conf[i,j,:] = 0
                    conf[i,j,j] = 1 
                # otherwise letting the worker to be spammer for the j-th class. 
                # again to avoid numerical issues changing the spammer distribution slighltly 
                # by generating uniform random variable between 0.1 and 0.11
                else:
                    conf[i,j,:] = 1
                    conf[i,j,j] = 1 + np.random.uniform(0.1,0.11)
                    conf[i,j,:] = conf[i,j,:]/np.sum(conf[i,j,:])
    # returining the confusion matrices 
    return conf

def generate_labels_weight(fname,n,n1,repeat,conf):
    # extracting the number of workers and the number of classes from the confusion matrices
    m, k  = conf.shape[0], conf.shape[1]    
    # a numpy array to store true class of the training examples
    class_train = np.zeros((n), dtype = np.int)
    # reading the train.lst file and storing true class of each training example
    with open(fname[1],"r") as f1:
        content = f1.readlines()
    for i in range(n):
        content_lst = content[i].split("\t")
        class_train[i] = int(float(content_lst[1]))
    
    # a dictionary to store noisy labels generated using the worker confusion matrices for each training example  
    workers_train_label = {}
    # the dictionary contains "repeat" number of numpy arrays with keys named "softmax_0_label", where 0 varies
    # each array has the noisy labels for the training examples given by the workers
    for i in range(repeat):
        workers_train_label['softmax' + str(i) + '_label'] = np.zeros((n,k))   
    
    # Generating noisy labels according the worker confusion matrices and the true labels of the examples
    # a variable to store one-hot noisy label, note that each label belongs to one of the k classes
    resp = np.zeros((n,m,k))
    # a variable to store identity of the workers that are assigned to the i-th example
    # note that "repeat" number of workers are randomly chosen from the set of [m] workers and assigned to each example
    workers_this_example = np.zeros((n,repeat),dtype=np.int)
    
    # iterating over each training example
    for i in range(n):
        # randomly selecting "repeat" number of workers for the i-th example
        workers_this_example[i] = np.sort(np.random.choice(m,repeat,replace=False))
        count = 0
        # for each randomly chosen worker generating noisy label according to her confusion matrix and the true label
        for j in workers_this_example[i]:
            # using the row of the confusion matrix corresponding to the true label generating the noisy label
            temp_rand = np.random.multinomial(1,conf[j,class_train[i],:])
            # storing the noisy label in the resp variable 
            resp[i,j,:] = temp_rand
            # storing the noisy label in the dictionary
            workers_train_label['softmax' + str(count) + '_label'][i] = temp_rand
            count = count +1 
            
    # note that in the dictionary each numpy array is of size only (n,k). 
    # The dictionary is passed to the deep learning module
    # however, the resp variable is a numpy array of size (n,m,k).
    # it is used for performing expectation maximization on the noisy labels

    # initializing a dictionary to store one-hot representation of the true labels for the validation set
    workers_val_label = {}
    # the dictionary contains "repeat" number of numpy arrays with keys named "softmax_0_label", where 0 varies
    # each array has the true labels of the examples in the validation set
    workers_val_label['softmax' + str(0) + '_label'] = np.zeros((n1,k))  
    
    # reading the .lst file for the validation set
    content_val_lst = np.genfromtxt(fname[3], delimiter='\t')
    # storing the true labels of the examples in the validation set in the dictionary
    for i in range(n1):
        workers_val_label['softmax' + str(0) + '_label'][i][int(content_val_lst[i,1])] = 1
    
    # returning the noisy responses of the workers stored in the resp numpy array, 
    # the noisy labels stored in the dictionary that is used by the deep learning module
    # the true lables of the examples in the validation set stored in the dictionary
    # identity of the workers that are assigned to th each example in the training set
    return resp, workers_train_label, workers_val_label, workers_this_example

def majority_voting(resp):
    # computes majority voting label
    # ties are broken uniformly at random
    n = resp.shape[0]
    k = resp.shape[2]
    pred_mv = np.zeros((n), dtype = np.int)
    for i in range(n):
        # finding all labels that have got maximum number of votes
        poss_pred = np.where(np.sum(resp[i],0) == np.max(np.sum(resp[i],0)))[0]
        shuffle(poss_pred)
        # choosing a label randomly among all the labels that have got the highest number of votes
        pred_mv[i] = poss_pred[0]   
    pred_mv_vec = np.zeros((n,k))
    # returning one-hot representation of the majority vote label
    pred_mv_vec[np.arange(n), pred_mv] = 1
    return pred_mv_vec

def post_prob_DS(resp_org,e_class,workers_this_example):
    # computes posterior probability distribution of the true label given the noisy labels annotated by the workers
    # and model prediction
    n = resp_org.shape[0]
    m = resp_org.shape[1]
    k = resp_org.shape[2]
    repeat = workers_this_example.shape[1]
    
    temp_class = np.zeros((n,k))
    e_conf = np.zeros((m,k,k))
    temp_conf = np.zeros((m,k,k))
    
    #Estimating confusion matrices of each worker by assuming model prediction "e_class" is the ground truth label
    for i in range(n):
        for j in workers_this_example[i]: #range(m)
            temp_conf[j,:,:] = temp_conf[j,:,:] + np.outer(e_class[i],resp_org[i,j])
    #regularizing confusion matrices to avoid numerical issues
    for j in range(m):  
        for r in range(k):
            if (np.sum(temp_conf[j,r,:]) ==0):
                # assuming worker is spammer for the particular class if there is no estimation for that class for that worker
                temp_conf[j,r,:] = 1/k
            else:
                # assuming there is a non-zero probability of each worker assigning labels for all the classes
                temp_conf[j,r,:][temp_conf[j,r,:]==0] = 1e-10
        e_conf[j,:,:] = np.divide(temp_conf[j,:,:],np.outer(np.sum(temp_conf[j,:,:],axis =1),np.ones(k)))
    # Estimating posterior distribution of the true labels using confusion matrices of the workers and the original
    # noisy labels annotated by the workers
    for i in range(n):
        for j in workers_this_example[i]: 
            if (np.sum(resp_org[i,j]) ==1):
                temp_class[i] = temp_class[i] + np.log(np.dot(e_conf[j,:,:],np.transpose(resp_org[i,j])))
        temp_class[i] = np.exp(temp_class[i])
        temp_class[i] = np.divide(temp_class[i],np.outer(np.sum(temp_class[i]),np.ones(k)))
        e_class[i] = temp_class[i]           
    return e_class

# The following code implements ResNet using MXNET. It is copied from https://github.com/tornadomeet/ResNet/.
def train(gpus,fname,workers_train_label,workers_val_label,numepoch,batch_size,depth = 20,lr=0.5):    
    output_filename = "tr_err.txt"
    model_num = 1
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    hdlr = logging.FileHandler(output_filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 

    kv = mx.kvstore.create('device')
    ### training iterator
    train1 = mx.io.ImageRecordIter(
        path_imgrec         = fname[0],
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax0_label',
        data_shape          = (3, 32, 32), 
        batch_size          = batch_size,
        pad                 = 4, 
        fill_value          = 127,  
        rand_crop           = True,
        max_random_scale    = 1.0,  
        min_random_scale    = 1.0, 
        rand_mirror         = True,
        shuffle             = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)    
           
    ### Validation iterator
    val1 = mx.io.ImageRecordIter(
        path_imgrec         = fname[2],
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax0_label', 
        batch_size          = batch_size,
        data_shape          = (3, 32, 32), 
        rand_crop           = False,
        rand_mirror         = False,
        pad = 0,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)

    n = workers_train_label['softmax0_label'].shape[0]
    k = workers_train_label['softmax0_label'].shape[1]
    n1 = workers_val_label['softmax0_label'].shape[0]      
    train2 = mx.io.NDArrayIter(np.zeros(n), workers_train_label, batch_size, shuffle = False,)
    train_iter = MultiIter([train1,train2])          
    val2 = mx.io.NDArrayIter(np.zeros(n1), workers_val_label, batch_size = batch_size,shuffle = False,)
    val_iter = MultiIter([val1,val2]) 
        
    if((depth-2)%6 == 0 and depth < 164):
        per_unit = [int((depth-2)/6)]
        filter_list = [16, 16, 32, 64]
        bottle_neck = False
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))
    units = per_unit*3
    symbol = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=k,data_type="cifar10", 
                    bottle_neck = False, bn_mom=0.9, workspace=512,
                    memonger=False)
    
    devs = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
    epoch_size = max(int(n / batch_size / kv.num_workers), 1)
    if not os.path.exists("./model" + str(model_num)):
        os.mkdir("./model" + str(model_num))
    model_prefix = "model"+ str(model_num) + "/resnet-{}-{}-{}".format("cifar10", depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)

    def custom_metric(label,softmax):
        return len(np.where(np.argmax(softmax,1)==np.argmax(label,1))[0])/float(label.shape[0])
    #there is only one softmax layer with respect to which error of all the labels are computed
    output_names = []
    output_names = output_names + ['softmax' + str(0) + '_output']   
    eval_metrics = mx.metric.CustomMetric(custom_metric,name = 'accuracy', output_names=output_names, label_names=workers_train_label.keys())    
       
    model = mx.mod.Module(
        context             = devs,
        symbol              = mx.sym.Group(symbol),
        data_names          = ['data'],
        label_names         = workers_train_label.keys(),#['softmax0_label']
        )
    lr_scheduler = multi_factor_scheduler(0, epoch_size, step=[40, 50], factor=0.1)
    optimizer_params = {
        'learning_rate': lr,
        'momentum' : 0.9,
        'wd' : 0.0001,
        'lr_scheduler': lr_scheduler}
       
    model.fit(
        train_iter,
        eval_data          = val_iter,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(batch_size, 50),
        epoch_end_callback = checkpoint,
        optimizer           = 'nag',
        optimizer_params   = optimizer_params,        
        num_epoch           = numepoch, 
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        )
    
    epoch_max_val_acc, train_acc, val_acc = max_val_epoch(output_filename)
    #print "val-acc: " + str(val_acc) 
    
    # Prediction on Training data
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix,epoch_max_val_acc)
    model = mx.mod.Module(
        context             = devs,
        symbol              = sym,
        data_names          = ['data'], 
        label_names         = workers_train_label.keys(),#['softmax0_label']
        )
    model.bind(for_training=False, data_shapes=train_iter.provide_data, 
         label_shapes=train_iter.provide_label,)
    model.set_params(arg_params, aux_params, allow_missing=True)    

    outputs = model.predict(train_iter)
    if type(outputs) is list:
        return outputs[0].asnumpy(), val_acc
    else:
        return outputs.asnumpy(), val_acc

def max_val_epoch(filename):
    import re
    TR_RE = re.compile('.*?]\sTrain-accuracy=([\d\.]+)')
    VA_RE = re.compile('.*?]\sValidation-accuracy=([\d\.]+)')
    EPOCH_RE = re.compile('Epoch\[(\d+)\] V+?')
    log = open(filename, 'r').read()    
    val_acc = [float(x) for x in VA_RE.findall(log)]
    train_acc = [float(x) for x in TR_RE.findall(log)]
    index_max_val_acc = np.argmax([float(x) for x in VA_RE.findall(log)])
    epoch_max_val_acc = [int(x) for x in EPOCH_RE.findall(log)][index_max_val_acc]
    return epoch_max_val_acc+1, train_acc[index_max_val_acc], val_acc[index_max_val_acc]

class MultiIter(mx.io.DataIter):
    def __init__(self, iter_list):
        self.iters = iter_list 
        #self.batch_size = 500
    def next(self):
        batches = [i.next() for i in self.iters] 
        return mx.io.DataBatch(data=[t for t in batches[0].data],
                         label= [t for t in batches[1].label],pad=0)
    def reset(self):
        for i in self.iters:
            i.reset()
    @property
    def provide_data(self):
        return [t for t in self.iters[0].provide_data]
    @property
    def provide_label(self):
        return [t for t in self.iters[1].provide_label]
    
def multi_factor_scheduler(begin_epoch, epoch_size, step=[40, 50], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

    

'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(units, num_stage, filter_list, num_class, data_type, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
         raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    softmax0 = mx.sym.log_softmax(fc1)
    softmax0_output = mx.sym.BlockGrad(data = softmax0,name = 'softmax0')
    loss = [softmax0_output]
    label = mx.sym.Variable(name='softmax0_label')
    ce = -mx.sym.sum(mx.sym.sum(mx.sym.broadcast_mul(softmax0,label),1))
    loss[:] = loss +  [mx.symbol.MakeLoss(ce, normalization='batch')]
    return loss

# calling  function to generate confusion matrices of workers
conf = generate_workers(m,k,gamma,class_wise)  

# calling the main function that takes as input the following:
# name of .rec iterators and .lst files that to operate on,
# worker confusion matrices, 
# number of epochs for running ResNet model, depth of the model,
# number of gpus available on the machine,
# samples: number of samples to be used for training the model,
# repeat: the number of redundant noisy labels to be used for each training example, 
# that are generated using the worker confusion mtrices
# it prints the generalization error of the model on set aside test data
# note that the samples*repeat is approximately same for each pair
# which implies that the total annotation budget is fixed.
for repeat,samples in [[13,4000],[7,7000],[5,10000],[3,17000],[1,50000]]: 
    print "\nnumber of training examples: " + str(samples) + "\t redundancy: " + str(repeat)
    # calling the main function
    main(fname,n,n1,k,conf,samples,repeat,epochs,depth,gpus)


