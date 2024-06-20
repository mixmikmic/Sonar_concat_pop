get_ipython().magic('matplotlib inline')
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn


X = np.arange(-10,10,0.1)
X.shape


## Code taken from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


rolling_window(X[:5], 2)


X = np.arange(-10,10,0.1)
X = np.cos(np.mean(rolling_window(X, 5), -1))
#X = X[:-5+1]
print(X.shape)


plt.plot(X)


# ## Implementation from Pytorch tutorial
# 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size+hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x, h):
        inp = torch.cat((x,h), 1)
        hidden = self.tanh(self.i2h(inp))
        output = self.h2o(inp)
        return hidden, output
    
    
    def get_output(self, X):
        time_steps = X.size(0)
        batch_size = X.size(1)
        hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if torch.cuda.is_available() and X.is_cuda:
            hidden = hidden.cuda()
        outputs = []
        hiddens = []
        for t in range(time_steps):
            hidden, output = self.forward(X[t], hidden)
            outputs.append(output)
            hiddens.append(hidden)
        return torch.cat(hiddens, 1), torch.cat(outputs, 1)
    
## Helper functions

def get_variable_from_np(X):
    return Variable(torch.from_numpy(X)).float()


def get_training_data(X, batch_size=10, look_ahead=1):
    ## Lookahead will always be one as the prediction is for 1 step ahead
    inputs = []
    targets = []
    time_steps = X.shape[0]
    for i in range(0, time_steps-batch_size-look_ahead):
        inp = X[i:i+batch_size, np.newaxis, np.newaxis]
        inputs.append(get_variable_from_np(inp))
        target = X[i+look_ahead:i+batch_size+look_ahead, np.newaxis, np.newaxis]
        targets.append(get_variable_from_np(target))
        #print(inp.shape, target.shape)
    return torch.cat(inputs, 1), torch.cat(targets, 1)


print(torch.cat([get_variable_from_np(X[i:i+5, np.newaxis, np.newaxis]) for i in range(X.shape[0]-5-1)], 1).size())
print(torch.cat([get_variable_from_np(X[i:i+5, np.newaxis, np.newaxis]) for i in range(1, X.shape[0]-5)], 1).size())


inputs, targets = get_training_data(X, batch_size=5)
inputs.size(), targets.size()


inputs, targets = get_training_data(X, batch_size=3)
inputs.size(), targets.size()


rnn = RNN(30, 20, 1)


criterion = nn.MSELoss()
batch_size = 10
TIMESTEPS = 5

batch = Variable(torch.randn(batch_size, 30))
hidden = Variable(torch.randn(batch_size, 20))
target = Variable(torch.randn(batch_size, 1))

loss = 0
for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    loss += criterion(output, target)
    
loss.backward()    


rnn = RNN(30, 20, 1).cuda()


criterion = nn.MSELoss()
batch_size = 10
TIMESTEPS = 5

batch = Variable(torch.randn(batch_size, 30)).cuda()
hidden = Variable(torch.randn(batch_size, 20)).cuda()
target = Variable(torch.randn(batch_size, 1)).cuda()

loss = 0
for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    loss += criterion(output, target)
    
loss.backward()


# ### Run on own data
# 

rnn = RNN(1,3,1).cuda()


criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(rnn.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adadelta(rnn.parameters())


X[:, np.newaxis, np.newaxis].shape


batch = get_variable_from_np(X[:, np.newaxis, np.newaxis]).cuda()


batch.is_cuda


batch = get_variable_from_np(X[:, np.newaxis, np.newaxis]).cuda()
hiddens, outputs = rnn.get_output(batch)


outputs.size()


target = get_variable_from_np(X[np.newaxis, :])
target.size()


torch.cat([get_variable_from_np(X[i:i+10, np.newaxis, np.newaxis]) for i in range(5)], 1).size()


torch.cat([get_variable_from_np(X[i:i+10, np.newaxis, np.newaxis]) for i in range(5)], 1).size()


rnn = RNN(1,3,1)
if torch.cuda.is_available():
    rnn = rnn.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adadelta(rnn.parameters())


batch_size = 1
TIMESTEPS = X.shape[0]
epochs = 10000
print_every = 1000
inputs, targets = get_training_data(X, batch_size=100)
if torch.cuda.is_available() and rnn.is_cuda:
    inputs = inputs.cuda()
    targets = targets.cuda()
print(inputs.size(), targets.size())
losses = []

for i in range(epochs):
    optimizer.zero_grad()
    hiddens, outputs = rnn.get_output(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.data[0])
    if (i+1) % print_every == 0:
        print("Loss at epoch [%s]: %.3f" % (i, loss.data[0]))


inputs, targets = get_training_data(X, batch_size=5)


inputs.size()


inputs = inputs.cuda()


torch.cuda.is_available()


outputs[:, 0].size()


X.shape, outputs[:, 0].data.numpy().flatten().shape


plt.plot(X, '-b', label='data')
plt.plot(outputs[:, 0].data.numpy().flatten(), '-r', label='rnn') # add some offset to view each curve
plt.legend()


# # Try to use the LSTM cells and LSTM layer (Work in progress)
# 
# 
# 
# ## Use LSTM cell
# 

input_size, hidden_size, output_size = 1,3,1
lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
output_layer = nn.Linear(hidden_size, output_size)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD([
    {"params": lstm.parameters()},
    {"params": output_layer.parameters()}
], lr=0.001, momentum=0.9)


# ## Use LSTM Layer
# 

lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
output_layer = nn.Linear(hidden_size, output_size)


batch = get_variable_from_np(X[:, np.newaxis, np.newaxis])
batch.size()


hidden = Variable(torch.zeros(1, batch.size(1), hidden_size))
cell_state = Variable(torch.zeros(1, batch.size(1), hidden_size))


hx = (hidden, cell_state)
output, (h_n, c_n) = lstm.forward(batch, hx)


output.size()


out = output_layer.forward(output[0])
out


criterion = nn.MSELoss()
optimizer = torch.optim.SGD([
    {"params": lstm.parameters()},
    {"params": output_layer.parameters()}
], lr=0.001, momentum=0.9)


batch_size = 1
epochs = 10

inputs, targets = get_training_data(X, max_steps=1)

for i in range(epochs):
    loss = 0
    optimizer.zero_grad()
    hidden = Variable(torch.zeros(1, inputs.size(1), hidden_size))
    cell_state = Variable(torch.zeros(1, inputs.size(1), hidden_size))
    hx = (hidden, cell_state)
    output, (h_n, c_n) = lstm.forward(inputs, hx)
    losses = []
    for j in range(output.size()[0]):
        out = output_layer.forward(output[j])
        losses.append((out - targets[j])**2)
        #loss += criterion(out, target[i])
    loss = torch.mean(torch.cat(losses, 1))
    loss.backward()
    optimizer.step()
    print("Loss at epoch [%s]: %.3f" % (i, loss.squeeze().data[0]))


output.size()


out.size()


targets.size()


y_pred = []
for i in range(output.size()[1]):
        out = output_layer.forward(output[i])
        y_pred.append(out.squeeze().data[0])
y_pred = np.array(y_pred)


plt.plot(X, '-b', alpha=0.5, label='data')
plt.plot(y_pred + 0.1, '-r', label='rnn')
plt.legend()


y_pred





import numpy as np
np.random.seed(2017)

import torch
torch.manual_seed(2017)

from scipy.misc import logsumexp # Use it for reference checking implementation


seq_length, num_states=4, 2
emissions = np.random.randint(20, size=(seq_length,num_states))*1.
transitions = np.random.randint(10, size=(num_states, num_states))*1.
print("Emissions:", emissions, sep="\n")
print("Transitions:", transitions, sep="\n")


def viterbi_decoding(emissions, transitions):
    # Use help from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/crf/python/ops/crf.py
    scores = np.zeros_like(emissions)
    back_pointers = np.zeros_like(emissions, dtype="int")
    scores = emissions[0]
    # Generate most likely scores and paths for each step in sequence
    for i in range(1, emissions.shape[0]):
        score_with_transition = np.expand_dims(scores, 1) + transitions
        scores = emissions[i] + score_with_transition.max(axis=0)
        back_pointers[i] = np.argmax(score_with_transition, 0)
    # Generate the most likely path
    viterbi = [np.argmax(scores)]
    for bp in reversed(back_pointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(scores)
    return viterbi_score, viterbi


viterbi_decoding(emissions, transitions)


def viterbi_decoding_torch(emissions, transitions):
    scores = torch.zeros(emissions.size(1))
    back_pointers = torch.zeros(emissions.size()).int()
    scores = scores + emissions[0]
    # Generate most likely scores and paths for each step in sequence
    for i in range(1, emissions.size(0)):
        scores_with_transitions = scores.unsqueeze(1).expand_as(transitions) + transitions
        max_scores, back_pointers[i] = torch.max(scores_with_transitions, 0)
        scores = emissions[i] + max_scores
    # Generate the most likely path
    viterbi = [scores.numpy().argmax()]
    back_pointers = back_pointers.numpy()
    for bp in reversed(back_pointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = scores.numpy().max()
    return viterbi_score, viterbi
    


viterbi_decoding_torch(torch.Tensor(emissions), torch.Tensor(transitions))


viterbi_decoding(emissions, transitions)


def log_sum_exp(vecs, axis=None, keepdims=False):
    ## Use help from: https://github.com/scipy/scipy/blob/v0.18.1/scipy/misc/common.py#L20-L140
    max_val = vecs.max(axis=axis, keepdims=True)
    vecs = vecs - max_val
    if not keepdims:
        max_val = max_val.squeeze(axis=axis)
    out_val = np.log(np.exp(vecs).sum(axis=axis, keepdims=keepdims))
    return max_val + out_val


def score_sequence(emissions, transitions, tags):
    # Use help from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/crf/python/ops/crf.py
    score = emissions[0][tags[0]]
    for i, emission in enumerate(emissions[1:]):
        score = score + transitions[tags[i], tags[i+1]] + emission[tags[i+1]]
    return score


score_sequence(emissions, transitions, [1,1,0,0])


correct_seq = [0, 0, 1, 1]
[transitions[correct_seq[i],correct_seq[i+1]] for i in range(len(correct_seq) -1)]


sum([transitions[correct_seq[i], correct_seq[i+1]] for i in range(len(correct_seq) -1)])


viterbi_decoding(emissions, transitions)


score_sequence(emissions, transitions, [0, 0, 1, 1])


def score_sequence_torch(emissions, transitions, tags):
    score = emissions[0][tags[0]]
    for i, emission in enumerate(emissions[1:]):
        score = score + transitions[tags[i], tags[i+1]] + emission[tags[i+1]]
    return score


score_sequence_torch(torch.Tensor(emissions), torch.Tensor(transitions), [0, 0, 1, 1])


def get_all_tags(seq_length, num_labels):
    if seq_length == 0:
        yield []
        return
    for sequence in get_all_tags(seq_length-1, num_labels):
        #print(sequence, seq_length)
        for label in range(num_labels):
            yield [label] + sequence        
list(get_all_tags(4,2))


def get_all_tags_dp(seq_length, num_labels):
    prior_tags = [[]]
    for i in range(1, seq_length+1):
        new_tags = []
        for label in range(num_labels):
            for tags in prior_tags:
                new_tags.append([label] + tags)
        prior_tags = new_tags
    return new_tags
list(get_all_tags_dp(2,2))


def brute_force_score(emissions, transitions):
    # This is for ensuring the correctness of the dynamic programming method.
    # DO NOT run with very high values of number of labels or sequence lengths
    for tags in get_all_tags_dp(*emissions.shape):
        yield score_sequence(emissions, transitions, tags)

        
brute_force_sequence_scores = list(brute_force_score(emissions, transitions))
print(brute_force_sequence_scores)


max(brute_force_sequence_scores) # Best score calcuated using brute force


log_sum_exp(np.array(brute_force_sequence_scores)) # Partition function


def forward_algorithm_naive(emissions, transitions):
    scores = emissions[0]
    # Get the log sum exp score
    for i in range(1,emissions.shape[0]):
        print(scores)
        alphas_t = np.zeros_like(scores) # Forward vars at timestep t
        for j in range(emissions.shape[1]):
            emit_score = emissions[i,j]
            trans_score = transitions.T[j]
            next_tag_var = scores + trans_score
            alphas_t[j] = log_sum_exp(next_tag_var) + emit_score
        scores = alphas_t
    return log_sum_exp(scores)


forward_algorithm_naive(emissions, transitions)


def forward_algorithm_vec_check(emissions, transitions):
    # This is for checking the correctedness of log_sum_exp function compared to scipy
    scores = emissions[0]
    scores_naive = emissions[0]
    # Get the log sum exp score
    for i in range(1, emissions.shape[0]):
        print(scores, scores_naive)
        scores = emissions[i] + logsumexp(
            scores_naive + transitions.T,
            axis=1)
        scores_naive = emissions[i] + np.array([log_sum_exp(
            scores_naive + transitions.T[j]) for j in range(emissions.shape[1])])
    print(scores, scores_naive)
    return logsumexp(scores), log_sum_exp(scores_naive)


forward_algorithm_vec_check(emissions, transitions)


def forward_algorithm(emissions, transitions):
    scores = emissions[0]
    # Get the log sum exp score
    for i in range(1, emissions.shape[0]):
        scores = emissions[i] + log_sum_exp(
            scores + transitions.T,
            axis=1)
    return log_sum_exp(scores)


forward_algorithm(emissions, transitions)


tt = torch.Tensor(emissions)
tt_max, _ = tt.max(1)


tt_max.expand_as(tt)


tt.sum(0)


tt.squeeze(0)


tt.transpose(-1,-2)


tt.ndimension()


def log_sum_exp_torch(vecs, axis=None):
    ## Use help from: http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    if axis < 0:
        axis = vecs.ndimension()+axis
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.expand_as(vecs)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    #print(max_val, out_val)
    return max_val + out_val


def forward_algorithm_torch(emissions, transitions):
    scores = emissions[0]
    # Get the log sum exp score
    transitions = transitions.transpose(-1,-2)
    for i in range(1, emissions.size(0)):
        scores = emissions[i] + log_sum_exp_torch(
            scores.expand_as(transitions) + transitions,
            axis=1)
    return log_sum_exp_torch(scores, axis=-1)


forward_algorithm_torch(torch.Tensor(emissions), torch.Tensor(transitions))


# The core idea is to find the sequence of states $y = \{y_0, y_1, ..., y_N\}$ which have the highest probability given the input $X = \{X_0, X_1, ..., X_N\}$ as follows:
# 
# $$
# \begin{equation}
# p(y\mid X) = \prod_{i=0}^{N}{p(y_i\mid X_i)p(y_i \mid y_{i-1})}\\log{p(y\mid X)} = \sum_{i=0}^{N}{\log{p(y_i\mid X_i)} + \log{p(y_i \mid y_{i-1})}}\\end{equation}
# $$
# 
# Now $\log{p(y_i\mid X_i)}$ and $\log{p(y_i \mid y_{i-1})}$ can be parameterized as follows:
# 
# $$
# \begin{equation}
# \log{p(y_i\mid X_i)} = \sum_{l=0}^{L}{\sum_{k=0}^{K}{w_{k}^{l}*\phi_{k}^{l}(X_i, y_i)}}\\log{p(y_i\mid y_{y-1})} = \sum_{l=0}^{L}{\sum_{l'=0}^{L}{w_{l'}^{l}*\psi_{l'}^{l}(y_i, y_{i-1})}}\\implies \log{p(y\mid X)} = \sum_{i=0}^{N}{(\sum_{l=0}^{L}{\sum_{k=0}^{K}{w_{k}^{l}*\phi_{k}^{l}(X_i, y_i)}}
# + \sum_{l=0}^{L}{\sum_{l'=0}^{L}{w_{l'}^{l}*\psi_{l'}^{l}(y_i, y_{i-1})}})}\\implies \log{p(y\mid X)} = \sum_{i=0}^{N}{(\Phi(X_i)W_{emission} + \log{p(y_{i-1} \mid X_{i-1})}W_{transition})}
# \end{equation}
# $$
# 
# Where, 
# 
# * $N$ is the sequence length
# * $K$ is number of feature functions,
# * $L$ is number of states
# * $W_{emission}$ is $K*L$ matrix
# * $W_{transition}$ is $L*L$ matrix
# * $\Phi(X_i)$ is a feature vector of shape $1*K$
# * $(\Phi(X_i)W_{emission} + \log{p(y_{i-1} \mid X_{i-1})}W_{transition})$ gives the score for each label
# 
# 




