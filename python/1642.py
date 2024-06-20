get_ipython().magic('pylab inline')
figsize(10,5)
matplotlib.rcParams["image.interpolation"] = "none"
matplotlib.rcParams["image.cmap"] = "afmhot"


import clstm
import h5py


# # LSTM + CTC Training on MNIST
# 

# Let's start by getting the MNIST dataset. This version of MNIST in HDF5 represents images in a sequence format suitable for training with `clstm` command line models.
# 

get_ipython().system('test -f mnist_seq.h5 || curl http://www.tmbdev.net/ocrdata-hdf5/mnist_seq.h5 > mnist_seq.h5 || rm -f mnist_seq.h5')


# In HDF5 data files for CLSTM, row `t` represents the input vector at time step `t`. For MNIST, we scan through the original image left-to-right over time.
# 
# Image storage in HDF5 would have to be a rank 3 doubly ragged array, but HDF5 supports only rank 2 arrays. We therefore store image dimensions in a separate array.
# 

h5 = h5py.File("mnist_seq.h5","r")
imshow(h5["images"][0].reshape(*h5["images_dims"][0]))
print h5["images"].shape


# Let's use a bidirectional LSTM and a fiarly high learning rate.
# 

net = clstm.make_net_init("bidi","ninput=28:nhidden=10:noutput=11")
net.setLearningRate(1e-2,0.9)
print clstm.network_info(net)


# The class labels in the dataset are such that digit `0` has been assigned class `10`, since class 0 is reserved for epsilon states in CTC alignment.
# 

print [chr(c) for c in h5["codec"]]


index = 0
xs = array(h5["images"][index].reshape(28,28,1),'f')
cls = h5["transcripts"][index][0]
print cls
imshow(xs.reshape(28,28).T,cmap=cm.gray)


# Forward propagation is quite simple: we take the input data and put it into the input sequence of the network, call the `forward` method, and take the result out of the output sequence.
# 
# Note that all sequences (including `xs`) in clstm are of rank 3, with indexes giving the time step, the feature dimension, and the batch index, in order.
# 
# The output from the network is a vector of posterior probabilities at each time step.
# 

net.inputs.aset(xs)
net.forward()
pred = net.outputs.array()
imshow(pred.reshape(28,11).T, interpolation='none')


# We now construct a "target" array and perform CTC alignment with the output.
# 

target = zeros((3,11),'f')
target[0,0] = 1
target[2,0] = 1
target[1,cls] = 1
seq = clstm.Sequence()
seq.aset(target.reshape(3,11,1))
aligned = clstm.Sequence()
clstm.seq_ctc_align(aligned,net.outputs,seq)
aligned = aligned.array()
imshow(aligned.reshape(28,11).T, interpolation='none')


# Next, we take the aligned output, subtract the actual output, set that as the output deltas, and the propagate the error backwards and update.
# 

deltas = aligned - net.outputs.array()
net.d_outputs.aset(deltas)
net.backward()
net.update()


# If we repeat these steps over and over again, we eventually end up with a trained network.
# 

for i in range(60000):
    index = int(rand()*60000)
    xs = array(h5["images"][index].reshape(28,28,1),'f')
    cls = h5["transcripts"][index][0]
    net.inputs.aset(xs)
    net.forward()
    pred = net.outputs.array()
    target = zeros((3,11),'f')
    target[0,0] = 1
    target[2,0] = 1
    target[1,cls] = 1
    seq = clstm.Sequence()
    seq.aset(target.reshape(3,11,1))
    aligned = clstm.Sequence()
    clstm.seq_ctc_align(aligned,net.outputs,seq)
    aligned = aligned.array()
    deltas = aligned - net.outputs.array()
    net.d_outputs.aset(deltas)
    net.backward()
    net.update()


figsize(5,10)
subplot(211,aspect=1)
imshow(xs.reshape(28,28).T)
subplot(212,aspect=1)
imshow(pred.reshape(28,11).T, interpolation='none', vmin=0, vmax=1)


get_ipython().magic('pylab inline')
figsize(10,5)


import clstm


# Network creation and initialization is very similar to C++:
# 
#  - networks are created using the `make_net(name)` factory function
#  - the `net.set(key,value)` method is used to set up parameters
#  - the `.setLearningRate(lr,mom)` method is used to set learning rate and momentum
#  - `.initialize()` is called to create the network
# 
# As in C++, the combination of `make_net` and `set` does not allow arbitrary network architectures to be constructed. For anything complicated, you 
# 

net = clstm.make_net_init("lstm1","ninput=1:nhidden=4:noutput=2")
print net


net.setLearningRate(1e-4,0.9)
print clstm.network_info_as_string(net)


# You can navigate the network structure as you would in C++. You can use similar methods to create more complex network architectures than possible with `make_net`.
# 

print net.sub.size()
print net.sub[0]
print net.sub[0].kind


# This cell generally illustrates how to invoke the CLSTM library from Python:
# 
#  - `net.inputs`, `net.outputs`, `net.d_inputs`, and `net.d_outputs` are `Sequence` types
#  - `Sequence` objects can be converted to rank 3 arrays using the .array() method
#  - The values in a `Sequence` can be set with the `.aset(array)` method
# 

N = 20
xs = array(randn(N,1,1)<0.2, 'f')
net.inputs.aset(xs)
net.forward()


# Here is a training loop that generates a delayed-by-one from a random input sequence and trains the network to learn this task.
# 

N = 20
test = array(rand(N)<0.3, 'f')
plot(test, '--', c="black")
ntrain = 30000
for i in range(ntrain):
    xs = array(rand(N)<0.3, 'f')
    ys = roll(xs, 1)
    ys[0] = 0
    ys = array([1-ys, ys],'f').T.copy()
    net.inputs.aset(xs.reshape(N,1,1))
    net.forward()
    net.outputs.dset(ys.reshape(N,2,1)-net.outputs.array())
    net.backward()
    clstm.sgd_update(net)
    if i%1000==0:
        net.inputs.aset(test.reshape(N,1,1))
        net.forward()
        plot(net.outputs.array()[:,1,0],c=cm.jet(i*1.0/ntrain))





get_ipython().magic('pylab inline')


# # Test Cases for LSTM Training
# 

# This worksheet contains code that generates a variety of LSTM test cases. The output files are suitable for use with `clstmseq`.
# 

from pylab import *
from scipy.ndimage import filters

default_ninput = 2
default_n = 29


# Here is a simple utility class to write out sequence data to an HDF5 file quickly.
# 

import h5py
import numpy as np

class H5SeqData:
    def __init__(self,fname,N=None):
        self.fname = fname
        h5 = h5py.File("rnntest-"+fname+".h5","w")
        self.h5 = h5
        dt = h5py.special_dtype(vlen=np.dtype('float32'))
        it = np.dtype('int32')
        self.inputs = h5.create_dataset("inputs",(1,), maxshape=(None,),compression="gzip",dtype=dt)
        self.inputs_dims = h5.create_dataset("inputs_dims",(1,2), maxshape=(None,2), dtype=it)
        self.outputs = h5.create_dataset("outputs",(1,),maxshape=(None,),compression="gzip",dtype=dt)
        self.outputs_dims = h5.create_dataset("outputs_dims",(1,2), maxshape=(None,2), dtype=it)
        self.fill = 0
        if N is not None: self.resize(N)
    def close(self):
        self.h5.close()
        self.h5 = None
    def __enter__(self):
        print "writing",self.fname
        return self
    def __exit__(self, type, value, traceback):
        self.close()
        print "done writing",self.fname
    def resize(self,n):
        self.inputs.resize((n,))
        self.inputs_dims.resize((n,2))
        self.outputs.resize((n,))
        self.outputs_dims.resize((n,2))
    def add(self,inputs,outputs):
        self.inputs[self.fill] = inputs.ravel()
        self.inputs_dims[self.fill] = array(inputs.shape,'i')
        self.outputs[self.fill] = outputs.ravel()
        self.outputs_dims[self.fill] = array(outputs.shape,'i')
        self.fill += 1

N = 50000


def genfile(fname,f):
    with H5SeqData(fname,N) as db:
        for i in range(N):
            xs,ys = f()
            db.add(xs,ys)


def plotseq(fname,index=17):
    h5 = h5py.File(fname,"r")
    try:
        inputs = h5["inputs"][index].reshape(*h5["inputs_dims"][index])
        outputs = h5["outputs"][index].reshape(*h5["outputs_dims"][index])
        plot(inputs[:,0],'r-',linewidth=5,alpha=0.5)
        if inputs.shape[1]>1:
            plot(inputs[:,1:],'r-',linewidth=1,alpha=0.3)
        plot(outputs,'b--')
    finally:
        h5.close()


def generate_threshold(n=default_n,ninput=default_ninput,threshold=0.5,example=0):
    "No temporal dependencies, just threshold of the sum of the inputs."
    x = rand(n,ninput)
    y = 1.0*(sum(x,axis=1)>threshold*ninput).reshape(n,1)
    return x,y

genfile("threshold", generate_threshold)


plotseq("rnntest-threshold.h5")


def generate_mod(n=default_n,ninput=default_ninput,m=3,example=0):
    "Generate a regular beat every m steps. The input is random."
    x = rand(n,ninput)
    y = 1.0*(arange(n,dtype='i')%m==0).reshape(n,1)
    return x,y

genfile("mod3", generate_mod)


plotseq("rnntest-mod3.h5")


def generate_dmod(n=default_n,ninput=default_ninput,m=3,example=0):
    """Generate a regular beat every m steps, the input is random
    except for the first dimension, which contains a downbeat
    at the very beginning."""
    x = rand(n,ninput)
    y = 1.0*(arange(n,dtype='i')%m==0).reshape(n,1)
    x[:,0] = 0
    x[0,0] = 1
    return x,y

genfile("dmod3", generate_dmod)
genfile("dmod4", lambda:generate_dmod(m=4))
genfile("dmod5", lambda:generate_dmod(m=5))
genfile("dmod6", lambda:generate_dmod(m=6))


plotseq("rnntest-dmod3.h5")


def generate_imod(n=default_n,ninput=default_ninput,m=3,p=0.2,example=0):
    """Generate an output for every m input pulses."""
    if example:
        x = array(arange(n)%4==1,'i')
    else:
        x = array(rand(n)<p,'i')
    y = (add.accumulate(x)%m==1)*x*1.0
    x = array(vstack([x]*ninput).T,'f')
    y = y.reshape(len(y),1)
    return x,y

genfile("imod3", generate_imod)
genfile("imod4", lambda:generate_imod(m=4))


plotseq("rnntest-imod3.h5")


def generate_smod(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Generate an output for every m input pulses. The input
    is band limited, so it's a little easier than generate_imod."""
    x = rand(n)
    x = filters.gaussian_filter(x,r)
    x = (x>roll(x,-1))*(x>roll(x,1))
    y = (add.accumulate(x)%m==1)*x*1.0
    x = array(vstack([x]*ninput).T,'f')
    y = y.reshape(len(y),1)
    return x,y

genfile("smod3", generate_smod)
genfile("smod4", lambda:generate_smod(m=4))
genfile("smod5", lambda:generate_smod(m=5))


plotseq("rnntest-smod3.h5")


def generate_anbn(ninput=default_ninput,n=default_n,k=default_n//3,example=0):
    """A simple detector for a^nb^n. Note that this does not
    train the network to distinguish this langugage from other languages."""
    inputs = zeros(n)
    outputs = zeros(n)
    if example:
        l = n//3
    else:
        l = 1+int((k-1)*rand())
    inputs[:l] = 1
    outputs[2*l] = 1
    outputs = outputs.reshape(len(outputs),1)
    return vstack([inputs]*ninput).T,outputs

genfile("anbn", generate_anbn)


plotseq("rnntest-anbn.h5")


def generate_timing(ninput=default_ninput,n=default_n,t=5,example=0):
    """A simple timing related task: output a spike if no spike occurred within
    t time steps before."""
    x = 0
    inputs = []
    while x<n:
        inputs.append(x)
        x += max(1,0.5*t*randn()+t)
    inputs = [-999990]+inputs
    outputs = []
    for i in range(1,len(inputs)):
        if inputs[i]-inputs[i-1]>t:
            outputs.append(inputs[i])
    inputs = inputs[1:]
    xs = zeros((n,ninput))
    xs[inputs,:] = 1.0
    ys = zeros((n,1))
    ys[outputs,:] = 1.0
    return xs,ys

genfile("timing", generate_timing)


def generate_revtiming(ninput=default_ninput,n=default_n,t=5,example=0):
    """A simple timing related task: output a spike if no spike occurs within
    t time steps after. This cannot be learned using a causal model (it requires
    a reverse model)."""
    x = 0
    inputs = []
    while x<n:
        inputs.append(x)
        x += max(1,0.5*t*randn()+t)
    inputs = inputs+[999999]
    outputs = []
    for i in range(len(inputs)-1):
        if inputs[i+1]-inputs[i]>t:
            outputs.append(inputs[i])
    inputs = inputs[:-1]
    xs = zeros((n,ninput))
    xs[inputs,:] = 1.0
    ys = zeros((n,1))
    ys[outputs,:] = 1.0
    return xs,ys

genfile("revtiming", generate_revtiming)


def generate_biditiming(ninput=default_ninput,n=default_n,t=5,example=0):
    x = 0
    inputs = []
    while x<n:
        inputs.append(x)
        x += max(1,0.5*t*randn()+t)
    inputs = [-999999]+inputs+[999999]
    outputs = []
    for i in range(1,len(inputs)-1):
        if inputs[i+1]-inputs[i]>=t and inputs[i]-inputs[i-1]>=t:
            outputs.append(inputs[i])
    inputs = inputs[1:-1]
    xs = zeros((n,ninput))
    xs[inputs,:] = 1.0
    ys = zeros((n,1))
    ys[outputs,:] = 1.0
    return xs,ys

genfile("biditiming", generate_biditiming)


def detect_12(x):
    n = len(x)
    y = zeros(n)
    state = 0
    for i in range(n):
        s = tuple(1*(x[i]>0.5))
        if s==(0,0): pass
        elif s==(1,0): state = 1
        elif s==(0,1) and state==1:
            y[i] = 1
            state = 0
        else: state = 0
    return y


def generate_detect(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Generates a random sequence of bits and outputs a "1" whenever there is
    a sequence of inputs 01-00*-10"""
    x = rand(n,2)
    x = filters.gaussian_filter(x,(r,0))
    x = 1.0*(x>roll(x,-1,0))*(x>roll(x,1,0))
    y = detect_12(x)
    return x,y.reshape(len(y),1)

genfile("detect", generate_detect)


def generate_revdetect(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Reverse of generate_detect."""
    xs,ys = generate_detect(n=n,ninput=ninput,m=m,r=r,example=example)
    return array(xs)[::-1],array(ys)[::-1]

genfile("revdetect", generate_revdetect)


def generate_bididetect(n=default_n,ninput=default_ninput,m=3,r=0.5,example=0):
    """Generate a particular pattern whenever there is some input trigger."""
    xs,ys = generate_detect(n=n,ninput=ninput,m=m,r=r,example=example)
    rys = detect_12(xs[::-1])[::-1].reshape(len(ys),1)
    return array(xs),array(ys*rys)

genfile("bididetect", generate_bididetect)


def generate_predict_and_sync():
    """Similar to smod, but the correct output is provided one step after
    the required prediction for resynchronization."""
    pass


def generate_distracted_recall():
    """Distracted sequence recall example."""
    pass


def generate_morse():
    """Morse code encoding/decoding."""
    pass


def genseq_timing1(n=30,threshold=0.2,m=4,example=0):
    """Returns an output for every input within m time steps.
    A 1 -> N -> 1 problem."""
    x = (rand(n)<threshold)
    l = find(x)
    y = zeros(len(x))
    for i in range(1,len(l)):
        if l[i]-l[i-1]<m: y[l[i]] = 1
    return (1.0*x).reshape(n,1),y.reshape(n,1)


def genseq_threshold1(n=30,d=3,threshold=0.5,c=0,scale=1.0):
    """Threshold on the first component only."""
    x = randn(n,d)
    y = (1.0*(x[:,c]>threshold)).reshape(len(x),1)
    x[:,c] *= scale
    return x,y


def genseq_delay(n=30,threshold=0.2,d=1):
    """Returns an output for every input within m time steps.
    A 1 -> N -> 1 problem."""
    x = array(rand(n)<threshold,'f')
    y = roll(x,d)
    if d>0: y[:d] = 0
    elif d<0: y[d:] = 0
    return x.reshape(n,1),y.reshape(n,1)

genfile("delay1", genseq_delay)
genfile("delay2", lambda:genseq_delay(d=2))
genfile("delay3", lambda:genseq_delay(d=3))
genfile("rdelay1", lambda:genseq_delay(d=-1))
genfile("rdelay2", lambda:genseq_delay(d=-2))
genfile("rdelay3", lambda:genseq_delay(d=-3))


plotseq("rnntest-delay2.h5")


# # Test Run with `clstmseq`
# 

# Here is a simple example of sequence training with `clstmseq`. It takes one of the HDF5 files we generated above as an example. By default, it uses every tenth training sample as part of a test set. The `TESTERR` it reports is MSE error and binary error rate (assuming a threshold of 0.5).
# 

get_ipython().system('lrate=1e-3 report_every=5000 ntrain=20000 test_every=10000 ../clstmseq rnntest-delay1.h5')





get_ipython().magic('pylab inline')
from pylab import *
import codecs,string,os,sys,os.path,glob,re


# The CLSTM command line tools take their training data in HDF5 files (you will evenutally also be able to train directly from images saved on disk, as in ocropy, but that's not quite implemented yet). This illustrates how to store images into an HDF5 file and then how to run the `clstmctc` training tool on the data.
# 

# # The UW3-500 Dataset
# 

# We illustrate loading data with the `uw3-500.tgz` dataset, available from `tmbdev.net`.
# 

get_ipython().system('test -f uw3-500.tgz || wget -nd http://www.tmbdev.net/ocrdata/uw3-500.tgz')


# Let's untar the file unless it has already been untarred.
# 

get_ipython().system('test -d book || tar -zxvf uw3-500.tgz')


# The UW3-500 dataset is a collection of text line images and corresponding ground truth transcription. It's organized as a directory tree of the form `book/<page_no>/<line_id>.bin.png` etc.
# 

get_ipython().system('ls book/0005/010001.*')


# Let's now run `clstmctc` training. We report every 100 training steps. Since we didn't dewarp or size-normalize the lines, we need to use a `dewarp=center` argument to training.
# 

get_ipython().system('dewarp=center report_every=500 save_name=test save_every=10000 ntrain=11000 ../clstmctc uw3-500.h5')


get_ipython().system('ls book/*/*.bin.png | sort -r > uw3.files')
get_ipython().system('sed 100q uw3.files > uw3-test.files')
get_ipython().system('sed 1,100d uw3.files > uw3-train.files')
get_ipython().system('wc -l uw3*.files')


get_ipython().system('params=1 save_name=uw3small save_every=1000 report_every=100 maxtrain=50000 test_every=1000 ../clstmocrtrain uw3-train.files uw3-test.files')





