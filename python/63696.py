# get matplotlib configuration
get_ipython().magic('run plot_conf.py')


import os
import numpy as np


# load network 001 and 002 train losses
nets = ('001', '002')
net_train_loss = dict()
for net in nets:
    with os.popen("awk '/data/{print $18,$21}' ../results/" + net + "/train.log") as pipe:
        net_train_loss[net] = np.loadtxt(pipe)
net_train_loss['001'] *= 100
net_train_loss['002'] *= 100


# load network > 002 train losses
nets = ('006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '018', '019')
for net in nets:
    with os.popen("awk '/data/{print $18,$21,$25}' ../results/" + net + "/train.log") as pipe:
        net_train_loss[net] = np.loadtxt(pipe)


b = net_train_loss['007'].shape[0] // 10
loss_names = ['MSE', 'CE', 'rpl MSE', 'per CE']


def plot_loss(loss, legend, lim, l_names):
    z = tuple(loss_names.index(n) for n in l_names)
    for l in loss:
        plt.plot(l[:,z])
    plt.legend([l1 + ' ' + l2 for l1 in legend for l2 in l_names])
    if lim[0]:
        if isinstance(lim[0], tuple): plt.xlim(lim[0])
        else: plt.xlim(xmax=lim[0])
    if lim[1]:
        if isinstance(lim[1], tuple): plt.ylim(lim[1])
        else: plt.ylim(ymax=lim[1])
            
def plot_batch_loss(loss, legend, lim, l_names):
    plot_loss(loss, legend, lim, l_names)
    
    # Add vertical dotted lines at the end of each epoch
    ylim = plt.ylim()
    upper_range = loss[0].shape[0] // b
    plt.vlines(
        [b*i for i in range(1, upper_range)],
        ymin=ylim[0], ymax=ylim[1], colors='y', linestyles=':', linewidth=0.75
    )
    plt.ylim(ylim)


def plot_batch_mse(losses, legend, lim=(None, None)):
    plot_batch_loss(losses, legend, lim, ('MSE', 'rpl MSE'))
    plt.xlabel('batch idx / 10')
    plt.title('MSE loss')
    plt.ylabel('mMSE')
    
def plot_epoch_mse(losses, legend, lim=(None, None)):
    plot_loss(losses, legend, lim, ('MSE', 'rpl MSE'))
    plt.xlabel('epoch idx')
    plt.title('MSE loss')
    plt.ylabel('mMSE')
    
    # Do start from Epoch #1
    xlim = plt.xlim()
    loc, _ = plt.xticks()
    plt.xticks(loc, [str(int(x) + 1) for x in loc])
    plt.xlim(xlim)


def plot_batch_ce(losses, legend, lim=(None, None)):
    plot_batch_loss(losses, legend, lim, ('CE','per CE'))
    plt.xlabel('batch idx / 10')
    plt.title('CE loss')
    plt.ylabel('nit')
    
def plot_epoch_ce(losses, legend, lim=(None, None)):
    plot_loss(losses, legend, lim, ('CE', 'per CE'))
    plt.xlabel('epoch idx')
    plt.title('CE loss')
    plt.ylabel('nit')
    
    # Do start from Epoch #1
    xlim = plt.xlim()
    loc, _ = plt.xticks()
    plt.xticks(loc, [str(int(x) + 1) for x in loc])
    plt.xlim(xlim)


plot_batch_loss((net_train_loss['001'], net_train_loss['002'], net_train_loss['006']),
                ('net 001', 'net 002', 'net 006'), (800, 70), ('MSE',))
plt.title('Train MSE loss')
plt.ylabel('mMSE')


plot_batch_mse(
    (net_train_loss['006'], net_train_loss['007']),
    ('net 006 train', 'net 007 train'),
    (None, 60)
)
# plt.xlim((1300, 1480))
plt.xlim((0, b*4))


# Proper state selective resetting

plot_batch_mse(
    (net_train_loss['007'], net_train_loss['008']),
    ('net 007 train', 'net 008 train'),
    (None, 60)
)


# lr /= 10 after 3 epochs

plot_batch_mse(
    (net_train_loss['008'], net_train_loss['009']),
    ('net 008 train', 'net 009 train'),
    (200, 60)
)


# brancing before summing, after G_n

plot_batch_mse(
    (net_train_loss['008'], net_train_loss['010']),
    ('net 008 train', 'net 010 train'),
    (500, 60)
)


# model_02 vs. model_01

plot_batch_mse(
    (net_train_loss['007'], net_train_loss['011']),
    ('model_01 (net 007) train', 'model_02 (net 011) train'),
    (None, 60)
)


np.mean(net_train_loss['011'][1200:,0])


# model_02

plt.subplot(121)
plot_batch_mse(
    (net_train_loss['011'],),
    ('net 011 train',),
    ((0, 150), (0, 10))
)

plt.subplot(122)
plot_batch_mse(
    (net_train_loss['011'],),
    ('net 011 train',),
    ((1310, 1470), (0, 10))
)


# proper double selective state and loss resetting

plt.subplot(121)
plot_batch_mse(
    (net_train_loss['011'], net_train_loss['012']),
    ('net 011 train', 'net 012 train'),
    ((0, 150), (0, 10))
)

plt.subplot(122)
plot_batch_mse(
    (net_train_loss['011'], net_train_loss['012']),
    ('net 011 train', 'net 012 train'),
    ((1310, 1470), (0, 10))
)


# Now that we are actually getting *proper* results, it makes sense to start checking the performance on the validation set.
# 

# load network > 011 validation losses
nets = ('011', '012', '014', '015', '016', '018', '019')
net_val_loss = dict()
net_mean_train_loss = dict()
for net in nets:
#     print(net)
    net_mean_train_loss[net] = np.mean(net_train_loss[net].reshape(-1, b, 3), axis=1)
    with os.popen("awk '/end/{print $11,$14,$18}' ../results/" + net + "/train.log") as pipe:
        net_val_loss[net] = np.loadtxt(pipe)
net_val_loss['012'][net_val_loss['012'] > 1e3] = np.nan
net_val_loss['012'][:,1] = 6.88
net_val_loss['014'][net_val_loss['014'] > 1e3] = np.nan
net_val_loss['015'][:,1] *= 1e3


# proper double selective state and loss resetting

plt.subplot(121)
plot_epoch_mse(
    (net_mean_train_loss['011'], net_mean_train_loss['012']),
    ('net 011 train', 'net 012 train'),
    (None, (0, 8))
)

plt.subplot(122)
plot_epoch_mse(
    (net_val_loss['011'], net_val_loss['012']),
    ('net 011 val', 'net 012 val'),
    (None, (0, 8))
)


# proper double selective state and loss resetting

plt.subplot(1,2,1)
plot_epoch_mse(
    (net_mean_train_loss['012'], net_val_loss['012']),
    ('net 012 train', 'net 012 val'),
    (None, (0, 8))
)


# lr = 0.1, √10x reduction every 3 epochs
# vs.
# lr = 0.1, 10x reduction every 10 epochs

plt.subplot(121)
plot_epoch_mse(
    (net_mean_train_loss['012'], net_mean_train_loss['014']),
    ('net 012 train', 'net 014 train'),
    (None, (0, 8))
)

plt.subplot(122)
plot_epoch_mse(
    (net_val_loss['012'], net_val_loss['014']),
    ('net 012 val', 'net 014 val'),
    (None, (0, 8))
)


# different hyperparameters

plot_epoch_mse(
    (net_mean_train_loss['014'], net_val_loss['014']),
    ('net 014 train', 'net 014 val'),
    (None, (0, 8))
)


plt.subplot(221)
plot_epoch_mse(
    (net_mean_train_loss['012'], net_mean_train_loss['015']),
    ('net 012 train', 'net 015 train'),
    (None, (2, 6))
)

plt.subplot(222)
plot_epoch_mse(
    (net_val_loss['012'], net_val_loss['015']),
    ('net 012 val', 'net 015 val'),
    (None, (2, 6))
)

plt.subplot(223)
plot_epoch_ce(
    (net_mean_train_loss['012'], net_mean_train_loss['015']),
    ('net 012 train', 'net 015 train'),
)

plt.subplot(224)
plot_epoch_ce(
    (net_val_loss['012'], net_val_loss['015']),
    ('net 012 val', 'net 015 val'),
)

plt.tight_layout()


plt.subplot(221)
plot_epoch_mse(
    (net_mean_train_loss['012'], net_mean_train_loss['018'], net_mean_train_loss['019']),
    ('net 012 train', 'net 018 train', 'net 019 train'),
    (None, (2, 6))
)

plt.subplot(222)
plot_epoch_mse(
    (net_val_loss['012'], net_val_loss['018'], net_val_loss['019']),
    ('net 012 val', 'net 018 val', 'net 019 val'),
    (None, (2, 6))
)

plt.subplot(223)
plot_epoch_ce(
    (net_mean_train_loss['015'], net_mean_train_loss['018'], net_mean_train_loss['019']),
    ('net 015 train', 'net 018 train', 'net 019 train'),
)

plt.subplot(224)
plot_epoch_ce(
    (net_val_loss['015'], net_val_loss['018'], net_val_loss['019']),
    ('net 015 val', 'net 018 val', 'net 019 val'),
)

plt.tight_layout()


# load network > 002 train losses
nets = ('020', '021', '022', '023', '024', '025')
for net in nets:
#     print(net)
    with os.popen("awk '/batches/{print $18,$21,$25,$29}' ../results/" + net + "/train.log") as pipe:
        net_train_loss[net] = np.loadtxt(pipe)
    net_mean_train_loss[net] = np.mean(net_train_loss[net].reshape(-1, b, 4), axis=1)
    with os.popen("awk '/end/{print $11,$14,$18,$22}' ../results/" + net + "/train.log") as pipe:
        net_val_loss[net] = np.loadtxt(pipe)


net_train_loss['025'].shape


net_train_loss['027'].shape


# altogether, μ = 1, λ = 1e−3, π = 1e−3 (no CE temporal gradient)

plt.subplot(2,1,1)
plot_batch_ce(
    (net_train_loss['020'],),
    ('net 020 train',),
)
plt.subplot(2,1,2)
plot_batch_mse(
    (net_train_loss['020'],),
    ('net 020 train',),
    (None, (0, 8))
)
plt.tight_layout()


# altogether, μ = 0, λ = 0.01, π = 0 (CE temporal gradient)

plt.subplot(2,1,1)
plot_batch_ce(
    (net_train_loss['022'],),
    ('net 022 train',),
)
plt.subplot(2,1,2)
plot_epoch_ce(
    (net_mean_train_loss['022'], net_val_loss['022']),
    ('net 022 train', 'net 022 val'),
)
plt.tight_layout()


# altogether, μ = 0, λ = 0, π = 0.01 (CE temporal gradient)

plt.subplot(2,1,1)
plot_batch_ce(
    (net_train_loss['023'],),
    ('net 023 train',),
)
plt.subplot(2,1,2)
plot_epoch_ce(
    (net_mean_train_loss['023'], net_val_loss['023']),
    ('net 023 train', 'net 023 val'),
)
plt.tight_layout()


# altogether, μ = 0, λ = 0, π = 0.0316 (CE temporal gradient)

plt.subplot(2,1,1)
plot_batch_ce(
    (net_train_loss['025'],),
    ('net 025 train',),
)
plt.subplot(2,1,2)
plot_epoch_ce(
    (net_mean_train_loss['025'], net_val_loss['025']),
    ('net 025 train', 'net 025 val'),
)
plt.tight_layout()


# load network > 002 train losses
b = 140
nets = ('027', '028', '033')
for net in nets:
#     print(net)
    with os.popen("awk '/batches/{print $18,$21,$25,$29}' ../results/" + net + "/train.log") as pipe:
        net_train_loss[net] = np.loadtxt(pipe)
    net_mean_train_loss[net] = np.mean(net_train_loss[net].reshape(-1, b, 4), axis=1)
    with os.popen("awk '/end/{print $11,$14,$18,$22}' ../results/" + net + "/train.log") as pipe:
        net_val_loss[net] = np.loadtxt(pipe)


plt.subplot(2,2,1)
plot_batch_ce(
    (net_train_loss['028'],),
    ('net 028 train',),
    (None, (6.5, 7.2))
)
plt.subplot(2,2,3)
plot_batch_mse(
    (net_train_loss['028'],),
    ('net 028 train',),
    (None, (0, 20))
)
plt.subplot(2,2,2)
plot_epoch_ce(
    (net_mean_train_loss['028'], net_val_loss['028']),
    ('net 028 train', 'net 028 val'),
)
plt.subplot(2,2,4)
plot_epoch_mse(
    (net_mean_train_loss['028'], net_val_loss['028']),
    ('net 028 train', 'net 028 val'),
)
plt.tight_layout()


plt.subplot(2,2,1)
plot_batch_ce(
    (net_train_loss['027'],),
    ('net 028 train',),
#     (None, (6.5, 7.2))
)
plt.subplot(2,2,3)
plot_batch_mse(
    (net_train_loss['027'],),
    ('net 028 train',),
#     (None, (0, 20))
)
plt.subplot(2,2,2)
plot_epoch_ce(
    (net_mean_train_loss['027'], net_val_loss['027']),
    ('net 027 train', 'net 027 val'),
)
plt.subplot(2,2,4)
plot_epoch_mse(
    (net_mean_train_loss['027'], net_val_loss['027']),
    ('net 027 train', 'net 027 val'),
)
plt.tight_layout()


plt.subplot(2,2,1)
plot_batch_ce(
    (net_train_loss['033'],),
    ('net 028 train',),
    (None, (0, 7.5))
)
plt.subplot(2,2,3)
plot_batch_mse(
    (net_train_loss['033'],),
    ('net 028 train',),
    (None, (0, 30))
)
plt.subplot(2,2,2)
plot_epoch_ce(
    (net_mean_train_loss['033'], net_val_loss['033']),
    ('net 033 train', 'net 033 val'),
    (None, (0, 7))
)
plt.subplot(2,2,4)
plot_epoch_mse(
    (net_mean_train_loss['033'], net_val_loss['033']),
    ('net 033 train', 'net 033 val'),
    (None, (0, 45))
)
plt.tight_layout()





import torch
import torchvision
from torch.autograd import Variable as V
from utils.visualise import make_dot


resnet_18 = torchvision.models.resnet18(pretrained=True)
resnet_18.eval();


# by setting the volatile flag to True, intermediate caches are not saved
# making the inspection of the graph pretty boring / useless
torch.manual_seed(0)
x = V(torch.randn(1, 3, 224, 224))#, volatile=True)
h_x = resnet_18(x)


dot = make_dot(h_x)  # generate network graph
dot.render('net.dot');  # save DOT and PDF in the current directory
# dot  # uncomment for displaying the graph in the notebook


# explore network graph
print('h_x creator ->',h_x.creator)
print('h_x creator prev fun type ->', type(h_x.creator.previous_functions))
print('h_x creator prev fun length ->', len(h_x.creator.previous_functions))
print('\n--- content of h_x creator prev fun ---')
for a, b in enumerate(h_x.creator.previous_functions): print(a, '-->', b)
print('---------------------------------------\n')


# The current node is a `torch.nn._functions.linear.Linear` object, fed by
# 
# - 0 --> output of `torch.autograd._functions.tensor.View` object
# - 1 --> weight matrix of size `(1000, 512)`
# - 2 --> bias vector of size `(1000)`
# 

print(resnet_18)


resnet_18._modules.keys()


avgpool_layer = resnet_18._modules.get('avgpool')
h = avgpool_layer.register_forward_hook(
        lambda m, i, o: \
        print(
            'm:', type(m),
            '\ni:', type(i),
                '\n   len:', len(i),
                '\n   type:', type(i[0]),
                '\n   data size:', i[0].data.size(),
                '\n   data type:', i[0].data.type(),
            '\no:', type(o),
                '\n   data size:', o.data.size(),
                '\n   data type:', o.data.type(),
        )
)
h_x = resnet_18(x)
h.remove()


my_embedding = torch.zeros(512)
def fun(m, i, o): my_embedding.copy_(o.data)
h = avgpool_layer.register_forward_hook(fun)
h_x = resnet_18(x)
h.remove()


# print first values of the embedding
my_embedding[:10].view(1, -1)


# get matplotlib configuration
get_ipython().magic('run plot_conf.py')


from data.VideoFolder import VideoFolder


my_data = VideoFolder('data/256min_data_set/')


def process(data):
    
    nb_videos = len(data.videos)
    frames_per_video = tuple(last - first + 1 for ((last, first), _) in data.videos)
    sorted_frames = sorted(frames_per_video)
    plt.plot(sorted_frames)
    plt.ylabel('Number of frames')
    plt.xlabel('Sorted video index')
    plt.ylim(ymin=0)
    print('There are', len(frames_per_video))
    print('The 10 shortest videos have', *sorted_frames[:10], 'frames')
    print('The 10 longest videos have', *sorted_frames[-10:], 'frames')
    
    return nb_videos, frames_per_video

(nb_videos, frames_per_video) = process(my_data)


def fit_distribution(frames_count, nb_videos, enough=1e3):
    plt.figure(1)
    n, bins, patches = plt.hist(frames_count, bins=50)
    bin_width = bins[1] - bins[0]
    plt.xlabel('Frame count')
    plt.ylabel('Video count')

    from scipy.stats import t
    import numpy as np
    param = t.fit(frames_count)
    x = np.linspace(min(frames_count),max(frames_count),500)
    area = len(frames_count) * bin_width
    pdf_fitted = t.pdf(x, param[0], loc=param[1], scale=param[2]) * area
    plt.plot(x, pdf_fitted)

    from scipy.stats import norm
    normal_pdf = norm.pdf(x, loc=param[-2], scale=param[-1]) * area
    plt.plot(x, normal_pdf, c='.6', linewidth=.5)
    p = norm.fit(frames_count)
    normal_pdf = norm.pdf(x, loc=p[-2], scale=p[-1]) * area
    plt.plot(x, normal_pdf, c='.3', linewidth=.5)

    plt.legend(('t-student', 'norm1', 'norm2', 'hist'))
    plt.title('Frame count distribution')


    # draw limits
    plt.figure(2)
    plt.axhline(y=0)
    plt.axhline(y=nb_videos)

    y = n.cumsum()

    plt.step(bins[1:], y)
    plt.title('Frame count cumulative distribution')
    plt.xlabel('Frame count *fc*')
    plt.ylabel('Nb of video with at least *fc* frames');
    
    print('nu: {:.2f}'.format(param[0]))
    print('Average length (frames): {:.0f}'.format(param[1]))
    print('90% interval: [{:.0f}, {:.0f}]'.format(*t.interval(0.90, param[0], loc=param[1], scale=param[2])))
    print('95% interval: [{:.0f}, {:.0f}]'.format(*t.interval(0.95, param[0], loc=param[1], scale=param[2])))
    print('\n')
    
    for i, p in enumerate(zip(n, bins)):
        print('{:2d} {:3.0f} {:3.0f}'.format(i, *p))
        if i >= enough - 1: break
    
fit_distribution(frames_per_video, nb_videos, enough=25)


# getting new stats
my_train_data = VideoFolder('data/processed-data/train/')


(nb_train_videos, frames_per_train_video) = process(my_train_data)


fit_distribution(frames_per_train_video, nb_train_videos, enough=25)


# get videos length and name
a = tuple((last - first + 1, i) for (i, ((last, first), _)) in enumerate(my_train_data.videos))
b = sorted(a)  # sort by length
print('5 longest videos', *b[-5:], sep='\n')
v = my_train_data.videos[b[-1][1]]
print('The longest video is:', v[1][0])
print('which has length: ', v[0][0] - v[0][1] + 1)


# In origin, this was `floor/VID_20160605_094332.mp4`, with variable frame rate, and `1149` output frames.
# 

# getting new stats
my_val_data = VideoFolder('data/processed-data/val/')


(nb_val_videos, frames_per_val_video) = process(my_val_data)


# Damn `ffmpeg` with its shitty `-sseof -<seconds>` option. Better to use `-filter_complex select` and output one input file and stream to two single-stream files.
# 

my_sampled_data = VideoFolder('data/sampled-data/train/')


(nb_sampled_videos, frames_per_sampled_video) = process(my_sampled_data)


my_sampled_data.frames_per_class[:5]


[sum(my_sampled_data.frames_per_video[4*a:4*a+4]) for a in range(0,5)]


len(my_sampled_data.classes)


my_sampled_val_data = VideoFolder('data/sampled-data/val/', init_shuffle='init')


(nb_sampled_val_videos, frames_per_sampled_val_video) = process(my_sampled_val_data)


len(my_sampled_val_data.videos)


len(my_sampled_data.videos)


my_sampled_val_data.videos[:10]


my_sampled_data.videos[:10]





