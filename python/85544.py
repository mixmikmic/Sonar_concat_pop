# # Batch Sinkhorn Iteration Wasserstein Distance
# 
# Thomas Viehmann
# 
# This notebook implements sinkhorn iteration wasserstein distance layers.
# 
# ## Important note: This is under construction and does not yet work as well as it should.
# 

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function
from torchvision import datasets, transforms


# The following is a "plain sinkhorn" implementation that could be used in
# [C. Frogner et. al.: Learning with a Wasserstein Loss](https://arxiv.org/abs/1506.05439)
# 
# Note that we use a different convention for $\lambda$ (i.e. we use $\lambda$ as the weight for the regularisation, later versions of the above use $\lambda^-1$ as the weight).
# 
# The implementation has benefitted from
# 
# - Chiyuan Zhang's implementation in [Mocha](https://github.com/pluskid/Mocha.jl),
# - Rémi Flamary's implementation of various sinkhorn algorithms in [Python Optimal Transport](https://github.com/rflamary/POT)
# 
# Thank you!
# 

class WassersteinLossVanilla(Function):
    def __init__(self,cost, lam = 1e-3, sinkhorn_iter = 50):
        super(WassersteinLossVanilla,self).__init__()
        
        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost/self.lam)
        self.KM = self.cost*self.K
        self.stored_grad = None
        
    def forward(self, pred, target):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        assert pred.size(1)==self.na
        assert target.size(1)==self.nb

        nbatch = pred.size(0)
        
        u = self.cost.new(nbatch, self.na).fill_(1.0/self.na)
        
        for i in range(self.sinkhorn_iter):
            v = target/(torch.mm(u,self.K.t())) # double check K vs. K.t() here and next line
            u = pred/(torch.mm(v,self.K))
            #print ("stability at it",i, "u",(u!=u).sum(),u.max(),"v", (v!=v).sum(), v.max())
            if (u!=u).sum()>0 or (v!=v).sum()>0 or u.max()>1e9 or v.max()>1e9: # u!=u is a test for NaN...
                # we have reached the machine precision
                # come back to previous solution and quit loop
                raise Exception(str(('Warning: numerical errrors',i+1,"u",(u!=u).sum(),u.max(),"v",(v!=v).sum(),v.max())))

        loss = (u*torch.mm(v,self.KM.t())).mean(0).sum() # double check KM vs KM.t()...
        grad = self.lam*u.log()/nbatch # check whether u needs to be transformed        
        grad = grad-torch.mean(grad,dim=1).expand_as(grad)
        grad = grad-torch.mean(grad,dim=1).expand_as(grad) # does this help over only once?
        self.stored_grad = grad

        dist = self.cost.new((loss,))
        return dist
    def backward(self, grad_output):
        #print (grad_output.size(), self.stored_grad.size())
        return self.stored_grad*grad_output[0],None


# The following is a variant of the "log-stabilized sinkhorn" algorithm as described by [B. Schmitzer: Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems](https://arxiv.org/abs/1610.06519).
# However, the author (for his application of computing the transport map for a single pair of measures) uses a form that modifies the $K$ matrix. This makes is less suitable for processing (mini-) batches, where we want to avoid the additional dimension.
# 
# To the best of my knowledge, this is the first implementation of a batch stabilized sinkhorn algorithm and I would appreciate if you find it useful, you could credit
# *Thomas Viehmann: Batch Sinkhorn Iteration Wasserstein Distance*, [https://github.com/t-vi/pytorch-tvmisc/wasserstein-distance/Pytorch_Wasserstein.ipynb](https://github.com/t-vi/pytorch-tvmisc/wasserstein-distance/Pytorch_Wasserstein.ipynb).
# 

class WassersteinLossStab(Function):
    def __init__(self,cost, lam = 1e-3, sinkhorn_iter = 50):
        super(WassersteinLossStab,self).__init__()
        
        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost/self.lam)
        self.KM = self.cost*self.K
        self.stored_grad = None
        
    def forward(self, pred, target):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        assert pred.size(1)==self.na
        assert target.size(1)==self.nb

        batch_size = pred.size(0)
        
        log_a, log_b = torch.log(pred), torch.log(target)
        log_u = self.cost.new(batch_size, self.na).fill_(-numpy.log(self.na))
        log_v = self.cost.new(batch_size, self.nb).fill_(-numpy.log(self.nb))
        
        for i in range(self.sinkhorn_iter):
            log_u_max = torch.max(log_u, dim=1)[0]
            u_stab = torch.exp(log_u-log_u_max.expand_as(log_u))
            log_v = log_b - torch.log(torch.mm(self.K.t(),u_stab.t()).t()) - log_u_max.expand_as(log_v)
            log_v_max = torch.max(log_v, dim=1)[0]
            v_stab = torch.exp(log_v-log_v_max.expand_as(log_v))
            log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max.expand_as(log_u)

        log_v_max = torch.max(log_v, dim=1)[0]
        v_stab = torch.exp(log_v-log_v_max.expand_as(log_v))
        logcostpart1 = torch.log(torch.mm(self.KM,v_stab.t()).t())+log_v_max.expand_as(log_u)
        wnorm = torch.exp(log_u+logcostpart1).mean(0).sum() # sum(1) for per item pair loss...
        grad = log_u*self.lam
        grad = grad-torch.mean(grad,dim=1).expand_as(grad)
        grad = grad-torch.mean(grad,dim=1).expand_as(grad) # does this help over only once?
        grad = grad/batch_size
        
        self.stored_grad = grad

        return self.cost.new((wnorm,))
    def backward(self, grad_output):
        #print (grad_output.size(), self.stored_grad.size())
        #print (self.stored_grad, grad_output)
        res = grad_output.new()
        res.resize_as_(self.stored_grad).copy_(self.stored_grad)
        if grad_output[0] != 1:
            res.mul_(grad_output[0])
        return res,None


# We may test our implementation against Rémi Flamary's algorithms in [Python Optimal Transport](https://github.com/rflamary/POT).
# 

import ot
import numpy
from matplotlib import pyplot
get_ipython().magic('matplotlib inline')


# test problem from Python Optimal Transport
n=100
a=ot.datasets.get_1D_gauss(n,m=20,s=10).astype(numpy.float32)
b=ot.datasets.get_1D_gauss(n,m=60,s=30).astype(numpy.float32)
c=ot.datasets.get_1D_gauss(n,m=40,s=20).astype(numpy.float32)
a64=ot.datasets.get_1D_gauss(n,m=20,s=10).astype(numpy.float64)
b64=ot.datasets.get_1D_gauss(n,m=60,s=30).astype(numpy.float64)
c64=ot.datasets.get_1D_gauss(n,m=40,s=20).astype(numpy.float64)
# distance function
x=numpy.arange(n,dtype=numpy.float32)
M=(x[:,numpy.newaxis]-x[numpy.newaxis,:])**2
M/=M.max()
x64=numpy.arange(n,dtype=numpy.float64)
M64=(x64[:,numpy.newaxis]-x64[numpy.newaxis,:])**2
M64/=M64.max()


transp = ot.bregman.sinkhorn(a,b,M,reg=1e-3)
transp2 = ot.bregman.sinkhorn_stabilized(a,b,M,reg=1e-3)


(transp*M).sum(), (transp2*M).sum()


cabt = Variable(torch.from_numpy(numpy.stack((c,a,b),axis=0)))
abct = Variable(torch.from_numpy(numpy.stack((a,b,c),axis=0)))


lossvanilla = WassersteinLossVanilla(torch.from_numpy(M), lam=0.1)
loss = lossvanilla
losses = loss(cabt,abct), loss(cabt[:1],abct[:1]), loss(cabt[1:2],abct[1:2]), loss(cabt[2:],abct[2:])
sum(losses[1:])/3, losses


loss = WassersteinLossStab(torch.from_numpy(M), lam=0.1)
losses = loss(cabt,abct), loss(cabt[:1],abct[:1]), loss(cabt[1:2],abct[1:2]), loss(cabt[2:],abct[2:])
sum(losses[1:])/3, losses


# The stabilized version can handle the extended range needed to get closer to the Python Optimal Transport loss.
# 

transp3 = ot.bregman.sinkhorn_stabilized(a,b,M,reg=1e-2)
loss = WassersteinLossStab(torch.from_numpy(M), lam=0.01)
(transp3*M).sum(), loss(cabt[1:2],abct[1:2]).data[0]


# By the linear expansion, we should have
# $$
# L(x2) \approx L(x1)+\nabla L(\frac{x1+x2}{2})(x2-x1),
# $$
# so in particular we can see if for an example
# $L(x+\epsilon \nabla L)-L(x1) / \epsilon \|\nabla L\|^2 \approx 1$.
# 
# This seems to be the case ... sometimes.
# 

theloss = WassersteinLossStab(torch.from_numpy(M), lam=0.01, sinkhorn_iter=50)
cabt = Variable(torch.from_numpy(numpy.stack((c,a,b),axis=0)))
abct = Variable(torch.from_numpy(numpy.stack((a,b,c),axis=0)),requires_grad=True)
lossv1 = theloss(abct,cabt)
lossv1.backward()
grv = abct.grad
epsilon = 1e-5
abctv2 = Variable(abct.data-epsilon*grv.data, requires_grad=True)
lossv2 = theloss(abctv2, cabt)
lossv2.backward()
grv2 = abctv2.grad
(lossv1.data-lossv2.data)/(epsilon*((0.5*(grv.data+grv2.data))**2).sum()) # should be around 1


# Naturally, one has to check whether the abctv2 is a valid probability distribution (i.e. all entries $>0$). It seems that the range of $\lambda$ in which the gradient works well is somewhat limited. This may point to a bug in the implementation.
# 
# Note also that feeding the same distribution in both arguments results in a NaN, when 0 is the correct answer.
# 

# ## Straightforward port of Python Optimal Transport's Sinkhorn routines
# 
# These are more straightforward ports of Rémi Flamary's algorithms in [Python Optimal Transport](https://github.com/rflamary/POT) useful for investigating stability.
# 

def sinkhorn(a,b, M, reg, numItermax = 1000, stopThr=1e-9, verbose=False, log=False):
    # seems to explode terribly fast with 32 bit floats...
    if a is None:
        a = M.new(M.size(0)).fill_(1/m.size(0))
    if b is None:
        b = M.new(M.size(0)).fill_(1/M.size(1))

    # init data
    Nini = a.size(0)
    Nfin = b.size(0)

    cpt = 0
    if log:
        log={'err':[]}

    # we assume that no distances are null except those of the diagonal of distances
    u = M.new(Nfin).fill_(1/Nfin)
    v = M.new(Nfin).fill_(1/Nfin)
    uprev=M.new(Nini).zero_()
    vprev=M.new(Nini).zero_()

    K = torch.exp(-M/reg)

    Kp = K/(a[:,None].expand_as(K))
    transp = K
    cpt = 0
    err=1
    while (err>stopThr and cpt<numItermax):
        Kt_dot_u = torch.mv(K.t(),u)
        if (Kt_dot_u==0).sum()>0 or (u!=u).sum()>0 or (v!=v).sum()>0: # u!=u is a test for NaN...
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev
            break
        uprev = u
        vprev = v
        v = b/Kt_dot_u
        u = 1./torch.mv(Kp,v)
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp =   (u[:,None].expand_as(K))*K*(v[None,:].expand_as(K))
            err = torch.dist(transp.sum(0),b)**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt%200 ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))
        cpt = cpt +1
    if log:
        log['u']=u
        log['v']=v
    #print 'err=',err,' cpt=',cpt
    if log:
        return (u[:,None].expand_as(K))*K*(v[None,:].expand_as(K)),log
    else:
        return (u[:,None].expand_as(K))*K*(v[None,:].expand_as(K))


# test 32 bit vs. 64 bit for unstabilized
typ = numpy.float64
dist_torch64 =  sinkhorn(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
typ = numpy.float32
dist_torch32 =  sinkhorn(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
dist_pot     = ot.bregman.sinkhorn(a,b,M,reg=1e-3)
numpy.abs(dist_torch64.numpy()-dist_pot).max(), numpy.abs(dist_torch32.numpy()-dist_pot).max()


def sinkhorn_stabilized(a,b, M, reg, numItermax = 1000,tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False,print_period=20, log=False):
    if a is None:
        a = M.new(m.size(0)).fill_(1/m.size(0))
    if b is None:
        b = M.new(m.size(0)).fill_(1/m.size(1))

    # init data
    na = a.size(0)
    nb = b.size(0)

    cpt = 0
    if log:
        log={'err':[]}


    # we assume that no distances are null except those of the diagonal of distances
    if warmstart is None:
        alpha,beta=M.new(na).zero_(),M.new(nb).zero_()
    else:
        alpha,beta=warmstart
    u,v = M.new(na).fill_(1/na),M.new(nb).fill_(1/nb)
    uprev,vprev=M.new(na).zero_(),M.new(nb).zero_()

    def get_K(alpha,beta):
        """log space computation"""
        return torch.exp(-(M-alpha[:,None].expand_as(M)-beta[None,:].expand_as(M))/reg)

    def get_Gamma(alpha,beta,u,v):
        """log space gamma computation"""
        return torch.exp(-(M-alpha[:,None].expand_as(M)-beta[None,:].expand_as(M))/reg+torch.log(u)[:,None].expand_as(M)+torch.log(v)[None,:].expand_as(M))

    K=get_K(alpha,beta)
    transp = K
    loop=True
    cpt = 0
    err=1
    while loop:

        if  u.abs().max()>tau or  v.abs().max()>tau:
            alpha, beta = alpha+reg*torch.log(u), beta+reg*torch.log(v)
            u,v = M.new(na).fill_(1/na),M.new(nb).fill_(1/nb)
            K=get_K(alpha,beta)

        uprev = u
        vprev = v
        
        Kt_dot_u = torch.mv(K.t(),u)
        v = b/Kt_dot_u
        u = a/torch.mv(K,v)

        if cpt%print_period==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = get_Gamma(alpha,beta,u,v)
            err = torch.dist(transp.sum(0),b)**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt%(print_period*20) ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))


        if err<=stopThr:
            loop=False

        if cpt>=numItermax:
            loop=False


        if (Kt_dot_u==0).sum()>0 or (u!=u).sum()>0 or (v!=v).sum()>0: # u!=u is a test for NaN...
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev
            break

        cpt = cpt +1
    #print 'err=',err,' cpt=',cpt
    if log:
        log['logu']=alpha/reg+torch.log(u)
        log['logv']=beta/reg+torch.log(v)
        log['alpha']=alpha+reg*torch.log(u)
        log['beta']=beta+reg*torch.log(v)
        log['warmstart']=(log['alpha'],log['beta'])
        return get_Gamma(alpha,beta,u,v),log
    else:
        return get_Gamma(alpha,beta,u,v)


# test 32 bit vs. 64 bit for stabilized
typ = numpy.float64
dist_torch64 =  sinkhorn_stabilized(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
typ = numpy.float32
dist_torch32 =  sinkhorn_stabilized(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
dist_pot     = ot.bregman.sinkhorn_stabilized(a,b,M,reg=1e-3)
numpy.abs(dist_torch64.numpy()-dist_pot).max(), numpy.abs(dist_torch32.numpy()-dist_pot).max()





# # Regularized gradient descent GAN optimization
# *Thomas Viehmann* <tv@lernapparat.de>
# 
# This is a pytorch adaptation of a toy problem in the [code](https://github.com/locuslab/gradient_regularized_gan/blob/master/gaussian-toy-regularized.py) accompanying
# [V. Nagarajan, J.Z. Kolter: Gradient descent GAN optimization is locally stable](https://arxiv.org/abs/1706.04156).
# 
# This notebook regularizes the generator gradient descent for the vanilla (DCGAN) GAN.
# 
# In the paper's notation the score function (minimized by G and maximized by D) is defined as
# $$
# V(\theta_G, \theta_D) := E_{x\sim p_{data}}[f(D(x))]+E_{z \sim p_{noise}}[f(-D(G(z))].
# $$
# In our case the of vanilla GAN, the logsigmoid scoring function $f:=-\log(1+exp(-x))$, see the python function `score` below.
# Note that this is equivalent to the [`binary_cross_entropy with logits`](http://pytorch.org/docs/master/nn.html#torch.nn.functional.binary_cross_entropy_with_logits) function with a target indicating "realness" - the formulation the original code uses.
# 
# As usual, $\theta_G$ and $\theta_D$ denote the generator's and discriminator's parameters and the expectations are actually sampled in minibatches in the stochastic updates.
# 
# The authors propose to replace the vanilla GAN generator loss $V$ by the regularized
# $$
# \tilde V := V + \eta \| \nabla_{\theta_D} V \|^2 
# $$
# and train the generator and the discriminator with alternating single steps.
# This is done below.
# 

import itertools
import types
get_ipython().magic('matplotlib inline')
import seaborn
from matplotlib import pyplot
import torch
from torch.autograd import Variable
import IPython
import numpy


def sample_mog(batch_size, n_mixture=8, std=0.02, radius=2.0):
    thetas = torch.arange(0, 2*numpy.pi, 2*numpy.pi/n_mixture, out=torch.cuda.FloatTensor())
    centers = torch.stack([radius * torch.sin(thetas), radius * torch.cos(thetas)],dim=1)
    cat = torch.distributions.Categorical(torch.cuda.FloatTensor(1,n_mixture).fill_(1.0)).sample_n(batch_size).squeeze(1)
    sample = torch.cuda.FloatTensor(batch_size,2).normal_()*std+centers[cat]
    return sample
 


params = types.SimpleNamespace()
for k,v in dict(
    batch_size=512,
    disc_learning_rate=1e-4,
    gen_learning_rate=1e-4,
    beta1=0.5,
    epsilon=1e-8,
    max_iter=100001,
    viz_every=1000,
    z_dim=256,
    x_dim=2,
    unrolling_steps=0,
    regularizer_weight=0.5,
  ).items():
  setattr(params,k,v)


class Generator(torch.nn.Sequential):
    def __init__(self, input_dim=params.z_dim, output_dim=2, n_hidden=128, n_layer=2):
        super().__init__(*(sum(([torch.nn.Linear((input_dim if i==0 else n_hidden), n_hidden),
                              torch.nn.ReLU()] for i in range(n_layer)),[])
                             +[torch.nn.Linear(n_hidden, output_dim)]))

class Lambda(torch.nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
    def forward(self, *x):
        return self.lam(*x)

# I don't know why the /4 is strictly necessary... Just copying locuslab's version
class Discriminator(torch.nn.Sequential):
    def __init__(self, input_dim=2, n_hidden=128, n_layer=1):
        super().__init__(*([Lambda(lambda x: x/4.0)]
                           +sum(([torch.nn.Linear((input_dim if i==0 else n_hidden), n_hidden),
                                  torch.nn.ReLU()] for i in range(n_layer)),[])
                           +[torch.nn.Linear(n_hidden, 1)]))

generator = Generator()
discriminator = Discriminator()
generator.cuda()
discriminator.cuda()

for p in itertools.chain(generator.parameters(), discriminator.parameters()):
    if p.data.dim()>1:
        torch.nn.init.orthogonal(p, 0.8)
#with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=0.8)):

def score(real_score, fake_score):
    # D maximizes this, G minimizes this + a regularizer
    return torch.nn.functional.logsigmoid(real_score).mean()+torch.nn.functional.logsigmoid(-fake_score).mean()

d_opt = torch.optim.Adam(discriminator.parameters(), lr=params.disc_learning_rate, betas = (params.beta1,0.999), eps=params.epsilon)
g_opt = torch.optim.Adam(generator.parameters(), lr=params.gen_learning_rate, betas = (params.beta1,0.999), eps=params.epsilon)
all_d_gr_norms = []
all_scores = []


for i in range(params.max_iter):
    noise = Variable(torch.cuda.FloatTensor(params.batch_size, params.z_dim).normal_())
    real_data = Variable(sample_mog(params.batch_size))
    
    # Discriminator update
    generator.eval()
    discriminator.train()

    fake_data = generator(noise).detach()
    fake_score = discriminator(fake_data)
    real_score = discriminator(real_data)
    d_scores = -score(real_score, fake_score)    
    d_opt.zero_grad()
    d_scores.backward()
    d_opt.step()

    # Generator Update
    generator.train()
    discriminator.eval()
    
    fake_data = generator(noise)
    fake_score = discriminator(fake_data)
    real_score = discriminator(real_data)
    scores = score(real_score, fake_score)

    grads = torch.autograd.grad(scores, discriminator.parameters(), retain_graph=True, create_graph=True)
    gr_norm_sq = 0.0
    for gr in grads:
        gr_norm_sq += (gr**2).sum()

    all_scores.append(scores.data[0])
    all_d_gr_norms.append(gr_norm_sq.data[0]**0.5)

    g_loss = scores+params.regularizer_weight*gr_norm_sq

    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()

    if i % params.viz_every == 0:
        IPython.display.clear_output(True)
        pyplot.figure(figsize=(10,5))
        pyplot.subplot(1,2,1)
        xx = fake_data.data.cpu().numpy()
        yy = real_data.data.cpu().numpy()
        pyplot.scatter(xx[:, 0], xx[:, 1], edgecolor='none',alpha=0.6)
        pyplot.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none', alpha=0.6)
        pyplot.subplot(1,2,2)
        generator.eval()
        noise = Variable(torch.cuda.FloatTensor(params.batch_size*10, params.z_dim).normal_(), volatile=True)
        fake_data = generator(noise)
        fake_data_cpu = fake_data.data.cpu().numpy()
        seaborn.kdeplot(fake_data_cpu[:, 0], fake_data_cpu[:, 1], shade=True, cmap='Greens', n_levels=20, clip=[[-2.5,2.5]]*2)
        pyplot.gca().set_facecolor(seaborn.color_palette('Greens', n_colors=256)[0])
        pyplot.title('density estimation step %d'%i)
        pyplot.show()
        print(d_scores.data[0], gr_norm_sq.data[0], scores.data[0])





pyplot.figure(figsize=(10,5))        
pyplot.subplot(1,2,1)
pyplot.title('Discriminator Gradient L2 Norm')
pyplot.xlabel('Iteration')
pyplot.plot(all_d_gr_norms)
pyplot.subplot(1,2,2)
pyplot.title('Discriminator Score')
pyplot.xlabel('Iteration')
pyplot.plot(all_scores);


# I hope this notebook is useful for you. I appreciate your feedback and read every mail.
# 
# Thomas Viehmann <tv@lernapparat.de>
# 

# # Gaussian mixture density networks in one dimension
# *Thomas Viehmann* <tv@lernapparat.de>
# 
# Today we want to implement a simple Gaussian mixture density network. Mixture Density Networks have been introduced by [Bishop](http://publications.aston.ac.uk/373/1/NCRG_94_004.pdf) in the 1994 article of the same name.
# 
# The basic idea is simple: Given some input, say, $x$ we estimate the distribution of $Y | X=x$ as a mixture (in our case of Gaussians). In mixture density networks, the parameters of the mixture and its components are computed as a neural net. The canonical loss function is the negative log likelihood and we can backpropagate the loss through the parameters of the distribution to the network coefficients (so at the top there is what Kingma and Welling call the reparametrization trick and then there is plain neural network training).
# 
# Mixture density models are generative in the sense that you can sample from the estimated distribution $Y | X=x$.
# 
# Today we do this in 1d, but one popular application (in fact the one I implemented first) is the handwriting generation mode of [A. Graves: Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850).
# 
# First, we import the world
# 

from matplotlib import pyplot
get_ipython().magic('matplotlib inline')
import torch
import torch.utils.data
import numpy
from torch.autograd import Variable
import IPython
import itertools
import seaborn


# I used [David Ha's example](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/) which is almost twice as wiggly as Bishop's original example. Let's have a dataset and a data loader.
# 
# I will differ from Ha and Bishop in that I don't start with the case of the distribution to learn being the graph of a function of the condition $x$ (plus error), but we jump right into the non-graphical case.
# 

n_train = 1000
batch_size = 32
class DS(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
        self.y = torch.rand(n)*21-10.5
        self.x = torch.sin(0.75*self.y)*7.0+self.y*0.5+torch.randn(n)
    def __len__(self):
        return self.n
    def __getitem__(self,i):
        return (self.x[i],self.y[i])

train_ds = DS(n_train)
pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
pyplot.show()
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


class GaussianMixture1d(torch.nn.Module):
    def __init__(self, n_in, n_mixtures, eps=0):
        super(GaussianMixture1d, self).__init__()
        self.n_in = n_in
        self.eps = eps
        self.n_mixtures = n_mixtures        
        self.lin = torch.nn.Linear(n_in, 3*n_mixtures)
        self.log_2pi = numpy.log(2*numpy.pi)

    def params(self, inp, pi_bias=0, std_bias=0):
        # inp = batch x input
        p = self.lin(inp)
        pi = torch.nn.functional.softmax(p[:,:self.n_mixtures]*(1+pi_bias)) # mixture weights (probability weights)
        mu = p[:,self.n_mixtures:2*self.n_mixtures] # means of the 1d gaussians
        sigma = (p[:,2*self.n_mixtures:]-std_bias).exp() # stdevs of the 1d gaussians
        sigma = sigma+self.eps
        return pi,mu,sigma

    def forward(self, inp, x):
        # x = batch x 3 (=movement x,movement y,end of stroke)
        # loss, negative log likelihood
        pi,mu,sigma = self.params(inp)
        log_normal_likelihoods =  -0.5*((x.unsqueeze(1)-mu) / sigma)**2-0.5*self.log_2pi-torch.log(sigma) # batch x n_mixtures
        log_weighted_normal_likelihoods = log_normal_likelihoods+pi.log() # batch x n_mixtures
        maxes,_ = log_weighted_normal_likelihoods.max(1)
        mixture_log_likelihood = (log_weighted_normal_likelihoods-maxes.unsqueeze(1)).exp().sum(1).log()+maxes # log-sum-exp with stabilisation
        neg_log_lik = -mixture_log_likelihood
        return neg_log_lik

    def predict(self, inp, pi_bias=0, std_bias=0):
        # inp = batch x n_in
        pi,mu,sigma = self.params(inp, pi_bias=pi_bias, std_bias=std_bias)
        x = inp.data.new(inp.size(0)).normal_()
        mixture = pi.data.multinomial(1)       # batch x 1 , index to the mixture component
        sel_mu = mu.data.gather(1, mixture).squeeze(1)
        sel_sigma = sigma.data.gather(1, mixture).squeeze(1)
        x = x*sel_sigma+sel_mu
        return Variable(x)

class Model(torch.nn.Module):
    def __init__(self, n_inp = 1, n_hid = 24, n_mixtures = 24):
        super(Model, self).__init__()
        self.lin = torch.nn.Linear(n_inp, n_hid)
        self.mix = GaussianMixture1d(n_hid, n_mixtures)
    def forward(self, inp, x):
        h = torch.tanh(self.lin(inp))
        l = self.mix(h, x)
        return l.mean()
    def predict(self, inp, pi_bias=0, std_bias=0):
        h = torch.tanh(self.lin(inp))
        return self.mix.predict(h, std_bias=std_bias, pi_bias=pi_bias)


m = Model(1, 32, 20)     
opt = torch.optim.Adam(m.parameters(), 0.001)
m.cuda()
losses = []
for epoch in range(2000):
    thisloss  = 0
    for i,(x,y) in enumerate(train_dl):
        x = Variable(x.float().unsqueeze(1).cuda())
        y = Variable(y.float().cuda())
        opt.zero_grad()
        loss = m(x, y)
        loss.backward()
        thisloss += loss.data[0]/len(train_dl)
        opt.step()
    losses.append(thisloss)
    if epoch % 10 == 0:
        IPython.display.clear_output(wait=True)
        print (epoch, loss.data[0])
        x = Variable(torch.rand(1000,1).cuda()*30-15)
        y = m.predict(x)
        y2 = m.predict(x, std_bias=10)
        pyplot.subplot(1,2,1)
        pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
        pyplot.scatter(x.data.cpu().squeeze(1).numpy(), y.data.cpu().numpy(),facecolor='r', s=3)
        pyplot.scatter(x.data.cpu().squeeze(1).numpy(), y2.data.cpu().numpy(),facecolor='g', s=3)
        pyplot.subplot(1,2,2)
        pyplot.title("loss")
        pyplot.plot(losses)
        pyplot.show()


# Not bad!
# The green line basically gives the means of the mixture components rather than drawing from the mixture. This is implemented by biasing the variance. In some applications we might want to "clean" the output but not disable stochasticity completely. This can be achieved by biasing the variance of the components and biasing the component probabilites towards the largest one in the softmax.
# I learnt this trick from  Gaves' article.
# 
# ## Alternatives? Trying a GAN approach.
# 
# We might also try to reproduce a SLOGAN (Single sided Lipschitz Objective General Adversarial Network, a Wasserstein GAN variant).
# Note that while the MDN above tries to learn the distribution of $y$ given $x$, the GAN below tries to learn the unconditional distribution of pairs $(x,y)$.
# 
# I wrote a bit about the Wasserstein GAN and SLOGAN in two [blog](https://lernapparat.de/improved-wasserstein-gan/) [posts](https://lernapparat.de/more-improved-wgan/).
# 

class G(torch.nn.Module):
    def __init__(self, n_random=2, n_hidden=50):
        super(G, self).__init__()
        self.n_random = n_random
        self.l1 = torch.nn.Linear(n_random,n_hidden)
        self.l2 = torch.nn.Linear(n_hidden,n_hidden)
        #self.l2b = torch.nn.Linear(n_hidden,n_hidden)
        self.l3 = torch.nn.Linear(n_hidden,2)
    def forward(self, batch_size=32):
        x = Variable(self.l1.weight.data.new(batch_size, self.n_random).normal_())
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        #x = torch.nn.functional.relu(self.l2b(x))
        x = self.l3(x)
        return x

class D(torch.nn.Module):
    def __init__(self, lam=10.0, n_hidden=50):
        super(D, self).__init__()
        self.l1 = torch.nn.Linear(2,n_hidden)
        self.l2 = torch.nn.Linear(n_hidden,n_hidden)
        self.l3 = torch.nn.Linear(n_hidden,1)
        self.one = torch.FloatTensor([1]).cuda()
        self.mone = torch.FloatTensor([-1]).cuda()
        self.lam = lam
    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x
    def slogan_loss_and_backward(self, real, fake):
        self.zero_grad()
        f_real = self(real.detach())
        f_real_sum = f_real.sum()
        f_real_sum.backward(self.one, retain_graph=True)
        f_fake = self(fake.detach())
        f_fake_sum = f_fake.sum()
        f_fake_sum.backward(self.mone, retain_graph=True)
        f_mean = (f_fake_sum+f_real_sum)
        f_mean.abs().backward(retain_graph=True)
        dist = ((real.view(real.size(0),-1).unsqueeze(0)-fake.view(fake.size(0),-1).unsqueeze(1))**2).sum(2)**0.5
        f_diff = (f_real.unsqueeze(0)-f_fake.unsqueeze(1)).squeeze(2).abs()
        lip_dists = f_diff/(dist+1e-6)
        lip_penalty = (self.lam * (lip_dists.clamp(min=1)-1)**2).sum()
        lip_penalty.backward()
        return f_real_sum.data[0],f_fake_sum.data[0],lip_penalty.data[0], lip_dists.data.mean()

d = D()
d.cuda()
g = G(n_random=256, n_hidden=128)
g.cuda()
opt_d = torch.optim.Adam(d.parameters(), lr=1e-3)
opt_g = torch.optim.Adam(g.parameters(), lr=1e-3)

for p in itertools.chain(d.parameters(), g.parameters()):
    if p.data.dim()>1:
        torch.nn.init.orthogonal(p, 2.0)


def endless(dl):
    while True:
        for i in dl:
            yield i

batch_size=256
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
train_iter = endless(train_dl)

for i in range(30001):
    if i>0 and i%10000 == 0:
        for pg in opt_g.param_groups:
            pg['lr'] /= 3
        for pg in opt_d.param_groups:
            pg['lr'] /= 3
        print ("new lr",pg['lr'])
    for j in range(100 if i < 10 or i % 100 == 0 else 10):
        g.eval()
        d.train()
        for p in d.parameters():
            p.requires_grad = True
        real_x,real_y = next(train_iter)
        real = Variable(torch.stack([real_x.float().cuda(), real_y.float().cuda()],dim=1))
        fake = g(batch_size=batch_size)
        l_r, l_f, l_lip, lip_mean = d.slogan_loss_and_backward(real, fake)
        opt_d.step()
    if i % 100 == 0:
        print ("f_r:",l_r,"f_fake:",l_f, "lip loss:", l_lip, "lip_mean", lip_mean)
    g.train()
    d.eval()
    for p in d.parameters():
        p.requires_grad = False
    g.zero_grad()
    fake = g(batch_size=batch_size)
    f = d(fake)
    fsum = f.sum()
    fsum.backward()
    opt_g.step()
    if i % 1000 == 0:
        IPython.display.clear_output(wait=True)
        print (i)
        fake = g(batch_size=10000)
        fd = fake.data.cpu().numpy()
        pyplot.figure(figsize=(15,5))
        pyplot.subplot(1,3,1)
        pyplot.title("Generated Density")
        seaborn.kdeplot(fd[:,0], fd[:,1], shade=True, cmap='Greens', bw=0.01)
        pyplot.subplot(1,3,2)
        pyplot.title("Data Density Density")
        seaborn.kdeplot(train_ds.x.numpy(),train_ds.y.numpy(), shade=True, cmap='Greens', bw=0.01)
        pyplot.subplot(1,3,3)
        pyplot.title("Data (blue) and generated (red)")
        pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
        pyplot.scatter(fd[:,0], fd[:,1], facecolor='r',s=3, alpha=0.5)
        pyplot.show()


# We can try to see where $f$ sees the most discriminative power between the real sample and the generated fake distribution. Note that as $f$ depends on both distributions, it is not as simple as $f$ is low on the real data.
# 
# While we are far from perfectly resolving the path, it does not look too bad, either.
# 

N_dim = 100
x_test = torch.linspace(-20,20, N_dim)
y_test = torch.linspace(-15,15, N_dim)
x_test = (x_test.unsqueeze(0)*torch.ones(N_dim,1)).view(-1)
y_test = (y_test.unsqueeze(1)*torch.ones(1,N_dim)).view(-1)
xy_test = Variable(torch.stack([x_test, y_test], dim=1).cuda())
f_test = d(xy_test)
pyplot.imshow(f_test.data.view(N_dim,N_dim).cpu().numpy(), origin='lower', cmap=pyplot.cm.gist_heat_r, extent=(-20,20,-15,15))
pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), facecolor='b', s=2);


# **Update:** In a previous version, I expressed discontent with the GAN result. I revisited this notebook and after playing with the learning rate and increasing the noise dimension, the GAN seems to work reasonably well, too.
# 
# I appreciate your feedback at <tv@lernapparat.de>.
# 




# # Quick Shake-Shake Net for CIFAR10
# *Thomas Viehmann <tv@learnapparat.de>*
# 
# Prompted by a discussion on the Pytorch forums, here is a quick implementation of Shake-Shake Net in PyTorch
# 
# Reference: [Xavier Gastaldi, Shake-Shake regularization](https://arxiv.org/abs/1705.07485), [original (LUA-Torch) implementation by the author](https://github.com/xgastaldi/shake-shake)
# 
# Shortcomings:
# - I use the test set for setting the learning rate when I should use a validation set. If this were more than a little demo, this should not be done.
# - I don't do the cosine learning rate scheduling of the original implementation.
# - I am not quite certain whether the original implementation does data augmentation.
# - I have not added weight decay and the other usual tricks.
# - I only do 80 epochs.
# This only gets me 90% accuracy instead of >97% as in the reference.
# 
# The key thing is to how to implement the random weighting for forward and backward.
# I'm using a trick using `detach` I learned form [a post on Gumbel Softmax to the PyTorch forums by Hugh Perkins](https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/10).
# 
# (I'll add more commentary, but I wanted to share the code.)
# 

import torch
import torchvision
from matplotlib import pyplot
get_ipython().magic('matplotlib inline')
import os
import collections
import IPython
import itertools
import numpy
from torch.autograd import Variable


train_tr = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

test_tr = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


train_ds = torchvision.datasets.CIFAR10(root='/home/datasets/cifar10/', train=True, download=False, transform=train_tr)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)

test_ds = torchvision.datasets.CIFAR10(root='/home/datasets/cifar10/', train=False, download=False, transform=test_tr)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)


# shakeshake net leaning heavily on the original torch implementation https://github.com/xgastaldi/shake-shake/
class ShakeShakeBlock2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, per_image=True, rand_forward=True, rand_backward=True):
        super().__init__()
        self.same_width = (in_channels==out_channels)
        self.per_image = per_image
        self.rand_forward = rand_forward
        self.rand_backward = rand_backward
        self.stride = stride
        self.net1, self.net2 = [torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        torch.nn.BatchNorm2d(out_channels)) for i in range(2)]
        if not self.same_width:
            self.skip_conv1 = torch.nn.Conv2d(in_channels, out_channels//2, 1)
            self.skip_conv2 = torch.nn.Conv2d(in_channels, out_channels//2, 1)
            self.skip_bn = torch.nn.BatchNorm2d(out_channels)
    def forward(self, inp):
        if self.same_width:
            skip = inp
        else:
            # double check, this seems to be a fancy way to trow away the top-right and bottom-left of each 2x2 patch (with stride=2)
            x1 = torch.nn.functional.avg_pool2d(inp, 1, stride=self.stride)
            x1 = self.skip_conv1(x1)
            x2 = torch.nn.functional.pad(inp, (1,-1,1,-1))            # this makes the top and leftmost row 0. one could use -1,1
            x2 = torch.nn.functional.avg_pool2d(x2, 1, stride=self.stride)
            x2 = self.skip_conv2(x2)
            skip = torch.cat((x1,x2), dim=1)
            skip = self.skip_bn(skip)
        x1 = self.net1(inp)
        x2 = self.net2(inp)

        if self.training:
            if self.rand_forward:
                if self.per_image:
                    alpha = Variable(inp.data.new(inp.size(0),1,1,1).uniform_())
                else:
                    alpha = Variable(inp.data.new(1,1,1,1).uniform_())
            else:
                alpha = 0.5
            if self.rand_backward:
                if self.per_image:
                    beta = Variable(inp.data.new(inp.size(0),1,1,1).uniform_())
                else:
                    beta = Variable(inp.data.new(1,1,1,1).uniform_())
            else:
                beta = 0.5
            # this is the trick to get beta in the backward (because it does not see the detatched)
            # and alpha in the forward (when it sees the detached with the alpha and the beta cancel)
            x = skip+beta*x1+(1-beta)*x2+((alpha-beta)*x1).detach()+((beta-alpha)*x2).detach()
        else:
            x = skip+0.5*(x1+x2)
        return x

            
class ShakeShakeBlocks2d(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, depth, stride, per_image=True, rand_forward=True, rand_backward=True):
        super().__init__(*[
            ShakeShakeBlock2d(in_channels if i==0 else out_channels, out_channels, stride if i==0 else 1,
                              per_image, rand_forward, rand_backward) for i in range(depth)])

class ShakeShakeNet(torch.nn.Module):
    def __init__(self, depth=20, basewidth=32, per_image=True, rand_forward=True, rand_backward=True, num_classes=16):
        super().__init__()
        assert (depth - 2) % 6==0, "depth should be n*6+2"
        n = (depth - 2) // 6
        self.inconv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.s1 = ShakeShakeBlocks2d(16, basewidth, n, 1, per_image, rand_forward, rand_backward)
        self.s2 = ShakeShakeBlocks2d(basewidth, 2*basewidth, n, 2, per_image, rand_forward, rand_backward)
        self.s3 = ShakeShakeBlocks2d(2*basewidth, 4*basewidth, n, 2, per_image, rand_forward, rand_backward)
        self.fc = torch.nn.Linear(4*basewidth, num_classes)
    def forward(self, x):
        x = self.inconv(x)
        x = self.bn1(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.size(0), x.size(1), -1).mean(2)
        x = self.fc(x)
        return x



model = ShakeShakeNet()
model.cuda()


opt = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)


losses = []
corrects = []
last_corrects = 0


for epoch in range(80):
    model.train()
    print ("lr",opt.param_groups[0]['lr'])
    if opt.param_groups[0]['lr']<1e-6:
        print ("lr below 1e-5, exit")
        break
    for i,(data, target) in enumerate(train_dl):
        data = torch.autograd.Variable(data.cuda())
        target = torch.autograd.Variable(target.cuda())
        opt.zero_grad()
        pred = model(data)
        loss = torch.nn.functional.cross_entropy(pred, target)
        loss.backward()
        opt.step()
        if i % 50 == 49:
            print (epoch,i, len(train_dl), loss.data[0])
            losses.append(loss.data[0])
    model.eval()
    totalcorrect = 0
    total = 0
    corrects_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    for i,(data, target) in enumerate(test_dl):
        data = torch.autograd.Variable(data.cuda())
        pred = model(data).max(1)[1].data.cpu()
        correct = (target==pred).float()
        corrects_per_class += torch.sparse.FloatTensor(target.view(1,-1), correct, torch.Size((num_classes,))).to_dense()
        total_per_class += torch.sparse.FloatTensor(target.view(1,-1), torch.ones(target.size(0)), torch.Size((num_classes,))).to_dense()
        totalcorrect += correct.sum()
        total += target.size(0)
        if i%50==0: print(i, len(test_dl))
    corrects.append(totalcorrect/len(test_ds))
    if corrects[-1] < last_corrects*0.995:
        model.load_state_dict(last_state)
        for pg in opt.param_groups:
            pg['lr'] /= 2
    elif corrects[-1] > last_corrects:
        last_corrects = max(corrects)
        last_state = collections.OrderedDict([(k,v.cpu()) for k,v in model.state_dict().items()])
    IPython.display.clear_output(True)
    pyplot.figure(figsize=(15,5))
    pyplot.subplot(1,3,1)
    pyplot.plot(losses)
    pyplot.yscale('log')
    pyplot.subplot(1,3,2)
    pyplot.plot(corrects)
    pyplot.subplot(1,3,3)
    bars = pyplot.bar(torch.arange(0,num_classes),corrects_per_class/total_per_class)
    pyplot.xticks(numpy.arange(num_classes))
    pyplot.gca().set_xticklabels(classes)
    pyplot.show()
    print ("val",totalcorrect,len(test_ds),totalcorrect/len(test_ds), "max", max(corrects))
model.load_state_dict(last_state)
print ("max correct",max(corrects))


# If you find bugs or have suggestions or this is useful to you, I appreciate your feedback.
# 
# Thomas Viehmann, <tv@lernapparat.de>
# 




