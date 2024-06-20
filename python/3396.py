# This notebook will walk through the reason that it is necessary to model response times, and the various ways to model them. We will start by generating a design that has trials that vary in reaction time.  This is adapted from Poldrack (2014, Developmental Cognitive Neuroscience).
# 

import numpy
import nibabel
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import statsmodels.nonparametric.smoothers_lowess
import scipy
import nipype.algorithms.modelgen as model   # model generation
from nipype.interfaces.base import Bunch
from  nipype.interfaces import fsl
from statsmodels.tsa.arima_process import arma_generate_sample
import os,shutil
from IPython.display import display, HTML
import seaborn as sns
sns.set_style("white")

from nipype.caching import Memory
mem = Memory(base_dir='.')

def clearmem():
    for root, dirs, files in os.walk(mem.base_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    
# generate the design at a higher temporal resolution
tr=0.1


# First generate a design with four events that differ in their duration (by turning on the function for inreasing amounts of time).
# 

plt.plot(spm_hrf(tr))


variable_sf=numpy.zeros(1000)

variable_sf[100:102]=1
variable_sf[300:303]=1
variable_sf[500:505]=1
variable_sf[700:706]=1
plt.plot(variable_sf)
variable_sf_conv=numpy.convolve(variable_sf,
                                spm_hrf(tr,oversampling=1))[0:len(variable_sf)]
plt.plot(variable_sf_conv)


# This effect tapers off for trials longer than the HRF
# 

variable_sf_long=numpy.zeros(1000)

variable_sf_long[100:200]=1
variable_sf_long[500:900]=1
plt.plot(variable_sf_long)
variable_sf_long_conv=numpy.convolve(variable_sf_long,
                                spm_hrf(tr,oversampling=1))[0:len(variable_sf)]
plt.plot(variable_sf_long_conv)


# Generate a beta-series design matrix that fits a separate regressor for each of the four trials; this is equivalent to separately modeling the intensity of each trial (assuming a constant duration for each).
# 

hrf_bases=numpy.zeros((1000,4))
hrf_bases[100:104,0]=1
hrf_bases[300:304,1]=1
hrf_bases[500:504,2]=1
hrf_bases[700:704,3]=1
desmtx=numpy.zeros((1000,4))

for x in range(4):
    desmtx[:,x]=numpy.convolve(hrf_bases[:,x],
                               spm_hrf(tr,oversampling=1))[0:len(variable_sf)]
    
plt.matshow(desmtx, aspect='auto')


# Now fit the beta-series model, and generate the fitted regressor.
# 

b_est=numpy.linalg.inv(desmtx.T.dot(desmtx)).dot(desmtx.T).dot(variable_sf_conv)
print b_est
intensity_sf_conv=desmtx.dot(b_est)


plt.clf()
plt.plot(variable_sf_conv,color='k',linewidth=4)
plt.hold(True)
plt.plot(intensity_sf_conv,'c--')
#plt.plot(constant_sf_conv,color='b')
plt.plot(intensity_sf_conv - variable_sf_conv,color='b')
plt.text(10,-0.02,'RT')
plt.text(100,-0.02,'200 ms')
plt.text(300,-0.02,'300 ms')
plt.text(500,-0.02,'500 ms')
plt.text(700,-0.02,'600 ms')
plt.text(10,-0.03,'Beta')
plt.text(100,-0.03,'%0.2f'%b_est[0])
plt.text(300,-0.03,'%0.2f'%b_est[1])
plt.text(500,-0.03,'%0.2f'%b_est[2])
plt.text(700,-0.03,'%0.2f'%b_est[3])

plt.axis([0,1000,-0.05,0.15])
plt.legend(['Variable duration','Variable intensity (fitted)','Residual'],
           loc='upper left')


# The point to take away from this is that the variable duration and variable intensity have largely indistinguishable effects on the hemodynamic response, at least for relatively short events.
# 

# ###Modeling reaction times
# 
# Now let's look at the various ways that one can model response times. First let's generate a design with two conditions that differ in mean response times. We will use a lognormal distribution which is a reasonable approximation to the shape of RT distributions.
# 

diff=0.5 # difference in RT across conditions
ntrials=32 #  trials per condition
condition=numpy.zeros(ntrials)
condition[ntrials/2:]=1 
rt=numpy.zeros(len(condition))
rt[condition==0]=numpy.random.lognormal(0.0,0.2,ntrials/2)
rt[condition==1]=numpy.random.lognormal(diff,0.2,ntrials/2)

#rt[:ntrials/2]+=rtdiff
h1=numpy.histogram(rt[condition==0])
plt.plot((h1[1][1:]+h1[1][:1])/2.0,h1[0]/float(numpy.max(h1[0])))
h2=numpy.histogram(rt[condition==1])
plt.plot((h2[1][1:]+h2[1][:1])/2.0,h2[0]/float(numpy.max(h2[0])),color='red')
print 'Mean RT (condition 0):',numpy.mean(rt[condition==0])
print 'Mean RT (condition 1):',numpy.mean(rt[condition==1])
plt.legend(['condition 0','condition 1'])
meanrt=numpy.mean(rt)

# generate random onsets

trial_length=16 # length of each trial, including ISI
total_length=trial_length*ntrials
randonsets=numpy.arange(0,total_length,trial_length)
numpy.random.shuffle(randonsets)
onsets=numpy.zeros(len(randonsets))
onsets[condition==0]=numpy.sort(randonsets[condition==0])
onsets[condition==1]=numpy.sort(randonsets[condition==1])



# Now generate the data using these onsets and durations. We will generate three datasets:
# * constant event duration and activation across conditions (cd_ca)
# * variable event duration but constant activation across conditions (vd_ca)
# * constant event duration but variable activation across condition (cd_va)
# 

times=numpy.arange(0,total_length,1/100.)
deslen=len(times) # length of design in high-resolution (10 ms) space
sf_vd_ca=numpy.zeros(deslen) 
sf_cd_ca=numpy.zeros(deslen)
sf_cd_va=numpy.zeros(deslen)
activation_effect=1

for i in range(len(onsets)):
    start=onsets[i]*100.
    stop_var=onsets[i]*100 + round(rt[i]*10)
    stop_const=onsets[i]*100 + round(numpy.mean(rt)*10)
    sf_vd_ca[start:stop_var]=1
    sf_cd_ca[start:stop_const]=1
    sf_cd_va[start:stop_const]=1+condition[i]*activation_effect # add activation effect

noiselevel=0.25
noise=arma_generate_sample([1,0.4],[1,0.],total_length)*noiselevel

conv_sf_vd_ca=numpy.convolve(sf_vd_ca,spm_hrf(tr=0.01,oversampling=1.))[:len(sf_vd_ca)]
conv_sf_vd_ca=conv_sf_vd_ca[numpy.arange(0,len(conv_sf_vd_ca),100)]
data_vd_ca=conv_sf_vd_ca*50. + noise

conv_sf_cd_ca=numpy.convolve(sf_cd_ca,spm_hrf(tr=0.01,oversampling=1.))[:len(sf_cd_ca)]
conv_sf_cd_ca=conv_sf_cd_ca[numpy.arange(0,len(conv_sf_cd_ca),100)]
data_cd_ca=conv_sf_cd_ca*50. + noise

conv_sf_cd_va=numpy.convolve(sf_cd_va,spm_hrf(tr=0.01,oversampling=1.))[:len(sf_cd_va)]
conv_sf_cd_va=conv_sf_cd_va[numpy.arange(0,len(conv_sf_cd_va),100)]
data_cd_va=conv_sf_cd_va*50. + noise



plt.figure()
plt.plot(sf_vd_ca[1550:1650])
plt.ylim([0,2.2])
plt.title("variable duration / constant amplitude")
plt.figure()
plt.plot(sf_cd_ca[1550:1650])
plt.title("constant duration / constant amplitude")
plt.ylim([0,2.2])
plt.figure()
plt.plot(sf_cd_va[1550:1650])
plt.title("constant duration / variable amplitude")
plt.ylim([0,2.2])


plt.figure()
plt.plot(data_vd_ca[:50], label="variable duration / constant amplitude")
plt.plot(data_cd_ca[:50], label="constant duration / constant amplitude")
plt.plot(data_cd_va[:50], label="constant duration / variable amplitude")
plt.ylim([-1,3])
plt.legend()


plot


# First, build a model that assumes constant event durations
# 

info = [Bunch(conditions=['cond0',
                          'cond1'],
              onsets=[numpy.sort(onsets[condition==0]),
                      numpy.sort(onsets[condition==1])],
              durations=[[meanrt],
                         [meanrt]])]

# create a dummy image for SpecifyModel to look at
if not os.path.exists('tmp.nii.gz'):
    dummy=nibabel.Nifti1Image(numpy.zeros((12,12,12,total_length)),numpy.identity(4))
    dummy.to_filename('tmp.nii.gz')

s = model.SpecifyModel()
s.inputs.input_units = 'secs'
s.inputs.functional_runs = 'tmp.nii.gz'
s.inputs.time_repetition = 1.0
s.inputs.high_pass_filter_cutoff = 128.
s.inputs.subject_info = info
specify_model_results = s.run()

clearmem()

level1design = mem.cache(fsl.model.Level1Design)
level1design_results = level1design(interscan_interval = 1.0,
                                    bases = {'dgamma':{'derivs': False}},
                                    session_info = specify_model_results.outputs.session_info,
                                    model_serial_correlations=False)

modelgen = mem.cache(fsl.model.FEATModel)
modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                            ev_files=level1design_results.outputs.ev_files)


X=numpy.loadtxt(modelgen_results.outputs.design_file,skiprows=5)
X=numpy.hstack((X,numpy.ones((X.shape[0],1))))

print 'Model with constant event durations'

beta_hat_vd_ca_nort=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_vd_ca)

beta_hat_cd_ca_nort=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_cd_ca)

beta_hat_cd_va_nort=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_cd_va)

import pandas as pd
betas_nort=numpy.vstack((beta_hat_vd_ca_nort,beta_hat_cd_ca_nort,beta_hat_cd_va_nort))
df_nort=pd.DataFrame(betas_nort,columns=['Cond0','Cond1','Mean'],
                index=['variable duration/constant amplitude',
                       'constant duration/constant amplitude',
                      'constant duration/variable amplitude'])
HTML( df_nort.to_html() )
import seaborn as sns
sns.heatmap(X, vmin=0, vmax=0.8, xticklabels=df_nort.columns)


# Note that the first two datasets have equal activation strengths, and thus their betas should be the same.  
# 

# Now build a model using the actual reaction times as regressors.
# 

info = [Bunch(conditions=['cond0',
                          'cond1'],
              onsets=[onsets[condition==0],
                      onsets[condition==1]],
              durations=[rt[condition==0],
                         rt[condition==1]])]

# create a dummy image for SpecifyModel to look at
if not os.path.exists('tmp.nii.gz'):
    dummy=nibabel.Nifti1Image(numpy.zeros((12,12,12,total_length)),numpy.identity(4))
    dummy.to_filename('tmp.nii.gz')
    
s = model.SpecifyModel()
s.inputs.input_units = 'secs'
s.inputs.functional_runs = 'tmp.nii.gz'
s.inputs.time_repetition = 1.0
s.inputs.high_pass_filter_cutoff = 128.
s.inputs.subject_info = info
specify_model_results = s.run()

clearmem()
level1design = mem.cache(fsl.model.Level1Design)
level1design_results = level1design(interscan_interval = 1.0,
                                    bases = {'dgamma':{'derivs': False}},
                                    session_info = specify_model_results.outputs.session_info,
                                    model_serial_correlations=False)

modelgen = mem.cache(fsl.model.FEATModel)
modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                            ev_files=level1design_results.outputs.ev_files)


X=numpy.loadtxt(modelgen_results.outputs.design_file,skiprows=5)
X=numpy.hstack((X,numpy.ones((X.shape[0],1))))

print 'Model with variable durations'

beta_hat_vd_ca_rt=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_vd_ca)
beta_hat_cd_ca_rt=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_cd_ca)
beta_hat_cd_va_rt=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_cd_va)

import pandas as pd
betas_varrt=numpy.vstack((beta_hat_vd_ca_rt,beta_hat_cd_ca_rt,beta_hat_cd_va_rt))
df_varrt=pd.DataFrame(betas_varrt,columns=['Cond0','Cond1','Mean'],
                index=['variable duration/constant amplitude',
                       'constant duration/constant amplitude',
                      'constant duration/variable amplitude'])
sns.heatmap(X, vmin=0, vmax=0.8, xticklabels=df_nort.columns)
HTML( df_varrt.to_html() )


# There are two thigns to notice here.  First, the RT-unrelated region now has an artifactual difference between conditions, driven by the differences in RT across conditions that are now included in the regressor.  Second, notice that the difference in activation between conditions in the third dataset (where one actually exists) is reduced compared to the previous model, because some of the effect is being removed due to its correlation with the RT difference across conditions.
# 

# Now let's build a model that includes a separate parametric regressor for RT alongside the constant duration (unmodulated) regressor.
# 

info = [Bunch(conditions=['cond0-const',
                          'cond1-const',
                         'RT'],
              onsets=[onsets[condition==0],
                      onsets[condition==1],
                     onsets],
              durations=[[meanrt],
                         [meanrt],
                          [meanrt]],
              amplitudes=[[1],[1],rt-meanrt]
                         )]

# create a dummy image for SpecifyModel to look at
dummy=nibabel.Nifti1Image(numpy.zeros((12,12,12,total_length)),numpy.identity(4))
dummy.to_filename('tmp.nii.gz')
s = model.SpecifyModel()
s.inputs.input_units = 'secs'
s.inputs.functional_runs = 'tmp.nii.gz'
s.inputs.time_repetition = 1.0
s.inputs.high_pass_filter_cutoff = 128.
s.inputs.subject_info = info
specify_model_results = s.run()

clearmem()
level1design = mem.cache(fsl.model.Level1Design)
level1design_results = level1design(interscan_interval = 1.0,
                                    bases = {'dgamma':{'derivs': False}},
                                    session_info = specify_model_results.outputs.session_info,
                                    model_serial_correlations=False)

modelgen = mem.cache(fsl.model.FEATModel)
modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                            ev_files=level1design_results.outputs.ev_files)


X=numpy.loadtxt(modelgen_results.outputs.design_file,skiprows=5)
X=numpy.hstack((X,numpy.ones((X.shape[0],1))))

print 'Model with parametric RT effect'

beta_hat_vd_ca_param=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_vd_ca)
beta_hat_cd_ca_param=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_cd_ca)
beta_hat_cd_va_param=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_cd_va)


betas_param=numpy.vstack((beta_hat_vd_ca_param,beta_hat_cd_ca_param,beta_hat_cd_va_param))
df_param=pd.DataFrame(betas_param,columns=['Cond0','Cond1','RTparam','Mean'],
                index=['variable duration/constant amplitude',
                       'constant duration/constant amplitude',
                      'constant duration/variable amplitude'])
sns.heatmap(X, vmin=0, vmax=0.8, xticklabels=df_param.columns)
HTML( df_param.to_html() )


# Here you see that there are no big diferences between conditions (the first two colums) for the first two datasets, where there was no difference, and the model accurately detected the RT effect for the first dataset.  You also see that the differential activaiton effect is now roughtly as large as it was for the constant duration model, as it should be.
# 




# This notebook provides an introduction to some of the basic concepts of machine learning.
# 
# Let's start by generating some data to work with.  Let's say that we have a dataset that has tested people on two continuous measures (processing speed and age) and one discrete measure (diagnosis with any psychiatric disorder).  First let's create the continuous data assuming that there is a relationship between these two variables.  We will make a function to generate a new dataset, since we will need to do this multiple times.
# 

import numpy,pandas
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn.preprocessing import PolynomialFeatures,scale
from sklearn.linear_model import LinearRegression,LassoCV,Ridge
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.tools.tools import add_constant

recreate=True
if recreate:
    seed=20698
else:
    seed=numpy.ceil(numpy.random.rand()*100000).astype('int')
    print(seed)

numpy.random.seed(seed)

def make_continuous_data(mean=[45,100],var=[10,10],cor=-0.6,N=100):
    """
    generate a synthetic data set with two variables
    """
    cor=numpy.array([[1.,cor],[cor,1.]])
    var=numpy.array([[var[0],0],[0,var[1]]])
    cov=var.dot(cor).dot(var)
    return numpy.random.multivariate_normal(mean,cov,N)


n=50
d=make_continuous_data(N=n)
y=d[:,1]
plt.scatter(d[:,0],d[:,1])
plt.xlabel('age')
plt.ylabel('processing speed')

print('data R-squared: %f'%numpy.corrcoef(d.T)[0,1]**2)


# What is the simplest story that we could tell about processing speed these data?  Well, we could simply say that the variable is normal with a mean of zero and a standard deviation of 1.  Let's see how likely the observed processing speed values are given that set of parameters.  First, let's create a function that returns the normal log-likelihood of the data given a set of predicted values.
# 

def loglike(y,yhat,s2=None,verbose=True):
    N = len(y)
    SSR = numpy.sum((y-yhat)**2)
    if s2 is None:
        # use observed stdev
        s2 = SSR / float(N)
    logLike = -(n/2.)*numpy.log(s2) - (n/2.)*numpy.log(2*numpy.pi) - SSR/(2*s2)
    if verbose:
        print('SSR:',SSR)
        print('s2:',s2)
        print('logLike:',logLike)
    return logLike
    

logLike_null=loglike(y,numpy.zeros(len(y)),s2=1)


# We are pretty sure that the mean of our variables is not zero, so let's compute the mean and see if the likelihood of the data is higher.
# 

mean=numpy.mean(y)
print('mean:',mean)
pred=numpy.ones(len(y))*mean
logLike_mean=loglike(y,pred,s2=1)


# What about using the observed variance as well?

var=numpy.var(y)
print('variance',var)
pred=numpy.ones(len(y))*mean
logLike_mean_std=loglike(y,pred)


# Is there a relation between processing speed and age? Compute the linear regression equation to find out. 
# 

X=d[:,0]
X=add_constant(X)
result = sm.OLS( y, X ).fit()
print(result.summary())
intercept=result.params[0]
slope=result.params[1]
pred=result.predict(X)
logLike_ols=loglike(y,pred)
plt.scatter(y,pred)


print('processing speed = %f + %f*age'%(intercept,slope))
print('p =%f'%result.pvalues[1])

def get_RMSE(y,pred):
    return numpy.sqrt(numpy.mean((y - pred)**2))

def get_R2(y,pred):
    """ compute r-squared"""
    return numpy.corrcoef(y,pred)[0,1]**2

ax=plt.scatter(d[:,0],d[:,1])
plt.xlabel('age')
plt.ylabel('processing speed')
plt.plot(d[:,0], slope * d[:,0] + intercept, color='red')
# plot residual lines
d_predicted=slope*d[:,0] + intercept
for i in range(d.shape[0]):
    x=d[i,0]
    y=d[i,1]
    plt.plot([x,x],[d_predicted[i],y],color='blue')

RMSE=get_RMSE(d[:,1],d_predicted)
rsq=get_R2(d[:,1],d_predicted)
print('rsquared=%f'%rsq)


# This shows us that linear regression can provide a simple description of a complex dataset - we can describe the entire dataset in 2 numbers. Now let's ask how good this description is for a new dataset generated by the same process:
# 

d_new=make_continuous_data(N=n)
d_new_predicted=intercept + slope*d_new[:,0]
RMSE_new=get_RMSE(d_new[:,1],d_new_predicted)
rsq_new=get_R2(d_new[:,1],d_new_predicted)
print('R2 for new data: %f'%rsq_new)

ax=plt.scatter(d_new[:,0],d_new[:,1])
plt.xlabel('age')
plt.ylabel('processing speed')
plt.plot(d_new[:,0], slope * d_new[:,0] + intercept, color='red')


# Now let's do this 100 times and look at how variable the fits are.  
# 

nruns=100
slopes=numpy.zeros(nruns)
intercepts=numpy.zeros(nruns)
rsquared=numpy.zeros(nruns)

fig = plt.figure()
ax = fig.gca()

for i in range(nruns):
    data=make_continuous_data(N=n)
    slopes[i],intercepts[i],_,_,_=scipy.stats.linregress(data[:,0],data[:,1])
    ax.plot(data[:,0], slopes[i] * data[:,0] + intercepts[i], color='red', alpha=0.05)
    pred_orig=intercept + slope*data[:,0]
    rsquared[i]=get_R2(data[:,1],pred_orig)


print('Original R2: %f'%rsq)
print('Mean R2 for new datasets on original model: %f'%numpy.mean(rsquared))


# ### Cross-validation
# 
# The results above show that the fit of the model to the observed data overestimates our ability to predict on new data.  In many cases we would like to be able to quantify how well our model generalizes to new data, but it's often not possible to collect additional data.  The concept of *cross-validation* provides us with a way to measure how well a model generalizes.  The idea is to iteratively train the model on subsets of the data and then test the model on the left-out portion.  Let's first see what cross-validation looks like.  Perhaps the simplest version to understand is "leave-one-out" crossvalidation, so let's look at that.  Here is what the training and test datasets would look like for a dataset with 10 observations; in reality this is way too few observations, but we will use it as an exmaple
# 

# initialize the sklearn leave-one-out operator
loo=LeaveOneOut()  

for train,test in loo.split(range(10)):
    print('train:',train,'test:',test)


# It is often more common to use larger test folds, both to speed up performance (since LOO can require lots of model fitting when there are a large number of observations) and because LOO error estimates can have high variance due to the fact that the models are so highly correlated.  This is referred to as K-fold cross-validation; generally we want to choose K somewhere around 5-10.  It's generally a good idea to shuffle the order of the observations so that the folds are grouped randomly.
# 

# initialize the sklearn leave-one-out operator
kf=KFold(n_splits=5,shuffle=True)  

for train,test in kf.split(range(10)):
    print('train:',train,'test:',test)


# Now let's perform leave-one-out cross-validation on our original dataset, so that we can compare it to the performance on new datasets.  We expect that the correlation between LOO estimates and actual data should be very similar to the Mean R2 for new datasets.  We can also plot a histogram of the estimates, to see how they vary across folds.
# 

loo=LeaveOneOut()

slopes_loo=numpy.zeros(n)
intercepts_loo=numpy.zeros(n)
pred=numpy.zeros(n)

ctr=0
for train,test in loo.split(range(n)):
    slopes_loo[ctr],intercepts_loo[ctr],_,_,_=scipy.stats.linregress(d[train,0],d[train,1])
    pred[ctr]=intercepts_loo[ctr] + slopes_loo[ctr]*data[test,0]
    ctr+=1

print('R2 for leave-one-out prediction: %f'%get_R2(pred,data[:,1]))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
_=plt.hist(slopes_loo,20)
plt.xlabel('slope estimate')
plt.ylabel('frequency')
plt.subplot(1,2,2)
_=plt.hist(intercepts_loo,20)
plt.xlabel('intercept estimate')
plt.ylabel('frequency')


# Now let's look at the effect of outliers on in-sample correlation and out-of-sample prediction.
# 

# add an outlier
data_null=make_continuous_data(N=n,cor=0.0)
outlier_multiplier=2.0

data=numpy.vstack((data_null,[numpy.max(data_null[:,0])*outlier_multiplier,
                         numpy.max(data_null[:,1])*outlier_multiplier*-1]))
plt.scatter(data[:,0],data[:,1])
slope,intercept,r,p,se=scipy.stats.linregress(data[:,0],data[:,1])
plt.plot([numpy.min(data[:,0]),intercept + slope*numpy.min(data[:,0])],
         [numpy.max(data[:,0]),intercept + slope*numpy.max(data[:,0])])
rsq_outlier=r**2
print('R2 for regression with outlier: %f'%rsq_outlier)

loo=LeaveOneOut()

pred_outlier=numpy.zeros(data.shape[0])

ctr=0
for train,test in loo.split(range(data.shape[0])):
    s,i,_,_,_=scipy.stats.linregress(data[train,0],data[train,1])
    pred_outlier[ctr]=i + s*data[test,0]
    ctr+=1

print('R2 for leave-one-out prediction: %f'%get_R2(pred_outlier,data[:,1]))


# ### Model selection
# 
# Often when we are fitting models to data we have to make decisions about the complexity of the model; after all, if the model has as many parameters as there are data points then we can fit the data exactly, but as we saw above, this model will not generalize very well to other datasets.
# 
# 
# To see how we can use cross-validation to select our model complexity, let's generate some data with a certain polynomial order, and see whether crossvalidation can find the right model order.  
# 

# from https://gist.github.com/iizukak/1287876
def gram_schmidt_columns(X):
    Q, R = numpy.linalg.qr(X)
    return Q

def make_continuous_data_poly(mean=0,var=1,betaval=5,order=1,N=100):
    """
    generate a synthetic data set with two variables
    allowing polynomial functions up to 5-th order
    """
    x=numpy.random.randn(N)
    x=x-numpy.mean(x)
    pf=PolynomialFeatures(5,include_bias=False)

    x_poly=gram_schmidt_columns(pf.fit_transform(x[:,numpy.newaxis]))

    betas=numpy.zeros(5)
    betas[0]=mean
    for i in range(order):
        betas[i]=betaval
    func=x_poly.dot(betas)+numpy.random.randn(N)*var
    d=numpy.vstack((x,func)).T
    return d,x_poly

n=25
trueorder=2
data,x_poly=make_continuous_data_poly(N=n,order=trueorder)

# fit models of increasing complexity
npolyorders=7

plt.figure()
plt.scatter(data[:,0],data[:,1])
plt.title('fitted data')

xp=numpy.linspace(numpy.min(data[:,0]),numpy.max(data[:,0]),100)

for i in range(npolyorders):
    f = numpy.polyfit(data[:,0], data[:,1], i)
    p=numpy.poly1d(f)
    plt.plot(xp,p(xp))
plt.legend(['%d'%i for i in range(npolyorders)])

# compute in-sample and out-of-sample error using LOO
loo=LeaveOneOut()
pred=numpy.zeros((n,npolyorders))
mean_trainerr=numpy.zeros(npolyorders)
prederr=numpy.zeros(npolyorders)

for i in range(npolyorders):
    ctr=0
    trainerr=numpy.zeros(n)
    for train,test in loo.split(range(data.shape[0])):
        f = numpy.polyfit(data[train,0], data[train,1], i)
        p=numpy.poly1d(f)
        trainerr[ctr]=numpy.sqrt(numpy.mean((data[train,1]-p(data[train,0]))**2))
        pred[test,i]=p(data[test,0])
        ctr+=1
    mean_trainerr[i]=numpy.mean(trainerr)
    prederr[i]=numpy.sqrt(numpy.mean((data[:,1]-pred[:,i])**2))
    


plt.plot(range(npolyorders),mean_trainerr)
plt.plot(range(npolyorders),prederr,color='red')
plt.xlabel('Polynomial order')
plt.ylabel('root mean squared error')
plt.legend(['training error','test error'],loc=9)
plt.plot([numpy.argmin(prederr),numpy.argmin(prederr)],
         [numpy.min(mean_trainerr),numpy.max(prederr)],'k--')
plt.text(0.5,numpy.max(mean_trainerr),'underfitting')
plt.text(4.5,numpy.max(mean_trainerr),'overfitting')

print('True order:',trueorder)
print('Order estimated by cross validation:',numpy.argmin(prederr))


# ### Bias-variance tradeoffs
# 
# Another way to think about model complexity is in terms of bias-variance tradeoffs.  Bias is the average distance between the prediction of our model and the correct value, whereas variance is the average distance between different predictions from the model.  In standard statistics classes it is often taken as a given that an unbiased estimate is always best, but within machine learning we will often see that a bit of bias can go a long way towards reducing variance, and that some kinds of bias make particular sense.
# 
# Let's start with an example using linear regression.  First, we will generate a dataset with 20 variables and 100 observations, but only two of the variables are actually related to the outcome (the rest are simply random noise).  
# 

def make_larger_dataset(beta,n,sd=1):
    X=numpy.random.randn(n,len(beta)) # design matrix
    beta=numpy.array(beta)
    y=X.dot(beta)+numpy.random.randn(n)*sd
    return(y-numpy.mean(y),X)
    


# Now let's fit two different models to the data that we will generate.  First, we will fit a standard linear regression model, using ordinary least squares.  This is the best linear unbiased estimator for the regression model.  We will also fit a model that uses *regularization*, which places some constraints on the parameter estimates. In this case, we use the Lasso model, which minimizes the sum of squares while also constraining (or *penalizing*) the sum of the absolute parameter estimates (known as an L1 penalty).  The parameter estimates of this model will be biased towards zero, and will be *sparse*, meaning that most of the estimates will be exactly zero.
# 
# One complication of the Lasso model is that we need to select a value for the alpha parameter, which determines how much penalty there will be.  We will use crossvalidation within the training data set to do this; the sklearn LassoCV() function does it for us automatically.  Let's generate a function that can run both standard regression and Lasso regression.
# 

def compare_lr_lasso(n=100,nvars=20,n_splits=8,sd=1):
    beta=numpy.zeros(nvars)
    beta[0]=1
    beta[1]=-1
    y,X=make_larger_dataset(beta,100,sd=1)
    
    kf=KFold(n_splits=n_splits,shuffle=True)
    pred_lr=numpy.zeros(X.shape[0])
    coefs_lr=numpy.zeros((n_splits,X.shape[1]))
    pred_lasso=numpy.zeros(X.shape[0])
    coefs_lasso=numpy.zeros((n_splits,X.shape[1]))
    lr=LinearRegression()
    lasso=LassoCV()
    ctr=0
    for train,test in kf.split(X):
        Xtrain=X[train,:]
        Ytrain=y[train]
        lr.fit(Xtrain,Ytrain)
        lasso.fit(Xtrain,Ytrain)
        pred_lr[test]=lr.predict(X[test,:])
        coefs_lr[ctr,:]=lr.coef_
        pred_lasso[test]=lasso.predict(X[test,:])
        coefs_lasso[ctr,:]=lasso.coef_
        ctr+=1
    prederr_lr=numpy.sum((pred_lr-y)**2)
    prederr_lasso=numpy.sum((pred_lasso-y)**2)
    return [prederr_lr,prederr_lasso],numpy.mean(coefs_lr,0),numpy.mean(coefs_lasso,0),beta



# Let's run the simulation 100 times and look at the average parameter estimates.
# 

nsims=100
prederr=numpy.zeros((nsims,2))
lrcoef=numpy.zeros((nsims,20))
lassocoef=numpy.zeros((nsims,20))

for i in range(nsims):
    prederr[i,:],lrcoef[i,:],lassocoef[i,:],beta=compare_lr_lasso()
    
print('mean sum of squared error:')
print('linear regression:',numpy.mean(prederr,0)[0])
print('lasso:',numpy.mean(prederr,0)[1])


# The prediction error for the Lasso model is substantially less than the error for the linear regression model.  What about the parameters? Let's display the mean parameter estimates and their variabilty across runs.
# 

coefs_df=pandas.DataFrame({'True value':beta,'Regression (mean)':numpy.mean(lrcoef,0),'Lasso (mean)':numpy.mean(lassocoef,0),
                          'Regression(stdev)':numpy.std(lrcoef,0),'Lasso(stdev)':numpy.std(lassocoef,0)})
coefs_df


# Another place where regularization is essential is when your data are wider than they are tall - that is, when you have more variables than observations. This is almost always the case for brain imaging data, when the number of voxels far outweighs the number of subjects or events.  In this case, the ordinary least squares solution is ill-posed, meaning that it has an infinite number of possible solutions.  The sklearn LinearRegression() estimator will return an estimate even in this case, but the parameter estimates will be highly variable.  However, we can use a regularized regression technique to find more robust estimates in this case.  
# 
# Let's run the same simulation, but now put 1000 variables instead of 20. This will take a few minutes to execute.  
# 

nsims=100
prederr=numpy.zeros((nsims,2))
lrcoef=numpy.zeros((nsims,1000))
lassocoef=numpy.zeros((nsims,1000))

for i in range(nsims):
    prederr[i,:],lrcoef[i,:],lassocoef[i,:],beta=compare_lr_lasso(nvars=1000)
    
print('mean sum of squared error:')
print('linear regression:',numpy.mean(prederr,0)[0])
print('lasso:',numpy.mean(prederr,0)[1])





# In this notebook we will first implement a simplified version of the DCM model, in order to generate data for subsequent examples.
# 

import numpy
# use a consistent seed so that everyone has the same data
numpy.random.seed(1000)

import os,sys
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sys.path.insert(0,'../utils')
from mkdesign import create_design_singlecondition
from graph_utils import show_graph_from_adjmtx
import math
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
import scipy.interpolate

results_dir = os.path.abspath("../results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# first let's build the model without the bilinear influence (aka PPI)
# after http://spm.martinpyka.de/?p=81
nregions=5
z=numpy.zeros(nregions)

# intrinsic connectivity
A=numpy.zeros((z.shape[0],z.shape[0]))
A=numpy.diag(numpy.ones(z.shape[0])*-1)
# add some structure
#A=A + numpy.diag(numpy.ones(z.shape[0]-1),k=-1)
A[2,1]=1
A[3,1]=1
B=numpy.zeros(A.shape)
B[2,0]=1
B[4,0]=1

C=numpy.zeros((z.shape[0],1))
C[0]=1
u=0

print (A)
print (B)
print (C)

# we are assuming a 1 second TR for the resulting data
# but the neural data are at a 1/16 millisecond time resolution
stepsize=.01
tslength=300
timepoints=numpy.arange(0,tslength,stepsize)

# create a blocked design
d,design=create_design_singlecondition(blockiness=1.0,deslength=tslength,blocklength=20,offset=20)

u=scipy.interpolate.griddata(numpy.arange(1,d.shape[0]),d,timepoints,fill_value=0)

def dcm_model(t,z,A,B,C,u):
    ut=numpy.abs(timepoints - t).argmin() 
    return (A.dot(z)+ u[ut]*B.dot(z) + C.dot(u[ut]).T)[0] 

def mk_dcm_dataset(timepoints,z,noise_sd):
    data=numpy.zeros((len(timepoints),len(z)))
    for i in range(1,len(timepoints)):
        data[i,:]=data[i-1,:] + dcm_model(timepoints[i],data[i-1,:],A,B,C,u)  + numpy.random.randn(len(z))*noise_sd 
    hrf=spm_hrf(stepsize,oversampling=1)
    data_conv=numpy.zeros(data.shape)
    for i in range(len(z)):
        data_conv[:,i]=numpy.convolve(data[:,i],hrf)[:data.shape[0]]        
    return data,data_conv    


noise_sd=2
data,data_conv=mk_dcm_dataset(timepoints,z,noise_sd)
numpy.savez(os.path.join(results_dir,'dcmdata.npz'),data=data_conv,A=A,B=B,C=C,u=u,d=d,design=design)


plt.subplot(211)
plt.plot(data_conv)
cc=numpy.corrcoef(data_conv.T)
print ('correlation matrix')
print (cc)
from sklearn.covariance import GraphLassoCV
import matplotlib.colors


glasso=GraphLassoCV()
glasso.fit(data_conv)



from pcor_from_precision import pcor_from_precision
pcor=pcor_from_precision(glasso.precision_)
print ('partial r^2 matrix')
print (pcor**2)

plt.figure(figsize=(10,5))
plt.subplot(141)
plt.imshow(A,interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=-1,vmax=1))
plt.title('A mtx')
plt.subplot(142)
plt.imshow(B,interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=-1,vmax=1))
plt.title('B mtx')
plt.subplot(143)
plt.imshow(cc,interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=-1,vmax=1))
plt.title('correlation')
plt.subplot(144)
plt.imshow(pcor**2,interpolation='nearest',norm=matplotlib.colors.Normalize(vmin=-1,vmax=1))
plt.title('partial correlation')


# ###Show the true graph
# 

gr=show_graph_from_adjmtx(A,B,C)


# ## Run an analysis using SPM
# 
# Now we will run an actual DCM analysis on these data using SPM.
# 
# 1.Start spm using the command: 
# 
#     spm fmri
#     
# 2.Click the "Dynamic causal modelling" button on the main panel
# 
# 3.From the "Action" menu, choose "specify".  
# 
# #### Model 1: True model
# 
# The selection menu should automatically go to the directory where the relevant files are (/home/vagrant/fmri-analysis-vm/analysis/connectivity/dcmfiles).  Assuming that it does, then first select the SPM.mat file listed in the directory, and click "Done" at the bottom.  The SPM.mat file contains a description of the experimental design that DCM needs.
# 
# You will be asked for the name: enter "truemodel"
# 
# Another file selector will appear, which is asking you to select the ROI files.  You should select all 5 of the VOI\* files that are listed in the selector.  They need to be selected in order; the best way to do this is to hold the Shift key and click the last file in the list, which will select the entire set in order. Click "Done" again.  
# 
# You will be asked for a number of details about the design, for which you can just accept the default values.
# 
# Now we have to specify the three matrices in the SPM model (A,B, and C).  You will see a window that asks you to click radio buttons for the active connections.  For the A matrix, we need to include both the static and modulated connections from the A and B matrices above.  Your entry should look like this:
# 

from IPython.display import Image
Image(filename='DCM_Amtx.png',retina=True) 


# Click "done", which will take you to the next window to specify the B and C matrices.  Your entries should look like this:
# 

Image(filename='DCM_BCmtx.png',retina=True) 


# This will create a file called "DCM_truemodel.mat" in the same directory as the other files.
# 
# Now estimate the model by clicking the DCM button, choosing "estimate" from the Action menu, and selecting "DCM_truemodel.mat" from the selector.  This will take a few minutes to compute.
# 
# ####Model 2: no modulation
# 
# Now let's specify the same model, but without any modulated connections.  We can do this using MATLAB code rather than doing it by hand. 
# 
# From the Utils menu, choose "Run M-file".  Using the file selector, go one directory up ('..') and select "mk_model_noppi.m". This will specify the model without modulation (i.e. with no nonzero entries in the B matrix) and then estimate it, saving the results to a file called "DCM_noppi.mat'.
# 
# ####Model 3: incorrect modulation
# 
# Now let's specify the same model, but with two incorrect modulated connections (so that it has the same number of parameters as the true model). Using the "Run M-file" command, select "mk_model_wrongppi.m"
# 
# 

# ## Bayesian model comparison
# 
# A fundamental part of the DCM workflow is the comparison of models.  In this case, we can test for the PPI by comparing the three models that we just created.  
# 
# Choose "Dynamic causal modelling -> Action -> compare"
# 
# This will bring up a batch editor, in which we need to specify several variables.
# 
# 1. Double click on "Directory" and select the current directory (".")
# 2. Click on "Inference method" and choose "Fixed effects" and click "specify"
# 3. Click "Data" then in the window below click "New subject". Then click ". Subject" under Data above, and choose "New Session" below.  Then click on "Models" above, click "Specify", and choose the the three DCM\*.mat files that we created above, in the order truemodel, noppi, wrongppi.
# 4. Click the green arrow at the top to run the batch model.
# 

Image(filename='DCM_BMS.png',retina=True) 


# This tells us that the evidence for the first model (the true model) is much greater than that for the other (incorrect) models.
# 




# This notebook will perform analysis of functional connectivity on simulated data.
# 

import os,sys
import numpy
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sys.path.insert(0,'../utils')
from mkdesign import create_design_singlecondition
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
from make_data import make_continuous_data



data=make_continuous_data(N=200)
print('correlation without activation:',numpy.corrcoef(data.T)[0,1])

plt.plot(range(data.shape[0]),data[:,0],color='blue')
plt.plot(range(data.shape[0]),data[:,1],color='red')



# Now let's add on an activation signal to both voxels
# 

design_ts,design=create_design_singlecondition(blockiness=1.0,offset=30,blocklength=20,deslength=data.shape[0])
regressor,_=compute_regressor(design,'spm',numpy.arange(0,len(design_ts)))

regressor*=50.
data_act=data+numpy.hstack((regressor,regressor))
plt.plot(range(data.shape[0]),data_act[:,0],color='blue')
plt.plot(range(data.shape[0]),data_act[:,1],color='red')
print ('correlation with activation:',numpy.corrcoef(data_act.T)[0,1])


# How can we address this problem? A general solution is to first run a general linear model to remove the task effect and then compute the correlation on the residuals.
# 

X=numpy.vstack((regressor.T,numpy.ones(data.shape[0]))).T
beta_hat=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_act)
y_est=X.dot(beta_hat)
resid=data_act - y_est
print ('correlation of residuals:',numpy.corrcoef(resid.T)[0,1])


# What happens if we get the hemodynamic model wrong?  Let's use the temporal derivative model to generate an HRF that is lagged compared to the canonical.
# 

regressor_td,_=compute_regressor(design,'spm_time',numpy.arange(0,len(design_ts)))
regressor_lagged=regressor_td.dot(numpy.array([1,0.5]))*50


plt.plot(regressor_lagged)
plt.plot(regressor)


data_lagged=data+numpy.vstack((regressor_lagged,regressor_lagged)).T

beta_hat_lag=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data_lagged)
plt.subplot(211)
y_est_lag=X.dot(beta_hat_lag)
plt.plot(y_est)
plt.plot(data_lagged)
resid=data_lagged - y_est_lag
print ('correlation of residuals:',numpy.corrcoef(resid.T)[0,1])
plt.subplot(212)
plt.plot(resid)


# Let's see if using a more flexible basis set, like an FIR model, will allow us to get rid of the task-induced correlation.
# 

regressor_fir,_=compute_regressor(design,'fir',numpy.arange(0,len(design_ts)),fir_delays=range(28))


regressor_fir.shape


X_fir=numpy.vstack((regressor_fir.T,numpy.ones(data.shape[0]))).T
beta_hat_fir=numpy.linalg.inv(X_fir.T.dot(X_fir)).dot(X_fir.T).dot(data_lagged)
plt.subplot(211)
y_est_fir=X_fir.dot(beta_hat_fir)
plt.plot(y_est)
plt.plot(data_lagged)
resid=data_lagged - y_est_fir
print ('correlation of residuals:',numpy.corrcoef(resid.T)[0,1])
plt.subplot(212)
plt.plot(resid)








# In this notebook, we will work through a simulation of psychophysiological interaction
# 

import os,sys
import numpy
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sys.path.insert(0,'../')
from utils.mkdesign import create_design_singlecondition
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
from utils.make_data import make_continuous_data
from statsmodels.tsa.arima_process import arma_generate_sample
import scipy.stats
import seaborn as sns
sns.set_style("white")

results_dir = os.path.abspath("../results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# Load the data generated using the DCM forward model. In this model, there should be a significant PPI between roi 0 and rois 2 and 4 (see the B matrix in the DCM notebook)
# 

dcmdata=numpy.load(os.path.join(results_dir,'dcmdata.npz'))
data_conv=dcmdata['data']
# downsample to 1 second TR
data=data_conv[range(0,data_conv.shape[0],100)]
ntp=data.shape[0]

# create a blocked design
d,design=create_design_singlecondition(blockiness=1.0,deslength=ntp,blocklength=20,offset=20)

regressor,_=compute_regressor(design,'spm',numpy.arange(0,ntp))


for i in range(data.shape[1]):
    plt.plot(data[:,i], label="ROI %d"%i)
plt.legend(loc="best")
plt.xlim([-50,300])


# Set up the PPI model, using ROI 0 as the seed
# 

seed=0
X=numpy.vstack((regressor[:,0],data[:,seed],regressor[:,0]*data[:,seed],numpy.ones(data.shape[0]))).T
hat_mtx=numpy.linalg.inv(X.T.dot(X)).dot(X.T)

for i in range(data.shape[1]):
    beta_hat=hat_mtx.dot(data[:,i])
    resid=data[:,i] - X.dot(beta_hat)
    sigma2hat=(resid.dot(resid))/(X.shape[0] - X.shape[1])
    c=numpy.array([0,0,1,0])  # contrast for PPI
    t=c.dot(beta_hat)/numpy.sqrt(c.dot(numpy.linalg.inv(X.T.dot(X)).dot(c))*sigma2hat)
    print ('ROI %d:'%i, t, 1.0 - scipy.stats.t.cdf(t,X.shape[0] - X.shape[1]))
    
import seaborn as sns
sns.heatmap(X, vmin=0, xticklabels=["task", "seed", "task*seed", "mean"], 
            yticklabels=False)


# Let's plot the relation between the ROIs as a function of the task
# 

on_tp=numpy.where(regressor>0.9)[0]
off_tp=numpy.where(regressor<0.01)[0]
roinum=4

plt.scatter(data[on_tp,0],data[on_tp,roinum], label="task ON")
fit = numpy.polyfit(data[on_tp,0],data[on_tp,roinum],1)
plt.plot(data[on_tp,0],data[on_tp,0]*fit[0] +fit[1])
plt.scatter(data[off_tp,0],data[off_tp,roinum],color='red', label="task OFF")
fit = numpy.polyfit(data[off_tp,0],data[off_tp,roinum],1)
plt.plot(data[off_tp,0],data[off_tp,0]*fit[0] +fit[1],color='red')
plt.xlabel("activation in ROI 0")
plt.ylabel("activation in ROI %d"%roinum)
plt.legend(loc="best")





# In this notebook, we will work through a Bayes Net analysis using the GES algorithm with the TETRAD software (http://www.phil.cmu.edu/tetrad/).  We will use the same dataset used for the PPI and DCM examples.
# 

import os,sys
import numpy
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sys.path.insert(0,'../')
from utils.mkdesign import create_design_singlecondition
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
from utils.make_data import make_continuous_data
from utils.graph_utils import show_graph_from_adjmtx,show_graph_from_pattern
from statsmodels.tsa.arima_process import arma_generate_sample
import scipy.stats
from dcm_sim import sim_dcm_dataset

results_dir = os.path.abspath("../results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# Load the data generated using the DCM forward model. In this model, there is a significant static connectivity from 1->2 and 1->3 (A matrix), and a PPI for 0->2 and 0->4 (B matrix) and a significant input to ROI 0 (C matrix).
# 

_,data_conv,params=sim_dcm_dataset(verbose=True)

A_mtx=params['A']
B_mtx=params['B']
u=params['u']

# downsample design to 1 second TR
u=numpy.convolve(params['u'],spm_hrf(params['stepsize'],oversampling=1))
u=u[range(0,data_conv.shape[0],int(1./params['stepsize']))]
ntp=u.shape[0]




# ###Generate a set of synthetic datasets, referring to individual subjects
# 

tetrad_dir='/home/vagrant/data/tetrad_files'
if not os.path.exists(tetrad_dir):
    os.mkdir(tetrad_dir)

nfiles=10
for i in range(nfiles):
    _,data_conv,params=sim_dcm_dataset()


    # downsample to 1 second TR
    data=data_conv[range(0,data_conv.shape[0],int(1./params['stepsize']))]
    ntp=data.shape[0]

    imagesdata=numpy.hstack((numpy.array(u)[:,numpy.newaxis],data))
    numpy.savetxt(os.path.join(tetrad_dir,"data%03d.txt"%i),
              imagesdata,delimiter='\t',
             header='u\t0\t1\t2\t3\t4',comments='')


# ###Run iMAGES (using a shell script)
# 

get_ipython().system('bash run_images.sh')


# ###Show the graph estimated by iMAGES
# 

g=show_graph_from_pattern('images_test/test.pattern.dot')


# ### Show the true graph from the DCM forward model
# 

show_graph_from_adjmtx(A_mtx,B_mtx,params['C'])





# This notebook demonstrates the basics of Bayesian estimation of the general linear model.  This presentation is based on material from http://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/ .  First let's generate some data for a simple design.
# 

import os,sys
import numpy
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sys.path.insert(0,'../')
from utils.mkdesign import create_design_singlecondition
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
from statsmodels.tsa.arima_process import arma_generate_sample
import scipy.stats
import pymc3

tslength=300
d,design=create_design_singlecondition(blockiness=1.0,deslength=tslength,
                                       blocklength=20,offset=20)
regressor,_=compute_regressor(design,'spm',numpy.arange(0,tslength))


ar1_noise=arma_generate_sample([1,0.3],[1,0.],len(regressor))

X=numpy.hstack((regressor,numpy.ones((len(regressor),1))))
beta=numpy.array([4,100])
noise_sd=10
data = X.dot(beta) + ar1_noise*noise_sd


# First estimate the model using ordinary least squares
# 

beta_hat=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
resid=data - X.dot(beta_hat)
df=(X.shape[0] - X.shape[1])
mse=resid.dot(resid)
sigma2hat=(mse)/float(df)

xvar=X[:,0].dot(X[:,0])
c=numpy.array([1,0])  # contrast for PPI
t=c.dot(beta_hat)/numpy.sqrt(c.dot(numpy.linalg.inv(X.T.dot(X)).dot(c))*sigma2hat)
print ('betas [slope,intercept]:',beta_hat)
print ('t [for slope vs. zero]=',t, 'p=',1.0 - scipy.stats.t.cdf(t,X.shape[0] - X.shape[1]))



# Compute the frequentist 95% confidence intervals
# 

confs = [[beta_hat[0] - scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/xvar), 
         beta_hat[0] + scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/xvar)],
         [beta_hat[1] - scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/X.shape[0]), 
         beta_hat[1] + scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/X.shape[0])]]

print ('slope:',confs[0])
print ('intercept:',confs[1])


# Now let's estimate the same model using Bayesian estimation.  First we use the analytic framework described in the previous notebook.  
# 

prior_sd=10
v=numpy.identity(2)*(prior_sd**2)

beta_hat_bayes=numpy.linalg.inv(X.T.dot(X) + (sigma2hat/(prior_sd**2))*numpy.identity(2)).dot(X.T.dot(data))
print ('betas [slope,intercept]:',beta_hat_bayes)


# Now let's estimate it using Markov Chain Monte Carlo (MCMC) using the No U-turn Sampler (NUTS) (http://www.stat.columbia.edu/~gelman/research/unpublished/nuts.pdf) as implemented in PyMC3.
# 

with pymc3.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = pymc3.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pymc3.Normal('Intercept', 0, sd=prior_sd)
    x_coeff = pymc3.Normal('x', 0, sd=prior_sd)
    
    # Define likelihood
    likelihood = pymc3.Normal('y', mu=intercept + x_coeff * X[:,0], 
                        sd=sigma, observed=data)
    
    # Inference!
    start = pymc3.find_MAP() # Find starting value by optimization
    step = pymc3.NUTS(scaling=start) # Instantiate MCMC sampling algorithm
    trace = pymc3.sample(4000, step, start=start, progressbar=False) # draw 2000 posterior samples using NUTS sampling


# The starting point is the maximum a posteriori (MAP) estimate, which is the same as the one we just computed above.
# 

print(start)


# Now let's look at the results from the MCMC analysis for the slope parameter.  Note that we discard the first 100 steps of the MCMC trace in order to "burn in" the chain (http://stats.stackexchange.com/questions/88819/mcmc-methods-burning-samples).  
# 

plt.figure(figsize=(7, 7))
pymc3.traceplot(trace[100::5],'x')
plt.tight_layout();
pymc3.autocorrplot(trace[100::5])


# Let's look at a summary of the estimates.  How does the 95% highest probability density (HPD) region from the Bayesian analysis compare to the frequentist 95% confidence intervals?
# 

pymc3.summary(trace[100:])





# In this notebook we will work through a representational similarity analysis of the Haxby dataset.
# 

import numpy
import nibabel
import os
from haxby_data import HaxbyData
from nilearn.input_data import NiftiMasker
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import sklearn.manifold
import scipy.cluster.hierarchy

datadir='/home/vagrant/nilearn_data/haxby2001/subj2'

print('Using data from %s'%datadir)

haxbydata=HaxbyData(datadir)

modeldir=os.path.join(datadir,'blockmodel')
try:
    os.chdir(modeldir)
except:
    print('problem changing to %s'%modeldir)
    print('you may need to run the Classification Analysis script first')
    


use_whole_brain=False

if use_whole_brain:
    maskimg=haxbydata.brainmaskfile
else:
    maskimg=haxbydata.vtmaskfile
    
nifti_masker = NiftiMasker(mask_img=maskimg, standardize=False)
fmri_masked = nifti_masker.fit_transform(os.path.join(modeldir,'zstatdata.nii.gz'))


# Let's ask the following question: Are cats (condition 3) more similar to human faces (condition 2) than to chairs (condition 8)?  To do this, we compute the between-run similarity for all conditions against each other.
# 

print(ci,cj,i,j)


cc=numpy.zeros((8,8,12,12))

# loop through conditions
for ci in range(8):
    for cj in range(8):
        for i in range(12):
            for j in range(12):
                if i==6 or j==6:  # problem with run 6 - skip it
                    continue
                idx=numpy.where(numpy.logical_and(haxbydata.runs==i,haxbydata.condnums==ci+1))
                if len(idx[0])>0:
                    idx_i=idx[0][0]
                else:
                    print('problem',ci,cj,i,j)
                    idx_i=None
                idx=numpy.where(numpy.logical_and(haxbydata.runs==j,haxbydata.condnums==cj+1))
                if len(idx[0])>0:
                    idx_j=idx[0][0]
                else:
                    print('problem',ci,cj,i,j)
                    idx_j=None
                if not idx_i is None and not idx_j is None:
                    cc[ci,cj,i,j]=numpy.corrcoef(fmri_masked[idx_i,:],fmri_masked[idx_j,:])[0,1]
                else:
                    cc[ci,cj,i,j]=numpy.nan
meansim=numpy.zeros((8,8))
for ci in range(8):
    for cj in range(8):
        cci=cc[ci,cj,:,:]
        meansim[ci,cj]=numpy.nanmean(numpy.hstack((cci[numpy.triu_indices(12,1)],
                                            cci[numpy.tril_indices(12,1)])))


plt.imshow(meansim,interpolation='nearest')
plt.colorbar()


l=scipy.cluster.hierarchy.ward(1.0 - meansim)


cl=scipy.cluster.hierarchy.dendrogram(l,labels=haxbydata.condlabels,orientation='right')


# Let's test whether similarity is higher for faces across runs within-condition versus similarity between faces and all other categories. Note that we would generally want to compute this for each subject and do statistics on the means across subjects, rather than computing the statistics within-subject as we do below (which treats subject as a fixed effect)
# 

# within-condition

face_corr={}
corr_means=[]
corr_stderr=[]
corr_stimtype=[]
for k in haxbydata.cond_dict.keys():
    face_corr[k]=[]
    for i in range(12):
        for j in range(12):
            if i==6 or j==6:
                continue
            if i==j:
                continue
            face_corr[k].append(cc[haxbydata.cond_dict['face']-1,haxbydata.cond_dict[k]-1,i,j])

    corr_means.append(numpy.mean(face_corr[k]))
    corr_stderr.append(numpy.std(face_corr[k])/numpy.sqrt(len(face_corr[k])))
    corr_stimtype.append(k)


idx=numpy.argsort(corr_means)[::-1]
plt.bar(numpy.arange(0.5,8.),[corr_means[i] for i in idx],yerr=[corr_stderr[i] for i in idx]) #,yerr=corr_sterr[idx])
t=plt.xticks(numpy.arange(1,9), [corr_stimtype[i] for i in idx],rotation=70)
plt.ylabel('Mean between-run correlation with faces')


import sklearn.manifold
mds=sklearn.manifold.MDS()
#mds=sklearn.manifold.TSNE(early_exaggeration=10,perplexity=70,learning_rate=100,n_iter=5000)
encoding=mds.fit_transform(fmri_masked)


plt.figure(figsize=(12,12))
ax=plt.axes() #[numpy.min(encoding[0]),numpy.max(encoding[0]),numpy.min(encoding[1]),numpy.max(encoding[1])])
ax.scatter(encoding[:,0],encoding[:,1])
offset=0.01
for i in range(encoding.shape[0]):
    ax.annotate(haxbydata.conditions[i].split('-')[0],(encoding[i,0],encoding[i,1]),xytext=[encoding[i,0]+offset,encoding[i,1]+offset])
#for i in range(encoding.shape[0]):
#    plt.text(encoding[i,0],encoding[i,1],'%d'%haxbydata.condnums[i])


mdsmeans=numpy.zeros((2,8))
for i in range(8):
    mdsmeans[:,i]=numpy.mean(encoding[haxbydata.condnums==(i+1),:],0)


for i in range(2):
    print('Dimension %d:'%int(i+1))
    idx=numpy.argsort(mdsmeans[i,:])
    for j in idx:
        print('%s:\t%f'%(haxbydata.condlabels[j],mdsmeans[i,j]))
    print('')









# This notebook generates random synthetic fMRI data and a random behavioral regressor, and performs a standard univariate analysis to find correlations between the two.  It is meant to demonstrate how easy it is to find seemingly impressive correlations with fMRI data when multiple tests are not properly controlled for.  
# 
# In order to run this code, you must first install the standard Scientific Python stack (e.g. using [anaconda](https://www.continuum.io/downloads)) along with following additional dependencies:
# * [nibabel](http://nipy.org/nibabel/)
# * [nilearn](http://nilearn.github.io)
# * [statsmodels](http://statsmodels.sourceforge.net)
# * [nipype](http://nipype.readthedocs.io/en/latest/)
# 
# In addition, this notebook assumes that [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) is installed and that the FSLDIR environment variable is defined.
# 

import numpy
import nibabel
import os
import nilearn.plotting
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
import nipype.interfaces.fsl as fsl
import scipy.stats

if not 'FSLDIR' in os.environ.keys():
    raise Exception('This notebook requires that FSL is installed and the FSLDIR environment variable is set')

get_ipython().magic('matplotlib inline')


# Set up default parameters.  We use 28 subjects, which is the median sample size of the set of fMRI studies published in 2015 that were estimated from Neurosynth in the paper.  We use a heuristic correction for multiple comparisons of p<0.001 and 10 voxels, like that show by Eklund et al. (2016, PNAS) to result in Type I error rates of 0.6-0.9.
# 

pthresh=0.001  # cluster forming threshold
cthresh=10     # cluster extent threshold
nsubs=28       # number of subjects


# In order to recreate the figure from the paper exactly, we need to fix the random seed so that it will generate exactly the same random data.  If you wish to generate new data, then set the recreate_paper_figure variable to False and rerun the notebook.
# 

recreate_paper_figure=False
if recreate_paper_figure:
    seed=6636
else:
    seed=numpy.ceil(numpy.random.rand()*100000).astype('int')
    print(seed)

numpy.random.seed(seed)


# Use the standard MNI152 2mm brain mask as the mask for the generated data
# 
# 

maskimg=os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
mask=nibabel.load(maskimg)
maskdata=mask.get_data()
maskvox=numpy.where(maskdata>0)
print('Mask includes %d voxels'%len(maskvox[0]))


# Generate a dataset for each subject.  fMRI data within the mask are generated using a Gaussian distribution (mean=1000, standard deviation=100).  Behavioral data are generated using a Gaussian distribution (mean=100, standard deviation=1).
# 
# 

imgmean=1000    # mean activation within mask
imgstd=100      # standard deviation of noise within mask
behavmean=100   # mean of behavioral regressor
behavstd=1      # standard deviation of behavioral regressor

data=numpy.zeros((maskdata.shape + (nsubs,)))

for i in range(nsubs):
    tmp=numpy.zeros(maskdata.shape)
    tmp[maskvox]=numpy.random.randn(len(maskvox[0]))*imgstd+imgmean
    data[:,:,:,i]=tmp

newimg=nibabel.Nifti1Image(data,mask.get_affine(),mask.get_header())
newimg.to_filename('fakedata.nii.gz')
regressor=numpy.random.randn(nsubs,1)*behavstd+behavmean
numpy.savetxt('regressor.txt',regressor)


# Spatially smooth data using a 6 mm FWHM Gaussian kernel
# 

smoothing_fwhm=6 # FWHM in millimeters

smooth=fsl.IsotropicSmooth(fwhm=smoothing_fwhm,
                           in_file='fakedata.nii.gz',
                           out_file='fakedata_smooth.nii.gz')
smooth.run()


# Use FSL's GLM tool to run a regression at each voxel
# 

glm = fsl.GLM(in_file='fakedata_smooth.nii.gz', 
              design='regressor.txt', 
              out_t_name='regressor_tstat.nii.gz',
             demean=True)
glm.run()


# Use FSL's cluster tool to identify clusters of activation that exceed the specified cluster-forming threshold
# 

tcut=scipy.stats.t.ppf(1-pthresh,nsubs-1)
cl = fsl.Cluster()
cl.inputs.threshold = tcut
cl.inputs.in_file = 'regressor_tstat.nii.gz'
cl.inputs.out_index_file='tstat_cluster_index.nii.gz'
results=cl.run()


# Generate a plot showing the brain-behavior relation from the top cluster
# 

clusterimg=nibabel.load(cl.inputs.out_index_file)
clusterdata=clusterimg.get_data()
indices=numpy.unique(clusterdata)

clustersize=numpy.zeros(len(indices))
clustermean=numpy.zeros((len(indices),nsubs))
indvox={}
for c in range(1,len(indices)):
    indvox[c]=numpy.where(clusterdata==c)    
    clustersize[c]=len(indvox[c][0])
    for i in range(nsubs):
        tmp=data[:,:,:,i]
        clustermean[c,i]=numpy.mean(tmp[indvox[c]])
corr=numpy.corrcoef(regressor.T,clustermean[-1])

print('Found %d clusters exceeding p<%0.3f and %d voxel extent threshold'%(c,pthresh,cthresh))
print('Largest cluster: correlation=%0.3f, extent = %d voxels'%(corr[0,1],len(indvox[c][0])))

# set cluster to show - 0 is the largest, 1 the second largest, and so on
cluster_to_show=0

# translate this variable into the index of indvox
cluster_to_show_idx=len(indices)-cluster_to_show-1

# plot the (circular) relation between fMRI signal and 
# behavioral regressor in the chosen cluster

plt.scatter(regressor.T,clustermean[cluster_to_show_idx])
plt.title('Correlation = %0.3f'%corr[0,1],fontsize=14)
plt.xlabel('Fake behavioral regressor',fontsize=18)
plt.ylabel('Fake fMRI data',fontsize=18)
m, b = numpy.polyfit(regressor[:,0], clustermean[cluster_to_show_idx], 1)
axes = plt.gca()
X_plot = numpy.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-')
plt.savefig('scatter.png',dpi=600)


# Generate a thresholded statistics image for display
# 

tstat=nibabel.load('regressor_tstat.nii.gz').get_data()
thresh_t=clusterdata.copy()
cutoff=numpy.min(numpy.where(clustersize>cthresh))
thresh_t[thresh_t<cutoff]=0
thresh_t=thresh_t*tstat
thresh_t_img=nibabel.Nifti1Image(thresh_t,mask.get_affine(),mask.get_header())


# Generate a figure showing the location of the selected activation focus.
# 

mid=len(indvox[cluster_to_show_idx][0])/2
coords=numpy.array([indvox[cluster_to_show_idx][0][mid],
                    indvox[cluster_to_show_idx][1][mid],
                    indvox[cluster_to_show_idx][2][mid],1]).T
mni=mask.get_qform().dot(coords)
nilearn.plotting.plot_stat_map(thresh_t_img,
        os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz'),
                              threshold=cl.inputs.threshold,
                               cut_coords=mni[:3])
plt.savefig('slices.png',dpi=600)


# This notebook provides examples of analysis of resting fMRI. 
# 
# To grab the data, you should first do the following:
# 
# bash ~/fmri-analysis-vm/get_resting.sh
# 
# This will grab 4 sessions of resting data from ses-105, along with the field maps for this session.  These data have been motion-corrected to a common target so that they are aligned to one another. The motion parameter files are derived separately from each session, rather than using the common target.
# 
# Credit:
# - nilearn examples: https://nilearn.github.io/auto_examples/04_manipulating_images/plot_nifti_simple.html#sphx-glr-auto-examples-04-manipulating-images-plot-nifti-simple-py
# 

import os,glob
import nibabel
import numpy
import sklearn
import nilearn.input_data
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img
from nilearn.plotting import plot_stat_map, show
from nilearn.image import index_img,clean_img
from nilearn.decomposition import CanICA
from sklearn.decomposition import FastICA,PCA

import matplotlib.pyplot as plt
from nipype.interfaces import fsl, nipy
from nipype.caching import Memory
mem = Memory(base_dir='.')

import sys
sys.path.append('/home/vagrant/fmri-analysis-vm/analysis/utils')
from compute_fd_dvars import compute_fd,compute_dvars

get_ipython().magic('matplotlib inline')


# Load the data, clean it, and compute a mask using the nilearn NiftiMasker function
# 

rsfmri_basedir='/home/vagrant/data/ds031/sub-01/ses-105/mcflirt'
rsfmri_files=glob.glob(os.path.join(rsfmri_basedir,'sub*.nii.gz'))

rsfmri_files.sort()

# load the first image and create the masker object
rsfmri_img=nibabel.load(rsfmri_files[0])
masker= nilearn.input_data.NiftiMasker(mask_strategy='epi')
masker.fit(rsfmri_img)
mask_img = masker.mask_img_

rsfmri={}  # nifti handle to cleaned image
fmri_masked=None
# load and clean each image
for f in rsfmri_files:
    rsfmri_img=nibabel.load(f)
    runnum=int(f.split('_')[3].split('-')[1])
    rsfmri[runnum]=nilearn.image.smooth_img(nilearn.image.clean_img(rsfmri_img),'fast')
    print('loaded run',runnum)
    motparfile=f.replace('nii.gz','par')
    mp=numpy.loadtxt(motparfile)
    if fmri_masked is None:
        fmri_masked=masker.transform(rsfmri[runnum])
        motpars=mp
    else:
        fmri_masked=numpy.vstack((fmri_masked,masker.transform(rsfmri[runnum])))
        motpars=numpy.vstack((motpars,mp))


# Visualize the mask
# 

# calculate mean image for the background
mean_func_img = '/home/vagrant/data/ds031/sub-01/ses-105/mcflirt/mcflirt_target.nii.gz'

plot_roi(mask_img, mean_func_img, display_mode='y', 
         cut_coords=4, title="Mask")


# Compute framewise displacement and plot it.
# 

fd=compute_fd(motpars)
numpy.where(fd>1)
plt.figure(figsize=(12,4))
# remove first timepoint from each session
fd[240]=0
fd[480]=0
fd[720]=0
plt.plot(fd)
for c in [240,480,720]:
    plt.plot([c,c],
             [0,numpy.max(fd)*1.1],'k--')


# Run FastICA and PCA on the masked data and compare the components.
# 

ica=FastICA(n_components=5,max_iter=1000)
pca=PCA(n_components=5)
ica.fit(fmri_masked)
pca.fit(fmri_masked)
ica_components_img=masker.inverse_transform(ica.components_)
pca_components_img=masker.inverse_transform(pca.components_)


for i in range(5):
    img=index_img(ica_components_img, i)
    plot_stat_map(img, mean_func_img,
              display_mode='z', cut_coords=4, 
              title="ICA Component %d"%i)
        
for i in range(5):
    img=index_img(pca_components_img, i)
    plot_stat_map(img, mean_func_img,
              display_mode='z', cut_coords=4, 
              title="PCA Component %d"%i)


# Run ICA on the masked data, using the CanICA tool from nilearn.  Just arbitrarily set 10 components for now.
# 

n_components=10

canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)
canica.fit(rsfmri_files)

# Retrieve the independent components in brain space
components_img = canica.masker_.inverse_transform(canica.components_)


# Visualize results
for i in range(n_components):
    img=index_img(components_img, i)
    img_masked=masker.transform(img)
    ts=fmri_masked.dot(img_masked.T)
    plot_stat_map(img, mean_func_img,
              display_mode='z', cut_coords=4, threshold=0.01,
              title="Component %d"%i)
    plt.figure(figsize=(8,3))
    plt.plot(ts)
    for c in [240,480,720]:
        plt.plot([c,c],
                 [numpy.min(ts)*1.1,
                  numpy.max(ts)*1.1],
                'k--')
        





# This notebook provides examples of analysis of parcellated resting fMRI timeseries.
# 
# To grab the data, you should first do the following:
# 
# bash ~/fmri-analysis-vm/get_resting.sh
# 
# This will grab 6 sessions of parcellated resting timeseries, along with some information about the parcellation.
# 
# you will also need to install the python brain connectivity toolbox:
# 
# pip install git+git://github.com/aestrivex/bctpy
# 
# 

import os,glob
import nibabel
import numpy,pandas
import sklearn
import nilearn.input_data
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img
import bct
import scipy.stats
from collections import Counter

import matplotlib.pyplot as plt
import sys
sys.path.append('/home/vagrant/fmri-analysis-vm/analysis/utils')
from r2z import r_to_z,z_to_r

get_ipython().magic('matplotlib inline')


roidatadir='/home/vagrant/data/ds031/sub-01/roidata'
roifiles=glob.glob(os.path.join(roidatadir,'sub*txt'))
roifiles.sort()

columns=['roinum','hemis','X','Y','Z',
          'lobe','region','powernet','yeo7net','yeo17net']
parceldata=pandas.DataFrame.from_csv(os.path.join(roidatadir,'parcel_data.txt'),
                                     sep='\t',header=None,index_col=None)

parceldata.columns=columns

data={}

for f in roifiles:
    subcode=os.path.basename(f).split('.')[0]
    data[subcode]=numpy.loadtxt(f)
    datasize=data[subcode].shape

cc=numpy.zeros((len(roifiles),datasize[1],datasize[1]))
subcodes=list(data.keys())
meancc=numpy.zeros((datasize[1],datasize[1]))
for i,k in enumerate(subcodes):
    cc[i,:,:]=numpy.corrcoef(data[k].T)
    meancc+=r_to_z(cc[i,:,:])
meancc=z_to_r(meancc/len(subcodes))


import scipy.cluster.hierarchy as sch
import pylab

idx=parceldata.powernet.argsort().values
powernet=parceldata.powernet.copy()
powernet.values.tolist()
powernet_sorted=[powernet[i] for i in idx]
breakpoints=[i for i in range(1,len(powernet_sorted)) if powernet_sorted[i]!=powernet_sorted[i-1]]
meancc_reorder=meancc[idx,:]
meancc_reorder=meancc_reorder[:,idx]
plt.imshow(meancc_reorder)

for i,b in enumerate(breakpoints):
    if i==0:
        pos=b/2
    else:
        pos=(b+breakpoints[i-1])/2
    plt.text(700,pos,powernet_sorted[b])


density=10  # percentage of highest edges to keep
minsize=5
thr=scipy.stats.scoreatpercentile(meancc[numpy.triu_indices(meancc.shape[0],1)],100-density)
meancc_thresh=meancc>thr
communities,modularity=bct.community_louvain(meancc_thresh,gamma=2)
nc=[]
for c in numpy.unique(communities):
    count=numpy.sum(communities==c)
    nc.append(count)
    if count<minsize:
        communities[communities==c]=-1
        continue
    powernets_community=[parceldata.powernet.ix[i] for i in range(len(communities)) if communities[i]==c]
    print('community',c)
    pcount=Counter(powernets_community).most_common(4)
    for p in pcount:
        print([p[0],p[1]])
    print()





# This notebook provides an introduction to the basic ideas of graph theory.
# 
# Before running this you will  need to install the python brain connectivity toolbox:
# 
# pip install git+git://github.com/aestrivex/bctpy
# 

import sys,os
from collections import Counter

import numpy,pandas
import scipy.stats
import networkx
import nilearn.plotting
import matplotlib.pyplot as plt
sys.path.append('/home/vagrant/fmri-analysis-vm/analysis/utils')
from community.community_louvain import modularity,best_partition
import bct

get_ipython().magic('matplotlib inline')


# First let's generate a network by specifying it as a matrix.  We will start with a simple binary network with 5 nodes, where every node is either connected (1) or not (0). 
# 

adj=numpy.zeros((7,7))  # empty adjacency matrix
# now specify each 
edges=[[0,1],[0,2],[1,2],[0,3],[3,4],[3,5],[4,5],[3,6],[4,6],[5,6]]
for e in edges:
    adj[e[0],e[1]]=1
adj=adj+adj.T
print(adj)
plt.imshow(adj,interpolation='nearest',cmap='gray')


# Now we take this matrix and create a graph using networkx.
# 

G=networkx.from_numpy_matrix(adj).to_undirected()
print('nodes:',G.nodes())
print('edges:',G.edges())


# Visualize the network using a spring-embedded display.
# 

comm=best_partition(G)
membership=numpy.array([comm[i] for i in range(7)])
bc=networkx.betweenness_centrality(G)
participation=bct.participation_coef(adj,membership)
centrality=numpy.array([bc[i] for i in range(7)])
colorvals=['r','g']
networkx.draw_spring(G,with_labels=True,
                     node_color=[colorvals[i] for i in membership],
                    node_size=centrality*300+300)


# Print out some graph statistics about the nodes, and perform community detection using the Louvain algorithm.
# 

print('modularity=',modularity(comm,G))
nodeinfo=pandas.DataFrame({'degree':G.degree(),
                           'clustering':networkx.clustering(G)})
nodeinfo['partition']=membership
nodeinfo['betweeness_centrality']=centrality
nodeinfo['participation']=participation
print(nodeinfo)


# Now let's look at a more complex graph.  We will read in the adjacency matrix from the myconnectome data, threshold it at a particular density, binarize it, and then display it and compute its modularity.
# 

columns=['roinum','hemis','X','Y','Z',
          'lobe','region','powernet','yeo7net','yeo17net']
parceldata=pandas.DataFrame.from_csv('ds031_connectome/parcel_data.txt',
                                     sep='\t',header=None,index_col=None)

parceldata.columns=columns

cc_utr=numpy.load('ds031_connectome/meancorr_utr.npy')
cc=numpy.zeros((630,630))
cc[numpy.triu_indices(630,1)]=cc_utr
cc=cc+cc.T  # make it symmetric
density=0.1 # density of the graph
threshold=scipy.stats.scoreatpercentile(cc_utr,100-density*100)
cc_adj=(cc>threshold).astype('int')
G=networkx.from_numpy_matrix(cc_adj)
# add network labels
powernet_dict={}
yeo7_dict={}
coord_dict={}
for n in G.nodes():
    powernet_dict[n]=parceldata.powernet.ix[n]
    yeo7_dict[n]=parceldata.yeo7net.ix[n]
    coord_dict[n]=[parceldata.X.ix[n],parceldata.Y.ix[n],parceldata.Z.ix[n]]
networkx.set_node_attributes(G,'powernet',powernet_dict)
networkx.set_node_attributes(G,'yeo7',yeo7_dict)
networkx.set_node_attributes(G,'coords',coord_dict)

# get the giant component
Gcc=sorted(networkx.connected_component_subgraphs(G), key = len, reverse=True)
G0=Gcc[0]
adj_G0=numpy.array(networkx.to_numpy_matrix(G0))
pos=networkx.spring_layout(G0)
partition=best_partition(G0,resolution=0.5)
membership=numpy.array([partition[i] for i in G0.nodes()])
m=modularity(partition,G0)
print('modularity:',m)
eff=bct.efficiency_bin(cc_adj)
print('efficiency:',eff)
bc=networkx.betweenness_centrality(G)
centrality=numpy.array([bc[i] for i in range(len(G.nodes()))])
participation=bct.participation_coef(adj_G0,membership)


# figure out which networks are contained in each community

nc=[]
minsize=5
communities=numpy.array(list(partition.values()))
powernets=[G.node[i]['powernet'] for i in G.nodes()]
yeo7nets=[G.node[i]['yeo7'] for i in G.nodes()]
for c in numpy.unique(communities):
    count=numpy.sum(communities==c)
    nc.append(count)
    if count<minsize:
        communities[communities==c]=-1
        continue
    powernets_community=[powernets[i] for i in range(len(communities)) if communities[i]==c]
    print('community',c)
    pcount=Counter(powernets_community).most_common(4)
    for p in pcount:
        print([p[0],p[1]])
    print()


plt.figure(figsize=(10,10))
color_by='community'
if color_by=='community':
    colors=membership
elif color_by=='yeo7':
    colors=[]
    for i in G0.nodes():
        try:
            colors.append(int(G0.node[i]['yeo7'].split('_')[1]) )
        except:
            colors.append(0)
networkx.draw_networkx(G0,pos=pos,width=0.05,cmap=plt.get_cmap('gist_ncar'),
              node_color=colors, vmin=numpy.min(colors),vmax=numpy.max(colors),
                      with_labels=False,linewidths=0,
                      node_size=2000*centrality+5)

for i,c in enumerate(numpy.unique(colors)):
    y=0.85-i*0.05
    plt.plot([-0.15,-0.1],[y,y])
    plt.text(-0.08,y,'%d'%c)
plt.axis([-0.2,1.2,-0.1,0.9])


# We can also use the nilearn plotting tools to plot one of the networks.
# 

subnet=2
subnet_nodes=[i for i in range(len(colors)) if colors[i]==subnet]
sg=G.subgraph(subnet_nodes)
sg_adjmtx=networkx.to_numpy_matrix(sg).astype('float')
# this is a kludge - for some reason nilearn can't handle
# the matrix that networkx generates
adjmtx=numpy.zeros((sg_adjmtx.shape[0],sg_adjmtx.shape[0]))
for i in range(sg_adjmtx.shape[0]):
    for j in range(sg_adjmtx.shape[0]):
        adjmtx[i,j]=sg_adjmtx[i,j]
node_coords=numpy.array([sg.node[i]['coords'] for i in sg.nodes()])
nilearn.plotting.plot_connectome(adjmtx,node_coords,
                                edge_threshold='75%',edge_kwargs={'linewidth':1})


# Let's look at the same network as a matrix, sorting by the Power et al. networks assigned in the original analysis of the data.
# 

fig=plt.figure(figsize=(10,10))
idx=parceldata.powernet.argsort().values
powernet=parceldata.powernet.copy()
powernet.values.tolist()
powernet_sorted=[powernet[i] for i in idx]
breakpoints=[i for i in range(1,len(powernet_sorted)) if powernet_sorted[i]!=powernet_sorted[i-1]]
meancc_reorder=cc[idx,:]
meancc_reorder=meancc_reorder[:,idx]
plt.imshow(meancc_reorder,origin='upper')

for i,b in enumerate(breakpoints):
    if i==0:
        pos=b/2
    else:
        pos=(b+breakpoints[i-1])/2
    plt.text(700,pos,powernet_sorted[b])
    plt.plot([0,630],[b,b],'k',linewidth=0.5)
    plt.plot([b,b],[0,630],'k',linewidth=0.5)
plt.colorbar(orientation='horizontal',shrink=0.5)
plt.axis([0,630,630,0])


# Now let's randomize the connections and look at the graph.
# 

cc_rand=numpy.zeros((630,630))
cc_utr_rand=cc_utr.copy()
numpy.random.shuffle(cc_utr_rand)
cc_rand[numpy.triu_indices(630,1)]=cc_utr_rand
cc_rand=cc_rand+cc_rand.T  # make it symmetric
cc_rand_adj=(cc_rand>threshold).astype('int')
G_rand=networkx.from_numpy_matrix(cc_rand_adj)
# get the giant component
Gcc_rand=sorted(networkx.connected_component_subgraphs(G_rand), key = len, reverse=True)
G0_rand=Gcc_rand[0]
pos_rand=networkx.spring_layout(G0_rand)
partition=best_partition(G0_rand)
m_rand=modularity(partition,G0_rand)
print('modularity:',m_rand)
eff_rand=bct.efficiency_bin(cc_rand_adj)
print('efficiency:',eff_rand)
bc_rand=networkx.betweenness_centrality(G_rand)
centrality_rand=numpy.array([bc_rand[i] for i in range(len(G_rand.nodes()))])


plt.figure(figsize=(12,12))
colors=numpy.array([partition[i] for i in G0_rand.nodes()])
networkx.draw_networkx(G0_rand,pos=pos_rand,width=0.05,cmap=plt.get_cmap('gist_ncar'),
              node_color=colors, vmin=numpy.min(colors),vmax=numpy.max(colors),
                      with_labels=False,linewidths=0,
                      node_size=2000*centrality_rand)


# Plot the degree distributions for the origin and randomized networks
# 

degree_hist=numpy.histogram(list(G.degree().values()),50)
degree_hist_rand=numpy.histogram(list(G_rand.degree().values()),50)


plt.plot(degree_hist[1][1:],degree_hist[0])
plt.plot(degree_hist_rand[1][1:],degree_hist_rand[0])
plt.legend(['original','randomized'])
plt.ylabel('frequency')
plt.xlabel('degree')








# This notebook shows how anticorrelations are a reflection of conditioning on a common effect.
# 

import numpy
import matplotlib.pyplot as plt

npts=100
gs=numpy.random.randn(npts)

data=numpy.random.randn(npts,3)
for d in range(data.shape[1]):
    data[:,d]=data[:,d]+gs
data=data-numpy.mean(data,0)

origcor=numpy.corrcoef(data.T)
print(origcor)


meansig=numpy.mean(data,1)
meansig=meansig-numpy.mean(meansig)

r1_gsreg_beta=numpy.linalg.lstsq(meansig[:,numpy.newaxis],data[:,0])
r1_resid=data[:,0] - meansig*r1_gsreg_beta[0]

r2_gsreg_beta=numpy.linalg.lstsq(meansig[:,numpy.newaxis],data[:,1])
r2_resid=data[:,1] - meansig*r2_gsreg_beta[0]

residcor=numpy.corrcoef(r1_resid,r2_resid)[0,1]


get_ipython().magic('matplotlib inline')
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(data[:,0],data[:,1])
plt.title('original data: r = %0.3f'%origcor[0,1])
plt.subplot(1,2,2)  
plt.scatter(r1_resid,r2_resid)
plt.title('residuals from GSR: r = %0.3f'%residcor)


numpy.corrcoef(r1,r2)





# This notebook provides an example of the use of variational Bayesian estimation and inference.  The VB computations here are based on Kay Broderson's MATLAB demo at https://www.tnu.ethz.ch/de/software/tapas.html
# 
# I have contrived a simple example where we compute a one-sample t-test versus zero.
# 

import numpy,scipy
import time
from numpy.linalg import inv
from scipy.special import digamma,gammaln
from numpy import log,pi,trace
from numpy.linalg import det
import matplotlib.pyplot as plt
from pymc3 import Model,glm,find_MAP,NUTS,sample,Metropolis,HalfCauchy,Normal

get_ipython().magic('matplotlib inline')



# Set up code to estimate using VB
# 

# Create classes for prior and posterior

# %     a_0: shape parameter of the prior precision of coefficients
# %     b_0: rate  parameter of the prior precision of coefficients
# %     c_0: shape parameter of the prior noise precision
# %     d_0: rate  parameter of the prior noise precision

class Prior:
    def __init__(self,a_0=10,b_0=0.2,c_0=10,d_0=1):
        self.a_0=a_0
        self.b_0=b_0
        self.c_0=c_0
        self.d_0=d_0

class Posterior:
    def __init__(self,d,prior):
        self.mu_n=numpy.zeros((d,1))
        self.Lambda_n = numpy.eye(d)
        self.a_n      = prior.a_0
        self.b_n      = prior.b_0
        self.c_n      = prior.c_0
        self.d_n      = prior.d_0
        self.F        = -numpy.inf
        self.prior    = prior
        self.trace = []
        
# Returns the variational posterior q that maximizes the free energy.
def invert_model(y, X, prior,tolerance=10e-8,verbose=False):

    # Data shortcuts
    n,d = X.shape # observations x regressors

    # Initialize variational posterior
    q=Posterior(d,prior)
    q.F=free_energy(q,y,X,prior)

    # Variational algorithm
    nMaxIter = 30
    kX = X.T.dot(X)

    for i in range(nMaxIter):

        # (1) Update q(beta) - regression parameters
        q.Lambda_n = q.a_n/q.b_n + q.c_n/q.d_n * (X.T.dot(X))

        q.mu_n = q.c_n/q.d_n * numpy.linalg.inv(q.Lambda_n).dot(X.T.dot(y))

        # (2) Update q(alpha) - precision
        q.a_n = prior.a_0 + d/2
        q.b_n = prior.b_0 + 1/2 * (q.mu_n.T.dot(q.mu_n) + trace(inv(q.Lambda_n)));

        # (3) Update q(lambda)
        q.c_n = prior.c_0 + n/2
        pe = y - X.dot(q.mu_n)
        q.d_n = prior.d_0 + 0.5 * (pe.T.dot(pe) + trace(inv(q.Lambda_n).dot(kX))) ;

        # Compute free energy
        F_old = q.F;
        q.F = free_energy(q,y,X,prior);

        # Convergence?
        if (q.F - F_old) < tolerance:
            break
        if i == nMaxIter:
            print('tvblm: reached max iterations',i)
    if verbose:
        print('converged in %d iterations'%i)
    return q

# Computes the free energy of the model given the data.
def free_energy(q,y,X,prior):
    # Data shortcuts
    n,d = X.shape # observations x regressors

    # Expected log joint <ln p(y,beta,alpha,lambda)>_q
    J =(n/2*(digamma(q.c_n)-log(q.d_n)) - n/2*log(2*pi)
        - 0.5*q.c_n/q.d_n*(y.T.dot(y)) + q.c_n/q.d_n*(q.mu_n.T.dot(X.T.dot(y)))
        - 0.5*q.c_n/q.d_n*trace(X.T.dot(X) * (q.mu_n.dot(q.mu_n.T) + inv(q.Lambda_n)))
        - d/2*log(2*pi) + n/2*(digamma(q.a_n)-log(q.b_n))
        - 0.5*q.a_n/q.b_n * (q.mu_n.T.dot(q.mu_n) + trace(inv(q.Lambda_n)))
        + prior.a_0*log(prior.b_0) - gammaln(prior.a_0)
        + (prior.a_0-1)*(digamma(q.a_n)-log(q.b_n)) - prior.b_0*q.a_n/q.b_n
        + prior.c_0*log(prior.d_0) - gammaln(prior.c_0)
        + (prior.c_0-1)*(digamma(q.c_n)-log(q.d_n)) - prior.d_0*q.c_n/q.d_n)

    # Entropy H[q]
    H = (d/2*(1+log(2*pi)) + 1/2*log(det(inv(q.Lambda_n)))
      + q.a_n - log(q.b_n) + gammaln(q.a_n) + (1-q.a_n)*digamma(q.a_n)
      + q.c_n - log(q.d_n) + gammaln(q.c_n) + (1-q.c_n)*digamma(q.c_n))

    # Free energy
    F = J + H
    return(F)


# Now create synthetic data with a specified mean and perform a one sample t-test using either the t-test function from scipy.stats or using VB estimation.
# 

npts=64
std=1
nruns=1000

# one sample t test
X=numpy.ones((npts,1))
prior=Prior(a_0=10,c_0=10)
means=numpy.arange(0,0.501,0.1)
vb_siglevel=numpy.zeros(len(means))
t_siglevel=numpy.zeros(len(means))
vb_mean=numpy.zeros(len(means))
samp_mean=numpy.zeros(len(means))

t_time=0
vb_time=0

for j,mean in enumerate(means):
    vb_pvals=numpy.zeros(nruns)
    t_pvals=numpy.zeros(nruns)
    vb_means=numpy.zeros(nruns)
    samp_means=numpy.zeros(nruns)
    for i in range(nruns):
        y=numpy.random.randn(npts)*std+mean
        samp_means[i]=numpy.mean(y)
        t=time.time()
        q=invert_model(y, X, prior,verbose=False)
        vb_means[i]=q.mu_n
        # q.Lambda_n is the estimated precision, so we turn it into a standard deviation
        vb_pvals[i]=scipy.stats.norm.cdf(0,q.mu_n,1/numpy.sqrt(q.Lambda_n))
        vb_time+=time.time()-t
        
        t=time.time()
        _,t_pvals[i]=scipy.stats.ttest_1samp(y,0)
        t_time+=time.time()-t

    vb_siglevel[j]=numpy.mean(vb_pvals<0.05)
    t_siglevel[j]=numpy.mean(t_pvals<0.05)
    vb_mean[j]=numpy.mean(vb_means)
    samp_mean[j]=numpy.mean(samp_means)
    
print('Total elapsed time for %d analyses (seconds):'%int(len(means)*nruns))
print('t-test: %0.2f'%t_time)
print('VB: %0.2f'%vb_time)


# Now plot the error/power for the two approaches.  For the t-test, this is the proportion of statistically significant outcomes over the realizations (at p<0.05).  For the VB estimate, this is the proportion of tests for which zero falls below the 5%ile of the estimated posterior.  
# 

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(means,t_siglevel)
plt.plot(means,vb_siglevel)
plt.xlabel('Mean effect (in SD units)')
plt.ylabel('Proportion of exceedences')
plt.legend(['ttest','VB'],loc=2)

plt.subplot(1,2,2)
plt.plot(means,samp_mean)
plt.plot(means,vb_mean)
plt.xlabel('Mean effect (in SD units)')
plt.ylabel('Estimated mean')
plt.legend(['Sample mean','VB estimate'],loc=2)

print('False positive rate:')
print('t-test: %0.4f'%t_siglevel[0])
print('VB: %0.4f'%vb_siglevel[0])


# Play around with the prior values for precision of the coefficients - how do they affect the relative efficiency of VB versus the t-test?




# In the analysis of neuroimaging data using general linear models (GLMs), it is often common to find that regressors of interest
# are correlated with one another.  While this inflates the variance of the estimated parameters, the GLM ensures that the 
# estimated parameters only reflect the unique variance associated with the particular regressor; any shared variance
# between regressors, while accounted for in the total model variance, is not reflected in the individual parameter 
# estimates.  In general, this is as it should be; when it is not possible to uniquely attribute variance to any
# particular regressor, then it should be left out.  
# 
# Unfortunately, there is a tendency within the fMRI literature to overthrow this feature of the GLM by "orthogonalizing"
# variables that are correlated.  This, in effect, assigns the shared variance to one of the correlated variables based 
# on the experimenter's decision.  While statistically valid, this raises serious conceptual concerns about the 
# interpretation of the resulting parameter estimates.
# 
# The first point to make is that, contrary to claims often seen in fMRI papers, the presence of correlated regressors
# does not require the use of orthogonalization; in fact, in our opinion there are very few cases in which it is appropriate
# to use orthogonalization, and its use will most often result in problematic conclusions.
# 
# *What is orthogonalization?*
# 
# As an example of how the GLM deals with correlated regressors and how this is affected by orthogonalization,
# we first generate some synthetic data to work with.
# 

get_ipython().magic('pylab inline')
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

npts=100
X = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],npts)
X = X-np.mean(X,0)  

params  = [1,2]
y_noise = 0.2
Y = np.dot(X,params) + y_noise*np.random.randn(npts)
Y = Y-np.mean(Y)    # remove mean so we can skip ones in design mtx


# Plot the relations between the two columns in X and the Y variable.
# 

for i in range(2):
    print('correlation(X[%d],Y))'%i, '= %4.3f' % np.corrcoef(X[:,i],Y)[0,1])
    plt.subplot(1,2,i+1)
    plt.scatter(X[:,i],Y)


# Now let's compute the parameters for the two columns in X using linear regression.  They should come out very close
# to the values specified for params above.
# 

params_est =  np.linalg.lstsq(X,Y)[0]

print(params_est)


# Now let's orthogonalize the second regressor (X[1]) with respect to the first (X[0]) and create a new orthogonalized 
# design matrix X_orth.  One way to do this is to fit a regression and then take the residuals.
# 

x0_slope=numpy.linalg.lstsq(X[:,0].reshape((npts,1)),X[:,1].reshape((npts,1)))[0]

X_orth=X.copy()

X_orth[:,1]=X[:,1] - X[:,0]*x0_slope
print('Correlation matrix for original design matrix')
print (numpy.corrcoef(X.T))

print ('Correlation matrix for orthogonalized design matrix')
print (numpy.corrcoef(X_orth.T))


# As intended, the correlation between the two regressors is effectively zero after orthogonalization. Now 
# let's estimate the model parameters using the orthogonalized design matrix:
# 

params_est_orth =  numpy.linalg.lstsq(X_orth,Y)[0]

print (params_est_orth)


# Note that the parameter estimate for the orthogonalized regressor is exactly the same as it was in the original model;
# it is only the estimate for the other (orthogonalized-against) regressor that changes after orthogonalization.  That's
# because shared variance between the two regressors has been assigned to X[0], whereas previously it was unassigned.
# 

# Note also that testing the second regressor will yield exactly the same test value. Testing for the first regressor, on the contrary, will yield a much smaller p value as the variance explained by this regressor contains the shared variance of both regressors.  
# 

# More generally, orthogonalizing the two first regressors $X_0$ of the design matrix $X$ will look like:
# 

# Make X nptsx10
X = np.random.normal(0,1,(npts,10))
X = X - X.mean(axis=0)
X0 = X[:,:2]
X1 = X[:,2:]

# Orthogonolizing X0 with respect to X1: 
X0_orthog_wrt_X1 = X0 - np.dot(X1,np.linalg.pinv(X1)).dot(X0)

# reconstruct the new X matrix : Xorth
Xorth = np.hstack((X0_orthog_wrt_X1, X1))

# checking that the covariance of the two first regressors with others is 0
# look at the 5 first regressors
print (np.corrcoef(Xorth.T)[:5,:5])





# This code will load the model information, generate the model definition, and run the model estimation using FSL
# 

import nipype.algorithms.modelgen as model   # model generation
from  nipype.interfaces import fsl, ants      
from nipype.interfaces.base import Bunch
import os,json,glob,sys
import numpy
import nibabel
import nilearn.plotting

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

datadir='/home/vagrant/data/ds000114_R2.0.1/'
    
results_dir = os.path.abspath("../../results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

from nipype.caching import Memory
mem = Memory(base_dir='.')

print('Using data from',datadir)


from bids.grabbids import BIDSLayout
layout = BIDSLayout(datadir)
layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[0].filename


import pandas as pd
events = pd.read_csv(os.path.join(datadir, "task-fingerfootlips_events.tsv"), sep="\t")
events


for trial_type in events.trial_type.unique():
    print(events[events.trial_type == trial_type])


events[events.trial_type == 'Finger'].duration


source_epi = layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[5]

confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_confounds.tsv"%(source_epi.subject,
                                                                                                                           source_epi.session)),
           sep="\t", na_values="n/a")

info = [Bunch(conditions=['Finger',
                          'Foot',
                          'Lips'],
              onsets=[list(events[events.trial_type == 'Finger'].onset-10),
                      list(events[events.trial_type == 'Foot'].onset-10),
                      list(events[events.trial_type == 'Lips'].onset-10)],
              durations=[list(events[events.trial_type == 'Finger'].duration),
                          list(events[events.trial_type == 'Foot'].duration),
                          list(events[events.trial_type == 'Lips'].duration)],
             regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
                         list(confounds.aCompCor0[4:]),
                         list(confounds.aCompCor1[4:]),
                         list(confounds.aCompCor2[4:]),
                         list(confounds.aCompCor3[4:]),
                         list(confounds.aCompCor4[4:]),
                         list(confounds.aCompCor5[4:]),
                        ],
             regressor_names=['FramewiseDisplacement',
                              'aCompCor0',
                              'aCompCor1',
                              'aCompCor2',
                              'aCompCor3',
                              'aCompCor4',
                              'aCompCor5',])
       ]

skip = mem.cache(fsl.ExtractROI)
skip_results = skip(in_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(source_epi.subject,
                                                                                                                           source_epi.session)),
                     t_min=4, t_size=-1)

s = model.SpecifyModel()
s.inputs.input_units = 'secs'
s.inputs.functional_runs = skip_results.outputs.roi_file
s.inputs.time_repetition = layout.get_metadata(source_epi.filename)["RepetitionTime"]
s.inputs.high_pass_filter_cutoff = 128.
s.inputs.subject_info = info
specify_model_results = s.run()
s.inputs


finger_cond = ['Finger','T', ['Finger'],[1]]
foot_cond = ['Foot','T', ['Foot'],[1]]
lips_cond = ['Lips','T', ['Lips'],[1]]
lips_vs_others = ["Lips vs. others",'T', ['Finger', 'Foot', 'Lips'],[-0.5, -0.5, 1]]
all_motor = ["All motor", 'F', [finger_cond, foot_cond, lips_cond]]
contrasts=[finger_cond, foot_cond, lips_cond, lips_vs_others, all_motor]
           
level1design = mem.cache(fsl.model.Level1Design)
level1design_results = level1design(interscan_interval = layout.get_metadata(source_epi.filename)["RepetitionTime"],
                                    bases = {'dgamma':{'derivs': True}},
                                    session_info = specify_model_results.outputs.session_info,
                                    model_serial_correlations=True,
                                    contrasts=contrasts)

level1design_results.outputs


modelgen = mem.cache(fsl.model.FEATModel)
modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                            ev_files=level1design_results.outputs.ev_files)
modelgen_results.outputs


desmtx=numpy.loadtxt(modelgen_results.outputs.design_file,skiprows=5)
plt.imshow(desmtx,aspect='auto',interpolation='nearest',cmap='gray')


cc=numpy.corrcoef(desmtx.T)
plt.imshow(cc,aspect='auto',interpolation='nearest', cmap=plt.cm.viridis)
plt.colorbar()


mask = mem.cache(fsl.maths.ApplyMask)
mask_results = mask(in_file=skip_results.outputs.roi_file,
                    mask_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"%(source_epi.subject,
                                                                                                                             source_epi.session)))
mask_results.outputs


filmgls= mem.cache(fsl.FILMGLS)
filmgls_results = filmgls(in_file=mask_results.outputs.out_file,
                          design_file = modelgen_results.outputs.design_file,
                          tcon_file = modelgen_results.outputs.con_file,
                          fcon_file = modelgen_results.outputs.fcon_file,
                          autocorr_noestimate = True)
filmgls_results.outputs


for t_map in filmgls_results.outputs.zstats:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(t_map, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3)


for t_map in [filmgls_results.outputs.zfstats]:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(t_map, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3)


for t_map in filmgls_results.outputs.copes:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(t_map, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, vmax=30)


for t_map in filmgls_results.outputs.tstats:
    nilearn.plotting.plot_stat_map(nilearn.image.smooth_img(t_map, 8), colorbar=True, threshold=2.3)


# ## Repeat for all subjects
# 

# For the group level analysis we need to move results from all subjects into one common MNI space. Let's start with the EPI derived mask (we will use it later for group level mask)
# 

copes = {}
for i in range(10):
    source_epi = layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[i]

    confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep", 
                                            "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                            "sub-%s_ses-%s_task-fingerfootlips_bold_confounds.tsv"%(source_epi.subject,
                                                                                                                               source_epi.session)),
               sep="\t", na_values="n/a")

    info = [Bunch(conditions=['Finger',
                              'Foot',
                              'Lips'],
                  onsets=[list(events[events.trial_type == 'Finger'].onset-10),
                          list(events[events.trial_type == 'Foot'].onset-10),
                          list(events[events.trial_type == 'Lips'].onset-10)],
                  durations=[list(events[events.trial_type == 'Finger'].duration),
                              list(events[events.trial_type == 'Foot'].duration),
                              list(events[events.trial_type == 'Lips'].duration)],
                 regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
                             list(confounds.aCompCor0[4:]),
                             list(confounds.aCompCor1[4:]),
                             list(confounds.aCompCor2[4:]),
                             list(confounds.aCompCor3[4:]),
                             list(confounds.aCompCor4[4:]),
                             list(confounds.aCompCor5[4:]),
                            ],
                 regressor_names=['FramewiseDisplacement',
                                  'aCompCor0',
                                  'aCompCor1',
                                  'aCompCor2',
                                  'aCompCor3',
                                  'aCompCor4',
                                  'aCompCor5',])
           ]

    skip = mem.cache(fsl.ExtractROI)
    skip_results = skip(in_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                            "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                            "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(source_epi.subject,
                                                                                                                               source_epi.session)),
                         t_min=4, t_size=-1)

    s = model.SpecifyModel()
    s.inputs.input_units = 'secs'
    s.inputs.functional_runs = skip_results.outputs.roi_file
    s.inputs.time_repetition = layout.get_metadata(source_epi.filename)["RepetitionTime"]
    s.inputs.high_pass_filter_cutoff = 128.
    s.inputs.subject_info = info
    specify_model_results = s.run()
    
    finger_cond = ['Finger','T', ['Finger'],[1]]
    foot_cond = ['Foot','T', ['Foot'],[1]]
    lips_cond = ['Lips','T', ['Lips'],[1]]
    lips_vs_others = ["Lips vs. others",'T', ['Finger', 'Foot', 'Lips'],[-0.5, -0.5, 1]]
    all_motor = ["All motor", 'F', [finger_cond, foot_cond, lips_cond]]
    contrasts=[finger_cond, foot_cond, lips_cond, lips_vs_others, all_motor]

    level1design = mem.cache(fsl.model.Level1Design)
    level1design_results = level1design(interscan_interval = layout.get_metadata(source_epi.filename)["RepetitionTime"],
                                        bases = {'dgamma':{'derivs': True}},
                                        session_info = specify_model_results.outputs.session_info,
                                        model_serial_correlations=True,
                                        contrasts=contrasts)
    
    modelgen = mem.cache(fsl.model.FEATModel)
    modelgen_results = modelgen(fsf_file=level1design_results.outputs.fsf_files,
                                ev_files=level1design_results.outputs.ev_files)
    
    mask = mem.cache(fsl.maths.ApplyMask)
    mask_results = mask(in_file=skip_results.outputs.roi_file,
                        mask_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"%(source_epi.subject,
                                                                                                                             source_epi.session)))
    
    filmgls= mem.cache(fsl.FILMGLS)
    filmgls_results = filmgls(in_file=mask_results.outputs.out_file,
                              design_file = modelgen_results.outputs.design_file,
                              tcon_file = modelgen_results.outputs.con_file,
                              fcon_file = modelgen_results.outputs.fcon_file,
                              autocorr_noestimate = True)
                                                                                                                             
    copes[source_epi.subject] = list(filmgls_results.outputs.copes)                                                                                                                         



smooth_copes = []
for k,v in copes.items():
    smooth_cope = nilearn.image.smooth_img(v[3], 8)
    smooth_copes.append(smooth_cope)
    nilearn.plotting.plot_glass_brain(smooth_cope,
                                      display_mode='lyrz', 
                                      colorbar=True, 
                                      plot_abs=False, 
                                      vmax=30)


nilearn.plotting.plot_glass_brain(nilearn.image.mean_img(smooth_copes),
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False)


brainmasks = glob.glob(os.path.join(datadir, "derivatives", "fmriprep", "sub-*", "ses-test", "func", "*task-fingerfootlips_*space-MNI152NLin2009cAsym*_brainmask.nii*"))

for mask in brainmasks:
    nilearn.plotting.plot_roi(mask)
    
mean_mask = nilearn.image.mean_img(brainmasks)
nilearn.plotting.plot_stat_map(mean_mask)
group_mask = nilearn.image.math_img("a>=0.95", a=mean_mask)
nilearn.plotting.plot_roi(group_mask)


get_ipython().system('mkdir -p {datadir}/derivatives/custom_modelling/')

copes_concat = nilearn.image.concat_imgs(smooth_copes, auto_resample=True)
copes_concat.to_filename(os.path.join(datadir, "derivatives", "custom_modelling", "lips_vs_others_copes.nii.gz"))

group_mask = nilearn.image.resample_to_img(group_mask, copes_concat, interpolation='nearest')
group_mask.to_filename(os.path.join(datadir, "derivatives", "custom_modelling", "group_mask.nii.gz"))


group_mask.shape


randomise = mem.cache(fsl.Randomise)
randomise_results = randomise(in_file=os.path.join(datadir, "derivatives", "custom_modelling", "lips_vs_others_copes.nii.gz"),
                              mask=os.path.join(datadir, "derivatives", "custom_modelling", "group_mask.nii.gz"),
                              one_sample_group_mean=True,
                              tfce=True,
                              vox_p_values=True,
                              num_perm=500)
randomise_results.outputs


nilearn.plotting.plot_stat_map(randomise_results.outputs.t_corrected_p_files[0], threshold=0.95)


fig = nilearn.plotting.plot_stat_map(randomise_results.outputs.tstat_files[0], alpha=0.5, cut_coords=(-21, 0, 18))
fig.add_contours(randomise_results.outputs.t_corrected_p_files[0], levels=[0.95], colors='w')


get_ipython().magic('pinfo nilearn.plotting.plot_stat_map')





# This code will load the model information, generate the model definition, and run the model estimation using FSL
# 

import nipype.algorithms.modelgen as model   # model generation
from  nipype.interfaces import fsl, spm   
from nipype.interfaces.base import Bunch
import os,json,glob,sys
import numpy
import nibabel
import nilearn.plotting
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

datadir='/home/vagrant/data/ds000114_R2.0.1/'
    
results_dir = os.path.abspath("../../results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

from nipype.caching import Memory
mem = Memory(base_dir='.')

print('Using data from',datadir)


from bids.grabbids import BIDSLayout
layout = BIDSLayout(datadir)
layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[0].filename


import pandas as pd
events = pd.read_csv(os.path.join(datadir, "task-fingerfootlips_events.tsv"), sep="\t")
events


for trial_type in events.trial_type.unique():
    print(events[events.trial_type == trial_type])


events[events.trial_type == 'Finger'].duration


source_epi = layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[5]

confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_confounds.tsv"%(source_epi.subject,
                                                                                                                           source_epi.session)),
           sep="\t", na_values="n/a")

info = [Bunch(conditions=['Finger',
                          'Foot',
                          'Lips'],
              onsets=[list(events[events.trial_type == 'Finger'].onset-10),
                      list(events[events.trial_type == 'Foot'].onset-10),
                      list(events[events.trial_type == 'Lips'].onset-10)],
              durations=[list(events[events.trial_type == 'Finger'].duration),
                          list(events[events.trial_type == 'Foot'].duration),
                          list(events[events.trial_type == 'Lips'].duration)],
             regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
                         list(confounds.aCompCor0[4:]),
                         list(confounds.aCompCor1[4:]),
                         list(confounds.aCompCor2[4:]),
                         list(confounds.aCompCor3[4:]),
                         list(confounds.aCompCor4[4:]),
                         list(confounds.aCompCor5[4:]),
                        ],
             regressor_names=['FramewiseDisplacement',
                              'aCompCor0',
                              'aCompCor1',
                              'aCompCor2',
                              'aCompCor3',
                              'aCompCor4',
                              'aCompCor5'],
              amplitudes=None,
              tmod=None,
              pmod=None)
       ]

skip = mem.cache(fsl.ExtractROI)
skip_results = skip(in_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(source_epi.subject,
                                                                                                                           source_epi.session)),
                     t_min=4, t_size=-1, output_type="NIFTI")
s = model.SpecifySPMModel()
s.inputs.input_units = 'secs'
s.inputs.functional_runs = skip_results.outputs.roi_file
s.inputs.time_repetition = layout.get_metadata(source_epi.filename)["RepetitionTime"]
s.inputs.high_pass_filter_cutoff = 128.
s.inputs.concatenate_runs=False
s.inputs.output_units='secs'
s.inputs.subject_info = info
specify_model_results = s.run()
specify_model_results.outputs


finger_cond = ['Finger','T', ['Finger'],[1]]
foot_cond = ['Foot','T', ['Foot'],[1]]
lips_cond = ['Lips','T', ['Lips'],[1]]
lips_vs_others = ["Lips vs. others",'T', ['Finger', 'Foot', 'Lips'],[-0.5, -0.5, 1]]
all_motor = ["All motor", 'F', [finger_cond, foot_cond, lips_cond]]

contrasts=[finger_cond, foot_cond, lips_cond, lips_vs_others, all_motor]
    
matlab_cmd = '/home/vagrant/spm12/run_spm12.sh /home/vagrant/mcr/v85/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

level1design = mem.cache(spm.model.Level1Design)
level1design_results = level1design(interscan_interval = layout.get_metadata(source_epi.filename)["RepetitionTime"],
                                    bases = {'hrf':{'derivs': [0,0]}},
                                    session_info = specify_model_results.outputs.session_info,
                                    model_serial_correlations='AR(1)',
                                    timing_units='secs')

level1design_results.outputs


estimatemodel = mem.cache(spm.model.EstimateModel)
estimatemodel = estimatemodel(estimation_method={'Classical': 1}, 
                              spm_mat_file=level1design_results.outputs.spm_mat_file)
estimatemodel.outputs


estimatecontrasts = mem.cache(spm.model.EstimateContrast)
estimatecontrasts = estimatecontrasts(contrasts=contrasts,
                                      spm_mat_file=estimatemodel.outputs.spm_mat_file,
                                      beta_images=estimatemodel.outputs.beta_images,
                                      residual_image=estimatemodel.outputs.residual_image)
estimatecontrasts.outputs


for con_image in estimatecontrasts.outputs.spmT_images:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(con_image, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3)


for con_image in estimatecontrasts.outputs.con_images:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(con_image, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False)


copes = {}
for i in range(10):
    source_epi = layout.get(type="bold", task="fingerfootlips", session="test", extensions="nii.gz")[i]

    confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep", 
                                         "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                         "sub-%s_ses-%s_task-fingerfootlips_bold_confounds.tsv"%(source_epi.subject,
                                                                                                                               source_epi.session)),
               sep="\t", na_values="n/a")

    info = [Bunch(conditions=['Finger',
                              'Foot',
                              'Lips'],
                  onsets=[list(events[events.trial_type == 'Finger'].onset-10),
                          list(events[events.trial_type == 'Foot'].onset-10),
                          list(events[events.trial_type == 'Lips'].onset-10)],
                  durations=[list(events[events.trial_type == 'Finger'].duration),
                              list(events[events.trial_type == 'Foot'].duration),
                              list(events[events.trial_type == 'Lips'].duration)],
                 regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
                             list(confounds.aCompCor0[4:]),
                             list(confounds.aCompCor1[4:]),
                             list(confounds.aCompCor2[4:]),
                             list(confounds.aCompCor3[4:]),
                             list(confounds.aCompCor4[4:]),
                             list(confounds.aCompCor5[4:]),
                            ],
                 regressor_names=['FramewiseDisplacement',
                                  'aCompCor0',
                                  'aCompCor1',
                                  'aCompCor2',
                                  'aCompCor3',
                                  'aCompCor4',
                                  'aCompCor5'],
                  amplitudes=None,
                  tmod=None,
                  pmod=None)
           ]

    skip = mem.cache(fsl.ExtractROI)
    skip_results = skip(in_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                            "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                            "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(source_epi.subject,
                                                                                                                               source_epi.session)),
                         t_min=4, t_size=-1, output_type="NIFTI")
    s = model.SpecifySPMModel()
    s.inputs.input_units = 'secs'
    s.inputs.functional_runs = skip_results.outputs.roi_file
    s.inputs.time_repetition = layout.get_metadata(source_epi.filename)["RepetitionTime"]
    s.inputs.high_pass_filter_cutoff = 128.
    s.inputs.concatenate_runs=False
    s.inputs.output_units='secs'
    s.inputs.subject_info = info
    specify_model_results = s.run()
    
    finger_cond = ['Finger','T', ['Finger'],[1]]
    foot_cond = ['Foot','T', ['Foot'],[1]]
    lips_cond = ['Lips','T', ['Lips'],[1]]
    lips_vs_others = ["Lips vs. others",'T', ['Finger', 'Foot', 'Lips'],[-0.5, -0.5, 1]]
    all_motor = ["All motor", 'F', [finger_cond, foot_cond, lips_cond]]

    contrasts=[finger_cond, foot_cond, lips_cond, lips_vs_others, all_motor]

    matlab_cmd = '/home/vagrant/spm12/run_spm12.sh /home/vagrant/mcr/v85/ script'
    spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

    level1design = mem.cache(spm.model.Level1Design)
    level1design_results = level1design(interscan_interval = layout.get_metadata(source_epi.filename)["RepetitionTime"],
                                        bases = {'hrf':{'derivs': [1,1]}},
                                        session_info = specify_model_results.outputs.session_info,
                                        model_serial_correlations='AR(1)',
                                        timing_units='secs')
    
    estimatemodel = mem.cache(spm.model.EstimateModel)
    estimatemodel = estimatemodel(estimation_method={'Classical': 1}, 
                                  spm_mat_file=level1design_results.outputs.spm_mat_file)
    
    estimatecontrasts = mem.cache(spm.model.EstimateContrast)
    estimatecontrasts = estimatecontrasts(contrasts=contrasts,
                                          spm_mat_file=estimatemodel.outputs.spm_mat_file,
                                          beta_images=estimatemodel.outputs.beta_images,
                                          residual_image=estimatemodel.outputs.residual_image)
    
    copes[source_epi.subject] = list(estimatecontrasts.outputs.con_images)


smooth_copes = []
for k,v in copes.items():
    smooth_cope = nilearn.image.smooth_img(v[3], 8)
    smooth_copes.append(smooth_cope)
    nilearn.plotting.plot_glass_brain(smooth_cope,
                                      display_mode='lyrz', 
                                      colorbar=True, 
                                      plot_abs=False)


nilearn.plotting.plot_glass_brain(nilearn.image.mean_img(smooth_copes),
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False)


brainmasks = glob.glob(os.path.join(datadir, "derivatives", "fmriprep", "sub-*", "ses-test", "func", "*task-fingerfootlips_*space-MNI152NLin2009cAsym*_brainmask.nii*"))

for mask in brainmasks:
    nilearn.plotting.plot_roi(mask)
    
mean_mask = nilearn.image.mean_img(brainmasks)
nilearn.plotting.plot_stat_map(mean_mask)
group_mask = nilearn.image.math_img("a>=0.95", a=mean_mask)
nilearn.plotting.plot_roi(group_mask)


get_ipython().system('mkdir -p {datadir}/derivatives/custom_modelling_spm/')

copes_concat = nilearn.image.concat_imgs(smooth_copes, auto_resample=True)
copes_concat.to_filename(os.path.join(datadir, "derivatives", "custom_modelling_spm", "lips_vs_others_copes.nii.gz"))

group_mask = nilearn.image.resample_to_img(group_mask, copes_concat, interpolation='nearest')
group_mask.to_filename(os.path.join(datadir, "derivatives", "custom_modelling_spm", "group_mask.nii.gz"))


randomise = mem.cache(fsl.Randomise)
randomise_results = randomise(in_file=os.path.join(datadir, "derivatives", "custom_modelling_spm", "lips_vs_others_copes.nii.gz"),
                              mask=os.path.join(datadir, "derivatives", "custom_modelling_spm", "group_mask.nii.gz"),
                              one_sample_group_mean=True,
                              tfce=True,
                              vox_p_values=True,
                              num_perm=500)
randomise_results.outputs


fig = nilearn.plotting.plot_stat_map(randomise_results.outputs.tstat_files[0], alpha=0.5, cut_coords=(-21, 0, 18))
fig.add_contours(randomise_results.outputs.t_corrected_p_files[0], levels=[0.95], colors='w')





# Here we apply various matrix decompositions to a real fMRI dataset.  We will use a group fMRI dataset from a stop signal task (openfmri ds030, successful stop vs go contrast (cope6), control subjects only, n=125).
# 

import os
import nibabel
import numpy
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi, plot_stat_map,plot_prob_atlas
from nilearn.image import index_img,smooth_img
from sklearn.decomposition import PCA, FastICA
get_ipython().magic('matplotlib inline')


datafile='../../../data/ds030/ds030_all_stopsignal_cope6_control.nii.gz'
assert os.path.exists(datafile)
fsldir=os.environ['FSLDIR']
maskfile=os.path.join(fsldir,'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
assert os.path.exists(maskfile)
atlasfile=os.path.join(fsldir,'data/standard/MNI152_T1_2mm.nii.gz')

masker = NiftiMasker(mask_img=maskfile,smoothing_fwhm=8)
data=masker.fit_transform(datafile)
print(data.shape)


# First let's run PCA on the data and look at the resulting components
# 

pca=PCA(5)
pca_components=pca.fit_transform(data.T).T
print(pca_components.shape)


pca_components_img=masker.inverse_transform(pca_components)

show_pct=1  # percentage of top/bottom voxels to show

for i in range(5):
    img=index_img(pca_components_img, i)
    # clean up by setting very small values to zero
    imgdata=img.get_data()
    cutoffs=numpy.percentile(imgdata,[show_pct,100.-show_pct])
    img.dataobj[numpy.logical_and(imgdata<0,imgdata>cutoffs[0])]=0
    img.dataobj[numpy.logical_and(imgdata>0,imgdata<cutoffs[1])]=0
    plot_stat_map(img, atlasfile,threshold='auto',
              #display_mode='z', cut_coords=4, 
              title="PCA Component %d"%i)





# Now let's look at the same data modeled using ICA
# 

ica=FastICA(n_components=5,max_iter=10000)
components_masked = ica.fit_transform(data.T).T


ica_components_img=masker.inverse_transform(components_masked)

show_pct=1  # percentage of top/bottom voxels to show

for i in range(5):
    img=index_img(ica_components_img, i)
    # clean up by setting very small values to zero
    imgdata=img.get_data()
    cutoffs=numpy.percentile(imgdata,[show_pct,100.-show_pct])
    img.dataobj[numpy.logical_and(imgdata<0,imgdata>cutoffs[0])]=0
    img.dataobj[numpy.logical_and(imgdata>0,imgdata<cutoffs[1])]=0
    
    plt.figure(figsize=(10,5))
    plot_stat_map(img, atlasfile,#threshold='auto',
              title="ICA Component %d"%i)
    plt.figure(figsize=(10,4))
    plt.plot(ica.components_[i,:])


# Let's plot alongside one another to see how they differ.
# 

plot_prob_atlas(pca_components_img,title='All PCA components',view_type='filled_contours',
               display_mode='z',cut_coords=6)
plot_prob_atlas(ica_components_img,title='All ICA components',view_type='filled_contours',
                display_mode='z',cut_coords=6)





