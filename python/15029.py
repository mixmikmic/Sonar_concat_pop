# # 3D Vision - CVG ETHZ
# ## Object Recognition with Deep Neural Networks
# 

get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
from recognizer_voxnet import load_pc, detector_voxnet, voxilize, voxel_scatter
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#plot point_cloud
np_pc1 = load_pc("data/chairXYZ.mat")
fig1 = plt.figure(1,figsize=(8, 10))
ax1 = fig1.add_subplot(1,1,1, projection='3d')
ax1.scatter(np_pc1[:,0], np_pc1[:,1], np_pc1[:,2])


#voxelize
tic = time.time()
np_vox1 = voxilize(np_pc1)
tictoc = time.time() - tic
print("Plot1: Voxelizing took {0:.4f}sec for {1} points".format(tictoc, np_pc1.shape[0]))


#turn
vox_scat1 = voxel_scatter(np_vox1)
fig2 = plt.figure(2,figsize=(8, 10))
ax1 = fig2.add_subplot(1,1,1, projection='3d')
ax1.scatter(vox_scat1[:,0], vox_scat1[:,1], vox_scat1[:,2])
plt.draw()


tic = time.time()
obj_det = detector_voxnet("data/weights_modelnet40_acc0-8234_2016-5-20-3-47.h5")
tictoc = time.time() - tic
print("loading the Detector took {0:.4f}sec".format(tictoc))

tic = time.time()
label1, proba1 = obj_det.predict(X_pred=np_vox1)
tictoc = time.time() - tic
print("Plot1: Detection took {0:.4f}sec".format(tictoc))
print("plot1: A " + label1 + " was detected with {0:.2f}% certainty".format(proba1 * 100) )


np_pc2 = load_pc("data/stoolXYZ.mat")

#fig1 = plt.figure(1,figsize=(8, 10))
#ax2 = fig1.add_subplot(1,1,1, projection='3d')
#ax2.scatter(np_pc2[:,0], np_pc2[:,1], np_pc2[:,2])
#plt.draw()

tic = time.time()
np_vox2 = voxilize(np_pc2)
tictoc = time.time() - tic
print("Plot2: Voxelizing took {0:.4f}sec for {1} points".format(tictoc, np_pc2.shape[0]))

vox_scat2 = voxel_scatter(np_vox2)
fig2 = plt.figure(2,figsize=(8, 10))
ax2 = fig2.add_subplot(1,1,1, projection='3d')
ax2.scatter(vox_scat2[:,0], vox_scat2[:,1], vox_scat2[:,2])
plt.draw()

tic = time.time()
label2, proba2 = obj_det.predict(X_pred=np_vox2)
tictoc = time.time() - tic
print("Plot2: Detection took {0:.4f}sec".format(tictoc))
print("Plot2: A " + label2 + " was detected with {0:.2f}% certainty".format(proba2 * 100) )


np_pc3 = load_pc("data/deskXYZ.mat")

#fig1 = plt.figure(1,figsize=(8, 10))
#ax3 = fig1.add_subplot(1,1,1, projection='3d')
#ax3.scatter(np_pc3[:,0], np_pc3[:,1], np_pc3[:,2])
#plt.draw()

tic = time.time()
np_vox3 = voxilize(np_pc3)
tictoc = time.time() - tic
print("Plot3: Voxelizing took {0:.4f}sec for {1} points".format(tictoc, np_pc3.shape[0]))

vox_scat3 = voxel_scatter(np_vox3)
fig2 = plt.figure(2,figsize=(8, 10))
ax3 = fig2.add_subplot(1,1,1, projection='3d')
ax3.scatter(vox_scat3[:,0], vox_scat3[:,1], vox_scat3[:,2])
plt.draw()

tic = time.time()
label3, proba3 = obj_det.predict(X_pred=np_vox3)
tictoc = time.time() - tic
print("Plot3: Detection took {0:.4f}sec".format(tictoc))
print("Plot3: A " + label3 + " was detected with {0:.2f}% certainty".format(proba3 * 100) )





